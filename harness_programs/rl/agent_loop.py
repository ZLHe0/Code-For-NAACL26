"""A verl agent loop that runs a domain with the serving pipeline in the loop.

One call to :meth:`HarnessAgentLoop.run` plays one episode of a domain for one dataset row:

1. the row's task is rebuilt (:func:`task_from_row`) and its environment created;
2. each turn, the full transcript goes through the episode's
   :class:`~harness_programs.serving.serve.EpisodeContext` (compression, then the context budget, then
   the steering tail), the served messages are rendered with the tokenizer's chat template, and
   the policy is sampled token-in/token-out through verl's ``server_manager.generate``;
3. the decoded generation becomes an assistant message (:mod:`.parsing`), the environment
   steps on it, and its observations extend the transcript;
4. when the episode ends the environment's reward is attached, the episode is appended to the
   per-step dump (:mod:`.step_marker`), and the turns become training rows (:mod:`.traces`).

With the pipeline active each turn's prompt is rebuilt, so the episode returns **one row per
turn** (a list of ``AgentLoopOutput``); stock verl accepts one output per sample, and the
change it needs is described in ``docs/rl.md``. Without the pipeline the episode is a single
row in verl's usual form, with observation tokens masked out.

The episode logic lives in :class:`EpisodeRunner`, which depends only on four callables
(generate, render, render observations, decode), so it runs and is tested without verl. The
verl adapter is defined only when verl is importable; otherwise constructing it raises an
``ImportError`` that names what is missing. Rewards come from the environment, so every row
carries a score and verl needs no reward function.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

from ..domains import load as load_domain
from ..domains.base import Domain, Task
from ..rollout.runner import EpisodeRecord
from ..serving.budget import serving_limit_from
from ..serving.messages import Messages
from ..serving.serve import ContextPipeline, EpisodeContext
from ..serving.tokens import TokenCounter
from .parsing import to_assistant_message
from .step_marker import EpisodeDumpWriter, ReplicateCounter, read_global_step, rollout_meta
from .traces import Trace, TurnRecord, failure_trace, merged_trace, per_turn_traces

logger = logging.getLogger(__name__)

try:
    from verl.experimental.agent_loop import agent_loop as _verl_agent_loop
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
except ImportError as _exc:  # verl is optional: everything but the adapter works without it
    _VERL_IMPORT_ERROR: ImportError | None = _exc
    _verl_agent_loop = None
    AgentLoopBase = object  # type: ignore[assignment,misc]
    AgentLoopMetrics = AgentLoopOutput = None  # type: ignore[assignment,misc]
else:
    _VERL_IMPORT_ERROR = None

#: Module attribute the multi-output patch sets in verl's ``agent_loop.py`` (see ``docs/rl.md``).
MULTI_OUTPUT_FLAG = "SUPPORTS_MULTI_OUTPUT"
#: Name of the trainer-config block whose keys override the agent-loop YAML entry.
TRAINER_SECTION = "harness_programs"
TRACE_MODES = ("auto", "per_turn", "merged")
VALIDATION_SPLITS = ("val", "valid", "validation", "test", "dev")


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Recursively turn mapping-like config nodes (e.g. OmegaConf) into plain dicts and lists."""
    if isinstance(value, (str, bytes)) or value is None:
        return value
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and hasattr(value, "__len__")):
        return [_plain(v) for v in value]
    return value


def _get(node: Any, key: str, default: Any = None) -> Any:
    """``node[key]`` for dicts and config nodes alike, without materializing the whole node."""
    if node is None:
        return default
    getter = getattr(node, "get", None)
    return getter(key, default) if callable(getter) else getattr(node, key, default)


@dataclass(frozen=True)
class LoopSettings:
    """What the agent loop is asked to run. Built from the agent-loop YAML and the trainer config.

    Attributes:
        domain: the domain name (``harness_programs.domains.AVAILABLE``).
        domain_options: field overrides for the domain's dataclass (e.g. ``{reason: true}``).
        trace_mode: ``per_turn`` (one row per turn), ``merged`` (one row per episode) or ``auto``
            (per-turn exactly when the serving pipeline is active).
        budget_enable: run episodes through the serving pipeline.
        history_budget: history tokens allowed above the pinned head; ``None`` caps only at the
            serving limit.
        trim_overflow: cut the largest observation when even one turn exceeds the budget.
        program_path: the harness program document (compression + triggers).
        program_dynamic: the program may not exist yet; serve without it until it is installed.
        max_turns: assistant-turn cap when the rollout config sets none (else the domain's own).
        per_turn_max_tokens: generation cap for one turn (else the rollout response length).
        max_per_episode, max_per_trigger: caps on how often triggers may fire.
        run_dir: where the step marker is read and episode dumps are written.
        dump_episodes: append every episode to ``<run_dir>/rollouts/step_<N>.jsonl``.
        tasks_path: optional task file for rows that carry only a ``task_id``.
        require_multi_output_patch: refuse per-turn mode on a verl without the multi-output patch.
    """

    domain: str
    domain_options: tuple[tuple[str, Any], ...] = ()
    trace_mode: str = "auto"
    budget_enable: bool = False
    history_budget: int | None = None
    trim_overflow: bool = False
    program_path: str | None = None
    program_dynamic: bool = False
    max_turns: int | None = None
    per_turn_max_tokens: int | None = None
    max_per_episode: int = 0
    max_per_trigger: int = 0
    run_dir: str | None = None
    dump_episodes: bool = True
    tasks_path: str | None = None
    require_multi_output_patch: bool = True

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("the agent loop needs a `domain`")
        options = self.domain_options
        if isinstance(options, Mapping) or hasattr(options, "items"):
            options = _plain(options).items()
        object.__setattr__(self, "domain_options", tuple(sorted((str(k), v) for k, v in (options or ()))))
        if self.trace_mode not in TRACE_MODES:
            raise ValueError(f"trace_mode must be one of {TRACE_MODES}, got {self.trace_mode!r}")
        if not self.budget_enable:
            for name, value in (("context_budget.history_budget", self.history_budget),
                                ("context_budget.trim_overflow", self.trim_overflow or None),
                                ("program_path", self.program_path)):
                if value is not None:
                    raise ValueError(
                        f"{name} is set but context_budget.enable is not. The serving pipeline is what "
                        "applies it, so the run would silently be the unmanaged baseline.")
        if self.budget_enable and self.trace_mode == "merged":
            raise ValueError(
                "trace_mode=merged with the serving pipeline on would train each turn's tokens under a "
                "prompt they were not sampled from; use per_turn (or auto)")
        if self.program_dynamic and not self.program_path:
            raise ValueError("program_dynamic is set but program_path is not")

    @property
    def per_turn(self) -> bool:
        return self.trace_mode == "per_turn" or (self.trace_mode == "auto" and self.budget_enable)

    @classmethod
    def from_sources(cls, agent_kwargs: Mapping[str, Any] | None = None,
                     trainer_section: Mapping[str, Any] | None = None) -> LoopSettings:
        """Merge the agent-loop entry with the trainer config's ``harness_programs`` block (which wins).

        Both accept ``context_budget: {enable, history_budget, trim_overflow}`` as a nested block.
        """
        merged: dict[str, Any] = {}
        for source in (agent_kwargs, trainer_section):
            for key, value in (_plain(source) or {}).items():
                if key == "context_budget" and isinstance(value, dict):
                    for sub, sub_value in value.items():
                        merged["budget_enable" if sub == "enable" else sub] = sub_value
                else:
                    merged[key] = value
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(merged) - known)
        if unknown:
            raise ValueError(f"unknown agent-loop settings {unknown}; known: {sorted(known)}")
        if merged.get("history_budget") in ("null", "none", "None"):
            merged["history_budget"] = None
        for key in ("history_budget", "max_turns", "per_turn_max_tokens"):
            if merged.get(key) is not None:
                merged[key] = int(merged[key])
        return cls(**merged)


@dataclass(frozen=True)
class Limits:
    """Token and turn limits of the run, taken from verl's rollout config."""

    prompt_length: int
    response_length: int
    model_window: int
    per_turn_max_tokens: int
    max_turns: int
    max_parallel_calls: int | None = None
    want_logprobs: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.per_turn_max_tokens <= self.response_length:
            raise ValueError(f"per_turn_max_tokens={self.per_turn_max_tokens} must be in "
                             f"(0, response_length={self.response_length}]")
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")

    @property
    def serving_limit(self) -> int:
        return serving_limit_from(self.prompt_length, self.model_window, self.per_turn_max_tokens)

    @classmethod
    def from_rollout_config(cls, rollout: Any, settings: LoopSettings, domain: Domain) -> Limits:
        """Read ``actor_rollout_ref.rollout``; its turn cap wins over the YAML's and the domain's."""
        prompt_length, response_length = int(_get(rollout, "prompt_length")), int(_get(rollout, "response_length"))
        multi_turn = _get(rollout, "multi_turn")
        turns = _get(multi_turn, "max_assistant_turns") or settings.max_turns or domain.max_turns
        parallel = _get(multi_turn, "max_parallel_calls")
        return cls(prompt_length=prompt_length, response_length=response_length,
                   model_window=int(_get(rollout, "max_model_len") or prompt_length + response_length),
                   per_turn_max_tokens=int(settings.per_turn_max_tokens or response_length),
                   max_turns=int(turns), max_parallel_calls=None if parallel is None else int(parallel),
                   want_logprobs=bool(_get(rollout, "calculate_log_probs", False)))


def per_turn_config_errors(config: Any) -> list[str]:
    """Trainer settings that would silently mistrain with per-turn traces; ``[]`` when supported.

    A per-turn step holds a varying number of rows, so the advantage must be the instance-level
    GRPO one, the loss must not normalize by a fixed row count, and fixed-size micro-batches must
    not assume one. ``config`` is the full trainer config.
    """
    algorithm = _get(config, "algorithm")
    actor = _get(_get(config, "actor_rollout_ref"), "actor")
    data = _get(config, "data")
    errors = []
    estimator = _get(algorithm, "adv_estimator", "gae")
    estimator = str(getattr(estimator, "value", estimator)).strip().lower()
    if estimator != "grpo":
        errors.append(f"algorithm.adv_estimator must be grpo (got {estimator!r}): the instance-level "
                      "advantage is GRPO's")
    micro = _get(actor, "ppo_micro_batch_size_per_gpu")
    if not _get(actor, "use_dynamic_bsz", False) and micro is not None and int(micro) != 1:
        errors.append("set actor.use_dynamic_bsz=true (or ppo_micro_batch_size_per_gpu=1): fixed-size "
                      "micro-batches assume a fixed row count per step")
    agg = str(_get(actor, "loss_agg_mode", "token-mean")).strip().lower()
    if agg != "token-mean":
        errors.append(f"actor.loss_agg_mode must be token-mean (got {agg!r}): sequence-mean modes divide by "
                      "a fixed batch size while the row count varies, which scales the step size")
    train_bsz, mini_bsz = _get(data, "train_batch_size"), _get(actor, "ppo_mini_batch_size")
    if train_bsz is not None and mini_bsz is not None and (int(mini_bsz) <= 0 or int(train_bsz) % int(mini_bsz)):
        errors.append(f"data.train_batch_size ({train_bsz}) must be a multiple of actor.ppo_mini_batch_size "
                      f"({mini_bsz}) so each step makes a fixed number of optimizer updates")
    return errors


def configure_domain(settings: LoopSettings, *, max_turns: int | None = None) -> Domain:
    """The registered domain with ``settings.domain_options`` applied, and its turn cap set to ``max_turns``.

    The environment tells the policy how many turns it has, so its cap must match the loop's.
    """
    domain = load_domain(settings.domain)
    options = dict(settings.domain_options)
    if max_turns is not None and getattr(domain, "max_turns", None) != max_turns:
        options.setdefault("max_turns", int(max_turns))
    if not options:
        return domain
    if not dataclasses.is_dataclass(domain):
        raise TypeError(f"domain {settings.domain!r} is not a dataclass; domain_options cannot be applied")
    known = {f.name for f in dataclasses.fields(domain)}
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(f"unknown domain_options {unknown} for {settings.domain!r}; known: {sorted(known)}")
    return dataclasses.replace(domain, **options)


# ---------------------------------------------------------------------------------------------
# Dataset rows
# ---------------------------------------------------------------------------------------------


def _extra_info(row: Mapping[str, Any]) -> dict[str, Any]:
    return _plain(row.get("extra_info")) or {}


def task_to_row(task: Task, *, domain: str, index: int, split: str = "train", validate: bool = False,
                prompt_messages: Messages | None = None, agent_name: str | None = None) -> dict[str, Any]:
    """A verl dataset row for ``task``. The task travels as JSON in ``extra_info.task``.

    ``prompt`` is what verl's dataset filter measures; pass the environment's pinned head as
    ``prompt_messages`` to filter on the real first-turn prompt. The environment, not this
    column, builds the episode's messages.
    """
    row: dict[str, Any] = {
        "data_source": domain,
        "prompt": list(prompt_messages) if prompt_messages else [{"role": "user", "content": task.prompt}],
        "reward_model": {"style": "environment", "ground_truth": ""},
        "extra_info": {"index": int(index), "split": split, "validate": bool(validate), "task_id": task.task_id,
                       "task": json.dumps({"task_id": task.task_id, "prompt": task.prompt, "data": task.data},
                                          default=str)},
    }
    if agent_name:
        row["agent_name"] = agent_name
    return row


def task_from_row(row: Mapping[str, Any], *, lookup: Callable[[str, str], Task] | None = None) -> Task:
    """Rebuild a :class:`Task` from a dataset row's ``extra_info``.

    Accepts ``extra_info.task`` (JSON string or dict with ``task_id``, ``prompt``, ``data``), or
    flat ``task_id`` / ``prompt`` / ``data`` keys, or a bare ``task_id`` resolved via ``lookup``
    ``(task_id, split) -> Task``.
    """
    info = _extra_info(row)
    raw = info.get("task")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        info = {**info, **raw}
    if "task_id" not in info:
        raise ValueError("dataset row has no extra_info.task or extra_info.task_id")
    task_id = str(info["task_id"])
    if "prompt" in info:
        data = info.get("data") or {}
        if isinstance(data, str):
            data = json.loads(data)
        return Task(task_id=task_id, prompt=str(info["prompt"]), data=dict(data))
    if lookup is None:
        raise ValueError(f"row for task {task_id!r} carries no prompt and no tasks_path is configured")
    return lookup(task_id, str(info.get("split") or "train"))


def row_is_validation(row: Mapping[str, Any]) -> bool:
    info = _extra_info(row)
    if "validate" in info:
        return bool(info["validate"])
    return str(info.get("split") or "").lower() in VALIDATION_SPLITS


def make_rows(domain: str, tasks_path: str, *, split: str = "train", validate: bool | None = None,
              agent_name: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Dataset rows for every task of ``tasks_path``, ready to write as verl's parquet input."""
    is_val = split.lower() in VALIDATION_SPLITS if validate is None else bool(validate)
    tasks = load_domain(domain).load_tasks(tasks_path, split=split, limit=limit)
    return [task_to_row(t, domain=domain, index=i, split=split, validate=is_val, agent_name=agent_name)
            for i, t in enumerate(tasks)]


# ---------------------------------------------------------------------------------------------
# The episode
# ---------------------------------------------------------------------------------------------


@dataclass
class Generation:
    """One sampled turn, as returned by the inference server."""

    token_ids: list[int]
    log_probs: list[float] | None = None
    num_preempted: int | None = None


GenerateFn = Callable[[list[int], dict[str, Any]], Awaitable[Generation]]
RenderFn = Callable[[Messages], Awaitable[list[int]]]
DecodeFn = Callable[[list[int]], str]


@dataclass
class EpisodeOutcome:
    """A finished episode: its token-level turns, its record, and timing."""

    turns: list[TurnRecord]
    record: EpisodeRecord
    initial_messages: Messages
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.record.status == "error" or not self.turns


class EpisodeRunner:
    """Plays episodes of one domain under one set of settings; shared by all episodes of a worker.

    The pipeline is shared on purpose: it re-reads the program document when its modification
    time changes, so a program installed mid-run is picked up by the next episode that starts.
    """

    def __init__(self, domain: Domain, settings: LoopSettings, limits: Limits, *,
                 token_counter: TokenCounter | None = None) -> None:
        self.domain = domain
        self.settings = settings
        self.limits = limits
        self.per_turn = settings.per_turn
        self._pipelines: dict[bool, ContextPipeline] = {}
        if settings.budget_enable:
            if token_counter is None:
                raise ValueError("the serving pipeline needs a token counter")
            for with_program in (False, True) if settings.program_path else (False,):
                self._pipelines[with_program] = ContextPipeline(
                    domain.name, program_path=settings.program_path if with_program else None,
                    token_counter=token_counter, serving_limit=limits.serving_limit,
                    history_budget=settings.history_budget, max_steps=limits.max_turns,
                    trim_overflow=settings.trim_overflow, max_per_episode=settings.max_per_episode,
                    max_per_trigger=settings.max_per_trigger)
        self.dump = EpisodeDumpWriter(settings.run_dir) if settings.run_dir and settings.dump_episodes else None
        self.replicates = ReplicateCounter()
        self._tasks: dict[str, dict[str, Task]] = {}
        self._tasks_lock = threading.Lock()
        self.preflight()

    def preflight(self) -> None:
        """Fail at construction, not in every episode, on a missing or mismatched program."""
        path = self.settings.program_path
        if not path:
            return
        if not os.path.exists(path):
            if self.settings.program_dynamic:
                logger.info("program %s not installed yet; serving without it until it is", path)
                return
            raise FileNotFoundError(f"program_path {path} does not exist (set program_dynamic for a "
                                    "program installed during training)")
        self._pipelines[True].new_episode()

    def new_context(self) -> EpisodeContext | None:
        if not self._pipelines:
            return None
        installed = bool(self.settings.program_path) and os.path.exists(self.settings.program_path or "")
        return self._pipelines[installed].new_episode()

    def lookup_task(self, task_id: str, split: str) -> Task:
        if not self.settings.tasks_path:
            raise ValueError(f"task {task_id!r} needs tasks_path to resolve")
        with self._tasks_lock:
            if split not in self._tasks:
                tasks = self.domain.load_tasks(self.settings.tasks_path, split=split)
                self._tasks[split] = {t.task_id: t for t in tasks}
        return self._tasks[split][task_id]

    async def run(self, task: Task, *, generate: GenerateFn, render: RenderFn, decode: DecodeFn,
                  render_observations: RenderFn | None = None, sampling_params: Mapping[str, Any] | None = None,
                  rollout: Mapping[str, Any] | None = None) -> EpisodeOutcome:
        """Play one episode. Environment and pipeline calls run in the default executor."""
        if not self.per_turn and render_observations is None:
            raise ValueError("merged traces need render_observations to tokenize observations")
        loop = asyncio.get_running_loop()

        def off(fn: Callable[..., Any], *args: Any) -> Awaitable[Any]:
            return loop.run_in_executor(None, functools.partial(fn, *args))

        params = dict(sampling_params or {})
        params.pop("max_tokens", None)
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0, "serve": 0.0}
        context = self.new_context()
        turns: list[TurnRecord] = []
        served_log: list[dict[str, Any]] = []
        messages: Messages = []
        initial: Messages = []
        status, error = "max_turns", None
        env = None
        try:
            env = await off(self.domain.make_env, task)
            messages = list(await off(env.reset))
            initial = [dict(m) for m in messages]
            stream: list[int] | None = None
            used = 0
            for t in range(self.limits.max_turns):
                if self.per_turn:
                    prompt_messages = messages
                    if context is not None:
                        started = time.perf_counter()
                        try:
                            served = await off(context.serve, messages)
                        except ValueError as exc:
                            status, error = "error", f"unservable at turn {t}: {exc}"
                            break
                        metrics["serve"] += time.perf_counter() - started
                        prompt_messages = served.messages
                        served_log.append({**served.stats.to_dict(), "fired": list(served.fired)})
                    prompt_ids = list(await render(prompt_messages))
                    cap = self.limits.per_turn_max_tokens
                else:
                    if stream is None:
                        stream = list(await render(messages))
                    prompt_ids = stream
                    cap = min(self.limits.per_turn_max_tokens, self.limits.response_length - used)
                    if cap <= 0:
                        break
                if len(prompt_ids) > self.limits.prompt_length and (self.per_turn or t == 0):
                    status, error = "error", (f"turn {t}: prompt of {len(prompt_ids)} tokens exceeds "
                                              f"prompt_length={self.limits.prompt_length}")
                    break
                cap = min(cap, self.limits.model_window - len(prompt_ids))
                if cap <= 0:
                    status, error = "error", f"turn {t}: no room left in the model window"
                    break
                started = time.perf_counter()
                gen = await generate(list(prompt_ids), {**params, "max_tokens": cap})
                metrics["generate_sequences"] += time.perf_counter() - started
                if gen.num_preempted is not None and gen.num_preempted >= 0:
                    metrics["num_preempted"] = metrics.get("num_preempted", 0.0) + gen.num_preempted
                turn = TurnRecord(prompt_ids=list(prompt_ids), response_ids=list(gen.token_ids),
                                  response_logprobs=self._logprobs(gen))
                turns.append(turn)
                assistant, _ = to_assistant_message(decode(list(gen.token_ids)), parse_tools=bool(self.domain.tools),
                                                    turn=t, max_calls=self.limits.max_parallel_calls)
                messages.append(assistant)
                started = time.perf_counter()
                result = await off(env.step, assistant)
                metrics["tool_calls"] += time.perf_counter() - started
                messages.extend(result.observations)
                if result.done:
                    status = "answered"
                    break
                if not self.per_turn and render_observations is not None:
                    turn.observation_ids = list(await render_observations(list(result.observations))) \
                        if result.observations else []
                    stream = [*prompt_ids, *turn.response_ids, *turn.observation_ids]
                    used += len(turn.response_ids) + len(turn.observation_ids)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not take down the batch
            logger.warning("task %s: episode failed: %s", task.task_id, exc, exc_info=True)
            status, error = "error", f"{type(exc).__name__}: {exc}"
        record = await off(self._record, env, task, messages, status, context, served_log, rollout or {})
        if self.dump is not None:
            # ``error`` is an extra column for debugging; EpisodeRecord readers ignore it.
            row = {**record.to_dict(), **({"error": error} if error else {})}
            try:
                await off(self.dump.write, row)
            except OSError as exc:
                logger.warning("could not dump episode %s: %s", record.episode_id, exc)
        return EpisodeOutcome(turns=turns, record=record, initial_messages=initial, error=error, metrics=metrics)

    def _logprobs(self, gen: Generation) -> list[float] | None:
        if not self.limits.want_logprobs:
            return None
        if gen.log_probs is None or len(gen.log_probs) != len(gen.token_ids):
            return [0.0] * len(gen.token_ids)
        return [float(x) for x in gen.log_probs]

    def _record(self, env: Any, task: Task, messages: Messages, status: str, context: EpisodeContext | None,
                served_log: list[dict[str, Any]], rollout: Mapping[str, Any]) -> EpisodeRecord:
        def safe(fn: Callable[[], Any], default: Any) -> Any:
            if env is None:
                return default
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("task %s: %s failed: %s", task.task_id, getattr(fn, "__name__", "env call"), exc)
                return default

        return EpisodeRecord(
            episode_id=uuid.uuid4().hex[:12], harness=self.domain.name, task_id=task.task_id, status=status,
            num_steps=sum(1 for m in messages if m.get("role") == "assistant"), max_steps=self.limits.max_turns,
            messages=messages, reward=float(safe(lambda: env.reward(), 0.0)),
            final_output=str(safe(lambda: env.final_output(), "")), gold=safe(lambda: env.gold_view(), None),
            context_budget=self.settings.history_budget if self.settings.budget_enable else None,
            triggers={"setup": context.setup, "firings": context.firings} if context is not None else {},
            served=served_log, rollout=dict(rollout),
        )

    def traces(self, outcome: EpisodeOutcome, *, eos_token_id: int,
               fallback_prompt_ids: Sequence[int] = ()) -> list[Trace]:
        """Training rows for ``outcome``; a failed episode becomes one zero-gradient placeholder."""
        reward = float(outcome.record.reward)
        if not outcome.failed:
            if self.per_turn:
                return per_turn_traces(outcome.turns, reward=reward, prompt_length=self.limits.prompt_length,
                                       response_length=self.limits.response_length)
            return [merged_trace(outcome.turns, reward=reward, response_length=self.limits.response_length)]
        prompt = list(outcome.turns[0].prompt_ids) if outcome.turns else list(fallback_prompt_ids)
        prompt = prompt[-self.limits.prompt_length:] or [int(eos_token_id)]
        return [failure_trace(prompt, eos_token_id=eos_token_id, want_logprobs=self.limits.want_logprobs)]


# One runner per (settings, limits, tokenizer) per worker process: verl builds a fresh agent-loop
# object for every sample, so anything costly or stateful is kept here instead.
_RUNNERS: dict[tuple[Any, ...], EpisodeRunner] = {}
_RUNNERS_LOCK = threading.Lock()


def get_runner(settings: LoopSettings, limits: Limits, *, cache_key: Any = None,
               counter_factory: Callable[[Domain], TokenCounter] | None = None) -> EpisodeRunner:
    key = (settings, limits, cache_key)
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(key)
        if runner is None:
            domain = configure_domain(settings, max_turns=limits.max_turns)
            counter = counter_factory(domain) if counter_factory is not None else None
            runner = _RUNNERS[key] = EpisodeRunner(domain, settings, limits, token_counter=counter)
        return runner


# ---------------------------------------------------------------------------------------------
# The verl adapter
# ---------------------------------------------------------------------------------------------


def _token_ids(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping) or hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    return [int(t) for t in rendered]


def _settings_keys() -> set[str]:
    return {f.name for f in fields(LoopSettings)} | {"context_budget"}


if _VERL_IMPORT_ERROR is None:

    class HarnessAgentLoop(AgentLoopBase):  # type: ignore[valid-type,misc]
        """verl agent loop running one domain episode per sample.

        Registered through the agent-loop YAML (``configs/agent_loop.yaml``), not a decorator, so
        the YAML entry's keyword arguments reach the constructor. Keys under the trainer config's
        ``harness_programs`` block override that entry.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            own = {k: kwargs.pop(k) for k in list(kwargs) if k in _settings_keys()}
            super().__init__(*args, **kwargs)
            section = self.config.get(TRAINER_SECTION) if hasattr(self.config, "get") else None
            settings = LoopSettings.from_sources(own, section)
            domain = configure_domain(settings)
            limits = Limits.from_rollout_config(self.rollout_config, settings, domain)
            if (settings.per_turn and settings.require_multi_output_patch
                    and not getattr(_verl_agent_loop, MULTI_OUTPUT_FLAG, False)):
                raise RuntimeError(
                    "per-turn traces return several outputs per sample, which this verl does not accept. "
                    f"Apply the multi-output change described in docs/rl.md (it sets {MULTI_OUTPUT_FLAG}), "
                    "or run without the serving pipeline.")
            if settings.per_turn:
                errors = per_turn_config_errors(self.config)
                if errors:
                    raise ValueError("per-turn traces need: " + "; ".join(errors))
            self._template_kwargs = dict(_plain(getattr(self, "apply_chat_template_kwargs", None)) or {})
            self._runner = get_runner(settings, limits, cache_key=id(self.tokenizer),
                                      counter_factory=self._counter)

        # -- tokenization ------------------------------------------------------------------------

        def _render_sync(self, messages: Messages, tools: Sequence[dict[str, Any]] | None) -> list[int]:
            rendered = self.tokenizer.apply_chat_template(
                list(messages), tools=list(tools) if tools else None, add_generation_prompt=True, tokenize=True,
                **self._template_kwargs)
            return _token_ids(rendered)

        def _counter(self, domain: Domain) -> TokenCounter:
            return lambda messages: len(self._render_sync(messages, domain.tools))

        async def _render(self, messages: Messages) -> list[int]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._render_sync, messages, self._runner.domain.tools)

        async def _render_observations(self, messages: Messages) -> list[int]:
            """Observation tokens as a continuation, the way verl's tool loop renders tool replies."""
            loop = asyncio.get_running_loop()
            ids = await loop.run_in_executor(None, self._render_sync, messages, None)
            prefix = [int(t) for t in (getattr(self, "system_prompt", None) or [])]
            return ids[len(prefix):] if prefix and ids[:len(prefix)] == prefix else ids

        def _decode(self, token_ids: list[int]) -> str:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

        # -- identity ----------------------------------------------------------------------------

        def _identity(self, row: Mapping[str, Any], task: Task) -> dict[str, Any]:
            info = dict(getattr(self, "trajectory_info", None) or {})
            step = info.get("step")
            step = int(step) if step is not None else read_global_step(self._runner.settings.run_dir)
            validate = bool(info["validate"]) if "validate" in info else row_is_validation(row)
            replicate = info.get("rollout_n")
            if replicate is None:
                replicate = self._runner.replicates.next(step, task.task_id, validate)
            return rollout_meta(step=step, group_key=task.task_id, replicate=int(replicate), validate=validate)

        # -- the loop ----------------------------------------------------------------------------

        async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
            runner = self._runner
            request_id = uuid.uuid4().hex
            eos = self.tokenizer.eos_token_id
            eos = int(eos if eos is not None else (self.tokenizer.pad_token_id or 0))

            async def generate(prompt_ids: list[int], params: dict[str, Any]) -> Generation:
                out = await self.server_manager.generate(request_id=request_id, prompt_ids=prompt_ids,
                                                         sampling_params=params)
                return Generation(token_ids=[int(t) for t in out.token_ids],
                                  log_probs=None if out.log_probs is None else [float(x) for x in out.log_probs],
                                  num_preempted=getattr(out, "num_preempted", None))

            outcome: EpisodeOutcome | None = None
            try:
                task = task_from_row(kwargs, lookup=runner.lookup_task)
                outcome = await runner.run(
                    task, generate=generate, render=self._render, decode=self._decode,
                    render_observations=self._render_observations, sampling_params=sampling_params,
                    rollout=self._identity(kwargs, task))
                traces = runner.traces(outcome, eos_token_id=eos,
                                       fallback_prompt_ids=await self._fallback_prompt(kwargs, outcome))
            except Exception as exc:  # noqa: BLE001 - degrade this sample, never the batch
                logger.warning("sample failed before an episode could run: %s", exc, exc_info=True)
                prompt = (await self._fallback_prompt(kwargs, outcome))[-runner.limits.prompt_length:] or [eos]
                traces = [failure_trace(prompt, eos_token_id=eos, want_logprobs=runner.limits.want_logprobs)]
            metrics = self._metrics(outcome)
            outputs = [AgentLoopOutput(**t.output_kwargs(), metrics=metrics) for t in traces]
            return outputs if runner.per_turn else outputs[0]

        async def _fallback_prompt(self, row: Mapping[str, Any], outcome: EpisodeOutcome | None) -> list[int]:
            messages = (outcome.initial_messages if outcome is not None else None) or _plain(row.get("raw_prompt"))
            if not messages:
                return []
            try:
                return await self._render(list(messages))
            except Exception:  # noqa: BLE001 - the placeholder is masked out anyway
                return []

        @staticmethod
        def _metrics(outcome: EpisodeOutcome | None) -> Any:
            values = dict(outcome.metrics) if outcome is not None else {}
            known = set(getattr(AgentLoopMetrics, "model_fields", {}) or {})
            kept = {k: v for k, v in values.items() if k in known}
            if "num_preempted" in kept:
                kept["num_preempted"] = int(kept["num_preempted"])
            return AgentLoopMetrics(**kept)

else:

    class HarnessAgentLoop:  # type: ignore[no-redef]
        """Placeholder used when verl is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "HarnessAgentLoop needs verl (it subclasses verl.experimental.agent_loop.AgentLoopBase); "
                "install verl 0.8.x with the multi-output change in docs/rl.md") from _VERL_IMPORT_ERROR


__all__ = [
    "HarnessAgentLoop", "EpisodeOutcome", "EpisodeRunner", "Generation", "Limits", "LoopSettings",
    "MULTI_OUTPUT_FLAG", "TRAINER_SECTION", "configure_domain", "get_runner", "make_rows", "per_turn_config_errors",
    "row_is_validation", "task_from_row", "task_to_row",
]
