"""From one episode to training rows, and from rows to GRPO advantages.

**Why one episode can be several rows.** Stock multi-turn RL concatenates an episode into a
single row: the first prompt, then every generation interleaved with the observation tokens
that followed it (masked out of the loss). That is only correct when each turn's prompt is an
append-only extension of the previous prompt plus response. Under a context budget it is not:
the served prompt at turn ``t`` is the pinned head plus a compressed, partly evicted history plus
an ephemeral guidance tail, rebuilt from scratch every turn. Training on the concatenation would
assign each generation log-probabilities under a prompt it was never sampled from.

So with the serving pipeline in the loop an episode becomes **one trace per turn**: the trace's
prompt is exactly the prompt served that turn, its response is that turn's generation, and every
response token is trainable. The episode's outcome reward lives on one designated **reward
trace** (the last by default); its siblings carry a placeholder score of zero. The trainer then
computes an **instance-level GRPO advantage**: each episode (instance) is reduced to the score of
its reward trace, instances are grouped by task (``uid``), mean and standard deviation are taken
over instances rather than rows, and the episode's advantage is broadcast to all of its traces.
That weighting matters: stock GRPO over rows would count an episode that happened to take more
turns more times in its group's baseline. When every episode is a single trace the computation
is exactly stock GRPO.

Everything here is pure Python on lists, so it is testable without verl or torch; the verl
patch described in ``docs/rl.md`` performs the same computation on tensors.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: ``extra_fields`` key marking the reward trace of an episode (exactly one per episode).
IS_REWARD_KEY = "is_reward"
#: ``extra_fields`` key carrying the episode's assistant-turn count on every trace.
ASSISTANT_TURNS_KEY = "assistant_turns"
#: ``extra_fields`` key carrying the episode's outcome on every trace (for metrics only).
EPISODE_REWARD_KEY = "episode_reward"
#: Batch column the patched trainer adds: the row's episode (instance) index.
INSTANCE_ID_KEY = "__instance_id__"
#: Batch column the patched trainer adds: whether the row is its episode's reward trace.
IS_REWARD_TRACE_KEY = "__is_reward_trace__"
#: Batch column the patched trainer adds: whether the row is a neutralized padding clone.
IS_PAD_KEY = "__is_pad__"

RewardTraceFn = Callable[[Sequence[Any]], list[bool]]


@dataclass
class TurnRecord:
    """What happened at one turn of an episode, in tokens.

    Attributes:
        prompt_ids: the prompt the policy was sampled from at this turn.
        response_ids: the turn's generation.
        response_logprobs: sampler log-probabilities of ``response_ids`` (``None`` if not requested).
        observation_ids: tokens of the environment's reply, as appended in single-trace mode
            (empty in per-turn mode, where the next turn's prompt is re-rendered instead).
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float] | None = None
    observation_ids: list[int] = field(default_factory=list)


@dataclass
class Trace:
    """One training row: the inputs of verl's ``AgentLoopOutput`` minus metrics."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None = None
    is_reward: bool = True
    turn: int = 0
    num_turns: int = 1
    reward_score: float | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def output_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs for ``AgentLoopOutput`` (everything except ``metrics``)."""
        return {
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
            "response_logprobs": None if self.response_logprobs is None else list(self.response_logprobs),
            "reward_score": self.reward_score,
            "num_turns": self.num_turns,
            "extra_fields": dict(self.extra_fields),
        }


class PromptOverflowError(ValueError):
    """A served prompt is longer than the stored prompt width.

    verl pads prompts but never truncates them, and truncating after sampling would store a row
    whose prompt did not produce its response, so the episode is dropped instead.
    """


# ---------------------------------------------------------------------------------------------
# Episode -> rows
# ---------------------------------------------------------------------------------------------


def last_trace_is_reward(traces: Sequence[Any]) -> list[bool]:
    """Default reward-trace selector: the episode's outcome is attributed to its final trace."""
    flags = [False] * len(traces)
    if flags:
        flags[-1] = True
    return flags


def _check_reward_flags(flags: Sequence[bool], n: int) -> list[bool]:
    flags = [bool(f) for f in flags]
    if len(flags) != n or sum(flags) != 1:
        raise ValueError(f"a reward-trace selector must mark exactly one of {n} traces, got {flags}")
    return flags


def per_turn_traces(turns: Sequence[TurnRecord], *, reward: float, prompt_length: int | None = None,
                    response_length: int | None = None,
                    reward_trace_fn: RewardTraceFn = last_trace_is_reward) -> list[Trace]:
    """One trace per turn: the served prompt, that turn's generation, all of it trainable.

    The reward trace carries ``reward`` as its score; the others carry ``0.0`` so verl builds its
    reward tensor from the scores the loop supplied and never calls a reward function on them.
    Every trace carries the episode's turn count and outcome in ``extra_fields``.

    Raises:
        PromptOverflowError: a served prompt exceeds ``prompt_length``.
        ValueError: no turns, or a selector that does not pick exactly one trace.
    """
    if not turns:
        raise ValueError("an episode with no generated turns has no trace")
    flags = _check_reward_flags(reward_trace_fn(turns), len(turns))
    traces = []
    for i, (turn, is_reward) in enumerate(zip(turns, flags, strict=True)):
        if prompt_length is not None and len(turn.prompt_ids) > prompt_length:
            raise PromptOverflowError(
                f"turn {i}: served prompt of {len(turn.prompt_ids)} tokens exceeds prompt_length={prompt_length}")
        response = list(turn.response_ids)
        logprobs = None if turn.response_logprobs is None else list(turn.response_logprobs)
        if response_length is not None and len(response) > response_length:
            response = response[:response_length]
            logprobs = None if logprobs is None else logprobs[:response_length]
        traces.append(Trace(
            prompt_ids=list(turn.prompt_ids), response_ids=response, response_mask=[1] * len(response),
            response_logprobs=logprobs, is_reward=is_reward, turn=i, num_turns=len(turns),
            reward_score=float(reward) if is_reward else 0.0,
            extra_fields={IS_REWARD_KEY: is_reward, ASSISTANT_TURNS_KEY: float(len(turns)),
                          EPISODE_REWARD_KEY: float(reward)},
        ))
    return traces


def merged_trace(turns: Sequence[TurnRecord], *, reward: float | None,
                 response_length: int | None = None) -> Trace:
    """The stock single-row form: first prompt, then generations and masked observations.

    Valid only when every turn's prompt is the previous prompt plus response plus observation,
    i.e. no context management between turns. Observation tokens get mask 0 and log-prob 0; the
    observation after the final generation is dropped, since no trained token depends on it. The
    response is truncated to ``response_length`` from the right, as verl's own tool loop does.
    """
    if not turns:
        raise ValueError("an episode with no generated turns has no trace")
    want_logprobs = all(t.response_logprobs is not None for t in turns)
    response: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    for i, turn in enumerate(turns):
        response.extend(turn.response_ids)
        mask.extend([1] * len(turn.response_ids))
        if want_logprobs:
            logprobs.extend(turn.response_logprobs or [])
        if i < len(turns) - 1:
            response.extend(turn.observation_ids)
            mask.extend([0] * len(turn.observation_ids))
            if want_logprobs:
                logprobs.extend([0.0] * len(turn.observation_ids))
    if response_length is not None:
        response, mask = response[:response_length], mask[:response_length]
        logprobs = logprobs[:response_length]
    return Trace(
        prompt_ids=list(turns[0].prompt_ids), response_ids=response, response_mask=mask,
        response_logprobs=logprobs if want_logprobs else None, is_reward=True, turn=0, num_turns=len(turns),
        reward_score=None if reward is None else float(reward),
        extra_fields={IS_REWARD_KEY: True, ASSISTANT_TURNS_KEY: float(len(turns)),
                      **({} if reward is None else {EPISODE_REWARD_KEY: float(reward)})},
    )


def is_append_only(turns: Sequence[TurnRecord]) -> bool:
    """Whether every turn's prompt extends the previous prompt + response + observation.

    True exactly when :func:`merged_trace` loses nothing; a context budget, a compression or a
    guidance tail all make it false.
    """
    for prev, cur in zip(turns, turns[1:], strict=False):
        expected = [*prev.prompt_ids, *prev.response_ids, *prev.observation_ids]
        if cur.prompt_ids[:len(expected)] != expected:
            return False
    return True


def failure_trace(prompt_ids: Sequence[int], *, eos_token_id: int, want_logprobs: bool,
                  reward_score: float | None = 0.0) -> Trace:
    """A zero-gradient placeholder that keeps the batch well-formed when an episode fails.

    One EOS response token with ``response_mask=[0]`` trains on nothing. It is tagged as its
    episode's reward trace so every row of a per-turn batch carries the reward-trace column.
    """
    if not prompt_ids:
        raise ValueError("a failure placeholder still needs a non-empty prompt")
    return Trace(prompt_ids=[int(t) for t in prompt_ids], response_ids=[int(eos_token_id)], response_mask=[0],
                 response_logprobs=[0.0] if want_logprobs else None, is_reward=True, turn=0, num_turns=0,
                 reward_score=reward_score,
                 extra_fields={IS_REWARD_KEY: True, ASSISTANT_TURNS_KEY: 0.0, EPISODE_REWARD_KEY: 0.0})


# ---------------------------------------------------------------------------------------------
# Rows -> batch bookkeeping
# ---------------------------------------------------------------------------------------------


def _is_reward_of(row: Any) -> bool:
    if isinstance(row, Trace):
        return row.is_reward
    extra = row.get("extra_fields") if isinstance(row, dict) else getattr(row, "extra_fields", None)
    return bool((extra or {}).get(IS_REWARD_KEY, False))


def flatten_outputs(outputs: Sequence[Any]) -> tuple[list[Any], list[int], list[bool]]:
    """Flatten per-sample results (one row or a list of rows each) into rows plus bookkeeping.

    Returns ``(flat, parent_index, is_reward)``: ``parent_index[j]`` is the sample that row ``j``
    came from, and ``is_reward[j]`` reads the row's reward-trace tag. With one row per sample this
    is the identity, ``parent_index == range(n)``.
    """
    flat: list[Any] = []
    parent_index: list[int] = []
    is_reward: list[bool] = []
    for i, out in enumerate(outputs):
        for row in out if isinstance(out, list) else [out]:
            flat.append(row)
            parent_index.append(i)
            is_reward.append(_is_reward_of(row))
    return flat, parent_index, is_reward


def traces_per_instance_stats(parent_index: Sequence[int], n_instances: int) -> dict[str, float]:
    """Min / max / mean traces per episode for a step, plus the row and episode counts."""
    counts = Counter(int(p) for p in parent_index)
    per = [counts.get(i, 0) for i in range(max(int(n_instances), 0))]
    if not per:
        return {"multi_trace/traces_per_instance/min": 0.0, "multi_trace/traces_per_instance/max": 0.0,
                "multi_trace/traces_per_instance/mean": 0.0, "multi_trace/num_traces": 0.0,
                "multi_trace/num_instances": 0.0}
    return {"multi_trace/traces_per_instance/min": float(min(per)),
            "multi_trace/traces_per_instance/max": float(max(per)),
            "multi_trace/traces_per_instance/mean": sum(per) / len(per),
            "multi_trace/num_traces": float(len(parent_index)),
            "multi_trace/num_instances": float(len(per))}


def num_mini_batch(train_batch_size: int, ppo_mini_batch_size: int) -> int:
    """Optimizer steps per RL step, in prompt units, independent of how many rows the step has.

    With per-turn traces the row count varies step to step, so a fixed row-count mini-batch would
    change how many updates a step makes. Splitting the step's rows into
    ``train_batch_size // ppo_mini_batch_size`` mini-batches keeps the update count of the
    single-trace baseline.
    """
    if ppo_mini_batch_size <= 0:
        raise ValueError(f"ppo_mini_batch_size must be positive, got {ppo_mini_batch_size}")
    if train_batch_size % ppo_mini_batch_size:
        raise ValueError(f"train_batch_size ({train_batch_size}) must be divisible by "
                         f"ppo_mini_batch_size ({ppo_mini_batch_size})")
    return train_batch_size // ppo_mini_batch_size


def pad_size(n_rows: int, divisor: int) -> int:
    """How many padding clones bring ``n_rows`` to a multiple of ``divisor`` (dp size x mini-batches)."""
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return (-n_rows) % divisor


# ---------------------------------------------------------------------------------------------
# Advantages
# ---------------------------------------------------------------------------------------------


def _group_stats(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, std)`` with verl's conventions: a singleton group is ``(0, 1)``; std is unbiased."""
    if len(values) == 1:
        return 0.0, 1.0
    return statistics.fmean(values), statistics.stdev(values)


def instance_grpo_advantages(scores: Sequence[float], response_masks: Sequence[Sequence[int]],
                             uids: Sequence[Hashable], instance_ids: Sequence[Hashable],
                             is_reward: Sequence[bool], is_pad: Sequence[bool] | None = None, *,
                             norm_by_std: bool = True, epsilon: float = 1e-6) -> list[list[float]]:
    """Instance-level GRPO advantage, broadcast over each episode's traces.

    Args:
        scores: each row's scalar score (its summed token-level reward).
        response_masks: each row's response mask; the advantage is multiplied by it.
        uids: each row's GRPO group (the task instance it was sampled for).
        instance_ids: each row's episode; rows of one episode share it.
        is_reward: whether the row is its episode's reward trace.
        is_pad: whether the row is a padding clone (gets advantage 0).

    Returns:
        Per-row, per-token advantages. Rows whose episode has no reward trace, and pad rows, get 0.
    """
    n = len(scores)
    pad = list(is_pad) if is_pad is not None else [False] * n
    if not (len(response_masks) == len(uids) == len(instance_ids) == len(is_reward) == len(pad) == n):
        raise ValueError("all per-row inputs must have the same length")
    outcome: dict[Hashable, float] = {}
    group_of: dict[Hashable, Hashable] = {}
    for i in range(n):
        if pad[i] or not is_reward[i]:
            continue
        outcome[instance_ids[i]] = float(scores[i])
        group_of[instance_ids[i]] = uids[i]
    by_group: dict[Hashable, list[float]] = defaultdict(list)
    for iid, value in outcome.items():
        by_group[group_of[iid]].append(value)
    stats = {uid: _group_stats(values) for uid, values in by_group.items()}
    advantage: dict[Hashable, float] = {}
    for iid, value in outcome.items():
        mean, std = stats[group_of[iid]]
        advantage[iid] = (value - mean) / (std + epsilon) if norm_by_std else value - mean
    rows = []
    for i in range(n):
        a = 0.0 if pad[i] else advantage.get(instance_ids[i], 0.0)
        rows.append([a * float(m) for m in response_masks[i]])
    return rows


def grpo_advantages(scores: Sequence[float], response_masks: Sequence[Sequence[int]], uids: Sequence[Hashable], *,
                    norm_by_std: bool = True, epsilon: float = 1e-6) -> list[list[float]]:
    """Stock per-row GRPO advantage (every row is its own episode), for comparison and tests."""
    return instance_grpo_advantages(scores, response_masks, uids, list(range(len(scores))),
                                    [True] * len(scores), norm_by_std=norm_by_std, epsilon=epsilon)


__all__ = [
    "ASSISTANT_TURNS_KEY", "EPISODE_REWARD_KEY", "INSTANCE_ID_KEY", "IS_PAD_KEY", "IS_REWARD_KEY",
    "IS_REWARD_TRACE_KEY", "PromptOverflowError", "Trace", "TurnRecord", "failure_trace", "flatten_outputs",
    "grpo_advantages", "instance_grpo_advantages", "is_append_only", "last_trace_is_reward", "merged_trace",
    "num_mini_batch", "pad_size", "per_turn_traces", "traces_per_instance_stats",
]
