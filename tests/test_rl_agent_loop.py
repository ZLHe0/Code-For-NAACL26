"""The verl adapter (``HarnessAgentLoop``) against a minimal stand-in for verl's agent-loop module.

verl is not a test dependency. The stand-in reproduces only the surface the adapter touches --
``AgentLoopBase.__init__`` attributes, ``AgentLoopOutput``, ``AgentLoopMetrics`` and the
token-in/token-out ``server_manager.generate`` -- so the adapter's wiring is checked end to end.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from harness_programs.domains import toy
from harness_programs.domains.toy_agents import toy_policy
from harness_programs.rl.agent_loop import task_to_row
from harness_programs.rl.step_marker import read_step, write_global_step

AGENT_LOOP_FILE = Path(__file__).resolve().parents[1] / "harness_programs" / "rl" / "agent_loop.py"


class Node(dict):
    """A dict with attribute access, standing in for an OmegaConf node."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _node(value: Any) -> Any:
    return Node({k: _node(v) for k, v in value.items()}) if isinstance(value, dict) else value


@dataclass
class FakeMetrics:
    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    num_preempted: int = -1


FakeMetrics.model_fields = {"generate_sequences": None, "tool_calls": None, "num_preempted": None}


@dataclass
class FakeOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    metrics: FakeMetrics
    response_logprobs: list[float] | None = None
    reward_score: float | None = None
    num_turns: int = 0
    extra_fields: dict[str, Any] = field(default_factory=dict)


class FakeAgentLoopBase:
    def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.apply_chat_template_kwargs = data_config.config.get("apply_chat_template_kwargs", {})
        self.system_prompt = []


class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True, **kwargs):
        assert kwargs == {"enable_thinking": False}
        text = "".join(f"<|{m['role']}|>{m.get('content') or ''}\n" for m in messages)
        return [ord(c) for c in text + ("<|assistant|>" if add_generation_prompt else "")]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


@dataclass
class TokenOutput:
    token_ids: list[int]
    log_probs: list[float] | None
    num_preempted: int | None = None


class FakeServer:
    def __init__(self) -> None:
        self.request_ids: set[str] = set()
        self.calls = 0

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        self.request_ids.add(request_id)
        self.calls += 1
        parts = re.split(r"<\|(system|user|assistant)\|>", "".join(chr(i) for i in prompt_ids))
        messages = [{"role": r, "content": b.removesuffix("\n")} for r, b in zip(parts[1::2], parts[2::2])][:-1]
        ids = [ord(c) for c in toy_policy(messages)["content"]][: sampling_params["max_tokens"]]
        return TokenOutput(token_ids=ids, log_probs=[-1.0] * len(ids), num_preempted=0)


def _load_adapter(monkeypatch, *, multi_output: bool):
    module = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    module.AgentLoopBase = FakeAgentLoopBase
    module.AgentLoopMetrics = FakeMetrics
    module.AgentLoopOutput = FakeOutput
    if multi_output:
        module.SUPPORTS_MULTI_OUTPUT = True
    package = types.ModuleType("verl.experimental.agent_loop")
    package.agent_loop = module
    for name, mod in {"verl": types.ModuleType("verl"), "verl.experimental": types.ModuleType("verl.experimental"),
                      "verl.experimental.agent_loop": package,
                      "verl.experimental.agent_loop.agent_loop": module}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("harness_programs.rl._agent_loop_under_fake_verl", AGENT_LOOP_FILE)
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, adapter)
    spec.loader.exec_module(adapter)
    assert adapter._VERL_IMPORT_ERROR is None
    return adapter


def _make(adapter, *, section: dict, server: FakeServer, yaml_kwargs: dict | None = None,
          estimator: str = "grpo"):
    config = _node({
        "algorithm": {"adv_estimator": estimator},
        "data": {"train_batch_size": 4},
        "actor_rollout_ref": {
            "actor": {"ppo_mini_batch_size": 4, "use_dynamic_bsz": True, "loss_agg_mode": "token-mean"},
            "rollout": {
                "prompt_length": 16384, "response_length": 4096, "max_model_len": 20480, "calculate_log_probs": True,
                "multi_turn": {"max_assistant_turns": 8, "max_parallel_calls": 1}}},
        "harness_programs": section,
    })
    return adapter.HarnessAgentLoop(
        trainer_config=Node(config=config), server_manager=server, tokenizer=CharTokenizer(), processor=None,
        dataset_cls=None, data_config=Node(config=_node({"apply_chat_template_kwargs": {"enable_thinking": False}})),
        name="harness_toy", **(yaml_kwargs or {"domain": "toy", "per_turn_max_tokens": 512}))


def _row(task, *, validate=False):
    row = task_to_row(task, domain="toy", index=0, validate=validate, agent_name="harness_toy")
    return {**row, "raw_prompt": row["prompt"], "index": 0, "uid": "u0"}


def test_per_turn_loop_returns_one_output_per_turn_and_dumps(monkeypatch, tmp_path):
    adapter = _load_adapter(monkeypatch, multi_output=True)
    server = FakeServer()
    section = {"trace_mode": "per_turn", "context_budget": {"enable": True, "history_budget": 300},
               "run_dir": str(tmp_path)}
    loop = _make(adapter, section=section, server=server)
    write_global_step(tmp_path, 3)
    task = toy.make_tasks(1, n_files=4, seed=3)[0]
    outputs = asyncio.run(loop.run({"temperature": 1.0, "top_p": 1.0}, **_row(task)))
    assert isinstance(outputs, list) and len(outputs) == server.calls >= 2
    assert len(server.request_ids) == 1  # one sticky request id per episode
    assert all(o.response_mask == [1] * len(o.response_ids) for o in outputs)
    assert [o.extra_fields["is_reward"] for o in outputs] == [False] * (len(outputs) - 1) + [True]
    assert all(o.reward_score is not None for o in outputs)  # verl then skips its reward function
    assert outputs[-1].metrics.num_preempted == 0
    (record,) = read_step(tmp_path, 3)
    assert record.rollout == {"step": 3, "group_key": task.task_id, "replicate": 0, "validate": False}
    assert outputs[-1].reward_score == record.reward


def test_merged_loop_returns_a_single_output(monkeypatch):
    adapter = _load_adapter(monkeypatch, multi_output=False)
    loop = _make(adapter, section={"trace_mode": "merged"}, server=FakeServer())
    task = toy.make_tasks(1, n_files=3, seed=1)[0]
    output = asyncio.run(loop.run({"temperature": 1.0}, **_row(task)))
    assert isinstance(output, FakeOutput)
    assert 0 in output.response_mask and output.reward_score == 1.0


def test_per_turn_without_the_patch_is_refused(monkeypatch):
    adapter = _load_adapter(monkeypatch, multi_output=False)
    with pytest.raises(RuntimeError, match="multi-output"):
        _make(adapter, section={"context_budget": {"enable": True, "history_budget": 1024}}, server=FakeServer())


def test_bad_row_degrades_to_a_placeholder(monkeypatch):
    adapter = _load_adapter(monkeypatch, multi_output=True)
    loop = _make(adapter, section={"trace_mode": "per_turn"}, server=FakeServer())
    row = {"raw_prompt": [{"role": "user", "content": "hi"}], "extra_info": {"split": "train"}}
    (output,) = asyncio.run(loop.run({"temperature": 1.0}, **row))
    assert output.response_mask == [0] and output.reward_score == 0.0
    assert output.prompt_ids  # rendered from raw_prompt


def test_per_turn_refuses_a_non_grpo_trainer(monkeypatch):
    adapter = _load_adapter(monkeypatch, multi_output=True)
    with pytest.raises(ValueError, match="adv_estimator"):
        _make(adapter, section={"trace_mode": "per_turn"}, server=FakeServer(), estimator="gae")
    _make(adapter, section={"trace_mode": "merged"}, server=FakeServer(), estimator="gae")  # merged: stock verl
