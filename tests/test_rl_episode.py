"""The verl-free episode core of the agent loop (harness_programs.rl.agent_loop.EpisodeRunner) on the toy domain.

A character-level fake tokenizer and the scripted toy policy stand in for the model server, so the
whole loop -- serving pipeline, environment, traces, dump -- runs without verl or a GPU.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

import harness_programs.rl
from harness_programs.domains import toy
from harness_programs.domains.toy_agents import DIGEST_PROGRAM, UNOPENED_RENDER, UNOPENED_TRIGGER, toy_policy
from harness_programs.rl.agent_loop import (
    EpisodeRunner,
    Generation,
    Limits,
    LoopSettings,
    configure_domain,
    make_rows,
    per_turn_config_errors,
    row_is_validation,
    task_from_row,
    task_to_row,
)
from harness_programs.rl.step_marker import read_step, rollout_meta, write_global_step
from harness_programs.rl.traces import is_append_only
from harness_programs.serving.program import Program, contract_sha, write_program
from harness_programs.serving.steer import Trigger

EOS = 0


def _text(messages) -> str:
    return "".join(f"<|{m['role']}|>{m.get('content') or ''}\n" for m in messages)


def _encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def _parse(text: str) -> list[dict]:
    """Invert the fake chat template, so the policy sees exactly the prompt it was sampled from."""
    parts = re.split(r"<\|(system|user|assistant)\|>", text)
    messages = [{"role": role, "content": body.removesuffix("\n")} for role, body in zip(parts[1::2], parts[2::2])]
    assert messages[-1] == {"role": "assistant", "content": ""}
    return messages[:-1]


class FakeModel:
    """Char-level chat template plus the scripted toy policy behind a token-in/token-out API."""

    def __init__(self) -> None:
        self.prompts: list[list[int]] = []
        self.params: list[dict] = []

    async def render(self, messages) -> list[int]:
        return _encode(_text(messages) + "<|assistant|>")

    async def render_observations(self, messages) -> list[int]:
        return _encode(_text(messages) + "<|assistant|>")

    async def generate(self, prompt_ids, params) -> Generation:
        self.prompts.append(list(prompt_ids))
        self.params.append(dict(params))
        reply = toy_policy(_parse(self.decode(prompt_ids)))["content"]
        ids = _encode(reply)[: params["max_tokens"]]
        return Generation(token_ids=ids, log_probs=[-0.5] * len(ids), num_preempted=0)

    @staticmethod
    def decode(ids) -> str:
        return "".join(chr(i) for i in ids)

    @staticmethod
    def count(messages) -> int:
        return len(_text(messages)) + len("<|assistant|>")


def _limits(**overrides) -> Limits:
    values = dict(prompt_length=16384, response_length=4096, model_window=20480, per_turn_max_tokens=512,
                  max_turns=8, want_logprobs=True)
    values.update(overrides)
    return Limits(**values)


def _run(runner: EpisodeRunner, task, model: FakeModel, **kwargs):
    return asyncio.run(runner.run(task, generate=model.generate, render=model.render, decode=model.decode,
                                  render_observations=model.render_observations,
                                  sampling_params={"temperature": 1.0, "max_tokens": 99999}, **kwargs))


@pytest.fixture
def task():
    return toy.make_tasks(1, n_files=4, seed=3)[0]


# -- settings ------------------------------------------------------------------------------------


def test_package_imports_without_verl():
    assert harness_programs.rl.LoopSettings is LoopSettings
    with pytest.raises(ImportError):
        harness_programs.rl.HarnessAgentLoop()


def test_settings_merge_nested_budget_and_trainer_override():
    settings = LoopSettings.from_sources(
        {"domain": "toy", "max_turns": 6, "context_budget": {"enable": False}},
        {"trace_mode": "per_turn", "context_budget": {"enable": True, "history_budget": "1024"}, "run_dir": "out"})
    assert settings.budget_enable and settings.history_budget == 1024 and settings.max_turns == 6
    assert settings.per_turn and settings.run_dir == "out"
    null = LoopSettings.from_sources({"domain": "toy", "context_budget": {"enable": True, "history_budget": "null"}})
    assert null.history_budget is None and null.per_turn  # auto -> per-turn under the budget
    assert not LoopSettings(domain="toy").per_turn


@pytest.mark.parametrize("kwargs", [
    {"history_budget": 512},
    {"program_path": "p.json"},
    {"budget_enable": True, "trace_mode": "merged"},
    {"budget_enable": True, "program_dynamic": True},
    {"trace_mode": "nope"},
    {"domain": ""},
])
def test_settings_reject_contradictions(kwargs):
    with pytest.raises(ValueError):
        LoopSettings(**{"domain": "toy", **kwargs})


def test_settings_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        LoopSettings.from_sources({"domain": "toy", "typo_key": 1})


def test_limits_from_rollout_config_and_serving_limit():
    settings = LoopSettings(domain="toy", per_turn_max_tokens=512)
    domain = configure_domain(settings)
    rollout = {"prompt_length": 4096, "response_length": 2048, "max_model_len": 16384, "calculate_log_probs": True,
               "multi_turn": {"max_assistant_turns": 6, "max_parallel_calls": 1}}
    limits = Limits.from_rollout_config(rollout, settings, domain)
    assert (limits.max_turns, limits.per_turn_max_tokens, limits.max_parallel_calls) == (6, 512, 1)
    assert limits.serving_limit == min(4096, 16384 - 512)
    fallback = Limits.from_rollout_config({"prompt_length": 100, "response_length": 50},
                                          LoopSettings(domain="toy"), domain)
    assert fallback.max_turns == toy.MAX_TURNS and fallback.model_window == 150
    assert fallback.per_turn_max_tokens == 50
    with pytest.raises(ValueError):
        Limits(prompt_length=10, response_length=5, model_window=20, per_turn_max_tokens=6, max_turns=1)


def test_per_turn_config_guard():
    def cfg(**over):
        actor = {"ppo_mini_batch_size": 8, "use_dynamic_bsz": True, "loss_agg_mode": "token-mean",
                 "ppo_micro_batch_size_per_gpu": 1}
        actor.update(over.pop("actor", {}))
        return {"algorithm": {"adv_estimator": over.pop("estimator", "grpo")},
                "data": {"train_batch_size": over.pop("train_batch_size", 8)}, "actor_rollout_ref": {"actor": actor}}

    assert per_turn_config_errors(cfg()) == []
    assert per_turn_config_errors(cfg(actor={"use_dynamic_bsz": False})) == []  # micro-batch 1 is fine
    problems = per_turn_config_errors(cfg(estimator="remax", train_batch_size=12,
                                          actor={"use_dynamic_bsz": False, "ppo_micro_batch_size_per_gpu": 2,
                                                 "loss_agg_mode": "seq-mean-token-sum"}))
    assert [p.split()[0] for p in problems] == ["algorithm.adv_estimator", "set", "actor.loss_agg_mode",
                                                "data.train_batch_size"]


def test_domain_options_and_turn_cap_reach_the_domain():
    settings = LoopSettings(domain="toy", domain_options={"max_turns": 3})
    assert configure_domain(settings).max_turns == 3
    assert configure_domain(LoopSettings(domain="toy"), max_turns=5).max_turns == 5
    with pytest.raises(ValueError):
        configure_domain(LoopSettings(domain="toy", domain_options={"nope": 1}))
    hash(settings)  # settings key the per-process runner cache


# -- dataset rows --------------------------------------------------------------------------------


def test_row_round_trip(task):
    row = task_to_row(task, domain="toy", index=4, split="val", agent_name="harness_toy")
    assert row["agent_name"] == "harness_toy" and row["data_source"] == "toy"
    assert json.loads(json.dumps(row)) == row  # parquet/JSON safe
    rebuilt = task_from_row(row)
    assert (rebuilt.task_id, rebuilt.prompt, rebuilt.data) == (task.task_id, task.prompt, task.data)
    assert not row_is_validation(row)  # explicit validate flag wins over the split name
    assert row_is_validation({"extra_info": {"split": "val"}})
    by_id = task_from_row({"extra_info": {"task_id": task.task_id, "split": "train"}},
                          lookup=lambda tid, split: task)
    assert by_id is task
    with pytest.raises(ValueError):
        task_from_row({"extra_info": {"task_id": "x"}})


def test_make_rows_marks_validation_by_split():
    train = make_rows("toy", "synthetic", split="train", limit=3, agent_name="harness_toy")
    val = make_rows("toy", "synthetic", split="val", limit=2)
    assert len(train) == 3 and len(val) == 2
    assert [r["extra_info"]["index"] for r in train] == [0, 1, 2]
    assert not any(row_is_validation(r) for r in train) and all(row_is_validation(r) for r in val)
    assert all(r["agent_name"] == "harness_toy" for r in train) and "agent_name" not in val[0]
    assert task_from_row(train[1]).task_id == train[1]["extra_info"]["task_id"]


# -- episodes ------------------------------------------------------------------------------------


def test_per_turn_episode_with_budget_trains_on_served_prompts(tmp_path, task):
    settings = LoopSettings(domain="toy", trace_mode="per_turn", budget_enable=True, history_budget=300,
                            run_dir=str(tmp_path))
    limits = _limits()
    runner = EpisodeRunner(configure_domain(settings, max_turns=limits.max_turns), settings, limits,
                           token_counter=FakeModel.count)
    model = FakeModel()
    write_global_step(tmp_path, 12)
    outcome = _run(runner, task, model, rollout=rollout_meta(step=12, group_key=task.task_id, replicate=0,
                                                             validate=False))
    assert not outcome.failed and outcome.error is None
    turns = outcome.turns
    assert len(turns) == outcome.record.num_steps >= 3
    assert [t.prompt_ids for t in turns] == model.prompts
    assert not is_append_only(turns)  # the budget rebuilt at least one prompt
    assert all(p["max_tokens"] == limits.per_turn_max_tokens for p in model.params)
    assert all(p["temperature"] == 1.0 for p in model.params)
    full_prompt_tokens = FakeModel.count(outcome.record.messages[:-1])
    assert max(len(p) for p in model.prompts) < full_prompt_tokens

    traces = runner.traces(outcome, eos_token_id=EOS)
    assert len(traces) == len(turns)
    assert [t.prompt_ids for t in traces] == model.prompts
    assert all(t.response_mask == [1] * len(t.response_ids) for t in traces)
    assert [t.reward_score for t in traces] == [0.0] * (len(traces) - 1) + [outcome.record.reward]

    dumped = read_step(tmp_path, 12)
    assert len(dumped) == 1
    rec = dumped[0]
    assert rec.task_id == task.task_id and rec.harness == "toy" and rec.context_budget == 300
    assert rec.rollout == {"step": 12, "group_key": task.task_id, "replicate": 0, "validate": False}
    assert len(rec.served) == len(turns) and {"tokens_after", "evicted", "fired"} <= set(rec.served[0])
    assert any(s["evicted"] for s in rec.served)
    assert rec.messages[0]["role"] == "system"  # the full transcript, never the served view
    assert rec.status in ("answered", "max_turns")


def test_merged_episode_without_budget_is_one_append_only_row(task):
    settings = LoopSettings(domain="toy", trace_mode="merged")
    limits = _limits()
    runner = EpisodeRunner(configure_domain(settings), settings, limits)
    model = FakeModel()
    outcome = _run(runner, task, model)
    assert outcome.record.status == "answered" and outcome.record.reward == 1.0
    assert is_append_only(outcome.turns)
    (trace,) = runner.traces(outcome, eos_token_id=EOS)
    assert trace.prompt_ids == model.prompts[0]
    assert 0 in trace.response_mask and trace.reward_score == 1.0
    # Every later prompt is a prefix of the merged row: what the policy saw is what is trained.
    full = trace.prompt_ids + trace.response_ids
    for prompt in model.prompts:
        assert full[:len(prompt)] == prompt
    used = sum(len(t.response_ids) + len(t.observation_ids) for t in outcome.turns[:-1])
    assert len(trace.response_ids) == used + len(outcome.turns[-1].response_ids)


def test_installed_program_and_trigger_are_served(tmp_path, task):
    trigger = Trigger(name="unopened", code=UNOPENED_TRIGGER, guidance="Open a file you have not opened.",
                      render=UNOPENED_RENDER)
    path = tmp_path / "program.json"
    write_program(path, Program(harness="toy", compression={"name": "digest", "code": DIGEST_PROGRAM},
                                triggers=[trigger], contract_sha=contract_sha(toy.CONTRACT)))
    settings = LoopSettings(domain="toy", budget_enable=True, history_budget=None, program_path=str(path))
    runner = EpisodeRunner(configure_domain(settings), settings, _limits(), token_counter=FakeModel.count)
    outcome = _run(runner, task, FakeModel())
    assert outcome.record.triggers["setup"]["installed"][0]["name"] == "unopened"
    assert any("unopened" in fired for fired in outcome.record.triggers["firings"])


def test_dynamic_program_is_picked_up_by_the_next_episode(tmp_path, task):
    path = tmp_path / "program.json"
    settings = LoopSettings(domain="toy", budget_enable=True, program_path=str(path), program_dynamic=True)
    runner = EpisodeRunner(configure_domain(settings), settings, _limits(), token_counter=FakeModel.count)
    before = _run(runner, task, FakeModel())
    assert before.record.triggers["setup"]["installed"] == []
    trigger = Trigger(name="nudge", code="def check(ep):\n    return True\n", guidance="Be quick.")
    write_program(path, Program(harness="toy", triggers=[trigger]))
    after = _run(runner, task, FakeModel())
    assert [c["name"] for c in after.record.triggers["setup"]["installed"]] == ["nudge"]


def test_missing_static_program_fails_at_construction(tmp_path):
    settings = LoopSettings(domain="toy", budget_enable=True, program_path=str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        EpisodeRunner(configure_domain(settings), settings, _limits(), token_counter=FakeModel.count)


def test_prompt_overflow_becomes_a_dumped_error_and_a_zero_gradient_row(tmp_path, task):
    settings = LoopSettings(domain="toy", trace_mode="per_turn", run_dir=str(tmp_path))
    limits = _limits(prompt_length=600, response_length=512, model_window=4096)
    runner = EpisodeRunner(configure_domain(settings), settings, limits)
    outcome = _run(runner, task, FakeModel(), rollout=rollout_meta(step=1, group_key=task.task_id, replicate=0,
                                                                   validate=False))
    assert outcome.failed and outcome.record.status == "error" and "prompt_length" in outcome.error
    (trace,) = runner.traces(outcome, eos_token_id=EOS)
    assert trace.response_mask == [0] and trace.reward_score == 0.0
    assert len(trace.prompt_ids) <= limits.prompt_length
    raw = json.loads((tmp_path / "rollouts" / "step_1.jsonl").read_text().splitlines()[0])
    assert raw["status"] == "error" and "prompt_length" in raw["error"]
    assert read_step(tmp_path, 1)[0].status == "error"


def test_environment_exception_is_contained(task):
    settings = LoopSettings(domain="toy", trace_mode="per_turn")
    runner = EpisodeRunner(configure_domain(settings), settings, _limits())

    async def broken(prompt_ids, params):
        raise RuntimeError("server went away")

    model = FakeModel()
    outcome = asyncio.run(runner.run(task, generate=broken, render=model.render, decode=model.decode))
    assert outcome.failed and "server went away" in outcome.error
    (trace,) = runner.traces(outcome, eos_token_id=EOS, fallback_prompt_ids=[5, 6, 7])
    assert trace.prompt_ids == [5, 6, 7] and trace.response_mask == [0]


def test_turn_cap_ends_the_episode(task):
    settings = LoopSettings(domain="toy", trace_mode="per_turn")
    limits = _limits(max_turns=2)
    runner = EpisodeRunner(configure_domain(settings, max_turns=2), settings, limits)
    outcome = _run(runner, task, FakeModel())
    assert outcome.record.status == "max_turns" and len(outcome.turns) == 2
    assert outcome.record.max_steps == 2
    assert len(runner.traces(outcome, eos_token_id=EOS)) == 2
