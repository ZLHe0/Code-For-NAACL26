"""The serving core: sandbox, compression, budget, steering, program document, pipeline."""

from __future__ import annotations

import os
import time

import pytest

from harness_programs.domains import toy
from harness_programs.domains.toy_agents import DIGEST_PROGRAM
from harness_programs.serving import sandbox
from harness_programs.serving.budget import TRUNCATION_STUB, BudgetedContext, keep_recent_turns
from harness_programs.serving.compress import (CompressionError, HeadRewritten, MeasuredStage1, apply_program,
                                        build_stage1, compile_program, get_state_dsl, make_stage1)
from harness_programs.serving.program import (Program, ProgramError, check_contract, contract_sha, load_program,
                                       program_from_dict, write_program)
from harness_programs.serving.serve import ContextPipeline
from harness_programs.serving.steer import (GUIDANCE_BLOCK, Trigger, TriggerError, TriggerRuntime, build_runtime,
                                     episode_from_messages)
from harness_programs.serving.tokens import char_counter

COUNT = char_counter()


def long_episode(n_files: int = 5, body_chars: int = 600) -> list[dict]:
    msgs = toy.sample_messages()[:2]
    msgs += [{"role": "assistant", "content": "List.\n<action>list</action>"},
             {"role": "user", "content": "<observation>" + "\n".join(f"file_{i}.txt" for i in range(n_files))
              + "</observation>"}]
    for i in range(n_files):
        msgs += [{"role": "assistant", "content": f"Open {i}.\n<action>open file_{i}.txt</action>"},
                 {"role": "user", "content": f"<observation>{'x' * body_chars}</observation>"}]
    return msgs


# -- sandbox ---------------------------------------------------------------------------------


@pytest.mark.parametrize("code, reason", [
    ("import os\ndef f(x):\n    return x", "import of 'os'"),
    ("def f(x):\n    return open('a')", "'open'"),
    ("def f(x):\n    return x.__class__", "dunder"),
    ("def g(x):\n    return x", "no top-level"),
    ("async def f(x):\n    return x", "no top-level"),
])
def test_screen_rejects(code, reason):
    errors = sandbox.screen_source(code, sandbox.CodePolicy(entrypoint="f"))
    assert any(reason in e for e in errors)


def test_lenient_policy_allows_getattr_but_not_dunders():
    policy = sandbox.lenient_policy("f")
    fn = sandbox.compile_entrypoint("def f(x):\n    return getattr(x, 'real', 0)", policy)
    assert fn(3) == 3
    bad = sandbox.compile_entrypoint("def f(x):\n    return getattr(x, '__class__')", policy)
    with pytest.raises(ValueError):
        bad(3)


def test_call_with_deadline_interrupts_a_loop():
    def spin():
        while True:
            pass
    with pytest.raises(sandbox.CodeTimeout):
        sandbox.call_with_deadline(spin, seconds=0.2)


def test_code_cache_poisons_after_repeated_strikes():
    cache = sandbox.CodeCache(sandbox.CodePolicy(entrypoint="f"), poison_limit=2)
    code = "def f(x):\n    return x"
    assert cache.compiled(code) is not None
    assert not cache.strike(code) and cache.strike(code)
    assert cache.compiled(code) is None and cache.is_poisoned(code)
    assert cache.compiled("def f(x) return") is None


# -- compression -----------------------------------------------------------------------------


def test_digest_program_shrinks_old_turns_and_keeps_the_live_one():
    msgs = long_episode()
    out = apply_program(compile_program(DIGEST_PROGRAM), msgs, get_state_dsl("toy"))
    assert len(out) == len(msgs) and COUNT(out) < COUNT(msgs)
    assert out[-1] == msgs[-1] and out[:2] == msgs[:2]
    assert [m for m in out if m["role"] == "assistant"] == [m for m in msgs if m["role"] == "assistant"]


def test_a_turn_left_out_of_the_state_is_served_unchanged():
    dsl = get_state_dsl("toy")
    msgs = long_episode()
    program = ("def compress(state):\n    for t in state[:-1]:\n        t.digest = 'x'\n"
               "    return state[1:]")
    out = apply_program(compile_program(program), msgs, dsl)
    assert len(out) == len(msgs) and out[3] == msgs[3] and out[5]["content"] == "<observation>x</observation>"


def test_program_that_raises_degrades_to_pass_through():
    stage1 = make_stage1("def compress(state):\n    return state[0].nope", get_state_dsl("toy"),
                         cache=sandbox.CodeCache(sandbox.lenient_policy("compress")))
    msgs = long_episode()
    assert stage1(msgs) == msgs


def test_measured_stage1_counts_raises():
    stage1 = MeasuredStage1("toy", "def compress(state):\n    if len(state) > 3:\n        return 1\n    return state")
    stage1(long_episode())
    assert stage1.raises == 1


def test_build_stage1_fails_at_load_on_a_bad_program():
    with pytest.raises(CompressionError):
        build_stage1("toy", "def compress(state):\n    return None")


def test_head_rewrite_is_a_hard_error():
    dsl = get_state_dsl("toy")

    def rewrite(messages):
        out = [dict(m) for m in messages]
        out[1]["content"] = "something else"
        return out

    from harness_programs.serving import compress
    original = compress.reserialize
    compress.reserialize = lambda messages, turns, d: rewrite(original(messages, turns, d))
    try:
        with pytest.raises(HeadRewritten):
            apply_program(compile_program("def compress(state):\n    return state"), long_episode(), dsl)
    finally:
        compress.reserialize = original


# -- budget ----------------------------------------------------------------------------------


def test_keep_recent_turns():
    msgs = long_episode()
    assert keep_recent_turns(msgs, 99) == msgs
    kept = keep_recent_turns(msgs, 2)
    assert kept[2]["content"] == TRUNCATION_STUB and kept[3:] == msgs[-4:]


def test_budget_evicts_oldest_turns_and_compression_buys_them_back():
    msgs = long_episode()
    naive = BudgetedContext(None, token_counter=COUNT, serving_limit=8192, history_budget=400)
    _, naive_stats = naive(msgs)
    digest = BudgetedContext(build_stage1("toy", DIGEST_PROGRAM), token_counter=COUNT, serving_limit=8192,
                             history_budget=400)
    _, digest_stats = digest(msgs)
    assert naive_stats.evicted and naive_stats.turns_kept < naive_stats.turns_total
    assert digest_stats.turns_kept > naive_stats.turns_kept
    assert digest_stats.raw_tokens > digest_stats.tokens_before


def test_budget_reserves_room_for_the_guidance_tail():
    msgs = long_episode()
    ctx = BudgetedContext(None, token_counter=COUNT, serving_limit=8192, history_budget=400)
    _, without = ctx(msgs)
    ctx = BudgetedContext(None, token_counter=COUNT, serving_limit=8192, history_budget=400)
    _, with_tail = ctx(msgs, tail=[{"role": "user", "content": "y" * 400}])
    assert with_tail.turns_kept < without.turns_kept


def test_budget_refuses_an_undeliverable_history_budget():
    with pytest.raises(ValueError):
        BudgetedContext(None, token_counter=COUNT, serving_limit=50, history_budget=400)(long_episode())


# -- steering --------------------------------------------------------------------------------

REOPEN = Trigger(name="reopen", guidance="You already opened this file; open a different one.",
                 code="def check(ep):\n    acts = ep.actions\n    return len(acts) != len(set(acts))\n")


def reopening_episode():
    msgs = long_episode(n_files=2)
    return msgs + [{"role": "assistant", "content": "Again.\n<action>open file_0.txt</action>"},
                   {"role": "user", "content": "<observation>x</observation>"}]


def test_episode_from_messages_parses_actions_and_reasoning():
    ep = episode_from_messages(long_episode(n_files=2))
    assert ep.actions == ["list", "open file_0.txt", "open file_1.txt"]
    assert ep.reasonings[0] == "List." and ep.task_text.startswith("Which file")


def test_trigger_fires_only_when_its_trigger_holds():
    runtime = build_runtime([REOPEN])
    assert runtime.fire(long_episode(n_files=2)) == []
    tail, fired = runtime.tail(reopening_episode())
    assert fired == ["reopen"] and tail[0]["content"] == GUIDANCE_BLOCK.format(text=REOPEN.guidance)


def test_trigger_reading_gold_fails_at_load():
    peek = Trigger(name="peek", guidance="p", code="def check(ep):\n    return ep.gold['file'] == 'a'\n")
    with pytest.raises(TriggerError):
        build_runtime([peek])


def test_duplicate_trigger_names_fail_at_load():
    with pytest.raises(TriggerError):
        build_runtime([REOPEN, REOPEN])


def test_dose_cap_per_trigger():
    always = Trigger(name="always", guidance="p", code="def check(ep):\n    return ep.n_steps > 0\n")
    capped = TriggerRuntime([always], max_per_trigger=1)
    assert capped.fire(long_episode(n_files=0)[:3]) != []
    assert capped.fire(long_episode(n_files=2)) == []


def test_render_writes_the_guidance_from_the_episode():
    trigger = Trigger(name="r", guidance="static", code="def check(ep):\n    return True\n",
                      render="def render(ep):\n    return 'step ' + str(ep.n_steps)\n")
    assert build_runtime([trigger]).fire(long_episode(n_files=1)) == [("r", "step 2")]


# -- program document ------------------------------------------------------------------------


def test_program_round_trip_never_serves_the_fault(tmp_path):
    trigger = Trigger(name="c", code=REOPEN.code, guidance="p", fault="def fault(ep):\n    return True\n")
    program = Program(harness="toy", compression={"name": "d", "code": DIGEST_PROGRAM}, triggers=[trigger],
                      contract_sha=contract_sha(toy.CONTRACT))
    write_program(tmp_path / "p.json", program)
    loaded = load_program(tmp_path / "p.json")
    assert loaded.compression == program.compression and loaded.triggers[0].fault == ""
    assert "fault" not in (tmp_path / "p.json").read_text()


@pytest.mark.parametrize("doc", [{}, {"harness": ""}, {"harness": "toy", "compression": "x"},
                                 {"harness": "toy", "triggers": [{"name": "c"}]}])
def test_malformed_programs_are_refused(doc):
    with pytest.raises(ProgramError):
        program_from_dict(doc)


def test_contract_mismatch_is_refused():
    program = Program(harness="toy", compression={"name": "d", "code": DIGEST_PROGRAM}, contract_sha="0" * 16)
    with pytest.raises(ProgramError):
        check_contract(program, toy.CONTRACT)


# -- pipeline --------------------------------------------------------------------------------


def test_pipeline_serves_compression_and_guidance_and_hot_reloads(tmp_path):
    path = tmp_path / "program.json"
    write_program(path, Program(harness="toy", compression={"name": "d", "code": DIGEST_PROGRAM},
                                contract_sha=contract_sha(toy.CONTRACT)))
    pipeline = ContextPipeline("toy", program_path=path, token_counter=COUNT, serving_limit=8192,
                               history_budget=400, max_steps=8)
    first = pipeline.new_episode().serve(reopening_episode())
    assert first.fired == [] and first.stats.stage1_saved > 0

    write_program(path, Program(harness="toy", compression={"name": "d", "code": DIGEST_PROGRAM},
                                triggers=[REOPEN], contract_sha=contract_sha(toy.CONTRACT)))
    os.utime(path, (time.time() + 5, time.time() + 5))
    episode = pipeline.new_episode()
    turn = episode.serve(reopening_episode())
    assert turn.fired == ["reopen"] and turn.messages[-1]["content"].endswith(REOPEN.guidance)
    assert episode.setup["compression"] == "d" and episode.setup["installed"][0]["name"] == "reopen"


def test_pipeline_refuses_a_program_for_another_domain(tmp_path):
    path = tmp_path / "program.json"
    write_program(path, Program(harness="other"))
    with pytest.raises(ProgramError):
        ContextPipeline("toy", program_path=path, token_counter=COUNT, serving_limit=8192,
                        history_budget=None).new_episode()
