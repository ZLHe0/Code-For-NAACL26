"""Replay: every trigger over every stored rollout, per-step series, and the evidence E."""

from __future__ import annotations

import pytest
from test_steering_support import (
    ALWAYS_CHECK,
    NEVER_CHECK,
    REOPEN_CHECK,
    REOPEN_FAULT,
    REOPEN_GUIDANCE,
    scratch,  # noqa: F401
    toy_records,
    write_rollouts,
)

from harness_programs.serving.steer import Trigger
from harness_programs.steering.replay import (
    Evidence,
    ReplayError,
    prefix_episodes,
    replay_evidence,
    replay_many,
    turn_prefixes,
)
from harness_programs.steering.reservoir import Placed, load_records
from harness_programs.steering.screen import behaviour_screen

REOPEN = Trigger("reopen", REOPEN_CHECK, REOPEN_GUIDANCE, fault=REOPEN_FAULT)


def _placed(records):
    return [Placed(record=r, step=int(r.rollout["step"]), group_key=f"step{r.rollout['step']}-{r.task_id}")
            for r in records]


@pytest.fixture(scope="module")
def placed():
    # Step 10: half the replicates loop on one file; step 20: a quarter do.
    early = toy_records(steps=(10,), mix=("solve", "loop", "loop", "solve"))
    late = toy_records(steps=(20,), mix=("solve", "loop", "solve", "solve"))
    return _placed(early + late)


def test_turn_prefixes_end_before_each_assistant_turn():
    record = toy_records(n_tasks=1, mix=("solve",))[0]
    prefixes = turn_prefixes(record.messages)
    assert len(prefixes) == record.num_steps
    assert all(p[-1]["role"] != "assistant" for p in prefixes)


def test_series_per_step(placed):
    out = replay_many(placed, [REOPEN])
    assert not out.failures
    reading = out.readings["reopen"]
    series = reading.series()
    assert [p.step for p in series] == [10, 20]
    assert [p.episodes for p in series] == [16, 16]
    assert [p.fault_rate for p in series] == [0.5, 0.25]
    # A looping episode re-opens from its third turn on; the trigger sees the prefix only.
    loop = next(e for e in reading.episodes if e.fault)
    assert loop.fired_turns[0] == 3 and loop.turns == 8
    assert all(e.fires == 0 for e in reading.episodes if not e.fault)
    assert behaviour_screen(reading) == []


def test_never_and_always_fire_are_refused(placed):
    out = replay_many(placed, [Trigger("never", NEVER_CHECK, REOPEN_GUIDANCE),
                               Trigger("always", ALWAYS_CHECK, REOPEN_GUIDANCE, fault=REOPEN_FAULT)])
    assert any("fires on 0 of" in c for c in behaviour_screen(out.readings["never"]))
    assert any("fault holds on 0 of" in c for c in behaviour_screen(out.readings["never"]))
    assert any("fires on every one of" in c for c in behaviour_screen(out.readings["always"]))


def test_fault_that_never_holds_is_refused(placed):
    fault = 'def fault(ep):\n    return any(a.startswith("delete ") for a in ep.actions)\n'
    out = replay_many(placed, [Trigger("quiet", REOPEN_CHECK, REOPEN_GUIDANCE, fault=fault)])
    reasons = behaviour_screen(out.readings["quiet"])
    assert len(reasons) == 1 and "fault holds on 0 of" in reasons[0]


def test_a_trigger_that_raises_is_dropped_whole(placed):
    # Passes the load smoke (0 and 2 steps) and divides by zero on a 3-step prefix.
    check = "def check(ep):\n    return 1 / (len(ep.actions) - 3) > 0\n"
    out = replay_many(placed, [REOPEN, Trigger("fragile", check, REOPEN_GUIDANCE)])
    assert "fragile" not in out.readings and "ZeroDivisionError" in out.failures["fragile"]
    assert "reopen" in out.readings and len(out.readings["reopen"].episodes) == 32


def test_load_failures_are_refused_not_raised(placed):
    out = replay_many(placed, [Trigger("bad", "def check(ep):\n    return ep.steps[-1].command\n",
                                       REOPEN_GUIDANCE)])
    assert "could not be loaded" in out.failures["bad"]


def test_render_samples_are_kept(placed):
    render = ('def render(ep):\n'
              '    seen = sorted({a[5:] for a in ep.actions if a.startswith("open ")})\n'
              '    return "Already opened: " + ", ".join(seen) + ". Open another file."\n')
    out = replay_many(placed, [Trigger("reopen", REOPEN_CHECK, REOPEN_GUIDANCE, fault=REOPEN_FAULT,
                                       render=render)])
    reading = out.readings["reopen"]
    assert reading.render_calls == reading.fires and reading.render_raised == 0
    assert reading.render_samples and reading.render_samples[0].startswith("Already opened: file_")


def test_prefix_episodes_carry_no_outcome():
    from harness_programs.serving.steer import episode_from_messages
    record = toy_records(n_tasks=1, mix=("solve",))[0]
    whole = episode_from_messages(record.messages, max_steps=8)
    whole.final_output, whole.status, whole.gold = record.final_output, record.status, record.gold
    prefixes = prefix_episodes(whole)
    assert [len(p.steps) for p in prefixes] == list(range(record.num_steps))
    assert all(not p.final_output and not p.status and p.gold is None for p in prefixes)


def test_malformed_requests_raise(placed):
    with pytest.raises(ReplayError):
        replay_many(placed, [])
    with pytest.raises(ReplayError):
        replay_many(placed, [REOPEN, REOPEN])


def test_replay_covers_every_step_file_including_r0(scratch):  # noqa: F811
    write_rollouts(scratch, toy_records(steps=(0, 10, 20), n_tasks=2))
    reservoir = load_records(scratch)
    assert reservoir.files == [f"rollouts/step_{s:06d}.jsonl" for s in (0, 10, 20)]
    evidence, dropped = replay_evidence(reservoir.placed, [REOPEN])
    assert not dropped
    assert [p.step for p in evidence["reopen"].series] == [0, 10, 20]
    assert load_records(scratch, up_to=10).files[-1] == "rollouts/step_000010.jsonl"


def test_evidence_is_the_failure_series_of_every_trigger(placed):
    never = Trigger("never", NEVER_CHECK, REOPEN_GUIDANCE)
    evidence, dropped = replay_evidence(placed, [REOPEN, never])
    assert set(evidence) == {"reopen"} and "fires on 0 of" in dropped["never"][0]
    ev = evidence["reopen"]
    assert [p.fault_rate for p in ev.series] == [0.5, 0.25] and ev.current_rate == 0.25
    assert set(ev.to_dict()) == {"trigger", "series", "fires", "turns", "render_samples"}
    assert Evidence.from_dict(ev.to_dict()) == ev


def test_installed_triggers_are_recorded_not_screened(placed):
    never = Trigger("never", NEVER_CHECK, REOPEN_GUIDANCE)
    evidence, dropped = replay_evidence(placed, [never], installed={"never"})
    assert not dropped and evidence["never"].series[0].faulted == 0
