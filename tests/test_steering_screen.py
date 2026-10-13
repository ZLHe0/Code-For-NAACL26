"""Program admission (static screens), dedup, and the behaviour screens replay applies."""

from __future__ import annotations

from test_steering_support import REOPEN_CHECK, REOPEN_FAULT, REOPEN_GUIDANCE

from harness_programs.steering.replay import Replay, ReplayedEpisode
from harness_programs.steering.screen import (
    behaviour_screen,
    dedup,
    gold_strings,
    guidance_gold_leak,
    static_screen,
)


def _screen(check=REOPEN_CHECK, fault=REOPEN_FAULT, guidance=REOPEN_GUIDANCE, render="", gold=None):
    return static_screen(check, fault, guidance, render, gold=gold)


def test_a_clean_candidate_passes():
    assert _screen() == []


def test_fault_may_qualify_with_gold():
    fault = ('def fault(ep):\n'
             '    opened = [a for a in ep.actions if a.startswith("open ")]\n'
             '    return bool(ep.gold) and ep.gold["file"] not in " ".join(opened)\n')
    assert _screen(fault=fault) == []


def test_always_on_trigger_is_refused():
    found = _screen(check="def check(ep):\n    return True\n")
    assert any("reads nothing off the episode" in c for c in found)


def test_trigger_reading_the_outcome_is_refused():
    check = 'def check(ep):\n    return ep.status == "max_turns" and len(ep.actions) > 2\n'
    assert any("always empty while an episode is running" in c for c in _screen(check=check))
    check = 'def check(ep):\n    return not ep.final_output and len(ep.actions) > 2\n'
    assert any("final_output" in c for c in _screen(check=check))


def test_trigger_reading_gold_or_reward_is_refused():
    for check in ('def check(ep):\n    return ep.gold is None and len(ep.actions) > 1\n',
                  'def check(ep):\n    return getattr(ep, "reward", 0) < 1 and len(ep.actions) > 1\n'):
        assert any("do not exist at serving time" in c for c in _screen(check=check))


def test_fault_reading_raw_reward_is_refused():
    fault = 'def fault(ep):\n    return getattr(ep, "reward", 0) < 1 and len(ep.actions) > 2\n'
    assert any("only through `ep.gold`" in c for c in _screen(fault=fault))


def test_fault_testing_only_elapsed_time_is_refused():
    fault = "def fault(ep):\n    return ep.n_steps >= ep.max_steps\n"
    assert any("tests only elapsed turns" in c for c in _screen(fault=fault))


def test_fault_restating_the_reward_is_refused():
    fault = 'def fault(ep):\n    return ep.gold is not None\n'
    assert any("restates the reward" in c for c in _screen(fault=fault))


def test_entrypoint_and_arity():
    assert any("defines no top-level check" in c
               for c in _screen(check="def trigger(ep):\n    return len(ep.actions) > 2\n"))
    assert any("exactly one argument" in c
               for c in _screen(check="def check(ep, n):\n    return len(ep.actions) > n\n"))
    assert any("does not parse" in c for c in _screen(check="def check(ep):\n    return (\n"))


def test_sandbox_policy_applies():
    check = "import os\ndef check(ep):\n    return len(ep.actions) > 2\n"
    assert any("sandbox policy" in c for c in _screen(check=check))


def test_guidance_length_condition_and_imperative():
    assert any("characters; under" in c for c in _screen(guidance="If stuck, move on."))
    assert any("over the 300" in c for c in _screen(guidance="If stuck, open another file. " * 20))
    assert any("never says in what case" in c
               for c in _screen(guidance="Open a file you have not read yet, and answer with the file that has the word."))
    assert any("never names an action" in c
               for c in _screen(guidance="If the agent keeps reading the same file, that is a waste of turns."))


def test_guidance_naming_the_gold_is_refused():
    gold = sorted(gold_strings({"file": "file_d.txt", "word": "gamma"}))
    leaky = "If you have read every file without finding it, open file_d.txt and answer with it."
    assert any("from this task's gold answer" in c for c in _screen(guidance=leaky, gold=gold))
    assert _screen(gold=gold) == []
    assert guidance_gold_leak("If unsure, check the txt files again before you answer.", gold) == []


def _reading(name: str, fired: dict[str, list[int]], faulted: set[str]) -> Replay:
    r = Replay(name)
    for ep, turns in fired.items():
        r.episodes.append(ReplayedEpisode(ep, 10, "g", len(turns), 8, turns, ep in faulted, "max_turns"))
    return r


def test_dedup_drops_byte_identical_checks_keeping_the_first():
    sources = {"a": "def check(ep): return 1", "b": "def check(ep): return 2",
               "c": "def check(ep): return 1\n"}
    assert dedup(sources) == {"c": "its check is byte-identical to a's"}


def test_dedup_never_drops_existing_triggers():
    dropped = dedup({"new": "def check(ep): return 1"}, existing={"inc": "def check(ep): return 1"})
    assert dropped == {"new": "its check is byte-identical to inc's"}


def test_behaviour_screen_flags_degenerate_triggers():
    assert behaviour_screen(_reading("ok", {"e1": [3], "e2": []}, {"e1"})) == []
    silent = behaviour_screen(_reading("silent", {"e1": [], "e2": []}, {"e1"}))
    assert len(silent) == 1 and "fires on 0 of 2" in silent[0]
    everywhere = _reading("everywhere", {"e1": list(range(8))}, {"e1"})
    assert "fires on every one" in behaviour_screen(everywhere)[0]
    never = behaviour_screen(_reading("never", {"e1": [3]}, set()))
    assert len(never) == 1 and "fault holds on 0 of 1" in never[0]
    assert "no episode" in behaviour_screen(Replay("empty"))[0]
