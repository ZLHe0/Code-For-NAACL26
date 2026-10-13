"""The measurements of ProposeCompress (Algorithm 1, line 2): is a candidate unfaithful or inert?

A candidate runs on stored episodes from R0 through the real serving path. The creator is shown
the contexts before and after it (:func:`render_samples`) together with two numbers:

* **unfaithful** -- at up to ``DECISION_POINTS`` turn boundaries per episode, the items the policy
  needs later are the domain's gold strings plus the words (at least ``MIN_ITEM_CHARS`` long) of
  every action the agent takes from that point on. Counted are the needed items readable in the
  append-only context outside what the contract already protects (the pinned head, the agent's
  actions, the live turn); reported is the share of them no longer readable anywhere in the
  compressed context, with examples of lost items.
* **inert** -- at each episode's last decision, over the episodes where the budget binds
  (eviction drops a turn) or the context fills at least ``LARGE_CONTEXT`` of it, the share on
  which the served context is identical to the one served with no program, or the program
  saves less than ``INERT_SAVING_PCT`` percent of the tokens; reported with the mean saving.

The same pass checks the serving contract: the program compiles, never raises, keeps every
action, and leaves the live turn alone. A violation is returned as an error to repair.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..serving.budget import BudgetedContext, BudgetStats
from ..serving.compress import CompressionError, MeasuredStage1, pair_observations
from ..serving.messages import Messages, content_of, is_assistant, pinned_head, tool_calls_of, turn_starts
from .episodes import OptEpisode, Sizing, Target

DECISION_POINTS = 4
MIN_ITEM_CHARS = 4
LARGE_CONTEXT = 0.5
INERT_SAVING_PCT = 5.0
N_SAMPLES = 4
N_EXAMPLES = 6
SAMPLE_CLIP = 700
ERROR_CLIP = 300

ParseActions = Callable[[str], list[str]]
Stage1 = Callable[[Messages], Messages]

_ITEM_RE = re.compile(rf"[\w./-]{{{MIN_ITEM_CHARS},}}")


class _Recorded(MeasuredStage1):
    """A measured program that also keeps each raise's message, to hand back for repair."""

    def __init__(self, domain: str, code: str) -> None:
        self.errors: list[str] = []
        super().__init__(domain, code)

    def _count(self, exc: BaseException) -> None:
        super()._count(exc)
        self.errors.append(f"{type(exc).__name__}: {exc}"[:ERROR_CLIP])


# ---------------------------------------------------------------------------------------------
# Serving replay
# ---------------------------------------------------------------------------------------------


def served(stage1: Stage1 | None, sizing: Sizing, messages: Sequence[dict[str, Any]]
           ) -> tuple[Messages, BudgetStats] | None:
    """What the policy is served for this transcript, or ``None`` if it cannot be served at all."""
    budget = BudgetedContext(stage1, token_counter=sizing.counter, serving_limit=sizing.serving_limit,
                             history_budget=sizing.history_budget)
    try:
        return budget(list(messages))
    except ValueError:
        return None


def actions_of(messages: Sequence[dict[str, Any]], parse_actions: ParseActions | None = None) -> list[str]:
    """Every action the transcript took: tool calls with their arguments, or text parsed by the domain."""
    out: list[str] = []
    for m in messages:
        if not is_assistant(m):
            continue
        calls = tool_calls_of(m)
        if calls:
            out.extend(f"{c.get('name')}:{c.get('args')}" for c in calls)
        elif parse_actions is not None:
            out.extend(parse_actions(content_of(m)))
    return out


def decision_points(messages: Sequence[dict[str, Any]], dsl: Any) -> list[int]:
    """Up to ``DECISION_POINTS`` prefix lengths ending on a completed observation, the last included."""
    ends = [i + 1 for i, m in enumerate(messages)
            if dsl.is_observation(m) and (i + 1 == len(messages) or not dsl.is_observation(messages[i + 1]))]
    step = max(1, len(ends) // DECISION_POINTS)
    return sorted(ends[::-1][::step][:DECISION_POINTS])


def final_context(messages: Sequence[dict[str, Any]], dsl: Any) -> list[dict[str, Any]]:
    """The transcript as it stood at its last decision: up to the last completed observation."""
    points = decision_points(messages, dsl)
    return list(messages[:points[-1]]) if points else list(messages)


def live_start(prefix: Sequence[dict[str, Any]], dsl: Any) -> int:
    """Where the live turn -- the one the policy must answer next -- begins in ``prefix``."""
    obs_idx, ai_idx = pair_observations(prefix, dsl.is_observation)[-1]
    return ai_idx if 0 <= ai_idx < obs_idx else obs_idx


def live_turn_edited(prefix: Sequence[dict[str, Any]], out: Sequence[dict[str, Any]], dsl: Any) -> bool:
    return any(out[j] != prefix[j] for j in range(live_start(prefix, dsl), len(prefix)))


# ---------------------------------------------------------------------------------------------
# Unfaithful
# ---------------------------------------------------------------------------------------------


def _readable(messages: Sequence[dict[str, Any]]) -> str:
    """Everything the policy can read in ``messages``, case-folded: text and tool-call arguments."""
    parts = [content_of(m) for m in messages]
    parts += [str(c["args"]) for m in messages for c in tool_calls_of(m)]
    return "\n".join(parts).casefold()


def needed_items(target: Target, ep: OptEpisode, later: Sequence[dict[str, Any]]) -> set[str]:
    """The gold strings, plus the words of every action in ``later``."""
    gold_of = target.hook("gold_strings")
    items = {g for g in gold_of(ep.gold) if g} if gold_of else set()
    items.update(w for action in actions_of(later, target.hook("parse_actions")) for w in _ITEM_RE.findall(action))
    return items


def _protected(target: Target, prefix: Sequence[dict[str, Any]]) -> str:
    """What no valid program can remove: the pinned head, the actions taken, and the live turn."""
    start = live_start(prefix, target.dsl)
    actions = actions_of(prefix[:start], target.hook("parse_actions"))
    return "\n".join([_readable(pinned_head(prefix)), _readable(prefix[start:]), *actions]).casefold()


def lost_items(target: Target, needed: set[str], before: Sequence[dict[str, Any]],
               after: Sequence[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """``(needed items a program could remove from before, those no longer readable in after)``."""
    was, now, kept = _readable(before), _readable(after), _protected(target, before)
    readable = {i for i in needed if i.casefold() in was and i.casefold() not in kept}
    return readable, {i for i in readable if i.casefold() not in now}


# ---------------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------------


@dataclass
class _Tally:
    raising: int = 0
    actions_lost: int = 0
    live_edited: int = 0
    needed: int = 0
    lost: int = 0
    examples: dict[str, str] = field(default_factory=dict)
    pressured: list[tuple[bool, float, int, int]] = field(default_factory=list)
    diffs: list[tuple[int, str, str]] = field(default_factory=list)


def _edits(raw: Messages, compressed: Messages, dsl: Any) -> list[tuple[int, str, str]]:
    """``(chars removed, before, after)`` for every observation or rationale the program rewrote."""
    diffs = []
    for obs_idx, ai_idx in pair_observations(raw, dsl.is_observation):
        for idx in (ai_idx, obs_idx):
            if idx >= 0 and (before := content_of(raw[idx])) != (after := content_of(compressed[idx])):
                diffs.append((len(before) - len(after), before, after))
    return diffs


def _faithfulness(stage1: Stage1, target: Target, ep: OptEpisode, messages: Messages, tally: _Tally) -> bool:
    """Tally needed and lost items at each decision point; return whether the live turn was edited."""
    live = False
    for cut in decision_points(messages, target.dsl):
        prefix = messages[:cut]
        out = stage1(prefix)
        live = live or live_turn_edited(prefix, out, target.dsl)
        readable, lost = lost_items(target, needed_items(target, ep, messages[cut:]), prefix, out)
        tally.needed += len(readable)
        tally.lost += len(lost)
        for item in sorted(lost):
            tally.examples.setdefault(item, f"{item!r} (task {ep.task_id}, after turn {len(turn_starts(prefix))})")
    return live


def _pressure(stage1: Stage1, sizing: Sizing, messages: Messages, tally: _Tally) -> None:
    """If the budget binds or the context is large at ``messages``, record whether the program is inert there."""
    naive, compressed = served(None, sizing, messages), served(stage1, sizing, messages)
    if naive is None or compressed is None:
        return
    (naive_msgs, ns), (comp_msgs, cs) = naive, compressed
    if not (ns.evicted or ns.tokens_before >= LARGE_CONTEXT * ns.budget):
        return
    saving = 100.0 * (cs.raw_tokens - cs.tokens_before) / max(cs.raw_tokens, 1)
    inert = comp_msgs == naive_msgs or saving < INERT_SAVING_PCT
    tally.pressured.append((inert, saving, ns.turns_kept, cs.turns_kept))


def _episode(stage1: Stage1, target: Target, ep: OptEpisode, tally: _Tally) -> None:
    parse = target.hook("parse_actions")
    messages = list(ep.messages)
    compressed = stage1(messages)
    tally.actions_lost += bool(Counter(actions_of(messages, parse)) - Counter(actions_of(compressed, parse)))
    tally.diffs.extend(_edits(messages, compressed, target.dsl))
    tally.live_edited += _faithfulness(stage1, target, ep, messages, tally)
    _pressure(stage1, target.sizing, final_context(messages, target.dsl), tally)


def _mean(xs: Sequence[float]) -> float | None:
    return round(sum(xs) / len(xs), 1) if xs else None


def _violations(tally: _Tally, errors: list[str], n: int) -> list[str]:
    out = []
    if tally.raising:
        out.append(f"raised on {tally.raising} of {n} episodes, e.g. {errors[0]}")
    if tally.actions_lost:
        out.append(f"changed or dropped an action on {tally.actions_lost} episode(s); "
                   "every action must be served unchanged")
    if tally.live_edited:
        out.append(f"rewrote the live turn on {tally.live_edited} episode(s); "
                   "leave the last turn in `state` untouched")
    return out


def _clip(value: str, limit: int = SAMPLE_CLIP) -> str:
    return value if len(value) <= limit else f"{value[:limit]}\n[... {len(value) - limit} more chars]"


def _shape(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[A-Za-z0-9_]+", "w", value))[:120]


def _pick_samples(diffs: list[tuple[int, str, str]]) -> list[dict[str, Any]]:
    """The largest shrinks, one per distinct before/after shape."""
    seen, picked = set(), []
    for removed, before, after in sorted(diffs, key=lambda d: -d[0]):
        key = (_shape(before), _shape(after))
        if key in seen:
            continue
        seen.add(key)
        picked.append({"chars_removed": removed, "before": _clip(before), "after": _clip(after)})
        if len(picked) >= N_SAMPLES:
            break
    return picked


def measure(target: Target, code: str, eps: Sequence[OptEpisode]) -> dict[str, Any]:
    """Run ``code`` on ``eps``: contract violations, the two measurements, and before/after samples."""
    result: dict[str, Any] = {"compiles": False, "violations": [], "episodes": len(eps)}
    try:
        stage1 = _Recorded(target.domain, code)
    except Exception as exc:  # noqa: BLE001 -- a load failure is the finding
        result["violations"] = [f"does not compile: {type(exc).__name__}: {exc}"[:ERROR_CLIP]]
        return result
    tally = _Tally()
    for ep in eps:
        before = len(stage1.errors)
        try:
            _episode(stage1, target, ep, tally)
        except CompressionError as exc:
            stage1.errors.append(f"{type(exc).__name__}: {exc}"[:ERROR_CLIP])
        tally.raising += len(stage1.errors) > before

    rows = tally.pressured
    result.update(
        compiles=True,
        violations=_violations(tally, stage1.errors, len(eps)),
        unfaithful={"pct": round(100.0 * tally.lost / tally.needed, 1) if tally.needed else None,
                    "needed": tally.needed, "lost": tally.lost,
                    "examples": list(tally.examples.values())[:N_EXAMPLES]},
        inert={"pct": _mean([100.0 * r[0] for r in rows]), "episodes": len(rows),
               "token_saving_pct": _mean([r[1] for r in rows]),
               "turns_served": {"without": _mean([r[2] for r in rows]), "with": _mean([r[3] for r in rows])}},
        samples=_pick_samples(tally.diffs),
    )
    return result


# ---------------------------------------------------------------------------------------------
# Rendering for the creator
# ---------------------------------------------------------------------------------------------


def render_measurement(m: dict[str, Any]) -> str:
    u, i = m["unfaithful"], m["inert"]
    lines = [f"UNFAITHFUL: of the items the agent needs later (gold strings, words of its later actions) "
             f"that it could read without compression outside the pinned head, its actions and the live turn, "
             f"{u['pct'] if u['pct'] is not None else 'n/a'}% are gone after it "
             f"({u['lost']} of {u['needed']} over {m['episodes']} episodes)."]
    if u["examples"]:
        lines.append("  lost, e.g. " + "; ".join(u["examples"]))
    if i["episodes"]:
        lines.append(f"INERT: on {i['pct']}% of the {i['episodes']} episodes where the budget binds or the "
                     f"context is large, the served context is unchanged or shrinks by under "
                     f"{INERT_SAVING_PCT:g}%.")
        lines.append(f"  mean token saving {i['token_saving_pct']}%; turns served "
                     f"{i['turns_served']['without']} without compression -> {i['turns_served']['with']} with it")
    else:
        lines.append("INERT: not measurable; on no episode does the budget bind or the context fill "
                     f"{LARGE_CONTEXT:.0%} of it.")
    return "\n".join(lines)


def render_samples(m: dict[str, Any]) -> str:
    if not m.get("samples"):
        return "(the program changed no turn)"
    return "\n\n".join(f"--- sample {i}: {s['chars_removed']} characters removed\n"
                       f"BEFORE:\n{s['before']}\n\nAFTER:\n{s['after']}"
                       for i, s in enumerate(m["samples"], 1))
