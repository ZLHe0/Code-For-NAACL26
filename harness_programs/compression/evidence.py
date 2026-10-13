"""The creator's evidence for ProposeCompress (Algorithm 1, line 2): contrast first, format second.

The prompt leads with the per-task contrast. For tasks of R0 where some rollouts succeeded and
some failed, it shows one successful and one failed attempt as compact turn digests and asks
the creator which evidence the success used that the failure lost. Supporting material
follows so the programs can be coded against the real format: observation sizes, what the
context limit costs, and the largest observations verbatim.

Everything domain-specific comes from optional hooks on the domain's state-DSL module:

* ``DESC`` -- what an observation *is* (structure only, never which part is valuable);
* ``label(turn)`` -- the one-line digest of a turn;
* ``turn_text(turn)`` and ``gold_strings(gold)`` -- to mark turns that surface the answer;
* ``PINS_TURN0`` -- the first observation lives in the pinned head and is never evicted;
* ``evidence(episodes)`` -- a domain-specific structural block, replacing the generic one.

A domain without any of them still gets a working prompt.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..serving.compress import parse_state
from .episodes import OptEpisode, Target, by_task
from .measure import final_context, served

CONTRAST_TASKS = 3
DIGEST_TURNS = 24     # turns listed per attempt; the middle of a longer one is elided
LABEL_CHARS = 100
VERBATIM_TURNS = 2    # largest observations of the longest episode, shown exactly as served
VERBATIM_CHARS = 900


def percentile(values: list[int], fraction: float) -> int:
    return values[min(len(values) - 1, int(fraction * len(values)))] if values else 0


def _median(xs: list[float], default: float = 0.0) -> float:
    return sorted(xs)[len(xs) // 2] if xs else default


def _units(dsl: Any, turn: Any) -> list[Any]:
    """The observations a turn serves as; grouped turns are flattened by the domain."""
    expand = getattr(dsl, "expand_state", None)
    return list(expand([turn])) if expand is not None else [turn]


def _default_label(dsl: Any) -> Callable[[Any], str]:
    def label(turn: Any) -> str:
        flat = re.sub(r"\s+", " ", " ".join(dsl.render_turn(u) for u in _units(dsl, turn)))
        return f"{flat[:60]!r} ({len(flat)}c)"
    return label


# ---------------------------------------------------------------------------------------------
# The contrast
# ---------------------------------------------------------------------------------------------


def contrast_pairs(eps: list[OptEpisode]) -> tuple[list[tuple[OptEpisode, OptEpisode]], str]:
    """``(success, failure)`` pairs from mixed-outcome tasks, widest reward gap first, and a note.

    When no task has mixed outcomes, the best and the worst episode across tasks stand in.
    """
    groups = [[ep for ep in group if ep.outcome is not None] for group in by_task(eps).values()]
    pairs = [(max(g, key=lambda e: e.outcome), min(g, key=lambda e: e.outcome))
             for g in groups if len({e.outcome for e in g}) > 1]
    pairs.sort(key=lambda p: p[1].outcome - p[0].outcome)
    if pairs:
        return pairs[:CONTRAST_TASKS], (f"{len(pairs)} of {len(groups)} tasks have mixed outcomes; "
                                        f"showing {min(len(pairs), CONTRAST_TASKS)}.")
    scored = [ep for g in groups for ep in g]
    best = max(scored, key=lambda e: e.outcome, default=None)
    worst = min(scored, key=lambda e: e.outcome, default=None)
    if best is not None and worst is not None and best.outcome != worst.outcome:
        return [(best, worst)], ("No task has mixed outcomes; contrasting the best and the worst "
                                 "attempt across tasks instead.")
    return [], "Every attempt scored the same reward, so there is no contrast to show."


def digest(target: Target, ep: OptEpisode) -> str:
    """One attempt, a line per turn: kept or evicted with no program, and whether it shows the answer."""
    dsl = target.dsl
    turns = parse_state(ep.messages, dsl)
    window = served(None, target.sizing, final_context(ep.messages, dsl))
    kept = window[1].turns_kept if window else 0
    label = target.hook("label") or _default_label(dsl)
    gold_of, turn_text = target.hook("gold_strings"), target.hook("turn_text")
    golds = [g.casefold() for g in gold_of(ep.gold) if g] if gold_of else []
    pinned = bool(target.hook("PINS_TURN0", False))

    n = len(turns)
    lines = [f"reward={ep.outcome} status={ep.status or '?'} turns={n}; with no program the limit keeps "
             + (f"the last {min(kept, n)}" if window else "nothing (the transcript cannot be served)")]
    shown = list(range(n))
    if n > DIGEST_TURNS:
        head = DIGEST_TURNS // 3
        shown = shown[:head] + [-1] + shown[n - (DIGEST_TURNS - head):]
    for i in shown:
        if i < 0:
            lines.append(f"  ... {n - DIGEST_TURNS} turns elided ...")
            continue
        turn = turns[i]
        pin = pinned and i == 0
        state = "IN " if pin or i >= n - kept else "CUT"
        gold = "  [ANSWER]" if golds and turn_text and any(g in turn_text(turn).casefold() for g in golds) else ""
        lines.append(f"  turn {i:<2} {state}{' (pinned)' if pin else ''}  {str(label(turn))[:LABEL_CHARS]}{gold}")
    return "\n".join(lines)


def contrast_block(target: Target, eps: list[OptEpisode]) -> tuple[str, list[dict[str, Any]]]:
    pairs, note = contrast_pairs(eps)
    blocks = [note]
    for k, (success, failure) in enumerate(pairs, 1):
        task = success.task_id if success.task_id == failure.task_id else f"{success.task_id} vs {failure.task_id}"
        blocks.append(f"### Task {k} ({task})\nSUCCESSFUL attempt: {digest(target, success)}\n"
                      f"FAILED attempt: {digest(target, failure)}")
    shown = [{"success": s.task_id, "success_reward": s.outcome, "failure": f.task_id, "failure_reward": f.outcome}
             for s, f in pairs]
    return "\n\n".join(blocks), shown


# ---------------------------------------------------------------------------------------------
# Supporting evidence: format and sizes
# ---------------------------------------------------------------------------------------------


def structural_evidence(target: Target, eps: list[OptEpisode]) -> tuple[dict[str, Any], str]:
    """Observation sizes and turn counts: the shape of the pressure, with no content judgement."""
    dsl = target.dsl
    obs_chars: list[int] = []
    turn_counts: list[int] = []
    for ep in eps:
        turns = parse_state(ep.messages, dsl)
        if turns:
            turn_counts.append(len(turns))
            obs_chars.extend(len(dsl.render_turn(u)) for t in turns for u in _units(dsl, t))
    obs_chars.sort()
    turn_counts.sort()
    stats = {"n_episodes": len(turn_counts),
             "obs_chars": {"p50": percentile(obs_chars, .5), "p90": percentile(obs_chars, .9),
                           "max": obs_chars[-1] if obs_chars else 0},
             "turns": {"p50": percentile(turn_counts, .5), "max": turn_counts[-1] if turn_counts else 0}}
    block = (f"Measured over {stats['n_episodes']} episodes.\n"
             f"Per-turn observation size (chars): p50={stats['obs_chars']['p50']}, "
             f"p90={stats['obs_chars']['p90']}, max={stats['obs_chars']['max']}\n"
             f"Turns per episode: median={stats['turns']['p50']}, max={stats['turns']['max']}")
    return stats, block


def budget_brief(target: Target, eps: list[OptEpisode]) -> str:
    """The context limit as a per-turn price, and how much history it leaves with no program."""
    sizing = target.sizing
    budgets, fulls, prices, totals, kepts, evicted = [], [], [], [], [], 0
    for ep in eps:
        turns = parse_state(ep.messages, target.dsl)
        window = served(None, sizing, final_context(ep.messages, target.dsl)) if turns else None
        if window is None:
            continue
        stats = window[1]
        budgets.append(stats.budget)
        fulls.append(stats.raw_tokens)
        prices.append(max(stats.raw_tokens - stats.floor_tokens, 0) / len(turns))
        totals.append(len(turns))
        kepts.append(min(stats.turns_kept, len(turns)))
        evicted += stats.evicted
    if not budgets:
        return "No episode can be served at this context limit."
    rule = (f"the pinned head plus {sizing.history_budget} tokens of history, capped at {sizing.serving_limit}"
            if sizing.history_budget is not None else f"{sizing.serving_limit} tokens for the whole prompt")
    return (f"The context limit is {rule}: about {round(_median(budgets))} tokens on this data, against a full "
            f"transcript of about {round(_median(fulls))} tokens over {_median(totals):.0f} turns.\n"
            f"One turn costs about {_median(prices):.0f} tokens. With no program, eviction drops turns on "
            f"{round(100.0 * evicted / len(budgets))}% of episodes and keeps a median of "
            f"{_median(kepts):.0f} of {_median(totals):.0f} turns.")


def verbatim_turns(target: Target, eps: list[OptEpisode]) -> str:
    """The longest servable episode's largest observations exactly as served."""
    dsl = target.dsl
    candidates = [(len(turns), i, turns) for i, ep in enumerate(eps)
                  if (turns := parse_state(ep.messages, dsl)) and served(None, target.sizing, ep.messages)]
    if not candidates:
        return ""
    turns = max(candidates, key=lambda c: (c[0], -c[1]))[2]
    rendered = sorted(((len(body), t.index, body) for t in turns for u in _units(dsl, t)
                       for body in [dsl.render_turn(u)]), reverse=True)[:VERBATIM_TURNS]
    blocks = []
    for size, index, body in sorted(rendered, key=lambda item: item[1]):
        clipped = body if size <= VERBATIM_CHARS else f"{body[:VERBATIM_CHARS]}\n[... {size - VERBATIM_CHARS} more chars]"
        blocks.append(f"turn {index} ({size} chars):\n{clipped}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------------------------


def creator_prompt(target: Target, eps: list[OptEpisode], n: int) -> tuple[str, dict[str, Any]]:
    """The creator's drafting prompt for ``n`` candidates, and the structured evidence behind it."""
    contrast, shown = contrast_block(target, eps)
    custom = target.hook("evidence")
    stats, structure = custom(eps) if custom else structural_evidence(target, eps)
    prompt = f"""## Domain
{target.desc}

## Successful and failed attempts on the same task
Each line is one turn. IN/CUT is whether the context limit keeps that turn at the end of the
attempt when no program runs; [ANSWER] marks a turn whose observation shows the answer.

{contrast}

First, for each task above, identify the evidence the successful attempt used that the failed
attempt lost. Then write {n} DISTINCT compression programs, each keeping that kind of evidence
readable while freeing room for more history.

## Contract
{target.contract}

## Supporting measurements
{structure}

{budget_brief(target, eps)}

## Largest observations, verbatim
{verbatim_turns(target, eps)}

Return JSON ONLY:
{{"findings": ["<=40 words per task: the evidence the success used and the failure lost", ...],
 "proposals": [{{"name": "<slug>", "rationale": "<=40 words, tied to the findings",
                "code": "def compress(state):\\n    ...\\n    return state"}}, ... {n} total]}}"""
    return prompt, {"contrast": shown, "structure": stats}
