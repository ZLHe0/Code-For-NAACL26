"""Program admission for ProposeSteer (Algorithm 1, line 6), and Replay's behaviour screens.

The *static* screens read source only: required entry point, the sandbox policy (allowed
imports, forbidden names, unsafe constructs), and the rules on what a trigger, a fault and a
guidance may read or say. They run inside the proposer's repair loop, so a complaint is shown
to the creator model while the exchange is still open.

The *behaviour* screens need a replay over stored rollouts (Algorithm 1, line 7) and drop a
degenerate candidate: one that never fires, fires on every turn, or whose failure never
occurred. :func:`dedup` drops a proposal whose check is byte-identical to one already kept.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from ..serving.sandbox import screen_source
from ..serving.steer import TRIGGER_POLICY, FAULT_POLICY, RENDER_POLICY
from .prompts import GUIDANCE_MAX_CHARS, GUIDANCE_MIN_CHARS

if TYPE_CHECKING:
    from .replay import Replay

# Names that read what the agent did. Attribute access (``ep.observations``) and the string form
# (``getattr(s, 'observation', '')``) both count: they mean the same thing.
BEHAVIOURAL_NAMES = frozenset({
    "observation", "observations", "command", "action", "actions", "tool_calls", "task_text",
    "question", "queries", "reasoning", "reasonings", "final_output", "args", "status",
})

# Empty while an episode is running, so a trigger reading them is not unsafe but broken: the
# clause is dead in flight and the trigger fires on whatever remains.
OUTCOME_NAMES = frozenset({"final_output", "status"})

# Enough on their own to make a trigger a trigger: "in the last few turns" is a real choice.
TIMING_NAMES = frozenset({"n_steps", "max_steps", "steps", "remaining", "turn", "turns"})

# What no served predicate may read. A trigger runs mid-episode in a process that never loaded
# the dataset, so a clause reading one of these is a default, not a signal.
GOLD_NAMES = frozenset({
    "reward", "rewards", "score", "scores", "gold", "gold_answer", "ground_truth", "grade", "f1",
    "reward_model",
})

# The one gold name a fault may read, and the only route it has to the answer.
FAULT_GOLD_NAME = "gold"

# Shortest gold token worth looking for in a guidance; shorter ones collide with ordinary prose.
GOLD_TOKEN_MIN_CHARS = 4

# Share of firing turns on which a render may raise before it is refused.
RENDER_RAISE_SHARE = 0.05

CONDITION_HINTS = frozenset({"if", "when", "whenever", "once", "unless"})

IMPERATIVE_HINTS = frozenset({
    "call", "use", "run", "check", "give", "name", "submit", "stop", "avoid", "prefer", "start",
    "read", "verify", "confirm", "list", "try", "write", "return", "answer", "pick", "choose",
    "include", "add", "set", "do", "make", "quote", "cite", "search", "look", "re-run", "rerun",
    "open", "inspect", "find", "commit", "compare", "switch", "move", "go",
})


def names_mentioned(code: str) -> set[str]:
    """Every attribute name and string literal in ``code``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def static_entrypoint(code: str, *, entrypoint: str, role: str) -> list[str]:
    """Refuse source whose top-level entrypoint is missing or does not take exactly one arg."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"the {role} does not parse: {exc.msg} at line {exc.lineno}"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            args = node.args
            n_args = len(args.posonlyargs) + len(args.args)
            if n_args != 1 or args.vararg or args.kwarg or args.kwonlyargs:
                return [f"{entrypoint}() must take exactly one argument, the episode"]
            return []
    return [f"the {role} defines no top-level {entrypoint}(ep) -- helpers are fine, but the "
            f"entrypoint has to be a module-level def with that exact name"]


def static_policy(code: str, *, role: str) -> list[str]:
    """The sandbox policy the serving side will apply: imports, forbidden names, dunders."""
    policy = {"trigger": TRIGGER_POLICY, "fault": FAULT_POLICY, "render": RENDER_POLICY}[role]
    return [f"the {role} breaks the sandbox policy: {reason}"
            for reason in screen_source(code, policy)]


def static_always_on(code: str, *, role: str) -> list[str]:
    """Refuse a trigger that reads nothing off the episode: ``return True`` is no trigger."""
    if names_mentioned(code) & (BEHAVIOURAL_NAMES | TIMING_NAMES | OUTCOME_NAMES | GOLD_NAMES):
        return []
    return [f"the {role} reads nothing off the episode, so it is true on every turn of every "
            f"episode. Its guidance would be appended to the whole run rather than to a moment, "
            f"and a rate that is 1.0 everywhere cannot show an effect. Say when this should "
            f"speak: at a point in the episode, or on something the transcript shows."]


def static_outcome_leak(trigger: str) -> list[str]:
    """Refuse a trigger that consults the outcome, which is always empty in flight."""
    leaked = sorted(names_mentioned(trigger) & OUTCOME_NAMES)
    if not leaked:
        return []
    return [f"the trigger reads {', '.join(leaked)}, which is always empty while an episode is "
            f"running -- a trigger fires before the turn is generated. That clause is dead in "
            f"flight, so the trigger will fire on the remaining conditions alone. Move the "
            f"outcome test into the fault, which is graded offline and may read it."]


def static_gold_leak(code: str, *, role: str) -> list[str]:
    """Refuse a predicate reading the reward or gold; a fault may read ``ep.gold`` only."""
    mentioned = names_mentioned(code)
    allowed = {FAULT_GOLD_NAME} if role == "fault" else set()
    leaked = sorted((mentioned & GOLD_NAMES) - allowed)
    if not leaked:
        return []
    if role == "fault":
        return [f"the fault reads {', '.join(leaked)}. A fault may consult the oracle, but only "
                f"through `ep.gold` -- the curated answer view. The raw reward is not on the "
                f"episode, and a fault that recomputes a score is a private copy of the reward "
                f"function that will drift from the real one."]
    return [f"the {role} reads {', '.join(leaked)}. You are shown the reward and the gold answer "
            f"as evidence, but they do not exist at serving time -- a trigger runs mid-episode, "
            f"before any reward is computed, and the gold answer is never loaded into that "
            f"process. That clause reads a default on every episode, so it cannot distinguish "
            f"anything. Derive the {role} from what the transcript shows."]


def static_clock(fault: str) -> list[str]:
    """Refuse a fault that tests only elapsed turns: nothing the agent does can make it false."""
    mentioned = names_mentioned(fault)
    if mentioned & BEHAVIOURAL_NAMES or FAULT_GOLD_NAME in mentioned:
        return []
    if not mentioned & TIMING_NAMES:
        return ["the fault never reads what the agent did -- no observation, command, tool call "
                "or task text -- so nothing the agent does correctly can change its answer. Add "
                "a clause about what it actually did, or did not do."]
    return ["the fault tests only elapsed turns -- it never reads an observation, a command, a "
            "tool call or the task text, so nothing the agent does correctly can change its "
            "answer. Running out of turns is not itself a failure; what the agent failed to do "
            "with them is. Add a clause about what it actually did, or did not do."]


def static_gold_tautology(fault: str) -> list[str]:
    """Refuse a fault that reads ``ep.gold`` and nothing the agent did: the reward restated."""
    mentioned = names_mentioned(fault)
    if FAULT_GOLD_NAME not in mentioned or mentioned & BEHAVIOURAL_NAMES:
        return []
    return ["the fault reads ep.gold and nothing else about the episode, so it restates the "
            "reward rather than describing a behaviour. Every low-scoring episode trips it "
            "whatever the agent did. Gold may QUALIFY a fault -- say which episodes a behaviour "
            "actually cost something in -- but the behaviour has to be there too: name what the "
            "agent did, then use ep.gold to say when it mattered."]


def _has_imperative_clause(text: str) -> bool:
    """Does any clause *begin* with an instruction verb? "If ..., open ..." counts."""
    for clause in re.split(r"[.,;:!?\n]+", text):
        head = [w.strip("`\"'()").lower() for w in clause.split()]
        if head[:1] == ["then"]:
            head = head[1:]
        if head and head[0] in IMPERATIVE_HINTS:
            return True
    return False


def message_screen(guidance: str) -> list[str]:
    """Refuse guidance that cannot act on the agent: wrong length, no case, no instruction."""
    text = (guidance or "").strip()
    if len(text) < GUIDANCE_MIN_CHARS:
        return [f"the guidance is {len(text)} characters; under {GUIDANCE_MIN_CHARS} it cannot "
                f"state both the case it applies to and what to do in it"]
    out: list[str] = []
    if len(text) > GUIDANCE_MAX_CHARS:
        out.append(f"the guidance is {len(text)} characters, over the {GUIDANCE_MAX_CHARS} "
                   f"budget -- it is appended to a prompt whose history it displaces. Cut it to "
                   f"one case and one action.")
    words = {w.strip(".,;:!?`\"'()").lower() for w in text.split()}
    if not words & CONDITION_HINTS:
        out.append("the guidance never says in what case it applies. The trigger is loose, so it "
                   "lands on turns where this advice does not fit -- state the case as a "
                   "condition (\"If <what the agent has seen>, <what to do>\").")
    if not words & IMPERATIVE_HINTS:
        out.append("the guidance never names an action. Say what the agent should do in that "
                   "case, as an instruction.")
    elif not _has_imperative_clause(text):
        out.append("the guidance describes the problem but never issues an instruction -- lead "
                   "the action clause with the verb (\"If ..., open ...\").")
    return out


def render_screen(render: str) -> list[str]:
    """Refuse a ``render(ep)`` that reads what does not exist in flight."""
    out = static_entrypoint(render, entrypoint="render", role="render")
    if out:
        return out
    out += static_policy(render, role="render")
    leaked = sorted(names_mentioned(render) & (OUTCOME_NAMES | GOLD_NAMES))
    if leaked:
        out.append(f"the render reads {', '.join(leaked)}, which does not exist while an episode "
                   f"is running: it runs on the same prefix as the trigger. Quote only what the "
                   f"transcript so far shows.")
    return out


def static_screen(trigger: str, fault: str, guidance: str, render: str = "", *,
                  gold: Iterable[str] | None = None) -> list[str]:
    """Every source-level complaint about one candidate.

    ``gold`` is the string leaves of the gold views the candidate was written from
    (:func:`gold_strings`); a guidance that names one of them is refused, because guidance has to
    hold on tasks whose answer is different.
    """
    out = static_entrypoint(trigger, entrypoint="check", role="trigger")
    if not out:
        out += static_policy(trigger, role="trigger")
    out += static_always_on(trigger, role="trigger")
    out += static_outcome_leak(trigger)
    out += static_gold_leak(trigger, role="trigger")
    if fault and fault.strip():
        entry = static_entrypoint(fault, entrypoint="fault", role="fault")
        out += entry or static_policy(fault, role="fault")
        out += static_clock(fault)
        out += static_gold_leak(fault, role="fault")
        out += static_gold_tautology(fault)
    out += message_screen(guidance)
    out += guidance_gold_leak(guidance, gold)
    if render and render.strip():
        out += render_screen(render)
    return out


def gold_strings(gold: object) -> frozenset[str]:
    """The string leaves of a gold view, for the guidance leak check."""
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, Mapping):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for v in value:
                walk(v)

    walk(gold)
    return frozenset(found)


def guidance_gold_leak(guidance: str, gold: Iterable[str] | None = None) -> list[str]:
    """A guidance that names part of the gold answer it was written from.

    Tokens shorter than :data:`GOLD_TOKEN_MIN_CHARS` are ignored, so file extensions and short
    common words do not count as a leak.
    """
    if not gold:
        return []
    lowered = (guidance or "").lower()
    hits: set[str] = set()
    for name in gold:
        for token in {name, *re.split(r"[/:.\s]+", name)}:
            token = token.strip()
            if len(token) >= GOLD_TOKEN_MIN_CHARS and token.lower() in lowered:
                hits.add(token)
    if not hits:
        return []
    return [f"the guidance names {', '.join(sorted(hits)[:5])} from this task's gold answer. "
            f"Guidance is served on tasks whose answer is different (P2): describe the "
            f"behaviour, not this answer."]


# ---------------------------------------------------------------------------------------------
# Behaviour screens (need a replay)
# ---------------------------------------------------------------------------------------------

def behaviour_screen(reading: Replay) -> list[str]:
    """Every complaint that needs real episodes to see.

    Never fires, fires on every turn of every episode, fault never holds, render raises too
    often. Replays that raised are refused upstream, in :mod:`.replay`.
    """
    episodes = reading.episodes
    if not episodes:
        return ["no episode could be replayed, so nothing about this candidate was measured"]
    out: list[str] = []
    fires = sum(e.fires for e in episodes)
    turns = sum(e.turns for e in episodes)
    if not fires:
        out.append(f"the trigger fires on 0 of {len(episodes)} real episodes. The idea may be "
                   f"right and the matching too strict for full transcripts -- normalize before "
                   f"comparing, match on short stable anchors, and replace equality with "
                   f"containment.")
    elif turns and fires == turns:
        out.append(f"the trigger fires on every one of the {turns} turns of all "
                   f"{len(episodes)} replayed episodes, so it reads the episode without "
                   f"choosing anything. Its guidance would land on the whole run and a rate of "
                   f"1.0 cannot show an effect. Narrow it to the moment the advice is worth "
                   f"hearing.")
    if not any(e.fault for e in episodes):
        out.append(f"the fault holds on 0 of {len(episodes)} episodes, so on this evidence the "
                   f"behaviour never happened and there is nothing for a trigger to prevent.")
    if reading.render_calls and reading.render_raised > RENDER_RAISE_SHARE * reading.render_calls:
        out.append(f"the render raised on {reading.render_raised} of the {reading.render_calls} "
                   f"turns the trigger fired on. Serving falls back to the static guidance each "
                   f"time, so the rendered guidance would mostly not be what the agent is shown.")
    return out


def dedup(sources: Mapping[str, str], *, existing: Mapping[str, str] | None = None) -> dict[str, str]:
    """Drop every proposal whose check source repeats one kept before it. Returns ``{name: reason}``.

    ``sources`` maps proposal names to check sources, in proposal order; ``existing`` (installed
    and held triggers) is kept first and never dropped.
    """
    kept = {code.strip(): name for name, code in (existing or {}).items()}
    dropped: dict[str, str] = {}
    for name, code in sources.items():
        key = code.strip()
        if key in kept:
            dropped[name] = f"its check is byte-identical to {kept[key]}'s"
        else:
            kept[key] = name
    return dropped
