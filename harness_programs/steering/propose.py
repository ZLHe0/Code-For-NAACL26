"""ProposeSteer (Algorithm 1, line 6): failure hypotheses from the mixed-outcome groups of R_k.

One creator-model call per sampled group (:mod:`.reservoir`) returns one :class:`Hypothesis`:
a ``name``, the corrective ``insight``, the trigger ``check(ep)`` (run at serving time on the
episode prefix), an optional ``fault(ep)`` (the failure, run offline on the complete episode),
and the ``guidance`` written from the insight, optionally rendered by ``render(ep)``.

Program admission happens here, while the exchange is still open: every program is screened
statically (:func:`.screen.static_screen`) and executed once on sample episodes
(:func:`smoke`); a hypothesis with complaints is shown them and asked to revise, at most
:data:`DEFAULT_MAX_REVISIONS` times, and dropped if still refused. Proposals whose check repeats
an installed, held or earlier proposal byte for byte are dropped too (:func:`.screen.dedup`).
The admitted hypotheses' triggers form the round's proposal set P.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..llm import ChatModel, LLMError, complete, extract_json
from ..serving.steer import (
    Episode,
    Trigger,
    compile_fault,
    compile_render,
    compile_trigger,
    sample_episode,
)
from . import prompts
from .reservoir import Group, render_group
from .screen import dedup, gold_strings, static_screen

logger = logging.getLogger(__name__)

MAX_REVISIONS_ENV = "HARNESS_STEERING_MAX_REVISIONS"
DEFAULT_MAX_REVISIONS = 2
DEFAULT_WORKERS = 4
NAME_SEPARATOR = "__"


def max_revisions() -> int:
    """Repair rounds after the first reply; overridable through the environment."""
    raw = os.environ.get(MAX_REVISIONS_ENV, "")
    try:
        return max(0, int(raw)) if raw.strip() else DEFAULT_MAX_REVISIONS
    except ValueError:
        return DEFAULT_MAX_REVISIONS


@dataclass
class Hypothesis:
    """One failure hypothesis: its insight and the trigger written from it."""

    name: str
    insight: str
    code: str
    guidance: str
    fault: str = ""
    render: str = ""
    group: str = ""
    gold_strings: list[str] = field(default_factory=list, repr=False)

    def to_trigger(self) -> Trigger:
        return Trigger(name=self.name, code=self.code, guidance=self.guidance, fault=self.fault,
                       render=self.render)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("gold_strings")
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Hypothesis:
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


def reply_render(reply: dict[str, Any]) -> str:
    """The optional render source, or '' when the reply declined to write one.

    A model told a field is optional often fills it with a placeholder; anything that does not
    define ``render`` is read as absent rather than refused.
    """
    text = str(reply.get("render") or "").strip()
    return text if "def render" in text else ""


def reply_fault(reply: dict[str, Any]) -> str:
    """The optional fault source, or '' when the reply declined to write one."""
    text = str(reply.get("fault") or "").strip()
    return text if "def fault" in text else ""


def hypothesis_from(reply: dict[str, Any], *, group: str) -> Hypothesis | None:
    """A hypothesis from a parsed reply, or ``None`` for a skip or an incomplete reply."""
    if reply.get("skip"):
        logger.info("%s: skipped -- %s", group, reply["skip"])
        return None
    missing = [k for k in ("name", "insight", "check", "guidance") if not str(reply.get(k) or "").strip()]
    if missing:
        logger.info("%s: reply omitted %s", group, ", ".join(missing))
        return None
    return Hypothesis(name=str(reply["name"]).strip(), insight=str(reply["insight"]).strip(),
                      code=str(reply["check"]), guidance=str(reply["guidance"]).strip(),
                      fault=reply_fault(reply), render=reply_render(reply), group=group)


# ---------------------------------------------------------------------------------------------
# Program admission, inside the exchange
# ---------------------------------------------------------------------------------------------

def _repair_advice(exc: Exception) -> str:
    if isinstance(exc, NameError):
        return ("The check, the fault and the render are compiled separately and cannot call each "
                "other; a helper any of them uses must be defined in its own source.")
    if isinstance(exc, AttributeError):
        return 'Use only the attributes listed under "The episode object"; nothing else exists.'
    return "Guard every index and attribute that may be empty or None."


def smoke(hypothesis: Hypothesis, *, gold_views: Sequence[Any] = (None,)) -> list[str]:
    """Run the programs once on the episodes the replay will run them on, and say how they broke.

    The trigger and render run on an empty prefix (before the first turn) and on the two-step
    sample; the fault on the complete sample under each gold view and under both an empty and a
    non-empty final output.
    """
    empty = Episode(task_text="Sample task statement.", steps=[], max_steps=10)
    places = (("an episode with no steps yet (a trigger runs before EVERY turn, including the "
               "first)", empty), ("a two-step sample episode", sample_episode()))
    out: list[str] = []
    for role, source, compiler in (("check", hypothesis.code, compile_trigger),
                                   ("render", hypothesis.render, compile_render)):
        if not (source or "").strip():
            continue
        for where, episode in places:
            try:
                result = compiler(source)(episode)
                if role == "render" and not isinstance(result, str):
                    raise TypeError(f"render returned {type(result).__name__}, not a str")
            except Exception as exc:  # noqa: BLE001 -- every failure is a complaint to repair
                out.append(f"{role}(ep) raised {type(exc).__name__}: {exc} on {where}. "
                           f"{_repair_advice(exc)}")
                break
    if (hypothesis.fault or "").strip():
        complaint = _fault_smoke(hypothesis.fault, gold_views)
        if complaint:
            out.append(complaint)
    return out


def _fault_smoke(source: str, gold_views: Sequence[Any]) -> str:
    for view in gold_views:
        for answer in ("", "x"):
            episode = sample_episode(gold=view)
            episode.final_output = answer
            episode.status = "answered" if answer else "max_turns"
            try:
                compile_fault(source)(episode)
            except Exception as exc:  # noqa: BLE001
                return (f"fault(ep) raised {type(exc).__name__}: {exc} on a complete episode with "
                        f"final_output={answer!r} (always a str). {_repair_advice(exc)}")
    return ""


def complaints(hypothesis: Hypothesis, *, gold_views: Sequence[Any] = (None,)) -> list[str]:
    """Static complaints first; the execution smoke only for a hypothesis that screens clean."""
    found = static_screen(hypothesis.code, hypothesis.fault, hypothesis.guidance, hypothesis.render,
                          gold=hypothesis.gold_strings)
    return found or smoke(hypothesis, gold_views=gold_views)


def _ask(llm: ChatModel, prompt: str, system: str, label: str) -> dict[str, Any] | None:
    try:
        return extract_json(complete(llm, prompt, system=system))
    except LLMError as exc:
        logger.info("%s: unusable reply (%s)", label, exc)
        return None


def screened(llm: ChatModel, user: str, *, system: str,
             build: Callable[[dict[str, Any]], Hypothesis | None], label: str,
             gold_views: Sequence[Any] = (None,), revisions: int | None = None) -> Hypothesis | None:
    """Ask once, then let the model repair its own answer against program admission.

    A hypothesis returned here is servable and non-degenerate on its source alone; every
    behavioural question still needs a replay.
    """
    revisions = max_revisions() if revisions is None else revisions
    reply = _ask(llm, user, system, label)
    hypothesis = build(reply) if reply else None
    for _ in range(revisions):
        if hypothesis is None:
            return None
        found = complaints(hypothesis, gold_views=gold_views)
        if not found:
            return hypothesis
        logger.info("%s/%s: revising -- %s", label, hypothesis.name, "; ".join(found))
        prompt = prompts.fill(prompts.REVISE, {
            "name": hypothesis.name, "complaints": "\n".join(f"- {c}" for c in found),
            "fault": hypothesis.fault or "(none)", "check": hypothesis.code,
            "guidance": hypothesis.guidance})
        if hypothesis.render:
            prompt += prompts.fill(prompts.REVISE_RENDER, {"render": hypothesis.render})
        reply = _ask(llm, prompt, system, label)
        hypothesis = build(reply) if reply else None
    if hypothesis is None:
        return None
    found = complaints(hypothesis, gold_views=gold_views)
    if found:
        logger.info("%s/%s: refused after revision -- %s", label, hypothesis.name, "; ".join(found))
        return None
    return hypothesis


# ---------------------------------------------------------------------------------------------
# One round of proposals
# ---------------------------------------------------------------------------------------------

def propose(llm: ChatModel, group: Group, *, text: prompts.DomainText,
            revisions: int | None = None) -> Hypothesis | None:
    """One call (plus repairs) per group: an admitted hypothesis, or nothing."""
    strings = sorted(gold_strings(group.gold)) if group.gold is not None else []

    def build(reply: dict[str, Any]) -> Hypothesis | None:
        hypothesis = hypothesis_from(reply, group=group.key)
        return replace(hypothesis, gold_strings=strings) if hypothesis else None

    return screened(llm, render_group(group, scale=text.scale), system=prompts.system_prompt(text),
                    build=build, label=group.key, gold_views=(group.gold,), revisions=revisions)


def uniquify(hypotheses: Sequence[Hypothesis], *, taken: Sequence[str] = ()) -> list[Hypothesis]:
    """Give every proposal a name of its own.

    Two groups showing the same behaviour get the same name from the model, which a replay
    cannot accept. Every member of a collision is suffixed with its group, not just the later
    ones, so no proposal is privileged by read order. A name that collides with ``taken``
    (installed or held triggers) is suffixed too.
    """
    counts: dict[str, int] = {}
    for h in hypotheses:
        counts[h.name] = counts.get(h.name, 0) + 1
    reserved = set(taken)
    return [replace(h, name=f"{h.name}{NAME_SEPARATOR}{h.group}")
            if counts[h.name] > 1 or h.name in reserved else h for h in hypotheses]


@dataclass
class Proposals:
    """The proposal set P of one round, and what was dropped on the way."""

    hypotheses: list[Hypothesis]
    groups: list[str]
    duplicates: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"groups": self.groups, "hypotheses": [h.to_dict() for h in self.hypotheses],
                "duplicates": self.duplicates}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Proposals:
        return cls([Hypothesis.from_dict(h) for h in row["hypotheses"]], list(row["groups"]),
                   dict(row.get("duplicates") or {}))


def propose_steer(llm: ChatModel, groups: Sequence[Group], *, text: prompts.DomainText,
                  existing: Mapping[str, str] | None = None, workers: int = DEFAULT_WORKERS,
                  revisions: int | None = None) -> Proposals:
    """ProposeSteer: one admitted hypothesis per group at most, named uniquely and deduplicated.

    ``existing`` maps installed and held trigger names to their check sources.
    """
    def one(group: Group) -> Hypothesis | None:
        return propose(llm, group, text=text, revisions=revisions)

    if workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            proposed = list(pool.map(one, groups))
    else:
        proposed = [one(g) for g in groups]
    named = uniquify([h for h in proposed if h is not None], taken=list(existing or {}))
    duplicates = dedup({h.name: h.code for h in named}, existing=existing)
    kept = [h for h in named if h.name not in duplicates]
    logger.info("%d group(s) -> %d hypothesis(es), %d duplicate(s) dropped",
                len(groups), len(kept), len(duplicates))
    return Proposals(kept, [g.key for g in groups], duplicates)
