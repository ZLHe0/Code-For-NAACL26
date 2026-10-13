"""ProposeCompress (Algorithm 1, line 2): candidate programs from R0, refined until accepted.

The creator is shown successful and failed attempts on the same tasks (:mod:`.evidence`),
identifies the evidence the successes used and the failures lost, and drafts several
candidate ``compress(state)`` programs. Each is refined (:mod:`.refine`) on stored episodes
against the unfaithful and inert measurements (:mod:`.measure`); the accepted programs form
the proposal set P.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..llm.client import ChatModel
from . import refine
from .episodes import OptEpisode, Target
from .evidence import creator_prompt
from .gate import NAIVE_NAME
from .prompts import CREATOR_SYSTEM, ask

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES = 3


def _name(raw: Any, taken: set[str]) -> str:
    base = re.sub(r"[^\w.-]", "_", str(raw or "candidate"))[:60].strip("._") or "candidate"
    name, k = base, 1
    while name in taken or name == NAIVE_NAME:
        k += 1
        name = f"{base}-{k}"
    return name


def draft(llm: ChatModel, target: Target, eps: list[OptEpisode], n: int = DEFAULT_CANDIDATES
          ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``n`` distinct candidates from one creator call, and the findings and evidence behind them."""
    prompt, evidence = creator_prompt(target, eps, n)
    reply = ask(llm, prompt, CREATOR_SYSTEM)
    candidates: list[dict[str, Any]] = []
    for p in reply.get("proposals") or []:
        if not isinstance(p, dict) or not p.get("code"):
            continue
        if any(c["code"].strip() == p["code"].strip() for c in candidates):
            continue
        name = _name(p.get("name"), {c["name"] for c in candidates})
        candidates.append({"name": name, "rationale": str(p.get("rationale", "")), "code": p["code"]})
    findings = reply.get("findings") or []
    return candidates[:n], {"findings": findings if isinstance(findings, list) else [findings], "evidence": evidence}


def run(llm: ChatModel, target: Target, eps: list[OptEpisode], *, candidates: int = DEFAULT_CANDIDATES,
        rounds: int = refine.DEFAULT_ROUNDS) -> dict[str, Any]:
    """Draft, refine, and return the proposal set P under ``"accepted"``."""
    drafted, info = draft(llm, target, eps, candidates)
    if not drafted:
        raise RuntimeError(f"{target.domain}: the creator returned no parseable candidate programs")
    records = refine.run(llm, target, eps, drafted, rounds=rounds)
    accepted = [{"name": r["name"], "rationale": r["rationale"], "code": r["code"]} for r in records if r["accepted"]]
    logger.info("%s: %d of %d candidates accepted", target.domain, len(accepted), len(records))
    return {"domain": target.domain, "sizing": target.sizing.label, "episodes": len(eps), **info,
            "candidates": records, "accepted": accepted}
