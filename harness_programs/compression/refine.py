"""The refine loop of ProposeCompress (Algorithm 1, line 2): repair until it runs, then accept or revise.

Each candidate runs on stored episodes from R0 (:func:`~.measure.measure`). While it breaks
the serving contract -- does not compile, raises, drops an action, or edits the live turn --
the error goes back to the creator for repair, at most ``MAX_REPAIRS`` times in a row. Once it
runs, the creator sees the contexts before and after compression with the unfaithful and inert
measurements, and accepts the candidate or revises it. A candidate not accepted within
``rounds`` rounds, or that never runs, is dropped.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm.client import ChatModel
from .episodes import OptEpisode, Target
from .measure import measure, render_measurement, render_samples
from .prompts import CREATOR_SYSTEM, REPAIR_PROMPT, REVIEW_SYSTEM, ask

logger = logging.getLogger(__name__)

DEFAULT_ROUNDS = 3
MAX_REPAIRS = 2
PARALLEL = 4


def review_prompt(target: Target, candidate: dict[str, Any], m: dict[str, Any]) -> str:
    return f"""\
{target.desc}

{target.contract}

Name: {candidate['name']}
Stated intent: {candidate.get('rationale', '')}

```python
{candidate['code']}
```

{render_measurement(m)}

Real turns from stored episodes, BEFORE and AFTER the program ran: the ones it shrank most,
one per distinct pattern.

{render_samples(m)}

Accept the program, or revise it if a sample drops information the agent would plausibly need
later, or the measurements show it barely changes what the policy is served."""


def until_runs(llm: ChatModel, target: Target, code: str, eps: list[OptEpisode], record: dict[str, Any]
               ) -> tuple[str, dict[str, Any]]:
    """Measure ``code``, handing contract violations back to the creator until it runs or repairs run out."""
    m = measure(target, code, eps)
    for _ in range(MAX_REPAIRS):
        if not m["violations"]:
            break
        errors = "\n".join(f"- {v}" for v in m["violations"])
        fixed = ask(llm, REPAIR_PROMPT.format(code=code, errors=errors), CREATOR_SYSTEM).get("code")
        record["repairs"] += 1
        if not fixed or fixed == code:
            break
        code, m = fixed, measure(target, fixed, eps)
    return code, m


def refine_one(llm: ChatModel, target: Target, candidate: dict[str, Any], eps: list[OptEpisode],
               rounds: int = DEFAULT_ROUNDS) -> dict[str, Any]:
    """One candidate through the loop; ``record["accepted"]`` says whether it joins P."""
    code = candidate["code"]
    record: dict[str, Any] = {"name": candidate["name"], "rationale": candidate.get("rationale", ""),
                              "draft": code, "code": code, "measurement": None, "rounds": [], "repairs": 0,
                              "accepted": False, "dropped": ""}
    for _ in range(rounds):
        code, m = until_runs(llm, target, code, eps, record)
        record.update(code=code, measurement=m)
        if m["violations"]:
            record["dropped"] = "does not run: " + "; ".join(m["violations"])
            return record
        reply = ask(llm, review_prompt(target, {**candidate, "code": code}, m), REVIEW_SYSTEM)
        verdict, revised = reply.get("verdict") or "no reply", reply.get("code") or ""
        record["rounds"].append({"verdict": verdict, "diagnosis": reply.get("diagnosis", "")})
        if verdict == "accept":
            record["accepted"] = True
            return record
        if not revised or revised == code:
            break
        code = revised
    record["dropped"] = f"not accepted within {len(record['rounds'])} round(s)"
    return record


def run(llm: ChatModel, target: Target, eps: list[OptEpisode], candidates: list[dict[str, Any]], *,
        rounds: int = DEFAULT_ROUNDS) -> list[dict[str, Any]]:
    """Refine every candidate in parallel."""
    with ThreadPoolExecutor(max_workers=min(PARALLEL, max(len(candidates), 1))) as pool:
        records = list(pool.map(lambda c: refine_one(llm, target, c, eps, rounds), candidates))
    for r in records:
        logger.info("%s/%s: %s after %d round(s), %d repair(s)", target.domain, r["name"],
                    "accepted" if r["accepted"] else f"dropped ({r['dropped']})", len(r["rounds"]), r["repairs"])
    return records
