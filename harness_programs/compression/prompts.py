"""The creator's prompts for ProposeCompress (Algorithm 1, line 2): draft, repair, accept or revise.

The prompts state the objective, how serving runs the program, and the two ways a program
fails. They never say which content of an observation is worth keeping: that is for the
creator to infer from the contrast between successful and failed attempts, so the same
prompts serve every domain.
"""

from __future__ import annotations

from typing import Any

from ..llm.client import ChatModel, LLMError, complete, extract_json

SERVING = """\
How serving works:
- Before every policy call the program `compress(state)` runs ONCE on the whole transcript so
  far. `state` is the list of turns, oldest first, in the format of the contract below. The
  same function runs on every turn of every episode.
- After it, eviction drops whole oldest turns until the prompt fits the context limit. The
  pinned head (system prompt and task statement) is never dropped and never rewritten.
- The LAST turn in `state` is the observation the policy must answer now. It is not history;
  leave it untouched.
- Every action the agent took must be served unchanged; compress observations and rationale.
- Compressing a turn that eviction drops anyway buys nothing. Room comes from making the turns
  that are kept cheaper, so that older turns survive eviction."""

FAILURES = """\
A program fails in one of two ways, and they pull in opposite directions:
1. UNFAITHFUL -- it removes information the policy needs later. Whatever it drops is gone from
   the policy's context for the rest of the episode.
2. INERT -- it leaves the served context largely unchanged, so eviction does the work and the
   policy gets no more history than with no program at all.
Removing text is not the goal; it is the price paid for room. The way between the two failures
is to find content that is redundant given the rest of the transcript -- repeated, superseded
by a later turn, or never used again -- and compress that hard while keeping the evidence the
policy acts on readable."""

CREATOR_SYSTEM = f"""\
You write CONTEXT-COMPRESSION PROGRAMS for an agent whose context is limited. The goal is the
policy's reward: the program should keep the evidence the policy needs readable while freeing
room so that more of its history fits the limit.

{SERVING}

{FAILURES}

You are shown attempts of the same policy on the same task where some succeeded and some
failed. Contrast them: identify the evidence the successful attempts used and the failed
attempts lost, then write programs that keep exactly that kind of evidence readable and
compress the rest. Supporting measurements show the real format and sizes of the transcript,
so your code can parse what is actually served."""

REVIEW_SYSTEM = f"""\
You are reviewing a context-compression program you wrote, after it has been RUN on real agent
trajectories. You see what it actually did to the context and two measurements.

{SERVING}

{FAILURES}

UNFAITHFUL is measured directly: the share of items the agent needs later (the answer, the
words of its later actions) that are readable without compression and gone after it. INERT is
measured directly: the share of episodes under context pressure on which the served context is
unchanged or nearly so. Preserving everything is not a safe answer -- it is the second failure.

Loading and running without error says nothing about whether the program does the right thing.
Accept only if it is both faithful and materially changes what is served; otherwise revise it,
keeping the idea and fixing its execution.

Return JSON ONLY:
{{"verdict": "accept" | "revise", "diagnosis": "<=60 words, what the evidence shows",
 "code": "def compress(state):\\n    ...\\n    return state"}}
On "accept" the code is ignored. On "revise", return the complete corrected program."""

REPAIR_PROMPT = """\
This compression program does not run under the serving contract:

```python
{code}
```

Errors:
{errors}

Fix it without changing its idea. Return JSON ONLY:
{{"code": "def compress(state):\\n    ...\\n    return state"}}"""


def ask(llm: ChatModel, prompt: str, system: str) -> dict[str, Any]:
    """One JSON reply, or ``{}`` when nothing parseable came back."""
    try:
        return extract_json(complete(llm, prompt, system=system))
    except LLMError:
        return {}
