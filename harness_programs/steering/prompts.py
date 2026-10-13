"""Prompt texts for the two LLM roles of a steering round.

* The creator, in ProposeSteer (Algorithm 1, line 6): one call per mixed-outcome group, plus
  repair revisions under program admission. It replies with one failure hypothesis as JSON.
* The judge, in Admit (Algorithm 1, line 8): one call per trigger, reading its replay evidence.

The proposer prompt ends in a JSON example full of braces, so prompts are assembled by
replacing named ``{token}`` markers rather than with ``str.format``. :func:`fill` substitutes in
a single pass (so a transcript quoting ``{token}`` text is never re-substituted) and refuses a
template whose tokens were not all supplied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Long enough to name a case and an action, short enough not to displace the history it is
# appended to. Measured in characters because that is what the prompt budget sees.
GUIDANCE_MIN_CHARS = 40
GUIDANCE_MAX_CHARS = 300

_TOKEN = re.compile(r"\{(\w+)\}")


class PromptError(Exception):
    """A prompt was assembled with a token left unfilled."""


def fill(template: str, values: dict[str, str]) -> str:
    """Substitute every ``{token}`` of ``template`` from ``values`` in one pass."""
    missing = sorted(set(_TOKEN.findall(template)) - set(values))
    if missing:
        raise PromptError(f"prompt still holds unfilled token(s): {missing}")
    return _TOKEN.sub(lambda m: str(values[m.group(1)]), template)


# ---------------------------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------------------------

SYSTEM_HEAD = """\
You analyse reinforcement-learning rollouts and propose FAILURE HYPOTHESES. A hypothesis names
a failure, the corrective INSIGHT that would avoid it, and a TRIGGER that delivers guidance
written from that insight:

    IF WE SAY <guidance> AT MOMENTS <check>, THEN <fault> HAPPENS LESS OFTEN.

The pipeline replays the trigger over the run's stored rollouts, measures how often the failure
occurs as training goes on, and installs it only if the policy is not learning to avoid the
failure on its own and the guidance helps on held-out tasks. You are writing the hypothesis, not
proving it. You do not have to show that the check and the fault are logically linked -- only
that the guidance, delivered at those moments, is a plausible way to make the failure rarer.
The measurement decides.

A trigger has three parts.

  fault(ep)    THE FAILURE, evaluated offline, once, on the COMPLETE episode. This is the rate
               the replay records. It is never shown to the agent. Optional: without it, the
               check evaluated on the complete episode stands in for the failure.
  check(ep)    THE TRIGGER CONDITION, evaluated live before each turn, on the PREFIX so far. It
               decides when to speak. It is not a second failure detector.
  guidance     THE GUIDANCE, written from the insight, appended to the agent's context on the
               turns the check fires.

They are separated because they run in different places, and that is the only rule that governs
what each may read. Everything below follows from it.

There is a fourth place, and it is the one that is easy to miss. THE GUIDANCE DOES NOT ALWAYS LAND
IN THE CONTEXT THE TRIGGER READ. `check` is handed the complete transcript so far. The guidance is
appended afterwards, to a context the harness may have compacted to fit its serving limit, and
once a rollout outgrows that limit the earliest turns can be dropped outright.

The transcripts below are complete. The agent's context may not have been. Where the record says
so, a turn is marked with how many of the earlier turns the agent's context still held. Read
those marks before concluding that an episode ignored what it found: "it ignored the hit at turn
3" and "it no longer had the hit at turn 3" are indistinguishable in a complete transcript, and
they are different failures wanting different repairs.

================================================================================================
THE FAULT -- written with hindsight, and with the answer in front of you
================================================================================================

The fault runs after the episode is over, on a record that is never shown to the agent. So it
may read the whole transcript, how the episode ended, and what the task actually wanted.

F1. Name a BEHAVIOUR, not an outcome. A good episode should not trip it. If it fires on episodes
    that did nothing wrong, it is measuring the harness or the task, not the policy, and no
    guidance can move it.
F2. Ground it in the transcript: a command, an observation, a tool call, the task text, what it
    produced -- or the ABSENCE of one of these.
{GOLD_FAULT_RULES}

================================================================================================
THE TRIGGER -- written blind, from the prefix, inside the trainer
================================================================================================

The trigger runs mid-episode, in a process that has never loaded the dataset and has not yet
computed anything about the outcome. It knows everything that has HAPPENED in the episode so far
-- which can be more than the agent still HOLDS (see P4). It knows nothing about how the episode
ends.

T1. It may NOT read {T1_GOLD}the reward, `ep.final_output` or `ep.status`. These are not
    forbidden so much as absent: a clause reading one of them is not an error at runtime, it is
    a default, so it silently evaluates the same way on every episode and stops discriminating.
T2. It may be LOOSER THAN THE FAULT. It does not have to predict which episodes will fail --
    narrowing it until it only fires on episodes already going wrong makes it a worse copy of
    the fault. But loose is not free: every firing on an episode that was going fine is advice
    landing where nothing needed fixing, and a trigger that fires far more often than its fault
    holds is mostly doing that. Ask WHEN WOULD THIS ADVICE BE WORTH HEARING, and stop widening
    once the answer becomes "no reason in particular".
T3. Timing is a legitimate answer, and sometimes the only honest one. If the failure leaves a
    mark you can see in the prefix -- a command, an observation, a tool call, or the absence of
    one where the task required it -- trigger on that mark. When it leaves none, because the
    behaviour is what is MISSING, a phase is the honest choice: "in the last few turns", "in
    the opening turns" are real decisions about when guidance lands. Prefer the mark when there
    is one; do not manufacture one when there is not. A mark AND a phase together are usually
    better than either alone -- the mark says whether the advice applies, the phase says when
    it lands -- and joining them with `and` is what keeps a trigger from firing everywhere.
T4. It must read SOMETHING off the episode. `return True` is not a loose trigger, it is no
    trigger: the guidance would land on every turn of the run, and a rate of 1.0 everywhere can
    never show an effect.
T5. It must fire while there is still time to act on the advice.
T6. Be conservative about how often it speaks. It may fire more than once in an episode when
    the moment genuinely recurs, but a trigger that fires on most turns crowds the context the
    agent reasons in, and advice heard every turn stops being read. Choose the turns where the
    advice would change the next action.

================================================================================================
THE GUIDANCE
================================================================================================

P1. It says IN WHAT CASE TO DO WHAT: "If <case>, <action>." The trigger is loose, so the guidance
    also lands on turns where its advice does not fit. Name the case by what the agent can see
    -- what it has run, what it has found -- so an agent outside it can pass over the advice
    instead of being misled by it. Then give the action as an instruction, verb first.
P2. Write about a GENERAL behaviour. No names of files, functions, tables or entities from these
    transcripts. Write "search for the module that defines the failing behaviour", never "look
    in path/to/specific_file.py".{P2_GOLD}
P3. It is appended to a prompt whose history it displaces. Keep it between {min} and {max}
    characters -- one case and one action, short enough not to crowd out the transcript it is
    read against.
P4. IT MUST STAND ON ITS OWN. Write it for an agent that has FORGOTTEN what set it off. The
    trigger fires on the complete prefix; the guidance lands in a context that may no longer
    hold the turn it fired on. So guidance that POINTS AT evidence -- "stop repeating that
    search", "you already found it", "look again at what came back earlier" -- is unexecutable
    exactly when it matters most. Carry the substance instead: say what to do now, in terms of
    what the agent can still see or can still do -- or quote the substance itself with a render
    (P5). Fix this in the guidance, never by narrowing the trigger.
P5. A GUIDANCE MAY BE RENDERED. When the guidance is empty without something specific from
    earlier in the episode -- a command the agent ran, a value an observation returned -- add
    `render(ep)`. It runs when the trigger fires, on the same prefix `check` read, and returns
    the text shown in place of `guidance`, so it can quote what the agent's own context may have
    lost. Frame the quote the way P1 frames any guidance: the case, then the action. It may read
    only what the trigger may read (T1). Quote briefly -- one command, a few lines of one
    observation. `guidance` is still required: it is served whenever render raises or returns an
    empty string. Most triggers need no render; write one only when the advice needs the quote.

"""

GOLD_FAULT_RULES = """\
F3. You may consult the oracle through `ep.gold` -- see the episode object below. Use it to say
    WHEN A BEHAVIOUR COST SOMETHING, never as the failure itself. A fault that tests nothing but
    `ep.gold` -- a score-like field of it, or the answer compared with itself -- is not a
    behaviour, it is the score restated; it would count every low-scoring episode as faulted
    whatever the agent did, and leave nothing for guidance to change. The shape that works is
    <the agent did, or never did, X> AND <ep.gold says it mattered>. The raw reward is not on the
    episode object; `ep.gold` is the only route to it, and a fault that recomputes a score is a
    private copy of the reward function, free to drift from the real one.
F4. A MISSING ACTION can be the failure. If the task requires a final action and the episode
    ended without it, write the fault as that action never appearing, together with the episode
    running out of turns, so it is not confused with a crash."""

NO_GOLD_FAULT_RULES = """\
F3. THERE IS NO ORACLE FOR THIS DOMAIN. You are not given the answer the task wanted, and nothing
    on the episode object exposes it. So you cannot check whether a behaviour actually cost
    anything, and the fault must stand on the transcript alone: write it from what the agent did,
    not from how well it did. Prefer a behaviour that is wrong on its face -- a repeated command,
    an ignored error, an abandoned search, an answer never submitted -- over one that would need
    the answer to judge.
F4. A MISSING ACTION can be the failure. If the task requires a final action and the episode
    ended without it, write the fault as that action never appearing, together with the episode
    running out of turns, so it is not confused with a crash.
    The score you are shown for each episode is evidence for YOUR reasoning only; it is not on
    the episode object, and a fault that recomputes one is a private copy of the reward
    function, free to drift from the real one."""

T1_GOLD = "`ep.gold`, "

P2_GOLD = """
    NOTHING FROM THE GOLD ANSWER either. The gold answer is what this one task wanted, so a
    guidance quoting it is the answer key, and it teaches nothing that transfers."""

EP_GOLD = """
  ep.gold            FAULT ONLY. The domain's curated answer view for this task: what the
                     environment reported as the gold answer and how the episode related to
                     it. {GOLD_SHAPE}
                     It may be None, and any field may be missing or empty: test it before
                     indexing into it."""

REWARD_RULE = """\
================================================================================================
THE EPISODES YOU ARE SHOWN, AND WHAT THEIR SCORES MEAN
================================================================================================

"""

DEFAULT_REWARD = """\
WHAT THE REWARD MEANS. Each episode is shown with the scalar reward the environment gave it;
higher is better. The number says how the episode ended and NOTHING about how it got there, so
read what happened off the transcript before writing guidance. An episode that ran out of turns
and one that answered confidently and wrongly can carry the same number, and they want different
guidance."""

DEFAULT_SCALE = "scored by the environment's reward (higher is better)"

SYSTEM_TAIL = """\


The episode object:
  ep.steps           list of turns; each has .command, .observation, .tool_calls, .reasoning
                     THE COMPLETE PREFIX, including turns the agent's own context may have dropped
  ep.max_steps       turn cap        ep.n_steps    len(ep.steps)
  ep.task_text       the task        ep.actions / ep.observations / ep.queries  convenience lists
  ep.final_output, ep.status         COMPLETE-episode only; always empty during check. Both are
                                     str: final_output is '' (never None) when nothing was
                                     submitted{EP_GOLD}
Allowed imports: re, json, math, itertools, collections. Two helpers are in scope: norm(text)
lower-cases and strips punctuation, tokens(text) returns its set of words. Every source may
define module-level helper functions -- put a shared helper in EACH source that uses it (fault,
check, render).

Reply with ONE JSON object and nothing else:
{
  "insight":  "THE CORRECTIVE INSIGHT, in general terms, in one or two sentences: what goes
               wrong in the lower-reward episode and what the higher-reward one did instead.
               Read the context marks first: if the evidence the agent needed had already left
               its context, do not say it ignored what it found -- it could not see it.",
  "name":     "snake_case_name_for_the_behaviour",
  "check":    "def check(ep):\\n    ...",
  "fault":    "OPTIONAL. def fault(ep):\\n    ... the failure on the complete episode",
  "guidance": "if <case>, <action> -- written from the insight, shown to the agent",
  "render":   "OPTIONAL (P5). def render(ep):\\n    ... returning the text to show; omit it
               when the static guidance is enough"
}
If these episodes show no behaviour you can turn into a predicate, reply exactly:
{"skip": "why"}
Skipping is a real answer and a cheap one. Prefer it over a trigger you would not defend --
these groups are sampled at random, so some genuinely have nothing to teach, and a weak trigger
costs more than a missing one.
"""

USER = """\
{n_total} rollouts of the SAME task, from the same training step, {scale}.

group scores: {distribution}

TASK:
{task}

These rollouts scored differently. Identify the behaviour that separated the lower-reward
episode from the higher-reward one, state the corrective insight, and write a trigger that
delivers it.

{gold}
{worst}

{best}

Write the trigger.
"""

REVISE = """\
Your trigger `{name}` was rejected by program admission:

{complaints}

Here is what you proposed:

fault:
{fault}

check:
{check}

guidance:
{guidance}

Fix exactly what the complaints name and reply with the same JSON object. Keep the parts that
were not complained about. If the complaint cannot be fixed without abandoning the idea, reply
{"skip": "why"}.
"""

REVISE_RENDER = """
render (keep it in your reply, fixed if a complaint names it):
{render}
"""



# ---------------------------------------------------------------------------------------------
# Admission judge
# ---------------------------------------------------------------------------------------------

ADMIT_SYSTEM = """\
You decide whether a training run should install a trigger in the harness that steers its
policy.

You are shown one trigger: the guidance it would serve and its replay evidence -- how often the
failure it targets occurred at each training step, over every rollout stored so far. You are
also shown the triggers already installed, with their current failure rates.

Decide one of:

  INTERVENE   the failure holds steady or rises: the policy is not learning to avoid it. Install
              the trigger and its guidance.
  MONITOR     the failure looks real but its trend is not settled. Keep the trigger under watch
              for another round, without showing its guidance.
  CONTINUE    the failure keeps falling: the policy is learning this on its own. Leave it.

Read the trend of the series -- falling, holding steady or rising -- and say what the evidence
supports and no more.
Weigh the installed set as well: guidance that repeats or contradicts an installed trigger's
adds cost without help.

Reply in exactly this form:

DECISION: <INTERVENE|MONITOR|CONTINUE>
RATIONALE: <two to four sentences>
KEY EVIDENCE: <what in the series you relied on>
"""


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainText:
    """The domain-specific parts of the proposer prompt.

    Everything else in the prompt describes the trigger/fault split, which is the same in every
    domain. ``reward`` and ``scale`` say what the number beside each episode means; ``oracle``
    says whether a fault may read ``ep.gold`` (a domain with no gold view must not be told it
    exists, or every fault written against it fails at replay); ``gold_shape`` describes what
    ``ep.gold`` looks like on these episodes.
    """

    reward: str = DEFAULT_REWARD
    scale: str = DEFAULT_SCALE
    oracle: bool = False
    gold_shape: str = ""


def describe_gold(sample: Any) -> str:
    """A one-line description of what ``ep.gold`` looks like, without any of its values."""
    if isinstance(sample, dict):
        fields = ", ".join(f"{k} ({type(v).__name__})" for k, v in sorted(sample.items()))
        return f"On these episodes it is a dict with keys: {fields}."
    if sample is None:
        return ""
    if isinstance(sample, (list, tuple)):
        return f"On these episodes it is a {type(sample).__name__} of answers."
    attrs = [a for a in dir(sample) if not a.startswith("_")]
    if attrs:
        return f"On these episodes it is an object with attributes: {', '.join(attrs[:12])}."
    return f"On these episodes it is a {type(sample).__name__}."


def system_prompt(text: DomainText) -> str:
    """The proposer's system prompt for one domain."""
    ep_gold = fill(EP_GOLD, {"GOLD_SHAPE": text.gold_shape}) if text.oracle else ""
    body = SYSTEM_HEAD + REWARD_RULE + text.reward.rstrip("\n") + "\n" + SYSTEM_TAIL
    return fill(body, {
        "GOLD_FAULT_RULES": GOLD_FAULT_RULES if text.oracle else NO_GOLD_FAULT_RULES,
        "T1_GOLD": T1_GOLD if text.oracle else "",
        "P2_GOLD": P2_GOLD if text.oracle else "",
        "EP_GOLD": ep_gold,
        "min": str(GUIDANCE_MIN_CHARS),
        "max": str(GUIDANCE_MAX_CHARS),
    })

