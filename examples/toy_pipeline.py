"""The whole pipeline on the toy domain, offline, in a few seconds.

    python examples/toy_pipeline.py [--out DIR]

0. Split: carve D_hold out of the toy's training tasks. The policy trains on the rest; the
   evaluation tasks in step 3 are a separate pool that D_hold is checked against.
1. Stage 1: roll out a scripted policy with no program under a tight context limit, propose
   and refine compression programs from those rollouts, and gate them on D_hold, the held-out
   training tasks; emit ``program.json``. The creator LLM is scripted (:class:`ToyCreator`).
2. Stage 2 stand-in: install one steering trigger into the same ``program.json`` -- the
   document a live training run re-reads every episode.
3. Evaluate the policy, on the evaluation tasks, under no program, the trigger alone, the
   stage-1 program, and both.

Swap the scripted creator and policy for :class:`~harness_programs.llm.OpenAIChat` endpoints and the
toy domain for a real one to run the same code for real; see ``scripts/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from harness_programs.compression import Campaign, CampaignConfig, Sizing
from harness_programs.domains import load
from harness_programs.domains.toy import write_split
from harness_programs.domains.toy_agents import UNOPENED_RENDER, UNOPENED_TRIGGER, ToyCreator, toy_policy
from harness_programs.eval import summarize
from harness_programs.llm import ScriptedLLM
from harness_programs.rollout.runner import run_batch
from harness_programs.serving.program import Program, load_program, write_program
from harness_programs.serving.serve import ContextPipeline
from harness_programs.serving.steer import Trigger

SIZING = Sizing(serving_limit=4096, history_budget=350)

# A hand-written trigger of the shape stage 2 generates. Stage 2 mines, screens and admits these
# during training; here one is installed directly to show what serving does with it.
UNOPENED = Trigger(name="unopened_files", code=UNOPENED_TRIGGER, render=UNOPENED_RENDER,
                   guidance="Open a file you have not opened yet.")


def evaluate(program: Path | None) -> dict:
    """Score a program on the evaluation tasks, which never overlap D_hold."""
    domain = load("toy")
    pipeline = ContextPipeline("toy", program_path=program, token_counter=SIZING.counter,
                               serving_limit=SIZING.serving_limit, history_budget=SIZING.history_budget,
                               max_steps=domain.max_turns)
    records = run_batch(domain, domain.load_tasks("synthetic", split="val"), ScriptedLLM(toy_policy),
                        pipeline=pipeline, workers=1)
    return summarize(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logging.getLogger("harness_programs.serving.budget").setLevel(logging.ERROR)
    out = Path(args.out or tempfile.mkdtemp(prefix="harness-toy-"))

    train_tasks, holdout = write_split(out / "data", n_holdout=8)
    config = CampaignConfig(domain="toy", out_dir=str(out / "stage1"), sizing=SIZING,
                            tasks_path=str(train_tasks), holdout_path=str(holdout),
                            eval_tasks=("synthetic",), rollout_tasks=16, group_size=2, workers=1)
    result = Campaign(config, llm=ToyCreator(), policy=ScriptedLLM(toy_policy)).run()
    program_path = Path(result["program"])

    staged = out / "stage2" / "program.json"
    program = load_program(program_path)
    program.triggers.append(UNOPENED)
    write_program(staged, program)
    trigger_only = out / "trigger_only" / "program.json"
    write_program(trigger_only, Program(harness="toy", triggers=[UNOPENED]))

    rows = {"no program": evaluate(None), "trigger only": evaluate(trigger_only),
            "stage 1 (compression)": evaluate(program_path), "stage 1 + trigger": evaluate(staged)}
    print(f"\nselected compression: {result['winner']} (gate scores {result['scores']})")
    print(f"{'arm':<24}{'reward':>8}{'steps':>8}  firings")
    for name, row in rows.items():
        print(f"{name:<24}{row['mean_reward']:>8.3f}{row['mean_steps']:>8.2f}  {row['trigger_firings']}")
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nartifacts: {out}")


if __name__ == "__main__":
    main()
