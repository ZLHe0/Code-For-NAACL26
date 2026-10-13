"""``python -m harness_programs.steering``: the steer update of Algorithm 1, run beside an RL run.

Two subcommands:

* ``supervise`` -- watch ``<run-dir>/rollouts/step_*.jsonl`` and run a round every
  ``--round-every`` training steps until the round at ``--total-steps`` (or ``--max-rounds``).
* ``round`` -- run (or resume) one round at ``--step`` and print its summary.

The creator (proposer and judge) and the policy are OpenAI-compatible endpoints. The policy
endpoint must serve the run's latest checkpoint pi_k: the gate plays the held-out tasks with it.

``--holdout`` is D_hold, held-out *training* tasks carved out of the training split
(``python -m harness_programs.splits``). Before anything runs, it is checked to share no task with
``--tasks`` (the tasks the policy trains on) or with any ``--eval-tasks`` file.
Keys come from ``HARNESS_LLM_API_KEY`` and ``HARNESS_POLICY_API_KEY``; base URLs and model
names may come from the flags below or from ``HARNESS_LLM_*`` / ``HARNESS_POLICY_*``.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from .. import splits
from ..domains import load
from ..llm.client import OpenAIChat
from .gate import DEFAULT_SERVING_LIMIT, GateConfig
from .reservoir import MAX_GROUPS, latest_step
from .rounds import RoundConfig, run_round
from .supervisor import DEFAULT_POLL_SECONDS, ROUND_EVERY, supervise


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", required=True, help="the RL run directory (holds rollouts/)")
    p.add_argument("--program", required=True, help="the served program.json (H_k)")
    p.add_argument("--domain", required=True)
    p.add_argument("--total-steps", type=int, required=True,
                   help="the last training step; the supervisor runs its last round there")
    p.add_argument("--tasks", required=True,
                   help="training tasks D_train the policy trains on; D_hold is checked against them")
    p.add_argument("--holdout", required=True,
                   help="held-out TRAINING tasks D_hold for the gate: carved out of the training split "
                        "(python -m harness_programs.splits), never the validation file")
    p.add_argument("--eval-tasks", nargs="*", default=[], help="evaluation task files D_hold must be disjoint from")
    p.add_argument("--holdout-tasks", type=int, default=None, help="limit on D_hold tasks per gate arm")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--policy-base-url", default=None, help="endpoint serving the latest checkpoint")
    p.add_argument("--policy-model", default=None)
    p.add_argument("--serving-limit", type=int, default=DEFAULT_SERVING_LIMIT)
    p.add_argument("--history-budget", type=int, default=None,
                   help="history tokens above the pinned head in gate rollouts (match training)")
    p.add_argument("--admit-all", action="store_true",
                   help="ablation: skip the judge and send every replayed trigger to the gate")
    p.add_argument("--max-groups", type=int, default=MAX_GROUPS,
                   help="mixed-outcome groups the creator reads per round")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m harness_programs.steering", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sup = sub.add_parser("supervise", help="run a round every --round-every steps")
    _common(sup)
    sup.add_argument("--round-every", type=int, default=ROUND_EVERY)
    sup.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    sup.add_argument("--max-rounds", type=int, default=None)
    one = sub.add_parser("round", help="run or resume one round")
    _common(one)
    one.add_argument("--step", type=int, default=None, help="default: the newest dumped step")
    return p.parse_args(argv)


def config_from(args: argparse.Namespace) -> RoundConfig:
    """The round config, after checking that D_hold is held out from training and evaluation."""
    splits.check(load(args.domain), holdout=args.holdout, train=args.tasks, evaluation=args.eval_tasks)
    gate = GateConfig(holdout=args.holdout, holdout_tasks=args.holdout_tasks,
                      serving_limit=args.serving_limit, history_budget=args.history_budget,
                      workers=args.workers)
    return RoundConfig(run_dir=args.run_dir, program=args.program, domain=args.domain,
                       total_steps=args.total_steps, gate=gate, max_groups=args.max_groups,
                       seed=args.seed, admit_all=args.admit_all, workers=args.workers)


def main(argv: list[str] | None = None) -> Any:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = config_from(args)
    creator = OpenAIChat(base_url=args.llm_base_url, model=args.llm_model, prefix="HARNESS_LLM")
    policy = OpenAIChat(base_url=args.policy_base_url, model=args.policy_model,
                        prefix="HARNESS_POLICY")
    if args.command == "round":
        step = args.step if args.step is not None else latest_step(cfg.run_dir)
        if step < 0:
            raise SystemExit(f"no rollout dumps under {cfg.run_dir}/rollouts")
        result: Any = run_round(cfg, step, creator, policy)
    else:
        state = supervise(cfg, creator, policy, every=args.round_every,
                          poll_seconds=args.poll_seconds, max_rounds=args.max_rounds)
        result = {"rounds": state.rounds, "errors": state.errors}
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
