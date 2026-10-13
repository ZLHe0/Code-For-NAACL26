"""``python -m harness_programs.compression``: run stage 1 (Rollout -> ProposeCompress -> Gate) and emit ``program.json``.

The creator LLM and the policy are OpenAI-compatible endpoints. Keys are read from
``HARNESS_LLM_API_KEY`` and ``HARNESS_POLICY_API_KEY``; base URLs and model names may come
from the flags below or from ``HARNESS_LLM_*`` / ``HARNESS_POLICY_*``.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from ..llm.client import OpenAIChat
from ..serving.tokens import TokenCounter, char_counter, chat_template_counter
from .campaign import Campaign, CampaignConfig
from .episodes import Sizing


def _counter(tokenizer: str | None) -> TokenCounter:
    if not tokenizer:
        return char_counter()
    from transformers import AutoTokenizer

    return chat_template_counter(AutoTokenizer.from_pretrained(tokenizer))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = CampaignConfig(domain="", out_dir="", sizing=Sizing(0))
    p = argparse.ArgumentParser(prog="python -m harness_programs.compression", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True)
    p.add_argument("--out", required=True, help="artifact directory; program.json is written here")
    p.add_argument("--tasks", required=True,
                   help="training tasks D_train the policy trains on, for the rollout and the D_hold check")
    p.add_argument("--episodes", nargs="*", default=[], help="pre-collected rollouts (JSONL) to use as R0")
    p.add_argument("--holdout", required=True,
                   help="held-out TRAINING tasks D_hold, for the gate: carved out of the training split "
                        "(python -m harness_programs.splits), never the validation file")
    p.add_argument("--eval-tasks", nargs="*", default=[], help="evaluation task files D_hold must be disjoint from")
    p.add_argument("--serving-limit", type=int, required=True, help="the policy's context limit, in tokens")
    p.add_argument("--history-budget", type=int, default=None,
                   help="tokens of history allowed beyond the pinned head (the limited-context regime)")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer for exact counts; default ~4 chars/token")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--policy-base-url", default=None)
    p.add_argument("--policy-model", default=None)
    p.add_argument("--rollout-tasks", type=int, default=defaults.rollout_tasks, help="training tasks rolled out")
    p.add_argument("--group-size", type=int, default=defaults.group_size, help="rollouts per training task")
    p.add_argument("--candidates", type=int, default=defaults.candidates, help="candidate programs drafted")
    p.add_argument("--rounds", type=int, default=defaults.rounds, help="accept-or-revise rounds per candidate")
    p.add_argument("--holdout-tasks", type=int, default=defaults.holdout_tasks,
                   help="D_hold tasks each program is scored on")
    p.add_argument("--workers", type=int, default=defaults.workers)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    sizing = Sizing(args.serving_limit, args.history_budget, counter=_counter(args.tokenizer))
    config = CampaignConfig(domain=args.domain, out_dir=args.out, sizing=sizing, tasks_path=args.tasks,
                            holdout_path=args.holdout, eval_tasks=tuple(args.eval_tasks),
                            episodes=tuple(args.episodes),
                            rollout_tasks=args.rollout_tasks, group_size=args.group_size,
                            candidates=args.candidates, rounds=args.rounds, holdout_tasks=args.holdout_tasks,
                            workers=args.workers)
    llm = OpenAIChat(base_url=args.llm_base_url, model=args.llm_model, prefix="HARNESS_LLM")
    policy = OpenAIChat(base_url=args.policy_base_url, model=args.policy_model, prefix="HARNESS_POLICY")
    result = Campaign(config, llm=llm, policy=policy).run()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
