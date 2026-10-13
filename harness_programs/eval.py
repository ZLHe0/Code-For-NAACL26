"""``python -m harness_programs.eval``: score a policy on a held-out split, served through a program.

Runs every task in the split through the same :class:`ContextPipeline` training used, so a
checkpoint is evaluated under the context it was trained with. Writes the episodes and a
summary, and prints the summary.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from .domains import load as load_domain
from .llm.client import OpenAIChat
from .rollout.runner import EpisodeRecord, run_batch, write_jsonl
from .serving.serve import ContextPipeline
from .serving.tokens import char_counter, chat_template_counter


def summarize(records: list[EpisodeRecord], *, max_reward: float = 1.0) -> dict[str, Any]:
    """``mean_score`` is the reported metric: the reward rescaled to ``[0, 1]`` by the domain's
    ``max_reward`` (CodeSearch trains on the sum of its three F1s and reports their mean)."""
    n = len(records)
    mean_reward = sum(r.reward for r in records) / n if n else None
    fired = Counter(name for r in records for names in (r.triggers.get("firings") or []) for name in names)
    return {
        "episodes": n,
        "mean_score": round(mean_reward / max_reward, 4) if n else None,
        "mean_reward": round(mean_reward, 4) if n else None,
        "answered_pct": round(100.0 * sum(r.status == "answered" for r in records) / n, 1) if n else None,
        "mean_steps": round(sum(r.num_steps for r in records) / n, 2) if n else None,
        "evicting_turn_pct": round(100.0 * sum(s.get("evicted", False) for r in records for s in r.served)
                                   / max(1, sum(len(r.served) for r in records)), 1),
        "trigger_firings": dict(fired),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m harness_programs.eval", description=__doc__)
    p.add_argument("--domain", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--program", default=None, help="program.json; omit to serve with no program")
    p.add_argument("--serving-limit", type=int, required=True)
    p.add_argument("--history-budget", type=int, default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--policy-base-url", default=None)
    p.add_argument("--policy-model", default=None)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    domain = load_domain(args.domain)
    if args.tokenizer:
        from transformers import AutoTokenizer

        counter = chat_template_counter(AutoTokenizer.from_pretrained(args.tokenizer), domain.tools)
    else:
        counter = char_counter()
    pipeline = ContextPipeline(args.domain, program_path=args.program, token_counter=counter,
                               serving_limit=args.serving_limit, history_budget=args.history_budget,
                               max_steps=domain.max_turns)
    policy = OpenAIChat(base_url=args.policy_base_url, model=args.policy_model, prefix="HARNESS_POLICY")
    records = run_batch(domain, domain.load_tasks(args.tasks, split=args.split, limit=args.limit), policy,
                        pipeline=pipeline, replicates=args.replicates, workers=args.workers,
                        temperature=args.temperature, rollout={"split": args.split})
    out = Path(args.out)
    write_jsonl(out / "episodes.jsonl", records)
    summary = {"domain": args.domain, "split": args.split, "program": args.program,
               **summarize(records, max_reward=getattr(domain, "max_reward", 1.0))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
