"""D_hold: held-out TRAINING tasks, carved out of the training split and checked before any gate.

Algorithm 1 takes two task sets. The policy trains on D_train, and both Gates (lines 3 and 9)
score candidates on D_hold. D_hold is a slice of the *training* split that the policy never
trains on, so it is

* disjoint from D_train: a program judged on tasks the policy trains on is rewarded for helping
  the policy recite them;
* disjoint from every evaluation set: a gate that scores on evaluation tasks turns the reported
  numbers into a selection set.

The validation or evaluation file is therefore never a valid D_hold, and nothing in this release
defaults to it. :func:`carve` cuts D_hold out of the training file by task identity (the domain's
``task_key``: content, not ``task_id``, since ids are reassigned per file and collide across
splits); the policy trains on the remainder. :func:`check` refuses a D_hold that shares a task
with D_train or with an evaluation file, or that *is* an evaluation file. Both stage CLIs call it
before they do anything else, and ``scripts/run_pipeline.sh`` carves when no D_hold is given::

    python -m harness_programs.splits carve --domain sql --tasks data/sql/train.jsonl --n 128 \\
        --eval-tasks data/sql/val.jsonl
    # -> data/sql/train_policy.jsonl  (D_train: what the policy trains on)
    #    data/sql/train_holdout.jsonl (D_hold: what both gates score on)
    python -m harness_programs.splits check --domain sql --tasks data/sql/train_policy.jsonl \\
        --holdout data/sql/train_holdout.jsonl --eval-tasks data/sql/val.jsonl

Both files are written beside the source, because task files hold paths (repositories,
databases, game files) relative to their own directory.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

from .domains import Domain, Task, load

HOLDOUT_SPLIT = "train"  # D_hold is training data; this is the split label it is loaded under
EVAL_SPLIT = "val"
DEFAULT_HOLDOUT = 128
POLICY_SUFFIX = "_policy"
HOLDOUT_SUFFIX = "_holdout"


class HoldoutError(ValueError):
    """D_hold is not a held-out slice of the training data."""


def keys(domain: Domain, tasks: Sequence[Task]) -> list[str]:
    return [domain.task_key(t) for t in tasks]


def carve(domain: Domain, tasks: Sequence[Task], n: int, *, seed: int = 0) -> tuple[list[int], list[int]]:
    """Indices ``(policy, holdout)``: ``n`` task identities drawn at random go to D_hold.

    Every row sharing a drawn identity goes with it, so no task straddles the two sides.
    """
    ids = keys(domain, tasks)
    distinct = list(dict.fromkeys(ids))
    if not 0 < n < len(distinct):
        raise HoldoutError(f"cannot hold out {n} of {len(distinct)} distinct training tasks; "
                           "D_hold must be non-empty and leave tasks to train on")
    held = set(random.Random(seed).sample(distinct, n))
    policy = [i for i, k in enumerate(ids) if k not in held]
    holdout = [i for i, k in enumerate(ids) if k in held]
    return policy, holdout


def _same_file(a: str, b: str) -> bool:
    pa, pb = Path(a), Path(b)
    return pa.exists() and pb.exists() and pa.resolve() == pb.resolve()


def check(domain: Domain, *, holdout: str, train: str, evaluation: Sequence[str] = ()) -> dict[str, int]:
    """Refuse a D_hold that is not a held-out slice of the training data; return the task counts.

    Compares content (``task_key``), not only paths: two different files can still share tasks.
    """
    if not holdout:
        raise HoldoutError("no D_hold given: carve one out of the training tasks with python -m harness_programs.splits")
    if not train:
        raise HoldoutError("D_hold can only be checked against D_train: pass the training task file")
    for path in evaluation:
        if path and (path == holdout or _same_file(path, holdout)):
            raise HoldoutError(f"D_hold {holdout!r} is the evaluation file {path!r}; D_hold must be "
                               "carved out of the training split (python -m harness_programs.splits)")
    held = set(keys(domain, domain.load_tasks(holdout, split=HOLDOUT_SPLIT)))
    if not held:
        raise HoldoutError(f"D_hold {holdout!r} holds no tasks")
    counts = {"holdout": len(held)}
    sources = [("D_train", train, HOLDOUT_SPLIT)] + [("evaluation", p, EVAL_SPLIT) for p in evaluation if p]
    for label, path, split in sources:
        other = set(keys(domain, domain.load_tasks(path, split=split)))
        shared = held & other
        if shared:
            raise HoldoutError(f"D_hold {holdout!r} shares {len(shared)} of its {len(held)} tasks with "
                               f"{label} {path!r}; D_hold must be disjoint from the tasks the policy trains "
                               "on and from every evaluation set (python -m harness_programs.splits)")
        counts[label if label == "D_train" else f"evaluation:{path}"] = len(other)
    return counts


def split_paths(tasks_path: str | Path) -> tuple[Path, Path]:
    path = Path(tasks_path)
    return (path.with_name(f"{path.stem}{POLICY_SUFFIX}{path.suffix}"),
            path.with_name(f"{path.stem}{HOLDOUT_SUFFIX}{path.suffix}"))


def write_tasks(path: str | Path, tasks: Sequence[Task]) -> Path:
    """Write tasks as JSONL rows ``{task_id, prompt, data}``, the format every domain loads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"task_id": t.task_id, "prompt": t.prompt, "data": t.data}) + "\n"
                            for t in tasks))
    return path


def carve_file(domain: Domain, tasks_path: str | Path, n: int, *, seed: int = 0,
               evaluation: Sequence[str] = ()) -> tuple[Path, Path]:
    """Split a training JSONL into ``<stem>_policy.jsonl`` (D_train) and ``<stem>_holdout.jsonl`` (D_hold).

    The rows are copied verbatim, so relative paths inside them still resolve. The result is
    checked with :func:`check` before it is returned.
    """
    rows = [line for line in Path(tasks_path).read_text().splitlines() if line.strip()]
    tasks = domain.load_tasks(str(tasks_path), split=HOLDOUT_SPLIT)
    if len(tasks) != len(rows):
        raise HoldoutError(f"{tasks_path}: {len(rows)} rows but {len(tasks)} tasks; cannot split by row")
    policy, holdout = carve(domain, tasks, n, seed=seed)
    policy_path, holdout_path = split_paths(tasks_path)
    policy_path.write_text("".join(rows[i] + "\n" for i in policy))
    holdout_path.write_text("".join(rows[i] + "\n" for i in holdout))
    check(domain, holdout=str(holdout_path), train=str(policy_path), evaluation=evaluation)
    return policy_path, holdout_path


def main(argv: list[str] | None = None) -> dict[str, object]:
    p = argparse.ArgumentParser(prog="python -m harness_programs.splits", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    cut = sub.add_parser("carve", help="split the training file into D_train and D_hold")
    cut.add_argument("--tasks", required=True, help="the full training task file")
    cut.add_argument("--n", type=int, default=DEFAULT_HOLDOUT, help="training tasks held out as D_hold")
    cut.add_argument("--seed", type=int, default=0)
    chk = sub.add_parser("check", help="refuse a D_hold that overlaps D_train or an evaluation set")
    chk.add_argument("--tasks", required=True, help="the training tasks the policy trains on (D_train)")
    chk.add_argument("--holdout", required=True, help="the held-out training tasks (D_hold)")
    for q in (cut, chk):
        q.add_argument("--domain", required=True)
        q.add_argument("--eval-tasks", nargs="*", default=[],
                       help="evaluation task files D_hold must also be disjoint from")
    args = p.parse_args(argv)
    domain = load(args.domain)
    result: dict[str, object]
    if args.command == "carve":
        policy, holdout = carve_file(domain, args.tasks, args.n, seed=args.seed, evaluation=args.eval_tasks)
        result = {"policy": str(policy), "holdout": str(holdout)}
    else:
        result = check(domain, holdout=args.holdout, train=args.tasks, evaluation=args.eval_tasks)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
