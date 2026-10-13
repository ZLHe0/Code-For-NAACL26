"""D_hold is a slice of the training split, disjoint from D_train and from every evaluation set."""

from __future__ import annotations

import json

import pytest

from harness_programs.domains import Task, load
from harness_programs.domains.toy import write_split
from harness_programs.splits import HOLDOUT_SPLIT, HoldoutError, carve, carve_file, check, main, split_paths, write_tasks


def _toy():
    return load("toy")


def test_d_hold_is_loaded_as_training_data():
    assert HOLDOUT_SPLIT == "train"


def test_carve_is_disjoint_exhaustive_and_deterministic():
    domain = _toy()
    tasks = domain.load_tasks("synthetic", split="train")
    policy, holdout = carve(domain, tasks, 5, seed=3)
    assert len(holdout) == 5 and sorted(policy + holdout) == list(range(len(tasks)))
    assert {domain.task_key(tasks[i]) for i in policy}.isdisjoint(domain.task_key(tasks[i]) for i in holdout)
    assert carve(domain, tasks, 5, seed=3) == (policy, holdout)
    assert carve(domain, tasks, 5, seed=4) != (policy, holdout)


def test_carve_keeps_every_copy_of_a_task_on_one_side():
    domain = _toy()
    pool = domain.load_tasks("synthetic", split="train")[:4]
    tasks = pool + [Task(task_id=f"copy-{i}", prompt=t.prompt, data=t.data) for i, t in enumerate(pool)]
    policy, holdout = carve(domain, tasks, 2, seed=0)
    assert len(holdout) == 4 and len(policy) == 4
    held = {domain.task_key(tasks[i]) for i in holdout}
    assert all(domain.task_key(tasks[i]) not in held for i in policy)


@pytest.mark.parametrize("n", [0, 16, 17])
def test_carve_leaves_both_sides_non_empty(n):
    domain = _toy()
    with pytest.raises(HoldoutError):
        carve(domain, domain.load_tasks("synthetic", split="train"), n)


def test_carve_file_copies_rows_verbatim_beside_the_source(tmp_path):
    domain = _toy()
    source = write_tasks(tmp_path / "train.jsonl", domain.load_tasks("synthetic", split="train"))
    policy, holdout = carve_file(domain, source, 6, evaluation=["synthetic"])
    assert (policy, holdout) == split_paths(source) and policy.parent == source.parent
    rows = source.read_text().splitlines()
    out = policy.read_text().splitlines() + holdout.read_text().splitlines()
    assert sorted(out) == sorted(rows) and len(holdout.read_text().splitlines()) == 6


def test_check_accepts_a_carved_holdout(tmp_path):
    policy, holdout = write_split(tmp_path, n_holdout=6)
    counts = check(_toy(), holdout=str(holdout), train=str(policy), evaluation=["synthetic"])
    assert counts == {"holdout": 6, "D_train": 10, "evaluation:synthetic": 16}


def test_check_refuses_overlap_with_train_or_eval_and_the_eval_file_itself(tmp_path):
    domain = _toy()
    policy, holdout = write_split(tmp_path, n_holdout=6)
    with pytest.raises(HoldoutError, match="D_train"):
        check(domain, holdout=str(holdout), train=str(tmp_path / "train.jsonl"))
    with pytest.raises(HoldoutError, match="D_train"):
        check(domain, holdout="synthetic", train=str(policy))
    leaked = write_tasks(tmp_path / "eval.jsonl", domain.load_tasks(str(holdout))[:1])
    with pytest.raises(HoldoutError, match="evaluation"):
        check(domain, holdout=str(holdout), train=str(policy), evaluation=[str(leaked)])
    with pytest.raises(HoldoutError, match="is the evaluation file"):
        check(domain, holdout=str(holdout), train=str(policy), evaluation=[str(holdout)])
    with pytest.raises(HoldoutError, match="no D_hold"):
        check(domain, holdout="", train=str(policy))
    with pytest.raises(HoldoutError, match="D_train"):
        check(domain, holdout=str(holdout), train="")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(HoldoutError, match="no tasks"):
        check(domain, holdout=str(empty), train=str(policy))


def test_check_compares_content_not_ids(tmp_path):
    domain = _toy()
    policy, holdout = write_split(tmp_path, n_holdout=6)
    renamed = [Task(task_id=f"other-{i}", prompt=t.prompt, data=t.data)
               for i, t in enumerate(domain.load_tasks(str(holdout)))]
    disguised = write_tasks(tmp_path / "disguised.jsonl", renamed)
    with pytest.raises(HoldoutError, match="evaluation"):
        check(domain, holdout=str(holdout), train=str(policy), evaluation=[str(disguised)])


def test_cli_carves_then_checks(tmp_path, capsys):
    domain = _toy()
    source = write_tasks(tmp_path / "train.jsonl", domain.load_tasks("synthetic", split="train"))
    out = main(["carve", "--domain", "toy", "--tasks", str(source), "--n", "4", "--eval-tasks", "synthetic"])
    assert out == {"policy": str(tmp_path / "train_policy.jsonl"), "holdout": str(tmp_path / "train_holdout.jsonl")}
    assert json.loads(capsys.readouterr().out) == out
    counts = main(["check", "--domain", "toy", "--tasks", out["policy"], "--holdout", out["holdout"],
                   "--eval-tasks", "synthetic"])
    assert counts["holdout"] == 4
    with pytest.raises(HoldoutError):
        main(["check", "--domain", "toy", "--tasks", str(source), "--holdout", out["holdout"]])
