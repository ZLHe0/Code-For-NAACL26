"""Step marker and per-step episode dumps (harness_programs.rl.step_marker)."""

from __future__ import annotations

import json
import threading

from harness_programs.rl.step_marker import (
    EpisodeDumpWriter,
    ReplicateCounter,
    dump_path,
    marker_path,
    read_global_step,
    read_step,
    rollout_meta,
    write_global_step,
)
from harness_programs.rollout import read_jsonl
from harness_programs.rollout.runner import EpisodeRecord


def _record(task_id: str, *, step: int | None, replicate: int, validate: bool = False) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"{task_id}-{replicate}", harness="toy", task_id=task_id, status="answered", num_steps=2,
        max_steps=8, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        reward=1.0, final_output="file_a.txt", gold={"file": "file_a.txt"}, context_budget=1024,
        triggers={"setup": {"installed": []}, "firings": []}, served=[{"tokens_after": 10, "fired": []}],
        rollout=rollout_meta(step=step, group_key=task_id, replicate=replicate, validate=validate))


def test_marker_round_trip_and_missing(tmp_path):
    assert read_global_step(tmp_path) is None
    assert read_global_step(None) is None
    write_global_step(tmp_path, 7)
    assert marker_path(tmp_path).read_text() == "7\n"
    assert read_global_step(tmp_path) == 7
    write_global_step(tmp_path, 8)
    assert read_global_step(tmp_path) == 8
    assert sorted(p.name for p in tmp_path.iterdir()) == ["global_step"]  # no temp files left behind
    marker_path(tmp_path).write_text("garbage")
    assert read_global_step(tmp_path) is None


def test_dump_path_layout(tmp_path):
    assert dump_path(tmp_path, 3) == tmp_path / "rollouts" / "step_3.jsonl"
    assert dump_path(tmp_path, None) == tmp_path / "rollouts" / "step_unknown.jsonl"


def test_dump_appends_episode_records_readable_by_read_jsonl(tmp_path):
    writer = EpisodeDumpWriter(tmp_path)
    writer.write(_record("t0", step=4, replicate=0))
    writer.write({**_record("t0", step=4, replicate=1, validate=True).to_dict(), "error": "extra column"})
    writer.write(_record("t1", step=5, replicate=0))
    rows = read_jsonl(dump_path(tmp_path, 4))
    assert [r.episode_id for r in rows] == ["t0-0", "t0-1"]
    assert rows[0].rollout == {"step": 4, "group_key": "t0", "replicate": 0, "validate": False}
    assert rows[1].rollout["validate"] is True
    assert rows[0].messages[-1] == {"role": "assistant", "content": "a"}
    assert rows[0].served == [{"tokens_after": 10, "fired": []}]
    raw = [json.loads(line) for line in dump_path(tmp_path, 4).read_text().splitlines()]
    assert raw[1]["error"] == "extra column"
    assert [r.task_id for r in read_step(tmp_path, 5)] == ["t1"]
    assert read_step(tmp_path, 99) == []


def test_dump_without_step_goes_to_unknown(tmp_path):
    EpisodeDumpWriter(tmp_path).write(_record("t0", step=None, replicate=0))
    assert [r.rollout["step"] for r in read_step(tmp_path, None)] == [None]


def test_concurrent_writers_never_interleave_lines(tmp_path):
    writers = [EpisodeDumpWriter(tmp_path) for _ in range(4)]

    def work(w: EpisodeDumpWriter, k: int) -> None:
        for i in range(25):
            w.write(_record(f"t{k}", step=1, replicate=i))

    threads = [threading.Thread(target=work, args=(w, k)) for k, w in enumerate(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = read_step(tmp_path, 1)
    assert len(rows) == 100
    assert len({r.episode_id for r in rows}) == 100


def test_replicate_counter_numbers_per_step_group_and_mode():
    counter = ReplicateCounter()
    assert [counter.next(1, "a", False) for _ in range(3)] == [0, 1, 2]
    assert counter.next(1, "b", False) == 0
    assert counter.next(1, "a", True) == 0
    assert counter.next(2, "a", False) == 0
    assert counter.next(None, {"k": 1}, False) == 0


def test_dumps_feed_the_steering_reservoir(tmp_path):
    from harness_programs.steering.reservoir import latest_step, load_records

    writer = EpisodeDumpWriter(tmp_path)
    for replicate in range(3):
        writer.write(_record("t0", step=2, replicate=replicate))
    writer.write(_record("t1", step=3, replicate=0, validate=True))
    writer.write(_record("t2", step=None, replicate=0))  # no marker yet: kept out of the reservoir
    assert latest_step(tmp_path) == 3
    reservoir = load_records(tmp_path)
    assert reservoir.n_validation == 1
    assert [(p.step, p.group_key) for p in reservoir.placed] == [(2, "t0")] * 3
