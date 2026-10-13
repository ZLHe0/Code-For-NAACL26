"""Episode-to-row conversion and instance-level GRPO (harness_programs.rl.traces)."""

from __future__ import annotations

import pytest

from harness_programs.rl.traces import (
    EPISODE_REWARD_KEY,
    IS_REWARD_KEY,
    PromptOverflowError,
    Trace,
    TurnRecord,
    failure_trace,
    flatten_outputs,
    grpo_advantages,
    instance_grpo_advantages,
    is_append_only,
    merged_trace,
    num_mini_batch,
    pad_size,
    per_turn_traces,
    traces_per_instance_stats,
)


def _turns() -> list[TurnRecord]:
    # Turn prompts are rebuilt (not append-only): turn 1's prompt drops token 2 of turn 0's prompt.
    return [
        TurnRecord(prompt_ids=[1, 2, 3], response_ids=[10, 11], response_logprobs=[-0.1, -0.2],
                   observation_ids=[90]),
        TurnRecord(prompt_ids=[1, 3, 10, 11, 90], response_ids=[12], response_logprobs=[-0.3],
                   observation_ids=[91, 92]),
        TurnRecord(prompt_ids=[1, 90, 12, 91, 92], response_ids=[13, 14, 15], response_logprobs=[-0.4, -0.5, -0.6]),
    ]


# -- per-turn ------------------------------------------------------------------------------------


def test_per_turn_one_trace_per_turn_with_served_prompts_and_full_masks():
    turns = _turns()
    traces = per_turn_traces(turns, reward=1.0)
    assert len(traces) == 3
    for i, (trace, turn) in enumerate(zip(traces, turns, strict=True)):
        assert trace.prompt_ids == turn.prompt_ids
        assert trace.response_ids == turn.response_ids
        assert trace.response_mask == [1] * len(turn.response_ids)
        assert trace.response_logprobs == turn.response_logprobs
        assert trace.turn == i and trace.num_turns == 3


def test_per_turn_reward_on_last_trace_and_broadcast_in_extra_fields():
    traces = per_turn_traces(_turns(), reward=0.75)
    assert [t.is_reward for t in traces] == [False, False, True]
    assert [t.reward_score for t in traces] == [0.0, 0.0, 0.75]
    assert all(t.extra_fields[EPISODE_REWARD_KEY] == 0.75 for t in traces)
    assert [t.extra_fields[IS_REWARD_KEY] for t in traces] == [False, False, True]
    assert len({frozenset(t.extra_fields) for t in traces}) == 1  # verl unions keys per column


def test_per_turn_custom_selector_and_bad_selector():
    traces = per_turn_traces(_turns(), reward=1.0, reward_trace_fn=lambda ts: [True, False, False])
    assert [t.reward_score for t in traces] == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        per_turn_traces(_turns(), reward=1.0, reward_trace_fn=lambda ts: [True, True, False])
    with pytest.raises(ValueError):
        per_turn_traces([], reward=1.0)


def test_per_turn_prompt_overflow_raises_and_response_is_capped():
    with pytest.raises(PromptOverflowError):
        per_turn_traces(_turns(), reward=1.0, prompt_length=4)
    traces = per_turn_traces(_turns(), reward=1.0, prompt_length=5, response_length=2)
    assert traces[2].response_ids == [13, 14] and traces[2].response_mask == [1, 1]
    assert traces[2].response_logprobs == [-0.4, -0.5]


def test_output_kwargs_match_agent_loop_output_fields():
    kwargs = per_turn_traces(_turns(), reward=1.0)[0].output_kwargs()
    assert set(kwargs) == {"prompt_ids", "response_ids", "response_mask", "response_logprobs", "reward_score",
                           "num_turns", "extra_fields"}


# -- merged --------------------------------------------------------------------------------------


def _append_only_turns() -> list[TurnRecord]:
    t0 = TurnRecord(prompt_ids=[1, 2], response_ids=[10, 11], response_logprobs=[-0.1, -0.2], observation_ids=[90])
    t1 = TurnRecord(prompt_ids=[1, 2, 10, 11, 90], response_ids=[12], response_logprobs=[-0.3],
                    observation_ids=[91, 92])
    t2 = TurnRecord(prompt_ids=[1, 2, 10, 11, 90, 12, 91, 92], response_ids=[13], response_logprobs=[-0.4],
                    observation_ids=[93])
    return [t0, t1, t2]


def test_merged_interleaves_masked_observations_and_drops_the_trailing_one():
    trace = merged_trace(_append_only_turns(), reward=1.0)
    assert trace.prompt_ids == [1, 2]
    assert trace.response_ids == [10, 11, 90, 12, 91, 92, 13]
    assert trace.response_mask == [1, 1, 0, 1, 0, 0, 1]
    assert trace.response_logprobs == [-0.1, -0.2, 0.0, -0.3, 0.0, 0.0, -0.4]
    assert trace.reward_score == 1.0 and trace.is_reward and trace.num_turns == 3


def test_merged_truncates_from_the_right_and_handles_missing_logprobs():
    turns = _append_only_turns()
    trace = merged_trace(turns, reward=0.0, response_length=4)
    assert trace.response_ids == [10, 11, 90, 12] and trace.response_mask == [1, 1, 0, 1]
    turns[1].response_logprobs = None
    assert merged_trace(turns, reward=0.0).response_logprobs is None


def test_is_append_only():
    assert is_append_only(_append_only_turns())
    assert not is_append_only(_turns())
    assert is_append_only(_turns()[:1])


def test_merged_prefix_reproduces_each_turn_prompt_when_append_only():
    turns = _append_only_turns()
    trace = merged_trace(turns, reward=1.0)
    full = trace.prompt_ids + trace.response_ids
    for turn in turns:
        assert full[:len(turn.prompt_ids)] == turn.prompt_ids


# -- failure placeholder -------------------------------------------------------------------------


def test_failure_trace_trains_on_nothing():
    trace = failure_trace([5, 6], eos_token_id=2, want_logprobs=True)
    assert trace.response_ids == [2] and trace.response_mask == [0] and trace.response_logprobs == [0.0]
    assert trace.is_reward and trace.reward_score == 0.0 and trace.num_turns == 0
    assert failure_trace([5], eos_token_id=2, want_logprobs=False).response_logprobs is None
    with pytest.raises(ValueError):
        failure_trace([], eos_token_id=2, want_logprobs=False)


# -- batch bookkeeping ---------------------------------------------------------------------------


def test_flatten_outputs_tracks_parents_and_reward_flags():
    per_turn = per_turn_traces(_turns(), reward=1.0)
    single = merged_trace(_append_only_turns(), reward=0.0)
    as_dict = {"extra_fields": {IS_REWARD_KEY: True}}
    flat, parents, flags = flatten_outputs([per_turn, single, as_dict])
    assert len(flat) == 5
    assert parents == [0, 0, 0, 1, 2]
    assert flags == [False, False, True, True, True]
    stats = traces_per_instance_stats(parents, 3)
    assert stats["multi_trace/traces_per_instance/max"] == 3.0
    assert stats["multi_trace/traces_per_instance/min"] == 1.0
    assert stats["multi_trace/num_traces"] == 5.0


def test_flatten_outputs_is_identity_for_single_rows():
    rows = [Trace(prompt_ids=[1], response_ids=[2], response_mask=[1]) for _ in range(4)]
    flat, parents, flags = flatten_outputs(rows)
    assert flat == rows and parents == [0, 1, 2, 3] and flags == [True] * 4


def test_num_mini_batch_and_pad_size():
    assert num_mini_batch(16, 16) == 1
    assert num_mini_batch(128, 32) == 4
    with pytest.raises(ValueError):
        num_mini_batch(10, 4)
    with pytest.raises(ValueError):
        num_mini_batch(8, 0)
    assert pad_size(13, 8) == 3 and pad_size(16, 8) == 0
    with pytest.raises(ValueError):
        pad_size(3, 0)


# -- advantages ----------------------------------------------------------------------------------


def _flat(rows: list[list[float]]) -> list[float]:
    return [x for row in rows for x in row]


def test_instance_grpo_equals_stock_grpo_when_every_episode_is_one_trace():
    scores = [1.0, 0.0, 1.0, 0.5, 0.0, 0.0]
    masks = [[1, 1], [1], [1, 0, 1], [1], [1, 1], [0, 1]]
    uids = ["a", "a", "a", "b", "b", "b"]
    stock = grpo_advantages(scores, masks, uids)
    instance = instance_grpo_advantages(scores, masks, uids, list(range(6)), [True] * 6)
    assert _flat(instance) == pytest.approx(_flat(stock))
    # Hand-computed group a: mean 2/3, unbiased std sqrt(1/3).
    std_a = (1 / 3) ** 0.5
    assert stock[0][0] == pytest.approx((1 - 2 / 3) / (std_a + 1e-6))
    assert stock[1][0] == pytest.approx((0 - 2 / 3) / (std_a + 1e-6))


def test_instance_grpo_is_invariant_to_how_many_traces_an_episode_has():
    # Group "a" has two episodes: e0 (reward 1, three traces) and e1 (reward 0, one trace).
    rows = [("a", "e0", 0.0, False), ("a", "e0", 0.0, False), ("a", "e0", 1.0, True), ("a", "e1", 0.0, True)]
    uids = [r[0] for r in rows]
    instance_ids = [r[1] for r in rows]
    scores = [r[2] for r in rows]
    flags = [r[3] for r in rows]
    masks = [[1, 1]] * 4
    adv = instance_grpo_advantages(scores, masks, uids, instance_ids, flags)
    single = grpo_advantages([1.0, 0.0], [[1], [1]], ["a", "a"])
    assert _flat(adv[:3]) == pytest.approx([single[0][0]] * 6)
    assert adv[3] == pytest.approx([single[1][0]] * 2)
    # Stock per-row GRPO would wrongly see four samples with mean 1/4.
    naive = grpo_advantages(scores, masks, uids)
    assert naive[0][0] != pytest.approx(adv[0][0])


def test_pad_rows_and_episodes_without_reward_trace_get_zero():
    scores = [1.0, 0.0, 0.0, 1.0]
    masks = [[1], [1], [1], [1]]
    uids = ["a", "a", "a", "a"]
    instance_ids = [0, 1, 2, 3]
    flags = [True, True, False, True]
    pad = [False, False, False, True]
    adv = instance_grpo_advantages(scores, masks, uids, instance_ids, flags, pad)
    assert adv[2] == [0.0] and adv[3] == [0.0]
    ref = grpo_advantages([1.0, 0.0], [[1], [1]], ["a", "a"])
    assert adv[0] == pytest.approx(ref[0]) and adv[1] == pytest.approx(ref[1])


def test_singleton_group_and_masked_tokens():
    adv = instance_grpo_advantages([0.7], [[1, 0, 1]], ["x"], [0], [True])
    assert adv == [[pytest.approx(0.7 / (1 + 1e-6)), 0.0, pytest.approx(0.7 / (1 + 1e-6))]]
    no_std = instance_grpo_advantages([1.0, 0.0], [[1], [1]], ["a", "a"], [0, 1], [True, True], norm_by_std=False)
    assert _flat(no_std) == pytest.approx([0.5, -0.5])
    with pytest.raises(ValueError):
        instance_grpo_advantages([1.0], [[1], [1]], ["a"], [0], [True])
