from __future__ import annotations

import random

import pytest

from tokenserve.kv_cache import KVCacheManager
from tokenserve.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
    SchedulingPolicy,
    StaticScheduler,
    UpdateResult,
    make_scheduler,
)
from tokenserve.sequence import FinishReason, SamplingParams, Sequence, SequenceStatus


def make(
    *,
    num_blocks: int = 64,
    block_size: int = 4,
    max_num_seqs: int = 8,
    budget: int = 64,
    max_model_len: int = 128,
    chunked: bool = True,
    policy: SchedulingPolicy = SchedulingPolicy.CONTINUOUS,
    caching: bool = True,
    watermark_blocks: int = 0,
    eos: int | None = None,
) -> Scheduler:
    kv = KVCacheManager(
        num_blocks, block_size, enable_prefix_caching=caching, watermark_blocks=watermark_blocks
    )
    cfg = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        max_model_len=max_model_len,
        enable_chunked_prefill=chunked,
        policy=policy,
    )
    return make_scheduler(cfg, kv, eos_token_id=eos)


_next_id = iter(range(10**9))


def seq(prompt_len: int, max_tokens: int = 4, *, seq_id: int | None = None,
        tok: int = 1, ignore_eos: bool = False) -> Sequence:  # fmt: skip
    sid = next(_next_id) if seq_id is None else seq_id
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos)
    # Distinct prompts per sequence unless the caller wants sharing.
    return Sequence(sid, f"r{sid}", [tok + sid % 7] * prompt_len, params)


def step(s: Scheduler, token: int = 3) -> tuple[SchedulerOutput, UpdateResult]:
    out = s.schedule()
    res = s.update(out, {x.seq.seq_id: token for x in out.scheduled if x.samples})
    s.kv.check_invariants()
    return out, res


def planned(out: SchedulerOutput) -> dict[int, int]:
    return {x.seq.seq_id: x.num_new_tokens for x in out.scheduled}


def run_to_completion(s: Scheduler, max_steps: int = 10_000) -> int:
    n = 0
    while s.has_unfinished():
        step(s)
        n += 1
        assert n < max_steps, "scheduler made no progress"
    return n


def test_budget_and_max_num_seqs_are_respected() -> None:
    s = make(max_num_seqs=4, budget=10)
    seqs = [seq(6, max_tokens=5) for _ in range(10)]
    for q in seqs:
        s.add(q)
    while s.has_unfinished():
        out, _ = step(s)
        assert out.num_batched_tokens <= 10
        assert s.num_running <= 4
        assert len(out.scheduled) <= 4
    assert all(q.num_output_tokens == 5 and q.finish_reason is FinishReason.LENGTH for q in seqs)
    assert s.kv.num_free_blocks == s.kv.num_blocks


def test_chunked_prefill_splits_a_long_prompt_and_samples_only_at_the_end() -> None:
    s = make(budget=32, max_model_len=256, num_blocks=64)
    q = seq(100, max_tokens=2)
    s.add(q)
    chunks = []
    for _ in range(4):
        out, res = step(s)
        chunks.append(planned(out)[q.seq_id])
        assert bool(res.events) == (len(chunks) == 4)
    assert chunks == [32, 32, 32, 4]
    out, _ = step(s)
    assert planned(out) == {q.seq_id: 1}  # now decoding


def test_without_chunked_prefill_prompts_are_never_split() -> None:
    s = make(budget=64, max_model_len=64, chunked=False)
    a, b = seq(40), seq(40)
    s.add(a)
    s.add(b)
    out, _ = step(s)
    assert planned(out) == {a.seq_id: 40}  # b does not fit the rest of the budget
    out, _ = step(s)
    assert planned(out) == {a.seq_id: 1, b.seq_id: 40}


def test_decodes_are_scheduled_before_prefill_chunks() -> None:
    s = make(budget=16, max_model_len=128)
    young = seq(3, max_tokens=10, seq_id=10**8 + 5)
    s.add(young)
    step(s)  # young is now decoding
    old = seq(60, max_tokens=2, seq_id=10**8 + 1)  # older than `young` but arrives in prefill
    s.add(old)
    step(s)  # old admitted, first chunk
    out, _ = step(s)
    # FCFS alone would give the whole budget to the older prefill and stall `young`.
    assert planned(out) == {young.seq_id: 1, old.seq_id: 15}


def test_continuous_admits_new_requests_between_decode_steps() -> None:
    s = make()
    a = seq(4, max_tokens=10)
    s.add(a)
    step(s)
    step(s)
    b = seq(4, max_tokens=2)
    s.add(b)
    out, _ = step(s)
    assert planned(out) == {a.seq_id: 1, b.seq_id: 4}
    assert out.admitted == [b]


def test_static_batching_waits_for_the_whole_batch_to_drain() -> None:
    s = make(policy=SchedulingPolicy.STATIC)
    assert isinstance(s, StaticScheduler)
    a = seq(4, max_tokens=5)
    s.add(a)
    step(s)
    b = seq(4, max_tokens=2)
    s.add(b)
    while not a.is_finished:
        out, _ = step(s)
        assert b.seq_id not in planned(out)
        assert b.status is SequenceStatus.WAITING
    out, _ = step(s)
    assert planned(out) == {b.seq_id: 4}


def test_static_batch_members_leave_individually_but_nobody_joins() -> None:
    s = make(policy=SchedulingPolicy.STATIC)
    short, long_ = seq(4, max_tokens=1), seq(4, max_tokens=6)
    s.add(short)
    s.add(long_)
    step(s)
    assert short.is_finished
    late = seq(4, max_tokens=1)
    s.add(late)
    out, _ = step(s)
    assert planned(out) == {long_.seq_id: 1}  # batch of one, late still waits


def test_admission_is_fcfs_with_head_of_line_blocking() -> None:
    # 8 blocks x 4 tokens. `a` holds 5 blocks; `big` needs 6 and cannot fit;
    # `small` would fit but must not jump the queue.
    s = make(num_blocks=8, block_size=4, budget=64)
    a = seq(19, max_tokens=2)
    s.add(a)
    step(s)
    big, small = seq(24, max_tokens=1), seq(4, max_tokens=1)
    s.add(big)
    s.add(small)
    order = []
    while s.has_unfinished():
        out, _ = step(s)
        order += [q.seq_id for q in out.admitted]
        if not a.is_finished:
            assert small.status is SequenceStatus.WAITING
    assert order == [big.seq_id, small.seq_id]


def test_preemption_evicts_the_youngest_and_requeues_it_first() -> None:
    s = make(num_blocks=4, block_size=2, budget=64, caching=False)
    a, b = seq(3, max_tokens=6), seq(3, max_tokens=6)
    s.add(a)
    s.add(b)
    step(s)  # both prefill: 2 blocks each, cache full
    step(s)  # both decode within their second block
    late = seq(2, max_tokens=1)
    s.add(late)
    out, _ = step(s)  # a needs a 3rd block -> preempt b (youngest running)
    assert out.preempted == [b]
    assert planned(out) == {a.seq_id: 1}
    assert b.status is SequenceStatus.WAITING
    assert b.num_computed_tokens == 0
    assert s.kv.num_blocks_of(b.seq_id) == 0
    assert b.num_output_tokens == 2  # generated tokens are kept for recompute
    assert s.waiting[0] is b  # ahead of `late`, which arrived after it
    assert out.admitted == []  # no admissions in a step that preempted
    run_to_completion(s)
    assert a.num_output_tokens == b.num_output_tokens == 6
    assert s.num_preemptions >= 1
    assert s.kv.num_free_blocks == s.kv.num_blocks


def test_eos_finishes_unless_ignored() -> None:
    s = make(eos=3)
    a, b = seq(4, max_tokens=5), seq(4, max_tokens=5, ignore_eos=True)
    s.add(a)
    s.add(b)
    run_to_completion(s)  # step() always samples token 3
    assert a.finish_reason is FinishReason.STOP
    assert a.num_output_tokens == 1
    assert b.finish_reason is FinishReason.LENGTH
    assert b.num_output_tokens == 5


def test_prefix_cache_hits_are_counted_at_admission() -> None:
    s = make(block_size=4)
    p = [5] * 12
    a = Sequence(9001, "a", [*p, 1], SamplingParams(max_tokens=1))
    b = Sequence(9002, "b", [*p, 2], SamplingParams(max_tokens=1))
    s.add(a)
    run_to_completion(s)
    s.add(b)
    out, _ = step(s)
    assert planned(out) == {b.seq_id: 1}  # 12 of 13 prompt tokens came from cache
    assert s.prefix_hit_tokens == 12
    assert s.prefix_query_tokens == 26


def test_finish_removes_waiting_and_running_sequences() -> None:
    s = make()
    a, b = seq(4, max_tokens=10), seq(4, max_tokens=10)
    s.add(a)
    step(s)
    s.add(b)
    s.finish(a, FinishReason.ABORT)
    s.finish(b, FinishReason.ABORT)
    s.finish(b, FinishReason.ABORT)  # idempotent
    assert not s.has_unfinished()
    assert s.num_waiting_tokens == 0
    assert s.kv.num_free_blocks == s.kv.num_blocks


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        SchedulerConfig(
            max_num_batched_tokens=512, max_model_len=1024, enable_chunked_prefill=False
        )
    with pytest.raises(ValueError):
        SchedulerConfig(max_num_seqs=0)


@pytest.mark.parametrize("policy", list(SchedulingPolicy))
@pytest.mark.parametrize("seed", range(20))
def test_random_workloads_always_complete_and_preempt_youngest(
    policy: SchedulingPolicy, seed: int
) -> None:
    rng = random.Random(seed)
    s = make(
        num_blocks=rng.randint(6, 16),
        block_size=rng.choice([1, 2, 4]),
        max_num_seqs=rng.randint(1, 6),
        budget=rng.randint(8, 40),
        max_model_len=24,
        policy=policy,
    )
    pending = [seq(rng.randint(1, 12), max_tokens=rng.randint(1, 12)) for _ in range(25)]
    pending = [q for q in pending if s.kv.can_ever_fit(q.max_total_tokens)]
    everyone = list(pending)
    steps = 0
    while pending or s.has_unfinished():
        if pending and rng.random() < 0.5:
            s.add(pending.pop(0))
        running_before = {q.seq_id for q in s.running}
        out, _ = step(s)
        for victim in out.preempted:
            # A victim is never older than a sequence that kept running.
            assert all(victim.seq_id > q.seq_id for q in s.running if q.seq_id in running_before)
        steps += 1
        assert steps < 5000
    for q in everyone:
        assert q.finish_reason is FinishReason.LENGTH
        assert q.num_output_tokens == q.params.max_tokens
    assert s.kv.num_free_blocks == s.kv.num_blocks
