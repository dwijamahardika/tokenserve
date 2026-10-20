from __future__ import annotations

import itertools

import pytest

from tokenserve.kv_cache import KVCacheError, KVCacheManager, hash_block
from tokenserve.sequence import SamplingParams, Sequence

_ids = itertools.count()


def seq_of(tokens: list[int]) -> Sequence:
    return Sequence(next(_ids), "r", list(tokens), SamplingParams(max_tokens=8))


def admit(kv: KVCacheManager, seq: Sequence, *, watermark: bool = False) -> int:
    """Admit with prefix lookup and compute the whole prompt. Returns cached tokens."""
    hits, cached = kv.get_computed_blocks(seq)
    assert kv.allocate_slots(seq, seq.num_tokens - cached, hits, watermark=watermark)
    seq.num_computed_tokens = seq.num_tokens
    kv.cache_full_blocks(seq)
    return cached


def test_allocate_and_free_returns_every_block() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=4)
    s = seq_of(list(range(10)))
    admit(kv, s)
    assert kv.num_blocks_of(s.seq_id) == 3  # ceil(10 / 4)
    assert kv.num_free_blocks == 5
    kv.free(s.seq_id)
    assert kv.num_free_blocks == 8
    kv.check_invariants()


def test_decode_growth_allocates_one_block_per_block_size_tokens() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    s = seq_of([1, 2, 3])
    admit(kv, s)
    sizes = []
    for t in range(6):
        s.append_token(t, 0.0)
        assert kv.allocate_slots(s, 1)
        s.num_computed_tokens += 1
        sizes.append(kv.num_blocks_of(s.seq_id))
    # 4 tokens fit in 1 block; tokens 5..8 need a 2nd; 9 needs a 3rd.
    assert sizes == [1, 2, 2, 2, 2, 3]


def test_allocation_is_all_or_nothing() -> None:
    kv = KVCacheManager(num_blocks=4, block_size=4)
    big = seq_of(list(range(17)))  # needs 5 blocks
    assert not kv.allocate_slots(big, big.num_tokens)
    assert kv.num_free_blocks == 4
    assert kv.num_blocks_of(big.seq_id) == 0
    kv.check_invariants()


def test_watermark_only_applies_to_admission() -> None:
    kv = KVCacheManager(num_blocks=4, block_size=4, watermark_blocks=1)
    s = seq_of(list(range(16)))  # needs all 4 blocks
    assert not kv.allocate_slots(s, 16, watermark=True)
    assert kv.allocate_slots(s, 16, watermark=False)


def test_second_request_with_same_prefix_hits_and_shares_blocks() -> None:
    kv = KVCacheManager(num_blocks=16, block_size=4)
    prefix = [7] * 8
    a, b = seq_of([*prefix, 1, 2, 3]), seq_of([*prefix, 4, 5])
    assert admit(kv, a) == 0
    assert admit(kv, b) == 8
    assert kv.block_table(a.seq_id)[:2] == kv.block_table(b.seq_id)[:2]
    assert [kv.blocks[i].ref_count for i in kv.block_table(a.seq_id)[:2]] == [2, 2]
    # Only the diverging tail was allocated for b.
    assert kv.num_used_blocks == 3 + 1
    kv.check_invariants()


def test_full_prompt_hit_still_leaves_one_token_to_compute() -> None:
    kv = KVCacheManager(num_blocks=16, block_size=4)
    tokens = list(range(8))  # exactly two full blocks
    admit(kv, seq_of(tokens))
    hits, cached = kv.get_computed_blocks(seq_of(tokens))
    assert cached == 4  # not 8: the last block is recomputed to get logits
    assert len(hits) == 1


def test_freed_blocks_stay_reusable_until_evicted() -> None:
    kv = KVCacheManager(num_blocks=4, block_size=4)
    tokens = list(range(9))
    a = seq_of(tokens)
    admit(kv, a)
    kv.free(a.seq_id)
    assert kv.num_free_blocks == 4  # free...
    assert kv.num_cached_blocks == 2  # ...but still findable
    assert admit(kv, seq_of(tokens)) == 8
    kv.check_invariants()


def test_hash_chains_through_parent() -> None:
    blk = [1, 2, 3, 4]
    assert hash_block(None, blk) != hash_block(hash_block(None, [9, 9, 9, 9]), blk)
    kv = KVCacheManager(num_blocks=16, block_size=4)
    admit(kv, seq_of([9, 9, 9, 9, *blk, 0]))
    # Same second block, different first block: no hit.
    assert kv.get_computed_blocks(seq_of([8, 8, 8, 8, *blk, 0]))[1] == 0


def test_eviction_is_lru_and_tail_first() -> None:
    kv = KVCacheManager(num_blocks=6, block_size=2)
    a = seq_of([1, 1, 2, 2, 3, 3, 0])  # 3 full blocks + 1 partial
    admit(kv, a)
    kv.free(a.seq_id)
    # Three plain blocks are free; a 4-block request forces exactly one eviction,
    # and it must be a's *tail* block, so a's prefix stays cached.
    b = seq_of([5, 5, 6, 6, 7, 7, 8])
    admit(kv, b)
    assert kv.num_evictions == 1
    assert kv.get_computed_blocks(seq_of([1, 1, 2, 2, 3, 3, 0]))[1] == 4
    kv.check_invariants()


def test_plain_free_blocks_are_used_before_evicting_cached_ones() -> None:
    kv = KVCacheManager(num_blocks=4, block_size=2)
    a = seq_of([1, 1, 2])  # one full (cached) block + one partial
    admit(kv, a)
    kv.free(a.seq_id)
    b = seq_of([3, 3, 4, 4, 5])  # needs 3 blocks; 3 plain blocks exist
    admit(kv, b)
    assert kv.num_evictions == 0
    assert kv.get_computed_blocks(seq_of([1, 1, 2]))[1] == 2


def test_duplicate_content_is_not_registered_twice() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=2)
    a, b = seq_of([1, 2, 3]), seq_of([1, 2, 3])
    # Both allocate before either has computed: no hits, two copies.
    for s in (a, b):
        hits, cached = kv.get_computed_blocks(s)
        assert cached == 0
        assert kv.allocate_slots(s, s.num_tokens, hits)
    for s in (a, b):
        s.num_computed_tokens = s.num_tokens
        kv.cache_full_blocks(s)
    assert kv.num_cached_blocks == 1
    kv.check_invariants()
    kv.free(a.seq_id)
    kv.free(b.seq_id)
    kv.check_invariants()
    assert kv.num_free_blocks == 8


def test_prefix_caching_disabled() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=2, enable_prefix_caching=False)
    admit(kv, seq_of([1, 2, 3, 4, 5]))
    assert kv.get_computed_blocks(seq_of([1, 2, 3, 4, 5])) == ([], 0)
    assert kv.num_cached_blocks == 0


def test_free_is_idempotent_and_reset_drops_free_cached_blocks() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=2)
    a = seq_of([1, 2, 3, 4, 5])
    admit(kv, a)
    kv.free(a.seq_id)
    kv.free(a.seq_id)
    assert kv.reset_prefix_cache() == 2
    assert kv.num_cached_blocks == 0
    kv.check_invariants()


def test_invariant_checker_catches_corruption() -> None:
    kv = KVCacheManager(num_blocks=4, block_size=2)
    a = seq_of([1, 2, 3])
    admit(kv, a)
    kv.blocks[kv.block_table(a.seq_id)[0]].ref_count += 1
    with pytest.raises(KVCacheError, match="refcount"):
        kv.check_invariants()


def test_can_ever_fit_accounts_for_watermark() -> None:
    kv = KVCacheManager(num_blocks=10, block_size=4, watermark_blocks=2)
    assert kv.can_ever_fit(32)
    assert not kv.can_ever_fit(33)
