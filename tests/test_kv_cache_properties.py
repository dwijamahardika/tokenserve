"""Stateful property test for the KV-cache manager.

Hypothesis drives random interleavings of admit / decode / preempt / free over
prompts drawn from a tiny alphabet (so prefixes collide constantly) and a tiny
cache (so eviction and allocation failure are common). After every step:

* bookkeeping invariants hold (no leaked blocks, refcounts == references and
  never negative, free + used == total, registry consistent);
* each live sequence holds exactly the blocks its computed tokens need;
* every prefix-cache hit returns blocks whose *content* (tracked in a shadow
  copy, as if it were the KV) matches the tokens being looked up;
* freeing a sequence never makes its cached prefix unfindable.
"""

from __future__ import annotations

import itertools

from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    consumes,
    initialize,
    invariant,
    precondition,
    rule,
)

from tokenserve.kv_cache import KVBlock, KVCacheManager, cdiv
from tokenserve.sequence import SamplingParams, Sequence

tokens = st.lists(st.integers(0, 2), min_size=1, max_size=14)


class KVCacheMachine(RuleBasedStateMachine):
    seqs = Bundle("seqs")

    def __init__(self) -> None:
        super().__init__()
        self.ids = itertools.count()
        self.live: dict[int, Sequence] = {}
        # Shadow of what each block "holds": block_id -> {offset: token}.
        self.content: dict[int, dict[int, int]] = {}

    @initialize(
        num_blocks=st.integers(2, 12),
        block_size=st.integers(1, 4),
        caching=st.booleans(),
        watermark=st.integers(0, 1),
    )
    def setup(self, num_blocks: int, block_size: int, caching: bool, watermark: int) -> None:
        self.kv = KVCacheManager(
            num_blocks,
            block_size,
            enable_prefix_caching=caching,
            watermark_blocks=min(watermark, num_blocks - 1),
        )

    # ------------------------------------------------------------ helpers

    def _check_hits(self, seq: Sequence, hits: list[KVBlock]) -> None:
        bs = self.kv.block_size
        for i, block in enumerate(hits):
            expect = seq.token_ids[i * bs : (i + 1) * bs]
            got = [self.content[block.block_id][o] for o in range(bs)]
            assert got == expect, f"cache hit on block {block.block_id} holds {got}, want {expect}"

    def _compute(self, seq: Sequence, n: int) -> None:
        """Pretend the model computed n more tokens: write the shadow, register blocks."""
        table = self.kv.block_table(seq.seq_id)
        bs = self.kv.block_size
        for p in range(seq.num_computed_tokens, seq.num_computed_tokens + n):
            self.content.setdefault(table[p // bs], {})[p % bs] = seq.token_ids[p]
        seq.num_computed_tokens += n
        self.kv.cache_full_blocks(seq)

    def _admit(self, seq: Sequence) -> bool:
        free_before = self.kv.num_free_blocks
        hits, cached = self.kv.get_computed_blocks(seq)
        self._check_hits(seq, hits)
        n = seq.num_tokens - cached
        if not self.kv.allocate_slots(seq, n, hits, watermark=True):
            assert self.kv.num_free_blocks == free_before, "failed allocation changed state"
            self.kv.free(seq.seq_id)  # drop memoised hashes
            return False
        seq.num_computed_tokens = cached
        self._compute(seq, n)
        seq.append_token(seq.num_tokens % 3, 0.0)  # "sample"
        return True

    # -------------------------------------------------------------- rules

    @rule(target=seqs, prompt=tokens)
    def add(self, prompt: list[int]) -> Sequence | None:
        seq = Sequence(next(self.ids), "r", prompt, SamplingParams(max_tokens=64))
        if not self._admit(seq):
            return None
        self.live[seq.seq_id] = seq
        return seq

    @rule(seq=seqs)
    def decode(self, seq: Sequence | None) -> None:
        if seq is None or seq.seq_id not in self.live:
            return
        free_before = self.kv.num_free_blocks
        if self.kv.allocate_slots(seq, 1):
            self._compute(seq, 1)
            seq.append_token(seq.num_tokens % 3, 0.0)
        else:
            assert self.kv.num_free_blocks == free_before

    @rule(seq=seqs)
    def preempt_and_readmit(self, seq: Sequence | None) -> None:
        """Preemption by recompute: drop all blocks, admit prompt+output again."""
        if seq is None or seq.seq_id not in self.live:
            return
        self.kv.free(seq.seq_id)
        seq.num_computed_tokens = 0
        if not self._admit(seq):
            del self.live[seq.seq_id]

    @rule(seq=consumes(seqs))
    def free(self, seq: Sequence | None) -> None:
        if seq is None or seq.seq_id not in self.live:
            return
        probe = Sequence(next(self.ids), "probe", list(seq.token_ids), SamplingParams())
        hits_before, _ = self.kv.get_computed_blocks(probe)
        self.kv.free(probe.seq_id)
        self.kv.free(seq.seq_id)
        del self.live[seq.seq_id]
        # Freeing makes blocks evictable, not gone: the same prefix still hits.
        hits_after, _ = self.kv.get_computed_blocks(probe)
        self.kv.free(probe.seq_id)
        assert [b.block_id for b in hits_after] == [b.block_id for b in hits_before]

    @precondition(lambda self: hasattr(self, "kv"))
    @rule()
    def reset_prefix_cache(self) -> None:
        self.kv.reset_prefix_cache()

    # --------------------------------------------------------- invariants

    @precondition(lambda self: hasattr(self, "kv"))
    @invariant()
    def bookkeeping(self) -> None:
        self.kv.check_invariants()
        for seq in self.live.values():
            assert self.kv.num_blocks_of(seq.seq_id) == cdiv(
                seq.num_computed_tokens, self.kv.block_size
            )

    def teardown(self) -> None:
        if not hasattr(self, "kv"):
            return
        for seq_id in list(self.live):
            self.kv.free(seq_id)
        self.kv.check_invariants()
        assert self.kv.num_free_blocks == self.kv.num_blocks, "leaked blocks"


TestKVCacheMachine = KVCacheMachine.TestCase
