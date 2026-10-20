"""Paged KV-cache manager with refcounted prefix caching.

The KV cache is carved into ``num_blocks`` fixed-size blocks of ``block_size``
token slots. Each sequence owns a *block table*: logical block ``i`` (tokens
``[i*B, (i+1)*B)``) lives in physical block ``table[i]``. A paged-attention
kernel reads a sequence's context through that table, so blocks never need to
be contiguous and memory is wasted only in each sequence's last, partially
filled block.

Prefix caching
--------------
Once a block is full *and* its KV has been computed, it is registered under a
hash of ``(parent block hash, token ids in the block)``. Because the hash
chains through the parent, it identifies the entire prefix up to that block,
so a lookup is a walk down the new prompt's block hashes until the first miss.

Blocks are refcounted. When a sequence finishes, its blocks' refcounts drop;
a block that reaches zero is *free* but keeps its contents and stays findable
by hash until it is actually reused. Free blocks live in two pools:

* ``_free_plain``  — nothing reusable (partial blocks, duplicates); handed out first.
* ``_free_cached`` — still registered in the prefix cache, kept in LRU order;
  evicted (unregistered) only when the plain pool is empty.

A sequence's blocks are released tail-first, so under pressure the tail of an
old sequence is evicted before its prefix — prefixes are the part most likely
to be shared.

Invariants (checked by :meth:`check_invariants`, exercised by the
property-based tests):

* every block is either free (refcount 0, in exactly one free pool) or used
  (refcount > 0, in no pool) — so ``free + used == total``;
* a block's refcount equals the number of block tables that reference it;
* every registered hash maps to a block carrying that hash, and vice versa;
* a block in the plain pool has no hash; a block in the cached pool has one.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import Counter, OrderedDict, deque
from collections.abc import Sequence as Seq
from dataclasses import dataclass

from tokenserve.sequence import Sequence

BlockHash = bytes


def hash_block(parent: BlockHash | None, token_ids: Seq[int]) -> BlockHash:
    """Chained, collision-resistant hash of one full block.

    SHA-256 rather than Python's ``hash()``: with a 64-bit non-cryptographic
    hash an adversarial prompt could be crafted to collide with another
    tenant's prefix and be served KV computed for different tokens.
    """
    h = hashlib.sha256(parent or b"")
    h.update(array("I", token_ids).tobytes())
    return h.digest()


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(eq=False, slots=True)
class KVBlock:
    block_id: int
    ref_count: int = 0
    # Set once the block is full, computed, and registered in the prefix cache.
    block_hash: BlockHash | None = None

    def __repr__(self) -> str:
        h = self.block_hash.hex()[:8] if self.block_hash else None
        return f"KVBlock(id={self.block_id}, ref={self.ref_count}, hash={h})"


class KVCacheError(AssertionError):
    """An internal invariant of the KV-cache manager was violated."""


class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        enable_prefix_caching: bool = True,
        watermark_blocks: int = 0,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        if not 0 <= watermark_blocks < num_blocks:
            raise ValueError("watermark_blocks must be in [0, num_blocks)")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.watermark_blocks = watermark_blocks

        self.blocks = [KVBlock(i) for i in range(num_blocks)]
        self._free_plain: deque[KVBlock] = deque(self.blocks)
        self._free_cached: OrderedDict[int, KVBlock] = OrderedDict()
        self._cached: dict[BlockHash, KVBlock] = {}
        self._tables: dict[int, list[KVBlock]] = {}
        # Per sequence: memoised hashes of its full blocks, and how many leading
        # blocks have already been considered for registration.
        self._hashes: dict[int, list[BlockHash]] = {}
        self._num_registered: dict[int, int] = {}

        self.num_evictions = 0

    # ------------------------------------------------------------------ stats

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_plain) + len(self._free_cached)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def num_cached_blocks(self) -> int:
        """Blocks currently findable by prefix hash (used or free)."""
        return len(self._cached)

    @property
    def usage(self) -> float:
        return self.num_used_blocks / self.num_blocks

    @property
    def capacity_tokens(self) -> int:
        return (self.num_blocks - self.watermark_blocks) * self.block_size

    def can_ever_fit(self, num_tokens: int) -> bool:
        """Whether a sequence of this length could run if it had the cache to itself."""
        return cdiv(num_tokens, self.block_size) <= self.num_blocks - self.watermark_blocks

    def block_table(self, seq_id: int) -> list[int]:
        return [b.block_id for b in self._tables.get(seq_id, ())]

    def num_blocks_of(self, seq_id: int) -> int:
        return len(self._tables.get(seq_id, ()))

    # ------------------------------------------------------------ prefix cache

    def _block_hashes(self, seq: Sequence) -> list[BlockHash]:
        """Hashes of every full block of ``seq.token_ids`` (memoised, append-only)."""
        hashes = self._hashes.setdefault(seq.seq_id, [])
        bs = self.block_size
        for i in range(len(hashes), seq.num_tokens // bs):
            hashes.append(
                hash_block(hashes[-1] if hashes else None, seq.token_ids[i * bs : (i + 1) * bs])
            )
        return hashes

    def get_computed_blocks(self, seq: Sequence) -> tuple[list[KVBlock], int]:
        """Longest cached prefix of ``seq``: ``(blocks, num_cached_tokens)``.

        Read-only; pass the blocks to :meth:`allocate_slots` to take references.
        At least one token is always left uncached so the step that admits the
        sequence produces logits to sample from.
        """
        if not self.enable_prefix_caching:
            return [], 0
        if seq.seq_id in self._tables:
            raise KVCacheError(f"seq {seq.seq_id} already has blocks")
        max_blocks = (seq.num_tokens - 1) // self.block_size
        hits: list[KVBlock] = []
        for h in self._block_hashes(seq)[:max_blocks]:
            block = self._cached.get(h)
            if block is None:
                break
            hits.append(block)
        return hits, len(hits) * self.block_size

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Register blocks of ``seq`` that became full and computed this step."""
        if not self.enable_prefix_caching:
            return
        table = self._tables[seq.seq_id]
        num_full = seq.num_computed_tokens // self.block_size
        start = self._num_registered.get(seq.seq_id, 0)
        if num_full <= start:
            return
        hashes = self._block_hashes(seq)
        for i in range(start, num_full):
            block = table[i]
            if block.block_hash is not None:
                continue  # it was a cache hit, already registered
            h = hashes[i]
            if h not in self._cached:
                block.block_hash = h
                self._cached[h] = block
            # else: another sequence computed the same prefix concurrently and
            # registered first. This copy stays private and is freed normally.
        self._num_registered[seq.seq_id] = num_full

    # -------------------------------------------------------------- allocation

    def allocate_slots(
        self,
        seq: Sequence,
        num_new_tokens: int,
        computed_blocks: Seq[KVBlock] = (),
        *,
        watermark: bool = False,
    ) -> bool:
        """Grow ``seq``'s block table to hold ``num_new_tokens`` more tokens.

        ``computed_blocks`` (from :meth:`get_computed_blocks`) are prefix-cache
        hits placed at the front of a new sequence's table. With
        ``watermark=True`` (admission of a waiting sequence) the allocation must
        leave ``watermark_blocks`` free, so running sequences can still grow
        without immediately preempting the newcomer.

        All-or-nothing: returns ``False`` and changes nothing if the blocks are
        not available.
        """
        table = self._tables.get(seq.seq_id)
        if computed_blocks and table:
            raise KVCacheError("prefix hits can only be attached to a new sequence")
        num_existing = len(table) if table else 0
        num_computed = seq.num_computed_tokens + len(computed_blocks) * self.block_size
        num_required = cdiv(num_computed + num_new_tokens, self.block_size)
        num_new_blocks = max(0, num_required - num_existing - len(computed_blocks))
        # Hits that are currently free leave the free pool when we take a reference.
        num_revived = sum(1 for b in computed_blocks if b.ref_count == 0)
        reserve = self.watermark_blocks if watermark else 0
        if num_new_blocks + num_revived > self.num_free_blocks - reserve:
            return False

        if table is None:
            table = self._tables[seq.seq_id] = []
        for block in computed_blocks:
            self._take_ref(block)
            table.append(block)
        for _ in range(num_new_blocks):
            table.append(self._allocate_block())
        return True

    def free(self, seq_id: int) -> None:
        """Release all of a sequence's blocks (finish, abort, or preemption). Idempotent."""
        self._hashes.pop(seq_id, None)
        self._num_registered.pop(seq_id, None)
        table = self._tables.pop(seq_id, None)
        if not table:
            return
        # Tail first: the tail becomes least-recently-used and is evicted before
        # the prefix, which is the part other requests are likely to share.
        for block in reversed(table):
            block.ref_count -= 1
            if block.ref_count < 0:
                raise KVCacheError(f"negative refcount on {block}")
            if block.ref_count == 0:
                if block.block_hash is not None:
                    self._free_cached[block.block_id] = block
                else:
                    self._free_plain.append(block)

    def reset_prefix_cache(self) -> int:
        """Drop every *free* cached block from the prefix cache. Returns how many."""
        n = len(self._free_cached)
        while self._free_cached:
            _, block = self._free_cached.popitem(last=False)
            self._unregister(block)
            self._free_plain.append(block)
        return n

    def _take_ref(self, block: KVBlock) -> None:
        if block.ref_count == 0:
            del self._free_cached[block.block_id]
        block.ref_count += 1

    def _allocate_block(self) -> KVBlock:
        if self._free_plain:
            block = self._free_plain.popleft()
        else:
            _, block = self._free_cached.popitem(last=False)  # least recently used
            self._unregister(block)
            self.num_evictions += 1
        if block.ref_count != 0:
            raise KVCacheError(f"allocating a block that is in use: {block}")
        block.ref_count = 1
        return block

    def _unregister(self, block: KVBlock) -> None:
        h = block.block_hash
        if h is None or self._cached.get(h) is not block:
            raise KVCacheError(f"evicting a block that is not registered: {block}")
        del self._cached[h]
        block.block_hash = None

    # -------------------------------------------------------------- invariants

    def check_invariants(self) -> None:
        """Raise :class:`KVCacheError` if any bookkeeping invariant is broken."""

        def fail(msg: str) -> None:
            raise KVCacheError(msg)

        plain_ids = [b.block_id for b in self._free_plain]
        cached_ids = list(self._free_cached)
        if len(set(plain_ids)) != len(plain_ids):
            fail("duplicate block in plain free pool")
        if set(plain_ids) & set(cached_ids):
            fail("block in both free pools")
        plain_set = set(plain_ids)
        free_ids = plain_set | set(cached_ids)

        refs: Counter[int] = Counter()
        for seq_id, table in self._tables.items():
            ids = [b.block_id for b in table]
            if len(set(ids)) != len(ids):
                fail(f"seq {seq_id} references a block twice")
            refs.update(ids)

        num_used = 0
        for b in self.blocks:
            if b.ref_count < 0:
                fail(f"negative refcount: {b}")
            if b.ref_count != refs[b.block_id]:
                fail(f"refcount {b.ref_count} != {refs[b.block_id]} references: {b}")
            if (b.ref_count == 0) != (b.block_id in free_ids):
                fail(f"block is neither free nor used (or both): {b}")
            if b.block_id in self._free_cached and b.block_hash is None:
                fail(f"cached free block without hash: {b}")
            if b.block_id in plain_set and b.block_hash is not None:
                fail(f"plain free block with hash: {b}")
            if b.block_hash is not None and self._cached.get(b.block_hash) is not b:
                fail(f"block hash not registered: {b}")
            num_used += b.ref_count > 0
        for h, b in self._cached.items():
            if b.block_hash != h:
                fail(f"registry entry points at block with another hash: {b}")
        if self.num_free_blocks + num_used != self.num_blocks:
            fail(f"free {self.num_free_blocks} + used {num_used} != total {self.num_blocks}")
