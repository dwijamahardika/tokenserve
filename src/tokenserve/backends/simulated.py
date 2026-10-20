"""A deterministic stand-in for a GPU model executor.

Latency model (an *assumption*, not a measurement)::

    step_latency = a + b * num_batched_tokens        (x (1 + N(0, jitter)) if jitter > 0)

``a`` stands for the per-step cost that does not depend on batch size — on a
real GPU, mostly streaming the weights from HBM once per forward pass, plus
kernel launches. ``b`` stands for the per-token compute. Decode is dominated by
``a`` (memory-bound), prefill by ``b`` (compute-bound); that asymmetry is the
whole reason batching decode steps pays off. The defaults (a = 10 ms,
b = 0.1 ms/token) are in the regime of a 7B-class model on a single datacenter
GPU, chosen for illustration. Attention cost growing with context length is
not modelled.

Tokens are a pure function of ``(sequence seed, output position)``, so a
request produces the same text however it is batched, chunked or preempted —
which is what lets the tests assert that preemption-by-recompute is invisible
to the client.

With ``verify_kv=True`` the backend keeps a *shadow KV cache*: one slot per
``(block, offset)`` holding the token id whose KV would live there. Every step
it writes the new tokens through the block table and then reads each
sequence's whole context back through its table, failing loudly if any slot
holds the wrong token. That turns block-manager bugs (a shared block being
overwritten, a cached block evicted while referenced, a wrong table after a
prefix hit) into immediate test failures instead of silently wrong output.
"""

from __future__ import annotations

import asyncio
import random
import struct
import time
import zlib
from array import array
from collections.abc import Awaitable, Callable
from collections.abc import Sequence as Seq
from dataclasses import dataclass

from tokenserve.backends.base import Backend, BatchItem

# Generated text is drawn from this paragraph (ASCII, so one byte = one token).
SIM_TEXT = (
    b"Continuous batching lets new requests join between decode steps, so the "
    b"accelerator never idles waiting for the slowest sequence in a batch. Paged "
    b"attention keeps the KV cache in fixed-size blocks, which bounds fragmentation "
    b"to one partial block per sequence and lets requests that share a prompt share "
    b"its blocks. "
)


_MAX_CARRY_S = 0.005


class KVCorruptionError(AssertionError):
    """The shadow KV cache disagrees with a sequence's tokens."""


@dataclass(frozen=True, slots=True)
class SimulatedBackendConfig:
    base_latency_s: float = 0.010  # a
    per_token_latency_s: float = 0.0001  # b
    jitter: float = 0.0  # relative std-dev of multiplicative Gaussian noise
    eos_probability: float = 0.0  # chance of emitting EOS at each position
    verify_kv: bool = False
    seed: int = 0  # for jitter only; tokens depend on the request, not this

    def __post_init__(self) -> None:
        if self.base_latency_s < 0 or self.per_token_latency_s < 0 or self.jitter < 0:
            raise ValueError("latency parameters must be non-negative")
        if not 0.0 <= self.eos_probability <= 1.0:
            raise ValueError("eos_probability must be in [0, 1]")


class SimulatedBackend(Backend):
    name = "simulated"
    supports_prefix_caching = True

    def __init__(
        self,
        config: SimulatedBackendConfig,
        *,
        block_size: int,
        num_blocks: int,
        eos_token_id: int | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self._sleep = sleep or asyncio.sleep
        # Event-loop timers are only accurate to about 1 ms, i.e. ~10% of a
        # 10 ms step. Carrying each sleep's over/undershoot into the next one
        # keeps the *total* simulated time on the model. Off for injected
        # sleeps (tests), where wall time means nothing.
        self._compensate = sleep is None
        self._carry = 0.0
        self._rng = random.Random(config.seed)
        self._shadow = array("q", [-1]) * (num_blocks * block_size) if config.verify_kv else None
        # Counters, handy in tests.
        self.num_steps = 0
        self.num_tokens = 0
        self.total_latency_s = 0.0

    def step_latency(self, num_tokens: int) -> float:
        c = self.config
        t = c.base_latency_s + c.per_token_latency_s * num_tokens
        if c.jitter:
            t *= max(0.0, 1.0 + self._rng.gauss(0.0, c.jitter))
        return t

    async def execute(self, batch: Seq[BatchItem]) -> dict[int, int]:
        if self._shadow is not None:
            self._write_and_verify(batch, self._shadow)
        # "Sampling" happens before the sleep; the sleep is the GPU being busy.
        sampled = {item.seq_id: self._next_token(item) for item in batch if item.sample}
        n = sum(item.num_new_tokens for item in batch)
        latency = self.step_latency(n)
        self.num_steps += 1
        self.num_tokens += n
        self.total_latency_s += latency
        if not self._compensate:
            await self._sleep(latency)
            return sampled
        # carry = (actual time slept so far) - (modelled time so far)
        target = latency - self._carry
        t0 = time.perf_counter()
        if target > 0:
            await self._sleep(target)
        elapsed = time.perf_counter() - t0
        # Bounded, so a long stall (GC pause, blocked loop) is not "paid back"
        # with a burst of zero-length steps.
        self._carry = min(max(self._carry + elapsed - latency, -_MAX_CARRY_S), _MAX_CARRY_S)
        return sampled

    def _next_token(self, item: BatchItem) -> int:
        idx = item.output_index
        p = self.config.eos_probability
        if p and self.eos_token_id is not None:
            u = zlib.crc32(struct.pack("<QQ", item.seed & 0xFFFFFFFFFFFFFFFF, idx)) / 2**32
            if u < p:
                return self.eos_token_id
        return SIM_TEXT[(item.seed + idx) % len(SIM_TEXT)]

    def _write_and_verify(self, batch: Seq[BatchItem], shadow: array[int]) -> None:
        bs = self.block_size
        written: set[int] = set()
        for item in batch:
            if len(item.block_table) * bs < item.end:
                raise KVCorruptionError(
                    f"seq {item.seq_id}: block table covers {len(item.block_table) * bs} "
                    f"slots, step needs {item.end}"
                )
            for pos, slot in zip(range(item.start, item.end), item.slot_mapping(bs), strict=True):
                if slot in written:
                    raise KVCorruptionError(f"two sequences wrote KV slot {slot} in one step")
                written.add(slot)
                shadow[slot] = item.token_ids[pos]
        for item in batch:
            bt = item.block_table
            for pos in range(item.end):
                got = shadow[bt[pos // bs] * bs + pos % bs]
                if got != item.token_ids[pos]:
                    raise KVCorruptionError(
                        f"seq {item.seq_id} position {pos}: KV slot holds token {got}, "
                        f"expected {item.token_ids[pos]}"
                    )
