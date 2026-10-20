"""The interface between the engine and whatever executes the model."""

from __future__ import annotations

import abc
from collections.abc import Sequence as Seq
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchItem:
    """One sequence's share of a step.

    Positions ``[start, start + num_new_tokens)`` of ``token_ids`` are computed
    in this step; positions before ``start`` are already in the KV cache. The
    block table plus :meth:`slot_mapping` is exactly what a paged-attention
    kernel needs to write the new KV entries and gather the context.
    """

    seq_id: int
    token_ids: Seq[int]  # prompt + output so far (read-only view)
    start: int
    num_new_tokens: int
    num_prompt_tokens: int
    block_table: Seq[int]
    sample: bool  # whether a token is sampled from the last position
    seed: int
    temperature: float = 0.0
    top_p: float = 1.0

    @property
    def end(self) -> int:
        return self.start + self.num_new_tokens

    @property
    def output_index(self) -> int:
        """Index (within the output) of the token that will be sampled."""
        return self.end - self.num_prompt_tokens

    def slot_mapping(self, block_size: int) -> list[int]:
        """Physical KV slot for each new token: ``table[p // B] * B + p % B``."""
        bt = self.block_table
        return [
            bt[p // block_size] * block_size + p % block_size for p in range(self.start, self.end)
        ]


class Backend(abc.ABC):
    """Executes one batched forward step and samples next tokens."""

    name: str = "backend"
    #: Whether KV computed for one sequence can be reused by another (needs a
    #: real paged KV store). Backends that keep private per-sequence caches
    #: must say False; the engine then disables prefix caching.
    supports_prefix_caching: bool = True

    @abc.abstractmethod
    async def execute(self, batch: Seq[BatchItem]) -> dict[int, int]:
        """Run one step. Returns ``seq_id -> sampled token`` for items with ``sample``."""

    def free(self, seq_id: int) -> None:  # noqa: B027 - optional hook
        """Drop any backend-side state for a sequence (finished or preempted)."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""
