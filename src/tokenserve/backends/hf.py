"""Real model execution on CPU with Hugging Face ``transformers`` (optional extra).

This backend exists to show that the engine's contract is enough to drive a
real model: it receives exactly what the scheduler decided (which positions to
compute, whether to sample) and returns sampled tokens.

What it does *not* do is paged attention. ``transformers`` keeps a private,
contiguous KV cache per sequence, so here the block manager does the
accounting (admission, preemption, capacity) while the actual tensors live in
per-sequence ``past_key_values``. Consequences, all handled:

* KV cannot be shared across sequences → ``supports_prefix_caching = False``
  and the engine turns prefix caching off.
* After a preemption the engine calls :meth:`free`; the next step for that
  sequence starts at position 0 and recomputes prompt+output (recompute
  semantics, same as the paged path).
* Sequences in a step run one after another, not as one padded batch. On CPU
  the gain from batching is small, and correctness was the goal.

Install with ``pip install -e '.[hf]'`` and a CPU build of torch.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence as Seq
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokenserve.backends.base import Backend, BatchItem
from tokenserve.tokenizer import ChatMessage, default_chat_template


class HFBackend(Backend):
    name = "hf-transformers"
    supports_prefix_caching = False

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        # seq_id -> (past_key_values, number of positions it holds)
        self._caches: dict[int, tuple[Any, int]] = {}

    @property
    def max_position_embeddings(self) -> int | None:
        return getattr(self.model.config, "max_position_embeddings", None)

    async def execute(self, batch: Seq[BatchItem]) -> dict[int, int]:
        # Model code is synchronous and CPU-heavy: keep it off the event loop.
        return await asyncio.to_thread(self._execute, list(batch))

    def free(self, seq_id: int) -> None:
        self._caches.pop(seq_id, None)

    def _execute(self, batch: list[BatchItem]) -> dict[int, int]:
        sampled: dict[int, int] = {}
        with torch.inference_mode():
            for item in batch:
                past, num_cached = self._caches.pop(item.seq_id, (None, 0))
                if num_cached != item.start:
                    # The scheduler expects KV this backend does not hold
                    # (e.g. right after a preemption): recompute from scratch.
                    past, num_cached = None, 0
                ids = torch.tensor(
                    [list(item.token_ids[num_cached : item.end])],
                    dtype=torch.long,
                    device=self.device,
                )
                out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
                self._caches[item.seq_id] = (out.past_key_values, item.end)
                if item.sample:
                    sampled[item.seq_id] = self._sample(out.logits[0, -1], item)
        return sampled

    @staticmethod
    def _sample(logits: Any, item: BatchItem) -> int:
        if item.temperature <= 0.0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits.float() / item.temperature, dim=-1)
        if item.top_p < 1.0:
            sorted_probs, order = torch.sort(probs, descending=True)
            outside = torch.cumsum(sorted_probs, dim=-1) - sorted_probs > item.top_p
            sorted_probs[outside] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
        # Seeded per (sequence, position) so a recomputed sequence samples the same tokens.
        gen = torch.Generator(device="cpu")
        gen.manual_seed((item.seed * 1_000_003 + item.output_index) & 0x7FFF_FFFF_FFFF_FFFF)
        return int(torch.multinomial(probs.cpu(), 1, generator=gen).item())


class HFTokenizer:
    """Adapts a ``transformers`` tokenizer to :class:`tokenserve.tokenizer.Tokenizer`."""

    def __init__(self, tok: Any) -> None:
        self._tok = tok
        self.eos_token_id: int | None = tok.eos_token_id
        self._special = set(tok.all_special_ids)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=True))

    def decode(self, token_ids: Seq[int]) -> str:
        return str(self._tok.decode(list(token_ids), skip_special_tokens=True))

    def is_special(self, token_id: int) -> bool:
        return token_id in self._special

    def encode_chat(self, messages: Seq[ChatMessage]) -> list[int]:
        if getattr(self._tok, "chat_template", None):
            text = self._tok.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
            # The template already contains BOS where the model expects it.
            return list(self._tok.encode(text, add_special_tokens=False))
        return self.encode(default_chat_template(messages))


def load_hf(model_name: str, *, dtype: str = "float32") -> tuple[HFBackend, HFTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
    return HFBackend(model), HFTokenizer(tok)
