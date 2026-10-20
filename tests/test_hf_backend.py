"""Real-model backend. Skipped unless the optional ``hf`` extra (torch + transformers) is installed.

Uses a randomly initialised two-layer GPT-2 built from a config, so it needs
no network access: the point is that the engine's step contract (positions to
compute, when to sample, recompute after preemption) drives a real model
correctly, not that the text is meaningful.
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tests.conftest import collect  # noqa: E402
from tokenserve.backends.hf import HFBackend  # noqa: E402
from tokenserve.engine import Engine, EngineConfig  # noqa: E402
from tokenserve.scheduler import SchedulerConfig  # noqa: E402
from tokenserve.sequence import FinishReason, SamplingParams  # noqa: E402
from tokenserve.tokenizer import ByteTokenizer  # noqa: E402

TOK = ByteTokenizer()


def tiny_model() -> object:
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(
        vocab_size=TOK.vocab_size, n_positions=128, n_embd=32, n_layer=2, n_head=2
    )
    return transformers.GPT2LMHeadModel(cfg).eval()


def reference_greedy(model: object, prompt: list[int], n: int) -> list[int]:
    ids = torch.tensor([prompt])
    out = []
    with torch.inference_mode():
        for _ in range(n):  # no cache: recompute everything, the slow obvious way
            nxt = int(model(input_ids=ids).logits[0, -1].argmax())  # type: ignore[operator]
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


@pytest.mark.parametrize("num_blocks", [64, 10])  # 10 blocks forces preemption
async def test_engine_drives_a_real_model_greedily(num_blocks: int) -> None:
    model = tiny_model()
    cfg = EngineConfig(
        block_size=4,
        num_blocks=num_blocks,
        enable_prefix_caching=True,  # backend says no; engine must turn it off
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=8, max_model_len=64),
    )
    eng = Engine(cfg, HFBackend(model), TOK)
    assert not eng.kv.enable_prefix_caching
    await eng.start()
    prompts = [TOK.encode(f"prompt {i}") for i in range(4)]
    streams = [
        eng.add_request(f"r{i}", p, SamplingParams(max_tokens=12, ignore_eos=True))
        for i, p in enumerate(prompts)
    ]
    outs = await asyncio.gather(*(collect(s) for s in streams))
    await eng.wait_idle()
    if num_blocks == 10:
        assert eng.scheduler.num_preemptions > 0
    for p, o in zip(prompts, outs, strict=True):
        assert o.finish_reason is FinishReason.LENGTH
        assert o.token_ids == reference_greedy(model, p, 12)
    await eng.stop()
