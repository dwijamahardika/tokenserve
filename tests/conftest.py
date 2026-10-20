from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable

import pytest
from hypothesis import HealthCheck, settings

from tokenserve.backends.simulated import SimulatedBackend, SimulatedBackendConfig
from tokenserve.engine import Engine, EngineConfig, RequestOutput, RequestStream
from tokenserve.scheduler import SchedulerConfig, SchedulingPolicy
from tokenserve.tokenizer import ByteTokenizer

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci", max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def make_engine(
    *,
    block_size: int = 4,
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    max_model_len: int = 128,
    chunked_prefill: bool = True,
    prefix_caching: bool = True,
    policy: SchedulingPolicy = SchedulingPolicy.CONTINUOUS,
    max_waiting_seqs: int = 1000,
    max_waiting_tokens: int | None = 1_000_000,
    watermark: float = 0.0,
    base_latency_s: float = 0.0,
    per_token_latency_s: float = 0.0,
    eos_probability: float = 0.0,
    verify_kv: bool = True,
) -> Engine:
    """An engine on the simulated backend, with shadow-KV verification on."""
    tok = ByteTokenizer()
    cfg = EngineConfig(
        block_size=block_size,
        num_blocks=num_blocks,
        watermark=watermark,
        enable_prefix_caching=prefix_caching,
        max_waiting_seqs=max_waiting_seqs,
        max_waiting_tokens=max_waiting_tokens,
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            enable_chunked_prefill=chunked_prefill,
            policy=policy,
        ),
    )
    backend = SimulatedBackend(
        SimulatedBackendConfig(
            base_latency_s=base_latency_s,
            per_token_latency_s=per_token_latency_s,
            eos_probability=eos_probability,
            verify_kv=verify_kv,
        ),
        block_size=block_size,
        num_blocks=num_blocks,
        eos_token_id=tok.eos_token_id,
    )
    return Engine(cfg, backend, tok)


@pytest.fixture
async def engine() -> AsyncIterator[Engine]:
    eng = make_engine()
    await eng.start()
    yield eng
    await eng.stop()


async def collect(stream: RequestStream) -> RequestOutput:
    acc: RequestOutput | None = None
    async for out in stream:
        acc = out if acc is None else acc.merge(out)
    assert acc is not None
    return acc


async def wait_for(pred: Callable[[], bool], within_s: float = 5.0) -> None:
    """Poll a predicate over state the test cannot subscribe to (e.g. a counter)."""
    async with asyncio.timeout(within_s):
        while not pred():  # noqa: ASYNC110 - polling plain attributes is the point
            await asyncio.sleep(0.005)
