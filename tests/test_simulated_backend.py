from __future__ import annotations

import statistics
import time

import pytest

from tokenserve.backends.base import BatchItem
from tokenserve.backends.simulated import (
    SIM_TEXT,
    KVCorruptionError,
    SimulatedBackend,
    SimulatedBackendConfig,
)


def item(seq_id: int, tokens: list[int], start: int, n: int, table: list[int], *,
         sample: bool = True, prompt: int | None = None, seed: int = 0) -> BatchItem:  # fmt: skip
    return BatchItem(
        seq_id=seq_id,
        token_ids=tokens,
        start=start,
        num_new_tokens=n,
        num_prompt_tokens=len(tokens) if prompt is None else prompt,
        block_table=table,
        sample=sample,
        seed=seed,
    )


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, s: float) -> None:
        self.calls.append(s)


async def test_step_latency_is_a_plus_b_times_tokens() -> None:
    sleep = FakeSleep()
    cfg = SimulatedBackendConfig(base_latency_s=0.010, per_token_latency_s=0.001)
    be = SimulatedBackend(cfg, block_size=4, num_blocks=16, sleep=sleep)
    await be.execute([item(1, [1] * 8, 0, 8, [0, 1]), item(2, [2] * 3, 2, 1, [2])])
    await be.execute([item(1, [1] * 9, 8, 1, [0, 1, 3])])
    assert sleep.calls == pytest.approx([0.010 + 9 * 0.001, 0.010 + 0.001])
    assert be.num_steps == 2
    assert be.num_tokens == 10


def test_jitter_is_non_negative_and_centred() -> None:
    cfg = SimulatedBackendConfig(base_latency_s=0.01, per_token_latency_s=0.0, jitter=0.2, seed=1)
    be = SimulatedBackend(cfg, block_size=4, num_blocks=4)
    samples = [be.step_latency(10) for _ in range(5000)]
    assert min(samples) >= 0
    assert statistics.mean(samples) == pytest.approx(0.01, rel=0.02)
    assert statistics.stdev(samples) == pytest.approx(0.002, rel=0.1)


async def test_tokens_are_a_function_of_seed_and_position_only() -> None:
    be = SimulatedBackend(SimulatedBackendConfig(0, 0), block_size=4, num_blocks=16)
    # Output index 3 of a seq with a 5-token prompt, reached two different ways.
    a = await be.execute([item(1, [7] * 8, 7, 1, [0, 1], prompt=5, seed=42)])  # decode
    b = await be.execute([item(2, [7] * 8, 0, 8, [2, 3], prompt=5, seed=42)])  # recompute
    assert a[1] == b[2] == SIM_TEXT[(42 + 3) % len(SIM_TEXT)]
    c = await be.execute([item(3, [7] * 8, 0, 8, [4, 5], prompt=5, sample=False)])
    assert c == {}


async def test_eos_probability_one_always_emits_eos() -> None:
    be = SimulatedBackend(
        SimulatedBackendConfig(0, 0, eos_probability=1.0), block_size=4, num_blocks=4,
        eos_token_id=256,
    )  # fmt: skip
    assert await be.execute([item(1, [1, 2], 0, 2, [0])]) == {1: 256}


def test_slot_mapping() -> None:
    it = item(1, list(range(10)), 3, 6, [7, 2, 5])
    assert it.slot_mapping(4) == [7 * 4 + 3, 2 * 4 + 0, 2 * 4 + 1, 2 * 4 + 2, 2 * 4 + 3, 5 * 4 + 0]


async def test_shadow_kv_detects_block_table_bugs() -> None:
    cfg = SimulatedBackendConfig(0, 0, verify_kv=True)
    be = SimulatedBackend(cfg, block_size=2, num_blocks=8)
    await be.execute([item(1, [1, 2, 3], 0, 3, [0, 1])])
    # Correct prefix sharing: seq 2 reuses block 0 (same tokens) and decodes.
    await be.execute([item(2, [1, 2, 9], 2, 1, [0, 4])])
    # Bug: seq 3 claims block 0 holds its prefix, but its tokens differ.
    with pytest.raises(KVCorruptionError, match="position 0"):
        await be.execute([item(3, [5, 5, 9], 2, 1, [0, 5])])
    # Bug: two sequences write the same slot in one step.
    with pytest.raises(KVCorruptionError, match="wrote KV slot"):
        await be.execute([item(4, [1], 0, 1, [6]), item(5, [1], 0, 1, [6])])
    # Bug: block table too short for the step.
    with pytest.raises(KVCorruptionError, match="covers"):
        await be.execute([item(6, [1, 2, 3], 0, 3, [7])])


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SimulatedBackendConfig(base_latency_s=-1)
    with pytest.raises(ValueError, match="eos_probability"):
        SimulatedBackendConfig(eos_probability=2)


async def test_timer_compensation_keeps_total_simulated_time_on_the_model() -> None:
    # Event-loop sleeps are only ~1 ms accurate; the backend carries the error
    # forward so the sum stays on a + b*n.
    be = SimulatedBackend(SimulatedBackendConfig(0.0031, 0.0), block_size=4, num_blocks=4)
    t = time.perf_counter()
    for _ in range(60):
        await be.execute([item(1, [1], 0, 1, [0], sample=False)])
    assert time.perf_counter() - t == pytest.approx(be.total_latency_s, rel=0.15)
