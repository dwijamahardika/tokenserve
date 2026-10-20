from __future__ import annotations

import asyncio
from collections.abc import Sequence as Seq

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import collect, make_engine, wait_for
from tokenserve.backends.base import Backend, BatchItem
from tokenserve.backends.simulated import SIM_TEXT
from tokenserve.engine import (
    Engine,
    EngineConfig,
    EngineDeadError,
    QueueFullError,
    RequestOutput,
    RequestRejectedError,
    RequestStream,
)
from tokenserve.scheduler import SchedulingPolicy
from tokenserve.sequence import FinishReason, SamplingParams, Sequence
from tokenserve.tokenizer import ByteTokenizer

TOK = ByteTokenizer()


def expected_text(prompt: str, n: int) -> str:
    """What the simulated backend generates for this prompt (see SimulatedBackend)."""
    seed = Sequence(0, "x", TOK.encode(prompt), SamplingParams()).sampling_seed
    return bytes(SIM_TEXT[(seed + i) % len(SIM_TEXT)] for i in range(n)).decode()


async def run_requests(
    engine: Engine, prompts: list[str], max_tokens: int, **params: object
) -> list[RequestOutput]:
    streams = [
        engine.add_request(f"r{i}", TOK.encode(p), SamplingParams(max_tokens=max_tokens, **params))  # type: ignore[arg-type]
        for i, p in enumerate(prompts)
    ]
    return list(await asyncio.gather(*(collect(s) for s in streams)))


PROMPTS = [f"request number {i}: tell me about paged attention" for i in range(6)]


async def test_output_is_independent_of_batching_and_chunking() -> None:
    alone = make_engine(max_num_seqs=1, max_num_batched_tokens=128)
    packed = make_engine(max_num_seqs=8, max_num_batched_tokens=16)  # forces chunked prefill
    results = []
    for eng in (alone, packed):
        await eng.start()
        results.append([o.text for o in await run_requests(eng, PROMPTS, 30)])
        await eng.stop()
    assert results[0] == results[1] == [expected_text(p, 30) for p in PROMPTS]


@pytest.mark.parametrize("policy", list(SchedulingPolicy))
async def test_preempted_requests_still_complete_with_correct_output(
    policy: SchedulingPolicy,
) -> None:
    # 20 blocks x 4 tokens = 80 slots; six requests that each grow to 30 tokens
    # all fit at admission but not once they have generated their output.
    prompts = [f"req {i}" for i in range(6)]
    eng = make_engine(num_blocks=20, block_size=4, max_num_seqs=6, policy=policy)
    await eng.start()
    outs = await run_requests(eng, prompts, 24)
    await eng.wait_idle()
    assert eng.scheduler.num_preemptions > 0, "test did not exercise preemption"
    for p, o in zip(prompts, outs, strict=True):
        assert o.finish_reason is FinishReason.LENGTH
        assert o.completion_tokens == 24
        assert o.text == expected_text(p, 24)
    assert eng.kv.num_free_blocks == eng.kv.num_blocks
    eng.kv.check_invariants()
    await eng.stop()


async def test_abort_mid_generation_frees_blocks() -> None:
    eng = make_engine(base_latency_s=0.002, max_model_len=2048, num_blocks=1024)
    await eng.start()
    stream = eng.add_request("long", TOK.encode("hello"), SamplingParams(max_tokens=1500))
    seen = 0
    async for out in stream:
        seen += 1
        if seen == 3:
            assert eng.kv.num_used_blocks > 0
            eng.abort("long")
        if out.finished:
            assert out.finish_reason is FinishReason.ABORT
            assert out.completion_tokens < 1500
    await asyncio.wait_for(eng.wait_idle(), 2)
    assert eng.kv.num_free_blocks == eng.kv.num_blocks
    assert eng.metrics.requests_finished.labels("abort")._value.get() == 1
    await eng.stop()


async def test_abort_while_queued_never_runs() -> None:
    eng = make_engine()
    stream = eng.add_request("q", TOK.encode("hi"), SamplingParams(max_tokens=4))
    eng.abort("q")
    eng.abort("unknown-id")  # harmless
    await eng.start()
    out = await collect(stream)
    assert out.finish_reason is FinishReason.ABORT
    assert eng.backend.num_steps == 0  # type: ignore[attr-defined]
    await eng.stop()


async def test_stop_string_truncates_and_is_never_streamed(engine: Engine) -> None:
    prompt = "stop strings"
    ref = expected_text(prompt, 60)
    stop = ref[25:29]
    assert ref.find(stop) == 25, "pick a stop string that first occurs at 25"
    stream = engine.add_request(
        "s", TOK.encode(prompt), SamplingParams(max_tokens=60, stop=(stop,))
    )
    deltas = []
    async for out in stream:
        deltas.append(out.text)
        last = out
    assert "".join(deltas) == ref[:25]
    assert last.finish_reason is FinishReason.STOP
    await engine.wait_idle()
    assert engine.kv.num_free_blocks == engine.kv.num_blocks


async def test_multibyte_prompt_and_eos() -> None:
    eng = make_engine(eos_probability=1.0)
    await eng.start()
    out = await collect(
        eng.add_request("e", TOK.encode("héllo 日本"), SamplingParams(max_tokens=5))
    )
    assert out.finish_reason is FinishReason.STOP
    assert out.text == ""
    assert out.completion_tokens == 1  # the EOS itself
    out = await collect(
        eng.add_request("i", TOK.encode("x"), SamplingParams(max_tokens=5, ignore_eos=True))
    )
    assert out.finish_reason is FinishReason.LENGTH
    await eng.stop()


def test_backpressure_bounds_queue_length_and_queued_tokens() -> None:
    eng = make_engine(max_waiting_seqs=2, max_waiting_tokens=1_000)
    eng.add_request("a", [1], SamplingParams())
    eng.add_request("b", [1], SamplingParams())
    with pytest.raises(QueueFullError) as exc:
        eng.add_request("c", [1], SamplingParams())
    assert exc.value.retry_after_s >= 1

    eng = make_engine(max_waiting_seqs=100, max_waiting_tokens=10)
    eng.add_request("a", [1] * 8, SamplingParams())
    with pytest.raises(QueueFullError):
        eng.add_request("b", [1] * 5, SamplingParams())
    assert eng.metrics.requests_rejected.labels("queue_full")._value.get() == 1


def test_admission_control_rejects_requests_that_can_never_fit() -> None:
    eng = make_engine(max_model_len=64, num_blocks=8, block_size=4)  # 32 KV slots
    with pytest.raises(RequestRejectedError) as e1:
        eng.add_request("a", [1] * 60, SamplingParams(max_tokens=10))
    assert e1.value.code == "context_length_exceeded"
    with pytest.raises(RequestRejectedError) as e2:
        eng.add_request("b", [1] * 30, SamplingParams(max_tokens=10))
    assert e2.value.code == "kv_capacity_exceeded"
    eng.add_request("c", [1], SamplingParams())
    with pytest.raises(ValueError, match="duplicate"):
        eng.add_request("c", [1], SamplingParams())


class ExplodingBackend(Backend):
    async def execute(self, batch: Seq[BatchItem]) -> dict[int, int]:
        raise RuntimeError("device lost")


async def test_backend_failure_fails_requests_and_marks_engine_dead() -> None:
    eng = Engine(EngineConfig(num_blocks=16, block_size=4), ExplodingBackend(), TOK)
    await eng.start()
    out = await collect(eng.add_request("x", [1, 2, 3], SamplingParams()))
    assert out.finish_reason is FinishReason.ERROR
    assert isinstance(eng.dead, RuntimeError)
    with pytest.raises(EngineDeadError):
        eng.add_request("y", [1], SamplingParams())
    await eng.stop()


async def test_stream_coalesces_outputs_for_a_slow_consumer() -> None:
    s = RequestStream("r")
    for i in range(3):
        s.put(RequestOutput("r", str(i), [i], False, None, 1, i + 1))
    s.put(RequestOutput("r", "!", [9], True, FinishReason.LENGTH, 1, 4))
    out = await anext(s)
    assert (out.text, out.token_ids, out.finished) == ("012!", [0, 1, 2, 9], True)
    with pytest.raises(StopAsyncIteration):
        await anext(s)


async def test_queue_time_and_latency_metrics_are_recorded(engine: Engine) -> None:
    await run_requests(engine, PROMPTS[:3], 5)
    m = engine.metrics
    assert m.ttft._sum.get() > 0 or m.ttft._buckets[-1].get() == 3  # observed 3 times
    assert sum(b.get() for b in m.queue_time._buckets) == 3
    assert sum(b.get() for b in m.tpot._buckets) == 3
    assert sum(b.get() for b in m.itl._buckets) == 3 * 4
    assert m.generation_tokens._value.get() == 15


# ----------------------------------------------------------- property test

request = st.tuples(
    st.text(alphabet="ab", min_size=1, max_size=20),  # tiny alphabet: shared prefixes
    st.integers(1, 16),  # max_tokens
    st.sampled_from([None, 0, 2]),  # abort: never / immediately / after 2 outputs
)


@given(
    reqs=st.lists(request, min_size=1, max_size=10),
    num_blocks=st.integers(10, 24),
    block_size=st.sampled_from([1, 2, 4]),
    budget=st.integers(8, 48),
    max_num_seqs=st.integers(1, 6),
    policy=st.sampled_from(list(SchedulingPolicy)),
    caching=st.booleans(),
)
def test_engine_end_to_end_property(
    reqs: list[tuple[str, int, int | None]],
    num_blocks: int,
    block_size: int,
    budget: int,
    max_num_seqs: int,
    policy: SchedulingPolicy,
    caching: bool,
) -> None:
    """Random workloads on a tiny cache with the shadow-KV verifier on.

    Every request either completes with exactly the deterministic expected
    text, or (if aborted) gets a prefix of it; afterwards no block is leaked.
    """

    async def scenario() -> None:
        eng = make_engine(
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_batched_tokens=budget,
            max_num_seqs=max_num_seqs,
            max_model_len=40,
            policy=policy,
            prefix_caching=caching,
        )
        await eng.start()

        async def one(i: int, prompt: str, max_tokens: int, abort_after: int | None) -> None:
            rid = f"r{i}"
            try:
                stream = eng.add_request(
                    rid, TOK.encode(prompt), SamplingParams(max_tokens=max_tokens)
                )
            except RequestRejectedError:
                assert not eng.kv.can_ever_fit(len(prompt) + max_tokens)
                return
            if abort_after == 0:
                eng.abort(rid)
            text, n = "", 0
            async for out in stream:
                text += out.text
                n += 1
                if abort_after is not None and n == abort_after:
                    eng.abort(rid)
                last = out
            ref = expected_text(prompt, max_tokens)
            if last.finish_reason is FinishReason.ABORT:
                assert abort_after is not None
                assert ref.startswith(text)
            else:
                assert last.finish_reason is FinishReason.LENGTH
                assert text == ref

        await asyncio.gather(*(one(i, *r) for i, r in enumerate(reqs)))
        await asyncio.wait_for(eng.wait_idle(), 5)
        eng.kv.check_invariants()
        assert eng.kv.num_free_blocks == eng.kv.num_blocks
        assert eng.dead is None
        await eng.stop()

    asyncio.run(scenario())


async def test_wait_for_helper_times_out() -> None:
    with pytest.raises(TimeoutError):
        await wait_for(lambda: False, within_s=0.02)
