from __future__ import annotations

import asyncio
import itertools
import math
import statistics

import pytest
import uvicorn
from bench.loadgen import (
    LengthDist,
    RequestResult,
    Workload,
    build_requests,
    format_summary,
    percentile,
    run_load,
    summarize,
)

from tests.conftest import make_engine, wait_for
from tokenserve.api import create_app
from tokenserve.server import _Server


def test_percentile_is_linear_interpolation() -> None:
    xs = [float(x) for x in range(1, 11)]
    assert percentile(xs, 0) == 1
    assert percentile(xs, 50) == 5.5
    assert percentile(xs, 90) == pytest.approx(9.1)
    assert percentile(xs, 100) == 10
    assert percentile([3.0], 99) == 3
    assert math.isnan(percentile([], 50))
    assert percentile([1.0, math.nan, 3.0], 50) == 2


def test_length_distributions() -> None:
    import random

    rng = random.Random(0)
    assert LengthDist.parse("fixed:7").sample(rng) == 7
    u = LengthDist.parse("uniform:3:5")
    assert {u.sample(rng) for _ in range(200)} == {3, 4, 5}
    e = LengthDist.parse("exp:50")
    xs = [e.sample(rng) for _ in range(5000)]
    assert min(xs) >= 1
    assert statistics.mean(xs) == pytest.approx(50, rel=0.1)
    for bad in ("uniform:5:3", "fixed:0", "normal:1", "fixed:x", "exp"):
        with pytest.raises(ValueError):
            LengthDist.parse(bad)


def test_poisson_arrivals_have_the_requested_rate_and_are_reproducible() -> None:
    w = Workload(4000, 20.0, LengthDist.parse("fixed:4"), LengthDist.parse("fixed:2"), seed=3)
    reqs = build_requests(w)
    gaps = [b.arrival_s - a.arrival_s for a, b in itertools.pairwise(reqs)]
    assert min(gaps) >= 0
    assert statistics.mean(gaps) == pytest.approx(1 / 20, rel=0.05)
    assert statistics.stdev(gaps) == pytest.approx(1 / 20, rel=0.1)  # exponential: sd == mean
    assert build_requests(w) == reqs
    assert all(len(r.prompt) == 4 and r.max_tokens == 2 for r in reqs)


def test_shared_prefix_workload() -> None:
    w = Workload(
        200, 10.0, LengthDist.parse("uniform:5:9"), LengthDist.parse("fixed:3"),
        shared_prefix_len=50, num_shared_prefixes=3,
    )  # fmt: skip
    reqs = build_requests(w)
    prefixes = {r.prompt[:50] for r in reqs}
    assert len(prefixes) == 3
    assert all(55 <= len(r.prompt) <= 59 for r in reqs)
    assert len({r.prompt for r in reqs}) == 200  # suffixes keep prompts distinct


def test_summary_counts_rejections_and_failures() -> None:
    w = Workload(4, 1.0, LengthDist.parse("fixed:1"), LengthDist.parse("fixed:1"))
    results = [
        RequestResult(True, 200, 0.1, 0.01, 0.5, 10, 20, end_s=2.0),
        RequestResult(True, 200, 0.3, 0.03, 0.7, 10, 20, end_s=4.0),
        RequestResult(False, 429, end_s=1.0, error="queue full"),
        RequestResult(False, 0, end_s=1.0, error="ConnectError"),
    ]
    s = summarize(w, results)
    assert (s.completed, s.rejected_429, s.failed) == (2, 1, 1)
    assert s.duration_s == 4.0
    assert s.req_per_s == 0.5
    assert s.output_tok_per_s == 10
    assert s.ttft_s.p50 == pytest.approx(0.2)
    assert "queue full" in s.errors
    assert "429: 1" in format_summary(s)


async def test_load_generator_against_a_live_server() -> None:
    eng = make_engine(
        base_latency_s=0.002, per_token_latency_s=0.00001, max_model_len=512,
        num_blocks=512, max_num_batched_tokens=256,
    )  # fmt: skip
    server = _Server(
        uvicorn.Config(create_app(eng), host="127.0.0.1", port=0, lifespan="on", log_level="error")
    )
    task = asyncio.create_task(server.serve())
    await wait_for(lambda: server.started)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        w = Workload(
            20, 100.0, LengthDist.parse("uniform:10:40"), LengthDist.parse("uniform:2:20"),
            shared_prefix_len=64, num_shared_prefixes=2,
        )  # fmt: skip
        s = await run_load(f"http://127.0.0.1:{port}", w)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
    assert s.completed == 20, s.errors
    expected_out = sum(r.max_tokens for r in build_requests(w))
    assert s.output_tok_per_s * s.duration_s == pytest.approx(expected_out)
    assert 0 < s.ttft_s.p50 < 1
    assert s.tpot_s.p50 > 0
    assert eng.scheduler.prefix_hit_tokens > 0
