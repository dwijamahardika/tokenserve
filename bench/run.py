"""Benchmark matrix: static vs continuous batching, prefix cache off vs on.

Each scenario starts a fresh server process on the simulated backend, drives it
with the open-loop load generator from this process, then scrapes the server's
Prometheus endpoint for scheduler-side numbers (mean batch size, preemptions,
prefix-cache hit rate).

    python -m bench.run            # full matrix, seven runs (~5 min)
    python -m bench.run --quick    # smoke run (~1 min), numbers not meaningful

Results go to bench/results/ (git-ignored): one JSON file per run plus
``latest.md`` with the table.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from bench.loadgen import LengthDist, Summary, Workload, run_load

RESULTS = Path(__file__).parent / "results"

# The simulated backend's latency model (see SimulatedBackend). An assumption,
# not a measurement: a = per-step cost, b = per-token cost.
SIM_A_MS = 10.0
SIM_B_MS = 0.1

BASE_FLAGS = [
    "--sim-base-latency-ms", str(SIM_A_MS),
    "--sim-per-token-latency-ms", str(SIM_B_MS),
    "--num-blocks", "2048",
    "--block-size", "16",
    "--max-num-seqs", "64",
    "--max-num-batched-tokens", "2048",
    # Serve every request: we want queueing to show up as latency, not 429s.
    "--max-waiting-seqs", "100000",
    "--max-waiting-tokens", "1000000000",
    "--log-level", "warning",
]  # fmt: skip

ARRIVAL_WINDOW_S = 30  # each run offers qps * 30 requests

# Chat-like traffic: moderate prompts, widely varying output lengths. Wide
# output variance is what hurts static batching (a batch lasts as long as its
# longest member).
MIXED = Workload(
    num_requests=0,
    qps=0.0,
    prompt_len=LengthDist.parse("uniform:32:512"),
    output_len=LengthDist.parse("uniform:16:512"),
    seed=1,
)

# Many requests share one of a few long system prompts.
SHARED = Workload(
    num_requests=0,
    qps=0.0,
    prompt_len=LengthDist.parse("uniform:32:128"),
    output_len=LengthDist.parse("uniform:16:128"),
    shared_prefix_len=1024,
    num_shared_prefixes=4,
    seed=2,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    flags: list[str]
    workload: Workload


def scenarios(quick: bool) -> list[Scenario]:
    window = 5 if quick else ARRIVAL_WINDOW_S

    def at(w: Workload, qps: float) -> Workload:
        return replace(w, qps=qps, num_requests=round(qps * window))

    static, cont = ["--scheduler", "static"], ["--scheduler", "continuous"]
    return [
        Scenario("static batching", static, at(MIXED, 4)),
        Scenario("continuous batching", cont, at(MIXED, 4)),
        Scenario("static batching", static, at(MIXED, 8)),
        Scenario("continuous batching", cont, at(MIXED, 8)),
        # Same load, a quarter of the KV cache: memory, not compute, caps the batch.
        Scenario("continuous, KV cache / 4", [*cont, "--num-blocks", "512"], at(MIXED, 8)),
        Scenario("continuous, prefix cache off", ["--no-prefix-caching"], at(SHARED, 8)),
        Scenario("continuous, prefix cache on", ["--prefix-caching"], at(SHARED, 8)),
    ]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@contextlib.contextmanager
def server(flags: list[str], log_path: Path) -> Iterator[tuple[str, str]]:
    port, admin = free_port(), free_port()
    cmd = [sys.executable, "-m", "tokenserve", "--port", str(port), "--admin-port", str(admin),
           *BASE_FLAGS, *flags]  # fmt: skip
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            base, admin_base = f"http://127.0.0.1:{port}", f"http://127.0.0.1:{admin}"
            deadline = time.monotonic() + 20
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited early; see {log_path}")
                with contextlib.suppress(httpx.HTTPError):
                    if httpx.get(f"{admin_base}/readyz", timeout=1).status_code == 200:
                        break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"server did not become ready; see {log_path}")
                time.sleep(0.1)
            yield base, admin_base
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()


def scrape(admin_base: str) -> dict[str, float]:
    text = httpx.get(f"{admin_base}/metrics", timeout=5).text
    m: dict[str, float] = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if not s.labels:
                m[s.name] = s.value
    return m


def server_stats(m: dict[str, float]) -> dict[str, float]:
    m = dict(m)
    m["_modelled_step_seconds"] = (
        SIM_A_MS * m.get("tokenserve_batch_size_tokens_count", 0)
        + SIM_B_MS * m.get("tokenserve_batch_size_tokens_sum", 0)
    ) / 1000

    def ratio(a: str, b: str) -> float:
        return m.get(a, 0.0) / m[b] if m.get(b) else math.nan

    return {
        "mean_batch_seqs": ratio(
            "tokenserve_batch_size_seqs_sum", "tokenserve_batch_size_seqs_count"
        ),
        "mean_batch_tokens": ratio(
            "tokenserve_batch_size_tokens_sum", "tokenserve_batch_size_tokens_count"
        ),
        "steps": m.get("tokenserve_batch_size_seqs_count", math.nan),
        "preemptions": m.get("tokenserve_preemptions_total", math.nan),
        "prefix_hit_rate": ratio(
            "tokenserve_prefix_cache_hits_tokens_total",
            "tokenserve_prefix_cache_queries_tokens_total",
        ),
        "prefill_tokens_computed": m.get("tokenserve_prefill_tokens_computed_total", math.nan),
        # Engine wall time per step vs the latency model's a + b*tokens: the
        # scheduler/bookkeeping/streaming cost that is not hidden behind "GPU" time.
        "step_overhead": ratio(
            "tokenserve_step_duration_seconds_sum",
            "_modelled_step_seconds",
        )
        - 1,
    }


def _ms(x: float) -> str:
    if math.isnan(x):
        return "-"
    return f"{x * 1000:,.0f}" if x >= 0.1 else f"{x * 1000:.1f}"


def table(rows: list[tuple[Scenario, Summary, dict[str, float]]]) -> str:
    head = (
        "| Scenario | Workload | req/s | out tok/s | TTFT p50 / p90 / p99 (ms) "
        "| TPOT p50 / p90 / p99 (ms) | mean batch | preempt | prefix hit |\n"
        "|---|---|--:|--:|--:|--:|--:|--:|--:|"
    )
    lines = [head]
    for sc, s, st in rows:
        kind = "shared-prefix" if sc.workload.num_shared_prefixes else "mixed"
        wl = f"{kind} @ {sc.workload.qps:g} req/s"
        hit = "-" if math.isnan(st["prefix_hit_rate"]) else f"{st['prefix_hit_rate']:.0%}"
        if not sc.workload.num_shared_prefixes:
            hit = "n/a"  # random prompts: nothing to share
        lines.append(
            f"| {sc.name} | {wl} | {s.req_per_s:.2f} | {s.output_tok_per_s:,.0f} "
            f"| {_ms(s.ttft_s.p50)} / {_ms(s.ttft_s.p90)} / {_ms(s.ttft_s.p99)} "
            f"| {_ms(s.tpot_s.p50)} / {_ms(s.tpot_s.p90)} / {_ms(s.tpot_s.p99)} "
            f"| {st['mean_batch_seqs']:.1f} | {st['preemptions']:.0f} | {hit} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--quick", action="store_true", help="few requests; for CI smoke tests")
    p.add_argument("--only", help="run only scenarios whose name contains this")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows: list[tuple[Scenario, Summary, dict[str, float]]] = []
    for sc in scenarios(args.quick):
        if args.only and args.only not in sc.name:
            continue
        print(f"== {sc.name}: {sc.workload.describe()}", flush=True)
        t = time.monotonic()
        with server(sc.flags, RESULTS / f"server-{stamp}-{len(rows)}.log") as (base, admin):
            summary = asyncio.run(run_load(base, sc.workload))
            stats = server_stats(scrape(admin))
        print(
            f"   {summary.completed}/{summary.num_requests} ok, {summary.req_per_s:.2f} req/s, "
            f"TTFT p50 {_ms(summary.ttft_s.p50)} ms, p99 {_ms(summary.ttft_s.p99)} ms, "
            f"step overhead {stats['step_overhead']:+.1%} ({time.monotonic() - t:.0f}s wall)",
            flush=True,
        )
        rows.append((sc, summary, stats))

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "cpus": os.cpu_count(),
        "sim_a_ms": SIM_A_MS,
        "sim_b_ms_per_token": SIM_B_MS,
        "server_flags": BASE_FLAGS,
    }
    md = table(rows)
    print()
    print(md)
    payload: dict[str, Any] = {
        "env": env,
        "runs": [
            {"scenario": sc.name, "flags": sc.flags, "summary": s.as_dict(), "server": st}
            for sc, s, st in rows
        ],
    }
    (RESULTS / f"results-{stamp}.json").write_text(json.dumps(payload, indent=2))
    (RESULTS / "latest.md").write_text(md + "\n")
    failed = [sc.name for sc, s, _ in rows if s.completed != s.num_requests]
    if failed:
        sys.exit(f"incomplete runs: {failed}")


if __name__ == "__main__":
    main()
