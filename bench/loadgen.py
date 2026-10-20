"""Open-loop load generator for OpenAI-compatible ``/v1/completions`` servers.

Requests arrive as a Poisson process at a fixed rate, independent of how fast
the server answers (open loop): a closed-loop client that waits for responses
would slow down exactly when the server does and hide queueing. Latency is
measured from each request's *scheduled* arrival time, so client-side lag can
only make numbers worse, never better (no coordinated omission).

Lengths are in characters; with the simulated backend's byte tokenizer one
ASCII character is one token. ``--shared-prefix-len P --num-prefixes K`` makes
every prompt start with one of K random P-character "system prompts".

    python -m bench.loadgen --url http://127.0.0.1:8000 --qps 8 --num-requests 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import string
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

_ALPHABET = string.ascii_lowercase + " "


# ---------------------------------------------------------------- workload


@dataclass(frozen=True, slots=True)
class LengthDist:
    """``fixed:N`` | ``uniform:LO:HI`` (inclusive) | ``exp:MEAN`` (min 1)."""

    kind: str
    a: float
    b: float = 0.0

    @classmethod
    def parse(cls, spec: str) -> LengthDist:
        kind, *args = spec.split(":")
        try:
            nums = [float(x) for x in args]
        except ValueError:
            raise ValueError(f"bad length spec {spec!r}") from None
        if kind == "fixed" and len(nums) == 1 and nums[0] >= 1:
            return cls("fixed", nums[0])
        if kind == "uniform" and len(nums) == 2 and 1 <= nums[0] <= nums[1]:
            return cls("uniform", nums[0], nums[1])
        if kind == "exp" and len(nums) == 1 and nums[0] >= 1:
            return cls("exp", nums[0])
        raise ValueError(f"bad length spec {spec!r} (fixed:N | uniform:LO:HI | exp:MEAN)")

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return int(self.a)
        if self.kind == "uniform":
            return rng.randint(int(self.a), int(self.b))
        return max(1, round(rng.expovariate(1 / self.a)))

    def __str__(self) -> str:
        if self.kind == "uniform":
            return f"uniform:{self.a:g}:{self.b:g}"
        return f"{self.kind}:{self.a:g}"


@dataclass(frozen=True, slots=True)
class Workload:
    num_requests: int
    qps: float
    prompt_len: LengthDist
    output_len: LengthDist
    shared_prefix_len: int = 0
    num_shared_prefixes: int = 0
    seed: int = 0

    def describe(self) -> str:
        s = (
            f"{self.num_requests} requests, Poisson {self.qps:g} req/s, "
            f"prompt {self.prompt_len}, output {self.output_len}"
        )
        if self.num_shared_prefixes:
            s += f", {self.num_shared_prefixes} shared prefixes x {self.shared_prefix_len}"
        return s


@dataclass(frozen=True, slots=True)
class RequestSpec:
    arrival_s: float
    prompt: str
    max_tokens: int


def _text(rng: random.Random, n: int) -> str:
    return "".join(rng.choices(_ALPHABET, k=n))


def build_requests(w: Workload) -> list[RequestSpec]:
    rng = random.Random(w.seed)
    prefixes = [_text(rng, w.shared_prefix_len) for _ in range(w.num_shared_prefixes)]
    reqs, t = [], 0.0
    for _ in range(w.num_requests):
        prefix = rng.choice(prefixes) if prefixes else ""
        prompt = prefix + _text(rng, w.prompt_len.sample(rng))
        reqs.append(RequestSpec(t, prompt, w.output_len.sample(rng)))
        t += rng.expovariate(w.qps)
    return reqs


# ------------------------------------------------------------------ results


@dataclass(slots=True)
class RequestResult:
    ok: bool
    status: int
    ttft_s: float = math.nan
    tpot_s: float = math.nan
    e2e_s: float = math.nan
    prompt_tokens: int = 0
    completion_tokens: int = 0
    end_s: float = 0.0
    error: str | None = None


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]), NaN for empty input."""
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass(slots=True)
class Pctl:
    mean: float
    p50: float
    p90: float
    p99: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Pctl:
        xs = [v for v in values if not math.isnan(v)]
        mean = sum(xs) / len(xs) if xs else math.nan
        return cls(mean, percentile(xs, 50), percentile(xs, 90), percentile(xs, 99))


@dataclass(slots=True)
class Summary:
    workload: str
    num_requests: int
    completed: int
    rejected_429: int
    failed: int
    duration_s: float
    req_per_s: float
    output_tok_per_s: float
    total_tok_per_s: float
    ttft_s: Pctl
    tpot_s: Pctl
    e2e_s: Pctl
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(w: Workload, results: Sequence[RequestResult]) -> Summary:
    ok = [r for r in results if r.ok]
    duration = max((r.end_s for r in results), default=0.0) or math.nan
    out_tokens = sum(r.completion_tokens for r in ok)
    all_tokens = out_tokens + sum(r.prompt_tokens for r in ok)
    return Summary(
        workload=w.describe(),
        num_requests=len(results),
        completed=len(ok),
        rejected_429=sum(r.status == 429 for r in results),
        failed=sum(not r.ok and r.status != 429 for r in results),
        duration_s=duration,
        req_per_s=len(ok) / duration,
        output_tok_per_s=out_tokens / duration,
        total_tok_per_s=all_tokens / duration,
        ttft_s=Pctl.of([r.ttft_s for r in ok]),
        tpot_s=Pctl.of([r.tpot_s for r in ok]),
        e2e_s=Pctl.of([r.e2e_s for r in ok]),
        errors=sorted({r.error for r in results if r.error})[:5],
    )


# ------------------------------------------------------------------- client


async def _one(
    client: httpx.AsyncClient, spec: RequestSpec, t0: float, model: str | None
) -> RequestResult:
    loop = asyncio.get_running_loop()
    start = t0 + spec.arrival_s
    await asyncio.sleep(max(0.0, start - loop.time()))
    payload: dict[str, Any] = {
        "prompt": spec.prompt,
        "max_tokens": spec.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    if model:
        payload["model"] = model
    first = last = math.nan
    usage: dict[str, int] = {}
    try:
        async with client.stream("POST", "/v1/completions", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")[:200]
                return RequestResult(False, r.status_code, end_s=loop.time() - t0, error=body)
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    return RequestResult(
                        False, 200, end_s=loop.time() - t0, error=obj["error"]["message"]
                    )
                if obj.get("usage"):
                    usage = obj["usage"]
                if any(c.get("text") for c in obj.get("choices", ())):
                    now = loop.time()
                    if math.isnan(first):
                        first = now
                    last = now
    except httpx.HTTPError as e:
        return RequestResult(False, 0, end_s=loop.time() - t0, error=f"{type(e).__name__}: {e}")
    end = loop.time()
    n_out = usage.get("completion_tokens", 0)
    return RequestResult(
        ok=True,
        status=200,
        ttft_s=first - start,
        tpot_s=(last - first) / (n_out - 1) if n_out > 1 else math.nan,
        e2e_s=end - start,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=n_out,
        end_s=end - t0,
    )


async def run_load(
    base_url: str, workload: Workload, *, model: str | None = None, timeout_s: float = 600.0
) -> Summary:
    specs = build_requests(workload)
    headers = {}
    if key := os.environ.get("TOKENSERVE_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout_s, limits=limits, headers=headers
    ) as client:
        t0 = asyncio.get_running_loop().time() + 0.05
        results = await asyncio.gather(*(_one(client, s, t0, model) for s in specs))
    return summarize(workload, results)


# ---------------------------------------------------------------------- CLI


def _ms(x: float) -> str:
    return "-" if math.isnan(x) else f"{x * 1000:.0f}" if x >= 0.1 else f"{x * 1000:.1f}"


def format_summary(s: Summary) -> str:
    return "\n".join(
        [
            f"workload     {s.workload}",
            f"completed    {s.completed}/{s.num_requests}  (429: {s.rejected_429}, "
            f"failed: {s.failed})  in {s.duration_s:.1f}s",
            f"throughput   {s.req_per_s:.2f} req/s, {s.output_tok_per_s:.0f} output tok/s, "
            f"{s.total_tok_per_s:.0f} total tok/s",
            f"TTFT ms      p50 {_ms(s.ttft_s.p50)}  p90 {_ms(s.ttft_s.p90)}  "
            f"p99 {_ms(s.ttft_s.p99)}  mean {_ms(s.ttft_s.mean)}",
            f"TPOT ms      p50 {_ms(s.tpot_s.p50)}  p90 {_ms(s.tpot_s.p90)}  "
            f"p99 {_ms(s.tpot_s.p99)}  mean {_ms(s.tpot_s.mean)}",
            f"E2E  ms      p50 {_ms(s.e2e_s.p50)}  p90 {_ms(s.e2e_s.p90)}  p99 {_ms(s.e2e_s.p99)}",
            *(f"error        {e}" for e in s.errors),
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default=None)
    p.add_argument("--qps", type=float, default=4.0)
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument(
        "--prompt-len", type=LengthDist.parse, default=LengthDist.parse("uniform:64:512")
    )
    p.add_argument(
        "--output-len", type=LengthDist.parse, default=LengthDist.parse("uniform:16:256")
    )
    p.add_argument("--shared-prefix-len", type=int, default=0)
    p.add_argument("--num-prefixes", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write the summary as JSON to this file")
    args = p.parse_args(argv)
    w = Workload(
        num_requests=args.num_requests,
        qps=args.qps,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        shared_prefix_len=args.shared_prefix_len,
        num_shared_prefixes=args.num_prefixes,
        seed=args.seed,
    )
    summary = asyncio.run(run_load(args.url, w, model=args.model))
    print(format_summary(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary.as_dict(), f, indent=2)
    if summary.completed == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
