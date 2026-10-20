"""Command line entry point: ``tokenserve`` / ``python -m tokenserve``.

Every flag can also be set as ``TOKENSERVE_<FLAG>`` (dashes → underscores).
The API key is read from ``TOKENSERVE_API_KEY`` only: flags end up in shell
history and ``ps`` output.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from tokenserve.api import ApiSettings
from tokenserve.backends.simulated import SimulatedBackendConfig
from tokenserve.engine import EngineConfig
from tokenserve.scheduler import SchedulerConfig, SchedulingPolicy
from tokenserve.server import ServerSettings, serve


def _env(name: str, default: Any, cast: Callable[[str], Any] = str) -> Any:
    raw = os.environ.get("TOKENSERVE_" + name.upper().replace("-", "_"))
    return default if raw is None or raw == "" else cast(raw)


def _bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenserve",
        description="LLM inference server: continuous batching + paged KV cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def add(flag: str, default: Any, cast: Callable[[str], Any] = str, **kw: Any) -> None:
        name = flag.lstrip("-")
        p.add_argument(flag, default=_env(name, default, cast), type=cast, **kw)

    def toggle(flag: str, default: bool, help_: str) -> None:
        name = flag.lstrip("-")
        p.add_argument(
            flag,
            default=_env(name, default, _bool),
            action=argparse.BooleanOptionalAction,
            help=help_,
        )

    g = p.add_argument_group("listeners")
    add("--host", "127.0.0.1", help="API listen address")
    add("--port", 8000, int)
    add("--admin-host", "127.0.0.1", help="metrics/health listen address")
    add("--admin-port", 9090, int)

    g = p.add_argument_group("model")
    add("--backend", "sim", choices=["sim", "hf"])
    add("--served-model-name", "tokenserve-sim")
    add("--hf-model", None, help="HF model id or path (backend=hf)")
    add("--hf-dtype", "float32", choices=["float32", "bfloat16"])

    g = p.add_argument_group("scheduler")
    add("--scheduler", "continuous", choices=[s.value for s in SchedulingPolicy])
    add("--max-num-seqs", 64, int)
    add("--max-num-batched-tokens", 2048, int)
    add("--max-model-len", 4096, int)
    toggle("--chunked-prefill", True, "split long prompts across steps")
    add("--max-waiting-seqs", 256, int, help="queue bound; beyond it requests get 429")
    add("--max-waiting-tokens", None, int, help="queued prompt-token bound (default: KV size)")

    g = p.add_argument_group("kv cache")
    add("--block-size", 16, int)
    add("--num-blocks", 2048, int)
    add("--watermark", 0.01, float, help="fraction of blocks kept free at admission")
    toggle("--prefix-caching", True, "reuse KV blocks across requests with a shared prefix")

    g = p.add_argument_group("simulated backend (assumed latency model: a + b*tokens)")
    add("--sim-base-latency-ms", 10.0, float, help="a: per-step cost")
    add("--sim-per-token-latency-ms", 0.1, float, help="b: per batched token")
    add("--sim-jitter", 0.0, float, help="relative std-dev of step latency noise")
    add("--sim-eos-probability", 0.0, float)

    g = p.add_argument_group("api")
    add("--request-timeout", 300.0, float, help="seconds, per request")
    add("--max-tokens-limit", 2048, int, help="upper bound for max_tokens")
    add("--log-level", "info", choices=["debug", "info", "warning", "error"])
    toggle("--access-log", False, "log every HTTP request")
    del g
    return p


def settings_from_args(args: argparse.Namespace) -> ServerSettings:
    engine = EngineConfig(
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        watermark=args.watermark,
        enable_prefix_caching=args.prefix_caching,
        max_waiting_seqs=args.max_waiting_seqs,
        max_waiting_tokens=args.max_waiting_tokens,
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            enable_chunked_prefill=args.chunked_prefill,
            policy=SchedulingPolicy(args.scheduler),
        ),
    )
    sim = SimulatedBackendConfig(
        base_latency_s=args.sim_base_latency_ms / 1000,
        per_token_latency_s=args.sim_per_token_latency_ms / 1000,
        jitter=args.sim_jitter,
        eos_probability=args.sim_eos_probability,
    )
    api = ApiSettings(
        model_name=args.served_model_name,
        api_key=os.environ.get("TOKENSERVE_API_KEY") or None,
        request_timeout_s=args.request_timeout,
        max_tokens_limit=args.max_tokens_limit,
    )
    return ServerSettings(
        host=args.host,
        port=args.port,
        admin_host=args.admin_host,
        admin_port=args.admin_port,
        backend=args.backend,
        hf_model=args.hf_model,
        hf_dtype=args.hf_dtype,
        engine=engine,
        sim=sim,
        api=api,
        log_level=args.log_level,
        access_log=args.access_log,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = settings_from_args(args)
    except ValueError as e:
        raise SystemExit(f"invalid configuration: {e}") from e
    loop_factory = None
    try:
        import uvloop

        loop_factory = uvloop.new_event_loop
    except ImportError:
        pass
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(serve(settings))
