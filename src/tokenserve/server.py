"""Process wiring: build the engine, run the API and admin listeners on one loop."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import signal
from collections.abc import Iterator
from dataclasses import dataclass, field

import uvicorn
from prometheus_client import GC_COLLECTOR, PLATFORM_COLLECTOR, PROCESS_COLLECTOR

from tokenserve.api import ApiSettings, create_admin_app, create_app
from tokenserve.backends.simulated import SimulatedBackend, SimulatedBackendConfig
from tokenserve.engine import Engine, EngineConfig
from tokenserve.metrics import EngineMetrics
from tokenserve.tokenizer import ByteTokenizer

log = logging.getLogger("tokenserve")


@dataclass(frozen=True, slots=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    admin_host: str = "127.0.0.1"
    admin_port: int = 9090
    backend: str = "sim"  # "sim" | "hf"
    hf_model: str | None = None
    hf_dtype: str = "float32"
    engine: EngineConfig = field(default_factory=EngineConfig)
    sim: SimulatedBackendConfig = field(default_factory=SimulatedBackendConfig)
    api: ApiSettings = field(default_factory=ApiSettings)
    log_level: str = "info"
    access_log: bool = False
    graceful_shutdown_s: float = 10.0


def build_engine(settings: ServerSettings) -> Engine:
    metrics = EngineMetrics()
    for collector in (PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR):
        with contextlib.suppress(ValueError):  # already registered
            metrics.registry.register(collector)
    cfg = settings.engine
    if settings.backend == "sim":
        tokenizer = ByteTokenizer()
        backend = SimulatedBackend(
            settings.sim,
            block_size=cfg.block_size,
            num_blocks=cfg.num_blocks,
            eos_token_id=tokenizer.eos_token_id,
        )
        return Engine(cfg, backend, tokenizer, metrics=metrics)
    if settings.backend == "hf":
        try:
            from tokenserve.backends.hf import load_hf  # optional dependency
        except ImportError as e:
            raise SystemExit(
                f"--backend hf needs the optional extra: pip install -e '.[hf]' ({e})"
            ) from e

        if not settings.hf_model:
            raise SystemExit("--hf-model is required with --backend hf")
        hf_backend, hf_tokenizer = load_hf(settings.hf_model, dtype=settings.hf_dtype)
        limit = hf_backend.max_position_embeddings
        if limit and cfg.scheduler.max_model_len > limit:
            log.warning("capping max_model_len to the model's %d positions", limit)
            cfg = dataclasses.replace(
                cfg, scheduler=dataclasses.replace(cfg.scheduler, max_model_len=limit)
            )
        return Engine(cfg, hf_backend, hf_tokenizer, metrics=metrics)
    raise SystemExit(f"unknown backend {settings.backend!r}")


class _Server(uvicorn.Server):
    """uvicorn server that leaves signal handling to us (two servers share a loop)."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


async def serve(settings: ServerSettings) -> None:
    engine = build_engine(settings)
    app = create_app(engine, settings.api)
    admin = create_admin_app(engine)

    def config(app_: object, host: str, port: int, access_log: bool) -> uvicorn.Config:
        return uvicorn.Config(
            app_,  # type: ignore[arg-type]
            host=host,
            port=port,
            log_level=settings.log_level,
            access_log=access_log,
            lifespan="off",
            timeout_graceful_shutdown=int(settings.graceful_shutdown_s),
        )

    servers = [
        _Server(config(app, settings.host, settings.port, settings.access_log)),
        _Server(config(admin, settings.admin_host, settings.admin_port, False)),
    ]

    def shutdown() -> None:
        log.info("shutting down: draining in-flight requests")
        for s in servers:
            s.should_exit = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows
            loop.add_signal_handler(sig, shutdown)

    sc = settings.engine.scheduler
    log.info(
        "tokenserve: backend=%s policy=%s blocks=%dx%d max_num_seqs=%d "
        "max_num_batched_tokens=%d chunked_prefill=%s prefix_caching=%s auth=%s",
        engine.backend.name,
        sc.policy.value,
        settings.engine.num_blocks,
        settings.engine.block_size,
        sc.max_num_seqs,
        sc.max_num_batched_tokens,
        sc.enable_chunked_prefill,
        engine.kv.enable_prefix_caching,
        "on" if settings.api.api_key else "off",
    )
    await engine.start()
    try:
        await asyncio.gather(*(s.serve() for s in servers))
    finally:
        await engine.stop()
