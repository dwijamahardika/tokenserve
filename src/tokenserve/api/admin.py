"""Control-plane HTTP app, served on a separate (loopback by default) listener."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tokenserve.engine import Engine


def create_admin_app(engine: Engine) -> Starlette:
    async def metrics(_: Request) -> Response:
        return Response(generate_latest(engine.metrics.registry), media_type=CONTENT_TYPE_LATEST)

    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def readyz(_: Request) -> Response:
        ready = engine.is_running and engine.dead is None
        return JSONResponse({"ready": ready}, 200 if ready else 503)

    async def state(_: Request) -> Response:
        return JSONResponse(engine.snapshot())

    return Starlette(
        routes=[
            Route("/metrics", metrics),
            Route("/healthz", healthz),
            Route("/readyz", readyz),
            Route("/debug/state", state),
        ]
    )
