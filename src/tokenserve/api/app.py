"""OpenAI-compatible HTTP API: ``/v1/completions`` and ``/v1/chat/completions``.

Cancellation is the part that matters most for an inference server: a client
that goes away must stop costing KV blocks and compute. Every request path
ends in ``engine.abort(request_id)`` (a no-op if the request already finished):

* streaming   — :class:`EngineStreamResponse` watches for ``http.disconnect``
  while it streams and calls ``on_close`` however the response ends;
* non-streaming — the handler races the result against a disconnect watcher
  and the per-request deadline.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tokenserve.api.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    _Base,
    error_body,
)
from tokenserve.engine import (
    Engine,
    EngineDeadError,
    QueueFullError,
    RequestOutput,
    RequestRejectedError,
    RequestStream,
)
from tokenserve.sequence import FinishReason, SamplingParams

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@dataclass(frozen=True, slots=True)
class ApiSettings:
    model_name: str = "tokenserve-sim"
    api_key: str | None = None  # from the environment only, never a flag
    request_timeout_s: float = 300.0
    max_tokens_limit: int = 2048
    default_completion_max_tokens: int = 16
    max_body_bytes: int = 1 << 20


class ApiError(HTTPException):
    def __init__(
        self,
        status: int,
        message: str,
        type_: str = "invalid_request_error",
        *,
        code: str | int | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status, message, headers)
        self.body = error_body(message, type_, param=param, code=code)


# --------------------------------------------------------------------- ASGI bits


class BodySizeLimitMiddleware:
    """413 for bodies over the limit, including chunked uploads without Content-Length."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope["headers"]:
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                resp = JSONResponse(
                    error_body("request body too large", "invalid_request_error"), 413
                )
                await resp(scope, receive, send)
                return
        received = 0

        async def limited() -> Message:
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.max_bytes:
                    raise ApiError(413, "request body too large")
            return msg

        await self.app(scope, limited, send)


class EngineStreamResponse(StreamingResponse):
    """SSE response that always runs ``on_close`` — on completion, error or disconnect.

    Starlette's own disconnect handling cancels the body iterator, but whether
    the iterator's ``finally`` runs promptly depends on where the cancellation
    lands. Owning the task group makes the cleanup deterministic.
    """

    def __init__(self, content: AsyncIterator[str], on_close: Callable[[], None]) -> None:
        super().__init__(content, media_type="text/event-stream", headers=_SSE_HEADERS)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with anyio.create_task_group() as tg:

                async def run(fn: Callable[[], Awaitable[None]]) -> None:
                    with contextlib.suppress(OSError):
                        await fn()
                    tg.cancel_scope.cancel()

                tg.start_soon(run, lambda: self.stream_response(send))
                tg.start_soon(run, lambda: self.listen_for_disconnect(receive))
        finally:
            self._on_close()
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


async def _wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


def _sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


# ------------------------------------------------------------------ formatting


class _Formatter:
    """Builds OpenAI response objects for one request."""

    def __init__(self, kind: str, rid: str, model: str) -> None:
        self.chat = kind == "chat"
        self.rid = rid
        self.model = model
        self.created = int(time.time())

    def _envelope(self, obj: str, choices: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": self.rid,
            "object": obj,
            "created": self.created,
            "model": self.model,
            "choices": choices,
        }

    def chunk_obj(self) -> str:
        return "chat.completion.chunk" if self.chat else "text_completion"

    def preamble(self) -> list[dict[str, Any]]:
        if not self.chat:
            return []
        delta = {"role": "assistant", "content": ""}
        return [self._envelope(self.chunk_obj(), [self._choice(delta=delta, finish=None)])]

    def _choice(
        self, *, text: str | None = None, delta: dict[str, Any] | None = None, finish: str | None
    ) -> dict[str, Any]:
        if self.chat:
            return {"index": 0, "delta": delta or {}, "logprobs": None, "finish_reason": finish}
        return {"index": 0, "text": text or "", "logprobs": None, "finish_reason": finish}

    def delta(self, text: str, finish: str | None = None) -> dict[str, Any]:
        choice = self._choice(text=text, delta={"content": text} if text else {}, finish=finish)
        return self._envelope(self.chunk_obj(), [choice])

    def usage(self, prompt: int, completion: int) -> dict[str, int]:
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def usage_chunk(self, prompt: int, completion: int) -> dict[str, Any]:
        return {**self._envelope(self.chunk_obj(), []), "usage": self.usage(prompt, completion)}

    def full(self, text: str, finish: str, prompt: int, completion: int) -> dict[str, Any]:
        if self.chat:
            choice: dict[str, Any] = {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": finish,
            }
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
            obj = "text_completion"
        return {**self._envelope(obj, [choice]), "usage": self.usage(prompt, completion)}


def _client_error_for(out: RequestOutput) -> dict[str, Any]:
    if out.finish_reason is FinishReason.ERROR:
        return error_body("internal engine error", "internal_error", code=500)
    return error_body("request aborted by the server", "server_error", code=503)


# ------------------------------------------------------------------------- app


def create_app(engine: Engine, settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    tokenizer = engine.tokenizer
    max_model_len = engine.config.scheduler.max_model_len

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        owns_engine = not engine.is_running
        if owns_engine:
            await engine.start()
        try:
            yield
        finally:
            if owns_engine:
                await engine.stop()

    app = FastAPI(title="tokenserve", version="0.1.0", lifespan=lifespan, docs_url=None,
                  redoc_url=None)  # fmt: skip
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)

    # -------------------------------------------------------------- errors

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body, exc.status_code, headers=exc.headers)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        body = error_body(str(exc.detail), "invalid_request_error", code=exc.status_code)
        return JSONResponse(body, exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = [str(p) for p in first.get("loc", ()) if p != "body"]
        msg = first.get("msg", "invalid request")
        message = f"{'.'.join(loc)}: {msg}" if loc else msg
        body = error_body(message, "invalid_request_error", param=".".join(loc) or None)
        return JSONResponse(body, 400)

    # ---------------------------------------------------------------- auth

    async def require_api_key(request: Request) -> None:
        if not settings.api_key:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.encode(), settings.api_key.encode()
        ):
            raise ApiError(
                401,
                "invalid or missing API key",
                "authentication_error",
                code="invalid_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = [Depends(require_api_key)]

    # ------------------------------------------------------------- helpers

    def check_common(req: _Base) -> None:
        if req.model and req.model != settings.model_name:
            raise ApiError(404, f"model '{req.model}' does not exist", code="model_not_found",
                           param="model")  # fmt: skip
        if req.n != 1:
            raise ApiError(400, "only n=1 is supported", param="n")

    def resolve_max_tokens(requested: int | None, default: int, prompt_len: int) -> int:
        if requested is not None and requested > settings.max_tokens_limit:
            raise ApiError(
                400,
                f"max_tokens must be <= {settings.max_tokens_limit}",
                param="max_tokens",
            )
        if requested is not None:
            return requested
        return max(1, min(default, settings.max_tokens_limit, max_model_len - prompt_len))

    def submit(rid: str, prompt_ids: list[int], params: SamplingParams) -> RequestStream:
        try:
            return engine.add_request(rid, prompt_ids, params)
        except RequestRejectedError as e:
            raise ApiError(400, str(e), code=e.code) from e
        except QueueFullError as e:
            raise ApiError(
                429,
                str(e),
                "rate_limit_error",
                code="queue_full",
                headers={"Retry-After": str(e.retry_after_s)},
            ) from e
        except EngineDeadError as e:
            raise ApiError(503, str(e), "server_error", code="engine_unavailable") from e

    async def stream_sse(
        stream: RequestStream, fmt: _Formatter, include_usage: bool, deadline: float
    ) -> AsyncIterator[str]:
        for c in fmt.preamble():
            yield _sse(c)
        last: RequestOutput | None = None
        failed = False
        while True:
            try:
                # The timeout wraps only the wait, never a yield: cancelling a
                # task while it is suspended in the *consumer* would not land here.
                async with asyncio.timeout_at(deadline):
                    out = await anext(stream)
            except StopAsyncIteration:
                break
            except TimeoutError:
                engine.abort(fmt.rid)
                failed = True
                yield _sse(error_body("request timed out", "timeout", code=504))
                break
            last = out
            if out.finished and out.finish_reason in (FinishReason.ABORT, FinishReason.ERROR):
                failed = True
                yield _sse(_client_error_for(out))
                break
            if out.text:
                yield _sse(fmt.delta(out.text))
            if out.finished:
                assert out.finish_reason is not None
                yield _sse(fmt.delta("", out.finish_reason.value))
                break
        if include_usage and last is not None and not failed:
            yield _sse(fmt.usage_chunk(last.prompt_tokens, last.completion_tokens))
        yield _sse("[DONE]")

    async def collect(stream: RequestStream) -> RequestOutput:
        acc: RequestOutput | None = None
        async for out in stream:
            acc = out if acc is None else acc.merge(out)
        assert acc is not None
        return acc

    async def respond(
        request: Request,
        kind: str,
        prompt_ids: list[int],
        params: SamplingParams,
        stream_mode: bool,
        include_usage: bool,
    ) -> Response:
        rid = f"{'chatcmpl' if kind == 'chat' else 'cmpl'}-{uuid.uuid4().hex}"
        stream = submit(rid, prompt_ids, params)
        fmt = _Formatter(kind, rid, settings.model_name)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.request_timeout_s
        headers = {"X-Request-Id": rid}

        if stream_mode:
            resp = EngineStreamResponse(
                stream_sse(stream, fmt, include_usage, deadline),
                on_close=lambda: engine.abort(rid),
            )
            resp.headers.update(headers)
            return resp

        collector = asyncio.ensure_future(collect(stream))
        watcher = asyncio.ensure_future(_wait_for_disconnect(request.receive))
        try:
            done, _ = await asyncio.wait(
                {collector, watcher},
                timeout=max(0.0, deadline - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (collector, watcher):
                if not t.done():
                    t.cancel()
            engine.abort(rid)
        if collector not in done:
            if watcher in done:
                return Response(status_code=499)  # client gone; nobody will read this
            raise ApiError(504, "request timed out", "timeout", code="timeout")
        out = collector.result()
        if out.finish_reason in (FinishReason.ABORT, FinishReason.ERROR):
            body = _client_error_for(out)
            return JSONResponse(body, body["error"]["code"], headers=headers)
        assert out.finish_reason is not None
        body = fmt.full(out.text, out.finish_reason.value, out.prompt_tokens, out.completion_tokens)
        return JSONResponse(body, headers=headers)

    def sampling_params(req: _Base, max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop_list,
            ignore_eos=req.ignore_eos,
            seed=req.seed,
        )

    # -------------------------------------------------------------- routes

    @app.post("/v1/completions", dependencies=auth)
    async def completions(req: CompletionRequest, request: Request) -> Response:
        check_common(req)
        if req.echo:
            raise ApiError(400, "echo is not supported", param="echo")
        if req.logprobs is not None:
            raise ApiError(400, "logprobs are not supported", param="logprobs")
        prompt_ids = tokenizer.encode(req.prompt_text)
        max_tokens = resolve_max_tokens(
            req.max_tokens, settings.default_completion_max_tokens, len(prompt_ids)
        )
        params = sampling_params(req, max_tokens)
        return await respond(request, "text", prompt_ids, params, req.stream, req.include_usage)

    @app.post("/v1/chat/completions", dependencies=auth)
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Response:
        check_common(req)
        if req.logprobs:
            raise ApiError(400, "logprobs are not supported", param="logprobs")
        prompt_ids = tokenizer.encode_chat([{"role": m.role, "content": m.text}
                                            for m in req.messages])  # fmt: skip
        max_tokens = resolve_max_tokens(
            req.requested_max_tokens, settings.max_tokens_limit, len(prompt_ids)
        )
        params = sampling_params(req, max_tokens)
        return await respond(request, "chat", prompt_ids, params, req.stream, req.include_usage)

    @app.get("/v1/models", dependencies=auth)
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "tokenserve",
                    "max_model_len": max_model_len,
                }
            ],
        }

    @app.get("/health")
    async def health() -> Response:
        ok = engine.dead is None and engine.is_running
        return JSONResponse({"status": "ok" if ok else "unavailable"}, 200 if ok else 503)

    return app
