from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.types import Message

from tests.conftest import make_engine, wait_for
from tests.test_engine import expected_text
from tokenserve.api import ApiSettings, create_app
from tokenserve.engine import Engine
from tokenserve.sequence import SamplingParams
from tokenserve.server import _Server


def parse_sse(body: str) -> list[Any]:
    events: list[Any] = []
    assert body.endswith("\n\n")
    for block in body.split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: "), block
        data = block.removeprefix("data: ")
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


def client_for(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def api() -> AsyncIterator[tuple[Engine, httpx.AsyncClient]]:
    eng = make_engine(max_model_len=256, num_blocks=256)
    await eng.start()
    async with client_for(create_app(eng, ApiSettings(max_tokens_limit=200))) as c:
        yield eng, c
    await eng.stop()


# ------------------------------------------------------------------ happy path


async def test_completion(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    r = await c.post("/v1/completions", json={"prompt": "hello", "max_tokens": 12})
    assert r.status_code == 200
    assert r.headers["x-request-id"].startswith("cmpl-")
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "tokenserve-sim"
    choice = body["choices"][0]
    assert choice["text"] == expected_text("hello", 12)
    assert choice["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}


async def test_completion_default_max_tokens_is_16(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    r = await c.post("/v1/completions", json={"prompt": ["hello"]})
    assert r.json()["usage"]["completion_tokens"] == 16


async def test_completion_streaming_format(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    payload = {
        "prompt": "hello",
        "max_tokens": 12,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    r = await c.post("/v1/completions", json=payload)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = parse_sse(r.text)
    assert events[-1] == "[DONE]"
    chunks, usage = events[:-2], events[-2]
    assert all(e["object"] == "text_completion" for e in chunks)
    assert len({e["id"] for e in events[:-1]}) == 1
    assert "".join(e["choices"][0]["text"] for e in chunks) == expected_text("hello", 12)
    assert [e["choices"][0]["finish_reason"] for e in chunks] == [None] * 12 + ["length"]
    assert usage["choices"] == []
    assert usage["usage"]["completion_tokens"] == 12


async def test_chat_streaming_format(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    payload = {
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
        "max_completion_tokens": 5,
        "stream": True,
    }
    r = await c.post("/v1/chat/completions", json=payload)
    events = parse_sse(r.text)
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(e["object"] == "chat.completion.chunk" for e in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in chunks[1:])
    assert len(content) == 5
    assert chunks[-1]["choices"][0] == {
        "index": 0,
        "delta": {},
        "logprobs": None,
        "finish_reason": "length",
    }


async def test_chat_completion(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    # No max_tokens: generate up to the configured limit.
    assert body["usage"]["completion_tokens"] == 200


async def test_stop_sequence(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    ref = expected_text("stop here", 40)
    stop = ref[10:13]
    cut = ref.index(stop)
    r = await c.post(
        "/v1/completions", json={"prompt": "stop here", "max_tokens": 40, "stop": stop}
    )
    choice = r.json()["choices"][0]
    assert choice == {"index": 0, "text": ref[:cut], "logprobs": None, "finish_reason": "stop"}


async def test_models_and_health(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    models = (await c.get("/v1/models")).json()
    assert models["data"][0]["id"] == "tokenserve-sim"
    assert (await c.get("/health")).json() == {"status": "ok"}


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"max_tokens": 5}, "prompt"),
        ({"prompt": ""}, "prompt"),
        ({"prompt": ["a", "b"]}, "prompt"),
        ({"prompt": "x", "max_tokens": 0}, "max_tokens"),
        ({"prompt": "x", "temperature": 3}, "temperature"),
        ({"prompt": "x", "top_p": 0}, "top_p"),
        ({"prompt": "x", "n": 2}, "n"),
        ({"prompt": "x", "stop": ["a", "b", "c", "d", "e"]}, "stop"),
        ({"prompt": "x", "max_tokens": 201}, "max_tokens"),
        ({"prompt": "x", "echo": True}, "echo"),
        ({"prompt": "x", "logprobs": 2}, "logprobs"),
    ],
)
async def test_invalid_requests_get_openai_style_400(
    api: tuple[Engine, httpx.AsyncClient], payload: dict[str, Any], param: str
) -> None:
    _, c = api
    r = await c.post("/v1/completions", json=payload)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == param
    assert err["message"]


async def test_malformed_json_and_empty_messages(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    r = await c.post(
        "/v1/completions", content=b"{nope", headers={"content-type": "application/json"}
    )
    assert r.status_code == 400
    r = await c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


async def test_prompt_longer_than_context_is_rejected(
    api: tuple[Engine, httpx.AsyncClient],
) -> None:
    _, c = api
    r = await c.post("/v1/completions", json={"prompt": "x" * 250, "max_tokens": 10})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"


async def test_unknown_model_and_route(api: tuple[Engine, httpx.AsyncClient]) -> None:
    _, c = api
    r = await c.post("/v1/completions", json={"prompt": "x", "model": "gpt-9"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
    r = await c.get("/v1/nope")
    assert r.status_code == 404
    assert "error" in r.json()


async def test_body_size_limit() -> None:
    eng = make_engine()
    app = create_app(eng, ApiSettings(max_body_bytes=64))
    async with client_for(app) as c:
        r = await c.post("/v1/completions", json={"prompt": "x" * 100})
        assert r.status_code == 413

        async def chunked() -> AsyncIterator[bytes]:  # no Content-Length header
            yield b'{"prompt": "'
            yield b"x" * 100
            yield b'"}'

        r = await c.post(
            "/v1/completions", content=chunked(), headers={"content-type": "application/json"}
        )
        assert r.status_code == 413


async def test_api_key_auth() -> None:
    eng = make_engine()
    await eng.start()
    app = create_app(eng, ApiSettings(api_key="test-only-key"))
    async with client_for(app) as c:
        payload = {"prompt": "x", "max_tokens": 1}
        assert (await c.post("/v1/completions", json=payload)).status_code == 401
        r = await c.post("/v1/completions", json=payload, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        r = await c.post(
            "/v1/completions", json=payload, headers={"Authorization": "Bearer test-only-key"}
        )
        assert r.status_code == 200
        assert (await c.get("/health")).status_code == 200  # health stays open
    await eng.stop()


# ---------------------------------------------------------------- backpressure


async def test_queue_full_returns_429_with_retry_after() -> None:
    eng = make_engine(max_waiting_seqs=1)  # not started: nothing drains the queue
    eng.add_request("occupant", [1, 2, 3], SamplingParams())
    async with client_for(create_app(eng)) as c:
        r = await c.post("/v1/completions", json={"prompt": "x", "max_tokens": 1})
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["code"] == "queue_full"


async def test_timeout_returns_504_and_frees_blocks() -> None:
    eng = make_engine(base_latency_s=0.02, num_blocks=128)
    await eng.start()
    app = create_app(eng, ApiSettings(request_timeout_s=0.1))
    async with client_for(app) as c:
        r = await c.post("/v1/completions", json={"prompt": "slow", "max_tokens": 100})
        assert r.status_code == 504
        assert r.json()["error"]["type"] == "timeout"
        payload = {
            "prompt": "slow",
            "max_tokens": 100,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        events = parse_sse((await c.post("/v1/completions", json=payload)).text)
        assert events[-2]["error"]["type"] == "timeout"  # no usage chunk after an error
        assert events[-1] == "[DONE]"
    await asyncio.wait_for(eng.wait_idle(), 2)
    assert eng.kv.num_free_blocks == eng.kv.num_blocks
    await eng.stop()


# ------------------------------------------------------------------ disconnect


def _scope(path: str, body: bytes) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }


@pytest.mark.parametrize("stream", [True, False])
async def test_client_disconnect_aborts_and_frees_blocks(stream: bool) -> None:
    """Drive the ASGI app directly so the disconnect lands at a precise moment."""
    eng = make_engine(base_latency_s=0.002, max_model_len=4096, num_blocks=1024)
    await eng.start()
    app = create_app(eng)
    body = json.dumps({"prompt": "bye", "max_tokens": 2000, "stream": stream}).encode()
    gone = asyncio.Event()
    body_sent = False
    chunks = 0

    async def receive() -> Message:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(msg: Message) -> None:
        nonlocal chunks
        if msg["type"] == "http.response.body" and msg.get("body"):
            chunks += 1

    task = asyncio.create_task(app(_scope("/v1/completions", body), receive, send))
    await wait_for(lambda: eng.metrics.generation_tokens._value.get() >= 5 or task.done())
    assert not task.done(), "request finished before the client disconnected"
    assert eng.kv.num_used_blocks > 0
    gone.set()
    await asyncio.wait_for(task, 2)
    await asyncio.wait_for(eng.wait_idle(), 2)
    assert eng.kv.num_free_blocks == eng.kv.num_blocks
    assert eng.metrics.requests_finished.labels("abort")._value.get() == 1
    assert eng.metrics.generation_tokens._value.get() < 2000
    if stream:
        assert chunks >= 1
    await eng.stop()


async def test_disconnect_over_a_real_socket_frees_blocks() -> None:
    """Same property end-to-end: real uvicorn, real TCP connection closed mid-stream."""
    eng = make_engine(base_latency_s=0.002, max_model_len=4096, num_blocks=1024)
    server = _Server(
        uvicorn.Config(create_app(eng), host="127.0.0.1", port=0, lifespan="on", log_level="error")
    )
    serve_task = asyncio.create_task(server.serve())
    await wait_for(lambda: server.started)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
            payload = {"prompt": "hi", "max_tokens": 2000, "stream": True}
            async with c.stream("POST", "/v1/completions", json=payload) as r:
                assert r.status_code == 200
                n = 0
                async for line in r.aiter_lines():
                    n += line.startswith("data: ")
                    if n == 5:
                        break  # leaving the block closes the connection
        await wait_for(lambda: eng.metrics.requests_finished.labels("abort")._value.get() == 1)
        await asyncio.wait_for(eng.wait_idle(), 2)
        assert eng.kv.num_free_blocks == eng.kv.num_blocks
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, 5)
    assert not eng.is_running  # lifespan stopped the engine it started
