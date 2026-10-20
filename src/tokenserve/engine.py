"""The engine loop: schedule → execute → update → stream, once per step.

The engine runs as one asyncio task on the same event loop as the HTTP
server. ``add_request`` and ``abort`` are synchronous and only touch queues;
all scheduler and KV-cache mutation happens inside the loop, between steps,
so a backend step never sees its blocks change underneath it. Aborts
(client disconnect, timeout) are therefore *deferred* to the next step
boundary.

The backend's ``execute`` is awaited; the simulated backend sleeps and a real
one runs in a worker thread, so HTTP handling and SSE writes for the previous
step overlap with the current step's "GPU" time.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tokenserve.backends.base import Backend, BatchItem
from tokenserve.kv_cache import KVCacheManager, cdiv
from tokenserve.metrics import EngineMetrics
from tokenserve.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
    make_scheduler,
)
from tokenserve.sequence import FinishReason, SamplingParams, Sequence
from tokenserve.tokenizer import IncrementalDetokenizer, StopStringMatcher, Tokenizer

log = logging.getLogger("tokenserve.engine")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    block_size: int = 16
    num_blocks: int = 2048
    #: Fraction of blocks admission must leave free for running sequences to grow into.
    watermark: float = 0.01
    enable_prefix_caching: bool = True
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    #: Backpressure: bound on queued sequences …
    max_waiting_seqs: int = 256
    #: … and on queued prompt tokens (default: one full KV cache worth).
    max_waiting_tokens: int | None = None

    @property
    def watermark_blocks(self) -> int:
        return math.floor(self.watermark * self.num_blocks)


class EngineError(Exception):
    """Base class for errors surfaced to API clients."""


class RequestRejectedError(EngineError):
    """The request can never be served with this configuration (→ 400)."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class QueueFullError(EngineError):
    """Backpressure: the waiting queue is at capacity (→ 429)."""

    def __init__(self, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class EngineDeadError(EngineError):
    """The engine loop crashed or is shut down (→ 503)."""


@dataclass(slots=True)
class RequestOutput:
    request_id: str
    text: str  # delta since the previous output
    token_ids: list[int]
    finished: bool
    finish_reason: FinishReason | None
    prompt_tokens: int
    completion_tokens: int

    def merge(self, newer: RequestOutput) -> RequestOutput:
        return RequestOutput(
            self.request_id,
            self.text + newer.text,
            self.token_ids + newer.token_ids,
            newer.finished,
            newer.finish_reason,
            newer.prompt_tokens,
            newer.completion_tokens,
        )


class RequestStream:
    """Async iterator of :class:`RequestOutput` for one request.

    If the consumer falls behind (slow client), pending deltas are coalesced
    into one output instead of piling up one message per token.
    """

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue[RequestOutput] = asyncio.Queue()
        self._done = False

    def put(self, out: RequestOutput) -> None:
        self._queue.put_nowait(out)

    def __aiter__(self) -> RequestStream:
        return self

    async def __anext__(self) -> RequestOutput:
        if self._done:
            raise StopAsyncIteration
        out = await self._queue.get()
        while not out.finished and not self._queue.empty():
            out = out.merge(self._queue.get_nowait())
        self._done = out.finished
        return out


@dataclass(eq=False, slots=True)
class _Request:
    seq: Sequence
    stream: RequestStream
    detok: IncrementalDetokenizer
    stop: StopStringMatcher


class Engine:
    def __init__(
        self,
        config: EngineConfig,
        backend: Backend,
        tokenizer: Tokenizer,
        *,
        metrics: EngineMetrics | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.backend = backend
        self.tokenizer = tokenizer
        self.metrics = metrics or EngineMetrics()
        self.clock = clock
        prefix_caching = config.enable_prefix_caching and backend.supports_prefix_caching
        if config.enable_prefix_caching and not prefix_caching:
            log.warning("backend %s cannot share KV; prefix caching disabled", backend.name)
        self.kv = KVCacheManager(
            config.num_blocks,
            config.block_size,
            enable_prefix_caching=prefix_caching,
            watermark_blocks=config.watermark_blocks,
        )
        self.scheduler: Scheduler = make_scheduler(
            config.scheduler, self.kv, eos_token_id=tokenizer.eos_token_id, clock=clock
        )
        self.max_waiting_tokens = (
            config.max_waiting_tokens
            if config.max_waiting_tokens is not None
            else config.num_blocks * config.block_size
        )
        self._requests: dict[str, _Request] = {}
        self._pending_aborts: set[str] = set()
        self._seq_ids = itertools.count()
        self._wakeup = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self.dead: BaseException | None = None
        self.num_steps = 0
        # EWMA of queue time, used to suggest Retry-After under backpressure.
        self._queue_time_ewma = 0.0
        self.metrics.kv_blocks_total.set(config.num_blocks)
        self._update_gauges()

    # ------------------------------------------------------------- lifecycle

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="tokenserve-engine")

    async def stop(self) -> None:
        self._stopping = True
        self._wakeup.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for req in list(self._requests.values()):
            self._finish(req, FinishReason.ABORT)
        await self.backend.close()

    async def wait_idle(self) -> None:
        """Wait until every request has finished and its blocks are released."""
        await self._idle.wait()

    # -------------------------------------------------------------- requests

    def add_request(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams
    ) -> RequestStream:
        """Queue a request. Raises :class:`EngineError` subclasses on rejection."""
        if self.dead is not None or self._stopping:
            raise EngineDeadError("engine is not running")
        if request_id in self._requests:
            raise ValueError(f"duplicate request id {request_id}")
        sc = self.config.scheduler
        total = len(prompt_token_ids) + params.max_tokens
        if not prompt_token_ids:
            raise RequestRejectedError("prompt is empty", "empty_prompt")
        if total > sc.max_model_len:
            self.metrics.requests_rejected.labels("too_long").inc()
            raise RequestRejectedError(
                f"prompt ({len(prompt_token_ids)} tokens) + max_tokens ({params.max_tokens}) "
                f"exceeds the maximum context length of {sc.max_model_len} tokens",
                "context_length_exceeded",
            )
        if not self.kv.can_ever_fit(total):
            self.metrics.requests_rejected.labels("kv_capacity").inc()
            raise RequestRejectedError(
                f"request needs {cdiv(total, self.kv.block_size)} KV blocks; the cache has "
                f"{self.kv.num_blocks - self.kv.watermark_blocks} usable",
                "kv_capacity_exceeded",
            )
        if (
            self.scheduler.num_waiting >= self.config.max_waiting_seqs
            or self.scheduler.num_waiting_tokens + len(prompt_token_ids) > self.max_waiting_tokens
        ):
            self.metrics.requests_rejected.labels("queue_full").inc()
            raise QueueFullError("server is at capacity; retry later", self._retry_after())

        seq = Sequence(
            seq_id=next(self._seq_ids),
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            params=params,
            arrival_time=self.clock(),
        )
        stream = RequestStream(request_id)
        self._requests[request_id] = _Request(
            seq,
            stream,
            IncrementalDetokenizer(self.tokenizer, prompt_token_ids),
            StopStringMatcher(params.stop),
        )
        self.scheduler.add(seq)
        self.metrics.prompt_tokens.inc(len(prompt_token_ids))
        self._idle.clear()
        self._wakeup.set()
        return stream

    def abort(self, request_id: str) -> None:
        """Cancel a request; its KV blocks are freed at the next step boundary.

        Safe to call for unknown or already-finished requests.
        """
        if request_id in self._requests:
            self._pending_aborts.add(request_id)
            self._wakeup.set()

    def _retry_after(self) -> int:
        return max(1, min(60, math.ceil(self._queue_time_ewma)))

    # ------------------------------------------------------------------ loop

    async def _run(self) -> None:
        try:
            while not self._stopping:
                self._apply_aborts()
                if not self.scheduler.has_unfinished():
                    self._wakeup.clear()
                    self._update_gauges()
                    self._idle.set()
                    await self._wakeup.wait()
                    continue
                await self.step()
                # Let request handlers run even if the backend returned instantly.
                await asyncio.sleep(0)
        except Exception as exc:
            log.exception("engine loop crashed")
            self.dead = exc
            for req in list(self._requests.values()):
                self._finish(req, FinishReason.ERROR)

    async def step(self) -> None:
        t0 = self.clock()
        out = self.scheduler.schedule()
        for seq in out.preempted:
            self.backend.free(seq.seq_id)
        if out.is_empty:
            # Nothing runnable this step (should not happen with admission
            # control in place); avoid spinning.
            await asyncio.sleep(0.001)
            return
        self._observe_admissions(out)
        sampled = await self.backend.execute(self._build_batch(out))
        result = self.scheduler.update(out, sampled)
        now = self.clock()
        self.num_steps += 1

        m = self.metrics
        m.step_duration.observe(now - t0)
        m.batch_seqs.observe(len(out.scheduled))
        m.batch_tokens.observe(out.num_batched_tokens)
        m.computed_prefill_tokens.inc(out.num_prefill_tokens)
        m.generation_tokens.inc(len(result.events))

        for ev in result.events:
            req = self._requests.get(ev.seq.request_id)
            if req is None:
                continue
            seq = req.seq
            if seq.last_itl is not None and seq.num_output_tokens > 1:
                m.itl.observe(seq.last_itl)
            text, stopped = req.stop.push(req.detok.add(ev.token_id))
            if stopped:
                self.scheduler.finish(seq, FinishReason.STOP)  # no-op if already finished
            if seq.is_finished:
                text += req.stop.flush()
                assert seq.finish_reason is not None
                self._finish(req, seq.finish_reason, text=text, token=ev.token_id)
            else:
                req.stream.put(
                    RequestOutput(
                        seq.request_id,
                        text,
                        [ev.token_id],
                        False,
                        None,
                        seq.num_prompt_tokens,
                        seq.num_output_tokens,
                    )
                )
        self._update_gauges()

    def _build_batch(self, out: SchedulerOutput) -> list[BatchItem]:
        return [
            BatchItem(
                seq_id=s.seq.seq_id,
                token_ids=s.seq.token_ids,
                start=s.start,
                num_new_tokens=s.num_new_tokens,
                num_prompt_tokens=s.seq.num_prompt_tokens,
                block_table=self.kv.block_table(s.seq.seq_id),
                sample=s.samples,
                seed=s.seq.sampling_seed,
                temperature=s.seq.params.temperature,
                top_p=s.seq.params.top_p,
            )
            for s in out.scheduled
        ]

    def _apply_aborts(self) -> None:
        if not self._pending_aborts:
            return
        for rid in self._pending_aborts:
            req = self._requests.get(rid)
            if req is not None:
                self._finish(req, FinishReason.ABORT)
        self._pending_aborts.clear()
        self._update_gauges()

    def _finish(
        self, req: _Request, reason: FinishReason, *, text: str = "", token: int | None = None
    ) -> None:
        """Release everything a request holds and send its final output."""
        seq = req.seq
        self.scheduler.finish(seq, reason)  # no-op if the scheduler already did
        self.backend.free(seq.seq_id)
        self._requests.pop(seq.request_id, None)
        final_reason = seq.finish_reason or reason
        req.stream.put(
            RequestOutput(
                seq.request_id,
                text,
                [token] if token is not None else [],
                True,
                final_reason,
                seq.num_prompt_tokens,
                seq.num_output_tokens,
            )
        )
        m = self.metrics
        m.requests_finished.labels(final_reason.value).inc()
        if final_reason not in (FinishReason.LENGTH, FinishReason.STOP):
            return  # aborted/failed requests would skew the latency histograms
        end = seq.finished_time or self.clock()
        m.e2e.observe(end - seq.arrival_time)
        if seq.first_token_time is not None:
            m.ttft.observe(seq.first_token_time - seq.arrival_time)
            if seq.num_output_tokens > 1 and seq.last_token_time is not None:
                m.tpot.observe(
                    (seq.last_token_time - seq.first_token_time) / (seq.num_output_tokens - 1)
                )

    # --------------------------------------------------------------- metrics

    def _observe_admissions(self, out: SchedulerOutput) -> None:
        for seq in out.admitted:
            assert seq.first_scheduled_time is not None
            qt = seq.first_scheduled_time - seq.arrival_time
            self.metrics.queue_time.observe(qt)
            self._queue_time_ewma = 0.8 * self._queue_time_ewma + 0.2 * qt

    def _update_gauges(self) -> None:
        m = self.metrics
        s = self.scheduler
        m.running.set(s.num_running)
        m.waiting.set(s.num_waiting)
        m.kv_usage.set(self.kv.usage)
        m.kv_blocks_free.set(self.kv.num_free_blocks)
        m.sync_counters(
            preemptions=s.num_preemptions,
            prefix_queries=s.prefix_query_tokens,
            prefix_hits=s.prefix_hit_tokens,
        )

    # ----------------------------------------------------------------- debug

    def snapshot(self) -> dict[str, object]:
        """JSON-able view of scheduler and KV state for the admin endpoint."""
        s = self.scheduler

        def seq_info(seq: Sequence) -> dict[str, object]:
            return {
                "request_id": seq.request_id,
                "seq_id": seq.seq_id,
                "prompt_tokens": seq.num_prompt_tokens,
                "output_tokens": seq.num_output_tokens,
                "computed_tokens": seq.num_computed_tokens,
                "blocks": self.kv.num_blocks_of(seq.seq_id),
                "preemptions": seq.num_preemptions,
            }

        return {
            "policy": self.config.scheduler.policy.value,
            "steps": self.num_steps,
            "running": [seq_info(q) for q in s.running],
            "waiting": [seq_info(q) for q in s.waiting],
            "kv": {
                "block_size": self.kv.block_size,
                "blocks_total": self.kv.num_blocks,
                "blocks_free": self.kv.num_free_blocks,
                "blocks_cached": self.kv.num_cached_blocks,
                "usage": round(self.kv.usage, 4),
                "evictions": self.kv.num_evictions,
                "prefix_caching": self.kv.enable_prefix_caching,
            },
            "preemptions": s.num_preemptions,
            "prefix_cache_hit_tokens": s.prefix_hit_tokens,
            "prefix_cache_query_tokens": s.prefix_query_tokens,
        }
