"""Iteration-level (continuous) and request-level (static) batch schedulers.

Every engine step the scheduler decides, for each sequence, how many tokens
to push through the model in that step:

* a running sequence that is decoding gets 1 token;
* a sequence in (chunked) prefill gets up to its remaining prompt;
* a sequence resumed after preemption-by-recompute gets prompt+output again
  (minus whatever the prefix cache still holds).

The sum is capped by ``max_num_batched_tokens`` and the number of sequences by
``max_num_seqs``. KV-cache blocks are the binding constraint: before a sequence
is scheduled its block table must cover the new tokens, and if the cache is
full the youngest running sequence is preempted to make room.

Ordering — the no-starvation argument
-------------------------------------
``waiting`` and ``running`` are both kept sorted by ``seq_id`` (arrival order).
Running sequences are served oldest first, victims are taken youngest first,
and admission from ``waiting`` is strict FCFS with head-of-line blocking: if
the oldest waiting request does not fit, nobody behind it jumps ahead. So the
oldest request in the system is never preempted (unless it is alone and
cannot fit at all, which admission control rejects up front) and always makes
progress; by induction every request finishes.

Within a step, decodes are scheduled before prefill chunks (Sarathi-style
"stall-free" batching) so a long prompt arriving cannot freeze the token
stream of requests that are already generating.
"""

from __future__ import annotations

import bisect
import enum
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from tokenserve.kv_cache import KVCacheManager
from tokenserve.sequence import FinishReason, Sequence, SequenceStatus


class SchedulingPolicy(enum.StrEnum):
    CONTINUOUS = "continuous"
    STATIC = "static"


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    policy: SchedulingPolicy = SchedulingPolicy.CONTINUOUS

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1 or self.max_model_len < 2:
            raise ValueError("scheduler limits must be positive")
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            # After a preemption the whole prompt+output is recomputed in one
            # step, so without chunking a full-length sequence must fit a step.
            raise ValueError(
                "without chunked prefill, max_num_batched_tokens must be >= max_model_len"
            )


@dataclass(slots=True)
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int
    start: int  # num_computed_tokens before this step = first position computed

    @property
    def samples(self) -> bool:
        """Whether this step completes the sequence's known tokens (→ sample one)."""
        return self.start + self.num_new_tokens == self.seq.num_tokens

    @property
    def is_decode(self) -> bool:
        return self.num_new_tokens == 1 and self.seq.num_output_tokens > 0


@dataclass(slots=True)
class SchedulerOutput:
    token_budget: int
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    admitted: list[Sequence] = field(default_factory=list)  # scheduled for the first time

    def add(self, seq: Sequence, num_new_tokens: int) -> None:
        self.scheduled.append(ScheduledSeq(seq, num_new_tokens, seq.num_computed_tokens))
        self.token_budget -= num_new_tokens

    def unschedule(self, seq: Sequence) -> None:
        for i, item in enumerate(self.scheduled):
            if item.seq is seq:
                self.token_budget += item.num_new_tokens
                del self.scheduled[i]
                return

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled)

    @property
    def num_decode_tokens(self) -> int:
        return sum(1 for s in self.scheduled if s.is_decode)

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_batched_tokens - self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


@dataclass(slots=True)
class TokenEvent:
    seq: Sequence
    token_id: int


@dataclass(slots=True)
class UpdateResult:
    events: list[TokenEvent] = field(default_factory=list)
    finished: list[Sequence] = field(default_factory=list)


def _by_id(seq: Sequence) -> int:
    return seq.seq_id


class Scheduler:
    """Continuous batching: sequences join and leave the batch every step."""

    def __init__(
        self,
        config: SchedulerConfig,
        kv: KVCacheManager,
        *,
        eos_token_id: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.kv = kv
        self.eos_token_id = eos_token_id
        self.clock = clock
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self._waiting_tokens = 0
        # Counters (read by the engine for metrics).
        self.num_preemptions = 0
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0

    # ---------------------------------------------------------------- queries

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_waiting_tokens(self) -> int:
        return self._waiting_tokens

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # -------------------------------------------------------------- lifecycle

    def add(self, seq: Sequence) -> None:
        if seq.status is not SequenceStatus.WAITING or seq.num_computed_tokens:
            raise ValueError("only fresh sequences can be added")
        self._push_waiting(seq)

    def finish(self, seq: Sequence, reason: FinishReason) -> None:
        """Remove ``seq`` from the scheduler and free its KV blocks. Idempotent."""
        if seq.is_finished:
            return
        if seq.status is SequenceStatus.RUNNING:
            self.running.remove(seq)
        else:
            self.waiting.remove(seq)
            self._waiting_tokens -= seq.num_tokens
        self.kv.free(seq.seq_id)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finished_time = self.clock()

    def _push_waiting(self, seq: Sequence) -> None:
        bisect.insort(self.waiting, seq, key=_by_id)
        self._waiting_tokens += seq.num_tokens

    def _pop_waiting(self) -> Sequence:
        seq = self.waiting.pop(0)
        self._waiting_tokens -= seq.num_tokens
        return seq

    def _start(self, seq: Sequence, num_cached_tokens: int, out: SchedulerOutput) -> None:
        seq.status = SequenceStatus.RUNNING
        seq.num_computed_tokens = num_cached_tokens
        if seq.first_scheduled_time is None:
            seq.first_scheduled_time = self.clock()
            out.admitted.append(seq)
        if self.kv.enable_prefix_caching:
            self.prefix_query_tokens += seq.num_tokens
            self.prefix_hit_tokens += num_cached_tokens
        bisect.insort(self.running, seq, key=_by_id)

    def _preempt(self, victim: Sequence, out: SchedulerOutput) -> None:
        """Preemption by recompute: drop the KV, re-queue, recompute later."""
        self.running.remove(victim)
        out.unschedule(victim)
        out.preempted.append(victim)
        self.kv.free(victim.seq_id)
        victim.num_computed_tokens = 0
        victim.status = SequenceStatus.WAITING
        victim.num_preemptions += 1
        self.num_preemptions += 1
        self._push_waiting(victim)

    # -------------------------------------------------------------- scheduling

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput(token_budget=self.config.max_num_batched_tokens)
        self._schedule_running(out)
        # Admitting new work in a step that had to preempt would just thrash.
        if not out.preempted:
            self._schedule_waiting(out)
        return out

    def _schedule_running(self, out: SchedulerOutput) -> None:
        # Pass 1: decodes, so ongoing streams never stall behind a prefill.
        # Pass 2: prefill chunks (new prompts or recompute) with what is left.
        for decode_pass in (True, False):
            for seq in list(self.running):
                if out.token_budget <= 0:
                    return
                if seq.status is not SequenceStatus.RUNNING:
                    continue  # preempted earlier in this loop
                remaining = seq.num_tokens - seq.num_computed_tokens
                if (remaining == 1 and seq.num_output_tokens > 0) != decode_pass:
                    continue
                n = min(remaining, out.token_budget)
                if n < remaining and not self.config.enable_chunked_prefill:
                    continue  # wait for a step with room for the whole prompt
                if self._allocate_or_preempt(seq, n, out):
                    out.add(seq, n)

    def _allocate_or_preempt(self, seq: Sequence, n: int, out: SchedulerOutput) -> bool:
        while not self.kv.allocate_slots(seq, n):
            victim = self.running[-1]  # youngest
            self._preempt(victim, out)
            if victim is seq:
                return False
        return True

    def _schedule_waiting(self, out: SchedulerOutput) -> None:
        while (
            self.waiting and out.token_budget > 0 and len(self.running) < self.config.max_num_seqs
        ):
            seq = self.waiting[0]
            hits, num_cached = self.kv.get_computed_blocks(seq)
            remaining = seq.num_tokens - num_cached
            n = min(remaining, out.token_budget)
            if n < remaining and not self.config.enable_chunked_prefill:
                break  # head-of-line blocking keeps admission strictly FCFS
            if not self.kv.allocate_slots(seq, n, hits, watermark=True):
                break
            self._pop_waiting()
            self._start(seq, num_cached, out)
            out.add(seq, n)

    # ------------------------------------------------------------------ update

    def update(self, out: SchedulerOutput, sampled: Mapping[int, int]) -> UpdateResult:
        """Apply a finished step: advance computed tokens, append samples, finish."""
        now = self.clock()
        result = UpdateResult()
        for item in out.scheduled:
            seq = item.seq
            if seq.status is not SequenceStatus.RUNNING:
                continue  # aborted while the step was in flight
            sample = item.samples
            seq.num_computed_tokens += item.num_new_tokens
            self.kv.cache_full_blocks(seq)
            if not sample:
                continue  # an intermediate chunk of a chunked prefill
            token = sampled[seq.seq_id]
            seq.append_token(token, now)
            result.events.append(TokenEvent(seq, token))
            reason: FinishReason | None = None
            if (
                token == self.eos_token_id
                and self.eos_token_id is not None
                and not seq.params.ignore_eos
            ):
                reason = FinishReason.STOP
            elif (
                seq.num_output_tokens >= seq.params.max_tokens
                or seq.num_tokens >= self.config.max_model_len
            ):
                reason = FinishReason.LENGTH
            if reason is not None:
                self.finish(seq, reason)
                result.finished.append(seq)
        return result


class StaticScheduler(Scheduler):
    """Request-level batching baseline.

    A batch is formed only when the previous one has fully drained; members
    then run until the *last* of them finishes, and newcomers wait. Prompt
    blocks are allocated for the whole batch at formation time.

    This baseline is deliberately generous: finished members stop consuming
    compute (a real static server keeps padding them until the batch ends),
    and it shares the paged KV cache, prefix cache and chunked prefill. The
    only thing it lacks is iteration-level admission.
    """

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput(token_budget=self.config.max_num_batched_tokens)
        if not self.running:
            self._form_batch(out)
        self._schedule_running(out)
        return out

    def _form_batch(self, out: SchedulerOutput) -> None:
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            hits, num_cached = self.kv.get_computed_blocks(seq)
            if not self.kv.allocate_slots(seq, seq.num_tokens - num_cached, hits, watermark=True):
                break
            self._pop_waiting()
            self._start(seq, num_cached, out)


def make_scheduler(
    config: SchedulerConfig,
    kv: KVCacheManager,
    *,
    eos_token_id: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Scheduler:
    cls = StaticScheduler if config.policy is SchedulingPolicy.STATIC else Scheduler
    return cls(config, kv, eos_token_id=eos_token_id, clock=clock)
