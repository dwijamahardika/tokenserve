"""Sequence state shared by the scheduler, the KV-cache manager and backends.

A *sequence* is one prompt plus the tokens generated for it so far. The two
counters that drive everything are:

* ``num_tokens``          — prompt + output tokens known so far.
* ``num_computed_tokens`` — how many of those already have KV entries in the cache.

``num_tokens - num_computed_tokens`` is the work left before the next token can
be sampled: the whole prompt for a fresh request, one token while decoding, or
prompt+output again after a preemption-by-recompute.
"""

from __future__ import annotations

import enum
import time
import zlib
from array import array
from dataclasses import dataclass, field


class FinishReason(enum.StrEnum):
    LENGTH = "length"  # hit max_tokens
    STOP = "stop"  # EOS token or a stop string
    ABORT = "abort"  # client disconnect, timeout, shutdown
    ERROR = "error"  # backend failure


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")


@dataclass(eq=False, slots=True)
class Sequence:
    seq_id: int  # monotonic; doubles as the FCFS priority (lower = older)
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)

    output_token_ids: list[int] = field(default_factory=list)
    # token_ids = prompt + output, kept as one list so backends can slice it
    token_ids: list[int] = field(init=False)
    sampling_seed: int = field(init=False)
    num_computed_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: FinishReason | None = None
    num_preemptions: int = 0
    # Timestamps (monotonic seconds) for latency metrics.
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    last_itl: float | None = None  # gap before the most recent token
    finished_time: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.token_ids = list(self.prompt_token_ids)
        # Per-sequence seed so outputs are a pure function of the request. The
        # default (CRC of the prompt) makes identical prompts produce identical
        # outputs, like greedy decoding on a real model, regardless of batch
        # composition or preemption.
        if self.params.seed is not None:
            self.sampling_seed = self.params.seed
        else:
            self.sampling_seed = zlib.crc32(array("I", self.prompt_token_ids).tobytes())

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    @property
    def max_total_tokens(self) -> int:
        """Upper bound on the sequence length; used for admission control."""
        return self.num_prompt_tokens + self.params.max_tokens

    def append_token(self, token_id: int, now: float) -> None:
        self.output_token_ids.append(token_id)
        self.token_ids.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = now
        else:
            assert self.last_token_time is not None
            self.last_itl = now - self.last_token_time
        self.last_token_time = now
