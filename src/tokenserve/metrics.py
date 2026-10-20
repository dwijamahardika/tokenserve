"""Prometheus metrics. One registry per engine so tests stay isolated."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    20.0,
    30.0,
    60.0,
)
_TOKEN_LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.0125,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    1.0,
    2.5,
)
_SEQ_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256)
_TOKEN_BUCKETS = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


class EngineMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = r = registry if registry is not None else CollectorRegistry()

        # Per-request latency.
        self.ttft = Histogram(
            "tokenserve_time_to_first_token_seconds",
            "Arrival at the engine to first generated token.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.tpot = Histogram(
            "tokenserve_time_per_output_token_seconds",
            "Per request: (last token time - first token time) / (output tokens - 1).",
            buckets=_TOKEN_LATENCY_BUCKETS,
            registry=r,
        )
        self.itl = Histogram(
            "tokenserve_inter_token_latency_seconds",
            "Gap between consecutive tokens of one request.",
            buckets=_TOKEN_LATENCY_BUCKETS,
            registry=r,
        )
        self.queue_time = Histogram(
            "tokenserve_queue_time_seconds",
            "Arrival to first scheduled step.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.e2e = Histogram(
            "tokenserve_e2e_request_latency_seconds",
            "Arrival to finish.",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )

        # Per-step.
        self.step_duration = Histogram(
            "tokenserve_step_duration_seconds",
            "Wall time of one engine step (schedule + execute + update).",
            buckets=_TOKEN_LATENCY_BUCKETS,
            registry=r,
        )
        self.batch_seqs = Histogram(
            "tokenserve_batch_size_seqs",
            "Sequences in a step.",
            buckets=_SEQ_BUCKETS,
            registry=r,
        )
        self.batch_tokens = Histogram(
            "tokenserve_batch_size_tokens",
            "Tokens in a step (prefill + decode).",
            buckets=_TOKEN_BUCKETS,
            registry=r,
        )

        # Scheduler / KV state.
        self.running = Gauge(
            "tokenserve_num_requests_running", "Sequences in the running batch.", registry=r
        )
        self.waiting = Gauge(
            "tokenserve_num_requests_waiting", "Sequences queued (new or preempted).", registry=r
        )
        self.kv_usage = Gauge(
            "tokenserve_kv_cache_usage_ratio",
            "Fraction of KV blocks referenced by a sequence.",
            registry=r,
        )
        self.kv_blocks_total = Gauge("tokenserve_kv_cache_blocks_total", "KV blocks.", registry=r)
        self.kv_blocks_free = Gauge(
            "tokenserve_kv_cache_blocks_free",
            "Free KV blocks (including evictable cached ones).",
            registry=r,
        )
        self.prefix_queries = Counter(
            "tokenserve_prefix_cache_queries_tokens",
            "Tokens looked up in the prefix cache at admission.",
            registry=r,
        )
        self.prefix_hits = Counter(
            "tokenserve_prefix_cache_hits_tokens",
            "Tokens served from the prefix cache at admission.",
            registry=r,
        )
        self.prefix_hit_rate = Gauge(
            "tokenserve_prefix_cache_hit_rate", "Lifetime hits / queries (tokens).", registry=r
        )
        self.preemptions = Counter(
            "tokenserve_preemptions",
            "Sequences preempted (recompute) for lack of KV blocks.",
            registry=r,
        )

        # Throughput / outcomes.
        self.prompt_tokens = Counter(
            "tokenserve_prompt_tokens", "Prompt tokens admitted.", registry=r
        )
        self.generation_tokens = Counter(
            "tokenserve_generation_tokens", "Tokens generated.", registry=r
        )
        self.computed_prefill_tokens = Counter(
            "tokenserve_prefill_tokens_computed",
            "Prefill tokens actually run through the model (excludes prefix-cache hits, "
            "includes recompute after preemption).",
            registry=r,
        )
        self.requests_finished = Counter(
            "tokenserve_requests_finished", "Finished requests.", ["finish_reason"], registry=r
        )
        self.requests_rejected = Counter(
            "tokenserve_requests_rejected", "Requests refused at admission.", ["reason"], registry=r
        )

        # Export every known label value from the start, so rate() and alerts
        # see a zero rather than a missing series.
        for reason in ("length", "stop", "abort", "error"):
            self.requests_finished.labels(reason)
        for reason in ("queue_full", "too_long", "kv_capacity"):
            self.requests_rejected.labels(reason)

        self._last_preemptions = 0
        self._last_prefix_queries = 0
        self._last_prefix_hits = 0

    def sync_counters(self, *, preemptions: int, prefix_queries: int, prefix_hits: int) -> None:
        """Advance counters from the scheduler's monotonic totals."""
        self.preemptions.inc(preemptions - self._last_preemptions)
        self.prefix_queries.inc(prefix_queries - self._last_prefix_queries)
        self.prefix_hits.inc(prefix_hits - self._last_prefix_hits)
        self._last_preemptions = preemptions
        self._last_prefix_queries = prefix_queries
        self._last_prefix_hits = prefix_hits
        if prefix_queries:
            self.prefix_hit_rate.set(prefix_hits / prefix_queries)
