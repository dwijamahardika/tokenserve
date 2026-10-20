from __future__ import annotations

import asyncio

import httpx
from prometheus_client.parser import text_string_to_metric_families

from tests.conftest import collect, make_engine
from tokenserve.api import create_admin_app
from tokenserve.sequence import SamplingParams
from tokenserve.tokenizer import ByteTokenizer

TOK = ByteTokenizer()

EXPECTED = {
    "tokenserve_time_to_first_token_seconds",
    "tokenserve_time_per_output_token_seconds",
    "tokenserve_inter_token_latency_seconds",
    "tokenserve_queue_time_seconds",
    "tokenserve_e2e_request_latency_seconds",
    "tokenserve_step_duration_seconds",
    "tokenserve_batch_size_seqs",
    "tokenserve_batch_size_tokens",
    "tokenserve_num_requests_running",
    "tokenserve_num_requests_waiting",
    "tokenserve_kv_cache_usage_ratio",
    "tokenserve_kv_cache_blocks_total",
    "tokenserve_kv_cache_blocks_free",
    "tokenserve_prefix_cache_queries_tokens",
    "tokenserve_prefix_cache_hits_tokens",
    "tokenserve_prefix_cache_hit_rate",
    "tokenserve_preemptions",
    "tokenserve_prompt_tokens",
    "tokenserve_generation_tokens",
    "tokenserve_prefill_tokens_computed",
    "tokenserve_requests_finished",
    "tokenserve_requests_rejected",
}


def scrape(text: str) -> dict[str, float]:
    """Flatten an exposition into {sample name{labels}: value}."""
    out: dict[str, float] = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            labels = ",".join(f'{k}="{v}"' for k, v in sorted(s.labels.items()))
            out[f"{s.name}{{{labels}}}" if labels else s.name] = s.value
    return out


async def test_admin_endpoints_expose_metrics_health_and_state() -> None:
    # Tight cache + shared prefix => preemptions and prefix-cache hits both happen.
    eng = make_engine(num_blocks=24, block_size=4, max_num_seqs=6)
    await eng.start()
    shared = "system prompt shared by all requests. "
    streams = [
        eng.add_request(f"r{i}", TOK.encode(shared + str(i)), SamplingParams(max_tokens=20))
        for i in range(6)
    ]
    await asyncio.gather(*(collect(s) for s in streams))
    await eng.wait_idle()

    transport = httpx.ASGITransport(app=create_admin_app(eng))
    async with httpx.AsyncClient(transport=transport, base_url="http://admin") as c:
        r = await c.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        m = scrape(r.text)
        for name in EXPECTED:
            assert any(k.startswith(name) for k in m), f"{name} not exported"
        assert m["tokenserve_time_to_first_token_seconds_count"] == 6
        assert m["tokenserve_generation_tokens_total"] == 120
        assert m['tokenserve_requests_finished_total{finish_reason="length"}'] == 6
        assert m['tokenserve_requests_rejected_total{reason="queue_full"}'] == 0
        assert m["tokenserve_preemptions_total"] > 0
        assert m["tokenserve_prefix_cache_hits_tokens_total"] > 0
        assert 0 < m["tokenserve_prefix_cache_hit_rate"] < 1
        assert m["tokenserve_kv_cache_blocks_total"] == 24
        assert m["tokenserve_kv_cache_usage_ratio"] == 0  # all requests done
        assert m["tokenserve_batch_size_seqs_count"] == eng.num_steps

        assert (await c.get("/healthz")).status_code == 200
        assert (await c.get("/readyz")).json() == {"ready": True}
        state = (await c.get("/debug/state")).json()
        assert state["kv"]["blocks_free"] == 24
        assert state["preemptions"] == eng.scheduler.num_preemptions

        await eng.stop()
        assert (await c.get("/readyz")).status_code == 503
