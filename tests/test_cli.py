from __future__ import annotations

import pytest

from tokenserve.cli import build_parser, settings_from_args
from tokenserve.scheduler import SchedulingPolicy
from tokenserve.server import build_engine


def test_defaults_bind_loopback_and_use_the_simulated_backend() -> None:
    s = settings_from_args(build_parser().parse_args([]))
    assert s.host == "127.0.0.1"
    assert s.admin_host == "127.0.0.1"
    assert s.backend == "sim"
    assert s.api.api_key is None
    assert s.engine.scheduler.policy is SchedulingPolicy.CONTINUOUS
    assert s.sim.base_latency_s == pytest.approx(0.010)
    assert s.sim.per_token_latency_s == pytest.approx(0.0001)


def test_flags_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENSERVE_NUM_BLOCKS", "128")
    monkeypatch.setenv("TOKENSERVE_PREFIX_CACHING", "false")
    monkeypatch.setenv("TOKENSERVE_API_KEY", "from-env")
    args = build_parser().parse_args(
        ["--scheduler", "static", "--no-chunked-prefill", "--max-num-batched-tokens", "4096",
         "--sim-base-latency-ms", "5"]
    )  # fmt: skip
    s = settings_from_args(args)
    assert s.engine.num_blocks == 128
    assert s.engine.enable_prefix_caching is False
    assert s.engine.scheduler.policy is SchedulingPolicy.STATIC
    assert s.engine.scheduler.enable_chunked_prefill is False
    assert s.sim.base_latency_s == pytest.approx(0.005)
    assert s.api.api_key == "from-env"
    engine = build_engine(s)
    assert engine.kv.num_blocks == 128
    assert not engine.kv.enable_prefix_caching


def test_inconsistent_configuration_is_rejected() -> None:
    args = build_parser().parse_args(["--no-chunked-prefill", "--max-num-batched-tokens", "512"])
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        settings_from_args(args)
