import pytest

from agentix.tito.config import TITOGatewayConfig
from agentix.tito import discovery


def test_explicit_backend_url_wins_over_environment_and_probe(monkeypatch):
    env = {"TITO_BACKEND_URL": "http://env.example:3000"}

    def fail_if_probed(*args, **kwargs):
        raise AssertionError("explicit backend URL must not probe candidates")

    monkeypatch.setattr(discovery, "probe_backend_url", fail_if_probed)

    assert (
        discovery.discover_backend_url(
            "localhost:8000",
            env=env,
            probe_candidates=("http://probe.example:8000",),
        )
        == "http://localhost:8000"
    )


def test_environment_precedence_is_deterministic_and_wins_over_probe(monkeypatch):
    env = {
        "OPENAI_BASE_URL": "http://openai.example:8000",
        "SGLANG_BASE_URL": "http://sglang.example:8000",
    }

    def fail_if_probed(*args, **kwargs):
        raise AssertionError("environment backend URL must not probe candidates")

    monkeypatch.setattr(discovery, "probe_backend_url", fail_if_probed)

    assert (
        discovery.discover_backend_url(env=env, probe_candidates=("http://probe.example:8000",))
        == "http://openai.example:8000"
    )


def test_probe_selects_health_success(monkeypatch):
    calls = []

    def endpoint_probe(url, timeout):
        calls.append((url, timeout))
        return url == "http://candidate.example:8000/health"

    monkeypatch.setattr(discovery, "_probe_endpoint", endpoint_probe)

    assert (
        discovery.discover_backend_url(
            env={},
            probe_candidates=("candidate.example:8000",),
            probe_timeout=1.5,
        )
        == "http://candidate.example:8000"
    )
    assert calls == [("http://candidate.example:8000/health", 1.5)]


def test_probe_falls_back_to_models_endpoint(monkeypatch):
    calls = []

    def endpoint_probe(url, timeout):
        calls.append(url)
        return url == "http://candidate.example:8000/v1/models"

    monkeypatch.setattr(discovery, "_probe_endpoint", endpoint_probe)

    assert (
        discovery.discover_backend_url(env={}, probe_candidates=("http://candidate.example:8000",))
        == "http://candidate.example:8000"
    )
    assert calls == [
        "http://candidate.example:8000/health",
        "http://candidate.example:8000/v1/models",
    ]


def test_probe_selection_uses_first_live_candidate(monkeypatch):
    calls = []

    def endpoint_probe(url, timeout):
        calls.append(url)
        return url == "http://second.example:8000/health"

    monkeypatch.setattr(discovery, "_probe_endpoint", endpoint_probe)

    assert (
        discovery.discover_backend_url(
            env={},
            probe_candidates=("http://first.example:8000", "http://second.example:8000"),
        )
        == "http://second.example:8000"
    )
    assert calls == [
        "http://first.example:8000/health",
        "http://first.example:8000/v1/models",
        "http://second.example:8000/health",
    ]


def test_missing_backend_url_fails_clearly():
    with pytest.raises(RuntimeError, match="backend URL not found"):
        discovery.discover_backend_url(env={}, probe_candidates=())


def test_no_live_probe_candidate_fails_clearly(monkeypatch):
    monkeypatch.setattr(discovery, "_probe_endpoint", lambda url, timeout: False)

    with pytest.raises(RuntimeError, match="start a live backend"):
        discovery.discover_backend_url(env={}, probe_candidates=("http://dead.example:8000",))


def test_from_cli_values_maps_fields():
    config = TITOGatewayConfig.from_cli_values(
        hf_checkpoint="model",
        backend_url="http://backend",
        chat_template_path=None,
        tito_model="qwen3",
        tito_allowed_append_roles=["tool", "user"],
        session_server_ip="127.0.0.1",
        session_server_port=30000,
        router_timeout=30,
        backend_probe_candidates=["http://probe-a:8000", "probe-b:8001"],
        backend_probe_timeout=2.0,
    )

    assert config.router_timeout == 30
    assert config.tito_allowed_append_roles == ("tool", "user")
    assert config.backend_probe_candidates == ("http://probe-a:8000", "probe-b:8001")
    assert config.backend_probe_timeout == 2.0


def test_invalid_append_role_fails():
    with pytest.raises(ValueError, match="unsupported tito append roles"):
        TITOGatewayConfig(hf_checkpoint="model", tito_allowed_append_roles=("assistant",))
