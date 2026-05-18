"""Tests for the framework's pydantic wire types."""

from __future__ import annotations

import pytest

from agentix.deployment.base import SandboxConfig
from agentix.runtime.shared.models import RemoteError, RemoteRequest, RemoteResponse


def test_remote_request_defaults():
    r = RemoteRequest(callable_payload=b"payload", display_name="agentix.echo::echo", shape="unary")
    assert r.args == []
    assert r.kwargs == {}
    assert r.callable_payload == b"payload"
    assert r.display_name == "agentix.echo::echo"
    assert r.shape == "unary"


def test_remote_response_ok_shape():
    resp = RemoteResponse(ok=True, value={"x": 1})
    assert resp.error is None


def test_remote_response_error_shape():
    err = RemoteError(type="ValueError", message="bad")
    resp = RemoteResponse(ok=False, error=err)
    assert resp.value is None
    assert resp.error.type == "ValueError"


def test_sandbox_config_two_images():
    cfg = SandboxConfig(image="ubuntu:24.04", runtime_image="my-agent:0.1.0")
    assert cfg.image == "ubuntu:24.04"
    assert cfg.runtime_image == "my-agent:0.1.0"
    assert cfg.env is None


def test_sandbox_config_with_env():
    cfg = SandboxConfig(
        image="ubuntu:24.04",
        runtime_image="my-agent:0.1.0",
        env={"FOO": "bar"},
    )
    assert cfg.env == {"FOO": "bar"}


def test_sandbox_config_requires_both_images():
    with pytest.raises(Exception):
        SandboxConfig()  # type: ignore[call-arg]
    with pytest.raises(Exception):
        SandboxConfig(image="ubuntu:24.04")  # type: ignore[call-arg]
    with pytest.raises(Exception):
        SandboxConfig(runtime_image="my-agent:0.1.0")  # type: ignore[call-arg]
