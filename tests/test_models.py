"""Tests for the framework's pydantic wire types."""

from __future__ import annotations

import pickle

import pytest

from agentix.deployment.base import SandboxConfig
from agentix.runtime.shared.callables import RemoteCallable
from agentix.runtime.shared.models import RemoteError, RemoteRequest, RemoteResponse


def _example_fn(a: int) -> int:
    return a + 1


def test_remote_request_round_trips():
    args_payload = pickle.dumps(((1, 2), {"k": "v"}))
    rc = RemoteCallable._resolve(_example_fn)
    r = RemoteRequest(callable=rc, arguments=args_payload)
    assert isinstance(r.callable, str)  # str subclass
    assert r.callable.resolve()(2) == 3  # round-trip back to fn
    assert pickle.loads(r.arguments) == ((1, 2), {"k": "v"})


def test_remote_callable_rejects_non_callable():
    with pytest.raises(TypeError):
        RemoteCallable._resolve(42)  # type: ignore[arg-type]


def test_remote_response_ok_shape():
    resp = RemoteResponse(ok=True, value=pickle.dumps({"x": 1}))
    assert resp.error is None
    assert pickle.loads(resp.value) == {"x": 1}


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


def test_sandbox_config_with_platform():
    cfg = SandboxConfig(
        image="ubuntu:24.04",
        runtime_image="my-agent:0.1.0",
        platform="linux/amd64",
    )
    assert cfg.platform == "linux/amd64"


def test_sandbox_config_requires_both_images():
    with pytest.raises(Exception):
        SandboxConfig()  # type: ignore[call-arg]
    with pytest.raises(Exception):
        SandboxConfig(image="ubuntu:24.04")  # type: ignore[call-arg]
    with pytest.raises(Exception):
        SandboxConfig(runtime_image="my-agent:0.1.0")  # type: ignore[call-arg]
