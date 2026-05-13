"""Tests for agentix.models — pydantic validation and serialization."""

from __future__ import annotations

import pytest

from agentix.models import (
    AGENTIX_CLOSURE_ABI,
    ClosureManifest,
    RemoteError,
    RemoteRequest,
    RemoteResponse,
    SandboxConfig,
)


def test_closure_manifest_minimal():
    m = ClosureManifest(
        abi=AGENTIX_CLOSURE_ABI,
        name="core",
        version="0.1.0",
        package="agentix_closures.core",
    )
    assert m.package == "agentix_closures.core"
    assert m.kind is None


def test_closure_manifest_extra_allow():
    m = ClosureManifest.model_validate(
        {
            "abi": AGENTIX_CLOSURE_ABI,
            "name": "mock-agent",
            "version": "0.1.0",
            "package": "agentix_closures.mock_agent",
            "kind": "agent",
            "extra_field": "ignored-but-preserved",
        }
    )
    assert m.kind == "agent"


def test_closure_manifest_requires_abi_name_version_package():
    with pytest.raises(Exception):
        ClosureManifest(name="x", version="0.0.0", package="x")  # type: ignore[call-arg]


def test_remote_request_defaults():
    r = RemoteRequest(package="agentix_closures.echo", method="echo")
    assert r.args == []
    assert r.kwargs == {}


def test_remote_response_ok_shape():
    resp = RemoteResponse(ok=True, value={"x": 1})
    assert resp.error is None


def test_remote_response_error_shape():
    err = RemoteError(type="ValueError", message="bad")
    resp = RemoteResponse(ok=False, error=err)
    assert resp.value is None
    assert resp.error.type == "ValueError"


def test_sandbox_config_closures_is_list():
    cfg = SandboxConfig(
        image="ubuntu:24.04",
        runtime="agentix/runtime:0.1.0",
        closures=["agentix/claude-code:1.0.0", "agentix/swebench:1.0.0"],
    )
    assert cfg.closures == ["agentix/claude-code:1.0.0", "agentix/swebench:1.0.0"]
    assert cfg.env is None


def test_sandbox_config_default_closures_empty():
    cfg = SandboxConfig(image="ubuntu:24.04", runtime="agentix/runtime:0.1.0")
    assert cfg.closures == []


def test_sandbox_config_requires_runtime():
    with pytest.raises(Exception):
        SandboxConfig(image="ubuntu:24.04")  # type: ignore[call-arg]
