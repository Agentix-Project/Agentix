"""Tests for the framework's pydantic wire types."""

from __future__ import annotations

import pickle

import pytest

from agentix.provider.base import SandboxConfig, SandboxResource
from agentix.runtime.shared.callables import RemoteCallable
from agentix.runtime.shared.codec import pack, unpack
from agentix.runtime.shared.models import RemoteError, RemoteRequest, RemoteResponse

BUNDLE_REF = "/cache/agentix/bundles/sha256-deadbeef"


def _example_fn(a: int) -> int:
    return a + 1


def test_remote_request_round_trips():
    args_payload = pickle.dumps(((1, 2), {"k": "v"}))
    rc = RemoteCallable._resolve(_example_fn)
    r = RemoteRequest(callable=rc, arguments=args_payload)
    assert isinstance(r.callable, str)  # str subclass
    assert r.callable == "tests.runtime.shared.test_models::_example_fn"
    assert r.callable.resolve()(2) == 3  # round-trip back to fn
    assert pickle.loads(r.arguments) == ((1, 2), {"k": "v"})


def test_remote_callable_rejects_non_callable():
    with pytest.raises(TypeError):
        RemoteCallable._resolve(42)  # type: ignore[arg-type]


def test_remote_response_ok_shape():
    resp = RemoteResponse(ok=True, value=pack({"x": 1}))
    assert resp.error is None
    assert unpack(resp.value) == {"x": 1}


def test_remote_response_error_shape():
    err = RemoteError(type="ValueError", message="bad")
    resp = RemoteResponse(ok=False, error=err)
    assert resp.value is None
    assert resp.error.type == "ValueError"


def test_sandbox_config_image_and_bundle_ref():
    cfg = SandboxConfig(image="ubuntu:24.04", bundle=BUNDLE_REF)
    assert cfg.image == "ubuntu:24.04"
    assert cfg.bundle == BUNDLE_REF
    assert cfg.env is None


def test_sandbox_config_with_env():
    cfg = SandboxConfig(
        image="ubuntu:24.04",
        bundle=BUNDLE_REF,
        env={"FOO": "bar"},
    )
    assert cfg.env == {"FOO": "bar"}


def test_sandbox_config_with_resource():
    cfg = SandboxConfig(
        image="ubuntu:24.04",
        bundle=BUNDLE_REF,
        resource=SandboxResource(cpu=4, memory="16g", gpu=2),
    )
    assert cfg.resource is not None
    assert cfg.resource.cpu == 4
    assert cfg.resource.memory == "16g"
    assert cfg.resource.gpu == 2


def test_sandbox_resource_validates_positive_values():
    with pytest.raises(Exception):
        SandboxResource(cpu=0)
    with pytest.raises(Exception):
        SandboxResource(memory=0)
    with pytest.raises(Exception):
        SandboxResource(memory="")
    with pytest.raises(Exception):
        SandboxResource(gpu=0)


def test_sandbox_config_with_platform():
    cfg = SandboxConfig(
        image="ubuntu:24.04",
        bundle=BUNDLE_REF,
        platform="linux/amd64",
    )
    assert cfg.platform == "linux/amd64"


def test_sandbox_config_requires_both_images():
    with pytest.raises(Exception):
        SandboxConfig()  # type: ignore[call-arg]
    with pytest.raises(Exception):
        SandboxConfig(image="ubuntu:24.04")  # type: ignore[call-arg]
    with pytest.raises(Exception):
        SandboxConfig(bundle=BUNDLE_REF)  # type: ignore[call-arg]
