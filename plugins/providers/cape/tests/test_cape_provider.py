"""Unit tests for `CapeProvider` using a fake `cape` binary.

Follows the apptainer provider test pattern: a fake executable staged
in `tmp_path` records every invocation as one JSON line, and a real
local HTTP server answers `GET /health` so the provider's raw-TCP
probe succeeds. The point is to lock in the *assumed* CAPE CLI surface
the provider emits (verbs, flags, workload shape) — no network beyond
127.0.0.1, no real `cape`.

The fake models the assumed contract with a small file state machine:

  * `run` classifies each request by its workload — a workload that
    execs the bundle bootstrap is a *server* request (stays RUNNING
    until cancelled; overridable via `FAKE_CAPE_SERVER_STATE`); any
    other workload completes immediately. Multiple sandboxes therefore
    work in one test.
  * `logs` prints noisy banner lines plus the `AGENTIX_ENDPOINT` marker
    (port from `FAKE_CAPE_PORT`). `FAKE_CAPE_LOGS_EMPTY=N` withholds
    the marker for the first N calls (exercising the discovery retry
    loop); `FAKE_CAPE_LOGS_FAILURES=N` / `FAKE_CAPE_STATUS_FAILURES=N`
    make the first N invocations of that verb exit non-zero
    (exercising the transient-failure tolerance).
  * `cancel` records a per-request cancelled marker;
    `FAKE_CAPE_CANCEL_RC` forces a failing exit code (exercising the
    delete-retry path).
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from agentix.provider.cape import (
    CapeProvider,
    CapeProviderConfig,
    _boot_script,
    _cpu_cores,
    _memory_gb,
    _parse_endpoint,
)

from agentix.provider.base import SandboxConfig, SandboxId, SandboxProvider, SandboxResource

_FAKE_CAPE_BODY = '''
"""Recording fake `cape` CLI driven by FAKE_CAPE_* env vars."""
import json
import os
import sys

LOG = os.environ["FAKE_CAPE_LOG"]
STATE = os.environ["FAKE_CAPE_STATE"]


def log(argv):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": argv}) + "\\n")


def bump(name):
    """Increment and return a per-state-dir invocation counter."""
    path = os.path.join(STATE, name)
    n = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            n = int(f.read().strip() or "0")
    n += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(n))
    return n


def req_kind(req):
    path = os.path.join(STATE, "kind-" + req)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return "control"


def main():
    argv = sys.argv[1:]
    log(argv)
    verb = argv[0] if argv else ""
    if verb == "run":
        stderr = os.environ.get("FAKE_CAPE_RUN_STDERR")
        if stderr:
            sys.stderr.write(stderr + "\\n")
            sys.exit(1)
        n = bump("run")
        req = "req-fake-%d" % n
        sep = argv.index("--")
        workload = " ".join(argv[sep + 1 :])
        kind = "server" if "bootstrap.sh" in workload else "control"
        with open(os.path.join(STATE, "kind-" + req), "w", encoding="utf-8") as f:
            f.write(kind)
        print(req)
    elif verb == "status":
        req = argv[1]
        fails = int(os.environ.get("FAKE_CAPE_STATUS_FAILURES", "0"))
        if bump("status") <= fails:
            sys.stderr.write("fake cape: transient controller error\\n")
            sys.exit(1)
        if os.path.exists(os.path.join(STATE, "cancelled-" + req)):
            print(json.dumps({"state": "CANCELLED", "exit_code": 124, "request_id": req}))
        elif req_kind(req) == "server":
            state = os.environ.get("FAKE_CAPE_SERVER_STATE", "RUNNING")
            print(json.dumps({"state": state, "request_id": req}))
        else:
            print(json.dumps({"state": "COMPLETED", "exit_code": 0, "request_id": req}))
    elif verb == "logs":
        req = argv[1]
        n = bump("logs")
        lf = int(os.environ.get("FAKE_CAPE_LOGS_FAILURES", "0"))
        if n <= lf:
            sys.stderr.write("fake cape: transient log fetch error\\n")
            sys.exit(1)
        print("booting runtime...")
        print("GPU 0")
        le = int(os.environ.get("FAKE_CAPE_LOGS_EMPTY", "0"))
        if n - lf > le:
            print("AGENTIX_ENDPOINT 127.0.0.1 %s" % os.environ["FAKE_CAPE_PORT"])
    elif verb == "cancel":
        req = argv[1]
        rc = int(os.environ.get("FAKE_CAPE_CANCEL_RC", "0"))
        if rc:
            sys.stderr.write("fake cape: cancel rejected\\n")
            sys.exit(rc)
        with open(os.path.join(STATE, "cancelled-" + req), "w", encoding="utf-8"):
            pass
    else:
        sys.stderr.write("fake cape: unsupported verb %r\\n" % verb)
        sys.exit(2)


if __name__ == "__main__":
    main()
'''


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: object, **kwargs: object) -> None:
        pass


@pytest.fixture
def health_port():
    """A real loopback HTTP server answering 200 on `/health`."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def cape_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, health_port: int) -> dict[str, Any]:
    state = tmp_path / "state"
    state.mkdir()
    log = tmp_path / "cape.log.jsonl"
    fake = tmp_path / "fake-bin" / "cape"
    fake.parent.mkdir()
    fake.write_text(f"#!{sys.executable}\n{_FAKE_CAPE_BODY}")
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_CAPE_LOG", str(log))
    monkeypatch.setenv("FAKE_CAPE_STATE", str(state))
    monkeypatch.setenv("FAKE_CAPE_PORT", str(health_port))
    monkeypatch.setenv("CAPE_TOKEN", "unit-test-token")
    for var in (
        "CAPE_TOKEN_FILE",
        "CAPE_BINARY",
        "CAPE_CONTROLLER_URL",
        "FAKE_CAPE_SERVER_STATE",
        "FAKE_CAPE_RUN_STDERR",
        "FAKE_CAPE_CANCEL_RC",
        "FAKE_CAPE_LOGS_EMPTY",
        "FAKE_CAPE_LOGS_FAILURES",
        "FAKE_CAPE_STATUS_FAILURES",
    ):
        monkeypatch.delenv(var, raising=False)
    return {"binary": fake, "log": log, "state": state, "port": health_port}


def _provider(cape_env: dict[str, Any], **overrides: Any) -> CapeProvider:
    settings: dict[str, Any] = {
        "binary": str(cape_env["binary"]),
        "controller_url": "http://cape-controller.test:9000",
        "poll_interval_seconds": 0.05,
        "create_timeout_seconds": 30.0,
        "client_timeout_seconds": 30.0,
    }
    settings.update(overrides)
    return CapeProvider(CapeProviderConfig(**settings))


def _sandbox_config(**overrides: Any) -> SandboxConfig:
    defaults: dict[str, Any] = {
        "image": "docker://task-image:1",
        "bundle": "/mnt/shared/bundles/sha256-abc",
    }
    defaults.update(overrides)
    return SandboxConfig(**defaults)


def _log_entries(cape_env: dict[str, Any], verb: str | None = None) -> list[dict[str, Any]]:
    log: Path = cape_env["log"]
    if not log.exists():
        return []
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if verb is not None:
        entries = [e for e in entries if e["argv"] and e["argv"][0] == verb]
    return entries


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _flags(argv: list[str], name: str) -> list[str]:
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == name]


# ── create / run face ─────────────────────────────────────────────────────


async def test_create_returns_sandbox_and_emits_assumed_run_face(cape_env: dict[str, Any]) -> None:
    provider = _provider(cape_env)
    config = _sandbox_config(env={"HF_HOME": "/tmp/hf"}, resource=SandboxResource(gpu=2))
    sandbox = await provider.create(config)
    try:
        assert sandbox.status == "running"
        assert sandbox.runtime_url == f"http://127.0.0.1:{cape_env['port']}"

        runs = _log_entries(cape_env, verb="run")
        # One sandbox = one request; discovery must not submit a second
        # same-session command (the assumed session model is serial).
        assert len(runs) == 1
        argv = runs[0]["argv"]
        assert _flag(argv, "--controller-url") == "http://cape-controller.test:9000"
        assert _flag(argv, "--token") == "unit-test-token"
        assert _flag(argv, "--session-key") == f"agentix-{sandbox.sandbox_id}"
        assert _flag(argv, "--workspace") == f"/workspace/{sandbox.sandbox_id}"
        assert _flag(argv, "--cwd") == f"/workspace/{sandbox.sandbox_id}"
        assert _flag(argv, "--image") == "docker://task-image:1"
        assert _flag(argv, "--gpus") == "2"
        assert _flag(argv, "--gpu-mode") == "whole"
        assert _flag(argv, "--isolation-policy") == "default"
        assert _flag(argv, "--max-duration-seconds") == "14400"
        assert _flags(argv, "--bind") == ["/mnt/shared/bundles/sha256-abc:/nix:ro"]
        env_args = _flags(argv, "--env")
        assert "AGENTIX_BIND_PORT=8710" in env_args
        assert "HF_HOME=/tmp/hf" in env_args
        # Optional flags absent from a default config.
        for absent in ("--pool", "--user", "--cpu-cores", "--memory-gb", "--runtime-adapter"):
            assert absent not in argv
        # Workload sits after the `--` separator: print the endpoint
        # marker to stdout, then exec the bundle entry point.
        workload = argv[argv.index("--") + 1 :]
        assert workload[:2] == ["sh", "-c"]
        assert "AGENTIX_ENDPOINT" in workload[2]
        assert workload[2].endswith("exec /nix/runtime/bootstrap.sh")
        assert "mkdir" not in workload[2]
        assert ".agentix-endpoint" not in workload[2]
        # Discovery consumed status + logs of the server request only.
        assert _log_entries(cape_env, verb="logs")
    finally:
        await provider.delete(sandbox.sandbox_id)


async def test_optional_flags_emitted_when_configured(cape_env: dict[str, Any]) -> None:
    provider = _provider(
        cape_env,
        pool="pool-a",
        user="alice",
        runtime_adapter="apptainer",
        extra_binds=["/data:/data:rw"],
        isolation_policy="strict",
    )
    config = _sandbox_config(resource=SandboxResource(cpu=2.5, memory="16g", gpu=1))
    sandbox = await provider.create(config)
    try:
        argv = _log_entries(cape_env, verb="run")[0]["argv"]
        assert _flag(argv, "--pool") == "pool-a"
        assert _flag(argv, "--user") == "alice"
        assert _flag(argv, "--cpu-cores") == "3"  # ceil(2.5)
        assert _flag(argv, "--memory-gb") == "16"
        assert _flag(argv, "--gpus") == "1"
        assert _flag(argv, "--runtime-adapter") == "apptainer"
        assert _flag(argv, "--isolation-policy") == "strict"
        assert _flags(argv, "--bind") == [
            "/mnt/shared/bundles/sha256-abc:/nix:ro",
            "/data:/data:rw",
        ]
    finally:
        await provider.delete(sandbox.sandbox_id)


async def test_config_resource_defaults_used_when_resource_unset(cape_env: dict[str, Any]) -> None:
    provider = _provider(cape_env, cpu_cores=8, memory_gb=32)
    sandbox = await provider.create(_sandbox_config())
    try:
        argv = _log_entries(cape_env, verb="run")[0]["argv"]
        assert _flag(argv, "--cpu-cores") == "8"
        assert _flag(argv, "--memory-gb") == "32"
        assert _flag(argv, "--gpus") == "0"
    finally:
        await provider.delete(sandbox.sandbox_id)


# ── multi-sandbox ─────────────────────────────────────────────────────────


async def test_two_sandboxes_get_distinct_sessions_ports_and_cancels(
    cape_env: dict[str, Any],
) -> None:
    provider = _provider(cape_env)
    sb1 = await provider.create(_sandbox_config())
    sb2 = await provider.create(_sandbox_config())
    assert sb1.sandbox_id != sb2.sandbox_id
    assert len(provider._sandboxes) == 2

    runs = _log_entries(cape_env, verb="run")
    assert len(runs) == 2
    keys = {_flag(r["argv"], "--session-key") for r in runs}
    assert keys == {f"agentix-{sb1.sandbox_id}", f"agentix-{sb2.sandbox_id}"}
    ports = {
        e for r in runs for e in _flags(r["argv"], "--env") if e.startswith("AGENTIX_BIND_PORT=")
    }
    assert ports == {"AGENTIX_BIND_PORT=8710", "AGENTIX_BIND_PORT=8711"}

    # Deleting sandbox 1 cancels only its own request.
    await provider.delete(sb1.sandbox_id)
    cancels = _log_entries(cape_env, verb="cancel")
    assert [c["argv"][1] for c in cancels] == ["req-fake-1"]
    info = await provider.get(sb2.sandbox_id)
    assert info.status == "running"

    await provider.delete(sb2.sandbox_id)
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()


# ── delete ────────────────────────────────────────────────────────────────


async def test_delete_cancels_server_request_and_is_idempotent(cape_env: dict[str, Any]) -> None:
    provider = _provider(cape_env)
    sandbox = await provider.create(_sandbox_config())
    await provider.delete(sandbox.sandbox_id)

    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels, "no cape cancel recorded"
    assert cancels[-1]["argv"][1] == "req-fake-1"
    assert _flag(cancels[-1]["argv"], "--reason") == "agentix_delete"
    assert provider._inflight_ports == set()

    # Second delete of the same (now unknown) id is a no-op: no raise,
    # no extra cancel.
    await provider.delete(sandbox.sandbox_id)
    assert len(_log_entries(cape_env, verb="cancel")) == len(cancels)
    with pytest.raises(KeyError):
        await provider.get(sandbox.sandbox_id)


async def test_delete_keeps_bookkeeping_when_cancel_fails_then_retries(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(cape_env)
    sandbox = await provider.create(_sandbox_config())

    monkeypatch.setenv("FAKE_CAPE_CANCEL_RC", "1")
    await provider.delete(sandbox.sandbox_id)  # must not raise
    # Cancel was not confirmed: the record (and its port) stay so a
    # later delete() can retry instead of silently leaking the lease.
    assert sandbox.sandbox_id in provider._sandboxes
    info = await provider.get(sandbox.sandbox_id)
    assert info.status == "running"

    monkeypatch.delenv("FAKE_CAPE_CANCEL_RC")
    await provider.delete(sandbox.sandbox_id)
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()


# ── create failure / cancellation paths ───────────────────────────────────


async def test_create_failure_cancels_request_and_clears_bookkeeping(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CAPE_SERVER_STATE", "FAILED")
    provider = _provider(cape_env)
    with pytest.raises(RuntimeError, match="FAILED"):
        await provider.create(_sandbox_config())

    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels, "failed create must attempt a cancel"
    assert cancels[-1]["argv"][1] == "req-fake-1"
    assert _flag(cancels[-1]["argv"], "--reason") == "agentix_create_failed"
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()


async def test_discovery_retries_until_marker_appears(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Marker withheld for the first 2 logs calls, plus 2 transient status
    # failures: create must retry through both and still succeed.
    monkeypatch.setenv("FAKE_CAPE_LOGS_EMPTY", "2")
    monkeypatch.setenv("FAKE_CAPE_STATUS_FAILURES", "2")
    provider = _provider(cape_env)
    sandbox = await provider.create(_sandbox_config())
    try:
        assert len(_log_entries(cape_env, verb="logs")) >= 3
    finally:
        await provider.delete(sandbox.sandbox_id)


async def test_discovery_fails_after_consecutive_transient_failures(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CAPE_STATUS_FAILURES", "100000")
    provider = _provider(cape_env, transient_failure_limit=3)
    with pytest.raises(RuntimeError, match="3 consecutive"):
        await provider.create(_sandbox_config())
    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels and _flag(cancels[-1]["argv"], "--reason") == "agentix_create_failed"
    assert provider._sandboxes == {}


async def test_discovery_times_out_when_marker_never_appears(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CAPE_LOGS_EMPTY", "100000")
    provider = _provider(cape_env, create_timeout_seconds=1.0)
    with pytest.raises(TimeoutError, match="did not publish its endpoint"):
        await provider.create(_sandbox_config())
    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels and _flag(cancels[-1]["argv"], "--reason") == "agentix_create_failed"
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()


async def test_health_timeout_fails_create_and_cancels(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point the marker at a port nobody listens on: discovery succeeds,
    # the health probe burns the remaining budget, create fails + cancels.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    monkeypatch.setenv("FAKE_CAPE_PORT", str(dead_port))
    provider = _provider(cape_env, create_timeout_seconds=3.0)
    with pytest.raises(TimeoutError, match="not alive"):
        await provider.create(_sandbox_config())
    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels and _flag(cancels[-1]["argv"], "--reason") == "agentix_create_failed"
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()


async def test_cancelled_create_cancels_submitted_request(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep discovery spinning so the cancellation lands mid-create.
    monkeypatch.setenv("FAKE_CAPE_LOGS_EMPTY", "100000")
    provider = _provider(cape_env)
    task = asyncio.ensure_future(provider.create(_sandbox_config()))
    while not _log_entries(cape_env, verb="logs"):
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels, "cancelled create must attempt to cancel the submitted request"
    assert _flag(cancels[-1]["argv"], "--reason") in {
        "agentix_create_failed",
        "agentix_create_cancelled",
    }
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()


# ── get ───────────────────────────────────────────────────────────────────


async def test_get_unknown_raises_and_known_reports_running(cape_env: dict[str, Any]) -> None:
    provider = _provider(cape_env)
    with pytest.raises(KeyError, match="Sandbox not found"):
        await provider.get(SandboxId("cape-does-not-exist"))

    sandbox = await provider.create(_sandbox_config())
    try:
        info = await provider.get(sandbox.sandbox_id)
        assert info.status == "running"
        assert info.runtime_url == sandbox.runtime_url
        assert info.sandbox_id == sandbox.sandbox_id
    finally:
        await provider.delete(sandbox.sandbox_id)


async def test_get_maps_terminal_state_to_exited(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(cape_env)
    sandbox = await provider.create(_sandbox_config())
    try:
        monkeypatch.setenv("FAKE_CAPE_SERVER_STATE", "COMPLETED")
        info = await provider.get(sandbox.sandbox_id)
        assert info.status == "exited"
    finally:
        monkeypatch.delenv("FAKE_CAPE_SERVER_STATE", raising=False)
        await provider.delete(sandbox.sandbox_id)


# ── token hygiene ─────────────────────────────────────────────────────────


async def test_token_is_redacted_from_exception_text(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CAPE_RUN_STDERR", "authentication failed for token unit-test-token")
    provider = _provider(cape_env)
    with pytest.raises(RuntimeError) as excinfo:
        await provider.create(_sandbox_config())
    text = str(excinfo.value)
    assert "unit-test-token" not in text
    assert "<redacted>" in text


async def test_whitespace_containing_token_is_rejected(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPE_TOKEN", "bad token")
    provider = _provider(cape_env)
    with pytest.raises(RuntimeError, match="CAPE_TOKEN.*whitespace"):
        await provider.create(_sandbox_config())
    assert _log_entries(cape_env) == []  # rejected before any CLI call


async def test_token_file_rejected_when_group_or_other_accessible(
    cape_env: dict[str, Any], tmp_path: Path
) -> None:
    token_file = tmp_path / "cape-token"
    token_file.write_text("file-token-abc\n")
    token_file.chmod(0o644)
    provider = _provider(cape_env, token_file=str(token_file))
    with pytest.raises(RuntimeError, match="group/other"):
        await provider.create(_sandbox_config())
    assert _log_entries(cape_env) == []


async def test_token_file_wins_over_env_and_expands_tilde(
    cape_env: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    token_file = home / "cape-token"
    token_file.write_text("file-token-abc\n")
    token_file.chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    # CAPE_TOKEN stays set from the fixture; the file must win.
    provider = _provider(cape_env, token_file="~/cape-token")
    sandbox = await provider.create(_sandbox_config())
    try:
        argv = _log_entries(cape_env, verb="run")[0]["argv"]
        assert _flag(argv, "--token") == "file-token-abc"
    finally:
        await provider.delete(sandbox.sandbox_id)


async def test_empty_token_file_is_rejected(cape_env: dict[str, Any], tmp_path: Path) -> None:
    token_file = tmp_path / "cape-token"
    token_file.write_text("\n")
    token_file.chmod(0o600)
    provider = _provider(cape_env, token_file=str(token_file))
    with pytest.raises(RuntimeError, match="empty"):
        await provider.create(_sandbox_config())
    assert _log_entries(cape_env) == []


# ── pure helpers ──────────────────────────────────────────────────────────


def test_cpu_cores_mapping() -> None:
    cfg = CapeProviderConfig()
    assert _cpu_cores(SandboxResource(cpu=0.5), cfg) == 1
    assert _cpu_cores(SandboxResource(cpu=2.5), cfg) == 3
    assert _cpu_cores(SandboxResource(cpu=4), cfg) == 4
    assert _cpu_cores(None, cfg) is None
    assert _cpu_cores(None, CapeProviderConfig(cpu_cores=8)) == 8


def test_memory_gb_mapping() -> None:
    cfg = CapeProviderConfig()
    assert _memory_gb(SandboxResource(memory="16g"), cfg) == 16
    assert _memory_gb(SandboxResource(memory="512m"), cfg) == 1  # rounds up to whole GiB
    assert _memory_gb(SandboxResource(memory=1 << 30), cfg) == 1  # int = bytes
    assert _memory_gb(SandboxResource(memory=(1 << 30) + 1), cfg) == 2
    assert _memory_gb(None, cfg) is None
    assert _memory_gb(None, CapeProviderConfig(memory_gb=32)) == 32
    with pytest.raises(RuntimeError, match="cannot map memory"):
        _memory_gb(SandboxResource(memory="sixteen gigs"), cfg)


def test_parse_endpoint_requires_marker_prefix() -> None:
    marker = "booting runtime...\nGPU 0\nAGENTIX_ENDPOINT 10.0.0.5 8710\n"
    assert _parse_endpoint(marker) == ("10.0.0.5", 8710)
    # Two-token noise ("GPU 0") must never be misread as an endpoint.
    assert _parse_endpoint("GPU 0\n") is None
    assert _parse_endpoint("") is None
    assert _parse_endpoint("AGENTIX_ENDPOINT host notaport\n") is None
    assert _parse_endpoint("AGENTIX_ENDPOINT 10.0.0.5\n") is None


def test_boot_script_marker_roundtrip(tmp_path: Path) -> None:
    """Run the generated boot script under a real `sh` (with a stubbed
    multi-IP `hostname`) and feed its stdout to `_parse_endpoint` —
    the producer/consumer pair must agree."""
    stub = tmp_path / "stub-bin"
    stub.mkdir()
    hostname = stub / "hostname"
    hostname.write_text("#!/bin/sh\necho '10.0.0.5 172.17.0.1'\n")
    hostname.chmod(0o755)
    env = {
        "PATH": f"{stub}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "AGENTIX_BIND_PORT": "9999",
    }
    # `exec /nix/runtime/bootstrap.sh` fails in the test environment —
    # the marker must already be on stdout by then.
    proc = subprocess.run(["sh", "-c", _boot_script()], env=env, capture_output=True, text=True)
    assert _parse_endpoint(proc.stdout) == ("10.0.0.5", 9999)


# ── registry contract ─────────────────────────────────────────────────────


def test_zero_arg_constructor_for_plugin_registry() -> None:
    # The plugin registry instantiates providers with `cls()` — every
    # config field must be optional or defaulted.
    provider = CapeProvider()
    assert isinstance(provider, SandboxProvider)
    assert provider.config.workspace_root == "/workspace"
    assert provider.config.url_template == "http://{host}:{port}"
    assert provider.config.runtime_port_base == 8710
