"""Unit tests for `CapeProvider` using a fake `cape` binary.

Follows the apptainer provider test pattern: a fake executable staged
in `tmp_path` records every invocation as one JSON line, and a real
local HTTP server answers `GET /health` so the provider's raw-TCP
probe succeeds. The fake models the REAL CAPE CLI contract (verified
against the CAPE source) — no network beyond 127.0.0.1, no real `cape`.

Real-contract facts the fake reproduces:

  * `run` parses argv with the real flag table: `--pool` / `--user` /
    `--task-id` are required and there is NO `--workspace` and NO
    `--gpu-mode` flag (removed upstream; requests inherit
    exclusive/shared from the target pool) — violations exit 2 with an
    argparse-style usage error, exactly like the real parser.
  * Request ids look like the real ones (`req-%06d`).
  * `logs` returns EMPTY stdout/stderr while the request is running
    (CAPE has no streaming logs); output appears only once the request
    is terminal. Endpoint discovery must therefore never depend on
    `cape logs` — it reads the marker file the boot script writes.
  * `run` actually EXECUTES a server workload (the provider's boot
    script) under `sh -c` with a stubbed node environment:
    `AGENTIX_BIND_PORT` is overridden with `FAKE_CAPE_PORT` (where the
    test's health server listens) and `FAKE_CAPE_STUB_PATH` is
    prepended to PATH (stub `hostname` resolving to loopback). The
    boot script then really writes the endpoint marker into the tmp
    meta dir the provider polls. `FAKE_CAPE_EXEC_WORKLOAD=0` skips
    execution so the marker never appears.
  * `cancel` prints the post-cancel status JSON; `FAKE_CAPE_CANCEL_RC`
    forces a failing exit code (delete-retry path) and
    `FAKE_CAPE_CANCEL_404` emits the CLI's single-line HTTP-error JSON
    (`{"status": 404, ...}`) on stderr with rc=1 (already-gone path).
  * `FAKE_CAPE_STATUS_FAILURES=N` makes the first N `status` calls
    exit non-zero (transient-failure tolerance);
    `FAKE_CAPE_SERVER_STATE` overrides the server request's state.

The fake keeps the status JSON minimal ({state, request_id,
exit_code}) plus the new upstream cotenancy fields (`gpu_mode`,
`gpu_cotenant_count`, `gpu_cotenant_counts`) which the provider must
tolerate — it only ever reads `state` and the `request_id` echo.
`cape wait` is deliberately not modeled: the provider must never call
it (its exit code is untrustworthy).
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
"""Recording fake `cape` CLI modeling the REAL contract (see test module)."""
import argparse
import json
import os
import subprocess
import sys

LOG = os.environ["FAKE_CAPE_LOG"]
STATE = os.environ["FAKE_CAPE_STATE"]
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "LOST", "INFEASIBLE"}


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


def read_state(name, default=""):
    path = os.path.join(STATE, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_state(name, text):
    with open(os.path.join(STATE, name), "w", encoding="utf-8") as f:
        f.write(text)


def build_parser():
    parser = argparse.ArgumentParser(prog="cape")
    sub = parser.add_subparsers(dest="verb", required=True)
    run = sub.add_parser("run")
    # Mirrors the real run parser flag for flag: --pool/--user/--task-id
    # required, NO --workspace, REMAINDER workload. Sharing argparse with
    # the real CLI makes rejection behavior (rc=2, usage on stderr)
    # match by construction.
    run.add_argument("--controller-url")
    run.add_argument("--token")
    run.add_argument("--pool", required=True)
    run.add_argument("--user", required=True)
    run.add_argument("--task-id", required=True)
    run.add_argument("--image")
    run.add_argument("--profile")
    run.add_argument("--gpus", type=int, default=0)
    run.add_argument("--cpu-cores", type=int, default=1)
    run.add_argument("--memory-gb", type=int, default=1)
    run.add_argument("--max-duration-seconds", type=float)
    run.add_argument("--runtime-adapter")
    run.add_argument("--hard-gpu-enforcement", action="store_true")
    run.add_argument("--port", type=int, action="append", default=[])
    run.add_argument("--locality")
    # NO --gpu-mode: upstream removed the flag (requests inherit
    # exclusive/shared from the target pool), so sending it makes
    # argparse exit 2 with a usage error, like any unknown flag.
    run.add_argument("--isolation-policy", default="default")
    run.add_argument("--cwd")
    run.add_argument("--env", action="append", default=[])
    run.add_argument("--bind", action="append", default=[])
    run.add_argument("--log-path")
    run.add_argument("--artifact-path", action="append", default=[])
    run.add_argument("--session-key")
    run.add_argument("--wait", action="store_true")
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("workload_command", nargs=argparse.REMAINDER)
    for name in ("status", "logs", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("request_id")
        p.add_argument("--controller-url")
        p.add_argument("--token")
        if name == "cancel":
            p.add_argument("--reason", default="cancelled")
    return parser


def request_state(req):
    if read_state("cancelled-" + req):
        return "CANCELLED"
    if read_state("kind-" + req).strip() == "server":
        return os.environ.get("FAKE_CAPE_SERVER_STATE", "RUNNING")
    return "COMPLETED"


def do_run(args):
    forced_stderr = os.environ.get("FAKE_CAPE_RUN_STDERR")
    if forced_stderr:
        sys.stderr.write(forced_stderr + "\\n")
        sys.exit(1)
    req = "req-%06d" % bump("run")
    workload = list(args.workload_command)
    if workload and workload[0] == "--":
        workload = workload[1:]
    if not workload:
        sys.stderr.write("fake cape: empty workload command\\n")
        sys.exit(1)
    kind = "server" if "bootstrap.sh" in " ".join(workload) else "control"
    write_state("kind-" + req, kind)
    if kind == "server" and os.environ.get("FAKE_CAPE_EXEC_WORKLOAD", "1") != "0":
        # Model the node agent: run the workload with the request's env
        # applied, in a stubbed node environment (health-server port,
        # loopback `hostname`). The boot script really writes the
        # endpoint marker into the meta dir the provider polls.
        env = dict(os.environ)
        for item in args.env:
            key, _, value = item.partition("=")
            env[key] = value
        if os.environ.get("FAKE_CAPE_PORT"):
            env["AGENTIX_BIND_PORT"] = os.environ["FAKE_CAPE_PORT"]
        if os.environ.get("FAKE_CAPE_STUB_PATH"):
            env["PATH"] = os.environ["FAKE_CAPE_STUB_PATH"] + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(workload, env=env, capture_output=True, text=True, timeout=60)
        write_state("out-" + req, proc.stdout)
        write_state("err-" + req, proc.stderr)
    print(req)


def do_status(args):
    fails = int(os.environ.get("FAKE_CAPE_STATUS_FAILURES", "0"))
    if bump("status") <= fails:
        sys.stderr.write("fake cape: transient controller error\\n")
        sys.exit(1)
    req = args.request_id
    state = request_state(req)
    record = {"state": state, "request_id": req}
    if state == "CANCELLED":
        record["exit_code"] = 124
    elif state in TERMINAL:
        record["exit_code"] = 0
    # New upstream cotenancy fields: gpu_mode is DERIVED from the target
    # pool (null | "exclusive" | "shared"); the provider must tolerate
    # (ignore) all three.
    record["gpu_mode"] = None
    record["gpu_cotenant_count"] = 0
    record["gpu_cotenant_counts"] = []
    print(json.dumps(record))


def do_logs(args):
    req = args.request_id
    # Real contract: stdout/stderr stay EMPTY until the command exits;
    # the node agent reports output once, on terminal states only.
    if request_state(req) in TERMINAL:
        sys.stdout.write(read_state("out-" + req, "fake-terminal-stdout\\n"))
        sys.stderr.write(read_state("err-" + req, "fake-terminal-stderr\\n"))


def do_cancel(args):
    req = args.request_id
    if os.environ.get("FAKE_CAPE_CANCEL_404"):
        payload = {"status": 404, "error": {"detail": "unknown request: " + req}}
        sys.stderr.write(json.dumps(payload) + "\\n")
        sys.exit(1)
    rc = int(os.environ.get("FAKE_CAPE_CANCEL_RC", "0"))
    if rc:
        sys.stderr.write("fake cape: cancel rejected\\n")
        sys.exit(rc)
    write_state("cancelled-" + req, "1")
    print(json.dumps({"state": "CANCELLED", "request_id": req, "exit_code": None}))


def main():
    argv = sys.argv[1:]
    log(argv)
    args = build_parser().parse_args(argv)
    {"run": do_run, "status": do_status, "logs": do_logs, "cancel": do_cancel}[args.verb](args)


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
    meta_root = tmp_path / "meta"
    meta_root.mkdir()
    workspace_root = tmp_path / "ws"
    log = tmp_path / "cape.log.jsonl"
    fake = tmp_path / "fake-bin" / "cape"
    fake.parent.mkdir()
    fake.write_text(f"#!{sys.executable}\n{_FAKE_CAPE_BODY}")
    fake.chmod(0o755)
    # Stub `hostname` so the boot script's endpoint marker resolves to
    # loopback regardless of the test host's real hostname/-i support.
    stub = tmp_path / "stub-bin"
    stub.mkdir()
    hostname = stub / "hostname"
    hostname.write_text('#!/bin/sh\necho "127.0.0.1 10.0.0.5"\n')
    hostname.chmod(0o755)
    monkeypatch.setenv("FAKE_CAPE_LOG", str(log))
    monkeypatch.setenv("FAKE_CAPE_STATE", str(state))
    monkeypatch.setenv("FAKE_CAPE_PORT", str(health_port))
    monkeypatch.setenv("FAKE_CAPE_STUB_PATH", str(stub))
    monkeypatch.setenv("CAPE_TOKEN", "unit-test-token")
    for var in (
        "CAPE_TOKEN_FILE",
        "CAPE_BINARY",
        "CAPE_CONTROLLER_URL",
        "FAKE_CAPE_SERVER_STATE",
        "FAKE_CAPE_RUN_STDERR",
        "FAKE_CAPE_CANCEL_RC",
        "FAKE_CAPE_CANCEL_404",
        "FAKE_CAPE_EXEC_WORKLOAD",
        "FAKE_CAPE_STATUS_FAILURES",
    ):
        monkeypatch.delenv(var, raising=False)
    return {
        "binary": fake,
        "log": log,
        "state": state,
        "port": health_port,
        "meta_root": meta_root,
        "workspace_root": workspace_root,
    }


def _provider(cape_env: dict[str, Any], **overrides: Any) -> CapeProvider:
    settings: dict[str, Any] = {
        "binary": str(cape_env["binary"]),
        "controller_url": "http://cape-controller.test:9000",
        "pool": "unit-pool",
        "user": "unit-user",
        "meta_root": str(cape_env["meta_root"]),
        "workspace_root": str(cape_env["workspace_root"]),
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


def _meta_dir(cape_env: dict[str, Any], sandbox_id: str) -> Path:
    return Path(cape_env["meta_root"]) / sandbox_id


# ── create / run face ─────────────────────────────────────────────────────


async def test_create_returns_sandbox_and_emits_real_run_face(cape_env: dict[str, Any]) -> None:
    provider = _provider(cape_env)
    config = _sandbox_config(env={"HF_HOME": "/tmp/hf"}, resource=SandboxResource(gpu=2))
    sandbox = await provider.create(config)
    meta_dir = _meta_dir(cape_env, str(sandbox.sandbox_id))
    try:
        assert sandbox.status == "running"
        assert sandbox.runtime_url == f"http://127.0.0.1:{cape_env['port']}"

        runs = _log_entries(cape_env, verb="run")
        # One sandbox = one request; discovery must not submit a second
        # same-session command (the session model is serial).
        assert len(runs) == 1
        argv = runs[0]["argv"]
        assert _flag(argv, "--controller-url") == "http://cape-controller.test:9000"
        assert _flag(argv, "--token") == "unit-test-token"
        assert _flag(argv, "--pool") == "unit-pool"
        assert _flag(argv, "--user") == "unit-user"
        # --task-id is required by the real parser; the provider passes
        # the sandbox id.
        assert _flag(argv, "--task-id") == str(sandbox.sandbox_id)
        assert _flag(argv, "--session-key") == f"agentix-{sandbox.sandbox_id}"
        # The real CLI has no --workspace flag, and --cwd would point at
        # a directory nothing creates — the boot script mkdir+cd's.
        assert "--workspace" not in argv
        assert "--cwd" not in argv
        assert _flag(argv, "--image") == "docker://task-image:1"
        assert _flag(argv, "--gpus") == "2"
        # Upstream removed --gpu-mode (pool-level exclusive/shared);
        # sending it would make the real CLI exit 2 before submitting.
        assert "--gpu-mode" not in argv
        assert _flag(argv, "--isolation-policy") == "default"
        assert _flag(argv, "--max-duration-seconds") == "14400"
        assert _flags(argv, "--bind") == [
            "/mnt/shared/bundles/sha256-abc:/nix:ro",
            f"{meta_dir}:/agentix-meta:rw",
        ]
        env_args = _flags(argv, "--env")
        assert "AGENTIX_BIND_PORT=8710" in env_args
        assert "HF_HOME=/tmp/hf" in env_args
        # Optional flags absent from a default config.
        for absent in ("--cpu-cores", "--memory-gb", "--runtime-adapter"):
            assert absent not in argv
        # Workload sits after the `--` separator: create the workspace,
        # write the endpoint marker file, then exec the bundle entry point.
        workload = argv[argv.index("--") + 1 :]
        assert workload[:2] == ["sh", "-c"]
        script = workload[2]
        assert f"mkdir -p {cape_env['workspace_root']}/{sandbox.sandbox_id}" in script
        assert "AGENTIX_ENDPOINT" in script
        assert "/agentix-meta" in script
        assert "/nix/runtime/bootstrap.sh" in script
        # The boot script really ran and wrote the marker the provider
        # discovered; `cape logs` was never needed (real CAPE returns
        # empty logs while the request runs).
        assert (meta_dir / "endpoint").is_file()
        assert _log_entries(cape_env, verb="logs") == []
    finally:
        await provider.delete(sandbox.sandbox_id)
    assert not meta_dir.exists()  # delete removes the per-sandbox meta dir


async def test_optional_flags_emitted_when_configured(cape_env: dict[str, Any]) -> None:
    provider = _provider(
        cape_env,
        runtime_adapter="apptainer",
        extra_binds=["/data:/data:rw"],
        isolation_policy="host-stateless",
    )
    config = _sandbox_config(resource=SandboxResource(cpu=2.5, memory="16g", gpu=1))
    sandbox = await provider.create(config)
    try:
        argv = _log_entries(cape_env, verb="run")[0]["argv"]
        assert _flag(argv, "--cpu-cores") == "3"  # ceil(2.5)
        assert _flag(argv, "--memory-gb") == "16"
        assert _flag(argv, "--gpus") == "1"
        assert _flag(argv, "--runtime-adapter") == "apptainer"
        assert _flag(argv, "--isolation-policy") == "host-stateless"
        assert _flags(argv, "--bind") == [
            "/mnt/shared/bundles/sha256-abc:/nix:ro",
            f"{_meta_dir(cape_env, str(sandbox.sandbox_id))}:/agentix-meta:rw",
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


async def test_missing_pool_user_meta_root_fail_fast(cape_env: dict[str, Any]) -> None:
    # The real `cape run` exits 2 (argparse) without --pool/--user, and
    # discovery cannot work without meta_root: fail fast, before any CLI
    # call or meta dir creation.
    for missing in ("pool", "user", "meta_root"):
        provider = _provider(cape_env, **{missing: None})
        with pytest.raises(RuntimeError, match=missing):
            await provider.create(_sandbox_config())
        assert provider._inflight_ports == set()
    assert _log_entries(cape_env) == []
    assert list(Path(cape_env["meta_root"]).iterdir()) == []


# ── fake-CLI contract (documents the real parser behavior) ───────────────


def test_fake_cape_rejects_workspace_flag_and_requires_pool_user_task_id(
    cape_env: dict[str, Any],
) -> None:
    binary = str(cape_env["binary"])
    # The legacy (pre-verification) run face carried --workspace: the
    # real parser rejects it with an argparse usage error, rc=2.
    legacy = [
        binary, "run", "--pool", "p", "--user", "u", "--task-id", "t",
        "--workspace", "/ws", "--image", "img", "--", "sh", "-c", "true",
    ]  # fmt: skip
    proc = subprocess.run(legacy, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "--workspace" in proc.stderr
    # --pool/--user/--task-id are required=True in the real parser.
    proc = subprocess.run(
        [binary, "run", "--image", "img", "--", "sh", "-c", "true"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    for flag in ("--pool", "--user", "--task-id"):
        assert flag in proc.stderr


def test_fake_cape_rejects_dropped_gpu_mode_flag(cape_env: dict[str, Any]) -> None:
    # Upstream removed `cape run --gpu-mode` entirely (requests now
    # inherit exclusive/shared from the target pool). Sending the flag
    # is an unknown-argument error: argparse exits 2 with usage on
    # stderr, so a provider that still emits it never submits anything.
    binary = str(cape_env["binary"])
    legacy = [
        binary, "run", "--pool", "p", "--user", "u", "--task-id", "t",
        "--gpu-mode", "whole", "--image", "img", "--", "sh", "-c", "true",
    ]  # fmt: skip
    proc = subprocess.run(legacy, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "--gpu-mode" in proc.stderr


async def test_run_face_never_emits_dropped_gpu_mode_flag(cape_env: dict[str, Any]) -> None:
    # Regression for the upstream flag removal: the run argv must not
    # contain --gpu-mode for any GPU count (0 or N). The fake parser
    # would exit 2 on it, but assert explicitly so the failure reads as
    # a contract violation rather than a create() error.
    provider = _provider(cape_env)
    for resource, gpus in ((None, 0), (SandboxResource(gpu=4), 4)):
        sandbox = await provider.create(_sandbox_config(resource=resource))
        try:
            argv = _log_entries(cape_env, verb="run")[-1]["argv"]
            assert "--gpu-mode" not in argv
            assert _flag(argv, "--gpus") == str(gpus)
        finally:
            await provider.delete(sandbox.sandbox_id)


async def test_fake_cape_logs_empty_while_running_populated_when_terminal(
    cape_env: dict[str, Any],
) -> None:
    provider = _provider(cape_env)
    sandbox = await provider.create(_sandbox_config())
    binary = str(cape_env["binary"])
    logs_argv = [binary, "logs", "req-000001", "--controller-url", "u", "--token", "t"]
    # Real contract: rc=0 but EMPTY output while the request runs.
    proc = subprocess.run(logs_argv, capture_output=True, text=True)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")
    await provider.delete(sandbox.sandbox_id)
    # Terminal (cancelled) → the workload's recorded output is available.
    proc = subprocess.run(logs_argv, capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stderr  # the boot script's exec failure landed on stderr


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
    task_ids = {_flag(r["argv"], "--task-id") for r in runs}
    assert task_ids == {str(sb1.sandbox_id), str(sb2.sandbox_id)}
    ports = {
        e for r in runs for e in _flags(r["argv"], "--env") if e.startswith("AGENTIX_BIND_PORT=")
    }
    assert ports == {"AGENTIX_BIND_PORT=8710", "AGENTIX_BIND_PORT=8711"}

    # Deleting sandbox 1 cancels only its own request and removes only
    # its own meta dir.
    await provider.delete(sb1.sandbox_id)
    cancels = _log_entries(cape_env, verb="cancel")
    assert [c["argv"][1] for c in cancels] == ["req-000001"]
    assert not _meta_dir(cape_env, str(sb1.sandbox_id)).exists()
    assert _meta_dir(cape_env, str(sb2.sandbox_id)).is_dir()
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
    assert cancels[-1]["argv"][1] == "req-000001"
    assert _flag(cancels[-1]["argv"], "--reason") == "agentix_delete"
    assert provider._inflight_ports == set()
    assert not _meta_dir(cape_env, str(sandbox.sandbox_id)).exists()

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
    meta_dir = _meta_dir(cape_env, str(sandbox.sandbox_id))

    monkeypatch.setenv("FAKE_CAPE_CANCEL_RC", "1")
    await provider.delete(sandbox.sandbox_id)  # must not raise
    # Cancel was not confirmed: the record (and its port and meta dir)
    # stay so a later delete() can retry instead of leaking the lease.
    assert sandbox.sandbox_id in provider._sandboxes
    assert meta_dir.is_dir()
    info = await provider.get(sandbox.sandbox_id)
    assert info.status == "running"

    monkeypatch.delenv("FAKE_CAPE_CANCEL_RC")
    await provider.delete(sandbox.sandbox_id)
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()
    assert not meta_dir.exists()


async def test_delete_treats_cancel_404_as_already_gone(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A journal-less controller restart forgets the request: cancel then
    # fails with the CLI's stderr JSON `{"status": 404, ...}` and rc=1.
    # That request is gone — bookkeeping must not stick forever.
    provider = _provider(cape_env)
    sandbox = await provider.create(_sandbox_config())
    meta_dir = _meta_dir(cape_env, str(sandbox.sandbox_id))

    monkeypatch.setenv("FAKE_CAPE_CANCEL_404", "1")
    await provider.delete(sandbox.sandbox_id)
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()
    assert not meta_dir.exists()


# ── create failure / cancellation paths ───────────────────────────────────


async def test_terminal_state_fails_fast_with_terminal_logs(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The workload dies before writing the marker: discovery must fail
    # fast on the terminal state and enrich the error with `cape logs`
    # output (populated once the request is terminal).
    monkeypatch.setenv("FAKE_CAPE_EXEC_WORKLOAD", "0")
    monkeypatch.setenv("FAKE_CAPE_SERVER_STATE", "FAILED")
    provider = _provider(cape_env)
    with pytest.raises(RuntimeError, match="terminal state FAILED") as excinfo:
        await provider.create(_sandbox_config())
    text = str(excinfo.value)
    assert "fake-terminal-stdout" in text
    assert "fake-terminal-stderr" in text

    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels, "failed create must attempt a cancel"
    assert cancels[-1]["argv"][1] == "req-000001"
    assert _flag(cancels[-1]["argv"], "--reason") == "agentix_create_failed"
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()
    assert list(Path(cape_env["meta_root"]).iterdir()) == []


async def test_discovery_tolerates_transient_status_failures(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Marker withheld (workload not executed) plus 2 transient status
    # failures: discovery must retry through the blips and still succeed
    # once the marker file appears.
    monkeypatch.setenv("FAKE_CAPE_EXEC_WORKLOAD", "0")
    monkeypatch.setenv("FAKE_CAPE_STATUS_FAILURES", "2")
    provider = _provider(cape_env)
    task = asyncio.ensure_future(provider.create(_sandbox_config()))
    while len(_log_entries(cape_env, verb="status")) < 2:
        assert not task.done()
        await asyncio.sleep(0.02)
    # Write the marker the way the workload's boot script would.
    [meta_dir] = [p for p in Path(cape_env["meta_root"]).iterdir() if p.is_dir()]
    (meta_dir / "endpoint").write_text(f"AGENTIX_ENDPOINT 127.0.0.1 {cape_env['port']}\n")
    sandbox = await task
    try:
        assert sandbox.status == "running"
        assert len(_log_entries(cape_env, verb="status")) >= 2
    finally:
        await provider.delete(sandbox.sandbox_id)


async def test_discovery_fails_after_consecutive_transient_failures(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CAPE_EXEC_WORKLOAD", "0")
    monkeypatch.setenv("FAKE_CAPE_STATUS_FAILURES", "100000")
    provider = _provider(cape_env, transient_failure_limit=3)
    with pytest.raises(RuntimeError, match="3 consecutive"):
        await provider.create(_sandbox_config())
    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels and _flag(cancels[-1]["argv"], "--reason") == "agentix_create_failed"
    assert provider._sandboxes == {}
    assert list(Path(cape_env["meta_root"]).iterdir()) == []


async def test_discovery_times_out_when_marker_never_appears(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CAPE_EXEC_WORKLOAD", "0")
    provider = _provider(cape_env, create_timeout_seconds=1.0)
    with pytest.raises(TimeoutError, match="did not publish its endpoint"):
        await provider.create(_sandbox_config())
    cancels = _log_entries(cape_env, verb="cancel")
    assert cancels and _flag(cancels[-1]["argv"], "--reason") == "agentix_create_failed"
    # Logs are useless on a still-running request (empty by contract) —
    # the timeout path must not have fetched them.
    assert _log_entries(cape_env, verb="logs") == []
    assert provider._sandboxes == {}
    assert provider._inflight_ports == set()
    assert list(Path(cape_env["meta_root"]).iterdir()) == []


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
    assert list(Path(cape_env["meta_root"]).iterdir()) == []


async def test_marker_written_but_request_terminal_fails_with_logs(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The workload wrote its marker and then crashed (dead endpoint,
    # terminal request): the health wait must fail fast with the
    # request's logs instead of probing for the whole budget.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    monkeypatch.setenv("FAKE_CAPE_PORT", str(dead_port))
    monkeypatch.setenv("FAKE_CAPE_SERVER_STATE", "FAILED")
    provider = _provider(cape_env, create_timeout_seconds=30.0)
    with pytest.raises(RuntimeError, match="terminal state FAILED") as excinfo:
        await provider.create(_sandbox_config())
    assert "workload stderr" in str(excinfo.value)
    assert provider._sandboxes == {}
    assert list(Path(cape_env["meta_root"]).iterdir()) == []


async def test_cancelled_create_cancels_submitted_request(
    cape_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep discovery spinning (marker never appears) so the cancellation
    # lands mid-create.
    monkeypatch.setenv("FAKE_CAPE_EXEC_WORKLOAD", "0")
    provider = _provider(cape_env)
    task = asyncio.ensure_future(provider.create(_sandbox_config()))
    while not _log_entries(cape_env, verb="status"):
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
    assert list(Path(cape_env["meta_root"]).iterdir()) == []


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
    marker = "AGENTIX_ENDPOINT 10.0.0.5 8710\n"
    assert _parse_endpoint(marker) == ("10.0.0.5", 8710)
    # Two-token noise ("GPU 0") must never be misread as an endpoint.
    assert _parse_endpoint("GPU 0\n") is None
    assert _parse_endpoint("") is None
    assert _parse_endpoint("AGENTIX_ENDPOINT host notaport\n") is None
    assert _parse_endpoint("AGENTIX_ENDPOINT 10.0.0.5\n") is None


def test_boot_script_writes_marker_and_creates_workspace(tmp_path: Path) -> None:
    """Run the generated boot script under a real `sh -c` (with a stubbed
    multi-IP `hostname` and a tmp meta dir) and feed the marker file it
    writes to `_parse_endpoint` — the producer/consumer pair must agree.
    """
    stub = tmp_path / "stub-bin"
    stub.mkdir()
    hostname = stub / "hostname"
    hostname.write_text("#!/bin/sh\necho '10.0.0.5 172.17.0.1'\n")
    hostname.chmod(0o755)
    workspace = tmp_path / "ws" / "cape-x"
    meta_dir = tmp_path / "meta" / "cape-x"
    meta_dir.mkdir(parents=True)
    bundle = tmp_path / "bundle"  # deliberately has no runtime/bootstrap.sh
    script = _boot_script(workspace=str(workspace), meta_dir=str(meta_dir), bundle=str(bundle))
    env = {
        "PATH": f"{stub}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "AGENTIX_BIND_PORT": "9999",
    }
    proc = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True)
    # The final exec of the (absent) bundle bootstrap fails — but only
    # after the workspace was created and the marker was written.
    assert proc.returncode != 0
    assert workspace.is_dir()
    marker_file = meta_dir / "endpoint"
    assert marker_file.is_file()
    assert _parse_endpoint(marker_file.read_text()) == ("10.0.0.5", 9999)
    assert proc.stdout == ""  # the marker goes to the meta dir, not stdout


# ── registry contract ─────────────────────────────────────────────────────


def test_zero_arg_constructor_for_plugin_registry() -> None:
    # The plugin registry instantiates providers with `cls()` — every
    # config field must be optional or defaulted. pool/user/meta_root
    # are then checked at first use.
    provider = CapeProvider()
    assert isinstance(provider, SandboxProvider)
    assert provider.config.workspace_root == "/workspace"
    assert provider.config.url_template == "http://{host}:{port}"
    assert provider.config.runtime_port_base == 8710
    assert provider.config.pool is None
    assert provider.config.meta_root is None
