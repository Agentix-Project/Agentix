"""Unit tests for `ApptainerProvider` using a fake `apptainer` binary.

The fake binary is a small shell script the test stages on PATH at the
start of each test. It records every call to a file and acts as both
the `apptainer pull` (creates an empty `.sif`) and `apptainer exec`
(starts a stand-in HTTP server with a `/health` endpoint on
`AGENTIX_BIND_PORT`).

The point of these tests is to lock in the CLI surface the provider
uses (flags, order, env propagation), not to exercise a real container
runtime.
"""

from __future__ import annotations

import io
import json
import os
import stat
import tarfile
from pathlib import Path

import httpx
import pytest
from agentix.provider.apptainer import (
    ApptainerProvider,
    _bundle_digest,
    _extract_bundle,
)

from agentix.provider.base import SandboxConfig

# ── helpers ───────────────────────────────────────────────────────────────


def _write_bundle_tar(tmp_path: Path, *, digest: str = "sha256:abc") -> Path:
    """Build a minimal portable bundle tar containing manifest + nix/."""
    bundle_root = tmp_path / "bundle"
    nix_root = bundle_root / "nix"
    runtime = nix_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "bootstrap.sh").write_text("#!/bin/sh\necho fake\n")
    manifest = {"digest": digest, "name": "test-bundle", "tag": "0.0.1"}
    (bundle_root / "manifest.json").write_text(json.dumps(manifest))
    tar_path = tmp_path / "bundle.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(bundle_root / "manifest.json", arcname="manifest.json")
        tar.add(nix_root, arcname="nix")
    return tar_path


_FAKE_APPTAINER_SOURCE = '''#!{python}
"""Recording shim for `apptainer` used by ApptainerProvider tests.

Logs every invocation as one JSON line to `LOG_PATH`. On `exec`,
launches a tiny HTTP server that responds 200 on `/health` so the
provider health probe succeeds.
"""
import http.server
import json
import os
import sys

LOG_PATH = {log_path!r}


def _log(argv):
    rec = {{
        "argv": argv,
        "env": {{
            k: v
            for k, v in os.environ.items()
            if k.startswith("AGENTIX") or k.startswith("APPTAINERENV")
        }},
    }}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\\n")


def _pull(argv):
    sif = None
    it = iter(argv)
    for tok in it:
        if tok == "--force":
            continue
        if sif is None:
            sif = tok
        else:
            # apptainer pull <sif> <ref>
            break
    if sif is None:
        sys.stderr.write("fake apptainer: missing sif arg\\n")
        sys.exit(2)
    os.makedirs(os.path.dirname(sif) or ".", exist_ok=True)
    open(sif, "wb").close()


def _exec(argv):
    port = None
    for tok in argv:
        if tok.startswith("AGENTIX_BIND_PORT="):
            port = int(tok.split("=", 1)[1])
            break
    if port is None:
        sys.stderr.write("fake apptainer: no AGENTIX_BIND_PORT in args\\n")
        sys.exit(2)

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = b'{{"status":"ok"}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a, **kw):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), H)
    server.serve_forever()


def main():
    argv = sys.argv[1:]
    _log(argv)
    if not argv:
        sys.exit(0)
    sub = argv[0]
    rest = argv[1:]
    if sub == "pull":
        _pull(rest)
    elif sub == "exec":
        _exec(rest)
    elif sub == "--version":
        print("fake apptainer 0.0")
    else:
        sys.stderr.write(f"fake apptainer: unsupported subcommand {{sub}}\\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
'''


def _install_fake_apptainer(tmp_path: Path) -> tuple[Path, Path]:
    """Drop a Python recording shim named `apptainer` on disk.

    Returns `(fake_bin_path, log_path)`. The shim writes every
    invocation as one JSON line to `log_path` and, on `exec`, runs a
    tiny HTTP server on the requested port so the provider's
    `/health` poll succeeds.
    """
    fake_dir = tmp_path / "fake-bin"
    fake_dir.mkdir(exist_ok=True)
    log_path = tmp_path / "apptainer.log.jsonl"
    script = fake_dir / "apptainer"
    script.write_text(_FAKE_APPTAINER_SOURCE.format(python=os.sys.executable, log_path=str(log_path)))
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script, log_path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("AGENTIX_APPTAINER_CACHE", str(cache))
    fake_bin, log_path = _install_fake_apptainer(tmp_path)
    monkeypatch.setenv("AGENTIX_APPTAINER_BIN", str(fake_bin))
    return {"tmp_path": tmp_path, "cache": cache, "log": log_path}


# ── tests ─────────────────────────────────────────────────────────────────


def test_bundle_digest_uses_manifest_when_present(tmp_path: Path) -> None:
    bundle = _write_bundle_tar(tmp_path, digest="sha256:deadbeef")
    assert _bundle_digest(bundle) == "sha256_deadbeef"


def test_bundle_digest_falls_back_to_file_hash(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "nix").mkdir()
    (bundle_root / "nix" / "runtime").mkdir()
    tar_path = tmp_path / "bundle.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(bundle_root / "nix", arcname="nix")
    digest = _bundle_digest(tar_path)
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_extract_bundle_skips_when_runtime_already_present(tmp_path: Path) -> None:
    bundle = _write_bundle_tar(tmp_path)
    target = tmp_path / "out"
    nix = _extract_bundle(bundle, target)
    assert (nix / "runtime" / "bootstrap.sh").exists()

    # Mutate the file so the second extract would overwrite it if it ran.
    sentinel = nix / "runtime" / "bootstrap.sh"
    sentinel.write_text("MUTATED")
    _extract_bundle(bundle, target)
    assert sentinel.read_text() == "MUTATED"


def test_extract_bundle_rejects_path_traversal_member(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.tar"
    with tarfile.open(bundle, "w") as tar:
        data = b"owned"
        info = tarfile.TarInfo("nix/../../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    target = tmp_path / "cache" / "bundle"
    with pytest.raises(RuntimeError, match="unsafe member"):
        _extract_bundle(bundle, target)
    assert not (tmp_path / "cache" / "escape.txt").exists()


@pytest.mark.anyio
async def test_create_and_delete_roundtrip(env, tmp_path: Path) -> None:
    bundle = _write_bundle_tar(tmp_path)
    config = SandboxConfig(image="docker://python:3.13-slim", bundle=str(bundle))
    provider = ApptainerProvider()

    async with provider.session(config) as sandbox:
        info = await provider.get(sandbox.sandbox_id)
        assert info.status == "running"
        async with httpx.AsyncClient(base_url=sandbox.runtime_url, timeout=5) as c:
            r = await c.get("/health")
            assert r.status_code == 200

    # After session exit, the process is gone and a follow-up get reflects that.
    with pytest.raises(KeyError):
        await provider.get(sandbox.sandbox_id)


@pytest.mark.anyio
async def test_create_records_expected_apptainer_cli(env, tmp_path: Path) -> None:
    bundle = _write_bundle_tar(tmp_path)
    config = SandboxConfig(
        image="docker://python:3.13-slim",
        bundle=str(bundle),
        env={"HF_HOME": "/tmp/hf"},
    )
    provider = ApptainerProvider()
    async with provider.session(config) as _:
        pass

    log_lines = [
        json.loads(line) for line in env["log"].read_text().splitlines() if line.strip()
    ]
    pulls = [r for r in log_lines if r["argv"] and r["argv"][0] == "pull"]
    execs = [r for r in log_lines if r["argv"] and r["argv"][0] == "exec"]
    assert len(pulls) == 1, log_lines
    assert len(execs) == 1, log_lines

    exec_argv = execs[0]["argv"]
    # Default isolation: --userns + --cleanenv (works in
    # capability-restricted hosts; --cleanenv strips host LD_PRELOAD
    # / GPU-runtime noise that the task image doesn't have libs for).
    # --containall is opt-in via AGENTIX_APPTAINER_FLAGS.
    assert "--userns" in exec_argv
    assert "--no-init" in exec_argv
    assert "--writable-tmpfs" in exec_argv
    assert "--cleanenv" in exec_argv
    assert "--bind" in exec_argv
    bind_target = exec_argv[exec_argv.index("--bind") + 1]
    assert bind_target.endswith(":/nix:ro")
    # AGENTIX_BIND_PORT must be set; HF_HOME pass-through preserved.
    flat = " ".join(exec_argv)
    assert "AGENTIX_BIND_PORT=" in flat
    assert "HF_HOME=/tmp/hf" in flat
    # Container entry point: the bundle's /nix/runtime/bootstrap.sh, not
    # an inline `sh -c '...'` indirection.
    assert exec_argv[-1] == "/nix/runtime/bootstrap.sh"
    assert "-c" not in exec_argv


@pytest.mark.anyio
async def test_apptainer_flags_env_override(env, tmp_path: Path, monkeypatch) -> None:
    """`AGENTIX_APPTAINER_FLAGS` overrides the default isolation flags."""
    monkeypatch.setenv(
        "AGENTIX_APPTAINER_FLAGS",
        "--containall --no-init --writable-tmpfs",
    )
    bundle = _write_bundle_tar(tmp_path)
    config = SandboxConfig(image="docker://python:3.13-slim", bundle=str(bundle))
    provider = ApptainerProvider()
    async with provider.session(config) as _:
        pass

    log_lines = [
        json.loads(line) for line in env["log"].read_text().splitlines() if line.strip()
    ]
    execs = [r for r in log_lines if r["argv"] and r["argv"][0] == "exec"]
    exec_argv = execs[0]["argv"]
    assert "--containall" in exec_argv
    assert "--userns" not in exec_argv


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
