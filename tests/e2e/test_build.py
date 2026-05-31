"""End-to-end test for `agentix build` + `agentix deploy docker`.

Marked `e2e`: excluded from the default `pytest` run (`addopts` in
pyproject) and from the unit CI job. Run it explicitly with `-m e2e`.

Needs Docker only — Nix runs *inside* the build container, so the host
(or CI runner) needs no Nix. The build takes minutes, so one
module-scoped fixture builds the bundle once and the assertions
inspect the resulting deployed cache through a task container.

What it proves end to end:
  * `agentix build` stages the repo + drives `docker buildx build`
  * the in-container pipeline runs (toolchain → uv venv/sync →
    closure discovery → runtime)
  * `agentix deploy docker` deploys the portable tar as a local
    cache directory that can be bind-mounted into a task container
  * the interpreter is Nix-provided (`/nix/store`), not a stray host
    Python — the property that makes the bundle libc-hermetic
  * the project's remote target imports and runs
  * plugin and project system closures (`bash` and `ripgrep`) are
    merged into `/nix/runtime`
  * the bundle's `/nix/runtime/bootstrap.sh` entry point is wired and
    the runtime ASGI app it ultimately launches can be imported
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker is required for the bundle build"),
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO_ROOT / "examples" / "hello-world"
_BUNDLE_NAME = "agentix-build-e2e:pytest"
_TASK_IMAGE = "python:3.13-slim"
_PLATFORM = "linux/amd64"


def _sh(bundle: Path, script: str) -> str:
    """Run `sh -c <script>` with the bundle mounted at /nix."""
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--mount",
            f"type=bind,source={(bundle / 'nix').resolve()},target=/nix,readonly",
            "--entrypoint",
            "sh",
            _TASK_IMAGE,
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"in-image command failed ({script!r}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


@pytest.fixture(scope="module")
def bundle() -> Iterator[Path]:
    """Build `examples/hello-world` into a tar, then deploy it into the Docker cache."""
    tar_path = _REPO_ROOT / "dist" / "agentix-build-e2e.bundle.tar"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentix.cli",
            "build",
            str(_EXAMPLE),
            "--name",
            _BUNDLE_NAME,
            "--platform",
            _PLATFORM,
            "--output",
            str(tar_path),
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise AssertionError(f"`agentix build` failed:\n{build.stdout}\n{build.stderr}")
    deploy = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentix.cli",
            "deploy",
            "docker",
            str(tar_path),
            "--platform",
            _PLATFORM,
        ],
        capture_output=True,
        text=True,
    )
    if deploy.returncode != 0:
        raise AssertionError(f"`agentix deploy docker` failed:\n{deploy.stdout}\n{deploy.stderr}")
    bundle_path = _bundle_path_from_deploy_output(deploy.stdout)
    yield bundle_path
    shutil.rmtree(bundle_path, ignore_errors=True)
    tar_path.unlink(missing_ok=True)


def _bundle_path_from_deploy_output(output: str) -> Path:
    for line in output.splitlines():
        if line.startswith("bundle -> "):
            return Path(line.removeprefix("bundle -> ")).expanduser()
    raise AssertionError(f"`agentix deploy` did not print a bundle ref:\n{output}")


def test_bundle_deployed(bundle: Path) -> None:
    assert (bundle / "nix" / "runtime" / "bootstrap.sh").is_file()


def test_runtime_layout(bundle: Path) -> None:
    entries = set(_sh(bundle, "ls /nix/runtime").split())
    assert {"venv", "bin"} <= entries, entries


def test_venv_python_is_nix_provided(bundle: Path) -> None:
    """The venv's interpreter must resolve into `/nix/store` — that is
    what makes the bundle hermetic against the task image's libc."""
    real = _sh(bundle, "readlink -f /nix/runtime/venv/bin/python").strip()
    assert real.startswith("/nix/store/"), real
    version = _sh(bundle, "/nix/runtime/venv/bin/python --version").strip()
    assert version.startswith("Python 3.11"), version


def test_remote_target_importable(bundle: Path) -> None:
    """The project module + the framework + a plugin all import, and
    the remote callable runs — the venv is a real, working closure."""
    out = _sh(
        bundle,
        "/nix/runtime/venv/bin/python -c "
        "'import main, agentix, agentix.bash; print(main.hello())'",
    )
    assert "ripgrep" in out


def test_system_closures_merged(bundle: Path) -> None:
    """Plugin and project system closures must be merged into `/nix/runtime`."""
    assert "ok" in _sh(bundle, "test -x /nix/runtime/bin/bash && echo ok")
    assert "ripgrep" in _sh(bundle, "/nix/runtime/bin/rg --version")


def test_entrypoint_wired(bundle: Path) -> None:
    # Bundle entry point: /nix/runtime/bootstrap.sh, exec'd by every
    # provider backend. Verify the script exists + is executable, and
    # that the ASGI app it ultimately launches can be imported from the
    # bundle venv (catches dep-resolution bugs without actually starting
    # uvicorn, which would block forever).
    assert "ok" in _sh(bundle, "test -x /nix/runtime/bootstrap.sh && echo ok")
    assert "ok" in _sh(
        bundle,
        "/nix/runtime/venv/bin/python -c "
        "'from agentix.runtime.server.app import app; print(\"ok\")'",
    )
