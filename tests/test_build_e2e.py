"""End-to-end test for `agentix build` — builds a real bundle image.

Marked `e2e`: excluded from the default `pytest` run (`addopts` in
pyproject) and from the unit CI job. Run it explicitly with `-m e2e`.

Needs Docker only — Nix runs *inside* the build container, so the host
(or CI runner) needs no Nix. The build takes minutes, so one
module-scoped fixture builds the bundle once and the assertions
inspect the resulting image.

What it proves end to end:
  * `agentix build` stages the repo + drives `docker buildx build`
  * the in-container pipeline runs (toolchain → uv venv/sync →
    closure discovery → runtime)
  * the interpreter is Nix-provided (`/nix/store`), not a stray host
    Python — the property that makes the bundle libc-hermetic
  * the project's remote target imports and runs
  * plugin and project system closures (`bash` and `ripgrep`) are
    merged into `/nix/runtime`
  * the `agentix-server` entry point is wired
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
_IMAGE = "agentix-build-e2e:pytest"


def _sh(image: str, script: str) -> str:
    """Run `sh -c <script>` in `image`; return stdout, fail on non-zero."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", script],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"in-image command failed ({script!r}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


@pytest.fixture(scope="module")
def bundle() -> Iterator[str]:
    """Build `examples/hello-world` into a bundle image once."""
    build = subprocess.run(
        [sys.executable, "-m", "agentix.cli", "build", str(_EXAMPLE), "--name", _IMAGE],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise AssertionError(f"`agentix build` failed:\n{build.stdout}\n{build.stderr}")
    yield _IMAGE
    name = _IMAGE.split(":", 1)[0]
    subprocess.run(["docker", "rmi", "-f", _IMAGE, f"{name}:latest"], capture_output=True)


def test_image_built(bundle: str) -> None:
    ids = subprocess.run(
        ["docker", "images", "-q", bundle],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert ids, f"{bundle} not present in the local daemon"


def test_runtime_layout(bundle: str) -> None:
    entries = set(_sh(bundle, "ls /nix/runtime").split())
    assert {"venv", "bin"} <= entries, entries


def test_venv_python_is_nix_provided(bundle: str) -> None:
    """The venv's interpreter must resolve into `/nix/store` — that is
    what makes the bundle hermetic against the task image's libc."""
    real = _sh(bundle, "readlink -f /nix/runtime/venv/bin/python").strip()
    assert real.startswith("/nix/store/"), real
    version = _sh(bundle, "/nix/runtime/venv/bin/python --version").strip()
    assert version.startswith("Python 3.11"), version


def test_remote_target_importable(bundle: str) -> None:
    """The project module + the framework + a plugin all import, and
    the remote callable runs — the venv is a real, working closure."""
    out = _sh(
        bundle,
        "/nix/runtime/venv/bin/python -c "
        "'import main, agentix, agentix.bash; print(main.hello())'",
    )
    assert "ripgrep" in out


def test_system_closures_merged(bundle: str) -> None:
    """Plugin and project system closures must be merged into `/nix/runtime`."""
    assert "ok" in _sh(bundle, "test -x /nix/runtime/bin/bash && echo ok")
    assert "ripgrep" in _sh(bundle, "/nix/runtime/bin/rg --version")


def test_entrypoint_wired(bundle: str) -> None:
    assert "agentix-server" in _sh(bundle, "/nix/runtime/venv/bin/agentix-server --help")
