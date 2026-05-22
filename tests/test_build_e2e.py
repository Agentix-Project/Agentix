"""End-to-end test for `agentix build` — builds a real bundle image.

Marked `e2e`: excluded from the default `pytest` run (`addopts` in
pyproject) and from the unit CI job. Run it explicitly with `-m e2e`,
or via the `e2e` CI job.

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
  * a plugin's system closure (`agentix-runtime-basic` → bash) is
    merged into `/nix/runtime`
  * the `agentix-server` entry point is wired
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentix.cli import build as build_cli

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker is required for the bundle build"),
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO_ROOT / "examples" / "hello-bundle"
_IMAGE = "agentix-build-e2e:pytest"
_SELECTED_PLATFORM = "AGENTIX_E2E_PLATFORM"


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


def _sh_platform(image: str, platform: str, script: str) -> str:
    """Run `sh -c <script>` in `image` for a specific Docker platform."""
    proc = subprocess.run(
        ["docker", "run", "--rm", "--platform", platform, "--entrypoint", "sh", image, "-c", script],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"in-image command failed for {platform} ({script!r}):\n{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout


def _docker_buildx_platforms() -> set[str]:
    proc = subprocess.run(
        ["docker", "buildx", "inspect", "--bootstrap"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return set()

    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if line.startswith("Platforms:"):
            return {item.strip() for item in line.partition(":")[2].split(",") if item.strip()}
    return set()


def _supports_platform(supported: set[str], platform: str) -> bool:
    return platform in supported or any(item.startswith(f"{platform}/") for item in supported)


@pytest.fixture(scope="module")
def bundle() -> Iterator[str]:
    """Build `examples/hello-bundle` into a bundle image once."""
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
        "'import hello_bundle, agentix, agentix.bash; print(hello_bundle.run())'",
    )
    assert "hello, world" in out


def test_plugin_closure_merged(bundle: str) -> None:
    """`agentix-runtime-basic` ships a bash system closure; it must be
    merged into `/nix/runtime` via the discovered `agentix.nix` file."""
    assert "ok" in _sh(bundle, "test -x /nix/runtime/bin/bash && echo ok")


def test_entrypoint_wired(bundle: str) -> None:
    assert "agentix-server" in _sh(bundle, "/nix/runtime/venv/bin/agentix-server --help")


@pytest.mark.parametrize(
    ("platform", "machine"),
    [
        ("linux/amd64", "x86_64"),
        ("linux/arm64", "aarch64"),
    ],
)
def test_platform_bundle_builds_and_runs(platform: str, machine: str) -> None:
    selected = os.environ.get(_SELECTED_PLATFORM)
    if selected and selected != platform:
        pytest.skip(f"{_SELECTED_PLATFORM}={selected} selects a different platform")

    supported = _docker_buildx_platforms()
    if not _supports_platform(supported, platform):
        msg = f"Docker buildx does not report support for {platform}: {sorted(supported)}"
        if selected:
            pytest.fail(msg)
        pytest.skip(msg)

    arch = platform.rsplit("/", 1)[1]
    image = f"agentix-build-e2e-{arch}:pytest"
    name = image.split(":", 1)[0]

    try:
        assert build_cli.main([str(_EXAMPLE), "--name", image, "--platform", platform]) == 0
        out = _sh_platform(
            image,
            platform,
            "set -eu; "
            "uname -m; "
            "test -d /nix/runtime/venv; "
            "test -x /nix/runtime/bin/bash; "
            "readlink -f /nix/runtime/venv/bin/python; "
            "/nix/runtime/venv/bin/python --version; "
            "/nix/runtime/venv/bin/python -c "
            "'import agentix, hello_bundle, agentix.bash; print(hello_bundle.run())'; "
            "/nix/runtime/venv/bin/agentix-server --help",
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        assert lines[0] == machine
        assert any(line.startswith("/nix/store/") for line in lines)
        assert "Python 3.11" in out
        assert "hello, world" in out
        assert "agentix-server" in out
    finally:
        subprocess.run(["docker", "rmi", "-f", image, f"{name}:latest"], capture_output=True)
