"""Tests for `agentix build` — the host-side bundle pipeline.

Covered: pyproject metadata extraction, image name/tag parsing, git
context-root resolution, build-context staging, `--dry-run`, error
paths, and the in-container closure discovery (`agentix.cli.build.closures`).

NOT covered here: the actual `docker buildx build` + `nix build`
execution — that needs Docker + Nix and is exercised by a real
`agentix build` run, not the unit suite. Everything testable without a
container is tested.
"""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import pytest

from agentix.cli import build
from agentix.cli.build import bundle, context, docker
from agentix.cli.build.bundle import (
    _bundle_manifest,
    _validate_bundle_tree,
    _write_bundle_tar,
)
from agentix.cli.build.closures import (
    Closure,
    assemble,
    discover_plugin_closures,
    discover_project_closure,
    stage_closures,
)
from agentix.cli.build.naming import _default_tar_name, _tar_cache_image_ref
from agentix.cli.build.pyproject import (
    derive_tag,
    detect_python_version,
    project_nix,
    read_pyproject,
    short_name,
)

# ── fixtures / helpers ─────────────────────────────────────────────


def _pyproject(
    *,
    name: str = "demo-agent",
    version: str = "0.1.0",
    requires_python: str = ">=3.11",
    nix: str | None = None,
    deps: list[str] | None = None,
) -> str:
    lines = [
        "[project]",
        f'name = "{name}"',
        f'version = "{version}"',
        f'requires-python = "{requires_python}"',
    ]
    deps = deps if deps is not None else ["agentixx"]
    dep_lines = ", ".join(f'"{d}"' for d in deps)
    lines.append(f"dependencies = [{dep_lines}]")
    if nix is not None:
        lines += ["", "[tool.agentix]", f'nix = "{nix}"']
    return "\n".join(lines) + "\n"


def _make_project(
    root: Path,
    *,
    with_lock: bool = True,
    **pyproject_kw: object,
) -> Path:
    """Write a minimal project at `root`; return `root`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(_pyproject(**pyproject_kw))  # type: ignore[arg-type]
    if with_lock:
        (root / "uv.lock").write_text("version = 1\n")
    (root / "agent.py").write_text("def run():\n    return 'ok'\n")
    return root


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


# ── _resolve: detect_python_version ────────────────────────────────


class TestDetectPythonVersion:
    @pytest.mark.parametrize(
        ("requires", "expected"),
        [
            (">=3.11", "311"),
            (">=3.12", "312"),
            (">=3.13", "313"),
            (">=3.10", "310"),
            (">=3.13,<3.14", "313"),
            ("==3.12.*", "312"),
            ("~=3.11", "311"),
        ],
    )
    def test_parsed_from_requires_python(self, requires: str, expected: str) -> None:
        assert detect_python_version({"project": {"requires-python": requires}}) == expected

    def test_missing_requires_python_defaults_to_311(self) -> None:
        assert detect_python_version({"project": {}}) == "311"
        assert detect_python_version({}) == "311"

    def test_unsupported_minor_falls_back_to_default(self) -> None:
        # 3.9 is below the supported range; 3.99 above it.
        assert detect_python_version({"project": {"requires-python": ">=3.9"}}) == "311"
        assert detect_python_version({"project": {"requires-python": ">=3.99"}}) == "311"

    def test_non_string_requires_python(self) -> None:
        assert detect_python_version({"project": {"requires-python": None}}) == "311"


# ── _resolve: project_nix ──────────────────────────────────────────


class TestProjectNix:
    def test_declared(self) -> None:
        pp = {"tool": {"agentix": {"nix": "system.nix"}}}
        assert project_nix(pp) == "system.nix"

    def test_absent_returns_none(self) -> None:
        assert project_nix({}) is None
        assert project_nix({"tool": {}}) is None
        assert project_nix({"tool": {"agentix": {}}}) is None

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(SystemExit):
            project_nix({"tool": {"agentix": {"nix": ""}}})

    def test_non_string_rejected(self) -> None:
        with pytest.raises(SystemExit):
            project_nix({"tool": {"agentix": {"nix": 123}}})


# ── _resolve: read_pyproject / short_name / derive_tag ─────────────


class TestPyprojectMetadata:
    def test_read_pyproject(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        pp = read_pyproject(tmp_path)
        assert pp["project"]["name"] == "demo-agent"

    def test_read_pyproject_missing(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            read_pyproject(tmp_path)

    def test_short_name_strips_agentix_prefix(self) -> None:
        assert short_name({"project": {"name": "agentix-bridge"}}) == "bridge"
        assert short_name({"project": {"name": "demo-agent"}}) == "demo-agent"

    def test_short_name_requires_name(self) -> None:
        with pytest.raises(SystemExit):
            short_name({"project": {}})

    def test_derive_tag(self) -> None:
        assert derive_tag({"project": {"name": "demo-agent", "version": "1.2.3"}}) == "demo-agent:1.2.3"

    def test_derive_tag_requires_version(self) -> None:
        with pytest.raises(SystemExit):
            derive_tag({"project": {"name": "demo-agent"}})


# ── build: parse_name ──────────────────────────────────────────────


class TestParseName:
    def test_none_derives_from_pyproject(self) -> None:
        pp = {"project": {"name": "agentix-demo", "version": "2.0.0"}}
        assert build.parse_name(None, pp) == ("demo", "2.0.0")

    def test_bare_name_keeps_pyproject_version(self) -> None:
        pp = {"project": {"name": "demo", "version": "2.0.0"}}
        assert build.parse_name("custom", pp) == ("custom", "2.0.0")

    def test_name_colon_tag_used_verbatim(self) -> None:
        pp = {"project": {"name": "demo", "version": "2.0.0"}}
        assert build.parse_name("custom:dev", pp) == ("custom", "dev")

    def test_missing_version_defaults_latest(self) -> None:
        pp = {"project": {"name": "demo"}}
        assert build.parse_name(None, pp) == ("demo", "latest")

    @pytest.mark.parametrize("bad", ["name:", ":tag", ":"])
    def test_malformed_name_tag_rejected(self, bad: str) -> None:
        with pytest.raises(SystemExit):
            build.parse_name(bad, {"project": {"name": "demo", "version": "1.0"}})


# ── build: platform resolution ─────────────────────────────────────


class TestPlatform:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("linux/amd64", "linux/amd64"),
            ("amd64", "linux/amd64"),
            ("x86_64", "linux/amd64"),
            ("linux/x86_64", "linux/amd64"),
            ("linux/arm64", "linux/arm64"),
            ("linux/arm64/v8", "linux/arm64"),
            ("arm64", "linux/arm64"),
            ("aarch64", "linux/arm64"),
            ("linux/aarch64", "linux/arm64"),
        ],
    )
    def test_normalize_platform(self, value: str, expected: str) -> None:
        assert build.normalize_platform(value) == expected

    def test_normalize_platform_rejects_unknown(self) -> None:
        with pytest.raises(SystemExit):
            build.normalize_platform("darwin/arm64")

    @pytest.mark.parametrize(
        ("machine", "expected"),
        [
            ("x86_64", "linux/amd64"),
            ("amd64", "linux/amd64"),
            ("arm64", "linux/arm64"),
            ("aarch64", "linux/arm64"),
        ],
    )
    def test_detect_default_platform(self, machine: str, expected: str) -> None:
        assert build.detect_default_platform(machine) == expected

    def test_detect_default_platform_rejects_unknown(self) -> None:
        with pytest.raises(SystemExit):
            build.detect_default_platform("sparc")

    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            ("linux/amd64", "x86_64-linux"),
            ("linux/arm64", "aarch64-linux"),
        ],
    )
    def test_nix_system_for_platform(self, platform: str, expected: str) -> None:
        assert build.nix_system_for_platform(platform) == expected


# ── build: git context resolution ──────────────────────────────────


class TestResolveContext:
    def test_project_at_repo_root(self, tmp_path: Path) -> None:
        repo = _make_project(tmp_path / "repo")
        _git_init(repo)
        root, subpath = build.resolve_context(repo)
        assert root == repo.resolve()
        assert subpath == Path(".")

    def test_project_in_repo_subdir(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        proj = _make_project(repo / "examples" / "demo")
        root, subpath = build.resolve_context(proj)
        assert root == repo.resolve()
        assert subpath == Path("examples/demo")

    def test_project_not_in_git_is_its_own_context(self, tmp_path: Path) -> None:
        # tmp_path lives under /tmp — not a git repo.
        proj = _make_project(tmp_path / "standalone")
        root, subpath = build.resolve_context(proj)
        assert root == proj.resolve()
        assert subpath == Path(".")

    def test_git_toplevel_outside_repo_is_none(self, tmp_path: Path) -> None:
        assert context.git_toplevel(tmp_path) is None


# ── build: stage_context ───────────────────────────────────────────


class TestStageContext:
    def _staged(self, tmp_path: Path, python_version: str = "311", platform: str = "linux/amd64") -> Path:
        repo = _make_project(tmp_path / "repo")
        _git_init(repo)  # creates .git/ — must be skip-listed
        (repo / ".venv").mkdir()
        (repo / ".venv" / "junk").write_text("x")
        (repo / "__pycache__").mkdir()
        (repo / "__pycache__" / "c.pyc").write_text("x")
        stage = tmp_path / "stage"
        build.stage_context(stage, context_root=repo, python_version=python_version, platform=platform)
        return stage

    def test_repo_copied(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path)
        assert (stage / "repo" / "pyproject.toml").is_file()
        assert (stage / "repo" / "agent.py").is_file()

    def test_skip_list_applied(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path)
        assert not (stage / "repo" / ".git").exists()
        assert not (stage / "repo" / ".venv").exists()
        assert not (stage / "repo" / "__pycache__").exists()

    def test_nested_build_dir_not_skipped(self, tmp_path: Path) -> None:
        """The repo-root `build/` dir is a `python -m build` / dry-run
        output that's correctly skipped — but a *nested* `build/` like
        `agentix/cli/build/` is a real package that must survive into
        the build context. `shutil.ignore_patterns` would strip both;
        the staging copy uses a path-aware ignore instead.
        """
        repo = _make_project(tmp_path / "repo")
        _git_init(repo)
        (repo / "build").mkdir()
        (repo / "build" / "stale_output.txt").write_text("dry-run leftovers")
        nested = repo / "agentix" / "cli" / "build"
        nested.mkdir(parents=True)
        (nested / "__init__.py").write_text("# the click command lives here\n")
        (nested / "context.py").write_text("# real source\n")

        stage = tmp_path / "stage"
        build.stage_context(stage, context_root=repo, python_version="311", platform="linux/amd64")

        # Top-level `build/` skipped — same as before.
        assert not (stage / "repo" / "build").exists()
        # Nested `build/` package preserved — the bug we're regressing
        # against would have stripped it, leaving the in-container
        # `uv sync` unable to install `agentix.cli.build.closures`.
        assert (stage / "repo" / "agentix" / "cli" / "build" / "__init__.py").is_file()
        assert (stage / "repo" / "agentix" / "cli" / "build" / "context.py").is_file()

    def test_builder_files_staged(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path)
        for name in ("flake.nix", "flake.lock", "Dockerfile", "bundle-build.sh", "bootstrap.sh"):
            assert (stage / name).is_file(), f"{name} missing from stage"

    def test_python_version_file(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path, python_version="312")
        assert (stage / "python-version").read_text() == "312\n"

    def test_nix_system_file(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path, platform="linux/arm64")
        assert (stage / "nix-system").read_text() == "aarch64-linux\n"

    def test_closures_dir_created(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path)
        assert (stage / "closures").is_dir()

    def test_flake_nix_is_valid_shape(self, tmp_path: Path) -> None:
        # Staged flake.nix must be the verbatim shipped builder, with
        # the toolchain + runtime outputs the in-container script uses.
        stage = self._staged(tmp_path)
        flake = (stage / "flake.nix").read_text()
        assert "toolchain" in flake
        assert "runtime" in flake
        assert "builtins.readFile ./nix-system" in flake
        # Python is uv's job — no uv2nix machinery in the flake.
        assert "uv2nix.lib" not in flake
        assert "mkPyprojectOverlay" not in flake


# ── build: docker command ──────────────────────────────────────────


class TestDockerBuild:
    def test_docker_build_image_passes_platform(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def _fake_run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
            del cwd, capture
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(docker, "_run", _fake_run)

        docker._docker_build_image(
            tmp_path,
            tags=["demo:1.0.0"],
            project_subpath=Path("."),
            platform="linux/amd64",
        )

        cmd = calls[0]
        assert cmd[:4] == ["docker", "buildx", "build", "--platform"]
        assert cmd[4] == "linux/amd64"
        assert "--load" in cmd
        assert "-t" in cmd

    def test_podman_build_uses_plain_build_and_extra_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        def _fake_run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
            del cwd, capture
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setenv("https_proxy", "http://proxy.example:3128")
        monkeypatch.setattr(docker, "_run", _fake_run)

        docker._docker_build_image(
            tmp_path,
            tags=["demo:1.0.0"],
            project_subpath=Path("."),
            platform="linux/amd64",
            config=docker.ContainerBuildConfig(
                container_bin="podman",
                container_args=("--isolation=chroot",),
                nix_args=("--cores=4",),
                uv_args=("--no-build-isolation",),
            ),
        )

        cmd = calls[0]
        assert cmd[:4] == ["podman", "build", "--platform", "linux/amd64"]
        assert "--isolation=chroot" in cmd
        build_arg_pairs = list(zip(cmd, cmd[1:], strict=False))
        assert ("--build-arg", "https_proxy=http://proxy.example:3128") in build_arg_pairs
        assert ("--build-arg", "AGENTIX_NIX_ARGS=--cores=4") in build_arg_pairs
        assert ("--build-arg", "AGENTIX_UV_ARGS=--no-build-isolation") in build_arg_pairs
        assert "buildx" not in cmd
        assert "--load" not in cmd


# ── build: tar bundle artifacts ────────────────────────────────────


class TestTarBundle:
    def test_default_tar_name_includes_platform(self) -> None:
        assert _default_tar_name("ghcr.io/acme/demo", "1.0.0", "linux/amd64") == (
            "ghcr.io-acme-demo-1.0.0-linux-amd64.bundle.tar"
        )

    def test_tar_cache_image_ref_is_stable_across_output_tags(self) -> None:
        ref = _tar_cache_image_ref(
            name="ghcr.io/acme/Demo Agent",
            project_subpath=Path("examples/demo"),
            platform="linux/amd64",
        )

        assert ref.startswith("agentix-bundle-cache:ghcr.io-acme-demo-agent-linux-amd64-")
        assert ref == _tar_cache_image_ref(
            name="ghcr.io/acme/Demo Agent",
            project_subpath=Path("examples/demo"),
            platform="amd64",
        )

    def test_output_directory_gets_default_name(self, tmp_path: Path) -> None:
        out = build._tar_output_path(str(tmp_path / "dist"), name="demo", tag="1.0.0", platform="linux/arm64")
        assert out == tmp_path / "dist" / "demo-1.0.0-linux-arm64.bundle.tar"

    def test_output_file_used_verbatim(self, tmp_path: Path) -> None:
        out = build._tar_output_path(str(tmp_path / "bundle.tar"), name="demo", tag="1.0.0", platform="linux/arm64")
        assert out == tmp_path / "bundle.tar"

    def test_write_bundle_tar_preserves_absolute_symlink(self, tmp_path: Path) -> None:
        bundle_root = tmp_path / "bundle"
        store = bundle_root / "nix" / "store" / "hash-python" / "bin"
        runtime_bin = bundle_root / "nix" / "runtime" / "bin"
        store.mkdir(parents=True)
        runtime_bin.mkdir(parents=True)
        (store / "python").write_text("#!/bin/sh\n")
        (runtime_bin / "python").symlink_to("/nix/store/hash-python/bin/python")
        (bundle_root / "manifest.json").write_text("{}\n")

        tar_path = tmp_path / "demo.bundle.tar"
        _write_bundle_tar(bundle_root, tar_path)

        with tarfile.open(tar_path) as tar:
            members = {member.name: member for member in tar.getmembers()}
        assert "manifest.json" in members
        link = members["nix/runtime/bin/python"]
        assert link.issym()
        assert link.linkname == "/nix/store/hash-python/bin/python"

    def test_validate_bundle_tree_accepts_absolute_nix_symlink(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        store = nix / "store" / "hash-python" / "bin"
        runtime = nix / "runtime"
        runtime_bin = runtime / "bin"
        store.mkdir(parents=True)
        runtime_bin.mkdir(parents=True)
        (store / "python").write_text("#!/bin/sh\n")
        # Bundle entry point: a regular file installed at /nix/runtime/.
        (runtime / "bootstrap.sh").write_text("#!/bin/sh\nexec python\n")
        # Absolute /nix/store/... symlink the validator must accept.
        (runtime_bin / "python").symlink_to("/nix/store/hash-python/bin/python")

        _validate_bundle_tree(nix)

    def test_validate_bundle_tree_rejects_symlink_outside_nix(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        runtime = nix / "runtime"
        runtime.mkdir(parents=True)
        (runtime / "bootstrap.sh").write_text("#!/bin/sh\n")
        (runtime / "bad").symlink_to("/bin/sh")

        with pytest.raises(SystemExit, match="outside /nix"):
            _validate_bundle_tree(nix)

    def test_validate_bundle_tree_ignores_store_symlinks(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        runtime = nix / "runtime"
        store_dir = nix / "store" / "hash-base-system" / "etc" / "ssl" / "certs"
        runtime.mkdir(parents=True)
        store_dir.mkdir(parents=True)
        (runtime / "bootstrap.sh").write_text("#!/bin/sh\n")
        # Symlinks under /nix/store/ are validator-ignored (they're
        # store-internal). A target outside /nix/ would be fine here.
        (store_dir / "ca-bundle.crt").symlink_to("/nix/var/nix/profiles/default/etc/ssl/certs/ca-bundle.crt")

        _validate_bundle_tree(nix)

    def test_validate_bundle_tree_rejects_broken_absolute_nix_symlink(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        runtime = nix / "runtime"
        runtime_bin = runtime / "bin"
        runtime.mkdir(parents=True)
        runtime_bin.mkdir(parents=True)
        (runtime / "bootstrap.sh").write_text("#!/bin/sh\n")
        # Broken /nix/store/... symlink inside /nix/runtime/ — validator
        # must surface this as a build defect.
        (runtime_bin / "python").symlink_to("/nix/store/missing/bin/python")

        with pytest.raises(SystemExit, match="broken symlink"):
            _validate_bundle_tree(nix)

    def test_bundle_manifest_records_runtime_contract(self) -> None:
        manifest = _bundle_manifest(name="demo", tag="1.0.0", platform="linux/amd64", digest="abc123")
        assert manifest["schema_version"] == 1
        assert manifest["name"] == "demo"
        assert manifest["tag"] == "1.0.0"
        assert manifest["platform"] == "linux/amd64"
        assert manifest["nix_system"] == "x86_64-linux"
        assert manifest["digest"] == "sha256:abc123"
        assert manifest["entrypoint"] == "/nix/runtime/bootstrap.sh"
        assert manifest["runtime_env"]["PATH"] == "/nix/runtime/venv/bin:/nix/runtime/bin"  # type: ignore[index]

    def test_copy_nix_from_image_copies_from_stopped_container_with_platform(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            capture: bool = False,
            check: bool = True,
        ) -> subprocess.CompletedProcess:
            del cwd, capture, check
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        def fake_export(
            container: str,
            bundle_root: Path,
            *,
            config: docker.ContainerBuildConfig,
        ) -> None:
            del config
            calls.append(["export", container])
            (bundle_root / "nix" / "runtime").mkdir(parents=True)
            (bundle_root / "nix" / "runtime" / "bootstrap.sh").write_text("#!/bin/sh\n")

        monkeypatch.setattr(bundle, "_run", fake_run)
        monkeypatch.setattr(bundle, "_export_nix_from_container", fake_export)
        bundle._copy_nix_from_image("tmp:latest", tmp_path, platform="linux/amd64")

        create = calls[0]
        container = create[create.index("--name") + 1]
        assert create[:6] == ["docker", "create", "--platform", "linux/amd64", "--network", "none"]
        assert create[-1] == "tmp:latest"
        assert calls[1] == ["export", container]
        assert calls[2] == ["docker", "rm", "-f", container]
        assert (tmp_path / "nix" / "runtime" / "bootstrap.sh").is_file()

    def test_copy_nix_from_image_uses_configured_container_runner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            capture: bool = False,
            check: bool = True,
        ) -> subprocess.CompletedProcess:
            del cwd, capture, check
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(bundle, "_run", fake_run)
        monkeypatch.setattr(bundle, "_export_nix_from_container", lambda *args, **kwargs: None)
        bundle._copy_nix_from_image(
            "tmp:latest",
            tmp_path,
            platform="linux/amd64",
            config=docker.ContainerBuildConfig(
                container_bin="podman",
                container_run_args=("--runtime=crun", "--cgroups=disabled"),
            ),
        )

        create = calls[0]
        assert create[:6] == ["podman", "create", "--platform", "linux/amd64", "--network", "none"]
        assert "--runtime=crun" in create
        assert "--cgroups=disabled" in create

    def test_copy_nix_from_image_removes_container_on_copy_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(
            cmd: list[str],
            *,
            cwd: Path | None = None,
            capture: bool = False,
            check: bool = True,
        ) -> subprocess.CompletedProcess:
            del cwd, capture, check
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        def fake_export(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise SystemExit(23)

        monkeypatch.setattr(bundle, "_run", fake_run)
        monkeypatch.setattr(bundle, "_export_nix_from_container", fake_export)
        with pytest.raises(SystemExit, match="23"):
            bundle._copy_nix_from_image("tmp:latest", tmp_path, platform="linux/amd64")
        container = calls[0][calls[0].index("--name") + 1]
        assert calls[-1] == ["docker", "rm", "-f", container]

    def test_tar_build_keeps_cache_image(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        built_tags: list[list[str]] = []
        copied_refs: list[str] = []

        def fake_build_image(
            _stage: Path,
            *,
            tags: list[str],
            project_subpath: Path,
            platform: str,
            config: docker.ContainerBuildConfig | None = None,
        ) -> None:
            del project_subpath, platform, config
            built_tags.append(tags)

        monkeypatch.setattr(bundle, "_docker_build_image", fake_build_image)

        def fake_copy_nix(
            _image_ref: str,
            bundle_root: Path,
            *,
            platform: str,
            config: docker.ContainerBuildConfig | None = None,
        ) -> None:
            del platform, config
            copied_refs.append(_image_ref)
            runtime = bundle_root / "nix" / "runtime"
            runtime.mkdir(parents=True)
            (runtime / "bootstrap.sh").write_text("#!/bin/sh\n")

        monkeypatch.setattr(bundle, "_copy_nix_from_image", fake_copy_nix)
        build._build_tar_bundle(
            tmp_path,
            output_path=tmp_path / "bundle.tar",
            name="demo",
            tag="1.0.0",
            project_subpath=Path("."),
            platform="linux/amd64",
        )

        cache_ref = _tar_cache_image_ref(name="demo", project_subpath=Path("."), platform="linux/amd64")
        assert built_tags == [[cache_ref]]
        assert copied_refs == [cache_ref]


# ── build: main / --dry-run ────────────────────────────────────────


class TestMainDryRun:
    def test_dry_run_stages_without_docker(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        proj = _make_project(repo / "examples" / "demo", requires_python=">=3.12")

        out = tmp_path / "out"
        monkeypatch.setattr("agentix.cli.build.REPO_ROOT", out)
        rc = build.main([str(proj), "--dry-run"])
        assert rc == 0

        # project name "demo-agent" → image short name "demo-agent"
        staged = out / "build" / "demo-agent"
        assert (staged / "repo" / "examples" / "demo" / "pyproject.toml").is_file()
        assert (staged / "python-version").read_text() == "312\n"
        assert (staged / "nix-system").read_text()
        assert (staged / "Dockerfile").is_file()

    def test_dry_run_custom_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        proj = _make_project(tmp_path / "proj")
        out = tmp_path / "out"
        monkeypatch.setattr("agentix.cli.build.REPO_ROOT", out)
        rc = build.main([str(proj), "--name", "myimg:dev", "--platform", "linux/arm64", "--dry-run"])
        assert rc == 0
        assert (out / "build" / "myimg").is_dir()
        assert (out / "build" / "myimg" / "nix-system").read_text() == "aarch64-linux\n"

    def test_dry_run_does_not_invoke_docker(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        proj = _make_project(tmp_path / "proj")
        monkeypatch.setattr("agentix.cli.build.REPO_ROOT", tmp_path / "out")

        def _boom(*_a: object, **_k: object) -> None:
            raise AssertionError("docker must not be invoked on --dry-run")

        monkeypatch.setattr("agentix.cli.build._build_tar_bundle", _boom)
        assert build.main([str(proj), "--dry-run"]) == 0


class TestMainBuild:
    def test_build_writes_tar_bundle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        proj = _make_project(tmp_path / "proj")
        calls: list[Path] = []

        def fake_tar_bundle(
            stage: Path,
            *,
            output_path: Path,
            name: str,
            tag: str,
            project_subpath: Path,
            platform: str,
            config: docker.ContainerBuildConfig | None = None,
        ) -> Path:
            del config
            calls.append(output_path)
            assert stage.is_dir()
            assert name == "demo-agent"
            assert tag == "0.1.0"
            assert project_subpath == Path(".")
            assert platform in {"linux/amd64", "linux/arm64"}
            return output_path

        monkeypatch.setattr(build, "_build_tar_bundle", fake_tar_bundle)

        assert build.main([str(proj), "--output", str(tmp_path / "bundle.tar")]) == 0
        assert calls == [tmp_path / "bundle.tar"]

    def test_build_accepts_container_cli_options(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        proj = _make_project(tmp_path / "proj")
        configs: list[docker.ContainerBuildConfig | None] = []

        def fake_tar_bundle(
            stage: Path,
            *,
            output_path: Path,
            name: str,
            tag: str,
            project_subpath: Path,
            platform: str,
            config: docker.ContainerBuildConfig | None = None,
        ) -> Path:
            del stage, output_path, name, tag, project_subpath, platform
            configs.append(config)
            return tmp_path / "bundle.tar"

        monkeypatch.setattr(build, "_build_tar_bundle", fake_tar_bundle)

        assert (
            build.main(
                [
                    str(proj),
                    "--name",
                    "demo:dev",
                    "--backend",
                    "podman",
                    "--container-arg",
                    "--isolation=chroot",
                    "--nix-arg",
                    "--cores=4",
                    "--uv-arg",
                    "--no-build-isolation",
                ]
            )
            == 0
        )
        assert configs == [
            docker.ContainerBuildConfig(
                container_bin="podman",
                container_args=("--isolation=chroot",),
                nix_args=("--cores=4",),
                uv_args=("--no-build-isolation",),
            )
        ]


# ── build: error paths ─────────────────────────────────────────────


class TestMainErrors:
    def test_path_not_a_directory(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            build.main([str(tmp_path / "nonexistent"), "--dry-run"])

    def test_missing_pyproject(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit):
            build.main([str(empty), "--dry-run"])

    def test_missing_uv_lock(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", with_lock=False)
        with pytest.raises(SystemExit):
            build.main([str(proj), "--dry-run"])


# ── closures: plugin closure discovery ─────────────────────────────


class TestDiscoverPluginClosures:
    def test_finds_runtime_basic(self) -> None:
        """`agentix-runtime-basic` registers the `agentix.nix` group and
        ships `default.nix` files — it must be discovered in this venv."""
        found = discover_plugin_closures()
        labels = {c.label for c in found}
        # bash + files both ship a default.nix.
        assert any("runtime-basic" in label for label in labels), labels

    def test_closures_carry_content_and_origin(self) -> None:
        found = discover_plugin_closures()
        assert found, "expected at least the runtime-basic closures"
        for c in found:
            assert c.content, f"{c.label} has empty content"
            assert b"pkgs" in c.content  # a `{ pkgs }: drv`
            assert c.origin

    def test_labels_are_unique(self) -> None:
        found = discover_plugin_closures()
        labels = [c.label for c in found]
        assert len(labels) == len(set(labels))


# ── closures: project closure ──────────────────────────────────────


class TestDiscoverProjectClosure:
    def test_no_tool_agentix_nix_returns_none(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj")
        assert discover_project_closure(proj) is None

    def test_declared_closure_collected(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="system.nix")
        (proj / "system.nix").write_text("{ pkgs }: pkgs.hello\n")
        closure = discover_project_closure(proj)
        assert closure is not None
        assert closure.label == "project"
        assert b"pkgs.hello" in closure.content

    def test_missing_declared_file_errors(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="absent.nix")
        with pytest.raises(SystemExit):
            discover_project_closure(proj)

    def test_escaping_path_rejected(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="../escape.nix")
        (tmp_path / "escape.nix").write_text("{ pkgs }: pkgs.hello\n")
        with pytest.raises(SystemExit):
            discover_project_closure(proj)


# ── closures: staging + end-to-end ─────────────────────────────────


class TestStageClosures:
    def test_writes_each_closure(self, tmp_path: Path) -> None:
        items = [
            Closure(label="a", origin="x", content=b"{ pkgs }: 1\n"),
            Closure(label="b", origin="y", content=b"{ pkgs }: 2\n"),
        ]
        out = tmp_path / "closures"
        written = stage_closures(items, out)
        assert {p.name for p in written} == {"a.nix", "b.nix"}
        assert (out / "a.nix").read_bytes() == b"{ pkgs }: 1\n"

    def test_assemble_collects_plugins_and_project(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="sys.nix")
        (proj / "sys.nix").write_text("{ pkgs }: pkgs.git\n")
        out = tmp_path / "closures"
        collected = assemble(proj, out)

        labels = {c.label for c in collected}
        assert "project" in labels
        assert any("runtime-basic" in label for label in labels)
        # every collected closure was staged as a .nix file
        for c in collected:
            assert (out / f"{c.label}.nix").is_file()

    def test_assemble_pure_python_project(self, tmp_path: Path) -> None:
        """A project with no `[tool.agentix] nix` still gets plugin
        closures — assemble never returns empty when plugins are present."""
        proj = _make_project(tmp_path / "proj")
        out = tmp_path / "closures"
        collected = assemble(proj, out)
        assert all(c.label != "project" for c in collected)
