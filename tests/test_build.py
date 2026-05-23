"""Tests for `agentix build` — the host-side bundle pipeline.

Covered: pyproject metadata extraction, image name/tag parsing, git
context-root resolution, build-context staging, `--dry-run`, error
paths, and the in-container `_assemble` closure discovery.

NOT covered here: the actual `docker buildx build` + `nix build`
execution — that needs Docker + Nix and is exercised by a real
`agentix build` run, not the unit suite. Everything testable without a
container is tested.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from agentix.cli import _assemble, build
from agentix.cli._resolve import (
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
        assert build.git_toplevel(tmp_path) is None


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

    def test_builder_files_staged(self, tmp_path: Path) -> None:
        stage = self._staged(tmp_path)
        for name in ("flake.nix", "flake.lock", "Dockerfile", "bundle-build.sh"):
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
    def test_docker_build_passes_platform(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def _fake_run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(build, "_run", _fake_run)

        ref = build._docker_build(
            tmp_path,
            name="demo",
            tag="1.0.0",
            project_subpath=Path("."),
            platform="linux/amd64",
        )

        assert ref == "demo:1.0.0"
        cmd = calls[0]
        assert cmd[:4] == ["docker", "buildx", "build", "--platform"]
        assert cmd[4] == "linux/amd64"
        assert "--load" in cmd
        assert "-t" in cmd


# ── build: tar bundle artifacts ────────────────────────────────────


class TestTarBundle:
    def test_default_tar_name_includes_platform(self) -> None:
        assert build._default_tar_name("ghcr.io/acme/demo", "1.0.0", "linux/amd64") == (
            "ghcr.io-acme-demo-1.0.0-linux-amd64.bundle.tar"
        )

    def test_tar_cache_image_ref_is_stable_across_output_tags(self) -> None:
        ref = build._tar_cache_image_ref(
            name="ghcr.io/acme/Demo Agent",
            project_subpath=Path("examples/demo"),
            platform="linux/amd64",
        )

        assert ref.startswith("agentix-bundle-cache:ghcr.io-acme-demo-agent-linux-amd64-")
        assert ref == build._tar_cache_image_ref(
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
        bundle = tmp_path / "bundle"
        store = bundle / "nix" / "store" / "hash-python" / "bin"
        runtime_bin = bundle / "nix" / "runtime" / "bin"
        store.mkdir(parents=True)
        runtime_bin.mkdir(parents=True)
        (store / "python").write_text("#!/bin/sh\n")
        (runtime_bin / "python").symlink_to("/nix/store/hash-python/bin/python")
        (bundle / "manifest.json").write_text("{}\n")

        tar_path = tmp_path / "demo.bundle.tar"
        build._write_bundle_tar(bundle, tar_path)

        with tarfile.open(tar_path) as tar:
            members = {member.name: member for member in tar.getmembers()}
        assert "manifest.json" in members
        link = members["nix/runtime/bin/python"]
        assert link.issym()
        assert link.linkname == "/nix/store/hash-python/bin/python"

    def test_validate_bundle_tree_accepts_absolute_nix_symlink(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        store = nix / "store" / "hash-agentix" / "bin"
        entry_dir = nix / "runtime" / "venv" / "bin"
        store.mkdir(parents=True)
        entry_dir.mkdir(parents=True)
        (store / "agentix-server").write_text("#!/bin/sh\n")
        (entry_dir / "agentix-server").symlink_to("/nix/store/hash-agentix/bin/agentix-server")

        build._validate_bundle_tree(nix)

    def test_validate_bundle_tree_rejects_symlink_outside_nix(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        entry_dir = nix / "runtime" / "venv" / "bin"
        entry_dir.mkdir(parents=True)
        (entry_dir / "agentix-server").write_text("#!/bin/sh\n")
        (entry_dir / "bad").symlink_to("/bin/sh")

        with pytest.raises(SystemExit, match="outside /nix"):
            build._validate_bundle_tree(nix)

    def test_validate_bundle_tree_ignores_store_symlinks(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        entry_dir = nix / "runtime" / "venv" / "bin"
        store_dir = nix / "store" / "hash-base-system" / "etc" / "ssl" / "certs"
        entry_dir.mkdir(parents=True)
        store_dir.mkdir(parents=True)
        (entry_dir / "agentix-server").write_text("#!/bin/sh\n")
        (store_dir / "ca-bundle.crt").symlink_to("/nix/var/nix/profiles/default/etc/ssl/certs/ca-bundle.crt")

        build._validate_bundle_tree(nix)

    def test_validate_bundle_tree_rejects_broken_absolute_nix_symlink(self, tmp_path: Path) -> None:
        nix = tmp_path / "nix"
        entry_dir = nix / "runtime" / "venv" / "bin"
        entry_dir.mkdir(parents=True)
        (entry_dir / "agentix-server").symlink_to("/nix/store/missing/bin/agentix-server")

        with pytest.raises(SystemExit, match="broken symlink"):
            build._validate_bundle_tree(nix)

    def test_bundle_manifest_records_runtime_contract(self) -> None:
        manifest = build._bundle_manifest(name="demo", tag="1.0.0", platform="linux/amd64", digest="abc123")
        assert manifest["schema_version"] == 1
        assert manifest["name"] == "demo"
        assert manifest["tag"] == "1.0.0"
        assert manifest["platform"] == "linux/amd64"
        assert manifest["nix_system"] == "x86_64-linux"
        assert manifest["digest"] == "sha256:abc123"
        assert manifest["entrypoint"] == "/nix/runtime/venv/bin/agentix-server"
        assert manifest["runtime_env"]["PATH"] == "/nix/runtime/venv/bin:/nix/runtime/bin"  # type: ignore[index]

    def test_copy_nix_from_image_streams_tar_with_platform(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[list[str]] = []

        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as tar:
            for dirname in ("nix", "nix/runtime", "nix/runtime/venv", "nix/runtime/venv/bin"):
                info = tarfile.TarInfo(dirname)
                info.type = tarfile.DIRTYPE
                info.mode = 0o555
                tar.addfile(info)
            data = b"#!/bin/sh\n"
            info = tarfile.TarInfo("nix/runtime/venv/bin/agentix-server")
            info.mode = 0o755
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        class FakeProc:
            def __init__(self, cmd: list[str], stdout: object, stderr: object) -> None:
                del stdout, stderr
                self.stdout = io.BytesIO(payload.getvalue())
                self.returncode = 0
                calls.append(cmd)

            def wait(self) -> int:
                return self.returncode

        def fake_popen(cmd: list[str], *, stdout: object, stderr: object) -> FakeProc:
            return FakeProc(cmd, stdout, stderr)

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        build._copy_nix_from_image("tmp:latest", tmp_path, platform="linux/amd64")

        create = calls[0]
        assert create[:6] == ["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint"]
        assert create[-5:] == ["-C", "/", "-cf", "-", "nix"]
        assert (tmp_path / "nix" / "runtime" / "venv" / "bin" / "agentix-server").is_file()
        assert (tmp_path / "nix" / "runtime").stat().st_mode & 0o777 == 0o555

    def test_copy_nix_from_image_rejects_tar_members_outside_nix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as tar:
            info = tarfile.TarInfo("etc/passwd")
            info.size = 0
            tar.addfile(info, io.BytesIO())

        class FakeProc:
            def __init__(self, cmd: list[str], stdout: object, stderr: object) -> None:
                del cmd, stdout, stderr
                self.stdout = io.BytesIO(payload.getvalue())
                self.returncode = 0

            def wait(self) -> int:
                return self.returncode

        def fake_popen(cmd: list[str], *, stdout: object, stderr: object) -> FakeProc:
            return FakeProc(cmd, stdout, stderr)

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        with pytest.raises(SystemExit, match="non-/nix tar member"):
            build._copy_nix_from_image("tmp:latest", tmp_path, platform="linux/amd64")

    def test_tar_build_keeps_cache_image(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        built_tags: list[list[str]] = []
        copied_refs: list[str] = []

        def fake_build_image(
            _stage: Path,
            *,
            tags: list[str],
            project_subpath: Path,
            platform: str,
        ) -> None:
            del project_subpath, platform
            built_tags.append(tags)

        monkeypatch.setattr(build, "_docker_build_image", fake_build_image)

        def fake_copy_nix(_image_ref: str, bundle_root: Path, *, platform: str) -> None:
            del platform
            copied_refs.append(_image_ref)
            entry_dir = bundle_root / "nix" / "runtime" / "venv" / "bin"
            entry_dir.mkdir(parents=True)
            (entry_dir / "agentix-server").write_text("#!/bin/sh\n")

        monkeypatch.setattr(build, "_copy_nix_from_image", fake_copy_nix)
        build._build_tar_bundle(
            tmp_path,
            output_path=tmp_path / "bundle.tar",
            name="demo",
            tag="1.0.0",
            project_subpath=Path("."),
            platform="linux/amd64",
        )

        cache_ref = build._tar_cache_image_ref(name="demo", project_subpath=Path("."), platform="linux/amd64")
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

        monkeypatch.setattr("agentix.cli.build._docker_build", _boom)
        assert build.main([str(proj), "--dry-run"]) == 0

    def test_output_rejected_for_oci_image(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj")
        with pytest.raises(SystemExit, match="--output"):
            build.main([str(proj), "--format", "oci-image", "--output", str(tmp_path / "out.tar"), "--dry-run"])


class TestMainBuildFormats:
    def test_default_build_format_is_tar(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        ) -> Path:
            calls.append(output_path)
            assert stage.is_dir()
            assert name == "demo-agent"
            assert tag == "0.1.0"
            assert project_subpath == Path(".")
            assert platform in {"linux/amd64", "linux/arm64"}
            return output_path

        def boom_docker(*_a: object, **_k: object) -> str:
            raise AssertionError("default build format must not publish an OCI image")

        monkeypatch.setattr(build, "_build_tar_bundle", fake_tar_bundle)
        monkeypatch.setattr(build, "_docker_build", boom_docker)

        assert build.main([str(proj), "--output", str(tmp_path / "bundle.tar")]) == 0
        assert calls == [tmp_path / "bundle.tar"]

    def test_oci_image_format_keeps_docker_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        proj = _make_project(tmp_path / "proj")
        calls: list[str] = []

        def fake_docker_build(
            stage: Path,
            *,
            name: str,
            tag: str,
            project_subpath: Path,
            platform: str,
        ) -> str:
            assert stage.is_dir()
            assert name == "demo"
            assert tag == "dev"
            assert project_subpath == Path(".")
            assert platform in {"linux/amd64", "linux/arm64"}
            calls.append(f"{name}:{tag}")
            return f"{name}:{tag}"

        def boom_tar(*_a: object, **_k: object) -> Path:
            raise AssertionError("oci-image format must not build a bundle tar")

        monkeypatch.setattr(build, "_docker_build", fake_docker_build)
        monkeypatch.setattr(build, "_build_tar_bundle", boom_tar)

        assert build.main([str(proj), "--name", "demo:dev", "--format", "oci-image"]) == 0
        assert calls == ["demo:dev"]


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


# ── _assemble: plugin closure discovery ────────────────────────────


class TestDiscoverPluginClosures:
    def test_finds_runtime_basic(self) -> None:
        """`agentix-runtime-basic` registers the `agentix.nix` group and
        ships `default.nix` files — it must be discovered in this venv."""
        closures = _assemble.discover_plugin_closures()
        labels = {c.label for c in closures}
        # bash + files both ship a default.nix.
        assert any("runtime-basic" in label for label in labels), labels

    def test_closures_carry_content_and_origin(self) -> None:
        closures = _assemble.discover_plugin_closures()
        assert closures, "expected at least the runtime-basic closures"
        for c in closures:
            assert c.content, f"{c.label} has empty content"
            assert b"pkgs" in c.content  # a `{ pkgs }: drv`
            assert c.origin

    def test_labels_are_unique(self) -> None:
        closures = _assemble.discover_plugin_closures()
        labels = [c.label for c in closures]
        assert len(labels) == len(set(labels))


# ── _assemble: project closure ─────────────────────────────────────


class TestDiscoverProjectClosure:
    def test_no_tool_agentix_nix_returns_none(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj")
        assert _assemble.discover_project_closure(proj) is None

    def test_declared_closure_collected(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="system.nix")
        (proj / "system.nix").write_text("{ pkgs }: pkgs.hello\n")
        closure = _assemble.discover_project_closure(proj)
        assert closure is not None
        assert closure.label == "project"
        assert b"pkgs.hello" in closure.content

    def test_missing_declared_file_errors(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="absent.nix")
        with pytest.raises(SystemExit):
            _assemble.discover_project_closure(proj)

    def test_escaping_path_rejected(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="../escape.nix")
        (tmp_path / "escape.nix").write_text("{ pkgs }: pkgs.hello\n")
        with pytest.raises(SystemExit):
            _assemble.discover_project_closure(proj)


# ── _assemble: staging + end-to-end ────────────────────────────────


class TestStageClosures:
    def test_writes_each_closure(self, tmp_path: Path) -> None:
        closures = [
            _assemble.Closure(label="a", origin="x", content=b"{ pkgs }: 1\n"),
            _assemble.Closure(label="b", origin="y", content=b"{ pkgs }: 2\n"),
        ]
        out = tmp_path / "closures"
        written = _assemble.stage_closures(closures, out)
        assert {p.name for p in written} == {"a.nix", "b.nix"}
        assert (out / "a.nix").read_bytes() == b"{ pkgs }: 1\n"

    def test_assemble_collects_plugins_and_project(self, tmp_path: Path) -> None:
        proj = _make_project(tmp_path / "proj", nix="sys.nix")
        (proj / "sys.nix").write_text("{ pkgs }: pkgs.git\n")
        out = tmp_path / "closures"
        collected = _assemble.assemble(proj, out)

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
        collected = _assemble.assemble(proj, out)
        assert all(c.label != "project" for c in collected)
