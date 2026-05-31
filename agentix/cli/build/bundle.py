"""Build the portable Agentix bundle tar artifact.

A bundle is the deliverable of `agentix build`. Its layout is fixed and
host-portable:

    manifest.json          bundle identity + runtime contract
    nix/store/...          the closures: interpreter, uv, system deps
    nix/runtime/venv       the uv venv (all Python deps)
    nix/runtime/{bin,...}  symlinkJoin of every closure

The bundle is produced indirectly: the builder Dockerfile runs inside a
transient cache image, and this module exports a stopped container created
from that image, extracts only `nix/`, validates the runtime tree's symlinks,
hashes the result for content identity, and writes a portable tar.

Symlink validation is deliberately narrow — only `/nix/runtime` is
audited, since that's the surface providers execute. `/nix/store`
can contain incidental store-internal symlinks that aren't part of
Agentix's runtime contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from uuid import uuid4

from agentix.cli.build.docker import ContainerBuildConfig, _build_container_run_args, _docker_build_image, _run
from agentix.cli.build.naming import _tar_cache_image_ref
from agentix.cli.build.platform import nix_system_for_platform, normalize_platform
from agentix.runtime.shared.env import (
    BUNDLE_NIX_ROOT,
    BUNDLE_RUNTIME_ENTRYPOINT,
    BUNDLE_RUNTIME_ENV,
    BUNDLE_RUNTIME_ROOT,
)


def _tree_digest(root: Path) -> str:
    """Stable content digest for a bundle tree.

    The digest is over relative paths, file modes, symlink targets, and
    file bytes. It deliberately ignores uid/gid/mtime so the same bundle
    copied between hosts keeps the same identity.
    """
    h = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        st = path.lstat()
        mode = st.st_mode & 0o7777
        h.update(rel.encode())
        h.update(b"\0")
        h.update(str(mode).encode())
        h.update(b"\0")
        if path.is_symlink():
            h.update(b"L\0")
            h.update(os.readlink(path).encode())
            h.update(b"\0")
        elif path.is_dir():
            h.update(b"D\0")
        elif path.is_file():
            h.update(b"F\0")
            h.update(str(st.st_size).encode())
            h.update(b"\0")
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        else:
            raise SystemExit(f"{path}: unsupported file type in bundle")
    return h.hexdigest()


def _mapped_symlink_target(nix_root: Path, link: Path, target: str) -> Path:
    if target.startswith("/"):
        if target == "/nix":
            return nix_root
        if target.startswith("/nix/"):
            return nix_root / target.removeprefix("/nix/")
        raise SystemExit(f"{link}: symlink target {target!r} points outside /nix")

    parent = link.parent.relative_to(nix_root).as_posix()
    rel = posixpath.normpath(posixpath.join(parent, target))
    if rel == ".":
        return nix_root
    if rel == ".." or rel.startswith("../"):
        raise SystemExit(f"{link}: symlink target {target!r} escapes /nix")
    return nix_root / rel


def _validate_bundle_tree(nix_root: Path) -> None:
    """Validate the runtime-facing part of the extracted /nix tree.

    Nix store paths can contain incidental or profile-oriented symlinks
    that are not part of Agentix's runtime contract. Keep validation
    focused on `/nix/runtime`, which is what providers execute and
    prepend to the sandbox environment.
    """
    entrypoint = nix_root / BUNDLE_RUNTIME_ENTRYPOINT.removeprefix(f"{BUNDLE_NIX_ROOT}/")
    if not os.path.lexists(entrypoint):
        raise SystemExit(f"bundle missing runtime entrypoint: {BUNDLE_RUNTIME_ENTRYPOINT}")

    runtime_root = nix_root / BUNDLE_RUNTIME_ROOT.removeprefix(f"{BUNDLE_NIX_ROOT}/")
    if not runtime_root.is_dir():
        raise SystemExit(f"bundle missing runtime tree: {BUNDLE_RUNTIME_ROOT}")

    for path in sorted(runtime_root.rglob("*"), key=lambda p: p.relative_to(nix_root).as_posix()):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        mapped = _mapped_symlink_target(nix_root, path, target)
        if not os.path.lexists(mapped):
            raise SystemExit(f"{path}: broken symlink target {target!r} in bundle /nix tree")


def _bundle_manifest(*, name: str, tag: str, platform: str, digest: str) -> dict[str, object]:
    runtime_env = dict(BUNDLE_RUNTIME_ENV)
    added_env = {f"AGENTIX_ADDED_{key}": value for key, value in runtime_env.items()}
    return {
        "schema_version": 1,
        "format": "agentix-bundle",
        "name": name,
        "tag": tag,
        "platform": normalize_platform(platform),
        "nix_system": nix_system_for_platform(platform),
        "digest": f"sha256:{digest}",
        "entrypoint": BUNDLE_RUNTIME_ENTRYPOINT,
        "runtime_env": runtime_env,
        "agentix_added_env": added_env,
    }


def _portable_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_bundle_tar(bundle_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with tarfile.open(tmp_path, "w") as tar:
            tar.add(bundle_root / "manifest.json", arcname="manifest.json", filter=_portable_tarinfo)
            tar.add(bundle_root / "nix", arcname="nix", recursive=True, filter=_portable_tarinfo)
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _nix_export_member_name(name: str) -> str | None:
    normalized = posixpath.normpath(name)
    if normalized in {"", ".", ".."} or normalized.startswith("/") or normalized.startswith("../"):
        raise SystemExit(f"builder output produced unsafe tar member: {name!r}")
    if normalized == "nix" or normalized.startswith("nix/"):
        return normalized
    return None


def _ensure_safe_parent(root: Path, path: Path) -> None:
    current = root
    for part in path.parent.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise SystemExit(f"builder output produced unsafe parent path: {current.relative_to(root)}")
        else:
            current.mkdir()


def _remove_existing_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


def _extract_nix_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    bundle_root: Path,
    deferred_dirs: list[tuple[Path, int]],
) -> None:
    name = _nix_export_member_name(member.name)
    if name is None:
        return

    target = bundle_root / name
    _ensure_safe_parent(bundle_root, target)
    if member.isdir():
        if os.path.lexists(target):
            if target.is_symlink() or not target.is_dir():
                raise SystemExit(f"builder output cannot replace non-directory with directory: {name}")
        else:
            target.mkdir()
        deferred_dirs.append((target, member.mode & 0o7777))
        return

    _remove_existing_path(target)
    if member.issym():
        os.symlink(member.linkname, target)
        return
    if member.islnk():
        link_name = _nix_export_member_name(member.linkname)
        if link_name is None:
            raise SystemExit(f"builder output produced hard link outside /nix: {member.linkname!r}")
        os.link(bundle_root / link_name, target)
        return
    if member.isfile():
        source = tar.extractfile(member)
        if source is None:
            raise SystemExit(f"builder output produced unreadable tar member: {name}")
        with source, target.open("wb") as f:
            shutil.copyfileobj(source, f)
        os.chmod(target, member.mode & 0o7777)
        return

    raise SystemExit(f"builder output produced unsupported tar member: {name}")


def _export_nix_from_container(
    container: str,
    bundle_root: Path,
    *,
    config: ContainerBuildConfig,
) -> None:
    engine = config.container_engine
    cmd = [engine, "export", container]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    with TemporaryFile() as stderr:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        if proc.stdout is None:
            raise SystemExit(f"{engine} export did not provide a stdout pipe")
        deferred_dirs: list[tuple[Path, int]] = []
        try:
            try:
                with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                    for member in tar:
                        _extract_nix_member(tar, member, bundle_root, deferred_dirs)
            except tarfile.TarError as exc:
                proc.wait()
                stderr.seek(0)
                message = stderr.read().decode(errors="replace")
                raise SystemExit(message or f"{engine} export produced an invalid tar stream: {exc}") from exc
        finally:
            proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            stderr.seek(0)
            message = stderr.read().decode(errors="replace")
            raise SystemExit(message or rc)

    for path, mode in sorted(deferred_dirs, key=lambda item: len(item[0].parts), reverse=True):
        os.chmod(path, mode)


def _copy_nix_from_image(
    image_ref: str,
    bundle_root: Path,
    *,
    platform: str,
    config: ContainerBuildConfig | None = None,
) -> None:
    config = config or ContainerBuildConfig()
    engine = config.container_engine
    container = f"agentix-bundle-copy-{uuid4().hex[:12]}"
    create_cmd = [
        engine,
        "create",
        "--platform",
        normalize_platform(platform),
        "--network",
        "none",
        *_build_container_run_args(config),
        "--name",
        container,
        image_ref,
    ]
    try:
        _run(create_cmd)
        _export_nix_from_container(container, bundle_root, config=config)
    finally:
        _run([engine, "rm", "-f", container], check=False)


def _build_tar_bundle(
    stage: Path,
    *,
    output_path: Path,
    name: str,
    tag: str,
    project_subpath: Path,
    platform: str,
    config: ContainerBuildConfig | None = None,
) -> Path:
    """Build a portable Agentix bundle tar containing manifest.json + nix/."""
    config = config or ContainerBuildConfig()
    cache_ref = _tar_cache_image_ref(name=name, project_subpath=project_subpath, platform=platform)
    with TemporaryDirectory(prefix="agentix-bundle-") as tmp:
        bundle_root = Path(tmp) / "bundle"
        bundle_root.mkdir()
        _docker_build_image(stage, tags=[cache_ref], project_subpath=project_subpath, platform=platform, config=config)
        _copy_nix_from_image(cache_ref, bundle_root, platform=platform, config=config)
        if not (bundle_root / "nix").is_dir():
            raise SystemExit(f"container image {cache_ref!r} did not contain /nix")
        _validate_bundle_tree(bundle_root / "nix")
        digest = _tree_digest(bundle_root / "nix")
        manifest = _bundle_manifest(name=name, tag=tag, platform=platform, digest=digest)
        (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _write_bundle_tar(bundle_root, output_path)
    return output_path


__all__ = [
    "_build_tar_bundle",
    "_bundle_manifest",
    "_copy_nix_from_image",
    "_validate_bundle_tree",
    "_write_bundle_tar",
]
