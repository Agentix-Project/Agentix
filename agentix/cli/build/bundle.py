"""Build the portable Agentix bundle tar artifact.

A bundle is the `--format tar` deliverable of `agentix build`. Its
layout is fixed and host-portable:

    manifest.json          bundle identity + runtime contract
    nix/store/...          the closures: interpreter, uv, system deps
    nix/runtime/venv       the uv venv (all Python deps)
    nix/runtime/{bin,...}  symlinkJoin of every closure

The bundle is produced indirectly: the same Dockerfile that powers
`--format oci-image` runs inside a transient cache image, and this
module streams `/nix` out of that image (`docker run --entrypoint tar
... | tarfile.open`), validates the runtime tree's symlinks, hashes
the result for content identity, and writes a portable tar.

Symlink validation is deliberately narrow — only `/nix/runtime` is
audited, since that's the surface deployments execute. `/nix/store`
can contain incidental store-internal symlinks that aren't part of
Agentix's runtime contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from uuid import uuid4

from agentix.cli.build.docker import _docker_build_image
from agentix.cli.build.naming import _tar_cache_image_ref
from agentix.cli.build.platform import nix_system_for_platform, normalize_platform

_RUNTIME_ENTRYPOINT = "/nix/runtime/bootstrap.sh"

_RUNTIME_ENV = {
    "PATH": "/nix/runtime/venv/bin:/nix/runtime/bin",
    "LD_LIBRARY_PATH": "/nix/runtime/lib",
    "LIBRARY_PATH": "/nix/runtime/lib",
    "CPATH": "/nix/runtime/include",
    "C_INCLUDE_PATH": "/nix/runtime/include",
    "CPLUS_INCLUDE_PATH": "/nix/runtime/include",
    "PKG_CONFIG_PATH": "/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig",
    "CMAKE_PREFIX_PATH": "/nix/runtime",
}


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
    focused on `/nix/runtime`, which is what deployments execute and
    prepend to the sandbox environment.
    """
    entrypoint = nix_root / _RUNTIME_ENTRYPOINT.removeprefix("/nix/")
    if not os.path.lexists(entrypoint):
        raise SystemExit(f"bundle missing runtime entrypoint: {_RUNTIME_ENTRYPOINT}")

    runtime_root = nix_root / "runtime"
    if not runtime_root.is_dir():
        raise SystemExit("bundle missing runtime tree: /nix/runtime")

    for path in sorted(runtime_root.rglob("*"), key=lambda p: p.relative_to(nix_root).as_posix()):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        mapped = _mapped_symlink_target(nix_root, path, target)
        if not os.path.lexists(mapped):
            raise SystemExit(f"{path}: broken symlink target {target!r} in bundle /nix tree")


def _bundle_manifest(*, name: str, tag: str, platform: str, digest: str) -> dict[str, object]:
    runtime_env = dict(_RUNTIME_ENV)
    added_env = {f"AGENTIX_ADDED_{key}": value for key, value in runtime_env.items()}
    return {
        "schema_version": 1,
        "format": "agentix-bundle",
        "name": name,
        "tag": tag,
        "platform": normalize_platform(platform),
        "nix_system": nix_system_for_platform(platform),
        "digest": f"sha256:{digest}",
        "entrypoint": _RUNTIME_ENTRYPOINT,
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


def _checked_nix_member_name(name: str) -> str:
    normalized = posixpath.normpath(name)
    if normalized in {"", "."} or normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise SystemExit(f"bundle image produced unsafe tar member: {name!r}")
    if normalized != "nix" and not normalized.startswith("nix/"):
        raise SystemExit(f"bundle image produced non-/nix tar member: {name!r}")
    return normalized


def _extract_nix_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    bundle_root: Path,
    deferred_dirs: list[tuple[Path, int]],
) -> None:
    member.name = _checked_nix_member_name(member.name)
    if member.islnk():
        _checked_nix_member_name(member.linkname)
    if member.isdir():
        original_mode = member.mode
        deferred_dirs.append((bundle_root / member.name, original_mode))
        member.mode = original_mode | 0o700
    tar.extract(member, bundle_root)


def _copy_nix_from_image(image_ref: str, bundle_root: Path, *, platform: str) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        normalize_platform(platform),
        "--entrypoint",
        "tar",
        image_ref,
        "-C",
        "/",
        "-cf",
        "-",
        "nix",
    ]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    with TemporaryFile() as stderr:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        if proc.stdout is None:
            raise SystemExit("docker run did not provide a stdout pipe")
        deferred_dirs: list[tuple[Path, int]] = []
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    _extract_nix_member(tar, member, bundle_root, deferred_dirs)
        finally:
            proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            stderr.seek(0)
            message = stderr.read().decode(errors="replace")
            raise SystemExit(message or rc)
    for path, mode in sorted(deferred_dirs, key=lambda item: len(item[0].parts), reverse=True):
        os.chmod(path, mode)


def _build_tar_bundle(
    stage: Path,
    *,
    output_path: Path,
    name: str,
    tag: str,
    project_subpath: Path,
    platform: str,
) -> Path:
    """Build a portable Agentix bundle tar containing manifest.json + nix/."""
    cache_ref = _tar_cache_image_ref(name=name, project_subpath=project_subpath, platform=platform)
    with TemporaryDirectory(prefix="agentix-bundle-") as tmp:
        bundle_root = Path(tmp) / "bundle"
        bundle_root.mkdir()
        _docker_build_image(stage, tags=[cache_ref], project_subpath=project_subpath, platform=platform)
        _copy_nix_from_image(cache_ref, bundle_root, platform=platform)
        if not (bundle_root / "nix").is_dir():
            raise SystemExit(f"docker image {cache_ref!r} did not contain /nix")
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
