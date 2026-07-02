"""Hardened extraction of a portable bundle tar's `nix/` tree.

Shared by host-side provider backends that materialize a bundle on the
host filesystem (the docker/podman deploy cache, the apptainer
per-create cache). The tar normally comes from `agentix build`, but the
extractor treats it as untrusted input: every member is validated
before any write, so a crafted bundle cannot touch paths outside the
extraction root.

Guarantees:

- member names must normalize to `nix` or `nix/...` (GNU-tar `./nix/...`
  spellings included); a name that merely claims the `nix/` prefix but
  normalizes elsewhere (`nix/../../x`) is rejected, not skipped.
- no write ever routes through a symlinked parent directory.
- symlink targets must stay inside the nix tree: absolute targets must
  point into `/nix` (valid once the tree is mounted there), relative
  targets must resolve under the extraction root.
- hard links never follow a symlink out of the extraction root.
- extraction lands in a temporary sibling directory and is committed
  with one atomic rename only after the tree proves to contain
  `nix/runtime/bootstrap.sh` — a failed or malformed bundle leaves no
  partial cache behind.
"""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

__all__ = ["extract_nix_tree"]


def _bootstrap_path(root: Path) -> Path:
    return root / "nix" / "runtime" / "bootstrap.sh"


def _checked_member_name(name: str) -> str:
    normalized = posixpath.normpath(name)
    if normalized in {"", "."} or normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise RuntimeError(f"bundle tar produced unsafe member: {name!r}")
    if normalized != "nix" and not normalized.startswith("nix/"):
        raise RuntimeError(f"bundle tar produced non-/nix member: {name!r}")
    return normalized


def _checked_symlink_target(name: str, linkname: str) -> str:
    """Contain a symlink member's target inside the nix tree.

    Absolute targets must point into `/nix` — that is where the tree is
    mounted at runtime, so `/nix/store/...` links are the normal Nix
    shape. Relative targets must resolve (from the member's directory)
    to somewhere under the extracted tree.
    """
    if not linkname:
        raise RuntimeError(f"bundle tar symlink has an empty target: {name!r}")
    if linkname.startswith("/"):
        normalized = posixpath.normpath(linkname)
        if normalized != "/nix" and not normalized.startswith("/nix/"):
            raise RuntimeError(f"bundle tar symlink escapes /nix: {name!r} -> {linkname!r}")
        return linkname
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), linkname))
    if resolved != "nix" and not resolved.startswith("nix/"):
        raise RuntimeError(f"bundle tar symlink escapes the nix tree: {name!r} -> {linkname!r}")
    return linkname


def _checked_hardlink_target(root: Path, name: str, linkname: str) -> Path:
    try:
        link_name = _checked_member_name(linkname)
    except RuntimeError as exc:
        raise RuntimeError(f"bundle tar produced unsafe hard-link target: {linkname!r}") from exc
    link_target = root / link_name
    if not os.path.lexists(link_target):
        raise RuntimeError(f"bundle tar hard-link target does not exist: {linkname!r}")
    # A symlink anywhere on the target path could route the new link
    # outside the tree — contain the fully resolved path.
    real_target = Path(os.path.realpath(link_target))
    real_root = Path(os.path.realpath(root))
    if real_target != real_root and real_root not in real_target.parents:
        raise RuntimeError(f"bundle tar hard-link escapes extraction root: {name!r} -> {linkname!r}")
    return link_target


def _ensure_safe_parent(root: Path, path: Path) -> None:
    current = root
    for part in path.parent.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise RuntimeError(f"bundle tar member parent is not a directory: {current}")
        else:
            current.mkdir()


def _remove_existing_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


def _extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, root: Path) -> None:
    name = _checked_member_name(member.name)
    target = root / name
    _ensure_safe_parent(root, target)

    if member.isdir():
        if os.path.lexists(target):
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError(f"bundle tar cannot replace non-directory with directory: {name}")
        else:
            target.mkdir()
        return

    _remove_existing_path(target)
    if member.issym():
        os.symlink(_checked_symlink_target(name, member.linkname), target)
        return
    if member.islnk():
        link_target = _checked_hardlink_target(root, name, member.linkname)
        # follow_symlinks=False: when the target's final component is an
        # (in-tree) symlink, link the symlink itself rather than what it
        # points at, preserving the tar's structure.
        os.link(link_target, target, follow_symlinks=False)
        return
    if member.isfile():
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"bundle tar has unreadable file member: {name}")
        with source, target.open("wb") as f:
            shutil.copyfileobj(source, f)
        os.chmod(target, member.mode & 0o7777)
        return

    raise RuntimeError(f"bundle tar contains unsupported member type: {name}")


def extract_nix_tree(bundle_tar: Path, root: Path, *, manifest: dict[str, object] | None = None) -> Path:
    """Materialize `bundle_tar`'s `nix/` tree at `root/nix` and return it.

    Idempotent: returns immediately when `root` already holds a
    bootstrapped tree. When `manifest` is given, it is recorded as
    `manifest.json` next to the extracted tree before the atomic commit.
    """
    if _bootstrap_path(root).is_file():
        return root / "nix"
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{root.name}.", dir=root.parent) as tmp:
        tmp_root = Path(tmp)
        with tarfile.open(bundle_tar, "r:*") as tar:
            for member in tar:
                if posixpath.normpath(member.name) == "manifest.json":
                    continue
                _extract_member(tar, member, tmp_root)
        if not _bootstrap_path(tmp_root).is_file():
            raise RuntimeError(f"bundle {bundle_tar} does not contain nix/runtime/bootstrap.sh")
        if manifest is not None:
            (tmp_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _remove_existing_path(root)
        tmp_root.replace(root)
    return root / "nix"
