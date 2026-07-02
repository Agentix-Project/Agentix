"""Containment tests for the shared hardened bundle extractor.

Every case builds a crafted tar in-memory and asserts either the tree
lands correctly under `<root>/nix` or the extractor refuses loudly and
leaves no partial cache behind.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from agentix.provider._extract import extract_nix_tree


def _dir(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info


def _file(name: str, data: bytes = b"x", mode: int = 0o644) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    return info, io.BytesIO(data)


def _symlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info


def _hardlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    return info


_BASE = ["nix", "nix/runtime"]
_BOOTSTRAP = "nix/runtime/bootstrap.sh"


def _write_tar(path: Path, members: list, *, base: bool = True) -> Path:
    """Write a tar of `members`; with `base`, prepend the minimal valid
    skeleton (nix/, nix/runtime/, bootstrap.sh) so validation passes."""
    with tarfile.open(path, "w") as tar:
        if base:
            for name in _BASE:
                tar.addfile(_dir(name))
            info, data = _file(_BOOTSTRAP, b"#!/bin/sh\n", mode=0o755)
            tar.addfile(info, data)
        for member in members:
            if isinstance(member, tuple):
                tar.addfile(member[0], member[1])
            else:
                tar.addfile(member)
    return path


def test_extracts_tree_with_links_and_exec_bits(tmp_path: Path) -> None:
    bundle = _write_tar(
        tmp_path / "bundle.tar",
        [
            _dir("nix/store"),
            _file("nix/store/tool", b"bin", mode=0o755),
            _symlink("nix/runtime/tool", "/nix/store/tool"),
            _symlink("nix/runtime/rel", "../store/tool"),
            _hardlink("nix/store/tool2", "nix/store/tool"),
        ],
    )
    nix = extract_nix_tree(bundle, tmp_path / "out")
    assert (nix / "runtime" / "bootstrap.sh").is_file()
    assert os.access(nix / "store" / "tool", os.X_OK)
    assert os.readlink(nix / "runtime" / "tool") == "/nix/store/tool"
    assert os.readlink(nix / "runtime" / "rel") == "../store/tool"
    assert (nix / "store" / "tool2").stat().st_ino == (nix / "store" / "tool").stat().st_ino


def test_idempotent_when_bootstrap_present(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [])
    nix = extract_nix_tree(bundle, tmp_path / "out")
    sentinel = nix / "runtime" / "bootstrap.sh"
    sentinel.write_text("MUTATED")
    extract_nix_tree(bundle, tmp_path / "out")
    assert sentinel.read_text() == "MUTATED"


def test_manifest_recorded_next_to_tree(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [])
    extract_nix_tree(bundle, tmp_path / "out", manifest={"digest": "sha256:abc"})
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest == {"digest": "sha256:abc"}


def test_gnu_dot_prefixed_names_extract(tmp_path: Path) -> None:
    members = [_dir("./nix"), _dir("./nix/runtime")]
    info, data = _file("./nix/runtime/bootstrap.sh", b"#!/bin/sh\n", mode=0o755)
    bundle = _write_tar(tmp_path / "bundle.tar", [*members, (info, data)], base=False)
    nix = extract_nix_tree(bundle, tmp_path / "out")
    assert (nix / "runtime" / "bootstrap.sh").is_file()


def test_traversal_member_rejected_loudly(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [_file("nix/../../escape.txt")])
    with pytest.raises(RuntimeError, match="unsafe member"):
        extract_nix_tree(bundle, tmp_path / "cache" / "out")
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "cache" / "out").exists()


def test_non_nix_member_rejected(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [_file("stray.txt")])
    with pytest.raises(RuntimeError, match="non-/nix member"):
        extract_nix_tree(bundle, tmp_path / "out")


def test_symlink_absolute_escape_rejected(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [_symlink("nix/runtime/evil", "/etc/passwd")])
    with pytest.raises(RuntimeError, match="symlink escapes /nix"):
        extract_nix_tree(bundle, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_symlink_relative_escape_rejected(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [_symlink("nix/runtime/evil", "../../../etc/passwd")])
    with pytest.raises(RuntimeError, match="symlink escapes the nix tree"):
        extract_nix_tree(bundle, tmp_path / "out")


def test_hardlink_through_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("s")
    bundle = _write_tar(
        tmp_path / "bundle.tar",
        [
            # In-tree symlink pointing at /nix is allowed as a member; the
            # hard link routed through it must still be contained. Craft the
            # escape with a linkname whose real path leaves the tmp root.
            _symlink("nix/runtime/door", "/nix/runtime/.."),
            _hardlink("nix/runtime/steal", "nix/runtime/door/secret"),
        ],
    )
    with pytest.raises(RuntimeError, match="hard-link"):
        extract_nix_tree(bundle, tmp_path / "out")


def test_hardlink_outside_nix_rejected(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [_hardlink("nix/runtime/steal", "../../etc/passwd")])
    with pytest.raises(RuntimeError, match="unsafe hard-link target"):
        extract_nix_tree(bundle, tmp_path / "out")


def test_hardlink_missing_target_rejected(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [_hardlink("nix/runtime/steal", "nix/absent")])
    with pytest.raises(RuntimeError, match="does not exist"):
        extract_nix_tree(bundle, tmp_path / "out")


def test_write_through_symlink_parent_rejected(tmp_path: Path) -> None:
    bundle = _write_tar(
        tmp_path / "bundle.tar",
        [
            _symlink("nix/runtime/dir", "/nix/store"),
            _file("nix/runtime/dir/planted", b"x"),
        ],
    )
    with pytest.raises(RuntimeError, match="parent is not a directory"):
        extract_nix_tree(bundle, tmp_path / "out")


def test_missing_bootstrap_commits_nothing(tmp_path: Path) -> None:
    bundle = _write_tar(tmp_path / "bundle.tar", [_dir("nix"), _dir("nix/runtime")], base=False)
    with pytest.raises(RuntimeError, match="bootstrap.sh"):
        extract_nix_tree(bundle, tmp_path / "cache" / "out")
    # Failed validation must leave no committed cache and no temp litter.
    assert not (tmp_path / "cache" / "out").exists()
    assert list((tmp_path / "cache").iterdir()) == []


def test_unsupported_member_type_rejected(tmp_path: Path) -> None:
    fifo = tarfile.TarInfo("nix/runtime/pipe")
    fifo.type = tarfile.FIFOTYPE
    bundle = _write_tar(tmp_path / "bundle.tar", [fifo])
    with pytest.raises(RuntimeError, match="unsupported member type"):
        extract_nix_tree(bundle, tmp_path / "out")
