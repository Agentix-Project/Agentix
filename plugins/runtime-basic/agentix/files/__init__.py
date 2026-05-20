"""Files primitive — sandbox file upload / download as an Agentix namespace.

Usage:

    from agentix import RuntimeClient
    from agentix import files

    async with RuntimeClient(sandbox.runtime_url) as c:
        r = await c.remote(files.upload, path="/workspace/input.txt", content=b"hello")
        print(r.size)

        data = await c.remote(files.download, path="/workspace/output.txt")

Files are encoded as pydantic `bytes` (base64 in the JSON wire form).
Suitable for kB–MB sized files; very large blobs should ship via a
purpose-built binary `WirePattern` rather than the unary JSON path.

The package IS the namespace — `upload` and `download` are top-level
async functions, `UploadResult` is a regular dataclass callers can
import for type hints.

Writes/reads are confined to `$AGENTIX_UPLOAD_ROOT` (default
`/workspace`). Paths outside that root raise `PermissionError`.
"""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from pathlib import Path

UPLOAD_ROOT = Path(os.environ.get("AGENTIX_UPLOAD_ROOT", "/workspace")).resolve()


@dataclass
class UploadResult:
    """What `upload` returns — resolved sandbox-side path + bytes written."""

    path: str
    size: int


def _resolve_within(path: str) -> Path:
    """Return `path` lexically under `UPLOAD_ROOT`.

    Actual open/read/write calls below walk from an already-open root
    directory fd with O_NOFOLLOW, so symlink swaps cannot redirect the
    final file operation outside the upload root.
    """
    return UPLOAD_ROOT / Path(*_relative_parts(path))


def _relative_parts(path: str) -> tuple[str, ...]:
    raw = os.path.normpath(path)
    if os.path.isabs(raw):
        root = os.fspath(UPLOAD_ROOT)
        try:
            if os.path.commonpath([root, raw]) != root:
                raise PermissionError(f"Path {raw} outside allowed root {UPLOAD_ROOT}")
        except ValueError as exc:
            raise PermissionError(f"Path {raw} outside allowed root {UPLOAD_ROOT}") from exc
        raw = os.path.relpath(raw, root)
    parts = tuple(p for p in Path(raw).parts if p not in ("", "."))
    if not parts or any(p == ".." for p in parts):
        raise PermissionError(f"Path {path!r} outside allowed root {UPLOAD_ROOT}")
    return parts


def _open_parent(parts: tuple[str, ...], *, create: bool) -> tuple[int, str]:
    """Open the parent directory under UPLOAD_ROOT without following symlinks.

    Returns `(parent_fd, filename)`. The caller owns `parent_fd`.
    """
    root_fd = os.open(UPLOAD_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    fd = root_fd
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, mode=0o777, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=fd,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise PermissionError(
                        f"Refusing to follow symlink inside {UPLOAD_ROOT}"
                    ) from exc
                raise
            os.close(fd)
            fd = next_fd
        return fd, parts[-1]
    except Exception:
        os.close(fd)
        raise


def _open_file(path: str, flags: int, *, create_parents: bool, mode: int = 0o666) -> tuple[int, Path]:
    parts = _relative_parts(path)
    parent_fd, name = _open_parent(parts, create=create_parents)
    try:
        try:
            fd = os.open(name, flags | os.O_NOFOLLOW, mode, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise PermissionError(
                    f"Refusing to follow symlink inside {UPLOAD_ROOT}"
                ) from exc
            raise
        return fd, _resolve_within(path)
    finally:
        os.close(parent_fd)


async def upload(path: str, content: bytes) -> UploadResult:
    """Write `content` to `path` inside the sandbox.

    Creates parent directories as needed. `path` must resolve under
    the upload-root; otherwise raises `PermissionError`.
    """
    fd, p = _open_file(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        create_parents=True,
    )
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return UploadResult(path=str(p), size=len(content))


async def download(path: str) -> bytes:
    """Read the contents of `path` from inside the sandbox.

    Raises `FileNotFoundError` / `IsADirectoryError` /
    `PermissionError` for the corresponding filesystem conditions.
    """
    fd, _ = _open_file(path, os.O_RDONLY, create_parents=False)
    with os.fdopen(fd, "rb") as f:
        return f.read()
