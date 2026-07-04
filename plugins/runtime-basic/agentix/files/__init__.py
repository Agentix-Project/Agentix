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
`/workspace`). The invariant is on the RESOLVED target: symlinks are
followed as long as every hop stays under the root (merged-usr images
make `/bin` a symlink; real repos contain symlinks), and any hop whose
target lands outside the root raises `PermissionError`.
"""

from __future__ import annotations

import errno
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

UPLOAD_ROOT = Path(os.environ.get("AGENTIX_UPLOAD_ROOT", "/workspace")).resolve()

# Linux's own resolution cap is 40; a cycle inside the root would
# otherwise loop the walk forever.
_MAX_SYMLINK_HOPS = 40


@dataclass
class UploadResult:
    """What `upload` returns — resolved sandbox-side path + bytes written."""

    path: str
    size: int


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


def _is_symlink(exc: OSError, part: str, dir_fd: int) -> bool:
    """Did an O_NOFOLLOW open fail because `part` is a symlink?

    Linux reports ELOOP; macOS reports ENOTDIR for the O_DIRECTORY case —
    and ENOTDIR is ambiguous with a genuine regular-file-in-the-middle, so
    readlink is the arbiter.
    """
    if exc.errno not in (errno.ELOOP, errno.ENOTDIR):
        return False
    try:
        os.readlink(part, dir_fd=dir_fd)
    except OSError:
        return False
    return True


def _link_components(dir_fd: int, name: str) -> tuple[bool, list[str]]:
    """`(is_absolute, components)` of the symlink `name` under `dir_fd`.

    Components keep `..` verbatim — resolution happens against real directory
    fds in the walk, never by lexical normalization (which would apply `..`
    before the preceding symlink is resolved and pick the wrong file)."""
    target = os.readlink(name, dir_fd=dir_fd)
    comps = [c for c in target.split("/") if c not in ("", ".")]
    return os.path.isabs(target), comps


def _open_file(path: str, flags: int, *, create_parents: bool, mode: int = 0o666) -> tuple[int, Path]:
    """Open `path` confined under `UPLOAD_ROOT`, following in-root symlinks.

    A kernel-faithful (``RESOLVE_IN_ROOT``-style) walk that makes escape
    structurally impossible rather than lexically checked:

    - the root dir is opened ONCE with ``O_NOFOLLOW`` and retained for the
      whole walk (never reopened by path), so a concurrent rename/replace of
      the root cannot redirect a later step outside the jail;
    - every component opens with ``O_NOFOLLOW``; a symlink is read and its
      target's components are pushed onto the queue — absolute targets
      re-anchor at the root (chroot-like), relative targets resolve against
      the current directory;
    - ``..`` pops the open-dir stack, clamped at the root — you can never
      ascend above it — so ``..`` is applied against actually-resolved
      directories, not collapsed lexically.
    """
    parts = _relative_parts(path)
    root_fd = os.open(UPLOAD_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    dir_stack = [root_fd]  # owned fds; dir_stack[-1] is the current directory
    name_stack: list[str] = []  # resolved component names from the root
    queue = deque(parts)
    hops = 0
    try:
        while queue:
            comp = queue.popleft()
            if comp == "..":
                if len(dir_stack) > 1:
                    os.close(dir_stack.pop())
                    name_stack.pop()
                continue  # at root, `..` clamps to root
            cur = dir_stack[-1]
            is_last = not queue

            if is_last:
                try:
                    final_fd = os.open(comp, flags | os.O_NOFOLLOW, mode, dir_fd=cur)
                except OSError as exc:
                    if not _is_symlink(exc, comp, cur):
                        raise
                    hops = _bump_hops(hops, path)
                    absolute, comps = _link_components(cur, comp)
                    if absolute:
                        _reset_to_root(dir_stack, name_stack)
                    queue.extendleft(reversed(comps))
                    continue
                return final_fd, UPLOAD_ROOT.joinpath(*name_stack, comp)

            if create_parents:
                try:
                    os.mkdir(comp, mode=0o777, dir_fd=cur)
                except FileExistsError:
                    pass
            try:
                next_fd = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
            except OSError as exc:
                if not _is_symlink(exc, comp, cur):
                    raise
                hops = _bump_hops(hops, path)
                absolute, comps = _link_components(cur, comp)
                if absolute:
                    _reset_to_root(dir_stack, name_stack)
                queue.extendleft(reversed(comps))
                continue
            dir_stack.append(next_fd)
            name_stack.append(comp)
        # The path resolved to a directory (the root itself, or a resolved
        # dir) with no final file component.
        raise IsADirectoryError(f"{path!r} resolves to a directory under {UPLOAD_ROOT}")
    finally:
        for fd in dir_stack:
            os.close(fd)


def _bump_hops(hops: int, path: str) -> int:
    hops += 1
    if hops > _MAX_SYMLINK_HOPS:
        raise PermissionError(f"Too many symlink hops resolving {path!r} under {UPLOAD_ROOT}")
    return hops


def _reset_to_root(dir_stack: list[int], name_stack: list[str]) -> None:
    """Drop every resolved directory back to the retained root fd — an
    absolute symlink target is interpreted relative to the jail root."""
    while len(dir_stack) > 1:
        os.close(dir_stack.pop())
    name_stack.clear()


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
