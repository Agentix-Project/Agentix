"""Files primitive — sandbox file upload / download as an Agentix closure.

Stub-only module. Callers import the typed `Files` Namespace; the impl
lives in `_impl.py` and runs only inside the sandbox. Stub and impl
are independent classes, composed by `_register.py`.

Usage:

    from agentix import RuntimeClient
    from agentix_closures import files

    async with RuntimeClient(sandbox.runtime_url) as c:
        r = await c.remote(
            files.Files.upload,
            path="/workspace/input.txt",
            content=b"hello",
        )
        print(r.size)

        data = await c.remote(files.Files.download, path="/workspace/output.txt")

Files are encoded as pydantic `bytes` (base64 in the JSON wire form).
Suitable for kB–MB sized files; very large blobs should ship via a
purpose-built binary `WirePattern` rather than the unary JSON path.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentix.namespace import Namespace


@dataclass
class UploadResult:
    """What `Files.upload` returns — the resolved sandbox-side path and
    the number of bytes written."""

    path: str
    size: int


class Files(Namespace):
    """Sandbox file I/O primitive.

    The sandbox limits writes/reads to `$AGENTIX_UPLOAD_ROOT` (default
    `/workspace`). Paths outside that root raise `PermissionError`.
    """

    async def upload(self, path: str, content: bytes) -> UploadResult:
        """Write `content` to `path` inside the sandbox.

        Creates parent directories as needed. `path` must resolve under
        the upload-root; otherwise raises `PermissionError`.
        """
        ...

    async def download(self, path: str) -> bytes:
        """Read the contents of `path` from inside the sandbox.

        Raises `FileNotFoundError` / `IsADirectoryError` / `PermissionError`
        for the corresponding filesystem conditions.
        """
        ...
