"""Agentix runtime server.

In-process closure dispatch. The runtime is a single Python process serving:

- built-in operations (exec/upload/download) mounted at root
- `POST /_remote` — direct dispatch to a bound impl, body specifies the
  closure's Python package path + method name
- `GET /closures` — inventory
- `GET /health`

There are no caller-chosen namespaces: each closure's Python import path
(`manifest.package`) is its routing key. Two images shipping the same
package collide; the second is skipped with a warning.

Discovery: on startup, scan /mnt/* for `entry/manifest.json`, validate
against ClosureManifest, prepend each closure's `entry/python` to sys.path,
import its declared `package`, and call `<package>._register.register()`
to obtain a Dispatcher. Closures with no/invalid/abi-mismatched manifest
are skipped — non-closure mounts (task data, caches) can coexist under
/mnt without tripping discovery.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from agentix import __version__
from agentix.dispatch import Dispatcher, Registry
from agentix.models import (
    AGENTIX_CLOSURE_ABI,
    ClosureInfo,
    ClosureManifest,
    HealthResponse,
    RemoteRequest,
    RemoteResponse,
)
from agentix.runtime.builtins import router as builtins_router

logger = logging.getLogger("agentix.runtime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

CLOSURE_MOUNT_ROOT = Path(os.environ.get("AGENTIX_CLOSURE_MOUNT_ROOT", "/mnt"))

registry = Registry()
_manifests: dict[str, ClosureManifest] = {}  # package -> manifest
_mount_paths: dict[str, Path] = {}            # package -> /mnt/<dir>


async def _auto_load() -> None:
    """Scan /mnt for closures and register each one's Dispatcher.

    For each mount with a valid `entry/manifest.json`:
      1. Prepend `<mount>/entry/python` to sys.path.
      2. Import the package named in `manifest.package`.
      3. Import `<package>._register` and call `register()` -> Dispatcher.
      4. Add to the Registry keyed by `manifest.package`.

    Failures at any step skip that closure with an error log; runtime keeps
    going. The mount named `runtime` is reserved (the server itself).
    """
    if not CLOSURE_MOUNT_ROOT.is_dir():
        return
    for mount in sorted(CLOSURE_MOUNT_ROOT.iterdir()):
        if mount.name == "runtime" or not mount.is_dir():
            continue
        manifest = _read_manifest(mount)
        if manifest is None:
            continue
        if manifest.package in registry:
            logger.error(
                "skip mount %s: package %r already registered from %s",
                mount, manifest.package, _mount_paths[manifest.package],
            )
            continue
        try:
            dispatcher = _load_dispatcher(mount, manifest)
        except Exception as exc:
            logger.exception("skip mount %s: failed to load package '%s': %s",
                             mount, manifest.package, exc)
            continue
        registry.add(manifest.package, dispatcher)
        _manifests[manifest.package] = manifest
        _mount_paths[manifest.package] = mount
        logger.info("registered closure '%s' from %s (methods=%s)",
                    manifest.package, mount, dispatcher.methods())


def _read_manifest(mount: Path) -> ClosureManifest | None:
    """Read and validate <mount>/entry/manifest.json."""
    mf_path = mount / "entry" / "manifest.json"
    if not mf_path.is_file():
        logger.warning("skip mount %s: missing entry/manifest.json", mount)
        return None
    try:
        manifest = ClosureManifest.model_validate_json(mf_path.read_text())
    except ValidationError as exc:
        logger.error("skip mount %s: invalid manifest.json: %s", mount, exc)
        return None
    if manifest.abi != AGENTIX_CLOSURE_ABI:
        logger.warning(
            "skip mount %s: abi=%d, runtime supports %d",
            mount, manifest.abi, AGENTIX_CLOSURE_ABI,
        )
        return None
    return manifest


def _load_dispatcher(mount: Path, manifest: ClosureManifest) -> Dispatcher:
    """Import the closure's package and obtain its Dispatcher.

    Convention: `<package>._register.register() -> Dispatcher`. The closure
    image arranges for the package to be importable by dropping it (and any
    deps not provided by the runtime image) under `<mount>/entry/python/`.
    """
    py_root = mount / "entry" / "python"
    if py_root.is_dir():
        sys.path.insert(0, str(py_root))
    pkg = importlib.import_module(manifest.package)
    register_mod = importlib.import_module(f"{manifest.package}._register")
    if not hasattr(register_mod, "register"):
        raise AttributeError(f"{manifest.package}._register has no register()")
    dispatcher = register_mod.register()
    if not isinstance(dispatcher, Dispatcher):
        raise TypeError(
            f"{manifest.package}._register.register() returned "
            f"{type(dispatcher).__name__}, expected Dispatcher"
        )
    _ = pkg  # imported for side effects (stub module must exist)
    return dispatcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _auto_load()
    yield


app = FastAPI(title="agentix", version=__version__, lifespan=lifespan)
app.state.registry = registry
app.include_router(builtins_router)


# ── Health & inventory ──────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.get("/closures")
async def list_closures() -> list[ClosureInfo]:
    return [
        ClosureInfo(path=str(_mount_paths[pkg]), manifest=_manifests[pkg])
        for pkg in registry.packages()
    ]


# ── Remote dispatch ─────────────────────────────────────────────


@app.post("/_remote", response_model=RemoteResponse)
async def remote_call(request: RemoteRequest) -> RemoteResponse:
    dispatcher = registry.get(request.package)
    if dispatcher is None:
        raise HTTPException(
            status_code=404,
            detail=f"closure not loaded: package={request.package!r}",
        )
    return await dispatcher.dispatch(request)


# ── Entry point (invoked as /mnt/runtime/entry/bin/start) ───────


def main() -> None:
    """Entry point the closure convention expects at
    /mnt/runtime/entry/bin/start. Port via AGENTIX_BIND_PORT (env, default
    8000); dev shell can override via --port.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="agentix runtime server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGENTIX_BIND_PORT", "8000")),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-port", type=int, default=5678)
    parser.add_argument("--debug-wait", action="store_true")
    args = parser.parse_args()

    if args.debug:
        import debugpy

        debugpy.listen(("0.0.0.0", args.debug_port))
        print(f"debugpy listening on 0.0.0.0:{args.debug_port}")
        if args.debug_wait:
            print("Waiting for debugger to attach...")
            debugpy.wait_for_client()

    uvicorn.run("agentix.runtime.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
