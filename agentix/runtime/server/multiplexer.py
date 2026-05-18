"""Multiplexer — manages one worker subprocess per namespace.

Sits between the FastAPI/Socket.IO transports and the namespace workers.
A worker is a child Python process (typically running in its own venv,
for dep isolation) that dispatches one namespace's methods. The
multiplexer's job is to:

  1. **Discover** what namespaces exist in the bundle. In production
     this walks each `/venvs/<short>/` for entry points; in tests it
     accepts in-process registrations via `_register_inprocess(...)`.
  2. **Spawn workers lazily.** First dispatch for a namespace forks
     `python -m agentix.runtime.server.worker --target <pkg>:<class>` using
     that namespace's venv interpreter, plumbs stdin/stdout for frames.
  3. **Route frames** between transports (POST /_remote, Socket.IO) and
     workers, correlated by `call_id`.
  4. **Forward trace events** from workers up to the runtime's
     Socket.IO trace bridge.
  5. **Tear down** workers on shutdown.

Two backends share one routing layer:

  - `SubprocessEntry` — `target_module:Class` + python interpreter path;
    real isolated process per namespace. Production path.
  - `InProcessEntry` — already-bound Dispatcher held in this process.
    Test fixture path; lets pytest exercise the multiplexer's wire
    protocol without forcing every test class to live in an importable
    module + venv.

Both look identical to the transports — the multiplexer dispatches
through a thin `_WorkerLike` protocol that either ships frames to a
subprocess or feeds them to an in-process Dispatcher directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentix.dispatch import NAMESPACE_ENTRY_POINT_GROUP, Dispatcher
from agentix.idents import PackageName
from agentix.models import NamespaceManifest
from agentix.runtime.shared import frames as F
from agentix.runtime.shared.models import RemoteError, RemoteRequest, RemoteResponse
from agentix.runtime.shared.rpc import read_frame, write_frame

logger = logging.getLogger("agentix.runtime.server.multiplexer")

_WORKER_START_TIMEOUT = 15.0

# ── trace forwarder ─────────────────────────────────────────────────


TraceForwarder = Callable[[str, dict[str, Any], str | None, str | None], None]
"""Callback the multiplexer invokes for every trace event from any worker.
The runtime plugs in a function that publishes to the Socket.IO `traces`
room."""


class _WorkerLike(Protocol):
    """Both _InProcessWorker and _SubprocessWorker satisfy this surface;
    the multiplexer routes through it without caring which backs the call."""

    async def call_unary(self, request: RemoteRequest) -> RemoteResponse: ...
    def iter_stream(self, request: RemoteRequest) -> AsyncIterator[dict[str, Any]]: ...
    def iter_bidi(
        self, request: RemoteRequest, input_iter: AsyncIterator[Any],
    ) -> AsyncIterator[dict[str, Any]]: ...
    async def shutdown(self) -> None: ...


# ── entries (one per discovered namespace) ──────────────────────────


@dataclass
class _NamespaceEntry:
    """Common fields. Either `target`+`python` (subprocess) or
    `dispatcher` (in-process) is set, never both."""

    package: str                     # python import path, e.g. "agentix.bash"
    dist_name: str                   # pyproject [project].name
    dist_version: str

    # Subprocess fields
    target: str | None = None         # "module" or "module:attr"
    python: str | None = None         # path to interpreter for this venv
    bin_dir: Path | None = None       # /nix/<short>/bin — prepended to worker PATH

    # In-process fields (tests)
    dispatcher: Dispatcher | None = None

    # Spawned worker state (lazy)
    worker: _WorkerLike | None = field(default=None, repr=False)
    spawn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


# ── worker variants ────────────────────────────────────────────────


class _InProcessWorker:
    """In-process worker — for tests. Routes through Dispatcher directly."""

    def __init__(self, dispatcher: Dispatcher) -> None:
        self._dispatcher = dispatcher

    async def call_unary(self, request: RemoteRequest) -> RemoteResponse:
        return await self._dispatcher.dispatch(request)

    async def iter_stream(self, request: RemoteRequest) -> AsyncIterator[dict[str, Any]]:
        async for ev in self._dispatcher.dispatch_stream(request):
            yield ev

    async def iter_bidi(
        self, request: RemoteRequest, input_iter: AsyncIterator[Any],
    ) -> AsyncIterator[dict[str, Any]]:
        # Coerce input items via the dispatcher's bound adapter so the
        # impl receives typed values, matching subprocess-worker semantics
        # (where the worker process owns the same coercion).
        adapter = self._dispatcher.input_adapter_for(request.method)  # type: ignore[arg-type]

        async def _coerced():
            async for raw in input_iter:
                if adapter is not None:
                    raw = adapter.validate_python(raw)
                yield raw

        async for ev in self._dispatcher.dispatch_bidi(request, _coerced()):
            yield ev

    async def shutdown(self) -> None:
        return


class _SubprocessWorker:
    """Subprocess worker — spawns `python -m agentix.runtime.server.worker`."""

    def __init__(
        self,
        package: str,
        target: str,
        python: str,
        trace_forwarder: TraceForwarder | None,
        ns_bin_dir: Path | None = None,
    ) -> None:
        self._package = package
        self._target = target
        self._python = python
        self._trace_forwarder = trace_forwarder
        # If provided, prepended to the worker's PATH so user code can call
        # Nix-provided binaries (`git`, `claude`, …) by bare name without
        # knowing the absolute /nix/<short>/bin path.
        self._ns_bin_dir = ns_bin_dir

        self._proc: asyncio.subprocess.Process | None = None
        self._send_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._boot_error: dict[str, Any] | None = None
        self._read_task: asyncio.Task | None = None
        self._closed = asyncio.Event()

        # Per-call state: futures for unary, queues for stream/bidi.
        self._unary: dict[str, asyncio.Future] = {}
        self._streams: dict[str, asyncio.Queue] = {}
        # Best-effort cancel sends kept alive (asyncio's task GC).
        self._cancel_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        env = dict(os.environ)
        if self._ns_bin_dir is not None:
            # Prepend the namespace's bin dir so subprocess.run("git", …)
            # in user code resolves to /nix/<short>/bin/git transparently.
            existing = env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
            env["PATH"] = f"{self._ns_bin_dir}:{existing}"
        self._proc = await asyncio.create_subprocess_exec(
            self._python, "-m", "agentix.runtime.server.worker", "--target", self._target,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,  # logs straight through to runtime stderr
            env=env,
        )
        self._read_task = asyncio.create_task(self._read_loop())
        ready_task = asyncio.create_task(self._ready.wait())
        closed_task = asyncio.create_task(self._closed.wait())
        assert self._proc is not None
        proc_task = asyncio.create_task(self._proc.wait())
        try:
            done, pending = await asyncio.wait(
                {ready_task, closed_task, proc_task},
                timeout=_WORKER_START_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if not done:
                await self.shutdown()
                raise TimeoutError(
                    f"worker for {self._package!r} did not become ready "
                    f"within {_WORKER_START_TIMEOUT:.0f}s"
                )
            if ready_task not in done:
                rc = self._proc.returncode
                await self.shutdown()
                detail = f"exit code {rc}" if rc is not None else "stdout closed"
                raise RuntimeError(
                    f"worker for {self._package!r} exited before ready ({detail})"
                )
        finally:
            for task in (ready_task, closed_task, proc_task):
                if not task.done():
                    task.cancel()
        if self._boot_error is not None:
            await self.shutdown()
            raise RuntimeError(
                f"worker for {self._package!r} failed to boot: "
                f"{self._boot_error.get('type')}: {self._boot_error.get('message')}"
            )

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                frame = await read_frame(self._proc.stdout)
                if frame is None:
                    break
                self._on_frame(frame)
        except Exception:
            logger.exception("worker %r read loop crashed", self._package)
        finally:
            self._closed.set()
            # Fail any pending calls.
            for fut in list(self._unary.values()):
                if not fut.done():
                    fut.set_exception(RuntimeError(f"worker {self._package!r} exited"))
            self._unary.clear()
            for q in list(self._streams.values()):
                q.put_nowait({"type": "error", "error": RemoteError(
                    type="WorkerExited", message=f"worker {self._package!r} exited",
                ).model_dump()})

    def _on_frame(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == F.READY:
            self._ready.set()
        elif kind == F.BOOT_ERROR:
            self._boot_error = frame.get("error") or {"type": "Unknown", "message": ""}
            self._ready.set()
        elif kind == F.RESULT:
            cid = frame.get("call_id", "")
            fut = self._unary.pop(cid, None)
            if fut and not fut.done():
                fut.set_result(RemoteResponse(ok=True, value=frame.get("value")))
        elif kind == F.ERROR:
            cid = frame.get("call_id", "")
            err_payload = frame.get("error") or {"type": "Unknown", "message": ""}
            err = RemoteError(**err_payload)
            fut = self._unary.pop(cid, None)
            if fut and not fut.done():
                fut.set_result(RemoteResponse(ok=False, error=err))
                return
            q = self._streams.get(cid)
            if q is not None:
                q.put_nowait({"type": "error", "error": err_payload})
        elif kind == F.STREAM_ITEM:
            q = self._streams.get(frame.get("call_id", ""))
            if q is not None:
                q.put_nowait({"type": "item", "value": frame.get("value")})
        elif kind == F.STREAM_END:
            q = self._streams.get(frame.get("call_id", ""))
            if q is not None:
                q.put_nowait({"type": "end"})
        elif kind == F.TRACE:
            if self._trace_forwarder is not None:
                self._trace_forwarder(
                    frame.get("kind", ""),
                    frame.get("payload") or {},
                    frame.get("call_id"),
                    frame.get("source"),
                )
        else:
            logger.warning("worker %r: unknown frame %r", self._package, kind)

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        async with self._send_lock:
            await write_frame(self._proc.stdin, payload)

    def _call_frame(self, kind: str, cid: str, request: RemoteRequest) -> dict[str, Any]:
        return {
            "type": F.CALL, "kind": kind, "call_id": cid,
            "method": request.method, "args": request.args, "kwargs": request.kwargs,
        }

    def _schedule_cancel(self, cid: str) -> None:
        """Tell the worker to abort a call. Best-effort; the send is fired
        off as a background task tracked in `_cancel_tasks` so asyncio
        doesn't GC the reference before the frame lands."""
        t = asyncio.create_task(self._send_cancel(cid))
        self._cancel_tasks.add(t)
        t.add_done_callback(self._cancel_tasks.discard)

    async def _send_cancel(self, cid: str) -> None:
        try:
            await self._send({"type": F.CANCEL, "call_id": cid})
        except Exception:
            logger.debug("cancel send failed for call %r", cid)

    async def call_unary(self, request: RemoteRequest) -> RemoteResponse:
        cid = request.call_id or _new_id()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._unary[cid] = fut
        try:
            await self._send(self._call_frame(F.KIND_UNARY, cid, request))
            return await fut
        finally:
            self._unary.pop(cid, None)
            # Caller bailed before the worker replied — tell it to stop.
            if not fut.done():
                self._schedule_cancel(cid)

    async def iter_stream(self, request: RemoteRequest) -> AsyncIterator[dict[str, Any]]:
        cid = request.call_id or _new_id()
        q: asyncio.Queue = asyncio.Queue()
        self._streams[cid] = q
        terminated = False
        try:
            await self._send(self._call_frame(F.KIND_STREAM, cid, request))
            while True:
                ev = await q.get()
                yield ev
                if ev.get("type") in ("end", "error"):
                    terminated = True
                    return
        finally:
            self._streams.pop(cid, None)
            if not terminated:
                self._schedule_cancel(cid)

    async def iter_bidi(
        self, request: RemoteRequest, input_iter: AsyncIterator[Any],
    ) -> AsyncIterator[dict[str, Any]]:
        cid = request.call_id or _new_id()
        q: asyncio.Queue = asyncio.Queue()
        self._streams[cid] = q
        input_task: asyncio.Task | None = None
        terminated = False
        try:
            await self._send(self._call_frame(F.KIND_BIDI, cid, request))

            async def _pump_input():
                try:
                    async for item in input_iter:
                        await self._send({"type": F.BIDI_IN, "call_id": cid, "item": item})
                finally:
                    await self._send({"type": F.BIDI_END_IN, "call_id": cid})

            input_task = asyncio.create_task(_pump_input())
            while True:
                ev = await q.get()
                yield ev
                if ev.get("type") in ("end", "error"):
                    terminated = True
                    return
        finally:
            self._streams.pop(cid, None)
            if input_task is not None:
                input_task.cancel()
                with contextlib.suppress(BaseException):
                    await input_task
            if not terminated:
                self._schedule_cancel(cid)

    async def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            await self._send({"type": F.SHUTDOWN})
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except TimeoutError:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2)
            except TimeoutError:
                self._proc.kill()
                # SIGKILL is guaranteed; wait so we reap the zombie.
                await self._proc.wait()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(BaseException):
                await self._read_task


def _new_id() -> str:
    import uuid
    return uuid.uuid4().hex


# ── multiplexer ─────────────────────────────────────────────────────


class NamespaceMultiplexer:
    """Owns the package → worker mapping; routes dispatches.

    Two dispatch paths:

    1. **Plugins**: discovered via the `agentix.namespace` entry-point
       group at startup. Every plugin gets a pre-registered entry —
       `c.remote(bash.run, …)` immediately knows which venv to spawn
       the worker in.
    2. **Arbitrary user modules**: any importable Python module is a
       valid dispatch target. On first call to a package not in
       `_entries`, the multiplexer probes each known venv interpreter
       to see if the module imports there; first match wins and gets
       cached as a new `_NamespaceEntry`. No entry point, no
       `agentix.*` prefix, no boilerplate — `c.remote(my_app.tasks.run, …)`
       works as long as `my_app.tasks` is `pip install`-ed somewhere
       the runtime can see.
    """

    def __init__(self, trace_forwarder: TraceForwarder | None = None) -> None:
        self._entries: dict[str, _NamespaceEntry] = {}    # package → entry
        # Every venv interpreter we know about (from entry-point discovery
        # or test registrations). On unknown-package dispatch, we try each
        # one in turn — first that can import the module gets registered.
        self._venv_pythons: list[tuple[str, Path | None]] = []  # (python, bin_dir)
        self._trace_forwarder = trace_forwarder

    # ── discovery ───────────────────────────────────────────────────

    def discover_entry_points(self) -> None:
        """Walk installed `agentix.namespace` entry points and pre-register.

        Two layouts:

          * **Bundle.** A `/nix/runtime/` directory exists with the
            framework + every pip-installed plugin in its
            site-packages. Discovery walks that venv's entry points.
          * **Dev / test.** No bundle layout; fall back to walking the
            current Python's installed entry points.

        Tests using `_register_inprocess()` skip this entirely.
        """
        nix_runtime = Path("/nix/runtime")
        if nix_runtime.is_dir():
            self._discover_from_nix(nix_runtime)
        else:
            self._discover_from_current_env()

    def _discover_from_nix(self, venv: Path) -> None:
        """Walk the bundle's single venv (/nix/runtime/) for entry points."""
        python = venv / "bin" / "python"
        bin_dir = venv / "bin"
        site_pkgs_candidates = list(venv.glob("lib/python*/site-packages"))
        if not python.exists() or not site_pkgs_candidates:
            return
        site_pkgs = site_pkgs_candidates[0]
        # Record the venv even if it carries no entry points — on-demand
        # registration will probe it for arbitrary user modules.
        self._venv_pythons.append((str(python), bin_dir))
        for dist in importlib.metadata.distributions(path=[str(site_pkgs)]):
            for ep in dist.entry_points:
                if ep.group != NAMESPACE_ENTRY_POINT_GROUP:
                    continue
                # See _discover_from_current_env: package routing key is
                # the left-of-colon portion (the module path).
                package = ep.value.split(":", 1)[0]
                self._entries[package] = _NamespaceEntry(
                    package=package,
                    dist_name=dist.metadata["Name"] or "",
                    dist_version=dist.version or "",
                    target=ep.value, python=str(python),
                    bin_dir=bin_dir,
                )

    def _discover_from_current_env(self) -> None:
        # Always make sys.executable available for on-demand registration,
        # even if no plugin entry points are installed.
        self._venv_pythons.append((sys.executable, None))
        eps = importlib.metadata.entry_points()
        selected = (
            eps.select(group=NAMESPACE_ENTRY_POINT_GROUP)
            if hasattr(eps, "select") else
            eps.get(NAMESPACE_ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
        )
        for ep in selected:
            dist = ep.dist
            dist_name = getattr(dist, "name", "") if dist else ""
            dist_version = getattr(dist, "version", "") if dist else ""
            # Entry-point value is either `module` (package-as-namespace,
            # recommended) or `module:attr` (legacy class-style). Either
            # way the package routing key is the module path on the left.
            package = ep.value.split(":", 1)[0]
            self._entries[package] = _NamespaceEntry(
                package=package, dist_name=dist_name, dist_version=dist_version,
                target=ep.value, python=sys.executable,
            )

    # ── test-only registration paths ────────────────────────────────
    # Underscore-prefixed because production code never reaches these:
    # entry-point discovery (`discover_entry_points`) and on-demand
    # registration (`_auto_register`) are the only production paths.
    # Tests use these to bypass venv discovery for fixtures.

    def _register_inprocess(self, cls: type) -> None:
        """Bind a class in-process — bypasses subprocess + venv discovery.

        Used in pytest fixtures (see tests/conftest.py) to make a
        regular Python class a dispatchable namespace inside the same
        process the runtime runs in.
        """
        package = cls.__module__
        dispatcher = Dispatcher().bind_namespace(cls)
        self._entries[package] = _NamespaceEntry(
            package=package, dist_name=package.replace(".", "-"), dist_version="0.0.0",
            dispatcher=dispatcher,
        )

    def _register_subprocess(
        self,
        package: str,
        target: str,
        python: str,
        *,
        dist_name: str = "",
        dist_version: str = "0.0.0",
        bin_dir: Path | None = None,
    ) -> None:
        """Register a subprocess-backed namespace explicitly.

        Used by tests that need to exercise the real subprocess path
        rather than `_InProcessWorker`. Production code never calls
        this — entries are built from installed entry points.
        """
        self._entries[package] = _NamespaceEntry(
            package=package, dist_name=dist_name, dist_version=dist_version,
            target=target, python=python, bin_dir=bin_dir,
        )

    def has(self, package: str) -> bool:
        return package in self._entries

    def manifests(self) -> list[NamespaceManifest]:
        out: list[NamespaceManifest] = []
        for entry in self._entries.values():
            out.append(NamespaceManifest(
                name=entry.dist_name or entry.package.rsplit(".", 1)[-1],
                version=entry.dist_version or "0.0.0",
                package=PackageName(entry.package),
            ))
        return out

    # ── on-demand registration for arbitrary modules ────────────────

    def _auto_register(self, package: str) -> bool:
        """Try to register `package` against any known venv.

        For dispatch to an unknown package, we check each discovered
        interpreter for whether the module is importable. The bundle
        carries one venv at `/nix/runtime/`; dev/test mode uses
        `sys.executable`. First match wins.

        Fast path: the runtime's own Python tries `importlib.util.find_spec`
        in-process (no subprocess). Slow path: any auxiliary venv we
        know about gets a `python -c 'import <pkg>'` probe.

        Returns True iff the module was registered.
        """
        # Fast path: this Python.
        try:
            if importlib.util.find_spec(package) is not None:
                self._entries[package] = _NamespaceEntry(
                    package=package, dist_name="", dist_version="",
                    target=package, python=sys.executable,
                )
                return True
        except (ImportError, ValueError):
            pass

        # Slow path: other venvs we know about.
        import subprocess  # local: not used elsewhere in hot path
        for python, bin_dir in self._venv_pythons:
            if python == sys.executable:
                continue   # already tried above
            try:
                rc = subprocess.run(
                    [python, "-c", f"import {package}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=5,
                ).returncode
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
            if rc == 0:
                self._entries[package] = _NamespaceEntry(
                    package=package, dist_name="", dist_version="",
                    target=package, python=python, bin_dir=bin_dir,
                )
                return True
        return False

    # ── worker lifecycle ────────────────────────────────────────────

    async def _get_worker(self, package: str):
        entry = self._entries.get(package)
        if entry is None and self._auto_register(package):
            entry = self._entries[package]
        if entry is None:
            raise KeyError(package)
        if entry.worker is not None:
            return entry.worker
        async with entry.spawn_lock:
            if entry.worker is not None:
                return entry.worker
            if entry.dispatcher is not None:
                entry.worker = _InProcessWorker(entry.dispatcher)
            else:
                assert entry.target is not None and entry.python is not None
                w = _SubprocessWorker(
                    package, entry.target, entry.python, self._trace_forwarder,
                    ns_bin_dir=entry.bin_dir,
                )
                await w.start()
                entry.worker = w
            return entry.worker

    async def shutdown(self) -> None:
        await asyncio.gather(
            *(e.worker.shutdown() for e in self._entries.values() if e.worker is not None),
            return_exceptions=True,
        )

    # ── dispatch ────────────────────────────────────────────────────

    async def dispatch_unary(self, request: RemoteRequest) -> RemoteResponse:
        try:
            worker = await self._get_worker(request.package)
        except KeyError:
            return RemoteResponse(ok=False, error=RemoteError(
                type="PackageNotLoaded",
                message=f"namespace not loaded: {request.package!r}",
            ))
        return await worker.call_unary(request)

    async def dispatch_stream(self, request: RemoteRequest) -> AsyncIterator[dict[str, Any]]:
        try:
            worker = await self._get_worker(request.package)
        except KeyError:
            yield {"type": "error", "error": RemoteError(
                type="PackageNotLoaded",
                message=f"namespace not loaded: {request.package!r}",
            ).model_dump()}
            return
        async for ev in worker.iter_stream(request):
            yield ev

    async def dispatch_bidi(
        self, request: RemoteRequest, input_iter: AsyncIterator[Any],
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            worker = await self._get_worker(request.package)
        except KeyError:
            yield {"type": "error", "error": RemoteError(
                type="PackageNotLoaded",
                message=f"namespace not loaded: {request.package!r}",
            ).model_dump()}
            return
        async for ev in worker.iter_bidi(request, input_iter):
            yield ev


__all__ = ["NamespaceMultiplexer"]
