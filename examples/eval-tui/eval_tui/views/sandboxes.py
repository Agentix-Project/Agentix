"""Sandboxes — provider backends and their readiness.

Lists the known SandboxProvider backends and actively probes whether each is
usable *here* (binary on PATH, daemon reachable, SDK installed). Live sessions
and `client.remote(...)` need a built bundle; this screen tells you where you
can run. Probes are real (subprocess / import checks) but degrade gracefully
when nothing is installed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import DataTable, Static

# name -> kind: how the backend ships / is reached.
_BACKENDS = ["docker", "podman", "apptainer", "daytona", "e2b"]


class SandboxesView(Vertical):
    """Provider backends + live readiness."""

    def compose(self):
        yield Static("Sandbox provider backends — readiness on this host", id="sb-title")
        yield DataTable(id="sb-table", zebra_stripes=True, cursor_type="row")
        yield Static(_explainer(), id="sb-explainer")

    def on_mount(self) -> None:
        table = self.query_one("#sb-table", DataTable)
        table.add_column("Backend", width=14)
        table.add_column("Status", width=22)
        table.add_column("Detail")
        for name in _BACKENDS:
            table.add_row(name, Text("· checking…", style="dim"), "", key=name)
        self.run_worker(self._probe(), name="sb-probe", exclusive=True)

    async def _probe(self) -> None:
        for name in _BACKENDS:
            ok, status, detail = await _probe_backend(name)
            style = "bold green" if ok else "yellow" if status == "daemon down" else "dim"
            self._set(name, Text(f"{'✓' if ok else '○'} {status}", style=style), detail)

    def _set(self, name: str, status: Text, detail: str) -> None:
        try:
            table = self.query_one("#sb-table", DataTable)
            columns = list(table.columns.keys())
            table.update_cell(name, columns[1], status, update_width=False)
            table.update_cell(name, columns[2], detail, update_width=False)
        except Exception:
            pass


async def _probe_backend(name: str) -> tuple[bool, str, str]:
    if name == "docker":
        return await _probe_daemon("docker")
    if name == "podman":
        return await _probe_daemon("podman")
    if name == "apptainer":
        binary = shutil.which("apptainer") or shutil.which("singularity")
        return (bool(binary), "installed" if binary else "not installed", binary or "apptainer/singularity not on PATH")
    if name in ("daytona", "e2b"):
        spec = importlib.util.find_spec(name)
        key = os.environ.get(f"{name.upper()}_API_KEY")
        if spec is None:
            return (False, "SDK not installed", f"pip install agentix-provider-{name}")
        return (bool(key), "ready" if key else "SDK installed", "API key set" if key else f"set {name.upper()}_API_KEY")
    return (False, "unknown", "")


async def _probe_daemon(binary: str) -> tuple[bool, str, str]:
    path = shutil.which(binary)
    if not path:
        return (False, "not installed", f"{binary} not on PATH")
    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "info", "--format", "{{.ServerVersion}}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=8)
    except Exception:
        return (False, "daemon down", f"{binary} installed; daemon unreachable")
    if proc.returncode == 0:
        return (True, "ready", f"server {out.decode(errors='replace').strip()}")
    return (False, "daemon down", f"{binary} installed; daemon unreachable")


def _explainer() -> Text:
    return Text.assemble(
        ("Two stable APIs: ", "dim"),
        ("provider.session(config)", "bold"),
        (" yields a sandbox; ", "dim"),
        ("sandbox.remote(fn, …)", "bold"),
        (" runs an importable callable inside it.\n", "dim"),
        ("A live session needs a bundle from ", "dim"),
        ("agentix build", "bold"),
        (" overlaid on a task image.", "dim"),
    )
