"""Build — plan an `agentix build` and see what would be bundled.

`agentix build` packages a project + its declared deps into a deploy-ready
runtime image (uv owns Python, Nix owns system binaries). This view builds the
exact command from a project path + target platform (live), and lists the
`agentix.nix` closures that would be staged into the bundle (real entry-point
introspection — empty in a minimal env, populated in a full workspace).
"""

from __future__ import annotations

import importlib.metadata as md

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Input, Static

_DEFAULT_PLATFORM = "linux/amd64"


class BuildView(Vertical):
    """Interactive `agentix build` planner."""

    _command = ""  # last constructed command (for tests/inspection)

    def compose(self):
        yield Static("Plan an `agentix build`", id="build-title")
        yield Input(value=".", placeholder="project path (a dir with pyproject.toml)", id="build-path")
        yield Static(id="build-cmd")
        yield Static(id="build-info")

    def on_mount(self) -> None:
        self._refresh(".")
        self.query_one("#build-info", Static).update(_info())

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "build-path":
            self._refresh(event.value or ".")

    def _refresh(self, path: str) -> None:
        name = _project_name(path)
        self._command = (
            f"agentix build {path} "
            f"--platform {_DEFAULT_PLATFORM} "
            f"--output dist/{name}.bundle.tar"
        )
        self.query_one("#build-cmd", Static).update(
            Text.assemble(("$ ", "dim"), (self._command, "bold"))
        )


def _project_name(path: str) -> str:
    cleaned = path.rstrip("/")
    base = cleaned.rsplit("/", 1)[-1]
    return base if base and base != "." else "bundle"


def _nix_closures() -> list[str]:
    try:
        eps = md.entry_points(group="agentix.nix")
    except TypeError:  # pragma: no cover - very old importlib.metadata
        eps = md.entry_points().get("agentix.nix", [])  # type: ignore[attr-defined]
    return sorted(ep.name for ep in eps)


def _info() -> Text:
    closures = _nix_closures()
    closure_line = ", ".join(closures) if closures else "none discovered in this environment"
    return Text.assemble(
        ("How it builds\n", "bold"),
        ("  uv owns Python (uv venv + uv sync); Nix owns system binaries\n", "dim"),
        ("  (interpreter + uv + each plugin's default.nix). No uv2nix.\n\n", "dim"),
        ("Nix closures that would be staged\n", "bold"),
        (f"  {closure_line}\n\n", ""),
        ("The host needs only docker + git; every heavy step runs in-container.", "dim italic"),
    )
