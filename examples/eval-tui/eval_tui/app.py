"""Agentix TUI — a modern Textual control room for Agentix.

A tabbed shell that surfaces each Agentix area as its own view: an Overview
landing dashboard, live **Rollouts** over `agentix.runner`, a plugin
**Catalog**, **Sandboxes** readiness, a **Build** planner, and live
**Observability**. See `DESIGN.md` for the rubrics this iterates against.

Run a no-Docker demo with `agentix-eval-tui --demo 40`, point it at real
adapters like `agentix-run`, or launch it bare to browse the Catalog.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .models import RunSpec
from .views import (
    BuildView,
    CatalogView,
    ObservabilityView,
    OverviewView,
    RolloutsView,
    SandboxesView,
)


class AgentixTUI(App):
    """Tabbed control room for Agentix."""

    TITLE = "Agentix"
    SUB_TITLE = "agent ↔ environment control room"

    CSS = """
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; }

    #rollouts-summary {
        height: 3;
        padding: 0 2;
        content-align: left middle;
        border: round $primary;
        background: $panel;
    }
    #rollouts-body { height: 1fr; }
    #rollouts-table { width: 3fr; height: 1fr; border: round $primary; }
    #rollouts-side { width: 2fr; height: 1fr; }
    #rollouts-detail { height: 2fr; border: round $primary; padding: 0 1; }
    #rollouts-log { height: 3fr; border: round $primary; padding: 0 1; }

    #catalog-title { height: 1; padding: 0 1; }
    #catalog-filter { margin: 0 1; }
    #catalog-table { height: 1fr; border: round $primary; }

    #placeholder { text-align: center; width: auto; }

    #ov-banner { height: auto; padding: 1 2; content-align: center middle; text-align: center; }
    #ov-cards { height: 7; padding: 0 1; }
    .ov-card {
        width: 1fr;
        height: 5;
        border: round $primary;
        padding: 1 1;
        margin: 0 1;
        content-align: center middle;
        text-align: center;
    }
    #ov-hints { height: auto; padding: 1 2; }

    #sb-title { height: 1; padding: 0 1; }
    #sb-table { height: 1fr; border: round $primary; }
    #sb-explainer { height: auto; padding: 1 1; }

    #obs-title { height: 1; padding: 0 1; }
    #obs-body { height: 1fr; }
    #obs-trace { width: 1fr; height: 1fr; border: round $primary; padding: 0 1; }
    #obs-log { width: 1fr; height: 1fr; border: round $primary; padding: 0 1; }

    #build-title { height: 1; padding: 0 1; }
    #build-path { margin: 0 1; }
    #build-cmd { height: auto; padding: 1 2; }
    #build-info { height: 1fr; padding: 0 2; }
    """

    BINDINGS = [
        ("1", "show_tab('overview')", "Overview"),
        ("2", "show_tab('rollouts')", "Rollouts"),
        ("3", "show_tab('catalog')", "Catalog"),
        ("4", "show_tab('sandboxes')", "Sandboxes"),
        ("5", "show_tab('build')", "Build"),
        ("6", "show_tab('observability')", "Obs"),
        ("s", "save", "Save"),
        ("t", "cycle_theme", "Theme"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, *, rollout_spec: RunSpec | None = None) -> None:
        super().__init__()
        self._spec = rollout_spec

    def action_save(self) -> None:
        """Persist the collected rollout summaries to a JSON file in the cwd."""
        view = self.query_one(RolloutsView)
        payload = view.export_payload()
        if not payload["rollouts"]:
            self.notify("nothing to save yet — run some rollouts first", severity="warning", timeout=3)
            return
        path = view.export_to(Path.cwd() / "agentix-rollouts.json")
        self.notify(f"saved {len(payload['rollouts'])} rollouts → {path.name}", timeout=3)

    def on_mount(self) -> None:
        # Build the set of cycleable themes: a best-effort branded theme (falls
        # back gracefully if the Textual version's theme API differs) plus a few
        # always-available built-ins. Cycle with `t`.
        self._themes: list[str] = []
        try:
            from textual.theme import Theme

            self.register_theme(
                Theme(name="agentix", primary="#cc785c", secondary="#a45a45", accent="#e08a6d", dark=True)
            )
            self._themes.append("agentix")
        except Exception:
            pass
        for name in ("tokyo-night", "gruvbox", "nord", "dracula", "textual-light"):
            if name in self.available_themes:
                self._themes.append(name)
        if self._themes:
            self.theme = self._themes[0]

    def action_cycle_theme(self) -> None:
        if not self._themes:
            return
        index = self._themes.index(self.theme) if self.theme in self._themes else -1
        self.theme = self._themes[(index + 1) % len(self._themes)]
        self.notify(f"theme: {self.theme}", timeout=2)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="overview"):
            with TabPane("Overview", id="overview"):
                yield OverviewView()
            with TabPane("Rollouts", id="rollouts"):
                yield RolloutsView(self._spec)
            with TabPane("Catalog", id="catalog"):
                yield CatalogView()
            with TabPane("Sandboxes", id="sandboxes"):
                yield SandboxesView()
            with TabPane("Build", id="build"):
                yield BuildView()
            with TabPane("Observability", id="observability"):
                yield ObservabilityView()
        yield Footer()

    def action_show_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab
