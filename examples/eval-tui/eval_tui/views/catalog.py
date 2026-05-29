"""Catalog view — the installed Agentix ecosystem at a glance.

Introspects the running environment with `importlib.metadata`: every installed
`agentix*` distribution, plus the `agentix.provider` (sandbox backends) and
`agentix.nix` (agents/datasets that ship a Nix closure) entry points. A live
filter narrows by name / kind / detail. Pure introspection — no Docker — so it
always has something to show and is fully testable headlessly.
"""

from __future__ import annotations

import importlib.metadata as md

from textual.containers import Vertical
from textual.widgets import DataTable, Input, Static


class CatalogView(Vertical):
    """Installed Agentix distributions and entry points, with a live filter."""

    def __init__(self) -> None:
        super().__init__()
        self._rows = discover_catalog()

    def compose(self):
        yield Static(id="catalog-title")
        yield Input(placeholder="filter… (name / kind / detail)", id="catalog-filter")
        yield DataTable(id="catalog-table", zebra_stripes=True, cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        table.add_column("Name", width=38)
        table.add_column("Kind", width=14)
        table.add_column("Version", width=12)
        table.add_column("Detail")
        self._apply_filter("")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "catalog-filter":
            self._apply_filter(event.value)

    def _apply_filter(self, query: str) -> None:
        q = query.strip().lower()
        rows = [row for row in self._rows if not q or q in " ".join(row).lower()]
        table = self.query_one("#catalog-table", DataTable)
        table.clear()
        for name, kind, version, detail in rows:
            table.add_row(name, kind, version, detail)
        self.query_one("#catalog-title", Static).update(
            f"Installed Agentix distributions & entry points  ([b]{len(rows)}[/]/{len(self._rows)})"
        )


def discover_catalog() -> list[tuple[str, str, str, str]]:
    """Return (name, kind, version, detail) rows for the installed ecosystem."""
    rows: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").strip()
        if name.startswith("agentix") and name not in seen:
            seen.add(name)
            summary = (dist.metadata.get("Summary") or "").strip()
            rows.append((name, "package", dist.version or "", summary))

    for kind, group in (("provider", "agentix.provider"), ("nix-closure", "agentix.nix")):
        try:
            entry_points = md.entry_points(group=group)
        except TypeError:  # pragma: no cover - very old importlib.metadata
            entry_points = md.entry_points().get(group, [])  # type: ignore[attr-defined]
        for ep in entry_points:
            rows.append((ep.name, kind, "", ep.value))

    rows.sort(key=lambda r: (r[1], r[0]))
    return rows
