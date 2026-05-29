"""Rollouts view — a live batch-rollout dashboard over `agentix.runner`.

Per-instance phase grid (pending -> setup -> agent -> scoring -> PASS/FAIL/skip/
error), a live summary bar, an event log, and a **detail pane** that drills into
the highlighted instance. Phase transitions are observed by wrapping the
dataset/agent adapters (see `.._adapters`), so `agentix.runner` is unchanged.
With no `RunSpec` the view shows an idle state (other tabs still work).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agentix.runner import Rollout, run_rollouts
from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, RichLog, Static

from .._adapters import TracingAgent, TracingDataset, instance_id
from ..models import RunSpec

_PHASE = {
    "pending": ("· pending", "dim"),
    "setup": ("⟳ setup", "yellow"),
    "agent": ("⟳ agent", "cyan"),
    "score": ("⟳ scoring", "magenta"),
}


class RolloutsView(Vertical):
    """Live dashboard for a batch of rollouts."""

    def __init__(self, spec: RunSpec | None) -> None:
        super().__init__()
        self._spec = spec
        self._instances = list(spec.instances) if spec else []
        self._t0 = 0.0
        self._done = 0
        self._resolved = 0
        self._failed = 0
        self._running = 0
        self._results: dict[str, Rollout] = {}
        self._phase_of: dict[str, str] = {}
        self._selected: str | None = None
        self._detail_text = ""  # plain text of the detail pane (for tests/inspection)

    def compose(self):
        yield Static(id="rollouts-summary")
        with Horizontal(id="rollouts-body"):
            yield DataTable(id="rollouts-table", zebra_stripes=True, cursor_type="row")
            with Vertical(id="rollouts-side"):
                yield Static(id="rollouts-detail")
                yield RichLog(id="rollouts-log", markup=True, wrap=True)

    def on_mount(self) -> None:
        table = self.query_one("#rollouts-table", DataTable)
        table.add_column("Instance", key="iid", width=34)
        table.add_column("Status", key="status", width=16)
        table.add_column("Time", key="time", width=9)
        table.add_column("Result", key="result")
        self.query_one("#rollouts-detail", Static).update(
            Text("Select an instance to inspect.", style="dim")
        )

        if self._spec is None or not self._instances:
            self.query_one("#rollouts-summary", Static).update(
                Text("No run configured — launch with --demo N, or --dataset/--agent/--bundle.", style="dim")
            )
            self.query_one("#rollouts-log", RichLog).write("[dim]Idle — the Catalog tab works without a run.[/]")
            return

        for inst in self._instances:
            iid = instance_id(inst)
            self._phase_of[iid] = "pending"
            table.add_row(iid, _phase("pending"), "", "", key=iid)
        self._t0 = time.monotonic()
        self._refresh_summary()
        self.run_worker(self._drive(), name="rollouts-drive", exclusive=True)

    async def _drive(self) -> None:
        assert self._spec is not None
        log = self.query_one("#rollouts-log", RichLog)
        log.write(f"[b]▶ starting[/] {len(self._instances)} rollouts · concurrency {self._spec.n_concurrent}")
        rollouts = await run_rollouts(
            dataset=TracingDataset(self._spec.dataset, self._on_phase),
            agent=TracingAgent(self._spec.agent, self._on_phase),
            provider=self._spec.provider,
            bundle=self._spec.bundle,
            model=self._spec.model,
            instances=self._instances,
            n_concurrent=self._spec.n_concurrent,
            on_result=self._on_result,
        )
        dt = time.monotonic() - self._t0
        log.write(f"[b green]■ done[/] — {self._resolved}/{len(rollouts)} resolved in {dt:.1f}s")

    def _on_phase(self, iid: str, phase: str) -> None:
        if phase == "setup":
            self._running += 1
        self._phase_of[iid] = phase
        self._set_cell(iid, "status", _phase(phase))
        if iid == self._selected:
            self._render_detail(iid)
        self._refresh_summary()

    def _on_result(self, rollout: Rollout) -> None:
        self._running = max(0, self._running - 1)
        self._done += 1
        self._results[rollout.instance_id] = rollout
        if rollout.resolved:
            self._resolved += 1
            label, style, result = "✓ PASS", "bold green", "resolved"
        elif rollout.error:
            self._failed += 1
            label, style, result = "✗ error", "bold red", rollout.error[:48]
        elif rollout.skipped:
            self._failed += 1
            label, style, result = f"⊘ {rollout.skipped}", "yellow", rollout.skipped
        else:
            self._failed += 1
            label, style, result = "✗ FAIL", "red", "unresolved"
        self._set_cell(rollout.instance_id, "status", Text(label, style=style))
        self._set_cell(rollout.instance_id, "time", f"{rollout.duration_s:.1f}s")
        self._set_cell(rollout.instance_id, "result", result)
        self.query_one("#rollouts-log", RichLog).write(
            f"[{style}]{label}[/] {rollout.instance_id} · {rollout.duration_s:.1f}s"
        )
        if rollout.instance_id == self._selected:
            self._render_detail(rollout.instance_id)
        self._refresh_summary()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key.value
        if key is not None:
            self._selected = str(key)
            self._render_detail(self._selected)

    def _render_detail(self, iid: str) -> None:
        text = self._detail_for(iid)
        self._detail_text = text.plain
        self.query_one("#rollouts-detail", Static).update(text)

    def _detail_for(self, iid: str) -> Text:
        rollout = self._results.get(iid)
        if rollout is None:
            phase = self._phase_of.get(iid, "pending")
            label, style = _PHASE.get(phase, (phase, "white"))
            return Text.assemble((f"{iid}\n\n", "bold"), ("status: ", "dim"), (label, style))
        lines: list[tuple[str, str] | str] = [(f"{iid}\n\n", "bold")]
        verdict = "PASS" if rollout.resolved else (rollout.skipped or ("error" if rollout.error else "FAIL"))
        verdict_style = "bold green" if rollout.resolved else "bold red"
        lines += [("verdict: ", "dim"), (f"{verdict}\n", verdict_style)]
        lines += [("duration: ", "dim"), (f"{rollout.duration_s:.1f}s\n", "")]
        lines += [("agent exit: ", "dim"), (f"{rollout.agent_exit}\n", "")]
        lines += [("patch: ", "dim"), (f"{len(rollout.patch)} bytes\n", "")]
        if rollout.score:
            lines += [("\nscore:\n", "dim")]
            for k, v in list(rollout.score.items())[:8]:
                lines += [(f"  {k}: ", "dim"), (f"{_short(v)}\n", "")]
        if rollout.error:
            lines += [("\nerror:\n", "dim"), (rollout.error[:300], "red")]
        return Text.assemble(*lines)

    def _set_cell(self, iid: str, column: str, value: object) -> None:
        try:
            self.query_one("#rollouts-table", DataTable).update_cell(iid, column, value, update_width=False)
        except Exception:
            pass

    def _refresh_summary(self) -> None:
        total = len(self._instances)
        dt = max(1e-6, time.monotonic() - self._t0)
        rate = self._done / dt * 60
        text = Text.assemble(
            (_bar(self._done, total), "bold"),
            "   ",
            (f"{self._done}/{total} done", "bold"),
            "    ",
            ("✓ ", "dim"),
            (str(self._resolved), "bold green"),
            "    ",
            ("✗ ", "dim"),
            (str(self._failed), "bold red"),
            "    ",
            ("⟳ ", "dim"),
            (f"{self._running} running", "bold cyan"),
            "    ",
            (f"{rate:.1f}/min", "dim"),
        )
        self.query_one("#rollouts-summary", Static).update(text)

    def export_payload(self) -> dict[str, Any]:
        """A JSON-friendly snapshot of the run: the per-instance `Rollout`
        summaries collected so far plus a small aggregate. This is the unit an
        RL/eval loop persists for offline analysis or replay."""
        return {
            "summary": {
                "total": len(self._instances),
                "done": self._done,
                "resolved": self._resolved,
                "failed": self._failed,
            },
            "rollouts": [r.to_dict() for r in self._results.values()],
        }

    def export_to(self, path: Path) -> Path:
        """Write `export_payload()` to `path` as pretty JSON; returns the path."""
        path.write_text(json.dumps(self.export_payload(), indent=2))
        return path


def _short(value: object, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _phase(phase: str) -> Text:
    label, style = _PHASE.get(phase, (phase, "white"))
    return Text(label, style=style)


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "─" * width + "]"
    filled = round(width * done / total)
    return "[" + "█" * filled + "·" * (width - filled) + "]"
