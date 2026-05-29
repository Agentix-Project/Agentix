"""A live Textual dashboard for Agentix batch rollouts.

Drives `agentix.runner.run_rollouts(...)` and renders each instance's progress
in real time: a per-instance grid (pending -> setup -> agent -> scoring ->
PASS/FAIL/skip/error), a live summary bar (done / resolved / failed / running
+ throughput), and an event log. Phase transitions are observed by wrapping
the dataset/agent adapters (see `_adapters`), so the runner needs no changes.

Run a no-Docker synthetic demo with `agentix-eval-tui --demo 40`, or point it
at real adapters exactly like `agentix-run` (see `cli`).
"""

from __future__ import annotations

import time
from typing import Any

from agentix.runner import Rollout, run_rollouts
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from ._adapters import TracingAgent, TracingDataset, instance_id

_PHASE = {
    "pending": ("· pending", "dim"),
    "setup": ("⟳ setup", "yellow"),
    "agent": ("⟳ agent", "cyan"),
    "score": ("⟳ scoring", "magenta"),
}


class EvalDashboard(App):
    """Live dashboard for a batch of rollouts."""

    CSS = """
    Screen { layout: vertical; background: $surface; }
    #summary {
        height: 3;
        padding: 0 2;
        content-align: left middle;
        border: round $primary;
        background: $panel;
    }
    #body { height: 1fr; }
    #table { width: 3fr; height: 1fr; border: round $primary; }
    #log { width: 2fr; height: 1fr; border: round $primary; padding: 0 1; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Dark/Light"),
    ]

    def __init__(
        self,
        *,
        dataset: Any,
        agent: Any,
        provider: Any,
        bundle: str,
        model: str | None = None,
        instances: list[dict[str, Any]] | None = None,
        n_concurrent: int = 4,
        run_title: str = "Agentix · eval",
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._agent = agent
        self._provider = provider
        self._bundle = bundle
        self._model = model
        self._n_concurrent = n_concurrent
        self._instances = list(instances) if instances is not None else list(dataset.instances())
        self._run_title = run_title
        self._t0 = 0.0
        self._done = 0
        self._resolved = 0
        self._failed = 0
        self._running = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="summary")
        with Horizontal(id="body"):
            yield DataTable(id="table", zebra_stripes=True, cursor_type="row")
            yield RichLog(id="log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = self._run_title
        self.sub_title = f"{len(self._instances)} instances · concurrency {self._n_concurrent}"
        table = self.query_one("#table", DataTable)
        table.add_column("Instance", key="iid", width=34)
        table.add_column("Status", key="status", width=16)
        table.add_column("Time", key="time", width=9)
        table.add_column("Result", key="result")
        for inst in self._instances:
            iid = instance_id(inst)
            table.add_row(iid, _phase("pending"), "", "", key=iid)
        self._t0 = time.monotonic()
        self._refresh_summary()
        self.run_worker(self._drive(), name="drive", exclusive=True)

    async def _drive(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write(f"[b]▶ starting[/] {len(self._instances)} rollouts · concurrency {self._n_concurrent}")
        rollouts = await run_rollouts(
            dataset=TracingDataset(self._dataset, self._on_phase),
            agent=TracingAgent(self._agent, self._on_phase),
            provider=self._provider,
            bundle=self._bundle,
            model=self._model,
            instances=self._instances,
            n_concurrent=self._n_concurrent,
            on_result=self._on_result,
        )
        dt = time.monotonic() - self._t0
        log.write(f"[b green]■ done[/] — {self._resolved}/{len(rollouts)} resolved in {dt:.1f}s")
        self.sub_title = f"done · {self._resolved}/{len(rollouts)} resolved · {dt:.1f}s"

    # ── callbacks (run on the app event loop, via the driver worker) ──

    def _on_phase(self, iid: str, phase: str) -> None:
        if phase == "setup":
            self._running += 1
        self._set_cell(iid, "status", _phase(phase))
        self._refresh_summary()

    def _on_result(self, rollout: Rollout) -> None:
        self._running = max(0, self._running - 1)
        self._done += 1
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
        self.query_one("#log", RichLog).write(
            f"[{style}]{label}[/] {rollout.instance_id} · {rollout.duration_s:.1f}s"
        )
        self._refresh_summary()

    # ── helpers ──

    def _set_cell(self, iid: str, column: str, value: Any) -> None:
        try:
            self.query_one("#table", DataTable).update_cell(iid, column, value, update_width=False)
        except Exception:
            pass  # row may have been filtered out

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
        self.query_one("#summary", Static).update(text)

    def action_toggle_dark(self) -> None:
        try:
            self.dark = not self.dark  # type: ignore[attr-defined]
        except Exception:
            pass


def _phase(phase: str) -> Text:
    label, style = _PHASE.get(phase, (phase, "white"))
    return Text(label, style=style)


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "─" * width + "]"
    filled = round(width * done / total)
    return "[" + "█" * filled + "·" * (width - filled) + "]"
