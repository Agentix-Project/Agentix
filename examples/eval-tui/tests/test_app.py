"""Headless pilot test: the dashboard runs a synthetic batch to completion."""

from __future__ import annotations

from eval_tui.app import EvalDashboard
from eval_tui.demo import DemoAgent, DemoDataset, DemoProvider
from textual.widgets import DataTable


async def test_dashboard_runs_demo_to_completion() -> None:
    n = 8
    app = EvalDashboard(
        dataset=DemoDataset(n, seed=3, dur_scale=0.03),
        agent=DemoAgent(),
        provider=DemoProvider(),
        bundle="demo",
        n_concurrent=4,
    )
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        table = app.query_one("#table", DataTable)
        assert table.row_count == n
        assert app._done == n
        assert app._resolved + app._failed == n
