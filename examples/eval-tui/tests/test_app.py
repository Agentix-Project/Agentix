"""Headless pilot tests for the Agentix TUI (no Docker)."""

from __future__ import annotations

from eval_tui.app import AgentixTUI
from eval_tui.demo import DemoAgent, DemoDataset, DemoProvider
from eval_tui.models import RunSpec
from eval_tui.views.build import BuildView
from eval_tui.views.catalog import CatalogView, discover_catalog
from eval_tui.views.observability import ObservabilityView
from eval_tui.views.overview import OverviewView
from eval_tui.views.rollouts import RolloutsView
from textual.widgets import DataTable, Input, TabbedContent


def _demo_spec(n: int = 8) -> RunSpec:
    dataset = DemoDataset(n, seed=3, dur_scale=0.03)
    return RunSpec(
        dataset=dataset,
        agent=DemoAgent(),
        provider=DemoProvider(),
        bundle="demo",
        instances=dataset.instances(),
        n_concurrent=4,
    )


async def test_tui_runs_demo_and_lists_catalog() -> None:
    n = 8
    app = AgentixTUI(rollout_spec=_demo_spec(n))
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        view = app.query_one(RolloutsView)
        assert view._done == n
        assert view._resolved + view._failed == n

        catalog = app.query_one("#catalog-table", DataTable)
        assert catalog.row_count >= 1  # at least agentixx / agentix-runner are installed


async def test_tui_idle_without_spec_does_not_crash() -> None:
    app = AgentixTUI(rollout_spec=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(RolloutsView)._done == 0
        assert app.query_one("#catalog-table", DataTable).row_count >= 1


async def test_rollouts_drilldown_shows_instance_detail() -> None:
    app = AgentixTUI(rollout_spec=_demo_spec(6))
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        view = app.query_one(RolloutsView)
        view.query_one("#rollouts-table", DataTable).move_cursor(row=0)
        await pilot.pause()

        assert "demo__task-000" in view._detail_text
        assert "verdict" in view._detail_text  # a finished instance renders its verdict


async def test_overview_dashboard_shows_ecosystem_counts() -> None:
    app = AgentixTUI(rollout_spec=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        overview = app.query_one(OverviewView)
        assert overview._counts["packages"] >= 1  # at least agentixx is installed


async def test_sandboxes_view_lists_backends() -> None:
    app = AgentixTUI(rollout_spec=None)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()  # readiness probe worker
        await pilot.pause()
        table = app.query_one("#sb-table", DataTable)
        assert table.row_count == 5  # docker, podman, apptainer, daytona, e2b


async def test_observability_demo_streams_events() -> None:
    app = AgentixTUI(rollout_spec=None)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query_one(ObservabilityView)._emitted > 0


async def test_build_view_constructs_command_from_path() -> None:
    app = AgentixTUI(rollout_spec=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        build = app.query_one(BuildView)
        build.query_one("#build-path", Input).value = "examples/run-swe-rollouts"
        await pilot.pause()
        assert build._command.startswith("agentix build examples/run-swe-rollouts")
        assert "run-swe-rollouts.bundle.tar" in build._command


async def test_tab_keybinding_switches_active_tab() -> None:
    app = AgentixTUI(rollout_spec=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("3")  # -> Catalog
        await pilot.pause()
        assert app.query_one(TabbedContent).active == "catalog"


async def test_catalog_filter_narrows_rows() -> None:
    app = AgentixTUI(rollout_spec=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        catalog = app.query_one(CatalogView)
        table = app.query_one("#catalog-table", DataTable)
        assert table.row_count == len(catalog._rows)
        catalog.query_one("#catalog-filter", Input).value = "runner"
        await pilot.pause()
        assert 1 <= table.row_count <= len(catalog._rows)


def test_discover_catalog_finds_agentix_distributions() -> None:
    rows = discover_catalog()
    names = {name for name, *_ in rows}
    assert any(n.startswith("agentix") for n in names)
