from __future__ import annotations

import asyncio

import agentix.agents.mini_swe_agent as mini_swe


def test_run_success(monkeypatch):
    def fake_run_sync(**kwargs):
        return mini_swe._SyncResult(
            details={"exit_status": "submitted", "submission": "diff --git ..."},
            output_path="/tmp/out.jsonl",
            cost=1.25,
            n_calls=3,
        )

    monkeypatch.setattr(mini_swe, "_run_sync", fake_run_sync)
    result = asyncio.run(mini_swe.run("fix bug"))
    assert result.exit_status == "submitted"
    assert result.submission == "diff --git ..."
    assert result.output_path == "/tmp/out.jsonl"
    assert result.cost == 1.25
    assert result.n_calls == 3


def test_run_error(monkeypatch):
    def fake_run_sync(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mini_swe, "_run_sync", fake_run_sync)
    result = asyncio.run(mini_swe.run("fix bug"))
    assert result.exit_status == "error"
    assert "boom" in str(result.details.get("error", ""))
