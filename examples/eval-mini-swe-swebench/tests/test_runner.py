"""Lightweight tests for the eval orchestrator.

These tests do not boot any sandboxes — they verify the pieces of
`runner.py` that we can exercise without Docker / apptainer / real
SWE-bench images:

  * CLI parses correctly.
  * `_instance_image` produces the right SWE-bench image name on
    different architectures and namespaces.
  * `_load_instances` honours `--limit` and `--instances`.
  * `_serialise_score` handles dataclass-shaped and dict-shaped score
    objects.
  * `_summarise` produces the expected aggregate.

A full end-to-end run is documented in the README and expects
Docker (locally) or apptainer (on a Ray cluster).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import runner  # type: ignore[import-not-found]


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["runner.py"])
    args = runner.parse_args()
    assert args.dataset == "princeton-nlp/SWE-bench_Verified"
    assert args.split == "test"
    assert args.deployment == "local"
    assert args.limit == 1  # smoke default
    assert args.concurrency == 1


def test_instance_image_swap_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_image = "swebench/sweb.eval.x86_64.django__django-12345:latest"

    class _Spec:
        instance_image_key = fake_image

    def fake_make_spec(_inst, namespace=None):  # type: ignore[no-untyped-def]
        return _Spec()

    monkeypatch.setattr("swebench.harness.test_spec.test_spec.make_test_spec", fake_make_spec)

    instance = {"instance_id": "django__django-12345"}
    # x86_64 host
    assert (
        runner._instance_image(instance, namespace="swebench", tag="latest", arch="x86_64")
        == "swebench/sweb.eval.x86_64.django__django-12345:latest"
    )
    # arm64 host
    assert (
        runner._instance_image(instance, namespace="swebench", tag="latest", arch="arm64")
        == "swebench/sweb.eval.arm64.django__django-12345:latest"
    )
    # custom tag
    assert (
        runner._instance_image(
            instance, namespace="swebench", tag="v0.1", arch="x86_64"
        )
        == "swebench/sweb.eval.x86_64.django__django-12345:v0.1"
    )


def test_load_instances_limit_and_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_rows = [
        {"instance_id": "a"},
        {"instance_id": "b"},
        {"instance_id": "c"},
        {"instance_id": "d"},
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda *_args, **_kw: fake_rows)
    assert [
        r["instance_id"]
        for r in runner._load_instances("ds", split="test", limit=2, allow_ids=None)
    ] == ["a", "b"]
    assert [
        r["instance_id"]
        for r in runner._load_instances("ds", split="test", limit=0, allow_ids=["c"])
    ] == ["c"]


def test_serialise_score_handles_dataclass_and_dict() -> None:
    @dataclass
    class _Score:
        resolved: bool
        log: str = "x"

    assert runner._serialise_score(_Score(True)) == {"resolved": True, "log": "x"}
    assert runner._serialise_score({"resolved": False, "extra": 1}) == {
        "value": "{'resolved': False, 'extra': 1}"
    }


def test_summarise_aggregates_correctly(tmp_path: Path) -> None:
    rows = [
        {
            "instance_id": "a",
            "agent": {"ok": True},
            "score": {"ok": True, "result": {"resolved": True}},
        },
        {
            "instance_id": "b",
            "agent": {"ok": True},
            "score": {"ok": True, "result": {"resolved": False}},
        },
        {"instance_id": "c", "agent": {"ok": False}, "score": None},
    ]
    runner._summarise(rows, out_dir=tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary == {
        "total": 3,
        "agent_ok": 2,
        "scored_ok": 2,
        "resolved": 1,
        "resolved_rate": 1 / 3,
    }
