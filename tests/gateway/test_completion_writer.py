"""Tests for the gateway's completion-writer sinks."""

from __future__ import annotations

import json
from pathlib import Path

from agentix.gateway.completion_writer import (
    JsonlCompletionWriter,
    NullCompletionWriter,
)


def test_null_writer_is_noop() -> None:
    w = NullCompletionWriter()
    w.write({"a": 1})
    w.close()


def test_jsonl_writer_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    w = JsonlCompletionWriter(path)
    w.write({"a": 1})
    w.write({"a": 2})
    w.close()
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"a": 2}]


def test_jsonl_writer_batched_flush(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    w = JsonlCompletionWriter(path, flush_every=2)
    w.write({"a": 1})
    w.write({"a": 2})  # triggers flush
    w.write({"a": 3})
    w.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 3
