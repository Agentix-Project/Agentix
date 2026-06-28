"""Regenerate golden fixtures by running each input through LiteLLM (oracle).

Writes:
    tests/fixtures/requests/openai_<name>.json       (translated request)
    tests/fixtures/requests/tool_map_<name>.json     (LiteLLM tool name map)
    tests/fixtures/responses/anthropic_<name>.json   (translated response — hand-built)
    tests/fixtures/streams/anthropic_<name>.jsonl    (translated event stream — hand-built)

For requests, LiteLLM is the source of truth.

For responses and streams, LiteLLM exposes only an `AnthropicStreamWrapper`
that needs a full LiteLLM ModelResponse to drive — we do NOT depend on that.
Instead, we either:
  - use the Rust translator itself to produce the golden (it has been
    unit-tested against LiteLLM's behaviour for each rule), OR
  - leave the response/stream goldens stubbed out for now and rely on the
    Rust-side unit tests we already wrote.

This script currently regenerates request goldens only. Run it again after
each behavioural change to the request translator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _import_litellm():
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
    except ImportError as e:
        print(f"litellm not installed: {e}", file=sys.stderr)
        print("Install: pip install 'litellm>=1.0'", file=sys.stderr)
        sys.exit(1)
    return LiteLLMAnthropicMessagesAdapter()


def regen_requests(adapter) -> int:
    req_dir = ROOT / "requests"
    count = 0
    for input_path in sorted(req_dir.glob("anthropic_*.json")):
        name = input_path.stem.removeprefix("anthropic_")
        anthropic_req = json.loads(input_path.read_text())
        try:
            openai_req, tool_map = adapter.translate_anthropic_to_openai(anthropic_req)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        out_req = req_dir / f"openai_{name}.json"
        out_map = req_dir / f"tool_map_{name}.json"
        out_req.write_text(json.dumps(openai_req, indent=2, sort_keys=True) + "\n")
        out_map.write_text(json.dumps(tool_map or {}, indent=2, sort_keys=True) + "\n")
        count += 1
        print(f"  ✓ {name}")
    return count


def main() -> None:
    adapter = _import_litellm()
    n = regen_requests(adapter)
    print(f"\nregenerated {n} request goldens via LiteLLM")


if __name__ == "__main__":
    main()
