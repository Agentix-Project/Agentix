"""Non-streaming round-trip runner.

For each fixture under playground/requests/, send the full pipeline:

    1. anthropic_request.json   ← source (copied verbatim from requests/)
    2. oai_request.json         ← cc_convert.translate_request() output
    3. oai_response.json        ← upstream server's raw response
    4. anthropic_response.json  ← cc_convert.translate_response() output

All four files for one fixture land in:

    playground/runs/<UTC-timestamp>/<fixture_name>/

Plus a `meta.json` (status, latency, http_status, error if any) and a
top-level `_summary.json` with all fixtures' status at a glance.

This is the ONLY script you need to run to see the complete round-trip
data. No streaming, no synthetic, no offline mocks — just the real pipeline.

Usage:
    python playground/run_roundtrip.py --upstream http://YOUR_UPSTREAM_HOST:8000

If --model is omitted, /v1/models is probed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cc_convert

PLAYGROUND = Path(__file__).resolve().parent
REQ_DIR = PLAYGROUND / "requests"


def _opener_no_proxy() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def derive_chat_url(base: str) -> str:
    b = base.rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    if re.search(r"/v\d+$", b):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


def base_from_chat(chat_url: str) -> str:
    for suf in ("/v1/chat/completions", "/chat/completions"):
        if chat_url.endswith(suf):
            return chat_url[: -len(suf)]
    return chat_url.rstrip("/")


def probe_models(base: str) -> List[str]:
    for path in ("/v1/models", "/models"):
        try:
            r = _opener_no_proxy().open(base + path, timeout=5)
            data = json.loads(r.read())
        except (urllib.error.URLError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [m["id"] if isinstance(m, dict) and "id" in m else str(m) for m in data["data"]]
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            return [
                m if isinstance(m, str) else (m.get("id") if isinstance(m, dict) else str(m))
                for m in data["models"]
            ]
        if isinstance(data, list):
            return [
                m if isinstance(m, str) else (m.get("id", str(m)) if isinstance(m, dict) else str(m))
                for m in data
            ]
    return []


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run_one(
    fx_name: str,
    anthropic_req: Dict[str, Any],
    chat_url: str,
    api_key: Optional[str],
    target_model: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run the full pipeline for one fixture; return meta dict."""
    fx_dir = out_dir / fx_name
    fx_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save the original Anthropic request verbatim.
    write_json(fx_dir / "1_anthropic_request.json", anthropic_req)

    # 2) Translate to OpenAI shape + save.
    openai_req, tool_map = cc_convert.translate_request(
        anthropic_req, target_model=target_model
    )
    # Force non-streaming.
    openai_req.pop("stream", None)
    openai_req.pop("stream_options", None)
    write_json(fx_dir / "2_oai_request.json", openai_req)
    if tool_map:
        write_json(fx_dir / "tool_map.json", tool_map)

    meta: Dict[str, Any] = {
        "fixture": fx_name,
        "upstream": chat_url,
        "model_sent": target_model,
        "original_model": anthropic_req.get("model"),
        "tool_map_size": len(tool_map),
        "stream": False,
    }

    # 3) POST to the upstream.
    body = json.dumps(openai_req).encode()
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    t0 = time.time()
    try:
        req = urllib.request.Request(chat_url, data=body, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        resp = _opener_no_proxy().open(req, timeout=300)
        raw = resp.read()
        openai_resp = json.loads(raw or b"{}")
        meta.update(
            {
                "status": "ok",
                "http_status": resp.status,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        )
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        meta.update(
            {
                "status": "http_error",
                "http_status": e.code,
                "error": body_text[:500],
                "latency_ms": int((time.time() - t0) * 1000),
            }
        )
        write_json(fx_dir / "meta.json", meta)
        (fx_dir / "error.txt").write_text(f"HTTP {e.code}\n\n{body_text}\n")
        return meta
    except urllib.error.URLError as e:
        meta.update(
            {
                "status": "url_error",
                "error": str(e),
                "latency_ms": int((time.time() - t0) * 1000),
            }
        )
        write_json(fx_dir / "meta.json", meta)
        (fx_dir / "error.txt").write_text(f"URLError: {e}\n")
        return meta
    except Exception as e:  # noqa: BLE001
        meta.update(
            {"status": "exception", "error": f"{type(e).__name__}: {e}",
             "latency_ms": int((time.time() - t0) * 1000)}
        )
        write_json(fx_dir / "meta.json", meta)
        (fx_dir / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
        return meta

    # 3') Save raw OAI response.
    write_json(fx_dir / "3_oai_response.json", openai_resp)

    # 4) Translate back to Anthropic shape + save.
    try:
        anthropic_resp = cc_convert.translate_response(
            openai_resp,
            original_model=anthropic_req.get("model", "claude-opus-4-7"),
            tool_name_map=tool_map,
        )
        write_json(fx_dir / "4_anthropic_response.json", anthropic_resp)
    except Exception as e:  # noqa: BLE001
        meta.update({"status": "reverse_translate_error", "error": str(e)})
        (fx_dir / "error.txt").write_text(f"reverse translate: {type(e).__name__}: {e}\n")

    write_json(fx_dir / "meta.json", meta)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="upstream URL (bare host OK; /v1/chat/completions auto-appended)")
    ap.add_argument("--model", default=None, help="model name to send (auto-probes /v1/models if omitted)")
    ap.add_argument("--api-key", default=os.environ.get("CC_CONVERT_UPSTREAM_API_KEY"))
    ap.add_argument("--only", help="only run fixtures whose name contains this substring")
    args = ap.parse_args()

    chat_url = derive_chat_url(args.upstream)
    base = base_from_chat(chat_url)

    model = args.model
    if not model:
        models = probe_models(base)
        print(f"[probe] /v1/models: {models}", file=sys.stderr)
        if not models:
            print("[error] no model name and probe returned empty; pass --model", file=sys.stderr)
            return 2
        model = models[0]
    print(f"[ok] model={model!r} url={chat_url}", file=sys.stderr)

    fixtures = sorted(REQ_DIR.glob("*.json"))
    if args.only:
        fixtures = [f for f in fixtures if args.only in f.stem]
    if not fixtures:
        print(f"[error] no fixtures matched in {REQ_DIR}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = PLAYGROUND / "runs" / stamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"[ok] outputs -> {out}\n", file=sys.stderr)

    summary = []
    for fx_path in fixtures:
        name = fx_path.stem
        req = json.loads(fx_path.read_text())
        print(f"--- {name} (max_tokens={req.get('max_tokens')}) ---", file=sys.stderr)
        m = run_one(name, req, chat_url, args.api_key, model, out)
        summary.append(m)
        ok = m.get("status") == "ok"
        mark = "✓" if ok else "✗"
        info = (
            f"http={m.get('http_status')} ms={m.get('latency_ms')}"
            if ok
            else f"status={m.get('status')} http={m.get('http_status')} error={(m.get('error') or '')[:80]}"
        )
        print(f"   {mark} {info}", file=sys.stderr)

    write_json(out / "_summary.json", summary)

    print("\n=== Summary ===", file=sys.stderr)
    print(f"{'Fixture':36s} {'Status':18s} HTTP   ms", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    for m in summary:
        ms = m.get("latency_ms")
        ms_s = f"{ms}" if ms is not None else "—"
        print(
            f"{m['fixture']:36s} {m.get('status', '?'):18s} "
            f"{str(m.get('http_status') or '—'):>5s}   {ms_s}",
            file=sys.stderr,
        )
    print(f"\nfull data: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
