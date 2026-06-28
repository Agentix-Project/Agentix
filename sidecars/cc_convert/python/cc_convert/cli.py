"""CLI for cc_convert.

Subcommands:

  ``serve``             - run an HTTP server (sidecar) in one of two modes:
                          ``proxy`` (default) — Anthropic-shape in, transparently
                          forwarded to an OpenAI-compatible upstream, Anthropic-shape
                          out. ``rpc`` — pure translation, no upstream call.
  ``translate``         - one-shot JSON-to-JSON conversion in either direction:
                          ``--direction cc-to-oai`` (request side, Anthropic → OpenAI)
                          or ``--direction oai-to-cc`` (response side, OpenAI →
                          Anthropic).

Examples:

  # Run as a sidecar in front of OpenAI
  cc_convert serve --listen 0.0.0.0:8787 \\
      --upstream-url https://api.openai.com/v1/chat/completions \\
      --upstream-key sk-...

  # Run as a sidecar in front of a vLLM / SGLang / DeepSeek backend
  cc_convert serve --upstream-url http://localhost:8000/v1/chat/completions \\
      --log-level debug

  # Pure-translation RPC server (no upstream)
  cc_convert serve --mode rpc --listen 127.0.0.1:8788

  # One-shot: Anthropic request -> OpenAI request
  cat anthropic_req.json | cc_convert translate --direction cc-to-oai > openai_req.json

  # One-shot: OpenAI response -> Anthropic response (need original model name + tool_map)
  cc_convert translate --direction oai-to-cc \\
      --input openai_resp.json --original-model claude-opus-4-7 \\
      --tool-map tool_map.json --output anthropic_resp.json
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import socketserver
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional

import cc_convert

log = logging.getLogger("cc_convert")


# ---------- helpers ----------


def _read_json(path: Optional[str]) -> Any:
    if path and path != "-":
        with open(path) as f:
            return json.load(f)
    return json.load(sys.stdin)


def _write_json(payload: Any, path: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path and path != "-":
        with open(path, "w") as f:
            f.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")


def _normalize_upstream_url(raw: str) -> str:
    """Auto-complete OpenAI-compatible upstream URLs so users don't have to
    remember the exact suffix.

    Accepts (all map to the same thing):
        https://api.openai.com
        https://api.openai.com/
        https://api.openai.com/v1
        https://api.openai.com/v1/
        https://api.openai.com/v1/chat/completions      (verbatim)

    The suffix `/chat/completions` is what OpenAI-style servers (OpenAI,
    vLLM, SGLang, DeepSeek, Together, Anyscale, Fireworks, Moonshot, ...)
    listen on for non-streaming + streaming chat. If the URL already ends
    in that, we leave it. Otherwise we append `/chat/completions`, inserting
    `/v1` if neither `/v1` nor any other obvious version prefix is present.
    """
    url = raw.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    # Already has a version segment like /v1 or /v2 → just append the suffix.
    import re
    if re.search(r"/v\d+$", url):
        return url + "/chat/completions"
    # Bare host or /something else → assume /v1/chat/completions.
    return url + "/v1/chat/completions"


def _path_is_anthropic_messages(path: str) -> bool:
    """True if `path` looks like an Anthropic `messages` endpoint, regardless
    of any prefix the client (or an upstream load balancer) tacked on.

    Accepts:  /v1/messages, /messages, /anthropic/v1/messages,
              /some/prefix/v1/messages?stream=true ...
    Rejects:  /v1/messages/foo (trailing segment), /healthz, /version.
    """
    # Strip query string.
    p = path.split("?", 1)[0].rstrip("/")
    if p.endswith("/v1/messages") or p == "/v1/messages":
        return True
    if p.endswith("/messages") or p == "/messages":
        return True
    return False


def _setup_logging(level: str, fmt: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    if fmt == "json":
        # Minimal JSON formatter: one record per line, easy to grep/jq.
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                    "level": record.levelname.lower(),
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if hasattr(record, "extra_fields"):
                    payload.update(record.extra_fields)
                return json.dumps(payload, ensure_ascii=False)

        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        logging.basicConfig(level=numeric, handlers=[h], force=True)
    else:
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            force=True,
            stream=sys.stderr,
        )


def _log(level: int, msg: str, **fields: Any) -> None:
    """Log with structured extras when JSON format is on."""
    if fields:
        log.log(level, msg, extra={"extra_fields": fields})
    else:
        log.log(level, msg)


# ---------- translate (one-shot) ----------


def cmd_translate(args: argparse.Namespace) -> int:
    payload = _read_json(args.input)
    if args.direction == "cc-to-oai":
        openai_req, tool_map = cc_convert.translate_request(
            payload, target_model=args.target_model, mode=args.compat_mode
        )
        if args.tool_map_out:
            with open(args.tool_map_out, "w") as f:
                json.dump(tool_map, f, indent=2, sort_keys=True)
                f.write("\n")
            log.info("wrote tool_map to %s", args.tool_map_out)
        _write_json(openai_req, args.output)
    elif args.direction == "oai-to-cc":
        tool_map: Dict[str, str] = {}
        if args.tool_map:
            with open(args.tool_map) as f:
                tool_map = json.load(f)
        anthropic = cc_convert.translate_response(
            payload,
            original_model=args.original_model,
            tool_name_map=tool_map,
        )
        _write_json(anthropic, args.output)
    return 0


# ---------- serve ----------


class _Handler(http.server.BaseHTTPRequestHandler):
    """Single handler for both proxy and rpc modes."""

    server_version = "cc_convert/0.1"

    # Injected by build_router below
    mode: str = "proxy"
    compat_mode: str = "pragmatic"
    upstream_url: str = ""
    upstream_key: Optional[str] = None
    auth_passthrough: bool = False
    cc_path: str = "/v1/messages"            # Anthropic-shape entry point
    cc_to_oai_path: str = "/translate/cc-to-oai"
    oai_to_cc_path: str = "/translate/oai-to-cc"
    request_log: bool = True

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default access log
        return

    # ---- routing ----
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send_text(200, "ok")
            return
        if self.path == "/version":
            self._send_json(200, {"name": "cc_convert", "version": cc_convert.__version__})
            return
        self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        rid = uuid.uuid4().hex[:12]
        t0 = time.time()
        try:
            length = int(self.headers.get("content-length") or "0")
            raw = self.rfile.read(length) if length > 0 else b""
            payload = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._anthropic_error(400, "invalid_request_error", f"bad json: {e}")
            self._access_log(rid, 400, t0, route="<bad-json>")
            return

        path_no_query = self.path.split("?", 1)[0].rstrip("/")

        if self.mode == "proxy" and _path_is_anthropic_messages(self.path):
            status = self._handle_proxy(rid, payload)
            self._access_log(rid, status, t0, route="proxy")
        elif self.mode == "rpc" and (
            path_no_query == self.cc_to_oai_path.rstrip("/")
            or path_no_query.endswith("/translate/cc-to-oai")
        ):
            try:
                openai_req, tool_map = cc_convert.translate_request(
                    payload, mode=self.compat_mode
                )
                self._send_json(200, {"openai_request": openai_req, "tool_map": tool_map})
                status = 200
            except Exception as e:  # noqa: BLE001
                self._anthropic_error(400, "invalid_request_error", str(e))
                status = 400
            self._access_log(rid, status, t0, route="rpc cc-to-oai")
        elif self.mode == "rpc" and (
            path_no_query == self.oai_to_cc_path.rstrip("/")
            or path_no_query.endswith("/translate/oai-to-cc")
        ):
            try:
                openai_resp = payload.get("openai_response") or payload
                original_model = payload.get("original_model", "unknown-model")
                tool_map = payload.get("tool_map") or {}
                out = cc_convert.translate_response(
                    openai_resp, original_model=original_model, tool_name_map=tool_map
                )
                self._send_json(200, out)
                status = 200
            except Exception as e:  # noqa: BLE001
                self._anthropic_error(400, "invalid_request_error", str(e))
                status = 400
            self._access_log(rid, status, t0, route="rpc oai-to-cc")
        else:
            hint = ""
            if self.mode == "proxy":
                hint = (
                    f" hint: this proxy accepts POST on any URL ending in "
                    f"/messages or /v1/messages (got {self.path!r})"
                )
            elif self.mode == "rpc":
                hint = (
                    f" hint: this RPC server accepts POST on {self.cc_to_oai_path!r} "
                    f"or {self.oai_to_cc_path!r} (got {self.path!r})"
                )
            self._send_json(
                404, {"error": {"message": f"not found.{hint}", "type": "not_found"}}
            )
            self._access_log(rid, 404, t0, route=self.path)

    # ---- proxy mode handler ----
    def _handle_proxy(self, rid: str, req_value: Dict[str, Any]) -> int:
        stream_mode = bool(req_value.get("stream"))
        original_model = req_value.get("model", "unknown")
        _log(
            logging.DEBUG,
            "request received",
            rid=rid, model=original_model, stream=stream_mode,
            messages=len(req_value.get("messages", [])),
        )

        try:
            openai_req, tool_map = cc_convert.translate_request(
                req_value, mode=self.compat_mode
            )
        except Exception as e:  # noqa: BLE001
            self._anthropic_error(400, "invalid_request_error", str(e))
            return 400

        auth_header = self._build_auth_header()
        upstream_body = json.dumps(openai_req).encode()
        upstream_req = urllib.request.Request(
            self.upstream_url,
            data=upstream_body,
            method="POST",
            headers={"content-type": "application/json"},
        )
        if auth_header:
            upstream_req.add_header("authorization", auth_header)

        _log(
            logging.DEBUG,
            "upstream POST",
            rid=rid, url=self.upstream_url,
            tool_map_size=len(tool_map), body_bytes=len(upstream_body),
        )

        try:
            upstream_resp = urllib.request.urlopen(upstream_req, timeout=600)  # noqa: S310
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            _log(logging.WARNING, "upstream error", rid=rid, status=e.code)
            self._anthropic_error(e.code, "api_error", body_text or "upstream error")
            return e.code
        except urllib.error.URLError as e:
            _log(logging.ERROR, "upstream unreachable", rid=rid, error=str(e))
            self._anthropic_error(502, "api_error", str(e))
            return 502

        if not stream_mode:
            try:
                raw = upstream_resp.read()
                openai_resp = json.loads(raw or b"{}")
                anthropic_resp = cc_convert.translate_response(
                    openai_resp,
                    original_model=original_model,
                    tool_name_map=tool_map,
                )
            except Exception as e:  # noqa: BLE001
                _log(logging.ERROR, "response translation failed", rid=rid, error=str(e))
                self._anthropic_error(502, "api_error", str(e))
                return 502
            self._send_json(200, anthropic_resp)
            return 200

        # ---- streaming ----
        translator = cc_convert.StreamTranslator(original_model, tool_map)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers()

        buffer = b""
        events_sent = 0
        try:
            while True:
                chunk = upstream_resp.read(8192)
                if not chunk:
                    break
                buffer += chunk
                while b"\n\n" in buffer:
                    event_blob, buffer = buffer.split(b"\n\n", 1)
                    events_sent += self._emit_anthropic_events(translator, event_blob)
            # tail flush
            if buffer.strip():
                events_sent += self._emit_anthropic_events(translator, buffer)
            for ev in translator.finish():
                self._write_sse(ev)
                events_sent += 1
        except (BrokenPipeError, ConnectionResetError):
            _log(logging.INFO, "client disconnected mid-stream", rid=rid, events_sent=events_sent)
        try:
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        _log(logging.DEBUG, "stream done", rid=rid, events=events_sent)
        return 200

    def _emit_anthropic_events(self, translator: Any, blob: bytes) -> int:
        count = 0
        for line in blob.splitlines():
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                openai_chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for ev in translator.push(openai_chunk):
                self._write_sse(ev)
                count += 1
        return count

    def _build_auth_header(self) -> Optional[str]:
        if self.auth_passthrough:
            client_auth = self.headers.get("authorization") or self.headers.get("x-api-key")
            if client_auth:
                if not client_auth.startswith("Bearer "):
                    client_auth = f"Bearer {client_auth}"
                return client_auth
            return None
        if self.upstream_key:
            return f"Bearer {self.upstream_key}"
        return None

    # ---- IO helpers ----
    def _write_sse(self, event: Dict[str, Any]) -> None:
        name = event.get("type", "message")
        data = json.dumps(event, separators=(",", ":"))
        self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
        self.wfile.flush()

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _anthropic_error(self, status: int, type_: str, message: str) -> None:
        self._send_json(status, {"type": "error", "error": {"type": type_, "message": message}})

    def _access_log(self, rid: str, status: int, t0: float, route: str) -> None:
        if not self.request_log:
            return
        dur_ms = int((time.time() - t0) * 1000)
        _log(
            logging.INFO,
            f"{self.command} {self.path} {status} ({dur_ms} ms)",
            rid=rid, status=status, route=route, dur_ms=dur_ms,
            remote=self.client_address[0],
        )


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def cmd_serve(args: argparse.Namespace) -> int:
    host, _, port_s = args.listen.rpartition(":")
    if not host:
        host = "0.0.0.0"
    port = int(port_s)

    if args.mode == "proxy" and not args.upstream_url:
        print(
            "error: proxy mode requires --upstream-url (or $CC_CONVERT_UPSTREAM_URL).\n"
            "examples:\n"
            "  cc_convert serve --upstream-url https://api.openai.com --upstream-key sk-...\n"
            "  cc_convert serve --upstream-url http://localhost:8000               # vLLM/SGLang\n"
            "or run in RPC mode (no upstream):\n"
            "  cc_convert serve --mode rpc",
            file=sys.stderr,
        )
        return 2

    # Auto-complete the upstream URL: bare host / /v1 / etc -> .../v1/chat/completions
    normalized_upstream = (
        _normalize_upstream_url(args.upstream_url) if args.upstream_url else ""
    )
    if normalized_upstream and normalized_upstream != args.upstream_url:
        log.info(
            "upstream URL %r -> %r (auto-completed)",
            args.upstream_url, normalized_upstream,
        )
    args.upstream_url = normalized_upstream

    if args.mode == "proxy" and not (args.upstream_key or args.auth_passthrough):
        log.warning(
            "proxy mode running WITHOUT --upstream-key and WITHOUT "
            "--auth-passthrough; the upstream call will be unauthenticated"
        )

    handler_cls = type(
        "_BoundHandler",
        (_Handler,),
        {
            "mode": args.mode,
            "compat_mode": args.compat_mode,
            "upstream_url": args.upstream_url,
            "upstream_key": args.upstream_key,
            "auth_passthrough": args.auth_passthrough,
            "cc_path": args.cc_path,
            "cc_to_oai_path": args.cc_to_oai_path,
            "oai_to_cc_path": args.oai_to_cc_path,
            "request_log": not args.quiet,
        },
    )

    server = _ThreadedServer((host, port), handler_cls)
    log.info(
        "cc_convert serve: mode=%s listen=http://%s:%d cc_path=%s upstream=%s",
        args.mode,
        host,
        port,
        args.cc_path if args.mode == "proxy" else f"{args.cc_to_oai_path} | {args.oai_to_cc_path}",
        args.upstream_url if args.mode == "proxy" else "<none, rpc mode>",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.shutdown()
    return 0


# ---------- argument parser ----------


def _add_global_opts(parser: argparse.ArgumentParser) -> None:
    """Add log-level / log-format / -v / --version. We add these to BOTH the
    top-level parser and every subparser so users don't get tripped up by
    'unrecognized argument' errors when they write `cc_convert serve
    --log-level debug` instead of `cc_convert --log-level debug serve`."""
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error"],
        help="log level (default: $CC_CONVERT_LOG_LEVEL or 'info')",
    )
    parser.add_argument(
        "--log-format",
        default=None,
        choices=["text", "json"],
        help="log format (default: $CC_CONVERT_LOG_FORMAT or 'text')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v = info, -vv = debug (overrides --log-level if higher)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cc_convert {cc_convert.__version__}",
        help="show version and exit",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cc_convert",
        description=(
            "Anthropic <-> OpenAI Chat Completions protocol converter. "
            "Run as a sidecar (`cc_convert serve`) or do one-shot JSON-in / "
            "JSON-out translations (`cc_convert translate`)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    _add_global_opts(p)

    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- translate ----
    pt = sub.add_parser(
        "translate",
        help="one-shot JSON-in / JSON-out conversion in either direction",
    )
    _add_global_opts(pt)
    pt.add_argument(
        "--direction",
        required=True,
        choices=["cc-to-oai", "oai-to-cc"],
        help="cc-to-oai = Anthropic request -> OpenAI request; "
        "oai-to-cc = OpenAI response -> Anthropic response",
    )
    pt.add_argument("-i", "--input", help="input file (default: stdin)")
    pt.add_argument("-o", "--output", help="output file (default: stdout)")
    pt.add_argument(
        "--target-model",
        help="(cc-to-oai) override the model name in the translated request",
    )
    pt.add_argument(
        "--compat-mode",
        default=os.environ.get("CC_CONVERT_COMPAT_MODE", "pragmatic"),
        choices=["pragmatic", "litellm_compat"],
        help="(cc-to-oai) 'pragmatic' (default) collapses single-text content "
        "to a string, drops top_k, etc. — best for real OAI-compat upstreams "
        "(SGLang/vLLM strict mode). 'litellm_compat' is byte-equivalent to "
        "LiteLLM's AnthropicAdapter — use when replacing LiteLLM.",
    )
    pt.add_argument(
        "--original-model",
        help="(oai-to-cc) model name to set on the translated Anthropic response",
    )
    pt.add_argument(
        "--tool-map",
        help="(oai-to-cc) path to a tool-name map JSON saved by a prior translate",
    )
    pt.add_argument(
        "--tool-map-out",
        help="(cc-to-oai) write the tool-name map to this path",
    )
    pt.set_defaults(func=cmd_translate)

    # ---- serve ----
    ps = sub.add_parser(
        "serve",
        help="run as an HTTP server (sidecar) — proxy mode or pure-RPC mode",
    )
    _add_global_opts(ps)
    ps.add_argument(
        "--mode",
        default=os.environ.get("CC_CONVERT_MODE", "proxy"),
        choices=["proxy", "rpc"],
        help=(
            "proxy: terminate Anthropic-shape requests and forward to "
            "--upstream-url; rpc: stateless translation endpoints, no "
            "upstream call (default: $CC_CONVERT_MODE or 'proxy')"
        ),
    )
    ps.add_argument(
        "--listen",
        default=os.environ.get("CC_CONVERT_LISTEN_ADDR", "0.0.0.0:8787"),
        help="host:port to listen on (default: $CC_CONVERT_LISTEN_ADDR or 0.0.0.0:8787)",
    )
    ps.add_argument(
        "--cc-path",
        default=os.environ.get("CC_CONVERT_CC_PATH", "/v1/messages"),
        help="(proxy) the path that receives Anthropic-shape requests "
        "(default: $CC_CONVERT_CC_PATH or /v1/messages)",
    )
    ps.add_argument(
        "--cc-to-oai-path",
        default=os.environ.get("CC_CONVERT_RPC_REQUEST_PATH", "/translate/cc-to-oai"),
        help="(rpc) path for Anthropic-request to OpenAI-request translation",
    )
    ps.add_argument(
        "--oai-to-cc-path",
        default=os.environ.get("CC_CONVERT_RPC_RESPONSE_PATH", "/translate/oai-to-cc"),
        help="(rpc) path for OpenAI-response to Anthropic-response translation",
    )
    ps.add_argument(
        "--upstream-url",
        default=os.environ.get("CC_CONVERT_UPSTREAM_URL"),
        help=(
            "(proxy) full URL of the upstream OpenAI-compatible "
            "/v1/chat/completions endpoint "
            "(default: $CC_CONVERT_UPSTREAM_URL)"
        ),
    )
    ps.add_argument(
        "--upstream-key",
        default=os.environ.get("CC_CONVERT_UPSTREAM_API_KEY"),
        help=(
            "(proxy) bearer token sent to the upstream "
            "(default: $CC_CONVERT_UPSTREAM_API_KEY)"
        ),
    )
    ps.add_argument(
        "--auth-passthrough",
        action="store_true",
        default=os.environ.get("CC_CONVERT_AUTH_PASSTHROUGH") == "1",
        help="forward the CLIENT's Authorization header instead of --upstream-key",
    )
    ps.add_argument(
        "--compat-mode",
        default=os.environ.get("CC_CONVERT_COMPAT_MODE", "pragmatic"),
        choices=["pragmatic", "litellm_compat"],
        help="translation profile: 'pragmatic' (default) collapses "
        "single-text content to a string, drops top_k, etc. — best for "
        "real OAI-compat upstreams (SGLang/vLLM strict mode). "
        "'litellm_compat' is byte-equivalent to LiteLLM's AnthropicAdapter.",
    )
    ps.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-request access logs (errors are still logged)",
    )
    ps.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    # Resolve log options: -vv > -v > subcommand --log-level > top-level > env > default.
    verbose = getattr(args, "verbose", 0)
    if verbose >= 2:
        level = "debug"
    elif verbose >= 1:
        level = "info"
    else:
        level = args.log_level or os.environ.get("CC_CONVERT_LOG_LEVEL") or "info"
    fmt = args.log_format or os.environ.get("CC_CONVERT_LOG_FORMAT") or "text"
    _setup_logging(level, fmt)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130
    except FileNotFoundError as e:
        log.error("file not found: %s", e.filename or e)
        return 2
    except json.JSONDecodeError as e:
        log.error("invalid JSON input: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
