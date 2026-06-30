"""Command-line entrypoint for the Agentix TITO gateway."""

from __future__ import annotations

import argparse
import sys

from .config import TITOGatewayConfig
from .gateway import TITOGateway
from .tokenizer import TITOTokenizerType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentix-tito",
        description="Agentix TITO gateway — token-in-token-out session-recording proxy.",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_serve_parser(subparsers)
    return parser


def _add_serve_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    serve = subparsers.add_parser("serve", help="Start the TITO gateway server.")
    _add_serve_arguments(serve)
    return serve


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hf-checkpoint", required=True, help="HuggingFace model ID or local checkpoint path.")
    parser.add_argument("--backend-url", default=None, help="OpenAI-compatible backend URL to proxy to.")
    parser.add_argument("--chat-template-path", default=None, help="Optional fixed chat template path.")
    parser.add_argument(
        "--tito-model",
        choices=[item.value for item in TITOTokenizerType],
        default=TITOTokenizerType.DEFAULT.value,
        help="TITO tokenizer family (qwen3, or default for the tokenizer's own template).",
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        choices=["tool", "user", "system"],
        default=["tool"],
        help="Roles allowed after an assistant turn; tool is the default.",
    )
    parser.add_argument("--session-server-ip", default="127.0.0.1", help="Gateway bind host.")
    parser.add_argument("--session-server-port", type=int, default=30000, help="Gateway bind port.")
    parser.add_argument("--router-timeout", type=float, default=600.0, help="Proxy timeout in seconds.")
    parser.add_argument(
        "--backend-probe-candidate",
        action="append",
        default=None,
        metavar="URL",
        help="Local backend URL candidate to probe after explicit and environment URLs; repeatable.",
    )
    parser.add_argument(
        "--backend-probe-timeout",
        type=float,
        default=0.25,
        help="Per-endpoint backend probe timeout in seconds.",
    )


def _serve(args: argparse.Namespace) -> int:
    config = TITOGatewayConfig.from_cli_values(
        hf_checkpoint=args.hf_checkpoint,
        backend_url=args.backend_url,
        chat_template_path=args.chat_template_path,
        tito_model=args.tito_model,
        tito_allowed_append_roles=args.tito_allowed_append_roles,
        session_server_ip=args.session_server_ip,
        session_server_port=args.session_server_port,
        router_timeout=args.router_timeout,
        backend_probe_candidates=args.backend_probe_candidate,
        backend_probe_timeout=args.backend_probe_timeout,
    )
    TITOGateway(config).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args[0] not in {"serve", "-h", "--help"}:
        raw_args.insert(0, "serve")
    args = parser.parse_args(raw_args)
    try:
        return _serve(args)
    except Exception as exc:  # noqa: BLE001
        print(f"agentix-tito: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
