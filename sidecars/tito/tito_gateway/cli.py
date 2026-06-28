"""Command-line entrypoint for TITO Gateway."""

from __future__ import annotations

import argparse
import json
import sys

from tito_gateway.config import TITOGatewayConfig
from tito_gateway.gateway import TITOGateway
from tito_gateway.tokenizer import TITOTokenizerType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tito-gateway",
        description="Standalone wrapper around Miles TITO session gateway work.",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_serve_parser(subparsers)
    _add_verify_chat_template_parser(subparsers)
    _add_verify_session_parser(subparsers)
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
        "--apply-chat-template-kwargs",
        default=None,
        help="JSON object forwarded as chat-template kwargs, matching Miles convention.",
    )
    parser.add_argument(
        "--tito-model",
        choices=[item.value for item in TITOTokenizerType],
        default=TITOTokenizerType.DEFAULT.value,
        help="Miles TITO tokenizer family.",
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
    parser.add_argument("--miles-router-timeout", type=float, default=600.0, help="Proxy timeout in seconds.")
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


def _add_verify_chat_template_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "verify-chat-template",
        help="Verify that a chat template is append-only after last user message.",
    )
    parser.add_argument("--template", metavar="PATH")
    parser.add_argument("--model", metavar="MODEL_ID")
    parser.add_argument(
        "--tito-model",
        choices=[item.value for item in TITOTokenizerType],
        default=None,
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        default=["tool"],
        choices=["tool", "user", "system"],
        metavar="ROLE",
    )
    parser.add_argument("--thinking", choices=["off", "on", "both"], default="on")
    parser.add_argument("--chat-template-kwargs", type=json.loads, default=None, metavar="JSON")
    parser.set_defaults(verify_command="chat-template")


def _add_verify_session_parser(subparsers: argparse._SubParsersAction) -> None:
    from miles.utils.test_utils.session_verify_runner import (
        ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD,
        SESSION_VERIFY_INVARIANT_ARGS,
    )

    parser = subparsers.add_parser(
        "verify-session-tito-tokenizer",
        help="Run the optional Miles/SGLang session-server TITO verifier.",
    )
    parser.add_argument("--hf-checkpoint", required=True, help="HuggingFace model ID or local checkpoint path.")
    parser.add_argument("--chat-template-path", default=None, help="Optional fixed chat template path.")
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        default=False,
        help="Miles-compatible flag for applying the chat template in data preprocessing.",
    )
    parser.add_argument(
        "--apply-chat-template-kwargs",
        type=json.loads,
        default={},
        metavar="JSON",
        help="JSON object forwarded as Miles chat-template kwargs.",
    )
    parser.add_argument(
        "--tito-model",
        choices=[item.value for item in TITOTokenizerType],
        required=True,
        help="Miles TITO tokenizer family.",
    )
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        default=["tool"],
        choices=["tool", "user", "system"],
        metavar="ROLE",
    )
    parser.add_argument("--prompt-data", default=SESSION_VERIFY_INVARIANT_ARGS["prompt_data"])
    parser.add_argument("--input-key", default=SESSION_VERIFY_INVARIANT_ARGS["input_key"])
    parser.add_argument(
        "--custom-generate-function-path",
        default=SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"],
    )
    parser.add_argument(
        "--custom-agent-function-path",
        default=SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"],
    )
    parser.add_argument("--backend-url", default=None, help="Optional OpenAI-compatible backend URL for session tests.")
    parser.add_argument("--session-server-ip", default="127.0.0.1")
    parser.add_argument("--session-server-port", type=int, default=30000)
    parser.add_argument("--miles-router-timeout", type=float, default=600.0)
    parser.add_argument("--sglang-reasoning-parser", default=None)
    parser.add_argument("--sglang-tool-call-parser", default=None)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--sglang-expert-parallel-size", type=int, default=1)
    parser.add_argument("--num-rollout", type=int, default=SESSION_VERIFY_INVARIANT_ARGS["num_rollout"])
    parser.add_argument("--rollout-batch-size", type=int, default=SESSION_VERIFY_INVARIANT_ARGS["rollout_batch_size"])
    parser.add_argument(
        "--rollout-max-response-len",
        type=int,
        default=SESSION_VERIFY_INVARIANT_ARGS["rollout_max_response_len"],
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=SESSION_VERIFY_INVARIANT_ARGS["rollout_temperature"],
    )
    parser.add_argument("--global-batch-size", type=int, default=SESSION_VERIFY_INVARIANT_ARGS["global_batch_size"])
    parser.add_argument("--rm-type", default=SESSION_VERIFY_INVARIANT_ARGS["rm_type"])
    parser.add_argument("--actor-num-nodes", type=int, default=1)
    parser.add_argument("--actor-num-gpus-per-node", type=int, default=1)
    parser.add_argument("--n-samples-per-prompt", type=int, default=4)
    parser.add_argument("--session-verify-cycles", type=int, default=3)
    parser.add_argument("--tool-call-failure-mode", default="rollback")
    parser.add_argument(
        "--assistant-text-threshold",
        type=float,
        default=ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD,
        help=(
            "Soft threshold for assistant_text mismatch ratio. "
            f"Default {ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--train-backend",
        choices=["megatron", "fsdp"],
        default=SESSION_VERIFY_INVARIANT_ARGS["train_backend"],
    )
    parser.add_argument("--use-session-server", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-rollout-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ci-test", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--colocate", action=argparse.BooleanOptionalAction, default=None)
    parser.set_defaults(**SESSION_VERIFY_INVARIANT_ARGS)
    parser.set_defaults(verify_command="session-tito-tokenizer")


def _serve(args: argparse.Namespace) -> int:
    config = TITOGatewayConfig.from_cli_values(
        hf_checkpoint=args.hf_checkpoint,
        backend_url=args.backend_url,
        chat_template_path=args.chat_template_path,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        tito_model=args.tito_model,
        tito_allowed_append_roles=args.tito_allowed_append_roles,
        session_server_ip=args.session_server_ip,
        session_server_port=args.session_server_port,
        miles_router_timeout=args.miles_router_timeout,
        backend_probe_candidates=args.backend_probe_candidate,
        backend_probe_timeout=args.backend_probe_timeout,
    )
    TITOGateway(config).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    commands = {"serve", "verify-chat-template", "verify-session-tito-tokenizer", "-h", "--help"}
    if not raw_args or raw_args[0] not in commands:
        raw_args.insert(0, "serve")
    args = parser.parse_args(raw_args)

    if getattr(args, "verify_command", None):
        if args.verify_command == "chat-template":
            from tito_gateway.verify_chat_template import run_from_args

            try:
                return run_from_args(args)
            except Exception as exc:
                print(f"tito-gateway verify-chat-template: error: {exc}", file=sys.stderr)
                return 1
        from tito_gateway.verify_session_tito_tokenizer import run_from_args

        return run_from_args(args)

    try:
        return _serve(args)
    except Exception as exc:
        print(f"tito-gateway: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
