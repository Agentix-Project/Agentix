import os

from miles.utils.test_utils import session_verify_runner as runner
from tito_gateway.cli import build_parser, main


def test_cli_top_level_help(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    assert "Standalone wrapper around Miles TITO" in capsys.readouterr().out


def test_cli_serve_help(capsys):
    try:
        main(["serve", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    assert "--hf-checkpoint" in capsys.readouterr().out


def test_cli_verify_session_returns_clear_dependency_error(capsys, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.example:8888")
    monkeypatch.setenv("HTTPS_PROXY", "http://secure-proxy.example:8888")

    code = main(
        [
            "verify-session-tito-tokenizer",
            "--hf-checkpoint",
            "Qwen/Qwen3-4B",
            "--tito-model",
            "qwen3",
            "--tito-allowed-append-roles",
            "tool",
            "user",
            "--sglang-reasoning-parser",
            "qwen3",
            "--sglang-tool-call-parser",
            "qwen25",
            "--rollout-num-gpus-per-engine",
            "1",
        ]
    )

    assert code == 2
    assert "requires the optional Miles/SGLang training stack" in capsys.readouterr().out
    assert os.environ.get("http_proxy") == "http://proxy.example:8888"
    assert os.environ.get("HTTPS_PROXY") == "http://secure-proxy.example:8888"


def test_cli_verify_session_help(capsys):
    try:
        main(["verify-session-tito-tokenizer", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert "--hf-checkpoint" in out
    assert "--rollout-num-gpus-per-engine" in out
    assert "--assistant-text-threshold" in out
    assert f"Default {runner.ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD}" in out


def test_cli_verify_session_default_threshold_matches_runner_constant():
    args = build_parser().parse_args(
        [
            "verify-session-tito-tokenizer",
            "--hf-checkpoint",
            "Qwen/Qwen3-4B",
            "--tito-model",
            "qwen3",
        ]
    )

    assert args.assistant_text_threshold == runner.ASSISTANT_TEXT_MISMATCH_RATIO_THRESHOLD


def test_cli_verify_session_parses_representative_miles_style_invocation(tmp_path):
    prompt_data = tmp_path / "session.jsonl"
    template_path = tmp_path / "template.jinja"
    args = build_parser().parse_args(
        [
            "verify-session-tito-tokenizer",
            "--hf-checkpoint",
            "Qwen/Qwen3-4B",
            "--chat-template-path",
            str(template_path),
            "--apply-chat-template",
            "--apply-chat-template-kwargs",
            '{"enable_thinking": false}',
            "--tito-model",
            "qwen3",
            "--tito-allowed-append-roles",
            "tool",
            "user",
            "--prompt-data",
            str(prompt_data),
            "--input-key",
            "messages",
            "--backend-url",
            "http://127.0.0.1:8000",
            "--session-server-ip",
            "127.0.0.1",
            "--session-server-port",
            "31000",
            "--miles-router-timeout",
            "12.5",
            "--sglang-reasoning-parser",
            "qwen3",
            "--sglang-tool-call-parser",
            "qwen25",
            "--rollout-num-gpus-per-engine",
            "2",
            "--sglang-expert-parallel-size",
            "4",
            "--num-rollout",
            "2",
            "--rollout-batch-size",
            "8",
            "--rollout-max-response-len",
            "4096",
            "--rollout-temperature",
            "0.2",
            "--global-batch-size",
            "32",
            "--actor-num-nodes",
            "2",
            "--actor-num-gpus-per-node",
            "4",
            "--n-samples-per-prompt",
            "6",
            "--session-verify-cycles",
            "5",
            "--tool-call-failure-mode",
            "skip",
            "--assistant-text-threshold",
            "0.4",
            "--train-backend",
            "fsdp",
            "--custom-generate-function-path",
            "pkg.verify.generate",
            "--custom-agent-function-path",
            "pkg.verify.run_agent",
            "--rm-type",
            "random",
            "--use-session-server",
            "--debug-rollout-only",
            "--ci-test",
            "--colocate",
        ]
    )

    assert args.verify_command == "session-tito-tokenizer"
    assert args.chat_template_path == str(template_path)
    assert args.apply_chat_template is True
    assert args.apply_chat_template_kwargs == {"enable_thinking": False}
    assert args.backend_url == "http://127.0.0.1:8000"
    assert args.session_server_port == 31000
    assert args.miles_router_timeout == 12.5
    assert args.assistant_text_threshold == 0.4

    train_args = runner.namespace_to_train_args(args)
    assert f"--prompt-data {prompt_data}" in train_args
    assert "--input-key messages" in train_args
    assert "--rollout-batch-size 8" in train_args
    assert "--n-samples-per-prompt 6" in train_args
    assert "--rollout-max-response-len 4096" in train_args
    assert "--rollout-temperature 0.2" in train_args
    assert "--global-batch-size 32" in train_args
    assert "--custom-generate-function-path pkg.verify.generate" in train_args
    assert "--custom-agent-function-path pkg.verify.run_agent" in train_args
    assert "--session-verify-cycles 5" in train_args
    assert "--tool-call-failure-mode skip" in train_args
    assert "--rollout-num-gpus-per-engine 2" in train_args
    assert "--sglang-expert-parallel-size 4" in train_args
    assert "--actor-num-nodes 2" in train_args
    assert "--actor-num-gpus-per-node 4" in train_args
    assert "--train-backend fsdp" in train_args
    assert "--use-session-server" in train_args
    assert "--debug-rollout-only" in train_args
    assert "--ci-test" in train_args
    assert "--colocate" in train_args


def test_cli_verify_chat_template_help(capsys):
    try:
        main(["verify-chat-template", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert "--template" in out
    assert "--tito-allowed-append-roles" in out


def test_cli_verify_chat_template_runs_real_verifier(tmp_path, capsys):
    template = tmp_path / "simple.jinja"
    template.write_text(
        "{%- for message in messages -%}"
        "{{ '<|' + message['role'] + '|>' + (message.get('content') or '') }}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{ '<|assistant|>' }}{%- endif -%}"
    )

    code = main(["verify-chat-template", "--template", str(template), "--thinking", "off"])

    captured = capsys.readouterr()
    assert code == 0
    assert "Verdict: PASS - template IS append-only" in captured.out
