import pytest
from agentix.tito.cli import build_parser, main


def test_cli_top_level_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "serve" in capsys.readouterr().out


def test_cli_serve_help(capsys):
    with pytest.raises(SystemExit):
        main(["serve", "--help"])
    out = capsys.readouterr().out
    assert "--hf-checkpoint" in out
    assert "--tito-model" in out
    assert "--tito-allowed-append-roles" in out


def test_cli_serve_parses_args():
    args = build_parser().parse_args(
        [
            "serve",
            "--hf-checkpoint", "Qwen/Qwen3-0.6B",
            "--backend-url", "http://127.0.0.1:8000",
            "--tito-model", "qwen3",
            "--tito-allowed-append-roles", "tool", "user",
        ]
    )
    assert args.command == "serve"
    assert args.hf_checkpoint == "Qwen/Qwen3-0.6B"
    assert args.tito_model == "qwen3"
    assert args.tito_allowed_append_roles == ["tool", "user"]


def test_cli_tito_model_choices_are_qwen3_and_default():
    args = build_parser().parse_args(["serve", "--hf-checkpoint", "X"])
    assert args.tito_model == "default"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--hf-checkpoint", "X", "--tito-model", "glm47"])


def test_cli_backend_url_is_repeatable_for_a_pool():
    args = build_parser().parse_args(
        ["serve", "--hf-checkpoint", "X",
         "--backend-url", "http://r1:8000", "--backend-url", "http://r2:8000",
         "--routing-policy", "round_robin"]
    )
    assert args.backend_url == ["http://r1:8000", "http://r2:8000"]
    assert args.routing_policy == "round_robin"


def test_cli_values_plumb_pool_and_trust_flags():
    from agentix.tito.config import TITOGatewayConfig

    cfg = TITOGatewayConfig.from_cli_values(
        hf_checkpoint="X",
        backend_url=None,
        backend_urls=["http://r1:8000", "http://r2:8000"],
        routing_policy="round_robin",
        trust_remote_code=True,
        chat_template_path=None,
        tito_model="default",
        tito_allowed_append_roles=["tool"],
        session_server_ip="127.0.0.1",
        session_server_port=30000,
        router_timeout=1.0,
    )
    assert cfg.backend_urls == ("http://r1:8000", "http://r2:8000")
    assert cfg.routing_policy == "round_robin"
    assert cfg.trust_remote_code is True
    assert cfg.as_session_args().trust_remote_code is True


def test_cli_trust_remote_code_defaults_off():
    args = build_parser().parse_args(["serve", "--hf-checkpoint", "X"])
    assert args.trust_remote_code is False


def test_cli_backend_kind_defaults_to_sglang_and_validates_choices():
    args = build_parser().parse_args(["serve", "--hf-checkpoint", "X"])
    assert args.backend_kind == "sglang"
    args = build_parser().parse_args(["serve", "--hf-checkpoint", "X", "--backend-kind", "vllm"])
    assert args.backend_kind == "vllm"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--hf-checkpoint", "X", "--backend-kind", "tgi"])
