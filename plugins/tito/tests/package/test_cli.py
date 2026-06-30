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
