import argparse
import importlib
import json

from miles.utils.test_utils import session_verify_runner as runner


def _load_dotted(path):
    module_name, attr_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr_name)


def _build_args(tmp_path):
    values = {
        **runner.SESSION_VERIFY_INVARIANT_ARGS,
        "hf_checkpoint": str(tmp_path / "local-model"),
        "tito_model": "qwen3",
        "tito_allowed_append_roles": ["tool", "user"],
        "rollout_num_gpus_per_engine": 1,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "n_samples_per_prompt": 4,
        "session_verify_cycles": 3,
        "tool_call_failure_mode": "rollback",
        "sglang_reasoning_parser": "qwen3",
        "sglang_tool_call_parser": "qwen25",
        "assistant_text_threshold": 0.1,
        "sglang_expert_parallel_size": 1,
    }
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    return argparse.Namespace(**values)


def test_session_verify_agent_function_paths_are_importable():
    generate = _load_dotted(runner.SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"])
    run_agent = _load_dotted(runner.SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"])

    assert callable(generate)
    assert callable(run_agent)


def test_run_session_verify_cpu_fast_positive_path(tmp_path, monkeypatch):
    import miles.utils.external_utils.command_utils as command_utils

    calls = []

    def fake_execute_train(**kwargs):
        calls.append(kwargs)
        metrics_path = kwargs["extra_env_vars"]["MILES_SESSION_VERIFY_METRICS_PATH"]
        with open(metrics_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "driver_events": ["initial", "append_tool", "rollback"],
                        "had_assistant_mismatch": False,
                    }
                )
                + "\n"
            )

    monkeypatch.setattr(command_utils, "execute_train", fake_execute_train)
    monkeypatch.setattr(runner, "PROMPT_DATA_PATH", str(tmp_path / "session_multi_role_verify.jsonl"))

    args = _build_args(tmp_path)
    runner.run_session_verify(args)

    assert len(calls) == 1
    assert calls[0]["num_gpus_per_node"] == 1
    assert calls[0]["megatron_model_type"] is None
    train_args = calls[0]["train_args"]
    assert f"--hf-checkpoint {tmp_path / 'local-model'}" in train_args
    assert "--custom-generate-function-path miles.utils.test_utils.session_verify_agent.generate" in train_args
    assert "--custom-agent-function-path miles.utils.test_utils.session_verify_agent.run_agent" in train_args
