import importlib
import sys

import pytest

from miles._upstream_loader import UpstreamModuleLoadError, load_upstream_module


TARGET_MODULES = {
    "miles.utils.external_utils.command_utils",
    "miles.rollout.generate_hub.agentic_tool_call",
    "miles.rollout.base_types",
}


def _clear_target_modules():
    for name in list(sys.modules):
        if name in TARGET_MODULES or name.startswith("_tito_gateway_upstream_"):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def clear_target_modules():
    _clear_target_modules()
    yield
    _clear_target_modules()


def _write_fake_upstream(root):
    command_utils = root / "miles" / "utils" / "external_utils"
    command_utils.mkdir(parents=True)
    (command_utils / "command_utils.py").write_text(
        "SOURCE = 'fake-upstream-command-utils'\n"
        "def exec_command(*args, **kwargs):\n"
        "    return ('upstream-exec', args, kwargs)\n"
        "def execute_train(*args, **kwargs):\n"
        "    return ('upstream-train', args, kwargs)\n"
    )

    generate_hub = root / "miles" / "rollout" / "generate_hub"
    generate_hub.mkdir(parents=True)
    (generate_hub / "agentic_tool_call.py").write_text(
        "SOURCE = 'fake-upstream-agentic-tool-call'\n"
        "async def generate(input):\n"
        "    return ('upstream-generate', input)\n"
        "def _add_arguments(parser):\n"
        "    parser.add_argument('--fake-upstream-agentic-flag')\n"
        "generate.add_arguments = _add_arguments\n"
    )

    rollout = root / "miles" / "rollout"
    (rollout / "base_types.py").write_text(
        "SOURCE = 'fake-upstream-base-types'\n"
        "class GenerateFnInput:\n"
        "    ORIGIN = 'upstream'\n"
        "class GenerateFnOutput:\n"
        "    ORIGIN = 'upstream'\n"
    )


def test_exact_name_wrappers_delegate_to_later_upstream_sys_path(tmp_path, monkeypatch):
    fake_root = tmp_path / "fake_upstream"
    _write_fake_upstream(fake_root)
    monkeypatch.setattr(sys, "path", [*sys.path, str(fake_root)])
    importlib.invalidate_caches()

    command_utils = importlib.import_module("miles.utils.external_utils.command_utils")
    agentic_tool_call = importlib.import_module("miles.rollout.generate_hub.agentic_tool_call")
    base_types = importlib.import_module("miles.rollout.base_types")

    assert command_utils.SOURCE == "fake-upstream-command-utils"
    assert command_utils.exec_command("x")[0] == "upstream-exec"
    assert command_utils.execute_train(train_args="--debug")[0] == "upstream-train"

    assert agentic_tool_call.SOURCE == "fake-upstream-agentic-tool-call"
    assert callable(agentic_tool_call.generate)
    assert callable(agentic_tool_call.generate.add_arguments)

    assert base_types.SOURCE == "fake-upstream-base-types"
    assert base_types.GenerateFnInput.ORIGIN == "upstream"
    assert base_types.GenerateFnOutput.ORIGIN == "upstream"


def test_loader_considers_upstream_candidate_under_shared_install_root(tmp_path, monkeypatch):
    shared_root = tmp_path / "site-packages"
    upstream = shared_root / "miles" / "utils" / "external_utils"
    upstream.mkdir(parents=True)
    (upstream / "command_utils.py").write_text("SOURCE = 'shared-root-upstream'\n")

    local_file = shared_root / "tito_gateway_wrapper" / "miles" / "utils" / "external_utils" / "command_utils.py"
    local_file.parent.mkdir(parents=True)
    local_file.write_text("SOURCE = 'local-wrapper'\n")

    monkeypatch.setattr(sys, "path", [str(shared_root)])

    module = load_upstream_module("miles.utils.external_utils.command_utils", str(local_file))

    assert module is not None
    assert module.SOURCE == "shared-root-upstream"


def test_present_upstream_import_failure_is_not_masked(tmp_path, monkeypatch):
    fake_root = tmp_path / "broken_upstream"
    command_utils = fake_root / "miles" / "utils" / "external_utils"
    command_utils.mkdir(parents=True)
    (command_utils / "command_utils.py").write_text("raise RuntimeError('upstream exploded')\n")
    monkeypatch.setattr(sys, "path", [*sys.path, str(fake_root)])
    importlib.invalidate_caches()

    with pytest.raises(UpstreamModuleLoadError, match="Found upstream candidate") as exc_info:
        importlib.import_module("miles.utils.external_utils.command_utils")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "upstream exploded" in str(exc_info.value.__cause__)


def test_command_utils_fallback_remains_clear_without_upstream():
    command_utils = importlib.import_module("miles.utils.external_utils.command_utils")

    with pytest.raises(command_utils.MissingMilesTrainingStackError, match="not bundled with tito-gateway"):
        command_utils.execute_train(train_args="--debug", num_gpus_per_node=1, megatron_model_type=None)
