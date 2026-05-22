from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SWE_PLUGIN = ROOT / "plugins" / "datasets" / "swe"
sys.path.insert(0, str(SWE_PLUGIN))

import agentix.plugins.datasets.swe as swe  # noqa: E402
from agentix.plugins.datasets.swe import swe as swe_module  # noqa: E402


def test_swe_public_exports() -> None:
    assert swe.prepare_env is swe_module.prepare_env
    assert swe.score is swe_module.score


def test_new_file_only_test_patch_reset_preserves_setup_commit() -> None:
    modified, touched = swe_module._extract_test_patch_paths(
        "diff --git a/tests/test_new.py b/tests/test_new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_new.py\n"
        "@@ -0,0 +1 @@\n"
        "+def test_new(): pass\n"
    )

    assert modified == []
    assert touched == ["tests/test_new.py"]


def test_modified_test_patch_reset_is_left_to_swebench() -> None:
    modified, touched = swe_module._extract_test_patch_paths(
        "diff --git a/tests/test_existing.py b/tests/test_existing.py\n"
        "--- a/tests/test_existing.py\n"
        "+++ b/tests/test_existing.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    assert modified == ["tests/test_existing.py"]
    assert touched == ["tests/test_existing.py"]


def test_eval_export_commands_update_subprocess_env() -> None:
    env: dict[str, str] = {}

    assert swe_module._apply_export_command("export LANG=en_US.UTF-8", env) is True
    assert env == {"LANG": "en_US.UTF-8"}
    assert (
        swe_module._apply_export_command("sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen", env) is False
    )


def test_status_names_fallback_to_test_modules() -> None:
    assert swe_module._test_directives_from_status(
        {
            "FAIL_TO_PASS": '["test_a (auth_tests.test_templates.AuthTemplateTests)"]',
            "PASS_TO_PASS": (
                '["tests/test_example.py::test_ok", '
                '"test_b (auth_tests.test_templates.AuthTemplateTests)"]'
            ),
        }
    ) == ["auth_tests.test_templates.AuthTemplateTests", "tests/test_example.py::test_ok"]


def test_status_fallback_reports_unmapped_docstring_names() -> None:
    directives, missing = swe_module._test_directives_from_status_with_missing(
        {
            "FAIL_TO_PASS": '["Named URLs should be reversible"]',
            "PASS_TO_PASS": '["test_b (auth_tests.test_templates.AuthTemplateTests)"]',
        }
    )

    assert directives == ["auth_tests.test_templates.AuthTemplateTests"]
    assert missing == 1


def test_eval_plan_uses_testspec_without_swe_repo_specs() -> None:
    spec = SimpleNamespace(
        instance_id="repo__repo-1",
        eval_script_list=[
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "cd /testbed",
            "export FOO=bar",
            "git config --global --add safe.directory /testbed",
            "cd /testbed",
            "git status",
            "git show",
            "git -c core.fileMode=false diff abc123",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            "python -m pip install -e . --verbose",
            "git checkout abc123 tests/test_example.py",
            "git apply -v - <<'EOF_114329324912'\npatch\nEOF_114329324912",
            ": '>>>>> Start Test Output'",
            "pytest -rA tests/test_example.py",
            ": '>>>>> End Test Output'",
            "git checkout abc123 tests/test_example.py",
        ],
    )

    plan = swe_module._eval_plan_from_test_spec(
        spec,
        {
            "repo": "pallets/flask",
            "base_commit": "abc123",
            "test_patch": "diff --git a/tests/test_example.py b/tests/test_example.py\n",
        },
    )

    assert plan.setup_commands == ["export FOO=bar"]
    assert plan.install_commands == ["python -m pip install -e . --verbose"]
    assert plan.test_cmd == "pytest -rA"
    assert plan.test_files == ["tests/test_example.py"]


def test_local_score_uses_testspec_gold_lists() -> None:
    spec = SimpleNamespace(
        repo="pallets/flask",
        FAIL_TO_PASS=["tests/test_app.py::test_fixed"],
        PASS_TO_PASS=["tests/test_app.py::test_still_ok"],
    )
    log = (
        ">>>>> Applied Patch\n"
        ">>>>> Start Test Output\n"
        "PASSED tests/test_app.py::test_fixed\n"
        "PASSED tests/test_app.py::test_still_ok\n"
        ">>>>> End Test Output\n"
    )

    entry, fixes = swe_module._score_eval_log(spec=spec, patch="diff", patch_applied=True, combined_log=log)

    assert fixes == []
    assert entry["resolved"] is True
    assert entry["tests_status"]["FAIL_TO_PASS"]["success"] == ["tests/test_app.py::test_fixed"]
    assert entry["tests_status"]["PASS_TO_PASS"]["failure"] == []


def test_local_score_treats_missing_required_test_as_failure() -> None:
    spec = SimpleNamespace(
        repo="pallets/flask",
        FAIL_TO_PASS=["tests/test_app.py::test_fixed"],
        PASS_TO_PASS=[],
    )
    log = (
        ">>>>> Applied Patch\n"
        ">>>>> Start Test Output\n"
        "PASSED tests/test_app.py::test_other\n"
        ">>>>> End Test Output\n"
    )

    entry, _ = swe_module._score_eval_log(spec=spec, patch="diff", patch_applied=True, combined_log=log)

    assert entry["resolved"] is False
    assert entry["tests_status"]["FAIL_TO_PASS"]["failure"] == ["tests/test_app.py::test_fixed"]


def test_empty_pytest_param_alias_is_grading_only_normalization() -> None:
    entry = {
        "resolved": False,
        "tests_status": {
            "FAIL_TO_PASS": {"success": [], "failure": []},
            "PASS_TO_PASS": {
                "success": ["pkg/test_mod.py::test_roundtrip[x]"],
                "failure": ["pkg/test_mod.py::test_roundtrip[]"],
            },
        },
    }
    log = "pkg/test_mod.py::test_roundtrip[unit3] PASSED\npkg/test_mod.py::test_roundtrip[x] PASSED\n"

    fixed, fixes = swe_module._apply_grading_normalizations(entry, log)

    assert fixes == ["pytest:empty-param-id-alias"]
    assert fixed["resolved"] is True
    assert fixed["tests_status"]["PASS_TO_PASS"]["failure"] == []
    assert "pkg/test_mod.py::test_roundtrip[]" in fixed["tests_status"]["PASS_TO_PASS"]["success"]


def test_empty_pytest_param_alias_requires_unique_passed_alias() -> None:
    entry = {
        "resolved": False,
        "tests_status": {
            "FAIL_TO_PASS": {"success": [], "failure": []},
            "PASS_TO_PASS": {"success": [], "failure": ["pkg/test_mod.py::test_roundtrip[]"]},
        },
    }
    log = (
        "pkg/test_mod.py::test_roundtrip[unit0] PASSED\n"
        "pkg/test_mod.py::test_roundtrip[param1] PASSED\n"
    )

    fixed, fixes = swe_module._apply_grading_normalizations(entry, log)

    assert fixes == []
    assert fixed["resolved"] is False
    assert fixed["tests_status"]["PASS_TO_PASS"]["failure"] == ["pkg/test_mod.py::test_roundtrip[]"]


def test_compatibility_predicates_are_repo_version_scoped() -> None:
    assert swe_module._is_old_astropy({"repo": "astropy/astropy", "version": "3.1"}) is True
    assert swe_module._is_old_astropy({"repo": "astropy/astropy", "version": "5.1"}) is False
    assert swe_module._is_old_astropy({"repo": "django/django", "version": "3.1"}) is False


def test_requests_tarpit_patch_is_scoped_to_legacy_address(tmp_path: Path) -> None:
    target = tmp_path / "requests/packages/urllib3/util"
    target.mkdir(parents=True)
    connection = target / "connection.py"
    connection.write_text(
        "def create_connection(address):\n"
        "    host, port = address\n"
        "    err = None\n"
        "    return host, port, err\n"
    )

    assert swe_module._patch_requests_tarpit_connect_timeout(tmp_path) is True
    text = connection.read_text()

    assert 'if host == "10.255.255.1":' in text
    assert "raise socket.timeout" in text
    assert swe_module._patch_requests_tarpit_connect_timeout(tmp_path) is False


def test_requests_httpbin_retry_shim_is_scoped_to_transient_httpbin_failures(tmp_path: Path) -> None:
    env = {"PYTHONPATH": "/existing"}

    assert swe_module._install_requests_httpbin_retry_shim(tmp_path, env) is True

    shim_dir = tmp_path / "requests-httpbin-retry"
    shim = shim_dir / "sitecustomize.py"
    text = shim.read_text()

    assert env["PYTHONPATH"] == os.pathsep.join([str(shim_dir), "/existing"])
    assert 'host == "httpbin.org" or host.endswith(".httpbin.org")' in text
    assert "response.status_code not in (502, 503, 504)" in text
    assert "from requests.adapters import HTTPAdapter" in text
    assert "HTTPAdapter.send = _agentix_send_with_httpbin_retry" in text


def test_requests_httpbin_retry_shim_does_not_duplicate_pythonpath(tmp_path: Path) -> None:
    env: dict[str, str] = {}

    assert swe_module._install_requests_httpbin_retry_shim(tmp_path, env) is True
    assert swe_module._install_requests_httpbin_retry_shim(tmp_path, env) is True

    shim_dir = str(tmp_path / "requests-httpbin-retry")
    assert env["PYTHONPATH"].split(os.pathsep) == [shim_dir]


def test_requests_httpbin_retry_shim_retries_only_transient_httpbin_statuses(monkeypatch) -> None:
    responses = [_FakeResponse(502), _FakeResponse(503), _FakeResponse(200)]
    queued_responses = list(responses)
    calls: list[str] = []

    def send(self, request, **kwargs):
        calls.append(request.url)
        return queued_responses.pop(0)

    adapter_cls = _exec_requests_httpbin_retry_shim(monkeypatch, send)

    response = adapter_cls().send(SimpleNamespace(url="https://httpbin.org/post"))

    assert response.status_code == 200
    assert calls == ["https://httpbin.org/post"] * 3
    assert responses[0].closed is True
    assert responses[1].closed is True
    assert responses[2].closed is False


def test_requests_httpbin_retry_shim_does_not_retry_asserted_or_other_hosts(monkeypatch) -> None:
    calls: list[str] = []

    def send(self, request, **kwargs):
        calls.append(request.url)
        return _FakeResponse(401 if "httpbin.org" in request.url else 502)

    adapter_cls = _exec_requests_httpbin_retry_shim(monkeypatch, send)

    asserted = adapter_cls().send(SimpleNamespace(url="https://httpbin.org/status/401"))
    other_host = adapter_cls().send(SimpleNamespace(url="https://example.com/status/502"))

    assert asserted.status_code == 401
    assert other_host.status_code == 502
    assert calls == ["https://httpbin.org/status/401", "https://example.com/status/502"]


def test_requests_httpbin_retry_shim_retries_httpbin_connection_errors(monkeypatch) -> None:
    calls: list[str] = []

    def send(self, request, **kwargs):
        calls.append(request.url)
        if len(calls) == 1:
            raise _FakeConnectionError("temporary disconnect")
        return _FakeResponse(200)

    adapter_cls = _exec_requests_httpbin_retry_shim(monkeypatch, send)

    response = adapter_cls().send(SimpleNamespace(url="https://httpbin.org/get"))

    assert response.status_code == 200
    assert calls == ["https://httpbin.org/get", "https://httpbin.org/get"]


class _FakeConnectionError(Exception):
    pass


class _FakeTimeout(Exception):
    pass


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _exec_requests_httpbin_retry_shim(monkeypatch, send):
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    class FakeHTTPAdapter:
        pass

    FakeHTTPAdapter.send = send

    requests_module = ModuleType("requests")
    adapters_module = ModuleType("requests.adapters")
    exceptions_module = ModuleType("requests.exceptions")
    adapters_module.HTTPAdapter = FakeHTTPAdapter
    exceptions_module.ConnectionError = _FakeConnectionError
    exceptions_module.Timeout = _FakeTimeout
    requests_module.adapters = adapters_module
    requests_module.exceptions = exceptions_module

    monkeypatch.setitem(sys.modules, "requests", requests_module)
    monkeypatch.setitem(sys.modules, "requests.adapters", adapters_module)
    monkeypatch.setitem(sys.modules, "requests.exceptions", exceptions_module)

    exec(swe_module.REQUESTS_HTTPBIN_RETRY_SHIM, {})
    return FakeHTTPAdapter
