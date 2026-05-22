"""Sandbox-side: SWE-bench Verified dataset primitives.

Two remote-call entry points, mirroring the SWE-bench eval contract:

    1. `swe.prepare_env(workdir, base_commit)`
       Reset `/testbed` to `base_commit`, drop any extra commits, wipe
       the working tree. Run before `cc.run` so the model starts from a
       known state regardless of what the prebuilt image happened to
       ship at HEAD.

    2. `swe.score(instance, patch)`
       Reproduce the SWE-bench scoring contract without executing the
       generated `test_spec.eval_script`:
         (a) write `patch` to `/tmp/patch.diff`, try the GIT_APPLY_CMDS
             fallback chain; emit APPLY_PATCH_PASS / APPLY_PATCH_FAIL.
         (b) derive the repository setup/install/base test command from the
             `TestSpec`, reset/apply the test patch, and invoke the targeted
             tests.
         (c) parse and grade the test log locally against
             `TestSpec.FAIL_TO_PASS/PASS_TO_PASS`.
       Returns `{resolved, patch_applied, tests_status}`.

The intended host flow is two sandboxes per instance: one for the cc
agent (`prepare_env` → agent → generic patch capture), tear down, then
a fresh sandbox for `score`. Both sandboxes use the per-instance
SWE-bench eval image (`swebench/sweb.eval.x86_64.<id>:latest`) as the
base.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

WORKROOT = Path(os.environ.get("AGENTIX_UPLOAD_ROOT", "/tmp")) / ".cache" / "swebench-eval"
TESTBED = "/testbed"
PATCH_DOCKER_PATH = "/tmp/patch.diff"

# Same fallback chain as swebench.harness.run_evaluation.GIT_APPLY_CMDS.
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"

FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
PASSED = "PASSED"
FAILED = "FAILED"
SKIPPED = "SKIPPED"
ERROR = "ERROR"
XFAIL = "XFAIL"

NON_TEST_EXTS = (".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml")

REQUESTS_HTTPBIN_RETRY_SHIM = r'''
"""Retry transient httpbin.org failures in legacy Requests SWE-bench tests."""

import time
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

try:
    from requests.adapters import HTTPAdapter
    from requests.exceptions import ConnectionError, Timeout
except Exception:
    HTTPAdapter = None
    ConnectionError = Timeout = Exception


def _agentix_is_httpbin(url):
    host = urlparse(url).hostname or ""
    return host == "httpbin.org" or host.endswith(".httpbin.org")


if HTTPAdapter is not None and not getattr(HTTPAdapter, "_agentix_httpbin_retry", False):
    _agentix_orig_send = HTTPAdapter.send

    def _agentix_send_with_httpbin_retry(self, request, **kwargs):
        attempts = 4 if _agentix_is_httpbin(getattr(request, "url", "")) else 1
        for attempt in range(attempts):
            try:
                response = _agentix_orig_send(self, request, **kwargs)
            except (ConnectionError, Timeout):
                if attempt + 1 >= attempts:
                    raise
            else:
                if response.status_code not in (502, 503, 504) or attempt + 1 >= attempts:
                    return response
                try:
                    response.close()
                except Exception:
                    pass
            time.sleep(0.25 * (attempt + 1))

    HTTPAdapter.send = _agentix_send_with_httpbin_retry
    HTTPAdapter._agentix_httpbin_retry = True
'''


@dataclass
class PrepareEnvResult:
    ok: bool
    head: str
    log: str


@dataclass
class ScoreResult:
    resolved: bool
    patch_applied: bool
    apply_cmd: str | None
    known_fixes: list[str] = field(default_factory=list)
    fail_to_pass: dict[str, list[str]] = field(default_factory=dict)
    pass_to_pass: dict[str, list[str]] = field(default_factory=dict)
    git_diff_before: str = ""
    git_diff_after: str = ""
    apply_log: str = ""
    test_log: str = ""


@dataclass
class EvalPlan:
    setup_commands: list[str]
    install_commands: list[str]
    test_cmd: str
    test_files: list[str]


async def prepare_env(workdir: str = TESTBED, base_commit: str | None = None) -> PrepareEnvResult:
    """Reset `workdir` to `base_commit` and drop everything else.

    Mirrors SWE-bench's expectation that /testbed enters the eval at
    exactly base_commit with a clean working tree.
    """
    parts = [
        f"cd {workdir}",
        "git -c core.fileMode=false update-index --refresh >/dev/null 2>&1 || true",
    ]
    if base_commit:
        parts.append(f"git reset --hard {base_commit}")
    else:
        parts.append("git reset --hard HEAD")
    parts.append("git clean -fdx")
    parts.append("git rev-parse HEAD")

    proc = await asyncio.create_subprocess_shell(
        " && ".join(parts),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="replace")
    head = text.strip().splitlines()[-1] if proc.returncode == 0 else ""
    return PrepareEnvResult(ok=proc.returncode == 0, head=head, log=text)


async def score(
    *,
    instance: dict[str, Any],
    patch: str,
    workdir: str = TESTBED,
    apply_timeout: float = 120,
    eval_timeout: float = 1800,
) -> ScoreResult:
    """Apply `patch` in `workdir`, run targeted SWE-bench tests, grade."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(cast(Any, instance))
    workroot = WORKROOT / spec.instance_id
    if workroot.exists():
        shutil.rmtree(workroot)
    workroot.mkdir(parents=True)

    # (a) Stage the patch and try GIT_APPLY_CMDS in order — same chain
    # as swebench/harness/run_evaluation.py.
    Path(PATCH_DOCKER_PATH).write_text(patch or "")
    apply_log_parts: list[str] = []
    applied_with: str | None = None
    for cmd in GIT_APPLY_CMDS:
        proc = await asyncio.create_subprocess_shell(
            f"cd {workdir} && {cmd} {PATCH_DOCKER_PATH}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=apply_timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            apply_log_parts.append(f"[{cmd}] timed out\n")
            continue
        apply_log_parts.append(f"[{cmd}] exit={proc.returncode}\n{out.decode(errors='replace')}\n")
        if proc.returncode == 0:
            applied_with = cmd
            break

    if applied_with:
        apply_log_parts.append(f"{APPLY_PATCH_PASS}\n")
    else:
        apply_log_parts.append(f"{APPLY_PATCH_FAIL}\n")

    apply_log = "".join(apply_log_parts)
    (workroot / "apply.log").write_text(apply_log)

    git_diff_before = await _capture_diff(workdir)

    test_log_path = workroot / "test_output.log"
    test_log, known_fixes = await _run_swebench_tests(
        spec=spec,
        instance=instance,
        workdir=workdir,
        workroot=workroot,
        timeout=eval_timeout,
    )

    # The grading function expects APPLY_PATCH_PASS / APPLY_PATCH_FAIL
    # in the same log file it parses for test results.
    combined_log = apply_log + test_log
    test_log_path.write_text(combined_log)

    git_diff_after = await _capture_diff(workdir)

    # (c) Grade locally against the gold lists carried by TestSpec.
    entry, grading_fixes = _score_eval_log(
        spec=spec,
        patch=patch,
        patch_applied=applied_with is not None,
        combined_log=combined_log,
    )
    known_fixes.extend(grading_fixes)
    tests = entry.get("tests_status", {}) or {}
    ftp = tests.get("FAIL_TO_PASS", {"success": [], "failure": []})
    ptp = tests.get("PASS_TO_PASS", {"success": [], "failure": []})

    return ScoreResult(
        resolved=bool(entry.get("resolved", False)),
        patch_applied=bool(entry.get("patch_successfully_applied", False)),
        apply_cmd=applied_with,
        known_fixes=known_fixes,
        fail_to_pass={
            "success": list(ftp.get("success", [])),
            "failure": list(ftp.get("failure", [])),
        },
        pass_to_pass={
            "success": list(ptp.get("success", [])),
            "failure": list(ptp.get("failure", [])),
        },
        git_diff_before=git_diff_before,
        git_diff_after=git_diff_after,
        apply_log=apply_log,
        test_log=test_log,
    )


async def _capture_diff(workdir: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        f"cd {workdir} && git -c core.fileMode=false diff",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return out.decode(errors="replace").strip()


async def _run_swebench_tests(
    *,
    spec: Any,
    instance: dict[str, Any],
    workdir: str,
    workroot: Path,
    timeout: float,
) -> tuple[str, list[str]]:
    """Apply SWE-bench's test patch and run targeted tests without generated eval scripts."""
    from agentix.runtime.env import get_env_without_agentix

    env = get_env_without_agentix()
    plan = _eval_plan_from_test_spec(spec, instance)
    log_parts: list[str] = []
    known_fixes: list[str] = []
    run_full_test_command = False
    test_files = list(plan.test_files)
    if not test_files:
        test_files, missing_status_directives = _test_directives_from_status_with_missing(instance)
        if missing_status_directives:
            test_files = []
            run_full_test_command = True
            known_fixes.append("swebench:full-test-command-fallback")
    modified_paths, touched_paths = _extract_test_patch_paths(str(instance["test_patch"]))
    base_commit = str(instance["base_commit"])

    async def shell(command: str, *, command_timeout: float | None = None) -> int:
        code, out = await _run_conda_shell(command, workdir=workdir, env=env, timeout=command_timeout or timeout)
        log_parts.append(f"$ {command}\n{out}\n")
        return code

    async def git(*args: str) -> int:
        code, out = await _run_exec(("git", *args), workdir=workdir, env=env, timeout=timeout)
        log_parts.append(f"$ git {' '.join(shlex.quote(arg) for arg in args)}\n{out}\n")
        return code

    async def abort(reason: str) -> tuple[str, list[str]]:
        log_parts.append(f"aborting SWE-bench test run: {reason}\n")
        return "".join(log_parts), known_fixes

    code = await git("config", "--global", "--add", "safe.directory", workdir)
    if code != 0:
        return await abort("failed to configure git safe.directory")

    for command in plan.setup_commands:
        if _apply_export_command(command, env):
            log_parts.append(f"$ {command}\n")
        else:
            code = await shell(command)
            if code != 0:
                return await abort(f"eval command failed: {command}")

    if _is_old_astropy(instance):
        code = await shell("python -m pip install 'pytest<7' --verbose")
        if code != 0:
            return await abort("failed to install legacy Astropy pytest dependency")
        env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
        known_fixes.append("astropy:legacy-test-deps")
    if _patch_django_legacy_sqlite_schema_editor(Path(workdir), instance):
        known_fixes.append("django:legacy-sqlite-alter-table")
    requests_fixes = await _prepare_legacy_requests_network(Path(workdir), workroot, instance, env, log_parts)
    known_fixes.extend(requests_fixes)

    install_commands = [] if instance["repo"] == "scikit-learn/scikit-learn" else plan.install_commands
    for install_command in install_commands:
        code = await shell(install_command)
        if code != 0:
            return await abort(f"install command failed: {install_command}")

    if modified_paths:
        code = await git("checkout", base_commit, *modified_paths)
        if code != 0:
            return await abort("failed to reset tracked test-patch files")
    if touched_paths and not modified_paths:
        known_fixes.append("swebench:new-file-only-test-patch-reset")
    await _cleanup_untracked_paths(Path(workdir), touched_paths, log_parts, env)

    Path("/tmp/test_patch.diff").write_text(str(instance["test_patch"]))
    code = await git("apply", "--check", "/tmp/test_patch.diff")
    if code != 0:
        return await abort("test patch failed git apply --check")
    code = await git("apply", "/tmp/test_patch.diff")
    if code != 0:
        return await abort("test patch failed git apply")

    quoted_test_files = " ".join(shlex.quote(path) for path in test_files)
    if not quoted_test_files and not run_full_test_command:
        return await abort("could not derive targeted test directives")
    test_invocation = plan.test_cmd if run_full_test_command else f"{plan.test_cmd} {quoted_test_files}"
    log_parts.append(f": '{START_TEST_OUTPUT}'\n")
    await shell(test_invocation, command_timeout=timeout)
    log_parts.append(f": '{END_TEST_OUTPUT}'\n")

    if modified_paths:
        await git("checkout", base_commit, *modified_paths)
    await _cleanup_untracked_paths(Path(workdir), touched_paths, log_parts, env)
    return "".join(log_parts), known_fixes


def _eval_plan_from_test_spec(spec: Any, instance: dict[str, Any]) -> EvalPlan:
    """Extract the commands we need from TestSpec without executing its script."""
    commands = [str(command).strip() for command in spec.eval_script_list if str(command).strip()]
    start_marker = f": '{START_TEST_OUTPUT}'"
    try:
        test_marker_index = commands.index(start_marker)
        generated_test_command = commands[test_marker_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"TestSpec for {spec.instance_id} does not contain a runnable test command") from exc

    test_files = _test_directives_from_patch(instance["repo"], str(instance["test_patch"]))
    test_cmd = _strip_test_directive_suffix(generated_test_command, test_files)

    try:
        setup_end = next(
            index
            for index, command in enumerate(commands)
            if command.startswith("git config --global --add safe.directory")
        )
    except StopIteration as exc:
        raise ValueError(f"TestSpec for {spec.instance_id} does not contain the safe.directory command") from exc

    setup_commands = [
        command
        for command in commands[:setup_end]
        if not _is_eval_script_boilerplate(command) and not _is_git_info_command(command)
    ]

    try:
        apply_index = next(index for index, command in enumerate(commands) if command.startswith("git apply -v - <<"))
        reset_index = next(
            index
            for index, command in enumerate(commands[:apply_index])
            if command.startswith(f"git checkout {instance['base_commit']}")
        )
    except StopIteration as exc:
        raise ValueError(f"TestSpec for {spec.instance_id} does not contain a test-patch apply sequence") from exc

    install_commands = [
        command
        for command in commands[setup_end + 1 : reset_index]
        if not _is_eval_script_boilerplate(command) and not _is_git_info_command(command)
    ]

    return EvalPlan(
        setup_commands=setup_commands,
        install_commands=install_commands,
        test_cmd=test_cmd,
        test_files=test_files,
    )


def _is_eval_script_boilerplate(command: str) -> bool:
    return (
        command == "source /opt/miniconda3/bin/activate"
        or command.startswith("conda activate ")
        or command.startswith("cd ")
    )


def _is_git_info_command(command: str) -> bool:
    return command in {"git status", "git show"} or command.startswith("git -c core.fileMode=false diff ")


def _test_directives_from_patch(repo: str, test_patch: str) -> list[str]:
    directives = re.findall(r"diff --git a/.* b/(.*)", test_patch)
    directives = [directive for directive in directives if not any(directive.endswith(ext) for ext in NON_TEST_EXTS)]
    if repo == "django/django":
        transformed = []
        for directive in directives:
            if directive.endswith(".py"):
                directive = directive[: -len(".py")]
            if directive.startswith("tests/"):
                directive = directive[len("tests/") :]
            transformed.append(directive.replace("/", "."))
        directives = transformed
    return list(dict.fromkeys(directives))


def _strip_test_directive_suffix(command: str, directives: list[str]) -> str:
    suffix = " ".join(directives)
    if suffix and command.endswith(f" {suffix}"):
        return command[: -(len(suffix) + 1)].rstrip()
    return command


def _extract_test_patch_paths(test_patch: str) -> tuple[list[str], list[str]]:
    preimage_paths = [
        path
        for path in re.findall(r"^--- a/(.*)$", test_patch, re.MULTILINE)
        if path != "/dev/null"
    ]
    postimage_paths = [
        path
        for path in re.findall(r"^\+\+\+ b/(.*)$", test_patch, re.MULTILINE)
        if path != "/dev/null"
    ]
    return list(dict.fromkeys(preimage_paths)), list(dict.fromkeys(preimage_paths + postimage_paths))


def _test_directives_from_status(instance: dict[str, Any]) -> list[str]:
    directives, _ = _test_directives_from_status_with_missing(instance)
    return directives


def _test_directives_from_status_with_missing(instance: dict[str, Any]) -> tuple[list[str], int]:
    directives: list[str] = []
    missing = 0
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        for test_name in _load_test_status_list(instance.get(key, [])):
            directive = _test_directive_from_status_name(test_name)
            if directive:
                directives.append(directive)
            else:
                missing += 1
    return list(dict.fromkeys(directives)), missing


def _load_test_status_list(value: Any) -> list[str]:
    if isinstance(value, str):
        loaded = json.loads(value)
    else:
        loaded = value
    return [str(item) for item in loaded]


def _test_directive_from_status_name(test_name: str) -> str | None:
    if "::" in test_name:
        return test_name
    match = re.search(r"\(([^()]+)\)", test_name)
    if not match:
        return None
    dotted = match.group(1)
    if "." not in dotted or " " in dotted:
        return None
    return dotted


def _is_old_astropy(instance: dict[str, Any]) -> bool:
    if instance["repo"] != "astropy/astropy":
        return False
    major_minor = _major_minor_version(str(instance["version"]))
    return major_minor is not None and major_minor <= (3, 1)


def _major_minor_version(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _patch_django_legacy_sqlite_schema_editor(workdir: Path, instance: dict[str, Any]) -> bool:
    if instance["repo"] != "django/django":
        return False
    major_minor = _major_minor_version(str(instance["version"]))
    if major_minor is None or major_minor > (2, 2):
        return False

    path = workdir / "django/db/backends/sqlite3/schema.py"
    text = path.read_text()
    if "PRAGMA legacy_alter_table = ON" in text:
        return False
    old = """    def __enter__(self):
        # Some SQLite schema alterations need foreign key constraints to be
        # disabled. Enforce it here for the duration of the transaction.
        self.connection.disable_constraint_checking()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.connection.enable_constraint_checking()
"""
    new = """    def __enter__(self):
        # Some SQLite schema alterations need foreign key constraints to be
        # disabled. Enforce it here for the duration of the transaction.
        self.connection.disable_constraint_checking()
        with self.connection.cursor() as cursor:
            cursor.execute('PRAGMA legacy_alter_table = ON')
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            with self.connection.cursor() as cursor:
                cursor.execute('PRAGMA legacy_alter_table = OFF')
            self.connection.enable_constraint_checking()
"""
    if old not in text:
        return False
    path.write_text(text.replace(old, new))
    return True


async def _prepare_legacy_requests_network(
    workdir: Path,
    workroot: Path,
    instance: dict[str, Any],
    env: dict[str, str],
    log_parts: list[str],
) -> list[str]:
    if instance["repo"] != "psf/requests":
        return []

    fixes = ["requests:https-httpbin"]
    env["HTTPBIN_URL"] = "https://httpbin.org/"
    if _install_requests_httpbin_retry_shim(workroot, env):
        fixes.append("requests:httpbin-transient-retry")

    needs_tarpit_patch = await _requests_tarpit_needs_connect_timeout_patch(workdir, env, log_parts)
    if needs_tarpit_patch and _patch_requests_tarpit_connect_timeout(workdir):
        fixes.append("requests:docker-tarpit-connect-timeout")
    return fixes


def _install_requests_httpbin_retry_shim(workroot: Path, env: dict[str, str]) -> bool:
    shim_dir = workroot / "requests-httpbin-retry"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(REQUESTS_HTTPBIN_RETRY_SHIM.lstrip())
    pythonpath = env.get("PYTHONPATH")
    shim_path = str(shim_dir)
    if pythonpath:
        paths = pythonpath.split(os.pathsep)
        if shim_path not in paths:
            env["PYTHONPATH"] = os.pathsep.join([shim_path, *paths])
    else:
        env["PYTHONPATH"] = shim_path
    return True


async def _requests_tarpit_needs_connect_timeout_patch(
    workdir: Path,
    env: dict[str, str],
    log_parts: list[str],
) -> bool:
    command = """python - <<'PY'
import socket
import sys

s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("10.255.255.1", 80))
except socket.timeout:
    sys.exit(0)
except OSError:
    sys.exit(1)
else:
    sys.exit(1)
finally:
    s.close()
PY"""
    code, out = await _run_conda_shell(command, workdir=str(workdir), env=env, timeout=5)
    log_parts.append(f"$ requests tarpit connect-timeout probe\n{out}\n")
    return code != 0


def _patch_requests_tarpit_connect_timeout(workdir: Path) -> bool:
    path = workdir / "requests/packages/urllib3/util/connection.py"
    if not path.exists():
        return False
    text = path.read_text()
    if "Agentix Docker Desktop tarpit compatibility" in text:
        return False
    old = """    host, port = address
    err = None
"""
    new = """    host, port = address
    # Agentix Docker Desktop tarpit compatibility: old Requests tests expect
    # 10.255.255.1 to hang during connect, but some local Docker networks
    # accept the TCP connection and hang only on read.
    if host == "10.255.255.1":
        raise socket.timeout("timed out")
    err = None
"""
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1))
    return True


def _score_eval_log(
    *,
    spec: Any,
    patch: str | None,
    patch_applied: bool,
    combined_log: str,
) -> tuple[dict[str, Any], list[str]]:
    if patch is None:
        return {
            "patch_is_None": True,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "resolved": False,
            "tests_status": _eval_tests_report({}, spec),
        }, []

    test_output = _extract_marked_test_output(combined_log)
    if patch_applied and test_output is not None:
        status_map = _parse_eval_statuses(test_output or combined_log, spec)
    else:
        status_map = {}
    report = _eval_tests_report(status_map, spec)
    entry = {
        "patch_is_None": False,
        "patch_exists": True,
        "patch_successfully_applied": patch_applied,
        "resolved": patch_applied and TESTS_TIMEOUT not in combined_log and _report_is_resolved(report),
        "tests_status": report,
    }
    entry, fixes = _apply_grading_normalizations(entry, combined_log)
    if TESTS_TIMEOUT in combined_log:
        entry["resolved"] = False
    return entry, fixes


def _extract_marked_test_output(log: str) -> str | None:
    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        return None
    return log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]


def _eval_tests_report(status_map: dict[str, str], spec: Any) -> dict[str, dict[str, list[str]]]:
    return {
        FAIL_TO_PASS: _score_gold_tests(list(spec.FAIL_TO_PASS), status_map),
        PASS_TO_PASS: _score_gold_tests(list(spec.PASS_TO_PASS), status_map),
    }


def _score_gold_tests(gold_tests: list[str], status_map: dict[str, str]) -> dict[str, list[str]]:
    success: list[str] = []
    failure: list[str] = []
    for test_name in gold_tests:
        status = status_map.get(test_name)
        if status in {PASSED, XFAIL}:
            success.append(test_name)
        elif status is None or status in {FAILED, ERROR}:
            failure.append(test_name)
    return {"success": success, "failure": failure}


def _report_is_resolved(report: dict[str, dict[str, list[str]]]) -> bool:
    return _score_ratio(report[FAIL_TO_PASS]) == 1 and _score_ratio(report[PASS_TO_PASS]) == 1


def _score_ratio(status: dict[str, list[str]]) -> float:
    total = len(status["success"]) + len(status["failure"])
    if total == 0:
        return 1
    return len(status["success"]) / total


def _parse_eval_statuses(log: str, spec: Any) -> dict[str, str]:
    parser = _PARSERS_BY_REPO.get(spec.repo, _parse_log_pytest_v2)
    return parser(log, spec)


def _parse_log_pytest(log: str, spec: Any) -> dict[str, str]:
    status_map: dict[str, str] = {}
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not _line_starts_status(line):
            continue
        if line.startswith(FAILED):
            line = line.replace(" - ", " ")
        parts = line.split()
        if len(parts) >= 2:
            status_map[parts[1]] = parts[0]
    return status_map


def _parse_log_pytest_options(log: str, spec: Any) -> dict[str, str]:
    status_map: dict[str, str] = {}
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not _line_starts_status(line):
            continue
        if line.startswith(FAILED):
            line = line.replace(" - ", " ")
        parts = line.split()
        if len(parts) < 2:
            continue
        test_name = parts[1]
        match = option_pattern.search(test_name)
        if match:
            main, option = match.groups()
            if option.startswith("/") and not option.startswith("//") and "*" not in option:
                option = "/" + option.split("/")[-1]
            test_name = f"{main}[{option}]"
        status_map[test_name] = parts[0]
    return status_map


def _parse_log_pytest_v2(log: str, spec: Any) -> dict[str, str]:
    status_map: dict[str, str] = {}
    escapes = "".join(chr(char) for char in range(1, 32))
    for raw_line in log.splitlines():
        line = re.sub(r"\[(\d+)m", "", raw_line.strip()).translate(str.maketrans("", "", escapes))
        if _line_starts_status(line):
            if line.startswith(FAILED):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                status_map[parts[1]] = parts[0]
        elif _line_ends_status(line):
            parts = line.split()
            if len(parts) >= 2:
                status_map[parts[0]] = parts[1]
    return status_map


def _parse_log_django(log: str, spec: Any) -> dict[str, str]:
    status_map: dict[str, str] = {}
    prev_test: str | None = None
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if "--version is equivalent to version" in line:
            status_map["--version is equivalent to version"] = PASSED
        if " ... " in line:
            prev_test = line.split(" ... ")[0]
        for suffix in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suffix):
                if line.startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                    line = line.split("...", 1)[-1].strip()
                status_map[line.rsplit(suffix, 1)[0]] = PASSED
                break
        if " ... skipped" in line:
            status_map[line.split(" ... skipped")[0]] = SKIPPED
        if line.endswith(" ... FAIL"):
            status_map[line.split(" ... FAIL")[0]] = FAILED
        if line.startswith("FAIL:"):
            status_map[line.split()[1].strip()] = FAILED
        if line.endswith(" ... ERROR"):
            status_map[line.split(" ... ERROR")[0]] = ERROR
        if line.startswith("ERROR:"):
            status_map[line.split()[1].strip()] = ERROR
        if line.lstrip().startswith("ok") and prev_test is not None:
            status_map[prev_test] = PASSED

    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            status_map[match.group(1)] = PASSED
    return status_map


def _parse_log_seaborn(log: str, spec: Any) -> dict[str, str]:
    status_map: dict[str, str] = {}
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if line.startswith(FAILED):
            parts = line.split()
            if len(parts) >= 2:
                status_map[parts[1]] = FAILED
        elif f" {PASSED} " in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == PASSED:
                status_map[parts[0]] = PASSED
        elif line.startswith(PASSED):
            parts = line.split()
            if len(parts) >= 2:
                status_map[parts[1]] = PASSED
    return status_map


def _parse_log_matplotlib(log: str, spec: Any) -> dict[str, str]:
    return _parse_log_pytest(log.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3"), spec)


def _parse_log_sympy(log: str, spec: Any) -> dict[str, str]:
    status_map: dict[str, str] = {}
    for match in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        status_map[f"{match[1]}.py:{match[2]}"] = FAILED
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not line.startswith("test_"):
            continue
        test_name = line.split()[0]
        if line.endswith(" E"):
            status_map[test_name] = ERROR
        if line.endswith(" F"):
            status_map[test_name] = FAILED
        if line.endswith(" ok"):
            status_map[test_name] = PASSED
    return status_map


def _line_starts_status(line: str) -> bool:
    return any(line.startswith(status) for status in (FAILED, PASSED, SKIPPED, ERROR, XFAIL))


def _line_ends_status(line: str) -> bool:
    return any(line.endswith(status) for status in (FAILED, PASSED, SKIPPED, ERROR, XFAIL))


_PARSERS_BY_REPO = {
    "astropy/astropy": _parse_log_pytest_v2,
    "django/django": _parse_log_django,
    "matplotlib/matplotlib": _parse_log_matplotlib,
    "mwaskom/seaborn": _parse_log_seaborn,
    "pallets/flask": _parse_log_pytest,
    "psf/requests": _parse_log_pytest_options,
    "pydata/xarray": _parse_log_pytest,
    "pylint-dev/pylint": _parse_log_pytest_options,
    "pytest-dev/pytest": _parse_log_pytest,
    "scikit-learn/scikit-learn": _parse_log_pytest_v2,
    "sphinx-doc/sphinx": _parse_log_pytest_v2,
    "sympy/sympy": _parse_log_sympy,
}


def _apply_grading_normalizations(entry: dict[str, Any], combined_log: str) -> tuple[dict[str, Any], list[str]]:
    entry = dict(entry)
    tests = dict(entry.get("tests_status", {}) or {})
    known_fixes: list[str] = []
    changed = False

    for bucket in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        status = dict(tests.get(bucket, {}) or {})
        success = list(status.get("success", []))
        failure = list(status.get("failure", []))
        remaining_failure: list[str] = []

        for test_name in failure:
            if _pytest_empty_param_alias_passed(test_name, combined_log):
                success.append(test_name)
                changed = True
            else:
                remaining_failure.append(test_name)

        status["success"] = list(dict.fromkeys(success))
        status["failure"] = remaining_failure
        tests[bucket] = status

    if changed:
        entry["tests_status"] = tests
        ftp = tests.get("FAIL_TO_PASS", {}) or {}
        ptp = tests.get("PASS_TO_PASS", {}) or {}
        entry["resolved"] = not ftp.get("failure") and not ptp.get("failure")
        known_fixes.append("pytest:empty-param-id-alias")

    return entry, known_fixes


def _pytest_empty_param_alias_passed(test_name: str, combined_log: str) -> bool:
    if not test_name.endswith("[]"):
        return False
    base = test_name[:-2]
    aliases = set(
        re.findall(
            rf"(?m)^{re.escape(base)}\[[A-Za-z_]\w*\d+\] PASSED(?:\s|$)",
            combined_log,
        )
    )
    return len(aliases) == 1


async def _run_conda_shell(
    command: str,
    *,
    workdir: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str]:
    wrapped = f"source /opt/miniconda3/bin/activate\nconda activate testbed\n{command}"
    return await _run_exec(("/bin/bash", "-lc", wrapped), workdir=workdir, env=env, timeout=timeout)


async def _run_exec(
    args: tuple[str, ...],
    *,
    workdir: str,
    env: dict[str, str],
    timeout: float,
) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=workdir,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, out.decode(errors="replace")
    except TimeoutError:
        proc.kill()
        out, _ = await proc.communicate()
        text = out.decode(errors="replace")
        return 124, f"{text}\n{TESTS_TIMEOUT}\ncommand timed out after {timeout}s: {shlex.join(args)}\n"


def _apply_export_command(command: str, env: dict[str, str]) -> bool:
    if not command.startswith("export "):
        return False
    assignments = shlex.split(command[len("export ") :])
    if not assignments or any("=" not in item for item in assignments):
        return False
    for item in assignments:
        key, value = item.split("=", 1)
        env[key] = value
    return True


async def _cleanup_untracked_paths(
    root: Path,
    paths: list[str],
    log_parts: list[str],
    env: dict[str, str],
) -> None:
    root = root.resolve()
    for path in paths:
        target = (root / path).resolve()
        if root != target and root not in target.parents:
            log_parts.append(f"skip cleanup outside testbed: {path}\n")
            continue
        if not target.exists():
            continue
        code, out = await _run_exec(
            ("git", "ls-files", "--error-unmatch", "--", path),
            workdir=str(root),
            env=env,
            timeout=30,
        )
        log_parts.append(f"$ git ls-files --error-unmatch -- {shlex.quote(path)}\n{out}\n")
        if code == 0:
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        log_parts.append(f"removed untracked test patch path: {path}\n")



__all__ = ["prepare_env", "score", "PrepareEnvResult", "ScoreResult"]
