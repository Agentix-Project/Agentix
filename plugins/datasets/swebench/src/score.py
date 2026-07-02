from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
from pathlib import Path
from typing import Any, cast

from swebench.harness.constants import (
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.log_parsers.python import parse_log_pytest_v2
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import make_test_spec
from unidiff import PatchSet

from .env import prepare_env

logger = logging.getLogger(__name__)

TESTBED = "/testbed"
MODEL_PATCH = "/tmp/agentix_model.patch"
TEST_PATCH = "/tmp/agentix_test.patch"

# Every unresolved result carries `failure_stage` plus a bounded
# `log_tail`, so a nightly artifact can be attributed (harness bug vs
# environment drift vs genuinely failing tests) without re-running the
# instance. The bound keeps per-instance JSON small at batch scale.
LOG_TAIL_CHARS = 8000


def _tail(log: str) -> str:
    return log[-LOG_TAIL_CHARS:]


def _get_test_command(instance: dict) -> str:
    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
    directives = [shlex.quote(str(d)) for d in get_test_directives(cast(Any, instance))]
    return " ".join([str(specs["test_cmd"]), *directives])


def _get_log_parser(instance: dict):
    spec = make_test_spec(cast(Any, instance))
    parser = MAP_REPO_TO_PARSER.get(instance["repo"], parse_log_pytest_v2)
    return lambda log: parser(log, spec)


def _make_report(instance: dict, test_status: dict) -> dict:
    spec = make_test_spec(cast(Any, instance))
    gold = {FAIL_TO_PASS: spec.FAIL_TO_PASS, PASS_TO_PASS: spec.PASS_TO_PASS}
    eval_type = EvalType.FAIL_ONLY if instance["repo"] in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
    report = get_eval_tests_report(test_status, cast(Any, gold), eval_type=eval_type)
    return {
        "resolved": get_resolution_status(report) == ResolvedStatus.FULL.value,
        "patch_applied": True,
        "timed_out": False,
        "test_status": test_status,
    }


async def score(
    instance: dict,
    patch: str,
    workdir: str = TESTBED,
    apply_timeout: float = 120,
    eval_timeout: float = 1800,
) -> dict:
    logger.info("scoring %s", instance["instance_id"])
    logger.debug("submitted patch:\n%s", patch)
    logger.debug("groundtruth patch:\n%s", instance.get("patch", ""))

    env = _get_env()
    prepared = await prepare_env(workdir=workdir, base_commit=str(instance["base_commit"]))
    if not prepared.ok:
        logger.error("prepare env failed:\n%s", prepared.log)
        return _empty_result(patch_applied=False, stage="prepare_env", log=prepared.log)

    patch_applied, apply_log = await _apply_model_patch(patch, workdir, env, apply_timeout)
    if not patch_applied:
        return _empty_result(patch_applied=False, stage="apply_patch", log=apply_log)

    prepared_ok, prepare_log = await _prepare_tests(instance, workdir, env, eval_timeout)
    if not prepared_ok:
        return _empty_result(patch_applied=True, stage="prepare_tests", log=prepare_log)

    code, output, timed_out = await _run(_get_test_command(instance), workdir, env, eval_timeout, conda=True)
    logger.debug("test output:\n%s", output)
    await _cleanup_tests(instance, workdir, env, eval_timeout)

    test_status = _get_log_parser(instance)(output)
    result = _make_report(instance, test_status)
    result["timed_out"] = timed_out
    if timed_out:
        result["resolved"] = False
    if not result["resolved"]:
        result["failure_stage"] = "tests"
        result["exit_code"] = code
        result["log_tail"] = _tail(output)
    logger.info(
        "scored %s: resolved=%s tests=%d exit=%d",
        instance["instance_id"],
        result["resolved"],
        len(test_status),
        code,
    )
    return result


async def _apply_model_patch(patch: str, workdir: str, env: dict[str, str], timeout: float) -> tuple[bool, str]:
    if not patch.strip():
        logger.error("empty patch")
        return False, "empty patch"
    Path(MODEL_PATCH).write_text(patch)
    attempts: list[str] = []
    for command in GIT_APPLY_CMDS:
        code, output, _ = await _run(f"{command} {MODEL_PATCH}", workdir, env, timeout)
        logger.debug("$ %s\n%s", command, output)
        if code == 0:
            logger.info("patch applied with: %s", command)
            return True, ""
        attempts.append(f"$ {command}\n{output}")
    logger.error("failed to apply patch")
    return False, "\n".join(attempts)


async def _prepare_tests(instance: dict, workdir: str, env: dict[str, str], timeout: float) -> tuple[bool, str]:
    specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]
    commands = [f"git config --global --add safe.directory {shlex.quote(workdir)}"]
    # Task images bake `pre_install` edits into /testbed's tracked files
    # at build time (sphinx: `sed 's/pytest/pytest -rA/' tox.ini` so the
    # log parser sees per-test lines; astropy: a setuptools pin that the
    # editable reinstall needs). `prepare_env`'s `git reset --hard`
    # reverts those edits, so re-apply them before install/eval.
    commands.extend(str(command) for command in specs.get("pre_install", []))
    commands.extend(str(command) for command in specs.get("eval_commands", []))

    install = str(specs.get("install") or "").strip()
    if install:
        commands.append(install)

    modified, touched = _test_patch_paths(str(instance["test_patch"]))
    if modified:
        commands.append(f"git checkout {shlex.quote(str(instance['base_commit']))} {shlex.join(modified)}")

    for command in commands:
        if _apply_export(command, env):
            continue
        code, output, _ = await _run(command, workdir, env, timeout, conda=not command.startswith("git "))
        logger.debug("$ %s\n%s", command, output)
        if code != 0:
            logger.error("prepare command failed: %s", command)
            return False, f"$ {command}\n{output}"

    await _remove_untracked_paths(workdir, touched, env, timeout)
    Path(TEST_PATCH).write_text(str(instance["test_patch"]))
    for command in (f"git apply --check {TEST_PATCH}", f"git apply {TEST_PATCH}"):
        code, output, _ = await _run(command, workdir, env, timeout)
        logger.debug("$ %s\n%s", command, output)
        if code != 0:
            logger.error("test patch command failed: %s", command)
            return False, f"$ {command}\n{output}"
    return True, ""


async def _cleanup_tests(instance: dict, workdir: str, env: dict[str, str], timeout: float) -> None:
    modified, touched = _test_patch_paths(str(instance["test_patch"]))
    if modified:
        command = f"git checkout {shlex.quote(str(instance['base_commit']))} {shlex.join(modified)}"
        await _run(command, workdir, env, timeout)
    await _remove_untracked_paths(workdir, touched, env, timeout)


def _test_patch_paths(test_patch: str) -> tuple[list[str], list[str]]:
    modified: list[str] = []
    touched: list[str] = []

    def add_unique(paths: list[str], path: str | None) -> None:
        if path and path not in paths:
            paths.append(path)

    for file in PatchSet(test_patch):
        source = _strip_patch_prefix(file.source_file)
        target = _strip_patch_prefix(file.target_file)
        if not file.is_added_file:
            add_unique(modified, source)
        add_unique(touched, source)
        add_unique(touched, target)

    return modified, touched


def _strip_patch_prefix(path: str) -> str | None:
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


async def _remove_untracked_paths(workdir: str, paths: list[str], env: dict[str, str], timeout: float) -> None:
    root = Path(workdir).resolve()
    for path in paths:
        target = (root / path).resolve()
        if root != target and root not in target.parents:
            logger.warning("skip cleanup outside testbed: %s", path)
            continue
        if not target.exists():
            continue
        code, _, _ = await _run(f"git ls-files --error-unmatch -- {shlex.quote(path)}", workdir, env, timeout)
        if code != 0:
            shutil.rmtree(target) if target.is_dir() else target.unlink()


async def _run(command: str, workdir: str, env: dict[str, str], timeout: float, *, conda: bool = False):
    prefix = "source /opt/miniconda3/bin/activate\nconda activate testbed\n" if conda else ""
    proc = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-c",
        f"set -o pipefail\n{prefix}{command}",
        cwd=workdir,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, out.decode(errors="replace"), False
    except TimeoutError:
        proc.kill()
        out, _ = await proc.communicate()
        return 124, out.decode(errors="replace"), True


def _get_env() -> dict[str, str]:
    from agentix.runtime.shared.env import get_env_without_agentix

    env = get_env_without_agentix()
    bashrc = Path.home() / ".bashrc"
    if bashrc.is_file() and "BASH_ENV" not in env:
        env["BASH_ENV"] = str(bashrc)
    return env


def _apply_export(command: str, env: dict[str, str]) -> bool:
    if not command.startswith("export "):
        return False
    for item in shlex.split(command.removeprefix("export ")):
        key, value = item.split("=", 1)
        env[key] = value
    return True


def _empty_result(*, patch_applied: bool, stage: str, log: str) -> dict:
    return {
        "resolved": False,
        "patch_applied": patch_applied,
        "timed_out": False,
        "test_status": {},
        "failure_stage": stage,
        "log_tail": _tail(log),
    }
