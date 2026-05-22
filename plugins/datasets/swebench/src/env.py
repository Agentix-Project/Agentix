from __future__ import annotations

import asyncio
from dataclasses import dataclass

TESTBED = "/testbed"


@dataclass
class PrepareEnvResult:
    ok: bool
    head: str
    log: str


async def prepare_env(workdir: str = TESTBED, base_commit: str | None = None) -> PrepareEnvResult:
    """Reset `workdir` to `base_commit` with a clean working tree."""
    commands = [
        "git -c core.fileMode=false update-index --refresh >/dev/null 2>&1 || true",
        f"git reset --hard {base_commit or 'HEAD'}",
        "git clean -fdx",
        "git rev-parse HEAD",
    ]
    code, log = await _run(" && ".join(commands), workdir=workdir)
    head = log.strip().splitlines()[-1] if code == 0 and log.strip() else ""
    return PrepareEnvResult(ok=code == 0, head=head, log=log)


async def _run(command: str, *, workdir: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=workdir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace")
