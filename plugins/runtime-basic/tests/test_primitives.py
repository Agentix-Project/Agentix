from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path

import agentix.bash as bash
import agentix.files as files
import pytest


@pytest.mark.asyncio
async def test_files_upload_symlink_to_outside_cannot_touch_the_real_target(tmp_path, monkeypatch):
    """An absolute symlink pointing outside the root is re-anchored at the root
    (chroot / RESOLVE_IN_ROOT semantics): the write lands inside the jail and
    the real out-of-jail file is never touched — escape is structurally
    impossible, not merely refused."""
    outside = tmp_path / "outside.txt"
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(files, "UPLOAD_ROOT", root.resolve())

    link = root / "link"
    link.symlink_to(outside)  # absolute target, outside the root
    result = await files.upload("link", b"escape")
    assert not outside.exists()  # the real out-of-jail path is untouched
    assert Path(result.path).resolve().is_relative_to(root.resolve())


@pytest.mark.asyncio
async def test_bash_run_drains_stderr_after_output_cap():
    code = (
        "import sys; "
        "sys.stderr.write('x' * 200000); "
        "sys.stderr.flush(); "
        "print('done')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    result = await asyncio.wait_for(
        bash.run(command, max_output=1024),
        timeout=5,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "done"
    assert "[truncated at 1024 bytes]" in result.stderr


@pytest.mark.asyncio
async def test_bash_run_honors_bash_env(tmp_path: Path):
    bash_env = tmp_path / "bash_env"
    bash_env.write_text("export FROM_BASH_ENV=loaded\n")

    result = await bash.run(
        "printf '%s' \"$FROM_BASH_ENV\"",
        env={**os.environ, "BASH_ENV": str(bash_env)},
    )

    assert result.exit_code == 0
    assert result.stdout == "loaded"


@pytest.mark.asyncio
@pytest.mark.parametrize("executable", ["bash", "zsh", "fish"])
async def test_bash_run_can_use_explicit_executable_names(tmp_path: Path, executable: str):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    shell_path = fakebin / executable
    shell_path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" != \"-c\" ]; then exit 64; fi\n"
        "shift\n"
        "export AGENTIX_TEST_SHELL=\"$(basename \"$0\")\"\n"
        "exec /bin/sh -c \"$1\"\n"
    )
    shell_path.chmod(0o755)

    result = await bash.run(
        "printf '%s' \"$AGENTIX_TEST_SHELL\"",
        env={"PATH": os.pathsep.join([str(fakebin), os.environ.get("PATH", "")])},
        executable=executable,
    )

    assert result.exit_code == 0
    assert result.stdout == executable


@pytest.mark.asyncio
async def test_bash_run_stream_can_use_explicit_executable(tmp_path: Path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    shell_path = fakebin / "zsh"
    shell_path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" != \"-c\" ]; then exit 64; fi\n"
        "shift\n"
        "export AGENTIX_TEST_SHELL=\"$(basename \"$0\")\"\n"
        "exec /bin/sh -c \"$1\"\n"
    )
    shell_path.chmod(0o755)

    events = [
        event
        async for event in bash.run_stream(
            "printf '%s' \"$AGENTIX_TEST_SHELL\"",
            env={"PATH": os.pathsep.join([str(fakebin), os.environ.get("PATH", "")])},
            executable="zsh",
        )
    ]

    assert [event.data for event in events if isinstance(event, bash.BashStdout)] == ["zsh"]
    assert [event.exit_code for event in events if isinstance(event, bash.BashExit)] == [0]


@pytest.mark.asyncio
async def test_bash_run_none_env_inherits_worker_environment(monkeypatch):
    """subprocess.run parity: no env → inherit the worker's environment."""
    monkeypatch.setenv("AGENTIX_TEST_MARKER", "inherited")
    result = await bash.run(command='printf %s "$AGENTIX_TEST_MARKER"')
    assert result.stdout == "inherited"


@pytest.mark.asyncio
async def test_bash_run_env_dict_replaces_the_environment(monkeypatch):
    """subprocess.run parity: an explicit env dict REPLACES the environment;
    a worker var not in it is gone from the child (no merge)."""
    monkeypatch.setenv("AGENTIX_TEST_MARKER", "should-not-leak")
    result = await bash.run(
        command='printf %s "${AGENTIX_TEST_MARKER-absent}"',
        env={"PATH": os.environ.get("PATH", "/usr/bin")},
    )
    assert result.stdout == "absent"


def test_image_env_subtracts_additions_and_restores_saved(monkeypatch):
    """The task-image env util: recorded bundle PATH additions subtracted,
    stripped vars restored from their AGENTIX_SAVED_* snapshot."""
    from agentix.bash import image_env

    from agentix.runtime.shared.env import AGENTIX_ADDED_PATH

    monkeypatch.setenv("PATH", os.pathsep.join(["/nix/runtime/bin", "/usr/bin"]))
    monkeypatch.setenv(AGENTIX_ADDED_PATH, "/nix/runtime/bin")
    monkeypatch.setenv("AGENTIX_SAVED_PYTHONPATH", "/testbed/src")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = image_env()
    assert env["PATH"] == "/usr/bin"
    assert env["PYTHONPATH"] == "/testbed/src"
    assert AGENTIX_ADDED_PATH not in env
    assert "AGENTIX_SAVED_PYTHONPATH" not in env


@pytest.mark.asyncio
async def test_bash_run_with_image_env_runs_in_the_image_world(monkeypatch):
    """The blessed recipe: bash.run(cmd, env=image_env()) drops the bundle's
    injected PATH and restores the image's stripped vars."""
    from agentix.bash import image_env

    from agentix.runtime.shared.env import AGENTIX_ADDED_PATH

    real_path = os.environ.get("PATH", "/usr/bin")
    monkeypatch.setenv("PATH", os.pathsep.join(["/agentix-injected/bin", real_path]))
    monkeypatch.setenv(AGENTIX_ADDED_PATH, "/agentix-injected/bin")
    monkeypatch.setenv("AGENTIX_SAVED_PYTHONPATH", "/testbed/src")
    monkeypatch.delenv("PYTHONPATH", raising=False)

    polluted = await bash.run(command='printf %s "$PATH"')
    assert polluted.stdout.startswith("/agentix-injected/bin")

    clean = await bash.run(command='printf %s "$PATH:$PYTHONPATH"', env=image_env())
    assert "/agentix-injected/bin" not in clean.stdout
    assert clean.stdout.endswith("/testbed/src")


def _set_files_root(monkeypatch, root):
    # UPLOAD_ROOT is computed from env at import time; patch the module
    # attribute directly — reloading the module would re-create its classes
    # and break identity for everything that imported them.
    monkeypatch.setattr(files, "UPLOAD_ROOT", root.resolve())


@pytest.mark.asyncio
async def test_files_follows_directory_symlinks_inside_root(tmp_path, monkeypatch):
    """merged-usr layout: /bin -> usr/bin. Symlinks whose targets stay under
    the root are legitimate filesystem structure, not an escape."""
    root = tmp_path / "rootfs"
    (root / "usr" / "bin").mkdir(parents=True)
    (root / "bin").symlink_to("usr/bin")  # relative, as real merged-usr images store it
    _set_files_root(monkeypatch, root)

    result = await files.upload("bin/tool.sh", b"#!/bin/sh\n")
    assert (root / "usr" / "bin" / "tool.sh").read_bytes() == b"#!/bin/sh\n"
    assert result.size == 10
    assert await files.download("bin/tool.sh") == b"#!/bin/sh\n"


@pytest.mark.asyncio
async def test_files_follows_relative_symlinks_inside_root(tmp_path, monkeypatch):
    root = tmp_path / "rootfs"
    (root / "usr" / "lib").mkdir(parents=True)
    (root / "lib").symlink_to("usr/lib")  # relative target, like real images
    _set_files_root(monkeypatch, root)

    await files.upload("lib/marker", b"x")
    assert (root / "usr" / "lib" / "marker").read_bytes() == b"x"


@pytest.mark.asyncio
async def test_files_follows_file_symlink_inside_root(tmp_path, monkeypatch):
    root = tmp_path / "rootfs"
    root.mkdir()
    (root / "real.txt").write_bytes(b"data")
    (root / "alias.txt").symlink_to("real.txt")
    _set_files_root(monkeypatch, root)

    assert await files.download("alias.txt") == b"data"
    await files.upload("alias.txt", b"new")
    assert (root / "real.txt").read_bytes() == b"new"


@pytest.mark.asyncio
async def test_files_symlinks_pointing_outside_cannot_read_or_write_the_real_target(tmp_path, monkeypatch):
    """Absolute out-of-jail symlinks re-anchor at the root: reads miss the real
    file and writes land inside the jail — the real out-of-jail data is
    never reached."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"SECRET")
    root = tmp_path / "rootfs"
    root.mkdir()
    (root / "evil").symlink_to(outside)  # absolute -> re-anchored under root
    (root / "evil_file").symlink_to(outside / "secret.txt")
    _set_files_root(monkeypatch, root)

    await files.upload("evil/x.txt", b"boom")
    assert not (outside / "x.txt").exists()  # write stayed inside the jail

    with pytest.raises(FileNotFoundError):
        await files.download("evil_file")
    assert (outside / "secret.txt").read_bytes() == b"SECRET"  # untouched


@pytest.mark.asyncio
async def test_files_symlink_cycle_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "rootfs"
    root.mkdir()
    (root / "a").symlink_to("b")
    (root / "b").symlink_to("a")
    _set_files_root(monkeypatch, root)

    with pytest.raises(PermissionError, match="symlink"):
        await files.download("a")


@pytest.mark.asyncio
async def test_files_nested_link_with_dotdot_resolves_like_posix(tmp_path, monkeypatch):
    """`alias -> inner/../victim.txt` where `inner -> dest/sub`: POSIX resolves
    to dest/victim.txt (`..` applies AFTER the inner symlink), not root-level
    victim.txt. Lexical normpath of the target gets this wrong."""
    root = tmp_path / "rootfs"
    (root / "dest" / "sub").mkdir(parents=True)
    (root / "dest" / "victim.txt").write_bytes(b"correct")
    (root / "victim.txt").write_bytes(b"WRONG")
    (root / "inner").symlink_to("dest/sub")
    (root / "alias").symlink_to("inner/../victim.txt")
    _set_files_root(monkeypatch, root)

    assert await files.download("alias") == b"correct"


@pytest.mark.asyncio
async def test_files_link_to_link_chain_resolves(tmp_path, monkeypatch):
    root = tmp_path / "rootfs"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "real.txt").write_bytes(b"deep")
    (root / "hop1").symlink_to("hop2")
    (root / "hop2").symlink_to("a/b/real.txt")
    _set_files_root(monkeypatch, root)

    assert await files.download("hop1") == b"deep"


@pytest.mark.asyncio
async def test_files_symlink_target_dotdot_cannot_escape_root(tmp_path, monkeypatch):
    """A `..`-bearing target that would climb above root is clamped at root
    (kernel RESOLVE_IN_ROOT semantics), never followed outside."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"SECRET")
    root = tmp_path / "rootfs"
    root.mkdir()
    (root / "escape").symlink_to("../outside/secret.txt")
    _set_files_root(monkeypatch, root)

    with pytest.raises(FileNotFoundError):
        # ../outside clamps to root/outside (does not exist) — never the real
        # sibling outside the jail.
        await files.download("escape")
    assert (outside / "secret.txt").read_bytes() == b"SECRET"  # untouched


@pytest.mark.asyncio
async def test_files_absolute_symlink_target_is_reanchored_at_root(tmp_path, monkeypatch):
    """An absolute symlink target is interpreted relative to the jail root
    (chroot-like), so /etc/x -> root/etc/x, never the host's /etc."""
    root = tmp_path / "rootfs"
    (root / "etc").mkdir(parents=True)
    (root / "etc" / "conf").write_bytes(b"jailed")
    (root / "link").symlink_to("/etc/conf")
    _set_files_root(monkeypatch, root)

    assert await files.download("link") == b"jailed"


@pytest.mark.asyncio
async def test_files_link_resolving_exactly_to_root_then_suffix(tmp_path, monkeypatch):
    """A dir symlink whose target is the root itself must still allow a suffix
    under it (empty resolved prefix is valid, not an escape)."""
    root = tmp_path / "rootfs"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "up").symlink_to("..")  # -> root
    (root / "target.txt").write_bytes(b"hit")
    _set_files_root(monkeypatch, root)

    assert await files.download("sub/up/target.txt") == b"hit"


def test_image_env_bashrc_follows_the_env_home(tmp_path, monkeypatch):
    """BASH_ENV must point at the RESULT env's ~/.bashrc — including when the
    image's HOME was stripped+restored or overridden via extra — not at the
    worker process's ambient HOME."""
    from agentix.bash import image_env

    (tmp_path / ".bashrc").write_text("export FROM_IMAGE_RC=1\n")
    env = image_env({"HOME": str(tmp_path)})
    assert env["BASH_ENV"] == str(tmp_path / ".bashrc")

    explicit = image_env({"HOME": str(tmp_path), "BASH_ENV": "/custom/rc"})
    assert explicit["BASH_ENV"] == "/custom/rc"  # caller's value is preserved


def test_shell_executable_follows_the_env_path(tmp_path):
    """The shell resolves from the env's PATH like everything else the
    command runs: an image_env() PATH selects the image's bash (its restored
    LD_PRELOAD etc. target that binary); no bash on PATH falls back."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    image_bash = fakebin / "bash"
    image_bash.write_text("#!/bin/sh\n")
    image_bash.chmod(0o755)

    assert bash._shell_executable(None, {"PATH": str(fakebin)}) == str(image_bash)
    # no bash anywhere on the supplied PATH -> /bin/bash fallback
    # (BUNDLE_RUNTIME_BASH does not exist outside a bundle)
    assert bash._shell_executable(None, {"PATH": str(tmp_path)}) == "/bin/bash"
