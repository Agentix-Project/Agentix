"""`agentix build` — package a Python project into a bundle artifact.

Usage:

    agentix build                         # current directory's project
    agentix build path/to/project         # explicit project root
    agentix build . --name hello-agentix  # bundle tar (auto-appends version)
    agentix build . --name hello:dev      # bundle tar tagged as dev
    agentix build . --platform linux/amd64
    agentix build . --format oci-image    # Docker-compatible image path
    agentix build . --format tar          # Agentix bundle tar (default)
    agentix build . --dry-run             # stage the build context only

The argument is a project root — a directory with `pyproject.toml` +
`uv.lock`. The build splits cleanly along one line:

  * **Python deps** are uv's job. Inside the build container `uv venv`
    + `uv sync` materialize the project's full dependency closure into
    `/nix/runtime/venv` — exactly the venv uv would produce anywhere.
  * **System deps** are Nix's job. The interpreter + uv come from a
    Nix toolchain closure; plugins and the project contribute
    `{ pkgs }: drv` files that Nix builds and `symlinkJoin`s into
    `/nix/runtime`. Nix never touches Python packaging — no uv2nix.

The host side is deliberately thin: find the project's git repo, copy
it into a build context, and hand the context to a Docker build executor.
Every heavy step — `uv venv`, `uv sync`, `nix build` — happens inside
the container. The host needs only `agentixx`, `docker`, and `git`;
no project venv, no uv, no nix.

The platform is the sandbox runtime platform, not the host platform.
On macOS, for example, `--platform linux/amd64` builds a Linux x86_64
bundle for a remote x86 sandbox.

A project that path-depends on siblings (a uv workspace, a cookbook
example) needs those siblings in the build context — so the context is
the whole git repository, with the project addressed by its subpath.

The portable bundle tar layout is:

    manifest.json          bundle identity + runtime contract
    nix/store/...          the closures: interpreter, uv, system deps
    nix/runtime/venv       the uv venv (all Python deps)
    nix/runtime/{bin,lib,...}   symlinkJoin of every closure

The Docker-compatible `oci-image` layout is the same `/nix` tree inside
an image, optimized for `DockerDeployment`'s `--volumes-from` fast path:

    /nix/store/...        the closures: interpreter, uv, system deps
    /nix/runtime/venv     the uv venv (all Python deps)
    /nix/runtime/{bin,lib,...}   symlinkJoin of every closure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as host_platform
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from uuid import uuid4

from agentix.cli._resolve import REPO_ROOT, detect_python_version, read_pyproject, short_name

# Directories never copied into the build context — caches, build
# outputs, virtualenvs, VCS metadata. The context is hashed by Docker;
# keeping it lean keeps builds fast and cacheable.
_SOURCE_SKIP = frozenset({
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "result",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".direnv",
    "node_modules",
})

# Files staged verbatim from `agentix/nix/` into the build context.
_BUILDER_FILES = ("flake.nix", "flake.lock", "Dockerfile", "bundle-build.sh")

_BUILD_FORMATS = ("tar", "oci-image")

_RUNTIME_ENTRYPOINT = "/nix/runtime/venv/bin/agentix-server"

_RUNTIME_ENV = {
    "PATH": "/nix/runtime/venv/bin:/nix/runtime/bin",
    "LD_LIBRARY_PATH": "/nix/runtime/lib",
    "LIBRARY_PATH": "/nix/runtime/lib",
    "CPATH": "/nix/runtime/include",
    "C_INCLUDE_PATH": "/nix/runtime/include",
    "CPLUS_INCLUDE_PATH": "/nix/runtime/include",
    "PKG_CONFIG_PATH": "/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig",
    "CMAKE_PREFIX_PATH": "/nix/runtime",
}

_DOCKER_TO_NIX_SYSTEM = {
    "linux/amd64": "x86_64-linux",
    "linux/arm64": "aarch64-linux",
}

_PLATFORM_ALIASES = {
    "linux/amd64": "linux/amd64",
    "linux/x86-64": "linux/amd64",
    "amd64": "linux/amd64",
    "x86-64": "linux/amd64",
    "linux/arm64": "linux/arm64",
    "linux/arm64/v8": "linux/arm64",
    "linux/aarch64": "linux/arm64",
    "arm64": "linux/arm64",
    "aarch64": "linux/arm64",
}


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command; raise SystemExit on failure."""
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True)
    if check and proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stderr or "")
        raise SystemExit(proc.returncode)
    return proc


def normalize_platform(value: str) -> str:
    """Normalize a user platform into Docker's OS/arch form."""
    key = value.strip().lower().replace("_", "-")
    platform = _PLATFORM_ALIASES.get(key)
    if platform is None:
        supported = ", ".join(sorted(_DOCKER_TO_NIX_SYSTEM))
        raise SystemExit(f"--platform {value!r}: supported values are {supported}")
    return platform


def detect_default_platform(machine: str | None = None) -> str:
    """Best-effort default Docker platform for the current build host.

    Agentix builds Linux container images even when invoked from macOS,
    so only the CPU architecture is inherited from the host.
    """
    raw = (machine or host_platform.machine()).strip().lower().replace("_", "-")
    if raw in {"amd64", "x86-64"}:
        return "linux/amd64"
    if raw in {"arm64", "aarch64"}:
        return "linux/arm64"
    raise SystemExit(f"cannot auto-detect Docker platform from machine {raw!r}; pass --platform")


def nix_system_for_platform(platform: str) -> str:
    """Return the Nix system matching a normalized Docker platform."""
    platform = normalize_platform(platform)
    return _DOCKER_TO_NIX_SYSTEM[platform]


def git_toplevel(path: Path) -> Path | None:
    """The git work-tree root containing `path`, or None when `path`
    is not inside a git repository."""
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    top = proc.stdout.strip()
    return Path(top).resolve() if top else None


def resolve_context(src: Path) -> tuple[Path, Path]:
    """Return `(context_root, project_subpath)` for a project at `src`.

    The context root is the project's git repository — copying the
    whole repo is what lets in-container `uv sync` resolve path
    dependencies that point outside the project directory (`../..`,
    `../../plugins/*`). `project_subpath` locates the project within
    the staged copy.

    A project not in a git repo is its own context (`project_subpath`
    is `.`); that supports only registry/git Python deps, since there
    is nothing outside the directory to copy.
    """
    src = src.resolve()
    top = git_toplevel(src)
    if top is None:
        return src, Path(".")
    return top, src.relative_to(top)


def _shipped(name: str) -> bytes:
    """Read a builder file shipped as `agentix/nix/<name>` package data."""
    f = resources.files("agentix") / "nix" / name
    if not f.is_file():
        raise SystemExit(f"shipped builder file {name!r} missing — reinstall agentixx")
    return f.read_bytes()


def stage_context(
    stage: Path,
    *,
    context_root: Path,
    python_version: str,
    platform: str,
) -> None:
    """Lay out the Docker build context under `stage`.

        stage/repo/            copy of the git repo (skip-listed)
        stage/flake.nix        Nix builder (verbatim)
        stage/flake.lock       pinned nixpkgs (verbatim)
        stage/Dockerfile       build container (verbatim)
        stage/bundle-build.sh  in-container orchestration (verbatim)
        stage/python-version   interpreter minor, read by flake.nix
        stage/nix-system       target Nix system, read by flake.nix
        stage/closures/        empty — filled in-container by `_assemble`
    """
    platform = normalize_platform(platform)
    stage.mkdir(parents=True, exist_ok=True)

    repo_dest = stage / "repo"
    shutil.copytree(
        context_root,
        repo_dest,
        ignore=shutil.ignore_patterns(*_SOURCE_SKIP),
        symlinks=True,
    )

    for name in _BUILDER_FILES:
        (stage / name).write_bytes(_shipped(name))

    (stage / "python-version").write_text(f"{python_version}\n")
    (stage / "nix-system").write_text(f"{nix_system_for_platform(platform)}\n")
    (stage / "closures").mkdir(exist_ok=True)
    # git won't track an empty dir; the flake guards on pathExists, but
    # a marker keeps the dir present in the context tarball.
    (stage / "closures" / ".keep").write_text("")


def _docker_build_image(stage: Path, *, tags: list[str], project_subpath: Path, platform: str) -> None:
    """`docker buildx build --load` the staged context with explicit tags."""
    if not tags:
        raise SystemExit("internal error: docker image build requires at least one tag")
    _run([
        "docker",
        "buildx",
        "build",
        "--platform",
        normalize_platform(platform),
        "--load",
        *(arg for tag in tags for arg in ("-t", tag)),
        "--build-arg",
        f"AGENTIX_PROJECT_SUBPATH={project_subpath}",
        "--progress=plain",
        str(stage),
    ])


def _docker_build(stage: Path, *, name: str, tag: str, project_subpath: Path, platform: str) -> str:
    """Build the Docker-compatible bundle image; return the primary image ref.

    A bare `NAME` is also tagged `NAME:latest` for convenience.
    """
    ref = f"{name}:{tag}"
    tags = [ref]
    if tag != "latest":
        tags.append(f"{name}:latest")
    _docker_build_image(stage, tags=tags, project_subpath=project_subpath, platform=platform)
    return ref


def _platform_slug(platform: str) -> str:
    return normalize_platform(platform).replace("/", "-")


def _artifact_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return component or "bundle"


def _docker_tag_component(value: str) -> str:
    component = _artifact_component(value).lower().strip(".-")
    if not component:
        return "bundle"
    if not re.match(r"[a-z0-9_]", component):
        component = f"_{component}"
    return component


def _tar_cache_image_ref(*, name: str, project_subpath: Path, platform: str) -> str:
    platform = normalize_platform(platform)
    digest = hashlib.sha256(f"{name}\0{project_subpath.as_posix()}\0{platform}".encode()).hexdigest()[:12]
    platform_part = _platform_slug(platform)
    max_name_len = 128 - len(platform_part) - len(digest) - 2
    name_part = _docker_tag_component(name)[:max_name_len].rstrip(".-") or "bundle"
    return f"agentix-bundle-cache:{name_part}-{platform_part}-{digest}"


def _default_tar_name(name: str, tag: str, platform: str) -> str:
    return f"{_artifact_component(name)}-{_artifact_component(tag)}-{_platform_slug(platform)}.bundle.tar"


def _tar_output_path(output: str | None, *, name: str, tag: str, platform: str) -> Path:
    default_name = _default_tar_name(name, tag, platform)
    if output is None:
        return Path("dist") / default_name

    path = Path(output)
    if output.endswith(os.sep) or path.is_dir() or path.suffix == "":
        return path / default_name
    return path


def _tree_digest(root: Path) -> str:
    """Stable content digest for a bundle tree.

    The digest is over relative paths, file modes, symlink targets, and
    file bytes. It deliberately ignores uid/gid/mtime so the same bundle
    copied between hosts keeps the same identity.
    """
    h = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        st = path.lstat()
        mode = st.st_mode & 0o7777
        h.update(rel.encode())
        h.update(b"\0")
        h.update(str(mode).encode())
        h.update(b"\0")
        if path.is_symlink():
            h.update(b"L\0")
            h.update(os.readlink(path).encode())
            h.update(b"\0")
        elif path.is_dir():
            h.update(b"D\0")
        elif path.is_file():
            h.update(b"F\0")
            h.update(str(st.st_size).encode())
            h.update(b"\0")
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        else:
            raise SystemExit(f"{path}: unsupported file type in bundle")
    return h.hexdigest()


def _mapped_symlink_target(nix_root: Path, link: Path, target: str) -> Path:
    if target.startswith("/"):
        if target == "/nix":
            return nix_root
        if target.startswith("/nix/"):
            return nix_root / target.removeprefix("/nix/")
        raise SystemExit(f"{link}: symlink target {target!r} points outside /nix")

    parent = link.parent.relative_to(nix_root).as_posix()
    rel = posixpath.normpath(posixpath.join(parent, target))
    if rel == ".":
        return nix_root
    if rel == ".." or rel.startswith("../"):
        raise SystemExit(f"{link}: symlink target {target!r} escapes /nix")
    return nix_root / rel


def _validate_bundle_tree(nix_root: Path) -> None:
    """Validate the runtime-facing part of the extracted /nix tree.

    Nix store paths can contain incidental or profile-oriented symlinks
    that are not part of Agentix's runtime contract. Keep validation
    focused on `/nix/runtime`, which is what deployments execute and
    prepend to the sandbox environment.
    """
    entrypoint = nix_root / _RUNTIME_ENTRYPOINT.removeprefix("/nix/")
    if not os.path.lexists(entrypoint):
        raise SystemExit(f"bundle missing runtime entrypoint: {_RUNTIME_ENTRYPOINT}")

    runtime_root = nix_root / "runtime"
    if not runtime_root.is_dir():
        raise SystemExit("bundle missing runtime tree: /nix/runtime")

    for path in sorted(runtime_root.rglob("*"), key=lambda p: p.relative_to(nix_root).as_posix()):
        if not path.is_symlink():
            continue
        target = os.readlink(path)
        mapped = _mapped_symlink_target(nix_root, path, target)
        if not os.path.lexists(mapped):
            raise SystemExit(f"{path}: broken symlink target {target!r} in bundle /nix tree")


def _bundle_manifest(*, name: str, tag: str, platform: str, digest: str) -> dict[str, object]:
    runtime_env = dict(_RUNTIME_ENV)
    added_env = {f"AGENTIX_ADDED_{key}": value for key, value in runtime_env.items()}
    return {
        "schema_version": 1,
        "format": "agentix-bundle",
        "name": name,
        "tag": tag,
        "platform": normalize_platform(platform),
        "nix_system": nix_system_for_platform(platform),
        "digest": f"sha256:{digest}",
        "entrypoint": _RUNTIME_ENTRYPOINT,
        "runtime_env": runtime_env,
        "agentix_added_env": added_env,
    }


def _portable_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_bundle_tar(bundle_root: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with tarfile.open(tmp_path, "w") as tar:
            tar.add(bundle_root / "manifest.json", arcname="manifest.json", filter=_portable_tarinfo)
            tar.add(bundle_root / "nix", arcname="nix", recursive=True, filter=_portable_tarinfo)
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _checked_nix_member_name(name: str) -> str:
    normalized = posixpath.normpath(name)
    if normalized in {"", "."} or normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise SystemExit(f"bundle image produced unsafe tar member: {name!r}")
    if normalized != "nix" and not normalized.startswith("nix/"):
        raise SystemExit(f"bundle image produced non-/nix tar member: {name!r}")
    return normalized


def _extract_nix_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    bundle_root: Path,
    deferred_dirs: list[tuple[Path, int]],
) -> None:
    member.name = _checked_nix_member_name(member.name)
    if member.islnk():
        _checked_nix_member_name(member.linkname)
    if member.isdir():
        original_mode = member.mode
        deferred_dirs.append((bundle_root / member.name, original_mode))
        member.mode = original_mode | 0o700
    tar.extract(member, bundle_root)


def _copy_nix_from_image(image_ref: str, bundle_root: Path, *, platform: str) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--platform",
        normalize_platform(platform),
        "--entrypoint",
        "tar",
        image_ref,
        "-C",
        "/",
        "-cf",
        "-",
        "nix",
    ]
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    with TemporaryFile() as stderr:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        if proc.stdout is None:
            raise SystemExit("docker run did not provide a stdout pipe")
        deferred_dirs: list[tuple[Path, int]] = []
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    _extract_nix_member(tar, member, bundle_root, deferred_dirs)
        finally:
            proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            stderr.seek(0)
            message = stderr.read().decode(errors="replace")
            raise SystemExit(message or rc)
    for path, mode in sorted(deferred_dirs, key=lambda item: len(item[0].parts), reverse=True):
        os.chmod(path, mode)


def _build_tar_bundle(
    stage: Path,
    *,
    output_path: Path,
    name: str,
    tag: str,
    project_subpath: Path,
    platform: str,
) -> Path:
    """Build a portable Agentix bundle tar containing manifest.json + nix/."""
    cache_ref = _tar_cache_image_ref(name=name, project_subpath=project_subpath, platform=platform)
    with TemporaryDirectory(prefix="agentix-bundle-") as tmp:
        bundle_root = Path(tmp) / "bundle"
        bundle_root.mkdir()
        _docker_build_image(stage, tags=[cache_ref], project_subpath=project_subpath, platform=platform)
        _copy_nix_from_image(cache_ref, bundle_root, platform=platform)
        if not (bundle_root / "nix").is_dir():
            raise SystemExit(f"docker image {cache_ref!r} did not contain /nix")
        _validate_bundle_tree(bundle_root / "nix")
        digest = _tree_digest(bundle_root / "nix")
        manifest = _bundle_manifest(name=name, tag=tag, platform=platform, digest=digest)
        (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _write_bundle_tar(bundle_root, output_path)
    return output_path


def parse_name(arg: str | None, pyproject: dict) -> tuple[str, str]:
    """Parse `--name` into `(name, tag)`.

      * None       → (short_name, pyproject version)
      * "NAME"     → ("NAME", pyproject version)
      * "NAME:TAG" → ("NAME", "TAG")
    """
    project = pyproject.get("project", {})
    version = project.get("version")
    default_tag = version if isinstance(version, str) and version else "latest"

    if arg is None:
        return short_name(pyproject), default_tag
    if ":" in arg:
        name, _, tag = arg.partition(":")
        if not name or not tag:
            raise SystemExit(f"--name {arg!r}: both sides of ':' must be non-empty")
        return name, tag
    return arg, default_tag


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentix build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="project root with pyproject.toml + uv.lock (default: current dir)",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="bundle NAME or NAME:TAG. Bare NAME gets ':<pyproject-version>'; "
        "NAME:TAG is used verbatim. Default: derived from pyproject.",
    )
    parser.add_argument(
        "--format",
        choices=_BUILD_FORMATS,
        default="tar",
        help="artifact format: 'tar' writes manifest.json + nix/ (default); "
        "'oci-image' loads a Docker-compatible image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output file or directory for --format tar. Default: "
        "dist/<name>-<tag>-<platform>.bundle.tar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stage the build context to ./build/<name>/ and stop",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="target Linux container platform for the sandbox runtime "
        "(linux/amd64 or linux/arm64; default: auto-detect local CPU)",
    )
    args = parser.parse_args(argv)

    src = Path(args.path).resolve()
    if not src.is_dir():
        raise SystemExit(f"{src}: not a directory")

    pyproject = read_pyproject(src)
    if not (src / "uv.lock").is_file():
        raise SystemExit(f"{src}/uv.lock missing — run `uv lock` first")

    name, tag = parse_name(args.name, pyproject)
    python_version = detect_python_version(pyproject)
    platform = normalize_platform(args.platform) if args.platform else detect_default_platform()
    context_root, project_subpath = resolve_context(src)
    tar_output = _tar_output_path(args.output, name=name, tag=tag, platform=platform)

    if args.output and args.format != "tar":
        raise SystemExit("--output is only supported with --format tar")

    if args.dry_run:
        out = REPO_ROOT / "build" / name
        if out.exists():
            shutil.rmtree(out)
        stage_context(out, context_root=context_root, python_version=python_version, platform=platform)
        print(f"staged build context → {out}")
        print(f"  bundle           → {name}:{tag}")
        print(f"  format           → {args.format}")
        if args.format == "tar":
            print(f"  output           → {tar_output}")
        print(f"  platform         → {platform}")
        print(f"  nix system       → {nix_system_for_platform(platform)}")
        print(f"  python           → 3.{python_version[1:]}")
        print(f"  context root     → {context_root}")
        print(f"  project subpath  → {project_subpath}")
        return 0

    with TemporaryDirectory(prefix="agentix-build-") as tmp:
        stage = Path(tmp) / "ctx"
        stage_context(stage, context_root=context_root, python_version=python_version, platform=platform)
        if args.format == "oci-image":
            ref = _docker_build(stage, name=name, tag=tag, project_subpath=project_subpath, platform=platform)
            print(f"\nimage ready → {ref}", file=sys.stderr)
            if tag != "latest":
                print(f"            → {name}:latest", file=sys.stderr)
        else:
            path = _build_tar_bundle(
                stage,
                output_path=tar_output,
                name=name,
                tag=tag,
                project_subpath=project_subpath,
                platform=platform,
            )
            print(f"\nbundle ready → {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
