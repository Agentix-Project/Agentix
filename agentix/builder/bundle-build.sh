#!/bin/sh
# In-container build orchestration for `agentix build`. Run by the
# Dockerfile against the staged context (`repo/`, `flake.nix`,
# `closures/`, `python-version`).
#
#   1. nix build the toolchain  — interpreter + uv
#   2. uv venv + uv sync        — /nix/runtime/venv (all Python deps)
#   3. discover system closures — plugins → closures/plugins/<label>.nix
#                                 project → closures/project.nix
#   4. nix build the runtime    — toolchain + project, merged
#   5. nix build the plugins    — each plugin tree, kept independent
#   6. place both trees         — /nix/runtime/{bin,lib,...}
#                                 /nix/runtime/plugins/<label>/
#   7. write bootstrap.sh       — bundle's startup contract
set -eu

SUBPATH="${AGENTIX_PROJECT_SUBPATH:-.}"
PROJECT="/build/repo/${SUBPATH}"

if [ ! -f "${PROJECT}/pyproject.toml" ]; then
    echo "bundle-build: no pyproject.toml at ${PROJECT}" >&2
    exit 1
fi

# Flakes only see git-tracked files; the context is not a repo yet.
git init -q
git config user.email build@agentix.local
git config user.name agentix-build
git add -A

echo ">>> [1/5] building Nix toolchain (interpreter + uv)"
nix build .#toolchain ${AGENTIX_NIX_ARGS:-} -o toolchain-result --print-build-logs
TOOLCHAIN="$(readlink -f toolchain-result)"

echo ">>> [2/5] creating venv + syncing Python deps"
mkdir -p /nix/runtime
"${TOOLCHAIN}/bin/uv" venv /nix/runtime/venv --python "${TOOLCHAIN}/bin/python3"
# A committed uv.lock pins the closure (`--frozen`); without one uv resolves
# fresh. Either way the bundle gets a non-editable, prod-only dependency set.
if [ -f "${PROJECT}/uv.lock" ]; then UV_FROZEN="--frozen"; else UV_FROZEN=""; echo ">>>     no uv.lock — resolving dependencies fresh"; fi
( cd "${PROJECT}" \
  && VIRTUAL_ENV=/nix/runtime/venv "${TOOLCHAIN}/bin/uv" sync \
        --active ${UV_FROZEN} --no-dev --no-editable ${AGENTIX_UV_ARGS:-} )

echo ">>> [3/7] discovering system-dep closures"
/nix/runtime/venv/bin/python -m agentix.cli.build.closures \
    --project "${PROJECT}" --closures /build/closures
git add -A

echo ">>> [4/7] building Nix runtime closure (toolchain + project)"
nix build .#runtime ${AGENTIX_NIX_ARGS:-} -o runtime-result --print-build-logs

echo ">>> [5/7] building Nix plugin closures (independent trees)"
nix build .#plugins ${AGENTIX_NIX_ARGS:-} -o plugins-result --print-build-logs

echo ">>> [6/7] placing /nix/runtime + /nix/runtime/plugins"
cp -a runtime-result/. /nix/runtime/
mkdir -p /nix/runtime/plugins
# `cp -a` preserves the linkFarm's `<label> -> /nix/store/<plugin>`
# symlinks; the targets are already part of the image (every store
# path the linkFarm references is a build-time dep). Bootstrap globs
# `/nix/runtime/plugins/*/` and trusts the OS to resolve them.
cp -a plugins-result/. /nix/runtime/plugins/
rm -f toolchain-result runtime-result plugins-result

# Drop the bundle's startup contract at /nix/runtime/bootstrap.sh.
# Provider backends (docker, apptainer, future k8s/...) just exec
# this — they stay agnostic about Python venvs, LD_LIBRARY_PATH, or
# where the runtime server lives. The script is shipped verbatim as
# wheel data from `agentix/builder/bootstrap.sh` and staged into the
# build context next to bundle-build.sh.
echo ">>> [7/7] installing /nix/runtime/bootstrap.sh"
install -m 0755 /build/bootstrap.sh /nix/runtime/bootstrap.sh

echo ">>> bundle build complete"
