# syntax=docker/dockerfile:1
#
# agentix bundle builder. The transient build image contains `/nix`
# with everything a sandbox needs. The host build command copies that
# tree into the portable bundle tar; provider backends later make the
# same tree visible in the task container at `/nix`.
#
#   /nix/store/...              the closures (interpreter, uv, system deps)
#   /nix/runtime/venv           the uv venv — all Python deps
#   /nix/runtime/{bin,lib,...}  symlinkJoin of every closure
#
# Nix runs only inside this build — the host needs just docker.

# Pinned by digest: the floating `latest` tag moved to an image whose
# nix binary segfaults under qemu-user x86_64 emulation on arm64 hosts
# (breaks the cross-arch e2e matrix), and an unpinned base makes bundle
# builds unreproducible. Override with --build-arg to test newer images.
ARG AGENTIX_BUILDER_BASE=nixos/nix:latest@sha256:bf1d938835ab96312f098fa6c2e9cab367728e0aad0646ee3e02a787c80d8fb8
FROM ${AGENTIX_BUILDER_BASE}

# Flakes + the modern CLI; accept flake `nixConfig` for substituters.
# Syscall filtering can fail under Docker buildx emulation; the Docker
# build sandbox is already the outer isolation boundary here.
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf \
 && echo "accept-flake-config = true" >> /etc/nix/nix.conf \
 && echo "filter-syscalls = false" >> /etc/nix/nix.conf

# Where the project sits inside the copied repo (a uv-workspace member
# or cookbook example lives in a subdirectory). Passed by `agentix build`.
ARG AGENTIX_PROJECT_SUBPATH=.
ENV AGENTIX_PROJECT_SUBPATH=${AGENTIX_PROJECT_SUBPATH}

# Raw passthrough args from `agentix build --nix-arg ... --uv-arg ...`,
# word-split by bundle-build.sh onto `nix build` / `uv sync`. This is how nix
# knobs (substituters, trusted keys, cores, ...) reach the build — e.g.
# --nix-arg "--option extra-substituters https://mirror...".
ARG AGENTIX_NIX_ARGS=
ARG AGENTIX_UV_ARGS=
ENV AGENTIX_NIX_ARGS=${AGENTIX_NIX_ARGS} \
    AGENTIX_UV_ARGS=${AGENTIX_UV_ARGS}

COPY . /build
WORKDIR /build

# nix build (toolchain + runtime), uv venv + uv sync, closure discovery.
# HTTP/HTTPS proxies, when needed, are inherited from BuildKit via the
# host's `~/.docker/config.json` `proxies` section — no per-build plumbing.
RUN sh ./bundle-build.sh

ENV PATH="/nix/runtime/venv/bin:/nix/runtime/bin:${PATH}" \
    LD_LIBRARY_PATH="/nix/runtime/lib" \
    LIBRARY_PATH="/nix/runtime/lib" \
    CPATH="/nix/runtime/include" \
    C_INCLUDE_PATH="/nix/runtime/include" \
    CPLUS_INCLUDE_PATH="/nix/runtime/include" \
    PKG_CONFIG_PATH="/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig" \
    CMAKE_PREFIX_PATH="/nix/runtime" \
    AGENTIX_ADDED_PATH="/nix/runtime/venv/bin:/nix/runtime/bin" \
    AGENTIX_ADDED_LD_LIBRARY_PATH="/nix/runtime/lib" \
    AGENTIX_ADDED_LIBRARY_PATH="/nix/runtime/lib" \
    AGENTIX_ADDED_CPATH="/nix/runtime/include" \
    AGENTIX_ADDED_C_INCLUDE_PATH="/nix/runtime/include" \
    AGENTIX_ADDED_CPLUS_INCLUDE_PATH="/nix/runtime/include" \
    AGENTIX_ADDED_PKG_CONFIG_PATH="/nix/runtime/lib/pkgconfig:/nix/runtime/share/pkgconfig" \
    AGENTIX_ADDED_CMAKE_PREFIX_PATH="/nix/runtime" \
    AGENTIX_BIND_PORT=8000
EXPOSE 8000/tcp
VOLUME ["/nix"]
ENTRYPOINT ["/nix/runtime/bootstrap.sh"]
