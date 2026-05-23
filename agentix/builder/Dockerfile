# syntax=docker/dockerfile:1
#
# agentix bundle builder. The produced image is a *runtime carrier*:
# its `/nix` holds everything a sandbox needs and is mounted into a
# task container at `/nix` (`--mount type=image,subpath=nix`).
#
#   /nix/store/...              the closures (interpreter, uv, system deps)
#   /nix/runtime/venv           the uv venv — all Python deps
#   /nix/runtime/{bin,lib,...}  symlinkJoin of every closure
#
# Nix runs only inside this build — the host needs just docker.

FROM nixos/nix:latest

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

COPY . /build
WORKDIR /build

# nix build (toolchain + runtime), uv venv + uv sync, closure discovery.
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
