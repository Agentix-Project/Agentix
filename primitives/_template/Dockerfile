# Shared closure-image Dockerfile.
#
# Used by every closure under `primitives/<name>/`. `tools/build_closure.py`
# stages a build context that contains:
#
#   ./agentix_closures/<name>/{__init__.py, _impl.py}   — closure source
#   ./pyproject.toml                                    — generated stub
#   ./default.nix                                       — symlink/copy of shared template
#   ./gen_manifest.py                                   — copy of the framework tool
#
# Build context is therefore a self-contained tarball; no out-of-tree
# references inside the Dockerfile. The closure's name is supplied as
# a label only — the rest is determined by the staged content.
ARG NIX_IMAGE=nixos/nix:latest

FROM ${NIX_IMAGE} AS builder
RUN mkdir -p ~/.config/nix && \
    echo 'experimental-features = nix-command flakes' >> ~/.config/nix/nix.conf && \
    nix-channel --update 2>/dev/null || true
WORKDIR /src
COPY . .
RUN nix-build --no-out-link default.nix -o ./result && \
    STORE_PATH=$(readlink -f ./result) && \
    mkdir -p /export/nix/store && \
    for p in $(nix-store -qR "$STORE_PATH"); do cp -a "$p" /export/nix/store/; done && \
    ln -s "$STORE_PATH" /export/nix/entry

FROM busybox:stable
COPY --from=builder /export /
VOLUME /nix
ARG CLOSURE_NAME=unknown
LABEL org.agentix.closure=1
LABEL org.agentix.closure.kind=primitive
LABEL org.agentix.closure.name=${CLOSURE_NAME}
