# TITO Gateway Documentation

[English](index.md) | [中文](index.zh-CN.md)

TITO Gateway is a public package and CLI wrapper around Miles' Agentic Chat
Template / TITO session-server work. It reuses Miles logic and exposes it in a
standalone package.

## Start Here

- [Quickstart](quickstart.md): install, run, and test the package.
- [Concepts](concepts.md): runtime model, session flow, and invariants.
- [Python API](api.md): embedding the gateway beside a backend.
- [CLI Reference](cli.md): serve command and backend discovery.
- [Verification And Tests](verification.md): verifier commands and CPU-fast test setup.
- [Development Notes](development.md): source policy, attribution, and release checks.

## Public Attribution

This package explicitly credits Miles as the source of the underlying TITO
implementation. It is a wrapper and usage layer, not a reimplementation. The
source audit lives in `tito_gateway/VENDORED_MILES_AUDIT.md`.
