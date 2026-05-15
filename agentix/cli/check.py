"""`agentix check` — stub ↔ impl signature drift checker.

Usage:

    agentix check                          # default: primitives/
    agentix check primitives/bash
    agentix check primitives/ ../my-closures/

For each closure directory (one with `agentix_closures/<name>/__init__.py`):

  1. Builds the manifest in memory from `pyproject.toml` (no manifest.json
     required in source — `agentix build` generates it at image build time).
  2. Adds the package root to `sys.path` and dispatches via
     `agentix.dispatch._import_and_register`, which handles both
     explicit `_register.py` and convention-based auto-discovery.
  3. For every bound method, compares the stub's signature against the
     impl's: parameter names, kinds, defaults, annotations, return type.

Exits non-zero on any drift. This is the one class of bug the runtime
itself cannot catch until the first call — it gets caught in CI instead.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import typing
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentix.dispatch import _import_and_register
from agentix.models import AGENTIX_CLOSURE_ABI, ClosureManifest

# `agentix check` is a dev-time tool — `gen_manifest` lives in tools/ next
# to other build infra and is the canonical source of manifest derivation
# logic (it must work without the framework installed, for nix-side use).
REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from gen_manifest import generate as _gen_manifest  # noqa: E402


@dataclass
class Mismatch:
    closure: str           # closure package, e.g. "agentix_closures.bash"
    method: str            # bound method name
    field: str             # what differs: "param.<name>", "return", "kind"
    stub: str              # stub-side rendering
    impl: str              # impl-side rendering

    def render(self) -> str:
        return (
            f"  {self.closure}.{self.method} :: {self.field}\n"
            f"    stub: {self.stub}\n"
            f"    impl: {self.impl}"
        )


def _iter_closure_dirs(roots: Iterable[Path]) -> Iterator[Path]:
    """Yield closure directories under each root.

    A closure directory exposes `agentix_closures/<name>/__init__.py`.
    """
    def _has_closure(d: Path) -> bool:
        return any(d.glob("agentix_closures/*/__init__.py"))

    for r in roots:
        if _has_closure(r):
            yield r
            continue
        if r.is_dir():
            for child in sorted(r.iterdir()):
                if child.is_dir() and _has_closure(child):
                    yield child


def _load_manifest(closure_dir: Path) -> ClosureManifest:
    """Manifest from `pyproject.toml` (or a pre-shipped manifest.json)."""
    pre = closure_dir / "manifest.json"
    if pre.is_file():
        return ClosureManifest.model_validate_json(pre.read_text())
    pp = closure_dir / "pyproject.toml"
    if not pp.is_file():
        raise SystemExit(f"{closure_dir}: missing pyproject.toml")
    raw = _gen_manifest(pp)
    return ClosureManifest.model_validate(raw)


def _load_dispatcher(closure_dir: Path, manifest: ClosureManifest):
    """Make the closure importable, then build its dispatcher.

    Delegates to `agentix.dispatch._import_and_register`, which handles
    both shapes:
      * explicit `_register.py` (escape hatch)
      * convention-based auto-discovery from `__init__.py` + `_impl.py`
    """
    py_root = closure_dir
    # Two layouts: source tree's `agentix_closures/<name>/` directly, or the
    # docker image's `entry/python/agentix_closures/<name>/`. Try both.
    candidates = [closure_dir, closure_dir / "entry" / "python"]
    for cand in candidates:
        if (cand / Path(*manifest.package.split("."))).is_dir():
            py_root = cand
            break
    py_str = str(py_root)
    if py_str not in sys.path:
        sys.path.insert(0, py_str)
    return _import_and_register(manifest)


def _resolved_hints(fn: object) -> dict[str, typing.Any]:
    """Best-effort resolution of `fn`'s annotations to real types.

    Falls back to raw `__annotations__` when `get_type_hints` can't
    evaluate a forward ref — better to compare strings than to crash.
    """
    try:
        return typing.get_type_hints(fn)  # type: ignore[arg-type]
    except Exception:
        return dict(getattr(fn, "__annotations__", {}))


def _render(value: object) -> str:
    """Stable string rendering for diff output."""
    if value is inspect.Parameter.empty or value is inspect.Signature.empty:
        return "<empty>"
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _compare(
    closure: str,
    method: str,
    stub: object,
    impl: object,
) -> list[Mismatch]:
    """Compare two callables; one mismatch per differing attribute."""
    stub_sig = inspect.signature(stub, eval_str=True)  # type: ignore[arg-type]
    impl_sig = inspect.signature(impl, eval_str=True)  # type: ignore[arg-type]
    stub_hints = _resolved_hints(stub)
    impl_hints = _resolved_hints(impl)

    out: list[Mismatch] = []

    stub_params = list(stub_sig.parameters.values())
    impl_params = list(impl_sig.parameters.values())
    # `self` is class-method syntactic noise — drop from both sides. The
    # dispatcher strips `self` at bind time; mirror that here.
    if stub_params and stub_params[0].name == "self":
        stub_params = stub_params[1:]
    if impl_params and impl_params[0].name == "self":
        impl_params = impl_params[1:]

    if [p.name for p in stub_params] != [p.name for p in impl_params]:
        out.append(Mismatch(
            closure, method, "param.names",
            stub=", ".join(p.name for p in stub_params),
            impl=", ".join(p.name for p in impl_params),
        ))
        return out

    for sp, ip in zip(stub_params, impl_params):
        if sp.kind is not ip.kind:
            out.append(Mismatch(
                closure, method, f"param.{sp.name}.kind",
                stub=str(sp.kind), impl=str(ip.kind),
            ))
        # `==` not `is` — large ints live outside Python's small-int cache.
        if sp.default != ip.default:
            out.append(Mismatch(
                closure, method, f"param.{sp.name}.default",
                stub=_render(sp.default), impl=_render(ip.default),
            ))
        s_ann = stub_hints.get(sp.name, sp.annotation)
        i_ann = impl_hints.get(ip.name, ip.annotation)
        if s_ann != i_ann:
            out.append(Mismatch(
                closure, method, f"param.{sp.name}.annotation",
                stub=_render(s_ann), impl=_render(i_ann),
            ))

    s_ret = stub_hints.get("return", stub_sig.return_annotation)
    i_ret = impl_hints.get("return", impl_sig.return_annotation)
    if s_ret != i_ret:
        out.append(Mismatch(
            closure, method, "return",
            stub=_render(s_ret), impl=_render(i_ret),
        ))
    return out


def check_closure(closure_dir: Path) -> list[Mismatch]:
    manifest = _load_manifest(closure_dir)
    if manifest.abi != AGENTIX_CLOSURE_ABI:
        raise ValueError(
            f"{closure_dir}: manifest.abi={manifest.abi} but expected "
            f"{AGENTIX_CLOSURE_ABI}"
        )
    dispatcher = _load_dispatcher(closure_dir, manifest)
    mismatches: list[Mismatch] = []
    for method_name in dispatcher.methods():
        bound = dispatcher._methods[method_name]  # noqa: SLF001 — checker tool
        mismatches.extend(_compare(
            manifest.package, method_name, bound.stub, bound.impl,
        ))
    return mismatches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentix check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[REPO_ROOT / "primitives"],
        help="closure dirs or roots containing them (default: primitives/)",
    )
    args = parser.parse_args(argv)

    closures = list(_iter_closure_dirs(args.roots))
    if not closures:
        print(f"no closures found under {args.roots!r}", file=sys.stderr)
        return 2

    all_mismatches: list[Mismatch] = []
    for cdir in closures:
        try:
            all_mismatches.extend(check_closure(cdir))
        except Exception as exc:
            print(f"FAIL {cdir}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    if all_mismatches:
        print(f"stub↔impl drift in {len({m.closure for m in all_mismatches})} closure(s):")
        for m in all_mismatches:
            print(m.render())
        return 1

    print(f"checked {len(closures)} closure(s); no drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
