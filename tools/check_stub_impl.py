"""Stub ↔ impl signature drift checker.

Run as:

    python tools/check_stub_impl.py [primitives/bash ...]

For each closure directory (one that contains `manifest.json` + an
`agentix_closures/<name>/` package), the script:

  1. Reads the manifest to find the closure's Python import path.
  2. Adds the closure's package root to `sys.path`.
  3. Imports the closure and calls `_register.register()` to obtain a
     populated `Dispatcher`. This gives the same (stub, impl) pairings
     the live runtime would use.
  4. For every bound method, compares the stub's signature against the
     impl's: parameter names, annotations, defaults, return type.

Exits non-zero on any drift. The check works uniformly for the legacy
module-function style (current bash primitive) and for the upcoming
class-based `Namespace` style — both bottom out at `Dispatcher.bind()`.

The drift this catches is the one we can't catch any other way: a
stub author and an impl author renaming a parameter or adjusting a
default without each other. The dispatcher would only notice at the
first runtime call; this script notices at CI time.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import typing
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Resolve project root so we can import `agentix.*` when run from a checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentix.dispatch import Dispatcher, _import_and_register  # noqa: E402
from agentix.models import AGENTIX_CLOSURE_ABI, ClosureManifest  # noqa: E402
from tools.gen_manifest import generate as _gen_manifest  # noqa: E402


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
    """Yield every closure directory under each root.

    A closure directory is one whose layout exposes
    `agentix_closures/<name>/__init__.py`. `manifest.json` is no longer
    required in source — it's generated at build time from `__init__.py`
    metadata.
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
    """Build the manifest in memory from the closure's `__init__.py`.

    Falls back to a pre-generated `manifest.json` if present — useful
    for closures that live outside the convention or that ship a
    customized manifest.
    """
    pre = closure_dir / "manifest.json"
    if pre.is_file():
        return ClosureManifest.model_validate_json(pre.read_text())
    init_pys = sorted(closure_dir.glob("agentix_closures/*/__init__.py"))
    if not init_pys:
        raise SystemExit(f"{closure_dir}: no agentix_closures/<name>/__init__.py found")
    if len(init_pys) > 1:
        names = [str(p.relative_to(closure_dir)) for p in init_pys]
        raise SystemExit(
            f"{closure_dir}: multiple closure packages found ({names}); "
            f"one closure per directory please"
        )
    raw = _gen_manifest(init_pys[0])
    return ClosureManifest.model_validate(raw)


def _load_dispatcher(closure_dir: Path, manifest: ClosureManifest) -> Dispatcher:
    """Make the closure importable, then build its dispatcher.

    Delegates to `agentix.dispatch._import_and_register`, which handles
    both shapes:
      * explicit `_register.py` (legacy / escape hatch)
      * convention-based auto-discovery from `__init__.py` + `_impl.py`
    """
    py_root = closure_dir
    # Two conventional layouts: closure root contains `agentix_closures/<name>/`
    # directly (development tree), or the Docker image layout
    # `entry/python/agentix_closures/<name>/`. Try both.
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
    """Stable string rendering of a parameter / annotation for diff output."""
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
    # eval_str=True matches Dispatcher.bind so PEP 563 stringified
    # annotations resolve to real types.
    stub_sig = inspect.signature(stub, eval_str=True)  # type: ignore[arg-type]
    impl_sig = inspect.signature(impl, eval_str=True)  # type: ignore[arg-type]
    stub_hints = _resolved_hints(stub)
    impl_hints = _resolved_hints(impl)

    out: list[Mismatch] = []

    stub_params = list(stub_sig.parameters.values())
    impl_params = list(impl_sig.parameters.values())
    # `self` is class-method syntactic noise — drop from both sides. A class
    # stub's unbound method has `self` first; its impl, looked up via
    # `getattr(impl_instance, name)`, is bound and has it stripped. The
    # dispatcher strips `self` from the stub at `bind` time; mirror that here.
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
        return out  # name-level drift makes per-param diffs noise

    for sp, ip in zip(stub_params, impl_params):
        if sp.kind is not ip.kind:
            out.append(Mismatch(
                closure, method, f"param.{sp.name}.kind",
                stub=str(sp.kind), impl=str(ip.kind),
            ))
        # `==` not `is` — large ints (e.g. 10*1024*1024) live outside Python's
        # small-int cache and compare unequal by identity even when textually
        # the same. Equality on the sentinel `inspect.Parameter.empty` works
        # too: its `__eq__` falls back to identity for the singleton.
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

    # Return annotations may differ in async/sync surface — both inspected
    # via __annotations__ already strip the Coroutine wrapper, so compare
    # the resolved hints directly.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
