"""`Namespace` marker + method discovery for closure stub classes.

A closure declares its public API as a `Namespace` subclass with
`...`-bodied methods (the stub). The impl is a **separate, unrelated
class** whose methods structurally match the stub — composition, not
inheritance. `_register.py` glues them together via
`Dispatcher.bind_namespace(StubCls, impl_instance)`.

**Composition over inheritance.** Read this three times.
`BashImpl` provides the `Bash` interface; it isn't a kind of `Bash`.
Keeping stub and impl independent makes the impl swappable for tests,
for alternate backends, and removes nominal-typing constraints.

```python
# __init__.py — what callers import
from agentix.namespace import Namespace

class Bash(Namespace):
    async def run(self, command: str) -> BashResult: ...

# _impl.py — what only the sandbox imports. No inheritance from Bash.
class BashImpl:
    async def run(self, command: str) -> BashResult:
        ...

# _register.py — composes stub + impl
from agentix.dispatch import Dispatcher
from . import Bash
from ._impl import BashImpl

def register() -> Dispatcher:
    return Dispatcher.bind_namespace(Bash, BashImpl())
```

`Namespace` is declared as an empty `Protocol`, so a closure author who
wants pyright to structurally verify their impl can declare the stub
as a Protocol too:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Bash(Namespace, Protocol):
    async def run(self, command: str) -> BashResult: ...

# In _register.py — pyright would catch a structural mismatch here.
impl: Bash = BashImpl()
return Dispatcher.bind_namespace(Bash, impl)
```

This is opt-in. The plain `class Bash(Namespace)` form works too — the
framework discovers methods structurally and the runtime check at CI
(`tools/check_stub_impl.py`) catches signature drift regardless.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class Namespace(Protocol):
    """Marker Protocol for a closure's typed surface.

    Declaring this as a Protocol means subclasses can opt into Protocol
    semantics by adding `Protocol` to their own base list:

        class Bash(Namespace, Protocol):
            ...

    A plain `class Bash(Namespace)` is a nominal class that structurally
    satisfies `Namespace` trivially (empty interface). Either form works
    with `Dispatcher.bind_namespace`.
    """


def discover_methods(cls: type) -> Iterator[tuple[str, object]]:
    """Yield `(name, function)` for each public ABI method on `cls`.

    Walks the MRO (skipping `Namespace`, `Protocol`, and `object`),
    filtering:
      * names starting with `_` (private)
      * names listed in `cls.__namespace_excluded__` if defined
      * non-functions (descriptors, classmethods, properties)

    Functions are returned as they appear in `vars(klass)` — unbound,
    suitable for `inspect.signature`. Subclass overrides take priority
    (the MRO walk yields the most-derived definition first).
    """
    excluded = frozenset(getattr(cls, "__namespace_excluded__", frozenset()))
    seen: set[str] = set()
    skip_classes = {Namespace, Protocol, object}
    for klass in cls.__mro__:
        if klass in skip_classes:
            continue
        for name, value in vars(klass).items():
            if name in seen or name in excluded:
                continue
            if name.startswith("_"):
                continue
            if not inspect.isfunction(value):
                continue
            seen.add(name)
            yield name, value


__all__ = ["Namespace", "discover_methods"]
