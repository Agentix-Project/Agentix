"""`python -m agentix.cli.build` — invoke the build subcommand standalone.

Equivalent to `agentix build` but without going through the parent
`agentix` group. Mostly useful when the console script isn't on PATH
(e.g. inside a fresh checkout that hasn't been installed yet).
"""

from __future__ import annotations

import sys

from agentix.cli.build import main

if __name__ == "__main__":
    sys.exit(main())
