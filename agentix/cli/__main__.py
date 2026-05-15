"""`python -m agentix.cli` entry point — defers to `agentix.cli.main`."""

from __future__ import annotations

import sys

from agentix.cli import main

if __name__ == "__main__":
    sys.exit(main())
