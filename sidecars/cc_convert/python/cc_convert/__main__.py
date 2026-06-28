"""Allow ``python -m cc_convert ...``."""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
