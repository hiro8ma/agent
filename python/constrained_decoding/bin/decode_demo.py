from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `uv run python bin/decode_demo.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo import main  # noqa: E402

if __name__ == "__main__":
    main()
