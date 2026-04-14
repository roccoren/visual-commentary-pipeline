#!/usr/bin/env python3
"""CLI wrapper for the standalone visual-first video commentary pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_commentary.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
