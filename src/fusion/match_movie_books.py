"""Thin wrapper around the book/movie matcher placed under src/fusion.

This keeps the implementation in one place while exposing the requested
entry point under ``src/fusion``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.book.match_movie_books import main


if __name__ == "__main__":
    main()
