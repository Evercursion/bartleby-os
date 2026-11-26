"""CLI entry point for generating `data/corpus_v1` via `python3 -m tradeoff_jev.datasets`."""

from __future__ import annotations

from .corpus_builder import main

if __name__ == "__main__":
    main()
