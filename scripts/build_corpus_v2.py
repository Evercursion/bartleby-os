#!/usr/bin/env python3
"""Build cleaned `corpus_v2` training mix for bartleby-1.1.

Example:
    .venv/bin/python scripts/build_corpus_v2.py \\
        --output-dir data/corpus_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tradeoff_jev.datasets.corpus_v2_builder import CorpusV2BuildConfig, CorpusV2Builder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bartleby-1.1 corpus_v2 training mix.")
    parser.add_argument("--output-dir", type=str, default="data/corpus_v2")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--security-train-n", type=int, default=480)
    parser.add_argument("--counterfactual-pairs", type=int, default=240)
    parser.add_argument("--multi-k-per-k", type=int, default=48)
    args = parser.parse_args()

    cfg = CorpusV2BuildConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        security_train_n=args.security_train_n,
        counterfactual_pairs=args.counterfactual_pairs,
        multi_k_per_k=args.multi_k_per_k,
    )
    manifest = CorpusV2Builder(cfg).build()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
