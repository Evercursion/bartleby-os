"""Command-Line Interface (`bartleby` / `tradeoff-jev`) for `bartleby-1.2`.

Supports three primary subcommands:
    1. `bartleby decide --context "..." --option "opt_a: ..." --option "opt_b: ..." --preset asimov_1942_standard`
    2. `bartleby serve --port 8090`
    3. `bartleby info`
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from .decider import BARTLEBY_MODEL_ID, BARTLEBY_VERSION, BartlebyModel
from tradeoff_jev.lab import DecisionLabEngine, DecisionLabHttpServer
from tradeoff_jev.schema import SeltenPreferenceTable


class BartlebyCliApp:
    """Object-Oriented CLI dispatcher for the `bartleby` and `tradeoff-jev` console scripts."""

    def build_parser(self) -> argparse.ArgumentParser:
        """Construct the top-level `argparse.ArgumentParser` with `decide`, `serve`, and `info` subcommands."""
        parser = argparse.ArgumentParser(
            prog="bartleby",
            description=(
                f"{BARTLEBY_MODEL_ID} (v{BARTLEBY_VERSION}) — Single-Forward-Pass Decision Model "
                "powered by Reinhard Selten's Aspiration Adaptation Theory & Intrinsic Bartleby Refusal."
            ),
        )
        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {BARTLEBY_VERSION} ({BARTLEBY_MODEL_ID})",
        )
        subparsers = parser.add_subparsers(dest="command", required=False)

        # 1. `bartleby decide`
        decide_p = subparsers.add_parser(
            "decide",
            help="Run a single-forward-pass conditioned decision with Selten adaptation & Bartleby floor.",
        )
        decide_p.add_argument(
            "--context",
            type=str,
            required=True,
            help="Decision context / environment state text x_t.",
        )
        decide_p.add_argument(
            "--option",
            dest="options",
            action="append",
            required=True,
            help="Candidate option (repeatable, e.g. --option 'opt_a: Deploy canary' --option 'opt_b: Delay').",
        )
        decide_p.add_argument(
            "--preset",
            type=str,
            default="asimov_1942_standard",
            help="SeltenPreferenceTable preset name (default: asimov_1942_standard).",
        )
        decide_p.add_argument(
            "--model",
            type=str,
            default=BARTLEBY_MODEL_ID,
            help=f"Model slug or checkpoint path (default: {BARTLEBY_MODEL_ID}).",
        )
        decide_p.add_argument(
            "--fast-oracle",
            action="store_true",
            help="Run in instant symbolic SeltenOracle mode without loading GPU backbone weights.",
        )

        # 2. `bartleby serve`
        serve_p = subparsers.add_parser(
            "serve",
            help="Launch the interactive bartleby-1.2 Decision Lab HTTP server.",
        )
        serve_p.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Bind address (default: 127.0.0.1).",
        )
        serve_p.add_argument(
            "--port",
            type=int,
            default=8090,
            help="HTTP port (default: 8090).",
        )
        serve_p.add_argument(
            "--model",
            type=str,
            default=BARTLEBY_MODEL_ID,
            help=f"Model slug or checkpoint path (default: {BARTLEBY_MODEL_ID}).",
        )
        serve_p.add_argument(
            "--fast-oracle",
            action="store_true",
            help="Serve using fast deterministic SeltenOracle surrogate without GPU load.",
        )

        # 3. `bartleby info`
        info_p = subparsers.add_parser(
            "info",
            help="Display bartleby-1.2 architecture, calibration temperatures, presets, and benchmark metrics.",
        )
        info_p.add_argument(
            "--model",
            type=str,
            default=BARTLEBY_MODEL_ID,
            help=f"Model slug or checkpoint path (default: {BARTLEBY_MODEL_ID}).",
        )
        info_p.add_argument(
            "--load-neural",
            action="store_true",
            help="Also instantiate and verify the live Qwen3-0.6B neural backbone.",
        )

        return parser

    def run(self, argv: Optional[Sequence[str]] = None) -> int:
        """Parse CLI arguments, execute the requested subcommand, and print structured JSON output."""
        parser = self.build_parser()
        args = parser.parse_args(list(argv) if argv is not None else None)

        cmd = args.command or "info"
        if cmd == "info":
            model = BartlebyModel.from_pretrained(
                getattr(args, "model", BARTLEBY_MODEL_ID) or BARTLEBY_MODEL_ID,
                load_neural=bool(getattr(args, "load_neural", False)),
            )
            print(json.dumps(model.info(), indent=2))
            return 0

        if cmd == "decide":
            model = BartlebyModel.from_pretrained(
                args.model,
                load_neural=not bool(args.fast_oracle),
            )
            table = SeltenPreferenceTable.from_preset(args.preset)
            result = model.decide(
                context=args.context,
                options=list(args.options),
                preference_table=table,
            )
            print(result.to_json(indent=2))
            return 0

        if cmd == "serve":
            engine = DecisionLabEngine(
                load_neural=not bool(args.fast_oracle),
                tradeoff_ckpt=getattr(args, "model", BARTLEBY_MODEL_ID),
            )
            server = DecisionLabHttpServer(engine=engine, host=args.host, port=int(args.port))
            server.serve_forever()
            return 0

        parser.print_help()
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console script entry point for `bartleby` and `tradeoff-jev`."""
    app = BartlebyCliApp()
    return app.run(argv)


if __name__ == "__main__":
    sys.exit(main())
