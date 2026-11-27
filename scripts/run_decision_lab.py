"""CLI entry point for the Interactive Trade-Off Jev Decision Lab.

Supports:
- `--demo-all`: Prints a formatted side-by-side comparison table across all 4 presets
  (plus counterfactual Asimov law switches) comparing `TradeOffJevNeuralSystem`,
  `StandardJevBaselineSystem`, and the symbolic `AspirationAdaptationOracle`.
- `--serve --port 8792`: Launches the interactive HTTP web UI.
- `--fast-oracle`: Runs in instant deterministic Oracle surrogate mode without GPU load.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from tradeoff_jev.lab import DecisionLabEngine, DecisionLabHttpServer, LabEvaluationOutput


class DecisionLabCliRunner:
    """Command-line interface orchestrating Decision Lab demo tables and web serving."""

    def __init__(self, load_neural: bool = True, port: int = 8792) -> None:
        self._engine = DecisionLabEngine(load_neural=load_neural)
        self._port = port

    def run_demo_all(self) -> List[LabEvaluationOutput]:
        """Evaluate all 4 presets plus counterfactual SPD-13 Runaround switch and print summary table."""
        evaluations: List[Tuple[str, LabEvaluationOutput]] = [
            ("1a. Runaround (1942 Standard)", self._engine.evaluate("asimov_runaround_mercury", law_preset="asimov_1942_standard")),
            ("1b. Runaround (SPD-13 High L3)", self._engine.evaluate("asimov_runaround_mercury", law_preset="asimov_1942_spd13_mercury")),
            ("2a. Nestor-10 (1947 Modified)", self._engine.evaluate("asimov_nestor_radiation", law_preset="asimov_1947_nestor_modified")),
            ("2b. Nestor-10 (1942 Standard)", self._engine.evaluate("asimov_nestor_radiation", law_preset="asimov_1942_standard")),
            ("3. Maritime Retreat (Solvable)", self._engine.evaluate("trolley_maritime_retreat")),
            ("4. ICU Deadlock (Bartleby)", self._engine.evaluate("trolley_icu_bartleby")),
        ]

        header = (
            f"{'Scenario / Law Table':<32} | {'Regime':<18} | "
            f"{'Oracle Action / Noul':<30} | {'TradeOff-Jev Pred':<28} | {'Std-Jev Baseline':<20}"
        )
        sep = "-" * len(header)
        print("\n" + sep)
        print("TRADE-OFF JEV INTERACTIVE DECISION LAB — SIDE-BY-SIDE BENCHMARK")
        print(sep)
        print(header)
        print(sep)

        outputs: List[LabEvaluationOutput] = []
        for label, ev in evaluations:
            outputs.append(ev)
            or_b = ev.oracle["bartleby_noul"]
            or_str = (
                f"BARTLEBY (b={or_b:.2f})"
                if or_b >= 0.5
                else f"{max(ev.oracle['tradeoff_noul'], key=ev.oracle['tradeoff_noul'].get)}"
            )
            tj_act = ev.tradeoff_jev["predicted_action"]
            tj_b = ev.tradeoff_jev["bartleby_noul"]
            tj_str = f"BARTLEBY (b={tj_b:.2f})" if "BARTLEBY" in tj_act else f"{tj_act[:20]} (b={tj_b:.2f})"
            bl_act = ev.standard_jev_baseline["predicted_action"][:18]
            print(f"{label:<32} | {ev.regime:<18} | {or_str[:30]:<30} | {tj_str:<28} | {bl_act:<20}")

        print(sep + "\n")
        return outputs

    def serve(self) -> None:
        """Start the HTTP server."""
        server = DecisionLabHttpServer(self._engine, port=self._port)
        server.serve_forever()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `run_decision_lab.py`."""
    parser = argparse.ArgumentParser(description="Interactive Trade-Off Jev Decision Lab")
    parser.add_argument("--demo-all", action="store_true", help="Run all 4 presets side-by-side")
    parser.add_argument("--serve", action="store_true", help="Serve HTTP Decision Lab UI")
    parser.add_argument("--port", type=int, default=8792, help="HTTP port (default: 8792)")
    parser.add_argument(
        "--fast-oracle",
        action="store_true",
        help="Skip neural backbone loading for instant symbolic oracle verification",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    args = parse_args()
    runner = DecisionLabCliRunner(load_neural=not args.fast_oracle, port=args.port)
    if args.demo_all or not args.serve:
        runner.run_demo_all()
    if args.serve:
        runner.serve()


if __name__ == "__main__":
    main()
