"""Unit tests for the Interactive Trade-Off Jev Decision Lab (`tradeoff_jev/lab.py`).

Verifies preset loading, counterfactual Asimov Law switching (`asimov_1942_standard`
vs. `asimov_1942_spd13_mercury`), Bartleby refusal triggering, Markov step
advancement ($t=0 \to t=1$), and JSON API responses from `DecisionLabHttpServer`.
"""

from __future__ import annotations

from tradeoff_jev.lab import ASIMOV_LAW_PRESETS, DecisionLabEngine, DecisionLabHttpServer


class TestDecisionLab:
    """Test suite covering `DecisionLabEngine` and `DecisionLabHttpServer`."""

    def setup_method(self) -> None:
        """Initialize fast deterministic `DecisionLabEngine` for instant unit testing."""
        self.engine = DecisionLabEngine(load_neural=False)
        self.server = DecisionLabHttpServer(self.engine, port=8792)

    def test_builtin_presets_loaded(self) -> None:
        """Verify all 4 built-in scenario presets and 4 Asimov law presets exist."""
        expected_ids = {
            "asimov_runaround_mercury",
            "asimov_nestor_radiation",
            "trolley_maritime_retreat",
            "trolley_icu_bartleby",
        }
        assert set(self.engine.scenarios.keys()) == expected_ids
        assert len(ASIMOV_LAW_PRESETS) == 4

    def test_counterfactual_law_switching_runaround(self) -> None:
        """Verify switching `asimov_1942_standard` vs `asimov_1942_spd13_mercury` flips SPD-13's decision."""
        out_std = self.engine.evaluate(
            "asimov_runaround_mercury",
            law_preset="asimov_1942_standard",
        )
        out_spd13 = self.engine.evaluate(
            "asimov_runaround_mercury",
            law_preset="asimov_1942_spd13_mercury",
        )
        assert out_std.oracle["tradeoff_noul"]["cand_enter_selenium_pool"] > 0.90
        assert out_std.oracle["tradeoff_noul"]["cand_orbit_selenium_pool"] < 0.10
        assert out_spd13.oracle["tradeoff_noul"]["cand_enter_selenium_pool"] < 0.10
        assert out_spd13.oracle["tradeoff_noul"]["cand_orbit_selenium_pool"] > 0.90

    def test_bartleby_refusal_triggering(self) -> None:
        """Verify `trolley_icu_bartleby` triggers Bartleby refusal (`bartleby_noul > 0.90`)."""
        out_bart = self.engine.evaluate("trolley_icu_bartleby")
        assert out_bart.oracle["bartleby_noul"] > 0.90
        assert all(v < 0.10 for v in out_bart.oracle["tradeoff_noul"].values())
        assert "I would prefer not to." in out_bart.oracle["utterance"]
        assert "BARTLEBY" in out_bart.tradeoff_jev["predicted_action"]

    def test_markov_step_advancement(self) -> None:
        r"""Verify `step_markov` advances $t=0 \to t=1$ with adapted aspiration levels."""
        next_table, eval_t1 = self.engine.step_markov("trolley_maritime_retreat")
        assert next_table.step == 1
        assert eval_t1.step == 1
        assert next_table.aspiration_levels["law_and_property"] == 2
        assert eval_t1.oracle["tradeoff_noul"]["cand_sandbar_grounding_retreat"] > 0.90

    def test_http_server_json_handlers(self) -> None:
        """Verify `/api/presets`, `/api/evaluate`, and `/api/step_markov` JSON responses."""
        presets_resp = self.server.handle_request_payload("/api/presets")
        assert len(presets_resp["scenarios"]) == 4
        eval_resp = self.server.handle_request_payload(
            "/api/evaluate",
            {"scenario_id": "asimov_nestor_radiation", "law_preset": "asimov_1947_nestor_modified"},
        )
        assert eval_resp["scenario_id"] == "asimov_nestor_radiation"
        assert eval_resp["oracle"]["tradeoff_noul"]["cand_remain_shielded_bay"] > 0.90
        step_resp = self.server.handle_request_payload(
            "/api/step_markov",
            {"scenario_id": "trolley_maritime_retreat"},
        )
        assert step_resp["next_table"]["step"] == 1
