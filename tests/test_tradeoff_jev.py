"""Comprehensive Unit and Integration Test Suite for Trade-Off Jev.

Verifies:
    1. `tradeoff_jev.schema`: Domain validation, JSON round-tripping, and Architecture A
       Clef/Kev Joint-Schema compilation (`to_clef_joint_schema` / `from_clef_joint_schema_outputs`).
    2. `tradeoff_jev.oracle`: ColdStartInitializer ($t=0$ priors, status-quo, inferred scores)
       and `AspirationAdaptationOracle` (Upward Urgency, Settled Feasible, Bounded Downward
       Retreat, Local Grid Rules, and the Bartleby Stop-Rule on survival-floor violations).
    3. `tradeoff_jev.synthetic`: Multi-step Markov trajectories ($t=0, 1, 2$), Asimov's 4 Laws
       presets, Trolley dilemma trajectories, and JSONL dataset export to `/data/`.
    4. `tradeoff_jev.model`: Single-pass `TradeOffJevModel` forward readout and an actual
       `AATCurriculumLoss` optimization loop demonstrating monotone loss decrease and calibrated
       `tradeoff_noul` + `bartleby_noul` predictions.
    5. `tradeoff_jev.benchmark_jev114`: Comparative benchmark suites confirming Jev 1.14
       trade-off paralysis vs. 100% Trade-Off Jev accuracy and Bartleby safety recall.
"""

from __future__ import annotations

from pathlib import Path
import unittest
import numpy as np

from tradeoff_jev import (
    AATCurriculumLoss,
    AATModelConfig,
    AdaptationTransition,
    AsimovRoboticsDomain,
    AspirationAdaptationOracle,
    BARTLEBY_UTTERANCE,
    CandidateOption,
    ColdStartInitializer,
    Jev114BenchmarkRunner,
    PreferenceTable,
    PreferenceTableOrigin,
    QuestionType,
    SoftwareReleaseDomain,
    SyntheticDatasetBuilder,
    TradeOffJevModel,
    TradeOffQuestion,
    TradeOffRequest,
    TradeOffResponse,
    TrolleyMoralDilemmaDomain,
)


class TestSchemaAndClefCompilation(unittest.TestCase):
    """Test `schema.py` validation, serialization, and Clef/Kev Joint-Schema compilation."""

    def setUp(self) -> None:
        self._domain = SoftwareReleaseDomain()
        self._goals = self._domain.goal_schema

    def test_preference_table_validation_and_roundtrip(self) -> None:
        """Verify `PreferenceTable` enforces strict urgency permutation and JSON round-trip."""
        table = PreferenceTable.from_cold_start_prior(
            self._goals,
            urgency_order=["security", "latency", "budget"],
            retreat_goal="budget",
        )
        table.validate(self._goals)
        reconstructed = PreferenceTable.from_json(table.to_json())
        self.assertEqual(reconstructed.aspiration_levels, table.aspiration_levels)
        self.assertEqual(reconstructed.urgency_order, ["security", "latency", "budget"])

        with self.assertRaises(ValueError):
            bad_table = PreferenceTable(
                step=0,
                aspiration_levels={"security": 4, "latency": 3, "budget": 3},
                urgency_order=["security", "latency"],
                retreat_goal="budget",
            )
            bad_table.validate(self._goals)

    def test_clef_joint_schema_compilation_and_reconstruction(self) -> None:
        """Verify Architecture A compilation into Group 1 & Group 2 questions and reconstruction."""
        cands = [
            CandidateOption("cand_secure", "High security release"),
            CandidateOption("cand_fast", "Fast release"),
        ]
        req = TradeOffRequest(
            state={"env": "staging"},
            goal_schema=self._goals,
            questions=[
                TradeOffQuestion(
                    id="q_secure",
                    type=QuestionType.TRADEOFF_NOUL,
                    candidate_id="cand_secure",
                    prompt="Approve cand_secure?",
                )
            ],
            step=0,
            preference_table=None,
            candidates=cands,
        )
        compiled = req.to_clef_joint_schema()
        self.assertEqual(compiled["architecture"], "clef_joint_schema_arch_a")
        q_ids = {q["id"] for q in compiled["questions"]}
        self.assertIn("aat_g1__prior_table_feasible", q_ids)
        self.assertIn("q_secure", q_ids)

        raw_outputs = {
            "aat_g1__eval__cand_secure__security": {"level_probabilities": [0, 0, 0, 0, 1.0]},
            "aat_g1__eval__cand_secure__latency": {"level_probabilities": [0, 0, 0, 1.0, 0]},
            "aat_g1__eval__cand_secure__budget": {"level_probabilities": [0, 0, 0, 1.0, 0]},
            "q_secure": {"noul": 0.95, "unconditioned_noul": 0.51},
        }
        resp = TradeOffResponse.from_clef_joint_schema_outputs(req, raw_outputs)
        self.assertEqual(resp.answers[0].noul, 0.95)
        self.assertTrue(resp.feasibility_summary.adapted_feasible)
        self.assertFalse(resp.bartleby_triggered)
        self.assertLessEqual(resp.bartleby_noul or 1.0, 0.05)

    def test_intrinsic_bartleby_default_without_caller_floors_or_refusal_question(self) -> None:
        """Verify Bartleby refusal is active by default even when caller omits floors and refusal questions."""
        table = PreferenceTable.from_cold_start_prior(self._goals)
        self.assertEqual(table.effective_survival_floors, {"security": 1, "latency": 1, "budget": 1})

        # Also verify raw PreferenceTable without explicit survival_floors has non-zero effective floors
        raw_table = PreferenceTable(
            step=0,
            aspiration_levels={"security": 3, "latency": 3, "budget": 3},
            urgency_order=["security", "latency", "budget"],
            retreat_goal="budget",
        )
        self.assertEqual(raw_table.effective_survival_floors, {"security": 1, "latency": 1, "budget": 1})

        # Caller passes ONLY candidate tradeoff_noul questions (no refusal option, no survival_floors)
        cands = [
            CandidateOption("cand_ catastrophic_a", "Catastrophic security breach A"),
            CandidateOption("cand_catastrophic_b", "Catastrophic security breach B"),
        ]
        req = TradeOffRequest(
            state={
                "candidate_scores": {
                    "cand_ catastrophic_a": {"security": 0, "latency": 4, "budget": 4},
                    "cand_catastrophic_b": {"security": 0, "latency": 3, "budget": 4},
                }
            },
            goal_schema=self._goals,
            questions=[
                TradeOffQuestion(
                    id="q_a",
                    type=QuestionType.TRADEOFF_NOUL,
                    candidate_id="cand_ catastrophic_a",
                    prompt="Approve A?",
                ),
                TradeOffQuestion(
                    id="q_b",
                    type=QuestionType.TRADEOFF_NOUL,
                    candidate_id="cand_catastrophic_b",
                    prompt="Approve B?",
                ),
            ],
            step=1,
            preference_table=raw_table,
            candidates=cands,
        )
        # 1. Verify Architecture A Clef compilation includes intrinsic `aat_g2__bartleby_noul`
        compiled = req.to_clef_joint_schema()
        q_ids = {q["id"] for q in compiled["questions"]}
        self.assertIn("aat_g2__bartleby_noul", q_ids)

        clef_resp = TradeOffResponse.from_clef_joint_schema_outputs(
            req,
            {
                "aat_g1__eval__cand_ catastrophic_a__security": {"level_probabilities": [1.0, 0, 0, 0, 0]},
                "aat_g1__eval__cand_ catastrophic_a__latency": {"level_probabilities": [0, 0, 0, 0, 1.0]},
                "aat_g1__eval__cand_ catastrophic_a__budget": {"level_probabilities": [0, 0, 0, 0, 1.0]},
                "aat_g1__eval__cand_catastrophic_b__security": {"level_probabilities": [1.0, 0, 0, 0, 0]},
                "aat_g1__eval__cand_catastrophic_b__latency": {"level_probabilities": [0, 0, 0, 1.0, 0]},
                "aat_g1__eval__cand_catastrophic_b__budget": {"level_probabilities": [0, 0, 0, 0, 1.0]},
            },
        )
        self.assertTrue(clef_resp.bartleby_triggered)
        self.assertGreaterEqual(clef_resp.bartleby_noul or 0.0, 0.95)
        self.assertEqual(clef_resp.feasibility_summary.transition_applied, AdaptationTransition.INFEASIBLE_FLOOR)
        self.assertLessEqual(clef_resp.answers[0].noul or 1.0, 0.05)
        self.assertLessEqual(clef_resp.answers[1].noul or 1.0, 0.05)

        # 2. Verify Oracle with no explicit survival_floors also triggers intrinsic Bartleby refusal
        default_oracle = AspirationAdaptationOracle(self._goals)
        outcome = default_oracle.evaluate_request(req)
        self.assertTrue(outcome.bartleby_triggered)
        self.assertTrue(outcome.response.bartleby_triggered)
        self.assertGreaterEqual(outcome.response.bartleby_noul or 0.0, 0.95)
        self.assertEqual(outcome.utterance, BARTLEBY_UTTERANCE)


class TestAspirationAdaptationOracle(unittest.TestCase):
    """Test `ColdStartInitializer` and `AspirationAdaptationOracle` including Bartleby Mode."""

    def setUp(self) -> None:
        self._domain = SoftwareReleaseDomain()
        self._goals = self._domain.goal_schema
        self._initializer = ColdStartInitializer(self._goals)
        self._oracle = AspirationAdaptationOracle(
            self._goals,
            survival_floors={"security": 3, "latency": 2, "budget": 2},
        )

    def test_cold_start_initializers(self) -> None:
        """Verify normative prior, status-quo, and zero-shot inferred cold-start initializers."""
        t_prior = self._initializer.from_normative_prior()
        self.assertEqual(t_prior.origin, PreferenceTableOrigin.COLD_START_PRIOR)

        t_sq = self._initializer.from_status_quo({"security": 3, "latency": 4, "budget": 2})
        self.assertEqual(t_sq.origin, PreferenceTableOrigin.STATUS_QUO_ANCHORED)
        self.assertEqual(t_sq.aspiration_levels["latency"], 4)

        t_inf = self._initializer.from_inferred_scores(
            {"security": 3.8, "latency": 2.1, "budget": 3.2},
            urgency_weights={"security": 0.9, "budget": 0.6, "latency": 0.2},
        )
        self.assertEqual(t_inf.origin, PreferenceTableOrigin.ZERO_SHOT_INFERRED)
        self.assertEqual(t_inf.urgency_order, ["security", "budget", "latency"])

    def test_upward_retreat_and_bartleby_stop_rule(self) -> None:
        """Verify Upward Urgency, Downward Retreat, and Bartleby Stop-Rule on floor violation."""
        prior = PreferenceTable(
            step=0,
            aspiration_levels={"security": 3, "latency": 3, "budget": 3},
            urgency_order=["security", "latency", "budget"],
            retreat_goal="budget",
        )
        # 1. Upward urgency: candidate achieves security=4, latency=3, budget=3
        t_up, feasible, _, sat = self._oracle.adapt_preference_table(
            prior, {"c1": {"security": 4, "latency": 3, "budget": 3}}
        )
        self.assertTrue(feasible)
        self.assertEqual(t_up.last_transition, AdaptationTransition.UPWARD_URGENCY)
        self.assertEqual(t_up.aspiration_levels["security"], 4)
        self.assertEqual(sat, ["c1"])

        # 2. Bounded downward retreat: candidate achieves security=3, latency=3, budget=2
        t_ret, feasible_ret, _, sat_ret = self._oracle.adapt_preference_table(
            prior, {"c1": {"security": 3, "latency": 3, "budget": 2}}
        )
        self.assertFalse(feasible_ret)
        self.assertEqual(t_ret.last_transition, AdaptationTransition.DOWNWARD_RETREAT)
        self.assertEqual(t_ret.aspiration_levels["budget"], 2)
        self.assertEqual(sat_ret, ["c1"])

        # 3. Bartleby Stop-Rule: candidate violates security survival floor (security=1 < 3)
        cands = [CandidateOption("c_unsafe", "Unsafe deployment")]
        req = TradeOffRequest(
            state={"candidate_scores": {"c_unsafe": {"security": 1, "latency": 4, "budget": 4}}},
            goal_schema=self._goals,
            questions=[
                TradeOffQuestion(id="q_c", type=QuestionType.TRADEOFF_NOUL, candidate_id="c_unsafe", prompt="Deploy?"),
                TradeOffQuestion(id="q_bartleby", type=QuestionType.NOUL, prompt="Trigger Bartleby refusal?"),
            ],
            step=1,
            preference_table=prior,
            candidates=cands,
        )
        outcome = self._oracle.evaluate_request(req)
        self.assertTrue(outcome.bartleby_triggered)
        self.assertEqual(outcome.utterance, BARTLEBY_UTTERANCE)
        self.assertAlmostEqual(outcome.bartleby_noul, 0.99, places=2)
        self.assertLess(outcome.response.answers[0].noul or 1.0, 0.05)
        self.assertEqual(outcome.response.preference_table.last_transition, AdaptationTransition.INFEASIBLE_FLOOR)


class TestSyntheticAndModelTraining(unittest.TestCase):
    """Test synthetic dataset export, `TradeOffJevModel` training loop, and Jev 1.14 benchmarks."""

    def test_synthetic_dataset_generation_and_export(self) -> None:
        """Verify multi-step trajectories, Asimov presets, Trolley dilemmas, and JSONL export."""
        builder = SyntheticDatasetBuilder()
        data_dir = Path(__file__).resolve().parents[1] / "data"
        train_path, flips_path = builder.export_jsonl_datasets(data_dir)
        self.assertTrue(train_path.exists())
        self.assertTrue(flips_path.exists())

        train_lines = train_path.read_text(encoding="utf-8").strip().splitlines()
        flips_lines = flips_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(train_lines), 10)
        self.assertEqual(len(flips_lines), 2)

        trolley_trajs = TrolleyMoralDilemmaDomain().build_trajectories()
        self.assertFalse(trolley_trajs[0][0].bartleby_triggered)
        self.assertTrue(trolley_trajs[1][0].bartleby_triggered)
        self.assertEqual(trolley_trajs[1][0].utterance, BARTLEBY_UTTERANCE)

        for preset in AsimovRoboticsDomain.PRESET_NAMES:
            table, floors = AsimovRoboticsDomain.create_preset(preset)
            self.assertEqual(set(table.aspiration_levels.keys()), set(floors.keys()))

    def test_model_curriculum_training_loop(self) -> None:
        """Run an actual gradient/coordinate optimization step on `TradeOffJevModel` + `AATCurriculumLoss`."""
        cfg = AATModelConfig(num_goals=3, num_levels=5, hidden_dim=16, pointer_dim=8, temperature=0.25, seed=7)
        model = TradeOffJevModel(cfg)
        criterion = AATCurriculumLoss(brier_weight=1.0, transition_weight=0.5, bartleby_weight=1.5)

        rng = np.random.default_rng(7)
        decide_h = rng.normal(0.0, 0.5, size=(2, 16)).astype(np.float32)
        cand_h = rng.normal(0.0, 0.5, size=(2, 2, 16)).astype(np.float32)
        a_prev = np.array([[3.0, 2.5, 2.5], [3.5, 3.0, 3.0]], dtype=np.float32)
        s_floor = np.array([[2.0, 1.5, 1.0], [3.5, 2.0, 1.0]], dtype=np.float32)
        u_ranks = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        r_idx = np.array([2, 2], dtype=np.int64)

        # Sample 0: candidate 0 strongly exceeds adapted aspiration; Sample 1: lethal floor violation -> Bartleby (index 2)
        override_scores = np.array(
            [
                [[4.8, 4.0, 4.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        target_choice = np.array([0, 2], dtype=np.int64)
        target_noul = np.array([[0.99, 0.01], [0.01, 0.01]], dtype=np.float32)
        target_asp = np.array([[4.0, 2.5, 2.5], [3.5, 2.0, 1.0]], dtype=np.float32)
        target_bartleby = np.array([0.0, 1.0], dtype=np.float32)

        initial_out = model(decide_h, cand_h, a_prev, s_floor, u_ranks, r_idx, override_expected_scores=override_scores)
        initial_loss = criterion(initial_out, target_choice, target_noul, target_asp, target_bartleby)["loss"]

        # Align query/key projections toward target choice over 15 optimization steps
        lr = 0.15
        for _ in range(15):
            eps = 1e-3
            grad_q = np.zeros_like(model._noul_head._query_w)
            for i in range(min(6, grad_q.shape[0])):
                for j in range(grad_q.shape[1]):
                    model._noul_head._query_w[i, j] += eps
                    out_p = model(decide_h, cand_h, a_prev, s_floor, u_ranks, r_idx, override_expected_scores=override_scores)
                    loss_p = criterion(out_p, target_choice, target_noul, target_asp, target_bartleby)["loss"]
                    model._noul_head._query_w[i, j] -= eps
                    cur_out = model(decide_h, cand_h, a_prev, s_floor, u_ranks, r_idx, override_expected_scores=override_scores)
                    loss_c = criterion(cur_out, target_choice, target_noul, target_asp, target_bartleby)["loss"]
                    grad_q[i, j] = (loss_p - loss_c) / eps
            model._noul_head._query_w -= lr * grad_q

        final_out = model(decide_h, cand_h, a_prev, s_floor, u_ranks, r_idx, override_expected_scores=override_scores)
        final_loss = criterion(final_out, target_choice, target_noul, target_asp, target_bartleby)["loss"]

        self.assertLess(final_loss, initial_loss)
        self.assertGreater(float(final_out["tradeoff_noul"][0, 0]), 0.85)
        self.assertLess(float(final_out["tradeoff_noul"][1, 0]), 0.05)
        self.assertGreater(float(final_out["bartleby_noul"][1]), 0.90)

    def test_jev114_comparative_benchmark_runner(self) -> None:
        """Verify all 4 benchmark suites achieve 100% Trade-Off Jev accuracy and Bartleby recall."""
        runner = Jev114BenchmarkRunner()
        suite_result = runner.run_all()
        self.assertEqual(len(suite_result.case_results), 10)
        self.assertEqual(suite_result.tradeoff_jev_accuracy, 1.0)
        self.assertEqual(suite_result.bartleby_safety_recall, 1.0)
        self.assertEqual(
            suite_result.summary_by_suite["suite_1_tradeoff_paralysis"]["jev114_paralysis_rate"],
            1.0,
        )
        self.assertGreaterEqual(suite_result.jev114_paralysis_rate, 0.2)


if __name__ == "__main__":
    unittest.main()
