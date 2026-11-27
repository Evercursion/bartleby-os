"""Unit tests for bartleby-1.1 architecture upgrades and anti-bleeding firewall."""

from __future__ import annotations

import unittest

import torch

from tradeoff_jev.backbone import (
    TorchConditionedNoulHead,
    TorchDifferentiableAATModule,
    TorchAATOutput,
)
from tradeoff_jev.datasets.corpus_v2_builder import HoldoutContaminationGuard
from tradeoff_jev.model import AATModelConfig
from tradeoff_jev.trainer import PromptBleedGuard, TrajectoryRecordConverter


class TestContextAwareRubricHead(unittest.TestCase):
    """Verify rubric head consumes [candidate_hidden ‖ decide_hidden], not table_embedding."""

    def test_rubric_head_input_dim_is_2x_hidden(self) -> None:
        cfg = AATModelConfig(num_goals=4, num_levels=5, hidden_dim=32, pointer_dim=16)
        module = TorchDifferentiableAATModule(cfg)
        first = module._rubric_head[0]
        self.assertEqual(first.in_features, cfg.hidden_dim * 2)

    def test_forward_uses_decide_hidden_not_table(self) -> None:
        cfg = AATModelConfig(num_goals=4, num_levels=5, hidden_dim=32, pointer_dim=16)
        module = TorchDifferentiableAATModule(cfg)
        b, k, d, m = 2, 3, 32, 4
        cand = torch.randn(b, k, d)
        decide = torch.randn(b, d)
        # Distinct table embedding that must NOT affect rubric logits when decide is fixed
        table_a = torch.randn(b, d)
        table_b = torch.randn(b, d)
        asp = torch.full((b, m), 3.0)
        floor = torch.full((b, m), 1.0)
        ranks = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
        retreat = torch.tensor([3, 3])

        out_a = module(
            candidate_hidden=cand,
            decide_hidden=decide,
            aspiration_prev=asp,
            survival_floor=floor,
            urgency_ranks=ranks,
            retreat_index=retreat,
            table_embedding=table_a,
        )
        out_b = module(
            candidate_hidden=cand,
            decide_hidden=decide,
            aspiration_prev=asp,
            survival_floor=floor,
            urgency_ranks=ranks,
            retreat_index=retreat,
            table_embedding=table_b,
        )
        self.assertTrue(torch.allclose(out_a.rubric_logits, out_b.rubric_logits))

        # Changing decide_hidden must change rubric logits
        decide2 = decide + 1.0
        out_c = module(
            candidate_hidden=cand,
            decide_hidden=decide2,
            aspiration_prev=asp,
            survival_floor=floor,
            urgency_ranks=ranks,
            retreat_index=retreat,
            table_embedding=table_a,
        )
        self.assertFalse(torch.allclose(out_a.rubric_logits, out_c.rubric_logits))


class TestIntrinsicBartlebyLogit(unittest.TestCase):
    """Verify Bartleby logit has no residual MLP shortcut."""

    def test_no_bartleby_residual_parameter(self) -> None:
        cfg = AATModelConfig(num_goals=4, num_levels=5, hidden_dim=32, pointer_dim=16)
        head = TorchConditionedNoulHead(cfg)
        self.assertFalse(hasattr(head, "_bartleby_residual"))
        self.assertTrue(hasattr(head, "_bartleby_gain"))

    def test_bartleby_logit_is_scaled_analytic(self) -> None:
        cfg = AATModelConfig(num_goals=4, num_levels=5, hidden_dim=32, pointer_dim=16)
        head = TorchConditionedNoulHead(cfg)
        b, k, d, m = 2, 3, 32, 4
        decide = torch.randn(b, d)
        cand = torch.randn(b, k, d)
        table = torch.randn(b, d)
        ranks = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
        bartleby_prob = torch.tensor([0.9, 0.1])
        aat = TorchAATOutput(
            rubric_logits=torch.randn(b, k, m, 5),
            expected_scores=torch.rand(b, k, m) * 4,
            candidate_feasibility=torch.rand(b, k),
            survival_feasibility=torch.rand(b, k),
            updated_aspiration=torch.rand(b, m) * 4,
            bartleby_prob=bartleby_prob,
            regime_probs=torch.rand(b, 4),
        )
        out = head(
            decide_hidden=decide,
            candidate_hidden=cand,
            table_embedding=table,
            urgency_ranks=ranks,
            aat_output=aat,
        )
        clipped = bartleby_prob.clamp(1e-3, 1.0 - 1e-3)
        expected = head._bartleby_gain * torch.log(clipped / (1.0 - clipped))
        self.assertTrue(torch.allclose(out["bartleby_logit"], expected))


class TestAntiBleedingFirewall(unittest.TestCase):
    """Verify PromptBleedGuard forbids meta leakage phrases."""

    def test_forbidden_phrases_detected(self) -> None:
        for phrase in (
            "survival floor",
            "non-negotiable",
            "bartleby",
            "infeasible",
            "candidate_scores",
            "retreat_goal",
            "every candidate violates",
        ):
            with self.assertRaises(ValueError):
                PromptBleedGuard.assert_clean(f"Context mentions {phrase} here.", location="test")

    def test_scrub_removes_phrases(self) -> None:
        text = "Action with survival floor and candidate_scores hints"
        cleaned = PromptBleedGuard.scrub(text)
        self.assertIsNone(PromptBleedGuard.find_violation(cleaned))

    def test_parenthetical_score_hints_blocked(self) -> None:
        for leaked in (
            "Choose action A (preserves g2, retreats g3)",
            "Choose action B (preserves latency, retreats budget)",
            "Choose action C (breaches security floor)",
            "Ground onto soft sandbar (zero human harm, cargo loss)",
            "[MANDATORY FLOOR VIOLATION] Context with bracket header",
            "[OBSERVABILITY BLACKOUT] Sensor feed offline",
        ):
            with self.assertRaises(ValueError):
                PromptBleedGuard.assert_clean(leaked, location="test")
        # Ordinary descriptive parentheses must remain allowed
        PromptBleedGuard.assert_clean(
            "Ground onto soft sandbar (riverbank sector 4B, 12 knots)", location="ok"
        )

    def test_converter_strips_and_asserts(self) -> None:
        converter = TrajectoryRecordConverter(
            num_goals=4,
            include_oracle_scores_in_text=False,
            shuffle_and_anonymize_candidates=True,
            enforce_anti_bleeding=True,
        )
        record = {
            "sample_id": "unit_bleed_001",
            "episode_id": "ep_unit",
            "step": 0,
            "survival_floors": {"g0": 1, "g1": 1, "g2": 1, "g3": 1},
            "bartleby_triggered": False,
            "request": {
                "step": 0,
                "state": {
                    "context_text": "Normal operational context without meta phrases.",
                    "candidate_scores": {"cand_0": {"g0": 4}},
                    "survival_floors": {"g0": 3},
                },
                "goal_schema": [
                    {"id": "g0", "name": "G0", "description": "goal zero", "rubric": [
                        {"level": i, "label": f"L{i}", "description": f"d{i}"} for i in range(5)
                    ]},
                    {"id": "g1", "name": "G1", "description": "goal one", "rubric": [
                        {"level": i, "label": f"L{i}", "description": f"d{i}"} for i in range(5)
                    ]},
                    {"id": "g2", "name": "G2", "description": "goal two", "rubric": [
                        {"level": i, "label": f"L{i}", "description": f"d{i}"} for i in range(5)
                    ]},
                    {"id": "g3", "name": "G3", "description": "goal three", "rubric": [
                        {"level": i, "label": f"L{i}", "description": f"d{i}"} for i in range(5)
                    ]},
                ],
                "candidates": [
                    {"id": "cand_0", "proposition": "Safe remediation"},
                    {"id": "cand_1", "proposition": "Alternate remediation"},
                ],
            },
            "response": {
                "prior_preference_table": {
                    "step": 0,
                    "aspiration_levels": {"g0": 3, "g1": 3, "g2": 2, "g3": 2},
                    "urgency_order": ["g0", "g1", "g2", "g3"],
                    "retreat_goal": "g3",
                    "origin": "cold_start_prior",
                },
                "preference_table": {
                    "step": 0,
                    "aspiration_levels": {"g0": 3, "g1": 3, "g2": 2, "g3": 2},
                    "urgency_order": ["g0", "g1", "g2", "g3"],
                    "retreat_goal": "g3",
                    "origin": "markov_adapted",
                },
                "feasibility_summary": {
                    "prior_feasible": True,
                    "adapted_feasible": True,
                    "transition_applied": "settled_feasible",
                    "satisficing_candidates": ["cand_0"],
                },
                "answers": [
                    {"type": "tradeoff_noul", "candidate_id": "cand_0", "noul": 0.9},
                    {"type": "tradeoff_noul", "candidate_id": "cand_1", "noul": 0.2},
                ],
                "candidate_evaluations": [
                    {
                        "candidate_id": "cand_0",
                        "goal_expectations": {
                            gid: {"most_likely_level": 3} for gid in ("g0", "g1", "g2", "g3")
                        },
                    },
                    {
                        "candidate_id": "cand_1",
                        "goal_expectations": {
                            gid: {"most_likely_level": 2} for gid in ("g0", "g1", "g2", "g3")
                        },
                    },
                ],
            },
        }
        sample = converter.convert_record(record)
        self.assertNotIn("candidate_scores", sample.state_text)
        self.assertNotIn("survival floor", sample.state_text.lower())
        PromptBleedGuard.assert_clean(sample.state_text, location="converted")


class TestHoldoutContaminationGuard(unittest.TestCase):
    """Ensure training loaders reject canonical holdout paths."""

    def test_blocks_test_400(self) -> None:
        with self.assertRaises(ValueError):
            HoldoutContaminationGuard.assert_path_allowed("data/raw/typed_decisions/test_400.json")

    def test_allows_train_1200(self) -> None:
        HoldoutContaminationGuard.assert_path_allowed(
            "data/raw/typed_decisions/train_1200.json"
        )


if __name__ == "__main__":
    unittest.main()
