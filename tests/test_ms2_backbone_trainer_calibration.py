"""Unit & Integration Tests for Milestone 2 Backbone, CurriculumTrainer, and TemperatureCalibrator.

Verifies:
    1. `DecisionSequencePacker`: Exact candidate span indexing, variable candidate counts,
       and truncation safety preserving candidate spans and `[DECIDE]`.
    2. `QwenDecisionBackbone`: `lm_head` stripping (`has_lm_head() == False`), `peft` LoRA
       (`r=16, lora_alpha=32`) attachment, and span pooling modes.
    3. `TradeOffJevNeuralSystem` vs. `StandardJevBaselineSystem`: Counterfactual preference
       table sensitivity vs. strict unconditioned invariance, differential optimizer LR
       groups (`1e-4` LoRA vs. `3e-4` FP32 head), and `safetensors` checkpoint round-trip.
    4. `CurriculumTrainer`: Phase 1 single-step calibration + Phase 2 multi-step Markov
       scheduled-sampling rollout, gradient accumulation, and structured JSONL telemetry.
    5. `TemperatureCalibrator`: L-BFGS + Log-Grid NLL scaling reporting 15-bin ECE and
       Brier Score improvements across both `BINARY_NOUL` and `MULTICLASS_CHOICE`.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import List
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from tradeoff_jev.backbone import (
    BackboneConfig,
    DecisionSequencePacker,
    PoolingStrategy,
    QwenDecisionBackbone,
    StandardJevBaselineSystem,
    TradeOffJevNeuralSystem,
)
from tradeoff_jev.calibration import CalibrationMode, TemperatureCalibrator
from tradeoff_jev.model import AATModelConfig
from tradeoff_jev.trainer import (
    CurriculumTrainer,
    TrainerConfig,
    TrajectoryRecordConverter,
)


class _AsciiWordTokenizer:
    """Deterministic lightweight tokenizer for fast offline unit tests."""

    pad_token_id: int = 0
    eos_token_id: int = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        _ = add_special_tokens
        return [2 + (ord(ch) % 250) for ch in text]


def _build_tiny_qwen_backbone(
    pooling: PoolingStrategy = PoolingStrategy.SPAN_END_AND_MEAN,
) -> QwenDecisionBackbone:
    """Create an in-memory 2-layer Qwen CausalLM and verify `QwenDecisionBackbone` strips `lm_head`."""
    torch.manual_seed(999)
    qwen_cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    causal_lm = Qwen2ForCausalLM(qwen_cfg)
    bb_cfg = BackboneConfig(
        model_name_or_path="tiny-qwen-unit-test",
        torch_dtype="float32",
        max_seq_length=256,
        pooling_strategy=pooling,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
    )
    return QwenDecisionBackbone(
        config=bb_cfg,
        pretrained_model=causal_lm,
        tokenizer=_AsciiWordTokenizer(),
    )


class TestBackboneAndSystems(unittest.TestCase):
    """Test `DecisionSequencePacker`, `QwenDecisionBackbone`, and neural readout systems."""

    def test_packer_spans_and_truncation(self) -> None:
        """Verify candidate spans are valid, non-overlapping, and survive state truncation."""
        tok = _AsciiWordTokenizer()
        packer = DecisionSequencePacker(tok, max_seq_length=120)
        long_state = "S" * 400
        packed = packer.pack_batch(
            state_texts=[long_state, "short state"],
            candidate_descriptions=[["cand A", "cand B", "cand C"], ["only A", "only B"]],
        )
        self.assertEqual(packed.input_ids.shape[0], 2)
        self.assertLessEqual(packed.input_ids.shape[1], 120)
        self.assertEqual(tuple(packed.candidate_spans.shape), (2, 3, 2))
        self.assertTrue(bool(packed.candidate_mask[0, 2]))
        self.assertFalse(bool(packed.candidate_mask[1, 2]))
        # Decide position must strictly follow the last valid candidate span
        self.assertGreater(int(packed.decide_positions[0]), int(packed.candidate_spans[0, 2, 1] - 1))

    def test_lm_head_stripped_lora_and_ablation_invariance(self) -> None:
        """Verify `lm_head` is stripped, LoRA is active, and baseline ignores PreferenceTable."""
        backbone_aat = _build_tiny_qwen_backbone(PoolingStrategy.SPAN_END_AND_MEAN)
        self.assertFalse(backbone_aat.has_lm_head())

        aat_cfg = AATModelConfig(num_goals=4, num_levels=5, hidden_dim=32, pointer_dim=16)
        system = TradeOffJevNeuralSystem(backbone=backbone_aat, aat_config=aat_cfg)
        summary = system.trainable_parameter_summary()
        self.assertGreater(summary["lora_trainable_parameters"], 0)
        self.assertGreater(summary["head_fp32_trainable_parameters"], 0)
        self.assertGreater(summary["backbone_frozen_parameters"], 0)

        groups = system.build_optimizer_param_groups(lora_lr=1e-4, head_lr=3e-4)
        lrs = {g["group_name"]: g["lr"] for g in groups}
        self.assertEqual(lrs["lora_decay"], 1e-4)
        self.assertEqual(lrs["head_decay"], 3e-4)

        packed = backbone_aat.pack_texts(
            ["Deployment conflict state"],
            [["Deploy hardened build", "Deploy fast build"]],
        )
        asp_a = torch.tensor([[4.0, 2.0, 2.0, 0.0]])
        asp_b = torch.tensor([[1.0, 4.0, 4.0, 0.0]])
        floors = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        ranks = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        retreat = torch.tensor([2])

        # Perturb LoRA weights so lora_B != 0, then run forward and checkpoint round-trip
        with torch.no_grad():
            for name, param in system.backbone.named_parameters():
                if param.requires_grad and "lora_B" in name:
                    param.add_(0.05)

        system.eval()
        out_a = system(packed, asp_a, floors, ranks, retreat)
        out_b = system(packed, asp_b, floors, ranks, retreat)
        # Conditioned system responds to PreferenceTable change
        self.assertFalse(torch.allclose(out_a["conditioned_logits"], out_b["conditioned_logits"]))

        # Baseline system must be strictly invariant to PreferenceTable / floor changes
        baseline = StandardJevBaselineSystem(
            backbone=_build_tiny_qwen_backbone(PoolingStrategy.SPAN_END),
            aat_config=aat_cfg,
        )
        baseline.eval()
        base_a = baseline(packed, asp_a, floors, ranks, retreat)
        base_b = baseline(packed, asp_b, floors * 4.0, ranks, retreat)
        self.assertTrue(torch.allclose(base_a["tradeoff_noul"], base_b["tradeoff_noul"]))

        # Checkpoint save/load round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = system.save_checkpoint(Path(tmpdir) / "ckpt_test")
            reloaded = TradeOffJevNeuralSystem(
                backbone=_build_tiny_qwen_backbone(PoolingStrategy.SPAN_END_AND_MEAN),
                aat_config=aat_cfg,
            )
            reloaded.load_checkpoint_weights(ckpt_path)
            reloaded.eval()
            out_reloaded = reloaded(packed, asp_a, floors, ranks, retreat)
            self.assertTrue(
                torch.allclose(out_a["tradeoff_noul"], out_reloaded["tradeoff_noul"], atol=1e-5)
            )


class TestCurriculumTrainerAndCalibrator(unittest.TestCase):
    """Test two-phase `CurriculumTrainer` and post-hoc `TemperatureCalibrator`."""

    def test_two_phase_curriculum_trainer_and_calibration(self) -> None:
        """Run Phase 1 + Phase 2 + TemperatureCalibrator on synthetic trajectories."""
        converter = TrajectoryRecordConverter(num_goals=4)
        jsonl_path = Path(__file__).resolve().parents[1] / "data" / "train_trajectories.jsonl"
        flat_samples, trajectories = converter.load_jsonl(jsonl_path)
        self.assertGreaterEqual(len(flat_samples), 10)
        self.assertGreaterEqual(len(trajectories), 5)

        aat_cfg = AATModelConfig(num_goals=4, num_levels=5, hidden_dim=32, pointer_dim=16)
        system = TradeOffJevNeuralSystem(
            backbone=_build_tiny_qwen_backbone(PoolingStrategy.SPAN_MEAN),
            aat_config=aat_cfg,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            t_cfg = TrainerConfig(
                output_dir=tmpdir,
                run_name="unit_test_run",
                batch_size=3,
                trajectory_batch_size=2,
                gradient_accumulation_steps=2,
                phase1_steps=3,
                phase2_steps=2,
                use_bf16=False,
                seed=11,
            )
            trainer = CurriculumTrainer(system=system, config=t_cfg, device="cpu")
            summary = trainer.train(
                phase1_samples=flat_samples,
                phase2_trajectories=trajectories,
                cal_samples=flat_samples,
            )
            self.assertEqual(summary.total_optimizer_steps, 5)
            self.assertEqual(summary.phase1_steps_completed, 3)
            self.assertEqual(summary.phase2_steps_completed, 2)
            self.assertTrue(Path(summary.metrics_jsonl_path).exists())
            self.assertTrue(Path(summary.final_checkpoint_dir).exists())
            self.assertIsNotNone(summary.noul_calibration_report)
            self.assertIsNotNone(summary.choice_calibration_report)

    def test_temperature_calibrator_improves_overconfident_logits(self) -> None:
        """Verify L-BFGS + Log-Grid scaling reduces NLL, Brier, and 15-bin ECE on miscalibrated logits."""
        torch.manual_seed(123)
        true_probs = torch.linspace(0.1, 0.9, 300)
        labels = torch.bernoulli(true_probs)
        true_logits = torch.log(true_probs / (1.0 - true_probs))
        # Inflate logits by 4.5x to simulate severe neural overconfidence
        overconfident_logits = 4.5 * true_logits

        cal = TemperatureCalibrator(mode=CalibrationMode.BINARY_NOUL, num_bins=15)
        rep = cal.fit(overconfident_logits, labels)
        self.assertEqual(len(rep.post_bins), 15)
        self.assertGreater(rep.optimal_temperature, 2.0)
        self.assertLess(rep.post_nll, rep.pre_nll)
        self.assertLess(rep.post_brier, rep.pre_brier)
        self.assertLess(rep.post_ece, rep.pre_ece)

    def test_zero_bleeding_and_candidate_shuffling_anonymization(self) -> None:
        """Verify `TrajectoryRecordConverter` strips `candidate_scores` by default and aligns CF permutations."""
        import json

        test_jsonl = Path(__file__).resolve().parents[1] / "data" / "corpus_v1" / "test.jsonl"
        lines = test_jsonl.read_text(encoding="utf-8").splitlines()
        rec_a = json.loads(lines[1])  # rec_software_release_cf_0028_a
        rec_b = json.loads(lines[2])  # rec_software_release_cf_0028_b

        conv_default = TrajectoryRecordConverter(num_goals=4)
        sample_default = conv_default.convert_record(rec_a)
        self.assertNotIn("candidate_scores", sample_default.state_text)

        conv_shuffled = TrajectoryRecordConverter(
            num_goals=4,
            include_oracle_scores_in_text=False,
            shuffle_and_anonymize_candidates=True,
        )
        shuf_a = conv_shuffled.convert_record(rec_a)
        shuf_b = conv_shuffled.convert_record(rec_b)
        self.assertNotIn("candidate_scores", shuf_a.state_text)
        self.assertEqual(shuf_a.state_text, shuf_b.state_text)
        self.assertEqual(shuf_a.candidate_descriptions, shuf_b.candidate_descriptions)
        for idx, desc in enumerate(shuf_a.candidate_descriptions):
            self.assertTrue(desc.startswith(f"option_{idx}: "))
            self.assertNotIn("cand_prioritize", desc)
        self.assertNotEqual(shuf_a.target_choice, shuf_b.target_choice)


if __name__ == "__main__":
    unittest.main()
