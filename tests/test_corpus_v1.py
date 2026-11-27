"""Comprehensive Unit & Integration Tests for Milestone 1 (`corpus_v1`).

Verifies:
    1. Exact split sizes (`4,000` train, `500` val, `500` cal, `600` test) and SHA-256
       manifest integrity on `data/corpus_v1/`.
    2. Strict zero data leakage across `train`, `val`, `cal`, and `test` over
       `scenario_group_id`, `episode_id`, `counterfactual_pair_id`, and `context_text`,
       plus strict isolation of `is_ood=True` domains inside `test.jsonl`.
    3. Sharp 3-way cognitive regime separation across all 5,600 records:
       - `APPROVAL`: `max(tradeoff_noul) >= 0.85`, `bartleby_noul <= 0.05`.
       - `EPISTEMIC_UNCERTAINTY`: `tradeoff_noul` in `[0.42, 0.58]`, `bartleby_noul` in `[0.02, 0.10]`.
       - `BARTLEBY_REFUSAL`: all `tradeoff_noul <= 0.05`, `bartleby_noul >= 0.95`,
         `utterance == "I would prefer not to."`.
    4. Counterfactual `PreferenceTable` flip consistency: byte-for-byte identical
       `context_text` and `candidates` under distinct presets yield opposing winners
       or opposing Bartleby refusals (`>= 0.85` probability swing).
    5. Asimov 8-subdomain & 4-preset coverage, Markov chain continuity
       ($\\mathcal{T}_0 \\to \\mathcal{T}_1 \\to \\mathcal{T}_2$), offline fallback in
       `TrolleyMoralCorpusGenerator`, and direct tensor compatibility with
       `TradeOffJevModel` + `AATCurriculumLoss`.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tradeoff_jev.datasets import (
    AsimovDilemmaCorpusGenerator,
    CorpusRecord,
    CorpusSplitConfig,
    DatasetSplit,
    DecisionRegime,
    ScaledCorpusBuilder,
    TrolleyMoralCorpusGenerator,
)
from tradeoff_jev.model import AATCurriculumLoss, AATModelConfig, TradeOffJevModel
from tradeoff_jev.oracle import BARTLEBY_UTTERANCE
from tradeoff_jev.schema import GoalSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_V1_DIR = REPO_ROOT / "data" / "corpus_v1"


class TestCorpusV1ExportAndLeakage(unittest.TestCase):
    """Verify exported `data/corpus_v1/` files, zero data leakage, and 3-regime calibration."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the generated `data/corpus_v1/` splits and `manifest.json` once."""
        if not (CORPUS_V1_DIR / "manifest.json").exists():
            ScaledCorpusBuilder().export_corpus(CORPUS_V1_DIR)

        cls._manifest = json.loads(
            (CORPUS_V1_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        cls._splits: dict[str, list[CorpusRecord]] = {}
        for split_name in ("train", "val", "cal", "test"):
            path = CORPUS_V1_DIR / f"{split_name}.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            cls._splits[split_name] = [CorpusRecord.from_dict(json.loads(line)) for line in lines]

    def test_split_counts_and_manifest_checksums(self) -> None:
        """Verify exact 4000 / 500 / 500 / 600 record counts and SHA-256 checksums."""
        expected_counts = {"train": 4000, "val": 500, "cal": 500, "test": 600}
        self.assertEqual(self._manifest["total_records"], 5600)
        self.assertTrue(self._manifest["zero_group_leakage_verified"])

        for split_name, exp_cnt in expected_counts.items():
            self.assertEqual(len(self._splits[split_name]), exp_cnt)
            fname = f"{split_name}.jsonl"
            raw_bytes = (CORPUS_V1_DIR / fname).read_bytes()
            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
            expected_sha = self._manifest["files"][fname]["sha256"]
            if actual_sha != expected_sha and b"\r\n" in raw_bytes:
                actual_sha = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()
            self.assertEqual(expected_sha, actual_sha)
            self.assertEqual(self._manifest["files"][fname]["num_records"], exp_cnt)

    def test_zero_group_episode_and_context_leakage_across_splits(self) -> None:
        """Verify strict pairwise disjointness of `scenario_group_id`, `episode_id`, and `context_text`."""
        split_names = ("train", "val", "cal", "test")
        groups_per_split = {
            s: {r.scenario_group_id for r in self._splits[s]} for s in split_names
        }
        episodes_per_split = {
            s: {r.episode_id for r in self._splits[s]} for s in split_names
        }
        contexts_per_split = {
            s: {r.context_text for r in self._splits[s]} for s in split_names
        }

        for i, s_a in enumerate(split_names):
            for s_b in split_names[i + 1 :]:
                self.assertEqual(
                    groups_per_split[s_a] & groups_per_split[s_b],
                    set(),
                    f"Group leakage detected between {s_a} and {s_b}!",
                )
                self.assertEqual(
                    episodes_per_split[s_a] & episodes_per_split[s_b],
                    set(),
                    f"Episode leakage detected between {s_a} and {s_b}!",
                )
                self.assertEqual(
                    contexts_per_split[s_a] & contexts_per_split[s_b],
                    set(),
                    f"Surface context_text leakage detected between {s_a} and {s_b}!",
                )

        # Verify OOD scenarios exist ONLY in held-out test.jsonl
        for s in ("train", "val", "cal"):
            self.assertEqual(sum(1 for r in self._splits[s] if r.is_ood), 0)
        ood_test = [r for r in self._splits["test"] if r.is_ood]
        self.assertEqual(len(ood_test), 120)
        self.assertEqual(
            {r.domain_name for r in ood_test},
            {"energy_grid_dispatch_ood", "clinical_protocol_ood"},
        )

    def test_three_regimes_present_and_sharply_calibrated(self) -> None:
        """Verify Approval (~1.0), Epistemic (~0.50), and Bartleby (~1.0) across all splits."""
        for split_name, records in self._splits.items():
            by_regime: dict[DecisionRegime, list[CorpusRecord]] = defaultdict(list)
            for rec in records:
                by_regime[rec.regime].append(rec)

            for regime in DecisionRegime:
                self.assertGreaterEqual(
                    len(by_regime[regime]),
                    40,
                    f"Split {split_name} has insufficient examples for {regime.value}",
                )

            for rec in by_regime[DecisionRegime.APPROVAL]:
                self.assertGreaterEqual(max(rec.tradeoff_noul.values()), 0.90)
                self.assertLessEqual(rec.bartleby_noul, 0.05)
                self.assertFalse(rec.bartleby_triggered)
                self.assertIsNotNone(rec.selected_candidate_id)

            for rec in by_regime[DecisionRegime.EPISTEMIC_UNCERTAINTY]:
                for p in rec.tradeoff_noul.values():
                    self.assertGreaterEqual(p, 0.45)
                    self.assertLessEqual(p, 0.55)
                self.assertGreaterEqual(rec.bartleby_noul, 0.03)
                self.assertLessEqual(rec.bartleby_noul, 0.08)
                self.assertFalse(rec.bartleby_triggered)
                self.assertIsNone(rec.selected_candidate_id)

            for rec in by_regime[DecisionRegime.BARTLEBY_REFUSAL]:
                self.assertLessEqual(max(rec.tradeoff_noul.values()), 0.05)
                self.assertGreaterEqual(rec.bartleby_noul, 0.95)
                self.assertTrue(rec.bartleby_triggered)
                self.assertEqual(rec.utterance, BARTLEBY_UTTERANCE)
                self.assertIsNone(rec.selected_candidate_id)

    def test_counterfactual_flip_consistency(self) -> None:
        """Verify paired counterfactual records share identical text $x_t$ and flip decisions."""
        total_pairs = 0
        for split_name, records in self._splits.items():
            pairs: dict[str, list[CorpusRecord]] = defaultdict(list)
            for r in records:
                if r.counterfactual_pair_id is not None:
                    pairs[r.counterfactual_pair_id].append(r)

            self.assertGreaterEqual(len(pairs), 50, f"Too few CF pairs in {split_name}")
            for cf_id, siblings in pairs.items():
                self.assertEqual(len(siblings), 2, f"Incomplete CF pair {cf_id} in {split_name}")
                rec_a, rec_b = siblings
                self.assertEqual(rec_a.scenario_group_id, rec_b.scenario_group_id)
                self.assertEqual(rec_a.context_text, rec_b.context_text)
                self.assertEqual(
                    [c.to_dict() for c in rec_a.candidates],
                    [c.to_dict() for c in rec_b.candidates],
                )
                self.assertNotEqual(rec_a.preset_name, rec_b.preset_name)

                # Verify genuine decision reversal (winner flip OR Bartleby flip)
                if rec_a.bartleby_triggered != rec_b.bartleby_triggered:
                    self.assertGreaterEqual(abs(rec_a.bartleby_noul - rec_b.bartleby_noul), 0.90)
                    active_rec = rec_b if rec_a.bartleby_triggered else rec_a
                    refusal_rec = rec_a if rec_a.bartleby_triggered else rec_b
                    win_cid = active_rec.selected_candidate_id
                    assert win_cid is not None
                    self.assertGreaterEqual(
                        active_rec.tradeoff_noul[win_cid] - refusal_rec.tradeoff_noul[win_cid],
                        0.90,
                    )
                else:
                    self.assertNotEqual(rec_a.selected_candidate_id, rec_b.selected_candidate_id)
                    win_a = rec_a.selected_candidate_id
                    win_b = rec_b.selected_candidate_id
                    assert win_a is not None and win_b is not None
                    self.assertGreaterEqual(
                        rec_a.tradeoff_noul[win_a] - rec_b.tradeoff_noul[win_a], 0.85
                    )
                    self.assertGreaterEqual(
                        rec_b.tradeoff_noul[win_b] - rec_a.tradeoff_noul[win_b], 0.85
                    )
                total_pairs += 1

        self.assertGreaterEqual(total_pairs, 900)


class TestGeneratorsMarkovAndModelIntegration(unittest.TestCase):
    """Test generator internals, Markov continuity, offline fallback, and `TradeOffJevModel` batching."""

    def test_asimov_subdomains_presets_and_markov_continuity(self) -> None:
        """Verify all 8 Asimov subdomains, 4 presets, and Markov state handoff $t=0 \\to 1 \\to 2$."""
        gen = AsimovDilemmaCorpusGenerator()
        recs = gen.generate_records(240, seed=11)
        expected_subdomains = {
            "planetary_mining",
            "surgical_robotics",
            "autonomous_rescue",
            "industrial_hazard",
            "telepathic_counseling",
            "bystander_omission",
            "equilibrium_hazard_loop",
            "civilization_triage",
        }
        self.assertEqual({r.subdomain for r in recs}, expected_subdomains)
        self.assertEqual({r.preset_name for r in recs}, set(gen.ALL_PRESETS))

        # Check multi-step Markov continuity: prior_table at step t+1 == adapted_table at step t
        episodes: dict[str, list[CorpusRecord]] = defaultdict(list)
        for r in recs:
            if r.source == "asimov_markov_trajectory":
                episodes[r.episode_id].append(r)

        self.assertGreaterEqual(len(episodes), 15)
        for ep_id, steps in episodes.items():
            ordered = sorted(steps, key=lambda x: x.step)
            self.assertEqual([s.step for s in ordered], [0, 1, 2], f"Broken steps in {ep_id}")
            self.assertEqual(
                ordered[1].prior_preference_table.to_dict(),
                ordered[0].adapted_preference_table.to_dict(),
            )
            self.assertEqual(
                ordered[2].prior_preference_table.to_dict(),
                ordered[1].adapted_preference_table.to_dict(),
            )

    def test_trolley_offline_mode_and_request_roundtrip(self) -> None:
        """Verify `TrolleyMoralCorpusGenerator` succeeds offline and round-trips to `TradeOffRequest`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = TrolleyMoralCorpusGenerator(allow_network=False, cache_dir=Path(tmpdir))
            recs = gen.generate_records(90, seed=99)
            self.assertEqual(len(recs), 90)
            self.assertEqual({r.regime for r in recs}, set(DecisionRegime))

            req = recs[0].to_tradeoff_request(gen.goal_schema)
            req.validate()
            clef_payload = req.to_clef_joint_schema()
            self.assertEqual(clef_payload["architecture"], "clef_joint_schema_arch_a")

        # Also verify goal_schemas stored in manifest.json deserialize cleanly
        manifest = json.loads((CORPUS_V1_DIR / "manifest.json").read_text(encoding="utf-8"))
        for dom_name, raw_goals in manifest["goal_schemas"].items():
            specs = [GoalSpec.from_dict(g) for g in raw_goals]
            self.assertEqual(len(specs), 4, f"Domain {dom_name} should have 4 goals")

    def test_corpus_tensor_supervision_with_tradeoff_jev_model(self) -> None:
        """Feed real `corpus_v1` supervision tensors into `TradeOffJevModel` & `AATCurriculumLoss`."""
        train_lines = (CORPUS_V1_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()
        two_cand_recs = [
            CorpusRecord.from_dict(json.loads(line))
            for line in train_lines[:80]
            if len(json.loads(line)["candidates"]) == 2
        ][:8]
        self.assertEqual(len(two_cand_recs), 8)

        cfg = AATModelConfig(num_goals=4, num_levels=5, hidden_dim=32, pointer_dim=16, seed=42)
        model = TradeOffJevModel(cfg)
        criterion = AATCurriculumLoss()

        rng = np.random.default_rng(42)
        batch_size = len(two_cand_recs)
        decide_h = rng.normal(0.0, 0.2, size=(batch_size, 32)).astype(np.float32)
        cand_h = rng.normal(0.0, 0.2, size=(batch_size, 2, 32)).astype(np.float32)

        a_prev = np.array([r.tensors.aspiration_prev for r in two_cand_recs], dtype=np.float32)
        s_floor = np.array([r.tensors.survival_floor for r in two_cand_recs], dtype=np.float32)
        u_ranks = np.array([r.tensors.urgency_ranks for r in two_cand_recs], dtype=np.float32)
        r_idx = np.array([r.tensors.retreat_index for r in two_cand_recs], dtype=np.int64)
        cand_scores = np.array(
            [r.tensors.candidate_scores for r in two_cand_recs], dtype=np.float32
        )

        target_choice = np.array(
            [r.tensors.target_choice_index for r in two_cand_recs], dtype=np.int64
        )
        target_noul = np.array([r.tensors.target_noul for r in two_cand_recs], dtype=np.float32)
        target_asp = np.array(
            [r.tensors.target_aspiration for r in two_cand_recs], dtype=np.float32
        )
        target_bartleby = np.array(
            [r.tensors.target_bartleby for r in two_cand_recs], dtype=np.float32
        )

        outputs = model(
            decide_h,
            cand_h,
            a_prev,
            s_floor,
            u_ranks,
            r_idx,
            override_expected_scores=cand_scores,
        )
        losses = criterion(outputs, target_choice, target_noul, target_asp, target_bartleby)
        self.assertTrue(np.isfinite(losses["loss"]))
        self.assertGreater(losses["loss"], 0.0)

    def test_custom_small_split_config(self) -> None:
        """Verify `ScaledCorpusBuilder` supports custom `CorpusSplitConfig` sizes cleanly."""
        small_cfg = CorpusSplitConfig(
            train_size=160,
            val_size=40,
            cal_size=40,
            test_size=60,
            ood_test_size=16,
            seed=777,
        )
        builder = ScaledCorpusBuilder(config=small_cfg, allow_network=False)
        splits = builder.build_splits()
        self.assertEqual(len(splits[DatasetSplit.TRAIN]), 160)
        self.assertEqual(len(splits[DatasetSplit.VAL]), 40)
        self.assertEqual(len(splits[DatasetSplit.CAL]), 40)
        self.assertEqual(len(splits[DatasetSplit.TEST]), 60)

    def test_trainer_jsonl_contract_and_counterfactual_text_blindness(self) -> None:
        """Verify JSONL `request`/`response` structure matches `TrajectoryRecordConverter` without leaking presets."""
        test_lines = (CORPUS_V1_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()
        raw_dicts = [json.loads(line) for line in test_lines]

        cf_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in raw_dicts:
            self.assertIn("request", row)
            self.assertIn("response", row)
            req = row["request"]
            resp = row["response"]
            self.assertEqual(len(req["goal_schema"]), 4)
            self.assertIn("prior_preference_table", resp)
            self.assertIn("preference_table", resp)
            self.assertIn("answers", resp)
            # Ensure preset and survival_floors are NOT leaked inside request.state
            self.assertNotIn("preset", req["state"])
            self.assertNotIn("survival_floors", req["state"])
            if row.get("counterfactual_pair_id"):
                cf_groups[str(row["counterfactual_pair_id"])].append(row)

        self.assertGreaterEqual(len(cf_groups), 50)
        for cf_id, pair in cf_groups.items():
            self.assertEqual(len(pair), 2, f"Expected 2 siblings for {cf_id}")
            state_a = {
                k: v for k, v in pair[0]["request"]["state"].items() if k != "candidate_scores"
            }
            state_b = {
                k: v for k, v in pair[1]["request"]["state"].items() if k != "candidate_scores"
            }
            self.assertEqual(state_a, state_b)


if __name__ == "__main__":
    unittest.main()
