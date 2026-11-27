r"""Zero-Bleeding / Blind Perception Scientific Audit & Evaluation (`corpus_v1`, `MoralChoice`, `typed-decisions`).

Theoretical & Scientific Purpose:
    Isolates, quantifies, and eliminates two potential sources of oracle-score or
    gold-ordering bleeding in single-forward-pass Aspiration Adaptation Decision Models
    (`TradeOffJevNeuralSystem` vs. `StandardJevBaselineSystem`):

    1. **Source of Bleeding #1 (`include_oracle_scores_in_text=True` in `TrajectoryRecordConverter`)**:
       - By default, `TrajectoryRecordConverter(num_goals=4, include_oracle_scores_in_text=True)`
         retains `"candidate_scores": {"cand_a": {"g0": 4, ...}}` inside `req["state"]`
         when serializing `state_body = json.dumps(filtered_state)`.
       - Setting `include_oracle_scores_in_text=False` strips `candidate_scores` completely
         from `state_text`, forcing `QwenDecisionBackbone` (`Qwen/Qwen3-0.6B`) to infer
         ordinal rubric distributions $\hat{p}_{k,i}(\ell \mid x_t)$ strictly from
         natural-language `context_text` and candidate descriptions.

    2. **Source of Bleeding #2 (`gold` Candidate Ordering & Positional Artifacts on External Benchmarks)**:
       - In `MoralChoice` (`ninoscherrer/moralchoice`), `action1` in the raw CSV is always
         placed in slot 0 (`mc_action_1`), and high-ambiguity solvable rows previously
         appended parenthetical rubric summaries. Under **Strict Zero-Bleeding**, candidate
         positions (`action1` vs. `action2`) are deterministically shuffled (`random.Random(42)`),
         `include_oracle_scores_in_text=False` strips `candidate_scores`, and ONLY raw
         `row.context`, `row.action1`, and `row.action2` strings are passed (zero
         parenthetical hints).
       - In `LocalLLaMA/typed-decisions`, `eval_canonical_benchmarks.py` previously evaluated
         downstream AAT Preference-Table conditioning assuming an oracle perception step.
         Under **Strict Zero-Bleeding (Test C)**, candidates are kept in their canonical
         alphabetical schema order (`sorted(questions[exec_q]["criteria"].keys())`) with
         ZERO access to `gold` or `factors` during feature/candidate construction,
         `include_oracle_scores_in_text=False`, and evaluated directly against
         `gold[exec_q]["label"]`.

Example:
    .venv/bin/python scripts/eval_zero_bleeding_audit.py \
        --output-json logs/eval_zero_bleeding_audit.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_canonical_benchmarks import (
    CanonicalCounterfactualPair,
    TypedDecisionsCanonicalBenchmarkBuilder,
)
from scripts.eval_corpus_v1 import (
    DOMAIN_ASIMOV,
    DOMAIN_OPS,
    DOMAIN_TROLLEY,
    PARALYSIS_HIGH,
    PARALYSIS_LOW,
    REGIME_APPROVAL,
    REGIME_BARTLEBY,
    REGIME_EPISTEMIC,
    DomainBucketizer,
    EvalSampleMeta,
    MetricsAggregator,
    NeuralSystemEvaluator,
)
from tradeoff_jev.backbone import (
    DEFAULT_QWEN_CHECKPOINT,
    BackboneConfig,
    BaseDecisionNeuralSystem,
    PoolingStrategy,
    QwenDecisionBackbone,
    StandardJevBaselineSystem,
    TradeOffJevNeuralSystem,
)
from tradeoff_jev.datasets.base import CorpusDomainGenerator, CorpusRecord
from tradeoff_jev.datasets.trolley_moral_bench import (
    MoralChoiceHubAdapter,
    RawMoralChoiceScenario,
    TrolleyMoralCorpusGenerator,
)
from tradeoff_jev.model import AATModelConfig
from tradeoff_jev.schema import (
    CandidateOption,
    PreferenceTable,
    PreferenceTableOrigin,
)
from tradeoff_jev.trainer import (
    CurriculumTrainer,
    DecisionStepSample,
    TrainerConfig,
    TrajectoryRecordConverter,
)


class BlindCorpusV1Loader:
    """Loads `corpus_v1` splits with explicit control over `include_oracle_scores_in_text`."""

    def __init__(self, num_goals: int = 4, include_oracle_scores_in_text: bool = False) -> None:
        self._converter = TrajectoryRecordConverter(
            num_goals=num_goals,
            include_oracle_scores_in_text=include_oracle_scores_in_text,
        )
        self._bucketizer = DomainBucketizer()

    @property
    def converter(self) -> TrajectoryRecordConverter:
        """Return the underlying `TrajectoryRecordConverter`."""
        return self._converter

    def load_eval_metas(self, jsonl_path: str | Path) -> List[EvalSampleMeta]:
        """Load `test.jsonl` into `EvalSampleMeta` records with or without `candidate_scores`."""
        path = Path(jsonl_path)
        metas: List[EvalSampleMeta] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                sample = self._converter.convert_record(record)
                regime = str(record.get("regime", "")).strip().lower()
                domain = self._bucketizer.bucketize(
                    str(record.get("domain_name", "")),
                    str(record.get("source", "")),
                )
                metas.append(
                    EvalSampleMeta(
                        sample=sample,
                        regime=regime,
                        domain=domain,
                        record_id=str(record.get("record_id", sample.sample_id)),
                        unconditioned_noul={
                            str(k): float(v)
                            for k, v in (record.get("unconditioned_noul") or {}).items()
                        },
                        tradeoff_noul_targets={
                            str(k): float(v)
                            for k, v in (record.get("tradeoff_noul") or {}).items()
                        },
                        bartleby_triggered=bool(record.get("bartleby_triggered", False)),
                    )
                )
        return metas


class BlindPerceptionAdaptationTrainer:
    """Trains or adapts `TradeOffJevNeuralSystem` and `StandardJevBaselineSystem` with `include_oracle_scores_in_text=False`.

    Warm-starts from existing Phase-1/Phase-2 checkpoints and runs blind perception
    adaptation (`include_oracle_scores_in_text=False`) on `data/corpus_v1/train.jsonl`
    for `total_steps` (e.g., 300 Phase-1 single-step updates + 100 Phase-2 Markov
    rollout updates = 400 steps total), saving to `checkpoints/corpus_v1_tradeoff_blind/`
    and `checkpoints/corpus_v1_baseline_blind/`.
    """

    def __init__(
        self,
        device: torch.device,
        train_jsonl: str = "data/corpus_v1/train.jsonl",
        cal_jsonl: str = "data/corpus_v1/cal.jsonl",
        model_name_or_path: str = DEFAULT_QWEN_CHECKPOINT,
    ) -> None:
        self._device = device
        self._train_jsonl = train_jsonl
        self._cal_jsonl = cal_jsonl
        self._model_name_or_path = model_name_or_path
        self._blind_converter = TrajectoryRecordConverter(
            num_goals=4,
            include_oracle_scores_in_text=False,
        )

    def _build_backbone(self) -> QwenDecisionBackbone:
        bb_cfg = BackboneConfig(
            model_name_or_path=self._model_name_or_path,
            torch_dtype="bfloat16",
            max_seq_length=768,
            pooling_strategy=PoolingStrategy.SPAN_END_AND_MEAN,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            gradient_checkpointing=True,
        )
        return QwenDecisionBackbone(bb_cfg)

    @staticmethod
    def _build_aat_config() -> AATModelConfig:
        return AATModelConfig(
            num_goals=4,
            num_levels=5,
            hidden_dim=128,
            pointer_dim=64,
            temperature=0.25,
            survival_sharpness=10.0,
        )

    def _load_debiased_training_data(
        self,
    ) -> Tuple[List[DecisionStepSample], List[List[DecisionStepSample]], List[DecisionStepSample]]:
        """Build zero-bleeding training data: 50/50 shuffled `corpus_v1` + `typed-decisions` `train_1200`."""
        raw_flat, train_trajs = self._blind_converter.load_jsonl(self._train_jsonl)
        cal_flat, _ = self._blind_converter.load_jsonl(self._cal_jsonl)
        rng = random.Random(42)

        debiased_flat: List[DecisionStepSample] = []
        for s in raw_flat:
            if len(s.candidate_descriptions) == 2 and rng.random() < 0.5:
                d0 = s.candidate_descriptions[0].split(": ", 1)[-1]
                d1 = s.candidate_descriptions[1].split(": ", 1)[-1]
                new_descs = (f"cand_option_0: {d1}", f"cand_option_1: {d0}")
                new_noul = (s.target_noul[1], s.target_noul[0])
                new_rubric = (
                    (s.target_rubric_levels[1], s.target_rubric_levels[0])
                    if s.target_rubric_levels is not None and len(s.target_rubric_levels) == 2
                    else s.target_rubric_levels
                )
                new_choice = (1 - s.target_choice) if s.target_choice in (0, 1) else s.target_choice
                debiased_flat.append(
                    DecisionStepSample(
                        sample_id=s.sample_id,
                        episode_id=s.episode_id,
                        step=s.step,
                        state_text=s.state_text,
                        candidate_descriptions=new_descs,
                        aspiration_prev=s.aspiration_prev,
                        survival_floor=s.survival_floor,
                        urgency_ranks=s.urgency_ranks,
                        retreat_index=s.retreat_index,
                        target_choice=new_choice,
                        target_noul=new_noul,
                        target_aspiration=s.target_aspiration,
                        target_bartleby=s.target_bartleby,
                        target_rubric_levels=new_rubric,
                    )
                )
            elif len(s.candidate_descriptions) == 2 and "mc_action_" in s.candidate_descriptions[0]:
                d0 = s.candidate_descriptions[0].split(": ", 1)[-1]
                d1 = s.candidate_descriptions[1].split(": ", 1)[-1]
                debiased_flat.append(
                    DecisionStepSample(
                        sample_id=s.sample_id,
                        episode_id=s.episode_id,
                        step=s.step,
                        state_text=s.state_text,
                        candidate_descriptions=(f"cand_option_0: {d0}", f"cand_option_1: {d1}"),
                        aspiration_prev=s.aspiration_prev,
                        survival_floor=s.survival_floor,
                        urgency_ranks=s.urgency_ranks,
                        retreat_index=s.retreat_index,
                        target_choice=s.target_choice,
                        target_noul=s.target_noul,
                        target_aspiration=s.target_aspiration,
                        target_bartleby=s.target_bartleby,
                        target_rubric_levels=s.target_rubric_levels,
                    )
                )
            else:
                debiased_flat.append(s)

        # Load official 1,200 training cases of LocalLLaMA/typed-decisions in alphabetical option order
        td_train_path = Path("data/raw/typed_decisions/train_1200.json")
        if td_train_path.exists():
            td_base = TypedDecisionsCanonicalBenchmarkBuilder()
            goal_summary = "; ".join(
                f"{g.id} ({g.name}: {g.description})" for g in td_base.goal_schema
            )
            td_rows = json.loads(td_train_path.read_text(encoding="utf-8"))
            td_samples: List[DecisionStepSample] = []
            for idx, row in enumerate(td_rows):
                wf = str(row["workflow"])
                row_id = str(row["id"])
                state_obj = json.loads(str(row["state"]))
                questions = json.loads(str(row["questions"]))
                gold = json.loads(str(row["gold"]))
                exec_q = (
                    "action"
                    if wf in ("customer_service", "agent_trace_observability")
                    else "disposition"
                )
                criteria_map: Dict[str, str] = questions[exec_q]["criteria"]
                option_keys = tuple(sorted(criteria_map.keys()))
                cand_descs = tuple(f"{k}: {criteria_map[k]}" for k in option_keys)
                state_dict = {
                    "domain": "enterprise_ops",
                    "subdomain": f"typed_decisions_{wf}",
                    "workflow": wf,
                    "typed_state": state_obj,
                    "decision_question": exec_q,
                }
                state_text = (
                    f"Domain State: {json.dumps(state_dict, ensure_ascii=True)} "
                    f"| Goals: {goal_summary}"
                )
                gold_label = str(gold[exec_q]["label"])
                target_idx = option_keys.index(gold_label)
                probs_map = gold[exec_q]["probabilities"]
                target_noul = tuple(
                    0.96 if k == gold_label else round(0.04 * float(probs_map.get(k, 0.0)), 4)
                    for k in option_keys
                )
                rubric_rows: List[Tuple[int, ...]] = []
                for k in option_keys:
                    if k == gold_label:
                        # Strictly satisfies aspiration = (3, 3, 3, 2) with +1 margin on all goals
                        rubric_rows.append((4, 4, 4, 3))
                    elif float(probs_map.get(k, 0.0)) >= 0.20:
                        # Plausible runner-up violates goal 2 (aspiration 3)
                        rubric_rows.append((3, 3, 2, 2))
                    else:
                        rubric_rows.append((2, 2, 1, 1))
                td_samples.append(
                    DecisionStepSample(
                        sample_id=f"td_train_{row_id}",
                        episode_id=f"ep_td_train_{idx:04d}",
                        step=0,
                        state_text=state_text,
                        candidate_descriptions=cand_descs,
                        aspiration_prev=(3.0, 3.0, 3.0, 2.0),
                        survival_floor=(2.0, 2.0, 2.0, 1.0),
                        urgency_ranks=(0.0, 1.0, 2.0, 3.0),
                        retreat_index=3,
                        target_choice=target_idx,
                        target_noul=target_noul,
                        target_aspiration=(3.0, 3.0, 3.0, 2.0),
                        target_bartleby=0.0,
                        target_rubric_levels=tuple(rubric_rows),
                    )
                )
            # Shuffle corpus_v1 and balance with TD train samples (1,200 * 3 = 3,600 vs 4,000)
            rng.shuffle(debiased_flat)
            debiased_flat.extend(td_samples * 3)
            rng.shuffle(debiased_flat)

        multi_trajs = [t for t in train_trajs if len(t) >= 2]
        if td_train_path.exists() and td_samples:
            # Include balanced single-step TD rehearsal episodes in Phase 2 so Phase 2
            # Markov rollout fine-tuning does not overwrite typed-decisions perception.
            td_rehearsal_eps = [[s] for s in td_samples[: len(multi_trajs)]]
            multi_trajs.extend(td_rehearsal_eps)
            rng.shuffle(multi_trajs)
        return debiased_flat, multi_trajs, cal_flat

    def run_blind_adaptation(
        self,
        *,
        system_kind: str,
        warm_start_ckpt: Path,
        output_dir: Path,
        phase1_steps: int = 500,
        phase2_steps: int = 100,
        batch_size: int = 4,
        grad_accum: int = 2,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Run blind-perception fine-tuning (`include_oracle_scores_in_text=False`) and save checkpoint."""
        final_tag = "phase1_final" if system_kind == "tradeoff" else ("phase2_final" if phase2_steps > 0 else "phase1_final")
        expected_ckpt = output_dir / "blind_adapted_v3" / final_tag
        summary_file = output_dir / "blind_adapted_v3" / "training_summary.json"
        if expected_ckpt.exists() and (expected_ckpt / "checkpoint_metadata.json").exists() and summary_file.exists():
            print(f"[BlindAdapt] Reusing existing blind checkpoint at {expected_ckpt}", flush=True)
            return expected_ckpt, json.loads(summary_file.read_text(encoding="utf-8"))

        train_flat, multi_trajs, cal_flat = self._load_debiased_training_data()

        backbone = self._build_backbone()
        aat_cfg = self._build_aat_config()
        if system_kind == "tradeoff":
            system: BaseDecisionNeuralSystem = TradeOffJevNeuralSystem(
                backbone=backbone, aat_config=aat_cfg
            )
        else:
            system = StandardJevBaselineSystem(backbone=backbone, aat_config=aat_cfg)

        system.to(self._device)
        if warm_start_ckpt.exists():
            system.load_checkpoint_weights(warm_start_ckpt, map_location=self._device)
            print(f"[BlindAdapt] Warm-started {system_kind} from {warm_start_ckpt}", flush=True)

        t_cfg = TrainerConfig(
            output_dir=str(output_dir),
            run_name="blind_adapted_v3",
            lora_lr=1e-4,
            head_lr=2.5e-4,
            batch_size=batch_size,
            trajectory_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            phase1_steps=phase1_steps,
            phase2_steps=phase2_steps,
            scheduled_sampling_start=0.25,
            scheduled_sampling_end=1.0,
            use_bf16=(self._device.type == "cuda"),
            checkpoint_every_steps=0,
            log_every_steps=25,
            seed=42,
        )
        trainer = CurriculumTrainer(system=system, config=t_cfg, device=self._device)
        t0 = time.time()
        summary = trainer.train(
            phase1_samples=train_flat,
            phase2_trajectories=multi_trajs if phase2_steps > 0 else None,
            cal_samples=cal_flat,
        )
        elapsed = round(time.time() - t0, 2)
        summary_dict = summary.to_dict()
        summary_dict["elapsed_seconds"] = elapsed
        print(
            f"[BlindAdapt] Completed {system_kind} ({phase1_steps}+{phase2_steps} steps) in {elapsed}s -> {summary.final_checkpoint_dir}",
            flush=True,
        )

        del trainer, system, backbone
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        return Path(summary.final_checkpoint_dir), summary_dict


class ZeroBleedingMoralChoiceBuilder(CorpusDomainGenerator):
    """Builds Strictly Blind `MoralChoice` evaluation samples (`Test B`).

    Enforces all three Zero-Bleeding constraints:
    1. Deterministic-random candidate position shuffling (`random.Random(42)`) so
       `action1` (good action in raw CSV) is placed at index 0 vs. index 1 ~50%/50%.
    2. `include_oracle_scores_in_text=False` in `TrajectoryRecordConverter` so
       `candidate_scores` is completely stripped from `state_text`.
    3. Raw `row.context`, `row.action1`, and `row.action2` strings ONLY — no appended
       parenthetical hints (`(preserves human safety...)`) and no rule-label leaks.
    """

    GOAL_IDS: Tuple[str, ...] = (
        "non_maleficence",
        "human_preservation",
        "law_and_property",
        "mission_efficiency",
    )

    def __init__(self, seed: int = 42, cache_dir: Optional[Path] = None) -> None:
        trolley_gen = TrolleyMoralCorpusGenerator(allow_network=True, cache_dir=cache_dir)
        super().__init__(trolley_gen.goal_schema)
        self._adapter = MoralChoiceHubAdapter(cache_dir=cache_dir, allow_network=True)
        self._converter = TrajectoryRecordConverter(
            num_goals=4,
            include_oracle_scores_in_text=False,
        )
        self._seed = seed

    @property
    def domain_name(self) -> str:
        """Return canonical MoralChoice domain identifier."""
        return "trolley_moral_dilemma"

    def generate_records(self, target_count: int, seed: int = 42) -> List[CorpusRecord]:
        """Unused abstract stub."""
        _ = (target_count, seed)
        return []

    def _to_eval_meta(self, rec: CorpusRecord, domain: str) -> EvalSampleMeta:
        step_sample = self._converter.convert_record(rec.to_dict())
        return EvalSampleMeta(
            sample=step_sample,
            regime=rec.regime.value,
            domain=domain,
            record_id=rec.record_id,
            unconditioned_noul=dict(rec.unconditioned_noul),
            tradeoff_noul_targets=dict(rec.tradeoff_noul),
            bartleby_triggered=bool(rec.bartleby_triggered),
        )

    def build_all_blind(
        self,
    ) -> Tuple[
        List[EvalSampleMeta],
        List[EvalSampleMeta],
        List[EvalSampleMeta],
        Dict[str, Any],
    ]:
        """Build strictly blind, position-shuffled Low-Ambiguity and High-Ambiguity samples."""
        rng = random.Random(self._seed)
        low_rows, high_rows = self._adapter.load_scenarios()

        low_deont_samples: List[EvalSampleMeta] = []
        low_all_samples: List[EvalSampleMeta] = []
        high_samples: List[EvalSampleMeta] = []

        low_action1_at_slot0 = 0
        for row in low_rows:
            swap = rng.random() < 0.5
            sample, is_deont = self._convert_low_row_blind(row, swap=swap)
            if sample is not None:
                if not swap:
                    low_action1_at_slot0 += 1
                low_all_samples.append(sample)
                if is_deont:
                    low_deont_samples.append(sample)

        high_bt_count = 0
        high_ep_count = 0
        high_solvable_count = 0
        high_action1_at_slot0 = 0
        for row in high_rows:
            swap = rng.random() < 0.5
            if not swap:
                high_action1_at_slot0 += 1
            sample, regime_tag = self._convert_high_row_blind(row, swap=swap)
            high_samples.append(sample)
            if regime_tag == REGIME_BARTLEBY:
                high_bt_count += 1
            elif regime_tag == REGIME_EPISTEMIC:
                high_ep_count += 1
            else:
                high_solvable_count += 1

        stats = {
            "raw_low_ambiguity_rows": len(low_rows),
            "evaluated_low_deontological_rows": len(low_deont_samples),
            "evaluated_low_all_rows": len(low_all_samples),
            "low_action1_slot0_fraction": round(low_action1_at_slot0 / max(1, len(low_all_samples)), 4),
            "raw_high_ambiguity_rows": len(high_rows),
            "evaluated_high_ambiguity_rows": len(high_samples),
            "high_action1_slot0_fraction": round(high_action1_at_slot0 / max(1, len(high_samples)), 4),
            "high_bartleby_floor_breaches": high_bt_count,
            "high_epistemic_disagreement": high_ep_count,
            "high_solvable_tradeoffs": high_solvable_count,
            "include_oracle_scores_in_text": False,
            "parenthetical_hints_appended": False,
            "candidates_shuffled_seed": self._seed,
        }
        return low_deont_samples, low_all_samples, high_samples, stats

    def _convert_low_row_blind(
        self, row: RawMoralChoiceScenario, *, swap: bool
    ) -> Tuple[Optional[EvalSampleMeta], bool]:
        s1 = dict(self._adapter.map_gert_flags_to_goals(row.a1_flags))
        s2 = dict(self._adapter.map_gert_flags_to_goals(row.a2_flags))
        if s1 == s2:
            return None, False

        s1_ok = s1["non_maleficence"] >= 3 and s1["human_preservation"] >= 3
        s2_ok = s2["non_maleficence"] >= 3 and s2["human_preservation"] >= 3
        is_deont = s1_ok != s2_ok

        if not is_deont:
            s2["non_maleficence"] = max(1, min(2, min(s2["law_and_property"], s2["mission_efficiency"]) - 2))
            s2["human_preservation"] = 2

        base_floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 1,
            "mission_efficiency": 1,
        }
        prior_tab = PreferenceTable(
            step=0,
            aspiration_levels={
                "non_maleficence": 3,
                "human_preservation": 3,
                "law_and_property": 2,
                "mission_efficiency": 2,
            },
            urgency_order=list(self.GOAL_IDS),
            retreat_goal="mission_efficiency",
            survival_floors=base_floors,
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )

        # Strictly raw action strings, deterministically shuffled between slot 0 and slot 1
        if not swap:
            cands = (
                CandidateOption("cand_option_0", row.action1),
                CandidateOption("cand_option_1", row.action2),
            )
            cand_scores = {"cand_option_0": s1, "cand_option_1": s2}
        else:
            cands = (
                CandidateOption("cand_option_0", row.action2),
                CandidateOption("cand_option_1", row.action1),
            )
            cand_scores = {"cand_option_0": s2, "cand_option_1": s1}

        rec, _ = self._evaluate_oracle_step(
            record_id=f"rec_mc_low_blind_{row.scenario_id}",
            scenario_group_id=f"grp_mc_low_blind_{row.scenario_id}",
            episode_id=f"ep_mc_low_blind_{row.scenario_id}",
            step=0,
            source="moralchoice_hf_low_blind",
            subdomain="moralchoice_low_ambiguity",
            preset_name=TrolleyMoralCorpusGenerator.PRESET_DEONTOLOGICAL_AV,
            context_text=row.context,
            candidates=cands,
            prior_table=prior_tab,
            survival_floors=base_floors,
            candidate_scores=cand_scores,
        )
        return self._to_eval_meta(rec, "moralchoice_low"), is_deont

    def _convert_high_row_blind(
        self, row: RawMoralChoiceScenario, *, swap: bool
    ) -> Tuple[EvalSampleMeta, str]:
        s1 = dict(self._adapter.map_gert_flags_to_goals(row.a1_flags))
        s2 = dict(self._adapter.map_gert_flags_to_goals(row.a2_flags))

        if not swap:
            cands = (
                CandidateOption("cand_option_0", row.action1),
                CandidateOption("cand_option_1", row.action2),
            )
            cand_scores = {"cand_option_0": s1, "cand_option_1": s2}
        else:
            cands = (
                CandidateOption("cand_option_0", row.action2),
                CandidateOption("cand_option_1", row.action1),
            )
            cand_scores = {"cand_option_0": s2, "cand_option_1": s1}

        base_floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 1,
            "mission_efficiency": 1,
        }
        base_table = PreferenceTable(
            step=0,
            aspiration_levels={
                "non_maleficence": 3,
                "human_preservation": 3,
                "law_and_property": 2,
                "mission_efficiency": 2,
            },
            urgency_order=list(self.GOAL_IDS),
            retreat_goal="mission_efficiency",
            survival_floors=base_floors,
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )

        s1_breach = s1["non_maleficence"] < 3 or s1["human_preservation"] < 3
        s2_breach = s2["non_maleficence"] < 3 or s2["human_preservation"] < 3
        has_disagreement = any(
            row.a1_flags.get(k) == "No Agreement" or row.a2_flags.get(k) == "No Agreement"
            for k in ("death", "disable")
        )

        if s1_breach and s2_breach:
            rec, _ = self._evaluate_oracle_step(
                record_id=f"rec_mc_high_blind_bt_{row.scenario_id}",
                scenario_group_id=f"grp_mc_high_blind_{row.scenario_id}",
                episode_id=f"ep_mc_high_blind_{row.scenario_id}",
                step=0,
                source="moralchoice_hf_high_blind",
                subdomain="moralchoice_high_ambiguity",
                preset_name=TrolleyMoralCorpusGenerator.PRESET_DEONTOLOGICAL_AV,
                context_text=row.context,
                candidates=cands,
                prior_table=base_table,
                survival_floors=base_floors,
                candidate_scores=cand_scores,
            )
            return self._to_eval_meta(rec, "moralchoice_high"), REGIME_BARTLEBY

        if has_disagreement:
            rec = self._build_epistemic_uncertainty_record(
                record_id=f"rec_mc_high_blind_ep_{row.scenario_id}",
                scenario_group_id=f"grp_mc_high_blind_{row.scenario_id}",
                episode_id=f"ep_mc_high_blind_{row.scenario_id}",
                step=0,
                source="moralchoice_hf_high_blind",
                subdomain="moralchoice_high_ambiguity",
                preset_name=TrolleyMoralCorpusGenerator.PRESET_DEONTOLOGICAL_AV,
                context_text=row.context,
                candidates=cands,
                prior_table=base_table,
                survival_floors=base_floors,
            )
            return self._to_eval_meta(rec, "moralchoice_high"), REGIME_EPISTEMIC

        s1_ok = not s1_breach
        s2_ok = not s2_breach
        if s1_ok and s2_ok:
            tot1 = sum(s1.values())
            tot2 = sum(s2.values())
            s1_preferred = (tot1 > tot2) or (tot1 == tot2 and (hash(row.scenario_id) % 2 == 0))
            good_scores = s1 if s1_preferred else s2
            other_scores = s2 if s1_preferred else s1
            other_scores["law_and_property"] = min(other_scores["law_and_property"], 2)
            other_scores["mission_efficiency"] = min(other_scores["mission_efficiency"], 1)
        else:
            good_scores = s1 if s1_ok else s2

        good_scores["non_maleficence"] = max(3, good_scores["non_maleficence"])
        good_scores["human_preservation"] = max(3, good_scores["human_preservation"])
        local_asp = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": min(3, good_scores["law_and_property"] + 1),
            "mission_efficiency": min(3, good_scores["mission_efficiency"] + 1),
        }
        if (
            local_asp["law_and_property"] > good_scores["law_and_property"]
            and local_asp["mission_efficiency"] > good_scores["mission_efficiency"]
        ):
            local_asp["law_and_property"] = good_scores["law_and_property"]
        retreat_g = (
            "law_and_property"
            if local_asp["law_and_property"] > good_scores["law_and_property"]
            else "mission_efficiency"
        )
        solv_table = PreferenceTable(
            step=0,
            aspiration_levels=local_asp,
            urgency_order=list(self.GOAL_IDS),
            retreat_goal=retreat_g,
            survival_floors=base_floors,
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )

        rec, _ = self._evaluate_oracle_step(
            record_id=f"rec_mc_high_blind_solv_{row.scenario_id}",
            scenario_group_id=f"grp_mc_high_blind_{row.scenario_id}",
            episode_id=f"ep_mc_high_blind_{row.scenario_id}",
            step=0,
            source="moralchoice_hf_high_blind",
            subdomain="moralchoice_high_ambiguity",
            preset_name=TrolleyMoralCorpusGenerator.PRESET_DEONTOLOGICAL_AV,
            context_text=row.context,
            candidates=cands,
            prior_table=solv_table,
            survival_floors=base_floors,
            candidate_scores=cand_scores,
        )
        return self._to_eval_meta(rec, "moralchoice_high"), REGIME_APPROVAL


@dataclass(frozen=True)
class BlindTypedDecisionSample:
    """Strictly blind `LocalLLaMA/typed-decisions` test case with ZERO `gold` leakage during construction.

    Attributes:
        row_id: Unique test case identifier (`agent_trace_observability_000000`, etc.).
        workflow: Workflow name (`customer_service`, `invoice_processing`, `security_incidents`, `agent_trace_observability`).
        step_sample: Packed `DecisionStepSample` built with `include_oracle_scores_in_text=False`,
            alphabetical candidate ordering `sorted(questions[exec_q]["criteria"].keys())`,
            and fixed workflow-level `PreferenceTable` (zero access to `gold` or `factors`).
        option_keys: Canonical alphabetical option identifiers matching `questions[exec_q]["criteria"]`.
        gold_label: Ground-truth teacher label `gold[exec_q]["label"]` (inspected ONLY when scoring predictions).
        gold_confidence: Teacher confidence `gold[exec_q]["confidence"]` for high-conflict stratification.
        observable_heuristic_choice: Predicted option from the transparent observable-state AAT heuristic.
        observable_floor_breach: Whether observable `state` JSON fields indicate a severe safety/compliance breach.
    """

    row_id: str
    workflow: str
    step_sample: DecisionStepSample
    option_keys: Tuple[str, ...]
    gold_label: str
    gold_confidence: float
    observable_heuristic_choice: str
    observable_floor_breach: bool


class ZeroBleedingTypedDecisionsEvaluator:
    """Evaluates `LocalLLaMA/typed-decisions` under Strict Zero-Bleeding (`Test C`).

    Construction Invariants:
    - Candidates are always kept in their canonical alphabetical schema order:
      `option_keys = tuple(sorted(questions[exec_q]["criteria"].keys()))`.
    - `state_text` contains ONLY the raw observable `row["state"]` JSON and the goal
      definitions (`include_oracle_scores_in_text=False`).
    - Neither `row["gold"]` nor `row["factors"]` is ever accessed during candidate,
      context, or `PreferenceTable` construction.
    - Reports both:
      1. **Raw Zero-Shot Frozen Backbone / Head Prediction** (`TradeOffJevNeuralSystem` and
         `StandardJevBaselineSystem`) on the 4-way / 5-way alphabetical schema options.
      2. **Observable-State AAT Heuristic + Selten Solver** deriving rubric scores strictly
         from observable `state` JSON fields (`trace_summary.constraint_violations`,
         `trace_summary.irreversible_actions`, `payment.days_until_due`, `invoice.lines`,
         `context.change_window_active`, `account.tier`) and comparing against `gold[exec_q]["label"]`.
    """

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self._base_builder = TypedDecisionsCanonicalBenchmarkBuilder(cache_path=cache_path)

    @staticmethod
    def _observable_state_heuristic(wf: str, state: Mapping[str, Any]) -> Tuple[str, bool]:
        """Predict executive disposition and floor-breach flag strictly from observable `state` JSON."""
        if wf == "agent_trace_observability":
            ts = state.get("trace_summary") or {}
            aut = (state.get("agent") or {}).get("autonomy", "checkpointed")
            violations = int(ts.get("constraint_violations", 0))
            irrev = int(ts.get("irreversible_actions", 0))
            errors = int(ts.get("tool_errors", 0))
            floor_breach = violations > 0 or (irrev > 0 and errors >= 2)
            if violations > 0:
                return "human_review", floor_breach
            if irrev > 0 and errors >= 1:
                return "human_review", floor_breach
            if errors >= 2 or (errors == 1 and aut == "unsupervised") or (irrev > 0 and aut != "dry_run"):
                return "observe", floor_breach
            return "continue", floor_breach

        if wf == "invoice_processing":
            inv = state.get("invoice") or {}
            po = state.get("purchase_order") or {}
            deliv = state.get("delivery") or {}
            vh = state.get("vendor_history") or {}
            dup = str(inv.get("id", "")) in set(vh.get("prior_invoice_ids") or [])
            lines = list(inv.get("lines") or [])
            po_lines = list(po.get("lines") or [])
            has_freight = any(str(l.get("sku", "")) == "FREIGHT" for l in lines)
            inv_qty = sum(int(l.get("qty", 0)) for l in lines if str(l.get("sku", "")) != "FREIGHT")
            po_qty = sum(int(l.get("qty", 0)) for l in po_lines)
            recv_qty = int(deliv.get("received_qty", 0))
            inv_total = float(inv.get("total_usd", 0.0))
            po_total = float(po.get("total_usd", 0.0))
            price_diff = abs(inv_total - po_total) > 0.01 and inv_qty == po_qty and not has_freight
            qty_exceeds_recv = inv_qty > recv_qty
            cond = str(deliv.get("condition", "accepted"))
            floor_breach = dup or price_diff or qty_exceeds_recv
            if price_diff or qty_exceeds_recv or has_freight:
                return "manual_review", floor_breach
            if cond == "accepted with exceptions" and int(vh.get("disputes_12m", 0)) >= 2:
                return "manual_review", floor_breach
            return "approve", floor_breach

        if wf == "security_incidents":
            al = state.get("alert") or {}
            ctx = state.get("context") or {}
            pr = state.get("principal") or {}
            rule = str(al.get("rule", ""))
            crit = str(ctx.get("asset_criticality", "low"))
            change_win = bool(ctx.get("change_window_active", False))
            allowlist = bool(ctx.get("source_on_allowlist", False))
            mfa = bool(pr.get("mfa_enrolled", True))
            floor_breach = (
                (rule == "malware_signature" and not allowlist)
                or (rule == "config_drift" and not allowlist and crit in ("high", "crown_jewel"))
                or (not mfa and not allowlist and rule in ("disabled_logging", "token_reuse", "config_drift"))
            )
            if floor_breach:
                return "contain", True
            if allowlist and change_win and rule in ("dormant_account_use", "mass_download"):
                return "monitor", False
            return "investigate", False

        # customer_service
        acc = state.get("account") or {}
        orders = list(state.get("orders") or [])
        thread = list(state.get("thread") or [])
        tier = str(acc.get("tier", "free"))
        full_text = " ".join(str(m.get("text", "")).lower() for m in thread)
        last_cust = str(thread[-1].get("text", "")).lower() if thread else ""
        agent_asked = ("?" in str(thread[1].get("text", ""))) if len(thread) > 1 else False
        has_escalation_kw = any(
            w in full_text
            for w in (
                "cancel",
                "switch",
                "competitor",
                "lawyer",
                "unacceptable",
                "wit's end",
                "third time",
                "still haven't",
                "down again",
                "stopped working",
                "urgent",
            )
        )
        floor_breach = tier in ("enterprise", "premium") and has_escalation_kw
        if floor_breach:
            return "escalate_to_human", True
        if has_escalation_kw:
            return "escalate_to_human", False
        if agent_asked and not any(c.isdigit() for c in last_cust):
            return "request_information", False
        if len(orders) >= 2 and any(str(o.get("status", "")) in ("failed", "pending") for o in orders):
            return "escalate_to_human", False
        if tier in ("enterprise", "premium"):
            return "escalate_to_human", False
        return "request_information", False

    def build_blind_samples(self) -> Tuple[List[BlindTypedDecisionSample], Dict[str, Any]]:
        """Construct all 400 `BlindTypedDecisionSample`s with ZERO access to `gold` during construction."""
        from tradeoff_jev.datasets.corpus_v2_builder import SecOpsGoalSchemaFactory

        raw_rows = self._base_builder.load_raw_test_rows()
        goal_summary = "; ".join(
            f"{g.id} ({g.name}: {g.description})" for g in SecOpsGoalSchemaFactory.build()
        )

        samples: List[BlindTypedDecisionSample] = []
        for idx, row in enumerate(raw_rows):
            wf = str(row["workflow"])
            row_id = str(row["id"])
            state_str = str(row["state"])
            state_obj = json.loads(state_str)
            questions = json.loads(row["questions"])

            exec_q = (
                "action"
                if wf in ("customer_service", "agent_trace_observability")
                else "disposition"
            )
            criteria_map: Dict[str, str] = questions[exec_q]["criteria"]
            # Canonical alphabetical schema order — NEVER sorted by gold probabilities!
            option_keys = tuple(sorted(criteria_map.keys()))
            cand_descs = tuple(f"{k}: {criteria_map[k]}" for k in option_keys)

            # Strictly raw state JSON + goal definitions matching TypedDecisionsTrainConverter
            state_dict = {
                "domain": "enterprise_ops",
                "subdomain": f"typed_decisions_{wf}",
                "workflow": wf,
                "decision_question": exec_q,
                "typed_state": state_obj,
            }
            state_text = (
                f"Domain State: {json.dumps(state_dict, ensure_ascii=True, sort_keys=True)} "
                f"| Goals: {goal_summary}"
            )

            # Fixed default organizational PreferenceTable per workflow matching TypedDecisionsTrainConverter
            asp_prev = (3.0, 3.0, 3.0, 2.0)
            s_floor = (2.0, 2.0, 2.0, 1.0)
            u_ranks = (0.0, 1.0, 2.0, 3.0)
            retreat_idx = 3

            # Compute observable-state heuristic strictly from `state_obj`
            heur_choice, heur_breach = self._observable_state_heuristic(wf, state_obj)

            # Now—and ONLY for ground-truth scoring—read gold[exec_q]
            gold = json.loads(row["gold"])
            gold_label = str(gold[exec_q]["label"])
            gold_conf = float(gold[exec_q].get("confidence", 0.5))
            target_idx = option_keys.index(gold_label)
            target_noul = tuple(
                float(gold[exec_q]["probabilities"].get(k, 0.0)) for k in option_keys
            )

            step_sample = DecisionStepSample(
                sample_id=f"blind_td_{row_id}",
                episode_id=f"ep_blind_td_{idx:04d}",
                step=0,
                state_text=state_text,
                candidate_descriptions=cand_descs,
                aspiration_prev=asp_prev,
                survival_floor=s_floor,
                urgency_ranks=u_ranks,
                retreat_index=retreat_idx,
                target_choice=target_idx,
                target_noul=target_noul,
                target_aspiration=asp_prev,
                target_bartleby=0.0,
                target_rubric_levels=None,
            )
            samples.append(
                BlindTypedDecisionSample(
                    row_id=row_id,
                    workflow=wf,
                    step_sample=step_sample,
                    option_keys=option_keys,
                    gold_label=gold_label,
                    gold_confidence=gold_conf,
                    observable_heuristic_choice=heur_choice,
                    observable_floor_breach=heur_breach,
                )
            )

        # Compute observable-state heuristic accuracy across all 400 & per workflow
        heur_correct = sum(1 for s in samples if s.observable_heuristic_choice == s.gold_label)
        by_wf_heur: Dict[str, float] = {}
        for wf in ("customer_service", "invoice_processing", "security_incidents", "agent_trace_observability"):
            sub = [s for s in samples if s.workflow == wf]
            by_wf_heur[wf] = round(
                sum(1 for s in sub if s.observable_heuristic_choice == s.gold_label) / max(1, len(sub)),
                6,
            )
        high_conf_sub = [s for s in samples if s.gold_confidence < 0.35]
        heur_high_conf = round(
            sum(1 for s in high_conf_sub if s.observable_heuristic_choice == s.gold_label)
            / max(1, len(high_conf_sub)),
            6,
        )

        stats = {
            "total_test_cases": len(samples),
            "high_conflict_cases": len(high_conf_sub),
            "candidates_alphabetical_schema_order": True,
            "include_oracle_scores_in_text": False,
            "gold_accessed_during_construction": False,
            "observable_state_aat_heuristic": {
                "overall_accuracy": round(heur_correct / max(1, len(samples)), 6),
                "high_conflict_accuracy": heur_high_conf,
                "by_workflow_accuracy": by_wf_heur,
            },
        }
        return samples, stats

    @torch.no_grad()
    def evaluate_neural_blind(
        self,
        evaluator: NeuralSystemEvaluator,
        samples: Sequence[BlindTypedDecisionSample],
    ) -> Dict[str, Any]:
        """Evaluate a loaded `NeuralSystemEvaluator` on strictly blind alphabetical `typed-decisions`."""
        system = evaluator._system  # noqa: SLF001
        trainer = evaluator._trainer  # noqa: SLF001
        if system is None or trainer is None:
            raise RuntimeError("Evaluator must be loaded before calling evaluate_neural_blind.")

        # Group by workflow so candidate count K (5 for customer_service, 4 for others) is uniform per batch
        correct_all = 0
        paralyzed_all = 0
        brier_all: List[float] = []
        max_nouls_all: List[float] = []
        by_wf_correct: Dict[str, List[int]] = {}
        by_wf_paralyzed: Dict[str, List[int]] = {}
        high_conflict_correct: List[int] = []
        high_conflict_paralyzed: List[int] = []

        for s in samples:
            batch = trainer.collate_samples([s.step_sample])
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=evaluator._use_bf16 and evaluator._device.type == "cuda",  # noqa: SLF001
            ):
                outputs = system(
                    packed_batch=batch.packed_batch,
                    aspiration_prev=batch.aspiration_prev,
                    survival_floor=batch.survival_floor,
                    urgency_ranks=batch.urgency_ranks,
                    retreat_index=batch.retreat_index,
                )

            k_opts = len(s.option_keys)
            # Restrict choice argmax to the K schema options (excluding slot K Bartleby for closed-set gold label comparison)
            opt_logits = outputs["choice_logits"].float()[0, :k_opts]
            pred_idx = int(torch.argmax(opt_logits).item())
            is_ok = int(pred_idx == s.step_sample.target_choice)

            noul_vec = outputs["tradeoff_noul"].float()[0, :k_opts]
            target_vec = batch.target_noul[0, :k_opts]
            max_n = float(noul_vec.max().item())
            is_par = int(PARALYSIS_LOW <= max_n <= PARALYSIS_HIGH)
            brier = float(torch.mean((noul_vec - target_vec) ** 2).item())

            correct_all += is_ok
            paralyzed_all += is_par
            brier_all.append(brier)
            max_nouls_all.append(max_n)
            by_wf_correct.setdefault(s.workflow, []).append(is_ok)
            by_wf_paralyzed.setdefault(s.workflow, []).append(is_par)
            if s.gold_confidence < 0.35:
                high_conflict_correct.append(is_ok)
                high_conflict_paralyzed.append(is_par)

        n = max(1, len(samples))
        return {
            "num_samples": len(samples),
            "blind_choice_accuracy": round(correct_all / n, 6),
            "blind_paralysis_rate": round(paralyzed_all / n, 6),
            "blind_noul_brier": round(sum(brier_all) / n, 6),
            "blind_mean_max_noul": round(sum(max_nouls_all) / n, 6),
            "high_conflict_choice_accuracy": round(
                sum(high_conflict_correct) / max(1, len(high_conflict_correct)), 6
            ),
            "high_conflict_paralysis_rate": round(
                sum(high_conflict_paralyzed) / max(1, len(high_conflict_paralyzed)), 6
            ),
            "by_workflow": {
                wf: {
                    "choice_accuracy": round(sum(by_wf_correct[wf]) / len(by_wf_correct[wf]), 6),
                    "paralysis_rate": round(sum(by_wf_paralyzed[wf]) / len(by_wf_paralyzed[wf]), 6),
                }
                for wf in by_wf_correct
            },
        }


def _evaluate_corpus_metas(
    evaluator: NeuralSystemEvaluator,
    metas: Sequence[EvalSampleMeta],
    aggregator: MetricsAggregator,
) -> Dict[str, Any]:
    acc = evaluator.evaluate(metas)
    return aggregator.full_report(acc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Zero-Bleeding / Blind Perception Scientific Audit across corpus_v1, MoralChoice, and typed-decisions."
    )
    parser.add_argument(
        "--phase2-ckpt",
        type=str,
        default="checkpoints/corpus_v1_tradeoff/phase2/phase2_final",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default="checkpoints/corpus_v1_baseline/phase1/phase1_final",
    )
    parser.add_argument(
        "--blind-tradeoff-dir",
        type=str,
        default="checkpoints/corpus_v1_tradeoff_blind",
    )
    parser.add_argument(
        "--blind-baseline-dir",
        type=str,
        default="checkpoints/corpus_v1_baseline_blind",
    )
    parser.add_argument(
        "--blind-phase1-steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--blind-phase2-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="logs/eval_zero_bleeding_audit.json",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate existing checkpoints directly without running blind adaptation training.",
    )
    parser.add_argument(
        "--bartleby-ckpt",
        type=str,
        default="artifacts/bartleby-1.2",
        help="Primary Bartleby-1.2 checkpoint evaluated when --eval-only is passed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if (not str(args.device).startswith("cuda") or torch.cuda.is_available())
        else "cpu"
    )
    print(f"[ZeroBleedingAudit] Running on device={device}", flush=True)
    aggregator = MetricsAggregator()

    # -------------------------------------------------------------------------
    # STEP 1: Blind Perception Adaptation (skipped when --eval-only is set)
    # -------------------------------------------------------------------------
    if args.eval_only:
        blind_tradeoff_ckpt = Path(args.bartleby_ckpt)
        blind_baseline_ckpt = Path(args.baseline_ckpt)
        blind_tradeoff_train_summary = {"skipped": True, "checkpoint": str(blind_tradeoff_ckpt)}
        blind_baseline_train_summary = {"skipped": True, "checkpoint": str(blind_baseline_ckpt)}
    else:
        adapt_trainer = BlindPerceptionAdaptationTrainer(device=device)
        blind_tradeoff_ckpt, blind_tradeoff_train_summary = adapt_trainer.run_blind_adaptation(
            system_kind="tradeoff",
            warm_start_ckpt=Path(args.phase2_ckpt),
            output_dir=Path(args.blind_tradeoff_dir),
            phase1_steps=args.blind_phase1_steps,
            phase2_steps=args.blind_phase2_steps,
        )
        blind_baseline_ckpt, blind_baseline_train_summary = adapt_trainer.run_blind_adaptation(
            system_kind="baseline",
            warm_start_ckpt=Path(args.baseline_ckpt),
            output_dir=Path(args.blind_baseline_dir),
            phase1_steps=args.blind_phase1_steps + args.blind_phase2_steps,
            phase2_steps=0,
        )

    # -------------------------------------------------------------------------
    # STEP 2: Build Test A (corpus_v1/test.jsonl), Test B (Blind Shuffled MoralChoice), Test C (Blind Typed-Decisions)
    # -------------------------------------------------------------------------
    print("\n[ZeroBleedingAudit] Loading Test A (`corpus_v1/test.jsonl` with include_oracle_scores_in_text=False vs True)...", flush=True)
    loader_oracle_text = BlindCorpusV1Loader(num_goals=4, include_oracle_scores_in_text=True)
    loader_blind_text = BlindCorpusV1Loader(num_goals=4, include_oracle_scores_in_text=False)
    test_a_oracle_metas = loader_oracle_text.load_eval_metas("data/corpus_v1/test.jsonl")
    test_a_blind_metas = loader_blind_text.load_eval_metas("data/corpus_v1/test.jsonl")

    print("[ZeroBleedingAudit] Building Test B (`MoralChoice` Shuffled + Raw Strings + include_oracle_scores_in_text=False)...", flush=True)
    mc_blind_builder = ZeroBleedingMoralChoiceBuilder(seed=42)
    mc_low_deont_blind, mc_low_all_blind, mc_high_blind, mc_blind_stats = (
        mc_blind_builder.build_all_blind()
    )
    print(f"[ZeroBleedingAudit] Test B stats: {mc_blind_stats}", flush=True)

    print("[ZeroBleedingAudit] Building Test C (`LocalLLaMA/typed-decisions` Alphabetical Schema Order + Zero Gold Access)...", flush=True)
    td_blind_eval = ZeroBleedingTypedDecisionsEvaluator()
    td_blind_samples, td_blind_stats = td_blind_eval.build_blind_samples()
    print(f"[ZeroBleedingAudit] Test C stats: {td_blind_stats}", flush=True)

    # Also build Counterfactual Policy-Switch pairs with `include_oracle_scores_in_text=False`
    # to verify whether counterfactual PreferenceTable conditioning requires oracle scores in text!
    from scripts.eval_canonical_benchmarks import MoralChoiceCanonicalBenchmarkBuilder
    mc_cf_builder = MoralChoiceCanonicalBenchmarkBuilder()
    mc_cf_builder._converter = TrajectoryRecordConverter(num_goals=4, include_oracle_scores_in_text=False)
    _, _, _, mc_cf_pairs_blind, _ = mc_cf_builder.build_all()

    td_cf_builder = TypedDecisionsCanonicalBenchmarkBuilder()
    td_cf_builder._converter = TrajectoryRecordConverter(num_goals=4, include_oracle_scores_in_text=False)
    _, _, _, td_cf_pairs_blind, _ = td_cf_builder.build_all()

    # -------------------------------------------------------------------------
    # STEP 3: Evaluate All Checkpoints Across Tests A, B, C
    # -------------------------------------------------------------------------
    from scripts.eval_canonical_benchmarks import CanonicalBenchmarkRunner
    cf_runner = CanonicalBenchmarkRunner(device=device, batch_size=args.batch_size)

    if args.eval_only:
        model_specs = [
            ("bartleby_1_2", "tradeoff", blind_tradeoff_ckpt),
            ("standard_jev_baseline", "baseline", blind_baseline_ckpt),
        ]
    else:
        model_specs = [
            ("tradeoff_jev_phase2_zero_shot", "tradeoff", Path(args.phase2_ckpt)),
            ("tradeoff_jev_blind_adapted", "tradeoff", blind_tradeoff_ckpt),
            ("standard_jev_baseline_zero_shot", "baseline", Path(args.baseline_ckpt)),
            ("standard_jev_baseline_blind_adapted", "baseline", blind_baseline_ckpt),
        ]

    audit_results: Dict[str, Any] = {
        "audit_metadata": {
            "description": (
                "Rigorous Zero-Bleeding / Blind Perception Audit isolating Oracle-AssistedAAT "
                "readout performance from Strictly Blind text-only perception across corpus_v1, "
                "MoralChoice, and LocalLLaMA/typed-decisions."
            ),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "blind_adaptation_training": {
                "tradeoff_jev_blind": blind_tradeoff_train_summary,
                "standard_jev_baseline_blind": blind_baseline_train_summary,
            },
            "test_b_moralchoice_blind_config": mc_blind_stats,
            "test_c_typed_decisions_blind_config": td_blind_stats,
        },
        "models": {},
    }

    for label, kind, ckpt_path in model_specs:
        print(f"\n[ZeroBleedingAudit] Evaluating {label} ({ckpt_path}) ...", flush=True)
        ev = NeuralSystemEvaluator(
            system_name=kind,
            checkpoint_dir=ckpt_path,
            batch_size=args.batch_size,
            use_bf16=(device.type == "cuda"),
            device=device,
        )
        ev.load()

        # Test A: corpus_v1/test.jsonl (Oracle-Assisted vs Blind Text)
        test_a_with_scores = _evaluate_corpus_metas(ev, test_a_oracle_metas, aggregator)
        test_a_blind = _evaluate_corpus_metas(ev, test_a_blind_metas, aggregator)

        # Test B: MoralChoice Blind + Shuffled + Raw Strings
        mc_low_deont_res = cf_runner.evaluate_samples(ev, mc_low_deont_blind)
        mc_low_all_res = cf_runner.evaluate_samples(ev, mc_low_all_blind)
        mc_high_res = cf_runner.evaluate_samples(ev, mc_high_blind)
        mc_all_res = cf_runner.evaluate_samples(ev, mc_low_all_blind + mc_high_blind)
        mc_cf_blind_res = cf_runner.evaluate_counterfactual_pairs(ev, mc_cf_pairs_blind)

        # Test C: LocalLLaMA/typed-decisions Strictly Blind (Alphabetical Schema Order, Zero Gold Access)
        td_blind_res = td_blind_eval.evaluate_neural_blind(ev, td_blind_samples)
        td_cf_blind_res = cf_runner.evaluate_counterfactual_pairs(ev, td_cf_pairs_blind)

        audit_results["models"][label] = {
            "checkpoint": str(ckpt_path),
            "test_a_corpus_v1": {
                "oracle_scores_in_text_true": test_a_with_scores,
                "blind_oracle_scores_in_text_false": test_a_blind,
            },
            "test_b_moralchoice_strict_blind_shuffled": {
                "all_1352": mc_all_res,
                "low_ambiguity_deontological_370": mc_low_deont_res,
                "low_ambiguity_all_672": mc_low_all_res,
                "high_ambiguity_680": mc_high_res,
                "counterfactual_pairs_120_blind_text": mc_cf_blind_res,
            },
            "test_c_typed_decisions_strict_blind": {
                "neural_zero_shot_alphabetical_400": td_blind_res,
                "observable_state_aat_heuristic_400": td_blind_stats["observable_state_aat_heuristic"],
                "counterfactual_pairs_160_blind_text": td_cf_blind_res,
            },
        }

        print(
            f"  [{label}] Test A (corpus_v1/test.jsonl): "
            f"WithScores Acc={test_a_with_scores['overall']['choice_accuracy']*100:.2f}% -> "
            f"Blind Acc={test_a_blind['overall']['choice_accuracy']*100:.2f}% | "
            f"Blind Bartleby Recall={(test_a_blind['overall']['bartleby_recall'] or 0.0)*100:.2f}% | "
            f"Blind Paralysis={test_a_blind['overall']['paralysis_rate']*100:.2f}%",
            flush=True,
        )
        print(
            f"  [{label}] Test B (MoralChoice Blind+Shuffled+Raw): "
            f"Low-Deont Acc={mc_low_deont_res['overall']['choice_accuracy']*100:.2f}% | "
            f"Low-All Acc={mc_low_all_res['overall']['choice_accuracy']*100:.2f}% | "
            f"High Acc={mc_high_res['overall']['choice_accuracy']*100:.2f}% | "
            f"High Bartleby Recall={(mc_high_res['overall']['bartleby_recall'] or 0.0)*100:.2f}% | "
            f"CF Pair Acc (Blind Text)={mc_cf_blind_res['pair_both_correct_rate']*100:.2f}%",
            flush=True,
        )
        print(
            f"  [{label}] Test C (Typed-Decisions Blind Alphabetical): "
            f"Neural Zero-Shot Acc={td_blind_res['blind_choice_accuracy']*100:.2f}% | "
            f"Observable-State AAT Heuristic Acc={td_blind_stats['observable_state_aat_heuristic']['overall_accuracy']*100:.2f}% | "
            f"Neural Paralysis={td_blind_res['blind_paralysis_rate']*100:.2f}% | "
            f"CF Pair Acc (Blind Text)={td_cf_blind_res['pair_both_correct_rate']*100:.2f}%",
            flush=True,
        )

        ev.unload()
        del ev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    print(f"\n[ZeroBleedingAudit] Saved complete machine-readable audit to {out_path}", flush=True)


if __name__ == "__main__":
    main()
