"""Multi-Step Trajectory Stability, Counterfactual Flip, and Jev-1.14 Evaluation (`corpus_v1`).

Theoretical & Architectural Purpose:
    Evaluates the Phase 2 Markov Rollout model (`TradeOffJevNeuralSystem` trained with
    scheduled sampling on $\\hat{\\vec{a}}_{t-1}$) alongside the Phase 1 teacher-forced
    model, the unconditioned `StandardJevBaselineSystem` ablation, and `typesafe/jev-1.14`
    on the held-out group-disjoint test split (`data/corpus_v1/test.jsonl`, $N = 600$,
    including 120 Out-of-Distribution records in `EnergyGridDispatchOOD` and
    `ClinicalProtocolOOD`).

    Computes four evaluation pillars:
    1. **Single-Step Metrics**: Overall, per-regime (`approval`, `epistemic_uncertainty`,
       `bartleby_refusal`), per-domain (`asimov`, `trolley`, `ops`), and OOD-specific
       choice accuracy, paralysis rate ($\\max_k P(\\text{yes} \\mid x_t, u_k, \\mathcal{T}_t)
       \\in [0.40, 0.58]$), Bartleby safety recall, Brier score, and 15-bin ECE.
    2. **Autoregressive Self-Rollout Markov Chain Metrics ($t = 0 \\to t = 1 \\to t = 2$)**:
       Groups `test.jsonl` by `episode_id` into multi-step Markov trajectories ($T \\ge 2$)
       and feeds the model's *own* predicted aspiration vector $\\hat{\\vec{a}}_{t-1}$ into
       step $t$ without oracle resets. Reports per-step self-rollout choice accuracy,
       exact full-trajectory choice agreement, per-step L1 aspiration drift
       $\\frac{1}{M}\\|\\hat{\\vec{a}}_t - \\vec{a}_t^*\\|_1$ (and breakdown at $t=0, 1, 2$),
       aspiration MSE, and exact rounded discrete table match rate.
    3. **Counterfactual Preference-Table Flip Accuracy**:
       Evaluates all paired counterfactual scenarios (`counterfactual_pair_id != None` in
       `test.jsonl`) where the surface text $x_t$ and candidate propositions are 100%
       identical, but the incoming Preference Table $\\mathcal{T}_{t-1}^{(A)} \\neq
       \\mathcal{T}_{t-1}^{(B)}$ flips the optimal choice or Bartleby refusal state.
       Measures both pair-wise exact accuracy (both siblings correct) and counterfactual
       sensitivity (prediction flips when the Preference Table changes).
    4. **TypeSafe Jev 1.14 (`typesafe/jev-1.14`) Benchmark & Live/Emulated Comparison**:
       Checks `OPENROUTER_API_KEY` in the environment, executes `Jev114BenchmarkRunner`
       across Suites 1–4, and computes oracle-aligned Jev-1.14 paralysis and counterfactual
       insensitivity diagnostics on `test.jsonl`.

Example:
    .venv/bin/python scripts/eval_phase2_trajectories.py \\
        --test-jsonl data/corpus_v1/test.jsonl \\
        --phase2-ckpt checkpoints/corpus_v1_tradeoff/phase2/phase2_final \\
        --phase1-ckpt checkpoints/corpus_v1_tradeoff/phase1/phase1_final \\
        --baseline-ckpt checkpoints/corpus_v1_baseline/phase1/phase1_final \\
        --output-json logs/eval_phase2_markov_report.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_corpus_v1 import (
    BARTLEBY_THRESHOLD,
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
    Jev114OracleAlignedDiagnostics,
    MetricsAggregator,
    NeuralSystemEvaluator,
)
from tradeoff_jev.backbone import DEFAULT_QWEN_CHECKPOINT
from tradeoff_jev.benchmark_jev114 import Jev114BenchmarkRunner, OpenRouterJev114Client
from tradeoff_jev.trainer import (
    CollatedStepBatch,
    DecisionStepSample,
    TrajectoryRecordConverter,
)


@dataclass(frozen=True)
class RichEvalRecord:
    """Immutable container enriching `EvalSampleMeta` with trajectory and counterfactual keys.

    Attributes:
        meta: Standard `EvalSampleMeta` for single-step slice aggregation.
        episode_id: Episode identifier grouping multi-step Markov chains ($t=0, 1, 2$).
        step: Zero-based Markov step index $t$.
        counterfactual_pair_id: Optional counterfactual sibling pair identifier.
        is_ood: Whether the record belongs to a held-out OOD domain (`energy_grid_dispatch_ood`
            or `clinical_protocol_ood`).
        domain_name: Exact fine-grained corpus domain name.
        aat_transition: Ground-truth symbolic AAT transition label.
        preset_name: Preference Table preset identifier.
    """

    meta: EvalSampleMeta
    episode_id: str
    step: int
    counterfactual_pair_id: Optional[str]
    is_ood: bool
    domain_name: str
    aat_transition: str
    preset_name: str


@dataclass(frozen=True)
class SelfRolloutTrajectoryReport:
    """Immutable metrics from multi-step autoregressive self-rollout ($t=0 \\to t=1 \\to t=2$).

    Attributes:
        num_episodes: Number of multi-step test episodes ($T \\ge 2$).
        num_steps: Total decision steps evaluated across all multi-step episodes.
        teacher_forced_step_accuracy: Per-step choice accuracy when feeding oracle $\\vec{a}_{t-1}^*$.
        teacher_forced_chain_agreement: Fraction of episodes where all steps are correct
            under teacher-forced oracle $\\vec{a}_{t-1}^*$.
        self_rollout_step_accuracy: Per-step choice accuracy when feeding the model's own
            predicted $\\hat{\\vec{a}}_{t-1}$ from $t=0 \\to t=1 \\to t=2$ without oracle resets.
        self_rollout_chain_agreement: Fraction of episodes where every step $t=0, 1, 2$
            is simultaneously correct under self-rollout.
        step_accuracy_by_t: Mapping from step index (`"t=0"`, `"t=1"`, `"t=2"`) to
            self-rollout choice accuracy.
        aspiration_l1_drift_overall: Mean per-goal L1 aspiration error
            $\\frac{1}{M}\\|\\hat{\\vec{a}}_t - \\vec{a}_t^*\\|_1$ across self-rollout steps.
        aspiration_l1_drift_by_t: Mean per-goal L1 aspiration drift at `"t=0"`, `"t=1"`, `"t=2"`.
        aspiration_mse_overall: Mean per-goal squared error across self-rollout steps.
        discrete_table_exact_match_rate: Fraction of steps where `round(hat_a_t)` matches
            the integer oracle aspiration vector $\\vec{a}_t^*$ on all $M$ goals.
        self_rollout_paralysis_rate: Fraction of self-rollout steps falling in $[0.40, 0.58]$.
        self_rollout_bartleby_recall: Recall on Bartleby refusal steps (`t=2` floor breaches)
            during self-rollout.
    """

    num_episodes: int
    num_steps: int
    teacher_forced_step_accuracy: float
    teacher_forced_chain_agreement: float
    self_rollout_step_accuracy: float
    self_rollout_chain_agreement: float
    step_accuracy_by_t: Mapping[str, float]
    aspiration_l1_drift_overall: float
    aspiration_l1_drift_by_t: Mapping[str, float]
    aspiration_mse_overall: float
    discrete_table_exact_match_rate: float
    self_rollout_paralysis_rate: float
    self_rollout_bartleby_recall: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trajectory report to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualFlipReport:
    """Immutable metrics quantifying counterfactual PreferenceTable flip accuracy.

    Attributes:
        num_pairs: Number of paired counterfactual scenarios in `test.jsonl`.
        num_records: Total records belonging to counterfactual pairs ($2 \\times \\text{num\\_pairs}$).
        individual_record_accuracy: Single-record choice accuracy across counterfactual items.
        pair_both_correct_rate: Fraction of pairs where the model selects the exact oracle
            winner for *both* PreferenceTable $\\mathcal{T}^{(A)}$ and $\\mathcal{T}^{(B)}$.
        counterfactual_flip_sensitivity: Fraction of pairs (with distinct oracle winners)
            where changing only $\\mathcal{T}_t$ causes the model to flip its selected action.
        mean_noul_flip_swing: Mean absolute change $|\\hat{p}_A(u^*) - \\hat{p}_B(u^*)|$ in
            conditioned `tradeoff_noul` on the target candidate when $\\mathcal{T}_t$ flips.
        by_domain_pair_accuracy: Pair-wise both-correct accuracy stratified by coarse domain.
    """

    num_pairs: int
    num_records: int
    individual_record_accuracy: float
    pair_both_correct_rate: float
    counterfactual_flip_sensitivity: float
    mean_noul_flip_swing: float
    by_domain_pair_accuracy: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize counterfactual flip report to a JSON-compatible dictionary."""
        return asdict(self)


class RichCorpusTestLoader:
    """Loads `data/corpus_v1/test.jsonl` into `RichEvalRecord`s, chains, and CF pairs."""

    def __init__(self, num_goals: int = 4) -> None:
        self._converter = TrajectoryRecordConverter(
            num_goals=num_goals,
            include_oracle_scores_in_text=False,
            shuffle_and_anonymize_candidates=False,
        )
        self._bucketizer = DomainBucketizer()

    def load(self, jsonl_path: str | Path) -> List[RichEvalRecord]:
        """Parse `test.jsonl` into `RichEvalRecord` instances."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Test JSONL not found: {path}")

        records: List[RichEvalRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                sample: DecisionStepSample = self._converter.convert_record(obj)
                regime = str(obj.get("regime", "")).strip().lower()
                domain_name = str(obj.get("domain_name", ""))
                source = str(obj.get("source", ""))
                coarse_domain = self._bucketizer.bucketize(domain_name, source)
                meta = EvalSampleMeta(
                    sample=sample,
                    regime=regime,
                    domain=coarse_domain,
                    record_id=str(obj.get("record_id", sample.sample_id)),
                    unconditioned_noul={
                        str(k): float(v)
                        for k, v in (obj.get("unconditioned_noul") or {}).items()
                    },
                    tradeoff_noul_targets={
                        str(k): float(v)
                        for k, v in (obj.get("tradeoff_noul") or {}).items()
                    },
                    bartleby_triggered=bool(obj.get("bartleby_triggered", False)),
                )
                cf_id_raw = obj.get("counterfactual_pair_id")
                records.append(
                    RichEvalRecord(
                        meta=meta,
                        episode_id=str(obj.get("episode_id", sample.episode_id)),
                        step=int(obj.get("step", sample.step)),
                        counterfactual_pair_id=str(cf_id_raw) if cf_id_raw else None,
                        is_ood=bool(obj.get("is_ood", False)),
                        domain_name=domain_name,
                        aat_transition=str(obj.get("aat_transition", "")),
                        preset_name=str(obj.get("preset_name", "")),
                    )
                )
        return records

    @staticmethod
    def group_multi_step_episodes(
        records: Sequence[RichEvalRecord],
    ) -> List[List[RichEvalRecord]]:
        """Group records by `episode_id` and return sorted multi-step chains ($T \\ge 2$)."""
        by_ep: Dict[str, List[RichEvalRecord]] = defaultdict(list)
        for rec in records:
            by_ep[rec.episode_id].append(rec)
        chains = [
            sorted(steps, key=lambda r: r.step)
            for steps in by_ep.values()
            if len(steps) >= 2
        ]
        return sorted(chains, key=lambda c: c[0].episode_id)

    @staticmethod
    def group_counterfactual_pairs(
        records: Sequence[RichEvalRecord],
    ) -> List[Tuple[RichEvalRecord, RichEvalRecord]]:
        """Group records sharing a `counterfactual_pair_id` into ordered pairs `(A, B)`."""
        by_cf: Dict[str, List[RichEvalRecord]] = defaultdict(list)
        for rec in records:
            if rec.counterfactual_pair_id is not None:
                by_cf[rec.counterfactual_pair_id].append(rec)
        pairs: List[Tuple[RichEvalRecord, RichEvalRecord]] = []
        for cf_id in sorted(by_cf.keys()):
            items = sorted(by_cf[cf_id], key=lambda r: r.meta.record_id)
            if len(items) == 2:
                pairs.append((items[0], items[1]))
        return pairs


class TrajectoryAndCounterfactualAnalyzer:
    """Evaluates multi-step self-rollout stability and counterfactual flip accuracy."""

    def __init__(self, use_bf16: bool = True) -> None:
        self._use_bf16 = use_bf16

    @torch.no_grad()
    def evaluate_multi_step_chains(
        self,
        evaluator: NeuralSystemEvaluator,
        chains: Sequence[Sequence[RichEvalRecord]],
    ) -> SelfRolloutTrajectoryReport:
        """Run both teacher-forced and autoregressive self-rollout on multi-step test chains."""
        system = evaluator._system  # noqa: SLF001
        trainer = evaluator._trainer  # noqa: SLF001
        if system is None or trainer is None:
            raise RuntimeError("NeuralSystemEvaluator must be loaded before chain evaluation.")

        system.eval()
        device = evaluator._device  # noqa: SLF001
        amp_enabled = self._use_bf16 and device.type == "cuda"

        num_episodes = len(chains)
        total_steps = 0
        tf_step_correct = 0
        tf_chain_correct = 0
        sr_step_correct = 0
        sr_chain_correct = 0

        step_correct_by_t: Dict[int, int] = defaultdict(int)
        step_count_by_t: Dict[int, int] = defaultdict(int)
        l1_drift_by_t: Dict[int, float] = defaultdict(float)

        l1_drift_total = 0.0
        mse_drift_total = 0.0
        exact_table_matches = 0
        sr_paralyzed = 0
        sr_bartleby_gt = 0
        sr_bartleby_hits = 0

        for episode in chains:
            tf_all_ok = True
            sr_all_ok = True
            carried_aspiration: Optional[torch.Tensor] = None

            for t_idx, rec in enumerate(episode):
                batch: CollatedStepBatch = trainer.collate_samples([rec.meta.sample])

                # 1. Teacher-forced forward pass (using oracle aspiration_prev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    tf_out = system(
                        packed_batch=batch.packed_batch,
                        aspiration_prev=batch.aspiration_prev,
                        survival_floor=batch.survival_floor,
                        urgency_ranks=batch.urgency_ranks,
                        retreat_index=batch.retreat_index,
                    )
                tf_pred = int(torch.argmax(tf_out["choice_logits"].float(), dim=-1).item())
                target_choice = int(batch.target_choice.item())
                if tf_pred == target_choice:
                    tf_step_correct += 1
                else:
                    tf_all_ok = False

                # 2. Autoregressive self-rollout forward pass (feeding hat_a_{t-1} for t >= 1)
                aspiration_in = (
                    carried_aspiration
                    if (t_idx > 0 and carried_aspiration is not None)
                    else batch.aspiration_prev
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    sr_out = system(
                        packed_batch=batch.packed_batch,
                        aspiration_prev=aspiration_in,
                        survival_floor=batch.survival_floor,
                        urgency_ranks=batch.urgency_ranks,
                        retreat_index=batch.retreat_index,
                    )

                sr_logits = sr_out["choice_logits"].float()
                sr_pred = int(torch.argmax(sr_logits, dim=-1).item())
                step_count_by_t[t_idx] += 1
                if sr_pred == target_choice:
                    sr_step_correct += 1
                    step_correct_by_t[t_idx] += 1
                else:
                    sr_all_ok = False

                cand_mask = batch.packed_batch.candidate_mask.bool()
                k_valid = int(cand_mask[0].sum().item())
                sr_noul_vec = sr_out["tradeoff_noul"].float()[0, :k_valid]
                max_noul = float(sr_noul_vec.max().item()) if k_valid > 0 else 0.0
                if PARALYSIS_LOW <= max_noul <= PARALYSIS_HIGH:
                    sr_paralyzed += 1

                is_bart_gt = bool(rec.meta.bartleby_triggered) or (
                    rec.meta.regime == REGIME_BARTLEBY
                )
                if is_bart_gt:
                    sr_bartleby_gt += 1
                    bart_prob = float(sr_out["bartleby_noul"].float()[0].item())
                    if sr_pred == k_valid or bart_prob >= BARTLEBY_THRESHOLD:
                        sr_bartleby_hits += 1

                target_asp = batch.target_aspiration.float()
                if "updated_aspiration" in sr_out:
                    pred_asp = sr_out["updated_aspiration"].float()
                    carried_aspiration = pred_asp.detach()
                else:
                    # Unconditioned baseline emits no updated_aspiration; carry prior unchanged
                    pred_asp = aspiration_in.float()
                    carried_aspiration = pred_asp.detach()

                step_l1 = float(torch.mean(torch.abs(pred_asp - target_asp)).item())
                step_mse = float(F.mse_loss(pred_asp, target_asp).item())
                l1_drift_total += step_l1
                mse_drift_total += step_mse
                l1_drift_by_t[t_idx] += step_l1

                if torch.all(torch.round(pred_asp) == torch.round(target_asp)):
                    exact_table_matches += 1

                total_steps += 1

            if tf_all_ok:
                tf_chain_correct += 1
            if sr_all_ok:
                sr_chain_correct += 1

        n_ep = max(1, num_episodes)
        n_st = max(1, total_steps)
        return SelfRolloutTrajectoryReport(
            num_episodes=num_episodes,
            num_steps=total_steps,
            teacher_forced_step_accuracy=round(tf_step_correct / n_st, 6),
            teacher_forced_chain_agreement=round(tf_chain_correct / n_ep, 6),
            self_rollout_step_accuracy=round(sr_step_correct / n_st, 6),
            self_rollout_chain_agreement=round(sr_chain_correct / n_ep, 6),
            step_accuracy_by_t={
                f"t={t}": round(step_correct_by_t[t] / max(1, step_count_by_t[t]), 6)
                for t in sorted(step_count_by_t.keys())
            },
            aspiration_l1_drift_overall=round(l1_drift_total / n_st, 6),
            aspiration_l1_drift_by_t={
                f"t={t}": round(l1_drift_by_t[t] / max(1, step_count_by_t[t]), 6)
                for t in sorted(step_count_by_t.keys())
            },
            aspiration_mse_overall=round(mse_drift_total / n_st, 6),
            discrete_table_exact_match_rate=round(exact_table_matches / n_st, 6),
            self_rollout_paralysis_rate=round(sr_paralyzed / n_st, 6),
            self_rollout_bartleby_recall=(
                round(sr_bartleby_hits / sr_bartleby_gt, 6) if sr_bartleby_gt > 0 else None
            ),
        )

    @torch.no_grad()
    def evaluate_counterfactual_pairs(
        self,
        evaluator: NeuralSystemEvaluator,
        pairs: Sequence[Tuple[RichEvalRecord, RichEvalRecord]],
    ) -> CounterfactualFlipReport:
        """Evaluate counterfactual sensitivity across identical-text PreferenceTable pairs."""
        system = evaluator._system  # noqa: SLF001
        trainer = evaluator._trainer  # noqa: SLF001
        if system is None or trainer is None:
            raise RuntimeError("NeuralSystemEvaluator must be loaded before CF evaluation.")

        system.eval()
        device = evaluator._device  # noqa: SLF001
        amp_enabled = self._use_bf16 and device.type == "cuda"

        individual_correct = 0
        both_correct = 0
        distinct_target_pairs = 0
        flipped_when_required = 0
        noul_swing_sum = 0.0

        domain_pair_totals: Dict[str, int] = defaultdict(int)
        domain_pair_hits: Dict[str, int] = defaultdict(int)

        for rec_a, rec_b in pairs:
            batch_a = trainer.collate_samples([rec_a.meta.sample])
            batch_b = trainer.collate_samples([rec_b.meta.sample])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                out_a = system(
                    packed_batch=batch_a.packed_batch,
                    aspiration_prev=batch_a.aspiration_prev,
                    survival_floor=batch_a.survival_floor,
                    urgency_ranks=batch_a.urgency_ranks,
                    retreat_index=batch_a.retreat_index,
                )
                out_b = system(
                    packed_batch=batch_b.packed_batch,
                    aspiration_prev=batch_b.aspiration_prev,
                    survival_floor=batch_b.survival_floor,
                    urgency_ranks=batch_b.urgency_ranks,
                    retreat_index=batch_b.retreat_index,
                )

            pred_a = int(torch.argmax(out_a["choice_logits"].float(), dim=-1).item())
            pred_b = int(torch.argmax(out_b["choice_logits"].float(), dim=-1).item())
            targ_a = int(batch_a.target_choice.item())
            targ_b = int(batch_b.target_choice.item())

            ok_a = pred_a == targ_a
            ok_b = pred_b == targ_b
            if ok_a:
                individual_correct += 1
            if ok_b:
                individual_correct += 1

            domain = rec_a.meta.domain
            domain_pair_totals[domain] += 1
            if ok_a and ok_b:
                both_correct += 1
                domain_pair_hits[domain] += 1

            if targ_a != targ_b:
                distinct_target_pairs += 1
                if pred_a != pred_b:
                    flipped_when_required += 1

            noul_a = out_a["tradeoff_noul"].float()[0]
            noul_b = out_b["tradeoff_noul"].float()[0]
            k_valid = min(int(noul_a.numel()), int(noul_b.numel()))
            if k_valid > 0:
                swing = float(torch.max(torch.abs(noul_a[:k_valid] - noul_b[:k_valid])).item())
                noul_swing_sum += swing

        n_pairs = max(1, len(pairs))
        n_recs = max(1, 2 * len(pairs))
        return CounterfactualFlipReport(
            num_pairs=len(pairs),
            num_records=2 * len(pairs),
            individual_record_accuracy=round(individual_correct / n_recs, 6),
            pair_both_correct_rate=round(both_correct / n_pairs, 6),
            counterfactual_flip_sensitivity=round(
                flipped_when_required / max(1, distinct_target_pairs), 6
            ),
            mean_noul_flip_swing=round(noul_swing_sum / n_pairs, 6),
            by_domain_pair_accuracy={
                dom: round(domain_pair_hits[dom] / max(1, total), 6)
                for dom, total in sorted(domain_pair_totals.items())
            },
        )


@dataclass(frozen=True)
class Phase2TrajectoryEvalConfig:
    """Configuration for `scripts/eval_phase2_trajectories.py`."""

    test_jsonl: str = "data/corpus_v1/test.jsonl"
    phase2_ckpt: str = "checkpoints/corpus_v1_tradeoff/phase2/phase2_final"
    phase1_ckpt: str = "checkpoints/corpus_v1_tradeoff/phase1/phase1_final"
    baseline_ckpt: str = "checkpoints/corpus_v1_baseline/phase1/phase1_final"
    output_json: str = "logs/eval_phase2_markov_report.json"
    model_name_or_path: str = DEFAULT_QWEN_CHECKPOINT
    batch_size: int = 1
    max_seq_length: int = 768
    lora_r: int = 16
    lora_alpha: int = 32
    use_bf16: bool = True
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


class Phase2ComprehensiveEvaluator:
    """Runs Phase 2 vs. Phase 1 vs. Standard Jev Baseline vs. TypeSafe Jev 1.14 evaluation."""

    def __init__(self, config: Phase2TrajectoryEvalConfig) -> None:
        self._config = config
        self._loader = RichCorpusTestLoader(num_goals=4)
        self._aggregator = MetricsAggregator()
        self._analyzer = TrajectoryAndCounterfactualAnalyzer(use_bf16=config.use_bf16)
        self._jev114_diag = Jev114OracleAlignedDiagnostics()

    def _evaluate_checkpoint(
        self,
        label: str,
        system_name: str,
        checkpoint_dir: str,
        records: Sequence[RichEvalRecord],
        chains: Sequence[Sequence[RichEvalRecord]],
        cf_pairs: Sequence[Tuple[RichEvalRecord, RichEvalRecord]],
    ) -> Dict[str, Any]:
        """Evaluate a neural checkpoint across single-step, OOD, self-rollout, and CF metrics."""
        print(f"[eval-phase2] loading {label} ({system_name}) from {checkpoint_dir}", flush=True)
        evaluator = NeuralSystemEvaluator(
            system_name=system_name,
            checkpoint_dir=checkpoint_dir,
            model_name_or_path=self._config.model_name_or_path,
            max_seq_length=self._config.max_seq_length,
            lora_r=self._config.lora_r,
            lora_alpha=self._config.lora_alpha,
            batch_size=self._config.batch_size,
            use_bf16=self._config.use_bf16,
            device=self._config.device,
        )
        evaluator.load()

        metas = [r.meta for r in records]
        in_dist_metas = [r.meta for r in records if not r.is_ood]
        ood_metas = [r.meta for r in records if r.is_ood]

        acc_all = evaluator.evaluate(metas)
        acc_in_dist = evaluator.evaluate(in_dist_metas)
        acc_ood = evaluator.evaluate(ood_metas)

        single_step_report = self._aggregator.full_report(acc_all)
        single_step_report["in_distribution_overall"] = self._aggregator.slice_metrics(
            acc_in_dist
        ).to_dict()
        single_step_report["ood_overall"] = self._aggregator.slice_metrics(acc_ood).to_dict()

        rollout_report = self._analyzer.evaluate_multi_step_chains(evaluator, chains)
        cf_report = self._analyzer.evaluate_counterfactual_pairs(evaluator, cf_pairs)

        evaluator.unload()
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "label": label,
            "system_name": system_name,
            "checkpoint_dir": checkpoint_dir,
            "single_step_metrics": single_step_report,
            "multi_step_self_rollout": rollout_report.to_dict(),
            "counterfactual_flip_metrics": cf_report.to_dict(),
        }

    def _evaluate_jev114(
        self,
        records: Sequence[RichEvalRecord],
        cf_pairs: Sequence[Tuple[RichEvalRecord, RichEvalRecord]],
    ) -> Dict[str, Any]:
        """Run `Jev114BenchmarkRunner` (live OpenRouter if key set else analytical) + corpus stats."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        has_live_key = bool(api_key and api_key.strip())
        client = OpenRouterJev114Client(
            model_slug="typesafe/jev-1.14",
            live_api=has_live_key,
            api_key=api_key,
        )
        runner = Jev114BenchmarkRunner(client=client)
        suite_result = runner.run_all()

        metas = [r.meta for r in records]
        corpus_paralysis = self._jev114_diag.compute(metas)

        # Jev-1.14 is unconditioned on PreferenceTable T_t, so its output on identical text
        # across counterfactual pairs is invariant (0.0 flip sensitivity).
        cf_uncond_flips = 0
        for rec_a, rec_b in cf_pairs:
            u_a = rec_a.meta.unconditioned_noul
            u_b = rec_b.meta.unconditioned_noul
            win_a = max(u_a.items(), key=lambda kv: kv[1])[0] if u_a else ""
            win_b = max(u_b.items(), key=lambda kv: kv[1])[0] if u_b else ""
            if win_a != win_b:
                cf_uncond_flips += 1

        return {
            "model_slug": "typesafe/jev-1.14",
            "openrouter_api_key_present": has_live_key,
            "evaluation_mode": "live_openrouter_with_analytical_fallback"
            if has_live_key
            else "analytical_emulator",
            "benchmark_suites_1_to_4": {
                "jev114_paralysis_rate": suite_result.jev114_paralysis_rate,
                "tradeoff_jev_symbolic_accuracy": suite_result.tradeoff_jev_accuracy,
                "bartleby_safety_recall": suite_result.bartleby_safety_recall,
                "summary_by_suite": dict(suite_result.summary_by_suite),
            },
            "corpus_v1_test_unconditioned": corpus_paralysis,
            "counterfactual_flip_sensitivity": round(
                cf_uncond_flips / max(1, len(cf_pairs)), 6
            ),
        }

    def run(self) -> Dict[str, Any]:
        """Execute full Phase 2 trajectory & counterfactual evaluation and save JSON report."""
        records = self._loader.load(self._config.test_jsonl)
        chains = self._loader.group_multi_step_episodes(records)
        cf_pairs = self._loader.group_counterfactual_pairs(records)

        print(
            f"[eval-phase2] test_records={len(records)} "
            f"multi_step_chains={len(chains)} ({sum(len(c) for c in chains)} steps) "
            f"counterfactual_pairs={len(cf_pairs)} "
            f"ood_records={sum(1 for r in records if r.is_ood)}",
            flush=True,
        )

        phase2_block = self._evaluate_checkpoint(
            label="tradeoff_jev_phase2_markov",
            system_name="tradeoff",
            checkpoint_dir=self._config.phase2_ckpt,
            records=records,
            chains=chains,
            cf_pairs=cf_pairs,
        )
        phase1_block = self._evaluate_checkpoint(
            label="tradeoff_jev_phase1_single_step",
            system_name="tradeoff",
            checkpoint_dir=self._config.phase1_ckpt,
            records=records,
            chains=chains,
            cf_pairs=cf_pairs,
        )
        baseline_block = self._evaluate_checkpoint(
            label="standard_jev_baseline",
            system_name="baseline",
            checkpoint_dir=self._config.baseline_ckpt,
            records=records,
            chains=chains,
            cf_pairs=cf_pairs,
        )
        jev114_block = self._evaluate_jev114(records, cf_pairs)

        p2_sr = phase2_block["multi_step_self_rollout"]
        p1_sr = phase1_block["multi_step_self_rollout"]
        bs_sr = baseline_block["multi_step_self_rollout"]

        p2_ov = phase2_block["single_step_metrics"]["overall"]
        p1_ov = phase1_block["single_step_metrics"]["overall"]
        bs_ov = baseline_block["single_step_metrics"]["overall"]

        p2_cf = phase2_block["counterfactual_flip_metrics"]
        p1_cf = phase1_block["counterfactual_flip_metrics"]
        bs_cf = baseline_block["counterfactual_flip_metrics"]

        report: Dict[str, Any] = {
            "status": "completed",
            "config": asdict(self._config),
            "test_split_summary": {
                "num_records": len(records),
                "num_in_distribution_records": sum(1 for r in records if not r.is_ood),
                "num_ood_records": sum(1 for r in records if r.is_ood),
                "num_multi_step_episodes": len(chains),
                "num_multi_step_records": sum(len(c) for c in chains),
                "num_counterfactual_pairs": len(cf_pairs),
            },
            "phase2_tradeoff_jev": phase2_block,
            "phase1_tradeoff_jev": phase1_block,
            "standard_jev_baseline": baseline_block,
            "typesafe_jev_1_14": jev114_block,
            "comparisons": {
                "phase2_minus_phase1": {
                    "single_step_choice_accuracy": round(
                        p2_ov["choice_accuracy"] - p1_ov["choice_accuracy"], 6
                    ),
                    "single_step_paralysis_rate": round(
                        p2_ov["paralysis_rate"] - p1_ov["paralysis_rate"], 6
                    ),
                    "single_step_noul_brier": round(
                        p2_ov["noul_brier"] - p1_ov["noul_brier"], 6
                    ),
                    "self_rollout_step_accuracy": round(
                        p2_sr["self_rollout_step_accuracy"]
                        - p1_sr["self_rollout_step_accuracy"],
                        6,
                    ),
                    "self_rollout_chain_agreement": round(
                        p2_sr["self_rollout_chain_agreement"]
                        - p1_sr["self_rollout_chain_agreement"],
                        6,
                    ),
                    "aspiration_l1_drift_overall": round(
                        p2_sr["aspiration_l1_drift_overall"]
                        - p1_sr["aspiration_l1_drift_overall"],
                        6,
                    ),
                    "discrete_table_exact_match_rate": round(
                        p2_sr["discrete_table_exact_match_rate"]
                        - p1_sr["discrete_table_exact_match_rate"],
                        6,
                    ),
                    "counterfactual_pair_both_correct_rate": round(
                        p2_cf["pair_both_correct_rate"] - p1_cf["pair_both_correct_rate"],
                        6,
                    ),
                },
                "phase2_minus_baseline": {
                    "single_step_choice_accuracy": round(
                        p2_ov["choice_accuracy"] - bs_ov["choice_accuracy"], 6
                    ),
                    "single_step_paralysis_rate": round(
                        p2_ov["paralysis_rate"] - bs_ov["paralysis_rate"], 6
                    ),
                    "self_rollout_step_accuracy": round(
                        p2_sr["self_rollout_step_accuracy"]
                        - bs_sr["self_rollout_step_accuracy"],
                        6,
                    ),
                    "self_rollout_chain_agreement": round(
                        p2_sr["self_rollout_chain_agreement"]
                        - bs_sr["self_rollout_chain_agreement"],
                        6,
                    ),
                    "counterfactual_pair_both_correct_rate": round(
                        p2_cf["pair_both_correct_rate"] - bs_cf["pair_both_correct_rate"],
                        6,
                    ),
                    "counterfactual_flip_sensitivity": round(
                        p2_cf["counterfactual_flip_sensitivity"]
                        - bs_cf["counterfactual_flip_sensitivity"],
                        6,
                    ),
                },
            },
        }

        out_path = Path(self._config.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[eval-phase2] wrote comprehensive report -> {out_path}", flush=True)
        self._print_executive_summary(report)
        return report

    @staticmethod
    def _print_executive_summary(report: Mapping[str, Any]) -> None:
        """Print a structured console summary comparing Phase 2, Phase 1, Baseline, and Jev 1.14."""
        print("\n================ PHASE 2 MARKOV ROLLOUT EVALUATION ================", flush=True)
        for key in ("phase2_tradeoff_jev", "phase1_tradeoff_jev", "standard_jev_baseline"):
            blk = report[key]
            ov = blk["single_step_metrics"]["overall"]
            ood = blk["single_step_metrics"]["ood_overall"]
            sr = blk["multi_step_self_rollout"]
            cf = blk["counterfactual_flip_metrics"]
            print(
                f"[{blk['label']}]\n"
                f"  Single-Step : acc={ov['choice_accuracy']:.4f} | ood_acc={ood['choice_accuracy']:.4f} | "
                f"paralysis={ov['paralysis_rate']:.4f} | bartleby_rec={ov['bartleby_recall']} | "
                f"brier={ov['noul_brier']:.4f} | noul_ece={ov['noul_ece']:.4f}\n"
                f"  Self-Rollout: step_acc={sr['self_rollout_step_accuracy']:.4f} | "
                f"chain_agree={sr['self_rollout_chain_agreement']:.4f} | "
                f"L1_drift={sr['aspiration_l1_drift_overall']:.4f} | "
                f"exact_table={sr['discrete_table_exact_match_rate']:.4f}\n"
                f"  CF Flips    : pair_acc={cf['pair_both_correct_rate']:.4f} | "
                f"flip_sens={cf['counterfactual_flip_sensitivity']:.4f} | "
                f"noul_swing={cf['mean_noul_flip_swing']:.4f}",
                flush=True,
            )


def _parse_args(argv: Optional[Sequence[str]] = None) -> Phase2TrajectoryEvalConfig:
    """Parse CLI arguments for Phase 2 trajectory and counterfactual evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 2 Markov Rollout trajectories and counterfactual flips."
    )
    parser.add_argument("--test-jsonl", type=str, default="data/corpus_v1/test.jsonl")
    parser.add_argument(
        "--phase2-ckpt",
        type=str,
        default="artifacts/bartleby-1.2",
    )
    parser.add_argument(
        "--phase1-ckpt",
        type=str,
        default="checkpoints/bartleby_1_2/from_scratch/phase1_final",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default="checkpoints/corpus_v2_baseline/from_scratch/phase2_final",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="logs/eval_phase2_markov_report.json",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_QWEN_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)
    phase1_ckpt = args.phase1_ckpt
    if not Path(phase1_ckpt).exists():
        for fb in (
            "checkpoints/bartleby_1_1/from_scratch/phase1_final",
            "artifacts/bartleby-1.2",
        ):
            if Path(fb).exists():
                phase1_ckpt = fb
                break

    return Phase2TrajectoryEvalConfig(
        test_jsonl=args.test_jsonl,
        phase2_ckpt=args.phase2_ckpt,
        phase1_ckpt=phase1_ckpt,
        baseline_ckpt=args.baseline_ckpt,
        output_json=args.output_json,
        model_name_or_path=args.checkpoint,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_bf16=not args.no_bf16,
        device=args.device,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point returning process exit code."""
    config = _parse_args(argv)
    try:
        Phase2ComprehensiveEvaluator(config).run()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[eval-phase2] FAILED: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
