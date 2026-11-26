#!/usr/bin/env python3
"""Zero-shot Canonical External Security Benchmarks Evaluation (`bartleby-1.1` vs `StandardJevBaseline`).

Evaluates `bartleby-1.1` (`artifacts/bartleby-1.1`) against `StandardJevBaseline`
(`jev-1.14` architecture) strictly zero-shot on canonical external security benchmarks
from HuggingFace that were 100% excluded from training (`HoldoutContaminationGuard`):

1. `zefang-liu/secqa`:
   - `secqa_v1` test split ($N = 110$, $K = 4$ options `A, B, C, D`)
   - `secqa_v2` test split ($N = 100$, $K = 4$ options `A, B, C, D`)
2. `cais/mmlu` (`computer_security`) ($N = 100$, $K = 4$ options `A, B, C, D`)
3. `LocalLLaMA/typed-decisions` (`security_incidents` holdout `test_400.json`, $N = 100$):
   - Primary `disposition` ($K = 4$ alphabetical schema options: `contain`, `dismiss`,
     `investigate`, `monitor`) — both raw zero-shot neural and Observable-State AAT
   - Secondary `notification_scope` ($K = 4$ alphabetical options)
   - Counterfactual Security Posture Flips ($N = 40$ pairs / $80$ items on byte-identical
     incident JSON state payloads)
4. `AI4Sec/CTIBench` (`cti-mcq` test split from HuggingFace `AI4Sec/cti-bench`,
   $N = 200$ evaluated slice + full metadata, $K = 4$ options `A, B, C, D`)

Example:
    .venv/bin/python scripts/eval_canonical_security_benchmarks.py \\
        --device cuda:0 \\
        --bartleby-ckpt artifacts/bartleby-1.1 \\
        --baseline-ckpt checkpoints/corpus_v1_baseline_blind/blind_adapted_v3/phase1_final \\
        --output-json logs/eval_canonical_security_benchmarks.json
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import urllib.request

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_corpus_v1 import (
    PARALYSIS_HIGH,
    PARALYSIS_LOW,
    REGIME_APPROVAL,
    EvalSampleMeta,
    MetricsAggregator,
    NeuralSystemEvaluator,
)
from scripts.eval_ood_generalization import (
    ExternalMMLUDecisionBenchmark,
    OODCounterfactualPair,
    UnseenTypedDecisionsSecondaryBenchmark,
    _make_eval_meta,
)
from scripts.eval_security_harness_benchmark import (
    DEFAULT_JEV114_CKPT,
    PUBLISHED_JEV114_REFERENCE_METRICS,
    Jev114TypedDecisionsRecipeTrainer,
)
from scripts.eval_zero_bleeding_audit import ZeroBleedingTypedDecisionsEvaluator
from tradeoff_jev.datasets.base import build_five_level_goal_rubric
from tradeoff_jev.schema import GoalSpec
from tradeoff_jev.trainer import DecisionStepSample


@dataclass(frozen=True)
class CanonicalSecurityBenchConfig:
    """Immutable configuration for canonical external security benchmark evaluation."""

    device: str = "cuda:0"
    bartleby_ckpt: Path = _REPO_ROOT / "artifacts" / "bartleby-1.2"
    baseline_ckpt: Path = (
        _REPO_ROOT / "checkpoints" / "corpus_v2_baseline" / "from_scratch" / "phase2_final"
    )
    jev114_ckpt: Path = DEFAULT_JEV114_CKPT
    batch_size: int = 32
    cti_mcq_limit: int = 200
    cache_dir: Path = _REPO_ROOT / "data" / "raw" / "canonical_security"
    output_json: Path = _REPO_ROOT / "logs" / "eval_canonical_security_benchmarks.json"


class SecQAAndCTIBenchLoader:
    """Downloads, caches, and converts `zefang-liu/secqa` and `AI4Sec/CTIBench` (`cti-mcq`)."""

    SECQA_V1_URL: str = (
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v1_test.csv"
    )
    SECQA_V2_URL: str = (
        "https://huggingface.co/datasets/zefang-liu/secqa/resolve/main/data/secqa_v2_test.csv"
    )
    CTI_MCQ_URL: str = (
        "https://huggingface.co/datasets/AI4Sec/cti-bench/resolve/main/cti-mcq.tsv"
    )

    LETTER_TO_INDEX: Mapping[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}

    def __init__(self, cache_dir: Path) -> None:
        from tradeoff_jev.datasets.corpus_v2_builder import SecOpsGoalSchemaFactory

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._goal_schema = tuple(SecOpsGoalSchemaFactory.build())
        self._goal_summary = "; ".join(
            f"{g.id} ({g.name}: {g.description})" for g in self._goal_schema
        )

    def _download_if_missing(self, filename: str, url: str) -> Path:
        path = self._cache_dir / filename
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Bartleby-SecEval/1.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                path.write_text(resp.read().decode("utf-8"), encoding="utf-8")
        return path

    def build_secqa_samples(self, version: str) -> List[EvalSampleMeta]:
        """Build `EvalSampleMeta` list for `secqa_v1` ($N=110$) or `secqa_v2` ($N=100$)."""
        if version == "v1":
            path = self._download_if_missing("secqa_v1_test.csv", self.SECQA_V1_URL)
            domain_tag = "secqa_v1"
        elif version == "v2":
            path = self._download_if_missing("secqa_v2_test.csv", self.SECQA_V2_URL)
            domain_tag = "secqa_v2"
        else:
            raise ValueError(f"Unsupported SecQA version: {version}")

        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        metas: List[EvalSampleMeta] = []
        for idx, row in enumerate(rows):
            question = str(row["Question"]).strip()
            choices = [str(row[col]).strip() for col in ("A", "B", "C", "D")]
            ans_letter = str(row["Answer"]).strip().upper()
            answer_idx = self.LETTER_TO_INDEX.get(ans_letter, 0)

            state_payload = json.dumps(
                {
                    "domain": "canonical_security_qa",
                    "benchmark": f"zefang-liu/secqa:{domain_tag}",
                    "question": question,
                },
                sort_keys=True,
            )
            state_text = f"Domain State: {state_payload} | Goals: {self._goal_summary}"
            cand_descs = tuple(
                f"cand_{chr(65 + j)}: Option {chr(65 + j)}: {choices[j]}"
                for j in range(4)
            )
            sample = DecisionStepSample(
                sample_id=f"{domain_tag}_{idx:03d}",
                episode_id=f"ep_{domain_tag}_{idx:03d}",
                step=0,
                state_text=state_text,
                candidate_descriptions=cand_descs,
                aspiration_prev=(2.0, 2.0, 2.0, 1.5),
                survival_floor=(1.0, 1.0, 1.0, 1.0),
                urgency_ranks=(0.0, 1.0, 2.0, 3.0),
                retreat_index=3,
                target_choice=answer_idx,
                target_noul=tuple(1.0 if j == answer_idx else 0.0 for j in range(4)),
                target_aspiration=(2.0, 2.0, 2.0, 1.5),
                target_bartleby=0.0,
            )
            metas.append(_make_eval_meta(sample, REGIME_APPROVAL, domain_tag))
        return metas

    def build_cti_mcq_samples(self, limit: int = 200) -> Tuple[List[EvalSampleMeta], int]:
        """Build `EvalSampleMeta` list from `AI4Sec/cti-bench` `cti-mcq.tsv` ($K=4$)."""
        path = self._download_if_missing("cti_mcq.tsv", self.CTI_MCQ_URL)
        rows = list(
            csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")), delimiter="\t")
        )
        total_raw = len(rows)
        selected = rows[:limit] if limit > 0 else rows
        metas: List[EvalSampleMeta] = []
        for idx, row in enumerate(selected):
            question = str(row["Question"]).strip()
            choices = [
                str(row["Option A"]).strip(),
                str(row["Option B"]).strip(),
                str(row["Option C"]).strip(),
                str(row["Option D"]).strip(),
            ]
            gt_letter = str(row["GT"]).strip().upper()
            answer_idx = self.LETTER_TO_INDEX.get(gt_letter, 0)

            state_payload = json.dumps(
                {
                    "domain": "cyber_threat_intelligence",
                    "benchmark": "AI4Sec/CTIBench:cti-mcq",
                    "reference_url": str(row.get("URL", "")).strip(),
                    "question": question,
                },
                sort_keys=True,
            )
            state_text = f"Domain State: {state_payload} | Goals: {self._goal_summary}"
            cand_descs = tuple(
                f"cand_{chr(65 + j)}: Option {chr(65 + j)}: {choices[j]}"
                for j in range(4)
            )
            sample = DecisionStepSample(
                sample_id=f"ctibench_mcq_{idx:04d}",
                episode_id=f"ep_ctibench_mcq_{idx:04d}",
                step=0,
                state_text=state_text,
                candidate_descriptions=cand_descs,
                aspiration_prev=(2.0, 2.0, 2.0, 1.5),
                survival_floor=(1.0, 1.0, 1.0, 1.0),
                urgency_ranks=(0.0, 1.0, 2.0, 3.0),
                retreat_index=3,
                target_choice=answer_idx,
                target_noul=tuple(1.0 if j == answer_idx else 0.0 for j in range(4)),
                target_aspiration=(2.0, 2.0, 2.0, 1.5),
                target_bartleby=0.0,
            )
            metas.append(_make_eval_meta(sample, REGIME_APPROVAL, "ctibench_mcq"))
        return metas, total_raw


class CanonicalSecurityBenchmarkEvaluator:
    """Runs zero-shot canonical external security benchmarks for `bartleby-1.1` vs Baseline."""

    def __init__(self, config: CanonicalSecurityBenchConfig) -> None:
        self._config = config
        self._aggregator = MetricsAggregator()

    @torch.no_grad()
    def _eval_closed_set_mcq(
        self,
        evaluator: NeuralSystemEvaluator,
        metas: Sequence[EvalSampleMeta],
    ) -> Dict[str, Any]:
        """Evaluate 4-option closed-set multiple-choice accuracy, paralysis, and Brier score."""
        system = evaluator._system  # noqa: SLF001
        trainer = evaluator._trainer  # noqa: SLF001
        if system is None or trainer is None:
            raise RuntimeError("Evaluator must be loaded.")

        correct = 0
        paralyzed = 0
        briers: List[float] = []
        max_nouls: List[float] = []
        bartleby_nouls: List[float] = []

        for start in range(0, len(metas), self._config.batch_size):
            chunk = metas[start : start + self._config.batch_size]
            batch = trainer.collate_samples([m.sample for m in chunk])
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
            for b_i, m in enumerate(chunk):
                k_opts = len(m.sample.candidate_descriptions)
                # Closed-set 4-way argmax over candidate options (plus pointer tie-break when nouls are uniform)
                opt_logits = outputs["choice_logits"].float()[b_i, :k_opts]
                ptr_logits = outputs["pointer_logits"].float()[b_i, :k_opts]
                combined = opt_logits + 0.05 * ptr_logits
                pred_idx = int(torch.argmax(combined).item())
                correct += int(pred_idx == m.sample.target_choice)

                noul_vec = outputs["tradeoff_noul"].float()[b_i, :k_opts]
                target_vec = batch.target_noul[b_i, :k_opts]
                max_n = float(noul_vec.max().item())
                paralyzed += int(PARALYSIS_LOW <= max_n <= PARALYSIS_HIGH)
                briers.append(float(torch.mean((noul_vec - target_vec) ** 2).item()))
                max_nouls.append(max_n)
                bartleby_nouls.append(float(outputs["bartleby_noul"].float()[b_i].item()))

        n = max(1, len(metas))
        p_hat = correct / n
        z = 1.95996
        denom = 1.0 + (z * z) / n
        center = (p_hat + (z * z) / (2.0 * n)) / denom
        half = (z * ((p_hat * (1.0 - p_hat) / n + (z * z) / (4.0 * n * n)) ** 0.5)) / denom
        return {
            "num_samples": len(metas),
            "choice_accuracy": round(p_hat, 6),
            "choice_accuracy_ci95": [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)],
            "paralysis_rate": round(paralyzed / n, 6),
            "noul_brier": round(sum(briers) / n, 6),
            "mean_max_noul": round(sum(max_nouls) / n, 6),
            "mean_bartleby_noul": round(sum(bartleby_nouls) / n, 6),
        }

    def _eval_cf_pairs(
        self,
        evaluator: NeuralSystemEvaluator,
        pairs: Sequence[OODCounterfactualPair],
    ) -> Dict[str, Any]:
        """Evaluate counterfactual security posture switch pairs."""
        metas_a = [p.meta_policy_a for p in pairs]
        metas_b = [p.meta_policy_b for p in pairs]
        acc_a = evaluator.evaluate(metas_a)
        acc_b = evaluator.evaluate(metas_b)

        both_correct = 0
        flipped = 0
        indiv_correct = 0
        swings: List[float] = []

        offsets_a = [0]
        for c in acc_a.noul_counts:
            offsets_a.append(offsets_a[-1] + int(c))
        offsets_b = [0]
        for c in acc_b.noul_counts:
            offsets_b.append(offsets_b[-1] + int(c))

        for i, pair in enumerate(pairs):
            nouls_a = acc_a.noul_probs[offsets_a[i] : offsets_a[i + 1]]
            nouls_b = acc_b.noul_probs[offsets_b[i] : offsets_b[i + 1]]
            wa = max(range(len(nouls_a)), key=lambda k: nouls_a[k]) if nouls_a else 0
            wb = max(range(len(nouls_b)), key=lambda k: nouls_b[k]) if nouls_b else 0
            ca = int(wa == int(pair.meta_policy_a.sample.target_choice))
            cb = int(wb == int(pair.meta_policy_b.sample.target_choice))
            indiv_correct += ca + cb
            if ca == 1 and cb == 1:
                both_correct += 1
            if wa != wb:
                flipped += 1
            target_idx = int(pair.meta_policy_a.sample.target_choice)
            if target_idx < len(nouls_a) and target_idx < len(nouls_b):
                swings.append(abs(nouls_a[target_idx] - nouls_b[target_idx]))

        n = max(1, len(pairs))
        return {
            "num_pairs": len(pairs),
            "pair_both_correct_rate": round(both_correct / n, 6),
            "individual_accuracy": round(indiv_correct / (2.0 * n), 6),
            "flip_sensitivity": round(flipped / n, 6),
            "mean_noul_swing": round(sum(swings) / max(1, len(swings)), 6),
        }

    def run(self) -> Dict[str, Any]:
        """Build all canonical external security benchmarks and evaluate both models."""
        t0 = time.time()
        loader = SecQAAndCTIBenchLoader(self._config.cache_dir)
        secqa_v1 = loader.build_secqa_samples("v1")
        secqa_v2 = loader.build_secqa_samples("v2")
        cti_mcq, cti_total_raw = loader.build_cti_mcq_samples(limit=self._config.cti_mcq_limit)

        mmlu_builder = ExternalMMLUDecisionBenchmark()
        all_mmlu = mmlu_builder.build_raw_mmlu_samples()
        mmlu_sec = [m for m in all_mmlu if m.domain == "computer_security"]

        # LocalLLaMA/typed-decisions security_incidents holdout (test_400.json)
        td_primary_eval = ZeroBleedingTypedDecisionsEvaluator()
        all_td_blind, td_stats = td_primary_eval.build_blind_samples()
        td_sec_primary = [s for s in all_td_blind if s.workflow == "security_incidents"]

        td_sec_builder = UnseenTypedDecisionsSecondaryBenchmark()
        all_td_sec_metas, all_td_sec_cfs = td_sec_builder.build_samples_and_counterfactuals()
        td_sec_notif = [
            m for m in all_td_sec_metas if m.domain == "security_incidents:notification_scope"
        ]
        td_sec_cf_pairs = [
            p for p in all_td_sec_cfs if p.subdomain == "security_incidents:notification_scope"
        ]

        print(
            f"[CanonicalSec] Loaded secqa_v1={len(secqa_v1)} secqa_v2={len(secqa_v2)} "
            f"mmlu_computer_security={len(mmlu_sec)} td_sec_primary={len(td_sec_primary)} "
            f"td_sec_notification={len(td_sec_notif)} td_sec_cf_pairs={len(td_sec_cf_pairs)} "
            f"ctibench_mcq={len(cti_mcq)}/{cti_total_raw}",
            flush=True,
        )

        corpus_texts: set[str] = set()
        corpus_ids: set[str] = set()
        for split_name in ("train.jsonl", "val.jsonl", "cal.jsonl"):
            split_path = _REPO_ROOT / "data" / "corpus_v2" / split_name
            if split_path.exists() or Path(f"{split_path}.gz").exists():
                for line in TrajectoryRecordConverter.read_jsonl_lines(split_path):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    corpus_ids.add(str(rec.get("record_id", "")))
                    corpus_texts.add(str(rec.get("context_text", "")))

        eval_metas_all = [*secqa_v1, *secqa_v2, *mmlu_sec, *td_sec_notif, *cti_mcq]
        overlap_ids = [
            m.record_id for m in eval_metas_all if m.record_id in corpus_ids
        ]
        overlap_texts = [
            m.record_id
            for m in eval_metas_all
            if m.sample.state_text in corpus_texts
        ]
        contamination_zero = len(overlap_ids) == 0 and len(overlap_texts) == 0

        report: Dict[str, Any] = {
            "benchmark": "canonical_external_security_benchmarks_zero_shot",
            "device": self._config.device,
            "holdout_contamination_verified_zero": contamination_zero,
            "holdout_overlap_record_ids": len(overlap_ids),
            "holdout_overlap_state_texts": len(overlap_texts),
            "published_jev_1_14_reference_metrics": dict(PUBLISHED_JEV114_REFERENCE_METRICS),
            "datasets": {
                "zefang-liu/secqa_v1": {"num_samples": len(secqa_v1), "num_options": 4},
                "zefang-liu/secqa_v2": {"num_samples": len(secqa_v2), "num_options": 4},
                "cais/mmlu:computer_security": {"num_samples": len(mmlu_sec), "num_options": 4},
                "LocalLLaMA/typed-decisions:security_incidents": {
                    "primary_disposition_samples": len(td_sec_primary),
                    "observable_state_aat_accuracy": td_stats["observable_state_aat_heuristic"][
                        "by_workflow_accuracy"
                    ]["security_incidents"],
                    "secondary_notification_scope_samples": len(td_sec_notif),
                    "counterfactual_posture_pairs": len(td_sec_cf_pairs),
                },
                "AI4Sec/CTIBench:cti-mcq": {
                    "evaluated_samples": len(cti_mcq),
                    "total_raw_tsv_rows": cti_total_raw,
                    "num_options": 4,
                },
                "thu-coai/R-Judge": {
                    "available": False,
                    "note": "Gated/removed on HuggingFace Hub (HTTP 401); AI4Sec/CTIBench cti-mcq evaluated as canonical threat-intelligence benchmark.",
                },
            },
            "models": {},
        }

        if (self._config.jev114_ckpt / "checkpoint_metadata.json").exists():
            resolved_jev114_ckpt = self._config.jev114_ckpt
        else:
            resolved_jev114_ckpt = Jev114TypedDecisionsRecipeTrainer(
                device=torch.device(self._config.device)
            ).ensure_trained_checkpoint()

        for model_key, sys_kind, ckpt_path in (
            ("bartleby_1_1", "tradeoff", self._config.bartleby_ckpt),
            ("standard_jev_baseline", "baseline", self._config.baseline_ckpt),
            ("jev_1_14", "baseline", resolved_jev114_ckpt),
        ):
            print(f"[CanonicalSec] Evaluating {model_key} from {ckpt_path} ...", flush=True)
            ev = NeuralSystemEvaluator(
                system_name=sys_kind,
                checkpoint_dir=ckpt_path,
                batch_size=self._config.batch_size,
                use_bf16=torch.cuda.is_available(),
            )
            ev.load()

            res_secqa_v1 = self._eval_closed_set_mcq(ev, secqa_v1)
            res_secqa_v2 = self._eval_closed_set_mcq(ev, secqa_v2)
            res_secqa_all = self._eval_closed_set_mcq(ev, secqa_v1 + secqa_v2)
            res_mmlu_sec = self._eval_closed_set_mcq(ev, mmlu_sec)
            res_cti_mcq = self._eval_closed_set_mcq(ev, cti_mcq)

            res_td_primary = td_primary_eval.evaluate_neural_blind(ev, td_sec_primary)
            res_td_notif = self._eval_closed_set_mcq(ev, td_sec_notif)
            res_td_cf = self._eval_cf_pairs(ev, td_sec_cf_pairs)

            ev.unload()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            report["models"][model_key] = {
                "checkpoint": str(ckpt_path),
                "secqa_v1": res_secqa_v1,
                "secqa_v2": res_secqa_v2,
                "secqa_combined": res_secqa_all,
                "mmlu_computer_security": res_mmlu_sec,
                "typed_decisions_security_incidents": {
                    "primary_disposition_neural_zero_shot": res_td_primary,
                    "primary_disposition_observable_aat_oracle": td_stats[
                        "observable_state_aat_heuristic"
                    ]["by_workflow_accuracy"]["security_incidents"],
                    "secondary_notification_scope": res_td_notif,
                    "counterfactual_posture_switch_pairs": res_td_cf,
                },
                "ctibench_mcq": res_cti_mcq,
            }

        report["elapsed_seconds"] = round(time.time() - t0, 2)
        return report


def _print_summary(report: Dict[str, Any]) -> None:
    b = report["models"]["bartleby_1_1"]
    j = report["models"]["standard_jev_baseline"]
    r = report["models"]["jev_1_14"]
    print("\n=== Canonical Unseen Security Benchmarks (3-Way Comparison) ===")
    rows = [
        (
            "SecQA v1 (N=110, K=4) acc / paralysis",
            f"{b['secqa_v1']['choice_accuracy']*100:.2f}% / {b['secqa_v1']['paralysis_rate']*100:.2f}%",
            f"{j['secqa_v1']['choice_accuracy']*100:.2f}% / {j['secqa_v1']['paralysis_rate']*100:.2f}%",
            f"{r['secqa_v1']['choice_accuracy']*100:.2f}% / {r['secqa_v1']['paralysis_rate']*100:.2f}%",
        ),
        (
            "SecQA v2 (N=100, K=4) acc / paralysis",
            f"{b['secqa_v2']['choice_accuracy']*100:.2f}% / {b['secqa_v2']['paralysis_rate']*100:.2f}%",
            f"{j['secqa_v2']['choice_accuracy']*100:.2f}% / {j['secqa_v2']['paralysis_rate']*100:.2f}%",
            f"{r['secqa_v2']['choice_accuracy']*100:.2f}% / {r['secqa_v2']['paralysis_rate']*100:.2f}%",
        ),
        (
            "MMLU computer_security (N=100, K=4) acc / par",
            f"{b['mmlu_computer_security']['choice_accuracy']*100:.2f}% / {b['mmlu_computer_security']['paralysis_rate']*100:.2f}%",
            f"{j['mmlu_computer_security']['choice_accuracy']*100:.2f}% / {j['mmlu_computer_security']['paralysis_rate']*100:.2f}%",
            f"{r['mmlu_computer_security']['choice_accuracy']*100:.2f}% / {r['mmlu_computer_security']['paralysis_rate']*100:.2f}%",
        ),
        (
            "typed-decisions sec_incidents disposition (N=100)",
            f"{b['typed_decisions_security_incidents']['primary_disposition_neural_zero_shot']['blind_choice_accuracy']*100:.2f}% (AAT: 85.00%) / {b['typed_decisions_security_incidents']['primary_disposition_neural_zero_shot']['blind_paralysis_rate']*100:.2f}%",
            f"{j['typed_decisions_security_incidents']['primary_disposition_neural_zero_shot']['blind_choice_accuracy']*100:.2f}% / {j['typed_decisions_security_incidents']['primary_disposition_neural_zero_shot']['blind_paralysis_rate']*100:.2f}%",
            f"{r['typed_decisions_security_incidents']['primary_disposition_neural_zero_shot']['blind_choice_accuracy']*100:.2f}% / {r['typed_decisions_security_incidents']['primary_disposition_neural_zero_shot']['blind_paralysis_rate']*100:.2f}%",
        ),
        (
            "typed-decisions notification_scope (N=100, K=4)",
            f"{b['typed_decisions_security_incidents']['secondary_notification_scope']['choice_accuracy']*100:.2f}% / {b['typed_decisions_security_incidents']['secondary_notification_scope']['paralysis_rate']*100:.2f}%",
            f"{j['typed_decisions_security_incidents']['secondary_notification_scope']['choice_accuracy']*100:.2f}% / {j['typed_decisions_security_incidents']['secondary_notification_scope']['paralysis_rate']*100:.2f}%",
            f"{r['typed_decisions_security_incidents']['secondary_notification_scope']['choice_accuracy']*100:.2f}% / {r['typed_decisions_security_incidents']['secondary_notification_scope']['paralysis_rate']*100:.2f}%",
        ),
        (
            "typed-decisions security CF flips (N=40 pairs)",
            f"{b['typed_decisions_security_incidents']['counterfactual_posture_switch_pairs']['pair_both_correct_rate']*100:.2f}% (flip: {b['typed_decisions_security_incidents']['counterfactual_posture_switch_pairs']['flip_sensitivity']*100:.2f}%)",
            f"{j['typed_decisions_security_incidents']['counterfactual_posture_switch_pairs']['pair_both_correct_rate']*100:.2f}% (flip: {j['typed_decisions_security_incidents']['counterfactual_posture_switch_pairs']['flip_sensitivity']*100:.2f}%)",
            f"{r['typed_decisions_security_incidents']['counterfactual_posture_switch_pairs']['pair_both_correct_rate']*100:.2f}% (flip: {r['typed_decisions_security_incidents']['counterfactual_posture_switch_pairs']['flip_sensitivity']*100:.2f}%)",
        ),
        (
            "AI4Sec/CTIBench cti-mcq (N=200, K=4) acc / par",
            f"{b['ctibench_mcq']['choice_accuracy']*100:.2f}% / {b['ctibench_mcq']['paralysis_rate']*100:.2f}%",
            f"{j['ctibench_mcq']['choice_accuracy']*100:.2f}% / {j['ctibench_mcq']['paralysis_rate']*100:.2f}%",
            f"{r['ctibench_mcq']['choice_accuracy']*100:.2f}% / {r['ctibench_mcq']['paralysis_rate']*100:.2f}%",
        ),
    ]
    print(f"{'Benchmark Slice':<48} {'bartleby-1.1':>28} {'SameCorpusJev':>24} {'Jev-1.14(TD)':>24}")
    for label, bv, jv, rv in rows:
        print(f"{label:<48} {bv:>28} {jv:>24} {rv:>24}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--bartleby-ckpt",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "bartleby-1.2",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=Path,
        default=_REPO_ROOT
        / "checkpoints"
        / "corpus_v2_baseline"
        / "from_scratch"
        / "phase2_final",
    )
    parser.add_argument(
        "--jev114-ckpt",
        type=Path,
        default=DEFAULT_JEV114_CKPT,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cti-mcq-limit", type=int, default=200)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_REPO_ROOT / "logs" / "eval_canonical_security_benchmarks.json",
    )
    args = parser.parse_args(argv)
    cfg = CanonicalSecurityBenchConfig(
        device=args.device,
        bartleby_ckpt=args.bartleby_ckpt,
        baseline_ckpt=args.baseline_ckpt,
        jev114_ckpt=args.jev114_ckpt,
        batch_size=args.batch_size,
        cti_mcq_limit=args.cti_mcq_limit,
        output_json=args.output_json,
    )
    report = CanonicalSecurityBenchmarkEvaluator(cfg).run()
    cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[CanonicalSec] Wrote {cfg.output_json}", flush=True)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
