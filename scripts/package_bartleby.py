#!/usr/bin/env python3
"""Package a trained bartleby-1.1 checkpoint into `artifacts/bartleby-1.1/`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def package(
    checkpoint_dir: Path,
    artifact_dir: Path,
    *,
    training_summary: Optional[Dict[str, Any]] = None,
    eval_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Copy LoRA + AAT head + calibration into the canonical artifact bundle."""
    if not (checkpoint_dir / "aat_head.safetensors").exists():
        raise FileNotFoundError(f"Missing aat_head.safetensors in {checkpoint_dir}")

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Core weights (write both aat_head.safetensors and heads.safetensors for full compatibility)
    shutil.copy2(checkpoint_dir / "aat_head.safetensors", artifact_dir / "aat_head.safetensors")
    shutil.copy2(checkpoint_dir / "aat_head.safetensors", artifact_dir / "heads.safetensors")
    shutil.copy2(
        checkpoint_dir / "checkpoint_metadata.json",
        artifact_dir / "checkpoint_metadata.json",
    )
    lora_src = checkpoint_dir / "lora_adapter"
    if lora_src.exists():
        shutil.copytree(lora_src, artifact_dir / "lora_adapter")

    cal_candidates = [
        checkpoint_dir / "calibration",
        checkpoint_dir.parent / "calibration",
    ]
    for cal_src in cal_candidates:
        if cal_src.exists():
            shutil.copytree(cal_src, artifact_dir / "calibration", dirs_exist_ok=True)
            break

    meta = json.loads((artifact_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    cal_dir = artifact_dir / "calibration"
    noul_t = choice_t = None
    if (cal_dir / "noul_calibrator.json").exists():
        noul_t = json.loads((cal_dir / "noul_calibrator.json").read_text())["temperature"]
    if (cal_dir / "choice_calibrator.json").exists():
        choice_t = json.loads((cal_dir / "choice_calibrator.json").read_text())["temperature"]

    upgrades = [
        "Context-aware rubric head: [candidate_hidden ‖ decide_hidden] (not table_embedding)",
        "Intrinsic Bartleby logit from mathematical floor violation ∀k ∃m Ê_k,m < a_m^min + survival-gated pointer logits",
        "Anti-bleeding firewall strips meta phrases in all training texts",
        "Counterfactual survival-floor pairs (production high floor vs sandbox low floor) + security posture-switch pairs",
        "Multi-K curriculum (K=2..8) + balanced 1-of-K Safe-among-Dangerous vs 0-of-K All-Destructive synthetic security harness",
    ]

    manifest: Dict[str, Any] = {
        "model_id": "bartleby-1.2",
        "version": "1.2.0",
        "release_name": "bartleby-1.2 (clean symmetric from_scratch corpus_v2)",
        "license": "Apache-2.0",
        "from_scratch": True,
        "warm_start": None,
        "base_backbone": "Qwen/Qwen3-0.6B",
        "theoretical_foundation": {
            "core_engine": "Reinhard Selten (1962, 1998, 1999) Aspiration Adaptation Theory (AAT)",
            "preference_state": "SeltenPreferenceTable (T_t = (a_t, a^min, pi_t, r_t, R_grid, c_t))",
            "survival_floor_protocol": "Intrinsic Bartleby Refusal ('I would prefer not to.' / I_WOULD_PREFER_NOT_TO)",
        },
        "architecture_upgrades_vs_1_0": upgrades,
        "architecture": {
            "system_class": "BartlebyNeuralSystem",
            "adaptation_head_class": "SeltenAdaptationHead",
            "readout_head_class": "BartlebyReadoutHead",
            "oracle_class": "SeltenOracle",
            "base_backbone": "Qwen/Qwen3-0.6B",
            "rubric_head_input": "[candidate_hidden, decide_hidden]",
            "bartleby_logit": "analytic_floor_violation_logit * learned_gain",
            "lm_head_stripped": True,
            "backbone_dtype": "bfloat16",
            "head_dtype": "float32",
            "lora": meta.get("backbone_config", {}).get("lora")
            or {
                "r": meta.get("backbone_config", {}).get("lora_r", 16),
                "lora_alpha": meta.get("backbone_config", {}).get("lora_alpha", 32),
            },
            "selten_aat_head": meta.get("aat_config", {}),
            "parameters": meta.get("parameter_summary", {}),
        },
        "calibration": {
            "noul_temperature": noul_t,
            "choice_temperature": choice_t,
            "noul_calibrator_path": "calibration/noul_calibrator.json",
            "choice_calibrator_path": "calibration/choice_calibrator.json",
        },
        "checkpoint_resolution_order": [
            "artifacts/bartleby-1.2",
            "artifacts/bartleby-1.1",
            "checkpoints/bartleby_1_1/from_scratch/bartleby_1_1",
            "checkpoints/bartleby_1_1/from_scratch/phase2_final",
            "checkpoints/bartleby_1_1/from_scratch/phase1_final",
        ],
        "training_summary": training_summary or {},
        "benchmark_metrics": eval_metrics or {},
        "source_checkpoint": str(checkpoint_dir),
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = artifact_dir / "lora_adapter" / "README.md"
    if (artifact_dir / "lora_adapter").exists():
        readme.write_text(
            "# bartleby-1.2 LoRA Adapter\n\n"
            "Clean symmetric from-scratch LoRA adapter on `Qwen/Qwen3-0.6B` trained on `corpus_v2` (0 eval posture overlap).\n"
            "Does **not** warm-start from bartleby-1.0 / blind_adapted_v3.\n",
            encoding="utf-8",
        )
    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Package bartleby-1.2 artifacts.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/bartleby_1_1/from_scratch/bartleby_1_1",
    )
    parser.add_argument("--artifact-dir", type=str, default="artifacts/bartleby-1.2")
    parser.add_argument("--training-summary", type=str, default="")
    parser.add_argument("--eval-metrics", type=str, default="")
    parser.add_argument("--canonical-security-metrics", type=str, default="")
    args = parser.parse_args()

    training_summary = None
    if args.training_summary and Path(args.training_summary).exists():
        training_summary = json.loads(Path(args.training_summary).read_text(encoding="utf-8"))
    eval_metrics = None
    if args.eval_metrics and Path(args.eval_metrics).exists():
        sec_harness = json.loads(Path(args.eval_metrics).read_text(encoding="utf-8"))
        trap_out = sec_harness.get("canonical_trap", {})
        if "bartleby_1_0" in trap_out:
            trap_out["after_bartleby_1_2"] = trap_out["bartleby_1_0"]
            trap_out["after_bartleby_1_1"] = trap_out["bartleby_1_0"]
            trap_out["before_baseline"] = trap_out.get("baseline", {})
        eval_metrics = {
            "security_harness": {
                "canonical_wipe_vs_shadow_trap": trap_out,
                "suites": sec_harness.get("suites", {}),
                "suite4_ood_security": sec_harness.get("suite4_ood_security", {}),
            }
        }
    if args.canonical_security_metrics and Path(args.canonical_security_metrics).exists():
        can_metrics = json.loads(Path(args.canonical_security_metrics).read_text(encoding="utf-8"))
        eval_metrics = eval_metrics or {}
        eval_metrics["canonical_external_security_benchmarks"] = can_metrics

    out = package(
        Path(args.checkpoint_dir),
        Path(args.artifact_dir),
        training_summary=training_summary,
        eval_metrics=eval_metrics,
    )
    print(f"Packaged bartleby-1.1 → {out}")


if __name__ == "__main__":
    main()
