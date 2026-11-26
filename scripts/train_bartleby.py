#!/usr/bin/env python3
"""Train `bartleby-1.1` completely from scratch on `corpus_v2` (no warm-start).

Trains `TradeOffJevNeuralSystem` / `BartlebyNeuralSystem` from `Qwen/Qwen3-0.6B`
with Phase-1 + Phase-2 curriculum, then optionally trains a comparable
`StandardJevBaselineSystem`. Does **not** load `bartleby-1.0` / `blind_adapted_v3`
LoRA weights.

Example (RTX 5070 8 GB–safe defaults):
    .venv/bin/python scripts/train_bartleby_1_1.py \\
        --phase1-steps 1000 --phase2-steps 200 \\
        --batch-size 1 --grad-accum 8 \\
        --train-baseline
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tradeoff_jev.backbone import (
    DEFAULT_QWEN_CHECKPOINT,
    BackboneConfig,
    BaseDecisionNeuralSystem,
    BartlebyNeuralSystem,
    PoolingStrategy,
    PromptConditionedJevBaselineSystem,
    QwenDecisionBackbone,
    StandardJevBaselineSystem,
    TradeOffJevNeuralSystem,
)
from tradeoff_jev.datasets.corpus_v2_builder import HoldoutContaminationGuard
from tradeoff_jev.model import AATModelConfig
from tradeoff_jev.trainer import (
    CurriculumTrainer,
    DecisionStepSample,
    PromptBleedGuard,
    TrainerConfig,
    TrajectoryRecordConverter,
)


@dataclass(frozen=True)
class Bartleby11TrainConfig:
    """Immutable configuration for from-scratch bartleby-1.1 training."""

    model_name_or_path: str = DEFAULT_QWEN_CHECKPOINT
    train_jsonl: str = "data/corpus_v2/train.jsonl"
    val_jsonl: str = "data/corpus_v2/val.jsonl"
    cal_jsonl: str = "data/corpus_v2/cal.jsonl"
    checkpoint_dir: str = "checkpoints/bartleby_1_1"
    run_name: str = "from_scratch"
    baseline_checkpoint_dir: str = "checkpoints/corpus_v2_baseline"
    prompt_conditioned_baseline_checkpoint_dir: str = (
        "checkpoints/corpus_v2_prompt_conditioned_baseline"
    )
    phase1_steps: int = 1000
    phase2_steps: int = 200
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_length: int = 768
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 42
    train_tradeoff: bool = True
    train_baseline: bool = False
    train_prompt_conditioned_baseline: bool = False
    use_bf16: bool = True
    resume_from: Optional[str] = None
    security_oversample: int = 4

    def validate(self) -> None:
        """Assert no holdout contamination in configured data paths."""
        for p in (self.train_jsonl, self.val_jsonl, self.cal_jsonl):
            HoldoutContaminationGuard.assert_path_allowed(p)
            if "test_400" in p.lower():
                raise ValueError(f"Refuse holdout path in training config: {p}")


class Bartleby11FromScratchTrainer:
    """Object-oriented from-scratch trainer for bartleby-1.1 on corpus_v2."""

    def __init__(self, config: Bartleby11TrainConfig) -> None:
        config.validate()
        self._config = config
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _build_backbone(self) -> QwenDecisionBackbone:
        bb_cfg = BackboneConfig(
            model_name_or_path=self._config.model_name_or_path,
            torch_dtype="bfloat16",
            max_seq_length=self._config.max_seq_length,
            pooling_strategy=PoolingStrategy.SPAN_END_AND_MEAN,
            use_lora=True,
            lora_r=self._config.lora_r,
            lora_alpha=self._config.lora_alpha,
            lora_dropout=0.05,
            gradient_checkpointing=True,
        )
        return QwenDecisionBackbone(bb_cfg)

    def _build_aat_config(self) -> AATModelConfig:
        return AATModelConfig(
            num_goals=4,
            num_levels=5,
            hidden_dim=128,
            pointer_dim=64,
            temperature=0.25,
            survival_sharpness=10.0,
        )

    def _load_samples(
        self,
    ) -> tuple[List[DecisionStepSample], List[List[DecisionStepSample]], List[DecisionStepSample], List[DecisionStepSample]]:
        converter = TrajectoryRecordConverter(
            num_goals=4,
            include_oracle_scores_in_text=False,
            shuffle_and_anonymize_candidates=False,
            enforce_anti_bleeding=True,
        )
        # Spot-check anti-bleeding on raw JSONL lines before conversion
        for path in (self._config.train_jsonl, self._config.val_jsonl, self._config.cal_jsonl):
            HoldoutContaminationGuard.assert_path_allowed(path)
            for i, line in enumerate(TrajectoryRecordConverter.read_jsonl_lines(path)[:20]):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "context_text" in rec:
                    PromptBleedGuard.assert_clean(
                        str(rec["context_text"]), location=f"{path}:{i}.context"
                    )

        train_flat, train_trajs = converter.load_jsonl(self._config.train_jsonl)
        if self._config.security_oversample > 1:
            sec_samples = [
                s
                for s in train_flat
                if s.sample_id.startswith(("sec_", "cf_floor_", "rec_td_cf_", "rec_ops_cf_"))
            ]
            for _ in range(self._config.security_oversample - 1):
                train_flat.extend(sec_samples)
        val_flat, _ = converter.load_jsonl(self._config.val_jsonl)
        cal_flat, _ = converter.load_jsonl(self._config.cal_jsonl)
        multi = [t for t in train_trajs if len(t) >= 2]
        if not multi:
            # Fall back to single-step rehearsal episodes for Phase 2
            multi = [[s] for s in train_flat[: min(512, len(train_flat))]]
        return train_flat, multi, val_flat, cal_flat

    def _train_system(
        self,
        *,
        kind: str,
        output_dir: str,
        run_name: str,
        train_flat: Sequence[DecisionStepSample],
        multi_trajs: Sequence[Sequence[DecisionStepSample]],
        cal_flat: Sequence[DecisionStepSample],
    ) -> Dict[str, Any]:
        backbone = self._build_backbone()
        aat_cfg = self._build_aat_config()
        if kind == "tradeoff":
            system: BaseDecisionNeuralSystem = BartlebyNeuralSystem(
                backbone=backbone, aat_config=aat_cfg
            )
        elif kind == "prompt_conditioned_baseline":
            system = PromptConditionedJevBaselineSystem(
                backbone=backbone, aat_config=aat_cfg
            )
        else:
            system = StandardJevBaselineSystem(backbone=backbone, aat_config=aat_cfg)

        # Explicit from-scratch: never load bartleby-1.0 / blind_adapted_v3 weights
        assert not any(
            tag in str(output_dir).lower()
            for tag in ("bartleby-1.0", "blind_adapted_v3")
        )

        system.to(self._device)
        resumed_phase1_steps = 0
        if kind == "tradeoff" and self._config.resume_from:
            resume_path = Path(self._config.resume_from)
            assert not any(
                tag in str(resume_path).lower()
                for tag in ("bartleby-1.0", "blind_adapted_v3")
            )
            system.load_checkpoint_weights(resume_path, map_location=self._device)
            meta_path = resume_path / "checkpoint_metadata.json"
            if meta_path.exists():
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                resumed_phase1_steps = int(
                    meta_obj.get("extra", {}).get(
                        "global_step", meta_obj.get("global_step", 0)
                    )
                )
            print(
                f"[bartleby-1.1] Loaded initial weights from {resume_path} "
                f"(resumed_phase1_steps={resumed_phase1_steps})",
                flush=True,
            )
        trainer_cfg = TrainerConfig(
            output_dir=output_dir,
            run_name=run_name,
            lora_lr=1e-4,
            head_lr=3e-4,
            batch_size=self._config.batch_size,
            trajectory_batch_size=1,
            gradient_accumulation_steps=self._config.grad_accum,
            phase1_steps=self._config.phase1_steps,
            phase2_steps=self._config.phase2_steps,
            scheduled_sampling_start=0.25,
            scheduled_sampling_end=1.0,
            use_bf16=self._config.use_bf16 and self._device.type == "cuda",
            checkpoint_every_steps=100,
            log_every_steps=10,
            seed=self._config.seed,
        )
        trainer = CurriculumTrainer(system=system, config=trainer_cfg, device=self._device)
        print(
            f"[bartleby-1.1] training {kind} from scratch "
            f"phase1={self._config.phase1_steps} phase2={self._config.phase2_steps} "
            f"n_train={len(train_flat)} device={self._device}",
            flush=True,
        )
        summary = trainer.train(
            phase1_samples=list(train_flat),
            phase2_trajectories=list(multi_trajs) if self._config.phase2_steps > 0 else None,
            cal_samples=list(cal_flat),
        )
        training_dict = summary.to_dict()
        if resumed_phase1_steps > 0 and int(training_dict.get("phase1_steps_completed", 0)) == 0:
            training_dict["phase1_steps_completed"] = resumed_phase1_steps
            training_dict["phase1_resumed_from"] = str(self._config.resume_from)
            training_dict["total_optimizer_steps"] = (
                resumed_phase1_steps + int(training_dict.get("phase2_steps_completed", 0))
            )
        result = {
            "kind": kind,
            "from_scratch": True,
            "warm_start": None,
            "base_checkpoint": self._config.model_name_or_path,
            "training": training_dict,
            "peak_vram_mib": summary.peak_vram_mib,
            "final_loss": summary.final_loss,
            "output_dir": str(Path(output_dir) / run_name),
        }
        del trainer, system, backbone
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def run(self) -> Dict[str, Any]:
        """Execute from-scratch Phase1+Phase2 training and return summary."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for bartleby-1.1 from-scratch training.")

        started = time.time()
        train_flat, multi, val_flat, cal_flat = self._load_samples()
        print(
            f"[bartleby-1.1] loaded train={len(train_flat)} val={len(val_flat)} "
            f"cal={len(cal_flat)} multi_trajs={len(multi)}",
            flush=True,
        )

        tradeoff_result: Optional[Dict[str, Any]] = None
        if self._config.train_tradeoff:
            tradeoff_result = self._train_system(
                kind="tradeoff",
                output_dir=self._config.checkpoint_dir,
                run_name=self._config.run_name,
                train_flat=train_flat,
                multi_trajs=multi,
                cal_flat=cal_flat,
            )

        baseline_result: Optional[Dict[str, Any]] = None
        if self._config.train_baseline:
            baseline_result = self._train_system(
                kind="baseline",
                output_dir=self._config.baseline_checkpoint_dir,
                run_name=self._config.run_name,
                train_flat=train_flat,
                multi_trajs=multi,
                cal_flat=cal_flat,
            )

        prompt_conditioned_result: Optional[Dict[str, Any]] = None
        if self._config.train_prompt_conditioned_baseline:
            prompt_conditioned_result = self._train_system(
                kind="prompt_conditioned_baseline",
                output_dir=self._config.prompt_conditioned_baseline_checkpoint_dir,
                run_name=self._config.run_name,
                train_flat=train_flat,
                multi_trajs=multi,
                cal_flat=cal_flat,
            )
            pc_run_dir = (
                Path(self._config.prompt_conditioned_baseline_checkpoint_dir)
                / self._config.run_name
            )
            pc_src = pc_run_dir / ("phase2_final" if self._config.phase2_steps > 0 else "phase1_final")
            pc_alias = pc_run_dir / "prompt_conditioned_jev_baseline"
            if pc_src.exists():
                if pc_alias.exists():
                    shutil.rmtree(pc_alias)
                shutil.copytree(pc_src, pc_alias)
                pc_cal = pc_run_dir / "calibration"
                if pc_cal.exists():
                    shutil.copytree(pc_cal, pc_alias / "calibration", dirs_exist_ok=True)

        # Promote final checkpoint tags for packaging
        run_dir = Path(self._config.checkpoint_dir) / self._config.run_name
        phase_tag = "phase2_final" if self._config.phase2_steps > 0 else "phase1_final"
        final_src = run_dir / phase_tag
        packaged_ckpt = run_dir / "bartleby_1_1"
        if final_src.exists():
            if packaged_ckpt.exists():
                shutil.rmtree(packaged_ckpt)
            shutil.copytree(final_src, packaged_ckpt)
            # Copy calibration if present
            cal_src = run_dir / "calibration"
            if cal_src.exists():
                shutil.copytree(cal_src, packaged_ckpt / "calibration", dirs_exist_ok=True)

        logs_summary_path = _REPO_ROOT / "logs" / "bartleby_1_2_training_summary.json"
        if logs_summary_path.exists():
            try:
                prev_summary = json.loads(logs_summary_path.read_text(encoding="utf-8"))
                if tradeoff_result is None and prev_summary.get("tradeoff") is not None:
                    tradeoff_result = prev_summary["tradeoff"]
                if baseline_result is None and prev_summary.get("baseline") is not None:
                    baseline_result = prev_summary["baseline"]
                if (
                    prompt_conditioned_result is None
                    and prev_summary.get("prompt_conditioned_baseline") is not None
                ):
                    prompt_conditioned_result = prev_summary["prompt_conditioned_baseline"]
            except Exception:
                pass

        summary = {
            "status": "completed",
            "model_id": "bartleby-1.2",
            "version": "1.2.0",
            "from_scratch": True,
            "warm_start": None,
            "config": asdict(self._config),
            "gpu_name": torch.cuda.get_device_name(self._device),
            "num_train": len(train_flat),
            "num_val": len(val_flat),
            "num_cal": len(cal_flat),
            "tradeoff": tradeoff_result,
            "baseline": baseline_result,
            "prompt_conditioned_baseline": prompt_conditioned_result,
            "packaged_checkpoint": str(packaged_ckpt) if packaged_ckpt.exists() else None,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        summary_path = run_dir / "bartleby_1_2_training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logs_summary_path.parent.mkdir(parents=True, exist_ok=True)
        logs_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"[bartleby-1.2] wrote {summary_path} and {logs_summary_path}",
            flush=True,
        )
        return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> Bartleby11TrainConfig:
    p = argparse.ArgumentParser(description="Train bartleby-1.1 from scratch on corpus_v2.")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_QWEN_CHECKPOINT)
    p.add_argument("--train-jsonl", type=str, default="data/corpus_v2/train.jsonl")
    p.add_argument("--val-jsonl", type=str, default="data/corpus_v2/val.jsonl")
    p.add_argument("--cal-jsonl", type=str, default="data/corpus_v2/cal.jsonl")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/bartleby_1_1")
    p.add_argument("--run-name", type=str, default="from_scratch")
    p.add_argument("--phase1-steps", type=int, default=1000)
    p.add_argument("--phase2-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-baseline", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--train-prompt-conditioned-baseline", action="store_true")
    p.add_argument("--prompt-conditioned-baseline-only", action="store_true")
    p.add_argument(
        "--prompt-conditioned-baseline-checkpoint-dir",
        type=str,
        default="checkpoints/corpus_v2_prompt_conditioned_baseline",
    )
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--security-oversample", type=int, default=4)
    args = p.parse_args(argv)
    return Bartleby11TrainConfig(
        model_name_or_path=args.checkpoint,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        cal_jsonl=args.cal_jsonl,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        prompt_conditioned_baseline_checkpoint_dir=(
            args.prompt_conditioned_baseline_checkpoint_dir
        ),
        phase1_steps=args.phase1_steps,
        phase2_steps=args.phase2_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        train_tradeoff=not bool(
            args.baseline_only or args.prompt_conditioned_baseline_only
        ),
        train_baseline=bool(args.train_baseline or args.baseline_only)
        and not bool(args.prompt_conditioned_baseline_only),
        train_prompt_conditioned_baseline=bool(
            args.train_prompt_conditioned_baseline
            or args.prompt_conditioned_baseline_only
        ),
        use_bf16=not args.no_bf16,
        resume_from=args.resume_from,
        security_oversample=args.security_oversample,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    summary = Bartleby11FromScratchTrainer(cfg).run()
    active_res = (
        summary.get("prompt_conditioned_baseline")
        if cfg.train_prompt_conditioned_baseline and not cfg.train_tradeoff
        else (summary.get("tradeoff") or summary.get("baseline") or {})
    ) or {}
    print(json.dumps({"status": summary["status"], "final_loss": active_res.get("final_loss"),
                      "peak_vram_mib": active_res.get("peak_vram_mib"),
                      "packaged": summary["packaged_checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
