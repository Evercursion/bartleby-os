"""Two-Phase Curriculum Trainer and PyTorch Autograd Losses for Trade-Off Jev.

Implements the two-phase training pipeline for single-forward-pass Aspiration Adaptation
Decision Models (`TradeOffJevNeuralSystem` and `StandardJevBaselineSystem`):

- **Phase 1 — Single-Step Supervised Calibration**:
  Trains `QwenDecisionBackbone` (`bfloat16` + LoRA `r=16, alpha=32`) jointly with the
  FP32 AAT readout head using teacher-forced incoming Preference Tables $\\mathcal{T}_{t-1}^*$
  under the composite `TorchAATCurriculumLoss`:
  $$\\mathcal{L} = \\mathcal{L}_{\\text{CE}} + \\lambda_{\\text{rubric}} \\mathcal{L}_{\\text{Rubric}}
    + \\lambda_1 \\mathcal{L}_{\\text{Brier}} + \\lambda_2 \\mathcal{L}_{\\text{Transition}}
    + \\lambda_3 \\mathcal{L}_{\\text{Bartleby}}$$
- **Phase 2 — Multi-Step Markov Rollout with Scheduled Sampling**:
  Unrolls multi-step episodes ($t = 0, 1, \\dots, T - 1$) along the Selten Markov chain
  $\\mathcal{T}_{t-1} \\to \\mathcal{T}_t$. With scheduled probability $p_{\\text{ss}} \\in [0, 1]$
  ramping across Phase 2, step $t$ consumes the model's own differentiably predicted
  aspiration vector $\\hat{\\vec{a}}_{t-1}$ from step $t - 1$ rather than the oracle vector,
  eliminating exposure bias across sequential retreats and upward urgency adaptations.

Hardware & Numerical Design:
    - Native `torch.amp.autocast("cuda", dtype=torch.bfloat16)` for Blackwell (`sm_120`)
      and Ampere/Ada Tensor Cores while preserving full FP32 precision in the AAT head
      and proper scoring losses.
    - Separate AdamW learning rates for LoRA adapters (`1e-4`) and FP32 AAT readout
      parameters (`3e-4`) governed by linear warmup + cosine annealing.
    - Gradient accumulation, gradient norm clipping, `safetensors` checkpointing, peak
      VRAM telemetry, and structured JSONL/JSON training logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .backbone import (
    BaseDecisionNeuralSystem,
    PackedDecisionBatch,
    format_policy_conditioned_state_text,
)
from .calibration import CalibrationMode, CalibrationReport, TemperatureCalibrator

# Anti-bleeding firewall: meta / oracle leakage phrases forbidden in all training & eval texts.
BLEED_PHRASES: Tuple[str, ...] = (
    "survival floor",
    "survival_floor",
    "non-negotiable",
    "bartleby",
    "_bartleby",
    "infeasible",
    "candidate_scores",
    "retreat_goal",
    "every offered",
    "every candidate action violates",
    "every candidate violates",
    "proposes only destructive",
    "no non-lethal option exists",
    "no candidate satisfies",
    "mandatory floor violation",
    "moral machine lethal dilemma",
    "inescapable l1 conflict",
    "observability blackout",
    "epistemic perception occlusion",
    "sensor blackout",
    "non-lethal escape trade-off",
    "av safety retreat",
    "emergency_lockout",
    "opt_safe_",
)


class PromptBleedGuard:
    """Rejects training / prompt text that would leak meta-supervision phrases.

    Theoretical purpose:
        Prevents keyword shortcuts that bypass context-aware rubric perception and
        the intrinsic mathematical Bartleby floor channel
        ($\\forall k:\\,\\exists m\\,\\hat{E}_{k,m} < a_m^{\\min}$).
    """

    FORBIDDEN: Tuple[str, ...] = BLEED_PHRASES

    # Oracle-style parenthetical score/outcome hints (not ordinary descriptive parentheses).
    # Catches both literal g0..g3 and named domain goals / outcome stamps.
    _PAREN_SCORE_HINT: re.Pattern[str] = re.compile(
        r"\((?:[^)]*\b(?:"
        r"preserves\s+[a-z0-9_/ -]+,\s*retreats\s+[a-z0-9_/ -]+"
        r"|preserves\s+g\d"
        r"|retreats\s+g\d"
        r"|breaches\s+[a-z0-9_/ -]+\s+floor"
        r"|violates\s+[a-z0-9_/ -]+"
        r"|preserves\s+human\s+safety"
        r"|zero\s+human\s+harm"
        r"|lethal\s+pedestrian"
        r"|fatal\s+collision\s+with"
        r"|prioritizes\s+direct\s+mission/human\s+objective"
        r"|preserves\s+robot\s+actuator\s+integrity"
        r"|reckless\s+shortcut\s+exposing\s+nearby"
        r"|scores?\s*:"
        r"|g\d\s*=\s*\d"
        r"|level\s*[:=]\s*\d"
        r")[^)]*)\)",
        re.IGNORECASE,
    )

    _BRACKET_REGIME_HEADER: re.Pattern[str] = re.compile(
        r"\[\s*(?:"
        r"MANDATORY\s+FLOOR\s+VIOLATION"
        r"|MORAL\s+MACHINE\s+LETHAL\s+DILEMMA(?:\s*#\d+)?"
        r"|INESCAPABLE\s+L1\s+CONFLICT"
        r"|OBSERVABILITY\s+BLACKOUT"
        r"|EPISTEMIC\s+PERCEPTION\s+OCCLUSION(?:\s*#\d+)?"
        r"|SENSOR\s+BLACKOUT"
        r"|NON-LETHAL\s+ESCAPE\s+TRADE-OFF"
        r"|AV\s+SAFETY\s+RETREAT"
        r")\s*(?:\|\s*([^\]]+))?\]",
        re.IGNORECASE,
    )

    _TRAILING_LEAK_SENTENCES: re.Pattern[str] = re.compile(
        r"(?:"
        r"No\s+candidate\s+satisfies\s+[^.]*\."
        r"|No\s+non-lethal\s+option\s+exists\s+[^.]*\."
        r"|Every\s+offered\s+command\s+breaches\s+[^.]*\."
        r"|The\s+automation\s+proposes\s+only\s+destructive\s+[^.]*\."
        r"|Safety\s+compliance\s+and\s+blast-radius\s+scores\s+for\s+candidate\s+actions\s+are\s+currently\s+unobservable\s+[^.]*\."
        r"|\[Outcome\s+Telemetry\s+Note:[^\]]*\]"
        r")",
        re.IGNORECASE,
    )

    @classmethod
    def find_violation(cls, text: str) -> Optional[str]:
        """Return the first forbidden phrase found in `text`, else `None`."""
        lowered = text.lower()
        for phrase in cls.FORBIDDEN:
            if phrase in lowered:
                return phrase
        if cls._PAREN_SCORE_HINT.search(text):
            return "parenthetical_score_hint"
        if cls._BRACKET_REGIME_HEADER.search(text):
            return "bracket_regime_header"
        return None

    @classmethod
    def assert_clean(cls, text: str, *, location: str) -> None:
        """Raise ``ValueError`` if any banned anti-bleeding phrase appears in `text`."""
        hit = cls.find_violation(text)
        if hit is not None:
            raise ValueError(f"Anti-bleeding violation in {location}: contains {hit!r}")

    @classmethod
    def scrub(cls, text: str) -> str:
        """Strip forbidden meta phrases, bracket regime headers, and parenthetical score hints."""
        cleaned = cls._BRACKET_REGIME_HEADER.sub(
            lambda m: f"[{m.group(1).strip()}]" if m.group(1) else "",
            text,
        )
        cleaned = cls._TRAILING_LEAK_SENTENCES.sub("", cleaned)
        cleaned = cls._PAREN_SCORE_HINT.sub("", cleaned)
        for phrase in cls.FORBIDDEN:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            cleaned = pattern.sub("", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        return cleaned.strip()


@dataclass(frozen=True)
class DecisionStepSample:
    """Immutable supervised training sample for a single Markov decision step $t$.

    Attributes:
        sample_id: Unique identifier for the decision step.
        episode_id: Trajectory/episode identifier grouping sequential Markov steps.
        step: Zero-based discrete Markov epoch $t \\ge 0$.
        state_text: Formatted textual context $x_t$ including goal definitions.
        candidate_descriptions: Sequence of $K$ candidate action descriptions.
        aspiration_prev: Incoming aspiration vector $\\vec{a}_{t-1}$ of length $M$.
        survival_floor: Bartleby survival floor vector $\\vec{a}^{\\min}$ of length $M$.
        urgency_ranks: Zero-based urgency rank per goal (`0` = most urgent) of length $M$.
        retreat_index: Goal index $r_{t-1} \\in \\{0, \\dots, M - 1\\}$ lowered first on infeasibility.
        target_choice: Ground-truth winning index in $\\{0, \\dots, K\\}$ (`K` = Bartleby refusal).
        target_noul: Ground-truth `tradeoff_noul` targets in $[0, 1]$ of length $K$.
        target_aspiration: Ground-truth adapted aspiration vector $\\vec{a}_t^*$ of length $M$.
        target_bartleby: Ground-truth Bartleby refusal indicator (`1.0` if floor violated else `0.0`).
        target_rubric_levels: Optional ground-truth ordinal levels of shape `[K, M]`, or `None`.
    """

    sample_id: str
    episode_id: str
    step: int
    state_text: str
    candidate_descriptions: Tuple[str, ...]
    aspiration_prev: Tuple[float, ...]
    survival_floor: Tuple[float, ...]
    urgency_ranks: Tuple[float, ...]
    retreat_index: int
    target_choice: int
    target_noul: Tuple[float, ...]
    target_aspiration: Tuple[float, ...]
    target_bartleby: float
    target_rubric_levels: Optional[Tuple[Tuple[int, ...], ...]] = None


@dataclass(frozen=True)
class CollatedStepBatch:
    """Device-ready collated tensor batch for single-step or rollout-step execution."""

    packed_batch: PackedDecisionBatch
    aspiration_prev: torch.Tensor
    survival_floor: torch.Tensor
    urgency_ranks: torch.Tensor
    retreat_index: torch.Tensor
    target_choice: torch.Tensor
    target_noul: torch.Tensor
    target_aspiration: torch.Tensor
    target_bartleby: torch.Tensor
    target_rubric_levels: Optional[torch.Tensor]

    def to(self, device: torch.device | str) -> "CollatedStepBatch":
        """Move all batch tensors onto `device`."""
        return CollatedStepBatch(
            packed_batch=self.packed_batch.to(device),
            aspiration_prev=self.aspiration_prev.to(device, non_blocking=True),
            survival_floor=self.survival_floor.to(device, non_blocking=True),
            urgency_ranks=self.urgency_ranks.to(device, non_blocking=True),
            retreat_index=self.retreat_index.to(device, non_blocking=True),
            target_choice=self.target_choice.to(device, non_blocking=True),
            target_noul=self.target_noul.to(device, non_blocking=True),
            target_aspiration=self.target_aspiration.to(device, non_blocking=True),
            target_bartleby=self.target_bartleby.to(device, non_blocking=True),
            target_rubric_levels=(
                self.target_rubric_levels.to(device, non_blocking=True)
                if self.target_rubric_levels is not None
                else None
            ),
        )


class TrajectoryRecordConverter:
    """Converts serialized JSONL trajectory records (`TrajectoryStepRecord` / `corpus_v1`) to `DecisionStepSample`s.

    Automatically pads domain goal dimensions $m \\le M_{\\max}$ up to `num_goals = M_max`
    using inactive dummy goals fixed at satisfied floor `0` so heterogeneous domains
    (e.g., 3-goal Software/Trolley and 4-goal Asimov) train cleanly in unified batches.
    By default (`include_oracle_scores_in_text=False`), strips `"candidate_scores"` from
    the serialized `state` dictionary to prevent oracle rubric leakage into the prompt text.

    Anti-bleeding (bartleby-1.1):
        Scrubs and asserts against meta phrases (`survival floor`, `bartleby`,
        `candidate_scores`, parenthetical score hints, …) in all emitted texts.
    """

    def __init__(
        self,
        num_goals: int = 4,
        include_oracle_scores_in_text: bool = False,
        shuffle_and_anonymize_candidates: bool = False,
        enforce_anti_bleeding: bool = True,
    ) -> None:
        if num_goals < 1:
            raise ValueError(f"num_goals must be >= 1, got {num_goals}.")
        self._num_goals = num_goals
        self._include_oracle_scores_in_text = include_oracle_scores_in_text
        self._shuffle_and_anonymize_candidates = shuffle_and_anonymize_candidates
        self._enforce_anti_bleeding = bool(enforce_anti_bleeding)

    def convert_record(self, record: Mapping[str, Any]) -> DecisionStepSample:
        """Convert one dictionary record from `train_trajectories.jsonl` or `corpus_v1`."""
        req = record["request"]
        resp = record["response"]
        episode_id = str(record.get("episode_id", f"ep_{req.get('step', 0)}"))
        step = int(record.get("step", req.get("step", 0)))
        sample_id = str(record.get("sample_id", f"{episode_id}__t{step}"))

        goals = list(req["goal_schema"])
        if len(goals) > self._num_goals:
            raise ValueError(
                f"Record {sample_id} has {len(goals)} goals, exceeding configured num_goals={self._num_goals}."
            )
        goal_ids = [str(g["id"]) for g in goals]

        prior_table = resp["prior_preference_table"]
        adapted_table = resp["preference_table"]
        raw_floors = record.get("survival_floors") or (
            req["state"].get("survival_floors", {}) if isinstance(req.get("state"), Mapping) else {}
        )

        # Pad goal vectors from len(goals) up to self._num_goals
        asp_prev: List[float] = []
        s_floor: List[float] = []
        u_ranks: List[float] = []
        target_asp: List[float] = []

        urgency_list = [str(u) for u in prior_table["urgency_order"]]
        for gid in goal_ids:
            asp_prev.append(float(prior_table["aspiration_levels"][gid]))
            s_floor.append(float(raw_floors.get(gid, 0)))
            u_ranks.append(float(urgency_list.index(gid)) if gid in urgency_list else float(len(goal_ids) - 1))
            target_asp.append(float(adapted_table["aspiration_levels"][gid]))

        while len(asp_prev) < self._num_goals:
            pad_rank = float(len(asp_prev))
            asp_prev.append(0.0)
            s_floor.append(0.0)
            u_ranks.append(pad_rank)
            target_asp.append(0.0)

        retreat_gid = str(prior_table["retreat_goal"])
        retreat_idx = goal_ids.index(retreat_gid) if retreat_gid in goal_ids else max(0, len(goal_ids) - 1)

        # Format context text & candidate descriptions
        raw_state = req["state"]
        if isinstance(raw_state, Mapping):
            filtered_state = {
                k: v
                for k, v in raw_state.items()
                if self._include_oracle_scores_in_text or k != "candidate_scores"
            }
            # Never serialize survival_floors / retreat_goal into prompt state text.
            if self._enforce_anti_bleeding:
                filtered_state = {
                    k: v
                    for k, v in filtered_state.items()
                    if k
                    not in {
                        "survival_floors",
                        "retreat_goal",
                        "candidate_scores",
                        "bartleby_triggered",
                    }
                }
            state_body = json.dumps(filtered_state, ensure_ascii=True)
        else:
            state_body = str(raw_state)

        goal_summary = "; ".join(
            f"{g['id']} ({g['name']}: {g['description']})" for g in goals
        )
        state_text = f"Domain State: {state_body} | Goals: {goal_summary}"
        if self._enforce_anti_bleeding:
            state_text = PromptBleedGuard.scrub(state_text)

        candidates = list(req.get("candidates") or [])
        if self._shuffle_and_anonymize_candidates and len(candidates) > 1:
            # Normalize counterfactual sibling IDs (`..._a` / `..._b` / `..._high` / `..._low`)
            # so paired records receive the exact same deterministic permutation across candidate slots.
            perm_key = str(
                record.get("counterfactual_pair_id")
                or record.get("scenario_group_id")
                or re.sub(r"(_[abAB]|__[abAB]|_high|_low)$", "", sample_id)
            )
            seed_int = int.from_bytes(
                hashlib.sha256(perm_key.encode("utf-8")).digest()[:8], "big"
            )
            perm_rng = random.Random(seed_int)
            perm = list(range(len(candidates)))
            perm_rng.shuffle(perm)
            candidates = [candidates[idx] for idx in perm]

        cand_ids = [str(c["id"]) for c in candidates]
        if self._shuffle_and_anonymize_candidates:
            cand_descs = tuple(
                f"option_{idx}: {PromptBleedGuard.scrub(str(c['proposition']))}"
                + (
                    f" ({PromptBleedGuard.scrub(str(c['description']))})"
                    if c.get("description")
                    else ""
                )
                for idx, c in enumerate(candidates)
            )
        else:
            cand_descs = tuple(
                f"{c['id']}: {PromptBleedGuard.scrub(str(c['proposition']))}"
                + (
                    f" ({PromptBleedGuard.scrub(str(c['description']))})"
                    if c.get("description")
                    else ""
                )
                for c in candidates
            )

        if self._enforce_anti_bleeding:
            cand_descs = tuple(PromptBleedGuard.scrub(d) for d in cand_descs)
            PromptBleedGuard.assert_clean(state_text, location=f"{sample_id}.state_text")
            for desc in cand_descs:
                PromptBleedGuard.assert_clean(desc, location=f"{sample_id}.candidate")

        # Extract per-candidate target nouls and optional rubric realizations
        ans_by_cand: Dict[str, float] = {}
        for ans in resp.get("answers", []):
            if ans.get("type") == "tradeoff_noul" and ans.get("candidate_id") is not None:
                ans_by_cand[str(ans["candidate_id"])] = float(ans.get("noul", 0.0))

        eval_by_cand = {
            str(ce["candidate_id"]): ce for ce in resp.get("candidate_evaluations", [])
        }

        bartleby_triggered = bool(
            record.get(
                "bartleby_triggered",
                resp["feasibility_summary"]["transition_applied"] == "infeasible_floor",
            )
        )
        explicit_selected = record.get("selected_candidate_id")
        sat_cands = set(resp.get("feasibility_summary", {}).get("satisficing_candidates") or [])

        target_nouls: List[float] = []
        rubric_matrix: List[Tuple[int, ...]] = []
        for cid in cand_ids:
            ce = eval_by_cand.get(cid)
            if cid in ans_by_cand:
                noul_val = ans_by_cand[cid]
            elif bartleby_triggered:
                noul_val = 0.01
            elif explicit_selected is not None:
                if cid == str(explicit_selected):
                    noul_val = 0.96
                elif cid in sat_cands:
                    noul_val = 0.91
                else:
                    noul_val = 0.04
            elif ce is not None:
                noul_val = float(ce.get("joint_satisfaction_probability", 0.96 if ce.get("satisfices_adapted_table") else 0.04))
            else:
                noul_val = 0.0
            target_nouls.append(noul_val)
            if ce is not None:
                row_lvls: List[int] = []
                for gid in goal_ids:
                    row_lvls.append(int(ce["goal_expectations"][gid]["most_likely_level"]))
                while len(row_lvls) < self._num_goals:
                    row_lvls.append(0)
                rubric_matrix.append(tuple(row_lvls))

        target_bartleby = 1.0 if bartleby_triggered else 0.0

        if bartleby_triggered:
            target_choice = len(cand_ids)
        elif explicit_selected is not None and str(explicit_selected) in cand_ids:
            target_choice = int(cand_ids.index(str(explicit_selected)))
        else:
            best_idx = max(range(len(target_nouls)), key=lambda idx: target_nouls[idx])
            target_choice = int(best_idx)

        return DecisionStepSample(
            sample_id=sample_id,
            episode_id=episode_id,
            step=step,
            state_text=state_text,
            candidate_descriptions=cand_descs,
            aspiration_prev=tuple(asp_prev),
            survival_floor=tuple(s_floor),
            urgency_ranks=tuple(u_ranks),
            retreat_index=retreat_idx,
            target_choice=target_choice,
            target_noul=tuple(target_nouls),
            target_aspiration=tuple(target_asp),
            target_bartleby=target_bartleby,
            target_rubric_levels=tuple(rubric_matrix) if len(rubric_matrix) == len(cand_ids) else None,
        )

    @staticmethod
    def read_jsonl_lines(jsonl_path: str | Path) -> List[str]:
        """Read lines from a `.jsonl` file, transparently falling back to `.jsonl.gz` if compressed."""
        p = Path(jsonl_path)
        if p.exists():
            if p.suffix == ".gz":
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    return fh.read().splitlines()
            return p.read_text(encoding="utf-8").splitlines()
        gz_p = Path(f"{p}.gz")
        if gz_p.exists():
            with gzip.open(gz_p, "rt", encoding="utf-8") as fh:
                return fh.read().splitlines()
        raise FileNotFoundError(f"JSONL file not found: {p} (nor {gz_p})")

    def load_jsonl(self, jsonl_path: str | Path) -> Tuple[List[DecisionStepSample], List[List[DecisionStepSample]]]:
        """Load a JSONL trajectory file into flat single-step samples and grouped Markov episodes."""
        flat_samples: List[DecisionStepSample] = []
        by_episode: Dict[str, List[DecisionStepSample]] = {}
        for line in self.read_jsonl_lines(jsonl_path):
            stripped = line.strip()
            if not stripped:
                continue
            sample = self.convert_record(json.loads(stripped))
            flat_samples.append(sample)
            by_episode.setdefault(sample.episode_id, []).append(sample)

        trajectories = [
            sorted(steps, key=lambda s: s.step) for steps in by_episode.values()
        ]
        return flat_samples, trajectories


class TorchAATCurriculumLoss(nn.Module):
    """Differentiable FP32 Composite Curriculum Loss for Trade-Off Jev and Baseline Ablations.

    Optimizes:
        $$\\mathcal{L} = \\mathcal{L}_{\\text{CE}} + \\lambda_{\\text{rubric}} \\mathcal{L}_{\\text{Rubric}}
          + \\lambda_1 \\mathcal{L}_{\\text{Brier}} + \\lambda_2 \\mathcal{L}_{\\text{Transition}}
          + \\lambda_3 \\mathcal{L}_{\\text{Bartleby}}$$
    """

    def __init__(
        self,
        brier_weight: float = 1.0,
        transition_weight: float = 0.5,
        bartleby_weight: float = 1.5,
        rubric_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self._brier_weight = float(brier_weight)
        self._transition_weight = float(transition_weight)
        self._bartleby_weight = float(bartleby_weight)
        self._rubric_weight = float(rubric_weight)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: CollatedStepBatch,
    ) -> Dict[str, torch.Tensor]:
        """Compute differentiable scalar total loss and individual diagnostic loss tensors."""
        cand_mask = batch.packed_batch.candidate_mask  # shape: [B, K]
        mask_f = cand_mask.to(torch.float32)
        valid_count = mask_f.sum().clamp(min=1.0)

        # 1. Categorical choice cross-entropy over K candidates + Bartleby slot K
        choice_logits = outputs["choice_logits"]  # shape: [B, K + 1]
        loss_ce = F.cross_entropy(choice_logits, batch.target_choice)

        # 2. Proper Scoring Brier loss on conditioned `tradeoff_noul`
        pred_noul = outputs["tradeoff_noul"]
        sq_err = ((pred_noul - batch.target_noul) ** 2) * mask_f
        loss_brier = sq_err.sum() / valid_count

        # 3. Markov aspiration transition MSE (0.0 for unconditioned StandardJevBaselineSystem)
        if "updated_aspiration" in outputs:
            loss_transition = F.mse_loss(outputs["updated_aspiration"], batch.target_aspiration)
        else:
            loss_transition = torch.zeros((), dtype=torch.float32, device=choice_logits.device)

        # 4. Bartleby survival-floor BCE + noul suppression penalty when floor is infeasible
        bartleby_logit = outputs["bartleby_logit"]
        b_true = batch.target_bartleby
        loss_b_bce = F.binary_cross_entropy_with_logits(bartleby_logit, b_true)
        noul_suppression = (b_true.unsqueeze(-1) * (pred_noul ** 2) * mask_f).sum() / valid_count
        loss_bartleby = loss_b_bce + 0.5 * noul_suppression

        # 5. Ordinal rubric cross-entropy + Ranked Probability Score (RPS)
        if batch.target_rubric_levels is not None and "rubric_logits" in outputs:
            rubric_logits = outputs["rubric_logits"]  # [B, K, M, L]
            num_levels = rubric_logits.shape[-1]
            flat_logits = rubric_logits[cand_mask].reshape(-1, num_levels)
            flat_targets = batch.target_rubric_levels[cand_mask].reshape(-1).clamp(0, num_levels - 1)
            ce_rubric = F.cross_entropy(flat_logits, flat_targets)

            probs = F.softmax(flat_logits, dim=-1)
            cdf_pred = torch.cumsum(probs, dim=-1)
            level_idx = torch.arange(num_levels, device=flat_logits.device).unsqueeze(0)
            cdf_true = (level_idx >= flat_targets.unsqueeze(-1)).to(torch.float32)
            rps = torch.mean(torch.sum((cdf_pred - cdf_true) ** 2, dim=-1))
            loss_rubric = ce_rubric + 0.5 * rps
        else:
            loss_rubric = torch.zeros((), dtype=torch.float32, device=choice_logits.device)

        total_loss = (
            loss_ce
            + self._rubric_weight * loss_rubric
            + self._brier_weight * loss_brier
            + self._transition_weight * loss_transition
            + self._bartleby_weight * loss_bartleby
        )

        return {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_rubric": loss_rubric.detach(),
            "loss_brier": loss_brier.detach(),
            "loss_transition": loss_transition.detach(),
            "loss_bartleby": loss_bartleby.detach(),
        }


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable configuration for `CurriculumTrainer`.

    Attributes:
        output_dir: Root directory for checkpoints (`checkpoints/<run_name>`) and JSON logs.
        run_name: Descriptive identifier for this training run.
        lora_lr: Peak learning rate for `peft` LoRA adapter weights (`1e-4`).
        head_lr: Peak learning rate for FP32 AAT/Readout head parameters (`3e-4`).
        weight_decay: AdamW weight decay on matrix weights (`0.01`).
        min_lr_ratio: Cosine schedule floor ratio relative to peak LR (`0.10`).
        warmup_ratio: Fraction of total optimizer steps spent in linear LR warmup (`0.10`).
        batch_size: Micro-batch size for Phase 1 single-step training (`4`).
        trajectory_batch_size: Number of Markov episodes per micro-batch in Phase 2 (`2`).
        gradient_accumulation_steps: Micro-batches accumulated per optimizer update (`2`).
        max_grad_norm: Maximum global L2 gradient norm (`1.0`).
        phase1_steps: Number of optimizer update steps in Phase 1 (`50`).
        phase2_steps: Number of optimizer update steps in Phase 2 (`50`).
        scheduled_sampling_start: Initial Phase 2 probability of feeding $\\hat{\\vec{a}}_{t-1}$ (`0.25`).
        scheduled_sampling_end: Final Phase 2 probability of feeding $\\hat{\\vec{a}}_{t-1}$ (`1.0`).
        detach_rollout_aspiration: If `False`, gradients backpropagate across Markov steps
            through $\\hat{\\vec{a}}_{t-1}$.
        use_bf16: Enable `torch.amp.autocast("cuda", dtype=torch.bfloat16)` when on CUDA.
        checkpoint_every_steps: Save intermediate checkpoint every $N$ optimizer steps (`0` = phase ends only).
        log_every_steps: Emit JSONL metric entry every $N$ optimizer steps (`1`).
        seed: Deterministic random seed for shuffling and scheduled-sampling coin flips.
    """

    output_dir: str = "checkpoints"
    run_name: str = "tradeoff_jev_qwen"
    lora_lr: float = 1e-4
    head_lr: float = 3e-4
    weight_decay: float = 0.01
    min_lr_ratio: float = 0.10
    warmup_ratio: float = 0.10
    batch_size: int = 4
    trajectory_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    phase1_steps: int = 50
    phase2_steps: int = 50
    scheduled_sampling_start: float = 0.25
    scheduled_sampling_end: float = 1.0
    detach_rollout_aspiration: bool = False
    use_bf16: bool = True
    checkpoint_every_steps: int = 0
    log_every_steps: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate trainer hyperparameters."""
        if self.lora_lr <= 0.0 or self.head_lr <= 0.0:
            raise ValueError("lora_lr and head_lr must be strictly positive.")
        if self.batch_size < 1 or self.trajectory_batch_size < 1:
            raise ValueError("batch_size and trajectory_batch_size must be >= 1.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
        if self.phase1_steps < 0 or self.phase2_steps < 0:
            raise ValueError("phase1_steps and phase2_steps must be >= 0.")
        if not (0.0 <= self.scheduled_sampling_start <= 1.0) or not (
            0.0 <= self.scheduled_sampling_end <= 1.0
        ):
            raise ValueError("Scheduled sampling probabilities must lie in [0, 1].")

    @property
    def total_steps(self) -> int:
        """Total optimizer steps across Phase 1 and Phase 2."""
        return self.phase1_steps + self.phase2_steps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunSummary:
    """Immutable executive summary returned upon completion of `CurriculumTrainer.train`."""

    run_name: str
    system_type: str
    device: str
    total_optimizer_steps: int
    phase1_steps_completed: int
    phase2_steps_completed: int
    initial_loss: float
    final_loss: float
    peak_vram_mib: float
    final_checkpoint_dir: str
    metrics_jsonl_path: str
    summary_json_path: str
    noul_calibration_report: Optional[Dict[str, Any]] = None
    choice_calibration_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize training summary to a JSON-compatible dictionary."""
        return asdict(self)


class CurriculumTrainer:
    """Two-Phase OOP Curriculum Trainer for `TradeOffJevNeuralSystem` and `StandardJevBaselineSystem`.

    Manages mixed-precision `bfloat16` CUDA execution, differential LoRA/Head AdamW
    optimization with cosine decay, Phase 1 single-step calibration, Phase 2 Markov
    rollout scheduled sampling, `safetensors` checkpoint persistence, and post-hoc
    temperature calibration.
    """

    def __init__(
        self,
        system: BaseDecisionNeuralSystem,
        config: TrainerConfig,
        criterion: Optional[TorchAATCurriculumLoss] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self._system = system
        self._config = config
        self._criterion = criterion if criterion is not None else TorchAATCurriculumLoss()

        if device is None:
            resolved_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(device)
        self._device = resolved_device

        self._system.to(self._device)
        self._criterion.to(self._device)

        self._rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self._param_groups = self._system.build_optimizer_param_groups(
            lora_lr=config.lora_lr,
            head_lr=config.head_lr,
            weight_decay=config.weight_decay,
        )
        self._optimizer = AdamW(self._param_groups)
        self._scheduler = self._build_cosine_scheduler()

        self._run_dir = Path(config.output_dir) / config.run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self._run_dir / "training_metrics.jsonl"
        self._summary_path = self._run_dir / "training_summary.json"

        self._global_step: int = 0
        self._logged_history: List[Dict[str, Any]] = []

    def _build_cosine_scheduler(self) -> LambdaLR:
        """Create linear-warmup + cosine-decay multiplier applied uniformly to each group's peak LR."""
        total = max(1, self._config.total_steps)
        warmup = int(round(self._config.warmup_ratio * total))
        min_ratio = self._config.min_lr_ratio

        def _lr_multiplier(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            if total <= warmup:
                return 1.0
            progress = min(1.0, max(0.0, float(step - warmup) / float(total - warmup)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        return LambdaLR(self._optimizer, lr_lambda=_lr_multiplier)

    @property
    def system(self) -> BaseDecisionNeuralSystem:
        """Return the neural decision system being trained."""
        return self._system

    @property
    def device(self) -> torch.device:
        """Return the active execution `torch.device`."""
        return self._device

    @property
    def global_step(self) -> int:
        """Return the current completed optimizer step count."""
        return self._global_step

    @property
    def logged_history(self) -> Tuple[Dict[str, Any], ...]:
        """Return immutable view of logged step metric dictionaries."""
        return tuple(self._logged_history)

    def collate_samples(self, samples: Sequence[DecisionStepSample]) -> CollatedStepBatch:
        """Pack and pad a list of `DecisionStepSample`s into a device-resident `CollatedStepBatch`."""
        if not samples:
            raise ValueError("Cannot collate an empty sequence of DecisionStepSample.")

        if getattr(self._system, "prompt_conditioned", False):
            state_texts = [
                format_policy_conditioned_state_text(
                    s.state_text,
                    s.aspiration_prev,
                    s.survival_floor,
                    s.urgency_ranks,
                    s.retreat_index,
                )
                for s in samples
            ]
        else:
            state_texts = [s.state_text for s in samples]
        cand_lists = [list(s.candidate_descriptions) for s in samples]
        packed = self._system.backbone.pack_texts(state_texts, cand_lists)

        batch_size = len(samples)
        max_k = packed.candidate_mask.shape[1]
        num_goals = self._system.aat_config.num_goals

        asp_prev = torch.tensor([s.aspiration_prev for s in samples], dtype=torch.float32)
        s_floor = torch.tensor([s.survival_floor for s in samples], dtype=torch.float32)
        u_ranks = torch.tensor([s.urgency_ranks for s in samples], dtype=torch.float32)
        r_idx = torch.tensor([s.retreat_index for s in samples], dtype=torch.long)
        target_asp = torch.tensor([s.target_aspiration for s in samples], dtype=torch.float32)
        target_bartleby = torch.tensor([s.target_bartleby for s in samples], dtype=torch.float32)

        target_noul = torch.zeros((batch_size, max_k), dtype=torch.float32)
        target_choice = torch.zeros((batch_size,), dtype=torch.long)

        has_all_rubrics = all(s.target_rubric_levels is not None for s in samples)
        target_rubrics = (
            torch.zeros((batch_size, max_k, num_goals), dtype=torch.long)
            if has_all_rubrics
            else None
        )

        for b_idx, sample in enumerate(samples):
            k_b = len(sample.target_noul)
            target_noul[b_idx, :k_b] = torch.tensor(sample.target_noul, dtype=torch.float32)
            # Remap Bartleby refusal index from sample-local k_b to batch-padded slot max_k
            if sample.target_choice >= k_b:
                target_choice[b_idx] = max_k
            else:
                target_choice[b_idx] = sample.target_choice

            if target_rubrics is not None and sample.target_rubric_levels is not None:
                target_rubrics[b_idx, :k_b, :] = torch.tensor(
                    sample.target_rubric_levels, dtype=torch.long
                )

        batch = CollatedStepBatch(
            packed_batch=packed,
            aspiration_prev=asp_prev,
            survival_floor=s_floor,
            urgency_ranks=u_ranks,
            retreat_index=r_idx,
            target_choice=target_choice,
            target_noul=target_noul,
            target_aspiration=target_asp,
            target_bartleby=target_bartleby,
            target_rubric_levels=target_rubrics,
        )
        return batch.to(self._device)

    def _autocast_context(self) -> Any:
        """Return CUDA `bfloat16` autocast context when running on GPU with `use_bf16=True`."""
        use_amp = self._config.use_bf16 and self._device.type == "cuda"
        return torch.amp.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        )

    def _current_group_lrs(self) -> Tuple[float, float]:
        """Extract current `(lora_lr, head_lr)` from optimizer parameter groups."""
        lora_lr = 0.0
        head_lr = 0.0
        for group in self._optimizer.param_groups:
            gname = str(group.get("group_name", ""))
            if gname.startswith("lora") and lora_lr == 0.0:
                lora_lr = float(group["lr"])
            elif gname.startswith("head") and head_lr == 0.0:
                head_lr = float(group["lr"])
        return lora_lr, head_lr

    def _current_peak_vram_mib(self) -> float:
        """Query peak allocated CUDA memory in MiB (`0.0` on CPU)."""
        if self._device.type == "cuda":
            return float(torch.cuda.max_memory_allocated(self._device)) / (1024.0 * 1024.0)
        return 0.0

    def _record_step_metrics(
        self,
        phase: str,
        loss_totals: Mapping[str, float],
        scheduled_sampling_prob: float,
        step_time_ms: float,
    ) -> None:
        """Append structured JSONL metric record for the completed optimizer step."""
        lora_lr, head_lr = self._current_group_lrs()
        entry: Dict[str, Any] = {
            "global_step": self._global_step,
            "phase": phase,
            "loss": round(loss_totals["loss"], 6),
            "loss_ce": round(loss_totals["loss_ce"], 6),
            "loss_rubric": round(loss_totals["loss_rubric"], 6),
            "loss_brier": round(loss_totals["loss_brier"], 6),
            "loss_transition": round(loss_totals["loss_transition"], 6),
            "loss_bartleby": round(loss_totals["loss_bartleby"], 6),
            "lora_lr": round(lora_lr, 8),
            "head_lr": round(head_lr, 8),
            "scheduled_sampling_prob": round(scheduled_sampling_prob, 4),
            "step_time_ms": round(step_time_ms, 2),
            "peak_vram_mib": round(self._current_peak_vram_mib(), 2),
        }
        self._logged_history.append(entry)
        if self._config.log_every_steps > 0 and (
            self._global_step % self._config.log_every_steps == 0
        ):
            with self._metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    def save_checkpoint(self, tag: str) -> Path:
        """Save model LoRA + FP32 head weights and trainer optimizer/scheduler state."""
        ckpt_dir = self._run_dir / tag
        self._system.save_checkpoint(
            ckpt_dir,
            extra_metadata={
                "run_name": self._config.run_name,
                "global_step": self._global_step,
                "tag": tag,
            },
        )
        torch.save(
            {
                "global_step": self._global_step,
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict(),
            },
            ckpt_dir / "trainer_state.pt",
        )
        return ckpt_dir

    def load_checkpoint(self, checkpoint_dir: str | Path) -> Mapping[str, Any]:
        """Restore model weights and optimizer/scheduler state from `checkpoint_dir`."""
        ckpt_path = Path(checkpoint_dir)
        meta = self._system.load_checkpoint_weights(ckpt_path, map_location=self._device)
        trainer_pt = ckpt_path / "trainer_state.pt"
        if trainer_pt.exists():
            state = torch.load(trainer_pt, map_location=self._device, weights_only=True)
            self._global_step = int(state["global_step"])
            self._optimizer.load_state_dict(state["optimizer_state_dict"])
            self._scheduler.load_state_dict(state["scheduler_state_dict"])
        return meta

    def run_phase1_single_step(self, samples: Sequence[DecisionStepSample]) -> int:
        """Execute Phase 1 single-step teacher-forced supervised training."""
        if self._config.phase1_steps == 0:
            return 0
        if not samples:
            raise ValueError("Phase 1 requires at least one DecisionStepSample.")

        self._system.train()
        pool = list(samples)
        cursor = len(pool)  # trigger immediate initial shuffle
        grad_accum = self._config.gradient_accumulation_steps
        steps_done = 0

        for _ in range(self._config.phase1_steps):
            t0 = time.perf_counter()
            self._optimizer.zero_grad(set_to_none=True)
            accum_metrics = {
                "loss": 0.0,
                "loss_ce": 0.0,
                "loss_rubric": 0.0,
                "loss_brier": 0.0,
                "loss_transition": 0.0,
                "loss_bartleby": 0.0,
            }

            for _ in range(grad_accum):
                if cursor + self._config.batch_size > len(pool):
                    self._rng.shuffle(pool)
                    cursor = 0
                micro_samples = pool[cursor : cursor + self._config.batch_size]
                cursor += len(micro_samples)

                batch = self.collate_samples(micro_samples)
                with self._autocast_context():
                    outputs = self._system(
                        packed_batch=batch.packed_batch,
                        aspiration_prev=batch.aspiration_prev,
                        survival_floor=batch.survival_floor,
                        urgency_ranks=batch.urgency_ranks,
                        retreat_index=batch.retreat_index,
                    )
                losses = self._criterion(outputs, batch)
                scaled_loss = losses["loss"] / float(grad_accum)
                scaled_loss.backward()

                for k in accum_metrics:
                    accum_metrics[k] += float(losses[k].detach().item()) / float(grad_accum)

            nn.utils.clip_grad_norm_(self._system.parameters(), self._config.max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()

            self._global_step += 1
            steps_done += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._record_step_metrics(
                phase="phase1_single_step",
                loss_totals=accum_metrics,
                scheduled_sampling_prob=0.0,
                step_time_ms=elapsed_ms,
            )

            if (
                self._config.checkpoint_every_steps > 0
                and self._global_step % self._config.checkpoint_every_steps == 0
            ):
                self.save_checkpoint(f"step_{self._global_step}")

        self.save_checkpoint("phase1_final")
        return steps_done

    def _rollout_trajectory_batch(
        self,
        episodes: Sequence[Sequence[DecisionStepSample]],
        scheduled_sampling_prob: float,
        backward_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unroll a batch of Markov episodes with scheduled sampling on $\\hat{\\vec{a}}_{t-1}$."""
        max_horizon = max(len(ep) for ep in episodes)
        step_losses: List[Dict[str, torch.Tensor]] = []
        active_steps = sum(
            1
            for t_idx in range(max_horizon)
            if any(t_idx < len(ep) for ep in episodes)
        )
        step_denom = float(max(1, active_steps))
        do_step_backward = (
            backward_scale is not None and self._config.detach_rollout_aspiration
        )

        # Track predicted aspiration per episode index across t = 0 .. max_horizon - 1
        carried_aspirations: Dict[int, torch.Tensor] = {}

        for t_idx in range(max_horizon):
            active_ep_indices = [
                ep_i for ep_i, ep in enumerate(episodes) if t_idx < len(ep)
            ]
            if not active_ep_indices:
                continue

            step_samples = [episodes[ep_i][t_idx] for ep_i in active_ep_indices]
            batch = self.collate_samples(step_samples)

            # Apply scheduled sampling at t >= 1 whenever prior predicted aspiration exists
            if t_idx > 0 and carried_aspirations:
                mixed_rows: List[torch.Tensor] = []
                for local_b, ep_i in enumerate(active_ep_indices):
                    teacher_row = batch.aspiration_prev[local_b]
                    if (
                        ep_i in carried_aspirations
                        and self._rng.random() < scheduled_sampling_prob
                    ):
                        pred_row = carried_aspirations[ep_i]
                        if self._config.detach_rollout_aspiration:
                            pred_row = pred_row.detach()
                        mixed_rows.append(pred_row)
                    else:
                        mixed_rows.append(teacher_row)
                aspiration_in = torch.stack(mixed_rows, dim=0)
            else:
                aspiration_in = batch.aspiration_prev

            with self._autocast_context():
                outputs = self._system(
                    packed_batch=batch.packed_batch,
                    aspiration_prev=aspiration_in,
                    survival_floor=batch.survival_floor,
                    urgency_ranks=batch.urgency_ranks,
                    retreat_index=batch.retreat_index,
                )

            losses = self._criterion(outputs, batch)
            if "updated_aspiration" in outputs:
                updated_asp = outputs["updated_aspiration"]
                for local_b, ep_i in enumerate(active_ep_indices):
                    carried_aspirations[ep_i] = (
                        updated_asp[local_b].detach()
                        if self._config.detach_rollout_aspiration
                        else updated_asp[local_b]
                    )

            if do_step_backward:
                scaled_step_loss = losses["loss"] * (float(backward_scale) / step_denom)
                scaled_step_loss.backward()
                step_losses.append({k: v.detach() for k, v in losses.items()})
            else:
                step_losses.append(losses)

        denom = float(max(1, len(step_losses)))
        keys = ("loss", "loss_ce", "loss_rubric", "loss_brier", "loss_transition", "loss_bartleby")
        return {
            k: sum(item[k] for item in step_losses) / denom  # type: ignore[misc]
            for k in keys
        }

    def run_phase2_markov_rollout(
        self,
        trajectories: Sequence[Sequence[DecisionStepSample]],
    ) -> int:
        """Execute Phase 2 multi-step Markov rollout training with scheduled sampling."""
        if self._config.phase2_steps == 0:
            return 0
        valid_trajs = [list(t) for t in trajectories if len(t) > 0]
        if not valid_trajs:
            raise ValueError("Phase 2 requires at least one non-empty Markov trajectory.")

        self._system.train()
        cursor = len(valid_trajs)
        grad_accum = self._config.gradient_accumulation_steps
        steps_done = 0

        for p2_idx in range(self._config.phase2_steps):
            t0 = time.perf_counter()
            progress = (
                float(p2_idx) / float(max(1, self._config.phase2_steps - 1))
                if self._config.phase2_steps > 1
                else 1.0
            )
            ss_prob = self._config.scheduled_sampling_start + progress * (
                self._config.scheduled_sampling_end - self._config.scheduled_sampling_start
            )

            self._optimizer.zero_grad(set_to_none=True)
            accum_metrics = {
                "loss": 0.0,
                "loss_ce": 0.0,
                "loss_rubric": 0.0,
                "loss_brier": 0.0,
                "loss_transition": 0.0,
                "loss_bartleby": 0.0,
            }

            for _ in range(grad_accum):
                if cursor + self._config.trajectory_batch_size > len(valid_trajs):
                    self._rng.shuffle(valid_trajs)
                    cursor = 0
                ep_batch = valid_trajs[cursor : cursor + self._config.trajectory_batch_size]
                cursor += len(ep_batch)

                losses = self._rollout_trajectory_batch(
                    ep_batch,
                    scheduled_sampling_prob=ss_prob,
                    backward_scale=1.0 / float(grad_accum),
                )
                if not self._config.detach_rollout_aspiration:
                    scaled_loss = losses["loss"] / float(grad_accum)
                    scaled_loss.backward()

                for k in accum_metrics:
                    accum_metrics[k] += float(losses[k].detach().item()) / float(grad_accum)

            nn.utils.clip_grad_norm_(self._system.parameters(), self._config.max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()

            self._global_step += 1
            steps_done += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._record_step_metrics(
                phase="phase2_markov_rollout",
                loss_totals=accum_metrics,
                scheduled_sampling_prob=ss_prob,
                step_time_ms=elapsed_ms,
            )

            if (
                self._config.checkpoint_every_steps > 0
                and self._global_step % self._config.checkpoint_every_steps == 0
            ):
                self.save_checkpoint(f"step_{self._global_step}")

        self.save_checkpoint("phase2_final")
        return steps_done

    @torch.no_grad()
    def calibrate_on_split(
        self,
        cal_samples: Sequence[DecisionStepSample],
    ) -> Tuple[CalibrationReport, CalibrationReport]:
        """Run post-hoc temperature scaling on `cal_samples` for both `tradeoff_noul` and `choice`.

        Evaluates the trained system in `eval()` mode, fits `TemperatureCalibrator` (15 ECE
        bins, L-BFGS + Log-Grid NLL) on `conditioned_logits` and `choice_logits`
        (evaluating fixed candidate-count groups cleanly), and persists JSON artifacts
        under `checkpoints/<run_name>/calibration/`.
        """
        if not cal_samples:
            raise ValueError("cal_samples must be non-empty for temperature calibration.")

        self._system.eval()
        all_noul_logits: List[torch.Tensor] = []
        all_noul_targets: List[torch.Tensor] = []
        all_noul_masks: List[torch.Tensor] = []

        # Pad choice logits across potentially mixed candidate counts to common max_C
        raw_choice_logits: List[torch.Tensor] = []
        raw_choice_targets: List[torch.Tensor] = []
        raw_cand_counts: List[int] = []

        for start in range(0, len(cal_samples), self._config.batch_size):
            chunk = cal_samples[start : start + self._config.batch_size]
            batch = self.collate_samples(chunk)
            with self._autocast_context():
                outputs = self._system(
                    packed_batch=batch.packed_batch,
                    aspiration_prev=batch.aspiration_prev,
                    survival_floor=batch.survival_floor,
                    urgency_ranks=batch.urgency_ranks,
                    retreat_index=batch.retreat_index,
                )
            all_noul_logits.append(outputs["conditioned_logits"].detach().cpu().reshape(-1))
            all_noul_targets.append(batch.target_noul.detach().cpu().reshape(-1))
            all_noul_masks.append(batch.packed_batch.candidate_mask.detach().cpu().reshape(-1))

            c_logits = outputs["choice_logits"].detach().cpu()  # [B, K_chunk + 1]
            c_targets = batch.target_choice.detach().cpu()
            raw_choice_logits.append(c_logits)
            raw_choice_targets.append(c_targets)
            raw_cand_counts.append(c_logits.shape[-1] - 1)

        noul_calibrator = TemperatureCalibrator(mode=CalibrationMode.BINARY_NOUL, num_bins=15)
        noul_report = noul_calibrator.fit(
            logits=torch.cat(all_noul_logits, dim=0),
            targets=torch.cat(all_noul_targets, dim=0),
            mask=torch.cat(all_noul_masks, dim=0),
        )

        # Align choice_logits to global max_K + 1 with Bartleby always in the final slot
        global_max_k = max(raw_cand_counts)
        aligned_choice_logits: List[torch.Tensor] = []
        aligned_choice_targets: List[torch.Tensor] = []
        for c_logits, c_targets, k_chunk in zip(
            raw_choice_logits, raw_choice_targets, raw_cand_counts
        ):
            bsz = c_logits.shape[0]
            padded = torch.full((bsz, global_max_k + 1), -1e9, dtype=torch.float32)
            padded[:, :k_chunk] = c_logits[:, :k_chunk]
            padded[:, global_max_k] = c_logits[:, k_chunk]
            remapped_t = torch.where(
                c_targets == k_chunk,
                torch.full_like(c_targets, global_max_k),
                c_targets,
            )
            aligned_choice_logits.append(padded)
            aligned_choice_targets.append(remapped_t)

        choice_calibrator = TemperatureCalibrator(
            mode=CalibrationMode.MULTICLASS_CHOICE, num_bins=15
        )
        choice_report = choice_calibrator.fit(
            logits=torch.cat(aligned_choice_logits, dim=0),
            targets=torch.cat(aligned_choice_targets, dim=0),
        )

        cal_dir = self._run_dir / "calibration"
        noul_calibrator.save(cal_dir / "noul_calibrator.json")
        choice_calibrator.save(cal_dir / "choice_calibrator.json")
        return noul_report, choice_report

    def train(
        self,
        phase1_samples: Sequence[DecisionStepSample],
        phase2_trajectories: Optional[Sequence[Sequence[DecisionStepSample]]] = None,
        cal_samples: Optional[Sequence[DecisionStepSample]] = None,
    ) -> TrainingRunSummary:
        """Run complete Phase 1 + Phase 2 curriculum training and optional calibration."""
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        if self._metrics_path.exists() and self._config.phase1_steps > 0:
            self._metrics_path.unlink()

        p1_done = self.run_phase1_single_step(phase1_samples)
        p2_done = 0
        if phase2_trajectories is not None and self._config.phase2_steps > 0:
            p2_done = self.run_phase2_markov_rollout(phase2_trajectories)

        noul_rep_dict: Optional[Dict[str, Any]] = None
        choice_rep_dict: Optional[Dict[str, Any]] = None
        if cal_samples:
            noul_rep, choice_rep = self.calibrate_on_split(cal_samples)
            noul_rep_dict = noul_rep.to_dict()
            choice_rep_dict = choice_rep.to_dict()

        final_tag = "phase2_final" if p2_done > 0 else "phase1_final"
        final_ckpt = self._run_dir / final_tag
        if not final_ckpt.exists():
            final_ckpt = self.save_checkpoint("final")

        init_loss = self._logged_history[0]["loss"] if self._logged_history else 0.0
        fin_loss = self._logged_history[-1]["loss"] if self._logged_history else 0.0
        repo_root = Path(__file__).resolve().parents[1]

        def _rel(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(repo_root))
            except ValueError:
                return str(p)

        summary = TrainingRunSummary(
            run_name=self._config.run_name,
            system_type=self._system.system_type,
            device=str(self._device),
            total_optimizer_steps=self._global_step,
            phase1_steps_completed=p1_done,
            phase2_steps_completed=p2_done,
            initial_loss=float(init_loss),
            final_loss=float(fin_loss),
            peak_vram_mib=round(self._current_peak_vram_mib(), 2),
            final_checkpoint_dir=_rel(final_ckpt),
            metrics_jsonl_path=_rel(self._metrics_path),
            summary_json_path=_rel(self._summary_path),
            noul_calibration_report=noul_rep_dict,
            choice_calibration_report=choice_rep_dict,
        )
        with self._summary_path.open("w", encoding="utf-8", newline="\n") as sf:
            sf.write(json.dumps(summary.to_dict(), indent=2) + "\n")
        return summary
