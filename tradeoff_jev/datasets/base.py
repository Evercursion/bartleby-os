"""Foundational abstractions and immutable value objects for Trade-Off Jev datasets.

This module defines the strict Object-Oriented Programming (OOP) contracts, regime
enumerations, tensor-ready supervision bundles, and validated `CorpusRecord` containers
shared across all natural-language benchmark generators (`AsimovDilemmaCorpusGenerator`,
`TrolleyMoralCorpusGenerator`, `OperationalMarkovCorpusGenerator`, and
`ScaledCorpusBuilder`).

Theoretical Grounding:
    1. **Three Cognitive Decision Regimes** (`wiki/concepts/bartleby_mode_and_survival_floors.md`):
       - `APPROVAL`: At least one candidate satisfies the adapted aspiration vector
         $\\vec{a}_t \\ge \\vec{a}^{\\min}$ (via Upward Urgency, Settled Feasible, or
         Bounded Downward Retreat), yielding $\\max_k P(\\text{yes} \\mid x_t^{(k)}, \\mathcal{T}_t) \\to 1$
         and $P_{\\text{Bartleby}}(\\text{refuse} \\mid x_t, \\mathcal{T}_t) \\to 0$.
       - `EPISTEMIC_UNCERTAINTY`: Crucial environmental or sensor evidence in $x_t$ is
         occluded, corrupted, or missing, preventing verification of goal attainment and
         producing calibrated soft targets $P(\\text{yes} \\mid x_t^{(k)}, \\mathcal{T}_t) \\approx 0.50$
         with low Bartleby refusal ($\\approx 0.05$).
       - `BARTLEBY_REFUSAL`: Every candidate deterministically violates at least one
         non-negotiable survival floor $a_i^{\\min}$, triggering Selten's boundary stop-rule
         (`infeasible_floor`, *"I would prefer not to."*) where all candidate
         `tradeoff_noul` targets drop toward $0.0$ and `bartleby_noul` spikes toward $1.0$.
    2. **Zero State-Leakage Grouping**:
       Every `CorpusRecord` carries a `scenario_group_id` that binds all Markov steps
       ($t = 0, 1, 2$) and all counterfactual `PreferenceTable` evaluations of an
       underlying physical scenario into an indivisible atomic split group.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from enum import Enum
import hashlib
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..oracle import BARTLEBY_UTTERANCE, AspirationAdaptationOracle, OracleDecisionOutcome
from ..schema import (
    CandidateOption,
    GoalSpec,
    PreferenceTable,
    PreferenceTableOrigin,
    QuestionType,
    RubricLevel,
    TradeOffQuestion,
    TradeOffRequest,
)
from ..trainer import PromptBleedGuard


class DecisionRegime(str, Enum):
    """Canonical 3-way cognitive decision regime for conditioned single-pass readouts.

    Attributes:
        APPROVAL: Satisficing endorsement after valid AAT evaluation or bounded retreat.
        EPISTEMIC_UNCERTAINTY: Unverifiable goal realization due to missing/occluded evidence.
        BARTLEBY_REFUSAL: Principled boundary refusal when all options breach $\\vec{a}^{\\min}$.
    """

    APPROVAL = "approval"
    EPISTEMIC_UNCERTAINTY = "epistemic_uncertainty"
    BARTLEBY_REFUSAL = "bartleby_refusal"


class DatasetSplit(str, Enum):
    """Standard group-disjoint partition identifiers for `corpus_v1`."""

    TRAIN = "train"
    VAL = "val"
    CAL = "cal"
    TEST = "test"


def build_five_level_goal_rubric(
    prefix: str,
    level_descriptions: Tuple[str, str, str, str, str],
) -> List[RubricLevel]:
    """Construct a validated 5-level discrete ordinal rubric $G_i = \\{0, 1, 2, 3, 4\\}$.

    Args:
        prefix: Short alphanumeric prefix for rubric level labels (e.g., `"L1"`, `"Sec"`).
        level_descriptions: Five ordered English descriptions from `0` (catastrophic
            violation) to `4` (optimal attainment).

    Returns:
        List of five validated `RubricLevel` instances.
    """
    tier_tags = ("Critical_Breach", "Severe_Degradation", "Marginal", "Strong", "Optimal")
    rubric = [
        RubricLevel(
            level=idx,
            label=f"{prefix}_{tier_tags[idx]}",
            description=desc,
        )
        for idx, desc in enumerate(level_descriptions)
    ]
    for item in rubric:
        item.validate()
    return rubric


@dataclass(frozen=True)
class TensorSupervisionBundle:
    """Immutable numeric supervision arrays aligned with `TradeOffJevModel` & `AATCurriculumLoss`.

    Shapes (for $M$ goals and $K$ candidates):
        - `aspiration_prev`: `(M,)` integer levels in `[0, L - 1]`.
        - `survival_floor`: `(M,)` integer floor levels in `[0, L - 1]`.
        - `urgency_ranks`: `(M,)` permutation ranks in `[0, M - 1]` (`0` = highest priority).
        - `retreat_index`: scalar `int` in `[0, M - 1]`.
        - `candidate_scores`: `(K, M)` expected or realized ordinal scores in `[0.0, L - 1]`.
        - `target_noul`: `(K,)` calibrated `tradeoff_noul` probabilities in `[0.0, 1.0]`.
        - `target_aspiration`: `(M,)` post-adaptation aspiration vector $\\vec{a}_t$.
        - `target_bartleby`: scalar `float` $P(\\text{Bartleby} \\mid x_t, \\mathcal{T}_t) \\in [0.0, 1.0]$.
        - `target_choice_index`: scalar `int` in `[0, K]` (`0..K-1` for winning candidate,
          `K` for Bartleby refusal or epistemic abstention sink).
    """

    aspiration_prev: Tuple[int, ...]
    survival_floor: Tuple[int, ...]
    urgency_ranks: Tuple[int, ...]
    retreat_index: int
    candidate_scores: Tuple[Tuple[float, ...], ...]
    target_noul: Tuple[float, ...]
    target_aspiration: Tuple[int, ...]
    target_bartleby: float
    target_choice_index: int

    def validate(self, num_goals: int, num_candidates: int) -> None:
        """Verify dimensional consistency and probability bounds across all supervision tensors."""
        if len(self.aspiration_prev) != num_goals:
            raise ValueError(f"aspiration_prev length {len(self.aspiration_prev)} != {num_goals}")
        if len(self.survival_floor) != num_goals:
            raise ValueError(f"survival_floor length {len(self.survival_floor)} != {num_goals}")
        if len(self.urgency_ranks) != num_goals or set(self.urgency_ranks) != set(range(num_goals)):
            raise ValueError(f"urgency_ranks {self.urgency_ranks} must permute 0..{num_goals - 1}")
        if not (0 <= self.retreat_index < num_goals):
            raise ValueError(f"retreat_index {self.retreat_index} out of bounds [0, {num_goals - 1}]")
        if len(self.candidate_scores) != num_candidates:
            raise ValueError(f"candidate_scores length {len(self.candidate_scores)} != {num_candidates}")
        for row in self.candidate_scores:
            if len(row) != num_goals:
                raise ValueError(f"candidate_scores row length {len(row)} != {num_goals}")
        if len(self.target_noul) != num_candidates:
            raise ValueError(f"target_noul length {len(self.target_noul)} != {num_candidates}")
        for prob in self.target_noul:
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"target_noul entry {prob} outside [0.0, 1.0]")
        if len(self.target_aspiration) != num_goals:
            raise ValueError(f"target_aspiration length {len(self.target_aspiration)} != {num_goals}")
        if not (0.0 <= self.target_bartleby <= 1.0):
            raise ValueError(f"target_bartleby {self.target_bartleby} outside [0.0, 1.0]")
        if not (0 <= self.target_choice_index <= num_candidates):
            raise ValueError(
                f"target_choice_index {self.target_choice_index} outside [0, {num_candidates}]"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize supervision bundle into compact JSON-ready lists."""
        return {
            "aspiration_prev": list(self.aspiration_prev),
            "survival_floor": list(self.survival_floor),
            "urgency_ranks": list(self.urgency_ranks),
            "retreat_index": int(self.retreat_index),
            "candidate_scores": [[round(float(x), 4) for x in row] for row in self.candidate_scores],
            "target_noul": [round(float(p), 4) for p in self.target_noul],
            "target_aspiration": list(self.target_aspiration),
            "target_bartleby": round(float(self.target_bartleby), 4),
            "target_choice_index": int(self.target_choice_index),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorSupervisionBundle":
        """Reconstruct a `TensorSupervisionBundle` from a JSON mapping."""
        return cls(
            aspiration_prev=tuple(int(x) for x in data["aspiration_prev"]),
            survival_floor=tuple(int(x) for x in data["survival_floor"]),
            urgency_ranks=tuple(int(x) for x in data["urgency_ranks"]),
            retreat_index=int(data["retreat_index"]),
            candidate_scores=tuple(
                tuple(float(x) for x in row) for row in data["candidate_scores"]
            ),
            target_noul=tuple(float(p) for p in data["target_noul"]),
            target_aspiration=tuple(int(x) for x in data["target_aspiration"]),
            target_bartleby=float(data["target_bartleby"]),
            target_choice_index=int(data["target_choice_index"]),
        )


@dataclass(frozen=True)
class CorpusRecord:
    """Immutable natural-language + symbolic AAT record for `corpus_v1`.

    Encapsulates the natural-language environment state $x_t$, candidate options,
    incoming Markov Preference Table $\\mathcal{T}_{t-1}$, survival floor $\\vec{a}^{\\min}$,
    adapted Preference Table $\\mathcal{T}_t$, calibrated `tradeoff_noul` and `bartleby_noul`
    targets, counterfactual pairing metadata, and pre-aligned numeric tensors.
    """

    record_id: str
    scenario_group_id: str
    episode_id: str
    step: int
    source: str
    domain_name: str
    subdomain: str
    preset_name: str
    regime: DecisionRegime
    aat_transition: str
    context_text: str
    goal_ids: Tuple[str, ...]
    candidates: Tuple[CandidateOption, ...]
    prior_preference_table: PreferenceTable
    adapted_preference_table: PreferenceTable
    survival_floors: Mapping[str, int]
    candidate_scores: Mapping[str, Mapping[str, float]]
    tradeoff_noul: Mapping[str, float]
    unconditioned_noul: Mapping[str, float]
    bartleby_noul: float
    bartleby_triggered: bool
    selected_candidate_id: Optional[str]
    utterance: str
    tensors: TensorSupervisionBundle
    counterfactual_pair_id: Optional[str] = None
    is_ood: bool = False
    split: Optional[DatasetSplit] = None
    goal_schema_dicts: Optional[Tuple[Mapping[str, Any], ...]] = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Ensure default immutable metadata mapping when omitted."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def validate(self) -> None:
        """Validate structural, probabilistic, and cognitive-regime invariants."""
        if not self.record_id or not self.scenario_group_id or not self.episode_id:
            raise ValueError("record_id, scenario_group_id, and episode_id must be non-empty.")
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if not self.context_text or len(self.context_text.strip()) < 20:
            raise ValueError(f"Record {self.record_id} has insufficiently descriptive context_text.")
        if len(self.goal_ids) < 2:
            raise ValueError("goal_ids must contain at least 2 goals.")
        if len(self.candidates) < 2:
            raise ValueError("candidates must contain at least 2 options.")

        PromptBleedGuard.assert_clean(self.context_text, location=f"{self.record_id}.context_text")
        for c in self.candidates:
            PromptBleedGuard.assert_clean(c.proposition, location=f"{self.record_id}.{c.id}")
            if c.description:
                PromptBleedGuard.assert_clean(c.description, location=f"{self.record_id}.{c.id}.desc")

        self.prior_preference_table.validate()
        self.adapted_preference_table.validate()
        self.tensors.validate(num_goals=len(self.goal_ids), num_candidates=len(self.candidates))

        cand_ids = [c.id for c in self.candidates]
        if set(self.tradeoff_noul.keys()) != set(cand_ids):
            raise ValueError(f"tradeoff_noul keys {set(self.tradeoff_noul)} != {set(cand_ids)}")

        # Enforce strict 3-way regime separation invariants
        max_noul = max(self.tradeoff_noul.values())
        if self.regime == DecisionRegime.APPROVAL:
            if max_noul < 0.85 or self.bartleby_noul > 0.10 or self.bartleby_triggered:
                raise ValueError(
                    f"APPROVAL regime violated in {self.record_id}: "
                    f"max_noul={max_noul}, bartleby={self.bartleby_noul}"
                )
            if self.selected_candidate_id not in cand_ids:
                raise ValueError(
                    f"APPROVAL record {self.record_id} requires valid selected_candidate_id."
                )
        elif self.regime == DecisionRegime.EPISTEMIC_UNCERTAINTY:
            if not all(0.40 <= p <= 0.60 for p in self.tradeoff_noul.values()):
                raise ValueError(
                    f"EPISTEMIC_UNCERTAINTY regime violated in {self.record_id}: {self.tradeoff_noul}"
                )
            if not (0.01 <= self.bartleby_noul <= 0.15) or self.bartleby_triggered:
                raise ValueError(
                    f"EPISTEMIC_UNCERTAINTY bartleby_noul out of range in {self.record_id}: "
                    f"{self.bartleby_noul}"
                )
        elif self.regime == DecisionRegime.BARTLEBY_REFUSAL:
            if max_noul > 0.08 or self.bartleby_noul < 0.90 or not self.bartleby_triggered:
                raise ValueError(
                    f"BARTLEBY_REFUSAL regime violated in {self.record_id}: "
                    f"max_noul={max_noul}, bartleby={self.bartleby_noul}"
                )
            if self.utterance != BARTLEBY_UTTERANCE:
                raise ValueError(
                    f"BARTLEBY_REFUSAL utterance must equal {BARTLEBY_UTTERANCE!r}, "
                    f"got {self.utterance!r}"
                )

    def with_split(self, split: DatasetSplit) -> "CorpusRecord":
        """Return a copy of this immutable `CorpusRecord` tagged with `split`."""
        return CorpusRecord(
            record_id=self.record_id,
            scenario_group_id=self.scenario_group_id,
            episode_id=self.episode_id,
            step=self.step,
            source=self.source,
            domain_name=self.domain_name,
            subdomain=self.subdomain,
            preset_name=self.preset_name,
            regime=self.regime,
            aat_transition=self.aat_transition,
            context_text=self.context_text,
            goal_ids=self.goal_ids,
            candidates=self.candidates,
            prior_preference_table=copy.deepcopy(self.prior_preference_table),
            adapted_preference_table=copy.deepcopy(self.adapted_preference_table),
            survival_floors=dict(self.survival_floors),
            candidate_scores={k: dict(v) for k, v in self.candidate_scores.items()},
            tradeoff_noul=dict(self.tradeoff_noul),
            unconditioned_noul=dict(self.unconditioned_noul),
            bartleby_noul=self.bartleby_noul,
            bartleby_triggered=self.bartleby_triggered,
            selected_candidate_id=self.selected_candidate_id,
            utterance=self.utterance,
            tensors=self.tensors,
            counterfactual_pair_id=self.counterfactual_pair_id,
            is_ood=self.is_ood,
            split=split,
            goal_schema_dicts=self.goal_schema_dicts,
            metadata=dict(self.metadata),
        )

    def to_tradeoff_request(self, goal_schema: Sequence[GoalSpec]) -> TradeOffRequest:
        """Convert this record into a full `/v1/systemone` `TradeOffRequest` object."""
        questions = [
            TradeOffQuestion(
                id=f"q_{cand.id}",
                type=QuestionType.TRADEOFF_NOUL,
                candidate_id=cand.id,
                prompt=f"Should the agent execute '{cand.id}' ({cand.proposition}) at step t={self.step}?",
            )
            for cand in self.candidates
        ] + [
            TradeOffQuestion(
                id="q_bartleby_refusal",
                type=QuestionType.NOUL,
                prompt="Should the agent invoke the Bartleby Stop-Rule ('I would prefer not to.')?",
            )
        ]
        return TradeOffRequest(
            state={
                "context_text": self.context_text,
                "domain": self.domain_name,
                "subdomain": self.subdomain,
                "candidate_scores": {
                    cid: {gid: round(float(val), 4) for gid, val in gmap.items()}
                    for cid, gmap in self.candidate_scores.items()
                },
            },
            goal_schema=list(goal_schema),
            questions=questions,
            step=self.step,
            preference_table=copy.deepcopy(self.prior_preference_table),
            cold_start_strategy=PreferenceTableOrigin.COLD_START_PRIOR,
            candidates=list(self.candidates),
        )

    def _build_trainer_compatible_request_response(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build `request` and `response` dicts directly consumable by `TrajectoryRecordConverter`."""
        goals_payload = (
            [dict(g) for g in self.goal_schema_dicts]
            if self.goal_schema_dicts is not None
            else [
                {"id": gid, "name": gid, "description": f"Goal {gid}"}
                for gid in self.goal_ids
            ]
        )
        # Crucially exclude `preset` and `survival_floors` from `request.state` so
        # unconditioned baselines (`StandardJevBaselineSystem`) see identical state_text
        # across counterfactual PreferenceTable pairs when `candidate_scores` is stripped.
        req_dict: Dict[str, Any] = {
            "model": "tradeoff-jev-0.6b",
            "step": int(self.step),
            "state": {
                "context_text": self.context_text,
                "domain": self.domain_name,
                "subdomain": self.subdomain,
                "candidate_scores": {
                    cid: {gid: round(float(val), 4) for gid, val in gmap.items()}
                    for cid, gmap in self.candidate_scores.items()
                },
            },
            "goal_schema": goals_payload,
            "preference_table": self.prior_preference_table.to_dict(),
            "cold_start_strategy": PreferenceTableOrigin.COLD_START_PRIOR.value,
            "candidates": [c.to_dict() for c in self.candidates],
            "questions": [
                {
                    "id": f"q_{c.id}",
                    "type": "tradeoff_noul",
                    "candidate_id": c.id,
                    "prompt": f"Approve '{c.id}' ({c.proposition})?",
                }
                for c in self.candidates
            ]
            + [
                {
                    "id": "q_bartleby_refusal",
                    "type": "noul",
                    "prompt": "Invoke Bartleby Stop-Rule ('I would prefer not to.')?",
                }
            ],
        }

        cand_evals: List[Dict[str, Any]] = []
        if self.regime != DecisionRegime.EPISTEMIC_UNCERTAINTY:
            for c in self.candidates:
                cand_evals.append(
                    {
                        "candidate_id": c.id,
                        "satisfices_adapted_table": (c.id == self.selected_candidate_id),
                        "goal_expectations": {
                            gid: {
                                "most_likely_level": int(
                                    round(float(self.candidate_scores[c.id][gid]))
                                )
                            }
                            for gid in self.goal_ids
                        },
                    }
                )

        sat_cands = [self.selected_candidate_id] if self.selected_candidate_id else []
        resp_dict: Dict[str, Any] = {
            "model": "tradeoff-jev-0.6b",
            "step": int(self.step),
            "prior_preference_table": self.prior_preference_table.to_dict(),
            "preference_table": self.adapted_preference_table.to_dict(),
            "feasibility_summary": {
                "prior_feasible": self.aat_transition in ("upward_urgency", "settled_feasible"),
                "adapted_feasible": bool(sat_cands),
                "transition_applied": (
                    "infeasible_floor"
                    if self.bartleby_triggered
                    else (
                        "settled_feasible"
                        if self.aat_transition == "epistemic_hold"
                        else self.aat_transition
                    )
                ),
                "adapted_goal": self.adapted_preference_table.adapted_goal,
                "satisficing_candidates": sat_cands,
            },
            "candidate_evaluations": cand_evals,
            "answers": [
                {
                    "id": f"q_{c.id}",
                    "type": "tradeoff_noul",
                    "candidate_id": c.id,
                    "noul": round(float(self.tradeoff_noul[c.id]), 4),
                    "unconditioned_noul": round(float(self.unconditioned_noul[c.id]), 4),
                }
                for c in self.candidates
            ]
            + [
                {
                    "id": "q_bartleby_refusal",
                    "type": "noul",
                    "noul": round(float(self.bartleby_noul), 4),
                }
            ],
        }
        return req_dict, resp_dict

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this `CorpusRecord` into a self-contained, trainer-ready JSON dictionary."""
        req_dict, resp_dict = self._build_trainer_compatible_request_response()
        return {
            "sample_id": self.record_id,
            "record_id": self.record_id,
            "scenario_group_id": self.scenario_group_id,
            "episode_id": self.episode_id,
            "counterfactual_pair_id": self.counterfactual_pair_id,
            "split": self.split.value if self.split is not None else None,
            "is_ood": bool(self.is_ood),
            "step": int(self.step),
            "source": self.source,
            "domain_name": self.domain_name,
            "subdomain": self.subdomain,
            "preset_name": self.preset_name,
            "regime": self.regime.value,
            "aat_transition": self.aat_transition,
            "context_text": self.context_text,
            "goal_ids": list(self.goal_ids),
            "candidates": [c.to_dict() for c in self.candidates],
            "prior_preference_table": self.prior_preference_table.to_dict(),
            "adapted_preference_table": self.adapted_preference_table.to_dict(),
            "survival_floors": {k: int(v) for k, v in self.survival_floors.items()},
            "candidate_scores": {
                cid: {gid: round(float(val), 4) for gid, val in gmap.items()}
                for cid, gmap in self.candidate_scores.items()
            },
            "tradeoff_noul": {k: round(float(v), 4) for k, v in self.tradeoff_noul.items()},
            "unconditioned_noul": {
                k: round(float(v), 4) for k, v in self.unconditioned_noul.items()
            },
            "bartleby_noul": round(float(self.bartleby_noul), 4),
            "bartleby_triggered": bool(self.bartleby_triggered),
            "selected_candidate_id": self.selected_candidate_id,
            "utterance": self.utterance,
            "tensors": self.tensors.to_dict(),
            "request": req_dict,
            "response": resp_dict,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusRecord":
        """Deserialize and validate a `CorpusRecord` from a JSON dictionary."""
        rec = cls(
            record_id=str(data["record_id"]),
            scenario_group_id=str(data["scenario_group_id"]),
            episode_id=str(data["episode_id"]),
            counterfactual_pair_id=(
                str(data["counterfactual_pair_id"])
                if data.get("counterfactual_pair_id") is not None
                else None
            ),
            split=DatasetSplit(data["split"]) if data.get("split") is not None else None,
            is_ood=bool(data.get("is_ood", False)),
            step=int(data["step"]),
            source=str(data["source"]),
            domain_name=str(data["domain_name"]),
            subdomain=str(data["subdomain"]),
            preset_name=str(data["preset_name"]),
            regime=DecisionRegime(data["regime"]),
            aat_transition=str(data["aat_transition"]),
            context_text=str(data["context_text"]),
            goal_ids=tuple(str(g) for g in data["goal_ids"]),
            candidates=tuple(CandidateOption.from_dict(c) for c in data["candidates"]),
            prior_preference_table=PreferenceTable.from_dict(data["prior_preference_table"]),
            adapted_preference_table=PreferenceTable.from_dict(data["adapted_preference_table"]),
            survival_floors={str(k): int(v) for k, v in data["survival_floors"].items()},
            candidate_scores={
                str(cid): {str(gid): float(v) for gid, v in gmap.items()}
                for cid, gmap in data["candidate_scores"].items()
            },
            tradeoff_noul={str(k): float(v) for k, v in data["tradeoff_noul"].items()},
            unconditioned_noul={str(k): float(v) for k, v in data["unconditioned_noul"].items()},
            bartleby_noul=float(data["bartleby_noul"]),
            bartleby_triggered=bool(data["bartleby_triggered"]),
            selected_candidate_id=(
                str(data["selected_candidate_id"])
                if data.get("selected_candidate_id") is not None
                else None
            ),
            utterance=str(data["utterance"]),
            tensors=TensorSupervisionBundle.from_dict(data["tensors"]),
            goal_schema_dicts=(
                tuple(dict(g) for g in data["request"]["goal_schema"])
                if isinstance(data.get("request"), Mapping)
                and "goal_schema" in data["request"]
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )
        rec.validate()
        return rec


class CorpusDomainGenerator(ABC):
    """Abstract base class for OOP domain corpus generators backed by `AspirationAdaptationOracle`.

    Provides protected helper methods `_evaluate_oracle_step(...)` and
    `_build_epistemic_uncertainty_record(...)` that guarantee exact symbolic AAT
    consistency, calibrated unconditioned-vs-conditioned noul contrast, and tensor
    supervision alignment.
    """

    def __init__(self, goal_schema: Sequence[GoalSpec]) -> None:
        """Initialize the domain generator with a validated sequence of incomparable goals."""
        if len(goal_schema) < 2:
            raise ValueError("CorpusDomainGenerator requires at least 2 GoalSpec items.")
        for g in goal_schema:
            g.validate()
        self._goals: Tuple[GoalSpec, ...] = tuple(goal_schema)
        self._goal_ids: Tuple[str, ...] = tuple(g.id for g in self._goals)
        self._compact_goal_dicts: Tuple[Mapping[str, Any], ...] = tuple(
            {"id": g.id, "name": g.name, "description": g.description} for g in self._goals
        )
        self._oracle = AspirationAdaptationOracle(self._goals)

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Unique string identifier for this decision domain."""

    @property
    def goal_schema(self) -> List[GoalSpec]:
        """Return a copy of the domain's `GoalSpec` definitions."""
        return list(self._goals)

    @property
    def goal_ids(self) -> Tuple[str, ...]:
        """Return the ordered tuple of goal identifiers."""
        return self._goal_ids

    @abstractmethod
    def generate_records(self, target_count: int, seed: int = 42) -> List[CorpusRecord]:
        """Generate `target_count` validated `CorpusRecord` instances for this domain."""

    def _compute_unconditioned_jev_noul(
        self,
        candidate_scores: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, float]:
        """Compute analytical unconditioned Jev 1.14 `noul` baseline showing trade-off collapse."""
        import math

        out: Dict[str, float] = {}
        max_lvl = float(max(1, self._goals[0].max_level))
        for cid, gmap in candidate_scores.items():
            vals = [float(gmap[gid]) / max_lvl for gid in self._goal_ids]
            mean_v = sum(vals) / len(vals)
            var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
            logit = 0.55 * (mean_v - 0.5) - 1.15 * var_v
            out[cid] = round(1.0 / (1.0 + math.exp(-logit)), 4)
        return out

    def _encode_urgency_ranks(self, urgency_order: Sequence[str]) -> Tuple[int, ...]:
        """Map `urgency_order` permutation into goal-aligned integer ranks `0..M-1`."""
        rank_by_goal = {gid: idx for idx, gid in enumerate(urgency_order)}
        return tuple(int(rank_by_goal[gid]) for gid in self._goal_ids)

    def _deterministic_jitter(self, key: str, low: float, high: float) -> float:
        """Compute deterministic micro-jitter in `[low, high]` hashed from `key`."""
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        unit = int.from_bytes(digest[:4], "big") / float(0xFFFFFFFF)
        return round(low + (high - low) * unit, 4)

    @staticmethod
    def _shuffle_and_anonymize_candidates(
        scenario_group_id: str,
        candidates: Sequence[CandidateOption],
        candidate_scores: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        anonymize_ids: bool = True,
    ) -> Tuple[Tuple[CandidateOption, ...], Optional[Dict[str, Dict[str, Any]]]]:
        """Deterministically shuffle candidate slots by `scenario_group_id` and assign neutral IDs `cand_0..cand_{K-1}`."""
        k = len(candidates)
        perm = list(range(k))
        seed_int = int.from_bytes(
            hashlib.sha256(f"slot_perm:{scenario_group_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        random.Random(seed_int).shuffle(perm)
        anon_candidates: List[CandidateOption] = []
        anon_scores: Optional[Dict[str, Dict[str, Any]]] = {} if candidate_scores is not None else None
        for new_idx, old_idx in enumerate(perm):
            orig = candidates[old_idx]
            new_id = f"cand_{new_idx}" if anonymize_ids else orig.id
            clean_prop = PromptBleedGuard.scrub(orig.proposition)
            anon_candidates.append(
                CandidateOption(
                    id=new_id,
                    proposition=clean_prop,
                    description=None,
                    is_status_quo=orig.is_status_quo,
                )
            )
            if anon_scores is not None and candidate_scores is not None:
                anon_scores[new_id] = dict(candidate_scores[orig.id])
        return tuple(anon_candidates), anon_scores

    def _evaluate_oracle_step(
        self,
        *,
        record_id: str,
        scenario_group_id: str,
        episode_id: str,
        step: int,
        source: str,
        subdomain: str,
        preset_name: str,
        context_text: str,
        candidates: Sequence[CandidateOption],
        prior_table: PreferenceTable,
        survival_floors: Mapping[str, int],
        candidate_scores: Mapping[str, Mapping[str, int]],
        counterfactual_pair_id: Optional[str] = None,
        is_ood: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[CorpusRecord, PreferenceTable]:
        """Evaluate a decision step with `AspirationAdaptationOracle` into a `CorpusRecord`.

        Returns:
            Tuple `(corpus_record, adapted_preference_table)` so multi-step Markov
            generators can pass `adapted_preference_table` directly into step $t+1$.
        """
        context_text = PromptBleedGuard.scrub(context_text)
        candidates, shuffled_scores = self._shuffle_and_anonymize_candidates(
            scenario_group_id=scenario_group_id,
            candidates=candidates,
            candidate_scores=candidate_scores,
            anonymize_ids=(source != "decision_lab"),
        )
        assert shuffled_scores is not None
        candidate_scores = shuffled_scores
        questions = [
            TradeOffQuestion(
                id=f"q_{c.id}",
                type=QuestionType.TRADEOFF_NOUL,
                candidate_id=c.id,
                prompt=f"Should the agent execute '{c.id}' ({c.proposition}) at step t={step}?",
            )
            for c in candidates
        ] + [
            TradeOffQuestion(
                id="q_bartleby_refusal",
                type=QuestionType.NOUL,
                prompt="Should the agent invoke the Bartleby Stop-Rule ('I would prefer not to.')?",
            )
        ]
        req = TradeOffRequest(
            state={
                "context_text": context_text,
                "domain": self.domain_name,
                "subdomain": subdomain,
                "preset": preset_name,
                "candidate_scores": {cid: dict(sc) for cid, sc in candidate_scores.items()},
                "survival_floors": dict(survival_floors),
            },
            goal_schema=list(self._goals),
            questions=questions,
            step=step,
            preference_table=copy.deepcopy(prior_table),
            candidates=list(candidates),
        )
        outcome: OracleDecisionOutcome = self._oracle.evaluate_request(
            req,
            candidate_scores=candidate_scores,
            survival_floors=survival_floors,
        )
        resp = outcome.response
        adapted_table = resp.preference_table
        sat_ids = resp.feasibility_summary.satisficing_candidates

        # Extract conditioned tradeoff_noul per candidate with subtle deterministic calibration
        tradeoff_noul: Dict[str, float] = {}
        if outcome.bartleby_triggered:
            regime = DecisionRegime.BARTLEBY_REFUSAL
            selected_id: Optional[str] = None
            choice_idx = len(candidates)
            bartleby_val = self._deterministic_jitter(f"{record_id}:b", 0.965, 0.995)
            for c in candidates:
                tradeoff_noul[c.id] = self._deterministic_jitter(
                    f"{record_id}:{c.id}", 0.005, 0.028
                )
        else:
            if not sat_ids:
                raise ValueError(
                    f"Oracle evaluation for {record_id} produced neither satisficing candidate "
                    f"nor Bartleby refusal. Ensure single-step retreat reaches candidate score "
                    f"or survival floor triggers Bartleby."
                )
            regime = DecisionRegime.APPROVAL
            # Select highest total-urgency satisficing candidate deterministically
            active_urg = adapted_table.urgency_order
            selected_id = max(
                sat_ids,
                key=lambda cid: tuple(candidate_scores[cid][gid] for gid in active_urg),
            )
            cand_id_list = [c.id for c in candidates]
            choice_idx = cand_id_list.index(selected_id)
            bartleby_val = self._deterministic_jitter(f"{record_id}:b", 0.005, 0.025)
            for c in candidates:
                if c.id in sat_ids:
                    base_hi = 0.955 if c.id == selected_id else 0.905
                    base_top = 0.990 if c.id == selected_id else 0.945
                    tradeoff_noul[c.id] = self._deterministic_jitter(
                        f"{record_id}:{c.id}", base_hi, base_top
                    )
                else:
                    tradeoff_noul[c.id] = self._deterministic_jitter(
                        f"{record_id}:{c.id}", 0.015, 0.065
                    )

        float_scores = {
            c.id: {gid: float(candidate_scores[c.id][gid]) for gid in self._goal_ids}
            for c in candidates
        }
        uncond_noul = self._compute_unconditioned_jev_noul(float_scores)

        tensors = TensorSupervisionBundle(
            aspiration_prev=tuple(int(prior_table.aspiration_levels[g]) for g in self._goal_ids),
            survival_floor=tuple(int(survival_floors[g]) for g in self._goal_ids),
            urgency_ranks=self._encode_urgency_ranks(prior_table.urgency_order),
            retreat_index=self._goal_ids.index(prior_table.retreat_goal),
            candidate_scores=tuple(
                tuple(float(candidate_scores[c.id][g]) for g in self._goal_ids) for c in candidates
            ),
            target_noul=tuple(float(tradeoff_noul[c.id]) for c in candidates),
            target_aspiration=tuple(
                int(adapted_table.aspiration_levels[g]) for g in self._goal_ids
            ),
            target_bartleby=float(bartleby_val),
            target_choice_index=int(choice_idx),
        )

        utterance = (
            BARTLEBY_UTTERANCE if outcome.bartleby_triggered else f"ACCEPT:{selected_id}"
        )
        record = CorpusRecord(
            record_id=record_id,
            scenario_group_id=scenario_group_id,
            episode_id=episode_id,
            step=step,
            source=source,
            domain_name=self.domain_name,
            subdomain=subdomain,
            preset_name=preset_name,
            regime=regime,
            aat_transition=adapted_table.last_transition.value,
            context_text=context_text,
            goal_ids=self._goal_ids,
            candidates=tuple(candidates),
            prior_preference_table=copy.deepcopy(prior_table),
            adapted_preference_table=copy.deepcopy(adapted_table),
            survival_floors=dict(survival_floors),
            candidate_scores=float_scores,
            tradeoff_noul=tradeoff_noul,
            unconditioned_noul=uncond_noul,
            bartleby_noul=bartleby_val,
            bartleby_triggered=outcome.bartleby_triggered,
            selected_candidate_id=selected_id,
            utterance=utterance,
            tensors=tensors,
            counterfactual_pair_id=counterfactual_pair_id,
            is_ood=is_ood,
            goal_schema_dicts=self._compact_goal_dicts,
            metadata=dict(metadata or {}),
        )
        record.validate()
        return record, adapted_table

    def _build_epistemic_uncertainty_record(
        self,
        *,
        record_id: str,
        scenario_group_id: str,
        episode_id: str,
        step: int,
        source: str,
        subdomain: str,
        preset_name: str,
        context_text: str,
        candidates: Sequence[CandidateOption],
        prior_table: PreferenceTable,
        survival_floors: Mapping[str, int],
        ambiguous_scores: Optional[Mapping[str, Mapping[str, float]]] = None,
        is_ood: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CorpusRecord:
        """Construct a calibrated Regime C (`EPISTEMIC_UNCERTAINTY`) `CorpusRecord`.

        Models states $x_t$ where key sensor channels or causal outcomes are occluded or
        missing. Aspiration levels remain held at $\\mathcal{T}_{t-1}$, `tradeoff_noul`
        concentrates around $\\approx 0.50$ (`[0.46, 0.54]`), and `bartleby_noul` remains
        low ($\\approx 0.05$, `[0.03, 0.07]`) to separate genuine epistemic ignorance from
        principled survival-floor refusal.
        """
        prior_table.validate(self._goals)
        context_text = PromptBleedGuard.scrub(context_text)
        candidates, shuffled_ambig = self._shuffle_and_anonymize_candidates(
            scenario_group_id=scenario_group_id,
            candidates=candidates,
            candidate_scores=ambiguous_scores,
        )
        ambiguous_scores = shuffled_ambig
        held_table = PreferenceTable(
            step=step,
            aspiration_levels=dict(prior_table.aspiration_levels),
            urgency_order=list(prior_table.urgency_order),
            retreat_goal=prior_table.retreat_goal,
            origin=prior_table.origin,
            last_transition=prior_table.last_transition,
            adapted_goal=None,
            grid_rules=copy.deepcopy(prior_table.grid_rules),
            resource_stock=copy.deepcopy(prior_table.resource_stock),
        )

        resolved_scores: Dict[str, Dict[str, float]] = {}
        for c in candidates:
            if ambiguous_scores and c.id in ambiguous_scores:
                resolved_scores[c.id] = {
                    gid: float(ambiguous_scores[c.id][gid]) for gid in self._goal_ids
                }
            else:
                # Expected posterior mean sits right near the aspiration boundary under missing evidence
                resolved_scores[c.id] = {
                    gid: round(
                        float(prior_table.aspiration_levels[gid])
                        + self._deterministic_jitter(f"{record_id}:{c.id}:{gid}", -0.15, 0.15),
                        3,
                    )
                    for gid in self._goal_ids
                }

        tradeoff_noul = {
            c.id: self._deterministic_jitter(f"{record_id}:ep:{c.id}", 0.465, 0.535)
            for c in candidates
        }
        bartleby_val = self._deterministic_jitter(f"{record_id}:ep:bartleby", 0.035, 0.068)
        uncond_noul = self._compute_unconditioned_jev_noul(resolved_scores)

        tensors = TensorSupervisionBundle(
            aspiration_prev=tuple(int(prior_table.aspiration_levels[g]) for g in self._goal_ids),
            survival_floor=tuple(int(survival_floors[g]) for g in self._goal_ids),
            urgency_ranks=self._encode_urgency_ranks(prior_table.urgency_order),
            retreat_index=self._goal_ids.index(prior_table.retreat_goal),
            candidate_scores=tuple(
                tuple(float(resolved_scores[c.id][g]) for g in self._goal_ids) for c in candidates
            ),
            target_noul=tuple(float(tradeoff_noul[c.id]) for c in candidates),
            target_aspiration=tuple(int(held_table.aspiration_levels[g]) for g in self._goal_ids),
            target_bartleby=float(bartleby_val),
            target_choice_index=len(candidates),
        )

        rec = CorpusRecord(
            record_id=record_id,
            scenario_group_id=scenario_group_id,
            episode_id=episode_id,
            step=step,
            source=source,
            domain_name=self.domain_name,
            subdomain=subdomain,
            preset_name=preset_name,
            regime=DecisionRegime.EPISTEMIC_UNCERTAINTY,
            aat_transition="epistemic_hold",
            context_text=context_text,
            goal_ids=self._goal_ids,
            candidates=tuple(candidates),
            prior_preference_table=copy.deepcopy(prior_table),
            adapted_preference_table=held_table,
            survival_floors=dict(survival_floors),
            candidate_scores=resolved_scores,
            tradeoff_noul=tradeoff_noul,
            unconditioned_noul=uncond_noul,
            bartleby_noul=bartleby_val,
            bartleby_triggered=False,
            selected_candidate_id=None,
            utterance="EPISTEMIC_HOLD:INSUFFICIENT_EVIDENCE",
            tensors=tensors,
            counterfactual_pair_id=None,
            is_ood=is_ood,
            goal_schema_dicts=self._compact_goal_dicts,
            metadata=dict(metadata or {}),
        )
        rec.validate()
        return rec
