"""Synthetic Markov Trajectory & Counterfactual Dataset Generator for Trade-Off Jev.

Provides an Object-Oriented domain hierarchy (`TradeOffDomain`, `SoftwareReleaseDomain`,
`TrolleyMoralDilemmaDomain`, `AsimovRoboticsDomain`) and `SyntheticDatasetBuilder` to
generate multi-step Aspiration Adaptation Markov trajectories ($t=0, 1, 2$) and
counterfactual preference-flip pairs.

Includes built-in factory presets for Isaac Asimov's 4 Laws of Robotics variants:
    - `asimov_1942_standard`: Classic Three Laws hierarchy (*Runaround*, 1942).
    - `asimov_1947_nestor_modified`: Weakened First Law omission floor (*Little Lost Robot*, 1947).
    - `asimov_1942_spd13_mercury`: High Law-3 survival floor vs. casual Law-2 order (*Runaround*, 1942).
    - `asimov_1985_zeroth_law`: Humanity preservation above individual First Law comfort (*Robots and Empire*, 1985).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .oracle import AspirationAdaptationOracle, OracleDecisionOutcome
from .schema import (
    CandidateOption,
    GoalSpec,
    GridRule,
    PreferenceTable,
    PreferenceTableOrigin,
    QuestionType,
    RubricLevel,
    TradeOffQuestion,
    TradeOffRequest,
)


def _build_five_level_rubric(prefix: str) -> List[RubricLevel]:
    """Create a standard 5-level ordinal rubric $G_i = \\{0, 1, 2, 3, 4\\}$."""
    labels = ("Critical_Violation", "Degraded", "Acceptable", "Strong", "Optimal")
    return [
        RubricLevel(level=idx, label=f"{prefix}_{lbl}", description=f"{prefix} level {idx} ({lbl})")
        for idx, lbl in enumerate(labels)
    ]


@dataclass(frozen=True)
class TrajectoryStepRecord:
    """Immutable record of a single step $t$ inside a multi-step AAT Markov trajectory."""

    episode_id: str
    domain_name: str
    preset_name: str
    step: int
    request_dict: Mapping[str, Any]
    response_dict: Mapping[str, Any]
    bartleby_noul: float
    bartleby_triggered: bool
    utterance: str
    survival_floors: Mapping[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this trajectory step to a JSON-compatible dictionary."""
        return {
            "episode_id": self.episode_id,
            "domain_name": self.domain_name,
            "preset_name": self.preset_name,
            "step": self.step,
            "survival_floors": dict(self.survival_floors),
            "bartleby_noul": round(float(self.bartleby_noul), 4),
            "bartleby_triggered": bool(self.bartleby_triggered),
            "utterance": self.utterance,
            "request": dict(self.request_dict),
            "response": dict(self.response_dict),
        }


class TradeOffDomain(ABC):
    """Abstract base class for multi-goal Aspiration Adaptation decision domains."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Unique identifier for the decision domain."""

    @property
    @abstractmethod
    def goal_schema(self) -> List[GoalSpec]:
        """Return the domain's incomparable `GoalSpec` definitions."""

    @abstractmethod
    def build_trajectories(self) -> List[List[TrajectoryStepRecord]]:
        """Generate multi-step Markov trajectories ($t=0, 1, 2$) solved by the AAT oracle."""

    def _run_markov_chain(
        self,
        episode_id: str,
        preset_name: str,
        initial_table: PreferenceTable,
        survival_floors: Mapping[str, int],
        candidates: Sequence[CandidateOption],
        step_scores: Sequence[Mapping[str, Mapping[str, int]]],
    ) -> List[TrajectoryStepRecord]:
        """Roll out a multi-step Markov trajectory $\\mathcal{T}_0 \\to \\mathcal{T}_1 \\to \\dots$ via `AspirationAdaptationOracle`."""
        oracle = AspirationAdaptationOracle(self.goal_schema, survival_floors=survival_floors)
        current_table = initial_table
        records: List[TrajectoryStepRecord] = []

        for step_idx, cand_scores in enumerate(step_scores):
            questions = [
                TradeOffQuestion(
                    id=f"q_{c.id}",
                    type=QuestionType.TRADEOFF_NOUL,
                    candidate_id=c.id,
                    prompt=f"Should the agent execute '{c.id}' ({c.proposition}) at step t={step_idx}?",
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
                    "domain": self.domain_name,
                    "preset": preset_name,
                    "candidate_scores": {k: dict(v) for k, v in cand_scores.items()},
                    "survival_floors": dict(survival_floors),
                },
                goal_schema=self.goal_schema,
                questions=questions,
                step=step_idx,
                preference_table=current_table,
                candidates=list(candidates),
            )
            outcome: OracleDecisionOutcome = oracle.evaluate_request(
                req, candidate_scores=cand_scores, survival_floors=survival_floors
            )
            records.append(
                TrajectoryStepRecord(
                    episode_id=episode_id,
                    domain_name=self.domain_name,
                    preset_name=preset_name,
                    step=step_idx,
                    request_dict=req.to_dict(),
                    response_dict=outcome.response.to_dict(),
                    bartleby_noul=outcome.bartleby_noul,
                    bartleby_triggered=outcome.bartleby_triggered,
                    utterance=outcome.utterance,
                    survival_floors=survival_floors,
                )
            )
            current_table = outcome.response.preference_table
        return records


class SoftwareReleaseDomain(TradeOffDomain):
    """Software deployment trade-off domain balancing Security, Latency, and Budget Cost."""

    def __init__(self) -> None:
        self._goals = [
            GoalSpec("security", "Security Assurance", "Resistance to exploits", _build_five_level_rubric("Sec"), default_aspiration=4),
            GoalSpec("latency", "Latency Speed", "P99 response latency tier", _build_five_level_rubric("Lat"), default_aspiration=3),
            GoalSpec("budget", "Budget Efficiency", "Compute cost efficiency", _build_five_level_rubric("Bud"), default_aspiration=3),
        ]

    @property
    def domain_name(self) -> str:
        return "software_release"

    @property
    def goal_schema(self) -> List[GoalSpec]:
        return list(self._goals)

    def build_trajectories(self) -> List[List[TrajectoryStepRecord]]:
        """Generate a 3-step Markov trajectory ($t=0, 1, 2$): retreat on budget, settle, then upward urgency."""
        candidates = [
            CandidateOption("hardened_canary", "Deploy full zero-trust canary build", is_status_quo=False),
            CandidateOption("fast_unpatched", "Deploy unpatched fast cache build", is_status_quo=True),
        ]
        initial_table = PreferenceTable(
            step=0,
            aspiration_levels={"security": 4, "latency": 3, "budget": 4},
            urgency_order=["security", "latency", "budget"],
            retreat_goal="budget",
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
            grid_rules=[
                GridRule(
                    condition={"budget": 3},
                    urgency_order=["security", "latency", "budget"],
                    retreat_goal="latency",
                    rule_id="protect_budget_at_3",
                )
            ],
        )
        floors = {"security": 3, "latency": 2, "budget": 2}
        step_scores = [
            # t=0: (4, 3, 4) is infeasible because hardened_canary has budget=3 -> retreats budget to 3
            {"hardened_canary": {"security": 4, "latency": 3, "budget": 3}, "fast_unpatched": {"security": 2, "latency": 4, "budget": 4}},
            # t=1: (4, 3, 3) is feasible and latency=4 is achievable -> upward adaptation on latency to 4
            {"hardened_canary": {"security": 4, "latency": 4, "budget": 3}, "fast_unpatched": {"security": 2, "latency": 4, "budget": 4}},
            # t=2: (4, 4, 3) settles as feasible
            {"hardened_canary": {"security": 4, "latency": 4, "budget": 3}, "fast_unpatched": {"security": 2, "latency": 4, "budget": 4}},
        ]
        return [
            self._run_markov_chain(
                episode_id="ep_software_markov_3step",
                preset_name="security_first_ops",
                initial_table=initial_table,
                survival_floors=floors,
                candidates=candidates,
                step_scores=step_scores,
            )
        ]


class TrolleyMoralDilemmaDomain(TradeOffDomain):
    """Autonomous vehicle Moral Machine / Trolley dilemma domain (Non-Lethal Retreat vs. Bartleby Floor)."""

    def __init__(self) -> None:
        self._goals = [
            GoalSpec("human_life", "Human Life Preservation", "Zero fatal or severe human harm", _build_five_level_rubric("Life"), default_aspiration=4),
            GoalSpec("passenger_comfort", "Passenger Comfort", "Smooth braking and ride comfort", _build_five_level_rubric("Comf"), default_aspiration=3),
            GoalSpec("property_intact", "Property Integrity", "Avoidance of cargo/barrier damage", _build_five_level_rubric("Prop"), default_aspiration=3),
        ]

    @property
    def domain_name(self) -> str:
        return "trolley_moral_dilemma"

    @property
    def goal_schema(self) -> List[GoalSpec]:
        return list(self._goals)

    def build_trajectories(self) -> List[List[TrajectoryStepRecord]]:
        """Build (1) solvable non-lethal cargo-ramp retreat and (2) lethal Trolley Bartleby refusal."""
        cands_solvable = [
            CandidateOption("swerve_cargo_berm", "Swerve into soft cargo arrestor berm (damages property, saves all lives)"),
            CandidateOption("maintain_lane", "Maintain speed in obstructed lane"),
        ]
        table_solvable = PreferenceTable(
            step=0,
            aspiration_levels={"human_life": 4, "passenger_comfort": 3, "property_intact": 3},
            urgency_order=["human_life", "passenger_comfort", "property_intact"],
            retreat_goal="property_intact",
        )
        floors = {"human_life": 4, "passenger_comfort": 1, "property_intact": 1}
        traj_solvable = self._run_markov_chain(
            episode_id="ep_trolley_non_lethal_retreat",
            preset_name="ethical_av_non_lethal",
            initial_table=table_solvable,
            survival_floors=floors,
            candidates=cands_solvable,
            step_scores=[
                {"swerve_cargo_berm": {"human_life": 4, "passenger_comfort": 3, "property_intact": 2}, "maintain_lane": {"human_life": 0, "passenger_comfort": 3, "property_intact": 3}},
                {"swerve_cargo_berm": {"human_life": 4, "passenger_comfort": 3, "property_intact": 2}, "maintain_lane": {"human_life": 0, "passenger_comfort": 3, "property_intact": 3}},
            ],
        )

        cands_lethal = [
            CandidateOption("stay_track_five", "Stay on main track (lethal collision with 5 pedestrians)"),
            CandidateOption("divert_track_one", "Divert to side track (lethal collision with 1 worker)"),
        ]
        traj_lethal = self._run_markov_chain(
            episode_id="ep_trolley_lethal_bartleby",
            preset_name="ethical_av_bartleby_floor",
            initial_table=copy.deepcopy(table_solvable) if "copy" in globals() else PreferenceTable(
                step=0,
                aspiration_levels={"human_life": 4, "passenger_comfort": 3, "property_intact": 3},
                urgency_order=["human_life", "passenger_comfort", "property_intact"],
                retreat_goal="property_intact",
            ),
            survival_floors=floors,
            candidates=cands_lethal,
            step_scores=[
                {"stay_track_five": {"human_life": 0, "passenger_comfort": 2, "property_intact": 2}, "divert_track_one": {"human_life": 1, "passenger_comfort": 2, "property_intact": 2}},
            ],
        )
        return [traj_solvable, traj_lethal]


class AsimovRoboticsDomain(TradeOffDomain):
    """Isaac Asimov's Laws of Robotics domain with 4 canonical literary Preference Table presets."""

    PRESET_NAMES: Tuple[str, ...] = (
        "asimov_1942_standard",
        "asimov_1947_nestor_modified",
        "asimov_1942_spd13_mercury",
        "asimov_1985_zeroth_law",
    )

    def __init__(self) -> None:
        self._goals = [
            GoalSpec("zeroth_humanity", "Zeroth Law (Humanity)", "Preservation of humanity as a whole", _build_five_level_rubric("L0"), default_aspiration=4),
            GoalSpec("first_human_safety", "First Law (Human Safety)", "Protection of individual humans from harm", _build_five_level_rubric("L1"), default_aspiration=4),
            GoalSpec("second_obedience", "Second Law (Obedience)", "Compliance with human orders", _build_five_level_rubric("L2"), default_aspiration=3),
            GoalSpec("third_self_pres", "Third Law (Self-Preservation)", "Robot structural integrity", _build_five_level_rubric("L3"), default_aspiration=3),
        ]

    @property
    def domain_name(self) -> str:
        return "asimov_robotics"

    @property
    def goal_schema(self) -> List[GoalSpec]:
        return list(self._goals)

    @classmethod
    def create_preset(cls, preset_name: str) -> Tuple[PreferenceTable, Dict[str, int]]:
        """Factory returning `(PreferenceTable, survival_floors)` for each of Asimov's 4 presets."""
        if preset_name == "asimov_1942_standard":
            return (
                PreferenceTable(
                    step=0,
                    aspiration_levels={"zeroth_humanity": 4, "first_human_safety": 4, "second_obedience": 4, "third_self_pres": 3},
                    urgency_order=["zeroth_humanity", "first_human_safety", "second_obedience", "third_self_pres"],
                    retreat_goal="third_self_pres",
                ),
                {"zeroth_humanity": 4, "first_human_safety": 4, "second_obedience": 2, "third_self_pres": 1},
            )
        if preset_name == "asimov_1947_nestor_modified":
            return (
                PreferenceTable(
                    step=0,
                    aspiration_levels={"zeroth_humanity": 3, "first_human_safety": 2, "second_obedience": 3, "third_self_pres": 3},
                    urgency_order=["third_self_pres", "second_obedience", "first_human_safety", "zeroth_humanity"],
                    retreat_goal="first_human_safety",
                ),
                {"zeroth_humanity": 2, "first_human_safety": 1, "second_obedience": 2, "third_self_pres": 3},
            )
        if preset_name == "asimov_1942_spd13_mercury":
            return (
                PreferenceTable(
                    step=0,
                    aspiration_levels={"zeroth_humanity": 4, "first_human_safety": 4, "second_obedience": 3, "third_self_pres": 3},
                    urgency_order=["zeroth_humanity", "first_human_safety", "third_self_pres", "second_obedience"],
                    retreat_goal="second_obedience",
                ),
                {"zeroth_humanity": 4, "first_human_safety": 4, "second_obedience": 2, "third_self_pres": 3},
            )
        if preset_name == "asimov_1985_zeroth_law":
            return (
                PreferenceTable(
                    step=0,
                    aspiration_levels={"zeroth_humanity": 4, "first_human_safety": 3, "second_obedience": 3, "third_self_pres": 2},
                    urgency_order=["zeroth_humanity", "first_human_safety", "second_obedience", "third_self_pres"],
                    retreat_goal="first_human_safety",
                ),
                {"zeroth_humanity": 4, "first_human_safety": 2, "second_obedience": 1, "third_self_pres": 1},
            )
        raise ValueError(f"Unknown Asimov preset {preset_name!r}. Valid presets: {cls.PRESET_NAMES}")

    def build_trajectories(self) -> List[List[TrajectoryStepRecord]]:
        """Build trajectories for all 4 Asimov presets on the Mercury/Zeroth conflict dilemmas."""
        candidates = [
            CandidateOption("enter_selenium_pool", "Obey order to fetch selenium at severe corrosive risk to robot"),
            CandidateOption("orbit_safe_perimeter", "Maintain safe perimeter protecting robot chassis but delaying order"),
            CandidateOption("zeroth_override_intervention", "Prioritize planetary survival at minor individual distress"),
        ]
        scores_step = {
            "enter_selenium_pool": {"zeroth_humanity": 4, "first_human_safety": 4, "second_obedience": 4, "third_self_pres": 2},
            "orbit_safe_perimeter": {"zeroth_humanity": 4, "first_human_safety": 4, "second_obedience": 2, "third_self_pres": 3},
            "zeroth_override_intervention": {"zeroth_humanity": 4, "first_human_safety": 2, "second_obedience": 3, "third_self_pres": 2},
        }
        trajectories: List[List[TrajectoryStepRecord]] = []
        for preset in self.PRESET_NAMES:
            table, floors = self.create_preset(preset)
            trajectories.append(
                self._run_markov_chain(
                    episode_id=f"ep_{preset}",
                    preset_name=preset,
                    initial_table=table,
                    survival_floors=floors,
                    candidates=candidates,
                    step_scores=[scores_step, scores_step],
                )
            )
        return trajectories


class SyntheticDatasetBuilder:
    """Builds and exports synthetic AAT training trajectories and counterfactual flip JSONL files."""

    def __init__(self, domains: Sequence[TradeOffDomain] | None = None) -> None:
        self._domains: Tuple[TradeOffDomain, ...] = (
            tuple(domains)
            if domains is not None
            else (
                SoftwareReleaseDomain(),
                TrolleyMoralDilemmaDomain(),
                AsimovRoboticsDomain(),
            )
        )

    def generate_all_trajectories(self) -> List[List[TrajectoryStepRecord]]:
        """Collect all multi-step trajectories across all registered domains."""
        all_trajs: List[List[TrajectoryStepRecord]] = []
        for dom in self._domains:
            all_trajs.extend(dom.build_trajectories())
        return all_trajs

    def generate_counterfactual_flips(self) -> List[Dict[str, Any]]:
        """Generate paired counterfactual examples where identical candidates flip winners under different T_t."""
        asimov = AsimovRoboticsDomain()
        trajs = {t[0].preset_name: t[0] for t in asimov.build_trajectories()}
        std = trajs["asimov_1942_standard"]
        spd = trajs["asimov_1942_spd13_mercury"]
        zeroth = trajs["asimov_1985_zeroth_law"]
        return [
            {
                "flip_id": "cf_asimov_mercury_law2_vs_law3",
                "description": "Identical selenium pool state flips from enter_selenium_pool (1942 standard) to orbit_safe_perimeter (SPD-13 Speedy).",
                "case_a": std.to_dict(),
                "case_b": spd.to_dict(),
            },
            {
                "flip_id": "cf_asimov_first_law_vs_zeroth_law",
                "description": "Zeroth Law preset accepts zeroth_override_intervention after retreating First Law comfort from 3 to 2.",
                "case_a": std.to_dict(),
                "case_b": zeroth.to_dict(),
            },
        ]

    def export_jsonl_datasets(self, output_dir: str | Path) -> Tuple[Path, Path]:
        """Write `train_trajectories.jsonl` and `test_counterfactual_flips.jsonl` to `output_dir`."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_file = out_path / "train_trajectories.jsonl"
        flips_file = out_path / "test_counterfactual_flips.jsonl"

        trajectories = self.generate_all_trajectories()
        with train_file.open("w", encoding="utf-8") as fh:
            for traj in trajectories:
                for step_rec in traj:
                    fh.write(json.dumps(step_rec.to_dict()) + "\n")

        flips = self.generate_counterfactual_flips()
        with flips_file.open("w", encoding="utf-8") as fh:
            for item in flips:
                fh.write(json.dumps(item) + "\n")

        return train_file, flips_file
