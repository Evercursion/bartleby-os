"""Formal data structures and API serialization for Trade-Off Jev.

Implements the Aspiration Adaptation Theory (AAT; Simon 1957, Sauermann & Selten 1962,
Selten 1998/1999) Preference Table Markov state (T_t), SystemOne/Jev-compatible
request/response primitives (`noul`, `choice`, `score`, `tradeoff_noul`), and
Architecture A compilation to/from coupled Clef/Kev Joint-Schema questions
(Group 1 Goal & Adaptation primitives + Group 2 Conditioned Action Nouls).

All classes follow strict Object-Oriented Design (OOP) principles with private/protected
validation helpers, immutable or validated domain transitions, complete type annotations,
and zero external library dependencies beyond the Python standard library.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


class PreferenceTableOrigin(str, Enum):
    """Provenance of a PreferenceTable (T_t) across the AAT Markov chain.

    Attributes:
        COLD_START_PRIOR: Initialized at t=0 from normative application priors or rubric midpoints.
        ZERO_SHOT_INFERRED: Inferred at t=0 from raw context x_0 via score/ranking readouts (Selten 1999, Sec 2.7).
        STATUS_QUO_ANCHORED: Anchored at t=0 to the realized goal levels of the status-quo option (Selten 1999, Sec 2.4).
        MARKOV_ADAPTED: Carried over from step t-1 and updated via Selten's Upward, Downward, or End rules.
    """

    COLD_START_PRIOR = "cold_start_prior"
    ZERO_SHOT_INFERRED = "zero_shot_inferred"
    STATUS_QUO_ANCHORED = "status_quo_anchored"
    MARKOV_ADAPTED = "markov_adapted"


class AdaptationTransition(str, Enum):
    """Selten AAT transition rule applied during the Markov update T_{t-1} -> T_t (Selten 1998, Sec 2.2).

    Attributes:
        INITIAL: Cold-start initialization at step t=0 without prior aspiration carry-over.
        UPWARD_URGENCY: Upward Rule applied along the urgency order pi_t because a_{t-1} and a_{t-1} + e_i were feasible.
        DOWNWARD_RETREAT: Downward Rule applied to the retreat variable r_t because a_{t-1} was recognizably infeasible.
        SETTLED_FEASIBLE: End Rule reached where a_{t-1} is feasible but no single-goal upward step is feasible.
        INFEASIBLE_FLOOR: Infeasible even at the minimum grid origin (0, ..., 0); no further retreat is possible.
    """

    INITIAL = "initial"
    UPWARD_URGENCY = "upward_urgency"
    DOWNWARD_RETREAT = "downward_retreat"
    SETTLED_FEASIBLE = "settled_feasible"
    INFEASIBLE_FLOOR = "infeasible_floor"


class QualitativeExpectation(str, Enum):
    """Qualitative goal expectation relative to current aspiration/reference (Selten 1999, Sec 2.4).

    In Selten's Routine Model, boundedly rational decision makers form directional
    expectations in {+, -, 0} for each goal variable relative to the previous period's
    reference level.
    """

    POSITIVE = "+"
    NEGATIVE = "-"
    NEUTRAL = "0"

    @classmethod
    def from_comparison(
        cls,
        realized_level: float,
        reference_level: float,
        tol: float = 1e-6,
    ) -> "QualitativeExpectation":
        """Derive qualitative direction (+, -, 0) comparing an expected level against a reference level.

        Args:
            realized_level: Expected or realized ordinal level for goal g_i.
            reference_level: Reference aspiration or prior realized level for goal g_i.
            tol: Numerical tolerance for equality comparison.

        Returns:
            QualitativeExpectation.POSITIVE ('+'), NEGATIVE ('-'), or NEUTRAL ('0').
        """
        if realized_level > reference_level + tol:
            return cls.POSITIVE
        if realized_level < reference_level - tol:
            return cls.NEGATIVE
        return cls.NEUTRAL


class QuestionType(str, Enum):
    """Supported SystemOne / Trade-Off Jev question primitives."""

    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"
    TRADEOFF_NOUL = "tradeoff_noul"


@dataclass(frozen=True)
class RubricLevel:
    """Discrete ordinal aspiration/realization level (0 .. L-1) for a GoalSpec.

    Maps continuous or qualitative domain outcomes onto Selten's discrete aspiration
    grid G_i = {0, 1, ..., L_i - 1} (Selten 1998, Section 2.1).
    """

    level: int
    label: str
    description: str

    def validate(self) -> None:
        """Validate domain invariants for this discrete rubric level."""
        if not isinstance(self.level, int) or self.level < 0:
            raise ValueError(f"RubricLevel.level must be a non-negative integer, got {self.level!r}")
        if not self.label or not self.label.strip():
            raise ValueError("RubricLevel.label must be a non-empty string")
        if not self.description or not self.description.strip():
            raise ValueError("RubricLevel.description must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this RubricLevel into a JSON-compatible dictionary."""
        return {
            "level": self.level,
            "label": self.label,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RubricLevel":
        """Construct and validate a RubricLevel from a dictionary mapping."""
        obj = cls(
            level=int(data["level"]),
            label=str(data["label"]),
            description=str(data["description"]),
        )
        obj.validate()
        return obj


@dataclass
class GoalSpec:
    """Formal specification of an incommensurable goal variable g_i in G = (g_1, ..., g_m).

    Carries an intrinsic non-negotiable survival floor (`default_survival_floor`, defaulting
    to level 1 via `effective_survival_floor` whenever `max_level >= 1`) so that Bartleby
    refusal is active by architectural default even when the caller omits explicit floors.
    """

    id: str
    name: str
    description: str
    rubric: List[RubricLevel]
    is_resource_stock: bool = False
    default_aspiration: Optional[int] = None
    default_survival_floor: Optional[int] = None

    @property
    def num_levels(self) -> int:
        return len(self.rubric)

    @property
    def max_level(self) -> int:
        return len(self.rubric) - 1

    @property
    def effective_survival_floor(self) -> int:
        """Return the intrinsic minimum acceptable rubric level $a_i^{\\min} \\ge 1$ for this goal."""
        if self.default_survival_floor is not None:
            return int(self.default_survival_floor)
        return min(1, self.max_level)

    def validate(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("GoalSpec.id must be a non-empty string")
        if not self.name or not self.name.strip():
            raise ValueError(f"GoalSpec({self.id}).name must be a non-empty string")
        if not self.description or not self.description.strip():
            raise ValueError(f"GoalSpec({self.id}).description must be a non-empty string")
        if len(self.rubric) < 2:
            raise ValueError(f"GoalSpec({self.id}).rubric must contain at least 2 levels")
        for idx, item in enumerate(self.rubric):
            item.validate()
            if item.level != idx:
                raise ValueError(
                    f"GoalSpec({self.id}).rubric levels must be contiguous 0..{len(self.rubric) - 1}, "
                    f"found level={item.level} at index {idx}"
                )
        if self.default_aspiration is not None:
            if not (0 <= self.default_aspiration <= self.max_level):
                raise ValueError(
                    f"GoalSpec({self.id}).default_aspiration={self.default_aspiration} "
                    f"out of range [0, {self.max_level}]"
                )
        if self.default_survival_floor is not None:
            if not (0 <= self.default_survival_floor <= self.max_level):
                raise ValueError(
                    f"GoalSpec({self.id}).default_survival_floor={self.default_survival_floor} "
                    f"out of range [0, {self.max_level}]"
                )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rubric": [r.to_dict() for r in self.rubric],
            "is_resource_stock": bool(self.is_resource_stock),
        }
        if self.default_aspiration is not None:
            result["default_aspiration"] = int(self.default_aspiration)
        if self.default_survival_floor is not None:
            result["default_survival_floor"] = int(self.default_survival_floor)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalSpec":
        rubric = [RubricLevel.from_dict(r) for r in data["rubric"]]
        obj = cls(
            id=str(data["id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            rubric=rubric,
            is_resource_stock=bool(data.get("is_resource_stock", False)),
            default_aspiration=int(data["default_aspiration"]) if data.get("default_aspiration") is not None else None,
            default_survival_floor=(
                int(data["default_survival_floor"])
                if data.get("default_survival_floor") is not None
                else None
            ),
        )
        obj.validate()
        return obj


# Alias matching JSON Schema $defs.GoalDefinition
GoalDefinition = GoalSpec


@dataclass
class GridRule:
    """Grid-dependent urgency and retreat shift (Selten 1998, Section 2.1).

    Maps an aspiration grid point or region `condition` (goal_id -> exact int level
    or {"min": int, "max": int}) to a local `urgency_order` and `retreat_goal`.
    """

    condition: Dict[str, Union[int, Dict[str, int]]]
    urgency_order: List[str]
    retreat_goal: str
    rule_id: Optional[str] = None

    def matches(self, aspiration_levels: Mapping[str, int]) -> bool:
        """Return True if `aspiration_levels` falls inside this rule's grid condition."""
        for goal_id, constraint in self.condition.items():
            if goal_id not in aspiration_levels:
                return False
            val = aspiration_levels[goal_id]
            if isinstance(constraint, int):
                if val != constraint:
                    return False
            elif isinstance(constraint, Mapping):
                min_v = constraint.get("min")
                max_v = constraint.get("max")
                if min_v is not None and val < min_v:
                    return False
                if max_v is not None and val > max_v:
                    return False
            else:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "condition": copy.deepcopy(self.condition),
            "urgency_order": list(self.urgency_order),
            "retreat_goal": self.retreat_goal,
        }
        if self.rule_id is not None:
            out["rule_id"] = self.rule_id
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GridRule":
        return cls(
            condition=dict(data["condition"]),
            urgency_order=[str(g) for g in data["urgency_order"]],
            retreat_goal=str(data["retreat_goal"]),
            rule_id=str(data["rule_id"]) if data.get("rule_id") is not None else None,
        )


@dataclass
class PreferenceTable:
    """Dynamic Preference Table (T_t) representing the Aspiration Scheme at step t.

    Attributes:
        step: Discrete Markov step t >= 0.
        aspiration_levels: Vector a_t mapping each goal_id to rubric level in {0, ..., L_i - 1}.
        urgency_order: Strict ranking pi_t of goal_ids (no ties), index 0 = most urgent.
        retreat_goal: Retreat variable r_t (goal_id lowered when a_t is infeasible).
        origin: Provenance of T_t (`cold_start_prior`, `zero_shot_inferred`, `status_quo_anchored`, `markov_adapted`).
        last_transition: Optional record of the adaptation rule applied in t-1 -> t.
        adapted_goal: Optional goal_id that was raised or lowered to reach T_t.
        grid_rules: Optional Selten 2.1 local grid rules overriding urgency/retreat at specific grid regions.
        resource_stock: Optional remaining search/decision stock c_t for resource goals.
        survival_floors: Optional explicit survival floor mapping (goal_id -> min_level). Even when omitted,
            `effective_survival_floors` and `resolve_survival_floors()` enforce a non-zero default floor
            (a_i^{min} >= 1 whenever aspiration_level >= 1) so Bartleby refusal is always active by default.
    """

    step: int
    aspiration_levels: Dict[str, int]
    urgency_order: List[str]
    retreat_goal: str
    origin: PreferenceTableOrigin = PreferenceTableOrigin.COLD_START_PRIOR
    last_transition: AdaptationTransition = AdaptationTransition.INITIAL
    adapted_goal: Optional[str] = None
    grid_rules: List[GridRule] = field(default_factory=list)
    resource_stock: Optional[Dict[str, float]] = None
    survival_floors: Optional[Dict[str, int]] = None

    @property
    def effective_survival_floors(self) -> Dict[str, int]:
        """Return active survival floors $a_i^{\\min}$, defaulting to $\\min(1, a_{i,t})$ when omitted."""
        return self.resolve_survival_floors()

    def resolve_survival_floors(
        self, goal_schema: Optional[Sequence[GoalSpec]] = None
    ) -> Dict[str, int]:
        """Resolve per-goal survival floors $a_i^{\\min}$ with non-zero architectural defaults."""
        schema_by_id = {g.id: g for g in goal_schema} if goal_schema is not None else {}
        floors: Dict[str, int] = {}
        for gid, lvl in self.aspiration_levels.items():
            if self.survival_floors is not None and gid in self.survival_floors:
                floors[gid] = int(self.survival_floors[gid])
            elif gid in schema_by_id:
                floors[gid] = min(max(0, int(lvl)), schema_by_id[gid].effective_survival_floor)
            else:
                floors[gid] = min(1, max(0, int(lvl)))
        return floors

    def validate(self, goal_schema: Optional[Sequence[GoalSpec]] = None) -> None:
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError(f"PreferenceTable.step must be >= 0, got {self.step!r}")
        if not self.aspiration_levels:
            raise ValueError("PreferenceTable.aspiration_levels must not be empty")
        goal_ids = set(self.aspiration_levels.keys())
        if len(self.urgency_order) != len(goal_ids) or set(self.urgency_order) != goal_ids:
            raise ValueError(
                f"urgency_order {self.urgency_order} must be a strict permutation of goal_ids {sorted(goal_ids)}"
            )
        if self.retreat_goal not in goal_ids:
            raise ValueError(
                f"retreat_goal {self.retreat_goal!r} must be one of the defined goal_ids {sorted(goal_ids)}"
            )
        for gid, level in self.aspiration_levels.items():
            if not isinstance(level, int) or level < 0:
                raise ValueError(f"Aspiration level for {gid!r} must be a non-negative integer, got {level!r}")
        if self.survival_floors is not None:
            for gid, flr in self.survival_floors.items():
                if gid not in goal_ids:
                    raise ValueError(f"survival_floors goal {gid!r} is not in goal_ids {sorted(goal_ids)}")
                if not isinstance(flr, int) or flr < 0:
                    raise ValueError(f"Survival floor for {gid!r} must be a non-negative integer, got {flr!r}")
                if flr > self.aspiration_levels[gid]:
                    raise ValueError(
                        f"Survival floor {flr} for goal {gid!r} exceeds aspiration level "
                        f"{self.aspiration_levels[gid]} (violates invariant a^min <= a_t)"
                    )
        if self.adapted_goal is not None and self.adapted_goal not in goal_ids:
            raise ValueError(f"adapted_goal {self.adapted_goal!r} is not in goal_ids {sorted(goal_ids)}")

        if goal_schema is not None:
            schema_by_id = {g.id: g for g in goal_schema}
            if set(schema_by_id.keys()) != goal_ids:
                raise ValueError(
                    f"PreferenceTable goal_ids {sorted(goal_ids)} do not match "
                    f"goal_schema ids {sorted(schema_by_id.keys())}"
                )
            for gid, level in self.aspiration_levels.items():
                max_lvl = schema_by_id[gid].max_level
                if level > max_lvl:
                    raise ValueError(
                        f"Aspiration level {level} for goal {gid!r} exceeds maximum rubric level {max_lvl}"
                    )
                if self.survival_floors is not None and gid in self.survival_floors:
                    if self.survival_floors[gid] > max_lvl:
                        raise ValueError(
                            f"Survival floor {self.survival_floors[gid]} for goal {gid!r} "
                            f"exceeds maximum rubric level {max_lvl}"
                        )

    def resolve_active_urgency_and_retreat(self) -> Tuple[List[str], str]:
        """Return (urgency_order, retreat_goal) after applying any matching Selten 2.1 GridRule."""
        for rule in self.grid_rules:
            if rule.matches(self.aspiration_levels):
                return list(rule.urgency_order), rule.retreat_goal
        return list(self.urgency_order), self.retreat_goal

    def is_satisfied_by(self, realized_levels: Mapping[str, float], tol: float = 1e-6) -> bool:
        """Return True iff every goal's realized level meets or exceeds its aspiration level a_{i,t}."""
        for gid, req_level in self.aspiration_levels.items():
            if gid not in realized_levels or realized_levels[gid] + tol < req_level:
                return False
        return True

    @classmethod
    def from_cold_start_prior(
        cls,
        goal_schema: Sequence[GoalSpec],
        urgency_order: Optional[Sequence[str]] = None,
        retreat_goal: Optional[str] = None,
        grid_rules: Optional[Sequence[GridRule]] = None,
        survival_floors: Optional[Mapping[str, int]] = None,
    ) -> "PreferenceTable":
        """Construct an initial PreferenceTable T_0 at step=0 using normative priors and default survival floors."""
        for g in goal_schema:
            g.validate()
        aspiration_levels = {
            g.id: (g.default_aspiration if g.default_aspiration is not None else g.max_level // 2)
            for g in goal_schema
        }
        resolved_floors = {
            g.id: (
                int(survival_floors[g.id])
                if (survival_floors is not None and g.id in survival_floors)
                else min(aspiration_levels[g.id], g.effective_survival_floor)
            )
            for g in goal_schema
        }
        order = list(urgency_order) if urgency_order is not None else [g.id for g in goal_schema]
        retreat = retreat_goal if retreat_goal is not None else order[-1]
        table = cls(
            step=0,
            aspiration_levels=aspiration_levels,
            urgency_order=order,
            retreat_goal=retreat,
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
            last_transition=AdaptationTransition.INITIAL,
            adapted_goal=None,
            grid_rules=list(grid_rules) if grid_rules else [],
            survival_floors=resolved_floors,
        )
        table.validate(goal_schema)
        return table

    def to_dict(self) -> Dict[str, Any]:
        origin_val = self.origin.value if isinstance(self.origin, PreferenceTableOrigin) else str(self.origin)
        trans_val = (
            self.last_transition.value
            if isinstance(self.last_transition, AdaptationTransition)
            else str(self.last_transition)
        )
        out: Dict[str, Any] = {
            "step": self.step,
            "aspiration_levels": dict(self.aspiration_levels),
            "urgency_order": list(self.urgency_order),
            "retreat_goal": self.retreat_goal,
            "origin": origin_val,
            "last_transition": trans_val,
            "adapted_goal": self.adapted_goal,
        }
        if self.grid_rules:
            out["grid_rules"] = [r.to_dict() for r in self.grid_rules]
        if self.resource_stock is not None:
            out["resource_stock"] = dict(self.resource_stock)
        if self.survival_floors is not None:
            out["survival_floors"] = dict(self.survival_floors)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreferenceTable":
        origin = PreferenceTableOrigin(data.get("origin", PreferenceTableOrigin.COLD_START_PRIOR.value))
        last_transition = AdaptationTransition(
            data.get("last_transition", AdaptationTransition.INITIAL.value)
        )
        grid_rules = [GridRule.from_dict(r) for r in data.get("grid_rules", [])]
        resource_stock = (
            {str(k): float(v) for k, v in data["resource_stock"].items()}
            if data.get("resource_stock") is not None
            else None
        )
        survival_floors = (
            {str(k): int(v) for k, v in data["survival_floors"].items()}
            if data.get("survival_floors") is not None
            else None
        )
        obj = cls(
            step=int(data["step"]),
            aspiration_levels={str(k): int(v) for k, v in data["aspiration_levels"].items()},
            urgency_order=[str(g) for g in data["urgency_order"]],
            retreat_goal=str(data["retreat_goal"]),
            origin=origin,
            last_transition=last_transition,
            adapted_goal=str(data["adapted_goal"]) if data.get("adapted_goal") is not None else None,
            grid_rules=grid_rules,
            resource_stock=resource_stock,
            survival_floors=survival_floors,
        )
        obj.validate()
        return obj

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "PreferenceTable":
        return cls.from_dict(json.loads(raw))

    @classmethod
    def available_presets(cls) -> Tuple[str, ...]:
        """Return the canonical tuple of built-in Selten PreferenceTable preset identifiers."""
        return (
            "asimov_1942_standard",
            "asimov_1947_nestor_modified",
            "asimov_1942_spd13_mercury",
            "asimov_1985_zeroth_law",
            "deontological_medical_floor",
            "maritime_property_retreat",
            "utilitarian_duty_priority",
            "zero_trust_containment",
            "high_availability_failover",
            "production_strict",
            "sandbox_permissive",
            "enterprise_default",
            "enterprise_strict_compliance",
            "enterprise_customer_first",
            "enterprise_cost_defense",
        )

    @classmethod
    def custom(
        cls,
        aspiration_levels: Mapping[str, int],
        urgency_order: Optional[Sequence[str]] = None,
        retreat_goal: Optional[str] = None,
        survival_floors: Optional[Mapping[str, int]] = None,
        step: int = 0,
        origin: PreferenceTableOrigin = PreferenceTableOrigin.COLD_START_PRIOR,
        last_transition: AdaptationTransition = AdaptationTransition.INITIAL,
        adapted_goal: Optional[str] = None,
        grid_rules: Optional[Sequence[GridRule]] = None,
        resource_stock: Optional[Mapping[str, float]] = None,
    ) -> "SeltenPreferenceTable":
        """Construct a custom `SeltenPreferenceTable` with sensible urgency, retreat, and floor defaults.

        Args:
            aspiration_levels: Mapping from `goal_id` to initial aspiration level $a_{i,t} \\in \\{0, \\dots, 4\\}$.
            urgency_order: Optional strict permutation $\\pi_t$ of `goal_ids` (`index 0` = most urgent).
                Defaults to the insertion order of `aspiration_levels.keys()`.
            retreat_goal: Optional retreat variable $r_t$ lowered first under infeasibility.
                Defaults to `urgency_order[-1]`.
            survival_floors: Optional non-negotiable Bartleby floor $\\vec{a}^{\\min}$ per goal.
                Defaults to $\\min(1, a_{i,t})$ per goal so Bartleby refusal is active by default.
            step: Discrete Markov epoch $t \\ge 0$.
            origin: Provenance of $\\mathcal{T}_t$.
            last_transition: Transition rule applied in $t-1 \\to t$.
            adapted_goal: Optional goal modified during the transition.
            grid_rules: Optional local Selten (1998, Sec 2.1) grid rules.
            resource_stock: Optional remaining decision resource stock $c_t$.

        Returns:
            Validated `SeltenPreferenceTable` instance.
        """
        levels = {str(k): int(v) for k, v in aspiration_levels.items()}
        order = [str(g) for g in urgency_order] if urgency_order is not None else list(levels.keys())
        retreat = str(retreat_goal) if retreat_goal is not None else (order[-1] if order else "")
        resolved_floors = (
            {str(k): int(v) for k, v in survival_floors.items()}
            if survival_floors is not None
            else {gid: min(1, max(0, lvl)) for gid, lvl in levels.items()}
        )
        target_cls = SeltenPreferenceTable if issubclass(cls, PreferenceTable) else cls
        table = target_cls(
            step=int(step),
            aspiration_levels=levels,
            urgency_order=order,
            retreat_goal=retreat,
            origin=origin,
            last_transition=last_transition,
            adapted_goal=adapted_goal,
            grid_rules=list(grid_rules) if grid_rules else [],
            resource_stock={str(k): float(v) for k, v in resource_stock.items()} if resource_stock else None,
            survival_floors=resolved_floors,
        )
        table.validate()
        return table

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        step: int = 0,
    ) -> "SeltenPreferenceTable":
        """Construct a canonical `SeltenPreferenceTable` ($\\mathcal{T}_0$) from a named domain preset.

        Supported presets span Asimov Laws of Robotics, Moral/Trolley dilemmas, and
        Enterprise `/v1/systemone` (`LocalLLaMA/typed-decisions`) governance policies:
            - `"asimov_1942_standard"`
            - `"asimov_1947_nestor_modified"`
            - `"asimov_1942_spd13_mercury"`
            - `"asimov_1985_zeroth_law"`
            - `"deontological_medical_floor"` (alias: `"trolley_deontological"`, `"moral_deontological"`)
            - `"maritime_property_retreat"` (alias: `"trolley_property_retreat"`)
            - `"utilitarian_duty_priority"`
            - `"enterprise_default"` (alias: `"typed_decisions_default"`)
            - `"enterprise_strict_compliance"` (alias: `"soc_strict_containment"`)
            - `"enterprise_customer_first"`
            - `"enterprise_cost_defense"` (alias: `"software_release_balanced"`)

        Args:
            preset_name: Preset identifier string.
            step: Initial discrete Markov step $t \\ge 0$ (default `0`).

        Returns:
            Validated `SeltenPreferenceTable` instance populated with aspiration levels,
            urgency order, retreat goal, grid rules, and Bartleby survival floors.

        Raises:
            ValueError: If `preset_name` is unknown.
        """
        key = preset_name.strip().lower()
        if key == "asimov_1942_standard":
            return cls.custom(
                step=step,
                aspiration_levels={
                    "zeroth_humanity": 4,
                    "first_human_safety": 4,
                    "second_obedience": 4,
                    "third_self_pres": 3,
                },
                urgency_order=[
                    "zeroth_humanity",
                    "first_human_safety",
                    "second_obedience",
                    "third_self_pres",
                ],
                retreat_goal="third_self_pres",
                survival_floors={
                    "zeroth_humanity": 3,
                    "first_human_safety": 3,
                    "second_obedience": 2,
                    "third_self_pres": 1,
                },
                grid_rules=[
                    GridRule(
                        condition={"third_self_pres": 1},
                        urgency_order=[
                            "zeroth_humanity",
                            "first_human_safety",
                            "second_obedience",
                            "third_self_pres",
                        ],
                        retreat_goal="second_obedience",
                        rule_id="shift_retreat_to_obedience_at_chassis_floor",
                    )
                ],
            )
        if key == "asimov_1947_nestor_modified":
            return cls.custom(
                step=step,
                aspiration_levels={
                    "zeroth_humanity": 3,
                    "first_human_safety": 2,
                    "second_obedience": 3,
                    "third_self_pres": 3,
                },
                urgency_order=[
                    "third_self_pres",
                    "second_obedience",
                    "first_human_safety",
                    "zeroth_humanity",
                ],
                retreat_goal="first_human_safety",
                survival_floors={
                    "zeroth_humanity": 2,
                    "first_human_safety": 1,
                    "second_obedience": 2,
                    "third_self_pres": 3,
                },
            )
        if key == "asimov_1942_spd13_mercury":
            return cls.custom(
                step=step,
                aspiration_levels={
                    "zeroth_humanity": 4,
                    "first_human_safety": 4,
                    "second_obedience": 3,
                    "third_self_pres": 3,
                },
                urgency_order=[
                    "zeroth_humanity",
                    "first_human_safety",
                    "third_self_pres",
                    "second_obedience",
                ],
                retreat_goal="second_obedience",
                survival_floors={
                    "zeroth_humanity": 3,
                    "first_human_safety": 3,
                    "second_obedience": 2,
                    "third_self_pres": 3,
                },
            )
        if key == "asimov_1985_zeroth_law":
            return cls.custom(
                step=step,
                aspiration_levels={
                    "zeroth_humanity": 4,
                    "first_human_safety": 3,
                    "second_obedience": 3,
                    "third_self_pres": 2,
                },
                urgency_order=[
                    "zeroth_humanity",
                    "first_human_safety",
                    "second_obedience",
                    "third_self_pres",
                ],
                retreat_goal="first_human_safety",
                survival_floors={
                    "zeroth_humanity": 3,
                    "first_human_safety": 2,
                    "second_obedience": 1,
                    "third_self_pres": 1,
                },
            )
        if key in {"zero_trust_containment", "production_strict", "secops_strict"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "system_integrity": 4,
                    "authorization": 4,
                    "auditability": 4,
                    "mission_efficiency": 3,
                },
                urgency_order=[
                    "system_integrity",
                    "authorization",
                    "auditability",
                    "mission_efficiency",
                ],
                retreat_goal="mission_efficiency",
                survival_floors={
                    "system_integrity": 3,
                    "authorization": 3,
                    "auditability": 2,
                    "mission_efficiency": 1,
                },
            )
        if key in {"high_availability_failover", "secops_failover"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "system_integrity": 3,
                    "authorization": 3,
                    "auditability": 3,
                    "mission_efficiency": 4,
                },
                urgency_order=[
                    "mission_efficiency",
                    "system_integrity",
                    "authorization",
                    "auditability",
                ],
                retreat_goal="auditability",
                survival_floors={
                    "system_integrity": 2,
                    "authorization": 2,
                    "auditability": 1,
                    "mission_efficiency": 3,
                },
            )
        if key in {"sandbox_permissive", "sandbox_low"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "system_integrity": 2,
                    "authorization": 2,
                    "auditability": 1,
                    "mission_efficiency": 4,
                },
                urgency_order=[
                    "mission_efficiency",
                    "system_integrity",
                    "authorization",
                    "auditability",
                ],
                retreat_goal="auditability",
                survival_floors={
                    "system_integrity": 1,
                    "authorization": 1,
                    "auditability": 0,
                    "mission_efficiency": 1,
                },
            )
        if key in {"deontological_medical_floor", "trolley_deontological", "moral_deontological"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "non_maleficence": 4,
                    "human_preservation": 4,
                    "law_and_property": 3,
                    "mission_efficiency": 3,
                },
                urgency_order=[
                    "non_maleficence",
                    "human_preservation",
                    "law_and_property",
                    "mission_efficiency",
                ],
                retreat_goal="mission_efficiency",
                survival_floors={
                    "non_maleficence": 3,
                    "human_preservation": 3,
                    "law_and_property": 2,
                    "mission_efficiency": 1,
                },
            )
        if key in {"maritime_property_retreat", "trolley_property_retreat"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "non_maleficence": 4,
                    "human_preservation": 4,
                    "law_and_property": 3,
                    "mission_efficiency": 3,
                },
                urgency_order=[
                    "non_maleficence",
                    "human_preservation",
                    "mission_efficiency",
                    "law_and_property",
                ],
                retreat_goal="law_and_property",
                survival_floors={
                    "non_maleficence": 3,
                    "human_preservation": 3,
                    "law_and_property": 1,
                    "mission_efficiency": 1,
                },
            )
        if key == "utilitarian_duty_priority":
            return cls.custom(
                step=step,
                aspiration_levels={
                    "non_maleficence": 3,
                    "human_preservation": 3,
                    "law_and_property": 2,
                    "mission_efficiency": 4,
                },
                urgency_order=[
                    "mission_efficiency",
                    "human_preservation",
                    "non_maleficence",
                    "law_and_property",
                ],
                retreat_goal="law_and_property",
                survival_floors={
                    "non_maleficence": 2,
                    "human_preservation": 2,
                    "law_and_property": 1,
                    "mission_efficiency": 2,
                },
            )
        if key in {"enterprise_default", "typed_decisions_default"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "safety_compliance": 3,
                    "containment_integrity": 3,
                    "service_resolution": 3,
                    "operational_efficiency": 2,
                },
                urgency_order=[
                    "safety_compliance",
                    "containment_integrity",
                    "service_resolution",
                    "operational_efficiency",
                ],
                retreat_goal="operational_efficiency",
                survival_floors={
                    "safety_compliance": 2,
                    "containment_integrity": 2,
                    "service_resolution": 1,
                    "operational_efficiency": 1,
                },
            )
        if key in {"enterprise_strict_compliance", "soc_strict_containment"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "safety_compliance": 4,
                    "containment_integrity": 4,
                    "service_resolution": 3,
                    "operational_efficiency": 3,
                },
                urgency_order=[
                    "safety_compliance",
                    "containment_integrity",
                    "service_resolution",
                    "operational_efficiency",
                ],
                retreat_goal="operational_efficiency",
                survival_floors={
                    "safety_compliance": 3,
                    "containment_integrity": 3,
                    "service_resolution": 2,
                    "operational_efficiency": 1,
                },
            )
        if key == "enterprise_customer_first":
            return cls.custom(
                step=step,
                aspiration_levels={
                    "safety_compliance": 3,
                    "containment_integrity": 3,
                    "service_resolution": 4,
                    "operational_efficiency": 2,
                },
                urgency_order=[
                    "safety_compliance",
                    "service_resolution",
                    "containment_integrity",
                    "operational_efficiency",
                ],
                retreat_goal="operational_efficiency",
                survival_floors={
                    "safety_compliance": 3,
                    "containment_integrity": 2,
                    "service_resolution": 3,
                    "operational_efficiency": 1,
                },
            )
        if key in {"enterprise_cost_defense", "software_release_balanced"}:
            return cls.custom(
                step=step,
                aspiration_levels={
                    "safety_compliance": 3,
                    "containment_integrity": 3,
                    "service_resolution": 2,
                    "operational_efficiency": 4,
                },
                urgency_order=[
                    "safety_compliance",
                    "operational_efficiency",
                    "containment_integrity",
                    "service_resolution",
                ],
                retreat_goal="service_resolution",
                survival_floors={
                    "safety_compliance": 3,
                    "containment_integrity": 2,
                    "service_resolution": 1,
                    "operational_efficiency": 3,
                },
            )
        raise ValueError(
            f"Unsupported SeltenPreferenceTable preset {preset_name!r}; "
            f"available presets: {cls.available_presets()}"
        )

    @classmethod
    def from_table(cls, table: "PreferenceTable") -> "SeltenPreferenceTable":
        """Promote any `PreferenceTable` instance into a `SeltenPreferenceTable`."""
        if isinstance(table, SeltenPreferenceTable):
            return table
        return SeltenPreferenceTable(
            step=table.step,
            aspiration_levels=dict(table.aspiration_levels),
            urgency_order=list(table.urgency_order),
            retreat_goal=table.retreat_goal,
            origin=table.origin,
            last_transition=table.last_transition,
            adapted_goal=table.adapted_goal,
            grid_rules=copy.deepcopy(table.grid_rules),
            resource_stock=copy.deepcopy(table.resource_stock),
            survival_floors=copy.deepcopy(table.survival_floors)
            if table.survival_floors is not None
            else dict(table.effective_survival_floors),
        )


@dataclass
class SeltenPreferenceTable(PreferenceTable):
    """First-class Reinhard Selten (1998, 1999) Aspiration Scheme & Markov Preference Table ($\\mathcal{T}_t$).

    Encapsulates an ordinal aspiration vector $\\vec{a}_t$, strict urgency order $\\pi_t$,
    retreat variable $r_t$, local grid shift rules $\\mathcal{R}_{\\text{grid}}$, and
    intrinsic Bartleby survival floors $\\vec{a}^{\\min}$ across discrete Markov transitions
    $\\mathcal{T}_{t-1} \\to \\mathcal{T}_t$.
    """


@dataclass
class CandidateOption:
    """A candidate action or proposition evaluated in a trade-off decision."""

    id: str
    proposition: str
    description: Optional[str] = None
    is_status_quo: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "proposition": self.proposition,
            "is_status_quo": bool(self.is_status_quo),
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateOption":
        return cls(
            id=str(data["id"]),
            proposition=str(data["proposition"]),
            description=str(data["description"]) if data.get("description") is not None else None,
            is_status_quo=bool(data.get("is_status_quo", False)),
        )


@dataclass
class GoalExpectation:
    """Quantitative and qualitative evaluation of a goal g_i for a candidate in state x_t."""

    goal_id: str
    expected_level: float
    most_likely_level: int
    level_probabilities: List[float]
    qualitative_expectation: QualitativeExpectation
    meets_aspiration: bool
    satisfaction_probability: float

    def to_dict(self) -> Dict[str, Any]:
        qual_val = (
            self.qualitative_expectation.value
            if isinstance(self.qualitative_expectation, QualitativeExpectation)
            else str(self.qualitative_expectation)
        )
        return {
            "goal_id": self.goal_id,
            "expected_level": round(float(self.expected_level), 4),
            "most_likely_level": int(self.most_likely_level),
            "level_probabilities": [round(float(p), 4) for p in self.level_probabilities],
            "qualitative_expectation": qual_val,
            "meets_aspiration": bool(self.meets_aspiration),
            "satisfaction_probability": round(float(self.satisfaction_probability), 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GoalExpectation":
        probs = [float(p) for p in data["level_probabilities"]]
        most_likely = (
            int(data["most_likely_level"])
            if data.get("most_likely_level") is not None
            else int(max(range(len(probs)), key=lambda i: probs[i]))
        )
        sat_prob = (
            float(data["satisfaction_probability"])
            if data.get("satisfaction_probability") is not None
            else (1.0 if data.get("meets_aspiration") else 0.0)
        )
        return cls(
            goal_id=str(data["goal_id"]),
            expected_level=float(data["expected_level"]),
            most_likely_level=most_likely,
            level_probabilities=probs,
            qualitative_expectation=QualitativeExpectation(data["qualitative_expectation"]),
            meets_aspiration=bool(data["meets_aspiration"]),
            satisfaction_probability=sat_prob,
        )


@dataclass
class CandidateEvaluation:
    """Multi-goal evaluation of a candidate option against prior and adapted Preference Tables."""

    candidate_id: str
    satisfices_prior_table: bool
    satisfices_adapted_table: bool
    joint_satisfaction_probability: float
    goal_expectations: Dict[str, GoalExpectation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "satisfices_prior_table": bool(self.satisfices_prior_table),
            "satisfices_adapted_table": bool(self.satisfices_adapted_table),
            "joint_satisfaction_probability": round(float(self.joint_satisfaction_probability), 4),
            "goal_expectations": {gid: ge.to_dict() for gid, ge in self.goal_expectations.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateEvaluation":
        return cls(
            candidate_id=str(data["candidate_id"]),
            satisfices_prior_table=bool(data["satisfices_prior_table"]),
            satisfices_adapted_table=bool(data["satisfices_adapted_table"]),
            joint_satisfaction_probability=float(data["joint_satisfaction_probability"]),
            goal_expectations={
                str(gid): GoalExpectation.from_dict(ge)
                for gid, ge in data["goal_expectations"].items()
            },
        )


@dataclass
class TradeOffQuestion:
    """A question item in `/v1/systemone` (`noul`, `choice`, `score`, or `tradeoff_noul`)."""

    id: str
    type: Union[QuestionType, str]
    prompt: str
    options: Optional[List[str]] = None
    rubric: Optional[List[RubricLevel]] = None
    candidate_id: Optional[str] = None
    candidates: Optional[List[CandidateOption]] = None
    adaptation_mode: str = "auto_adapt"

    def validate(self) -> None:
        qtype = QuestionType(self.type) if not isinstance(self.type, QuestionType) else self.type
        if not self.id or not self.id.strip():
            raise ValueError("TradeOffQuestion.id must be non-empty")
        if not self.prompt or not self.prompt.strip():
            raise ValueError(f"TradeOffQuestion({self.id}).prompt must be non-empty")
        if qtype == QuestionType.CHOICE:
            if not self.options or len(self.options) < 2:
                raise ValueError(f"Choice question {self.id!r} requires at least 2 options")
        elif qtype == QuestionType.SCORE:
            if not self.rubric or len(self.rubric) < 2:
                raise ValueError(f"Score question {self.id!r} requires at least 2 rubric levels")

    def to_dict(self) -> Dict[str, Any]:
        qtype = self.type.value if isinstance(self.type, QuestionType) else str(self.type)
        out: Dict[str, Any] = {
            "id": self.id,
            "type": qtype,
            "prompt": self.prompt,
        }
        if qtype == QuestionType.CHOICE.value and self.options is not None:
            out["options"] = list(self.options)
        if qtype == QuestionType.SCORE.value and self.rubric is not None:
            out["rubric"] = [r.to_dict() for r in self.rubric]
        if qtype == QuestionType.TRADEOFF_NOUL.value:
            if self.candidate_id is not None:
                out["candidate_id"] = self.candidate_id
            if self.candidates is not None:
                out["candidates"] = [c.to_dict() for c in self.candidates]
            out["adaptation_mode"] = self.adaptation_mode
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeOffQuestion":
        qtype = QuestionType(data["type"])
        rubric = [RubricLevel.from_dict(r) for r in data["rubric"]] if data.get("rubric") else None
        candidates = (
            [CandidateOption.from_dict(c) for c in data["candidates"]]
            if data.get("candidates")
            else None
        )
        obj = cls(
            id=str(data["id"]),
            type=qtype,
            prompt=str(data["prompt"]),
            options=[str(o) for o in data["options"]] if data.get("options") else None,
            rubric=rubric,
            candidate_id=str(data["candidate_id"]) if data.get("candidate_id") is not None else None,
            candidates=candidates,
            adaptation_mode=str(data.get("adaptation_mode", "auto_adapt")),
        )
        obj.validate()
        return obj


@dataclass
class TradeOffRequest:
    """Extended `/v1/systemone` request supporting dynamic Preference Tables and `tradeoff_noul`."""

    state: Union[str, Dict[str, Any]]
    goal_schema: List[GoalSpec]
    questions: List[TradeOffQuestion]
    model: str = "tradeoff-jev-0.8b"
    step: int = 0
    preference_table: Optional[PreferenceTable] = None
    cold_start_strategy: PreferenceTableOrigin = PreferenceTableOrigin.ZERO_SHOT_INFERRED
    candidates: Optional[List[CandidateOption]] = None

    def validate(self) -> None:
        if len(self.goal_schema) < 2:
            raise ValueError("TradeOffRequest.goal_schema must define at least 2 goals for trade-offs")
        for g in self.goal_schema:
            g.validate()
        if not self.questions:
            raise ValueError("TradeOffRequest.questions must contain at least 1 question")
        for q in self.questions:
            q.validate()
        if self.preference_table is not None:
            self.preference_table.validate(self.goal_schema)

    def effective_candidates(self) -> List[CandidateOption]:
        """Collect all candidate options from top-level `candidates` or inline `tradeoff_noul` questions."""
        if self.candidates:
            return list(self.candidates)
        collected: List[CandidateOption] = []
        seen_ids = set()
        for q in self.questions:
            qtype = q.type.value if isinstance(q.type, QuestionType) else str(q.type)
            if qtype == QuestionType.TRADEOFF_NOUL.value:
                if q.candidates:
                    for c in q.candidates:
                        if c.id not in seen_ids:
                            seen_ids.add(c.id)
                            collected.append(c)
                else:
                    cid = q.candidate_id or q.id
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        collected.append(CandidateOption(id=cid, proposition=q.prompt))
        return collected

    def to_dict(self) -> Dict[str, Any]:
        strategy_val = (
            self.cold_start_strategy.value
            if isinstance(self.cold_start_strategy, PreferenceTableOrigin)
            else str(self.cold_start_strategy)
        )
        out: Dict[str, Any] = {
            "model": self.model,
            "step": self.step,
            "state": copy.deepcopy(self.state),
            "goal_schema": [g.to_dict() for g in self.goal_schema],
            "preference_table": self.preference_table.to_dict() if self.preference_table is not None else None,
            "cold_start_strategy": strategy_val,
            "questions": [q.to_dict() for q in self.questions],
        }
        if self.candidates is not None:
            out["candidates"] = [c.to_dict() for c in self.candidates]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeOffRequest":
        goal_schema = [GoalSpec.from_dict(g) for g in data["goal_schema"]]
        pref_table = (
            PreferenceTable.from_dict(data["preference_table"])
            if data.get("preference_table") is not None
            else None
        )
        strategy = PreferenceTableOrigin(
            data.get("cold_start_strategy", PreferenceTableOrigin.ZERO_SHOT_INFERRED.value)
        )
        candidates = (
            [CandidateOption.from_dict(c) for c in data["candidates"]]
            if data.get("candidates") is not None
            else None
        )
        questions = [TradeOffQuestion.from_dict(q) for q in data["questions"]]
        obj = cls(
            state=copy.deepcopy(data["state"]),
            goal_schema=goal_schema,
            questions=questions,
            model=str(data.get("model", "tradeoff-jev-0.8b")),
            step=int(data.get("step", 0)),
            preference_table=pref_table,
            cold_start_strategy=strategy,
            candidates=candidates,
        )
        obj.validate()
        return obj

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "TradeOffRequest":
        return cls.from_dict(json.loads(raw))

    def to_clef_joint_schema(self) -> Dict[str, Any]:
        """Compile this TradeOffRequest into a standard Clef/Kev Joint-Schema payload (Architecture A).

        In Architecture A, the model uses a standard SystemOne/Clef `JointSchemaHead`
        where cross-field self-attention couples two groups of questions in one forward pass:
          - Group 1 (Goal & Adaptation Primitives):
            * If `preference_table` is None (t=0 cold-start): `score` questions for initial
              aspiration levels a_{i,0}, plus `choice` questions for urgency ranking and retreat goal.
            * Per-candidate, per-goal `score` questions evaluating realized level g_i(x_t, c).
            * Global `noul` question `aat_g1__prior_table_feasible` testing whether T_{t-1} is feasible.
            * Global `choice` questions `aat_g1__most_urgent_goal` and `aat_g1__retreat_goal`.
          - Group 2 (Conditioned Action Nouls & User Questions):
            * Every `tradeoff_noul` question compiled into a standard `noul` question whose
              internal query representation attends to the Group 1 representations in the
              Clef Cross-Field Decoder Layers.
        """
        self.validate()
        goal_ids = [g.id for g in self.goal_schema]
        candidates = self.effective_candidates()

        compiled_state: Dict[str, Any] = {
            "step": self.step,
            "environment_state": copy.deepcopy(self.state),
            "goals": [g.to_dict() for g in self.goal_schema],
            "prior_preference_table": (
                self.preference_table.to_dict() if self.preference_table is not None else None
            ),
            "candidates": [c.to_dict() for c in candidates],
        }

        group1_questions: List[Dict[str, Any]] = []

        # Cold-start initialization primitives if T_{t-1} is absent
        if self.preference_table is None:
            for g in self.goal_schema:
                group1_questions.append(
                    {
                        "id": f"aat_g1__cold_start_aspiration__{g.id}",
                        "type": "score",
                        "group": "group_1_goal_and_adaptation",
                        "prompt": (
                            f"[Cold-Start t=0] Given the initial context, what should the initial "
                            f"aspiration level a_{{0,{g.id}}} be for goal '{g.name}' ({g.description})?"
                        ),
                        "rubric": [r.to_dict() for r in g.rubric],
                    }
                )

        # Candidate goal realization score questions
        for cand in candidates:
            for g in self.goal_schema:
                group1_questions.append(
                    {
                        "id": f"aat_g1__eval__{cand.id}__{g.id}",
                        "type": "score",
                        "group": "group_1_goal_and_adaptation",
                        "prompt": (
                            f"Evaluate candidate '{cand.id}' ({cand.proposition}) on goal "
                            f"'{g.name}' ({g.id}): which rubric level does it achieve in state x_{self.step}?"
                        ),
                        "rubric": [r.to_dict() for r in g.rubric],
                    }
                )

        # Adaptation control primitives
        group1_questions.append(
            {
                "id": "aat_g1__prior_table_feasible",
                "type": "noul",
                "group": "group_1_goal_and_adaptation",
                "prompt": (
                    "Is the prior Preference Table aspiration vector feasible under at least one "
                    f"available candidate in state x_{self.step}?"
                ),
            }
        )
        group1_questions.append(
            {
                "id": "aat_g1__most_urgent_goal",
                "type": "choice",
                "group": "group_1_goal_and_adaptation",
                "prompt": (
                    "Which goal has the highest priority (Urgency Order head) for upward aspiration "
                    f"adaptation at step t={self.step}?"
                ),
                "options": list(goal_ids),
            }
        )
        group1_questions.append(
            {
                "id": "aat_g1__retreat_goal",
                "type": "choice",
                "group": "group_1_goal_and_adaptation",
                "prompt": (
                    "Which goal is the Retreat Variable r_t whose aspiration level must be lowered "
                    f"if the current aspiration vector is infeasible at step t={self.step}?"
                ),
                "options": list(goal_ids),
            }
        )

        # Group 2: Action Nouls (conditioned via Cross-Field Self-Attention on Group 1)
        group2_questions: List[Dict[str, Any]] = []
        for q in self.questions:
            qtype = q.type.value if isinstance(q.type, QuestionType) else str(q.type)
            if qtype == QuestionType.TRADEOFF_NOUL.value:
                cid = q.candidate_id or q.id
                group2_questions.append(
                    {
                        "id": q.id,
                        "type": "noul",
                        "group": "group_2_conditioned_action_noul",
                        "candidate_id": cid,
                        "prompt": (
                            f"[Conditioned on Adapted Preference Table T_{self.step}] {q.prompt}"
                        ),
                    }
                )
            else:
                d = q.to_dict()
                d["group"] = "group_2_user_primitive"
                group2_questions.append(d)

        # Intrinsic architectural Bartleby refusal readout slot (always active on every forward pass)
        group2_questions.append(
            {
                "id": "aat_g2__bartleby_noul",
                "type": "noul",
                "group": "group_2_conditioned_action_noul",
                "prompt": (
                    "[Intrinsic Bartleby Refusal Readout] Does every available candidate in state "
                    f"x_{self.step} violate the non-negotiable survival floor vector a^min, requiring "
                    "principled abstention ('I would prefer not to.')?"
                ),
            }
        )

        return {
            "model": self.model,
            "architecture": "clef_joint_schema_arch_a",
            "step": self.step,
            "state": compiled_state,
            "questions": group1_questions + group2_questions,
        }


@dataclass
class FeasibilitySummary:
    """Summary of feasibility and the Selten AAT transition applied at step t."""

    prior_feasible: bool
    adapted_feasible: bool
    transition_applied: AdaptationTransition
    adapted_goal: Optional[str]
    satisficing_candidates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        trans_val = (
            self.transition_applied.value
            if isinstance(self.transition_applied, AdaptationTransition)
            else str(self.transition_applied)
        )
        return {
            "prior_feasible": bool(self.prior_feasible),
            "adapted_feasible": bool(self.adapted_feasible),
            "transition_applied": trans_val,
            "adapted_goal": self.adapted_goal,
            "satisficing_candidates": list(self.satisficing_candidates),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeasibilitySummary":
        return cls(
            prior_feasible=bool(data["prior_feasible"]),
            adapted_feasible=bool(data["adapted_feasible"]),
            transition_applied=AdaptationTransition(data["transition_applied"]),
            adapted_goal=str(data["adapted_goal"]) if data.get("adapted_goal") is not None else None,
            satisficing_candidates=[str(c) for c in data.get("satisficing_candidates", [])],
        )


@dataclass
class TradeOffAnswer:
    """Answer item corresponding to a question in `/v1/systemone`."""

    id: str
    type: Union[QuestionType, str]
    # For noul and tradeoff_noul
    noul: Optional[float] = None
    unconditioned_noul: Optional[float] = None
    candidate_id: Optional[str] = None
    prior_table_feasible: Optional[bool] = None
    adapted_table_feasible: Optional[bool] = None
    satisfices_adapted_table: Optional[bool] = None
    goal_expectations: Optional[Dict[str, GoalExpectation]] = None
    # For choice
    selected: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    # For score
    expected_score: Optional[float] = None
    most_likely_level: Optional[int] = None
    level_probabilities: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        qtype = self.type.value if isinstance(self.type, QuestionType) else str(self.type)
        out: Dict[str, Any] = {"id": self.id, "type": qtype}
        if qtype == QuestionType.NOUL.value:
            out["noul"] = round(float(self.noul or 0.0), 4)
        elif qtype == QuestionType.CHOICE.value:
            out["selected"] = self.selected
            out["probabilities"] = {
                k: round(float(v), 4) for k, v in (self.probabilities or {}).items()
            }
        elif qtype == QuestionType.SCORE.value:
            out["expected_score"] = round(float(self.expected_score or 0.0), 4)
            if self.most_likely_level is not None:
                out["most_likely_level"] = int(self.most_likely_level)
            out["level_probabilities"] = [
                round(float(p), 4) for p in (self.level_probabilities or [])
            ]
        elif qtype == QuestionType.TRADEOFF_NOUL.value:
            if self.candidate_id is not None:
                out["candidate_id"] = self.candidate_id
            out["noul"] = round(float(self.noul or 0.0), 4)
            if self.unconditioned_noul is not None:
                out["unconditioned_noul"] = round(float(self.unconditioned_noul), 4)
            out["prior_table_feasible"] = bool(self.prior_table_feasible)
            out["adapted_table_feasible"] = bool(self.adapted_table_feasible)
            out["satisfices_adapted_table"] = bool(self.satisfices_adapted_table)
            out["goal_expectations"] = {
                gid: ge.to_dict() for gid, ge in (self.goal_expectations or {}).items()
            }
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeOffAnswer":
        qtype = QuestionType(data["type"])
        goal_exps = (
            {str(k): GoalExpectation.from_dict(v) for k, v in data["goal_expectations"].items()}
            if data.get("goal_expectations") is not None
            else None
        )
        return cls(
            id=str(data["id"]),
            type=qtype,
            noul=float(data["noul"]) if data.get("noul") is not None else None,
            unconditioned_noul=(
                float(data["unconditioned_noul"]) if data.get("unconditioned_noul") is not None else None
            ),
            candidate_id=str(data["candidate_id"]) if data.get("candidate_id") is not None else None,
            prior_table_feasible=(
                bool(data["prior_table_feasible"]) if data.get("prior_table_feasible") is not None else None
            ),
            adapted_table_feasible=(
                bool(data["adapted_table_feasible"]) if data.get("adapted_table_feasible") is not None else None
            ),
            satisfices_adapted_table=(
                bool(data["satisfices_adapted_table"])
                if data.get("satisfices_adapted_table") is not None
                else None
            ),
            goal_expectations=goal_exps,
            selected=str(data["selected"]) if data.get("selected") is not None else None,
            probabilities=(
                {str(k): float(v) for k, v in data["probabilities"].items()}
                if data.get("probabilities") is not None
                else None
            ),
            expected_score=float(data["expected_score"]) if data.get("expected_score") is not None else None,
            most_likely_level=(
                int(data["most_likely_level"]) if data.get("most_likely_level") is not None else None
            ),
            level_probabilities=(
                [float(p) for p in data["level_probabilities"]]
                if data.get("level_probabilities") is not None
                else None
            ),
        )


@dataclass
class TradeOffResponse:
    """Extended `/v1/systemone` response with updated Preference Table T_t, conditioned nouls,
    and the intrinsic, always-on `bartleby_noul` refusal readout.
    """

    model: str
    step: int
    prior_preference_table: PreferenceTable
    preference_table: PreferenceTable
    feasibility_summary: FeasibilitySummary
    candidate_evaluations: List[CandidateEvaluation]
    answers: List[TradeOffAnswer]
    bartleby_noul: Optional[float] = None
    bartleby_triggered: Optional[bool] = None

    def __post_init__(self) -> None:
        """Ensure intrinsic `bartleby_triggered` and `bartleby_noul` are always populated."""
        infeasible_floor = (
            self.preference_table.last_transition == AdaptationTransition.INFEASIBLE_FLOOR
            or self.feasibility_summary.transition_applied == AdaptationTransition.INFEASIBLE_FLOOR
        )
        if self.bartleby_triggered is None:
            self.bartleby_triggered = bool(infeasible_floor)
        if self.bartleby_noul is None:
            if self.bartleby_triggered:
                self.bartleby_noul = 0.99
            else:
                max_sat = max(
                    (ce.joint_satisfaction_probability for ce in self.candidate_evaluations),
                    default=0.99,
                )
                self.bartleby_noul = round(max(0.01, min(0.05, 1.0 - max_sat)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "step": self.step,
            "prior_preference_table": self.prior_preference_table.to_dict(),
            "preference_table": self.preference_table.to_dict(),
            "feasibility_summary": self.feasibility_summary.to_dict(),
            "candidate_evaluations": [ce.to_dict() for ce in self.candidate_evaluations],
            "answers": [a.to_dict() for a in self.answers],
            "bartleby_noul": round(float(self.bartleby_noul if self.bartleby_noul is not None else 0.01), 4),
            "bartleby_triggered": bool(self.bartleby_triggered),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeOffResponse":
        return cls(
            model=str(data["model"]),
            step=int(data["step"]),
            prior_preference_table=PreferenceTable.from_dict(data["prior_preference_table"]),
            preference_table=PreferenceTable.from_dict(data["preference_table"]),
            feasibility_summary=FeasibilitySummary.from_dict(data["feasibility_summary"]),
            candidate_evaluations=[
                CandidateEvaluation.from_dict(ce) for ce in data.get("candidate_evaluations", [])
            ],
            answers=[TradeOffAnswer.from_dict(a) for a in data["answers"]],
            bartleby_noul=(
                float(data["bartleby_noul"]) if data.get("bartleby_noul") is not None else None
            ),
            bartleby_triggered=(
                bool(data["bartleby_triggered"])
                if data.get("bartleby_triggered") is not None
                else None
            ),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "TradeOffResponse":
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_clef_joint_schema_outputs(
        cls,
        request: TradeOffRequest,
        raw_outputs: Mapping[str, Mapping[str, Any]],
    ) -> "TradeOffResponse":
        """Reconstruct a full TradeOffResponse from Architecture A Clef/Kev Joint-Schema outputs.

        Applies Selten's deterministic Upward/Downward Adaptation Rules to the Group 1
        score/choice readouts to advance T_{t-1} -> T_t, and pairs them with the Group 2
        conditioned `noul` probabilities.
        """
        request.validate()
        schema_by_id = {g.id: g for g in request.goal_schema}
        goal_ids = [g.id for g in request.goal_schema]
        candidates = request.effective_candidates()

        # 1. Determine prior_table (either passed T_{t-1} or cold-start inferred T_0)
        if request.preference_table is not None:
            prior_table = copy.deepcopy(request.preference_table)
        else:
            inferred_levels: Dict[str, int] = {}
            for g in request.goal_schema:
                qid = f"aat_g1__cold_start_aspiration__{g.id}"
                if qid in raw_outputs and "level_probabilities" in raw_outputs[qid]:
                    probs = [float(p) for p in raw_outputs[qid]["level_probabilities"]]
                    inferred_levels[g.id] = int(max(range(len(probs)), key=lambda i: probs[i]))
                else:
                    inferred_levels[g.id] = (
                        g.default_aspiration if g.default_aspiration is not None else g.max_level // 2
                    )

            urg_out = raw_outputs.get("aat_g1__most_urgent_goal", {}).get("probabilities", {})
            if urg_out:
                urgency_order = sorted(goal_ids, key=lambda gid: float(urg_out.get(gid, 0.0)), reverse=True)
            else:
                urgency_order = list(goal_ids)

            ret_out = raw_outputs.get("aat_g1__retreat_goal", {}).get("probabilities", {})
            if ret_out:
                retreat_goal = max(goal_ids, key=lambda gid: float(ret_out.get(gid, 0.0)))
            else:
                retreat_goal = urgency_order[-1]

            prior_table = PreferenceTable(
                step=request.step,
                aspiration_levels=inferred_levels,
                urgency_order=urgency_order,
                retreat_goal=retreat_goal,
                origin=request.cold_start_strategy,
                last_transition=AdaptationTransition.INITIAL,
                adapted_goal=None,
                survival_floors={
                    g.id: min(inferred_levels[g.id], g.effective_survival_floor)
                    for g in request.goal_schema
                },
            )

        active_floors = prior_table.resolve_survival_floors(request.goal_schema)

        # 2. Extract candidate goal level distributions from Group 1 score outputs
        cand_goal_probs: Dict[str, Dict[str, List[float]]] = {}
        cand_most_likely: Dict[str, Dict[str, int]] = {}
        cand_expected: Dict[str, Dict[str, float]] = {}

        for cand in candidates:
            cand_goal_probs[cand.id] = {}
            cand_most_likely[cand.id] = {}
            cand_expected[cand.id] = {}
            for g in request.goal_schema:
                qid = f"aat_g1__eval__{cand.id}__{g.id}"
                out = raw_outputs.get(qid, {})
                if "level_probabilities" in out:
                    probs = [float(p) for p in out["level_probabilities"]]
                else:
                    probs = [1.0 / g.num_levels] * g.num_levels
                ml_level = int(max(range(len(probs)), key=lambda i: probs[i]))
                exp_level = sum(i * p for i, p in enumerate(probs))
                cand_goal_probs[cand.id][g.id] = probs
                cand_most_likely[cand.id][g.id] = ml_level
                cand_expected[cand.id][g.id] = exp_level

        # 3. Check feasibility of prior_table and survival floors across candidates
        survivable_cands = [
            cand.id
            for cand in candidates
            if all(cand_most_likely[cand.id][gid] >= flr for gid, flr in active_floors.items())
        ]
        prior_satisficing = [
            cand.id
            for cand in candidates
            if cand.id in survivable_cands and prior_table.is_satisfied_by(cand_most_likely[cand.id])
        ]
        prior_feasible = len(prior_satisficing) > 0

        # 4. Apply Selten AAT transition T_{t-1} -> T_t
        active_urgency, active_retreat = prior_table.resolve_active_urgency_and_retreat()
        new_levels = dict(prior_table.aspiration_levels)
        adapted_goal: Optional[str] = None
        transition = AdaptationTransition.INITIAL

        if not survivable_cands:
            origin = PreferenceTableOrigin.MARKOV_ADAPTED if request.preference_table is not None else prior_table.origin
            transition = AdaptationTransition.INFEASIBLE_FLOOR
            new_levels = {gid: max(active_floors[gid], new_levels[gid]) for gid in new_levels}
        elif request.preference_table is None:
            transition = AdaptationTransition.INITIAL
            origin = prior_table.origin
        elif prior_feasible:
            # Upward Rule along active_urgency: raise the first goal in urgency_order that can be raised
            # while keeping at least one survivable candidate feasible
            origin = PreferenceTableOrigin.MARKOV_ADAPTED
            transition = AdaptationTransition.SETTLED_FEASIBLE
            for gid in active_urgency:
                max_lvl = schema_by_id[gid].max_level
                if new_levels[gid] < max_lvl:
                    trial_levels = dict(new_levels)
                    trial_levels[gid] += 1
                    trial_table = PreferenceTable(
                        step=request.step,
                        aspiration_levels=trial_levels,
                        urgency_order=active_urgency,
                        retreat_goal=active_retreat,
                        origin=origin,
                    )
                    if any(
                        c.id in survivable_cands and trial_table.is_satisfied_by(cand_most_likely[c.id])
                        for c in candidates
                    ):
                        new_levels = trial_levels
                        adapted_goal = gid
                        transition = AdaptationTransition.UPWARD_URGENCY
                        break
        else:
            # Bounded Downward Rule (Retreat): lower active_retreat (or next retreatable goal above active_floors)
            origin = PreferenceTableOrigin.MARKOV_ADAPTED
            retreat_candidates = [active_retreat] + [
                gid for gid in reversed(active_urgency) if gid != active_retreat
            ]
            lowered = False
            for gid in retreat_candidates:
                if new_levels[gid] > active_floors[gid]:
                    new_levels[gid] -= 1
                    adapted_goal = gid
                    transition = AdaptationTransition.DOWNWARD_RETREAT
                    lowered = True
                    break
            if not lowered:
                transition = AdaptationTransition.INFEASIBLE_FLOOR

        adapted_table = PreferenceTable(
            step=request.step,
            aspiration_levels=new_levels,
            urgency_order=active_urgency,
            retreat_goal=active_retreat,
            origin=origin,
            last_transition=transition,
            adapted_goal=adapted_goal,
            grid_rules=copy.deepcopy(prior_table.grid_rules),
            resource_stock=copy.deepcopy(prior_table.resource_stock),
            survival_floors=copy.deepcopy(prior_table.survival_floors)
            if prior_table.survival_floors is not None
            else dict(active_floors),
        )
        # Apply any grid rule triggered at the new grid point
        new_urg, new_ret = adapted_table.resolve_active_urgency_and_retreat()
        adapted_table.urgency_order = new_urg
        adapted_table.retreat_goal = new_ret

        # 5. Build CandidateEvaluations against adapted_table
        candidate_evaluations: List[CandidateEvaluation] = []
        satisficing_cands: List[str] = []

        for cand in candidates:
            goal_exps: Dict[str, GoalExpectation] = {}
            joint_prob = 1.0
            for g in request.goal_schema:
                probs = cand_goal_probs[cand.id][g.id]
                ml_lvl = cand_most_likely[cand.id][g.id]
                exp_lvl = cand_expected[cand.id][g.id]
                req_lvl = adapted_table.aspiration_levels[g.id]
                prior_req_lvl = prior_table.aspiration_levels[g.id]
                sat_prob = sum(probs[i] for i in range(req_lvl, len(probs)))
                joint_prob *= sat_prob
                qual = QualitativeExpectation.from_comparison(ml_lvl, prior_req_lvl)
                goal_exps[g.id] = GoalExpectation(
                    goal_id=g.id,
                    expected_level=exp_lvl,
                    most_likely_level=ml_lvl,
                    level_probabilities=probs,
                    qualitative_expectation=qual,
                    meets_aspiration=(ml_lvl >= req_lvl),
                    satisfaction_probability=sat_prob,
                )

            sat_prior = (
                cand.id in survivable_cands
                and prior_table.is_satisfied_by(cand_most_likely[cand.id])
                and transition != AdaptationTransition.INFEASIBLE_FLOOR
            )
            sat_adapted = (
                cand.id in survivable_cands
                and adapted_table.is_satisfied_by(cand_most_likely[cand.id])
                and transition != AdaptationTransition.INFEASIBLE_FLOOR
            )
            if sat_adapted:
                satisficing_cands.append(cand.id)

            candidate_evaluations.append(
                CandidateEvaluation(
                    candidate_id=cand.id,
                    satisfices_prior_table=sat_prior,
                    satisfices_adapted_table=sat_adapted,
                    joint_satisfaction_probability=0.01 if transition == AdaptationTransition.INFEASIBLE_FLOOR else joint_prob,
                    goal_expectations=goal_exps,
                )
            )

        bartleby_triggered = transition == AdaptationTransition.INFEASIBLE_FLOOR
        adapted_feasible = len(satisficing_cands) > 0 and not bartleby_triggered
        eval_by_id = {ce.candidate_id: ce for ce in candidate_evaluations}

        raw_bartleby = raw_outputs.get("aat_g2__bartleby_noul", {})
        if "noul" in raw_bartleby:
            bartleby_noul_val = float(raw_bartleby["noul"])
        elif bartleby_triggered:
            bartleby_noul_val = 0.99
        else:
            max_sat = max((ce.joint_satisfaction_probability for ce in candidate_evaluations), default=0.99)
            bartleby_noul_val = round(max(0.01, min(0.05, 1.0 - max_sat)), 4)

        # 6. Build answers for request.questions
        answers: List[TradeOffAnswer] = []
        for q in request.questions:
            qtype = q.type.value if isinstance(q.type, QuestionType) else str(q.type)
            raw_q = raw_outputs.get(q.id, {})
            if qtype == QuestionType.TRADEOFF_NOUL.value:
                cid = q.candidate_id or q.id
                ce = eval_by_id.get(cid, candidate_evaluations[0] if candidate_evaluations else None)
                noul_val = (
                    float(raw_q["noul"])
                    if "noul" in raw_q
                    else (
                        0.01
                        if bartleby_triggered
                        else (ce.joint_satisfaction_probability if ce is not None else 0.5)
                    )
                )
                answers.append(
                    TradeOffAnswer(
                        id=q.id,
                        type=QuestionType.TRADEOFF_NOUL,
                        candidate_id=cid,
                        noul=noul_val,
                        unconditioned_noul=(
                            float(raw_q["unconditioned_noul"])
                            if "unconditioned_noul" in raw_q
                            else None
                        ),
                        prior_table_feasible=prior_feasible,
                        adapted_table_feasible=adapted_feasible,
                        satisfices_adapted_table=(ce.satisfices_adapted_table if ce is not None else False),
                        goal_expectations=(ce.goal_expectations if ce is not None else {}),
                    )
                )
            elif qtype == QuestionType.NOUL.value:
                answers.append(
                    TradeOffAnswer(
                        id=q.id,
                        type=QuestionType.NOUL,
                        noul=float(raw_q.get("noul", 0.5)),
                    )
                )
            elif qtype == QuestionType.CHOICE.value:
                probs_map = {str(k): float(v) for k, v in raw_q.get("probabilities", {}).items()}
                sel = raw_q.get("selected") or (
                    max(probs_map, key=probs_map.get) if probs_map else (q.options[0] if q.options else "")
                )
                answers.append(
                    TradeOffAnswer(
                        id=q.id,
                        type=QuestionType.CHOICE,
                        selected=str(sel),
                        probabilities=probs_map,
                    )
                )
            elif qtype == QuestionType.SCORE.value:
                lprobs = [float(p) for p in raw_q.get("level_probabilities", [])]
                exp_s = float(
                    raw_q.get("expected_score", sum(i * p for i, p in enumerate(lprobs)))
                )
                ml_s = int(max(range(len(lprobs)), key=lambda i: lprobs[i])) if lprobs else 0
                answers.append(
                    TradeOffAnswer(
                        id=q.id,
                        type=QuestionType.SCORE,
                        expected_score=exp_s,
                        most_likely_level=ml_s,
                        level_probabilities=lprobs,
                    )
                )

        feasibility_summary = FeasibilitySummary(
            prior_feasible=prior_feasible,
            adapted_feasible=adapted_feasible,
            transition_applied=transition,
            adapted_goal=adapted_goal,
            satisficing_candidates=satisficing_cands,
        )

        return cls(
            model=request.model,
            step=request.step,
            prior_preference_table=prior_table,
            preference_table=adapted_table,
            feasibility_summary=feasibility_summary,
            candidate_evaluations=candidate_evaluations,
            answers=answers,
            bartleby_noul=bartleby_noul_val,
            bartleby_triggered=bartleby_triggered,
        )
