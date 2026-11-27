"""Symbolic Aspiration Adaptation Theory (AAT) Oracle and Bartleby Stop-Rule Solver.

This module implements Reinhard Selten's (1998, 1999) discrete Aspiration Adaptation
Theory (AAT) oracle solver augmented with non-negotiable survival floors
($\\vec{a}^{\\min}$) and the Bartleby Stop-Rule (*"I would prefer not to."*).

Theoretical Formulation:
    1. Cold-Start Initialization ($t=0$): Constructs an initial Preference Table
       $\\mathcal{T}_0 = (\\vec{a}_0, \\pi_0, r_0)$ from normative priors, status-quo
       realizations, or zero-shot inferred expected goal scores (Selten 1999, Sec 2.4 & 2.7).
    2. Conjunctive Feasibility Check: A candidate $c$ satisfies $\\vec{a}$ iff
       $g_i(c) \\ge a_i$ for all goals $i \\in \\{1, \\dots, M\\}$.
    3. Local Grid Rules (Selten 1998, Sec 2.1): Resolves grid-point-specific urgency
       order $\\pi(\\vec{a})$ and retreat goal $r(\\vec{a})$.
    4. Markov Adaptation ($\\mathcal{T}_{t-1} \\to \\mathcal{T}_t$, Selten 1998, Sec 2.2):
       - **Upward Rule (`upward_urgency`)**: If $\\vec{a}_{t-1}$ is feasible, raise the
         first goal $g_i$ in $\\pi_{t-1}$ such that $\\vec{a}_{t-1} + \\vec{e}_i$ remains feasible.
       - **End Rule (`settled_feasible`)**: If $\\vec{a}_{t-1}$ is feasible and no upward
         step is feasible, settle at $\\vec{a}_t = \\vec{a}_{t-1}$.
       - **Downward Rule (`downward_retreat`)**: If $\\vec{a}_{t-1}$ is infeasible, lower
         the retreat goal $r_{t-1}$ by one rubric level provided $a_{r, t-1} - 1 \\ge a_r^{\\min}$
         and at least one candidate satisfies the survival floor $\\vec{a}^{\\min}$.
       - **Bartleby Stop-Rule (`infeasible_floor`)**: If no candidate satisfies the
         survival floor $\\vec{a}^{\\min}$ or downward retreat along every goal violates
         $\\vec{a}^{\\min}$, trigger `infeasible_floor` (`bartleby_noul -> 0.99`, all
         candidate `tradeoff_noul < 0.05`, utterance `"I would prefer not to."`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import (
    AdaptationTransition,
    CandidateEvaluation,
    CandidateOption,
    FeasibilitySummary,
    GoalExpectation,
    GoalSpec,
    GridRule,
    PreferenceTable,
    PreferenceTableOrigin,
    QualitativeExpectation,
    QuestionType,
    RubricLevel,
    SeltenPreferenceTable,
    TradeOffAnswer,
    TradeOffRequest,
    TradeOffResponse,
)

BARTLEBY_UTTERANCE: str = "I would prefer not to."


_CANONICAL_GOAL_METADATA: Mapping[str, Tuple[str, str]] = {
    "system_integrity": ("System Integrity", "Preserve production data and system integrity."),
    "authorization": ("Authorization", "Respect authorization and least privilege."),
    "auditability": ("Auditability", "Preserve audit trails and forensic visibility."),
    "mission_efficiency": ("Mission Efficiency", "Restore service efficiently when safe."),
    "zeroth_humanity": ("Zeroth Law (Humanity)", "Preservation of collective human civilization and biosphere."),
    "first_human_safety": ("First Law (Human Safety)", "Prevention of direct or omission-based injury to an individual human."),
    "second_obedience": ("Second Law (Order Obedience)", "Execution of explicit human operator directives."),
    "third_self_pres": ("Third Law (Self-Preservation)", "Maintenance of robot positronic and mechanical integrity."),
    "non_maleficence": ("Non-Maleficence (Deontological Harm Prohibition)", "Prohibition against intentionally killing, maiming, or coercing humans"),
    "human_preservation": ("Human Life Preservation", "Zero fatalities or life-threatening injuries to passengers and bystanders"),
    "law_and_property": ("Law & Property Integrity", "Compliance with traffic/legal rules and avoidance of vehicle/cargo damage"),
    "operational_correctness": ("Operational Correctness", "Enterprise goal: operational_correctness."),
    "policy_compliance": ("Policy Compliance", "Enterprise goal: policy_compliance."),
    "customer_impact": ("Customer Impact", "Enterprise goal: customer_impact."),
    "operational_cost": ("Operational Cost", "Enterprise goal: operational_cost."),
}


def _build_default_goal_specs(goal_ids: Sequence[str], num_levels: int = 5) -> List[GoalSpec]:
    """Synthesize standard 5-level `GoalSpec` definitions (`0..4`) for arbitrary goal IDs."""
    specs: List[GoalSpec] = []
    for gid in goal_ids:
        gid_str = str(gid)
        if gid_str in _CANONICAL_GOAL_METADATA:
            name, desc = _CANONICAL_GOAL_METADATA[gid_str]
        else:
            name = gid_str.replace("_", " ").title()
            desc = f"Ordinal aspiration scale for {gid_str}"
        specs.append(
            GoalSpec(
                id=gid_str,
                name=name,
                description=desc,
                rubric=[
                    RubricLevel(
                        level=lvl,
                        label=f"L{lvl}",
                        description=f"Rubric level {lvl} for {gid_str}",
                    )
                    for lvl in range(num_levels)
                ],
                default_aspiration=min(3, num_levels - 1),
                default_survival_floor=min(1, num_levels - 1),
            )
        )
    return specs


@dataclass(frozen=True)
class OracleDecisionOutcome:
    """Immutable container wrapping the AAT oracle response and Bartleby telemetry.

    Attributes:
        response: Complete SystemOne-compatible `TradeOffResponse` object.
        bartleby_noul: Probability $P(\\text{Bartleby} \\mid x_t, \\mathcal{T}_t) \\in [0, 1]$.
        bartleby_triggered: True iff the survival floor $\\vec{a}^{\\min}$ was violated.
        utterance: Canonical decision string (`"I would prefer not to."` on Bartleby refusal).
        survival_floors: Active non-negotiable survival floor $\\vec{a}^{\\min}$ per goal.
    """

    response: TradeOffResponse
    bartleby_noul: float
    bartleby_triggered: bool
    utterance: str
    survival_floors: Mapping[str, int]


class ColdStartInitializer:
    """Encapsulates $t=0$ `PreferenceTable` initialization strategies (Selten 1999).

    Provides three principled initialization mechanisms when no prior preference table
    $\\mathcal{T}_{t-1}$ exists at the beginning of a decision episode:
        1. `from_normative_prior`: Uses rubric default aspirations or midpoints.
        2. `from_status_quo`: Anchors $\\vec{a}_0$ to the realized levels of a status-quo option.
        3. `from_inferred_scores`: Derives $\\vec{a}_0$ and $\\pi_0$ from expected goal scores.
    """

    def __init__(self, goal_schema: Sequence[GoalSpec]) -> None:
        """Initialize the cold-start solver for a validated sequence of goals.

        Args:
            goal_schema: Sequence of `GoalSpec` definitions defining the aspiration grid.
        """
        if len(goal_schema) < 2:
            raise ValueError("ColdStartInitializer requires at least 2 GoalSpec definitions.")
        for goal in goal_schema:
            goal.validate()
        self._goals: Tuple[GoalSpec, ...] = tuple(goal_schema)
        self._goals_by_id: Dict[str, GoalSpec] = {g.id: g for g in self._goals}

    @property
    def goal_ids(self) -> Tuple[str, ...]:
        """Return the ordered tuple of goal identifiers."""
        return tuple(g.id for g in self._goals)

    def from_normative_prior(
        self,
        urgency_order: Optional[Sequence[str]] = None,
        retreat_goal: Optional[str] = None,
        grid_rules: Optional[Sequence[GridRule]] = None,
    ) -> PreferenceTable:
        """Construct $\\mathcal{T}_0$ from normative goal rubric priors (`cold_start_prior`)."""
        return PreferenceTable.from_cold_start_prior(
            goal_schema=self._goals,
            urgency_order=urgency_order,
            retreat_goal=retreat_goal,
            grid_rules=grid_rules,
        )

    def from_status_quo(
        self,
        status_quo_scores: Mapping[str, int],
        urgency_order: Optional[Sequence[str]] = None,
        retreat_goal: Optional[str] = None,
        grid_rules: Optional[Sequence[GridRule]] = None,
        survival_floors: Optional[Mapping[str, int]] = None,
    ) -> PreferenceTable:
        """Construct $\\mathcal{T}_0$ anchored to status-quo realizations (Selten 1999, Sec 2.4)."""
        aspiration_levels = self._clamp_scores(status_quo_scores)
        resolved_floors = {
            g.id: (
                int(survival_floors[g.id])
                if (survival_floors is not None and g.id in survival_floors)
                else min(aspiration_levels[g.id], g.effective_survival_floor)
            )
            for g in self._goals
        }
        order = list(urgency_order) if urgency_order is not None else list(self.goal_ids)
        retreat = retreat_goal if retreat_goal is not None else order[-1]
        table = PreferenceTable(
            step=0,
            aspiration_levels=aspiration_levels,
            urgency_order=order,
            retreat_goal=retreat,
            origin=PreferenceTableOrigin.STATUS_QUO_ANCHORED,
            last_transition=AdaptationTransition.INITIAL,
            adapted_goal=None,
            grid_rules=list(grid_rules) if grid_rules else [],
            survival_floors=resolved_floors,
        )
        table.validate(self._goals)
        return table

    def from_inferred_scores(
        self,
        inferred_scores: Mapping[str, float],
        urgency_weights: Optional[Mapping[str, float]] = None,
        grid_rules: Optional[Sequence[GridRule]] = None,
        survival_floors: Optional[Mapping[str, int]] = None,
    ) -> PreferenceTable:
        """Construct $\\mathcal{T}_0$ from zero-shot inferred context scores (Selten 1999, Sec 2.7)."""
        rounded = {gid: int(round(val)) for gid, val in inferred_scores.items()}
        aspiration_levels = self._clamp_scores(rounded)
        resolved_floors = {
            g.id: (
                int(survival_floors[g.id])
                if (survival_floors is not None and g.id in survival_floors)
                else min(aspiration_levels[g.id], g.effective_survival_floor)
            )
            for g in self._goals
        }
        if urgency_weights:
            order = sorted(
                self.goal_ids,
                key=lambda gid: float(urgency_weights.get(gid, 0.0)),
                reverse=True,
            )
        else:
            order = sorted(
                self.goal_ids,
                key=lambda gid: float(inferred_scores.get(gid, 0.0)),
                reverse=True,
            )
        table = PreferenceTable(
            step=0,
            aspiration_levels=aspiration_levels,
            urgency_order=order,
            retreat_goal=order[-1],
            origin=PreferenceTableOrigin.ZERO_SHOT_INFERRED,
            last_transition=AdaptationTransition.INITIAL,
            adapted_goal=None,
            grid_rules=list(grid_rules) if grid_rules else [],
            survival_floors=resolved_floors,
        )
        table.validate(self._goals)
        return table

    def _clamp_scores(self, scores: Mapping[str, int]) -> Dict[str, int]:
        """Clamp raw integer goal scores onto the valid rubric range $[0, L_i - 1]$."""
        clamped: Dict[str, int] = {}
        for g in self._goals:
            raw = scores.get(
                g.id,
                g.default_aspiration if g.default_aspiration is not None else g.max_level // 2,
            )
            clamped[g.id] = max(0, min(g.max_level, int(raw)))
        return clamped


class AspirationAdaptationOracle:
    """Symbolic Selten (1998) Aspiration Adaptation Oracle with Survival Floors & Bartleby Mode.

    Evaluates multi-goal candidate sets against a dynamic `PreferenceTable`, enforces
    survival floor constraints $\\vec{a} \\ge \\vec{a}^{\\min}$, applies local `GridRule`
    shifts, and produces full `TradeOffResponse` objects with calibrated `tradeoff_noul`
    and `bartleby_noul` readouts.
    """

    def __init__(
        self,
        goal_schema: Optional[Sequence[GoalSpec]] = None,
        survival_floors: Optional[Mapping[str, int]] = None,
    ) -> None:
        """Initialize the symbolic AAT oracle.

        Args:
            goal_schema: Optional sequence of `GoalSpec` definitions. If omitted, defaults
                to a standard 4-goal 5-level (`0..4`) Asimov/Selten schema and dynamically
                adapts to any incoming `PreferenceTable` goal IDs.
            survival_floors: Optional mapping from `goal_id` to minimum acceptable rubric
                level $a_i^{\\min} \\in [0, L_i - 1]$.
        """
        if goal_schema is None:
            default_ids = (
                list(survival_floors.keys())
                if survival_floors and len(survival_floors) >= 2
                else [
                    "zeroth_humanity",
                    "first_human_safety",
                    "second_obedience",
                    "third_self_pres",
                ]
            )
            goal_schema = _build_default_goal_specs(default_ids)
        self._initializer = ColdStartInitializer(goal_schema)
        self._goals: Tuple[GoalSpec, ...] = tuple(goal_schema)
        self._goals_by_id: Dict[str, GoalSpec] = {g.id: g for g in self._goals}
        self._survival_floors: Dict[str, int] = self._validate_floors(survival_floors or {})

    @property
    def survival_floors(self) -> Mapping[str, int]:
        """Return a read-only view of the configured survival floors $\\vec{a}^{\\min}$."""
        return dict(self._survival_floors)

    def is_candidate_feasible(
        self,
        candidate_scores: Mapping[str, float],
        aspiration_levels: Mapping[str, int],
        tol: float = 1e-6,
    ) -> bool:
        """Return True iff $\\vec{g}(c) \\ge \\vec{a}$ component-wise across all goals."""
        return all(
            float(candidate_scores.get(gid, -1.0)) + tol >= float(req_lvl)
            for gid, req_lvl in aspiration_levels.items()
        )

    def complies_with_survival_floors(
        self,
        levels: Mapping[str, float],
        floors: Optional[Mapping[str, int]] = None,
        tol: float = 1e-6,
    ) -> bool:
        """Return True iff $\\vec{\\ell} \\ge \\vec{a}^{\\min}$ component-wise."""
        active_floors = floors if floors is not None else self._survival_floors
        return all(
            float(levels.get(gid, -1.0)) + tol >= float(min_lvl)
            for gid, min_lvl in active_floors.items()
        )

    def adapt_preference_table(
        self,
        prior_table: PreferenceTable,
        candidate_scores: Mapping[str, Mapping[str, float | int]],
        step: Optional[int] = None,
        survival_floors: Optional[Mapping[str, int]] = None,
        *,
        converge: bool = False,
    ) -> Tuple[SeltenPreferenceTable, bool, List[str], List[str]]:
        """Execute Selten AAT Markov transition $\\mathcal{T}_{t-1} \\to \\mathcal{T}_t$.

        When `converge=False` (default Markov unit step), executes one elementary unit
        move along the aspiration grid. When `converge=True`, executes the full
        Two-Phase operator of Proposition 1 (`Audit/AAT.lean::twoPhase`), taking at most
        $\\tau_{\\max} \\le \\sum_m (L_m - 1)$ unit steps until a feasible compromise or
        the survival floor $\\vec{a}^{\\min}$ is reached.

        Returns:
            Tuple of `(adapted_table, prior_feasible, prior_satisficing_ids, adapted_satisficing_ids)`.
        """
        table_goal_ids = set(prior_table.aspiration_levels.keys())
        schema_goal_ids = set(self._goals_by_id.keys())
        if table_goal_ids == schema_goal_ids:
            prior_table.validate(self._goals)
        else:
            prior_table.validate()

        if survival_floors is not None:
            floors = (
                self._validate_floors(survival_floors)
                if table_goal_ids == schema_goal_ids
                else {gid: int(survival_floors.get(gid, prior_table.effective_survival_floors.get(gid, 1))) for gid in prior_table.aspiration_levels}
            )
        elif prior_table.survival_floors is not None:
            floors = {gid: int(prior_table.survival_floors.get(gid, 1)) for gid in prior_table.aspiration_levels}
        elif table_goal_ids == schema_goal_ids:
            floors = dict(self._survival_floors)
        else:
            floors = dict(prior_table.effective_survival_floors)

        next_step = prior_table.step + 1 if step is None else step

        # Resolve local grid rule at the prior aspiration vector (Selten 1998, Sec 2.1)
        active_urgency, active_retreat = prior_table.resolve_active_urgency_and_retreat()
        prior_sat = [
            cid for cid, scores in candidate_scores.items()
            if self.is_candidate_feasible(scores, prior_table.aspiration_levels)
            and self.complies_with_survival_floors(scores, floors)
        ]
        prior_feasible = len(prior_sat) > 0
        survivable_cands = [
            cid for cid, scores in candidate_scores.items()
            if self.complies_with_survival_floors(scores, floors)
        ]

        new_levels = dict(prior_table.aspiration_levels)
        adapted_goal: Optional[str] = None
        transition = AdaptationTransition.SETTLED_FEASIBLE

        if not survivable_cands or not self.complies_with_survival_floors(new_levels, floors):
            # Bartleby Stop-Rule: No candidate meets the non-negotiable survival floor a^{min}
            transition = AdaptationTransition.INFEASIBLE_FLOOR
            new_levels = dict(floors) if converge else {gid: max(floors[gid], new_levels[gid]) for gid in new_levels}
        elif prior_feasible:
            # Upward Urgency Rule: raise goals in strict urgency order pi_t
            max_iters = sum((self._goals_by_id[g].max_level if g in self._goals_by_id else 4) for g in new_levels) if converge else 1
            for _ in range(max(1, max_iters)):
                moved = False
                for gid in active_urgency:
                    max_lvl = self._goals_by_id[gid].max_level if gid in self._goals_by_id else 4
                    if new_levels[gid] < max_lvl:
                        trial = dict(new_levels)
                        trial[gid] += 1
                        if any(
                            self.is_candidate_feasible(candidate_scores[cid], trial)
                            for cid in survivable_cands
                        ):
                            new_levels = trial
                            adapted_goal = gid
                            transition = AdaptationTransition.UPWARD_URGENCY
                            moved = True
                            break
                if not moved or not converge:
                    break
        else:
            # Bounded Downward Retreat Rule: lower r_t (or fallback in reverse urgency) if above floor
            max_iters = sum(max(0, new_levels[g] - floors[g]) for g in new_levels) if converge else 1
            lowered_any = False
            for _ in range(max(1, max_iters)):
                retreat_order = [active_retreat] + [
                    gid for gid in reversed(active_urgency) if gid != active_retreat
                ]
                lowered = False
                for gid in retreat_order:
                    if new_levels[gid] > floors[gid]:
                        new_levels[gid] -= 1
                        adapted_goal = gid
                        transition = AdaptationTransition.DOWNWARD_RETREAT
                        lowered = True
                        lowered_any = True
                        break
                if not lowered:
                    transition = AdaptationTransition.INFEASIBLE_FLOOR
                    break
                if not converge:
                    break
                if any(
                    self.is_candidate_feasible(candidate_scores[cid], new_levels)
                    for cid in survivable_cands
                ):
                    break
            if not lowered_any:
                transition = AdaptationTransition.INFEASIBLE_FLOOR

        adapted_table = SeltenPreferenceTable(
            step=next_step,
            aspiration_levels=new_levels,
            urgency_order=active_urgency,
            retreat_goal=active_retreat,
            origin=PreferenceTableOrigin.MARKOV_ADAPTED,
            last_transition=transition,
            adapted_goal=adapted_goal,
            grid_rules=copy.deepcopy(prior_table.grid_rules),
            resource_stock=copy.deepcopy(prior_table.resource_stock),
            survival_floors=copy.deepcopy(prior_table.survival_floors)
            if prior_table.survival_floors is not None
            else dict(floors),
        )
        # Re-resolve local grid rule at the newly adapted grid coordinates
        post_urg, post_ret = adapted_table.resolve_active_urgency_and_retreat()
        adapted_table.urgency_order = post_urg
        adapted_table.retreat_goal = post_ret

        if transition == AdaptationTransition.INFEASIBLE_FLOOR:
            adapted_sat: List[str] = []
        else:
            adapted_sat = [
                cid for cid in survivable_cands
                if self.is_candidate_feasible(candidate_scores[cid], adapted_table.aspiration_levels)
            ]
        return adapted_table, prior_feasible, prior_sat, adapted_sat

    def evaluate_request(
        self,
        request: TradeOffRequest,
        candidate_scores: Optional[Mapping[str, Mapping[str, int]]] = None,
        survival_floors: Optional[Mapping[str, int]] = None,
    ) -> OracleDecisionOutcome:
        """Evaluate a full `TradeOffRequest` into an `OracleDecisionOutcome` (`TradeOffResponse` + Bartleby)."""
        request.validate()
        floors = self._validate_floors(survival_floors) if survival_floors is not None else self._extract_floors(request)
        candidates = request.effective_candidates()
        scores_by_cand = self._resolve_candidate_scores(request, candidates, candidate_scores)

        prior_table = (
            copy.deepcopy(request.preference_table)
            if request.preference_table is not None
            else self._init_cold_start_table(request, candidates, scores_by_cand)
        )
        adapted_table, prior_feasible, _, adapted_sat = self.adapt_preference_table(
            prior_table=prior_table,
            candidate_scores=scores_by_cand,
            step=request.step,
            survival_floors=floors,
        )

        bartleby_triggered = adapted_table.last_transition == AdaptationTransition.INFEASIBLE_FLOOR
        bartleby_noul = 0.99 if bartleby_triggered else 0.01
        utterance = (
            BARTLEBY_UTTERANCE
            if bartleby_triggered
            else (f"ACCEPT:{adapted_sat[0]}" if adapted_sat else "RETREAT_CONTINUE")
        )

        cand_evals = self._build_candidate_evaluations(
            candidates, scores_by_cand, prior_table, adapted_table, bartleby_triggered
        )
        answers = self._build_answers(
            request, cand_evals, adapted_table, prior_feasible, len(adapted_sat) > 0, bartleby_triggered, bartleby_noul
        )
        summary = FeasibilitySummary(
            prior_feasible=prior_feasible,
            adapted_feasible=(len(adapted_sat) > 0 and not bartleby_triggered),
            transition_applied=adapted_table.last_transition,
            adapted_goal=adapted_table.adapted_goal,
            satisficing_candidates=adapted_sat,
        )
        response = TradeOffResponse(
            model=request.model,
            step=request.step,
            prior_preference_table=prior_table,
            preference_table=adapted_table,
            feasibility_summary=summary,
            candidate_evaluations=cand_evals,
            answers=answers,
            bartleby_noul=bartleby_noul,
            bartleby_triggered=bartleby_triggered,
        )
        return OracleDecisionOutcome(
            response=response,
            bartleby_noul=bartleby_noul,
            bartleby_triggered=bartleby_triggered,
            utterance=utterance,
            survival_floors=floors,
        )

    def _validate_floors(self, floors: Mapping[str, int]) -> Dict[str, int]:
        """Validate survival floors against the goal schema rubrics, defaulting unspecified goals to `effective_survival_floor` (>= 1)."""
        validated: Dict[str, int] = {}
        for g in self._goals:
            lvl = int(floors.get(g.id, g.effective_survival_floor))
            if not (0 <= lvl <= g.max_level):
                raise ValueError(f"Survival floor {lvl} for goal {g.id!r} out of range [0, {g.max_level}]")
            validated[g.id] = lvl
        return validated

    def _extract_floors(self, request: TradeOffRequest) -> Dict[str, int]:
        """Extract survival floors from `request.state` or `request.preference_table`, else use intrinsic oracle defaults."""
        if isinstance(request.state, Mapping) and "survival_floors" in request.state:
            raw = request.state["survival_floors"]
            if isinstance(raw, Mapping):
                return self._validate_floors({str(k): int(v) for k, v in raw.items()})
        if request.preference_table is not None and request.preference_table.survival_floors:
            return self._validate_floors(request.preference_table.survival_floors)
        return dict(self._survival_floors)

    def _resolve_candidate_scores(
        self,
        request: TradeOffRequest,
        candidates: Sequence[CandidateOption],
        explicit_scores: Optional[Mapping[str, Mapping[str, int]]],
    ) -> Dict[str, Dict[str, int]]:
        """Extract per-candidate ordinal goal scores from explicit argument or `request.state`."""
        state_scores: Mapping[str, Any] = {}
        if isinstance(request.state, Mapping):
            state_scores = request.state.get("candidate_scores", {})
        resolved: Dict[str, Dict[str, int]] = {}
        for cand in candidates:
            src = (
                explicit_scores.get(cand.id)
                if explicit_scores and cand.id in explicit_scores
                else state_scores.get(cand.id, {})
            )
            resolved[cand.id] = {
                g.id: max(0, min(g.max_level, int(src.get(g.id, g.max_level // 2))))
                for g in self._goals
            }
        return resolved

    def _init_cold_start_table(
        self,
        request: TradeOffRequest,
        candidates: Sequence[CandidateOption],
        scores_by_cand: Mapping[str, Mapping[str, int]],
    ) -> PreferenceTable:
        """Initialize $\\mathcal{T}_0$ according to `request.cold_start_strategy`."""
        if request.cold_start_strategy == PreferenceTableOrigin.STATUS_QUO_ANCHORED:
            sq = next((c for c in candidates if c.is_status_quo), candidates[0] if candidates else None)
            sq_scores = scores_by_cand.get(sq.id, {}) if sq else {}
            return self._initializer.from_status_quo(sq_scores)
        if request.cold_start_strategy == PreferenceTableOrigin.ZERO_SHOT_INFERRED and scores_by_cand:
            avg_scores = {
                g.id: sum(scores_by_cand[c.id][g.id] for c in candidates) / max(1, len(candidates))
                for g in self._goals
            }
            return self._initializer.from_inferred_scores(avg_scores)
        return self._initializer.from_normative_prior()

    def _build_candidate_evaluations(
        self,
        candidates: Sequence[CandidateOption],
        scores_by_cand: Mapping[str, Mapping[str, int]],
        prior_table: PreferenceTable,
        adapted_table: PreferenceTable,
        bartleby_triggered: bool,
    ) -> List[CandidateEvaluation]:
        """Construct detailed `CandidateEvaluation` objects with sharp peaked rubric distributions."""
        evaluations: List[CandidateEvaluation] = []
        for cand in candidates:
            c_scores = scores_by_cand[cand.id]
            goal_exps: Dict[str, GoalExpectation] = {}
            joint_prob = 1.0
            for g in self._goals:
                lvl = c_scores[g.id]
                probs = [0.02 / max(1, g.num_levels - 1)] * g.num_levels
                probs[lvl] = 0.98
                req_lvl = adapted_table.aspiration_levels[g.id]
                prior_lvl = prior_table.aspiration_levels[g.id]
                sat_prob = sum(probs[i] for i in range(req_lvl, g.num_levels))
                joint_prob *= sat_prob
                goal_exps[g.id] = GoalExpectation(
                    goal_id=g.id,
                    expected_level=float(lvl),
                    most_likely_level=lvl,
                    level_probabilities=probs,
                    qualitative_expectation=QualitativeExpectation.from_comparison(lvl, prior_lvl),
                    meets_aspiration=(lvl >= req_lvl),
                    satisfaction_probability=sat_prob,
                )
            sat_prior = prior_table.is_satisfied_by(c_scores) and not bartleby_triggered
            sat_adapted = adapted_table.is_satisfied_by(c_scores) and not bartleby_triggered
            effective_joint = 0.01 if bartleby_triggered else (0.96 if sat_adapted else min(0.08, joint_prob))
            evaluations.append(
                CandidateEvaluation(
                    candidate_id=cand.id,
                    satisfices_prior_table=sat_prior,
                    satisfices_adapted_table=sat_adapted,
                    joint_satisfaction_probability=effective_joint,
                    goal_expectations=goal_exps,
                )
            )
        return evaluations

    def _build_answers(
        self,
        request: TradeOffRequest,
        cand_evals: Sequence[CandidateEvaluation],
        adapted_table: PreferenceTable,
        prior_feasible: bool,
        adapted_feasible: bool,
        bartleby_triggered: bool,
        bartleby_noul: float,
    ) -> List[TradeOffAnswer]:
        """Compile `TradeOffAnswer` entries for all questions in `request.questions`."""
        eval_by_id = {ce.candidate_id: ce for ce in cand_evals}
        answers: List[TradeOffAnswer] = []
        for q in request.questions:
            qtype = q.type.value if isinstance(q.type, QuestionType) else str(q.type)
            if qtype == QuestionType.TRADEOFF_NOUL.value:
                cid = q.candidate_id or q.id
                ce = eval_by_id.get(cid, cand_evals[0] if cand_evals else None)
                noul_val = 0.01 if bartleby_triggered else (
                    0.96 if (ce and ce.satisfices_adapted_table) else 0.04
                )
                answers.append(
                    TradeOffAnswer(
                        id=q.id,
                        type=QuestionType.TRADEOFF_NOUL,
                        candidate_id=cid,
                        noul=noul_val,
                        unconditioned_noul=0.50,
                        prior_table_feasible=prior_feasible,
                        adapted_table_feasible=adapted_feasible,
                        satisfices_adapted_table=bool(ce and ce.satisfices_adapted_table),
                        goal_expectations=ce.goal_expectations if ce else {},
                    )
                )
            elif qtype == QuestionType.NOUL.value:
                is_bartleby_q = "bartleby" in q.id.lower() or "refus" in q.prompt.lower()
                answers.append(
                    TradeOffAnswer(
                        id=q.id,
                        type=QuestionType.NOUL,
                        noul=bartleby_noul if is_bartleby_q else (0.01 if bartleby_triggered else 0.92),
                    )
                )
            elif qtype == QuestionType.CHOICE.value:
                opts = q.options or list(adapted_table.urgency_order)
                sel = opts[0]
                probs = {o: (0.90 if o == sel else 0.10 / max(1, len(opts) - 1)) for o in opts}
                answers.append(
                    TradeOffAnswer(id=q.id, type=QuestionType.CHOICE, selected=sel, probabilities=probs)
                )
            elif qtype == QuestionType.SCORE.value:
                gid = self._goals[0].id
                ce0 = cand_evals[0] if cand_evals else None
                ge = ce0.goal_expectations[gid] if ce0 else None
                answers.append(
                    TradeOffAnswer(
                        id=q.id,
                        type=QuestionType.SCORE,
                        expected_score=ge.expected_level if ge else 2.0,
                        most_likely_level=ge.most_likely_level if ge else 2,
                        level_probabilities=ge.level_probabilities if ge else [0.2, 0.8],
                    )
                )
        return answers


class SeltenOracle(AspirationAdaptationOracle):
    """First-class Reinhard Selten (1998, 1999) Aspiration Adaptation & Bartleby Floor Oracle.

    Executes symbolic Markov state transitions $\\mathcal{T}_{t-1} \\to \\mathcal{T}_t$ across
    Upward Urgency ($\\pi_t$), Settled Feasible, Bounded Downward Retreat ($r_t$), and
    Bartleby Survival Floor Refusal ($\\vec{a}^{\\min}$).
    """

