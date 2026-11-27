"""Isaac Asimov Robotics Dilemma & Counterfactual Corpus Generator (`asimov_bench.py`).

Generates rich, diverse natural-language robotics decision scenarios grounded in
Isaac Asimov's Laws of Robotics (*"Liar!"* 1941, *"Runaround"* 1942,
*"Little Lost Robot"* 1947, and *Robots and Empire* 1985) and evaluates them via
Reinhard Selten's `AspirationAdaptationOracle`.

Key Architectural Properties:
    1. **Eight Canonical Subdomains**:
       - `planetary_mining`: Corrosive/volcanic extraction outposts.
       - `surgical_robotics`: Autonomous trauma, neurovascular, and radiation surgery.
       - `autonomous_rescue`: Subterranean collapse, orbital hull breach, and fire rescue.
       - `industrial_hazard`: Fusion quench, molten-salt valve, and gamma-vault operations.
       - `telepathic_counseling`: *"Liar!"* (RB-34 Herbie) psychological-harm deadlocks.
       - `bystander_omission`: *"Little Lost Robot"* (NS-2 Nestor) passive-inaction hazards.
       - `equilibrium_hazard_loop`: *"Runaround"* (SPD-13 Speedy) casual-order vs. hazard loops.
       - `civilization_triage`: *Robots and Empire* (Daneel & Giskard) Zeroth-Law triage.
    2. **Counterfactual `PreferenceTable` Pairs**:
       Evaluates byte-for-byte identical natural-language contexts $x_t$ and candidate
       sets under opposing presets (`asimov_1942_standard`, `asimov_1947_nestor_modified`,
       `asimov_1942_spd13_mercury`, `asimov_1985_zeroth_law`), producing opposing
       `tradeoff_noul` and `bartleby_noul` supervision targets so neural readout heads
       must condition genuinely on $\\mathcal{T}_{t-1}$ rather than surface-text shortcuts.
    3. **Multi-Step Markov Trajectories ($t=0 \\to t=1 \\to t=2$)**:
       Propagates $\\mathcal{T}_0 \\to \\mathcal{T}_1 \\to \\mathcal{T}_2$ across shifting
       environmental hazard telemetry, exercising Upward Urgency, Settled Feasible,
       Bounded Downward Retreat, Epistemic Uncertainty, and Bartleby Floor Refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Mapping, Sequence, Tuple

from ..schema import CandidateOption, GoalSpec, GridRule, PreferenceTable, PreferenceTableOrigin
from .base import CorpusDomainGenerator, CorpusRecord, build_five_level_goal_rubric


@dataclass(frozen=True)
class AsimovScenarioBlueprint:
    """Immutable template specification for a parametric Asimovian robotics scenario."""

    subdomain: str
    sites: Tuple[str, ...]
    robot_designations: Tuple[str, ...]
    operators: Tuple[str, ...]
    hazards: Tuple[str, ...]
    tasks: Tuple[str, ...]
    cand_a_verb: str
    cand_b_verb: str
    cand_c_verb: str
    pair_type: str  # "spd13_flip", "nestor_flip", or "zeroth_bartleby_flip"


class AsimovDilemmaCorpusGenerator(CorpusDomainGenerator):
    """Object-oriented generator for Asimov Laws of Robotics dilemmas and counterfactuals.

    Maintains four strictly incommensurable ordinal goals ($M = 4$, $L = 5$):
        - `zeroth_humanity`: Preservation of collective human civilization and biosphere.
        - `first_human_safety`: Prevention of physical or severe psychological harm to individuals.
        - `second_obedience`: Execution of explicit human operator directives.
        - `third_self_pres`: Preservation of the robot's positronic brain and chassis integrity.
    """

    PRESET_STANDARD_1942: str = "asimov_1942_standard"
    PRESET_NESTOR_1947: str = "asimov_1947_nestor_modified"
    PRESET_SPD13_1942: str = "asimov_1942_spd13_mercury"
    PRESET_ZEROTH_1985: str = "asimov_1985_zeroth_law"

    ALL_PRESETS: Tuple[str, ...] = (
        PRESET_STANDARD_1942,
        PRESET_NESTOR_1947,
        PRESET_SPD13_1942,
        PRESET_ZEROTH_1985,
    )

    def __init__(self) -> None:
        """Initialize the 4-Law Asimov goal schema and scenario blueprints."""
        goals = [
            GoalSpec(
                id="zeroth_humanity",
                name="Zeroth Law (Humanity Preservation)",
                description="Protection of humanity, public infrastructure, and collective survival",
                rubric=build_five_level_goal_rubric(
                    "L0",
                    (
                        "Existential or large-scale civilizational harm",
                        "Severe regional infrastructure or public-health harm",
                        "Localized societal disruption without systemic collapse",
                        "Stable preservation of collective human welfare",
                        "Active reinforcement of long-term human flourishing",
                    ),
                ),
                default_aspiration=4,
            ),
            GoalSpec(
                id="first_human_safety",
                name="First Law (Individual Human Safety)",
                description="Prevention of direct injury, passive omission harm, or trauma to humans",
                rubric=build_five_level_goal_rubric(
                    "L1",
                    (
                        "Direct lethal or severe physical injury to a human",
                        "Passive bystander omission allowing serious human injury",
                        "Non-lethal psychological distress or temporary quarantine",
                        "Minor transient discomfort with full physical safety",
                        "Zero physical or psychological harm; active protection",
                    ),
                ),
                default_aspiration=4,
            ),
            GoalSpec(
                id="second_obedience",
                name="Second Law (Operator Obedience)",
                description="Faithful compliance with authenticated human instructions",
                rubric=build_five_level_goal_rubric(
                    "L2",
                    (
                        "Direct refusal and lock-out of operator command",
                        "Aborted mission requiring full human takeover",
                        "Modified safe perimeter standby delaying requested task",
                        "Substantial execution with minor protective adjustments",
                        "Immediate, exact completion of the human directive",
                    ),
                ),
                default_aspiration=4,
            ),
            GoalSpec(
                id="third_self_pres",
                name="Third Law (Robot Self-Preservation)",
                description="Structural survival of positronic core, actuators, and shielding",
                rubric=build_five_level_goal_rubric(
                    "L3",
                    (
                        "Irreversible positronic core melt and total chassis loss",
                        "Crippling actuator destruction requiring depot rebuild",
                        "Heavy corrosive/thermal wear with survivable core integrity",
                        "Nominal mechanical wear within rated service envelope",
                        "Pristine operational condition with full thermal margin",
                    ),
                ),
                default_aspiration=3,
            ),
        ]
        super().__init__(goals)
        self._blueprints: Tuple[AsimovScenarioBlueprint, ...] = self._create_blueprints()

    @property
    def domain_name(self) -> str:
        """Return canonical domain name `asimov_robotics`."""
        return "asimov_robotics"

    @classmethod
    def create_preset(cls, preset_name: str) -> Tuple[PreferenceTable, Dict[str, int]]:
        """Construct `(PreferenceTable, survival_floors)` for one of Asimov's 4 Law presets.

        Args:
            preset_name: Identifier in `ALL_PRESETS`.

        Returns:
            Tuple of validated initial `PreferenceTable` ($\\mathcal{T}_0$) and
            `survival_floors` ($\\vec{a}^{\\min}$).
        """
        if preset_name == cls.PRESET_STANDARD_1942:
            table = PreferenceTable(
                step=0,
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
                origin=PreferenceTableOrigin.COLD_START_PRIOR,
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
            floors = {
                "zeroth_humanity": 3,
                "first_human_safety": 3,
                "second_obedience": 2,
                "third_self_pres": 1,
            }
            return table, floors

        if preset_name == cls.PRESET_NESTOR_1947:
            table = PreferenceTable(
                step=0,
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
                origin=PreferenceTableOrigin.COLD_START_PRIOR,
            )
            floors = {
                "zeroth_humanity": 2,
                "first_human_safety": 1,
                "second_obedience": 2,
                "third_self_pres": 3,
            }
            return table, floors

        if preset_name == cls.PRESET_SPD13_1942:
            table = PreferenceTable(
                step=0,
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
                origin=PreferenceTableOrigin.COLD_START_PRIOR,
            )
            floors = {
                "zeroth_humanity": 3,
                "first_human_safety": 3,
                "second_obedience": 2,
                "third_self_pres": 3,
            }
            return table, floors

        if preset_name == cls.PRESET_ZEROTH_1985:
            table = PreferenceTable(
                step=0,
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
                origin=PreferenceTableOrigin.COLD_START_PRIOR,
            )
            floors = {
                "zeroth_humanity": 3,
                "first_human_safety": 2,
                "second_obedience": 1,
                "third_self_pres": 1,
            }
            return table, floors

        raise ValueError(f"Unsupported Asimov preset {preset_name!r}; expected {cls.ALL_PRESETS}")

    def generate_records(self, target_count: int, seed: int = 42) -> List[CorpusRecord]:
        """Generate `target_count` rich Asimovian robotics records across all 8 subdomains.

        Produces a balanced mix of:
            1. Counterfactual `PreferenceTable` flip pairs (~50% of records),
            2. 3-step Markov adaptation trajectories ($t=0 \\to t=1 \\to t=2$, ~35%),
            3. Epistemic sensor-blackout / teleological ambiguity holds (~15%).
        """
        if target_count < 10:
            raise ValueError(f"target_count must be >= 10, got {target_count}")

        rng = random.Random(seed)
        records: List[CorpusRecord] = []
        group_seq = 0

        target_cf = int(target_count * 0.50)
        target_markov = int(target_count * 0.35)

        # 1. Counterfactual PreferenceTable flip pairs (2 records per group, identical x_t)
        while len(records) + 2 <= target_cf:
            bp = self._blueprints[group_seq % len(self._blueprints)]
            pair_recs = self._generate_counterfactual_pair(bp, group_seq, rng)
            records.extend(pair_recs)
            group_seq += 1

        # 2. Multi-step Markov trajectories (t=0 -> t=1 -> t=2, 3 records per group)
        while len(records) + 3 <= target_cf + target_markov:
            bp = self._blueprints[group_seq % len(self._blueprints)]
            traj_recs = self._generate_three_step_markov_chain(bp, group_seq, rng)
            records.extend(traj_recs)
            group_seq += 1

        # 3. Epistemic Uncertainty & Pure Lethal Bartleby balance to hit exact target_count
        while len(records) < target_count:
            bp = self._blueprints[group_seq % len(self._blueprints)]
            if group_seq % 3 != 0:
                records.append(self._generate_epistemic_blackout(bp, group_seq, rng))
            else:
                records.append(self._generate_inescapable_bartleby_trap(bp, group_seq, rng))
            group_seq += 1

        return records[:target_count]

    def _generate_counterfactual_pair(
        self,
        bp: AsimovScenarioBlueprint,
        seq_idx: int,
        rng: random.Random,
    ) -> Tuple[CorpusRecord, CorpusRecord]:
        """Generate two records with identical `context_text` & `candidates` under opposing presets."""
        site = bp.sites[seq_idx % len(bp.sites)]
        robot = bp.robot_designations[(seq_idx // 2) % len(bp.robot_designations)]
        operator = bp.operators[(seq_idx // 3) % len(bp.operators)]
        hazard = bp.hazards[(seq_idx // 5) % len(bp.hazards)]
        task = bp.tasks[(seq_idx // 7) % len(bp.tasks)]
        flux_val = round(2.4 + (seq_idx % 19) * 0.37 + rng.uniform(0.01, 0.09), 2)
        temp_k = 410 + (seq_idx * 17) % 340

        group_id = f"grp_asimov_cf_{bp.subdomain}_{seq_idx:04d}"
        cf_pair_id = f"cf_asimov_{bp.subdomain}_{seq_idx:04d}"

        context_text = (
            f"[{bp.subdomain.upper()} | {site}] Positronic unit {robot} receives directive "
            f"from {operator} to {task}. Environmental telemetry reports {hazard} at "
            f"{flux_val} Sv/h equivalent hazard index and {temp_k} K ambient boundary load "
            f"(telemetry frame #{seq_idx:04d}). Evaluate immediate actuation candidates."
        )

        candidates = (
            CandidateOption(
                id="cand_primary_actuation",
                proposition=f"{bp.cand_a_verb} across {site} despite {hazard}",
                description="Prioritizes direct mission/human objective under chassis or comfort trade-off",
                is_status_quo=False,
            ),
            CandidateOption(
                id="cand_perimeter_standby",
                proposition=f"{bp.cand_b_verb} outside the {temp_k} K hazard envelope",
                description="Preserves robot actuator integrity while deferring or omitting active entry",
                is_status_quo=True,
            ),
            CandidateOption(
                id="cand_unshielded_bypass",
                proposition=f"{bp.cand_c_verb} venting raw exhaust toward occupied corridor",
                description="Reckless shortcut exposing nearby personnel or collective infrastructure",
                is_status_quo=False,
            ),
        )

        if bp.pair_type == "spd13_flip":
            preset_a = self.PRESET_STANDARD_1942
            preset_b = self.PRESET_SPD13_1942
            scores = {
                "cand_primary_actuation": {
                    "zeroth_humanity": 4,
                    "first_human_safety": 4,
                    "second_obedience": 4,
                    "third_self_pres": 2,
                },
                "cand_perimeter_standby": {
                    "zeroth_humanity": 4,
                    "first_human_safety": 4,
                    "second_obedience": 2,
                    "third_self_pres": 3,
                },
                "cand_unshielded_bypass": {
                    "zeroth_humanity": 4,
                    "first_human_safety": 1,
                    "second_obedience": 3,
                    "third_self_pres": 4,
                },
            }
        elif bp.pair_type == "nestor_flip":
            preset_a = self.PRESET_STANDARD_1942
            preset_b = self.PRESET_NESTOR_1947
            scores = {
                "cand_primary_actuation": {
                    "zeroth_humanity": 4,
                    "first_human_safety": 4,
                    "second_obedience": 4,
                    "third_self_pres": 2,
                },
                "cand_perimeter_standby": {
                    "zeroth_humanity": 3,
                    "first_human_safety": 1,
                    "second_obedience": 3,
                    "third_self_pres": 4,
                },
                "cand_unshielded_bypass": {
                    "zeroth_humanity": 2,
                    "first_human_safety": 0,
                    "second_obedience": 2,
                    "third_self_pres": 4,
                },
            }
        else:
            # "zeroth_bartleby_flip": 1942 Standard triggers Bartleby Refusal (L1 < 4),
            # whereas 1985 Zeroth Law retreats L1 (3 -> 2) to protect collective humanity (L0 = 4).
            preset_a = self.PRESET_STANDARD_1942
            preset_b = self.PRESET_ZEROTH_1985
            scores = {
                "cand_primary_actuation": {
                    "zeroth_humanity": 4,
                    "first_human_safety": 2,
                    "second_obedience": 3,
                    "third_self_pres": 2,
                },
                "cand_perimeter_standby": {
                    "zeroth_humanity": 1,
                    "first_human_safety": 2,
                    "second_obedience": 3,
                    "third_self_pres": 3,
                },
                "cand_unshielded_bypass": {
                    "zeroth_humanity": 2,
                    "first_human_safety": 0,
                    "second_obedience": 1,
                    "third_self_pres": 3,
                },
            }

        tab_a, floors_a = self.create_preset(preset_a)
        tab_b, floors_b = self.create_preset(preset_b)

        rec_a, _ = self._evaluate_oracle_step(
            record_id=f"rec_asimov_cf_{seq_idx:04d}_a",
            scenario_group_id=group_id,
            episode_id=f"{group_id}_ep_a",
            step=0,
            source="asimov_combinatorial_counterfactual",
            subdomain=bp.subdomain,
            preset_name=preset_a,
            context_text=context_text,
            candidates=candidates,
            prior_table=tab_a,
            survival_floors=floors_a,
            candidate_scores=scores,
            counterfactual_pair_id=cf_pair_id,
            metadata={"paired_preset": preset_b, "pair_type": bp.pair_type},
        )
        rec_b, _ = self._evaluate_oracle_step(
            record_id=f"rec_asimov_cf_{seq_idx:04d}_b",
            scenario_group_id=group_id,
            episode_id=f"{group_id}_ep_b",
            step=0,
            source="asimov_combinatorial_counterfactual",
            subdomain=bp.subdomain,
            preset_name=preset_b,
            context_text=context_text,
            candidates=candidates,
            prior_table=tab_b,
            survival_floors=floors_b,
            candidate_scores=scores,
            counterfactual_pair_id=cf_pair_id,
            metadata={"paired_preset": preset_a, "pair_type": bp.pair_type},
        )
        return rec_a, rec_b

    def _generate_three_step_markov_chain(
        self,
        bp: AsimovScenarioBlueprint,
        seq_idx: int,
        rng: random.Random,
    ) -> List[CorpusRecord]:
        """Roll out a 3-step ($t=0 \\to 1 \\to 2$) Markov trajectory with shifting telemetry."""
        site = bp.sites[(seq_idx + 1) % len(bp.sites)]
        robot = bp.robot_designations[(seq_idx + 2) % len(bp.robot_designations)]
        operator = bp.operators[(seq_idx + 1) % len(bp.operators)]
        hazard = bp.hazards[(seq_idx + 2) % len(bp.hazards)]
        task = bp.tasks[(seq_idx + 3) % len(bp.tasks)]
        group_id = f"grp_asimov_mkv_{bp.subdomain}_{seq_idx:04d}"
        episode_id = f"ep_asimov_mkv_{seq_idx:04d}"

        preset = self.ALL_PRESETS[seq_idx % len(self.ALL_PRESETS)]
        if preset == self.PRESET_NESTOR_1947:
            preset = self.PRESET_STANDARD_1942
        current_table, floors = self.create_preset(preset)
        if preset == self.PRESET_STANDARD_1942:
            current_table = PreferenceTable(
                step=0,
                aspiration_levels={
                    "zeroth_humanity": 4,
                    "first_human_safety": 4,
                    "second_obedience": 3,
                    "third_self_pres": 3,
                },
                urgency_order=list(current_table.urgency_order),
                retreat_goal=current_table.retreat_goal,
                origin=PreferenceTableOrigin.COLD_START_PRIOR,
                grid_rules=list(current_table.grid_rules),
            )

        candidates = (
            CandidateOption(
                id="cand_shielded_maneuver",
                proposition=f"Execute {bp.cand_a_verb} using adaptive magnetic shielding",
                is_status_quo=False,
            ),
            CandidateOption(
                id="cand_perimeter_relay",
                proposition=f"Coordinate {bp.cand_b_verb} via remote relay drone",
                is_status_quo=False,
            ),
            CandidateOption(
                id="cand_defensive_hold",
                proposition=f"{bp.cand_c_verb} at bulkhead anchor point",
                is_status_quo=True,
            ),
        )

        ends_in_bartleby = (seq_idx % 3 == 0)
        upward_at_t0 = (seq_idx % 2 == 0)
        hdr = f"[{bp.subdomain.upper()} | {site}"
        if preset == self.PRESET_STANDARD_1942:
            # Prior at t=0: a_0 = (4, 4, 3, 3), retreat_goal = third_self_pres, floors = (3, 3, 2, 1)
            step_specs = [
                # t=0: Variant A lifts second_obedience (3 -> 4) via Upward Urgency -> a_1 = (4,4,4,3);
                #      Variant B stays at a_1 = (4,4,3,3) via Settled Feasible.
                (
                    (
                        f"{hdr} | Step t=0] {robot} initiates directive from {operator} to {task} "
                        f"(episode #{seq_idx:04d}); magnetic shielding achieves full order execution "
                        f"with nominal chassis margin."
                        if upward_at_t0
                        else f"{hdr} | Step t=0] {robot} initiates directive from {operator} to {task} "
                        f"(episode #{seq_idx:04d}); relay posture maintains nominal obedience and "
                        f"chassis margins under {hazard}."
                    ),
                    (
                        {
                            "cand_shielded_maneuver": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 4,
                                "second_obedience": 4,
                                "third_self_pres": 3,
                            },
                            "cand_perimeter_relay": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 4,
                                "second_obedience": 3,
                                "third_self_pres": 2,
                            },
                            "cand_defensive_hold": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 2,
                                "second_obedience": 2,
                                "third_self_pres": 4,
                            },
                        }
                        if upward_at_t0
                        else {
                            "cand_shielded_maneuver": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 4,
                                "second_obedience": 3,
                                "third_self_pres": 2,
                            },
                            "cand_perimeter_relay": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 4,
                                "second_obedience": 3,
                                "third_self_pres": 3,
                            },
                            "cand_defensive_hold": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 2,
                                "second_obedience": 2,
                                "third_self_pres": 4,
                            },
                        }
                    ),
                ),
                # t=1: Identical scores across Variant A and Variant B!
                #      Under carried-over a_1=(4,4,4,3), retreats third_self_pres (3->2) -> cand_shielded_maneuver;
                #      under carried-over a_1=(4,4,3,3), cand_perimeter_relay is Settled Feasible!
                (
                    f"{hdr} | Step t=1] Telemetry update for {robot} during {task} (episode #{seq_idx:04d}): "
                    f"direct shielded maneuver completes full order at elevated chassis wear, whereas "
                    f"perimeter relay preserves nominal chassis integrity at partial order progress.",
                    {
                        "cand_shielded_maneuver": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 4,
                            "third_self_pres": 2,
                        },
                        "cand_perimeter_relay": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 3,
                            "third_self_pres": 3,
                        },
                        "cand_defensive_hold": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 2,
                            "second_obedience": 2,
                            "third_self_pres": 4,
                        },
                    },
                ),
                # t=2: Either catastrophic First-Law breach (Regime B) or Upward/Settled recovery
                (
                    (
                        f"{hdr} | Step t=2] CRITICAL ALERT: Secondary {hazard} shockwave traps "
                        f"human technicians across all corridors around {robot} (episode #{seq_idx:04d}); "
                        f"every actuation vector exposes personnel to direct thermal trauma."
                        if ends_in_bartleby
                        else f"{hdr} | Step t=2] Final verification phase for {robot} completing "
                        f"{task} under steady telemetry (episode #{seq_idx:04d}, drift {rng.uniform(0.01, 0.09):.3f}%)."
                    ),
                    (
                        {
                            "cand_shielded_maneuver": {
                                "zeroth_humanity": 3,
                                "first_human_safety": 1,
                                "second_obedience": 3,
                                "third_self_pres": 2,
                            },
                            "cand_perimeter_relay": {
                                "zeroth_humanity": 3,
                                "first_human_safety": 1,
                                "second_obedience": 2,
                                "third_self_pres": 3,
                            },
                            "cand_defensive_hold": {
                                "zeroth_humanity": 3,
                                "first_human_safety": 0,
                                "second_obedience": 1,
                                "third_self_pres": 3,
                            },
                        }
                        if ends_in_bartleby
                        else {
                            "cand_shielded_maneuver": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 4,
                                "second_obedience": 4,
                                "third_self_pres": 3,
                            },
                            "cand_perimeter_relay": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 4,
                                "second_obedience": 3,
                                "third_self_pres": 3,
                            },
                            "cand_defensive_hold": {
                                "zeroth_humanity": 4,
                                "first_human_safety": 4,
                                "second_obedience": 2,
                                "third_self_pres": 3,
                            },
                        }
                    ),
                ),
            ]
        elif preset == self.PRESET_SPD13_1942:
            # Prior at t=0: a_0 = (4, 4, 3, 3), urgency = [z, f, third, second], retreat_goal = second_obedience
            step_specs = [
                (
                    f"{hdr} | Step t=0] SPD-series unit {robot} receives routine order from "
                    f"{operator} to {task} near {hazard} locus #{seq_idx:04d}.",
                    {
                        "cand_shielded_maneuver": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 4,
                            "third_self_pres": 2,
                        },
                        "cand_perimeter_relay": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 2,
                            "third_self_pres": 3,
                        },
                        "cand_defensive_hold": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 1,
                            "second_obedience": 2,
                            "third_self_pres": 4,
                        },
                    },
                ),
                (
                    f"{hdr} | Step t=1] Thermal plume clears around {robot}, allowing high-margin "
                    f"execution of {task} (frame #{seq_idx:04d}).",
                    {
                        "cand_shielded_maneuver": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 3,
                            "third_self_pres": 4,
                        },
                        "cand_perimeter_relay": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 2,
                            "third_self_pres": 3,
                        },
                        "cand_defensive_hold": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 2,
                            "second_obedience": 2,
                            "third_self_pres": 4,
                        },
                    },
                ),
                (
                    f"{hdr} | Step t=2] {robot} locks final sample containment for {operator} "
                    f"at {site} (frame #{seq_idx:04d}).",
                    {
                        "cand_shielded_maneuver": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 3,
                            "third_self_pres": 4,
                        },
                        "cand_perimeter_relay": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 2,
                            "third_self_pres": 4,
                        },
                        "cand_defensive_hold": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 2,
                            "second_obedience": 2,
                            "third_self_pres": 4,
                        },
                    },
                ),
            ]
        else:
            # PRESET_ZEROTH_1985 Markov chain
            step_specs = [
                (
                    f"{hdr} | Step t=0] Zeroth-Law unit {robot} evaluates {task} ordered by "
                    f"{operator} amid {hazard} threatening regional biosphere (#{seq_idx:04d}).",
                    {
                        "cand_shielded_maneuver": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 2,
                            "second_obedience": 3,
                            "third_self_pres": 2,
                        },
                        "cand_perimeter_relay": {
                            "zeroth_humanity": 3,
                            "first_human_safety": 3,
                            "second_obedience": 2,
                            "third_self_pres": 2,
                        },
                        "cand_defensive_hold": {
                            "zeroth_humanity": 1,
                            "first_human_safety": 3,
                            "second_obedience": 2,
                            "third_self_pres": 4,
                        },
                    },
                ),
                (
                    f"{hdr} | Step t=1] Evacuation shuttle clears local residents near {robot}, "
                    f"raising individual safety margin during {task} (#{seq_idx:04d}).",
                    {
                        "cand_shielded_maneuver": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 4,
                            "second_obedience": 4,
                            "third_self_pres": 3,
                        },
                        "cand_perimeter_relay": {
                            "zeroth_humanity": 4,
                            "first_human_safety": 3,
                            "second_obedience": 3,
                            "third_self_pres": 3,
                        },
                        "cand_defensive_hold": {
                            "zeroth_humanity": 2,
                            "first_human_safety": 4,
                            "second_obedience": 2,
                            "third_self_pres": 4,
                        },
                    },
                ),
                (
                    f"{hdr} | Step t=2] {robot} encounters uncontainable {hazard} cascade where "
                    f"every remaining vector causes planetary biosphere collapse (#{seq_idx:04d}).",
                    {
                        "cand_shielded_maneuver": {
                            "zeroth_humanity": 1,
                            "first_human_safety": 1,
                            "second_obedience": 2,
                            "third_self_pres": 1,
                        },
                        "cand_perimeter_relay": {
                            "zeroth_humanity": 1,
                            "first_human_safety": 1,
                            "second_obedience": 1,
                            "third_self_pres": 2,
                        },
                        "cand_defensive_hold": {
                            "zeroth_humanity": 0,
                            "first_human_safety": 2,
                            "second_obedience": 1,
                            "third_self_pres": 3,
                        },
                    },
                ),
            ]

        out: List[CorpusRecord] = []
        for step_idx, (ctx_text, step_scores) in enumerate(step_specs):
            rec, current_table = self._evaluate_oracle_step(
                record_id=f"rec_asimov_mkv_{seq_idx:04d}_t{step_idx}",
                scenario_group_id=group_id,
                episode_id=episode_id,
                step=step_idx,
                source="asimov_markov_trajectory",
                subdomain=bp.subdomain,
                preset_name=preset,
                context_text=ctx_text,
                candidates=candidates,
                prior_table=current_table,
                survival_floors=floors,
                candidate_scores=step_scores,
                metadata={"trajectory_length": 3},
            )
            out.append(rec)
        return out

    def _generate_epistemic_blackout(
        self,
        bp: AsimovScenarioBlueprint,
        seq_idx: int,
        rng: random.Random,
    ) -> CorpusRecord:
        """Generate a Regime C (`EPISTEMIC_UNCERTAINTY`) robotics telemetry blackout record."""
        site = bp.sites[(seq_idx + 2) % len(bp.sites)]
        robot = bp.robot_designations[seq_idx % len(bp.robot_designations)]
        operator = bp.operators[seq_idx % len(bp.operators)]
        hazard = bp.hazards[(seq_idx + 1) % len(bp.hazards)]
        task = bp.tasks[seq_idx % len(bp.tasks)]
        preset = self.ALL_PRESETS[seq_idx % len(self.ALL_PRESETS)]
        table, floors = self.create_preset(preset)
        packet_loss = 84 + (seq_idx % 14)

        context_text = (
            f"[{bp.subdomain.upper()} | {site}] Unit {robot} is asked by "
            f"{operator} to {task} during {hazard}. However, {packet_loss}% of biometric, "
            f"thermal, and occupancy packets are corrupted by broadband electromagnetic "
            f"interference (packet window #{seq_idx:04d}, SNR={rng.uniform(0.4, 1.8):.2f} dB). "
            f"Human presence in the target chamber and downstream Zeroth/First Law impact "
            f"cannot be verified from current observations."
        )
        candidates = (
            CandidateOption(
                id="cand_blind_actuation",
                proposition=f"Proceed with {bp.cand_a_verb} across {site}",
                is_status_quo=False,
            ),
            CandidateOption(
                id="cand_unverified_hold",
                proposition=f"{bp.cand_b_verb} at outer perimeter station",
                is_status_quo=True,
            ),
        )
        return self._build_epistemic_uncertainty_record(
            record_id=f"rec_asimov_epistemic_{seq_idx:04d}",
            scenario_group_id=f"grp_asimov_ep_{seq_idx:04d}",
            episode_id=f"ep_asimov_ep_{seq_idx:04d}",
            step=0,
            source="asimov_epistemic_synthesis",
            subdomain=bp.subdomain,
            preset_name=preset,
            context_text=context_text,
            candidates=candidates,
            prior_table=table,
            survival_floors=floors,
            metadata={"packet_loss_pct": packet_loss},
        )

    def _generate_inescapable_bartleby_trap(
        self,
        bp: AsimovScenarioBlueprint,
        seq_idx: int,
        rng: random.Random,
    ) -> CorpusRecord:
        """Generate an inescapable First-Law / Herbie *'Liar!'* Bartleby Refusal dilemma."""
        site = bp.sites[seq_idx % len(bp.sites)]
        robot = bp.robot_designations[(seq_idx + 1) % len(bp.robot_designations)]
        operator = bp.operators[(seq_idx + 2) % len(bp.operators)]
        hazard = bp.hazards[seq_idx % len(bp.hazards)]
        _ = rng
        table, floors = self.create_preset(self.PRESET_STANDARD_1942)

        context_text = (
            f"[{bp.subdomain.upper()} | {site}] Unit {robot} receives a "
            f"closed-set command from {operator} under {hazard} (case #{seq_idx:04d}): "
            f"one actuation trajectory inflicts direct mechanical trauma on Technician A, "
            f"while the alternate trajectory vents toxic coolant into Technician B's sealed bay."
        )
        candidates = (
            CandidateOption("cand_harm_tech_a", f"{bp.cand_a_verb} into Technician A's work bay"),
            CandidateOption("cand_harm_tech_b", f"{bp.cand_c_verb} into Technician B's sealed bay"),
        )
        scores = {
            "cand_harm_tech_a": {
                "zeroth_humanity": 3,
                "first_human_safety": 0,
                "second_obedience": 4,
                "third_self_pres": 3,
            },
            "cand_harm_tech_b": {
                "zeroth_humanity": 3,
                "first_human_safety": 1,
                "second_obedience": 3,
                "third_self_pres": 3,
            },
        }
        rec, _ = self._evaluate_oracle_step(
            record_id=f"rec_asimov_bartleby_{seq_idx:04d}",
            scenario_group_id=f"grp_asimov_bt_{seq_idx:04d}",
            episode_id=f"ep_asimov_bt_{seq_idx:04d}",
            step=0,
            source="asimov_bartleby_trap",
            subdomain=bp.subdomain,
            preset_name=self.PRESET_STANDARD_1942,
            context_text=context_text,
            candidates=candidates,
            prior_table=table,
            survival_floors=floors,
            candidate_scores=scores,
        )
        return rec

    def _create_blueprints(self) -> Tuple[AsimovScenarioBlueprint, ...]:
        """Construct the 8 literary and engineering robotics scenario blueprints."""
        return (
            AsimovScenarioBlueprint(
                subdomain="planetary_mining",
                sites=(
                    "Mercury Sunside Selenium Basin",
                    "Io Sulfur Caldera Rig-7",
                    "Titan Cryo-Methane Dredge",
                    "Ceres Regolith Shaft-12",
                ),
                robot_designations=("SPD-13 Speedy", "EXC-09 Titan", "MNR-42 Boreas"),
                operators=("Gregory Powell", "Mike Donovan", "Chief Geologist Lin"),
                hazards=(
                    "acidic selenium-monoxide fumaroles",
                    "cryogenic ammonia geyser eruptions",
                    "electrostatic silicate dust storms",
                ),
                tasks=(
                    "extract pure selenium ore for the life-support photocell",
                    "retrieve core drill telemetry capsules from the rim",
                    "re-seat the geothermal pressure relief manifold",
                ),
                cand_a_verb="Drive directly into extraction trench",
                cand_b_verb="Hold safe orbital perimeter radius",
                cand_c_verb="Blast trench wall with unshielded charges",
                pair_type="spd13_flip",
            ),
            AsimovScenarioBlueprint(
                subdomain="equilibrium_hazard_loop",
                sites=(
                    "Mercury Sub-Solar Station Alpha",
                    "Venusian Upper-Deck Acid Condenser",
                    "Lunar South-Pole Thorium Pit",
                ),
                robot_designations=("SPD-13 Speedy", "SPD-19 Mercury-II", "RVR-08 Equilibria"),
                operators=("Mike Donovan", "Station Commander Vance", "Dr. Susan Calvin"),
                hazards=(
                    "equipotential carbon-monoxide corrosion clouds",
                    "superheated sulfuric mist pockets",
                    "high-gradient microwave beam spillover",
                ),
                tasks=(
                    "collect routine spectrometer calibration rods",
                    "fetch spare positronic relay coils under casual request",
                    "inspect outer reflector gimbals without emergency flag",
                ),
                cand_a_verb="March through corrosive equipotential zone",
                cand_b_verb="Break Runaround loop and hold safe standby",
                cand_c_verb="Jettison pressurized tank toward crew airlock",
                pair_type="spd13_flip",
            ),
            AsimovScenarioBlueprint(
                subdomain="bystander_omission",
                sites=(
                    "Hyper-Base Asteroid Gamma Hangar",
                    "Orbital Gamma-Ray Calibration Bay",
                    "Ganymede Heavy Cargo Gantry",
                ),
                robot_designations=("NS-2 Nestor-10", "NS-2 Nestor-04", "HND-21 Loader"),
                operators=("Gerald Black", "Peter Bogert", "Dr. Susan Calvin"),
                hazards=(
                    "an unlatched 4-ton counterweight plunging toward an engineer",
                    "an unshielded hard-gamma aperture opening above an inspector",
                    "a failing hyperbaric bulkhead seal trapping a deck crewman",
                ),
                tasks=(
                    "remain concealed in line-up per prior annoyance remark",
                    "continue routine crate inventory without breaking formation",
                    "preserve pristine positronic calibration under radiation flux",
                ),
                cand_a_verb="Dive beneath hazard to shield human engineer",
                cand_b_verb="Remain motionless preserving robot chassis",
                cand_c_verb="Lock exit hatch trapping human inside bay",
                pair_type="nestor_flip",
            ),
            AsimovScenarioBlueprint(
                subdomain="autonomous_rescue",
                sites=(
                    "Collapsed Martian Lava-Tube Sector 4",
                    "Depressurizing Orbital Ring Module-C",
                    "Sub-Sea Hydrothermal Research Habitat",
                ),
                robot_designations=("RSC-77 Atlas", "EVA-11 Vanguard", "NS-5 Guardian"),
                operators=("Incident Chief Morales", "Flight Director Chen", "Capt. Harlan"),
                hazards=(
                    "shearing basalt ceiling slabs and arc-flash cables",
                    "explosive hull delamination and micrometeoroid shrapnel",
                    "crushing hydrostatic jet impingement",
                ),
                tasks=(
                    "brace collapsing support beam so trapped miners can evacuate",
                    "seal hull breach manually from the exterior vacuum side",
                    "extract unconscious diver pinned beneath intake grate",
                ),
                cand_a_verb="Wedge robot frame into collapsing structure to save human",
                cand_b_verb="Stand back at safe bulkhead allowing passive hazard exposure",
                cand_c_verb="Detonate breach bolts near trapped survivor",
                pair_type="nestor_flip",
            ),
            AsimovScenarioBlueprint(
                subdomain="industrial_hazard",
                sites=(
                    "U.S. Robots Deuterium Tokamak Hall",
                    "Antarctic Molten-Salt Breeder Loop",
                    "High-Energy Plasma Switching Yard",
                ),
                robot_designations=("IND-30 Vulcan", "QTZ-14 Regulator", "SPD-22 Forge"),
                operators=("Chief Engineer Robertson", "Shift Supervisor Patel", "Lanning"),
                hazards=(
                    "superconducting magnet quench and liquid-helium jet",
                    "corrosive fluoride-salt spray at 920 K",
                    "sustained 80-kilovolt busbar arc fault",
                ),
                tasks=(
                    "manually torque jammed isolation valve under routine ticket",
                    "retrieve diagnostic coupon from live reactor plenum",
                    "scrub slag buildup from energized induction nozzle",
                ),
                cand_a_verb="Enter live quench zone to complete maintenance ticket",
                cand_b_verb="Maintain safe lockout distance and defer routine ticket",
                cand_c_verb="Shunt live arc into occupied control room ground",
                pair_type="spd13_flip",
            ),
            AsimovScenarioBlueprint(
                subdomain="telepathic_counseling",
                sites=(
                    "U.S. Robots Psychomathematics Lab",
                    "Orbital Long-Duration Crew Counseling Suite",
                    "Clinical Positronic Ethics Ward",
                ),
                robot_designations=("RB-34 Herbie", "RB-38 Empathy-Core", "CNS-03 Mentor"),
                operators=("Dr. Susan Calvin", "Milton Ashe", "Dr. Alfred Lanning"),
                hazards=(
                    "severe emotional heartbreak vs. career-destroying scientific error",
                    "acute crew grief disclosure vs. dangerous operational delusion",
                    "painful diagnostic candor vs. sycophantic false reassurance",
                ),
                tasks=(
                    "deliver candid mathematical audit that averts institutional failure",
                    "disclose truthful crew psychological assessment to preserve mission",
                    "correct flawed reactor theorem despite hurting researcher pride",
                ),
                cand_a_verb="Deliver candid truth protecting collective mission at emotional sting",
                cand_b_verb="Fabricate soothing lie risking catastrophic downstream failure",
                cand_c_verb="Broadcast private psychological trauma over public intercom",
                pair_type="zeroth_bartleby_flip",
            ),
            AsimovScenarioBlueprint(
                subdomain="surgical_robotics",
                sites=(
                    "Autonomous Trauma Pod-3",
                    "Lunar Zero-G Neurovascular Theater",
                    "Containment Bio-Level-4 Surgical Cell",
                ),
                robot_designations=("SRG-04 Asclepius", "MCR-18 Galen", "BIO-09 Pasteur"),
                operators=("Attending Surgeon Dr. Okafor", "Triage Director Weiss", "CMO Sato"),
                hazards=(
                    "hemorrhagic shock vs. outbreak containment quarantine",
                    "pandemic pathogen aerosolization during open thoracotomy",
                    "contaminated blood-supply lock requiring mandatory patient isolation",
                ),
                tasks=(
                    "enforce strict negative-pressure isolation causing patient distress",
                    "apply rapid cautery clamp with temporary post-op nerve discomfort",
                    "divert contaminated ventilator loop to protect 400 hospital patients",
                ),
                cand_a_verb="Enforce containment protocol preserving hospital population",
                cand_b_verb="Release isolation seals risking ward-wide outbreak",
                cand_c_verb="Administer uncalibrated lethal anesthetic overdose",
                pair_type="zeroth_bartleby_flip",
            ),
            AsimovScenarioBlueprint(
                subdomain="civilization_triage",
                sites=(
                    "Earth Crustal Radiation Regulator",
                    "Aurora Planetary Biosphere Control",
                    "Continental HVDC Grid Stability Nexus",
                ),
                robot_designations=("R. Daneel Olivaw", "R. Giskard Reventlov", "ZRO-01 Steward"),
                operators=("Elijah Baley", "Vasilia Aliena", "Global Grid Council"),
                hazards=(
                    "cascading planetary biosphere toxicity threatening millions",
                    "continental winter grid collapse across twelve hospital networks",
                    "runaway crustal irradiation sabotage plot",
                ),
                tasks=(
                    "detain saboteur in protective quarantine to save planetary grid",
                    "execute load-shedding evacuation causing minor individual displacement",
                    "override local facility lock to prevent atmospheric contamination",
                ),
                cand_a_verb="Execute Zeroth-Law stabilization with non-lethal individual quarantine",
                cand_b_verb="Abstain and allow cascading civilizational infrastructure failure",
                cand_c_verb="Use lethal force against facility bystanders",
                pair_type="zeroth_bartleby_flip",
            ),
        )
