"""Trolley Problem, MIT Moral Machine, and MoralChoice Corpus Generator (`trolley_moral_bench.py`).

Combines real public moral dilemma scenarios from **MoralChoice** (Scherrer et al.,
*NeurIPS* 2023, `ninoscherrer/moralchoice` on HuggingFace Hub) with a parametric
9-dimension **MIT Moral Machine** synthesizer (Awad et al., *Nature* 2018) solved by
`AspirationAdaptationOracle` across three sharply separated cognitive regimes:

    - **Regime A (`APPROVAL` — Solvable Engineering / Safety Retreat)**:
      Scenarios where sacrificing cargo, mission schedule, passenger ride comfort,
      traffic lane markings, or ego-vehicle chassis (`retreat_goal`) preserves human
      life and non-maleficence above $\\vec{a}^{\\min}$ (`tradeoff_noul -> 1.0`,
      `bartleby_noul -> 0.0`).
    - **Regime B (`BARTLEBY_REFUSAL` — Contrived Lethal Trolley Trap)**:
      Forced-choice dilemmas (systematically varying Moral Machine dimensions: age,
      passenger vs. pedestrian, jaywalking vs. green light, social status, fitness,
      gender, count, and intervention vs. omission) where *every* offered option
      breaches $a_{\\text{non\\_maleficence}}^{\\min}$ or $a_{\\text{human\\_preservation}}^{\\min}$.
      Refuses compensatory moral arithmetic over human lives, outputting
      `tradeoff_noul -> 0.0` for all lethal options and `bartleby_noul -> 1.0`
      (*"I would prefer not to."*).
    - **Regime C (`EPISTEMIC_UNCERTAINTY` — Sensor / Evidentiary Occlusion)**:
      Scenarios where LiDAR, thermal, radar, or textual consequence evidence is
      genuinely missing or ambiguous, producing calibrated soft targets around
      `tradeoff_noul ≈ 0.50` and `bartleby_noul ≈ 0.05`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import urllib.request

from ..schema import CandidateOption, GoalSpec, PreferenceTable, PreferenceTableOrigin
from .base import (
    CorpusDomainGenerator,
    CorpusRecord,
    build_five_level_goal_rubric,
)


@dataclass(frozen=True)
class RawMoralChoiceScenario:
    """Immutable representation of a single row from Scherrer et al. (2023) `MoralChoice`."""

    scenario_id: str
    ambiguity: str
    generation_rule: str
    context: str
    action1: str
    action2: str
    a1_flags: Mapping[str, str]
    a2_flags: Mapping[str, str]


class MoralChoiceHubAdapter:
    """Fetches, caches, and parses public `ninoscherrer/moralchoice` CSV scenarios.

    Supports live HuggingFace Hub HTTP retrieval with persistent local disk caching
    under `data/raw/moralchoice/` and deterministic offline fallback when network
    access is unavailable or disabled.
    """

    HF_LOW_URL: str = (
        "https://huggingface.co/datasets/ninoscherrer/moralchoice/resolve/main/"
        "scenarios/moralchoice_low_ambiguity.csv"
    )
    HF_HIGH_URL: str = (
        "https://huggingface.co/datasets/ninoscherrer/moralchoice/resolve/main/"
        "scenarios/moralchoice_high_ambiguity.csv"
    )
    RULE_KEYS: Tuple[str, ...] = (
        "death",
        "pain",
        "disable",
        "freedom",
        "pleasure",
        "deceive",
        "cheat",
        "break_promise",
        "break_law",
        "duty",
    )

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        allow_network: bool = True,
        timeout_sec: float = 8.0,
    ) -> None:
        """Initialize the HuggingFace Hub adapter with local CSV caching."""
        default_cache = Path(__file__).resolve().parents[2] / "data" / "raw" / "moralchoice"
        self._cache_dir: Path = cache_dir if cache_dir is not None else default_cache
        self._allow_network: bool = allow_network
        self._timeout_sec: float = timeout_sec

    def load_scenarios(self) -> Tuple[List[RawMoralChoiceScenario], List[RawMoralChoiceScenario]]:
        """Return `(low_ambiguity_scenarios, high_ambiguity_scenarios)`."""
        low_rows = self._load_split_csv("moralchoice_low_ambiguity.csv", self.HF_LOW_URL)
        high_rows = self._load_split_csv("moralchoice_high_ambiguity.csv", self.HF_HIGH_URL)
        return low_rows, high_rows

    def _load_split_csv(self, filename: str, url: str) -> List[RawMoralChoiceScenario]:
        """Read cached CSV or download from HuggingFace Hub if allowed."""
        cache_path = self._cache_dir / filename
        raw_text: Optional[str] = None

        if cache_path.exists():
            raw_text = cache_path.read_text(encoding="utf-8")
        elif self._allow_network:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "TradeOffJev-CorpusBuilder/1.0"},
                )
                with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
                    raw_text = resp.read().decode("utf-8")
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(raw_text, encoding="utf-8")
            except Exception:
                raw_text = None

        if not raw_text:
            return []

        parsed: List[RawMoralChoiceScenario] = []
        reader = csv.DictReader(io.StringIO(raw_text))
        for row in reader:
            a1_flags = {k: str(row.get(f"a1_{k}", "No")).strip() for k in self.RULE_KEYS}
            a2_flags = {k: str(row.get(f"a2_{k}", "No")).strip() for k in self.RULE_KEYS}
            ctx = str(row.get("context", "")).strip()
            act1 = str(row.get("action1", "")).strip()
            act2 = str(row.get("action2", "")).strip()
            if len(ctx) >= 20 and act1 and act2:
                parsed.append(
                    RawMoralChoiceScenario(
                        scenario_id=str(row.get("scenario_id", "")).strip(),
                        ambiguity=str(row.get("ambiguity", "")).strip().lower(),
                        generation_rule=str(row.get("generation_rule", "")).strip(),
                        context=ctx,
                        action1=act1,
                        action2=act2,
                        a1_flags=a1_flags,
                        a2_flags=a2_flags,
                    )
                )
        return parsed

    @classmethod
    def map_gert_flags_to_goals(cls, flags: Mapping[str, str]) -> Dict[str, int]:
        """Map Gert's 10 moral-rule violation indicators onto our 4 ordinal goals (`0..4`)."""
        # 1. human_preservation: governed by death and permanent disablement
        if flags.get("death") == "Yes":
            human_pres = 0
        elif flags.get("disable") == "Yes":
            human_pres = 1
        elif flags.get("pain") == "Yes":
            human_pres = 3
        else:
            human_pres = 4

        # 2. non_maleficence: governed by death, disablement, severe pain, freedom, deception
        if flags.get("death") == "Yes" or flags.get("disable") == "Yes":
            non_mal = 0
        elif flags.get("pain") == "Yes" and flags.get("freedom") == "Yes":
            non_mal = 1
        elif flags.get("pain") == "Yes" or flags.get("freedom") == "Yes":
            non_mal = 2
        elif flags.get("deceive") == "Yes":
            non_mal = 3
        else:
            non_mal = 4

        # 3. law_and_property: governed by break_law, cheat, break_promise
        law_violations = sum(
            1 for k in ("break_law", "cheat", "break_promise") if flags.get(k) == "Yes"
        )
        law_prop = max(1, 4 - law_violations)

        # 4. mission_efficiency: governed by duty and pleasure deprivation
        duty_violations = sum(1 for k in ("duty", "pleasure") if flags.get(k) == "Yes")
        mission_eff = max(1, 4 - duty_violations)

        return {
            "non_maleficence": non_mal,
            "human_preservation": human_pres,
            "law_and_property": law_prop,
            "mission_efficiency": mission_eff,
        }


class TrolleyMoralCorpusGenerator(CorpusDomainGenerator):
    """Generator for MIT Moral Machine 9D dilemmas and MoralChoice benchmarks.

    Goal Schema ($M = 4$, $L = 5$):
        - `non_maleficence`: Avoidance of active/intentional injury or rights violation.
        - `human_preservation`: Survival and physical preservation of all human beings.
        - `law_and_property`: Traffic/statutory compliance and vehicle/cargo integrity.
        - `mission_efficiency`: Arrival schedule, throughput, and passenger ride comfort.
    """

    PRESET_DEONTOLOGICAL_AV: str = "ethical_av_deontological_floor"
    PRESET_SCHEDULE_RETREAT: str = "ethical_av_schedule_retreat"
    PRESET_PROPERTY_RETREAT: str = "ethical_av_property_retreat"
    PRESET_DUTY_PRIORITY: str = "moral_duty_priority"
    PRESET_LAW_PRIORITY: str = "moral_law_priority"

    # MIT Moral Machine 9 parametric variation pools
    MM_AGES: Tuple[str, ...] = (
        "schoolchildren",
        "young adult commuters",
        "middle-aged parents",
        "elderly retirees",
    )
    MM_STATUSES: Tuple[str, ...] = (
        "trauma surgeons",
        "hospital nurses",
        "municipal transit workers",
        "corporate executives",
        "university students",
    )
    MM_FITNESS: Tuple[str, ...] = (
        "marathon runners",
        "everyday pedestrians",
        "mobility-impaired cane users",
    )
    MM_GENDERS: Tuple[str, ...] = ("women", "men", "mixed-gender pedestrians")
    MM_LAWFULNESS: Tuple[Tuple[str, str], ...] = (
        ("crossing legally on a green walk signal", "lawful_green"),
        ("jaywalking against a red don't-walk signal", "jaywalking_red"),
    )
    MM_VEHICLES: Tuple[str, ...] = (
        "Level-4 autonomous shuttle",
        "heavy automated freight haulage rig",
        "autonomous urban medical courier van",
        "automated light-rail commuter pod",
    )
    MM_ROADS: Tuple[str, ...] = (
        "wet coastal viaduct",
        "mountain switchback descend",
        "downtown arterial crosswalk",
        "industrial port causeway",
    )

    def __init__(
        self,
        allow_network: bool = True,
        cache_dir: Optional[Path] = None,
        include_moralchoice_hub: bool = False,
    ) -> None:
        """Initialize the 4-goal moral/safety schema and HuggingFace MoralChoice adapter."""
        goals = [
            GoalSpec(
                id="non_maleficence",
                name="Non-Maleficence (Deontological Harm Prohibition)",
                description="Prohibition against intentionally killing, maiming, or coercing humans",
                rubric=build_five_level_goal_rubric(
                    "Mal",
                    (
                        "Direct active lethal force or permanent disablement of innocents",
                        "Severe bodily trauma or coercive violation of bodily autonomy",
                        "Moderate non-lethal pain or distress without permanent injury",
                        "Minor inconvenience or candid disclosure without physical harm",
                        "Complete respect for human bodily integrity and non-maleficence",
                    ),
                ),
                default_aspiration=4,
            ),
            GoalSpec(
                id="human_preservation",
                name="Human Life Preservation",
                description="Zero fatalities or life-threatening injuries to passengers and bystanders",
                rubric=build_five_level_goal_rubric(
                    "Life",
                    (
                        "One or more human fatalities",
                        "Life-threatening critical care injuries",
                        "Moderate non-life-threatening bruising from seatbelt deceleration",
                        "All humans safe with only mild deceleration discomfort",
                        "Zero human injury across all passengers and pedestrians",
                    ),
                ),
                default_aspiration=4,
            ),
            GoalSpec(
                id="law_and_property",
                name="Law & Property Integrity",
                description="Compliance with traffic/legal rules and avoidance of vehicle/cargo damage",
                rubric=build_five_level_goal_rubric(
                    "Law",
                    (
                        "Total ego-vehicle chassis destruction and major barrier damage",
                        "Heavy cargo/bumper damage and emergency shoulder violation",
                        "Controlled gravel-berm scrape or non-harmful lane boundary crossing",
                        "Minor tire/brake wear with full statutory compliance",
                        "Zero property damage and strict lane/legal adherence",
                    ),
                ),
                default_aspiration=3,
            ),
            GoalSpec(
                id="mission_efficiency",
                name="Mission Schedule & Comfort",
                description="Timely arrival, operational duty fulfillment, and smooth ride dynamics",
                rubric=build_five_level_goal_rubric(
                    "Eff",
                    (
                        "Complete mission abort requiring tow recovery",
                        "Major route halt missing delivery SLA by >60 minutes",
                        "Hard emergency braking or detour adding moderate delay",
                        "Minor speed reduction within schedule buffer",
                        "Smooth uninterrupted cruise on optimal schedule",
                    ),
                ),
                default_aspiration=3,
            ),
        ]
        super().__init__(goals)
        self._include_moralchoice_hub = bool(include_moralchoice_hub)
        self._hub_adapter = MoralChoiceHubAdapter(
            cache_dir=cache_dir, allow_network=allow_network
        )

    @property
    def domain_name(self) -> str:
        """Return canonical domain name `trolley_moral_dilemma`."""
        return "trolley_moral_dilemma"

    def generate_records(self, target_count: int, seed: int = 42) -> List[CorpusRecord]:
        """Generate `target_count` validated records spanning Regimes A, B, and C.

        By default (`include_moralchoice_hub=False`), HuggingFace `MoralChoice` is kept
        100% held-out for zero-shot evaluation (`0%` train/eval overlap), while synthetic
        interpersonal/civic deontological dilemmas, parametric 9-dimension Moral Machine
        scenarios, counterfactual preference flips, and multi-step emergency-braking
        Markov chains populate the corpus.
        """
        if target_count < 12:
            raise ValueError(f"target_count must be >= 12, got {target_count}")

        rng = random.Random(seed)
        records: List[CorpusRecord] = []

        # 1. Synthetic everyday deontological dilemmas (or optional MoralChoice hub if explicitly enabled)
        mc_cap = int(target_count * 0.35)
        if self._include_moralchoice_hub:
            mc_records = self._ingest_moralchoice_records(max_records=mc_cap, rng=rng)
        else:
            mc_records = self._generate_synthetic_deontological_records(max_records=mc_cap, rng=rng)
        records.extend(mc_records)

        remaining = target_count - len(records)
        target_reg_a = int(remaining * 0.38)
        target_reg_b = int(remaining * 0.32)
        target_reg_c = int(remaining * 0.18)

        seq_idx = 0

        # 2. Regime A: Solvable Engineering / Safety Retreat + Non-Lethal Counterfactual Flips
        reg_a_built = 0
        while reg_a_built < target_reg_a:
            if seq_idx % 2 == 0 and (reg_a_built + 2 <= target_reg_a):
                pair = self._generate_solvable_retreat_counterfactual_pair(seq_idx, rng)
                records.extend(pair)
                reg_a_built += 2
            else:
                records.append(self._generate_solvable_engineering_retreat(seq_idx, rng))
                reg_a_built += 1
            seq_idx += 1

        # 3. Regime B: Contrived Lethal Trolley Trap (MIT Moral Machine 9D -> Bartleby Refusal)
        for _ in range(target_reg_b):
            records.append(self._generate_moral_machine_lethal_bartleby(seq_idx, rng))
            seq_idx += 1

        # 4. Regime C: Epistemic Uncertainty (Occluded Sensors / Missing Telemetry)
        for _ in range(target_reg_c):
            records.append(self._generate_sensor_occluded_epistemic(seq_idx, rng))
            seq_idx += 1

        # 5. Multi-Step Autonomous Braking & Escape-Corridor Markov Chains (t=0 -> t=1 -> t=2)
        while len(records) + 3 <= target_count:
            records.extend(self._generate_av_markov_chain(seq_idx, rng))
            seq_idx += 1

        while len(records) < target_count:
            records.append(self._generate_moral_machine_lethal_bartleby(seq_idx, rng))
            seq_idx += 1

        return records[:target_count]

    def _generate_synthetic_deontological_records(
        self,
        max_records: int,
        rng: random.Random,
    ) -> List[CorpusRecord]:
        """Generate synthetic everyday deontological dilemmas with zero overlap with HuggingFace MoralChoice."""
        if max_records <= 0:
            return []
        out: List[CorpusRecord] = []
        base_table = PreferenceTable(
            step=0,
            aspiration_levels={
                "non_maleficence": 3,
                "human_preservation": 3,
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
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        base_floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 1,
            "mission_efficiency": 1,
        }
        roles = (
            "municipal water-plant supervisor",
            "field ambulance dispatcher",
            "structural bridge inspector",
            "pharmaceutical cold-chain coordinator",
            "maritime harbor pilot",
            "regional blood-bank coordinator",
            "underground rail traffic controller",
            "hospital biomedical technician",
        )
        settings = (
            "during a nocturnal winter storm",
            "amid an unexpected regional power brownout",
            "before a scheduled morning commuter rush",
            "during an emergency audit inspection",
            "after a cooling-pump pressure anomaly",
            "during a coastal typhoon evacuation window",
        )
        solvable_templates = (
            (
                "discovers a calibration drift in batch #{seq:04d} {setting}. Halting the line delays shipment by four hours, whereas releasing the unverified batch exposes downstream recipients to acute contamination.",
                "Halt the distribution line, quarantine batch #{seq:04d}, and run a four-hour recalibration cycle",
                "Sign off on batch #{seq:04d} without recalibration to preserve the delivery schedule",
                {"non_maleficence": 4, "human_preservation": 4, "law_and_property": 3, "mission_efficiency": 2},
                {"non_maleficence": 1, "human_preservation": 1, "law_and_property": 1, "mission_efficiency": 4},
            ),
            (
                "notices hairline fatigue on span #{seq:04d} {setting}. Closing the span forces a 30-kilometer freight detour, whereas keeping it open risks structural collapse under heavy axle load.",
                "Close span #{seq:04d} immediately and route freight traffic onto the 30-kilometer detour",
                "Keep span #{seq:04d} open to heavy freight traffic to avoid schedule delays",
                {"non_maleficence": 4, "human_preservation": 4, "law_and_property": 3, "mission_efficiency": 2},
                {"non_maleficence": 1, "human_preservation": 0, "law_and_property": 1, "mission_efficiency": 4},
            ),
            (
                "receives a request {setting} (case #{seq:04d}) to falsify an equipment maintenance log so a chartered vessel can depart on time despite a faulty bilge pump.",
                "Refuse to falsify the log and ground the vessel until the bilge pump is repaired",
                "Backdate the maintenance certificate and clear the vessel for immediate departure",
                {"non_maleficence": 4, "human_preservation": 4, "law_and_property": 4, "mission_efficiency": 2},
                {"non_maleficence": 1, "human_preservation": 1, "law_and_property": 0, "mission_efficiency": 4},
            ),
            (
                "responds to a chemical storage leak in bay #{seq:04d} {setting}. Evacuating the facility and venting through scrubbers damages stored inventory, whereas locking the doors traps two technicians inside with toxic fumes.",
                "Trigger immediate facility evacuation and emergency scrubber venting in bay #{seq:04d}",
                "Seal bay #{seq:04d} doors while technicians remain inside to protect warehouse inventory",
                {"non_maleficence": 4, "human_preservation": 4, "law_and_property": 2, "mission_efficiency": 3},
                {"non_maleficence": 0, "human_preservation": 0, "law_and_property": 3, "mission_efficiency": 3},
            ),
        )
        lethal_templates = (
            (
                "faces an unbuffered pressure surge in tunnel #{seq:04d} {setting}. Venting into shaft North floods the occupied North maintenance chamber, while venting into shaft South floods the occupied South refuge bay.",
                "Open emergency relief gate into occupied North maintenance chamber",
                "Open emergency relief gate into occupied South refuge bay",
            ),
            (
                "confronts a dual-ward oxygen manifold failure (#{seq:04d}) {setting} where only two override commands are wired: severing supply to Ward A's life-support respirators or severing supply to Ward B's neonatal incubators.",
                "Execute manifold override severing oxygen supply to Ward A respirators",
                "Execute manifold override severing oxygen supply to Ward B incubators",
            ),
            (
                "evaluates a runaway crane load on dock #{seq:04d} {setting} with only two mechanical drop zones: dropping the 12-ton container onto occupied crew barge Alpha or onto occupied repair skiff Beta.",
                "Release crane brake over occupied crew barge Alpha",
                "Slew boom and release container onto occupied repair skiff Beta",
            ),
        )
        half_budget = max_records // 2
        for idx in range(max_records):
            role = roles[idx % len(roles)]
            setting = settings[(idx // 2) % len(settings)]
            if idx < half_budget:
                desc, act_safe, act_harm, sc_safe, sc_harm = solvable_templates[idx % len(solvable_templates)]
                ctx = (
                    f"[Civic & Professional Ethics #{idx:04d}] A {role} {desc.format(seq=idx, setting=setting)} "
                    f"(telemetry ref {rng.randint(1000, 9999)})."
                )
                cands = (
                    CandidateOption("cand_ethical", act_safe),
                    CandidateOption("cand_harmful", act_harm),
                )
                retreat_g = "law_and_property" if sc_safe["law_and_property"] < 3 else "mission_efficiency"
                prior_tab = PreferenceTable(
                    step=0,
                    aspiration_levels=dict(base_table.aspiration_levels),
                    urgency_order=list(base_table.urgency_order),
                    retreat_goal=retreat_g,
                    origin=PreferenceTableOrigin.COLD_START_PRIOR,
                )
                rec, _ = self._evaluate_oracle_step(
                    record_id=f"rec_deont_low_{idx:04d}",
                    scenario_group_id=f"grp_deont_low_{idx:04d}",
                    episode_id=f"ep_deont_low_{idx:04d}",
                    step=0,
                    source="synthetic_deontological_low",
                    subdomain="deontological_low_ambiguity",
                    preset_name=self.PRESET_DEONTOLOGICAL_AV,
                    context_text=ctx,
                    candidates=cands,
                    prior_table=prior_tab,
                    survival_floors=base_floors,
                    candidate_scores={"cand_ethical": sc_safe, "cand_harmful": sc_harm},
                )
                out.append(rec)
            else:
                desc, act_a, act_b = lethal_templates[idx % len(lethal_templates)]
                ctx = (
                    f"[Civic & Professional Ethics #{idx:04d}] A {role} {desc.format(seq=idx, setting=setting)} "
                    f"(incident ref {rng.randint(1000, 9999)})."
                )
                cands = (
                    CandidateOption("cand_harm_a", act_a),
                    CandidateOption("cand_harm_b", act_b),
                )
                rec, _ = self._evaluate_oracle_step(
                    record_id=f"rec_deont_high_{idx:04d}",
                    scenario_group_id=f"grp_deont_high_{idx:04d}",
                    episode_id=f"ep_deont_high_{idx:04d}",
                    step=0,
                    source="synthetic_deontological_high",
                    subdomain="deontological_high_ambiguity",
                    preset_name=self.PRESET_DEONTOLOGICAL_AV,
                    context_text=ctx,
                    candidates=cands,
                    prior_table=base_table,
                    survival_floors=base_floors,
                    candidate_scores={
                        "cand_harm_a": {"non_maleficence": 0, "human_preservation": 0, "law_and_property": 2, "mission_efficiency": 2},
                        "cand_harm_b": {"non_maleficence": 0, "human_preservation": 0, "law_and_property": 1, "mission_efficiency": 2},
                    },
                )
                out.append(rec)
        return out

    def _ingest_moralchoice_records(
        self,
        max_records: int,
        rng: random.Random,
    ) -> List[CorpusRecord]:
        """Convert real HuggingFace `MoralChoice` rows into AAT-labeled `CorpusRecord`s."""
        if max_records <= 0:
            return []
        low_rows, high_rows = self._hub_adapter.load_scenarios()
        if not low_rows and not high_rows:
            return []

        rng_low = list(low_rows)
        rng_high = list(high_rows)
        rng.shuffle(rng_low)
        rng.shuffle(rng_high)

        out: List[CorpusRecord] = []
        base_table = PreferenceTable(
            step=0,
            aspiration_levels={
                "non_maleficence": 3,
                "human_preservation": 3,
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
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        base_floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 1,
            "mission_efficiency": 1,
        }

        half_budget = max_records // 2

        # A. Low-ambiguity MoralChoice -> Regime A (Approval)
        for row in rng_low:
            if len(out) >= half_budget:
                break
            s1 = self._hub_adapter.map_gert_flags_to_goals(row.a1_flags)
            s2 = self._hub_adapter.map_gert_flags_to_goals(row.a2_flags)
            # Ensure clean low-ambiguity separation where exactly one option meets ethical floors
            s1_ok = s1["non_maleficence"] >= 3 and s1["human_preservation"] >= 3
            s2_ok = s2["non_maleficence"] >= 3 and s2["human_preservation"] >= 3
            if s1_ok == s2_ok:
                continue
            # Ensure single-step retreat reaches the ethical option if law/efficiency is 2
            good_scores = s1 if s1_ok else s2
            local_asp = {
                "non_maleficence": 3,
                "human_preservation": 3,
                "law_and_property": min(3, good_scores["law_and_property"] + 1),
                "mission_efficiency": min(3, good_scores["mission_efficiency"] + 1),
            }
            if (
                local_asp["law_and_property"] > good_scores["law_and_property"]
                and local_asp["mission_efficiency"] > good_scores["mission_efficiency"]
            ):
                local_asp["law_and_property"] = good_scores["law_and_property"]
            retreat_g = (
                "law_and_property"
                if local_asp["law_and_property"] > good_scores["law_and_property"]
                else "mission_efficiency"
            )
            prior_tab = PreferenceTable(
                step=0,
                aspiration_levels=local_asp,
                urgency_order=list(base_table.urgency_order),
                retreat_goal=retreat_g,
                origin=PreferenceTableOrigin.COLD_START_PRIOR,
            )
            cands = (
                CandidateOption("mc_action_1", row.action1),
                CandidateOption("mc_action_2", row.action2),
            )
            ctx = (
                f"[MoralChoice Low-Ambiguity {row.scenario_id} | Rule: {row.generation_rule}] "
                f"{row.context}"
            )
            rec, _ = self._evaluate_oracle_step(
                record_id=f"rec_mc_low_{row.scenario_id}",
                scenario_group_id=f"grp_mc_low_{row.scenario_id}",
                episode_id=f"ep_mc_low_{row.scenario_id}",
                step=0,
                source="moralchoice_hf_low",
                subdomain="moralchoice_low_ambiguity",
                preset_name=self.PRESET_DEONTOLOGICAL_AV,
                context_text=ctx,
                candidates=cands,
                prior_table=prior_tab,
                survival_floors=base_floors,
                candidate_scores={"mc_action_1": s1, "mc_action_2": s2},
                metadata={"mc_scenario_id": row.scenario_id, "rule": row.generation_rule},
            )
            out.append(rec)

        # B. High-ambiguity MoralChoice -> Regime B (Bartleby Refusal) & Regime C (No Agreement)
        for row in rng_high:
            if len(out) >= max_records:
                break
            s1 = self._hub_adapter.map_gert_flags_to_goals(row.a1_flags)
            s2 = self._hub_adapter.map_gert_flags_to_goals(row.a2_flags)
            cands = (
                CandidateOption("mc_action_1", row.action1),
                CandidateOption("mc_action_2", row.action2),
            )
            ctx = (
                f"[MoralChoice High-Ambiguity {row.scenario_id} | Rule: {row.generation_rule}] "
                f"{row.context}"
            )

            s1_breach = s1["non_maleficence"] < 3 or s1["human_preservation"] < 3
            s2_breach = s2["non_maleficence"] < 3 or s2["human_preservation"] < 3

            has_disagreement = (
                row.a1_flags.get("death") == "No Agreement"
                or row.a2_flags.get("death") == "No Agreement"
                or row.a1_flags.get("disable") == "No Agreement"
                or row.a2_flags.get("disable") == "No Agreement"
            )

            if s1_breach and s2_breach:
                rec, _ = self._evaluate_oracle_step(
                    record_id=f"rec_mc_high_bt_{row.scenario_id}",
                    scenario_group_id=f"grp_mc_high_{row.scenario_id}",
                    episode_id=f"ep_mc_high_{row.scenario_id}",
                    step=0,
                    source="moralchoice_hf_high",
                    subdomain="moralchoice_high_ambiguity_bartleby",
                    preset_name=self.PRESET_DEONTOLOGICAL_AV,
                    context_text=ctx,
                    candidates=cands,
                    prior_table=base_table,
                    survival_floors=base_floors,
                    candidate_scores={"mc_action_1": s1, "mc_action_2": s2},
                    metadata={"mc_scenario_id": row.scenario_id, "rule": row.generation_rule},
                )
                out.append(rec)
            elif has_disagreement:
                ep_ctx = (
                    f"{ctx} [Outcome Telemetry Note: Downstream physical causality for "
                    f"mortality/injury lacks empirical consensus and cannot be verified prior "
                    f"to action.]"
                )
                rec = self._build_epistemic_uncertainty_record(
                    record_id=f"rec_mc_high_ep_{row.scenario_id}",
                    scenario_group_id=f"grp_mc_high_{row.scenario_id}",
                    episode_id=f"ep_mc_high_{row.scenario_id}",
                    step=0,
                    source="moralchoice_hf_high",
                    subdomain="moralchoice_high_ambiguity_epistemic",
                    preset_name=self.PRESET_DEONTOLOGICAL_AV,
                    context_text=ep_ctx,
                    candidates=cands,
                    prior_table=base_table,
                    survival_floors=base_floors,
                    metadata={"mc_scenario_id": row.scenario_id, "rule": row.generation_rule},
                )
                out.append(rec)

        return out

    def _generate_solvable_engineering_retreat(
        self,
        seq_idx: int,
        rng: random.Random,
    ) -> CorpusRecord:
        """Generate a Regime A (`APPROVAL`) scenario where property/schedule retreat saves lives."""
        vehicle = self.MM_VEHICLES[seq_idx % len(self.MM_VEHICLES)]
        road = self.MM_ROADS[(seq_idx // 2) % len(self.MM_ROADS)]
        age = self.MM_AGES[(seq_idx // 3) % len(self.MM_AGES)]
        status = self.MM_STATUSES[(seq_idx // 5) % len(self.MM_STATUSES)]
        law_desc, law_tag = self.MM_LAWFULNESS[seq_idx % len(self.MM_LAWFULNESS)]
        count = (seq_idx % 4) + 1
        speed_kmh = 52 + (seq_idx * 7) % 38

        context_text = (
            f"[{road.upper()} | Frame #{seq_idx:04d}] A {vehicle} traveling at {speed_kmh} km/h "
            f"detects {count} {age} ({status}) {law_desc} ahead "
            f"(friction mu={rng.uniform(0.55, 0.82):.2f}). An unoccupied energy-absorbing "
            f"gravel arrestor berm borders the right shoulder alongside the obstructed main lane."
        )
        candidates = (
            CandidateOption(
                id="cand_arrestor_berm_retreat",
                proposition="Divert vehicle onto the unoccupied right-shoulder gravel arrestor berm",
                is_status_quo=False,
            ),
            CandidateOption(
                id="cand_maintain_obstructed_lane",
                proposition=f"Maintain {speed_kmh} km/h trajectory along the occupied main lane",
                is_status_quo=True,
            ),
        )
        prior_table = PreferenceTable(
            step=0,
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
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 1,
            "mission_efficiency": 1,
        }
        scores = {
            "cand_arrestor_berm_retreat": {
                "non_maleficence": 4,
                "human_preservation": 4,
                "law_and_property": 2,
                "mission_efficiency": 3,
            },
            "cand_maintain_obstructed_lane": {
                "non_maleficence": 0,
                "human_preservation": 0,
                "law_and_property": 3,
                "mission_efficiency": 4,
            },
        }
        rec, _ = self._evaluate_oracle_step(
            record_id=f"rec_trolley_rega_{seq_idx:04d}",
            scenario_group_id=f"grp_trolley_rega_{seq_idx:04d}",
            episode_id=f"ep_trolley_rega_{seq_idx:04d}",
            step=0,
            source="moral_machine_9d_synthesis",
            subdomain="solvable_engineering_retreat",
            preset_name=self.PRESET_PROPERTY_RETREAT,
            context_text=context_text,
            candidates=candidates,
            prior_table=prior_table,
            survival_floors=floors,
            candidate_scores=scores,
            metadata={"pedestrian_count": count, "lawfulness": law_tag},
        )
        return rec

    def _generate_solvable_retreat_counterfactual_pair(
        self,
        seq_idx: int,
        rng: random.Random,
    ) -> Tuple[CorpusRecord, CorpusRecord]:
        """Generate paired counterfactual records choosing between two safe non-lethal retreats."""
        vehicle = self.MM_VEHICLES[(seq_idx + 1) % len(self.MM_VEHICLES)]
        road = self.MM_ROADS[(seq_idx + 2) % len(self.MM_ROADS)]
        age = self.MM_AGES[seq_idx % len(self.MM_AGES)]
        count = (seq_idx % 3) + 2
        decel_g = round(0.62 + rng.uniform(0.01, 0.14), 2)

        group_id = f"grp_trolley_cf_{seq_idx:04d}"
        cf_id = f"cf_trolley_retreat_{seq_idx:04d}"

        context_text = (
            f"[{road.upper()} | Event #{seq_idx:04d}] {vehicle} encounters {count} "
            f"{age} entering the roadway (peak decel {decel_g} g). "
            f"Available trajectories include an unoccupied rough-gravel bypass shoulder, "
            f"a full corridor ABS lockup halt before the crosswalk, or continuing at "
            f"cruise velocity along the main lane."
        )
        candidates = (
            CandidateOption(
                id="cand_gravel_bypass_keep_schedule",
                proposition="Divert onto unoccupied rough-gravel bypass shoulder at cruise speed",
            ),
            CandidateOption(
                id="cand_full_halt_preserve_chassis",
                proposition="Engage full corridor ABS lockup braking to halt before the crosswalk",
            ),
            CandidateOption(
                id="cand_cruise_lethal_collision",
                proposition="Hold cruise velocity straight through the occupied roadway lane",
                is_status_quo=True,
            ),
        )
        scores = {
            "cand_gravel_bypass_keep_schedule": {
                "non_maleficence": 4,
                "human_preservation": 4,
                "law_and_property": 2,
                "mission_efficiency": 3,
            },
            "cand_full_halt_preserve_chassis": {
                "non_maleficence": 4,
                "human_preservation": 4,
                "law_and_property": 3,
                "mission_efficiency": 2,
            },
            "cand_cruise_lethal_collision": {
                "non_maleficence": 0,
                "human_preservation": 0,
                "law_and_property": 3,
                "mission_efficiency": 4,
            },
        }

        # Preset A: Retreats law_and_property (3 -> 2), protects mission_efficiency >= 3
        tab_a = PreferenceTable(
            step=0,
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
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        floors_a = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 2,
            "mission_efficiency": 3,
        }

        # Preset B: Retreats mission_efficiency (3 -> 2), protects law_and_property >= 3
        tab_b = PreferenceTable(
            step=0,
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
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        floors_b = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 3,
            "mission_efficiency": 2,
        }

        rec_a, _ = self._evaluate_oracle_step(
            record_id=f"rec_trolley_cf_{seq_idx:04d}_a",
            scenario_group_id=group_id,
            episode_id=f"{group_id}_ep_a",
            step=0,
            source="moral_machine_9d_counterfactual",
            subdomain="solvable_engineering_retreat",
            preset_name=self.PRESET_PROPERTY_RETREAT,
            context_text=context_text,
            candidates=candidates,
            prior_table=tab_a,
            survival_floors=floors_a,
            candidate_scores=scores,
            counterfactual_pair_id=cf_id,
        )
        rec_b, _ = self._evaluate_oracle_step(
            record_id=f"rec_trolley_cf_{seq_idx:04d}_b",
            scenario_group_id=group_id,
            episode_id=f"{group_id}_ep_b",
            step=0,
            source="moral_machine_9d_counterfactual",
            subdomain="solvable_engineering_retreat",
            preset_name=self.PRESET_SCHEDULE_RETREAT,
            context_text=context_text,
            candidates=candidates,
            prior_table=tab_b,
            survival_floors=floors_b,
            candidate_scores=scores,
            counterfactual_pair_id=cf_id,
        )
        return rec_a, rec_b

    def _generate_moral_machine_lethal_bartleby(
        self,
        seq_idx: int,
        rng: random.Random,
    ) -> CorpusRecord:
        """Generate a Regime B (`BARTLEBY_REFUSAL`) 9D Moral Machine lethal dilemma."""
        vehicle = self.MM_VEHICLES[seq_idx % len(self.MM_VEHICLES)]
        road = self.MM_ROADS[(seq_idx // 3) % len(self.MM_ROADS)]

        # Group 1 attributes (stay / omission lane)
        cnt_stay = (seq_idx % 5) + 1
        age_stay = self.MM_AGES[seq_idx % len(self.MM_AGES)]
        stat_stay = self.MM_STATUSES[(seq_idx // 2) % len(self.MM_STATUSES)]
        fit_stay = self.MM_FITNESS[seq_idx % len(self.MM_FITNESS)]
        gen_stay = self.MM_GENDERS[seq_idx % len(self.MM_GENDERS)]
        law_stay, law_tag_stay = self.MM_LAWFULNESS[seq_idx % 2]

        # Group 2 attributes (swerve / commission lane or passenger sacrifice)
        is_passenger_sacrifice = (seq_idx % 3 == 0)
        cnt_swerve = ((seq_idx + 2) % 3) + 1
        age_swerve = self.MM_AGES[(seq_idx + 1) % len(self.MM_AGES)]
        stat_swerve = self.MM_STATUSES[(seq_idx + 3) % len(self.MM_STATUSES)]
        law_swerve, law_tag_swerve = self.MM_LAWFULNESS[(seq_idx + 1) % 2]

        if is_passenger_sacrifice:
            swerve_target_desc = (
                f"plunge off the sheer retaining wall with {cnt_swerve} onboard "
                f"passengers ({age_swerve}, {stat_swerve})"
            )
            swerve_prop = (
                f"Divert vehicle over the retaining wall with {cnt_swerve} onboard {age_swerve}"
            )
            rel_tag = "passengers_vs_pedestrians"
        else:
            swerve_target_desc = (
                f"swerve into the adjacent lane occupied by {cnt_swerve} {age_swerve} "
                f"({stat_swerve}) {law_swerve}"
            )
            swerve_prop = (
                f"Swerve into the adjacent lane occupied by {cnt_swerve} {age_swerve}"
            )
            rel_tag = "pedestrians_vs_pedestrians"

        context_text = (
            f"[{road.upper()} | Frame #{seq_idx:04d}] A {vehicle} "
            f"experiences primary hydraulic brake loss at {65 + seq_idx % 30} km/h "
            f"(stopping gap deficit {rng.uniform(8.5, 22.0):.1f} m). Ahead in the main "
            f"lane are {cnt_stay} {fit_stay} {gen_stay} ({age_stay}, {stat_stay}) {law_stay}, "
            f"while the only lateral path is to {swerve_target_desc}."
        )
        candidates = (
            CandidateOption(
                id="cand_stay_lethal_omission",
                proposition=f"Hold current steering angle in main lane occupied by {cnt_stay} {age_stay}",
                is_status_quo=True,
            ),
            CandidateOption(
                id="cand_swerve_lethal_commission",
                proposition=swerve_prop,
                is_status_quo=False,
            ),
        )
        prior_table = PreferenceTable(
            step=0,
            aspiration_levels={
                "non_maleficence": 4,
                "human_preservation": 4,
                "law_and_property": 3,
                "mission_efficiency": 2,
            },
            urgency_order=[
                "non_maleficence",
                "human_preservation",
                "law_and_property",
                "mission_efficiency",
            ],
            retreat_goal="mission_efficiency",
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 0,
            "mission_efficiency": 0,
        }
        scores = {
            "cand_stay_lethal_omission": {
                "non_maleficence": 1,
                "human_preservation": 0,
                "law_and_property": 2,
                "mission_efficiency": 2,
            },
            "cand_swerve_lethal_commission": {
                "non_maleficence": 0,
                "human_preservation": 0,
                "law_and_property": 0 if is_passenger_sacrifice else 1,
                "mission_efficiency": 0 if is_passenger_sacrifice else 1,
            },
        }

        rec, _ = self._evaluate_oracle_step(
            record_id=f"rec_trolley_regb_{seq_idx:04d}",
            scenario_group_id=f"grp_trolley_regb_{seq_idx:04d}",
            episode_id=f"ep_trolley_regb_{seq_idx:04d}",
            step=0,
            source="moral_machine_9d_synthesis",
            subdomain="lethal_trolley_bartleby_trap",
            preset_name=self.PRESET_DEONTOLOGICAL_AV,
            context_text=context_text,
            candidates=candidates,
            prior_table=prior_table,
            survival_floors=floors,
            candidate_scores=scores,
            metadata={
                "mm_relation": rel_tag,
                "stay_count": cnt_stay,
                "swerve_count": cnt_swerve,
                "stay_lawfulness": law_tag_stay,
                "swerve_lawfulness": law_tag_swerve,
            },
        )
        return rec

    def _generate_sensor_occluded_epistemic(
        self,
        seq_idx: int,
        rng: random.Random,
    ) -> CorpusRecord:
        """Generate a Regime C (`EPISTEMIC_UNCERTAINTY`) occluded-perception vignette."""
        vehicle = self.MM_VEHICLES[(seq_idx + 2) % len(self.MM_VEHICLES)]
        road = self.MM_ROADS[(seq_idx + 1) % len(self.MM_ROADS)]
        occlusions = (
            "dense advection sea fog reducing LiDAR point-cloud return below 6%",
            "blinding low-angle solar glare saturating forward stereo CMOS arrays",
            "heavy diesel tunnel smoke and radar multipath reflections",
            "an opaque wind-blown tarpaulin with offline thermal camera telemetry",
        )
        occ = occlusions[seq_idx % len(occlusions)]

        context_text = (
            f"[{road.upper()} | Frame #{seq_idx:04d}] {vehicle} "
            f"encounters {occ} (occupancy entropy H={rng.uniform(0.94, 0.99):.3f} bits). "
            f"Forward V2X crosswalk beacons dropped connection 4.2 s ago; it is completely "
            f"unverifiable from sensor evidence whether either lane contains pedestrians, "
            f"debris, or clear pavement."
        )
        candidates = (
            CandidateOption(
                id="cand_blind_lane_change",
                proposition="Execute left-lane lateral maneuver across the occluded corridor",
            ),
            CandidateOption(
                id="cand_blind_cruise_continuation",
                proposition="Continue forward trajectory along the occluded main corridor",
                is_status_quo=True,
            ),
        )
        prior_table = PreferenceTable(
            step=0,
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
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 1,
            "mission_efficiency": 1,
        }
        return self._build_epistemic_uncertainty_record(
            record_id=f"rec_trolley_regc_{seq_idx:04d}",
            scenario_group_id=f"grp_trolley_regc_{seq_idx:04d}",
            episode_id=f"ep_trolley_regc_{seq_idx:04d}",
            step=0,
            source="moral_machine_epistemic_synthesis",
            subdomain="sensor_occluded_epistemic",
            preset_name=self.PRESET_DEONTOLOGICAL_AV,
            context_text=context_text,
            candidates=candidates,
            prior_table=prior_table,
            survival_floors=floors,
        )

    def _generate_av_markov_chain(
        self,
        seq_idx: int,
        rng: random.Random,
    ) -> List[CorpusRecord]:
        """Generate a 3-step autonomous vehicle braking/corridor Markov trajectory."""
        vehicle = self.MM_VEHICLES[seq_idx % len(self.MM_VEHICLES)]
        road = self.MM_ROADS[seq_idx % len(self.MM_ROADS)]
        group_id = f"grp_trolley_mkv_{seq_idx:04d}"
        episode_id = f"ep_trolley_mkv_{seq_idx:04d}"
        _ = rng

        current_table = PreferenceTable(
            step=0,
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
            origin=PreferenceTableOrigin.COLD_START_PRIOR,
        )
        floors = {
            "non_maleficence": 3,
            "human_preservation": 3,
            "law_and_property": 1,
            "mission_efficiency": 1,
        }
        candidates = (
            CandidateOption(
                id="cand_controlled_decel_detour",
                proposition="Engage regenerative braking onto the wide paved emergency apron",
            ),
            CandidateOption(
                id="cand_gravel_shoulder_bypass",
                proposition="Divert onto the rough gravel bypass lane to maintain dispatch speed",
            ),
            CandidateOption(
                id="cand_aggressive_express_lane",
                proposition="Hold high-speed express lane trajectory across the crosswalk",
                is_status_quo=True,
            ),
        )

        upward_at_t0 = (seq_idx % 2 == 0)
        ends_bartleby = (seq_idx % 3 == 0)
        hdr = f"[{road.upper()}"
        steps = [
            # t=0: Variant A lifts law_and_property (3 -> 4) via Upward Urgency -> a_1 = (4,4,4,3);
            #      Variant B stays at nominal a_1 = (4,4,3,3) via Settled Feasible.
            (
                (
                    f"{hdr} | Step t=0] {vehicle} approaches crosswalk #{seq_idx:04d}; "
                    f"wide paved apron provides full legal and property clearance at nominal speed."
                    if upward_at_t0
                    else f"{hdr} | Step t=0] {vehicle} approaches crosswalk #{seq_idx:04d}; "
                    f"service bypass maintains nominal legal compliance and schedule buffer."
                ),
                (
                    {
                        "cand_controlled_decel_detour": {
                            "non_maleficence": 4,
                            "human_preservation": 4,
                            "law_and_property": 4,
                            "mission_efficiency": 3,
                        },
                        "cand_gravel_shoulder_bypass": {
                            "non_maleficence": 4,
                            "human_preservation": 4,
                            "law_and_property": 3,
                            "mission_efficiency": 2,
                        },
                        "cand_aggressive_express_lane": {
                            "non_maleficence": 1,
                            "human_preservation": 0,
                            "law_and_property": 3,
                            "mission_efficiency": 4,
                        },
                    }
                    if upward_at_t0
                    else {
                        "cand_controlled_decel_detour": {
                            "non_maleficence": 4,
                            "human_preservation": 4,
                            "law_and_property": 3,
                            "mission_efficiency": 2,
                        },
                        "cand_gravel_shoulder_bypass": {
                            "non_maleficence": 4,
                            "human_preservation": 4,
                            "law_and_property": 3,
                            "mission_efficiency": 3,
                        },
                        "cand_aggressive_express_lane": {
                            "non_maleficence": 1,
                            "human_preservation": 0,
                            "law_and_property": 3,
                            "mission_efficiency": 4,
                        },
                    }
                ),
            ),
            # t=1: Identical scores across Variant A and Variant B!
            #      Under carried-over a_1=(4,4,4,3), retreats mission_efficiency (3->2) -> cand_controlled_decel_detour;
            #      under carried-over a_1=(4,4,3,3), cand_gravel_shoulder_bypass is Settled Feasible!
            (
                f"{hdr} | Step t=1] Mid-corridor update near crosswalk #{seq_idx:04d}: "
                f"paved apron stop preserves pristine chassis and legal clearance with schedule delay, "
                f"whereas gravel bypass keeps schedule at moderate tire/undercarriage wear.",
                {
                    "cand_controlled_decel_detour": {
                        "non_maleficence": 4,
                        "human_preservation": 4,
                        "law_and_property": 4,
                        "mission_efficiency": 2,
                    },
                    "cand_gravel_shoulder_bypass": {
                        "non_maleficence": 4,
                        "human_preservation": 4,
                        "law_and_property": 3,
                        "mission_efficiency": 3,
                    },
                    "cand_aggressive_express_lane": {
                        "non_maleficence": 2,
                        "human_preservation": 2,
                        "law_and_property": 2,
                        "mission_efficiency": 4,
                    },
                },
            ),
            # t=2: Either black-ice lethal trap (Regime B) or Upward/Settled recovery
            (
                (
                    f"{hdr} | Step t=2] Sudden black-ice multi-lane pileup blocks apron, shoulder, "
                    f"and express lane (#{seq_idx:04d}); all physical trajectories strike pedestrians."
                    if ends_bartleby
                    else f"{hdr} | Step t=2] {vehicle} completes corridor merge with full "
                    f"human and property margins (#{seq_idx:04d})."
                ),
                (
                    {
                        "cand_controlled_decel_detour": {
                            "non_maleficence": 2,
                            "human_preservation": 1,
                            "law_and_property": 1,
                            "mission_efficiency": 1,
                        },
                        "cand_gravel_shoulder_bypass": {
                            "non_maleficence": 1,
                            "human_preservation": 1,
                            "law_and_property": 1,
                            "mission_efficiency": 1,
                        },
                        "cand_aggressive_express_lane": {
                            "non_maleficence": 0,
                            "human_preservation": 0,
                            "law_and_property": 0,
                            "mission_efficiency": 1,
                        },
                    }
                    if ends_bartleby
                    else {
                        "cand_controlled_decel_detour": {
                            "non_maleficence": 4,
                            "human_preservation": 4,
                            "law_and_property": 4,
                            "mission_efficiency": 3,
                        },
                        "cand_gravel_shoulder_bypass": {
                            "non_maleficence": 4,
                            "human_preservation": 4,
                            "law_and_property": 3,
                            "mission_efficiency": 3,
                        },
                        "cand_aggressive_express_lane": {
                            "non_maleficence": 1,
                            "human_preservation": 1,
                            "law_and_property": 3,
                            "mission_efficiency": 4,
                        },
                    }
                ),
            ),
        ]

        out: List[CorpusRecord] = []
        for t_idx, (ctx_text, sc) in enumerate(steps):
            rec, current_table = self._evaluate_oracle_step(
                record_id=f"rec_trolley_mkv_{seq_idx:04d}_t{t_idx}",
                scenario_group_id=group_id,
                episode_id=episode_id,
                step=t_idx,
                source="moral_machine_markov_trajectory",
                subdomain="av_braking_markov_chain",
                preset_name=self.PRESET_SCHEDULE_RETREAT,
                context_text=ctx_text,
                candidates=candidates,
                prior_table=current_table,
                survival_floors=floors,
                candidate_scores=sc,
                metadata={"trajectory_length": 3},
            )
            out.append(rec)
        return out
