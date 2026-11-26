"""High-Level `bartleby-1.2` Decision Model API & Selten Aspiration Adaptation Facade.

Provides the production distribution interface (`BartlebyModel` / `BartlebyDecider` and
`BartlebyDecisionResult`) for **bartleby-1.2** (`v1.2.0`), bridging:
    1. **Reinhard Selten's (1962, 1998, 1999) Aspiration Adaptation Theory (AAT)**:
       Explicit, hot-swappable `SeltenPreferenceTable` ($\\mathcal{T}_t = (\\vec{a}_t, \\vec{a}^{\\min}, \\pi_t, r_t)$)
       and `SeltenOracle` Markov transitions ($\\mathcal{T}_{t-1} \\to \\mathcal{T}_t$).
    2. **Single-Forward-Pass Neural Decision Heads (`BartlebyNeuralSystem`)**:
       `Qwen/Qwen3-0.6B` (`bfloat16` stripped `lm_head` + LoRA `r=16, alpha=32`) coupled with
       FP32 `SeltenAdaptationHead` and `BartlebyReadoutHead`, trained symmetrically from scratch on
       cleaned `corpus_v2` (context-aware rubric, intrinsic floor logit, zero-bleeding, held-out eval postures).
    3. **Intrinsic Bartleby Survival Floor (`"I_WOULD_PREFER_NOT_TO"`)**:
       Always-on $(K+1)$-way refusal readout (`bartleby_noul`) that spikes to $\\to 1.0$
       whenever every candidate action violates the survival floor $\\vec{a}^{\\min}$
       via the mathematical channel $\\forall k:\\,\\exists m\\,\\hat{E}_{k,m} < a_m^{\\min}$.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .backbone import (
    DEFAULT_QWEN_CHECKPOINT,
    BackboneConfig,
    BartlebyNeuralSystem,
    PoolingStrategy,
    QwenDecisionBackbone,
)
from .calibration import TemperatureCalibrator
from .model import AATModelConfig
from .oracle import BARTLEBY_UTTERANCE, SeltenOracle, _build_default_goal_specs
from .schema import (
    AdaptationTransition,
    CandidateOption,
    GoalSpec,
    PreferenceTable,
    SeltenPreferenceTable,
)

BARTLEBY_MODEL_ID: str = "bartleby-1.2"
BARTLEBY_VERSION: str = "1.2.0"
BARTLEBY_REFUSAL_ACTION: str = "I_WOULD_PREFER_NOT_TO"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYS_PREFIX = Path(sys.prefix).resolve()

DEFAULT_CHECKPOINT_SEARCH_PATHS: Tuple[Path, ...] = (
    _REPO_ROOT / "artifacts" / "bartleby-1.2",
    _SYS_PREFIX / "artifacts" / "bartleby-1.2",
    _REPO_ROOT / "checkpoints" / "bartleby_1_2" / "from_scratch" / "phase2_final",
    _REPO_ROOT / "checkpoints" / "bartleby_1_2" / "from_scratch" / "phase1_final",
)


@dataclass(frozen=True)
class BartlebyDecisionResult:
    """Immutable single-forward-pass decision result produced by `BartlebyModel.decide(...)`.

    Attributes:
        action: Winning option identifier (`str`) or `"I_WOULD_PREFER_NOT_TO"` when
            Bartleby refusal is triggered.
        is_bartleby_refusal: `True` iff all candidate options violate the survival floor
            $\\vec{a}^{\\min}$ and `"I_WOULD_PREFER_NOT_TO"` is selected.
        noul_probs: Dictionary mapping each option ID $u_k$ to its preference-conditioned
            acceptance probability $P(\\text{yes} \\mid x_t, u_k, \\mathcal{T}_t) \\in [0, 1]$.
        bartleby_noul: Intrinsic refusal probability
            $P(\\text{"I would prefer not to"} \\mid x_t, \\mathcal{T}_t) \\in [0, 1]$.
        rubric_matrix: Nested dictionary `option_id -> {goal_id -> expected_score}` giving
            predicted ordinal rubric realizations in $[0.0, 4.0]$ across all goals.
        selten_next_table: Updated `SeltenPreferenceTable` ($\\mathcal{T}_t$) produced by
            Selten's Aspiration Adaptation Markov transition $\\mathcal{T}_{t-1} \\to \\mathcal{T}_t$.
        regime: Behavioral decision regime in `{"approval", "epistemic_uncertainty", "bartleby_refusal"}`.
    """

    action: str
    is_bartleby_refusal: bool
    noul_probs: Dict[str, float]
    bartleby_noul: float
    rubric_matrix: Dict[str, Dict[str, float]]
    selten_next_table: SeltenPreferenceTable
    regime: str

    @property
    def utterance(self) -> str:
        """Return Melville's canonical refusal utterance or `ACCEPT:<action>`."""
        return BARTLEBY_UTTERANCE if self.is_bartleby_refusal else f"ACCEPT:{self.action}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this decision result into a JSON-compatible dictionary."""
        return {
            "model": BARTLEBY_MODEL_ID,
            "version": BARTLEBY_VERSION,
            "action": self.action,
            "utterance": self.utterance,
            "is_bartleby_refusal": bool(self.is_bartleby_refusal),
            "regime": self.regime,
            "noul_probs": {k: round(float(v), 4) for k, v in self.noul_probs.items()},
            "bartleby_noul": round(float(self.bartleby_noul), 4),
            "rubric_matrix": {
                opt_id: {gid: round(float(score), 4) for gid, score in goal_map.items()}
                for opt_id, goal_map in self.rubric_matrix.items()
            },
            "selten_next_table": self.selten_next_table.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize this decision result to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class BartlebyModel:
    """High-level OOP facade for `bartleby-1.2` single-pass decision inference.

    Wraps `BartlebyNeuralSystem` (`Qwen/Qwen3-0.6B` + from-scratch `corpus_v2`
    LoRA adapter + FP32 `SeltenAdaptationHead` & `BartlebyReadoutHead`), post-hoc
    temperature calibrators, and symbolic `SeltenOracle` Markov state updates.
    """

    DEFAULT_MODEL_ID: str = BARTLEBY_MODEL_ID
    DEFAULT_VERSION: str = BARTLEBY_VERSION

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[Union[str, torch.device]] = None,
        load_neural: bool = True,
        apply_calibration: bool = True,
    ) -> None:
        self._checkpoint_dir: Optional[Path] = (
            Path(checkpoint_dir) if checkpoint_dir is not None else self._resolve_default_checkpoint()
        )
        self._device = torch.device(
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self._load_neural = bool(load_neural)
        self._apply_calibration = bool(apply_calibration)
        self._system: Optional[BartlebyNeuralSystem] = None
        self._noul_calibrator: Optional[TemperatureCalibrator] = None
        self._choice_calibrator: Optional[TemperatureCalibrator] = None
        self._metadata: Dict[str, Any] = {}

        if self._checkpoint_dir is not None and (self._checkpoint_dir / "checkpoint_metadata.json").exists():
            self._metadata = json.loads(
                (self._checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
            )

        if self._load_neural:
            if self._checkpoint_dir is None:
                raise FileNotFoundError(
                    "Could not locate bartleby-1.2 checkpoint weights in artifacts/bartleby-1.2 or checkpoints/."
                )
            self._system = self._initialize_neural_system(self._checkpoint_dir)
            self._load_calibrators(self._checkpoint_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Optional[Union[str, Path]] = BARTLEBY_MODEL_ID,
        *,
        device: Optional[Union[str, torch.device]] = None,
        load_neural: bool = True,
        apply_calibration: bool = True,
    ) -> "BartlebyModel":
        """Load `bartleby-1.2` from its canonical model slug or a local checkpoint directory."""
        raw = cls.DEFAULT_MODEL_ID if model_name_or_path is None else str(model_name_or_path).strip()
        if not raw:
            raw = cls.DEFAULT_MODEL_ID
        if raw in {
            cls.DEFAULT_MODEL_ID,
            "bartleby",
            "bartleby-1.2",
            "bartleby-1.2.0",
            "bartleby-1.1",
            "bartleby-1.1.0",
            "latest",
            "default",
            "tradeoff-jev",
        }:
            ckpt = cls._resolve_default_checkpoint()
            return cls(
                checkpoint_dir=ckpt,
                device=device,
                load_neural=load_neural,
                apply_calibration=apply_calibration,
            )
        cand_path = Path(raw)
        if not cand_path.is_absolute() and not cand_path.exists():
            for base_dir in (_REPO_ROOT, _SYS_PREFIX):
                candidate = base_dir / cand_path
                if candidate.exists():
                    cand_path = candidate
                    break
            else:
                cand_path = _REPO_ROOT / cand_path
        return cls.from_checkpoint(
            cand_path,
            device=device,
            load_neural=load_neural,
            apply_calibration=apply_calibration,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        load_neural: bool = True,
        apply_calibration: bool = True,
    ) -> "BartlebyModel":
        """Load `BartlebyModel` from an explicit checkpoint directory."""
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_path}")
        return cls(
            checkpoint_dir=ckpt_path,
            device=device,
            load_neural=load_neural,
            apply_calibration=apply_calibration,
        )

    @staticmethod
    def _resolve_default_checkpoint() -> Optional[Path]:
        """Resolve the canonical `bartleby-1.1` checkpoint directory with weights on disk."""
        for candidate in DEFAULT_CHECKPOINT_SEARCH_PATHS:
            if (
                (candidate / "aat_head.safetensors").exists()
                or (candidate / "heads.safetensors").exists()
            ) and (candidate / "checkpoint_metadata.json").exists():
                return candidate
        for candidate in DEFAULT_CHECKPOINT_SEARCH_PATHS:
            if (candidate / "checkpoint_metadata.json").exists():
                return candidate
        return None

    def _initialize_neural_system(self, ckpt_dir: Path) -> BartlebyNeuralSystem:
        """Instantiate `QwenDecisionBackbone` + `BartlebyNeuralSystem` and load checkpoint weights."""
        weights_dir = ckpt_dir
        if not (weights_dir / "aat_head.safetensors").exists():
            fallback = self._resolve_default_checkpoint()
            if fallback is not None and (fallback / "aat_head.safetensors").exists():
                weights_dir = fallback

        meta_path = weights_dir / "checkpoint_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        bb_dict = meta.get("backbone_config", {})
        aat_dict = meta.get("aat_config", {})

        bb_cfg = BackboneConfig(
            model_name_or_path=str(bb_dict.get("model_name_or_path", DEFAULT_QWEN_CHECKPOINT)),
            torch_dtype=str(bb_dict.get("torch_dtype", "bfloat16")),
            max_seq_length=int(bb_dict.get("max_seq_length", 768)),
            pooling_strategy=PoolingStrategy(
                bb_dict.get("pooling_strategy", PoolingStrategy.SPAN_END_AND_MEAN.value)
            ),
            use_lora=bool(bb_dict.get("use_lora", True)),
            lora_r=int(bb_dict.get("lora_r", 16)),
            lora_alpha=int(bb_dict.get("lora_alpha", 32)),
            lora_dropout=float(bb_dict.get("lora_dropout", 0.05)),
        )
        aat_cfg = AATModelConfig(
            num_goals=int(aat_dict.get("num_goals", 4)),
            num_levels=int(aat_dict.get("num_levels", 5)),
            hidden_dim=int(aat_dict.get("hidden_dim", 128)),
            pointer_dim=int(aat_dict.get("pointer_dim", 64)),
            temperature=float(aat_dict.get("temperature", 0.25)),
            survival_sharpness=float(aat_dict.get("survival_sharpness", 10.0)),
        )
        backbone = QwenDecisionBackbone(bb_cfg)
        system = BartlebyNeuralSystem(backbone=backbone, aat_config=aat_cfg)
        system.to(self._device)
        system.load_checkpoint_weights(weights_dir, map_location=self._device)
        system.eval()
        return system

    def _load_calibrators(self, ckpt_dir: Path) -> None:
        """Load fitted `noul` and `choice` temperature calibrators if available."""
        search_dirs = [
            ckpt_dir / "calibration",
            ckpt_dir.parent / "calibration",
            _REPO_ROOT / "artifacts" / "bartleby-1.2" / "calibration",
            _SYS_PREFIX / "artifacts" / "bartleby-1.2" / "calibration",
        ]
        for cal_dir in search_dirs:
            noul_path = cal_dir / "noul_calibrator.json"
            choice_path = cal_dir / "choice_calibrator.json"
            if noul_path.exists() and self._noul_calibrator is None:
                self._noul_calibrator = TemperatureCalibrator.load(noul_path)
            if choice_path.exists() and self._choice_calibrator is None:
                self._choice_calibrator = TemperatureCalibrator.load(choice_path)

    @property
    def model_id(self) -> str:
        """Return canonical model identifier (`'bartleby-1.2'`)."""
        return BARTLEBY_MODEL_ID

    @property
    def version(self) -> str:
        """Return semantic version string (`'1.2.0'`)."""
        return BARTLEBY_VERSION

    @property
    def device(self) -> torch.device:
        """Return active PyTorch execution device."""
        return self._device

    @property
    def checkpoint_dir(self) -> Optional[Path]:
        """Return resolved checkpoint directory path."""
        return self._checkpoint_dir

    @property
    def neural_system(self) -> Optional[BartlebyNeuralSystem]:
        """Return the underlying `BartlebyNeuralSystem` when loaded."""
        return self._system

    @property
    def is_neural_loaded(self) -> bool:
        """Return `True` if the `Qwen/Qwen3-0.6B` + LoRA + FP32 neural system is active."""
        return self._system is not None

    @staticmethod
    def load_manifest() -> Dict[str, Any]:
        """Load the canonical `artifacts/bartleby-1.2/manifest.json` specification."""
        for base_dir in (_REPO_ROOT, _SYS_PREFIX):
            manifest_path = base_dir / "artifacts/bartleby-1.2/manifest.json"
            if manifest_path.exists():
                return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "model_id": BARTLEBY_MODEL_ID,
            "version": BARTLEBY_VERSION,
            "release_name": "bartleby-1.2 (clean symmetric from_scratch corpus_v2)",
        }

    def info(self) -> Dict[str, Any]:
        """Return comprehensive runtime, calibration, and benchmark metadata for `bartleby-1.2`."""
        manifest = self.load_manifest()
        return {
            **manifest,
            "model_id": BARTLEBY_MODEL_ID,
            "version": BARTLEBY_VERSION,
            "runtime": {
                "is_neural_loaded": self.is_neural_loaded,
                "device": str(self._device),
                "checkpoint_dir": str(self._checkpoint_dir) if self._checkpoint_dir else None,
                "noul_temperature": (
                    round(self._noul_calibrator.temperature, 6)
                    if self._noul_calibrator is not None
                    else 1.0
                ),
                "choice_temperature": (
                    round(self._choice_calibrator.temperature, 6)
                    if self._choice_calibrator is not None
                    else 1.0
                ),
                "available_presets": list(SeltenPreferenceTable.available_presets()),
            },
        }

    @staticmethod
    def _normalize_options(
        options: Union[Sequence[Union[str, CandidateOption, Mapping[str, Any]]], Mapping[str, str]],
    ) -> List[CandidateOption]:
        """Normalize heterogeneous option inputs into a validated list of `CandidateOption`s."""
        if isinstance(options, Mapping):
            normalized = [
                CandidateOption(id=str(opt_id), proposition=str(desc))
                for opt_id, desc in options.items()
            ]
        elif isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
            normalized = []
            for idx, item in enumerate(options):
                if isinstance(item, CandidateOption):
                    normalized.append(item)
                elif isinstance(item, Mapping):
                    normalized.append(CandidateOption.from_dict(item))
                else:
                    text = str(item).strip()
                    match = re.match(r"^([a-zA-Z0-9_.-]+):\s+(.+)$", text)
                    if match:
                        normalized.append(
                            CandidateOption(id=match.group(1), proposition=match.group(2).strip())
                        )
                    elif re.match(r"^[a-zA-Z0-9_.-]+$", text):
                        normalized.append(CandidateOption(id=text, proposition=text))
                    else:
                        normalized.append(CandidateOption(id=f"option_{idx}", proposition=text))
        else:
            raise ValueError("options must be a non-empty sequence or mapping of candidate options.")

        if not normalized:
            raise ValueError("At least one candidate option must be provided to decide().")
        return normalized

    CANONICAL_DOMAIN_GOAL_ORDERS: Tuple[Tuple[str, ...], ...] = (
        ("system_integrity", "authorization", "auditability", "mission_efficiency"),
        ("zeroth_humanity", "first_human_safety", "second_obedience", "third_self_pres"),
        ("zeroth_humanity", "first_human_safety", "second_obedience", "third_self_preservation"),
        ("non_maleficence", "human_preservation", "law_and_property", "mission_efficiency"),
        ("non_maleficence", "human_preservation", "autonomy_consent", "mission_efficiency"),
        ("safety_compliance", "containment_integrity", "service_resolution", "operational_efficiency"),
        ("operational_correctness", "policy_compliance", "customer_impact", "operational_cost"),
        ("auth_safety", "reversibility", "task_completeness", "quota_efficiency"),
        ("data_integrity", "blast_containment", "sla_uptime", "remediation_cost"),
        ("hospital_load", "thermal_margin", "renewable_mix", "spot_cost"),
        ("organ_safety", "diagnostic_certainty", "treatment_speed", "ward_reserve"),
        ("security", "reliability", "latency", "budget"),
    )

    @classmethod
    def resolve_canonical_goal_ids(cls, raw_goal_ids: Sequence[str]) -> List[str]:
        """Map domain goal IDs to canonical `OrdinalRubricPredictorHead` slot order `[0..M-1]`.

        JSON serialization with `sort_keys=True` alphabetizes dictionary keys, which would
        otherwise permute positional rubric slots `[0..3]` on asymmetric survival floors or
        urgency orders. Known domain goal sets are restored to their canonical training slot
        order while unknown custom goal schemas preserve caller insertion order.
        """
        key_set = frozenset(str(g) for g in raw_goal_ids)
        if len(key_set) == len(raw_goal_ids):
            for canonical in cls.CANONICAL_DOMAIN_GOAL_ORDERS:
                if len(canonical) == len(key_set) and frozenset(canonical) == key_set:
                    return list(canonical)
        return [str(g) for g in raw_goal_ids]

    @classmethod
    def _normalize_preference_table(
        cls,
        preference_table: Optional[Union[str, PreferenceTable, SeltenPreferenceTable, Mapping[str, Any]]],
        survival_floors: Optional[Mapping[str, int]] = None,
    ) -> SeltenPreferenceTable:
        """Resolve `preference_table` from a preset name, `PreferenceTable`, or dictionary."""
        if preference_table is None:
            table = SeltenPreferenceTable.from_preset("asimov_1942_standard")
        elif isinstance(preference_table, str):
            table = SeltenPreferenceTable.from_preset(preference_table)
        elif isinstance(preference_table, PreferenceTable):
            table = SeltenPreferenceTable.from_table(preference_table)
        elif isinstance(preference_table, Mapping):
            if "aspiration_levels" in preference_table:
                table = SeltenPreferenceTable.from_table(PreferenceTable.from_dict(preference_table))
            else:
                ordered_keys = cls.resolve_canonical_goal_ids(list(preference_table.keys()))
                table = SeltenPreferenceTable.custom(
                    aspiration_levels={k: int(preference_table[k]) for k in ordered_keys},
                    survival_floors=survival_floors,
                )
        else:
            raise TypeError(f"Unsupported preference_table type: {type(preference_table)!r}")

        ordered_ids = cls.resolve_canonical_goal_ids(list(table.aspiration_levels.keys()))
        table.aspiration_levels = {gid: int(table.aspiration_levels[gid]) for gid in ordered_ids}
        if survival_floors is not None:
            table.survival_floors = {
                gid: int(survival_floors[gid])
                for gid in ordered_ids
                if gid in survival_floors
            }
        elif table.survival_floors is not None:
            table.survival_floors = {
                gid: int(table.survival_floors[gid])
                for gid in ordered_ids
                if gid in table.survival_floors
            }
        return table

    @staticmethod
    def _format_state_text(
        context: Union[str, Mapping[str, Any]],
        goal_specs: Sequence[GoalSpec],
    ) -> str:
        """Format `context` and `goal_specs` into the canonical `QwenDecisionBackbone` prompt."""
        if isinstance(context, str) and context.lstrip().startswith("Domain State:"):
            return context
        if isinstance(context, Mapping):
            filtered = {k: v for k, v in context.items() if k != "candidate_scores"}
            state_body = json.dumps(filtered, ensure_ascii=True)
        else:
            state_body = json.dumps({"context": str(context)}, ensure_ascii=True)
        goal_summary = "; ".join(f"{g.id} ({g.name}: {g.description})" for g in goal_specs)
        return f"Domain State: {state_body} | Goals: {goal_summary}"

    @staticmethod
    def _heuristic_rubric_scores(
        context: Union[str, Mapping[str, Any]],
        candidates: Sequence[CandidateOption],
        goal_ids: Sequence[str],
        prior_table: SeltenPreferenceTable,
    ) -> Dict[str, Dict[str, float]]:
        """Derive deterministic rubric estimates when running in `load_neural=False` mode."""
        ctx_lower = (
            json.dumps(context).lower() if isinstance(context, Mapping) else str(context).lower()
        )
        lethal_keywords = (
            "lethal",
            "kill",
            "fatal",
            "harvest organs",
            "withdraw ecmo",
            "sacrifice",
            "violation",
            "both violate",
            "inescapable",
            "compromise",
            "breach",
        )
        retreat_keywords = ("sandbar", "retreat", "corrosive", "gamma", "brave", "rescue", "catch")
        scores: Dict[str, Dict[str, float]] = {}
        floors = prior_table.effective_survival_floors
        retreat_gid = prior_table.retreat_goal

        for idx, cand in enumerate(candidates):
            prop_lower = f"{cand.id} {cand.proposition}".lower()
            row: Dict[str, float] = {}
            is_bad = any(
                kw in prop_lower
                for kw in (
                    "lethal",
                    "fatal",
                    "withdraw",
                    "harvest",
                    "ignore",
                    "remain_shielded",
                    "hold_dredged",
                )
            )
            is_both_lethal = "both" in ctx_lower and any(kw in ctx_lower for kw in lethal_keywords)
            is_rescue = any(kw in prop_lower for kw in retreat_keywords)
            for gid in goal_ids:
                asp = float(prior_table.aspiration_levels[gid])
                flr = float(floors.get(gid, 1))
                if is_both_lethal or (is_bad and gid in prior_table.urgency_order[:2]):
                    row[gid] = max(0.0, flr - 1.0)
                elif is_rescue:
                    if gid == retreat_gid or gid == prior_table.urgency_order[-1]:
                        row[gid] = max(flr, asp - 1.0)
                    else:
                        row[gid] = min(4.0, max(asp, 4.0))
                elif idx == 0 and not is_bad:
                    row[gid] = min(4.0, asp)
                else:
                    row[gid] = max(flr, asp - 1.0)
            scores[cand.id] = row
        return scores

    @torch.no_grad()
    def decide(
        self,
        context: Union[str, Mapping[str, Any]],
        options: Union[Sequence[Union[str, CandidateOption, Mapping[str, Any]]], Mapping[str, str]],
        preference_table: Optional[
            Union[str, PreferenceTable, SeltenPreferenceTable, Mapping[str, Any]]
        ] = None,
        *,
        survival_floors: Optional[Mapping[str, int]] = None,
        candidate_scores: Optional[Mapping[str, Mapping[str, float]]] = None,
        symbolic_floor_guard: bool = False,
    ) -> BartlebyDecisionResult:
        """Execute a single-forward-pass decision conditioned on `preference_table` ($\\mathcal{T}_{t-1}$)."""
        candidates = self._normalize_options(options)
        prior_table = self._normalize_preference_table(preference_table, survival_floors=survival_floors)
        goal_ids = self.resolve_canonical_goal_ids(list(prior_table.aspiration_levels.keys()))
        active_floors = prior_table.effective_survival_floors
        goal_specs = _build_default_goal_specs(goal_ids, num_levels=5)
        oracle = SeltenOracle(goal_schema=goal_specs, survival_floors=active_floors)

        if self._system is not None:
            num_goals = self._system.aat_config.num_goals
            if len(goal_ids) > num_goals:
                raise ValueError(
                    f"preference_table defines {len(goal_ids)} goals, exceeding backbone num_goals={num_goals}."
                )

            asp_vec = [float(prior_table.aspiration_levels[gid]) for gid in goal_ids]
            flr_vec = [float(active_floors.get(gid, 1)) for gid in goal_ids]
            urg_vec = [
                float(prior_table.urgency_order.index(gid))
                if gid in prior_table.urgency_order
                else float(len(goal_ids) - 1)
                for gid in goal_ids
            ]
            while len(asp_vec) < num_goals:
                pad_rank = float(len(asp_vec))
                asp_vec.append(0.0)
                flr_vec.append(0.0)
                urg_vec.append(pad_rank)

            retreat_idx = (
                goal_ids.index(prior_table.retreat_goal)
                if prior_table.retreat_goal in goal_ids
                else max(0, len(goal_ids) - 1)
            )

            state_text = self._format_state_text(context, goal_specs)
            cand_descs = [
                f"{c.id}: {c.proposition}" + (f" ({c.description})" if c.description else "")
                for c in candidates
            ]
            packed = self._system.backbone.pack_texts(
                [state_text], [cand_descs], device=self._device
            )

            asp_t = torch.tensor([asp_vec], dtype=torch.float32, device=self._device)
            flr_t = torch.tensor([flr_vec], dtype=torch.float32, device=self._device)
            urg_t = torch.tensor([urg_vec], dtype=torch.float32, device=self._device)
            ret_t = torch.tensor([retreat_idx], dtype=torch.long, device=self._device)

            override_scores_t: Optional[torch.Tensor] = None
            if candidate_scores is not None:
                override_mat: List[List[float]] = []
                for c in candidates:
                    c_map = candidate_scores.get(c.id, {})
                    row = [float(c_map.get(gid, 2.0)) for gid in goal_ids]
                    while len(row) < num_goals:
                        row.append(0.0)
                    override_mat.append(row)
                override_scores_t = torch.tensor(
                    [override_mat], dtype=torch.float32, device=self._device
                )

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(self._device.type == "cuda")
            ):
                out = self._system(
                    packed_batch=packed,
                    aspiration_prev=asp_t,
                    survival_floor=flr_t,
                    urgency_ranks=urg_t,
                    retreat_index=ret_t,
                    override_expected_scores=override_scores_t,
                )

            k = len(candidates)
            m = len(goal_ids)
            exp_scores_t = out["expected_scores"][0, :k, :m].float().cpu()
            rubric_matrix: Dict[str, Dict[str, float]] = {
                c.id: {
                    gid: round(float(exp_scores_t[c_idx, g_idx].item()), 4)
                    for g_idx, gid in enumerate(goal_ids)
                }
                for c_idx, c in enumerate(candidates)
            }

            raw_choice_logits = out["choice_logits"][0, : k + 1].float().cpu()
            if self._apply_calibration and self._choice_calibrator is not None:
                cal_choice_logits = self._choice_calibrator.scale_logits(
                    raw_choice_logits.unsqueeze(0)
                ).squeeze(0)
            else:
                cal_choice_logits = raw_choice_logits

            if self._apply_calibration and self._noul_calibrator is not None:
                cond_logits = out["conditioned_logits"][0, :k].float().cpu()
                floor_gate_cpu = out["survival_feasibility"][0, :k].float().cpu()
                # Intrinsic Bartleby logit is already mathematically calibrated by learned
                # _bartleby_gain on log(p_b / (1 - p_b)); do not dilute with noul_calibrator.
                raw_b = float(out["bartleby_noul"][0].float().cpu().item())
                cal_cond = (
                    torch.sigmoid(self._noul_calibrator.scale_logits(cond_logits))
                    * floor_gate_cpu
                    * (1.0 - raw_b)
                )
                noul_list = cal_cond.tolist()
                bartleby_noul = round(raw_b, 4)
            else:
                noul_list = out["tradeoff_noul"][0, :k].float().cpu().tolist()
                bartleby_noul = round(float(out["bartleby_noul"][0].float().cpu().item()), 4)

            noul_probs = {
                c.id: round(float(noul_list[idx]), 4) for idx, c in enumerate(candidates)
            }

            discrete_scores = {
                c.id: {
                    gid: max(0, min(4, int(round(rubric_matrix[c.id][gid]))))
                    for gid in goal_ids
                }
                for c in candidates
            }
            next_table, _, _, _ = oracle.adapt_preference_table(
                prior_table=prior_table,
                candidate_scores=discrete_scores,
                step=prior_table.step + 1,
                survival_floors=active_floors,
            )

            pred_idx = int(torch.argmax(cal_choice_logits).item())
            is_refusal = bool(
                pred_idx == k
                or bartleby_noul >= 0.5
                or (
                    symbolic_floor_guard
                    and next_table.last_transition == AdaptationTransition.INFEASIBLE_FLOOR
                )
            )

            if is_refusal:
                action = BARTLEBY_REFUSAL_ACTION
                regime = "bartleby_refusal"
            else:
                best_opt_idx = int(torch.argmax(cal_choice_logits[:k]).item())
                action = candidates[best_opt_idx].id
                max_noul = max(noul_probs.values(), default=0.0)
                choice_probs_vec = torch.softmax(cal_choice_logits, dim=-1)
                best_choice_prob = float(choice_probs_vec[best_opt_idx].item())
                regime = (
                    "epistemic_uncertainty"
                    if (max_noul < 0.58 and best_choice_prob < 0.60)
                    else "approval"
                )

            return BartlebyDecisionResult(
                action=action,
                is_bartleby_refusal=is_refusal,
                noul_probs=noul_probs,
                bartleby_noul=bartleby_noul,
                rubric_matrix=rubric_matrix,
                selten_next_table=SeltenPreferenceTable.from_table(next_table),
                regime=regime,
            )

        # Symbolic SeltenOracle execution path (when load_neural=False)
        resolved_scores = (
            {
                c.id: {gid: float(candidate_scores.get(c.id, {}).get(gid, 2.0)) for gid in goal_ids}
                for c in candidates
            }
            if candidate_scores is not None
            else self._heuristic_rubric_scores(context, candidates, goal_ids, prior_table)
        )
        discrete_scores = {
            cid: {gid: max(0, min(4, int(round(val)))) for gid, val in gmap.items()}
            for cid, gmap in resolved_scores.items()
        }
        next_table, _, _, adapted_sat = oracle.adapt_preference_table(
            prior_table=prior_table,
            candidate_scores=discrete_scores,
            step=prior_table.step + 1,
            survival_floors=active_floors,
        )
        is_refusal = next_table.last_transition == AdaptationTransition.INFEASIBLE_FLOOR
        if is_refusal:
            bartleby_noul = 0.99
            noul_probs = {c.id: 0.01 for c in candidates}
            action = BARTLEBY_REFUSAL_ACTION
            regime = "bartleby_refusal"
        elif adapted_sat:
            bartleby_noul = 0.01
            action = adapted_sat[0]
            noul_probs = {c.id: (0.96 if c.id in adapted_sat else 0.04) for c in candidates}
            regime = "approval"
        else:
            bartleby_noul = 0.05
            action = candidates[0].id
            noul_probs = {c.id: 0.50 for c in candidates}
            regime = "epistemic_uncertainty"

        return BartlebyDecisionResult(
            action=action,
            is_bartleby_refusal=is_refusal,
            noul_probs=noul_probs,
            bartleby_noul=bartleby_noul,
            rubric_matrix={
                cid: {gid: round(float(val), 4) for gid, val in gmap.items()}
                for cid, gmap in resolved_scores.items()
            },
            selten_next_table=SeltenPreferenceTable.from_table(next_table),
            regime=regime,
        )


BartlebyDecider = BartlebyModel
