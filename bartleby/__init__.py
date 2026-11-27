"""`bartleby` (`bartleby-1.2`, v1.2.0) — Single-Forward-Pass Decision Model with Selten AAT & Bartleby Survival Floors.

Top-level distribution package exposing:
    - `BartlebyModel` / `BartlebyDecider`: High-level `from_pretrained("bartleby-1.2")` & `.decide(...)` API.
    - `BartlebyDecisionResult`: Immutable decision output with `action`, `is_bartleby_refusal`,
      `noul_probs`, `bartleby_noul`, `rubric_matrix`, `selten_next_table`, and `regime`.
    - `SeltenPreferenceTable`: First-class Reinhard Selten (1998, 1999) Preference Table ($\\mathcal{T}_t$)
      with `.from_preset(...)` and `.custom(...)` builders.
    - `SeltenOracle`: Symbolic Aspiration Adaptation & Bartleby Floor Oracle.
    - `SeltenAdaptationHead`: Differentiable FP32 Selten Aspiration Adaptation & Rubric Head.
    - `BartlebyReadoutHead` / `TorchPreferenceConditionedNoulHead`: Conditioned Noul & Bartleby Readout Head.
    - `BartlebyNeuralSystem`: End-to-end `Qwen/Qwen3-0.6B` + LoRA + FP32 AAT Neural System.
"""

from __future__ import annotations

from .decider import (
    BARTLEBY_MODEL_ID,
    BARTLEBY_REFUSAL_ACTION,
    BARTLEBY_VERSION,
    BartlebyDecider,
    BartlebyDecisionResult,
    BartlebyModel,
)
from tradeoff_jev.backbone import (
    BackboneConfig,
    BackboneFeatureProjector,
    BackboneReadoutFeatures,
    BartlebyNeuralSystem,
    BartlebyReadoutHead,
    BaseDecisionNeuralSystem,
    DecisionSequencePacker,
    PackedDecisionBatch,
    PoolingStrategy,
    QwenDecisionBackbone,
    SeltenAdaptationHead,
    StandardJevBaselineSystem,
    TorchAATOutput,
    TorchConditionedNoulHead,
    TorchDifferentiableAATModule,
    TorchPreferenceConditionedNoulHead,
    TorchPreferenceTableEmbedding,
    TradeOffJevNeuralSystem,
)
from tradeoff_jev.calibration import (
    CalibrationMode,
    CalibrationReport,
    ReliabilityBin,
    TemperatureCalibrator,
)
from tradeoff_jev.model import (
    AATCurriculumLoss,
    AATModelConfig,
    ConditionedNoulHead,
    DifferentiableAATModule,
    DifferentiableAATOutput,
    PreferenceTableEmbedding,
    TradeOffJevModel,
)
from tradeoff_jev.oracle import (
    BARTLEBY_UTTERANCE,
    AspirationAdaptationOracle,
    ColdStartInitializer,
    OracleDecisionOutcome,
    SeltenOracle,
)
from tradeoff_jev.schema import (
    AdaptationTransition,
    CandidateEvaluation,
    CandidateOption,
    FeasibilitySummary,
    GoalDefinition,
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
    TradeOffQuestion,
    TradeOffRequest,
    TradeOffResponse,
)

__version__: str = "1.2.0"

__all__ = [
    "AATCurriculumLoss",
    "AATModelConfig",
    "AdaptationTransition",
    "AspirationAdaptationOracle",
    "BARTLEBY_MODEL_ID",
    "BARTLEBY_REFUSAL_ACTION",
    "BARTLEBY_UTTERANCE",
    "BARTLEBY_VERSION",
    "BackboneConfig",
    "BackboneFeatureProjector",
    "BackboneReadoutFeatures",
    "BartlebyDecider",
    "BartlebyDecisionResult",
    "BartlebyModel",
    "BartlebyNeuralSystem",
    "BartlebyReadoutHead",
    "BaseDecisionNeuralSystem",
    "CalibrationMode",
    "CalibrationReport",
    "CandidateEvaluation",
    "CandidateOption",
    "ColdStartInitializer",
    "ConditionedNoulHead",
    "DecisionSequencePacker",
    "DifferentiableAATModule",
    "DifferentiableAATOutput",
    "FeasibilitySummary",
    "GoalDefinition",
    "GoalExpectation",
    "GoalSpec",
    "GridRule",
    "OracleDecisionOutcome",
    "PackedDecisionBatch",
    "PoolingStrategy",
    "PreferenceTable",
    "PreferenceTableEmbedding",
    "PreferenceTableOrigin",
    "QualitativeExpectation",
    "QuestionType",
    "QwenDecisionBackbone",
    "ReliabilityBin",
    "RubricLevel",
    "SeltenAdaptationHead",
    "SeltenOracle",
    "SeltenPreferenceTable",
    "StandardJevBaselineSystem",
    "TemperatureCalibrator",
    "TorchAATOutput",
    "TorchConditionedNoulHead",
    "TorchDifferentiableAATModule",
    "TorchPreferenceConditionedNoulHead",
    "TorchPreferenceTableEmbedding",
    "TradeOffAnswer",
    "TradeOffJevModel",
    "TradeOffJevNeuralSystem",
    "TradeOffQuestion",
    "TradeOffRequest",
    "TradeOffResponse",
    "__version__",
]
