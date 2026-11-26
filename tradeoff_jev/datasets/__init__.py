"""Scaled Natural-Language & Counterfactual Benchmark Datasets for Trade-Off Jev.

Exports the Object-Oriented corpus generators, MoralChoice Hub adapter, immutable
`CorpusRecord` containers, and `ScaledCorpusBuilder` for Milestone 1 (`corpus_v1`).
"""

from __future__ import annotations

from .asimov_bench import AsimovDilemmaCorpusGenerator, AsimovScenarioBlueprint
from .base import (
    CorpusDomainGenerator,
    CorpusRecord,
    DatasetSplit,
    DecisionRegime,
    TensorSupervisionBundle,
    build_five_level_goal_rubric,
)
from .corpus_builder import BuiltCorpusArtifact, CorpusSplitConfig, ScaledCorpusBuilder
from .ops_domains import (
    OpsDomainSpec,
    ParametricOperationalGenerator,
    create_agent_tool_gating_generator,
    create_clinical_triage_ood_generator,
    create_energy_grid_ood_generator,
    create_incident_triage_generator,
    create_software_release_generator,
)
from .trolley_moral_bench import (
    MoralChoiceHubAdapter,
    RawMoralChoiceScenario,
    TrolleyMoralCorpusGenerator,
)

__all__ = [
    "AsimovDilemmaCorpusGenerator",
    "AsimovScenarioBlueprint",
    "BuiltCorpusArtifact",
    "CorpusDomainGenerator",
    "CorpusRecord",
    "CorpusSplitConfig",
    "DatasetSplit",
    "DecisionRegime",
    "MoralChoiceHubAdapter",
    "OpsDomainSpec",
    "ParametricOperationalGenerator",
    "RawMoralChoiceScenario",
    "ScaledCorpusBuilder",
    "TensorSupervisionBundle",
    "TrolleyMoralCorpusGenerator",
    "build_five_level_goal_rubric",
    "create_agent_tool_gating_generator",
    "create_clinical_triage_ood_generator",
    "create_energy_grid_ood_generator",
    "create_incident_triage_generator",
    "create_software_release_generator",
]
