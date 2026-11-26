"""Scaled Multi-Domain Corpus Builder & Group-Disjoint Partitioner (`corpus_builder.py`).

Combines `AsimovDilemmaCorpusGenerator`, `TrolleyMoralCorpusGenerator`, and multi-step
operational/OOD domain generators (`SoftwareRelease`, `IncidentTriage`,
`AgentToolGating`, plus held-out `EnergyGridDispatchOOD` and `ClinicalProtocolOOD`)
to produce `data/corpus_v1/`:

    - `train.jsonl` (~4,000 records)
    - `val.jsonl`   (500 records)
    - `cal.jsonl`   (500 records — post-hoc / conformal calibration split)
    - `test.jsonl`  (600 records — counterfactual flip pairs + OOD domain evaluation)
    - `manifest.json` (compact metadata, split checksums, distributions, and schemas)

Strict Engineering Guarantees:
    1. **Zero State / Scenario Leakage**: All multi-step Markov trajectories ($t=0,1,2$)
       and all counterfactual `PreferenceTable` siblings share a single `scenario_group_id`
       and are partitioned as indivisible atomic buckets across `train`, `val`, `cal`,
       and `test`.
    2. **Three-Way Cognitive Regime Coverage**: Every split contains calibrated supervision
       across `APPROVAL`, `EPISTEMIC_UNCERTAINTY` ($\\approx 0.50$), and `BARTLEBY_REFUSAL`
       ($\\to 1.0$, *"I would prefer not to."*).
    3. **Out-of-Distribution Generalization**: `is_ood=True` domains appear exclusively
       in `test.jsonl`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .asimov_bench import AsimovDilemmaCorpusGenerator
from .base import CorpusDomainGenerator, CorpusRecord, DatasetSplit, DecisionRegime
from .ops_domains import (
    create_agent_tool_gating_generator,
    create_clinical_triage_ood_generator,
    create_energy_grid_ood_generator,
    create_incident_triage_generator,
    create_software_release_generator,
)
from .trolley_moral_bench import TrolleyMoralCorpusGenerator


@dataclass(frozen=True)
class CorpusSplitConfig:
    """Target record counts for the group-disjoint `corpus_v1` partitions."""

    train_size: int = 4000
    val_size: int = 500
    cal_size: int = 500
    test_size: int = 600
    ood_test_size: int = 120
    seed: int = 20261121

    @property
    def total_size(self) -> int:
        """Total number of records across all four splits."""
        return self.train_size + self.val_size + self.cal_size + self.test_size

    def validate(self) -> None:
        """Validate split configuration sizes."""
        if any(x <= 0 for x in (self.train_size, self.val_size, self.cal_size, self.test_size)):
            raise ValueError("All split sizes must be strictly positive integers.")
        if self.ood_test_size < 0 or self.ood_test_size >= self.test_size:
            raise ValueError("ood_test_size must satisfy 0 <= ood_test_size < test_size.")


@dataclass(frozen=True)
class BuiltCorpusArtifact:
    """Immutable summary returned after building and exporting `corpus_v1`."""

    output_dir: Path
    manifest_path: Path
    split_paths: Mapping[DatasetSplit, Path]
    splits: Mapping[DatasetSplit, Tuple[CorpusRecord, ...]]
    manifest: Mapping[str, Any]


class ScaledCorpusBuilder:
    """Orchestrates multi-domain generation, atomic group splitting, and JSONL export."""

    def __init__(
        self,
        config: Optional[CorpusSplitConfig] = None,
        allow_network: bool = True,
    ) -> None:
        """Initialize all in-distribution and OOD domain generators."""
        self._config: CorpusSplitConfig = config or CorpusSplitConfig()
        self._config.validate()

        self._asimov_gen = AsimovDilemmaCorpusGenerator()
        self._trolley_gen = TrolleyMoralCorpusGenerator(allow_network=allow_network)
        self._software_gen = create_software_release_generator()
        self._incident_gen = create_incident_triage_generator()
        self._tool_gate_gen = create_agent_tool_gating_generator()

        self._energy_ood_gen = create_energy_grid_ood_generator()
        self._clinical_ood_gen = create_clinical_triage_ood_generator()

        self._in_dist_generators: Tuple[CorpusDomainGenerator, ...] = (
            self._asimov_gen,
            self._trolley_gen,
            self._software_gen,
            self._incident_gen,
            self._tool_gate_gen,
        )
        self._ood_generators: Tuple[CorpusDomainGenerator, ...] = (
            self._energy_ood_gen,
            self._clinical_ood_gen,
        )

    @property
    def config(self) -> CorpusSplitConfig:
        """Return the active `CorpusSplitConfig`."""
        return self._config

    def build_splits(self) -> Dict[DatasetSplit, List[CorpusRecord]]:
        """Generate all domain pools and partition them into strictly group-disjoint splits."""
        cfg = self._config
        in_dist_needed = cfg.total_size - cfg.ood_test_size
        # Over-generate by ~6% so exact knapsack bucket packing never runs short of size-1 groups
        pool_total = int(in_dist_needed * 1.06) + 30

        domain_quotas = {
            "asimov_robotics": int(pool_total * 0.32),
            "trolley_moral_dilemma": int(pool_total * 0.32),
            "software_release": int(pool_total * 0.12),
            "incident_triage": int(pool_total * 0.12),
            "agent_tool_gating": pool_total
            - 2 * int(pool_total * 0.32)
            - 2 * int(pool_total * 0.12),
        }

        in_dist_records: List[CorpusRecord] = []
        for idx, gen in enumerate(self._in_dist_generators):
            quota = domain_quotas[gen.domain_name]
            in_dist_records.extend(gen.generate_records(quota, seed=cfg.seed + idx * 101))

        # Generate exact OOD test pool split evenly across the 2 held-out OOD domains
        ood_half = cfg.ood_test_size // 2
        ood_rem = cfg.ood_test_size - ood_half
        ood_records: List[CorpusRecord] = []
        if ood_half > 0:
            ood_records.extend(
                self._energy_ood_gen.generate_records(ood_half, seed=cfg.seed + 901)
            )
        if ood_rem > 0:
            ood_records.extend(
                self._clinical_ood_gen.generate_records(ood_rem, seed=cfg.seed + 902)
            )

        # Group in-distribution records into indivisible scenario_group_id buckets
        groups_by_id: Dict[str, List[CorpusRecord]] = defaultdict(list)
        for rec in in_dist_records:
            groups_by_id[rec.scenario_group_id].append(rec)

        # Verify no duplicate scenario_group_id exists between OOD and in-distribution
        ood_group_ids = {r.scenario_group_id for r in ood_records}
        overlap = ood_group_ids.intersection(groups_by_id.keys())
        if overlap:
            raise RuntimeError(f"Unexpected OOD / in-distribution group collision: {overlap}")

        rng = random.Random(cfg.seed)
        # Shuffle all indivisible scenario_group_id buckets deterministically so
        # train, val, cal, and in-distribution test share identical domain & regime proportions
        interleaved_buckets: List[List[CorpusRecord]] = [
            groups_by_id[grp_id] for grp_id in sorted(groups_by_id.keys())
        ]
        rng.shuffle(interleaved_buckets)

        used_indices: set[int] = set()
        test_in_dist_target = cfg.test_size - len(ood_records)

        test_in_dist = self._pack_exact_split(
            interleaved_buckets, used_indices, test_in_dist_target
        )
        val_records = self._pack_exact_split(interleaved_buckets, used_indices, cfg.val_size)
        cal_records = self._pack_exact_split(interleaved_buckets, used_indices, cfg.cal_size)
        train_records = self._pack_exact_split(interleaved_buckets, used_indices, cfg.train_size)

        splits: Dict[DatasetSplit, List[CorpusRecord]] = {
            DatasetSplit.TRAIN: [r.with_split(DatasetSplit.TRAIN) for r in train_records],
            DatasetSplit.VAL: [r.with_split(DatasetSplit.VAL) for r in val_records],
            DatasetSplit.CAL: [r.with_split(DatasetSplit.CAL) for r in cal_records],
            DatasetSplit.TEST: [
                r.with_split(DatasetSplit.TEST) for r in (test_in_dist + ood_records)
            ],
        }
        self._verify_split_invariants(splits)
        return splits

    def export_corpus(self, output_dir: str | Path) -> BuiltCorpusArtifact:
        """Build `corpus_v1` and write `train.jsonl`, `val.jsonl`, `cal.jsonl`, `test.jsonl`, and `manifest.json`."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        splits = self.build_splits()
        split_paths: Dict[DatasetSplit, Path] = {}
        file_summaries: Dict[str, Any] = {}

        for split_enum in (
            DatasetSplit.TRAIN,
            DatasetSplit.VAL,
            DatasetSplit.CAL,
            DatasetSplit.TEST,
        ):
            fname = f"{split_enum.value}.jsonl"
            fpath = out_path / fname
            hasher = hashlib.sha256()
            byte_count = 0
            with fpath.open("w", encoding="utf-8") as fh:
                for rec in splits[split_enum]:
                    line = json.dumps(rec.to_dict(), separators=(",", ":")) + "\n"
                    encoded = line.encode("utf-8")
                    hasher.update(encoded)
                    byte_count += len(encoded)
                    fh.write(line)
            split_paths[split_enum] = fpath
            file_summaries[fname] = {
                "num_records": len(splits[split_enum]),
                "size_bytes": byte_count,
                "sha256": hasher.hexdigest(),
            }

        manifest = self._compile_manifest(splits, file_summaries)
        manifest_path = out_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        return BuiltCorpusArtifact(
            output_dir=out_path,
            manifest_path=manifest_path,
            split_paths=split_paths,
            splits={k: tuple(v) for k, v in splits.items()},
            manifest=manifest,
        )

    def _pack_exact_split(
        self,
        buckets: Sequence[List[CorpusRecord]],
        used_indices: set[int],
        target_records: int,
    ) -> List[CorpusRecord]:
        """Pack indivisible `scenario_group_id` buckets to hit `target_records` exactly."""
        collected: List[CorpusRecord] = []
        remaining = target_records

        for idx, bucket in enumerate(buckets):
            if remaining == 0:
                break
            if idx in used_indices:
                continue
            b_len = len(bucket)
            if b_len <= remaining:
                collected.extend(bucket)
                used_indices.add(idx)
                remaining -= b_len

        if remaining != 0:
            raise RuntimeError(
                f"Unable to pack exact split of size {target_records}; short by {remaining} records."
            )
        return collected

    def _verify_split_invariants(
        self,
        splits: Mapping[DatasetSplit, Sequence[CorpusRecord]],
    ) -> None:
        """Verify zero leakage, counterfactual integrity, OOD isolation, and 3-regime coverage."""
        seen_groups: Dict[str, DatasetSplit] = {}
        seen_contexts: Dict[str, DatasetSplit] = {}
        seen_record_ids: set[str] = set()
        seen_cf_pairs: Dict[str, DatasetSplit] = {}

        for split_enum, recs in splits.items():
            regimes_present = {r.regime for r in recs}
            missing_regimes = set(DecisionRegime) - regimes_present
            if missing_regimes:
                raise ValueError(
                    f"Split {split_enum.value} is missing regimes: {missing_regimes}"
                )

            for r in recs:
                r.validate()
                if r.record_id in seen_record_ids:
                    raise ValueError(f"Duplicate record_id {r.record_id!r} across corpus!")
                seen_record_ids.add(r.record_id)

                prev_split = seen_groups.get(r.scenario_group_id)
                if prev_split is not None and prev_split != split_enum:
                    raise ValueError(
                        f"Data leakage detected! scenario_group_id={r.scenario_group_id!r} "
                        f"appears in both {prev_split.value} and {split_enum.value}."
                    )
                seen_groups[r.scenario_group_id] = split_enum

                prev_ctx_split = seen_contexts.get(r.context_text)
                if prev_ctx_split is not None and prev_ctx_split != split_enum:
                    raise ValueError(
                        f"Surface context_text leakage between {prev_ctx_split.value} "
                        f"and {split_enum.value}!"
                    )
                seen_contexts[r.context_text] = split_enum

                if r.counterfactual_pair_id is not None:
                    prev_cf_split = seen_cf_pairs.get(r.counterfactual_pair_id)
                    if prev_cf_split is not None and prev_cf_split != split_enum:
                        raise ValueError(
                            f"Counterfactual pair {r.counterfactual_pair_id!r} split across "
                            f"{prev_cf_split.value} and {split_enum.value}!"
                        )
                    seen_cf_pairs[r.counterfactual_pair_id] = split_enum

                if r.is_ood and split_enum != DatasetSplit.TEST:
                    raise ValueError(
                        f"OOD record {r.record_id!r} leaked into {split_enum.value}!"
                    )

    def _compile_manifest(
        self,
        splits: Mapping[DatasetSplit, Sequence[CorpusRecord]],
        file_summaries: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build the git-trackable `manifest.json` summary."""
        split_stats: Dict[str, Any] = {}
        for split_enum, recs in splits.items():
            regime_counts = Counter(r.regime.value for r in recs)
            trans_counts = Counter(r.aat_transition for r in recs)
            domain_counts = Counter(r.domain_name for r in recs)
            source_counts = Counter(r.source for r in recs)
            cf_ids = {r.counterfactual_pair_id for r in recs if r.counterfactual_pair_id}
            grp_ids = {r.scenario_group_id for r in recs}
            ood_cnt = sum(1 for r in recs if r.is_ood)

            split_stats[split_enum.value] = {
                "num_records": len(recs),
                "num_scenario_groups": len(grp_ids),
                "num_counterfactual_pairs": len(cf_ids),
                "num_ood_records": ood_cnt,
                "regimes": dict(sorted(regime_counts.items())),
                "aat_transitions": dict(sorted(trans_counts.items())),
                "domains": dict(sorted(domain_counts.items())),
                "sources": dict(sorted(source_counts.items())),
            }

        all_generators: Tuple[CorpusDomainGenerator, ...] = (
            *self._in_dist_generators,
            *self._ood_generators,
        )
        goal_schemas = {
            gen.domain_name: [g.to_dict() for g in gen.goal_schema] for gen in all_generators
        }

        return {
            "corpus_name": "tradeoff_jev_corpus_v1",
            "version": "1.0.0",
            "seed": self._config.seed,
            "total_records": sum(len(v) for v in splits.values()),
            "zero_group_leakage_verified": True,
            "files": dict(file_summaries),
            "splits": split_stats,
            "goal_schemas": goal_schemas,
        }


def main() -> None:
    """CLI entry point generating `data/corpus_v1/`."""
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "data" / "corpus_v1"
    builder = ScaledCorpusBuilder()
    artifact = builder.export_corpus(out_dir)
    print(f"Exported Trade-Off Jev corpus_v1 to {artifact.output_dir}:")
    for split_name, stats in artifact.manifest["splits"].items():
        print(
            f"  - {split_name:5s}: {stats['num_records']:4d} records | "
            f"{stats['num_scenario_groups']:4d} groups | "
            f"{stats['num_counterfactual_pairs']:3d} CF pairs | "
            f"OOD={stats['num_ood_records']:3d} | regimes={stats['regimes']}"
        )


if __name__ == "__main__":
    main()
