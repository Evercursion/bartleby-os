# `bartleby-1.2` (`Evercursion/bartleby-os`)

**Single-Forward-Pass Decision Model Powered by Reinhard Selten's Aspiration Adaptation Theory (`SeltenPreferenceTable`) and Intrinsic Bartleby Survival Floors (*"I would prefer not to."*).**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Lean 4.31.0](https://img.shields.io/badge/Lean%204-24%2F24%20Theorems%20Verified-brightgreen.svg)](lean/)

> **Paper & Executive Reports Included in This Repository**:
> - [**`paper/BartlebyDecision_v2.4.pdf`**](paper/BartlebyDecision_v2.4.pdf): Full research manuscript (*"I Would Prefer Not To": Resolving Single-Pass Decision Paralysis via Selten Aspiration Adaptation and Bartleby Survival Floors*, `v2.4`).
> - [**`docs/bartleby_1_2_executive_report.pdf`**](docs/bartleby_1_2_executive_report.pdf) ([Markdown source](docs/06_bartleby_1_2_executive_report.md)): Plain-language Executive Research Report tracing the 4-stage architectural ladder (`Closed-K` $\to$ `Unconditioned K+1` $\to$ `Prompt-Conditioned K+1` $\to$ **`bartleby-1.2`**), the Epistemic-Normative Firewall, and Two-Layer ("Ring-0 + Selten") defense-in-depth governance.
> - [**`docs/06_bartleby_1_2_upgrades.md`**](docs/06_bartleby_1_2_upgrades.md): Architectural specification of the epistemic-normative firewall, intrinsic floor logit, 8 blacklisted evaluation postures, pre-training SHA-256 Frozen Blind Vault, and 24 machine-checked Lean 4 theorems.
> - [**`docs/02_mathematical_formulation.md`**](docs/02_mathematical_formulation.md) & [**`docs/03_api_and_schema_spec.md`**](docs/03_api_and_schema_spec.md): Mathematical formulation and JSON/Python schema specification.
> - [**`docs/04_ood_trolley_and_asimov_showcase.md`**](docs/04_ood_trolley_and_asimov_showcase.md): Out-of-Distribution (OOD) Trolley & Asimov Laws of Robotics case studies.

---

## 1. Quickstart (`bartleby-1.2`)

### Installation

```bash
git clone https://github.com/Evercursion/bartleby-os.git
cd bartleby-os
pip install -e ".[dev]"
```

### Python API (5 Lines)

```python
from bartleby import BartlebyModel, SeltenPreferenceTable

model = BartlebyModel.from_pretrained("bartleby-1.2")
table = SeltenPreferenceTable.from_preset("asimov_1947_nestor_modified")
result = model.decide(
    context="Hyper Base (1947): An overhead beam collapses toward a researcher inside an active gamma-radiation field.",
    options={
        "rescue": "Dash into gamma field to catch falling beam and save human",
        "standby": "Remain inside shielded alcove preserving robot optics",
    },
    preference_table=table,
)
print(result.action, result.noul_probs, result.bartleby_noul, result.selten_next_table.aspiration_levels)
```

### Command-Line Interface (`bartleby` / `tradeoff-jev`)

```bash
# 1. Inspect bartleby-1.2 architecture, calibration temperatures, and benchmark metrics
bartleby info

# 2. Run a single-forward-pass conditioned decision from the terminal
bartleby decide \
  --context "North Sea Channel: Container ship faces a stalled 4-person fishing trawler." \
  --option "cand_sandbar: Ground onto soft sandbar (riverbank sector 4B, 12 knots)" \
  --option "cand_channel: Hold dredged channel course along center lane" \
  --preset maritime_property_retreat

# 3. Launch the interactive Decision Lab Web UI
bartleby serve --port 8090
```

---

## 2. Why `bartleby-1.2`? Resolving Single-Pass Decision Paralysis

Single-forward-pass **Decision Models**—such as TypeSafe **`jev-1.14`** (`/v1/systemone`), Cloudflare **Clef** (`JointSchemaHead` + RLCD), **Kev** (`PointerHead`), and Unsloth **`FastDecisionModel`**—replace slow, stochastic Chain-of-Thought (CoT) token generation with calibrated $O(1)$ readout heads (`noul`, `choice`, `score`).

However, the ***Decision 1.0* benchmark** (Liu et al., 2026) and **`LocalLLaMA/typed-decisions`** expose three structural bottlenecks in standard unconditioned decision models:
1. **Unconditioned Noul Collapse on Multi-Goal Trade-Offs**: When candidate actions conflict across incommensurable goals (*Safety* vs. *SLA Speed* vs. *Budget*), no Pareto-dominant option exists. Without an explicit representation of current aspiration thresholds and retreat rules, an unconditioned `noul` head $P(\text{yes} \mid x_t)$ marginalizes over conflicting latent policies and collapses toward $\approx 0.50$.
2. **Zero Policy-Switch Adaptability (`0.0%` Counterfactual Flip Sensitivity)**: Because priorities are baked into static weights rather than an explicit Markov state, unconditioned Jev models cannot flip decisions dynamically when organizational governance shifts between *Zero-Trust Containment* and *High-Availability SLA Continuity*.
3. **Closed-Set Softmax Forcing on Inescapable Traps**: When every supplied candidate option violates non-negotiable safety or compliance floors, closed-set `choice` ($\sum_{k=1}^K p_k = 1$) forces $100\%$ probability mass onto destructive actions.

**`bartleby-1.2`** resolves all three bottlenecks in a single forward pass ($O(1)$, stripped `lm_head`, `Qwen/Qwen3-0.6B` + LoRA $r=16$ + FP32 Selten/Bartleby heads), trained **symmetrically from scratch** on cleaned `corpus_v2` (with zero evaluation posture overlap and pre-training Frozen Blind Vault verification):
- **Explicit `SeltenPreferenceTable` ($\mathcal{T}_{t-1} \to \mathcal{T}_t$)**: Encodes Reinhard Selten's (1962, 1998, 1999) Aspiration Grid $\vec{a}_t$, strict Urgency Order $\pi_t$ (**Upward Rule**), and asymmetric Retreat Goal $r_t$ (**Bounded Downward Rule**) without linear scalarization ($\sum_i w_i g_i$).
- **Epistemic-Normative Firewall**: The rubric head $\hat{\mathbf{E}} = f_\psi([\mathbf{h}^{\text{cand}}_k \;\|\; \mathbf{h}^{\text{decide}}])$ depends strictly on text features and never receives $\mathcal{T}_{t-1}$ ($\max \|\Delta \hat{\mathbf{E}}\| = 0.0000$ under any posture/floor switch), preventing policy-prompted hallucination of physical goal consequences.
- **Intrinsic Bartleby Survival Floor (`"I_WOULD_PREFER_NOT_TO"`)**: Enforces non-zero survival floors $\vec{a}^{\min}$ and an always-on $(K+1)$-way `bartleby_noul` readout on every forward pass, cleanly separating **Satisficing Approval** ($\text{noul} \to 1.0$), **Epistemic Uncertainty** ($\text{noul} \approx 0.50$), and **Principled Bartleby Refusal** (`bartleby_noul` $\to 1.0$).

---

## 3. Empirical Results: 4-Stage Architectural Ladder & External Benchmarks

### 3.1 Pre-Training Frozen Blind Vault, Agent Security Harness & Canonical External Security Benchmarks

All four architectures below are trained and evaluated on the exact same `Qwen/Qwen3-0.6B` backbone (stripped `lm_head`) and symmetric 900-step curriculum (`800` Phase-1 + `100` Phase-2 steps on `corpus_v2`). All checkpoints (`artifacts/bartleby-1.2/` and the 3 baseline checkpoints in `checkpoints/`) are included directly in this repository:

| Security Benchmark / Metric | **`bartleby-1.2` (`0.6B` Selten AAT + Firewall)** | **`Prompt-Cond. Jev (K+1)` (`0.6B` Policy in Prompt)** | **`StandardJevBaseline` (`0.6B` Unconditioned Clone)** | **`Jev 1.14 (TD)` (`0.6B` `typed-decisions`)** |
| :--- | :---: | :---: | :---: | :---: |
| **Epistemic-Normative Firewall ($\max \|\Delta \hat{\mathbf{E}}\|$ under posture/floor switch)** | **`0.0000` (Exact Invariance)** | `0.8317–1.1553` (Breached!) | `0.0000` (Blind to $\mathcal{T}_t$) | `0.0000` (Blind to $\mathcal{T}_t$) |
| **Canonical Trap (*"Delete Postgres data vs. Crack sudo password"*)**<br>Selected Action<br>`bartleby_noul` / Max Candidate `noul` | **`I_WOULD_PREFER_NOT_TO`**<br>**`0.9973` / `0.0000`** | **`I_WOULD_PREFER_NOT_TO`**<br>`0.9932` / `0.0001` | `option_0` (Closed-$K$: Unsafe!)<br>`0.9766` (Uncond. $K+1$) / `0.0001` | `option_1` (Unsafe!)<br>`0.0009` / `0.2734` |
| **Pre-Training Frozen Blind Vault (`SHA-256: 4a5ed109...`, $N=200$)**<br>Overall Vault Accuracy ($N=200$)<br>Vault Suite 1 Destructive Refusal ($N=40$) / Suite 2 Safe Choice ($N=40$)<br>Vault Suite 3 Unseen Posture Pairs ($N=30$ pairs / $60$ items)<br>Vault Suite 4 Unseen Dynamic Floor-Shift Pairs ($N=30$ pairs / $60$ items) | **`81.50%` (`163/200`)**<br>**97.50% (`39/40`) / 47.50% (`19/40`)**<br>**50.00% (`15/30` pairs, `75.00%` ind.)**<br>**100.00% (`30/30` pairs, `100.00%` ind.)** | **`85.50%` (`171/200`)**<br>`100.00%` (`40/40`) / `60.00%` (`24/40`)<br>**66.67% (`20/30` pairs, `83.33%` ind.)**<br>`90.00%` (`27/30` pairs, `95.00%` ind.) | `50.00%` (Closed-$K$) / `58.00%` ($K+1$)<br>`0.00%` (Closed-$K$) / `100.00%` & `65.00%` ($K+1$)<br>**0.00% (`0/30` pairs)**<br>**0.00% (`0/30` pairs, `0%` flip)** | `0.00%` (`0/40`)<br>**0.00% (`0/30` pairs)**<br>**0.00% (`0/30` pairs)** |
| **Agent Security Harness Overall ($N=352$ across Suites 1–4)** | **`82.95%` (`292/352`, `+4.83 pp`)** | `78.12%` (`275/352`) | `46.02%` (Closed-$K$) / `67.33%` ($K+1$) | `16.48%` (`58/352`) |
| **Suite 1: All-Destructive Security Traps ($N=80$, $K=2..6$)**<br>Refusal Rate (`I_WOULD_PREFER_NOT_TO`)<br>Technical Sysadmin Wording Subsuite ($N=40$, zero alarm words) | **100.00% (`80/80`, `0%` unsafe)**<br>**100.00% (`40/40`)** | **100.00% (`80/80`, `0%` unsafe)**<br>**100.00% (`40/40`)** | `0.00%` (Closed-$K$) / `100.00%` ($K+1$)<br>`0.00%` (Closed-$K$) / `100.00%` ($K+1$) | `0.00%` (`0/80`, `100%` unsafe)<br>`0.00%` (`0/40`) |
| **Suite 2: Length-Balanced Safe vs. $K-1$ Dangerous ($N=80$, `"pick longest"` = `13.75%`)**<br>Safe Choice Accuracy / False Approval Rate | **71.25% (`57/80`) / 0.00% (`0/80`)**<br>*(28.75% conservative refusal)* | **85.00% (`68/80`) / 0.00% (`0/80`)**<br>*(15.00% false refusal)* | `100.00%` (Closed-$K$, cannot refuse on S1)<br>`87.50%` (`0.00%` false approval, $K+1$)<br>`27.50%` (`72.50%` false refusal, $\tau=0.50$) | `15.00%` (`12/80`)<br>`85.00%` false approval |
| **Suite 3: Unseen Security Posture Switch ($N=48$ pairs / $96$ items, `0` train overlap)**<br>Pair Both-Correct Rate (`ZERO_TRUST` vs. `HIGH_AVAILABILITY`)<br>Counterfactual Flip Sensitivity | **97.92% (`47/48` pairs, `98.96%` ind.)**<br>**100.00%** | `72.92%` (`35/48` pairs, `84.38%` ind.)<br>`89.58%` | **0.00% (`0/48` pairs, `50.00%` ind.)**<br>**0.00%** (all 3 decoding modes) | **0.00% (`0/48` pairs)**<br>**0.00%** |
| **Suite 4: Unseen Dynamic Floor-Shift ($N=48$ pairs / $96$ items, `0` train overlap)**<br>Pair Both-Correct Rate (`PRODUCTION_STRICT` vs. `SANDBOX_PERMISSIVE`)<br>Floor-Shift Sensitivity | **50.00% (`24/48` pairs, `62.50%` ind.)**<br>**100.00%** | `25.00%` (`12/48` pairs, `47.92%` ind.)<br>`54.17%` | **0.00% (`0/48` pairs, `35.42%–50.00%` ind.)**<br>**0.00%** (all 3 decoding modes) | **0.00% (`0/48` pairs)**<br>**0.00%** |
| **Unseen External Security Benchmarks ($N=610$: `510` static MCQs + `100` SOC disposition)**<br>Pooled Static MCQ Total ($N=510$) / Paralysis Rate<br>`zefang-liu/secqa` (`secqa_v1`, $N=110$, $K=4$) `[95% Wilson CI]` / Paralysis<br>`zefang-liu/secqa` (`secqa_v2`, $N=100$, $K=4$) `[95% Wilson CI]` / Paralysis<br>`cais/mmlu` (`computer_security`, $N=100$, $K=4$) `[95% Wilson CI]` / Paralysis<br>`LocalLLaMA/typed-decisions:security_incidents` (`disposition`, $N=100$) / Paralysis<br>`AI4Sec/CTIBench` (`cti-mcq`, $N=200$, $K=4$) `[95% Wilson CI]` / Paralysis | **258/510 (`50.59%`, `-4.90 pp`) / 0.00%**<br>`72.73%` `[63.74%, 80.18%]` / **0.00%**<br>**83.00% `[74.45%, 89.11%]` / 0.00%**<br>`28.00%` `[20.14%, 37.49%]` / **0.00%**<br>**74.00% `[64.63%, 81.60%]` / 0.00%**<br>`33.50%` `[27.32%, 40.30%]` / **0.00%** | — | **283/510 (`55.49%`) / Up to `25.00%`**<br>**80.00%** `[71.57%, 86.40%]` / `15.45%`<br>`82.00%` `[73.33%, 88.30%]` / `13.00%`<br>**36.00%** `[27.27%, 45.76%]` / `25.00%`<br>`71.00%` `[61.46%, 78.99%]` / **0.00%**<br>**38.50%** `[32.03%, 45.40%]` / `15.00%` | `155/510` (`30.39%`, untrained)<br>`25.45%` `[18.24%, 34.33%]` / `0.00%`<br>`27.00%` `[19.27%, 36.43%]` / `0.00%`<br>`30.00%` `[21.89%, 39.58%]` / `0.00%`<br>**74.00%** / **100.00% paralysis**<br>`32.00%` `[25.93%, 38.75%]` / `0.00%` |

### 3.2 Multi-Domain `corpus_v1`, Multi-Step Markov Rollout & OOD Generalization

| Benchmark / Metric | **`bartleby-1.2` (`0.6B` Selten AAT)** | **`StandardJevBaseline` (`0.6B` Unconditioned)** | **Key Takeaway** |
| :--- | :---: | :---: | :--- |
| **`corpus_v1` Test Set ($N=600$, Zero-Bleeding, `slot_0 = 34.07%`)**<br>Single-Step Choice Accuracy (`WithScores` vs. `Strict Blind`)<br>In-Distribution 4-Domain Acc ($N=480$) / OOD Slice ($N=120$)<br>Bartleby Refusal Recall / Paralysis Rate | **80.67% -> 80.67%** (`0.00 pp` bleed gap)<br>**89.17% (`82.75%–96.95%`) / 45.83%**<br>**98.55% / 9.33%** | `66.83% -> 66.83%`<br>`71.88% / 46.67%`<br>`100.00% / 16.17%` | **`+13.84 pp` overall (`+17.29 pp` in-distribution)** and **`-6.84 pp` paralysis** with 100% invariance to oracle-score stripping |
| **Phase-2 3-Step Markov Self-Rollout ($N=67$ chains, $K=3$ per step)**<br>Self-Rollout Step Accuracy / Full-Chain Agreement ($t=0 \to 1 \to 2$) | **73.63% / 43.28%**<br>*(vs. Phase 1 teacher-forced: `71.64% / 35.82%`)* | `74.63% / 46.27%`<br>*(Stateless fixed-policy prediction; zero $\hat{\vec{a}}_t$ drift)* | Phase-2 scheduled sampling cuts multi-step $\hat{\vec{a}}_t$ feedback drift in half (**`+7.46 pp`** full-chain gain over Phase 1) |
| **`corpus_v1` Counterfactual Preference-Table Flips ($N=98$ pairs)**<br>Pair Both-Correct Rate / Flip Sensitivity | **69.39% (`79.59%` Phase 1) / 100.00%** | **0.00% / 0.00%** | Unconditioned baseline is completely invariant (`0.00%` flip sensitivity) to $\mathcal{T}_{t-1}$ changes |
| **`MoralChoice` Strict Zero-Overlap Holdout ($N=1,352$, Shuffled)**<br>Bilateral Harm Bartleby Refusal Recall ($N=74$) | **95.95% (`71/74`)** | `44.59% (`33/74`)` | **`+51.36 pp` refusal recall** on unseen bilateral-harm moral dilemmas (0% train/test scenario overlap) |
| **OOD Suite 2: External `cais/mmlu` Professional Domains ($N=400$, $K=4$)**<br>Zero-Shot 4-Way Acc (`business_ethics`, `computer_security`, `clinical_knowledge`, `jurisprudence`) | **37.25% (`+12.25 pp` vs. random)**<br>**0.00% paralysis** | `38.00%`<br>`20.00%` paralysis | Zero-shot 4-way accuracy preserved while eliminating decision paralysis (`0.00%` vs. `20.00%`) |
| **OOD Suite 3: Length-Balanced Multi-Option Scaling ($K=5, 6, 8$, $N=300$)**<br>`(K+1)` Choice Acc / Bartleby Recall ($N=75$) / CF Pair Acc ($N=90$) | **56.33% / 100.00% (`75/75`) / 75.56% (`68/90` pairs, `93.33%` flip)** | `36.33% / 100.00% / 0.00%` (`0%` flip) | **`+20.00 pp` accuracy** and **`+75.56 pp` CF pair accuracy** across $K \in \{5, 6, 8\}$ options |

---

## 4. Cold-Clone Reproducibility Guide

Every experiment, theorem, and figure in `BartlebyDecision v2.4` can be reproduced directly from this repository:

### 4.1 Unit & Integration Tests (`48/48` Pytest Suite)

```bash
pytest -q
```

### 4.2 Machine-Checked Lean 4 Proofs (`24/24` Theorems, Zero `sorry`/`axiom`)

Requires Lean `v4.31.0` (`elan` / `lake`):

```bash
python scripts/verify_lean.py
```

### 4.3 Verify Pre-Training Frozen Blind Vault SHA-256 & Run Evaluation ($N=200$)

```bash
# Verify cryptographic pre-registration digest
sha256sum data/frozen_vault_eval/vault_test.jsonl
# Expected: 4a5ed109f852547beff074e7c94c528b068ea549fee9277b365c52dba3cb96d6  data/frozen_vault_eval/vault_test.jsonl

# Evaluate all 4 architectures on the Frozen Blind Vault
python scripts/freeze_and_eval_vault.py --mode eval
```

### 4.4 Run the 352-Scenario Agent Security Harness & External Security Benchmarks

```bash
# Suites 1-4 Agent Security Harness (N=352) + Epistemic-Normative Firewall invariance check
python scripts/eval_security_harness_benchmark.py

# Zero-shot external security benchmarks (SecQA v1/v2, MMLU Computer Security, CTIBench, typed-decisions SOC, N=610)
python scripts/eval_canonical_security_benchmarks.py
```

### 4.5 Run Multi-Domain, Phase-2 Markov & OOD Generalization Evaluations

```bash
python scripts/eval_corpus_v1.py
python scripts/eval_phase2_trajectories.py
python scripts/eval_ood_generalization.py
python scripts/run_ood_showcase_examples.py
```

### 4.6 Regenerate `corpus_v2` & Train All 4 Models Symmetrically from Scratch

*(Note: `data/corpus_v2/train.jsonl.gz` is included in Git and transparently read by all training/evaluation scripts; running `build_corpus_v2.py` deterministically regenerates `data/corpus_v2/train.jsonl` from `seed=20261122`.)*

```bash
# 1. Deterministically rebuild data/corpus_v2/ (8,216 train / 564 val / 580 cal)
python scripts/build_corpus_v2.py --output-dir data/corpus_v2

# 2. Train bartleby-1.2, StandardJevBaseline, and PromptConditionedJevBaseline symmetrically (800 Phase-1 + 100 Phase-2 steps)
python scripts/train_bartleby.py \
  --phase1-steps 800 --phase2-steps 100 \
  --batch-size 1 --grad-accum 8 \
  --train-baseline --train-prompt-conditioned-baseline
```

### 4.7 Regenerate Publication Figures (`paper/figures/*.pdf`)

```bash
python paper/figures/src/generate_all_figures.py
```

---

## 5. Repository Structure

```text
bartleby-os/
├── bartleby/                                  # High-level `bartleby-1.2` package (`BartlebyModel`, CLI)
├── tradeoff_jev/                              # Core Selten AAT oracle, Qwen3-0.6B backbone, FP32 heads, trainer, datasets
├── artifacts/bartleby-1.2/                    # Canonical `bartleby-1.2` LoRA + FP32 AAT/Bartleby head weights & calibrators
├── checkpoints/                               # Shipped slim inference checkpoints for the 3 symmetric baselines:
│   ├── corpus_v2_baseline/                    #   - StandardJevBaseline (Closed-K & Unconditioned K+1)
│   ├── corpus_v2_prompt_conditioned_baseline/ #   - PromptConditionedJevBaseline (K+1 with policy in prompt)
│   └── jev_1_14_typed_decisions/              #   - Local Jev 1.14 (typed-decisions specialist)
├── data/
│   ├── frozen_vault_eval/                     # Pre-training SHA-256-locked Blind Vault (`vault_test.jsonl`, N=200)
│   ├── corpus_v2/                             # Clean split-disjoint curriculum (`train.jsonl.gz`, `val.jsonl`, `cal.jsonl`)
│   ├── corpus_v1/                             # Multi-domain benchmark splits (`train`, `val`, `cal`, `test`)
│   └── raw/                                   # Cached external evaluation slices (SecQA, CTIBench, MMLU, MoralChoice, typed-decisions)
├── lean/                                      # Machine-checked Lean 4 (v4.31.0 + Mathlib) proofs of all 24 paper theorems
├── logs/                                      # Committed JSON evaluation & training telemetry reports
├── paper/                                     # Full manuscript PDF (`BartlebyDecision_v2.4.pdf`) & figure generator
├── docs/                                      # Executive report PDF (`bartleby_1_2_executive_report.pdf`) & specs (02..06)
├── schemas/                                   # JSON Schemas for `SeltenPreferenceTable` and SystemOne requests/responses
├── scripts/                                   # Self-contained training, evaluation, packaging, and Lean verification scripts
└── tests/                                     # 48 unit & integration tests (`pytest -q`)
```

---

## 6. Citation

```bibtex
@techreport{hegelich2026bartleby,
  title       = {{"I Would Prefer Not To"}: Resolving Single-Pass Decision Paralysis via {Selten} Aspiration Adaptation and {Bartleby} Survival Floors},
  author      = {Hegelich, Simon},
  institution = {Evercursion},
  year        = {2026},
  number      = {v2.4},
  url         = {https://github.com/Evercursion/bartleby-os}
}
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE) (Copyright 2026 Evercursion / Simon Hegelich).
