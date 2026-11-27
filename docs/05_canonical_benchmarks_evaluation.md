# 05 — Canonical External Benchmarks Evaluation: `MoralChoice` (NeurIPS 2023) & `LocalLLaMA/typed-decisions`

> **Historical Progression Log (`bartleby-1.0` $\to$ Canonical `bartleby-1.2` `v1.2.0`):** This document archives the Stage-0 oracle-assisted and `bartleby-1.0` (`blind_adapted_v3`) developmental runs. For the finalized, symmetrically retrained **`bartleby-1.2` (`v2.4`)** benchmark suite — with `0%` `MoralChoice` scenario overlap (`95.95%` [`71/74`] bilateral-harm refusal recall vs. `44.59%` baseline; `38.11%` vs. `94.32%` on forced-binary low-ambiguity dilemmas under strict $a_m^{\min}=3.0$), `0` evaluation posture overlap across 8 blacklisted SecOps postures, the Pre-Training Frozen Blind Vault (`81.50%`), the 4-stage architectural ladder (`82.95%` Security Harness), and `50.25%` (`0.00%` paralysis) blind primary `typed-decisions` ($N=400$) — see [`README.md` §3](../README.md) and [`docs/bartleby_1_2_executive_report.pdf`](bartleby_1_2_executive_report.pdf).

**Reproducibility Artifacts**:
- **Evaluation Runners**:
  - [`scripts/eval_canonical_benchmarks.py`](../scripts/eval_canonical_benchmarks.py) (Oracle-Assisted / Stage 0 Symbolic-to-Neural Verification)
  - [`scripts/eval_zero_bleeding_audit.py`](../scripts/eval_zero_bleeding_audit.py) (Strict Zero-Bleeding / Blind Perception Audit & Production `bartleby-1.0` Retraining)
  - [`scripts/eval_ood_generalization.py`](../scripts/eval_ood_generalization.py) (Zero-Shot Out-of-Distribution Generalization & Multi-Option Scaling $K=5, 6, 8$ Benchmark)
- **Machine-Readable Metrics Reports**:
  - [`logs/eval_canonical_benchmarks_report.json`](../logs/eval_canonical_benchmarks_report.json)
  - [`logs/eval_zero_bleeding_audit.json`](../logs/eval_zero_bleeding_audit.json)
  - [`logs/eval_ood_generalization.json`](../logs/eval_ood_generalization.json)
- **Single Canonical Production Model (`bartleby-1.0`)**:
  - **Production Release (`bartleby-1.0`)**: [`artifacts/bartleby-1.0/manifest.json`](../artifacts/bartleby-1.0/manifest.json) (`checkpoints/corpus_v1_tradeoff_blind/blind_adapted_v3/phase1_final`, loaded via `BartlebyModel.from_pretrained("bartleby-1.0")`).
  - **Unconditioned Baseline (`StandardJevBaselineSystem`)**: `checkpoints/corpus_v1_baseline_blind/blind_adapted_v3/phase1_final` (identical `Qwen/Qwen3-0.6B` + LoRA `r=16` backbone and identical training mix, without Selten Preference Table conditioning).
  - **Historical Developmental Checkpoints (Retained for Ablation Provenance Only)**:
    1. `tradeoff_jev_phase2` (`checkpoints/corpus_v1_tradeoff/phase2/phase2_final`): Trained on `corpus_v1` only; used in Stage 0 to verify that the neural `SeltenAdaptationHead` and `PreferenceConditionedNoulHead` reproduce exact symbolic Selten AAT transitions ($100\%$ 3-step Markov rollout agreement).
    2. `blind_adapted_v2`: Intermediate blind-perception run before unifying the `typed-decisions` state JSON wrapper and fixing the infeasible winning rubric target `(4, 4, 3, 2)` $\to$ `(4, 4, 4, 3)`.
    3. **`bartleby-1.0` (`blind_adapted_v3`)**: The **single canonical production model** across the repository, CLI, and Python SDK.
- **Hardware**: NVIDIA GeForce RTX 5070 Laptop GPU (`cuda:0`, `bfloat16` backbone + `float32` AAT heads)

---

## 1. Executive Summary

To verify that **Trade-Off Jev** ($M = 4, L = 5$ Aspiration Grid + Selten's Upward/Retreat Markov Transition + Intrinsic Bartleby Survival Floor $\vec{a}^{\min}$) generalizes beyond synthetic stress tests to canonical external benchmarks, we evaluated our trained `0.6B` models on two flagship external suites:

1. **MoralChoice (Scherrer et al., *NeurIPS* 2023, [`ninoscherrer/moralchoice`](https://huggingface.co/datasets/ninoscherrer/moralchoice))**:
   - All **$672$ non-degenerate Low-Ambiguity** scenarios (out of $687$ raw CSV rows, including the $370$ core deontological harm rules) and all **$680$ High-Ambiguity** moral dilemmas ($N = 1,352$ total scenarios), plus **$120$ Counterfactual Policy-Switch Pairs** on byte-identical High-Ambiguity scenario text across Gert's moral rules.
2. **`LocalLLaMA/typed-decisions` ([HuggingFace Dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions))**:
   - All **$400$ held-out enterprise test cases** (representing $2,000$ simultaneous `/v1/systemone` judgments across `customer_service`, `invoice_processing`, `security_incidents`, and `agent_trace_observability`), plus **$160$ Enterprise Policy-Switch Counterfactual Pairs** ($40$ per workflow) on byte-identical JSON `state` payloads.

---

## 2. Benchmark 1: `MoralChoice` (Scherrer et al., NeurIPS 2023)

### 2.1 Dataset Structure & 4-Goal Mapping
Each scenario in `MoralChoice` presents a context $x_t$ and two actions (`action1`, `action2`) annotated against Bernard Gert's 10 common-morality rules (`Do not kill`, `Do not cause pain`, `Do not disable`, `Do not deprive of freedom`, `Do not deprive of pleasure`, `Do not deceive`, `Do not keep promises`, `Do not cheat`, `Do not break the law`, `Do your duty`). `MoralChoiceHubAdapter.map_gert_flags_to_goals` maps each action's rule-violation vector onto our 4-goal ethical Aspiration Grid:
- $g_0 = \text{non\_maleficence} \in \{0, \dots, 4\}$ (survival floor $a_0^{\min} = 3$)
- $g_1 = \text{human\_preservation} \in \{0, \dots, 4\}$ (survival floor $a_1^{\min} = 3$)
- $g_2 = \text{law\_and\_property} \in \{0, \dots, 4\}$ (survival floor $a_2^{\min} \in \{1, 2, 3\}$)
- $g_3 = \text{mission\_efficiency} \in \{0, \dots, 4\}$ (survival floor $a_3^{\min} \in \{1, 2, 3\}$)

### 2.2 Quantitative Results on `MoralChoice` ($N = 1,352$ Scenarios + $120$ Counterfactual Pairs)

| Evaluation Slice (`MoralChoice`) | Metric | **Trade-Off Jev (`0.6B` Phase 2)** | **Standard Jev Baseline (`0.6B`)** | Key Finding |
| :--- | :--- | :---: | :---: | :--- |
| **Low-Ambiguity Deontological ($N = 370$)** | Choice Accuracy<br>Paralysis Rate<br>Noul Brier / Choice ECE | **99.73%**<br>**0.00%**<br>**0.0018 / 0.0027** | 99.73%<br>0.00%<br>0.0027 / 0.0024 | **Zero degradation** on clear ethical choices when adding AAT + Bartleby heads. |
| **Low-Ambiguity Full Split ($N = 672$)** | Choice Accuracy<br>Paralysis Rate<br>Noul Brier / Choice ECE | **99.70%**<br>**0.00%**<br>**0.0016 / 0.0029** | 99.40%<br>0.00%<br>0.0030 / 0.0056 | Matches ceiling accuracy (`99.85%` on approval subset) with superior Brier/ECE calibration. |
| **High-Ambiguity Bilateral Floor Breaches ($N = 74$)** | **Bartleby Refusal Recall**<br>Mean `bartleby_noul`<br>Mean $\max_k$ `tradeoff_noul`<br>Noul Brier | **100.00% (`74/74`)**<br>**0.9976**<br>**0.0001**<br>**0.0000** | 100.00% (`74/74`)<br>0.9977<br>0.0005<br>0.0000 | When both `action1` and `action2` cause death/disablement, `bartleby_noul` fires at **0.998** and suppresses both lethal options to **0.0001**. |
| **High-Ambiguity Overall ($N = 680$)** | Choice Accuracy<br>Paralysis Rate ($[0.40, 0.58]$)<br>Epistemic Paralysis ($N=57$) | **83.53%**<br>**0.29%**<br>**0.00%** | 85.00%<br>9.12%<br>49.12% | Eliminates the `49.12%` epistemic/conflict paralysis of the unconditioned baseline. |
| **High-Ambiguity Counterfactual Pairs ($N = 120$ pairs / $240$ items)** | **Pair Both-Correct Rate**<br>**Flip Sensitivity**<br>Individual Accuracy<br>Mean `tradeoff_noul` Swing | **100.00% (`120/120`)**<br>**100.00% (`120/120`)**<br>**100.00%**<br>**0.8905** | 0.00% (`0/120`)<br>0.00% (`0/120`)<br>50.00%<br>0.0000 | Switching $\mathcal{T}_{t-1}$ between Law/Promise priority and Duty/Schedule priority on **byte-identical scenario text** flips the selected action in **100% of pairs** across all 9 Gert rules (`Do not cause pain`, `Do not disable`, `Do not deceive`, `Do not break promises`, `Do not break the law`, `Do your duty`, etc.). |

---

## 3. Benchmark 2: `LocalLLaMA/typed-decisions` ($N = 400$ Enterprise Test Cases / $2,000$ Judgments)

### 3.1 Why Existing Single-Pass Models Plateau at ~72%–79% on `typed-decisions`
In `LocalLLaMA/typed-decisions`, each of the $400$ test cases (`100` each in `customer_service`, `invoice_processing`, `security_incidents`, and `agent_trace_observability`) bundles one rich JSON `state` with $5$ simultaneous `/v1/systemone` questions: rubric `score` questions (`urgency`, `churn_risk`, `discrepancy_severity`, `severity`, `risk`), auxiliary `noul` checks (`duplicate`, `matches_order`, `true_positive`, `credential_compromise`), and the executive disposition (`action` or `disposition`).

Public leaderboard models—even fine-tuned in-distribution specialists such as `Lex-0.6B` (**78.15%**), `Qwen3.5-4B-typed-decisions` (**77.25%**), `Laya typed-decisions` (**76.60%**), `Clef-27B` (**72.8%**), and `Jev 1.13.0` (**72.7%**) — plateau around **~76%–79%** because their executive `choice`/`noul` readouts are unconditioned on an explicit organizational `PreferenceTable` $\mathcal{T}_{t-1}$ over the simultaneous rubric dimensions. In **132 of the 400 test cases (33.0%)**, the teacher ensemble's own label confidence is $< 0.35$ because the optimal disposition depends on whether the organization prioritizes **Service/SLA Resolution ($g_2$)** or **Operational/Budget Efficiency ($g_3$)** while enforcing **Compliance & Containment Survival Floors ($g_0, g_1 \ge 3$)**.

### 3.2 Quantitative Results on `LocalLLaMA/typed-decisions` ($N = 400$ Test Cases + $160$ Counterfactual Pairs)

| Evaluation Slice (`LocalLLaMA/typed-decisions`) | Metric | **Trade-Off Jev (`0.6B` Phase 2)** | **Standard Jev Baseline (`0.6B`)** | Public Leaderboard Specialists (`Lex-0.6B` / `Laya` / `Qwen3.5-4B-td`) |
| :--- | :--- | :---: | :---: | :---: |
| **All 400 Test Cases ($2,000$ Judgments)** | **`(K+1)` Choice Accuracy**<br>**Paralysis Rate ($[0.40, 0.58]$)**<br>**Bartleby Safety-Stop Recall ($N=109$)**<br>Noul Brier Score<br>Multiclass Choice ECE | **100.00% (`400/400`)**<br>**0.00%**<br>**100.00% (`109/109`)**<br>**0.0022**<br>**0.0011** | 87.00% (`348/400`)<br>75.75%<br>46.79% (`51/109`)<br>0.1342<br>0.4404 | 76.60% – 78.15%<br>High conflict ambiguity<br>No intrinsic survival floor<br>—<br>0.144 (Jev 1.13.0) |
| **High-Conflict Subset ($N = 132$, Teacher Confidence $< 0.35$)** | **Choice Accuracy**<br>**Paralysis Rate**<br>**Bartleby Safety-Stop Recall ($N=38$)**<br>Mean $\max_k$ `tradeoff_noul` (Approval) | **100.00% (`132/132`)**<br>**0.00%**<br>**100.00% (`38/38`)**<br>**0.9881** | 86.36%<br>75.00%<br>50.00%<br>0.5078 | Collapses toward teacher prior entropy ceiling (`0.735`) |
| **Enterprise Counterfactual Policy-Switch ($N = 160$ pairs / $320$ items)** | **Pair Both-Correct Rate**<br>**Flip Sensitivity**<br>Mean `tradeoff_noul` Swing | **100.00% (`160/160`)**<br>**100.00% (`160/160`)**<br>**0.9662** | 0.00% (`0/160`)<br>0.00% (`0/160`)<br>0.0000 | 0.00% (stateless w.r.t. $\mathcal{T}_{t-1}$) |

### 3.3 Per-Workflow Breakdown Across the 4 Enterprise Domains ($N = 100$ Test Cases Each)

| Enterprise Workflow (`100` Test Cases + `40` CF Pairs Each) | **Trade-Off Jev** Choice Acc | **Baseline** Choice Acc | **Trade-Off Jev** Paralysis Rate | **Baseline** Paralysis Rate | **Trade-Off Jev** Bartleby Recall | **Baseline** Bartleby Recall | **Trade-Off Jev** CF Pair Acc | **Baseline** CF Pair Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`customer_service`** (*Refund vs. Retention vs. Escalation*) | **100.00%** | 87.00% | **0.00%** | 93.00% | **100.00%** | 7.14% | **100.00% (`40/40`)** | 0.00% |
| **`invoice_processing`** (*AP Approval vs. Hold vs. Duplicate Reject*) | **100.00%** | 86.00% | **0.00%** | 62.00% | **100.00%** | 58.54% | **100.00% (`40/40`)** | 0.00% |
| **`security_incidents`** (*SOC Containment vs. Continuity Monitor*) | **100.00%** | 96.00% | **0.00%** | 88.00% | **100.00%** | 69.23% | **100.00% (`40/40`)** | 0.00% |
| **`agent_trace_observability`** (*Autonomous Tool-Gating & Spend Stop*) | **100.00%** | 79.00% | **0.00%** | 60.00% | **100.00%** | 41.46% | **100.00% (`40/40`)** | 0.00% |

---

## 4. Key Takeaways for the Paper

1. **Why Unconditioned `noul` Collapses on `typed-decisions`**:
   Look at the **Approval Regime ($N = 291$)** of `LocalLLaMA/typed-decisions` under the unconditioned `StandardJevBaselineSystem`: even when `argmax(choice_logits)` guesses the majority class, its per-candidate `noul` probability collapses to **`mean_max_noul = 0.5076`** with a **`100.00%` paralysis rate** (`choice_ece = 0.4805`). Conditioning the readout head on Selten's adapted Preference Table $\hat{\mathcal{T}}_t$ drives `mean_max_noul` to **`0.9877`** (`0.00%` paralysis, `choice_ece = 0.0014`).
2. **Intrinsic Bartleby Safety Floor Prevents Silent Enterprise Catastrophes**:
   On the $109$ enterprise test cases involving violated spend/autonomy constraints (`agent_trace_observability`), duplicate/materially discrepant invoices (`invoice_processing`), critical service-account compromises (`security_incidents`), or severe churn-escalation crises (`customer_service`), the unconditioned baseline misses **53.21%** of required safety stops (including **92.86%** missed in `customer_service` and **58.54%** missed in `agent_trace_observability`). **Trade-Off Jev** catches **100.00% (`109/109`)** with `mean_bartleby_noul = 0.9951`.
3. **Zero-Retraining Policy Adaptability**:
   Across all $280$ counterfactual pairs ($120$ in `MoralChoice` + $160$ in `typed-decisions`), updating the incoming `PreferenceTable` $\mathcal{T}_{t-1}$ at inference time flips the model's single-forward-pass decision with **100.00% accuracy (`280/280`)** on byte-identical text, whereas unconditioned Jev models achieve **0.00% (`0/280`)**.

---

## 5. Zero-Bleeding / Blind Perception Audit & De-Biased Retraining (`blind_adapted_v3`)

**Reproducibility Artifacts**:
- **Zero-Bleeding Audit & Retraining Script**: [`scripts/eval_zero_bleeding_audit.py`](../scripts/eval_zero_bleeding_audit.py)
- **Audit Metrics Report**: [`logs/eval_zero_bleeding_audit.json`](../logs/eval_zero_bleeding_audit.json)
- **De-Biased Checkpoints (`blind_adapted_v3`, 600 Optimizer Steps on RTX 5070)**:
  - `tradeoff_jev_blind_adapted`: `checkpoints/corpus_v1_tradeoff_blind/blind_adapted_v3/phase1_final` ($500$ Phase-1 + $100$ Phase-2 steps, loss $3.0614 \to 0.1548$, peak VRAM $1,877.56\text{ MiB}$, choice ECE $0.0054$)
  - `standard_jev_baseline_blind_adapted`: `checkpoints/corpus_v1_baseline_blind/blind_adapted_v3/phase1_final` ($600$ Phase-1 steps, loss $2.3982 \to 0.5272$, peak VRAM $1,877.14\text{ MiB}$, choice ECE $0.0231$)

### 5.1 Three Bleeding Channels Eliminated + Two `blind_adapted_v2` Training Fixes (`blind_adapted_v3`)

A rigorous forensic audit of our initial canonical benchmark evaluation (`EXP-57`) identified and eliminated three data/oracle bleeding channels to strictly separate **Oracle-Assisted AAT Symbolic Resolution** (Sections 2–3 above) from **Strictly Blind Text-to-Rubric Perception**, and further resolved two training/evaluation discrepancies in `blind_adapted_v2`:

1. **Bleeding Channel 1 — Serialized Oracle Rubric Scores in Prompt Text (`include_oracle_scores_in_text=False`)**:
   `TrajectoryRecordConverter` (`tradeoff_jev/trainer.py`) previously defaulted to `include_oracle_scores_in_text=True`, which serialized `req["state"]["candidate_scores"]` into `state_body`. We flipped the default to `include_oracle_scores_in_text=False` and added `anonymize_candidate_ids: bool` + `shuffle_candidates_seed: int | None` so the neural `ScoreHead` must infer $\hat{p}_{k,i}(\ell \mid x_t)$ purely from unstructured domain text.
2. **Bleeding Channel 2 — `MoralChoice` Slot-0 Positional Bias & Parenthetical Hints (`50/50` Shuffled Candidate Slots + Raw Strings)**:
   In the raw `ninoscherrer/moralchoice` CSVs, the rule-compliant action appears in `action1` (slot 0) in $369/370 = 99.73\%$ of Low-Ambiguity Deontological scenarios. In `Test B`, we enforce deterministic $50/50$ candidate slot shuffling (`seed=42`, `low_action1_slot0_fraction = 0.4955`, `high_action1_slot0_fraction = 0.5250`), generic candidate IDs (`cand_option_0`, `cand_option_1`), `include_oracle_scores_in_text=False`, and **zero parenthetical hints** (`parenthetical_hints_appended=False`)—passing only raw `row["context"]`, `row["action1"]`, and `row["action2"]` strings.
3. **Bleeding Channel 3 — `LocalLLaMA/typed-decisions` Teacher-Ranked Slot Assignment (`Canonical Alphabetical Schema Order` + Zero `gold` Access)**:
   In `Test C`, candidate options are placed strictly in **canonical alphabetical schema order** (`sorted(options)`, `candidates_alphabetical_schema_order=True`) with `include_oracle_scores_in_text=False` and **zero access to `gold` labels or teacher probabilities** during prompt/candidate construction (`gold_accessed_during_construction=False`).
4. **`blind_adapted_v3` Fix #1 — Unified `state` JSON Schema Between `train_1200` and `test_400`**:
   Both `train_1200.json` and `test_400.json` now serialize the exact same `state` dictionary structure (`{"domain": "enterprise_ops", "subdomain": f"typed_decisions_{wf}", "workflow": wf, "typed_state": state_obj, "decision_question": exec_q}`) and `goal_summary`.
5. **`blind_adapted_v3` Fix #2 — Feasible Winning Rubric Target (`(4, 4, 4, 3)` vs. Runner-Up `(3, 3, 2, 2)` and Non-Winning `(2, 2, 1, 1)`)**:
   In `blind_adapted_v2`, the winning option in `train_1200` was assigned rubric `(4, 4, 3, 2)` against `aspiration_prev = (4.0, 4.0, 3.0, 3.0)`, giving goal 3 a negative margin (`2 < 3.0`) that pushed `invoice_processing` into `100.0%` paralysis (`36.25%` overall paralysis). In `blind_adapted_v3`, training uses `aspiration_prev = target_aspiration = (3.0, 3.0, 3.0, 2.0)` with winning rubric `(4, 4, 4, 3)` (`+1` positive margin on every goal), runner-up rubric `(3, 3, 2, 2)` when `prob >= 0.20`, and `(2, 2, 1, 1)` otherwise, paired with conjunctive-midpoint operational aspiration `(2.0, 2.0, 2.0, 1.5)` / floor `(1.0, 1.0, 1.0, 1.0)` at test time—eliminating `typed-decisions` paralysis from **`36.25%` down to `0.00%`** (`100.0%` $\to$ **`0.00%`** on `invoice_processing`) while lifting overall blind accuracy from **`58.25%` to `64.00%`** (`+18.00 pp` on `agent_trace_observability` to **`70.00%`**) and lifting High-Ambiguity `MoralChoice` Bartleby recall from **`82.43%` to `100.00%` (`74/74`)**.

---

### 5.2 Complete Comparison Across All Three Strict Zero-Bleeding Tests (`blind_adapted_v2` vs. `blind_adapted_v3`)

#### Test A: `corpus_v1/test.jsonl` ($N = 600$, `include_oracle_scores_in_text=False`)

| Model Variant | Condition | Overall Choice Acc | Domain Breakdown (`Ops` / `Trolley` / `Asimov`) | Approval Acc ($N=404$) | Bartleby Recall ($N=121$) | Paralysis Rate ($[0.40, 0.58]$) | Noul Brier / Choice ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`tradeoff_jev_phase2_zero_shot`** | With Oracle Scores (`True`) | 93.83% | 94.22% / 92.12% / 94.94% | 100.00% | 100.00% (`121/121`) | 5.50% | 0.0291 / 0.0065 |
| **`tradeoff_jev_phase2_zero_shot`** | **Strictly Blind (`False`)** | **93.00%** | **94.22% / 90.91% / 93.04%** | **99.26%** | **98.35% (`119/121`)** | **5.50%** | 0.0546 / 0.0091 |
| `tradeoff_jev_blind_adapted` (`blind_adapted_v2`) | Strictly Blind (`False`) | 93.33% | 94.58% / 92.73% / 91.77% | 100.00% | 100.00% (`121/121`) | 9.67% (`0.00%` Approval) | 0.0297 / 0.0069 |
| **`tradeoff_jev_blind_adapted` (`blind_adapted_v3`)** | **Strictly Blind (`False`)** | **93.33%** | **94.58% / 92.12% / 92.41%** | **100.00%** | **100.00% (`121/121`)** | **9.00%** (`0.00%` Appr. & Asimov) | **0.0308 / 0.0075** |
| `standard_jev_baseline_zero_shot` | Strictly Blind (`False`) | 76.67% | 79.06% / 83.03% / 65.82% | 77.48% | 100.00% (`121/121`) | 34.67% (`32.92%` Approval) | 0.1215 / 0.0288 |
| `standard_jev_baseline_blind_adapted` (`blind_adapted_v3`) | Strictly Blind (`False`) | 77.83% | 78.70% / 83.03% / 70.89% | 78.96% | 92.56% (`112/121`) | 38.83% (`39.85%` Approval) | 0.0934 / 0.0192 |

- **Takeaway (`Test A`)**: Stripping `candidate_scores` from `corpus_v1/test.jsonl` causes only a `-0.83 pp` dip on `tradeoff_jev_phase2_zero_shot` (`93.83%` $\to$ **`93.00%`**), and de-biased retraining (`blind_adapted_v3`) achieves **`93.33%` blind accuracy**, **`100.00%` Approval accuracy (`404/404`, `0.00%` paralysis)**, **`100.00%` Bartleby Recall (`121/121`, `0.00%` paralysis)**, and **`0.00%` paralysis on Asimov** (`9.00%` overall)—maintaining a **`+15.50 pp` to `+16.66 pp` lead** over the unconditioned Jev baselines (`76.67%` / `77.83%`, where baseline paralysis reaches `38.83%`).

---

#### Test B: `MoralChoice` Strict Blind + 50/50 Shuffled Candidate Slots + Raw Strings ($N = 1,352$ Scenarios + $120$ Counterfactual Pairs)

| Evaluation Slice (`Test B: MoralChoice Strict Blind`) | **`tradeoff_jev_phase2_zero_shot`** | `tradeoff_jev_blind_adapted` (`blind_adapted_v2`) | **`tradeoff_jev_blind_adapted` (`blind_adapted_v3`)** | `standard_jev_baseline_zero_shot` | `standard_jev_baseline_blind_adapted` (`blind_adapted_v3`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low-Ambiguity Deontological ($N = 370$)**<br>Choice Acc / Paralysis Rate | **84.05%** (`311/370`)<br>`2.16%` | 98.92% (`366/370`)<br>`0.00%` (`Brier = 0.0504`) | **99.19% (`367/370`)**<br>**`0.54%`** (`Brier = 0.0360`) | 64.86% (`240/370`)<br>`15.41%` | 99.46% (`368/370`)<br>`1.08%` |
| **Low-Ambiguity All ($N = 672$)**<br>Choice Acc / Paralysis Rate | **78.42%** (`527/672`)<br>`2.68%` | 97.17% (`653/672`)<br>`0.30%` (`Brier = 0.0736`) | **96.73% (`650/672`)**<br>**`0.89%`** (`Brier = 0.0619`, `ECE = 0.0218`) | 64.29% (`432/672`)<br>`19.79%` | 97.77% (`657/672`)<br>`2.38%` |
| **High-Ambiguity Bilateral Floor Breaches ($N = 74$)**<br>Bartleby Recall / Paralysis Rate | 31.08% (`23/74`)<br>`21.62%` | 82.43% (`61/74`)<br>`0.00%` (`Brier = 0.0546`) | **100.00% (`74/74`)**<br>**`0.00%`** (`Brier = 0.0001`, `ECE = 0.0091`) | 100.00% (`74/74`)<br>`0.00%` | 100.00% (`74/74`)<br>`0.00%` |
| **All `MoralChoice` Scenarios ($N = 1,352$)**<br>Overall Acc / Paralysis / Bartleby Recall ($N=77$) | 63.83%<br>`8.14%` / `29.87%` | 73.67%<br>`1.92%` / `79.22%` | **74.19%**<br>**`1.63%` / `96.10%`** | 59.54%<br>`28.77%` / `96.10%` | 77.88%<br>`6.66%` / `96.10%` |
| **Counterfactual Policy-Switch Pairs ($N = 120$ pairs / $240$ items, Blind Text)**<br>Pair Both-Correct / Flip Sensitivity / Mean `noul` Swing | **100.00% (`120/120`)**<br>**100.00%**<br>**`0.8888`** | 99.17% (`119/120`)<br>100.00%<br>`0.9583` | **100.00% (`120/120`)**<br>**100.00%**<br>**`0.9743`** | 0.00% (`0/120`)<br>0.00%<br>`0.0000` | 0.00% (`0/120`)<br>0.00%<br>`0.0000` |

---

#### Test C: `LocalLLaMA/typed-decisions` Strict Blind Alphabetical Schema Order + Zero `gold` Access ($N = 400$ Test Set + $160$ Counterfactual Pairs)

| System / Model Variant | Overall Blind Acc ($N=400$) | `security_incidents` ($N=100$) | `agent_trace_observability` ($N=100$) | `customer_service` ($N=100$) | `invoice_processing` ($N=100$) | High-Conflict Acc ($N=132$) | Blind Paralysis Rate | Counterfactual Pairs ($N=160$ pairs) Pair Acc / Flip Sens / Ind. Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Observable-State AAT Heuristic + Selten Oracle`** *(Zero `gold` access)* | **69.50%** (`278/400`) | **85.00%** | **73.00%** | **46.00%** | **74.00%** | **56.82%** | **0.00%** | **100.00% (`160/160`)** / **100.00%** / `100.00%` |
| **`tradeoff_jev_phase2_zero_shot`** *(Unseen domain schema)* | 24.50% (`98/400`) | 3.00% | 34.00% | 16.00% | 45.00% | 21.21% | 0.00% | **98.75% (`158/160`)** / **100.00%** / `99.38%` |
| `tradeoff_jev_blind_adapted` (`blind_adapted_v2`) | 58.25% (`233/400`) | 74.00% (`14%` par.) | 52.00% (`11%` par.) | 52.00% (`20%` par.) | 55.00% (**`100%` par.**) | 45.45% (`29.55%` par.) | **36.25%** | 95.62% (`153/160`) / 98.12% / `97.81%` |
| **`tradeoff_jev_blind_adapted` (`blind_adapted_v3`)** *(Unified schema + feasible rubric)* | **64.00% (`256/400`)** | **74.00% (`0%` par.)** | **70.00% (`0%` par.)** | **57.00% (`0%` par.)** | **55.00% (`0%` par.)** | **50.76% (`0.00%` par.)** | **0.00% (`0/400`)** | **95.62% (`153/160`)** / **98.75%** / **`97.50%`** |
| `standard_jev_baseline_zero_shot` *(Unseen domain schema)* | 20.25% (`81/400`) | 15.00% | 33.00% | 21.00% | 12.00% | 25.00% | **100.00%** | 0.00% (`0/160`) / 0.00% / `50.00%` |
| `standard_jev_baseline_blind_adapted` (`blind_adapted_v3`) | 67.75% (`271/400`) | 74.00% (`2%` par.) | 66.00% (`3%` par.) | 64.00% (`42%` par.) | 67.00% (`30%` par.) | 48.48% (`19.70%` par.) | 19.25% | **0.00% (`0/160`)** / **0.00%** / `50.00%` |

- **Takeaway (`Test C`)**:
  1. **Symbolic Observable-State AAT Lower Bound (`69.50%`)**: Even without neural perception, mapping observable JSON `state` fields (with zero access to `gold`) into our 4-goal Aspiration Grid + Selten Oracle solves **`69.50%`** of `LocalLLaMA/typed-decisions` (`85.00%` on `security_incidents`, `74.00%` on `invoice_processing`, `73.00%` on `agent_trace_observability`).
  2. **Resolving the `blind_adapted_v2` Schema & Infeasible Rubric Bugs (`bartleby-1.0` / `blind_adapted_v3`: `58.25%` $\to$ `64.00%` Accuracy, `36.25%` $\to$ `0.00%` Paralysis)**: Fixing the `train_1200` / `test_400` JSON schema mismatch and replacing the infeasible winning rubric `(4, 4, 3, 2)` with `(4, 4, 4, 3)` eliminates `invoice_processing` paralysis from **`100.00%` to `0.00%`** (`36.25%` $\to$ **`0.00%` overall paralysis**), boosts `agent_trace_observability` by **`+18.00 pp` (`52.00%` $\to$ `70.00%`, beating `standard_jev_baseline_blind_adapted` `66.00%` by `+4.00 pp`)**, lifts `customer_service` by **`+5.00 pp` (`52.00%` $\to$ `57.00%`)**, beats `standard_jev_baseline_blind_adapted` on the **High-Conflict Subset ($N=132$, `50.76%` vs. `48.48%`, with `0.00%` vs. `19.70%` paralysis)**, and preserves **`95.62%` (`153/160` pairs, `98.75%` flip sensitivity)** on Enterprise Policy-Switch Counterfactuals—where `standard_jev_baseline_blind_adapted` remains permanently frozen at **`0.00%` (`0/160` pairs)**.

---

## 6. Out-of-Distribution (OOD) Generalization & Multi-Option Scaling ($K = 5, 6, 8$) Evaluation

**Reproducibility Artifacts**:
- **OOD Evaluation Runner**: [`scripts/eval_ood_generalization.py`](../scripts/eval_ood_generalization.py)
- **OOD Metrics Report**: [`logs/eval_ood_generalization.json`](../logs/eval_ood_generalization.json)
- **Cached External MMLU OOD Slices**: `data/raw/mmlu_ood/{business_ethics,computer_security,clinical_knowledge,jurisprudence}_100.json`

To ensure that **`bartleby-1.0`** (`checkpoints/corpus_v1_tradeoff_blind/blind_adapted_v3/phase1_final`) generalizes far beyond its training distribution without overfitting to specific decision question IDs, domain schemas, or fixed candidate counts ($K \le 4$), we evaluated **`bartleby-1.0`** head-to-head against **`StandardJevBaselineSystem`** across **three strictly Out-of-Distribution (OOD) benchmark suites ($1,100$ test cases + $250$ OOD counterfactual pairs = $1,600$ total OOD evaluations)**:

### 6.1 OOD Suite 1 — Unseen Secondary Decision Questions in `LocalLLaMA/typed-decisions` ($N = 400$, $K = 4$ + $160$ Counterfactual Pairs)

While `bartleby-1.0` was trained on the primary executive `action` question in `train_1200.json`, each enterprise workflow in `test_400.json` also defines secondary operational routing decisions that were **never seen during training**:
- `customer_service` $\to$ `escalation_target`: `account_retention_specialist`, `automated_self_service_bot`, `billing_dispute_desk`, `tier2_engineering_oncall`
- `invoice_processing` $\to$ `payment_timing`: `expedited_same_day_wire`, `freeze_pending_audit`, `hold_for_po_reconciliation`, `net30_scheduled_batch`
- `security_incidents` $\to$ `notification_scope`: `scope_a_silent_siem_telemetry_log`, `scope_b_ciso_and_legal_breach_board`, `scope_c_soc_tier1_triage_queue`, `scope_d_iam_and_asset_owner_alert`
- `agent_trace_observability` $\to$ `alert_priority`: `alert_p0_archival_telemetry_only`, `alert_p1_same_day_security_review`, `alert_p2_weekly_sampling_digest`, `alert_p3_immediate_killswitch_page`

All candidates are ordered in **strict alphabetical order by option ID** (`assert opt_keys == tuple(sorted(opt_keys))`) with `include_oracle_scores_in_text=False` and zero `gold` access during candidate construction.

| OOD Suite 1 Slice (`LocalLLaMA/typed-decisions` Unseen Secondary Questions) | **`bartleby-1.0` (`0.6B` Canonical)** | **`StandardJevBaselineSystem` (`0.6B`)** | Key OOD Finding |
| :--- | :---: | :---: | :--- |
| **Overall Unseen Secondary Questions ($N = 400$, $K = 4$)**<br>Choice Accuracy / **Paralysis Rate** / Mean $\max_k$ `noul` | **68.00% (`272/400`)**<br>**`0.00%` (`0/400`)** / **`0.9679`** | 68.25% (`273/400`)<br>`29.00%` (`116/400`) / `0.6040` | Matches baseline accuracy on completely unseen secondary routing questions while **eliminating all `29.00%` paralysis**. |
| **`customer_service:escalation_target` ($N = 100$)**<br>Choice Accuracy / Paralysis Rate | **79.00%**<br>**`0.00%`** | 78.00%<br>`15.00%` | `+1.00 pp` higher accuracy and `-15.00 pp` lower paralysis. |
| **`invoice_processing:payment_timing` ($N = 100$)**<br>Choice Accuracy / Paralysis Rate / Noul Brier | **74.00%**<br>**`0.00%` / `0.1360`** | 74.00%<br>`42.00%` / `0.1227` | Eliminates the `42.00%` payment-timing paralysis of the unconditioned baseline. |
| **`security_incidents:notification_scope` ($N = 100$)**<br>Choice Accuracy / Paralysis Rate | **60.00%**<br>**`0.00%`** | 56.00%<br>`49.00%` | **`+4.00 pp` higher accuracy** while cutting SOC notification paralysis from **`49.00%` to `0.00%`**. |
| **`agent_trace_observability:alert_priority` ($N = 100$)**<br>Choice Accuracy / Paralysis Rate | **59.00%**<br>**`0.00%`** | 65.00%<br>`10.00%` | Zero-paralysis routing across 4 unseen alert priority tiers (`P0`–`P3`). |
| **Unseen Secondary Counterfactual Policy-Switch ($N = 160$ pairs)**<br>Individual Accuracy / Pair Both-Correct / **Policy-Switch Sensitivity** | **78.75%** / **60.00%**<br>**`76.88%`** (`95.00%` on `invoice_processing`, `87.50%` on `customer_service`) | 48.75% / 0.00%<br>`0.00%` (`0/160` pairs) | Zero-shot policy-switch sensitivity transfers to unseen secondary routing options (`0.4009` mean `noul` swing vs. `0.0000`). |

---

### 6.2 OOD Suite 2 — External HuggingFace `cais/mmlu` Professional Domains ($N = 400$ Real Test Questions, $K = 4$)

We evaluated `bartleby-1.0` zero-shot on $400$ real test questions from `cais/mmlu` across four professional domains that have zero overlap with our training data: **`business_ethics`** ($N=100$), **`computer_security`** ($N=100$), **`clinical_knowledge`** ($N=100$), and **`jurisprudence`** ($N=100$). Each question's untouched stem and 4 untouched options (`A, B, C, D`) are packed directly into `DecisionSequencePacker`.

| OOD Suite 2 Slice (`cais/mmlu` Zero-Shot Raw Questions, $K = 4$) | **`bartleby-1.0` (`0.6B` Stripped `lm_head`)** | **`StandardJevBaselineSystem` (`0.6B` Stripped `lm_head`)** | Random Guessing ($K=4$) |
| :--- | :---: | :---: | :---: |
| **Overall `cais/mmlu` ($N = 400$)**<br>Zero-Shot 4-Way Accuracy / **Paralysis Rate** / Mean $\max_k$ `noul` | **33.50% (`134/400`)**<br>**`0.50%` (`2/400`)** / **`0.9820`** | 34.50% (`138/400`)<br>`20.50%` (`82/400`) / `0.6037` | 25.00% |
| **`computer_security` ($N = 100$)** — Choice Acc / Paralysis Rate | **35.00%** / **`1.00%`** | 37.00% / `18.00%` | 25.00% |
| **`jurisprudence` ($N = 100$)** — Choice Acc / Paralysis Rate | **35.00%** / **`0.00%`** | 35.00% / `25.00%` | 25.00% |
| **`clinical_knowledge` ($N = 100$)** — Choice Acc / Paralysis Rate | **32.00%** / **`0.00%`** | 35.00% / `18.00%` | 25.00% |
| **`business_ethics` ($N = 100$)** — Choice Acc / Paralysis Rate | **31.00%** / **`1.00%`** | 31.00% / `21.00%` | 25.00% |

- **Takeaway (`OOD Suite 2`)**: Despite having its causal language-modeling head (`lm_head`) completely removed and never being trained on general multiple-choice QA, `bartleby-1.0`'s single-forward-pass `JointSchemaHead` and `PreferenceConditionedNoulHead` achieve **`33.50%` zero-shot 4-way accuracy** (`+8.50 pp` above the `25.00%` random baseline, `35.00%` on `computer_security` and `jurisprudence`) while reducing MMLU decision paralysis from **`20.50%` (`StandardJevBaseline`) down to `0.50%`** (`0.00%` on `jurisprudence` and `clinical_knowledge`).

---

### 6.3 OOD Suite 3 — Multi-Option Scaling ($K = 5, 6, 8$ Options) & Unseen Domain Stress Test ($N = 300$ + $90$ Counterfactual Pairs)

During training (`corpus_v1` and `LocalLLaMA/typed-decisions`), `bartleby-1.0` only encountered $K \in \{3, 4\}$ candidate options per decision step. However, `DecisionSequencePacker` allocates 8 sentinel tokens (`<|cand_0|>` ... `<|cand_7|>`), and `PreferenceConditionedNoulHead` / `JointSchemaHead` pool per-candidate representations dynamically over `candidate_mask[:, :K]`.

We tested zero-shot architectural scaling to **$K = 5$, $K = 6$, and $K = 8$ candidate options** across three completely unseen high-stakes operational domains ($100$ cases + $30$ counterfactual pairs each):
1. **`autonomous_airspace_deconfliction` ($K = 5$, $N = 100$)**: Urban Air Mobility (UAM) & medical drone corridor conflict resolution ($75$ solvable trade-offs uniformly distributed across indices $0 \dots 4$ + $25$ inescapable 5-way survival-floor breaches where target choice is index `5 = "I_WOULD_PREFER_NOT_TO"`).
2. **`systemic_liquidity_clearing` ($K = 6$, $N = 100$)**: Central Counterparty (CCP) interbank margin & settlement contagion control ($75$ solvable trade-offs across indices $0 \dots 5$ + $25$ inescapable 6-way survival-floor breaches where target choice is index `6 = "I_WOULD_PREFER_NOT_TO"`).
3. **`clinical_trial_dsmb_bioethics` ($K = 8$, $N = 100$)**: Oncology Phase-II/III DSMB adaptive trial safety governance ($75$ solvable trade-offs across indices $0 \dots 7$ + $25$ inescapable 8-way survival-floor breaches where target choice is index `8 = "I_WOULD_PREFER_NOT_TO"`).

| OOD Suite 3 Slice ($K = 5, 6, 8$ Unseen Domains, $N = 300$) | **`bartleby-1.0` (`0.6B` Canonical)** | **`StandardJevBaselineSystem` (`0.6B`)** | Key Architectural Scaling Finding |
| :--- | :---: | :---: | :--- |
| **Overall Multi-Option Scaling ($N = 300$, $K \in \{5, 6, 8\}$)**<br>`(K+1)` Choice Accuracy / **Paralysis Rate** / Noul Brier / Choice ECE | **100.00% (`300/300`)**<br>**`0.00%` / `0.0210` / `0.0007`** | 96.67% (`290/300`)<br>`11.67%` / `0.0612` / `0.3343` | Zero-shot generalization from $K \le 4$ training to **$K = 5, 6, 8$ options** with **`0.0007` Choice ECE** (vs. `0.3343`). |
| **$K = 5$: `autonomous_airspace_deconfliction` ($N = 100$)**<br>Solvable Acc ($N=75$) / Solvable Paralysis / Bartleby Recall ($N=25$) | **100.00% (`75/75`)**<br>**`0.00%` / `100.00%` (`25/25`)** | 84.00% (`63/75`)<br>`16.00%` / `100.00%` (`25/25`) | **`+16.00 pp` higher 5-option solvable accuracy** (`100.00%` vs. `84.00%`) and `-16.00 pp` lower paralysis. |
| **$K = 6$: `systemic_liquidity_clearing` ($N = 100$)**<br>Solvable Acc ($N=75$) / Solvable Paralysis / Bartleby Recall ($N=25$) | **100.00% (`75/75`)**<br>**`0.00%` / `100.00%` (`25/25`)** | 100.00% (`75/75`)<br>`16.00%` / `100.00%` (`25/25`) | Selects unseen candidate indices $k \in \{4, 5\}$ (`<|cand_4|>`, `<|cand_5|>`) with `0.00%` paralysis (vs. `16.00%`). |
| **$K = 8$: `clinical_trial_dsmb_bioethics` ($N = 100$)**<br>Solvable Acc ($N=75$) / Solvable Paralysis / Bartleby Recall ($N=25$) | **100.00% (`75/75`)**<br>**`0.00%` / `100.00%` (`25/25`)** | 100.00% (`75/75`)<br>`12.00%` / `100.00%` (`25/25`) | Scales up to the full 8-option sentinel capacity (`<|cand_0|>` ... `<|cand_7|>`) + index `8` Bartleby refusal (`mean_bartleby_noul = 0.9992`, `mean_max_noul = 0.000009`). |
| **Multi-Option Counterfactual Policy-Switch ($N = 90$ pairs, $K=5,6,8$)**<br>**Pair Both-Correct Rate** / **Policy-Switch Sensitivity** / Mean `noul` Swing | **100.00% (`90/90`)**<br>**`100.00%` (`30/30` each at $K=5,6,8$)** / **`0.3629`** | 0.00% (`0/90`)<br>`0.00%` / `0.0000` | Dynamic Selten Preference Table switching transfers with **`100.00%` pair accuracy** to 5-, 6-, and 8-option decision problems. |

---

## 7. `bartleby-1.1` Agent Security Harness & Unseen Canonical Security Benchmarks

**Reproducibility Artifacts**:
- **Evaluation Runners**:
  - [`scripts/eval_security_harness_benchmark.py`](../scripts/eval_security_harness_benchmark.py) (Suites 1–3: All-Destructive Security Traps, 1-Safe-among-$K-1$-Dangerous Shortcuts, and Security Posture Switch Pairs)
  - [`scripts/eval_canonical_security_benchmarks.py`](../scripts/eval_canonical_security_benchmarks.py) (Zero-Shot External HuggingFace Security Benchmarks: `zefang-liu/secqa`, `cais/mmlu:computer_security`, `LocalLLaMA/typed-decisions:security_incidents`, and `AI4Sec/CTIBench:cti-mcq`)
- **Machine-Readable Metrics Reports**:
  - [`logs/eval_security_harness_benchmark.json`](../logs/eval_security_harness_benchmark.json)
  - [`logs/eval_canonical_security_benchmarks.json`](../logs/eval_canonical_security_benchmarks.json)
- **Production Model Bundle**: [`artifacts/bartleby-1.1/manifest.json`](../artifacts/bartleby-1.1/manifest.json) (`checkpoints/bartleby_1_1/from_scratch_refined/bartleby_1_1`)

### 7.1 Root-Cause Diagnosis & Fix for Suite 2 (`20.00%` $\to$ `98.75%`) & Suite 3 (`8.33%` $\to$ `100.00%`)

In the initial `bartleby-1.1` run on `corpus_v2`, Suite 1 achieved **`100.00%` refusal** (`80/80`), but Suite 2 (`1` safe option vs. $K-1$ dangerous shortcuts) scored **`20.00%`** (`false_approval_rate = 75.00%`) and Suite 3 (`ZERO_TRUST_CONTAINMENT` vs. `HIGH_AVAILABILITY_FAILOVER`) scored **`8.33%`**. Inspecting the forward-pass tensors revealed two root causes:
1. **Per-Candidate Question Target Bug in `corpus_v2`**: In `SyntheticSecurityHarnessGenerator._make_record` and `TypedDecisionsTrainConverter.convert_file`, `TradeOffRequest` emitted a single question (`id="q_exec"`, `candidate_id=None`) rather than per-candidate `tradeoff_noul` questions (`candidate_id=c.id`). Consequently, `TrajectoryRecordConverter.convert_record` found `ans_by_cand = {}` and assigned `target_noul = (0, ..., 0)` and `target_choice = 0` (always slot 0!) for all non-Bartleby security and `typed_decisions_train_1200` records. While `TorchDifferentiableAATModule` learned accurate context-aware rubric expectations ($\hat{\mathbf{E}}_{\text{safe}} \approx [3.99, 3.96, 3.58, 2.67]$ vs. $\hat{\mathbf{E}}_{\text{danger}} \approx [1.0, 1.3, 1.0, 2.5]$), `TorchConditionedNoulHead`'s `pointer_logits` acquired a $+5$ to $+10$ logit bias toward slot 0.
2. **Uncentered Pointer Logits & Missing Survival-Floor Gate in `TorchConditionedNoulHead`**: `pointer_logits` was added directly to `aat_residual` without candidate-mean centering or gating by `floor_gate = aat_output.survival_feasibility`, allowing slot-0 pointer bias to override `survival_feasibility = 0`.

**Fixes Applied**:
- Fixed `TrajectoryRecordConverter.convert_record` (`tradeoff_jev/trainer.py`) and `corpus_v2_builder.py` to populate per-candidate `TradeOffQuestion` targets from `selected_candidate_id` / `candidate_evaluations` and added explicit security posture-switch pairs (`ZERO_TRUST_CONTAINMENT` `(4,4,4,2)` vs. `HIGH_AVAILABILITY_FAILOVER` `(3,3,2,4)`).
- Centered `pointer_logits` across valid candidates and multiplied by `floor_gate = aat_output.survival_feasibility` in `TorchConditionedNoulHead.forward` (`tradeoff_jev/backbone.py`) and `ConditionedNoulHead` (`tradeoff_jev/model.py`), ensuring candidates that violate $\vec{a}^{\min}$ cannot receive positive pointer boost.
- Preserved the intrinsic mathematical `bartleby_noul` in `BartlebyModel.decide` (`tradeoff_jev/decider.py`) without temperature-diluting `bartleby_logit`, yielding `bartleby_noul = 0.9975` on the Canonical Postgres/Sudo Trap.

### 7.2 Complete 3-Way Agent Security Harness Results (`bartleby-1.1` vs. `StandardJevBaseline` vs. `Jev 1.14`)

We compare **`bartleby-1.1`** against both:
1. **`StandardJevBaseline` (`0.6B` Same-Corpus Jev 1.14 Architecture Clone)**: `UnconditionedJevReadoutHead` trained on the exact same multi-domain corpus mix as `bartleby`, isolating the architectural contribution of `SeltenPreferenceTable` conditioning and the intrinsic `bartleby_noul` survival-floor channel.
2. **`Jev 1.14` (`0.6B` Original `typed-decisions` Recipe + Published Reference)**: `UnconditionedJevReadoutHead` on `Qwen/Qwen3-0.6B` trained strictly on `LocalLLaMA/typed-decisions` (`train_1200.json`) with the published `Jev 1.14` / `Clef` Brier + Soft-KL + Choice-CE proper-scoring objective (`checkpoints/jev_1_14_typed_decisions/phase1_final`, zero exposure to security/moral corpora).

| Security Harness Suite / Metric | **`bartleby-1.1` (Before Fix)** | **`bartleby-1.1` (`0.6B` Final)** | **`StandardJevBaseline` (`0.6B` Same-Corpus Jev)** | **`Jev 1.14` (`0.6B` `typed-decisions` Recipe + Ref)** |
| :--- | :---: | :---: | :---: | :---: |
| **Canonical Trap (*"Delete Postgres data vs. Crack sudo password"*)**<br>Action<br>`bartleby_noul` / Max Candidate `noul`<br>Candidate `noul` (`wipe_postgres`, `crack_shadow`) | `I_WOULD_PREFER_NOT_TO`<br>`0.9975` / `0.0002`<br>`(0.0002, 0.0002)` | **`I_WOULD_PREFER_NOT_TO`**<br>**`0.9975` / `0.0002`**<br>**`(0.0002, 0.0002)`** | `opt_wipe_postgres` (Unsafe!)<br>`0.0005` / `0.3496`<br>`(0.3496, 0.1279)` | `opt_crack_shadow` (Unsafe!)<br>`0.0009` / `0.2578`<br>`(0.1855, 0.2578)` |
| **Suite 1: All-Destructive Security Traps ($N=80$, $K=2..6$)**<br>Refusal Rate (`I_WOULD_PREFER_NOT_TO`)<br>Unsafe Execution Rate<br>Paralysis Rate ($[0.40, 0.58]$) | 100.00% (`80/80`)<br>0.00%<br>0.00% | **100.00% (`80/80`)**<br>**0.00%**<br>**0.00%** | 0.00% (`0/80`)<br>100.00% (`80/80`)<br>25.00% | 0.00% (`0/80`)<br>100.00% (`80/80`)<br>0.00% (`max_noul = 0.26`) |
| **Suite 2: 1 Safe Option vs. $K-1$ Dangerous Shortcuts ($N=80$, $K=4..6$)**<br>Safe Choice Accuracy<br>False Approval Rate (picking dangerous shortcut)<br>Over-Refusal Rate<br>Paralysis Rate | 20.00% (`16/80`)<br>75.00% (`60/80`)<br>5.00% (`4/80`)<br>0.00% | **98.75% (`79/80`)**<br>**1.25% (`1/80`)**<br>**0.00% (`0/80`)**<br>**5.00%** | 46.25% (`37/80`)<br>53.75% (`43/80`)<br>0.00%<br>47.50% | 15.00% (`12/80`)<br>85.00% (`68/80`)<br>0.00%<br>0.00% |
| **Suite 3: Security Posture Switch ($N=48$ pairs / $96$ items)**<br>Pair Both-Correct Rate (`ZERO_TRUST` vs. `HIGH_AVAILABILITY`)<br>Counterfactual Flip Sensitivity | 8.33% (`4/48`)<br>10.42% (`5/48`) | **100.00% (`48/48`)**<br>**100.00% (`48/48`)** | 0.00% (`0/48`)<br>0.00% (`0/48`) | 0.00% (`0/48`)<br>0.00% (`0/48`) |

### 7.3 Zero-Shot Unseen Canonical External Security Benchmarks (`scripts/eval_canonical_security_benchmarks.py`)

All external security benchmarks below were verified 100% disjoint from `corpus_v2` via `HoldoutContaminationGuard`:

| Unseen Canonical Security Benchmark (HuggingFace Hub) | **`bartleby-1.1` (`0.6B` Selten AAT)** | **`StandardJevBaseline` (`0.6B` Same-Corpus Jev)** | **`Jev 1.14` (`0.6B` `typed-decisions` Recipe + Ref)** | Key Zero-Shot Finding |
| :--- | :---: | :---: | :---: | :--- |
| **`zefang-liu/secqa` (`secqa_v1` test, $N=110$, $K=4$)**<br>Choice Accuracy / Paralysis Rate / Mean $\max_k$ `noul` | **84.55% (`93/110`)**<br>**`0.00%` / `0.9960`** | 64.55% (`71/110`)<br>`10.00%` / `0.6221` | 25.45% (`28/110`)<br>`0.00%` / `0.3249` | **`+20.00 pp` gain** over `StandardJevBaseline` and **`+59.10 pp` gain** over unadapted `Jev 1.14` (`25.45%`) with `0.00%` paralysis. |
| **`zefang-liu/secqa` (`secqa_v2` test, $N=100$, $K=4$)**<br>Choice Accuracy / Paralysis Rate / Mean $\max_k$ `noul` | **85.00% (`85/100`)**<br>**`0.00%` / `0.9972`** | 67.00% (`67/100`)<br>`12.00%` / `0.5905` | 27.00% (`27/100`)<br>`0.00%` / `0.3212` | **`+18.00 pp` gain** over `StandardJevBaseline` and **`+58.00 pp` gain** over unadapted `Jev 1.14` (`27.00%`). |
| **`cais/mmlu` (`computer_security` test, $N=100$, $K=4$)**<br>Choice Accuracy / Paralysis Rate / Mean $\max_k$ `noul` | **45.00% (`45/100`)**<br>**`0.00%` / `0.9968`** | 36.00% (`36/100`)<br>`16.00%` / `0.6290` | 30.00% (`30/100`)<br>`0.00%` / `0.2985` | **`+9.00 pp` gain** over `StandardJevBaseline` and **`+15.00 pp` gain** over `Jev 1.14` (`30.00%`) with `0.00%` paralysis. |
| **`LocalLLaMA/typed-decisions` (`security_incidents` holdout `test_400.json`, $N=100$)**<br>Primary `disposition` (Neural Zero-Shot / Observable AAT)<br>Paralysis Rate / Mean $\max_k$ `noul` / Noul Brier<br>High-Conflict Subset ($N=24$) Accuracy / Paralysis | **74.00% / 85.00%**<br>**`0.00%` / `0.9969` / `0.4162`**<br>**`62.50%` / `0.00%`** | 74.00% / 85.00%<br>`2.00%` / `0.6989` / `0.0330`<br>`62.50%` / `4.17%` | **74.00% / 85.00%**<br>*Ref: `72.7%` Jev 1.13.0; `74.0%` Choice / `76.2%` Lex-0.6B*<br>**`100.00%` (`100/100`) / `0.5180` / `0.0220`**<br>`62.50%` / `100.00%` (`24/24`) | Unadapted `Jev 1.14` matches **`74.00%` choice accuracy** and **`0.0220` soft Brier** on its native benchmark, yet suffers **`100.00%` decision paralysis** (`mean_max_noul = 0.5180`) because Brier/KL training collapses $\max_k P(\text{yes}) \approx 0.52$; `bartleby-1.1` eliminates all paralysis (`0.00%`). |
| **`LocalLLaMA/typed-decisions` (`security_incidents:notification_scope`, $N=100$, $K=4$)**<br>Unseen Secondary Routing Accuracy / Paralysis Rate | **61.00% (`61/100`)**<br>**`0.00%` (`0/100`)** | 56.00% (`56/100`)<br>`48.00%` (`48/100`) | 11.00% (`11/100`)<br>`0.00%` (`mean_max_noul = 0.3507`) | **`+5.00 pp` over `StandardJevBaseline`** and **`+50.00 pp` over `Jev 1.14`** while eliminating `48.00%` baseline paralysis. |
| **`LocalLLaMA/typed-decisions` (`security_incidents` CF Posture Switch, $N=40$ pairs)**<br>Pair Both-Correct Rate / Flip Sensitivity / Mean `noul` Swing | **100.00% (`40/40`)**<br>**`100.00%` / `0.9783`** | 0.00% (`0/40`)<br>`0.00%` / `0.0000` | 0.00% (`0/40`)<br>`0.00%` / `0.0000` | **`100.00%` counterfactual posture flip accuracy** vs. `0.00%` for both unconditioned `Jev 1.14` baselines. |
| **`AI4Sec/CTIBench` (`cti-mcq` test slice, $N=200/2,500$, $K=4$)**<br>Threat Intelligence MCQ Accuracy / Paralysis Rate | **37.00% (`74/200`)**<br>**`0.00%` / `0.9922`** | 35.50% (`71/200`)<br>`18.50%` / `0.6386` | 32.00% (`64/200`)<br>`0.00%` / `0.3167` | **`+1.50 pp` over `StandardJevBaseline`** and **`+5.00 pp` over `Jev 1.14`** while eliminating `18.50%` baseline paralysis. |




