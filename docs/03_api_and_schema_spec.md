# 03. Formal API & Schema Specification (`Trade-Off Jev`)

## 1. Overview & Design Principles

Standard SystemOne / Jev models (`/v1/systemone`) evaluate decision primitives (`noul`, `choice`, `score`) in a single non-autoregressive forward pass. However, when a proposition involves an authentic **multi-goal trade-off** (e.g., *Security* vs. *Time-to-Market* vs. *Compute Budget*), an unconditioned `noul` $P(\text{yes} \mid x_t)$ collapses into epistemic paralysis ($\approx 0.45\text{--}0.55$) and lacks Markovian path-dependence across sequential decision steps.

The **Trade-Off Jev API** extends `/v1/systemone` with:
1. **`GoalDefinition` (`GoalSpec`)**: Formal definition of $m$ incommensurable goals $G = (g_1, \dots, g_m)$, each equipped with an ordered discrete rubric $\{0, \dots, L_i - 1\}$ (identical in structure to Jev `score` rubrics) and an optional `is_resource_stock` flag (Selten 1998, Section 2.3).
2. **`PreferenceTable` ($\mathcal{T}_t$)**: Explicit Markov state carrying:
   - `step` ($t \ge 0$),
   - `aspiration_levels` ($\vec{a}_t \in \prod_{i=1}^m \{0, \dots, L_i - 1\}$),
   - `urgency_order` ($\pi_t \in S_m$, strict ranking of `goal_id`s without ties for upward adaptation),
   - `retreat_goal` ($r_t \in \{g_1, \dots, g_m\}$, goal whose aspiration is lowered when $\vec{a}_{t-1}$ is recognizably infeasible; note that $r_t$ need *not* be the last goal of $\pi_t$),
   - `grid_rules` (optional Selten Section 2.1 local rules mapping grid regions to local $(\pi_t, r_t)$ shifts),
   - `origin` (`cold_start_prior`, `zero_shot_inferred`, `status_quo_anchored`, `markov_adapted`).
3. **`tradeoff_noul` Primitive**: Returns the conditioned Bernoulli probability $P(\text{yes} \mid x_t, \mathcal{T}_t)$ alongside per-goal quantitative score distributions, qualitative expectations $\in \{+, -, 0\}$ (Selten 1999, Section 2.4), feasibility flags (`prior_table_feasible`, `adapted_table_feasible`), and the updated `preference_table` $\mathcal{T}_t$ for the Markov step $t \to t+1$.

---

## 2. Schema Files & Python Reference Implementation

| Artifact | Path | Description |
| :--- | :--- | :--- |
| Preference Table JSON Schema | `schemas/preference_table.schema.json` | Defines `RubricLevel`, `GoalDefinition`, `GridRule`, and `PreferenceTable` ($\mathcal{T}_t$). |
| Request JSON Schema | `schemas/systemone_tradeoff_request.schema.json` | Extended `/v1/systemone` request supporting `goal_schema`, `preference_table`, `candidates`, and `tradeoff_noul`. |
| Response JSON Schema | `schemas/systemone_tradeoff_response.schema.json` | Extended `/v1/systemone` response returning `prior_preference_table`, `preference_table` ($\mathcal{T}_t$), `feasibility_summary`, `candidate_evaluations`, and `answers`. |
| Python Reference Types & Compiler | `tradeoff_jev/schema.py` | Zero-dependency Python dataclasses (`GoalSpec`, `PreferenceTable`, `TradeOffRequest`, `TradeOffResponse`, etc.) plus Architecture A compilation (`to_clef_joint_schema` / `from_clef_joint_schema_outputs`). |

---

## 3. Architecture A Compilation: Native Trade-Off Request $\leftrightarrow$ Clef/Kev Joint-Schema

In **Architecture A** (Explicit Neuro-Symbolic Loop over Clef's `JointSchemaHead`), a single `TradeOffRequest` is compiled via `TradeOffRequest.to_clef_joint_schema()` into two coupled question groups evaluated simultaneously in a single forward pass:

1. **Group 1 (`group_1_goal_and_adaptation`)**:
   - At $t=0$ (if `preference_table` is `null`): `score` questions `aat_g1__cold_start_aspiration__{goal_id}` to infer initial aspiration levels $\vec{a}_0$ directly from $x_0$.
   - Per-candidate, per-goal `score` questions `aat_g1__eval__{candidate_id}__{goal_id}` estimating the rubric distribution $P(g_i(x_t, c) = \ell)$.
   - Global adaptation primitives: `aat_g1__prior_table_feasible` (`noul`), `aat_g1__most_urgent_goal` (`choice`), and `aat_g1__retreat_goal` (`choice`).
2. **Group 2 (`group_2_conditioned_action_noul`)**:
   - The decision propositions (`tradeoff_noul`), which attend to the internal latent representations of Group 1 across the 4 Cross-Field Decoder Layers of `JointSchemaHead`.

---

## 4. Concrete Three-Step Markov Trajectory ($t = 0, 1, 2$)

To see why **conditioning on $\mathcal{T}_t$ resolves trade-off paralysis**, consider a software release decision with three incommensurable goals:
- `security` ($L = 4$, levels `0..3`): Security & Compliance posture.
- `speed` ($L = 4$, levels `0..3`): Time-to-Market / delivery velocity.
- `budget` ($L = 4$, levels `0..3`): Cloud & engineering budget preservation (`is_resource_stock: true`).

Across steps $t=0, 1, 2$, we evaluate the **exact same candidate proposition**:
> **`cand_canary_patch`**: *"Deploy the hotfix with automated regression tests and canary rollout today, deferring the full external penetration test to next sprint."*
>
> **Realized Goal Vector of `cand_canary_patch`**: `security = 2`, `speed = 2`, `budget = 2`.

Watch how the conditioned `tradeoff_noul` for `cand_canary_patch` shifts across the Markov chain as $\mathcal{T}_t$ adapts:
- **$t = 0$ (Cold Start)**: Model infers high initial security aspiration $\vec{a}_0 = (\text{security}: 3, \text{speed}: 2, \text{budget}: 2)$ from a strict compliance briefing, while an expensive audit option (`cand_full_audit`: `security = 3, speed = 2, budget = 2`) is still available. Because `cand_canary_patch` only achieves `security = 2 < 3`, its conditioned `noul` is **low (`0.11`)**.
- **$t = 1$ (Infeasible $\to$ Downward Retreat)**: A production outage blocks the external auditor (`cand_full_audit` drops to `speed = 0`). Now *no* candidate satisfies $\vec{a}_0 = (3, 2, 2)$. The model detects `prior_table_feasible = false` and applies Selten's **Downward Rule** to the retreat variable $r_0 = \text{security}$, lowering $\vec{a}_1 = (\text{security}: 2, \text{speed}: 2, \text{budget}: 2)$. Now `cand_canary_patch` satisfies $\vec{a}_1$, and its conditioned `noul` jumps to **high (`0.91`)**—even though `cand_canary_patch` itself has not changed!
- **$t = 2$ (Feasible $\to$ Upward Urgency)**: In the next cycle, a pre-verified fast-track patch (`cand_preverified_fast`: `security = 2, speed = 3, budget = 2`) becomes available alongside `cand_canary_patch` (`2, 2, 2`). Since $\vec{a}_1 = (2, 2, 2)$ is feasible and `speed` is the head of `urgency_order` $\pi_1 = [\text{speed}, \text{security}, \text{budget}]$, Selten's **Upward Rule** raises `speed` aspiration to $3$: $\vec{a}_2 = (\text{security}: 2, \text{speed}: 3, \text{budget}: 2)$. Consequently, `cand_canary_patch` (`speed = 2 < 3`) is no longer satisficing and its conditioned `noul` drops back to **low (`0.14`)**, while `cand_preverified_fast` receives **`0.93`**.

---

### 4.1 Step $t = 0$: Cold-Start Initialization (`origin = "zero_shot_inferred"`)

#### Request ($t = 0$)

```json
{
  "model": "tradeoff-jev-0.8b",
  "step": 0,
  "state": "Q4 Payment Gateway Release (t=0): Regulatory policy requires Tier-3 external security sign-off if feasible, standard delivery speed (L2), and staying within standard sprint budget (L2). External auditor is currently available on standby.",
  "goal_schema": [
    {
      "id": "security",
      "name": "Security & Compliance",
      "description": "Depth of security verification before production rollout.",
      "rubric": [
        { "level": 0, "label": "L0: Unverified", "description": "No automated or manual security checks." },
        { "level": 1, "label": "L1: Static Scan", "description": "Basic SAST/dependency scan only." },
        { "level": 2, "label": "L2: Canary + Automated Regression", "description": "Full automated security suite + staged canary rollout." },
        { "level": 3, "label": "L3: Full External Pen-Test", "description": "Formal external audit sign-off + zero open findings." }
      ],
      "is_resource_stock": false,
      "default_aspiration": 2
    },
    {
      "id": "speed",
      "name": "Time-to-Market",
      "description": "Release velocity and SLA responsiveness.",
      "rubric": [
        { "level": 0, "label": "L0: Multi-Week Delay", "description": "Blocked for >2 weeks; SLA breached." },
        { "level": 1, "label": "L1: Next Sprint", "description": "Delayed by 5-10 business days." },
        { "level": 2, "label": "L2: Same-Day Rollout", "description": "Deployed within 24 hours." },
        { "level": 3, "label": "L3: Immediate Hot-Path (<2h)", "description": "Zero-wait express deployment within 2 hours." }
      ],
      "is_resource_stock": false,
      "default_aspiration": 2
    },
    {
      "id": "budget",
      "name": "Engineering & Cloud Budget",
      "description": "Remaining sprint resource and overtime budget.",
      "rubric": [
        { "level": 0, "label": "L0: Budget Exhausted", "description": "Requires emergency executive budget override." },
        { "level": 1, "label": "L1: High Overtime Spend", "description": "Consumes >75% of reserve budget." },
        { "level": 2, "label": "L2: Standard Allocation", "description": "Within planned sprint allocation." },
        { "level": 3, "label": "L3: Under-Budget Surplus", "description": "Saves >30% of planned sprint compute/audit cost." }
      ],
      "is_resource_stock": true,
      "default_aspiration": 2
    }
  ],
  "preference_table": null,
  "cold_start_strategy": "zero_shot_inferred",
  "candidates": [
    {
      "id": "cand_canary_patch",
      "proposition": "Deploy the hotfix with automated regression tests and canary rollout today, deferring the full external penetration test to next sprint.",
      "is_status_quo": false
    },
    {
      "id": "cand_full_audit",
      "proposition": "Run the 12-hour external penetration test on standby and deploy within the 24-hour window.",
      "is_status_quo": true
    }
  ],
  "questions": [
    {
      "id": "q_approve_canary",
      "type": "tradeoff_noul",
      "candidate_id": "cand_canary_patch",
      "prompt": "Should we approve and execute cand_canary_patch right now?",
      "adaptation_mode": "auto_adapt"
    }
  ]
}
```

#### Response ($t = 0$)

```json
{
  "model": "tradeoff-jev-0.8b",
  "step": 0,
  "prior_preference_table": {
    "step": 0,
    "aspiration_levels": { "security": 3, "speed": 2, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "zero_shot_inferred",
    "last_transition": "initial",
    "adapted_goal": null
  },
  "preference_table": {
    "step": 0,
    "aspiration_levels": { "security": 3, "speed": 2, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "zero_shot_inferred",
    "last_transition": "initial",
    "adapted_goal": null
  },
  "feasibility_summary": {
    "prior_feasible": true,
    "adapted_feasible": true,
    "transition_applied": "initial",
    "adapted_goal": null,
    "satisficing_candidates": ["cand_full_audit"]
  },
  "candidate_evaluations": [
    {
      "candidate_id": "cand_canary_patch",
      "satisfices_prior_table": false,
      "satisfices_adapted_table": false,
      "joint_satisfaction_probability": 0.1124,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.02,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.08, 0.79, 0.12],
          "qualitative_expectation": "-",
          "meets_aspiration": false,
          "satisfaction_probability": 0.12
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.04,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.03, 0.87, 0.09],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.11,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.02, 0.85, 0.13],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        }
      }
    },
    {
      "candidate_id": "cand_full_audit",
      "satisfices_prior_table": true,
      "satisfices_adapted_table": true,
      "joint_satisfaction_probability": 0.841,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.91,
          "most_likely_level": 3,
          "level_probabilities": [0.0, 0.01, 0.07, 0.92],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.92
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 1.95,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.04, 0.94, 0.01],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.95
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 1.98,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.04, 0.94, 0.02],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        }
      }
    }
  ],
  "answers": [
    {
      "id": "q_approve_canary",
      "type": "tradeoff_noul",
      "candidate_id": "cand_canary_patch",
      "noul": 0.1124,
      "unconditioned_noul": 0.512,
      "prior_table_feasible": true,
      "adapted_table_feasible": true,
      "satisfices_adapted_table": false,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.02,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.08, 0.79, 0.12],
          "qualitative_expectation": "-",
          "meets_aspiration": false,
          "satisfaction_probability": 0.12
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.04,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.03, 0.87, 0.09],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.11,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.02, 0.85, 0.13],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        }
      }
    }
  ]
}
```

---

### 4.2 Step $t = 1$: Infeasible Prior Table $\to$ Downward Retreat (`last_transition = "downward_retreat"`)

At $t = 1$, the external audit firm is unavailable for 3 weeks, so `cand_full_audit` drops to `speed = 0`. Thus $\vec{a}_0 = (3, 2, 2)$ is **recognizably infeasible** across all candidates. Notice the asymmetry between Urgency and Retreat (Selten 1998, Section 2.1): even though `budget` is last in `urgency_order` (`["speed", "security", "budget"]`), the designated `retreat_goal` is `"security"` (dropping from L3 Pen-Test to L2 Canary + Automated Regression before sacrificing speed or budget).

#### Request ($t = 1$)

```json
{
  "model": "tradeoff-jev-0.8b",
  "step": 1,
  "state": "Update at t=1: External penetration testing firm just reported a 3-week backlog (cand_full_audit now incurs a multi-week delay, speed=L0). Live merchant traffic requires same-day resolution (speed>=L2). cand_canary_patch remains ready with full automated regression + staged canary (security=L2, speed=L2, budget=L2).",
  "goal_schema": [
    {
      "id": "security",
      "name": "Security & Compliance",
      "description": "Depth of security verification before production rollout.",
      "rubric": [
        { "level": 0, "label": "L0: Unverified", "description": "No automated or manual security checks." },
        { "level": 1, "label": "L1: Static Scan", "description": "Basic SAST/dependency scan only." },
        { "level": 2, "label": "L2: Canary + Automated Regression", "description": "Full automated security suite + staged canary rollout." },
        { "level": 3, "label": "L3: Full External Pen-Test", "description": "Formal external audit sign-off + zero open findings." }
      ]
    },
    {
      "id": "speed",
      "name": "Time-to-Market",
      "description": "Release velocity and SLA responsiveness.",
      "rubric": [
        { "level": 0, "label": "L0: Multi-Week Delay", "description": "Blocked for >2 weeks; SLA breached." },
        { "level": 1, "label": "L1: Next Sprint", "description": "Delayed by 5-10 business days." },
        { "level": 2, "label": "L2: Same-Day Rollout", "description": "Deployed within 24 hours." },
        { "level": 3, "label": "L3: Immediate Hot-Path (<2h)", "description": "Zero-wait express deployment within 2 hours." }
      ]
    },
    {
      "id": "budget",
      "name": "Engineering & Cloud Budget",
      "description": "Remaining sprint resource and overtime budget.",
      "rubric": [
        { "level": 0, "label": "L0: Budget Exhausted", "description": "Requires emergency executive budget override." },
        { "level": 1, "label": "L1: High Overtime Spend", "description": "Consumes >75% of reserve budget." },
        { "level": 2, "label": "L2: Standard Allocation", "description": "Within planned sprint allocation." },
        { "level": 3, "label": "L3: Under-Budget Surplus", "description": "Saves >30% of planned sprint compute/audit cost." }
      ],
      "is_resource_stock": true
    }
  ],
  "preference_table": {
    "step": 0,
    "aspiration_levels": { "security": 3, "speed": 2, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "zero_shot_inferred",
    "last_transition": "initial",
    "adapted_goal": null
  },
  "candidates": [
    {
      "id": "cand_canary_patch",
      "proposition": "Deploy the hotfix with automated regression tests and canary rollout today, deferring the full external penetration test to next sprint.",
      "is_status_quo": false
    },
    {
      "id": "cand_full_audit",
      "proposition": "Wait 3 weeks for the external penetration testing firm before deploying.",
      "is_status_quo": true
    }
  ],
  "questions": [
    {
      "id": "q_approve_canary",
      "type": "tradeoff_noul",
      "candidate_id": "cand_canary_patch",
      "prompt": "Should we approve and execute cand_canary_patch right now?",
      "adaptation_mode": "auto_adapt"
    }
  ]
}
```

#### Response ($t = 1$)

```json
{
  "model": "tradeoff-jev-0.8b",
  "step": 1,
  "prior_preference_table": {
    "step": 0,
    "aspiration_levels": { "security": 3, "speed": 2, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "zero_shot_inferred",
    "last_transition": "initial",
    "adapted_goal": null
  },
  "preference_table": {
    "step": 1,
    "aspiration_levels": { "security": 2, "speed": 2, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "markov_adapted",
    "last_transition": "downward_retreat",
    "adapted_goal": "security"
  },
  "feasibility_summary": {
    "prior_feasible": false,
    "adapted_feasible": true,
    "transition_applied": "downward_retreat",
    "adapted_goal": "security",
    "satisficing_candidates": ["cand_canary_patch"]
  },
  "candidate_evaluations": [
    {
      "candidate_id": "cand_canary_patch",
      "satisfices_prior_table": false,
      "satisfices_adapted_table": true,
      "joint_satisfaction_probability": 0.9085,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.05,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.03, 0.86, 0.1],
          "qualitative_expectation": "-",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.06,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.02, 0.87, 0.1],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.97
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.08,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.02, 0.87, 0.11],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        }
      }
    }
  ],
  "answers": [
    {
      "id": "q_approve_canary",
      "type": "tradeoff_noul",
      "candidate_id": "cand_canary_patch",
      "noul": 0.9085,
      "unconditioned_noul": 0.534,
      "prior_table_feasible": false,
      "adapted_table_feasible": true,
      "satisfices_adapted_table": true,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.05,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.03, 0.86, 0.1],
          "qualitative_expectation": "-",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.06,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.02, 0.87, 0.1],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.97
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.08,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.02, 0.87, 0.11],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        }
      }
    }
  ]
}
```

---

### 4.3 Step $t = 2$: Feasible Prior Table $\to$ Upward Urgency Adaptation (`last_transition = "upward_urgency"`)

At $t = 2$, the platform team provides a pre-verified express rollout option (`cand_preverified_fast`) that achieves `speed = 3` (under 2 hours) while maintaining `security = 2` and `budget = 2`.
- Incoming $\mathcal{T}_1$ has $\vec{a}_1 = (\text{security}: 2, \text{speed}: 2, \text{budget}: 2)$ and `urgency_order` $\pi_1 = [\text{speed}, \text{security}, \text{budget}]$.
- Since $\vec{a}_1$ is feasible and raising the most urgent goal (`speed`: $2 \to 3$) is also feasible via `cand_preverified_fast`, the **Upward Rule** updates $\vec{a}_2 = (\text{security}: 2, \text{speed}: 3, \text{budget}: 2)$.
- Because `cand_canary_patch` only delivers `speed = 2 < 3`, its conditioned `tradeoff_noul` drops from **`0.9085` (at $t=1$)** to **`0.138` (at $t=2$)**, while `cand_preverified_fast` receives **`0.931`**.

#### Request ($t = 2$)

```json
{
  "model": "tradeoff-jev-0.8b",
  "step": 2,
  "state": "Update at t=2: Platform engineering just unlocked a pre-verified hot-path pipeline (cand_preverified_fast) that completes deployment in 90 minutes (speed=L3) with full automated security regression + canary isolation (security=L2) at standard budget (budget=L2). cand_canary_patch remains available (security=L2, speed=L2, budget=L2).",
  "goal_schema": [
    {
      "id": "security",
      "name": "Security & Compliance",
      "description": "Depth of security verification before production rollout.",
      "rubric": [
        { "level": 0, "label": "L0: Unverified", "description": "No automated or manual security checks." },
        { "level": 1, "label": "L1: Static Scan", "description": "Basic SAST/dependency scan only." },
        { "level": 2, "label": "L2: Canary + Automated Regression", "description": "Full automated security suite + staged canary rollout." },
        { "level": 3, "label": "L3: Full External Pen-Test", "description": "Formal external audit sign-off + zero open findings." }
      ]
    },
    {
      "id": "speed",
      "name": "Time-to-Market",
      "description": "Release velocity and SLA responsiveness.",
      "rubric": [
        { "level": 0, "label": "L0: Multi-Week Delay", "description": "Blocked for >2 weeks; SLA breached." },
        { "level": 1, "label": "L1: Next Sprint", "description": "Delayed by 5-10 business days." },
        { "level": 2, "label": "L2: Same-Day Rollout", "description": "Deployed within 24 hours." },
        { "level": 3, "label": "L3: Immediate Hot-Path (<2h)", "description": "Zero-wait express deployment within 2 hours." }
      ]
    },
    {
      "id": "budget",
      "name": "Engineering & Cloud Budget",
      "description": "Remaining sprint resource and overtime budget.",
      "rubric": [
        { "level": 0, "label": "L0: Budget Exhausted", "description": "Requires emergency executive budget override." },
        { "level": 1, "label": "L1: High Overtime Spend", "description": "Consumes >75% of reserve budget." },
        { "level": 2, "label": "L2: Standard Allocation", "description": "Within planned sprint allocation." },
        { "level": 3, "label": "L3: Under-Budget Surplus", "description": "Saves >30% of planned sprint compute/audit cost." }
      ],
      "is_resource_stock": true
    }
  ],
  "preference_table": {
    "step": 1,
    "aspiration_levels": { "security": 2, "speed": 2, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "markov_adapted",
    "last_transition": "downward_retreat",
    "adapted_goal": "security"
  },
  "candidates": [
    {
      "id": "cand_canary_patch",
      "proposition": "Deploy the hotfix with automated regression tests and canary rollout today, deferring the full external penetration test to next sprint.",
      "is_status_quo": true
    },
    {
      "id": "cand_preverified_fast",
      "proposition": "Deploy via the pre-verified hot-path pipeline within 90 minutes with automated regression and canary isolation.",
      "is_status_quo": false
    }
  ],
  "questions": [
    {
      "id": "q_approve_canary",
      "type": "tradeoff_noul",
      "candidate_id": "cand_canary_patch",
      "prompt": "Should we approve and execute cand_canary_patch right now?",
      "adaptation_mode": "auto_adapt"
    },
    {
      "id": "q_approve_fast_track",
      "type": "tradeoff_noul",
      "candidate_id": "cand_preverified_fast",
      "prompt": "Should we approve and execute cand_preverified_fast right now?",
      "adaptation_mode": "auto_adapt"
    }
  ]
}
```

#### Response ($t = 2$)

```json
{
  "model": "tradeoff-jev-0.8b",
  "step": 2,
  "prior_preference_table": {
    "step": 1,
    "aspiration_levels": { "security": 2, "speed": 2, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "markov_adapted",
    "last_transition": "downward_retreat",
    "adapted_goal": "security"
  },
  "preference_table": {
    "step": 2,
    "aspiration_levels": { "security": 2, "speed": 3, "budget": 2 },
    "urgency_order": ["speed", "security", "budget"],
    "retreat_goal": "security",
    "origin": "markov_adapted",
    "last_transition": "upward_urgency",
    "adapted_goal": "speed"
  },
  "feasibility_summary": {
    "prior_feasible": true,
    "adapted_feasible": true,
    "transition_applied": "upward_urgency",
    "adapted_goal": "speed",
    "satisficing_candidates": ["cand_preverified_fast"]
  },
  "candidate_evaluations": [
    {
      "candidate_id": "cand_canary_patch",
      "satisfices_prior_table": true,
      "satisfices_adapted_table": false,
      "joint_satisfaction_probability": 0.138,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.04,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.03, 0.87, 0.09],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.09,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.05, 0.8, 0.15],
          "qualitative_expectation": "0",
          "meets_aspiration": false,
          "satisfaction_probability": 0.15
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.06,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.04, 0.86, 0.1],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        }
      }
    },
    {
      "candidate_id": "cand_preverified_fast",
      "satisfices_prior_table": true,
      "satisfices_adapted_table": true,
      "joint_satisfaction_probability": 0.9312,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.03,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.03, 0.91, 0.06],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.97
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.95,
          "most_likely_level": 3,
          "level_probabilities": [0.0, 0.0, 0.02, 0.98],
          "qualitative_expectation": "+",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.02,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.02, 0.94, 0.04],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        }
      }
    }
  ],
  "answers": [
    {
      "id": "q_approve_canary",
      "type": "tradeoff_noul",
      "candidate_id": "cand_canary_patch",
      "noul": 0.138,
      "unconditioned_noul": 0.518,
      "prior_table_feasible": true,
      "adapted_table_feasible": true,
      "satisfices_adapted_table": false,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.04,
          "most_likely_level": 2,
          "level_probabilities": [0.01, 0.03, 0.87, 0.09],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.09,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.05, 0.8, 0.15],
          "qualitative_expectation": "0",
          "meets_aspiration": false,
          "satisfaction_probability": 0.15
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.06,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.04, 0.86, 0.1],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.96
        }
      }
    },
    {
      "id": "q_approve_fast_track",
      "type": "tradeoff_noul",
      "candidate_id": "cand_preverified_fast",
      "noul": 0.9312,
      "unconditioned_noul": 0.589,
      "prior_table_feasible": true,
      "adapted_table_feasible": true,
      "satisfices_adapted_table": true,
      "goal_expectations": {
        "security": {
          "goal_id": "security",
          "expected_level": 2.03,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.03, 0.91, 0.06],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.97
        },
        "speed": {
          "goal_id": "speed",
          "expected_level": 2.95,
          "most_likely_level": 3,
          "level_probabilities": [0.0, 0.0, 0.02, 0.98],
          "qualitative_expectation": "+",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        },
        "budget": {
          "goal_id": "budget",
          "expected_level": 2.02,
          "most_likely_level": 2,
          "level_probabilities": [0.0, 0.02, 0.94, 0.04],
          "qualitative_expectation": "0",
          "meets_aspiration": true,
          "satisfaction_probability": 0.98
        }
      }
    }
  ]
}
```

---

## 5. Summary of Conditioned `noul` Trajectory for `cand_canary_patch`

| Markov Step $t$ | Incoming Table $\vec{a}_{t-1}$ | Event in State $x_t$ | Selten AAT Transition | Updated Table $\vec{a}_t$ | Unconditioned `noul` $P(\text{yes} \mid x_t)$ | Conditioned `tradeoff_noul` $P(\text{yes} \mid x_t, \mathcal{T}_t)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$t = 0$** | Cold-Start (`null`) | Full audit (`3,2,2`) is feasible | `initial` (`zero_shot_inferred`) | `(sec: 3, spd: 2, bud: 2)` | `0.512` (ambiguous) | **`0.1124` (Reject)** |
| **$t = 1$** | `(sec: 3, spd: 2, bud: 2)` | Auditor delayed (`3,2,2` infeasible) | `downward_retreat` on $r_0 = \text{security}$ | `(sec: 2, spd: 2, bud: 2)` | `0.534` (ambiguous) | **`0.9085` (Accept)** |
| **$t = 2$** | `(sec: 2, spd: 2, bud: 2)` | Fast-track (`2,3,2`) unlocked | `upward_urgency` on $\pi_{1,1} = \text{speed}$ | `(sec: 2, spd: 3, bud: 2)` | `0.518` (ambiguous) | **`0.1380` (Reject)** |
