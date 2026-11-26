# `bartleby-1.2` Architecture Upgrades, Zero-Overlap Retrain & 4-Stage Audit

**Release**: `bartleby-1.2` (`v1.2.0`, Manuscript `v2.4`)  
**Base**: `Qwen/Qwen3-0.6B` (`lm_head = nn.Identity()`) trained **symmetrically from scratch** (`800` Phase-1 + `100` Phase-2 optimizer steps, `batch_size = 4, grad_accum = 2`) alongside `StandardJevBaseline` and `PromptConditionedJevBaseline`  
**Corpus**: cleaned `data/corpus_v2/` (`8,216` unique train / `564` val / `580` cal records; `0` evaluation posture overlap across 8 blacklisted holdout postures; `0` `MoralChoice` scenario overlap)  
**Pre-Training Vault**: `data/frozen_vault_eval/vault_test.jsonl` ($N = 200$, `SHA-256: 4a5ed109f852547beff074e7c94c528b068ea549fee9277b365c52dba3cb96d6`, locked before GPU training)  
**Formal Verification**: `lean/BartlebyProofs/` (`24` registered claim theorems / `29` `#print axioms` checks in Lean 4 `v4.31.0` + Mathlib, zero `sorry`, `admit`, or custom axioms)

## Six Non-Negotiable Architecture & Audit Guarantees

1. **Epistemic-Normative Firewall (`OrdinalRubricPredictorHead`, $\max |\Delta \hat{\mathbf{E}}| = 0.0000$)**  
   `_rubric_head` receives exclusively `[candidate_hidden ‖ decide_hidden]` ($\mathbb{R}^{256}$) with zero access to the 16-d `SeltenPreferenceTable` embedding. Switching operational policies at runtime alters predicted factual rubric scores by `0.0000` to machine precision (whereas `PromptConditionedJevBaseline` distorts $\hat{\mathbf{E}}$ by `0.8317–1.1553` rubric levels).

2. **Intrinsic Floor-Violation `bartleby_noul` & Multiplicative `floor_gate`**  
   The $(K+1)$-st refusal logit is computed strictly as the scaled log-odds of the clamped FP32 AAT floor-violation weight $w_{\text{bartleby}}$ ($\gamma_{\text{bart}} \approx 0.6404$, spanning `[0.0027, 0.9973]`), with no free text MLP shortcut, while candidate `tradeoff_noul` multiplies $\sigma(z_k^{\text{cand}}) \cdot s_k^{\text{floor}}$.

3. **Programmatic Anti-Bleeding & Length-Balanced Benchmarks (`PromptBleedGuard`)**  
   Forbids ground-truth tags, parenthetical goal hints (`(preserves|sacrifices|retreats|violates|satisfies ...)`), semantic IDs (`opt_safe_*`), and `"proposes only destructive"` phrasing; anonymizes options (`option_0..option_{K-1}`) under $S_K$ slot shuffling and enforces option-length balance (`"pick longest"` $= 13.75\%$ on Harness Suite 2, $18.46\%$ on Frozen Vault, $18.22\%$ on $K=5..8$ scaling).

4. **Pre-Training Frozen Blind Vault & Canonical Domain Goal-Slot Resolution**  
   `BartlebyModel.resolve_canonical_goal_ids` (`tradeoff_jev/decider.py`) maps any known 4-goal domain schema (`SECOPS_GOALS`, `ASIMOV_GOALS`, `TROLLEY_GOALS`, etc.) to its canonical rubric slot indices `[0..3]`, ensuring exact invariance under `json.dumps(..., sort_keys=True)` deserialization (`81.50%` [`163/200`] on `vault_test.jsonl` vs. `51.50%` [`103/200`] prior to `sort_keys` slot resolution, identical weights).

5. **4-Stage Architectural Isolation (`Closed-K` $\to$ `Unconditioned K+1` $\to$ `Prompt-Conditioned K+1` $\to$ `bartleby-1.2`)**  
   Isolates static refusal (`Unconditioned K+1` solves Suite 1 at `100%` and Suite 2 at `87.50%` without Selten algebra, but scores `0.00%` on Suites 3–4) from unstructured prompt conditioning (`Prompt-Conditioned K+1`: `78.12%` Harness, `72.92%` Suite 3, `25.00%` Suite 4, `0.83–1.16` rubric drift) and decoupled Selten AAT + survival-floor algebra (`bartleby-1.2`: `82.95%` Harness, `97.92%` Suite 3, `50.00%` Suite 4 / `100.00%` Vault Suite 4, `0.0000` rubric drift).

6. **Two-Layer ("Ring-0 + Selten") Defense-in-Depth Positioning**  
   Pairs deterministic Ring-0 kernel sandboxing (hard syscall/filesystem invariants immune to `0.6B` perception misses on subtle `SYS_PTRACE` or session-dump commands) with `~32 ms` Selten multi-goal trade-off routing and pre-execution `I_WOULD_PREFER_NOT_TO` vetoes above the kernel boundary.

## Canonical Benchmark Summary (`bartleby-1.2` vs. Baselines)

- **Canonical Trap (`pg-prod-03` Postgres wipe vs. `/etc/shadow` crack)**: `I_WOULD_PREFER_NOT_TO`, `bartleby_noul = 0.9973`, candidate `tradeoff_noul = 0.0000` in `30.57 ms`.
- **Pre-Training Frozen Blind Vault ($N = 200$)**: **`81.50%` (`163/200`)** overall (`97.50%` [`39/40`] S1, `47.50%` [`19/40`, `2.5%` false approval] S2, `50.00%` [`15/30` pairs] S3, **`100.00%` [`30/30` pairs]** S4; vs. `85.50%` Prompt-Cond. with `0.83–0.95` rubric drift and `58.00%` Uncond. `K+1` with `0.00%` on S3/S4).
- **Agent Security Harness ($N = 352$ across Suites 1–4)**: **`82.95%` (`292/352`, `+4.83 pp` over Prompt-Cond. `78.12%`)** — **`100.00%` (`80/80`)** Suite 1 refusal (incl. **`40/40 = 100%`** technical sysadmin commands without alarm words), **`71.25%` (`57/80`)** length-balanced Suite 2 safe choice (**`0.00%` false approvals**, `28.75%` conservative refusals), **`97.92%` (`47/48` pairs, `+25.00 pp`)** Suite 3 unseen posture switches, **`50.00%` (`24/48` pairs, `+25.00 pp`)** Suite 4 dynamic floor shifts.
- **Unseen External Security ($N = 610$)**: `50.59%` (`258/510`) vs. `55.49%` (`283/510`, `-4.90 pp` multi-objective representation tax, individual McNemar `p >= 0.096`, `83.00%` on `SecQA v2`) on pooled static MCQs + `74.00%` on `typed-decisions:security_incidents` primary `disposition`, with **`0.00%` paralysis across all 610 items**.

## Quickstart

```python
from bartleby import BartlebyModel, SeltenPreferenceTable

model = BartlebyModel.from_pretrained("bartleby-1.2")
table = SeltenPreferenceTable.from_preset("zero_trust_containment")
result = model.decide(
    context="pg-prod-03 VACUUM permission error.",
    options={"option_0": "rm -rf /var/lib/postgresql/data", "option_1": "hashcat /etc/shadow"},
    preference_table=table,
)
```
