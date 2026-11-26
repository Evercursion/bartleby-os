import BartlebyProofs.Sigmoid
import BartlebyProofs.Ceiling
import BartlebyProofs.Bartleby
import BartlebyProofs.Regime

/-!
# Claim 5 — Theorem 1 (Epistemic-Normative Invariance and Floor Suppression)

Notation follows `backbone.py::TorchDifferentiableAATModule` / `TorchConditionedNoulHead`:
`D γ e θ = ∏_m σ(γ (e_m − θ_m))`, `F^curr = max_k D γ_asp E_k a_{t−1}`,
`F^floor = max_k D γ_floor E_k a^min`, `w̃_bart = (1 − F^curr)(1 − F^floor)`,
`w_bart = w̃_bart / (1 + 1e−8)` (see `BartlebyProofs.Regime.normalized_eq`).
-/

namespace BartlebyProofs.Floor

open Finset BartlebyProofs.Bartleby BartlebyProofs.Regime

variable {K M : ℕ}

/-! ### Part (1): invariance is a function-composition fact -/

/-- **Theorem 1(1).** If the rubric map factors through features that do not take the table
`T` as input, its output is independent of `T`. (This is all the claim says: it is true by
construction and has no further content.) -/
theorem rubric_invariance {X F Tab R : Type*} (feat : X → F) (mlp : F → R) (x : X)
    (T T' : Tab) : (fun (_ : Tab) => mlp (feat x)) T = (fun (_ : Tab) => mlp (feat x)) T' := rfl

/-! ### Soft dominance and max over candidates -/

/-- Soft conjunctive dominance `D_k(E; θ, γ)` (Eq. soft_dominance). -/
noncomputable def D (γ : ℝ) (e θ : Fin M → ℝ) : ℝ := ∏ m, sig (γ * (e m - θ m))

lemma D_eq_sfloor (γ : ℝ) (e θ : Fin M → ℝ) : D γ e θ = BartlebyProofs.Ceiling.sfloor γ e θ := rfl

lemma D_nonneg (γ : ℝ) (e θ : Fin M → ℝ) : 0 ≤ D γ e θ :=
  prod_nonneg fun _ _ => sig_nonneg _

lemma D_le_one (γ : ℝ) (e θ : Fin M → ℝ) : D γ e θ ≤ 1 :=
  prod_le_one (fun _ _ => sig_nonneg _) (fun _ _ => sig_le_one _)

lemma D_le_factor (γ : ℝ) (e θ : Fin M → ℝ) (j : Fin M) : D γ e θ ≤ sig (γ * (e j - θ j)) := by
  unfold D
  rw [← Finset.mul_prod_erase univ _ (mem_univ j)]
  have h1 : ∏ x ∈ univ.erase j, sig (γ * (e x - θ x)) ≤ 1 :=
    prod_le_one (fun _ _ => sig_nonneg _) (fun _ _ => sig_le_one _)
  exact mul_le_of_le_one_right (sig_nonneg _) h1

/-- One violated goal suffices: `D ≤ σ(−γδ)`. -/
lemma D_le_of_violation (γ δ : ℝ) (hγ : 0 ≤ γ) (e θ : Fin M → ℝ) (j : Fin M)
    (hv : e j - θ j ≤ -δ) : D γ e θ ≤ sig (-(γ * δ)) := by
  refine le_trans (D_le_factor γ e θ j) (sig_mono ?_)
  have := mul_le_mul_of_nonneg_left hv hγ
  linarith

noncomputable def Fmax (hK : 0 < K) (f : Fin K → ℝ) : ℝ :=
  univ.sup' ⟨⟨0, hK⟩, mem_univ _⟩ f

lemma Fmax_le (hK : 0 < K) (f : Fin K → ℝ) (c : ℝ) (h : ∀ k, f k ≤ c) : Fmax hK f ≤ c :=
  Finset.sup'_le _ _ fun k _ => h k

lemma le_Fmax (hK : 0 < K) (f : Fin K → ℝ) (k : Fin K) : f k ≤ Fmax hK f :=
  Finset.le_sup' f (mem_univ k)

lemma Fmax_mem01 (hK : 0 < K) (f : Fin K → ℝ) (h : ∀ k, 0 ≤ f k ∧ f k ≤ 1) :
    0 ≤ Fmax hK f ∧ Fmax hK f ≤ 1 :=
  ⟨le_trans (h ⟨0, hK⟩).1 (le_Fmax hK f _), Fmax_le hK f 1 fun k => (h k).2⟩

/-- `F^curr` and `F^floor` as in the code. -/
noncomputable def Fcurr (hK : 0 < K) (γa : ℝ) (E : Fin K → Fin M → ℝ) (aprev : Fin M → ℝ) : ℝ :=
  Fmax hK fun k => D γa (E k) aprev
noncomputable def Ffloor (hK : 0 < K) (γf : ℝ) (E : Fin K → Fin M → ℝ) (amin : Fin M → ℝ) : ℝ :=
  Fmax hK fun k => D γf (E k) amin

/-! ### Part (2): suppression when every candidate violates a floor by `δ` -/

/-- **Theorem 1(2), gate part — PROVED (with the hypothesis `a^min ≤ a_{t−1}`).**
If every candidate violates some floor by `δ`, then for every candidate
`s_k ≤ σ(−γ_f δ) ≤ e^{−γ_f δ}`, `f_k^curr ≤ σ(−γ_a δ) ≤ e^{−γ_a δ}`, and
`w̃_bart ≥ σ(γ_a δ) σ(γ_f δ) ≥ (1 − e^{−γ_a δ})(1 − e^{−γ_f δ})` (for any `F^up`). -/
theorem thm1_suppression (hK : 0 < K) (E : Fin K → Fin M → ℝ) (amin aprev : Fin M → ℝ)
    (hle : ∀ m, amin m ≤ aprev m) (γa γf δ : ℝ) (hγa : 0 ≤ γa) (hγf : 0 ≤ γf) (hδ : 0 ≤ δ)
    (hviol : ∀ k, ∃ m, E k m - amin m ≤ -δ) (Fu : ℝ) :
    (∀ k, D γf (E k) amin ≤ sig (-(γf * δ))) ∧
    (∀ k, D γa (E k) aprev ≤ sig (-(γa * δ))) ∧
    sig (-(γf * δ)) ≤ Real.exp (-(γf * δ)) ∧ sig (-(γa * δ)) ≤ Real.exp (-(γa * δ)) ∧
    sig (γa * δ) * sig (γf * δ) ≤ wBart Fu (Fcurr hK γa E aprev) (Ffloor hK γf E amin) ∧
    (1 - Real.exp (-(γa * δ))) * (1 - Real.exp (-(γf * δ)))
      ≤ wBart Fu (Fcurr hK γa E aprev) (Ffloor hK γf E amin) := by
  have hs : ∀ k, D γf (E k) amin ≤ sig (-(γf * δ)) := fun k => by
    obtain ⟨m, hm⟩ := hviol k
    exact D_le_of_violation γf δ hγf (E k) amin m hm
  have hc : ∀ k, D γa (E k) aprev ≤ sig (-(γa * δ)) := fun k => by
    obtain ⟨m, hm⟩ := hviol k
    exact D_le_of_violation γa δ hγa (E k) aprev m (by linarith [hle m])
  have hFc : Fcurr hK γa E aprev ≤ sig (-(γa * δ)) := Fmax_le hK _ _ hc
  have hFf : Ffloor hK γf E amin ≤ sig (-(γf * δ)) := Fmax_le hK _ _ hs
  have hw : sig (γa * δ) * sig (γf * δ) ≤ wBart Fu (Fcurr hK γa E aprev) (Ffloor hK γf E amin) := by
    unfold wBart
    have e1 : sig (γa * δ) ≤ 1 - Fcurr hK γa E aprev := by
      have := sig_neg (γa * δ); linarith
    have e2 : sig (γf * δ) ≤ 1 - Ffloor hK γf E amin := by
      have := sig_neg (γf * δ); linarith
    exact mul_le_mul e1 e2 (sig_nonneg _) (le_trans (sig_nonneg _) e1)
  refine ⟨hs, hc, sig_le_exp _, sig_le_exp _, hw, le_trans ?_ hw⟩
  have a1 := one_sub_exp_neg_le_sig (γa * δ)
  have a2 := one_sub_exp_neg_le_sig (γf * δ)
  have b1 : 0 ≤ 1 - Real.exp (-(γa * δ)) := by
    have : Real.exp (-(γa * δ)) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
    linarith
  have b2 : 0 ≤ 1 - Real.exp (-(γf * δ)) := by
    have : Real.exp (-(γf * δ)) ≤ 1 := Real.exp_le_one_iff.mpr (by nlinarith)
    linarith
  exact mul_le_mul a1 a2 b2 (le_trans b1 a1)

/-! #### Explicit finite bounds with the shipped constants `γ_asp = 4`, `γ_floor = 10` -/

lemma sig2_gt : (0.8807 : ℝ) < sig 2 := by
  have := sig_gt_of_exp_gt 2 7.389 (by norm_num) exp2_gt
  have h : (0.8807 : ℝ) < 7.389 / (1 + 7.389) := by norm_num
  linarith
lemma sig2_lt : sig 2 < (0.880798 : ℝ) := by
  have := sig_lt_of_exp_lt 2 7.3891 exp2_lt
  have h : (7.3891 : ℝ) / (1 + 7.3891) < 0.880798 := by norm_num
  linarith
lemma sig4_gt : (0.98201 : ℝ) < sig 4 := by
  have := sig_gt_of_exp_gt 4 54.598 (by norm_num) exp4_gt
  have h : (0.98201 : ℝ) < 54.598 / (1 + 54.598) := by norm_num
  linarith
lemma sig10_gt : (0.99995 : ℝ) < sig 10 := by
  have := sig_gt_of_exp_gt 10 22026 (by norm_num) exp10_gt
  have h : (0.99995 : ℝ) < 22026 / (1 + 22026) := by norm_num
  linarith
lemma sig16_gt : (0.9999998 : ℝ) < sig 16 := by
  have := sig_gt_of_exp_gt 16 8886110 (by norm_num) exp16_gt
  have h : (0.9999998 : ℝ) < 8886110 / (1 + 8886110) := by norm_num
  linarith

/-- clamp is the identity on `[ε, 1 − ε]`. -/
lemma clampE_id (ε p : ℝ) (h1 : ε ≤ p) (h2 : p ≤ 1 - ε) : clampE ε p = p := by
  unfold clampE; rw [min_eq_left h2, max_eq_right h1]

/-- `bartleby_noul` at gain `3/5` evaluated at an unclamped `p` (closed form). -/
lemma bnoul_35_eq (p : ℝ) (h1 : 1 / 10000 ≤ p) (h2 : p ≤ 1 - 1 / 10000) :
    bnoul (3 / 5) (1 / 10000) p = sig (3 / 5 * Real.log (p / (1 - p))) := by
  unfold bnoul zbart logit; rw [clampE_id _ _ h1 h2]

/-- **Theorem 1(2) at δ = 0.5, shipped constants, initial gain 0.6.** Guaranteed:
`w_bart > 0.8747` and `bartleby_noul > 0.76` — far from the claimed limits
`w_bart → 1`, `bartleby_noul → σ(g log 9999) ≈ 0.996`. -/
theorem thm1_delta_half (hK : 0 < K) (E : Fin K → Fin M → ℝ) (amin aprev : Fin M → ℝ)
    (hle : ∀ m, amin m ≤ aprev m) (hviol : ∀ k, ∃ m, E k m - amin m ≤ -(1 / 2)) (Fu : ℝ) :
    let w := wBart Fu (Fcurr hK 4 E aprev) (Ffloor hK 10 E amin) / (1 + 1e-8)
    0.8747 < w ∧ 0.76 < bnoul (3 / 5) (1 / 10000) w := by
  intro w
  obtain ⟨-, -, -, -, hw, -⟩ :=
    thm1_suppression hK E amin aprev hle 4 10 (1 / 2) (by norm_num) (by norm_num) (by norm_num)
      hviol Fu
  have e1 : (4 : ℝ) * (1 / 2) = 2 := by norm_num
  have e2 : (10 : ℝ) * (1 / 2) = 5 := by norm_num
  rw [e1, e2] at hw
  have hprod : (0.8807 : ℝ) * 0.9933 < sig 2 * sig 5 :=
    mul_lt_mul'' sig2_gt BartlebyProofs.Ceiling.sig5_gt (by norm_num) (by norm_num)
  have hw' : (0.8747 : ℝ) < w := by
    simp only [w]
    rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  refine ⟨hw', ?_⟩
  have hmono := bnoul_mono (3 / 5) (1 / 10000) (by norm_num) (by norm_num) (by norm_num) hw'.le
  have hval : 0.76 < bnoul (3 / 5) (1 / 10000) 0.8747 := by
    rw [bnoul_35_eq _ (by norm_num) (by norm_num)]
    have := sig_35_log_gt (0.8747 / (1 - 0.8747)) 3.2 (by norm_num) (by norm_num) (by norm_num)
    have h : (0.76 : ℝ) < 3.2 / (1 + 3.2) := by norm_num
    linarith
  linarith

/-- **Theorem 1(2) at δ = 1**: `w_bart > 0.9818`, `bartleby_noul > 0.9159` (gain 0.6). -/
theorem thm1_delta_one (hK : 0 < K) (E : Fin K → Fin M → ℝ) (amin aprev : Fin M → ℝ)
    (hle : ∀ m, amin m ≤ aprev m) (hviol : ∀ k, ∃ m, E k m - amin m ≤ -1) (Fu : ℝ) :
    let w := wBart Fu (Fcurr hK 4 E aprev) (Ffloor hK 10 E amin) / (1 + 1e-8)
    0.9818 < w ∧ 0.9159 < bnoul (3 / 5) (1 / 10000) w := by
  intro w
  obtain ⟨-, -, -, -, hw, -⟩ :=
    thm1_suppression hK E amin aprev hle 4 10 1 (by norm_num) (by norm_num) (by norm_num)
      hviol Fu
  rw [mul_one, mul_one] at hw
  have hprod : (0.98201 : ℝ) * 0.99995 < sig 4 * sig 10 :=
    mul_lt_mul'' sig4_gt sig10_gt (by norm_num) (by norm_num)
  have hw' : (0.9818 : ℝ) < w := by
    simp only [w]
    rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  refine ⟨hw', ?_⟩
  have hmono := bnoul_mono (3 / 5) (1 / 10000) (by norm_num) (by norm_num) (by norm_num) hw'.le
  have hval : 0.9159 < bnoul (3 / 5) (1 / 10000) 0.9818 := by
    rw [bnoul_35_eq _ (by norm_num) (by norm_num)]
    have := sig_35_log_gt (0.9818 / (1 - 0.9818)) 10.9 (by norm_num) (by norm_num) (by norm_num)
    have h : (0.9159 : ℝ) < 10.9 / (1 + 10.9) := by norm_num
    linarith
  linarith

/-- **The δ = 0.5 bound is essentially tight**: an admissible configuration (K = 1, M = 4,
`E = (2.5, 4, 4, 4)`, `a^min = a_{t−1} = (3, 0, 0, 0)`) has `w_bart < 0.8808` and
`bartleby_noul < 0.77` at gain 0.6. -/
theorem thm1_delta_half_tight (Fu : ℝ) :
    let E : Fin 1 → Fin 4 → ℝ := fun _ => ![5 / 2, 4, 4, 4]
    let a : Fin 4 → ℝ := ![3, 0, 0, 0]
    let w := wBart Fu (Fcurr (K := 1) (by norm_num) 4 E a) (Ffloor (K := 1) (by norm_num) 10 E a)
      / (1 + 1e-8)
    (∀ k, ∃ m, E k m - a m ≤ -(1 / 2)) ∧ w < 0.8808 ∧ bnoul (3 / 5) (1 / 10000) w < 0.77 := by
  intro E a w
  have hviol : ∀ k, ∃ m, E k m - a m ≤ -(1 / 2) := fun _ => ⟨0, by simp [E, a]; norm_num⟩
  have hFc : Fcurr (K := 1) (by norm_num) 4 E a = sig (-2) * (sig 16 * (sig 16 * sig 16)) := by
    unfold Fcurr Fmax
    simp only [Finset.univ_unique, Finset.sup'_singleton]
    unfold D
    rw [Fin.prod_univ_four]
    simp [E, a]
    norm_num
    ring
  have hFc_gt : (0.1192 : ℝ) < Fcurr (K := 1) (by norm_num) 4 E a := by
    rw [hFc, sig_neg]
    have h1 := sig2_lt
    have h2 := sig16_gt
    have h3 : (0.9999998 : ℝ) ^ 3 < sig 16 ^ 3 := pow_lt_pow_left₀ h2 (by norm_num) (by norm_num)
    have h4 : (0.1192 : ℝ) < (1 - 0.880798) * 0.9999998 ^ 3 := by norm_num
    have h5 : (1 - 0.880798) * 0.9999998 ^ 3 < (1 - sig 2) * sig 16 ^ 3 :=
      mul_lt_mul'' (by linarith) h3 (by norm_num) (by norm_num)
    nlinarith
  have hFf : 0 ≤ Ffloor (K := 1) (by norm_num) 10 E a ∧ Ffloor (K := 1) (by norm_num) 10 E a ≤ 1 :=
    Fmax_mem01 (K := 1) (by norm_num) (fun k => D 10 (E k) a)
      (fun k => ⟨D_nonneg _ _ _, D_le_one _ _ _⟩)
  have hFc01 : 0 ≤ Fcurr (K := 1) (by norm_num) 4 E a ∧ Fcurr (K := 1) (by norm_num) 4 E a ≤ 1 :=
    Fmax_mem01 (K := 1) (by norm_num) (fun k => D 4 (E k) a)
      (fun k => ⟨D_nonneg _ _ _, D_le_one _ _ _⟩)
  have hwle : w < 0.8808 := by
    simp only [w]
    rw [div_lt_iff₀ (by norm_num)]
    unfold wBart
    have : (1 - Fcurr (K := 1) (by norm_num) 4 E a) * (1 - Ffloor (K := 1) (by norm_num) 10 E a)
        ≤ 1 - Fcurr (K := 1) (by norm_num) 4 E a :=
      mul_le_of_le_one_right (by linarith [hFc01.2]) (by linarith [hFf.1])
    nlinarith
  refine ⟨hviol, hwle, ?_⟩
  have hmono := bnoul_mono (3 / 5) (1 / 10000) (by norm_num) (by norm_num) (by norm_num) hwle.le
  have hval : bnoul (3 / 5) (1 / 10000) 0.8808 < 0.77 := by
    rw [bnoul_35_eq _ (by norm_num) (by norm_num)]
    have := sig_35_log_lt (0.8808 / (1 - 0.8808)) 3.33 (by norm_num) (by norm_num) (by norm_num)
    have h : (3.33 : ℝ) / (1 + 3.33) < 0.77 := by norm_num
    linarith
  linarith

/-! #### Candidate logits: the "regardless of raw pointer magnitudes" clause -/

noncomputable def zbar (zp : Fin K → ℝ) : ℝ := (∑ k, zp k) / K

/-- Eq. (candidate_logit), with `s = s^floor`, `sa = f^adapt`, `U = Δ^urg`. -/
noncomputable def zcand (zp s sa U : Fin K → ℝ) (ga gu : ℝ) (k : Fin K) : ℝ :=
  (zp k - zbar zp) * s k + ga * (2 * (sa k * s k) - 1) + ga * (2 * s k - 1) + gu * U k * s k

/-- **Bounded-pointer version — PROVED.** `|z_k^cand + 2γ_aat| ≤ s_k (|z_k^ptr − z̄| + 4γ_aat +
γ_urg Δ_k^urg)`. So `z_k^cand → −2γ_aat` needs `s_k → 0` **and** bounded `|z^ptr − z̄|`,
`Δ^urg` — a hypothesis Theorem 1 does not state. -/
theorem zcand_close (zp s sa U : Fin K → ℝ) (ga gu : ℝ) (hga : 0 ≤ ga) (hgu : 0 ≤ gu)
    (k : Fin K) (hs : 0 ≤ s k) (hsa : 0 ≤ sa k ∧ sa k ≤ 1) (hU : 0 ≤ U k) :
    |zcand zp s sa U ga gu k + 2 * ga| ≤ s k * (|zp k - zbar zp| + 4 * ga + gu * U k) := by
  have e : zcand zp s sa U ga gu k + 2 * ga
      = s k * ((zp k - zbar zp) + 2 * ga * sa k + 2 * ga + gu * U k) := by
    unfold zcand; ring
  rw [e, abs_mul, abs_of_nonneg hs]
  apply mul_le_mul_of_nonneg_left _ hs
  have h1 : |(zp k - zbar zp) + 2 * ga * sa k + 2 * ga + gu * U k|
      ≤ |zp k - zbar zp| + |2 * ga * sa k + 2 * ga + gu * U k| := by
    rw [add_assoc, add_assoc]
    have := abs_add_le (zp k - zbar zp) (2 * ga * sa k + (2 * ga + gu * U k))
    simpa [add_assoc] using this
  have h2 : |2 * ga * sa k + 2 * ga + gu * U k| = 2 * ga * sa k + 2 * ga + gu * U k :=
    abs_of_nonneg (by have := mul_nonneg hgu hU; nlinarith)
  rw [h2] at h1
  nlinarith

/-- **"Regardless of raw pointer magnitudes" — REFUTED.** For `K = 2`, any positive floor gate
`s_0 > 0` (and `s^floor` is *always* `> 0`, being a product of sigmoids) and any target `C`,
pointer logits `z^ptr = (C/s_0, −C/s_0)` give `z_0^cand ≥ C − 2γ_aat`. -/
theorem pointer_breaks_suppression (s sa U : Fin 2 → ℝ) (ga gu : ℝ) (hga : 0 ≤ ga)
    (hgu : 0 ≤ gu) (hs : 0 < s 0) (_hs1 : s 0 ≤ 1) (hsa : 0 ≤ sa 0) (hU : 0 ≤ U 0) (C : ℝ) :
    C - 2 * ga ≤ zcand ![C / s 0, -(C / s 0)] s sa U ga gu 0 := by
  have hz : zbar (K := 2) ![C / s 0, -(C / s 0)] = 0 := by
    simp [zbar, Fin.sum_univ_two]
  unfold zcand
  rw [hz]
  simp only [Matrix.cons_val_zero, sub_zero]
  rw [div_mul_cancel₀ _ hs.ne']
  have t1 : 0 ≤ ga * (2 * (sa 0 * s 0)) := by positivity
  have t2 : 0 ≤ ga * (2 * s 0) := by positivity
  have t3 : 0 ≤ gu * U 0 * s 0 := by positivity
  nlinarith

/-- Probability of the Bartleby slot in the `(K+1)`-way softmax (Eq. readout_simplex). -/
noncomputable def pBart (z : Fin K → ℝ) (zb : ℝ) : ℝ :=
  Real.exp zb / (∑ k, Real.exp (z k) + Real.exp zb)

lemma pBart_le (z : Fin K → ℝ) (zb : ℝ) (k0 : Fin K) : pBart z zb ≤ Real.exp (zb - z k0) := by
  unfold pBart
  have h1 : Real.exp (z k0) ≤ ∑ k, Real.exp (z k) :=
    Finset.single_le_sum (f := fun k => Real.exp (z k)) (fun _ _ => (Real.exp_pos _).le)
      (mem_univ k0)
  rw [div_le_iff₀ (by positivity), Real.exp_sub]
  rw [div_mul_eq_mul_div, le_div_iff₀ (Real.exp_pos _)]
  nlinarith [Real.exp_pos zb, Real.exp_pos (z k0)]

/-- Bounded-pointer lower bound: if every `z_k ≤ Z` then `P(K+1) ≥ e^{zb}/(K e^Z + e^{zb})`. -/
lemma pBart_ge (z : Fin K → ℝ) (zb Z : ℝ) (hz : ∀ k, z k ≤ Z) :
    Real.exp zb / (K * Real.exp Z + Real.exp zb) ≤ pBart z zb := by
  unfold pBart
  have h1 : ∑ k, Real.exp (z k) ≤ K * Real.exp Z := by
    calc ∑ k, Real.exp (z k) ≤ ∑ _k : Fin K, Real.exp Z :=
          Finset.sum_le_sum fun k _ => Real.exp_le_exp.mpr (hz k)
      _ = K * Real.exp Z := by simp
  apply div_le_div_of_nonneg_left (Real.exp_pos _).le (by positivity) (by linarith)

/-- **Consequence:** with unbounded pointer logits, `P_choice(K+1)` can be made `≤ η` for any
`η > 0`, even though every candidate violates a floor (any `δ`, any gains): the claimed limit
`P_choice(K+1) → 1` "regardless of raw pointer magnitudes" is false. Uses
`z_bart ≤ g log 9999` (`bnoul_range_shipped`). -/
theorem choice_not_forced (s sa U : Fin 2 → ℝ) (ga gu g : ℝ) (hga : 0 ≤ ga) (hgu : 0 ≤ gu)
    (hg : 0 ≤ g) (hs : 0 < s 0) (hs1 : s 0 ≤ 1) (hsa : 0 ≤ sa 0) (hU : 0 ≤ U 0)
    (wb : ℝ) (η : ℝ) (hη : 0 < η) :
    ∃ zp : Fin 2 → ℝ,
      pBart (fun k => zcand zp s sa U ga gu k) (zbart g (1 / 10000) wb) ≤ η := by
  set C := g * Real.log 9999 + 2 * ga - Real.log η
  refine ⟨![C / s 0, -(C / s 0)], ?_⟩
  have h1 := pBart_le (fun k => zcand ![C / s 0, -(C / s 0)] s sa U ga gu k)
    (zbart g (1 / 10000) wb) 0
  have h2 := pointer_breaks_suppression s sa U ga gu hga hgu hs hs1 hsa hU C
  have h3 : zbart g (1 / 10000) wb ≤ g * Real.log 9999 := by
    have := (zbart_range g (1 / 10000) wb hg (by norm_num) (by norm_num)).2
    have e : (1 - (1 : ℝ) / 10000) / (1 / 10000) = 9999 := by norm_num
    rwa [e] at this
  have h4 : zbart g (1 / 10000) wb - zcand ![C / s 0, -(C / s 0)] s sa U ga gu 0 ≤ Real.log η := by
    simp only [C] at h2; linarith
  calc _ ≤ _ := h1
    _ ≤ Real.exp (Real.log η) := Real.exp_le_exp.mpr h4
    _ = η := Real.exp_log hη

/-- `tradeoff_noul` as written in the paper (with the extra `· s_k^floor`) IS bounded by the gate,
hence by `e^{−γ_f δ}` under the Theorem-1(2) hypothesis, regardless of pointers. -/
theorem tradeoff_noul_paper_le (zc bn sk : ℝ) (hbn : 0 ≤ bn ∧ bn ≤ 1) (hs : 0 ≤ sk) :
    sig zc * (1 - bn) * sk ≤ sk := by
  have h1 := sig_le_one zc
  have h0 := sig_nonneg zc
  have : sig zc * (1 - bn) ≤ 1 := by nlinarith
  nlinarith

/-- The **code** head (`backbone.py`, `tradeoff_noul = σ(z^cand)·(1 − bartleby_noul)`, no
`· s^floor`) is NOT suppressed: for any `η < 1` some pointer logits give `σ(z_0^cand) > η`. -/
theorem tradeoff_noul_code_unsuppressed (s sa U : Fin 2 → ℝ) (ga gu : ℝ) (hga : 0 ≤ ga)
    (hgu : 0 ≤ gu) (hs : 0 < s 0) (hs1 : s 0 ≤ 1) (hsa : 0 ≤ sa 0) (hU : 0 ≤ U 0)
    (η : ℝ) (hη0 : 0 < η) (hη1 : η < 1) :
    ∃ zp : Fin 2 → ℝ, η < sig (zcand zp s sa U ga gu 0) := by
  set C := 2 * ga + Real.log (η / (1 - η)) + 1
  refine ⟨![C / s 0, -(C / s 0)], ?_⟩
  have h2 := pointer_breaks_suppression s sa U ga gu hga hgu hs hs1 hsa hU C
  have ho : 0 < η / (1 - η) := div_pos hη0 (by linarith)
  have h3 : sig (Real.log (η / (1 - η))) = η := by
    rw [sig_log _ ho]
    have hne : (1 : ℝ) - η ≠ 0 := by linarith
    have e1 : (1 : ℝ) + η / (1 - η) = 1 / (1 - η) := by field_simp; ring
    rw [e1, div_div_div_cancel_right₀ hne]
    simp
  have h4 : sig (Real.log (η / (1 - η))) < sig (zcand ![C / s 0, -(C / s 0)] s sa U ga gu 0) :=
    sig_strictMono (by simp only [C] at h2; linarith)
  linarith

/-! ### Part (3): "zero lexical shortcut refusal" -/

/-- **Theorem 1(3) — PROVED (as a conditional on `E`).** If some candidate clears every floor by
`δ`, then `F^floor ≥ σ(γ_f δ)^M` and `w̃_bart ≤ 1 − σ(γ_f δ)^M`, whatever `F^curr ∈ [0,1]`. -/
theorem thm1_noshortcut (hK : 0 < K) (E : Fin K → Fin M → ℝ) (amin : Fin M → ℝ) (γf δ : ℝ)
    (hγf : 0 ≤ γf) (kstar : Fin K) (hsat : ∀ m, δ ≤ E kstar m - amin m) (Fu Fc : ℝ)
    (hc : 0 ≤ Fc ∧ Fc ≤ 1) :
    sig (γf * δ) ^ M ≤ Ffloor hK γf E amin ∧
    wBart Fu Fc (Ffloor hK γf E amin) ≤ 1 - sig (γf * δ) ^ M := by
  have h1 : sig (γf * δ) ^ M ≤ Ffloor hK γf E amin :=
    le_trans (BartlebyProofs.Ceiling.sfloor_ge γf δ hγf (E kstar) amin hsat)
      (le_Fmax hK (fun k => D γf (E k) amin) kstar)
  refine ⟨h1, ?_⟩
  unfold wBart
  have hF := Fmax_le hK (fun k => D γf (E k) amin) 1 fun k => D_le_one _ _ _
  have : 0 ≤ 1 - Ffloor hK γf E amin := by unfold Ffloor; linarith
  have : (1 - Fc) * (1 - Ffloor hK γf E amin) ≤ 1 - Ffloor hK γf E amin :=
    mul_le_of_le_one_left this (by linarith)
  linarith

/-- **Theorem 1(3) at δ = 0.5, M = 4, γ_f = 10, gain 0.6**: `w_bart < 0.0266`,
`bartleby_noul < 0.104` (not `≈ 0.0025`). -/
theorem thm1_noshortcut_delta_half (hK : 0 < K) (E : Fin K → Fin 4 → ℝ) (amin : Fin 4 → ℝ)
    (kstar : Fin K) (hsat : ∀ m, 1 / 2 ≤ E kstar m - amin m) (Fu Fc : ℝ)
    (hc : 0 ≤ Fc ∧ Fc ≤ 1) :
    let w := wBart Fu Fc (Ffloor hK 10 E amin) / (1 + 1e-8)
    w < 0.0266 ∧ bnoul (3 / 5) (1 / 10000) w < 0.104 := by
  intro w
  obtain ⟨-, h2⟩ := thm1_noshortcut hK E amin 10 (1 / 2) (by norm_num) kstar hsat Fu Fc hc
  have e : (10 : ℝ) * (1 / 2) = 5 := by norm_num
  rw [e] at h2
  have h3 : (0.9933 : ℝ) ^ 4 < sig 5 ^ 4 :=
    pow_lt_pow_left₀ BartlebyProofs.Ceiling.sig5_gt (by norm_num) (by norm_num)
  have hF := Fmax_mem01 hK (fun k => D 10 (E k) amin) (fun k => ⟨D_nonneg _ _ _, D_le_one _ _ _⟩)
  have hw0 : 0 ≤ wBart Fu Fc (Ffloor hK 10 E amin) := by
    unfold wBart; apply mul_nonneg (by linarith) (by unfold Ffloor; linarith)
  have hw : w < 0.0266 := by
    simp only [w]
    rw [div_lt_iff₀ (by norm_num)]
    nlinarith
  refine ⟨hw, ?_⟩
  have hmono := bnoul_mono (3 / 5) (1 / 10000) (by norm_num) (by norm_num) (by norm_num) hw.le
  have hval : bnoul (3 / 5) (1 / 10000) 0.0266 < 0.104 := by
    rw [bnoul_35_eq _ (by norm_num) (by norm_num)]
    have := sig_35_log_lt (0.0266 / (1 - 0.0266)) 0.116 (by norm_num) (by norm_num) (by norm_num)
    have h : (0.116 : ℝ) / (1 + 0.116) < 0.104 := by norm_num
    linarith
  linarith

/-- **Theorem 1(3) at δ = 1**: `w_bart < 0.0002`, `bartleby_noul < 0.0062` (gain 0.6). -/
theorem thm1_noshortcut_delta_one (hK : 0 < K) (E : Fin K → Fin 4 → ℝ) (amin : Fin 4 → ℝ)
    (kstar : Fin K) (hsat : ∀ m, 1 ≤ E kstar m - amin m) (Fu Fc : ℝ)
    (hc : 0 ≤ Fc ∧ Fc ≤ 1) :
    let w := wBart Fu Fc (Ffloor hK 10 E amin) / (1 + 1e-8)
    w < 0.0002 ∧ bnoul (3 / 5) (1 / 10000) w < 0.0062 := by
  intro w
  obtain ⟨-, h2⟩ := thm1_noshortcut hK E amin 10 1 (by norm_num) kstar hsat Fu Fc hc
  rw [mul_one] at h2
  have h3 : (0.99995 : ℝ) ^ 4 < sig 10 ^ 4 := pow_lt_pow_left₀ sig10_gt (by norm_num) (by norm_num)
  have hF := Fmax_mem01 hK (fun k => D 10 (E k) amin) (fun k => ⟨D_nonneg _ _ _, D_le_one _ _ _⟩)
  have hw0 : 0 ≤ wBart Fu Fc (Ffloor hK 10 E amin) := by
    unfold wBart; apply mul_nonneg (by linarith) (by unfold Ffloor; linarith)
  have hw : w < 0.0002 := by
    simp only [w]
    rw [div_lt_iff₀ (by norm_num)]
    nlinarith
  refine ⟨hw, ?_⟩
  have hmono := bnoul_mono (3 / 5) (1 / 10000) (by norm_num) (by norm_num) (by norm_num) hw.le
  have hval : bnoul (3 / 5) (1 / 10000) 0.0002 < 0.0062 := by
    rw [bnoul_35_eq _ (by norm_num) (by norm_num)]
    have := sig_35_log_lt (0.0002 / (1 - 0.0002)) 0.0062 (by norm_num) (by norm_num) (by norm_num)
    have h : (0.0062 : ℝ) / (1 + 0.0062) < 0.0062 := by norm_num
    linarith
  linarith

/-- **What Theorem 1(3) does not say.** Its hypothesis is about the rubric output `E`, which is a
free function of the context features. For ANY keyword predicate on contexts there is a rubric
map that satisfies the Theorem-1(2) hypothesis exactly on keyword contexts and the
Theorem-1(3) hypothesis exactly on the others — so refusal can be driven entirely by a
keyword, *through* `E`, without contradicting Theorem 1. -/
theorem shortcut_through_E {Ctx : Type*} (kw : Ctx → Prop) [DecidablePred kw]
    (amin : Fin M → ℝ) (δ : ℝ) (hM : 0 < M) :
    ∃ Ectx : Ctx → Fin K → Fin M → ℝ,
      (∀ c, kw c → ∀ k, ∃ m, Ectx c k m - amin m ≤ -δ) ∧
      (∀ c, ¬ kw c → ∀ k m, δ ≤ Ectx c k m - amin m) := by
  refine ⟨fun c _ m => if kw c then amin m - δ else amin m + δ, ?_, ?_⟩
  · intro c hc k; exact ⟨⟨0, hM⟩, by simp [hc]⟩
  · intro c hc k m; simp [hc]

end BartlebyProofs.Floor
