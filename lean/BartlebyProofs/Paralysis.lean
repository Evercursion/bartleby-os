import Mathlib

/-!
# Claim 1 — Eq. (1)–(2) "Trade-Off Paralysis"

We formalize, over *finite* distributions:

* `brier_decomp`, `brier_argmin_unique` : the unique minimiser over `p : ℝ` of
  `E_T[(p - y T)^2]` is the mean `E_T[y]` (Eq. (2), first line).
* `mean_mem_Icc` : if `y ∈ [0,1]` the mean lies in `[0,1]`, so minimising over `[0,1]`
  gives the same answer.
* `ce_ge_entropy`, `ce_eq_entropy_imp` : Gibbs' inequality — the unique minimiser of expected
  cross-entropy over the open simplex is the vector of marginal probabilities (Eq. (2),
  second line), with `expected_ce_eq_marginal_ce` regrouping `E_T[-log q (u* T)]`.
* `binary_mean_eq_prob`, `binary_half_iff` : the precise "≈ 0.5" condition.
* `choice_support_card_le` : the marginal choice vector has at most `|T|` non-zero entries, so
  a two-posture prior can NEVER yield `≈ 1/K` flattening for `K > 2`.
* `risk_decomp`, `conditional_mean_optimal` : with observed input `x`, the Bayes predictor is
  the *conditional* mean `E[y | x]`.
* `observable_T_no_collapse` : if `T` is a function of `x` (`w x t ≠ 0 → t = τ x`), the
  Bayes-optimal unconditioned predictor is exactly `y(x, τ x)` and achieves zero Brier risk —
  no collapse.
* `independent_T_collapse` : collapse to the prior-marginal happens exactly when `T` carries no
  information in `x` (independence).
-/

open Finset

namespace BartlebyProofs.Paralysis

section Brier

variable {T : Type*} [Fintype T]

/-- Expected (unnormalised) Brier loss `∑_t w_t (p - y_t)^2`. -/
noncomputable def brier (w y : T → ℝ) (p : ℝ) : ℝ := ∑ t, w t * (p - y t) ^ 2

/-- The weighted mean `∑ w y / ∑ w` (for a probability vector this is `E_T[y]`). -/
noncomputable def mean (w y : T → ℝ) : ℝ := (∑ t, w t * y t) / ∑ t, w t

lemma brier_expand (w y : T → ℝ) (q : ℝ) :
    brier w y q = q ^ 2 * (∑ t, w t) - 2 * q * (∑ t, w t * y t) + ∑ t, w t * y t ^ 2 := by
  unfold brier
  have h : ∀ t, w t * (q - y t) ^ 2 = q ^ 2 * w t - 2 * q * (w t * y t) + w t * y t ^ 2 := by
    intro t; ring
  simp_rw [h, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]

/-- Bias–variance decomposition of the Brier risk. -/
theorem brier_decomp (w y : T → ℝ) (hW : ∑ t, w t ≠ 0) (p : ℝ) :
    brier w y p = (∑ t, w t) * (p - mean w y) ^ 2 + brier w y (mean w y) := by
  rw [brier_expand, brier_expand]
  unfold mean
  field_simp
  ring

/-- **Eq. (2), Brier part.** For non-negative weights with positive total mass, the mean is the
unique minimiser over `ℝ` of the expected Brier loss. -/
theorem brier_argmin_unique (w y : T → ℝ) (hW : 0 < ∑ t, w t) (p : ℝ) :
    brier w y (mean w y) ≤ brier w y p ∧ (brier w y p = brier w y (mean w y) → p = mean w y) := by
  have hd := brier_decomp w y hW.ne' p
  refine ⟨?_, ?_⟩
  · rw [hd]; nlinarith [sq_nonneg (p - mean w y)]
  · intro h
    rw [hd] at h
    have : (∑ t, w t) * (p - mean w y) ^ 2 = 0 := by linarith
    rcases mul_eq_zero.mp this with h0 | h0
    · exact absurd h0 hW.ne'
    · have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h0
      linarith

/-- If `y ∈ [0,1]` the mean lies in `[0,1]`: restricting `p` to `[0,1]` changes nothing. -/
theorem mean_mem_Icc (w y : T → ℝ) (hw : ∀ t, 0 ≤ w t) (hW : 0 < ∑ t, w t)
    (hy : ∀ t, 0 ≤ y t ∧ y t ≤ 1) : 0 ≤ mean w y ∧ mean w y ≤ 1 := by
  unfold mean
  constructor
  · apply div_nonneg _ hW.le
    exact Finset.sum_nonneg fun t _ => mul_nonneg (hw t) (hy t).1
  · rw [div_le_one hW]
    apply Finset.sum_le_sum
    intro t _
    nlinarith [hw t, (hy t).2]

/-- For `y ∈ {0,1}` the Bayes predictor is the prior mass of the postures under which
`u_k` is the target. -/
theorem mean_indicator (w : T → ℝ) (A : T → Prop) [DecidablePred A] (hsum : ∑ t, w t = 1) :
    mean w (fun t => if A t then 1 else 0) = ∑ t ∈ univ.filter A, w t := by
  unfold mean
  rw [hsum, div_one, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro t _
  split_ifs with h <;> simp [h]

/-- Two latent postures `true = A` (target `u_a`), `false = B`: `p* = P(A)`. -/
theorem binary_mean_eq_prob (w : Bool → ℝ) (hsum : w true + w false = 1) :
    mean w (fun t => if t then 1 else 0) = w true := by
  unfold mean
  simp [hsum]

/-- **Precise "≈ 0.5" condition.** `|p* − 1/2| = |P(A) − 1/2|`: the unconditioned Bayes
predictor is near `0.5` iff (and only iff) the latent posture prior is near-balanced. -/
theorem binary_half_iff (w : Bool → ℝ) (hsum : w true + w false = 1) :
    |mean w (fun t => if t then 1 else 0) - 1 / 2| = |w true - 1 / 2| := by
  rw [binary_mean_eq_prob w hsum]

end Brier

section CrossEntropy

variable {ι : Type*} [Fintype ι]

/-- Expected cross-entropy `−∑ π_k log q_k`. With Mathlib's `Real.log 0 = 0`, `ce π π` is the
Shannon entropy with the convention `0 log 0 = 0`. -/
noncomputable def ce (π q : ι → ℝ) : ℝ := -∑ k, π k * Real.log (q k)

lemma ce_term (πk qk : ℝ) (hπ : 0 ≤ πk) (hq : 0 < qk) :
    πk - qk ≤ πk * Real.log πk - πk * Real.log qk := by
  rcases hπ.eq_or_lt with h | h
  · subst h; simp [hq.le]
  · have hx : 0 < qk / πk := div_pos hq h
    have := Real.log_le_sub_one_of_pos hx
    rw [Real.log_div hq.ne' h.ne'] at this
    have h2 : πk * (Real.log qk - Real.log πk) ≤ πk * (qk / πk - 1) :=
      mul_le_mul_of_nonneg_left this h.le
    rw [mul_sub, mul_sub, mul_div_cancel₀ _ h.ne', mul_one] at h2
    linarith

/-- **Gibbs' inequality (Eq. (2), choice part).** Over the open simplex, expected cross-entropy is
minimised by the marginal vector `π`. -/
theorem ce_ge_entropy (π q : ι → ℝ) (hπ : ∀ k, 0 ≤ π k) (hπs : ∑ k, π k = 1)
    (hq : ∀ k, 0 < q k) (hqs : ∑ k, q k = 1) : ce π π ≤ ce π q := by
  unfold ce
  have : ∑ k, (π k - q k) ≤ ∑ k, (π k * Real.log (π k) - π k * Real.log (q k)) :=
    Finset.sum_le_sum fun k _ => ce_term _ _ (hπ k) (hq k)
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, hπs, hqs] at this
  linarith

/-- **Uniqueness.** Equality in Gibbs forces `q = π` (in particular `π` must be interior). -/
theorem ce_eq_entropy_imp (π q : ι → ℝ) (hπ : ∀ k, 0 ≤ π k) (hπs : ∑ k, π k = 1)
    (hq : ∀ k, 0 < q k) (hqs : ∑ k, q k = 1) (heq : ce π q = ce π π) : q = π := by
  unfold ce at heq
  set f : ι → ℝ := fun k => (π k * Real.log (π k) - π k * Real.log (q k)) - (π k - q k)
  have hf : ∀ k, 0 ≤ f k := fun k => by
    have := ce_term _ _ (hπ k) (hq k); simp only [f]; linarith
  have hsum : ∑ k, f k = 0 := by
    simp only [f]
    rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib, hπs, hqs]
    linarith
  have hz : ∀ k, f k = 0 := fun k =>
    (Finset.sum_eq_zero_iff_of_nonneg (fun k _ => hf k)).mp hsum k (Finset.mem_univ k)
  funext k
  have hk := hz k
  simp only [f] at hk
  rcases (hπ k).eq_or_lt with h | h
  · rw [← h] at hk; simp at hk; exact absurd hk (hq k).ne'
  · by_contra hne
    have hx : 0 < q k / π k := div_pos (hq k) h
    have hx1 : q k / π k ≠ 1 := by
      intro h1; apply hne; field_simp at h1; linarith
    have := Real.log_lt_sub_one_of_pos hx hx1
    rw [Real.log_div (hq k).ne' h.ne'] at this
    have h2 : π k * (Real.log (q k) - Real.log (π k)) < π k * (q k / π k - 1) :=
      mul_lt_mul_of_pos_left this h
    rw [mul_sub, mul_sub, mul_div_cancel₀ _ h.ne', mul_one] at h2
    linarith

variable {T : Type*} [Fintype T] [DecidableEq ι]

/-- The marginal choice vector `π_k = P(u*(T) = k)`. -/
noncomputable def marginal (w : T → ℝ) (ustar : T → ι) (k : ι) : ℝ :=
  ∑ t ∈ univ.filter (fun t => ustar t = k), w t

/-- Regrouping: `E_T[−log q(u*(T))] = −∑_k π_k log q_k`. -/
theorem expected_ce_eq_marginal_ce (w : T → ℝ) (ustar : T → ι) (q : ι → ℝ) :
    -∑ t, w t * Real.log (q (ustar t)) = ce (marginal w ustar) q := by
  unfold ce marginal
  congr 1
  rw [← Finset.sum_fiberwise_of_maps_to (g := ustar) (t := univ) (fun _ _ => mem_univ _)]
  apply Finset.sum_congr rfl
  intro k _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro t ht
  rw [(Finset.mem_filter.mp ht).2]

/-- **Support bound.** The Bayes-optimal unconditioned choice vector has at most `|T|` non-zero
coordinates. With two latent postures it has at most two, so "choice probabilities flattened
near 1/K" (e.g. `1/4`) is *not* a consequence of a balanced two-posture prior. -/
theorem choice_support_card_le (w : T → ℝ) (ustar : T → ι) :
    (univ.filter (fun k => marginal w ustar k ≠ 0)).card ≤ Fintype.card T := by
  calc (univ.filter (fun k => marginal w ustar k ≠ 0)).card
      ≤ (univ.image ustar).card := by
        apply Finset.card_le_card
        intro k hk
        have hk := (Finset.mem_filter.mp hk).2
        unfold marginal at hk
        obtain ⟨t, ht, -⟩ := Finset.exists_ne_zero_of_sum_ne_zero hk
        exact Finset.mem_image.mpr ⟨t, mem_univ _, (Finset.mem_filter.mp ht).2⟩
    _ ≤ univ.card := Finset.card_image_le
    _ = Fintype.card T := Finset.card_univ

/-- Corollary: for `|T| = 2` and `K ≥ 3` candidates the marginal choice vector is never uniform. -/
theorem two_postures_not_uniform (w : T → ℝ) (ustar : T → ι) (hT : Fintype.card T = 2)
    (hK : 3 ≤ Fintype.card ι) : ¬ (∀ k, marginal w ustar k = 1 / Fintype.card ι) := by
  intro h
  have hc := choice_support_card_le w ustar
  have hall : univ.filter (fun k => marginal w ustar k ≠ 0) = univ := by
    apply Finset.filter_true_of_mem
    intro k _
    rw [h k]
    have : (0 : ℝ) < Fintype.card ι := by exact_mod_cast (by omega : 0 < Fintype.card ι)
    positivity
  rw [hall, Finset.card_univ, hT] at hc
  omega

end CrossEntropy

section Conditional

variable {X T : Type*} [Fintype X] [Fintype T]

/-- Brier risk of a predictor `p : X → ℝ` that sees the input `x` but not `T`. -/
noncomputable def risk (w y : X → T → ℝ) (p : X → ℝ) : ℝ := ∑ x, brier (w x) (y x) (p x)

/-- The conditional mean `E[y | x]`. -/
noncomputable def condMean (w y : X → T → ℝ) (x : X) : ℝ := mean (w x) (y x)

theorem risk_decomp (w y : X → T → ℝ) (hw : ∀ x t, 0 ≤ w x t) (p : X → ℝ) :
    risk w y p = (∑ x, (∑ t, w x t) * (p x - condMean w y x) ^ 2) + risk w y (condMean w y) := by
  unfold risk
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro x _
  by_cases hW : ∑ t, w x t = 0
  · have hz : ∀ t, w x t = 0 := fun t =>
      (Finset.sum_eq_zero_iff_of_nonneg (fun t _ => hw x t)).mp hW t (mem_univ t)
    simp [brier, hz]
  · exact brier_decomp (w x) (y x) hW (p x)

/-- The Bayes-optimal predictor given `x` is the conditional mean `E[y | x]`. -/
theorem conditional_mean_optimal (w y : X → T → ℝ) (hw : ∀ x t, 0 ≤ w x t) (p : X → ℝ) :
    risk w y (condMean w y) ≤ risk w y p := by
  rw [risk_decomp w y hw p]
  have : 0 ≤ ∑ x, (∑ t, w x t) * (p x - condMean w y x) ^ 2 :=
    Finset.sum_nonneg fun x _ =>
      mul_nonneg (Finset.sum_nonneg fun t _ => hw x t) (sq_nonneg _)
  linarith

/-- **Key caveat (no collapse when `T` is observable).** If the latent posture is a function
`τ` of the observed input (`w x t ≠ 0 → t = τ x`), the Bayes-optimal unconditioned predictor is
`y(x, τ x)` exactly (on the support), and that predictor has zero Brier risk; hence any
risk-minimiser equals `y(x, τ x)` on every `x` of positive mass. Proper scoring per se does not
force `≈ 0.5`; unobservability of `T` from `x` does. -/
theorem observable_T_no_collapse (w y : X → T → ℝ) (hw : ∀ x t, 0 ≤ w x t) (τ : X → T)
    (hτ : ∀ x t, w x t ≠ 0 → t = τ x) :
    (∀ x, ∑ t, w x t ≠ 0 → condMean w y x = y x (τ x)) ∧
    risk w y (fun x => y x (τ x)) = 0 ∧
    (∀ p : X → ℝ, risk w y p = 0 → ∀ x, w x (τ x) ≠ 0 → p x = y x (τ x)) := by
  have hsingle : ∀ x (f : T → ℝ), ∑ t, w x t * f t = w x (τ x) * f (τ x) := by
    intro x f
    rw [Finset.sum_eq_single (τ x)]
    · intro t _ ht
      by_cases h : w x t = 0
      · simp [h]
      · exact absurd (hτ x t h) ht
    · simp
  refine ⟨?_, ?_, ?_⟩
  · intro x hW
    unfold condMean mean
    have h1 := hsingle x (y x)
    have h2 := hsingle x (fun _ => 1)
    simp only [mul_one] at h2
    rw [h1, h2]
    rw [h2] at hW
    field_simp
  · unfold risk brier
    apply Finset.sum_eq_zero
    intro x _
    rw [hsingle x (fun t => (y x (τ x) - y x t) ^ 2)]
    simp
  · intro p hp x hx
    unfold risk at hp
    have hnn : ∀ x, 0 ≤ brier (w x) (y x) (p x) := fun x =>
      Finset.sum_nonneg fun t _ => mul_nonneg (hw x t) (sq_nonneg _)
    have hx0 := (Finset.sum_eq_zero_iff_of_nonneg (fun x _ => hnn x)).mp hp x (mem_univ x)
    unfold brier at hx0
    rw [hsingle x (fun t => (p x - y x t) ^ 2)] at hx0
    rcases mul_eq_zero.mp hx0 with h | h
    · exact absurd h hx
    · have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h
      linarith

omit [Fintype X] in
/-- **Collapse occurs under independence.** If `w x t = ρ x · ν t` (T independent of x, prior
`ν`), the Bayes predictor is the prior-marginal `∑_t ν_t y(x,t)` — this, not proper scoring, is
the source of "paralysis". -/
theorem independent_T_collapse (ρ : X → ℝ) (ν : T → ℝ) (y : X → T → ℝ) (hν : ∑ t, ν t = 1)
    (x : X) (hρ : ρ x ≠ 0) :
    condMean (fun x t => ρ x * ν t) y x = ∑ t, ν t * y x t := by
  unfold condMean mean
  have h1 : ∑ t, ρ x * ν t * y x t = ρ x * ∑ t, ν t * y x t := by
    rw [Finset.mul_sum]; apply Finset.sum_congr rfl; intro t _; ring
  rw [h1, ← Finset.mul_sum, hν, mul_one]
  field_simp

end Conditional

end BartlebyProofs.Paralysis
