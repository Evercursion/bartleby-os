import Mathlib

/-!
# Claim 4 — regime weights (Eqs. regime_weights_1/2) and the updated aspiration
-/

namespace BartlebyProofs.Regime

noncomputable def wUp (Fu Fc _Ff : ℝ) : ℝ := Fu * Fc
noncomputable def wSet (Fu Fc _Ff : ℝ) : ℝ := (1 - Fu) * Fc
noncomputable def wRet (_Fu Fc Ff : ℝ) : ℝ := (1 - Fc) * Ff
noncomputable def wBart (_Fu Fc Ff : ℝ) : ℝ := (1 - Fc) * (1 - Ff)

/-- **The four unnormalised weights sum to 1 identically, for all real `F`'s.** -/
theorem regime_sum_one (Fu Fc Ff : ℝ) :
    wUp Fu Fc Ff + wSet Fu Fc Ff + wRet Fu Fc Ff + wBart Fu Fc Ff = 1 := by
  unfold wUp wSet wRet wBart; ring

/-- Partial sums: Regimes 1+2 carry exactly `F_curr`, Regimes 3+4 exactly `1 − F_curr`. -/
theorem regime_partial_sums (Fu Fc Ff : ℝ) :
    wUp Fu Fc Ff + wSet Fu Fc Ff = Fc ∧ wRet Fu Fc Ff + wBart Fu Fc Ff = 1 - Fc := by
  unfold wUp wSet wRet wBart; constructor <;> ring

theorem regime_nonneg (Fu Fc Ff : ℝ) (hu : 0 ≤ Fu ∧ Fu ≤ 1) (hc : 0 ≤ Fc ∧ Fc ≤ 1)
    (hf : 0 ≤ Ff ∧ Ff ≤ 1) :
    0 ≤ wUp Fu Fc Ff ∧ 0 ≤ wSet Fu Fc Ff ∧ 0 ≤ wRet Fu Fc Ff ∧ 0 ≤ wBart Fu Fc Ff := by
  unfold wUp wSet wRet wBart
  refine ⟨?_, ?_, ?_, ?_⟩ <;> apply mul_nonneg <;> linarith

/-- The code's normalisation `w̃ / (∑ w̃ + 1e−8)` is therefore division by the constant
`1 + 1e−8`: a uniform rescaling, a no-op up to a relative error of `1e−8`. -/
theorem normalized_eq (Fu Fc Ff x : ℝ) :
    x / (wUp Fu Fc Ff + wSet Fu Fc Ff + wRet Fu Fc Ff + wBart Fu Fc Ff + 1e-8)
      = x / (1 + 1e-8) := by
  rw [regime_sum_one]

theorem normalized_sum (Fu Fc Ff : ℝ) :
    let Z := wUp Fu Fc Ff + wSet Fu Fc Ff + wRet Fu Fc Ff + wBart Fu Fc Ff + 1e-8
    wUp Fu Fc Ff / Z + wSet Fu Fc Ff / Z + wRet Fu Fc Ff / Z + wBart Fu Fc Ff / Z
      = 1 / (1 + 1e-8) := by
  intro Z
  have hZ : Z = 1 + 1e-8 := by simp only [Z]; rw [regime_sum_one]
  rw [hZ, ← add_div, ← add_div, ← add_div, regime_sum_one]

theorem normalized_close (w : ℝ) (hw : 0 ≤ w ∧ w ≤ 1) : |w / (1 + 1e-8) - w| ≤ 1e-8 := by
  have : w / (1 + 1e-8) - w = -(1e-8 * w) / (1 + 1e-8) := by field_simp; ring
  rw [this, abs_div, abs_neg, abs_of_pos (by norm_num : (0:ℝ) < 1 + 1e-8)]
  rw [div_le_iff₀ (by norm_num)]
  rw [abs_of_nonneg (by nlinarith)]
  nlinarith

/-- `w̃_bart = 1 ⟺ F_curr = 0 ∧ F_floor = 0` (for `F ∈ [0,1]`). -/
theorem wBart_eq_one_iff (Fu Fc Ff : ℝ) (hc : 0 ≤ Fc ∧ Fc ≤ 1) (hf : 0 ≤ Ff ∧ Ff ≤ 1) :
    wBart Fu Fc Ff = 1 ↔ Fc = 0 ∧ Ff = 0 := by
  unfold wBart
  constructor
  · intro h
    constructor <;> nlinarith [mul_nonneg hc.1 hf.1]
  · rintro ⟨h1, h2⟩; rw [h1, h2]; ring

/-- Updated aspiration (Eq. updated_aspiration), one coordinate, with targets `aU, aP, aR, aF`. -/
noncomputable def ahat (Fu Fc Ff aU aP aR aF : ℝ) : ℝ :=
  (wUp Fu Fc Ff * aU + wSet Fu Fc Ff * aP + wRet Fu Fc Ff * aR + wBart Fu Fc Ff * aF) / (1 + 1e-8)

/-- Consequence: even if every target is `≥ a^min`, the normalised aspiration can sit strictly
below the floor (by a factor `1/(1+1e−8)`), so the Definition-3 invariant `a^min ≤ a_t` holds for
`â_t` only up to `1e−8` relative error (irrelevant in FP32, but not exact). -/
theorem ahat_below_floor (Fu Fc Ff amin : ℝ) (h : 0 < amin) :
    ahat Fu Fc Ff amin amin amin amin < amin := by
  unfold ahat
  have : wUp Fu Fc Ff * amin + wSet Fu Fc Ff * amin + wRet Fu Fc Ff * amin + wBart Fu Fc Ff * amin
      = amin := by
    have := regime_sum_one Fu Fc Ff
    linear_combination amin * this
  rw [this, div_lt_iff₀ (by norm_num)]
  nlinarith

end BartlebyProofs.Regime
