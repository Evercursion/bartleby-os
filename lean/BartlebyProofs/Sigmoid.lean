import Mathlib

/-!
# Shared: logistic sigmoid and certified numeric bounds on `exp`
-/

namespace BartlebyProofs

open Real

/-- Logistic sigmoid `σ(x) = 1 / (1 + e^{−x})`. -/
noncomputable def sig (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

lemma sig_pos (x : ℝ) : 0 < sig x := by unfold sig; positivity

lemma sig_lt_one (x : ℝ) : sig x < 1 := by
  unfold sig
  rw [div_lt_one (by positivity)]
  linarith [Real.exp_pos (-x)]

lemma sig_le_one (x : ℝ) : sig x ≤ 1 := (sig_lt_one x).le
lemma sig_nonneg (x : ℝ) : 0 ≤ sig x := (sig_pos x).le

lemma sig_strictMono : StrictMono sig := by
  intro x y h
  unfold sig
  apply one_div_lt_one_div_of_lt (by positivity)
  have : Real.exp (-y) < Real.exp (-x) := Real.exp_lt_exp.mpr (by linarith)
  linarith

lemma sig_mono : Monotone sig := sig_strictMono.monotone

lemma sig_neg (x : ℝ) : sig (-x) = 1 - sig x := by
  unfold sig
  rw [neg_neg]
  have h1 : 0 < 1 + Real.exp (-x) := by positivity
  have h2 : 0 < 1 + Real.exp x := by positivity
  have hx : Real.exp (-x) * Real.exp x = 1 := by rw [← Real.exp_add]; simp
  field_simp
  nlinarith [hx]

lemma sig_lt_half_iff (x : ℝ) : sig x < 1 / 2 ↔ x < 0 := by
  constructor
  · intro h
    by_contra hx
    have hx : 0 ≤ x := le_of_not_gt hx
    have : sig 0 ≤ sig x := sig_mono hx
    have h0 : sig 0 = 1 / 2 := by unfold sig; norm_num
    linarith
  · intro hx
    have : sig x < sig 0 := sig_strictMono hx
    have h0 : sig 0 = 1 / 2 := by unfold sig; norm_num
    linarith

/-- `σ(x) ≤ e^x`, i.e. `log σ(x) ≤ x` (the inequality used in the proof of Theorem 1). -/
lemma sig_le_exp (x : ℝ) : sig x ≤ Real.exp x := by
  unfold sig
  rw [div_le_iff₀ (by positivity)]
  have hx : Real.exp x * Real.exp (-x) = 1 := by rw [← Real.exp_add]; simp
  nlinarith [Real.exp_pos x, hx]

/-- `σ(x) ≥ 1 − e^{−x}`. -/
lemma one_sub_exp_neg_le_sig (x : ℝ) : 1 - Real.exp (-x) ≤ sig x := by
  have := sig_le_exp (-x)
  rw [sig_neg] at this
  linarith

/-- Closed form `σ(x) = e^x / (1 + e^x)`. -/
lemma sig_eq (x : ℝ) : sig x = Real.exp x / (1 + Real.exp x) := by
  unfold sig
  have hx : Real.exp (-x) = 1 / Real.exp x := by rw [Real.exp_neg]; ring
  rw [hx]
  have := Real.exp_pos x
  field_simp
  ring

/-- `σ(log o) = o / (1 + o)` for `o > 0`. -/
lemma sig_log (o : ℝ) (ho : 0 < o) : sig (Real.log o) = o / (1 + o) := by
  rw [sig_eq, Real.exp_log ho]

/-- `σ(g · log o) = o^g / (1 + o^g)` for `o > 0`. -/
lemma sig_mul_log (g o : ℝ) (ho : 0 < o) : sig (g * Real.log o) = o ^ g / (1 + o ^ g) := by
  rw [sig_eq, Real.rpow_def_of_pos ho, mul_comm]

/-- `σ(y) > c ⟺ e^{−y} < 1/c − 1` style helper: lower bound from a lower bound on `e^y`. -/
lemma sig_gt_of_exp_gt (y c : ℝ) (hc : 0 < c) (h : c < Real.exp y) :
    c / (1 + c) < sig y := by
  rw [sig_eq]
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith

lemma sig_lt_of_exp_lt (y c : ℝ) (h : Real.exp y < c) :
    sig y < c / (1 + c) := by
  rw [sig_eq]
  have := Real.exp_pos y
  rw [div_lt_div_iff₀ (by positivity) (by linarith)]
  nlinarith

/-! ### Certified bounds on `e^n` from `2.7182818283 < e < 2.7182818286` -/

lemma exp_nat_gt (n : ℕ) (hn : n ≠ 0) : (2.7182818283 : ℝ) ^ n < Real.exp n := by
  rw [← Real.exp_one_pow]
  exact pow_lt_pow_left₀ Real.exp_one_gt_d9 (by norm_num) hn

lemma exp_nat_lt (n : ℕ) (hn : n ≠ 0) : Real.exp n < (2.7182818286 : ℝ) ^ n := by
  rw [← Real.exp_one_pow]
  exact pow_lt_pow_left₀ Real.exp_one_lt_d9 (Real.exp_pos 1).le hn

lemma exp2_gt : (7.389 : ℝ) < Real.exp 2 := by
  have := exp_nat_gt 2 (by norm_num); push_cast at this; linarith [show (7.389:ℝ) < 2.7182818283^2 by norm_num]
lemma exp2_lt : Real.exp 2 < (7.3891 : ℝ) := by
  have := exp_nat_lt 2 (by norm_num); push_cast at this; linarith [show (2.7182818286:ℝ)^2 < 7.3891 by norm_num]
lemma exp4_gt : (54.598 : ℝ) < Real.exp 4 := by
  have := exp_nat_gt 4 (by norm_num); push_cast at this; linarith [show (54.598:ℝ) < 2.7182818283^4 by norm_num]
lemma exp5_gt : (148.41 : ℝ) < Real.exp 5 := by
  have := exp_nat_gt 5 (by norm_num); push_cast at this; linarith [show (148.41:ℝ) < 2.7182818283^5 by norm_num]
lemma exp5_lt : Real.exp 5 < (148.42 : ℝ) := by
  have := exp_nat_lt 5 (by norm_num); push_cast at this; linarith [show (2.7182818286:ℝ)^5 < 148.42 by norm_num]
lemma exp6_gt : (403.42 : ℝ) < Real.exp 6 := by
  have := exp_nat_gt 6 (by norm_num); push_cast at this; linarith [show (403.42:ℝ) < 2.7182818283^6 by norm_num]
lemma exp10_gt : (22026 : ℝ) < Real.exp 10 := by
  have := exp_nat_gt 10 (by norm_num); push_cast at this; linarith [show (22026:ℝ) < 2.7182818283^10 by norm_num]
lemma exp16_gt : (8886110 : ℝ) < Real.exp 16 := by
  have := exp_nat_gt 16 (by norm_num); push_cast at this; linarith [show (8886110:ℝ) < 2.7182818283^16 by norm_num]

end BartlebyProofs
