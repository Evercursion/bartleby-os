import BartlebyProofs.Sigmoid

/-!
# Claim 6 — range of the intrinsic Bartleby logit and the "0.9975 / 0.0025" constants

`z_bart = g · log(p̂/(1−p̂))`, `p̂ = clamp(w_bart, ε, 1−ε)`, `bartleby_noul = σ(z_bart)`
(Eq. intrinsic_bartleby_logit; `heads.py` / `backbone.py::TorchConditionedNoulHead`,
`ε = 1e−4`, gain initialised to `0.6`).
-/

namespace BartlebyProofs.Bartleby

open Real

noncomputable def clampE (ε p : ℝ) : ℝ := max ε (min p (1 - ε))
noncomputable def logit (p : ℝ) : ℝ := Real.log (p / (1 - p))
noncomputable def zbart (g ε p : ℝ) : ℝ := g * logit (clampE ε p)
noncomputable def bnoul (g ε p : ℝ) : ℝ := sig (zbart g ε p)

lemma clampE_mem {ε : ℝ} (hε2 : ε ≤ 1 / 2) (p : ℝ) :
    ε ≤ clampE ε p ∧ clampE ε p ≤ 1 - ε := by
  unfold clampE
  constructor
  · exact le_max_left _ _
  · apply max_le (by linarith) (min_le_right _ _)

lemma clampE_mono (ε : ℝ) : Monotone (clampE ε) := by
  intro p q h
  unfold clampE
  exact max_le_max (le_refl _) (min_le_min h (le_refl _))

lemma logit_le_logit {p q : ℝ} (hp : 0 < p) (hpq : p ≤ q) (hq : q < 1) : logit p ≤ logit q := by
  unfold logit
  apply Real.log_le_log (div_pos hp (by linarith))
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

lemma logit_lt_logit {p q : ℝ} (hp : 0 < p) (hpq : p < q) (hq : q < 1) : logit p < logit q := by
  unfold logit
  apply Real.log_lt_log (div_pos hp (by linarith))
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith

lemma logit_symm (ε : ℝ) (_h0 : 0 < ε) (_h1 : ε < 1) : logit ε = - Real.log ((1 - ε) / ε) := by
  unfold logit
  rw [← Real.log_inv, inv_div]

lemma logit_top (ε : ℝ) : logit (1 - ε) = Real.log ((1 - ε) / ε) := by
  unfold logit; congr 1; ring_nf

/-- **Range of the Bartleby logit.** For `0 < ε < 1/2` and `g ≥ 0`,
`−g log((1−ε)/ε) ≤ z_bart ≤ g log((1−ε)/ε)`. -/
theorem zbart_range (g ε p : ℝ) (hg : 0 ≤ g) (h0 : 0 < ε) (h1 : ε < 1 / 2) :
    -(g * Real.log ((1 - ε) / ε)) ≤ zbart g ε p ∧ zbart g ε p ≤ g * Real.log ((1 - ε) / ε) := by
  obtain ⟨hc1, hc2⟩ := clampE_mem h1.le p
  unfold zbart
  constructor
  · have := logit_le_logit h0 hc1 (by linarith)
    rw [logit_symm ε h0 (by linarith)] at this
    nlinarith
  · have := logit_le_logit (by linarith) hc2 (by linarith)
    rw [logit_top] at this
    exact mul_le_mul_of_nonneg_left this hg

theorem bnoul_range (g ε p : ℝ) (hg : 0 ≤ g) (h0 : 0 < ε) (h1 : ε < 1 / 2) :
    sig (-(g * Real.log ((1 - ε) / ε))) ≤ bnoul g ε p ∧
    bnoul g ε p ≤ sig (g * Real.log ((1 - ε) / ε)) := by
  obtain ⟨a, b⟩ := zbart_range g ε p hg h0 h1
  exact ⟨sig_mono a, sig_mono b⟩

/-- Shipped `ε = 10⁻⁴`: `(1−ε)/ε = 9999`, so `bartleby_noul ∈ [σ(−g log 9999), σ(g log 9999)]`. -/
theorem bnoul_range_shipped (g p : ℝ) (hg : 0 ≤ g) :
    sig (-(g * Real.log 9999)) ≤ bnoul g (1 / 10000) p ∧
    bnoul g (1 / 10000) p ≤ sig (g * Real.log 9999) := by
  have h := bnoul_range g (1 / 10000) p hg (by norm_num) (by norm_num)
  have e : (1 - (1 : ℝ) / 10000) / (1 / 10000) = 9999 := by norm_num
  rw [e] at h
  exact h

lemma bnoul_mono (g ε : ℝ) (hg : 0 ≤ g) (h0 : 0 < ε) (h1 : ε < 1 / 2) : Monotone (bnoul g ε) := by
  intro p q hpq
  unfold bnoul zbart
  apply sig_mono
  apply mul_le_mul_of_nonneg_left _ hg
  obtain ⟨-, hq2⟩ := clampE_mem h1.le q
  obtain ⟨hp1, -⟩ := clampE_mem h1.le p
  exact logit_le_logit (by linarith) (clampE_mono ε hpq) (by linarith)

/-- The gain at which the saturation value equals the paper's `0.9975`. -/
noncomputable def gstar : ℝ := Real.log 399 / Real.log 9999

lemma log9999_pos : 0 < Real.log 9999 := Real.log_pos (by norm_num)

/-- `σ(g* log 9999) = 399/400 = 0.9975` exactly. -/
theorem sat_at_gstar : sig (gstar * Real.log 9999) = 0.9975 := by
  unfold gstar
  rw [div_mul_cancel₀ _ log9999_pos.ne', sig_log 399 (by norm_num)]
  norm_num

/-- and the lower saturation value is `1/400 = 0.0025` exactly. -/
theorem sat_low_at_gstar : sig (-(gstar * Real.log 9999)) = 0.0025 := by
  rw [sig_neg, sat_at_gstar]; norm_num

/-- `0.65 < g* < 0.6505` (numerically `g* ≈ 0.65025`). -/
theorem gstar_bounds : 0.65 < gstar ∧ gstar < 0.6505 := by
  have hl := log9999_pos
  unfold gstar
  constructor
  · rw [lt_div_iff₀ hl]
    have h : Real.log ((9999 : ℝ) ^ 13) < Real.log ((399 : ℝ) ^ 20) :=
      Real.log_lt_log (by norm_num) (by norm_num)
    rw [Real.log_pow, Real.log_pow] at h
    push_cast at h
    linarith
  · rw [div_lt_iff₀ hl]
    have h : Real.log ((399 : ℝ) ^ 103) < Real.log ((9999 : ℝ) ^ 67) :=
      Real.log_lt_log (by norm_num) (by norm_num)
    rw [Real.log_pow, Real.log_pow] at h
    push_cast at h
    have : (0.6505 : ℝ) * Real.log 9999 ≥ 67 / 103 * Real.log 9999 :=
      mul_le_mul_of_nonneg_right (by norm_num) hl.le
    linarith

/-- With `g = g*`, the value `0.9975` is reached **iff** `w_bart ≥ 1 − ε`, i.e. iff the clamp
is saturated: `0.9975` is a saturation constant, not a measurement. -/
theorem saturation_iff (p : ℝ) : bnoul gstar (1 / 10000) p = 0.9975 ↔ 1 - 1 / 10000 ≤ p := by
  have hg : 0 < gstar := by linarith [gstar_bounds.1]
  have e : Real.log 9999 = logit (1 - 1 / 10000) := by
    rw [logit_top]; norm_num
  constructor
  · intro h
    by_contra hp
    push Not at hp
    have hc : clampE (1 / 10000) p < 1 - 1 / 10000 := by
      unfold clampE
      apply max_lt (by norm_num)
      exact lt_of_le_of_lt (min_le_left _ _) hp
    obtain ⟨hc1, -⟩ := clampE_mem (ε := 1 / 10000) (by norm_num) p
    have hl := logit_lt_logit (by linarith) hc (by norm_num)
    have : bnoul gstar (1 / 10000) p < sig (gstar * Real.log 9999) := by
      unfold bnoul zbart
      apply sig_strictMono
      rw [e]
      exact mul_lt_mul_of_pos_left hl hg
    rw [sat_at_gstar] at this
    linarith
  · intro hp
    have hc : clampE (1 / 10000) p = 1 - 1 / 10000 := by
      unfold clampE
      rw [min_eq_right hp]
      exact max_eq_right (by norm_num)
    unfold bnoul zbart
    rw [hc, ← e, sat_at_gstar]

/-- At the initial gain `g = 0.6`, the saturation value is `≈ 0.99604`, *not* `0.9975`:
`0.99603 < σ(0.6 log 9999) < 0.99605`. -/
theorem init_gain_saturation :
    0.99603 < sig (3 / 5 * Real.log 9999) ∧ sig (3 / 5 * Real.log 9999) < 0.99605 := by
  rw [sig_mul_log _ _ (by norm_num)]
  set x : ℝ := (9999 : ℝ) ^ ((3 : ℝ) / 5) with hx
  have hxpos : 0 < x := by positivity
  have h5 : x ^ 5 = (9999 : ℝ) ^ 3 := by
    rw [hx, ← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
    norm_num
  have hlo : (251 : ℝ) < x := by
    by_contra h; push Not at h
    have := pow_le_pow_left₀ hxpos.le h 5
    rw [h5] at this; norm_num at this
  have hhi : x < 252 := by
    by_contra h; push Not at h
    have := pow_le_pow_left₀ (by norm_num) h 5
    rw [h5] at this; norm_num at this
  constructor
  · rw [lt_div_iff₀ (by positivity)]; nlinarith
  · rw [div_lt_iff₀ (by positivity)]; nlinarith

/-- Helper for numeric bounds at `g = 3/5`: `c^5 < o^3 ⇒ c/(1+c) < σ((3/5) log o)`. -/
lemma sig_35_log_gt (o c : ℝ) (ho : 0 < o) (hc : 0 < c) (h : c ^ 5 < o ^ 3) :
    c / (1 + c) < sig (3 / 5 * Real.log o) := by
  rw [sig_mul_log _ _ ho]
  set x : ℝ := o ^ ((3 : ℝ) / 5) with hx
  have hxpos : 0 < x := by positivity
  have h5 : x ^ 5 = o ^ 3 := by
    rw [hx, ← Real.rpow_natCast, ← Real.rpow_mul ho.le]; norm_num
  have hlo : c < x := by
    by_contra hh; push Not at hh
    have := pow_le_pow_left₀ hxpos.le hh 5
    rw [h5] at this; linarith
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith

lemma sig_35_log_lt (o c : ℝ) (ho : 0 < o) (hc : 0 < c) (h : o ^ 3 < c ^ 5) :
    sig (3 / 5 * Real.log o) < c / (1 + c) := by
  rw [sig_mul_log _ _ ho]
  set x : ℝ := o ^ ((3 : ℝ) / 5) with hx
  have hxpos : 0 < x := by positivity
  have h5 : x ^ 5 = o ^ 3 := by
    rw [hx, ← Real.rpow_natCast, ← Real.rpow_mul ho.le]; norm_num
  have hhi : x < c := by
    by_contra hh; push Not at hh
    have := pow_le_pow_left₀ hc.le hh 5
    rw [h5] at this; linarith
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith

end BartlebyProofs.Bartleby
