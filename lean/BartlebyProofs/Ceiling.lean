import BartlebyProofs.Sigmoid

/-!
# Claim 3 — Proposition 2 "Interior Ceiling Bound"
-/

namespace BartlebyProofs.Ceiling

open Finset

/-- Expected ordinal score `E = ∑_ℓ ℓ p_ℓ` on levels `{0,…,L−1}`. -/
noncomputable def expScore {L : ℕ} (p : Fin L → ℝ) : ℝ := ∑ ℓ : Fin L, ((ℓ : ℕ) : ℝ) * p ℓ

/-- **Prop. 2(a).** For an interior distribution (`p_ℓ > 0`, `∑ p = 1`) and `L ≥ 2`,
`E < L − 1` strictly. (For `L = 1` it is false: `E = 0 = L − 1`.) -/
theorem expScore_lt_top (L : ℕ) (hL : 2 ≤ L) (p : Fin L → ℝ) (hp : ∀ ℓ, 0 < p ℓ)
    (hs : ∑ ℓ, p ℓ = 1) : expScore p < (L : ℝ) - 1 := by
  have key : ∑ ℓ : Fin L, ((L : ℝ) - 1 - ((ℓ : ℕ) : ℝ)) * p ℓ = ((L : ℝ) - 1) - expScore p := by
    unfold expScore
    have : ∑ ℓ : Fin L, ((L : ℝ) - 1 - ((ℓ : ℕ) : ℝ)) * p ℓ
        = ((L : ℝ) - 1) * ∑ ℓ, p ℓ - ∑ ℓ : Fin L, ((ℓ : ℕ) : ℝ) * p ℓ := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl; intro ℓ _; ring
    rw [this, hs, mul_one]
  have hpos : 0 < ∑ ℓ : Fin L, ((L : ℝ) - 1 - ((ℓ : ℕ) : ℝ)) * p ℓ := by
    apply Finset.sum_pos'
    · intro ℓ _
      have h1 : ((ℓ : ℕ) : ℝ) + 1 ≤ (L : ℝ) := by exact_mod_cast ℓ.isLt
      exact mul_nonneg (by linarith) (hp ℓ).le
    · refine ⟨⟨0, by omega⟩, mem_univ _, ?_⟩
      have h2 : (2 : ℝ) ≤ L := by exact_mod_cast hL
      exact mul_pos (by simp; linarith) (hp _)
  linarith

/-- **Prop. 2(b), round-half-up** (Mathlib's `round x = ⌊x + 1/2⌋`): for `E < L − 1`,
`round E = L − 1 ↔ E ≥ L − 1.5`. -/
theorem round_top_iff (L : ℕ) (E : ℝ) (hE : E < (L : ℝ) - 1) :
    round E = (L : ℤ) - 1 ↔ (L : ℝ) - 3 / 2 ≤ E := by
  rw [round_eq, Int.floor_eq_iff]
  push_cast
  constructor
  · rintro ⟨h1, -⟩; linarith
  · intro h; constructor <;> linarith

/-- Round-half-to-even (the convention of Python's `round`, `numpy.round`, `torch.round`). -/
noncomputable def roundHalfEven (x : ℝ) : ℤ :=
  if Int.fract x = 1 / 2 then (if Even ⌊x⌋ then ⌊x⌋ else ⌊x⌋ + 1) else round x

/-- **Prop. 2(b), half-even, L = 5 (shipped).** The paper's "iff" also holds. -/
theorem roundHalfEven_top_iff_L5 (E : ℝ) (hE : E < 4) :
    roundHalfEven E = 4 ↔ 7 / 2 ≤ E := by
  unfold roundHalfEven
  have hfl : (⌊E⌋ : ℝ) + Int.fract E = E := Int.floor_add_fract E
  split_ifs with hfr he
  · rw [hfr] at hfl
    constructor
    · intro h; rw [h] at hfl; push_cast at hfl; linarith
    · intro h
      have h1 : (⌊E⌋ : ℝ) < 4 := by linarith
      have h2 : (3 : ℝ) ≤ ⌊E⌋ := by linarith
      have h1' : ⌊E⌋ < 4 := by exact_mod_cast h1
      have h2' : 3 ≤ ⌊E⌋ := by exact_mod_cast h2
      have h3 : ⌊E⌋ = 3 := by omega
      rw [h3] at he; exact absurd he (by decide)
  · rw [hfr] at hfl
    constructor
    · intro h
      have h3 : ⌊E⌋ = 3 := by omega
      rw [h3] at hfl; push_cast at hfl; linarith
    · intro h
      have h1 : (⌊E⌋ : ℝ) < 4 := by linarith
      have h2 : (3 : ℝ) ≤ ⌊E⌋ := by linarith
      have h1' : ⌊E⌋ < 4 := by exact_mod_cast h1
      have h2' : 3 ≤ ⌊E⌋ := by exact_mod_cast h2
      omega
  · have := round_top_iff 5 E (by norm_num; linarith)
    norm_num at this
    exact this

/-- **Prop. 2(b), half-even, even `L` — REFUTED.** For `L = 4` the interior distribution
`p = (0.05, 0.15, 0.05, 0.75)` has `E = 2.5 = L − 1.5`, yet `roundHalfEven E = 2 ≠ L − 1 = 3`. -/
theorem roundHalfEven_counterexample_L4 :
    let p : Fin 4 → ℝ := ![0.05, 0.15, 0.05, 0.75]
    (∀ ℓ, 0 < p ℓ) ∧ ∑ ℓ, p ℓ = 1 ∧ expScore p = 4 - 3 / 2 ∧ roundHalfEven (expScore p) = 2 := by
  intro p
  have hE : expScore p = 5 / 2 := by
    simp [expScore, Fin.sum_univ_four, p]; norm_num
  have hfloor : ⌊(5 / 2 : ℝ)⌋ = 2 := by
    rw [Int.floor_eq_iff]; norm_num
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro ℓ; fin_cases ℓ <;> simp [p] <;> norm_num
  · simp [Fin.sum_univ_four, p]; norm_num
  · rw [hE]; norm_num
  · rw [hE]
    unfold roundHalfEven
    have hfr : Int.fract (5 / 2 : ℝ) = 1 / 2 := by
      rw [Int.fract, hfloor]; norm_num
    rw [if_pos hfr, hfloor, if_pos (by decide)]

/-- **Prop. 2(c).** At an unshifted ceiling floor `a^min = L − 1`, every interior output gives
`σ(γ (E − a^min)) < 1/2` (any `γ > 0`). -/
theorem sig_ceiling_lt_half (L : ℕ) (hL : 2 ≤ L) (p : Fin L → ℝ) (hp : ∀ ℓ, 0 < p ℓ)
    (hs : ∑ ℓ, p ℓ = 1) (γ : ℝ) (hγ : 0 < γ) :
    sig (γ * (expScore p - ((L : ℝ) - 1))) < 1 / 2 := by
  rw [sig_lt_half_iff]
  exact mul_neg_of_pos_of_neg hγ (by linarith [expScore_lt_top L hL p hp hs])

/-- Survival-floor gate `s_floor = ∏_m σ(γ (E_m − a^min_m))` (Eq. soft_dominance). -/
noncomputable def sfloor {M : ℕ} (γ : ℝ) (E amin : Fin M → ℝ) : ℝ :=
  ∏ m, sig (γ * (E m - amin m))

/-- Monotone lower bound: margins `≥ δ` on all `M` goals give `s_floor ≥ σ(γδ)^M`. -/
theorem sfloor_ge {M : ℕ} (γ δ : ℝ) (hγ : 0 ≤ γ) (E amin : Fin M → ℝ)
    (h : ∀ m, δ ≤ E m - amin m) : sig (γ * δ) ^ M ≤ sfloor γ E amin := by
  unfold sfloor
  calc sig (γ * δ) ^ M = ∏ _m : Fin M, sig (γ * δ) := by simp
    _ ≤ ∏ m, sig (γ * (E m - amin m)) := by
      apply Finset.prod_le_prod
      · intro m _; exact sig_nonneg _
      · intro m _; exact sig_mono (mul_le_mul_of_nonneg_left (h m) hγ)

lemma sig5_gt : (0.9933 : ℝ) < sig 5 := by
  have := sig_gt_of_exp_gt 5 148.41 (by norm_num) exp5_gt
  have h : (0.9933 : ℝ) < 148.41 / (1 + 148.41) := by norm_num
  linarith

lemma sig5_lt : sig 5 < (0.99331 : ℝ) := by
  have := sig_lt_of_exp_lt 5 148.42 exp5_lt
  have h : (148.42 : ℝ) / (1 + 148.42) < 0.99331 := by norm_num
  linarith

lemma sig6_gt : (0.997527 : ℝ) < sig 6 := by
  have := sig_gt_of_exp_gt 6 403.42 (by norm_num) exp6_gt
  have h : (0.997527 : ℝ) < 403.42 / (1 + 403.42) := by norm_num
  linarith

/-- **Prop. 2(d), per-goal reading — PROVED.** `E ∈ [3.5, 4)`, `a^min ≤ 3`, `γ = 10` give
`σ(10 (E − a^min)) > 0.99` (margin is `≥ 0.5`, not `> 0.5` as written: `E = 3.5, a = 3`). -/
theorem per_goal_gate_gt (E a : ℝ) (hE : 7 / 2 ≤ E) (ha : a ≤ 3) :
    (0.99 : ℝ) < sig (10 * (E - a)) := by
  have : sig 5 ≤ sig (10 * (E - a)) := sig_mono (by linarith)
  linarith [sig5_gt]

/-- **Prop. 2(d) for the product `s_floor` with M = 4 — REFUTED.** All four goals at
`E_m = 3.5`, `a^min_m = 3` (allowed: `a^min ≤ L − 2`, `E ∈ [3.5, 4)`) give
`s_floor = σ(5)^4 < 0.9736 < 0.99`. -/
theorem product_gate_counterexample :
    sfloor (M := 4) 10 (fun _ => 7 / 2) (fun _ => 3) < 0.9736 := by
  unfold sfloor
  have h : (10 : ℝ) * (7 / 2 - 3) = 5 := by norm_num
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  rw [h]
  have h0 := sig_pos 5
  have h1 := sig5_lt
  calc sig 5 ^ 4 < (0.99331 : ℝ) ^ 4 := pow_lt_pow_left₀ h1 h0.le (by norm_num)
    _ < 0.9736 := by norm_num

theorem product_gate_counterexample_lt_099 :
    ¬ (∀ (E a : Fin 4 → ℝ), (∀ m, 7 / 2 ≤ E m ∧ E m < 4 ∧ a m ≤ 3) → 0.99 < sfloor 10 E a) := by
  intro h
  have := h (fun _ => 7 / 2) (fun _ => 3) (fun _ => by norm_num)
  linarith [product_gate_counterexample]

/-- Already for `M = 2` the product claim fails: `σ(5)^M < 0.99` for every `M ≥ 2`. -/
theorem sig5_pow_lt (M : ℕ) (hM : 2 ≤ M) : sig 5 ^ M < 0.99 := by
  have h0 := sig_pos 5
  have h1 := sig_lt_one 5
  calc sig 5 ^ M ≤ sig 5 ^ 2 := pow_le_pow_of_le_one h0.le h1.le hM
    _ < (0.99331 : ℝ) ^ 2 := pow_lt_pow_left₀ sig5_lt h0.le (by norm_num)
    _ < 0.99 := by norm_num

/-- **Correct statement 1 (what the hypothesis actually guarantees for M = 4):**
`s_floor ≥ σ(5)^4 > 0.973`. -/
theorem product_gate_lower_bound (E a : Fin 4 → ℝ) (h : ∀ m, 7 / 2 ≤ E m ∧ a m ≤ 3) :
    (0.973 : ℝ) < sfloor 10 E a := by
  have hb := sfloor_ge 10 (1 / 2) (by norm_num) E a (fun m => by linarith [(h m).1, (h m).2])
  have : (10 : ℝ) * (1 / 2) = 5 := by norm_num
  rw [this] at hb
  have h2 : (0.9933 : ℝ) ^ 4 < sig 5 ^ 4 := pow_lt_pow_left₀ sig5_gt (by norm_num) (by norm_num)
  have h3 : (0.973 : ℝ) < 0.9933 ^ 4 := by norm_num
  linarith

/-- **Correct statement 2:** with M = 4 goals and `γ = 10`, margins `≥ 0.6` on every goal
suffice for `s_floor > 0.99`. -/
theorem product_gate_gt_099 (E a : Fin 4 → ℝ) (h : ∀ m, 3 / 5 ≤ E m - a m) :
    (0.99 : ℝ) < sfloor 10 E a := by
  have hb := sfloor_ge 10 (3 / 5) (by norm_num) E a h
  have : (10 : ℝ) * (3 / 5) = 6 := by norm_num
  rw [this] at hb
  have h2 : (0.997527 : ℝ) ^ 4 < sig 6 ^ 4 := pow_lt_pow_left₀ sig6_gt (by norm_num) (by norm_num)
  have h3 : (0.99 : ℝ) < 0.997527 ^ 4 := by norm_num
  linarith

end BartlebyProofs.Ceiling
