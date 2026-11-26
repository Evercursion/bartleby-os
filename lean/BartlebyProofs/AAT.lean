import Mathlib

/-!
# Claim 2 — Proposition 1 (finite termination and the four regimes)

Grid `G = ∏_m {0,…,L_m − 1}` (vectors `a : Fin M → ℕ` with `a m < L m`), survival floor
`amin`, urgency order `urg a` (most urgent first; a list that is assumed to contain every goal
where needed: `UrgComplete`), retreat variable `ret a`. Grid rules `R_grid` are subsumed by
letting `urg`, `ret` be arbitrary functions of the grid point.

Feasibility is an arbitrary Boolean predicate on grid points that is *antitone*
(`a ≤ b → feas b → feas a`); `feasOf` (integer scores) and `feasReal` (real scores, the paper's
`G ∈ ℝ^{K×M}`) are instances (`feasOf_antitone`, `feasReal_antitone`).

Rules (paper §2, incl. the floor-bounded Downward rule of line ~221):
* Downward (infeasible `a`): lower `ret a` if it has slack over `amin`; otherwise lower the
  first slack coordinate in reverse urgency order; if no coordinate has slack, stop (Bartleby).
* Upward (feasible `a`): raise the most urgent `m` with `a m + 1 < L m` and `feas (a + e_m)`.
* End: feasible and no upward step.

Three operators are formalised:
* `stepV1` — Selten-faithful: the rule applicable at the *current* point is applied at every
  step (so Upward steps can follow a retreat);
* `twoPhase` — the operator described in the paper's proof: Phase 1 (downward only) if `a_{t−1}`
  is infeasible, Phase 2 (upward only) if it is feasible;
* `oracleStep` — what `oracle.py::adapt_preference_table` actually does: ONE unit step per
  Markov transition.
-/

namespace BartlebyProofs.AAT

abbrev Vec (M : ℕ) := Fin M → ℕ

structure Scheme (M : ℕ) where
  L : Fin M → ℕ
  amin : Vec M
  urg : Vec M → List (Fin M)
  ret : Vec M → Fin M

variable {M : ℕ}

def inc (a : Vec M) (m : Fin M) : Vec M := Function.update a m (a m + 1)
def dec (a : Vec M) (m : Fin M) : Vec M := Function.update a m (a m - 1)

/-- Most urgent feasible upward coordinate. -/
def upChoice (S : Scheme M) (feas : Vec M → Bool) (a : Vec M) : Option (Fin M) :=
  (S.urg a).find? (fun m => decide (a m + 1 < S.L m) && feas (inc a m))

/-- Floor-bounded retreat coordinate: `ret a` while it has slack, otherwise the first slack
coordinate in reverse urgency order. -/
def downChoice (S : Scheme M) (a : Vec M) : Option (Fin M) :=
  if S.amin (S.ret a) < a (S.ret a) then some (S.ret a)
  else (S.urg a).reverse.find? (fun m => decide (S.amin m < a m))

/-- Selten-faithful step: the rule applicable at the current point. `none` = halt. -/
def stepV1 (S : Scheme M) (feas : Vec M → Bool) (a : Vec M) : Option (Vec M) :=
  if feas a then (upChoice S feas a).map (inc a) else (downChoice S a).map (dec a)

def downStep (S : Scheme M) (feas : Vec M → Bool) (a : Vec M) : Option (Vec M) :=
  if feas a then none else (downChoice S a).map (dec a)

def upStep (S : Scheme M) (feas : Vec M → Bool) (a : Vec M) : Option (Vec M) :=
  if feas a then (upChoice S feas a).map (inc a) else none

/-- Iterate a partial step function with fuel; returns final point and number of steps taken. -/
def run {α : Type*} (step : α → Option α) : ℕ → α → α × ℕ
  | 0, a => (a, 0)
  | n + 1, a =>
    match step a with
    | none => (a, 0)
    | some a' => ((run step n a').1, (run step n a').2 + 1)

/-- The two-phase operator of the paper's proof of Proposition 1. -/
def twoPhase (S : Scheme M) (feas : Vec M → Bool) (fuel : ℕ) (a0 : Vec M) : Vec M × ℕ :=
  if feas a0 then run (upStep S feas) fuel a0 else run (downStep S feas) fuel a0

/-- The single Markov step of `oracle.py` (one unit move; Bartleby check first). -/
def oracleStep (S : Scheme M) (feas : Vec M → Bool) (a0 : Vec M) : Vec M :=
  if feas S.amin then
    match stepV1 S feas a0 with
    | none => a0
    | some a' => a'
  else a0

/-! ### Feasibility predicates -/

def AntitoneFeas (feas : Vec M → Bool) : Prop :=
  ∀ a b : Vec M, a ≤ b → feas b = true → feas a = true

def UrgComplete (S : Scheme M) : Prop := ∀ a m, m ∈ S.urg a

/-- Satisficing-set non-emptiness for integer scores `G` (list of candidate score vectors). -/
def feasOf (G : List (Fin M → ℕ)) (a : Vec M) : Bool := G.any (fun g => decide (∀ m, a m ≤ g m))

theorem feasOf_antitone (G : List (Fin M → ℕ)) : AntitoneFeas (feasOf G) := by
  intro a b hab hb
  simp only [feasOf, List.any_eq_true, decide_eq_true_eq] at *
  obtain ⟨g, hg, h⟩ := hb
  exact ⟨g, hg, fun m => le_trans (hab m) (h m)⟩

open Classical in
/-- Satisficing-set non-emptiness for real scores `G ∈ ℝ^{K×M}` (paper's setting). -/
noncomputable def feasReal {K : ℕ} (G : Fin K → Fin M → ℝ) (a : Vec M) : Bool :=
  decide (∃ k, ∀ m, (a m : ℝ) ≤ G k m)

theorem feasReal_antitone {K : ℕ} (G : Fin K → Fin M → ℝ) : AntitoneFeas (feasReal G) := by
  intro a b hab hb
  simp only [feasReal, decide_eq_true_eq] at *
  obtain ⟨k, hk⟩ := hb
  exact ⟨k, fun m => le_trans (by exact_mod_cast hab m) (hk m)⟩

/-! ### Basic lemmas -/

lemma inc_apply_self (a : Vec M) (j : Fin M) : inc a j j = a j + 1 := by simp [inc]
lemma dec_apply_self (a : Vec M) (j : Fin M) : dec a j j = a j - 1 := by simp [dec]
lemma inc_apply_ne (a : Vec M) {j m : Fin M} (h : m ≠ j) : inc a j m = a m := by
  simp [inc, Function.update_of_ne h]
lemma dec_apply_ne (a : Vec M) {j m : Fin M} (h : m ≠ j) : dec a j m = a m := by
  simp [dec, Function.update_of_ne h]

lemma le_inc (a : Vec M) (j : Fin M) : a ≤ inc a j := by
  rw [Pi.le_def]
  intro m
  by_cases h : m = j
  · subst h; rw [inc_apply_self]; omega
  · rw [inc_apply_ne a h]

lemma inc_ne (a : Vec M) (j : Fin M) : inc a j ≠ a := by
  intro h
  have := congrFun h j
  rw [inc_apply_self] at this; omega

lemma dec_le (a : Vec M) (j : Fin M) : dec a j ≤ a := by
  rw [Pi.le_def]
  intro m
  by_cases h : m = j
  · subst h; rw [dec_apply_self]; omega
  · rw [dec_apply_ne a h]

lemma le_dec {b a : Vec M} {j : Fin M} (hba : b ≤ a) (hj : b j < a j) : b ≤ dec a j := by
  rw [Pi.le_def] at hba ⊢
  intro m
  by_cases h : m = j
  · subst h; rw [dec_apply_self]; have := hba m; omega
  · rw [dec_apply_ne a h]; exact hba m

/-- Downward slack potential `Φ(a) = ∑_m (a_m − a^min_m)`. -/
def slack (S : Scheme M) (a : Vec M) : ℕ := ∑ m, (a m - S.amin m)
/-- Upward headroom `∑_m (L_m − 1 − a_m)`. -/
def headroom (S : Scheme M) (a : Vec M) : ℕ := ∑ m, (S.L m - 1 - a m)

lemma slack_dec_lt (S : Scheme M) (a : Vec M) {j : Fin M} (hj : S.amin j < a j) :
    slack S (dec a j) < slack S a := by
  unfold slack
  apply Finset.sum_lt_sum
  · intro m _
    by_cases h : m = j
    · subst h; rw [dec_apply_self]; omega
    · rw [dec_apply_ne a h]
  · exact ⟨j, Finset.mem_univ _, by rw [dec_apply_self]; omega⟩

lemma headroom_inc_lt (S : Scheme M) (a : Vec M) {j : Fin M} (hj : a j + 1 < S.L j) :
    headroom S (inc a j) < headroom S a := by
  unfold headroom
  apply Finset.sum_lt_sum
  · intro m _
    by_cases h : m = j
    · subst h; rw [inc_apply_self]; omega
    · rw [inc_apply_ne a h]
  · exact ⟨j, Finset.mem_univ _, by rw [inc_apply_self]; omega⟩

lemma slack_pos (S : Scheme M) (a : Vec M) {j : Fin M} (hj : S.amin j < a j) :
    0 < slack S a := by
  unfold slack
  exact lt_of_lt_of_le (by omega : 0 < a j - S.amin j)
    (Finset.single_le_sum (f := fun m => a m - S.amin m) (fun _ _ => Nat.zero_le _)
      (Finset.mem_univ j))

lemma headroom_le_of_ge (S : Scheme M) {a b : Vec M} (h : b ≤ a) :
    headroom S a ≤ headroom S b :=
  Finset.sum_le_sum fun m _ => by have : b m ≤ a m := h m; omega

lemma downChoice_slack (S : Scheme M) (a : Vec M) {j : Fin M} (h : downChoice S a = some j) :
    S.amin j < a j := by
  unfold downChoice at h
  split_ifs at h with h1
  · cases h; exact h1
  · have := List.find?_some h
    simpa using this

lemma downChoice_none (S : Scheme M) (hU : UrgComplete S) (a : Vec M)
    (h : downChoice S a = none) : a ≤ S.amin := by
  unfold downChoice at h
  split_ifs at h with h1
  rw [List.find?_eq_none] at h
  rw [Pi.le_def]
  intro m
  have := h m (List.mem_reverse.mpr (hU a m))
  simp at this
  exact this

lemma upChoice_spec (S : Scheme M) (feas : Vec M → Bool) (a : Vec M) {j : Fin M}
    (h : upChoice S feas a = some j) : a j + 1 < S.L j ∧ feas (inc a j) = true := by
  unfold upChoice at h
  have := List.find?_some h
  simpa using this

/-- The set `F↑(a)` of feasible upward steps is non-empty. -/
def UpNonempty (S : Scheme M) (feas : Vec M → Bool) (a : Vec M) : Prop :=
  ∃ m, a m + 1 < S.L m ∧ feas (inc a m) = true

lemma upChoice_none_iff (S : Scheme M) (hU : UrgComplete S) (feas : Vec M → Bool) (a : Vec M) :
    upChoice S feas a = none ↔ ¬ UpNonempty S feas a := by
  constructor
  · intro h ⟨m, hm1, hm2⟩
    unfold upChoice at h
    rw [List.find?_eq_none] at h
    have := h m (hU a m)
    simp [hm1, hm2] at this
  · intro h
    cases hc : upChoice S feas a with
    | none => rfl
    | some j => exact absurd ⟨j, upChoice_spec S feas a hc⟩ h

lemma upChoice_none_of_not (S : Scheme M) (feas : Vec M → Bool) (a : Vec M)
    (h : ¬ UpNonempty S feas a) : upChoice S feas a = none := by
  cases hc : upChoice S feas a with
  | none => rfl
  | some j => exact absurd ⟨j, upChoice_spec S feas a hc⟩ h

/-! ### Generic termination lemma for `run` -/

theorem run_spec {α : Type*} (step : α → Option α) (μ : α → ℕ) (I : α → Prop)
    (hI : ∀ a a', I a → step a = some a' → I a')
    (hμ : ∀ a a', I a → step a = some a' → μ a' < μ a) :
    ∀ n a, I a → μ a ≤ n →
      step (run step n a).1 = none ∧ (run step n a).2 ≤ μ a ∧ I (run step n a).1 := by
  intro n
  induction n with
  | zero =>
    intro a ha hle
    cases h : step a with
    | none => simp [run, h, ha]
    | some a' => exact absurd (hμ a a' ha h) (by omega)
  | succ n ih =>
    intro a ha hle
    cases h : step a with
    | none => simp [run, h, ha]
    | some a' =>
      have h1 := hμ a a' ha h
      have h2 := hI a a' ha h
      obtain ⟨r1, r2, r3⟩ := ih a' h2 (by omega)
      simp only [run, h]
      exact ⟨r1, by omega, r3⟩

lemma run_of_none {α : Type*} (step : α → Option α) (a : α) (h : step a = none) (n : ℕ) :
    run step n a = (a, 0) := by
  cases n with
  | zero => rfl
  | succ n => simp [run, h]

lemma run_fst_of_some {α : Type*} (step : α → Option α) (a a' : α) (h : step a = some a')
    (n : ℕ) : run step (n + 1) a = ((run step n a').1, (run step n a').2 + 1) := by
  simp [run, h]

/-! ### (a)+(b) for the paper's two-phase operator: termination and `τ ≤ ∑ (L_m − 1)` -/

section TwoPhase

variable (S : Scheme M) (feas : Vec M → Bool)

lemma upStep_spec {a a' : Vec M} (h : upStep S feas a = some a') :
    ∃ j, a' = inc a j ∧ a j + 1 < S.L j ∧ feas a = true ∧ feas a' = true := by
  unfold upStep at h
  split_ifs at h with hf
  cases hc : upChoice S feas a with
  | none => simp [hc] at h
  | some j =>
    simp [hc] at h
    obtain ⟨h1, h2⟩ := upChoice_spec S feas a hc
    exact ⟨j, h.symm, h1, hf, h ▸ h2⟩

lemma downStep_spec {a a' : Vec M} (h : downStep S feas a = some a') :
    ∃ j, a' = dec a j ∧ S.amin j < a j ∧ feas a = false := by
  unfold downStep at h
  split_ifs at h with hf
  cases hc : downChoice S a with
  | none => simp [hc] at h
  | some j =>
    simp [hc] at h
    exact ⟨j, h.symm, downChoice_slack S a hc, by simpa using hf⟩

lemma upStep_none {a : Vec M} (hf : feas a = true) (h : upStep S feas a = none) :
    upChoice S feas a = none := by
  unfold upStep at h
  rw [if_pos hf] at h
  cases hc : upChoice S feas a with
  | none => rfl
  | some j => rw [hc] at h; simp at h

lemma headroom_le_total (a : Vec M) : headroom S a ≤ ∑ m, (S.L m - 1) :=
  Finset.sum_le_sum fun m _ => by omega

lemma slack_le_total (a : Vec M) (hg : ∀ m, a m < S.L m) : slack S a ≤ ∑ m, (S.L m - 1) :=
  Finset.sum_le_sum fun m _ => by have := hg m; omega

/-- **Prop. 1(a,b) for the two-phase operator.** It halts, and the number of unit steps is at
most `∑_m (L_m − 1)`. -/
theorem twoPhase_terminates_bound (a0 : Vec M) (hg : ∀ m, a0 m < S.L m) (N : ℕ)
    (hN : ∑ m, (S.L m - 1) ≤ N) :
    (twoPhase S feas N a0).2 ≤ ∑ m, (S.L m - 1) ∧
    (if feas a0 then upStep S feas (twoPhase S feas N a0).1 = none
     else downStep S feas (twoPhase S feas N a0).1 = none) := by
  unfold twoPhase
  by_cases hf : feas a0 = true
  · simp only [hf, if_true]
    have := run_spec (upStep S feas) (headroom S) (fun _ => True)
      (fun _ _ _ _ => trivial)
      (fun a a' _ h => by
        obtain ⟨j, rfl, hj, -, -⟩ := upStep_spec S feas h
        exact headroom_inc_lt S a hj)
      N a0 trivial (le_trans (headroom_le_total S a0) hN)
    exact ⟨le_trans this.2.1 (headroom_le_total S a0), this.1⟩
  · simp only [hf, if_false, Bool.false_eq_true]
    have := run_spec (downStep S feas) (slack S) (fun _ => True)
      (fun _ _ _ _ => trivial)
      (fun a a' _ h => by
        obtain ⟨j, rfl, hj, -⟩ := downStep_spec S feas h
        exact slack_dec_lt S a hj)
      N a0 trivial (le_trans (slack_le_total S a0 hg) hN)
    exact ⟨le_trans this.2.1 (slack_le_total S a0 hg), this.1⟩

/-! ### (c) The four regimes are exhaustive and mutually exclusive -/

def Regime1 (a0 : Vec M) : Prop := feas a0 = true ∧ UpNonempty S feas a0
def Regime2 (a0 : Vec M) : Prop := feas a0 = true ∧ ¬ UpNonempty S feas a0
def Regime3 (a0 : Vec M) : Prop := feas a0 = false ∧ feas S.amin = true
def Regime4 : Prop := feas S.amin = false

/-- **Prop. 1(c).** Under monotone feasibility and `amin ≤ a_{t−1}` exactly one regime holds. -/
theorem regimes_partition (hA : AntitoneFeas feas) (a0 : Vec M) (hmin : S.amin ≤ a0) :
    (Regime1 S feas a0 ∨ Regime2 S feas a0 ∨ Regime3 S feas a0 ∨ Regime4 S feas) ∧
    ¬ (Regime1 S feas a0 ∧ Regime2 S feas a0) ∧ ¬ (Regime1 S feas a0 ∧ Regime3 S feas a0) ∧
    ¬ (Regime1 S feas a0 ∧ Regime4 S feas) ∧ ¬ (Regime2 S feas a0 ∧ Regime3 S feas a0) ∧
    ¬ (Regime2 S feas a0 ∧ Regime4 S feas) ∧ ¬ (Regime3 S feas a0 ∧ Regime4 S feas) := by
  unfold Regime1 Regime2 Regime3 Regime4
  have mono : feas a0 = true → feas S.amin = true := hA _ _ hmin
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · by_cases h1 : feas a0 = true
    · by_cases h2 : UpNonempty S feas a0
      · exact Or.inl ⟨h1, h2⟩
      · exact Or.inr (Or.inl ⟨h1, h2⟩)
    · by_cases h3 : feas S.amin = true
      · exact Or.inr (Or.inr (Or.inl ⟨by simpa using h1, h3⟩))
      · exact Or.inr (Or.inr (Or.inr (by simpa using h3)))
  · rintro ⟨⟨-, h⟩, ⟨-, h'⟩⟩; exact h' h
  · rintro ⟨⟨h, -⟩, ⟨h', -⟩⟩; simp [h] at h'
  · rintro ⟨⟨h, -⟩, h'⟩; simp [mono h] at h'
  · rintro ⟨⟨h, -⟩, ⟨h', -⟩⟩; simp [h] at h'
  · rintro ⟨⟨h, -⟩, h'⟩; simp [mono h] at h'
  · rintro ⟨⟨-, h⟩, h'⟩; simp [h] at h'

/-! ### (d) Terminal points of the two-phase operator in each regime -/

/-- Regime 1: `a_t ≥ a_{t−1}`, `a_t ≠ a_{t−1}`, feasible, and the End rule holds at `a_t`. -/
theorem twoPhase_regime1 (hU : UrgComplete S) (a0 : Vec M) (N : ℕ)
    (hN : ∑ m, (S.L m - 1) ≤ N) (h : Regime1 S feas a0) :
    a0 ≤ (twoPhase S feas N a0).1 ∧ (twoPhase S feas N a0).1 ≠ a0 ∧
    feas (twoPhase S feas N a0).1 = true ∧ ¬ UpNonempty S feas (twoPhase S feas N a0).1 := by
  obtain ⟨hf, hup⟩ := h
  unfold twoPhase
  simp only [hf, if_true]
  have hr := run_spec (upStep S feas) (headroom S) (fun a => a0 ≤ a ∧ feas a = true)
    (fun a a' ⟨h1, _⟩ h => by
      obtain ⟨j, rfl, -, -, h4⟩ := upStep_spec S feas h
      exact ⟨le_trans h1 (le_inc a j), h4⟩)
    (fun a a' _ h => by
      obtain ⟨j, rfl, hj, -, -⟩ := upStep_spec S feas h
      exact headroom_inc_lt S a hj)
    N a0 ⟨le_refl _, hf⟩ (le_trans (headroom_le_total S a0) hN)
  obtain ⟨hstop, -, hle, hfe⟩ := hr
  have hnone : upChoice S feas (run (upStep S feas) N a0).1 = none :=
    upStep_none S feas hfe hstop
  refine ⟨hle, ?_, hfe, (upChoice_none_iff S hU feas _).mp hnone⟩
  intro heq
  rw [heq] at hnone
  exact ((upChoice_none_iff S hU feas a0).mp hnone) hup

/-- Regime 2: `a_t = a_{t−1}` and zero steps. -/
theorem twoPhase_regime2 (a0 : Vec M) (N : ℕ) (h : Regime2 S feas a0) :
    twoPhase S feas N a0 = (a0, 0) := by
  obtain ⟨hf, hup⟩ := h
  unfold twoPhase
  simp only [hf, if_true]
  apply run_of_none
  simp [upStep, hf, upChoice_none_of_not S feas a0 hup]

/-- Regime 3: `amin ≤ a_t ≤ a_{t−1}`, `a_t ≠ a_{t−1}`, and `a_t` feasible. (This is the
correct reading of the paper's `a_t ∈ [a^min, a_{t−1})`: product order with `≠`.) -/
theorem twoPhase_regime3 (hU : UrgComplete S) (a0 : Vec M) (hmin : S.amin ≤ a0)
    (hg : ∀ m, a0 m < S.L m) (N : ℕ) (hN : ∑ m, (S.L m - 1) ≤ N) (h : Regime3 S feas a0) :
    S.amin ≤ (twoPhase S feas N a0).1 ∧ (twoPhase S feas N a0).1 ≤ a0 ∧
    (twoPhase S feas N a0).1 ≠ a0 ∧ feas (twoPhase S feas N a0).1 = true := by
  obtain ⟨hf, hfm⟩ := h
  unfold twoPhase
  simp only [hf, if_false, Bool.false_eq_true]
  have hr := run_spec (downStep S feas) (slack S) (fun a => S.amin ≤ a ∧ a ≤ a0)
    (fun a a' ⟨h1, h2⟩ h => by
      obtain ⟨j, rfl, hj, -⟩ := downStep_spec S feas h
      exact ⟨le_dec h1 hj, le_trans (dec_le a j) h2⟩)
    (fun a a' _ h => by
      obtain ⟨j, rfl, hj, -⟩ := downStep_spec S feas h
      exact slack_dec_lt S a hj)
    N a0 ⟨hmin, le_refl _⟩ (le_trans (slack_le_total S a0 hg) hN)
  obtain ⟨hstop, -, h1, h2⟩ := hr
  set b := (run (downStep S feas) N a0).1
  have hfb : feas b = true := by
    by_contra hnf
    have hnf' : feas b = false := by simpa using hnf
    unfold downStep at hstop
    simp only [hnf', if_false, Bool.false_eq_true] at hstop
    cases hc : downChoice S b with
    | some j => simp [hc] at hstop
    | none =>
      have := le_antisymm (downChoice_none S hU b hc) h1
      rw [this] at hnf'
      simp [hnf'] at hfm
  refine ⟨h1, h2, ?_, hfb⟩
  intro heq
  rw [heq] at hfb
  have := hfb.symm.trans hf
  simp at this

/-- Regime 4: Phase 1 exhausts all slack and halts exactly at `amin`, which is infeasible. -/
theorem twoPhase_regime4 (hU : UrgComplete S) (hA : AntitoneFeas feas) (a0 : Vec M)
    (hmin : S.amin ≤ a0) (hg : ∀ m, a0 m < S.L m) (N : ℕ) (hN : ∑ m, (S.L m - 1) ≤ N)
    (h : Regime4 S feas) :
    (twoPhase S feas N a0).1 = S.amin ∧ feas (twoPhase S feas N a0).1 = false := by
  unfold Regime4 at h
  have infeas : ∀ a, S.amin ≤ a → feas a = false := by
    intro a ha
    by_contra hc
    have := hA _ _ ha (by simpa using hc)
    simp [this] at h
  have hf := infeas a0 hmin
  unfold twoPhase
  simp only [hf, if_false, Bool.false_eq_true]
  have hr := run_spec (downStep S feas) (slack S) (fun a => S.amin ≤ a)
    (fun a a' h1 h => by
      obtain ⟨j, rfl, hj, -⟩ := downStep_spec S feas h
      exact le_dec h1 hj)
    (fun a a' _ h => by
      obtain ⟨j, rfl, hj, -⟩ := downStep_spec S feas h
      exact slack_dec_lt S a hj)
    N a0 hmin (le_trans (slack_le_total S a0 hg) hN)
  obtain ⟨hstop, -, h1⟩ := hr
  set b := (run (downStep S feas) N a0).1
  have hfb := infeas b h1
  unfold downStep at hstop
  simp only [hfb, if_false, Bool.false_eq_true] at hstop
  cases hc : downChoice S b with
  | some j => simp [hc] at hstop
  | none => exact ⟨le_antisymm (downChoice_none S hU b hc) h1, hfb⟩

end TwoPhase

/-! ### Selten-faithful operator `stepV1`: terminates, but only with the weaker bound -/

section V1

variable (S : Scheme M) (feas : Vec M → Bool)

/-- Lexicographic potential: infeasible points sit above all feasible ones. -/
def potV1 (a : Vec M) : ℕ :=
  if feas a then headroom S a else headroom S S.amin + slack S a

/-- **Termination of the Selten-faithful operator** (Upward steps allowed after retreat).
Steps `≤ ∑ (a_{t−1,m} − a^min_m) + ∑ (L_m − 1 − a^min_m) ≤ 2 ∑ (L_m − 1)`. -/
theorem stepV1_terminates (a0 : Vec M) (hmin : S.amin ≤ a0) (N : ℕ)
    (hN : headroom S S.amin + slack S a0 ≤ N) :
    stepV1 S feas (run (stepV1 S feas) N a0).1 = none ∧
    (run (stepV1 S feas) N a0).2 ≤ headroom S S.amin + slack S a0 := by
  have key := run_spec (stepV1 S feas) (potV1 S feas) (fun a => S.amin ≤ a)
    (fun a a' h1 h => by
      unfold stepV1 at h
      split_ifs at h with hf
      · cases hc : upChoice S feas a with
        | none => simp [hc] at h
        | some j => simp [hc] at h; rw [← h]; exact le_trans h1 (le_inc a j)
      · cases hc : downChoice S a with
        | none => simp [hc] at h
        | some j => simp [hc] at h; rw [← h]; exact le_dec h1 (downChoice_slack S a hc))
    (fun a a' h1 h => by
      unfold stepV1 at h
      split_ifs at h with hf
      · cases hc : upChoice S feas a with
        | none => simp [hc] at h
        | some j =>
          simp [hc] at h
          obtain ⟨hj, hfj⟩ := upChoice_spec S feas a hc
          rw [← h]
          simp only [potV1, hf, hfj, if_true]
          exact headroom_inc_lt S a hj
      · cases hc : downChoice S a with
        | none => simp [hc] at h
        | some j =>
          simp [hc] at h
          have hj := downChoice_slack S a hc
          rw [← h]
          simp only [potV1, hf, if_false, Bool.false_eq_true]
          by_cases hf' : feas (dec a j) = true
          · simp only [hf', if_true]
            have := headroom_le_of_ge S (le_dec h1 hj)
            have := slack_pos S a hj
            omega
          · simp only [hf', if_false, Bool.false_eq_true]
            have := slack_dec_lt S a hj
            omega)
    N a0 hmin (by
      unfold potV1; split_ifs
      · exact le_trans (by have := headroom_le_of_ge S hmin; omega) hN
      · exact hN)
  refine ⟨key.1, le_trans key.2.1 ?_⟩
  unfold potV1; split_ifs
  · have := headroom_le_of_ge S hmin; omega
  · exact le_refl _

end V1

/-! ### Counterexamples (checked by `decide`), M = 2, L = (5,5), amin = 0, retreat goal 0 -/

def exS (urg : List (Fin 2)) : Scheme 2 :=
  { L := fun _ => 5, amin := fun _ => 0, urg := fun _ => urg, ret := fun _ => 0 }

/-- (A) **Step bound refuted for the Selten-faithful operator.** One candidate `g = (4,0)`,
`a_{t−1} = (4,4)`: 8 downward steps to `(0,0)` (4 along `r`, then 4 reverse-urgency fallback
steps), then 4 upward steps to `(4,0)`: 12 unit steps `> ∑ (L_m − 1) = 8`. The run halts. -/
theorem counterexample_step_bound :
    run (stepV1 (exS [0, 1]) (feasOf [![4, 0]])) 100 ![4, 4] = (![4, 0], 12) ∧
    stepV1 (exS [0, 1]) (feasOf [![4, 0]]) ![4, 0] = none ∧
    ∑ m : Fin 2, ((exS [0, 1]).L m - 1) = 8 := by
  refine ⟨by decide, by decide, by simp [exS]⟩

/-- (B) **Regime-3 terminal `a_t ≤ a_{t−1}` refuted for the Selten-faithful operator.**
`g = (0,4)`, `a_{t−1} = (2,2)`, urgency `(goal 1, goal 0)`: retreat to `(0,2)` (feasible), then
Upward to `(0,4)`; `(0,4) ≰ (2,2)`. -/
theorem counterexample_regime3_not_below :
    run (stepV1 (exS [1, 0]) (feasOf [![0, 4]])) 100 ![2, 2] = (![0, 4], 4) ∧
    stepV1 (exS [1, 0]) (feasOf [![0, 4]]) ![0, 4] = none ∧
    feasOf [![0, 4]] ![2, 2] = false ∧ feasOf [![0, 4]] ![0, 0] = true ∧
    ¬ (![0, 4] ≤ (![2, 2] : Vec 2)) := by
  refine ⟨by decide, by decide, by decide, by decide, ?_⟩
  intro h; have := h 1; simp at this

/-- (C) **Componentwise strict `a_t < a_{t−1}` is false even for the two-phase operator.**
`g = (1,2)`, `a_{t−1} = (2,2)`: terminal `(1,2)`; coordinate 1 is not lowered. -/
theorem counterexample_regime3_not_strict :
    twoPhase (exS [0, 1]) (feasOf [![1, 2]]) 100 ![2, 2] = (![1, 2], 1) ∧
    ¬ (∀ m : Fin 2, (![1, 2] : Vec 2) m < (![2, 2] : Vec 2) m) := by
  refine ⟨by decide, ?_⟩
  intro h; have := h 1; simp at this

/-- (D) **The code's single-step oracle does not reach a feasible compromise in Regime 3.**
`g = (0,2)`, `a_{t−1} = (2,2)`: one retreat step gives `(1,2)`, still infeasible. -/
theorem counterexample_oracle_regime3_infeasible :
    oracleStep (exS [0, 1]) (feasOf [![0, 2]]) ![2, 2] = ![1, 2] ∧
    feasOf [![0, 2]] ![1, 2] = false ∧ feasOf [![0, 2]] ![2, 2] = false ∧
    feasOf [![0, 2]] ![0, 0] = true := by
  refine ⟨by decide, by decide, by decide, by decide⟩

/-- (E) **The hypothesis `amin ≤ a_{t−1}` is necessary for exclusivity**: with
`a_{t−1} = (0,0)`, `amin = (1,1)`, `g = (0,0)`, Regime 2 and Regime 4 hold simultaneously. -/
theorem counterexample_exclusivity_needs_floor_le :
    let S : Scheme 2 := { L := fun _ => 5, amin := fun _ => 1, urg := fun _ => [0, 1],
                          ret := fun _ => 0 }
    Regime2 S (feasOf [![0, 0]]) ![0, 0] ∧ Regime4 S (feasOf [![0, 0]]) := by
  refine ⟨⟨by decide, ?_⟩, by unfold Regime4; decide⟩
  rintro ⟨m, -, hm⟩
  revert hm
  fin_cases m <;> decide

end BartlebyProofs.AAT
