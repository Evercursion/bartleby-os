import BartlebyProofs

-- Section 1 (Eq. 1-2): Preference-Blind Paralysis & Observability
#print axioms BartlebyProofs.Paralysis.brier_argmin_unique
#print axioms BartlebyProofs.Paralysis.ce_eq_entropy_imp
#print axioms BartlebyProofs.Paralysis.choice_support_card_le
#print axioms BartlebyProofs.Paralysis.two_postures_not_uniform
#print axioms BartlebyProofs.Paralysis.independent_T_collapse
#print axioms BartlebyProofs.Paralysis.observable_T_no_collapse

-- Section 2 (Proposition 1): Finite Termination & Exhaustive 4-Regime Partition
#print axioms BartlebyProofs.AAT.twoPhase_terminates_bound
#print axioms BartlebyProofs.AAT.regimes_partition
#print axioms BartlebyProofs.AAT.twoPhase_regime1
#print axioms BartlebyProofs.AAT.twoPhase_regime2
#print axioms BartlebyProofs.AAT.twoPhase_regime3
#print axioms BartlebyProofs.AAT.twoPhase_regime4
#print axioms BartlebyProofs.AAT.stepV1_terminates

-- Section 3 (Proposition 2): Softmax Ceiling Bound & Floor Gate Clearance
#print axioms BartlebyProofs.Ceiling.expScore_lt_top
#print axioms BartlebyProofs.Ceiling.round_top_iff
#print axioms BartlebyProofs.Ceiling.roundHalfEven_top_iff_L5
#print axioms BartlebyProofs.Ceiling.sig_ceiling_lt_half
#print axioms BartlebyProofs.Ceiling.per_goal_gate_gt
#print axioms BartlebyProofs.Ceiling.product_gate_lower_bound
#print axioms BartlebyProofs.Ceiling.product_gate_gt_099

-- Section 3 (Eq. 9-12): Exact Regime Partition of Unity & Bartleby Logit Clamp
#print axioms BartlebyProofs.Regime.regime_sum_one
#print axioms BartlebyProofs.Bartleby.bnoul_range_shipped
#print axioms BartlebyProofs.Bartleby.sat_at_gstar
#print axioms BartlebyProofs.Bartleby.sat_low_at_gstar
#print axioms BartlebyProofs.Bartleby.gstar_bounds
#print axioms BartlebyProofs.Bartleby.saturation_iff

-- Section 3 (Theorem 1): Rubric Invariance, Survival-Floor Gate Suppression & No Readout Bypass
#print axioms BartlebyProofs.Floor.rubric_invariance
#print axioms BartlebyProofs.Floor.thm1_suppression
#print axioms BartlebyProofs.Floor.thm1_delta_half
#print axioms BartlebyProofs.Floor.thm1_delta_one
#print axioms BartlebyProofs.Floor.zcand_close
#print axioms BartlebyProofs.Floor.pBart_ge
#print axioms BartlebyProofs.Floor.tradeoff_noul_paper_le
#print axioms BartlebyProofs.Floor.thm1_noshortcut
#print axioms BartlebyProofs.Floor.thm1_noshortcut_delta_half
#print axioms BartlebyProofs.Floor.thm1_noshortcut_delta_one
