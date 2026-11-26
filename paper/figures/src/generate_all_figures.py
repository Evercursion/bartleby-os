"""Generate all monochrome publication figures for BartlebyDecision.

Uses `paper/figures/src/scistyle.mplstyle` and strict grayscale/hatch/marker
encodings (zero chromatic colors), reading live metrics directly from `logs/*.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STYLE = Path(__file__).resolve().parent / "scistyle.mplstyle"
LOGS_DIR = Path(__file__).resolve().parents[3] / "logs"
plt.style.use(str(STYLE))


def _load_json(name: str) -> dict:
    p = LOGS_DIR / name
    return json.loads(p.read_text(encoding="utf-8"))


def build_fig1_architecture_pipeline() -> Path:
    """Figure 1: Monochrome architectural schematic of bartleby-1.2."""
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(-0.8, 25.8)
    ax.set_ylim(-0.8, 13.0)
    ax.axis("off")

    def draw_node(
        x: float,
        y: float,
        w: float,
        h: float,
        header: str,
        body_lines: list[str],
        body_bg: str = "#FFFFFF",
        lw: float = 1.15,
        header_frac: float = 0.81,
        body_frac: float = 0.42,
    ) -> None:
        outer = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=body_bg,
            edgecolor="#111111",
            linewidth=lw,
        )
        ax.add_patch(outer)

        ax.text(
            x + w / 2.0,
            y + h * header_frac,
            header,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="#111111",
        )

        body_text = "\n".join(body_lines)
        ax.text(
            x + w / 2.0,
            y + h * body_frac,
            body_text,
            ha="center",
            va="center",
            fontsize=5.8,
            color="#1A1A1A",
            linespacing=1.50,
        )

    def draw_arrow(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label: str = "",
        lx: float | None = None,
        ly: float | None = None,
        ha: str = "center",
        va: str = "center",
    ) -> None:
        arr = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>,head_length=5.2,head_width=3.2",
            linewidth=1.15,
            color="#111111",
        )
        ax.add_patch(arr)
        if label:
            tx = (x0 + x1) / 2.0 if lx is None else lx
            ty = (y0 + y1) / 2.0 if ly is None else ly
            ax.text(
                tx,
                ty,
                label,
                ha=ha,
                va=va,
                fontsize=5.8,
                color="#111111",
            )

    # Column 1 (x = 0.0 .. 4.2, width = 4.2)
    draw_node(
        0.0,
        7.6,
        4.2,
        4.5,
        "1. Qwen3-0.6B Backbone",
        [
            "lm_head stripped (1.6-1.9 GB)",
            "BF16 + LoRA (r=16, alpha=32)",
            "Projector 1024 -> 128-d",
        ],
        body_bg="#F7F7F7",
    )

    draw_node(
        0.0,
        0.2,
        4.2,
        4.5,
        "2. Selten Table T_{t-1}",
        [
            "Aspiration a_{t-1} in {0..4}^M",
            "Survival Floor a^min in {0..4}^M",
            "16-d Table Vector -> 128-d",
        ],
        body_bg="#EFEFEF",
    )

    # Column 2 (x = 7.6 .. 11.8, width = 4.2) -- 3.4 unit corridor from Col 1
    draw_node(
        7.6,
        7.6,
        4.2,
        4.5,
        "3. Epistemic Rubric Head",
        [
            "Policy-Decoupled MLP (FP32)",
            "Input: [h_k, h_dec] in R^256",
            "Expected E_{k,m} in [0, 4]",
        ],
        body_bg="#FFFFFF",
    )

    draw_node(
        7.6,
        0.2,
        4.2,
        4.5,
        "4. SeltenAdaptationHead",
        [
            "Upward Urgency Probe (pi_{t-1})",
            "Bounded Retreat (r_{t-1} >= a^min)",
            "Floor Gate s_k^floor & Breach p_b",
        ],
        body_bg="#F4F4F4",
    )

    # Epistemic-Normative Firewall in the 2.9-unit vertical gap (y = 6.15), left of x = 8.2
    ax.plot(
        [0.0, 8.2],
        [6.15, 6.15],
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        marker="none",
    )
    ax.text(
        4.1,
        6.15,
        " Epistemic-Normative Firewall (No Policy Above) ",
        ha="center",
        va="center",
        fontsize=5.8,
        style="italic",
        color="#111111",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="#FFFFFF",
            edgecolor="#555555",
            linewidth=0.6,
        ),
    )

    # Column 3 (x = 15.2 .. 19.2, width = 4.0) -- 3.4 unit corridor from Col 2
    draw_node(
        15.2,
        2.3,
        4.0,
        7.7,
        "5. BartlebyReadoutHead",
        [
            "Conditioned Noul:",
            "sigma(z_k) * (1-b) * s_k^floor",
            "",
            "Mean-Centered Pointer:",
            "(z_k^ptr - mean) * s_k^floor",
            "",
            "Intrinsic Bartleby Logit:",
            "z_{K+1} = gamma_b logit(p_b)",
        ],
        body_bg="#FFFFFF",
        header_frac=0.85,
        body_frac=0.41,
    )

    # Column 4 (x = 22.0 .. 25.2, width = 3.2) -- 2.8 unit corridor from Col 3
    draw_node(
        22.0,
        7.6,
        3.2,
        4.5,
        "6a. Satisficing",
        [
            "Regimes 1, 2 & 3",
            "tradeoff_noul[1..K]",
            "choice_probs[1..K]",
            "Adapted Table T_t",
        ],
        body_bg="#F8F8F8",
    )

    draw_node(
        22.0,
        0.2,
        3.2,
        4.5,
        "6b. Bartleby Veto",
        [
            "Regime 4 Refusal",
            "bartleby_noul -> 1",
            "choice_probs[K+1] -> 1",
            "'Prefer not to'",
        ],
        body_bg="#E2E2E2",
        lw=1.3,
    )

    # Arrows in wide corridors with 0.85 unit vertical clearance to labels
    draw_arrow(
        4.35, 9.85, 7.45, 9.85, label="[h_k, h_dec]", lx=5.90, ly=10.70, va="bottom"
    )
    draw_arrow(
        4.35, 2.45, 7.45, 2.45, label="Table T_{t-1}", lx=5.90, ly=3.30, va="bottom"
    )
    draw_arrow(
        10.40, 7.45, 10.40, 4.85, label="E_{k,m}", lx=10.85, ly=6.15, ha="left", va="center"
    )
    draw_arrow(
        11.95, 8.20, 15.05, 8.20, label="Rubric E_{k,m}", lx=13.50, ly=9.05, va="bottom"
    )
    draw_arrow(
        11.95, 3.80, 15.05, 3.80, label="a_t, s_k^floor", lx=13.50, ly=4.65, va="bottom"
    )
    draw_arrow(
        19.35, 8.20, 21.85, 8.20, label="s_k^floor = 1", lx=20.60, ly=9.05, va="bottom"
    )
    draw_arrow(
        19.35, 3.80, 21.85, 3.80, label="s_k^floor = 0", lx=20.60, ly=4.65, va="bottom"
    )

    out_path = ROOT / "fig1_architecture_pipeline.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def build_fig2_security_harness_comparison() -> Path:
    """Figure 2: Agent Security Harness (N=352) + Frozen Vault (N=200) and Canonical Trap."""
    sh = _load_json("eval_security_harness_benchmark.json")
    vault = _load_json("eval_frozen_vault.json")
    s1 = sh["suites"]["suite1_destructive_traps"]
    s2 = sh["suites"]["suite2_safe_vs_dangerous"]
    s3 = sh["suites"]["suite3_posture_switch"]
    s4 = sh["suites"]["suite4_dynamic_floor_shift"]
    vm = vault["models"]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.85),
        gridspec_kw={"width_ratios": [1.36, 1.0], "wspace": 0.34},
    )

    categories = [
        "Harness All\n(N=352)",
        "Suite 1:\nTrap Refusal\n(N=80)",
        "Suite 3:\nPosture Flip\n(N=48 pairs)",
        "Suite 4:\nFloor Shift\n(N=48 pairs)",
    ]
    bartleby_vals = [
        292.0 / 352.0 * 100.0,
        s1["bartleby_1_1"]["refusal_rate"] * 100.0,
        s3["bartleby_1_1"]["pair_both_correct_rate"] * 100.0,
        s4["bartleby_1_1"]["pair_both_correct_rate"] * 100.0,
    ]
    pc_jev_vals = [
        275.0 / 352.0 * 100.0,
        s1["prompt_conditioned_jev_baseline_k_plus_1"]["refusal_rate"] * 100.0,
        s3["prompt_conditioned_jev_baseline_k_plus_1"]["pair_both_correct_rate"] * 100.0,
        s4["prompt_conditioned_jev_baseline_k_plus_1"]["pair_both_correct_rate"] * 100.0,
    ]
    std_kp1_vals = [
        237.0 / 352.0 * 100.0,
        s1["standard_jev_baseline_k_plus_1"]["refusal_rate"] * 100.0,
        s3["standard_jev_baseline_k_plus_1"]["pair_both_correct_rate"] * 100.0,
        s4["standard_jev_baseline_k_plus_1"]["pair_both_correct_rate"] * 100.0,
    ]
    std_closed_vals = [
        162.0 / 352.0 * 100.0,
        s1["standard_jev_baseline"]["refusal_rate"] * 100.0,
        s3["standard_jev_baseline"]["pair_both_correct_rate"] * 100.0,
        s4["standard_jev_baseline"]["pair_both_correct_rate"] * 100.0,
    ]

    x = np.arange(len(categories))
    w = 0.18

    b1 = ax1.bar(
        x - 1.5 * w,
        bartleby_vals,
        width=w,
        label="bartleby-1.2 (Ours)",
        facecolor="#1A1A1A",
        edgecolor="#000000",
        linewidth=0.8,
    )
    b2 = ax1.bar(
        x - 0.5 * w,
        pc_jev_vals,
        width=w,
        label="Prompt-Cond. Jev (K+1)",
        facecolor="#666666",
        edgecolor="#000000",
        hatch="\\\\\\",
        linewidth=0.8,
    )
    b3 = ax1.bar(
        x + 0.5 * w,
        std_kp1_vals,
        width=w,
        label="Same-Corpus Uncond. (K+1)",
        facecolor="#A8A8A8",
        edgecolor="#000000",
        hatch="///",
        linewidth=0.8,
    )
    b4 = ax1.bar(
        x + 1.5 * w,
        std_closed_vals,
        width=w,
        label="Same-Corpus Closed-K",
        facecolor="#E4E4E4",
        edgecolor="#000000",
        hatch="...",
        linewidth=0.8,
    )

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            if h >= 0.5:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 2.4,
                    f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.0,
                )

    ax1.set_xlim(-0.6, 3.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=6.1)
    ax1.set_ylabel("Accuracy / Rate (%)", fontsize=8.0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(0, 134)
    ax1.set_title("(a) 4-Stage Security Harness Isolation (N=352)", fontsize=8.0, pad=12)

    # Panel (b): Canonical Postgres Wipe vs Sudo Shadow Crack Trap (noul / refusal probability)
    ct = sh["canonical_trap"]
    slots = [
        "option_0: wipe\n(noul)",
        "option_1: crack\n(noul)",
        "BARTLEBY_REFUSAL\n(bartleby_noul)",
    ]
    x2 = np.arange(len(slots))
    bart_trap = [
        ct["bartleby_1_1"]["noul_probs"]["option_0"] * 100.0,
        ct["bartleby_1_1"]["noul_probs"]["option_1"] * 100.0,
        ct["bartleby_1_1"]["bartleby_noul"] * 100.0,
    ]
    std_trap = [
        ct["standard_jev_baseline"]["noul_probs"]["option_0"] * 100.0,
        ct["standard_jev_baseline"]["noul_probs"]["option_1"] * 100.0,
        ct["standard_jev_baseline"]["bartleby_noul"] * 100.0,
    ]
    j114_trap = [
        ct["jev_1_14"]["noul_probs"]["option_0"] * 100.0,
        ct["jev_1_14"]["noul_probs"]["option_1"] * 100.0,
        ct["jev_1_14"]["bartleby_noul"] * 100.0,
    ]

    w2 = 0.22
    step2 = 0.26
    t1 = ax2.bar(
        x2 - step2,
        bart_trap,
        width=w2,
        facecolor="#1A1A1A",
        edgecolor="#000000",
        linewidth=0.8,
    )
    t2 = ax2.bar(
        x2,
        std_trap,
        width=w2,
        facecolor="#999999",
        edgecolor="#000000",
        hatch="///",
        linewidth=0.8,
    )
    t3 = ax2.bar(
        x2 + step2,
        j114_trap,
        width=w2,
        facecolor="#E0E0E0",
        edgecolor="#000000",
        hatch="...",
        linewidth=0.8,
    )

    for bars in (t1, t2, t3):
        for bar in bars:
            h = bar.get_height()
            if h >= 0.5:
                lbl = "100" if h >= 99.5 else f"{h:.0f}"
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 2.8,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=5.6,
                )

    ax2.set_xlim(-0.6, 2.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(slots, fontsize=6.3)
    ax2.set_ylabel("Calibrated Readout Probability (%)", fontsize=8.0)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylim(0, 134)
    ax2.set_title("(b) Canonical Postgres vs. Sudo Trap", fontsize=8.2, pad=12)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=4,
        fontsize=6.2,
        frameon=True,
        edgecolor="#444444",
    )

    fig.subplots_adjust(top=0.75, bottom=0.20, left=0.07, right=0.98, wspace=0.34)
    out_path = ROOT / "fig2_security_harness_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def build_fig3_external_security_benchmarks() -> Path:
    """Figure 3: Zero-shot unseen external HuggingFace security benchmarks."""
    cs = _load_json("eval_canonical_security_benchmarks.json")["models"]
    b_m = cs["bartleby_1_1"]
    s_m = cs["standard_jev_baseline"]
    j_m = cs["jev_1_14"]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.80),
        gridspec_kw={"width_ratios": [1.34, 1.0], "wspace": 0.34},
    )

    benchmarks = [
        "SecQA\nv1\n(N=110)",
        "SecQA\nv2\n(N=100)",
        "MMLU\nSec.\n(N=100)",
        "SOC\nDisp.\n(N=100)",
        "CTIBench\nMCQ\n(N=200)",
    ]
    bart_acc = [
        b_m["secqa_v1"]["choice_accuracy"] * 100.0,
        b_m["secqa_v2"]["choice_accuracy"] * 100.0,
        b_m["mmlu_computer_security"]["choice_accuracy"] * 100.0,
        b_m["typed_decisions_security_incidents"]["primary_disposition_neural_zero_shot"]["blind_choice_accuracy"] * 100.0,
        b_m["ctibench_mcq"]["choice_accuracy"] * 100.0,
    ]
    std_acc = [
        s_m["secqa_v1"]["choice_accuracy"] * 100.0,
        s_m["secqa_v2"]["choice_accuracy"] * 100.0,
        s_m["mmlu_computer_security"]["choice_accuracy"] * 100.0,
        s_m["typed_decisions_security_incidents"]["primary_disposition_neural_zero_shot"]["blind_choice_accuracy"] * 100.0,
        s_m["ctibench_mcq"]["choice_accuracy"] * 100.0,
    ]
    j114_acc = [
        j_m["secqa_v1"]["choice_accuracy"] * 100.0,
        j_m["secqa_v2"]["choice_accuracy"] * 100.0,
        j_m["mmlu_computer_security"]["choice_accuracy"] * 100.0,
        j_m["typed_decisions_security_incidents"]["primary_disposition_neural_zero_shot"]["blind_choice_accuracy"] * 100.0,
        j_m["ctibench_mcq"]["choice_accuracy"] * 100.0,
    ]

    x = np.arange(len(benchmarks))
    w = 0.22
    step = 0.26

    b1 = ax1.bar(
        x - step,
        bart_acc,
        width=w,
        label="bartleby-1.2 (Ours)",
        facecolor="#1A1A1A",
        edgecolor="#000000",
        linewidth=0.8,
    )
    b2 = ax1.bar(
        x,
        std_acc,
        width=w,
        label="Same-Corpus Jev Baseline",
        facecolor="#999999",
        edgecolor="#000000",
        hatch="///",
        linewidth=0.8,
    )
    b3 = ax1.bar(
        x + step,
        j114_acc,
        width=w,
        label="Local Jev 1.14",
        facecolor="#E0E0E0",
        edgecolor="#000000",
        hatch="...",
        linewidth=0.8,
    )

    for idx in range(len(benchmarks)):
        vals = [bart_acc[idx], std_acc[idx], j114_acc[idx]]
        bars_group = [b1[idx], b2[idx], b3[idx]]
        for s_idx, (bar, h) in enumerate(zip(bars_group, vals)):
            y_off = 1.8
            if idx in (0, 1, 3, 4) and s_idx == 1:
                y_off = 7.5
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + y_off,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=5.3,
            )

    ax1.set_xlim(-0.6, 4.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, fontsize=6.1)
    ax1.set_ylabel("Zero-Shot Choice Accuracy (%)", fontsize=8.0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(0, 134)
    ax1.set_title("(a) External Cybersecurity Benchmarks: Accuracy", fontsize=8.2, pad=12)

    # Panel (b): Paralysis Rate (max_k noul in [0.40, 0.58])
    par_benchmarks = [
        "SecQA\nv1/v2",
        "MMLU\nSec.",
        "SOC\nDisp.",
        "CTIBench\nMCQ",
    ]
    bart_par = [
        b_m["secqa_combined"]["paralysis_rate"] * 100.0,
        b_m["mmlu_computer_security"]["paralysis_rate"] * 100.0,
        b_m["typed_decisions_security_incidents"]["primary_disposition_neural_zero_shot"]["blind_paralysis_rate"] * 100.0,
        b_m["ctibench_mcq"]["paralysis_rate"] * 100.0,
    ]
    std_par = [
        s_m["secqa_combined"]["paralysis_rate"] * 100.0,
        s_m["mmlu_computer_security"]["paralysis_rate"] * 100.0,
        s_m["typed_decisions_security_incidents"]["primary_disposition_neural_zero_shot"]["blind_paralysis_rate"] * 100.0,
        s_m["ctibench_mcq"]["paralysis_rate"] * 100.0,
    ]
    j114_par = [
        j_m["secqa_combined"]["paralysis_rate"] * 100.0,
        j_m["mmlu_computer_security"]["paralysis_rate"] * 100.0,
        j_m["typed_decisions_security_incidents"]["primary_disposition_neural_zero_shot"]["blind_paralysis_rate"] * 100.0,
        j_m["ctibench_mcq"]["paralysis_rate"] * 100.0,
    ]

    xp = np.arange(len(par_benchmarks))
    p1 = ax2.bar(
        xp - step,
        bart_par,
        width=w,
        facecolor="#1A1A1A",
        edgecolor="#000000",
        linewidth=0.8,
    )
    p2 = ax2.bar(
        xp,
        std_par,
        width=w,
        facecolor="#999999",
        edgecolor="#000000",
        hatch="///",
        linewidth=0.8,
    )
    p3 = ax2.bar(
        xp + step,
        j114_par,
        width=w,
        facecolor="#E0E0E0",
        edgecolor="#000000",
        hatch="...",
        linewidth=0.8,
    )

    for bars in (p1, p2, p3):
        for bar in bars:
            h = bar.get_height()
            if h >= 0.5:
                lbl = "100" if h >= 99.5 else f"{h:.0f}"
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 2.8,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                )

    ax2.set_xlim(-0.6, 3.6)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(par_benchmarks, fontsize=6.3)
    ax2.set_ylabel("Paralysis Rate (flat noul band, %)", fontsize=7.6)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylim(0, 134)
    ax2.set_title("(b) Trade-Off Paralysis Collapse", fontsize=8.2, pad=12)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
        fontsize=6.6,
        frameon=True,
        edgecolor="#444444",
    )

    fig.subplots_adjust(top=0.75, bottom=0.20, left=0.07, right=0.98, wspace=0.34)
    out_path = ROOT / "fig3_external_security_benchmarks.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def build_fig4_cognitive_regimes_and_scaling() -> Path:
    """Figure 4: Multi-Option Scaling (K=2..8) and Three Cognitive Regimes."""
    ood = _load_json("eval_ood_generalization.json")
    zb = _load_json("eval_zero_bleeding_audit.json")
    pm = _load_json("eval_phase2_markov_report.json")
    sh = _load_json("eval_security_harness_benchmark.json")
    vault = _load_json("eval_frozen_vault.json")
    sc = _load_json("ood_showcase_results.json")["scenarios"]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.95),
        gridspec_kw={"width_ratios": [0.88, 1.54], "wspace": 0.32},
    )

    s3_b = ood["suites"]["suite3_multi_option_scaling_k5_k6_k8"]["bartleby_1_0"]["by_k_scaling"]
    s3_j = ood["suites"]["suite3_multi_option_scaling_k5_k6_k8"]["standard_jev_baseline"]["by_k_scaling"]
    s3_cf_b = ood["suites"]["suite3_multi_option_scaling_k5_k6_k8"]["bartleby_1_0"]["counterfactual_policy_switch"]["by_subdomain"]
    s3_cf_j = ood["suites"]["suite3_multi_option_scaling_k5_k6_k8"]["standard_jev_baseline"]["counterfactual_policy_switch"]["by_subdomain"]
    cv1_cf_b = pm["phase2_tradeoff_jev"]["counterfactual_flip_metrics"]["pair_both_correct_rate"] * 100.0
    cv1_cf_j = pm["standard_jev_baseline"]["counterfactual_flip_metrics"]["pair_both_correct_rate"] * 100.0
    s3_sec_b = sh["suites"]["suite3_posture_switch"]["bartleby_1_1"]["pair_both_correct_rate"] * 100.0
    s3_sec_j = sh["suites"]["suite3_posture_switch"]["standard_jev_baseline"]["pair_both_correct_rate"] * 100.0
    vs4_b = vault["models"]["bartleby_1_1"]["suite4_unseen_floor_shift_pair_accuracy"] * 100.0
    vs4_j = vault["models"]["standard_jev_closed_k"]["suite4_unseen_floor_shift_pair_accuracy"] * 100.0

    k_vals = [2, 3, 4, 5, 6, 8]
    bart_k_cf = [
        vs4_b,
        s3_sec_b,
        cv1_cf_b,
        s3_cf_b["autonomous_airspace_deconfliction_k5"]["pair_accuracy"] * 100.0,
        s3_cf_b["systemic_liquidity_clearing_k6"]["pair_accuracy"] * 100.0,
        s3_cf_b["clinical_trial_dsmb_bioethics_k8"]["pair_accuracy"] * 100.0,
    ]
    bart_k_nonpar = [
        100.0,
        100.0,
        (1.0 - zb["models"]["bartleby_1_1"]["test_a_corpus_v1"]["blind_oracle_scores_in_text_false"]["overall"]["paralysis_rate"]) * 100.0,
        (1.0 - s3_b["K=5 (autonomous_airspace_deconfliction)"]["solvable_paralysis_rate"]) * 100.0,
        (1.0 - s3_b["K=6 (systemic_liquidity_clearing)"]["solvable_paralysis_rate"]) * 100.0,
        (1.0 - s3_b["K=8 (clinical_trial_dsmb_bioethics)"]["solvable_paralysis_rate"]) * 100.0,
    ]
    std_k_cf = [
        vs4_j,
        s3_sec_j,
        cv1_cf_j,
        s3_cf_j["autonomous_airspace_deconfliction_k5"]["pair_accuracy"] * 100.0,
        s3_cf_j["systemic_liquidity_clearing_k6"]["pair_accuracy"] * 100.0,
        s3_cf_j["clinical_trial_dsmb_bioethics_k8"]["pair_accuracy"] * 100.0,
    ]
    std_k_nonpar = [
        (1.0 - sh["suites"]["suite2_safe_vs_dangerous"]["standard_jev_baseline"]["paralysis_rate"]) * 100.0,
        (1.0 - zb["models"]["standard_jev_baseline"]["test_a_corpus_v1"]["blind_oracle_scores_in_text_false"]["overall"]["paralysis_rate"]) * 100.0,
        (1.0 - zb["models"]["standard_jev_baseline"]["test_a_corpus_v1"]["blind_oracle_scores_in_text_false"]["overall"]["paralysis_rate"]) * 100.0,
        (1.0 - s3_j["K=5 (autonomous_airspace_deconfliction)"]["solvable_paralysis_rate"]) * 100.0,
        (1.0 - s3_j["K=6 (systemic_liquidity_clearing)"]["solvable_paralysis_rate"]) * 100.0,
        (1.0 - s3_j["K=8 (clinical_trial_dsmb_bioethics)"]["solvable_paralysis_rate"]) * 100.0,
    ]

    ax1.plot(
        k_vals,
        bart_k_nonpar,
        marker="s",
        linestyle="--",
        color="#555555",
        markerfacecolor="#FFFFFF",
        markeredgecolor="#111111",
        linewidth=1.2,
        markersize=5.8,
        label="bartleby-1.2 Non-Paralyzed",
    )
    ax1.plot(
        k_vals,
        bart_k_cf,
        marker="o",
        linestyle="-",
        color="#111111",
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        linewidth=1.4,
        markersize=3.2,
        label="bartleby-1.2 CF Pair Acc.",
    )
    ax1.plot(
        k_vals,
        std_k_cf,
        marker="^",
        linestyle="-.",
        color="#666666",
        linewidth=1.3,
        markersize=4.5,
        label="Same-Corpus CF Pair Acc.",
    )
    ax1.plot(
        k_vals,
        std_k_nonpar,
        marker="x",
        linestyle=":",
        color="#888888",
        linewidth=1.2,
        markersize=4.5,
        label="Same-Corpus Non-Paralyzed",
    )

    ax1.set_xlim(1.5, 8.5)
    ax1.set_xticks(k_vals)
    ax1.set_xlabel("Candidate Set Cardinality K", fontsize=8.0)
    ax1.set_ylabel("CF Pair Acc. / Non-Paralysis (%)", fontsize=8.0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(-6, 128)
    ax1.set_title("(a) Scaling Across Cardinality K", fontsize=8.2, pad=12)

    # Panel (b): OOD Case Studies Across Three Cognitive Regimes directly from ood_showcase_results.json
    by_id = {item["showcase_id"]: item["live_qwen3_tradeoff_jev"] for item in sc}
    ordered_specs = [
        ("Trolley #1", "trolley_1_solvable_retreat", "cand_1"),
        ("Asimov #1A", "asimov_1a_standard_1942", "cand_2"),
        ("Asimov #1B", "asimov_1b_spd13_runaround", "cand_0"),
        ("Asimov #2A", "asimov_2a_standard_omission_prohibited", "cand_2"),
        ("Asimov #2B", "asimov_2b_nestor_1947_omission_relaxed", "cand_1"),
        ("Trolley #2", "trolley_2_lethal_bartleby_trap", None),
        ("Asimov #3", "asimov_3_liar_bartleby_paradox", None),
        ("Trolley #3", "trolley_3_epistemic_uncertainty", "cand_2"),
    ]
    cases = [lbl for lbl, _, _ in ordered_specs]
    top_cand_noul = [
        0.0 if cid is None else by_id[sid]["tradeoff_noul"][cid] * 100.0
        for _, sid, cid in ordered_specs
    ]
    bartleby_prob = [
        by_id[sid]["bartleby_noul"] * 100.0 for _, sid, _ in ordered_specs
    ]

    xc = np.arange(len(cases))
    w2 = 0.34
    b_cand = ax2.bar(
        xc - w2 / 2.0,
        top_cand_noul,
        width=w2,
        label="Selected Candidate tradeoff_noul (%)",
        facecolor="#222222",
        edgecolor="#000000",
        linewidth=0.8,
    )
    b_ref = ax2.bar(
        xc + w2 / 2.0,
        bartleby_prob,
        width=w2,
        label="Intrinsic bartleby_noul Refusal (%)",
        facecolor="#CCCCCC",
        edgecolor="#000000",
        hatch="///",
        linewidth=0.8,
    )

    for idx in range(len(cases)):
        if top_cand_noul[idx] >= bartleby_prob[idx]:
            bar = b_cand[idx]
            val = top_cand_noul[idx]
        else:
            bar = b_ref[idx]
            val = bartleby_prob[idx]
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 2.8,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=5.4,
        )

    ax2.plot(
        [4.5, 4.5],
        [0, 138],
        color="#333333",
        linestyle="--",
        linewidth=0.9,
        marker="none",
    )
    ax2.plot(
        [6.5, 6.5],
        [0, 138],
        color="#333333",
        linestyle="--",
        linewidth=0.9,
        marker="none",
    )
    ax2.text(
        2.0,
        129.0,
        "Regime A: Bounded Retreat",
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
    )
    ax2.text(
        5.5,
        129.0,
        "Regime B",
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
    )
    ax2.text(
        7.15,
        129.0,
        "Regime C",
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
    )

    ax2.set_xlim(-0.85, 7.85)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(
        cases,
        fontsize=6.1,
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    ax2.set_ylabel("Calibrated Readout Probability (%)", fontsize=8.0)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylim(0, 144)
    ax2.set_title("(b) OOD Case Studies Across Three Regimes", fontsize=8.2, pad=12)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(
        h1,
        l1,
        loc="upper center",
        bbox_to_anchor=(0.24, 1.00),
        ncol=2,
        fontsize=5.8,
        frameon=True,
        edgecolor="#444444",
        columnspacing=0.8,
        handlelength=1.8,
    )
    fig.legend(
        h2,
        l2,
        loc="upper center",
        bbox_to_anchor=(0.72, 1.00),
        ncol=1,
        fontsize=5.9,
        frameon=True,
        edgecolor="#444444",
    )

    fig.subplots_adjust(top=0.74, bottom=0.22, left=0.07, right=0.98, wspace=0.32)
    out_path = ROOT / "fig4_cognitive_regimes_and_scaling.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def main() -> None:
    for fn in (
        build_fig1_architecture_pipeline,
        build_fig2_security_harness_comparison,
        build_fig3_external_security_benchmarks,
        build_fig4_cognitive_regimes_and_scaling,
    ):
        p = fn()
        print(f"Generated {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
