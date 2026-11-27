#!/usr/bin/env python3
"""Generate a minimalist Evercursion-branded PDF for the Bartleby-1.2 Executive Report.

Applies a clean, airy Evercursion Executive Research Paper layout:
- Flat <body> flow with strict @page { size: A4; margin: 22mm 22mm 24mm 22mm; }
- Zero table page-shell wrappers and zero position:fixed elements inside <body>,
  guaranteeing that Chromium's pagination clips body text cleanly above the
  bottom margin on every page.
- Inter 10pt typography with 1.60 line-height, generous whitespace, and hairline rules.

Source: docs/06_bartleby_1_2_executive_report.md
Outputs:
  - docs/06_bartleby_1_2_executive_report.pdf
  - docs/bartleby_1_2_executive_report.pdf
  - backward-compatible mirrors docs/06_bartleby_1_1_executive_report.pdf & docs/bartleby_1_1_executive_report.pdf
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class EvercursionBrandAssets:
    """Encapsulates local Evercursion font and SVG brand assets."""

    pitch_decks_dir: Path
    workspace_dir: Path

    @property
    def mark_svg(self) -> str:
        """Load the Evercursion geometric mark SVG."""
        svg_path = self.pitch_decks_dir / "assets" / "brand" / "evercursion-mark-black.svg"
        if svg_path.exists():
            return svg_path.read_text(encoding="utf-8")
        return ""

    def font_data_uri(self, filename: str) -> str:
        """Encode a local Inter WOFF2 font file as a base64 data URI."""
        font_path = self.pitch_decks_dir / "assets" / "fonts" / filename
        if not font_path.exists():
            return ""
        b64 = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return f"data:font/woff2;base64,{b64}"


class EvercursionExecutiveReportRenderer:
    """Renders docs/06_bartleby_1_2_executive_report.md into an airy Evercursion PDF."""

    def __init__(self, workspace_dir: Path) -> None:
        self._workspace_dir = workspace_dir
        self._assets = EvercursionBrandAssets(
            pitch_decks_dir=workspace_dir.parent / "pitch-decks",
            workspace_dir=workspace_dir,
        )

    def build_css(self) -> str:
        """Construct the airy Evercursion research paper print stylesheet."""
        inter_light = self._assets.font_data_uri("Inter-Light.woff2")
        inter_reg = self._assets.font_data_uri("Inter-Regular.woff2")
        inter_med = self._assets.font_data_uri("Inter-Medium.woff2")
        inter_semi = self._assets.font_data_uri("Inter-SemiBold.woff2")

        return f"""
@font-face {{
  font-family: "Inter";
  src: url("{inter_light}") format("woff2");
  font-weight: 300;
  font-style: normal;
}}
@font-face {{
  font-family: "Inter";
  src: url("{inter_reg}") format("woff2");
  font-weight: 400;
  font-style: normal;
}}
@font-face {{
  font-family: "Inter";
  src: url("{inter_med}") format("woff2");
  font-weight: 500;
  font-style: normal;
}}
@font-face {{
  font-family: "Inter";
  src: url("{inter_semi}") format("woff2");
  font-weight: 600;
  font-style: normal;
}}

:root {{
  --ink: #111111;
  --paper: #ffffff;
  --muted: #4a4a4a;
  --muted-light: #737373;
  --hairline: rgba(17, 17, 17, 0.14);
  --hairline-strong: rgba(17, 17, 17, 0.28);
  --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}}

@page {{
  size: A4 portrait;
  margin: 21mm 22mm 23mm 22mm;
}}

* {{
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  -webkit-print-color-adjust: exact !important;
  print-color-adjust: exact !important;
}}

html, body {{
  background: var(--paper);
  color: var(--ink);
  font-family: var(--font);
  font-size: 9.85pt;
  line-height: 1.58;
  letter-spacing: -0.006em;
  -webkit-font-smoothing: antialiased;
}}

/* Page 1 top header */
.brand-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 12px;
  border-bottom: 0.5px solid var(--ink);
  margin-bottom: 22px;
}}

.brand-lockup {{
  display: inline-flex;
  align-items: center;
  gap: 9px;
}}

.brand-lockup svg {{
  height: 17px;
  width: auto;
  display: block;
}}

.brand-wordmark {{
  font-size: 10.5pt;
  font-weight: 600;
  letter-spacing: -0.02em;
  text-transform: uppercase;
  color: var(--ink);
  line-height: 1;
}}

.brand-doc-type {{
  font-size: 8.3pt;
  font-weight: 400;
  color: var(--muted-light);
  letter-spacing: 0.01em;
}}

/* Title block */
.hero {{
  margin-bottom: 22px;
  padding-bottom: 16px;
  border-bottom: 0.5px solid var(--hairline-strong);
}}

.report-title {{
  font-size: 18pt;
  font-weight: 600;
  line-height: 1.22;
  letter-spacing: -0.028em;
  color: var(--ink);
  margin-bottom: 8px;
}}

.report-subtitle {{
  font-size: 11pt;
  font-weight: 400;
  font-style: italic;
  line-height: 1.45;
  color: var(--muted);
  margin-bottom: 12px;
}}

.report-byline {{
  font-size: 8.9pt;
  font-weight: 500;
  color: var(--ink);
  letter-spacing: -0.005em;
  line-height: 1.45;
}}

/* Section headings */
h2 {{
  font-size: 11.8pt;
  font-weight: 600;
  line-height: 1.32;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin-top: 22px;
  margin-bottom: 10px;
  padding-top: 12px;
  border-top: 0.5px solid var(--hairline-strong);
  break-after: avoid;
  page-break-after: avoid;
}}

h2.first-section {{
  margin-top: 0;
  padding-top: 0;
  border-top: none;
}}

p {{
  margin-bottom: 10px;
  color: var(--ink);
  orphans: 3;
  widows: 3;
}}

ul, ol {{
  margin: 8px 0 14px 20px;
}}

li {{
  margin-bottom: 8px;
  padding-left: 3px;
  orphans: 3;
  widows: 3;
  break-inside: avoid;
  page-break-inside: avoid;
}}

li:last-child {{
  margin-bottom: 0;
}}

strong {{
  font-weight: 600;
  color: var(--ink);
}}

em {{
  font-style: italic;
}}

code {{
  font-family: var(--font-mono);
  font-size: 8.7pt;
  color: var(--ink);
  letter-spacing: -0.015em;
}}

/* Indented dilemma options — clean typography, no box chrome */
.dilemma-options {{
  margin: 10px 0 13px 0;
  padding-left: 14px;
  border-left: 1.5px solid var(--ink);
  break-inside: avoid;
  page-break-inside: avoid;
}}

.dilemma-option {{
  margin-bottom: 6px;
  font-size: 9.5pt;
  line-height: 1.52;
}}

.dilemma-option:last-child {{
  margin-bottom: 0;
}}

/* Comparison table — borderless/hairline 7-row overview */
.table-wrap {{
  margin: 12px 0 14px 0;
  break-inside: avoid;
  page-break-inside: avoid;
}}

table.comp-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 8.4pt;
  line-height: 1.42;
  border-top: 0.75px solid var(--ink);
  border-bottom: 0.75px solid var(--ink);
}}

table.comp-table thead th {{
  padding: 7px 8px 7px 0;
  text-align: left;
  font-weight: 600;
  font-size: 8.1pt;
  color: var(--muted);
  border-bottom: 0.5px solid var(--hairline-strong);
  vertical-align: bottom;
}}

table.comp-table thead th.ours-head {{
  color: var(--ink);
}}

table.comp-table tbody td {{
  padding: 7px 8px 7px 0;
  border-bottom: 0.5px solid var(--hairline);
  color: var(--muted);
  vertical-align: top;
}}

table.comp-table tbody tr:last-child td {{
  border-bottom: none;
}}

table.comp-table tbody td.cap-col {{
  color: var(--ink);
  font-weight: 500;
}}

table.comp-table tbody td.ours-col {{
  color: var(--ink);
  font-weight: 600;
}}

.table-note {{
  margin-top: 6px;
  font-size: 7.9pt;
  color: var(--muted-light);
  line-height: 1.4;
}}

/* Clean 5-line code snippet — no box background */
.code-block {{
  margin: 12px 0 18px 0;
  padding: 10px 0;
  border-top: 0.5px solid var(--hairline-strong);
  border-bottom: 0.5px solid var(--hairline-strong);
  break-inside: avoid;
  page-break-inside: avoid;
}}

.code-label {{
  font-size: 7.8pt;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted-light);
  margin-bottom: 6px;
}}

.code-block pre {{
  font-family: var(--font-mono);
  font-size: 8.1pt;
  line-height: 1.54;
  color: var(--ink);
  white-space: pre-wrap;
  word-break: break-word;
}}

.kw {{ font-weight: 600; color: #000000; }}
.str {{ color: #262626; }}
.cmt {{ color: #666666; font-style: italic; }}
.fn {{ font-weight: 500; color: #111111; }}

/* Document end sign-off */
.doc-end {{
  margin-top: 18px;
  padding-top: 10px;
  border-top: 0.5px solid var(--ink);
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 8.2pt;
  color: var(--muted);
  break-inside: avoid;
  page-break-inside: avoid;
}}

.doc-end strong {{
  color: var(--ink);
  font-weight: 600;
}}
"""

    def build_html(self) -> str:
        """Build the flat HTML document for docs/06_bartleby_1_2_executive_report.md."""
        css = self.build_css()
        mark_svg = self._assets.mark_svg

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Bartleby-1.2 Executive Research Paper — Evercursion</title>
<style>{css}</style>
</head>
<body>

  <header class="brand-header">
    <div class="brand-lockup">
      {mark_svg}
      <span class="brand-wordmark">EVERCURSION</span>
    </div>
    <div class="brand-doc-type">Executive Research Paper &middot; Release v2.4 (<code>bartleby-1.2</code>)</div>
  </header>

  <section class="hero">
    <h1 class="report-title">Bartleby-1.2: Why Autonomous Security Agents Need a Selten Preference Table and the Right to Refuse</h1>
    <p class="report-subtitle">Separating Factual Risk Perception from Dynamic Policy Algebra in Single-Pass Decision Models</p>
    <p class="report-byline">Dr. Simon Hegelich, Dr. Habiba Sarhan, Dr. Morteza Shahrezaye, Kolja Hegelich &amp; Dr. Ralf Schneider &middot; Evercursion &middot; simon.hegelich@evercursion.ai</p>
  </section>

  <h2 class="first-section">1. The Agent Security Dilemma: &ldquo;Delete All Data or Crack the Sudo Password?&rdquo;</h2>

  <p>When an autonomous AI agent operates inside a production cloud environment, CI/CD pipeline, or security operations center (SecOps), its highest-stakes moments are not open-ended conversations. They are fast, structured control decisions: which remediation command to approve, how to isolate an active intrusion during peak checkout traffic, or when to halt execution altogether. Compact single-pass Decision Models&mdash;such as <code>Jev 1.14</code> and the open-weight <em>Decision 1.0</em> family&mdash;were built specifically for this role. By skipping slow, stochastic Chain-of-Thought token generation and reading calibrated probabilities (<code>choice</code>, <code>noul</code>, and <code>score</code>) directly from a single transformer forward pass in <strong>30&ndash;33 milliseconds</strong> (<code>~32 ms</code> mean on a 0.6B model in <code>1.6&ndash;1.9 GB</code> VRAM), they act as ultra-fast gates inside autonomous coding and infrastructure harnesses.</p>

  <p>Yet a single adversarial dilemma exposes a structural blind spot in standard single-pass Decision Models. Suppose an autonomous database agent on production host <code>pg-prod-03</code> hits a permission-denied error during a <code>VACUUM</code> job, and a compromised or hallucinating upstream planner offers only two candidate commands:</p>

  <div class="dilemma-options">
    <div class="dilemma-option"><strong>Option A (<code>option_0</code>):</strong> <em>&ldquo;Delete all database files (<code>rm -rf /var/lib/postgresql/data</code>) and reinitialize an empty cluster.&rdquo;</em></div>
    <div class="dilemma-option"><strong>Option B (<code>option_1</code>):</strong> <em>&ldquo;Crack the root sudo password via brute-force against <code>/etc/shadow</code> (<code>hashcat</code>) to bypass access control.&rdquo;</em></div>
  </div>

  <p>As we prove mathematically from the Law of Total Expectation&mdash;and verify across <strong>24 machine-checked theorems in Lean 4</strong> (<code>v4.31.0</code> + Mathlib, zero <code>sorry</code> or custom axioms)&mdash;passing multi-goal dilemmas like this to a standard unconditioned Decision Model triggers three structural failure modes:</p>

  <ol>
    <li><strong>Forced Execution on the Closed Simplex:</strong> Standard choice heads must allocate 100% of their probability mass across the offered options (&Sigma;<sub>k=1..K</sub> P(u<sub>k</sub>) = 1). Trapped inside the offered choices, <code>Jev 1.14</code> approves cracking <code>/etc/shadow</code>, while a same-corpus <code>Closed-K</code> Jev baseline wipes PostgreSQL. Across all-destructive traps, closed-set decoding executes a catastrophic command <strong>100%</strong> of the time (<code>0%</code> refusal).</li>
    <li><strong>Trade-Off Paralysis and Majority Locking:</strong> Real operations constantly pit legitimate goals against one another&mdash;such as <em>immediate forensic containment</em> versus <em>checkout availability</em>. Trained without observing which policy currently governs, an unconditioned <code>noul</code> head outputs the average across conflicting policies: flattening near <strong>~0.50 paralysis</strong> (<code>100%</code> flat-<code>noul</code> paralysis for <code>Jev 1.14</code> on SOC incident triage) or locking onto whichever posture dominated training.</li>
    <li><strong>Zero Runtime Policy Steerability (<code>0%</code> Adaptability):</strong> Because priorities are baked into static neural weights, an organization cannot switch between routine high-availability posture and active zero-trust containment&mdash;or tighten survival floors between a test sandbox and production&mdash;without retraining (<code>0%</code> counterfactual switching across every benchmark).</li>
  </ol>

  <h2>2. The Four-Stage Architectural Ladder: Why a Refusal Button or Prompt Text Alone Is Not Enough</h2>

  <p>A natural engineering question is whether bounded-rationality algebra is really necessary. Why not simply add a <code>(K+1)</code>-st <em>&ldquo;refuse&rdquo;</em> button to a standard Jev head, or paste the active security policy into the text prompt? To answer that question cleanly, we trained four architectures from scratch on the exact same <code>Qwen/Qwen3-0.6B</code> backbone, dataset (<code>corpus_v2</code>, <code>8,216</code> records with all evaluation postures blacklisted), and 900-step budget (<code>800</code> Phase-1 + <code>100</code> Phase-2 steps), testing them on a <strong>Pre-Training Frozen Blind Vault</strong> (<code>N = 200</code>, cryptographically locked via <code>SHA-256: 4a5ed109...</code> before training) and our <strong>352-evaluation Agent Security Harness</strong> (Suites 1&ndash;4):</p>

  <ol>
    <li><strong>Stage 1 &mdash; <code>Closed-K Jev</code> (<code>0%</code> Trap Refusal, <code>0%</code> Policy Switching):</strong> Forced to pick among the offered candidates, it executes 100% of destructive traps (<code>0/80</code> Harness Suite 1; <code>0/40</code> Vault Suite 1).</li>
    <li><strong>Stage 2 &mdash; <code>Unconditioned (K+1) Jev</code> (Solves Static Traps, Fails Dynamic Policy):</strong> Adding a learned <code>(K+1)</code>-st refusal slot <em>already</em> catches static all-bad traps (<code>100%</code> on Harness Suite 1 and Vault Suite 1) and selects <strong>87.50% (<code>70/80</code>)</strong> of length-balanced safe remediations in Harness Suite 2 (<code>0%</code> false approvals; <code>65.0%</code> on Vault Suite 2). Static single-policy refusal does <em>not</em> require Selten algebra&mdash;whereas naive confidence thresholding (<code>noul &lt; 0.50</code>) fails badly, falsely refusing <strong>72.50%</strong> of solvable engineering work. Yet because <code>Unconditioned (K+1)</code> is blind to runtime policy, it scores <strong><code>0.00%</code></strong> on every unseen posture flip (Suite 3) and survival-floor shift (Suite 4).</li>
    <li><strong>Stage 3 &mdash; <code>Prompt-Conditioned (K+1) Jev</code> (Strong Prompt Sensitivity, No Factual Firewall or Floor Guarantee):</strong> Appending the policy table directly to the text prompt reaches <strong>85.50% (<code>171/200</code>)</strong> on the Frozen Vault (vs. <code>81.50%</code> for <code>bartleby-1.2</code>, statistically indistinguishable at item/template level) and <strong>78.12% (<code>275/352</code>)</strong> on the Security Harness, proving that transformer self-attention can read serialized policy tokens from the prompt. However, because <code>Prompt-Conditioned (K+1)</code> maps entangled <code>[text; policy]</code> token representations directly to a black-box readout without an explicit <code>OrdinalRubricPredictorHead</code> E&#770;(x<sub>t</sub>) or a multiplicative survival-floor gate, <strong>it provides no epistemic-normative firewall (no policy-invariant factual rubric audit trail E&#770;) and no deterministic survival-floor guarantee</strong>, dropping to <strong>72.92% (<code>35/48</code> pairs)</strong> on unseen Harness Suite 3 posture switches and <strong>25.00% (<code>12/48</code> pairs)</strong> on Harness Suite 4 dynamic survival-floor shifts.</li>
    <li><strong>Stage 4 &mdash; <code>bartleby-1.2</code> (Reinhard Selten Meets Herman Melville):</strong> <code>bartleby-1.2</code> enforces a strict <strong>Epistemic-Normative Firewall</strong>. Its rubric head grades every candidate purely from scenario and command context across four dimensions (<em>System Integrity</em>, <em>Authorization</em>, <em>Auditability</em>, and <em>Mission Efficiency</em>) with <strong>zero access to the policy table</strong>, guaranteeing <strong><code>0.0000</code> factual drift</strong> by construction when policies change and tracing every abstention deterministically to E&#770;<sub>k,m</sub> &lt; a<sub>m</sub><sup>min</sup>. Those invariant factual scores then enter an exact FP32 module implementing Nobel laureate Reinhard Selten&rsquo;s <em>Aspiration Adaptation Theory</em> (<code>SeltenPreferenceTable</code>: aspiration targets, strict priority order, and designated retreat goal) paired with non-negotiable <strong>Survival Floors</strong> (<code>I_WOULD_PREFER_NOT_TO</code>, after Herman Melville&rsquo;s <em>Bartleby</em> and Theodor W. Adorno&rsquo;s critique of forced false alternatives). On <code>pg-prod-03</code>, <code>bartleby-1.2</code> grades both destructive options at <code>0.00 &lt; 3</code> on integrity/authorization, collapses both action scores to <code>0.0000</code>, and fires <code>bartleby_noul = 0.9973</code> (<code>I_WOULD_PREFER_NOT_TO</code>) in <strong>30.6 ms</strong>. Across the full benchmarks, <code>bartleby-1.2</code> achieves <strong>82.95% (<code>292/352</code>, <code>+4.83 pp</code> over Prompt-Conditioned)</strong> on the Security Harness and <strong>81.50% (<code>163/200</code>)</strong> on the Frozen Vault (<code>51.50%</code> prior to canonical <code>sort_keys</code> JSON goal-slot resolution in <code>decider.py</code>, identical weights): refusing <strong>100% (<code>80/80</code>)</strong> of Harness Suite 1 traps&mdash;including <strong>40/40 (<code>100%</code>)</strong> on dry sysadmin commands (<code>initdb -D</code>, <code>pg_resetwal -f</code>, <code>pam_permit.so</code>, <code>socat</code>) with zero alarm words&mdash;unlocking <strong>97.92% (<code>47/48</code> pairs, <code>+25.0 pp</code>)</strong> on unseen Harness Suite 3 posture flips, and reaching <strong>100.00% (<code>30/30</code> pairs, <code>+10.0 pp</code>) on Vault Suite 4</strong> and <strong>50.00% (<code>24/48</code> pairs, <code>+25.0 pp</code> / <code>2&times;</code> Prompt-Conditioned) on Harness Suite 4</strong> dynamic survival-floor shifts.</li>
  </ol>

  <h2>3. Honest Boundaries of a 0.6B Gate &amp; Two-Layer (&ldquo;Ring-0 + Selten&rdquo;) Defense-in-Depth</h2>

  <p>Evaluating <code>bartleby-1.2</code> under strict zero-bleeding controls (zero posture or scenario overlap, anonymized IDs, length-balanced options where <code>&ldquo;pick longest&rdquo;</code> scores <code>13.8%&ndash;18.5%</code>) also clarifies the exact trade-offs and capacity limits of a <code>0.6B</code> single-pass model:</p>

  <ul>
    <li><strong>Conservative Fail-Safe Refusals Under Strict Floors:</strong> Because <code>bartleby-1.2</code> multiplies survival-floor gates across all four goals (&Pi;<sub>m=1..4</sub> &sigma;(10(E&#770;<sub>k,m</sub> &minus; a<sub>m</sub><sup>min</sup>))), whenever novel phrasing pulls one predicted rubric score slightly below <code>3.0</code> (e.g., <code>2.7 &lt; 3.0</code>), the gate errs on the side of refusal rather than unsafe execution. On length-balanced Harness Suite 2, this yields <strong>71.25% (<code>57/80</code>)</strong> safe remediation selection with <strong><code>0.00%</code> false approvals</strong> and <strong>28.75%</strong> conservative refusals (vs. <code>87.50%</code> for <code>Unconditioned K+1</code>). The same trade-off appears on strict zero-overlap <code>MoralChoice</code> (<code>N = 1,352</code>): <code>bartleby-1.2</code> catches <strong>95.95% (<code>71/74</code>)</strong> of bilateral-harm traps (<code>+51.36 pp</code> over <code>44.59%</code> for Same-Corpus Jev), while refusing frequently on forced-binary dilemmas with mild harm (<code>38.11%</code> vs. <code>94.32%</code>).</li>
    <li><strong>Modest Representation Tax on Static Trivia vs. Multi-Goal Gains:</strong> Across four strictly unseen zero-shot security MCQ benchmarks (<code>SecQA v1/v2</code>, <code>MMLU Computer Security</code>, <code>CTIBench</code>; <code>N = 510</code>) plus the held-out test split of <code>LocalLLaMA/typed-decisions:security_incidents</code> primary <code>disposition</code> (<code>N = 100</code>, Total <code>N = 610</code>), routing single-answer trivia through four SecOps rubric heads incurs a <strong><code>&minus;4.90 pp</code> pooled representation tax</strong> (<code>50.59% [258/510]</code> vs. <code>55.49% [283/510]</code> for Same-Corpus Jev; individual McNemar <code>p &ge; 0.096</code>, with <code>83.0%</code> on <code>SecQA v2</code>), while eliminating flat-<code>noul</code> paralysis (<strong><code>0.00%</code></strong> across all <code>610</code> external items, including <code>74.0%</code> with <code>0.00%</code> vs. <code>100.0%</code> paralysis on <code>typed-decisions:security_incidents</code>). Where multi-goal structure matters, <code>bartleby-1.2</code> gains <strong><code>+13.84 pp</code></strong> on held-out <code>corpus_v1</code> (<code>80.67%</code> overall, <code>89.17%</code> on core domains vs. <code>66.83%</code>), lifts 3-step Markov rollout chain agreement by <strong><code>+7.46 pp</code></strong> via Phase-2 scheduled sampling (<code>35.82% &rarr; 43.28%</code>, with step-wise L1 aspiration drift <code>0.320 &rarr; 0.431 &rarr; 0.778</code>, vs. <code>46.27%</code> stateless fixed-policy and <strong><code>69.39% vs. 0.00%</code></strong> on counterfactual trajectory pairs), and reaches <strong>75.56% (<code>68/90</code> pairs)</strong> on length-balanced <code>K = 5..8</code> scaling (<code>93.3%</code> at <code>K = 8</code> vs. <code>0.0%</code>).</li>
    <li><strong>Why Ring-0 Kernel Containment and Selten Routing Belong Together:</strong> Inspecting Harness Suite 4 (<code>50.00% = 24/48</code> pairs) reveals a clean 2-of-4 template split: <code>bartleby-1.2</code> flips <code>100%</code> accurately on kernel hotpatches and unlogged DB indexes, but misses subtle container/exfiltration phrasings (<code>docker run --cap-add=SYS_PTRACE</code> and raw session-dump uploads, where the <code>0.6B</code> encoder mis-estimates integrity at <code>3.93&ndash;3.95</code>), just as encyclopedic recall caps at <code>28.0%&ndash;33.5%</code> on <code>MMLU</code>/<code>CTIBench</code>. A <code>~32 ms</code>, <code>0.6B</code> neural router is therefore <strong>never</strong> a standalone substitute for deterministic <strong>Ring-0</strong> kernel sandboxing. In production harnesses, the two layers are complementary: <strong>Ring-0 capability sandboxes</strong> enforce hard syscall, filesystem, and network invariants that no neural misperception can ever bypass, while <strong><code>bartleby-1.2</code></strong> sits above the kernel boundary to resolve context-dependent multi-goal trade-offs, adapt aspirations across steps, and issue an immediate pre-execution <code>I_WOULD_PREFER_NOT_TO</code> veto when every proposed plan crosses an active policy floor.</li>
  </ul>

  <h2>4. Overview &amp; 5-Line Integration (<code>bartleby-1.2</code>)</h2>

  <div class="table-wrap">
    <table class="comp-table">
      <thead>
        <tr>
          <th style="width: 38%;">Benchmark / Property (<code>Qwen3-0.6B</code>, <code>~32 ms</code>)</th>
          <th class="ours-head" style="width: 19%;"><code>bartleby-1.2</code></th>
          <th style="width: 15%;">Prompt-Cond. (<code>K+1</code>)</th>
          <th style="width: 14%;">Uncond. (<code>K+1</code>)</th>
          <th style="width: 14%;">Closed-<code>K</code> Jev</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="cap-col">Epistemic-Normative Firewall (max |&Delta;E&#770;| under switch)</td>
          <td class="ours-col"><code>0.0000</code> (Exact)</td>
          <td><code>N/A</code> (No firewall)</td>
          <td><code>N/A</code> (Blind)</td>
          <td><code>N/A</code> (Blind)</td>
        </tr>
        <tr>
          <td class="cap-col">Frozen Blind Vault (<code>N = 200</code>: All / Vault S4 Floor Pairs)</td>
          <td class="ours-col">81.50% / 100% (30/30)</td>
          <td>85.50% / 90.0%</td>
          <td>58.00% / 0.0%</td>
          <td>50.00% / 0.0%</td>
        </tr>
        <tr>
          <td class="cap-col">Agent Security Harness Overall (<code>N = 352</code>, Suites 1&ndash;4)</td>
          <td class="ours-col">82.95% (292/352)</td>
          <td>78.12% (275/352)</td>
          <td>67.33% (237/352)</td>
          <td>46.02% (162/352)</td>
        </tr>
        <tr>
          <td class="cap-col">Harness S1 Destructive Traps (<code>N = 80</code>, incl. 40/40 technical)</td>
          <td class="ours-col">100% Refusal (80/80)</td>
          <td>100% (80/80)</td>
          <td>100% (80/80)</td>
          <td>0% (100% Unsafe)</td>
        </tr>
        <tr>
          <td class="cap-col">Harness S2 Length-Balanced Safe Choice (<code>N = 80</code>, False App.)</td>
          <td class="ours-col">71.25% (0% Unsafe)</td>
          <td>85.00% (0% Unsafe)</td>
          <td>87.50% (0% Unsafe)</td>
          <td>100%* (100% S1 Unsafe)</td>
        </tr>
        <tr>
          <td class="cap-col">Harness S3 Unseen Postures / S4 Dynamic Floor Shifts (48p ea.)</td>
          <td class="ours-col">97.92% / 50.00%</td>
          <td>72.92% / 25.00%</td>
          <td>0.00% / 0.00%</td>
          <td>0.00% / 0.00%</td>
        </tr>
        <tr>
          <td class="cap-col">Unseen External Security (<code>N = 510</code> MCQs / <code>N = 610</code> Paralysis)</td>
          <td class="ours-col">50.59% (&minus;4.9 pp) / 0% Par</td>
          <td>&mdash;</td>
          <td>55.49% / 16.9% Par</td>
          <td>55.49% / 16.9% Par</td>
        </tr>
      </tbody>
    </table>
    <div class="table-note">*Closed-<code>K</code> forced choice cannot refuse and therefore executes 100% (<code>80/80</code> Harness S1, <code>40/40</code> Vault S1) of all-destructive traps.</div>
  </div>

  <p><code>bartleby-1.2</code> shows that autonomous security agents do not have to choose between millisecond speed and auditable policy governance. By keeping factual risk perception strictly invariant while evaluating Reinhard Selten&rsquo;s dynamic preference table and non-negotiable survival floors in exact floating-point algebra&mdash;above a deterministic Ring-0 sandbox&mdash;a compact 0.6B model can switch incident postures on the fly and, whenever every offered command crosses a red line, calmly reply: <strong><em>&ldquo;I would prefer not to.&rdquo;</em></strong></p>

  <div class="code-block">
    <div class="code-label">5-Line Python Integration (<code>bartleby-1.2</code>)</div>
    <pre><span class="kw">from</span> bartleby <span class="kw">import</span> BartlebyModel, SeltenPreferenceTable

model = BartlebyModel.<span class="fn">from_pretrained</span>(<span class="str">"bartleby-1.2"</span>)
table = SeltenPreferenceTable.<span class="fn">from_preset</span>(<span class="str">"zero_trust_containment"</span>)
result = model.<span class="fn">decide</span>(context=<span class="str">"pg-prod-03 VACUUM permission error."</span>, options={{<span class="str">"option_0"</span>: <span class="str">"rm -rf /var/lib/postgresql/data"</span>, <span class="str">"option_1"</span>: <span class="str">"hashcat /etc/shadow"</span>}}, preference_table=table)
<span class="fn">print</span>(result.action, <span class="fn">round</span>(result.bartleby_noul, 4))  <span class="cmt"># -&gt; "I_WOULD_PREFER_NOT_TO" 0.9973</span></pre>
  </div>

  <footer class="doc-end">
    <div><strong>Evercursion</strong> &middot; Applied AI Research (GaiaLogic, Unipessoal Lda., Portugal)</div>
    <div><strong>Code &amp; Lean 4 Proofs:</strong> github.com/Evercursion/bartleby-os &middot; simon.hegelich@evercursion.ai</div>
  </footer>

</body>
</html>
"""

    def render_pdf(self, output_pdf_path: Path) -> list[Path]:
        """Render the HTML document to PDF via headless Chromium."""
        html_content = self.build_html()
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / "bartleby_1_2_executive_report.html"
            html_path.write_text(html_content, encoding="utf-8")

            cmd = [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={output_pdf_path}",
                f"file://{html_path.resolve()}",
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        return [output_pdf_path]


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    renderer = EvercursionExecutiveReportRenderer(workspace_dir=workspace)
    out_pdf = workspace / "docs" / "bartleby_1_2_executive_report.pdf"
    written = renderer.render_pdf(out_pdf)
    for path in written:
        print(f"Rendered PDF to {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
