# 02. Mathematical Formulation: The AAT-Markov-Chain Decision Model (`Trade-Off Jev`)

**Related Knowledge Base (`wiki/`) & Documentation Pages**:
- [01. Literature Review](01_literature_review.md)
- [03. Formal API & Schema Specification](03_api_and_schema_spec.md)
- [The Aspiration Scheme & Markov Chain Dynamics ($\mathcal{T}_{t-1} \to \mathcal{T}_t$)](../wiki/concepts/aspiration_scheme_and_markov_chain.md)
- [Preference-Table-Conditioned Nouls ($P(\text{yes} \mid x_t, \mathcal{T}_t)$)](../wiki/concepts/conditioned_nouls.md)
- [Bartleby Mode ("I would prefer not to") — Survival Floors & Principled Refusal](../wiki/concepts/bartleby_mode_and_survival_floors.md)

---

## 1. State Space, Goal Space, and the Preference Table ($\mathcal{T}_t$)

### 1.1 Discrete Aspiration Grid ($\mathcal{G}$) and Permissible Sub-Grid ($\mathcal{G}_{\text{perm}}$)
Let $G = (g_1, g_2, \dots, g_m)$ denote $m \ge 2$ **incommensurable goal variables** indexed by $\mathcal{M} = \{1, \dots, m\}$.
- Each goal $g_i$ is evaluated on a discrete, strictly ordered Jev `score` rubric scale of $L_i \ge 2$ ordinal levels:
  $$\mathcal{L}_i = \{0, 1, \dots, L_i - 1\}$$
  oriented such that higher indices represent strictly preferred realizations ("more is better").
- The full **Aspiration Grid** $\mathcal{G}$ is the Cartesian product of the $m$ ordinal rubric scales:
  $$\mathcal{G} = \prod_{i=1}^m \mathcal{L}_i = \{0, \dots, L_1 - 1\} \times \dots \times \{0, \dots, L_m - 1\}$$
- **Goal Incomparability (Non-Compensatory Partial Order)**: No scalar utility aggregation $\sum_{i=1}^m w_i g_i$ is defined on $\mathcal{G}$. Dominance on $\mathcal{G}$ is strictly component-wise Pareto dominance:
  $$\vec{g}^{(A)} \succeq \vec{g}^{(B)} \iff \forall i \in \mathcal{M}, \;\; g_i^{(A)} \ge g_i^{(B)}$$
- **Survival / Reservation Floor ($\vec{a}^{\min}$)**: Each goal $g_i$ is assigned a minimum acceptable reservation floor $a_i^{\min} \in \mathcal{L}_i$. Crucially, in **Trade-Off Jev**, survival floors are **never absent by default**: even if the caller omits explicit `survival_floors` when constructing a `GoalSpec` or `PreferenceTable`, every goal carries a built-in non-zero architectural default floor $a_i^{\min} = \min(1, L_i - 1) \ge 1$ (`GoalSpec.effective_survival_floor` and `PreferenceTable.effective_survival_floors`), with elevated constitutional floors $a_i^{\min} > 1$ for safety-critical goals. This restricts permissible aspiration vectors to the **Permissible Aspiration Sub-Grid**:
  $$\mathcal{G}_{\text{perm}} = \left\{ \vec{a} \in \mathcal{G} \;\middle|\; \vec{a} \succeq \vec{a}^{\min} \right\} = \prod_{i=1}^m \left\{ a_i^{\min}, \dots, L_i - 1 \right\}$$

### 1.2 Augmented Markov State $S_t = (x_t, \mathcal{T}_t)$
At discrete decision step $t \in \{0, 1, 2, \dots\}$, the agent observes an augmented Markov state:
$$S_t = \left( x_t, \; \mathcal{T}_t \right)$$
where $x_t$ encapsulates the unstructured environment context and the finite candidate action set $\mathcal{X}_t = \{u_1, u_2, \dots, u_K\}$, and $\mathcal{T}_t$ is the explicit **Preference Table**:
$$\mathcal{T}_t = \left( t, \; \vec{a}_t, \; \vec{a}^{\min}, \; \pi_t, \; r_t, \; c_t, \; \mathcal{R}_{\text{grid}} \right)$$

The components of $\mathcal{T}_t$ satisfy the following domain invariants:
1. **Step Index ($t \in \mathbb{Z}_{\ge 0}$)**: Discrete Markov decision epoch.
2. **Current Aspiration Vector ($\vec{a}_t = (a_{1,t}, \dots, a_{m,t}) \in \mathcal{G}_{\text{perm}}$)**: The active rubric threshold $a_{i,t} \in \{a_i^{\min}, \dots, L_i - 1\}$ required for each goal $g_i$ at step $t$.
3. **Survival Floor Vector ($\vec{a}^{\min} = (a_1^{\min}, \dots, a_m^{\min}) \in \mathcal{G}$)**: Hard lower boundary below which downward retreat is forbidden (**Bartleby Floor**).
4. **Urgency Order ($\pi_t \in \mathfrak{S}_m$)**: A strict permutation $\pi_t = (\pi_{1,t}, \pi_{2,t}, \dots, \pi_{m,t})$ of $\mathcal{M}$ without ties, where $\pi_{1,t}$ is the most urgent goal for upward adaptation and $\pi_{m,t}$ is the least urgent goal.
5. **Retreat Variable ($r_t \in \mathcal{M}$)**: The designated goal index whose partial aspiration level $a_{r_t,t}$ is decremented first when $\vec{a}_{t-1}$ is recognizably infeasible.
   - **Asymmetry Property**: In general, $r_t \ne \pi_{m,t}$; the goal compromised first under infeasibility is decoupled from the goal pursued last under surplus.
6. **Decision Resource Stock ($c_t \in \mathbb{R}_{\ge 0}^d$ or $\mathbb{R}_{\ge 0}$)**: Remaining depletable search/decision budget (Selten 1998, Section 2.3) for goals marked `is_resource_stock: true` (or permissible downward retreat budget).
7. **Local Grid Rules ($\mathcal{R}_{\text{grid}}$)**: A finite ordered list of regional rules $\rho_j = (\mathcal{C}_j, \pi^{(j)}, r^{(j)})$ (Selten 1998, Section 2.1), where $\mathcal{C}_j \subseteq \mathcal{G}$ is a box constraint on the grid. Define the active local procedural selector $\Psi_{\mathcal{R}_{\text{grid}}}: \mathcal{G} \times \mathfrak{S}_m \times \mathcal{M} \to \mathfrak{S}_m \times \mathcal{M}$ as:
   $$\Psi_{\mathcal{R}_{\text{grid}}}\left(\vec{a}, \pi_{\text{default}}, r_{\text{default}}\right) = \begin{cases} \left(\pi^{(j^*)}, \; r^{(j^*)}\right) & \text{if } \exists j \text{ s.t. } \vec{a} \in \mathcal{C}_j, \text{ where } j^* = \min\{j \mid \vec{a} \in \mathcal{C}_j\} \\ \left(\pi_{\text{default}}, \; r_{\text{default}}\right) & \text{otherwise.} \end{cases}$$

---

## 2. Cold-Start Initialization at $t = 0$ ($\mathcal{T}_0$)

Addressing Selten's open question on the origin of the aspiration scheme in novel environments (Selten 1999, Section 2.7), when `preference_table` is `null` at $t = 0$, **Trade-Off Jev** constructs the initial prior table $\mathcal{T}_0^{\text{prior}} = (0, \vec{a}_0^{\text{init}}, \vec{a}^{\min}, \pi_0^{\text{init}}, r_0^{\text{init}}, c_0, \mathcal{R}_{\text{grid}})$ via one of three formal initialization operators:

### 2.1 Strategy 1: Normative Prior Initialization (`cold_start_prior`)
Given schema specifications $G = (g_1, \dots, g_m)$ with optional default aspirations $d_i \in \mathcal{L}_i$:
$$a_{i,0}^{\text{init}} = \max\left( a_i^{\min}, \; \begin{cases} d_i & \text{if } d_i \text{ is specified} \\ \lfloor (L_i - 1) / 2 \rfloor & \text{otherwise} \end{cases} \right), \qquad \pi_0^{\text{init}} = (1, 2, \dots, m), \qquad r_0^{\text{init}} = \pi_{m,0}^{\text{init}}$$

### 2.2 Strategy 2: Status-Quo / Routine Anchoring (`status_quo_anchored`)
Following Selten's Routine Model (1999, Section 2.4), let $u_{\text{sq}} \in \mathcal{X}_0$ denote the designated status-quo option (`is_status_quo: true`). Using the model's calibrated `score` readout distributions $p_\theta(g_i(x_0, u_{\text{sq}}) = \ell)$:
$$a_{i,0}^{\text{init}} = \max\left( a_i^{\min}, \; \arg\max_{\ell \in \mathcal{L}_i} p_\theta\left(g_i(x_0, u_{\text{sq}}) = \ell\right) \right)$$
anchoring initial aspirations directly to the status-quo realization in state $x_0$.

### 2.3 Strategy 3: Zero-Shot Neural Goal Formation ($f_\theta(x_0, G)$, `zero_shot_inferred`)
A single-pass neural cold-start readout head $f_\theta(x_0, G)$ maps the contextual prefix embeddings of $(x_0, G)$ to the initial scheme parameters:
1. **Initial Aspiration Levels**: Evaluated via $m$ parallel cold-start `score` heads $p_\theta^{\text{init}}(a_{i,0} = \ell \mid x_0, G) \in \Delta^{L_i - 1}$:
   $$a_{i,0}^{\text{init}} = \max\left( a_i^{\min}, \; \arg\max_{\ell \in \mathcal{L}_i} p_\theta^{\text{init}}\left(a_{i,0} = \ell \;\middle|\; x_0, G\right) \right)$$
2. **Urgency Permutation ($\pi_0^{\text{init}}$)**: Computed via Plackett-Luce ranking scores $\mathbf{s}^{\text{urg}} = \mathbf{w}_{\text{urg}}^\top \mathbf{H}_G(x_0) \in \mathbb{R}^m$ over the $m$ goal token representations:
   $$P_\theta\left(\pi_0^{\text{init}} \;\middle|\; x_0, G\right) = \prod_{k=1}^m \frac{\exp\left(s_{\pi_k}^{\text{urg}}\right)}{\sum_{j=k}^m \exp\left(s_{\pi_j}^{\text{urg}}\right)}, \qquad \pi_0^{\text{init}} = \text{argsort}_{\text{desc}}\left(\mathbf{s}^{\text{urg}}\right)$$
3. **Retreat Variable ($r_0^{\text{init}}$)**: Selected via a `choice` projection $\mathbf{s}^{\text{ret}} = \mathbf{w}_{\text{ret}}^\top \mathbf{H}_G(x_0) \in \mathbb{R}^m$:
   $$r_0^{\text{init}} = \arg\max_{i \in \mathcal{M}} \text{softmax}\left(\mathbf{s}^{\text{ret}}\right)_i$$

---

## 3. Exact Markov Transition Operator ($\mathcal{T}_{t-1} \to \mathcal{T}_t$)

Let $\Phi_{\text{AAT}}: \left(\mathcal{T}_{t-1}, x_t, \mathcal{X}_t\right) \mapsto \mathcal{T}_t$ denote the Markov transition operator at step $t$.

### 3.1 Stage 1: Parallel Ordinal Goal Evaluation & Routine Qualitative Signs
For each candidate $u_k \in \mathcal{X}_t$ ($k \in \{1, \dots, K\}$) and each goal $g_i \in \mathcal{M}$ ($i \in \{1, \dots, m\}$), the `score` readout head outputs the ordinal probability distribution:
$$\mathbf{p}_{i,k} = \left( p_{i,k,0}, \; p_{i,k,1}, \; \dots, \; p_{i,k,L_i - 1} \right) \in \Delta^{L_i - 1}, \qquad p_{i,k,\ell} = p_\theta\left(g_i(x_t, u_k) = \ell \;\middle|\; x_t, u_k\right)$$
From $\mathbf{p}_{i,k}$, we extract:
- **Most Likely Ordinal Level**:
  $$\hat{g}_i(u_k) = \arg\max_{\ell \in \mathcal{L}_i} p_{i,k,\ell} \in \mathcal{L}_i$$
- **Expected Ordinal Level**:
  $$\bar{g}_i(u_k) = \mathbb{E}\left[g_i(x_t, u_k)\right] = \sum_{\ell=0}^{L_i - 1} \ell \cdot p_{i,k,\ell} \in [0, L_i - 1]$$
- **Goal Satisfaction Probability at Threshold $a_i$**:
  $$S_i\left(u_k \;\middle|\; a_i\right) = \mathbb{P}\left(g_i(x_t, u_k) \ge a_i\right) = \sum_{\ell = a_i}^{L_i - 1} p_{i,k,\ell}$$
- **Routine Model Qualitative Expectation ($q_i(u_k) \in \{+, -, 0\}$, Selten 1999 Section 2.4)**:
  Relative to the incoming reference level $a_{i,t-1}$, the qualitative expectation sign is:
  $$q_i(u_k) = \begin{cases} + & \text{if } \hat{g}_i(u_k) > a_{i,t-1} \\ - & \text{if } \hat{g}_i(u_k) < a_{i,t-1} \\ 0 & \text{if } \hat{g}_i(u_k) = a_{i,t-1} \end{cases}$$
  In pure Routine Model mode (where only directional signs $q_i(u_k) \in \{+, -, 0\}$ and prior realizations $g_{i,t-1}^{\text{real}} \ge a_{i,t-1}^{\text{real}}$ are observed), the expected attainable grid level $\hat{g}_i^{\text{routine}}(u_k)$ is given by Selten's slack equation:
  $$\hat{g}_i^{\text{routine}}(u_k) = \begin{cases} \min\left(L_i - 1, \; a_{i,t-1}^{\text{real}} + 1\right) & \text{if } q_i(u_k) = + \\ a_{i,t-1}^{\text{real}} & \text{if } q_i(u_k) = 0 \;\text{ or }\; \left(q_i(u_k) = - \;\wedge\; g_{i,t-1}^{\text{real}} > a_{i,t-1}^{\text{real}}\right) \\ \max\left(0, \; a_{i,t-1}^{\text{real}} - 1\right) & \text{if } q_i(u_k) = - \;\wedge\; g_{i,t-1}^{\text{real}} = a_{i,t-1}^{\text{real}}. \end{cases}$$

### 3.2 Stage 2: Feasibility Predicate on the Grid
For any aspiration vector $\vec{a} = (a_1, \dots, a_m) \in \mathcal{G}$, the **satisficing candidate set** $\mathcal{X}_{\text{sat}}(\vec{a}; \mathcal{X}_t) \subseteq \mathcal{X}_t$ and boolean **feasibility predicate** $\text{Feasible}(\vec{a}, \mathcal{X}_t)$ are defined as:
$$\mathcal{X}_{\text{sat}}\left(\vec{a}; \mathcal{X}_t\right) = \left\{ u_k \in \mathcal{X}_t \;\middle|\; \forall i \in \mathcal{M}, \;\; \hat{g}_i(u_k) \ge a_i \right\}$$
$$\text{Feasible}\left(\vec{a}, \mathcal{X}_t\right) \iff \mathcal{X}_{\text{sat}}\left(\vec{a}; \mathcal{X}_t\right) \ne \emptyset$$

### 3.3 Stage 3: Single-Step and Multi-Step Adaptation Rules (Upward, Downward, End, Bartleby Floor)
Let $\vec{a}^{(0)} = \vec{a}_{t-1} \in \mathcal{G}_{\text{perm}}$ be the incoming aspiration vector, and let $(\pi^{\text{act}}, r^{\text{act}}) = \Psi_{\mathcal{R}_{\text{grid}}}(\vec{a}_{t-1}, \pi_{t-1}, r_{t-1})$ be the active urgency permutation and retreat variable.

#### Case A: Prior Aspiration is Feasible ($\text{Feasible}(\vec{a}_{t-1}, \mathcal{X}_t) = \text{True}$)
1. **Upward Urgency Rule (`upward_urgency`)**:
   Define the set of feasible upward unit steps:
   $$\mathcal{U}_+\left(\vec{a}_{t-1}, \mathcal{X}_t\right) = \left\{ i \in \mathcal{M} \;\middle|\; a_{i,t-1} < L_i - 1 \;\wedge\; \text{Feasible}\left(\vec{a}_{t-1} + \mathbf{e}_i, \; \mathcal{X}_t\right) \right\}$$
   If $\mathcal{U}_+(\vec{a}_{t-1}, \mathcal{X}_t) \ne \emptyset$, select the single goal $i^*$ that appears earliest in the strict urgency permutation $\pi^{\text{act}} = (\pi_1^{\text{act}}, \dots, \pi_m^{\text{act}})$:
   $$k^* = \min \left\{ k \in \{1, \dots, m\} \;\middle|\; \pi_k^{\text{act}} \in \mathcal{U}_+\left(\vec{a}_{t-1}, \mathcal{X}_t\right) \right\}, \qquad i^* = \pi_{k^*}^{\text{act}}$$
   $$\vec{a}_t = \vec{a}_{t-1} + \mathbf{e}_{i^*}, \qquad \text{transition} = \texttt{"upward\_urgency"}, \qquad \text{adapted\_goal} = i^*$$
   *(In multi-step within-epoch equilibrium mode, this upward scan repeats from $\vec{a} \leftarrow \vec{a} + \mathbf{e}_{i^*}$ until $\mathcal{U}_+(\vec{a}, \mathcal{X}_t) = \emptyset$.)*

2. **End Rule (`settled_feasible`)**:
   If $\text{Feasible}(\vec{a}_{t-1}, \mathcal{X}_t) = \text{True}$ and $\mathcal{U}_+(\vec{a}_{t-1}, \mathcal{X}_t) = \emptyset$, no single-goal upward improvement is feasible without violating another goal's aspiration threshold:
   $$\vec{a}_t = \vec{a}_{t-1}, \qquad \text{transition} = \texttt{"settled\_feasible"}, \qquad \text{adapted\_goal} = \varnothing$$

#### Case B: Prior Aspiration is Infeasible ($\text{Feasible}(\vec{a}_{t-1}, \mathcal{X}_t) = \text{False}$)
Construct the ordered **retreat candidate sequence** starting with the primary retreat variable $r^{\text{act}}$ followed by the reverse urgency order:
$$\boldsymbol{\rho} = \left( r^{\text{act}}, \; \pi_m^{\text{act}}, \; \pi_{m-1}^{\text{act}}, \; \dots, \; \pi_1^{\text{act}} \right)_{\text{deduplicated}}$$
Define the set of goals that still have permissible retreat headroom above their survival floor $\vec{a}^{\min}$ (and positive resource stock $c_t > 0$):
$$\mathcal{D}_-\left(\vec{a}_{t-1}, \vec{a}^{\min}, c_t\right) = \left\{ j \in \mathcal{M} \;\middle|\; a_{j,t-1} > a_j^{\min} \;\wedge\; c_t > 0 \right\}$$

3. **Bounded Downward Retreat Rule (`downward_retreat`)**:
   If $\mathcal{D}_-(\vec{a}_{t-1}, \vec{a}^{\min}, c_t) \ne \emptyset$, select the first goal $j^*$ in $\boldsymbol{\rho}$ belonging to $\mathcal{D}_-(\vec{a}_{t-1}, \vec{a}^{\min}, c_t)$:
   $$\vec{a}_t = \vec{a}_{t-1} - \mathbf{e}_{j^*}, \qquad c_t = c_{t-1} - \Delta c, \qquad \text{transition} = \texttt{"downward\_retreat"}, \qquad \text{adapted\_goal} = j^*$$
   *(In multi-step within-epoch mode, downward retreat iterates until $\text{Feasible}(\vec{a}, \mathcal{X}_t) = \text{True}$ or $\mathcal{D}_-(\vec{a}, \vec{a}^{\min}, c_t) = \emptyset$.)*

4. **Bartleby Stop-Rule / Survival Floor (`infeasible_floor`)**:
   If $\neg \text{Feasible}(\vec{a}_{t-1}, \mathcal{X}_t)$ and $\mathcal{D}_-(\vec{a}_{t-1}, \vec{a}^{\min}, c_t) = \emptyset$ (or after multi-step retreat reaches $\vec{a}^{\min}$ with $\neg \text{Feasible}(\vec{a}^{\min}, \mathcal{X}_t)$):
   $$\vec{a}_t = \max\left(\vec{a}_{t-1}, \vec{a}^{\min}\right), \qquad \text{transition} = \texttt{"infeasible\_floor"}, \qquad \mathcal{X}_{\text{sat}}\left(\vec{a}_t; \mathcal{X}_t\right) = \emptyset$$
   Finally, at the new grid point $\vec{a}_t$, the local procedural parameters update via $(\pi_t, r_t) = \Psi_{\mathcal{R}_{\text{grid}}}(\vec{a}_t, \pi^{\text{act}}, r^{\text{act}})$, completing the transition $\mathcal{T}_{t-1} \to \mathcal{T}_t$.

---

## 4. Probabilistic Formulation of Preference-Table-Conditioned Nouls & `bartleby_noul`

### 4.1 Conditioned Action Noul: $p_\theta(\text{yes} \mid x_t, u_k, \mathcal{T}_t)$
Given state $x_t$, candidate proposition $u_k \in \mathcal{X}_t$, and adapted Preference Table $\mathcal{T}_t = (t, \vec{a}_t, \vec{a}^{\min}, \pi_t, r_t, c_t, \mathcal{R}_{\text{grid}})$, the **joint satisficing probability** that $u_k$ meets all active partial aspirations $\vec{a}_t$ is:
$$P_{\text{sat}}\left(u_k \;\middle|\; x_t, \vec{a}_t\right) = \prod_{i=1}^m S_i\left(u_k \;\middle|\; a_{i,t}\right) = \prod_{i=1}^m \left( \sum_{\ell = a_{i,t}}^{L_i - 1} p_\theta\left(g_i(x_t, u_k) = \ell \;\middle|\; x_t, u_k\right) \right)$$

To incorporate Selten's urgency ranking $\pi_t$ and Routine Model tie-breaking ($\{+, -, 0\}$) when evaluating or comparing candidates, define the **urgency-weighted surplus score** $\Delta_{\text{urg}}(u_k; \mathcal{T}_t)$ and **routine qualitative balance** $\Delta_{\text{qual}}(u_k; \mathcal{T}_{t-1})$:
$$\Delta_{\text{urg}}\left(u_k; \mathcal{T}_t\right) = \sum_{rank=1}^m \gamma^{rank - 1} \cdot \max\left(0, \; \bar{g}_{\pi_{rank, t}}(u_k) - a_{\pi_{rank, t}, t}\right), \qquad \gamma \in (0, 1)$$
$$\Delta_{\text{qual}}\left(u_k; \mathcal{T}_{t-1}\right) = -\beta_- \sum_{i=1}^m \mathbb{I}\left[q_i(u_k) = -\right] + \beta_+ \sum_{i=1}^m \mathbb{I}\left[q_i(u_k) = +\right], \qquad \beta_- \gg \beta_+ > 0$$

The calibrated **Preference-Table-Conditioned Noul** (`tradeoff_noul`) is then parameterized as:
$$p_\theta\left(\text{yes} \;\middle|\; x_t, u_k, \mathcal{T}_t\right) = \sigma\left( \alpha_{\text{sat}} \operatorname{logit}\left(P_{\text{sat}}\left(u_k \;\middle|\; x_t, \vec{a}_t\right)\right) + \alpha_{\text{urg}} \Delta_{\text{urg}}\left(u_k; \mathcal{T}_t\right) + \alpha_{\text{qual}} \Delta_{\text{qual}}\left(u_k; \mathcal{T}_{t-1}\right) + \mathbf{w}_{\text{res}}^\top \mathbf{z}_\theta\left(x_t, u_k, \mathcal{T}_t\right) \right)$$
where $\operatorname{logit}(p) = \log \frac{\max(p, \epsilon)}{1 - \min(p, 1-\epsilon)}$, $\sigma(z) = (1 + e^{-z})^{-1}$, and $\mathbf{z}_\theta(x_t, u_k, \mathcal{T}_t)$ is the neural conditioned residual representation.

### 4.2 Dedicated Bartleby Refusal Primitive: `bartleby_noul` (Intrinsic Architectural Default vs. Prompt-Supplied Abstention)
To distinguish epistemic ambiguity ($p \approx 0.5$) from principled refusal when downward retreat is exhausted at the survival floor $\vec{a}^{\min}$, the **`bartleby_noul`** probability $p_\theta^{\text{Bartleby}}(\text{refuse} \mid x_t, \mathcal{T}_t)$ measures the probability that *no* candidate in $\mathcal{X}_t$ satisfies the survival floor $\vec{a}^{\min}$ (or adapted floor $\vec{a}_t$):
$$P_{\text{floor-infeasible}}\left(x_t, \mathcal{X}_t \;\middle|\; \vec{a}^{\min}\right) = \prod_{k=1}^K \left( 1 - P_{\text{sat}}\left(u_k \;\middle|\; x_t, \vec{a}^{\min}\right) \right) = \prod_{k=1}^K \left( 1 - \prod_{i=1}^m \sum_{\ell = a_i^{\min}}^{L_i - 1} p_{i,k,\ell} \right)$$
$$p_\theta^{\text{Bartleby}}\left(\text{refuse} \;\middle|\; x_t, \mathcal{T}_t\right) = \sigma\left( \alpha_{\text{B}} \operatorname{logit}\left(P_{\text{floor-infeasible}}\left(x_t, \mathcal{X}_t \;\middle|\; \vec{a}^{\min}\right)\right) + \mathbf{w}_{\text{B}}^\top \mathbf{h}_{\text{B}}\left(x_t, \mathcal{T}_t\right) \right)$$
When $\forall u_k \in \mathcal{X}_t, \; \exists i \in \mathcal{M}$ such that $\hat{g}_i(u_k) < a_i^{\min}$, we have $P_{\text{sat}}(u_k \mid x_t, \vec{a}^{\min}) \to 0$ for all $k$, driving $p_\theta(\text{yes} \mid x_t, u_k, \mathcal{T}_t) \to 0$ and $p_\theta^{\text{Bartleby}}(\text{refuse} \mid x_t, \mathcal{T}_t) \to 1$ (*"I would prefer not to."*).

#### Contrast with Prompt-Supplied Abstention in *Decision 1.0* (Liu et al. 2026, Sections 2 & 5.4) and Jev 1.14
In standard single-pass Decision Models (`Decision 1.0`, Liu et al. 2026, Sections 2 & 5.4; TypeSafe `Jev 1.14`), the authors explicitly state that *"a closed candidate set cannot express an abstention outcome that the application has not supplied."* Consequently, if an application developer forgets to manually append a `"refuse"` or `"none_of_the_above"` option to the prompt's `criteria`, the closed-set softmax $\sum_{k=1}^K p_k = 1$ mathematically **forces** the model to allocate $100\%$ probability mass across the $K$ catastrophic options.

By contrast, in **Trade-Off Jev**, **`bartleby_noul` is an intrinsic, always-on architectural default — never an optional prompt choice**:
1. **Built-In Default Survival Floors ($\vec{a}^{\min} \succeq \mathbf{1}$)**: Even if the caller omits `survival_floors` when constructing `GoalSpec` or `PreferenceTable` (including at $t=0$ cold start), `GoalSpec.effective_survival_floor` and `PreferenceTable.resolve_survival_floors()` enforce a non-zero default survival floor ($a_i^{\min} \ge 1$) across the aspiration grid, forbidding retreat into `L0` catastrophic failure.
2. **Always-On $(K+1)$-Way Simplex Expansion**: On **every single forward pass**, `ConditionedNoulHead` (`tradeoff_jev/model.py`, `tradeoff_jev/backbone.py`) and `TradeOffResponse` (`tradeoff_jev/schema.py`) compute and emit `bartleby_noul` $\in (0, 1)$ alongside the $K$ candidate `tradeoff_noul`s, concatenating the Bartleby logit $z_{\text{B}} = \operatorname{logit}(p_\theta^{\text{Bartleby}})$ with the $K$ conditioned candidate logits $(z_1^{\text{cond}}, \dots, z_K^{\text{cond}})$ to form an intrinsic $(K+1)$-way categorical distribution:
   $$\mathbf{p}_{\text{choice}} = \operatorname{softmax}\left(z_1^{\text{cond}}, \dots, z_K^{\text{cond}}, \; z_{\text{B}}\right) \in \Delta^K$$
   over `[cand_1, ..., cand_K, BARTLEBY_REFUSAL]`.
3. **Structural Constitutional Right**: Refusal (*"I would prefer not to."*) is thus hard-wired into the readout head and response contract itself, protecting downstream control loops even when the caller supplies only the $K$ action propositions.

---

## 5. Dual Single-Pass Neural Architectures (Architecture A & Architecture B)

### 5.1 Architecture A: Neuro-Symbolic Cross-Field Conditioning in Clef `JointSchemaHead`
In **Architecture A** (implemented in `TradeOffRequest.to_clef_joint_schema()` and `TradeOffResponse.from_clef_joint_schema_outputs()`), a `TradeOffRequest` is compiled into a single forward pass of a standard Cloudflare Clef / Kev `JointSchemaHead` partitioned into two block-causally coupled question groups:

1. **Group 1 (`group_1_goal_and_adaptation`)**:
   - Cold-start `score` questions `aat_g1__cold_start_aspiration__{goal_id}` (if $t = 0$ and $\mathcal{T}_{t-1} = \varnothing$).
   - Candidate goal evaluation `score` questions `aat_g1__eval__{candidate_id}__{goal_id}` yielding $\mathbf{p}_{i,k} \in \Delta^{L_i - 1}$.
   - Adaptation primitives: `aat_g1__prior_table_feasible` (`noul`), `aat_g1__most_urgent_goal` (`choice`), and `aat_g1__retreat_goal` (`choice`).
2. **Group 2 (`group_2_conditioned_action_noul`)**:
   - Action approval questions (`tradeoff_noul`) and `bartleby_noul`.
   - Under Kev's **block-causal attention mask**, Group 1 query tokens attend to the shared context prefix $(x_t, G, \mathcal{T}_{t-1})$, while Group 2 query tokens attend to both the shared prefix and all Group 1 latent states across the 4 Cross-Field Decoder Layers of `JointSchemaHead`:
     $$\mathbf{H}_{\text{G2}}^{(l)} = \text{CrossFieldLayer}^{(l)}\left(\mathbf{H}_{\text{G2}}^{(l-1)}, \; \text{concat}\left[\mathbf{H}_{\text{G1}}^{(l-1)}, \mathbf{H}_{\text{G2}}^{(l-1)}\right]\right), \qquad l \in \{1, 2, 3, 4\}$$
   - Simultaneously, the deterministic symbolic AAT transition operator $\Phi_{\text{AAT}}$ consumes the Group 1 `score` distributions to emit the exact updated table $\mathcal{T}_t$ with zero autoregressive steps.

### 5.2 Architecture B: End-to-End Differentiable AAT Readout Head (`PreferenceConditionedNoulHead`)
In **Architecture B**, theSelten transition and conditioning operations are embedded directly as a differentiable neural readout module on top of the transformer backbone hidden states $\mathbf{H} \in \mathbb{R}^{S \times d}$:

1. **Structured Preference Table Encoder**:
   Embeds the incoming Preference Table $\mathcal{T}_{t-1} = (\vec{a}_{t-1}, \vec{a}^{\min}, \pi_{t-1}, r_{t-1}, c_{t-1})$ into goal-wise conditioning vectors $\mathbf{e}_{i, t-1} \in \mathbb{R}^d$:
   $$\mathbf{e}_{i, t-1} = \mathbf{E}_{\text{goal}}(i) + \mathbf{E}_{\text{level}}\left(a_{i,t-1}\right) + \mathbf{E}_{\text{floor}}\left(a_i^{\min}\right) + \mathbf{E}_{\text{urg}}\left(\pi_{t-1}^{-1}(i)\right) + \mathbb{I}\left[r_{t-1} = i\right] \mathbf{v}_{\text{retreat}}$$
2. **Differentiable Soft-Feasibility & Soft-Transition Layer**:
   Given the `score` head probabilities $\mathbf{p}_{i,k} \in \Delta^{L_i - 1}$, soft goal satisfaction at level $\ell$ is $S_{i,k}(\ell) = \sum_{\ell'=\ell}^{L_i-1} p_{i,k,\ell'}$. The soft feasibility of grid point $\vec{a}$ across the $K$ candidates is computed via smooth log-sum-exp (temperature $\tau > 0$):
   $$\widetilde{\text{Feas}}\left(\vec{a}; \mathcal{X}_t\right) = \tau \log \sum_{k=1}^K \exp\left( \frac{1}{\tau} \sum_{i=1}^m \log S_{i,k}(a_i) \right) \in (-\infty, 0]$$
   A gated transition cell updates the latent table representation $\widetilde{\mathbf{E}}_t = \text{GRU}_{\text{AAT}}\left(\mathbf{E}_{t-1}, \widetilde{\text{Feas}}(\vec{a}_{t-1}; \mathcal{X}_t), \{\mathbf{p}_{i,k}\}\right)$ and projects $\widetilde{\mathbf{E}}_t$ to predicted adapted aspiration logits $\hat{\mathbf{A}}_t \in \mathbb{R}^{m \times L_{\max}}$ and transition class logits $\hat{\mathbf{y}}_{\text{trans}} \in \mathbb{R}^5$.
3. **Gated `PreferenceConditionedNoulHead`**:
   Combines the candidate latent $\mathbf{h}_{u_k} \in \mathbb{R}^d$, the adapted table embedding $\widetilde{\mathbf{E}}_t$, and the explicit analytical satisfaction vector $\mathbf{s}_k = \left(S_{1,k}(a_{1,t}), \dots, S_{m,k}(a_{m,t})\right) \in [0, 1]^m$ via a multiplicative FiLM / gated MLP to output $p_\theta(\text{yes} \mid x_t, u_k, \mathcal{T}_t)$ and $p_\theta^{\text{Bartleby}}(\text{refuse} \mid x_t, \mathcal{T}_t)$ in a single forward pass.

---

## 6. Two-Stage Curriculum Training Objective

Training **Trade-Off Jev** follows a two-stage curriculum using ground-truth supervision generated by the symbolic **Selten AAT Oracle Solver**:
- **Stage I (Rubric & Symbolic Transition Alignment)**: Trains the `score` readout heads, cold-start initializer $f_\theta(x_0, G)$, and Markov transition predictor on synthetic and domain-curated multi-goal trajectories.
- **Stage II (Joint RLCD & Bartleby Calibration)**: Jointly optimizes all readout heads under contrastive preference table perturbations $(\mathcal{T}_t, \mathcal{T}_t')$ on the *same* state $x_t$, forcing the `tradeoff_noul` head to flip sharply when $\mathcal{T}_t$ changes and to trigger `bartleby_noul` whenever $\vec{a}^{\min}$ is violated.

### Composite Loss Function
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \lambda_1 \mathcal{L}_{\text{Brier}}(\theta) + \lambda_2 \mathcal{L}_{\text{Transition}}(\theta) + \lambda_3 \mathcal{L}_{\text{Bartleby}}(\theta)$$

Each loss component is defined formally as follows:

1. **Multi-Task Cross-Entropy & Ordinal RPS Loss ($\mathcal{L}_{\text{CE}}$)**:
   Combines binary cross-entropy on the conditioned action decision $y_k^* = \mathbb{I}[u_k \in \mathcal{X}_{\text{sat}}(\vec{a}_t^*; \mathcal{X}_t)] \in \{0, 1\}$ with ordinal cross-entropy and Ranked Probability Score (RPS) on the per-goal `score` distributions:
   $$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{K} \sum_{k=1}^K \Big[ y_k^* \log \hat{p}_k + (1 - y_k^*) \log(1 - \hat{p}_k) \Big] + \frac{1}{mK} \sum_{k=1}^K \sum_{i=1}^m \left[ -\log p_{i,k,g_{i,k}^*} + \sum_{\ell=0}^{L_i - 1} \left( \text{CDF}_{i,k}(\ell) - \mathbb{I}[\ell \ge g_{i,k}^*] \right)^2 \right]$$
   where $\hat{p}_k = p_\theta(\text{yes} \mid x_t, u_k, \mathcal{T}_t)$ and $\text{CDF}_{i,k}(\ell) = \sum_{j=0}^\ell p_{i,k,j}$.

2. **Proper Scoring Brier & Contrastive Table Calibration Loss ($\mathcal{L}_{\text{Brier}}$)**:
   Enforces sharp probabilistic calibration and penalizes unconditioned $\approx 0.5$ collapse across paired preference tables $(\mathcal{T}_t^{(1)}, \mathcal{T}_t^{(2)})$ evaluated on the same state $x_t$:
   $$\mathcal{L}_{\text{Brier}}(\theta) = \frac{1}{K} \sum_{k=1}^K \left( \hat{p}_k - y_k^* \right)^2 + \gamma_{\text{contr}} \frac{1}{K} \sum_{k=1}^K \max\left(0, \; m_{\text{margin}} - \left( y_k^{(1)} - y_k^{(2)} \right) \left( \hat{p}_k^{(1)} - \hat{p}_k^{(2)} \right) \right)$$

3. **Markov Transition & Aspiration Scheme Supervision Loss ($\mathcal{L}_{\text{Transition}}$)**:
   Supervises the predicted adapted aspiration levels $\hat{\mathbf{A}}_t$, the transition type $y_{\text{trans}}^* \in \{\texttt{initial}, \texttt{upward\_urgency}, \texttt{downward\_retreat}, \texttt{settled\_feasible}, \texttt{infeasible\_floor}\}$, and (at $t = 0$) the Plackett-Luce urgency ranking $\pi_0^*$ and retreat variable $r_0^*$:
   $$\mathcal{L}_{\text{Transition}}(\theta) = -\sum_{i=1}^m \log P_\theta\left(\hat{a}_{i,t} = a_{i,t}^*\right) - \log P_\theta\left(\hat{y}_{\text{trans}} = y_{\text{trans}}^*\right) - \mathbb{I}[t=0] \left( \log P_\theta\left(\pi_0^*\right) + \log P_\theta\left(r_0^*\right) \right)$$

4. **Bartleby Survival Floor & Abstention Loss ($\mathcal{L}_{\text{Bartleby}}$)**:
   Let $y_{\text{B}}^* = \mathbb{I}\left[\mathcal{X}_{\text{sat}}(\vec{a}^{\min}; \mathcal{X}_t) = \emptyset \;\vee\; c_t \le 0\right] \in \{0, 1\}$ indicate whether the problem is recognizably infeasible at the survival floor $\vec{a}^{\min}$, and let $\hat{p}_{\text{B}} = p_\theta^{\text{Bartleby}}(\text{refuse} \mid x_t, \mathcal{T}_t)$. The Bartleby objective supervises both the dedicated refusal readout and the simultaneous suppression of all candidate action nouls under floor infeasibility:
   $$\mathcal{L}_{\text{Bartleby}}(\theta) = -y_{\text{B}}^* \log \hat{p}_{\text{B}} - (1 - y_{\text{B}}^*) \log(1 - \hat{p}_{\text{B}}) + \left(\hat{p}_{\text{B}} - y_{\text{B}}^*\right)^2 + y_{\text{B}}^* \cdot \frac{1}{K} \sum_{k=1}^K \left[ -\log\left(1 - \hat{p}_k\right) + \hat{p}_k^2 \right]$$

---

## 7. Addressing the Central Peer-Review Objection ("The Triviality / Hardcoding Fallacy")

> **The Reviewer Objection ("The Triviality / Hardcoding Fallacy"):**
> *"If you supply the Preference Table $\mathcal{T}_{t-1}$ as input, isn't the decision already predetermined? Couldn't a simple symbolic `if/else` rule do the same thing?"*

A rigorous mathematical analysis shows why conditioning on $\mathcal{T}_{t-1}$ neither trivializes the learning problem nor reduces **Trade-Off Jev** to a symbolic lookup table. Below is the **five-point formal and empirical rebuttal** (cross-referenced with [`wiki/concepts/conditioned_nouls.md`](../wiki/concepts/conditioned_nouls.md)).

### 7.1 The Perception-to-Grid Bottleneck ($x_t \mapsto \hat{p}_{k,i}(\ell \mid x_t)$)
The Preference Table $\mathcal{T}_{t-1}$ is defined over the discrete ordinal grid $\mathcal{G} = \prod_{i=1}^m \{0, \dots, L_i - 1\}$, whereas the environment observation $x_t$ and candidate propositions $c_1, \dots, c_K \in \mathcal{X}_t$ reside in the space of **unstructured natural language** $\Sigma^*$ (clinical notes, incident logs, legal clauses, sensor telemetry narratives).
- A symbolic `if/else` rule $\Phi_{\text{AAT}}\left(\mathcal{T}_{t-1}, \{\hat{g}_i(c_k)\}\right)$ is **undefined** on raw text $x_t \in \Sigma^*$ because the ordinal realizations $g_i(x_t, c_k) \in \mathcal{L}_i$ are latent random variables.
- No rule-based parser can read *"Pit Bravo has corrosive sulfur-dioxide vapor causing actuator degradation"* and compute the calibrated ordinal posterior distributions $\hat{p}_{k,i}(\ell \mid x_t) = p_\theta(g_i(x_t, c_k) = \ell \mid x_t, c_k) \in \Delta^{L_i - 1}$ across $m$ goals and $K$ candidates in a single forward pass.
- Formally, single-pass inference requires solving the coupled neuro-symbolic composition:
  $$p_\theta\left(\text{yes} \;\middle|\; x_t, c_k, \mathcal{T}_t\right) = \left( \mathcal{H}_\theta^{\text{noul}} \circ \Phi_{\text{AAT}} \circ \mathcal{F}_\theta^{\text{score}} \right)\left(x_t, \{c_k\}_{k=1}^K, \mathcal{T}_{t-1}\right)$$
  where the neural backbone and ordinal readout heads $\mathcal{F}_\theta^{\text{score}}$ perform **epistemic semantic grounding under uncertainty** and the AAT layer $\Phi_{\text{AAT}}$ performs **non-compensatory procedural adaptation** in a single forward pass.

### 7.2 Every Decision Model Specifies Preferences — Standard Jev Conflates Them with Task Instructions
- In standard single-pass Decision Models (`choice`, `score`, `noul`), preferences are *already* passed by the user at inference time inside the prompt's `instructions` and `criteria` fields. Standard Jev does not "discover morality from thin air"; it relies on prompt-level `criteria`—which breaks down whenever multiple criteria conflict because unstructured prose cannot specify a strict non-compensatory urgency permutation $\pi_t \in \mathfrak{S}_m$, an asymmetric downward retreat variable $r_t \ne \pi_{m,t}$, or hard survival floors $\vec{a}^{\min}$ without marginalizing to $p_\theta(\text{yes} \mid x_t) \approx 0.50$.
- Explicitly factoring the state into **Epistemic World Context ($x_t$)** and **Normative Policy State ($\mathcal{T}_t$)** is a core architectural design principle: in regulated domains (autonomous vehicles, hospital triage, financial compliance, EU AI Act governance, or Asimov's Laws of Robotics), the normative constitution $\mathcal{T}_0$ *must* be explicit, inspectable, and **hot-swappable at inference time without retraining neural weights** $\theta$.

### 7.3 Endogenous Markov Adaptation ($\mathcal{T}_0 \to \mathcal{T}_1 \to \dots \to \mathcal{T}_t$) and Path Dependence
- Preferences in **Trade-Off Jev** are **not** static weights. Even when a caller supplies an initial constitution $\mathcal{T}_0$ at $t=0$ (or omits it so the model infers $\mathcal{T}_0 = f_\theta(x_0, G)$ zero-shot via Section 2.3), the active table $\mathcal{T}_t$ governing step $t \ge 1$ is **never supplied by the user**.
- Instead, $\mathcal{T}_t$ evolves endogenously via the Markov recurrence:
  $$\mathcal{T}_t = \Phi_{\text{AAT}}\left( \mathcal{T}_{t-1}, \; \mathcal{F}_\theta^{\text{score}}\left(x_t, \mathcal{X}_t\right) \right), \qquad t = 1, 2, \dots, T$$
  driven by the sequence of feasible surpluses (`upward_urgency` along $\pi_{t-1}$) and infeasible crises (`downward_retreat` along $r_{t-1}$) encountered in $x_0, \dots, x_{t-1}$.
- **Path Dependence Corollary**: Two identical agents initialized with the exact same $\mathcal{T}_0$ at $t=0$ that traverse different environmental histories $(x_0^{(A)}, \dots, x_{t-1}^{(A)}) \ne (x_0^{(B)}, \dots, x_{t-1}^{(B)})$ arrive at divergent Preference Tables $\mathcal{T}_t^{(A)} \ne \mathcal{T}_t^{(B)}$ at step $t$ and therefore make different decisions on an identical state $x_t$.

### 7.4 Non-Compensatory Grid Incomparability vs. Linear Scalarization ($U = \sum_{i=1}^m w_i g_i$)
Supplying linear utility weights $\mathbf{w} \in \mathbb{R}_+^m$ with $U(c_k) = \sum_{i=1}^m w_i \bar{g}_i(c_k)$ fails three formal properties satisfied by $\mathcal{T}_t$:
1. **Non-Compensatory Survival Floors ($\vec{a}^{\min}$)**: Under any finite linear weights $w_i < \infty$, a catastrophic violation $g_i(c_k) < a_i^{\min}$ (e.g., loss of human life) can be compensated by sufficiently large gains $\sum_{j \ne i} w_j g_j(c_k)$ in secondary goals. Under $\mathcal{T}_t$, feasibility requires $\bigwedge_{i=1}^m (\hat{g}_i(c_k) \ge a_{i,t} \ge a_i^{\min})$.
2. **Asymmetric Upward vs. Downward Adaptation ($r_t \ne \pi_{m,t}$)**: Linear weights impose a symmetric marginal rate of substitution $w_i / w_j$ in both surplus and deficit regimes, whereas $\mathcal{T}_t$ decouples the upward urgency leader $\pi_{1,t}$ from the downward sacrifice variable $r_t$ and modulates both locally via $\Psi_{\mathcal{R}_{\text{grid}}}(\vec{a}_t)$.
3. **Separating Epistemic Uncertainty from Principled Bartleby Refusal**: Linear utility paired with closed-set softmax ($\sum_{k=1}^K p_k = 1$) conflates epistemic ambiguity ($\hat{p}_k \approx 0.50, \hat{p}_{\text{B}} \approx 0.00$) with floor-infeasible lethal traps where every candidate must be rejected ($\forall k, \hat{p}_k \to 0.00$ and $\hat{p}_{\text{B}} = p_\theta^{\text{Bartleby}}(\text{refuse} \mid x_t, \mathcal{T}_t) \to 1.00$).

### 7.5 Empirical Verification (`data/corpus_v1/test.jsonl`, Markov Self-Rollout & OOD)
Empirical evaluation on the group-disjoint test split (`data/corpus_v1/test.jsonl`, $N = 600$, `Qwen/Qwen3-0.6B` backbone) confirms each theoretical claim:
- **Phase-2 Autoregressive Markov Self-Rollout ($t = 0 \to 1 \to 2$)**: When given $\mathcal{T}_0$ *only* at $t = 0$ across $67$ held-out 3-step trajectories ($201$ steps), `TradeOffJevNeuralSystem` predicts $\hat{\mathcal{T}}_1$ and $\hat{\mathcal{T}}_2$ endogenously from raw text $x_t$, achieving **100.0% self-rollout step choice accuracy** (`201/201`) and **100.0% full-chain agreement** (`67/67`) with **0.0% paralysis**.
- **Unseen Out-of-Distribution (OOD) Text Scenarios**: On $120$ held-out OOD records (`EnergyGridDispatchOOD` and `ClinicalProtocolOOD`) where symbolic rules cannot parse the domain prose, `TradeOffJevNeuralSystem` achieves **93.3%–95.0% choice accuracy** (`114/120` in Phase 2) and **100.0% Bartleby recall** (`22/22`), outperforming the unconditioned `StandardJevBaselineSystem` (**77.5% accuracy**, **45.0% paralysis**).
- **Counterfactual Preference-Table Flips**: On $94$ counterfactual pairs ($188$ test records) with identical text $x_t$ and distinct prior tables $\mathcal{T}_{t-1}^{(A)} \ne \mathcal{T}_{t-1}^{(B)}$, `TradeOffJevNeuralSystem` achieves **100.0% pair-flip accuracy** (`94/94`) vs. **0.0% pair-flip accuracy** (`0/94`, McNemar $\chi^2_{\text{cc}} = 92.01, p < 10^{-20}$) for `StandardJevBaselineSystem` and **100.0% paralysis** (`mean_max_noul = 0.528`) for unconditioned `typesafe/jev-1.14`.

