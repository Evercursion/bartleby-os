"""Neural readout heads and differentiable Aspiration Adaptation modules for Trade-Off Jev.

This module implements the single-forward-pass neural architecture bridging
Selten's (1998, 1999) Aspiration Adaptation Theory (AAT) with 2026 SystemOne / Jev
decision models (`JointSchemaHead`, `PointerHead`, and `ConditionedNoulHead`).
Supports native PyTorch (`torch.nn.Module`) with an automatic NumPy tensor fallback
when running in lightweight CPU test environments without PyTorch installed.

References:
    - Selten, R. (1998). Aspiration Adaptation Theory. Journal of Mathematical Psychology.
    - Liu et al. (2026). Decision 1.0: Single-Pass Structured Readout Heads.
    - Wiki: `wiki/concepts/conditioned_nouls.md`, `wiki/concepts/aspiration_scheme_and_markov_chain.md`
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
    TensorLike = torch.Tensor
except ImportError:  # pragma: no cover - lightweight fallback when PyTorch is absent
    HAS_TORCH = False
    TensorLike = np.ndarray  # type: ignore[misc]

    class _BaseModule:
        """Lightweight PyTorch-compatible `nn.Module` base class backed by NumPy."""

        def __init__(self) -> None:
            self._buffers: dict[str, np.ndarray] = {}

        def register_buffer(
            self, name: str, tensor: np.ndarray, persistent: bool = False
        ) -> None:
            """Register a non-trainable buffer tensor."""
            _ = persistent
            setattr(self, name, tensor)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    class _DummyNN:
        Module = _BaseModule

    nn = _DummyNN()  # type: ignore[assignment]


def _to_array(x: Any) -> np.ndarray:
    """Convert any array-like or tensor input into a float32 NumPy ndarray."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise logistic sigmoid $\\sigma(z) = (1 + e^{-z})^{-1}$."""
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the specified axis."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / (np.sum(exps, axis=axis, keepdims=True) + 1e-8)


@dataclass(frozen=True)
class AATModelConfig:
    """Immutable configuration for the Trade-Off Jev neural readout architecture.

    Attributes:
        num_goals: Number of incomparable ordinal goal dimensions ($M$).
        num_levels: Maximum ordinal rubric levels per goal ($L_{\\max}$).
        hidden_dim: Backbone hidden representation dimension ($d_h$).
        pointer_dim: Bilinear pointer projection dimension ($d_p$).
        temperature: Softmax / sigmoid temperature ($\\tau > 0$) for soft threshold gates.
        survival_sharpness: Steepness parameter ($\\gamma > 0$) for the Bartleby survival floor.
        seed: Deterministic random seed for projection weight initialization.
    """

    num_goals: int = 4
    num_levels: int = 5
    hidden_dim: int = 64
    pointer_dim: int = 32
    temperature: float = 0.25
    survival_sharpness: float = 10.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate architectural hyperparameters."""
        if self.num_goals < 1:
            raise ValueError(f"num_goals must be >= 1, got {self.num_goals}")
        if self.num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {self.num_levels}")
        if self.hidden_dim < 4 or self.pointer_dim < 4:
            raise ValueError("hidden_dim and pointer_dim must be >= 4.")
        if self.temperature <= 0.0 or self.survival_sharpness <= 0.0:
            raise ValueError("temperature and survival_sharpness must be strictly positive.")


@dataclass(frozen=True)
class DifferentiableAATOutput:
    """Immutable container for differentiable AAT transition tensors at step $t$.

    Attributes:
        rubric_logits: Predicted ordinal rubric logits of shape `[B, K, M, L]`.
        expected_scores: Expected ordinal goal levels $\\hat{g}_m(x_k)$ of shape `[B, K, M]`.
        candidate_feasibility: Soft aspiration satisfaction $f_k \\in [0, 1]$ of shape `[B, K]`.
        survival_feasibility: Soft survival-floor satisfaction $s_k \\in [0, 1]$ of shape `[B, K]`.
        updated_aspiration: Differentiably adapted aspiration vector $\\hat{\\vec{a}}_t$ of shape `[B, M]`.
        bartleby_prob: Probability $P(\\text{Bartleby} \\mid x_t, \\mathcal{T}_t)$ of shape `[B]`.
        regime_probs: Soft probability distribution over `(upward, settled, retreat, bartleby)` `[B, 4]`.
    """

    rubric_logits: np.ndarray
    expected_scores: np.ndarray
    candidate_feasibility: np.ndarray
    survival_feasibility: np.ndarray
    updated_aspiration: np.ndarray
    bartleby_prob: np.ndarray
    regime_probs: np.ndarray


class PreferenceTableEmbedding(nn.Module):
    """Embeds Markov preference table state $\\mathcal{T}_{t-1} = (\\vec{a}_{t-1}, \\vec{a}^{\\min}, \\pi_{t-1}, r_{t-1})$.

    Maps discrete or continuous aspiration vectors, survival floors, urgency permutations,
    and retreat goal indices into a dense conditioning representation in $\\mathbb{R}^{d_h}$.
    """

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        rng = np.random.default_rng(config.seed)
        in_dim = 4 * config.num_goals
        self._weight = rng.normal(0.0, 1.0 / math.sqrt(in_dim), size=(in_dim, config.hidden_dim)).astype(
            np.float32
        )
        self._bias = np.zeros((config.hidden_dim,), dtype=np.float32)

    @property
    def config(self) -> AATModelConfig:
        """Return the model configuration."""
        return self._config

    def forward(
        self,
        aspiration_prev: TensorLike,
        survival_floor: TensorLike,
        urgency_ranks: TensorLike,
        retreat_index: TensorLike,
    ) -> np.ndarray:
        """Compute dense preference table embeddings $e(\\mathcal{T}_{t-1})$ of shape `[B, hidden_dim]`."""
        a_prev = _to_array(aspiration_prev)
        s_floor = _to_array(survival_floor)
        u_ranks = _to_array(urgency_ranks)
        r_idx = np.clip(_to_array(retreat_index).astype(np.int64), 0, self._config.num_goals - 1)

        scale = float(max(1, self._config.num_levels - 1))
        norm_a = a_prev / scale
        norm_floor = s_floor / scale
        norm_ranks = u_ranks / float(max(1, self._config.num_goals - 1))
        retreat_oh = np.eye(self._config.num_goals, dtype=np.float32)[r_idx]

        features = np.concatenate([norm_a, norm_floor, norm_ranks, retreat_oh], axis=-1)
        hidden = np.tanh(features @ self._weight + self._bias)
        return hidden


class DifferentiableAATModule(nn.Module):
    """Differentiable Selten (1998) Aspiration Adaptation transition and rubric evaluation head.

    Predicts per-candidate ordinal rubric distributions across $M$ goals, evaluates soft
    dominance against $\\vec{a}_{t-1}$ and the survival floor $\\vec{a}^{\\min}$, and computes
    a differentiable Markov update $\\mathcal{T}_{t-1} \\to \\hat{\\mathcal{T}}_t$ across four
    behavioral regimes:
        1. Upward Urgency Adaptation,
        2. Settled Feasible Selection,
        3. Bounded Downward Retreat along $r_{t-1}$,
        4. Bartleby Survival-Floor Refusal (`bartleby_noul`).

    Context-aware rubric perception (bartleby-1.1): rubric logits are produced from
    ``[candidate_hidden ‖ decide_hidden]`` only — never from ``table_embedding``.
    """

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        rng = np.random.default_rng(config.seed + 1)
        out_dim = config.num_goals * config.num_levels
        self._rubric_w = rng.normal(
            0.0, 1.0 / math.sqrt(config.hidden_dim * 2), size=(config.hidden_dim * 2, out_dim)
        ).astype(np.float32)
        self._level_grid = np.arange(config.num_levels, dtype=np.float32)

    def _compute_expected_scores(self, rubric_logits: np.ndarray) -> np.ndarray:
        """Compute expected ordinal score $\\hat{g}_m(x_k) = \\sum_{\\ell} \\ell \\cdot P(\\ell)$."""
        probs = _softmax(rubric_logits, axis=-1)
        return np.sum(probs * self._level_grid, axis=-1)

    def _compute_soft_dominance(
        self,
        expected_scores: np.ndarray,
        thresholds: np.ndarray,
        sharpness: float,
    ) -> np.ndarray:
        """Compute soft conjunctive dominance $\\prod_{m=1}^M \\sigma(\\gamma (\\hat{g}_{k,m} - \\theta_m))$."""
        margin = expected_scores - np.expand_dims(thresholds, axis=1)
        per_goal_sat = _sigmoid(sharpness * margin)
        return np.prod(per_goal_sat, axis=-1)

    def forward(
        self,
        candidate_hidden: TensorLike,
        decide_hidden: TensorLike,
        aspiration_prev: TensorLike,
        survival_floor: TensorLike,
        urgency_ranks: TensorLike,
        retreat_index: TensorLike,
        override_expected_scores: TensorLike | None = None,
        table_embedding: TensorLike | None = None,
    ) -> DifferentiableAATOutput:
        """Run single-pass rubric evaluation and differentiable aspiration adaptation."""
        _ = table_embedding
        cand_h = _to_array(candidate_hidden)
        decide_h = _to_array(decide_hidden)
        a_prev = _to_array(aspiration_prev)
        s_floor = _to_array(survival_floor)
        u_ranks = _to_array(urgency_ranks)
        r_idx = np.clip(_to_array(retreat_index).astype(np.int64), 0, self._config.num_goals - 1)

        batch_size, num_candidates, _ = cand_h.shape
        expanded_decide = np.broadcast_to(
            np.expand_dims(decide_h, axis=1), (batch_size, num_candidates, decide_h.shape[-1])
        )
        joint = np.concatenate([cand_h, expanded_decide], axis=-1)
        rubric_logits = (joint @ self._rubric_w).reshape(
            batch_size, num_candidates, self._config.num_goals, self._config.num_levels
        )

        if override_expected_scores is not None:
            expected_scores = _to_array(override_expected_scores)
        else:
            expected_scores = self._compute_expected_scores(rubric_logits)

        inv_temp = 1.0 / self._config.temperature
        candidate_feas = self._compute_soft_dominance(expected_scores, a_prev, sharpness=inv_temp)
        survival_feas = self._compute_soft_dominance(
            expected_scores, s_floor, sharpness=self._config.survival_sharpness
        )

        # Upward step along highest-urgency goal (minimum rank index)
        top_urgent = np.argmin(u_ranks, axis=-1)
        upward_delta = np.eye(self._config.num_goals, dtype=np.float32)[top_urgent]
        max_lvl = float(self._config.num_levels - 1)
        upward_aspiration = np.minimum(a_prev + upward_delta, max_lvl)
        upward_feas = self._compute_soft_dominance(
            expected_scores, upward_aspiration, sharpness=inv_temp
        )

        # Bounded downward retreat step along r_{t-1}, clamped at survival floor a^{min}
        retreat_delta = np.eye(self._config.num_goals, dtype=np.float32)[r_idx]
        retreat_aspiration = np.maximum(a_prev - retreat_delta, s_floor)

        any_upward = np.max(upward_feas, axis=-1)
        any_current = np.max(candidate_feas, axis=-1)
        any_survival = np.max(survival_feas, axis=-1)

        p_upward = any_upward * any_current
        p_settled = (1.0 - any_upward) * any_current
        p_retreat = (1.0 - any_current) * any_survival
        p_bartleby = (1.0 - any_current) * (1.0 - any_survival)

        regime_stack = np.stack([p_upward, p_settled, p_retreat, p_bartleby], axis=-1)
        regime_probs = regime_stack / (np.sum(regime_stack, axis=-1, keepdims=True) + 1e-8)

        updated_aspiration = (
            regime_probs[:, 0:1] * upward_aspiration
            + regime_probs[:, 1:2] * a_prev
            + regime_probs[:, 2:3] * retreat_aspiration
            + regime_probs[:, 3:4] * s_floor
        )

        return DifferentiableAATOutput(
            rubric_logits=rubric_logits,
            expected_scores=expected_scores,
            candidate_feasibility=candidate_feas,
            survival_feasibility=survival_feas,
            updated_aspiration=updated_aspiration,
            bartleby_prob=regime_probs[:, 3],
            regime_probs=regime_probs,
        )


class ConditionedNoulHead(nn.Module):
    """Preference-Table-Conditioned Noul and Pointer Readout Head.

    Combines standard Kev/Jev bilinear pointer compatibility
    $z_k^{\\text{ptr}} = \\frac{1}{\\sqrt{d_p}} k(h_k)^\\top q(h_{\\text{decide}})$
    with an AAT satisfaction residual conditioned on $\\hat{\\mathcal{T}}_t$ to produce
    `tradeoff_noul` $P(\\text{yes} \\mid x_t, \\mathcal{T}_t)$ and `bartleby_noul`.
    """

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        rng = np.random.default_rng(config.seed + 2)
        self._query_w = rng.normal(
            0.0, 1.0 / math.sqrt(config.hidden_dim * 2), size=(config.hidden_dim * 2, config.pointer_dim)
        ).astype(np.float32)
        self._key_w = rng.normal(
            0.0, 1.0 / math.sqrt(config.hidden_dim), size=(config.hidden_dim, config.pointer_dim)
        ).astype(np.float32)
        self._aat_gain = 4.0

    def forward(
        self,
        decide_hidden: TensorLike,
        candidate_hidden: TensorLike,
        table_embedding: TensorLike,
        aat_output: DifferentiableAATOutput,
    ) -> Mapping[str, np.ndarray]:
        """Compute conditioned `tradeoff_noul`, pointer probabilities, and `bartleby_noul`."""
        dec_h = _to_array(decide_hidden)
        cand_h = _to_array(candidate_hidden)
        tab_emb = _to_array(table_embedding)

        q_in = np.concatenate([dec_h, tab_emb], axis=-1)
        query = np.expand_dims(q_in @ self._query_w, axis=-1)  # shape: [B, d_p, 1]
        keys = cand_h @ self._key_w  # shape: [B, K, d_p]

        scale = 1.0 / math.sqrt(float(self._config.pointer_dim))
        pointer_logits = np.squeeze(np.matmul(keys, query), axis=-1) * scale  # shape: [B, K]

        inv_temp = 1.0 / self._config.temperature
        adapted_margin = aat_output.expected_scores - np.expand_dims(
            aat_output.updated_aspiration, axis=1
        )
        adapted_sat = np.prod(_sigmoid(inv_temp * adapted_margin), axis=-1)

        floor_gate = aat_output.survival_feasibility
        effective_sat = adapted_sat * floor_gate
        aat_residual = self._aat_gain * (2.0 * effective_sat - 1.0)

        conditioned_logits = pointer_logits + aat_residual
        tradeoff_noul = _sigmoid(conditioned_logits) * (
            1.0 - np.expand_dims(aat_output.bartleby_prob, axis=-1)
        )

        clipped_b = np.clip(aat_output.bartleby_prob, 1e-5, 1.0 - 1e-5)
        bartleby_logit = np.expand_dims(np.log(clipped_b / (1.0 - clipped_b)), axis=-1)
        extended_logits = np.concatenate([conditioned_logits, bartleby_logit], axis=-1)
        choice_probs = _softmax(extended_logits, axis=-1)

        return {
            "pointer_logits": pointer_logits,
            "conditioned_logits": conditioned_logits,
            "tradeoff_noul": tradeoff_noul,
            "choice_probs": choice_probs,
            "bartleby_noul": aat_output.bartleby_prob,
        }


class TradeOffJevModel(nn.Module):
    """End-to-end single-pass Trade-Off Jev model combining AAT and conditioned nouls."""

    def __init__(self, config: AATModelConfig) -> None:
        super().__init__()
        self._config = config
        self._table_embedding = PreferenceTableEmbedding(config)
        self._aat_module = DifferentiableAATModule(config)
        self._noul_head = ConditionedNoulHead(config)

    @property
    def config(self) -> AATModelConfig:
        """Return the model configuration."""
        return self._config

    def forward(
        self,
        decide_hidden: TensorLike,
        candidate_hidden: TensorLike,
        aspiration_prev: TensorLike,
        survival_floor: TensorLike,
        urgency_ranks: TensorLike,
        retreat_index: TensorLike,
        override_expected_scores: TensorLike | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Execute a single forward pass producing conditioned noul and Markov state outputs."""
        table_emb = self._table_embedding(
            aspiration_prev=aspiration_prev,
            survival_floor=survival_floor,
            urgency_ranks=urgency_ranks,
            retreat_index=retreat_index,
        )
        aat_out = self._aat_module(
            candidate_hidden=candidate_hidden,
            decide_hidden=decide_hidden,
            aspiration_prev=aspiration_prev,
            survival_floor=survival_floor,
            urgency_ranks=urgency_ranks,
            retreat_index=retreat_index,
            override_expected_scores=override_expected_scores,
        )
        readout = self._noul_head(
            decide_hidden=decide_hidden,
            candidate_hidden=candidate_hidden,
            table_embedding=table_emb,
            aat_output=aat_out,
        )
        return {
            **readout,
            "rubric_logits": aat_out.rubric_logits,
            "expected_scores": aat_out.expected_scores,
            "updated_aspiration": aat_out.updated_aspiration,
            "regime_probs": aat_out.regime_probs,
        }


class AATCurriculumLoss(nn.Module):
    """Composite curriculum loss for training Trade-Off Jev readout heads.

    Optimizes:
        $$\\mathcal{L} = \\mathcal{L}_{\\text{CE}} + \\lambda_1 \\mathcal{L}_{\\text{Brier}}
          + \\lambda_2 \\mathcal{L}_{\\text{Transition}} + \\lambda_3 \\mathcal{L}_{\\text{Bartleby}}$$
    """

    def __init__(
        self,
        brier_weight: float = 1.0,
        transition_weight: float = 0.5,
        bartleby_weight: float = 1.5,
    ) -> None:
        super().__init__()
        self._brier_weight = brier_weight
        self._transition_weight = transition_weight
        self._bartleby_weight = bartleby_weight

    def forward(
        self,
        outputs: Mapping[str, TensorLike],
        target_choice: TensorLike,
        target_noul: TensorLike,
        target_aspiration: TensorLike,
        target_bartleby: TensorLike,
    ) -> Mapping[str, float]:
        """Compute composite curriculum loss components and total scalar loss."""
        choice_probs = np.clip(_to_array(outputs["choice_probs"]), 1e-7, 1.0)
        targets = _to_array(target_choice).astype(np.int64)
        batch_indices = np.arange(choice_probs.shape[0])
        loss_ce = float(-np.mean(np.log(choice_probs[batch_indices, targets])))

        loss_brier = float(np.mean((_to_array(outputs["tradeoff_noul"]) - _to_array(target_noul)) ** 2))
        loss_transition = float(
            np.mean((_to_array(outputs["updated_aspiration"]) - _to_array(target_aspiration)) ** 2)
        )

        b_pred = np.clip(_to_array(outputs["bartleby_noul"]), 1e-6, 1.0 - 1e-6)
        b_true = _to_array(target_bartleby)
        loss_bartleby = float(
            -np.mean(b_true * np.log(b_pred) + (1.0 - b_true) * np.log(1.0 - b_pred))
        )

        total_loss = (
            loss_ce
            + self._brier_weight * loss_brier
            + self._transition_weight * loss_transition
            + self._bartleby_weight * loss_bartleby
        )
        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_brier": loss_brier,
            "loss_transition": loss_transition,
            "loss_bartleby": loss_bartleby,
        }
