"""Post-Hoc Temperature Scaling and Calibration Diagnostics for Trade-Off Jev.

Implements Guo et al. (2017) post-hoc temperature scaling ($z \\mapsto z / T$, $T > 0$)
for single-forward-pass SystemOne / Jev decision heads (`choice_logits`, `conditioned_logits`,
and `bartleby_logit`). Combines a global log-space grid search with PyTorch L-BFGS
refinement on Negative Log-Likelihood (NLL) over the held-out calibration (`cal`) split,
and reports:
    - Negative Log-Likelihood (NLL) before and after calibration,
    - Proper Scoring Brier Score before and after calibration,
    - Expected Calibration Error (ECE) over $B = 15$ equal-width probability bins before
      and after calibration, alongside per-bin reliability diagrams.

References:
    - Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
    - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern
      Neural Networks. ICML 2017.
    - Liu et al. (2026). Decision 1.0: Single-Pass Structured Readout Heads (arXiv:2611.18629).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class CalibrationMode(str, Enum):
    """Supported post-hoc temperature scaling modes.

    Attributes:
        BINARY_NOUL: Scales scalar Bernoulli logits $z \\mapsto \\sigma(z / T)$ against
            binary labels $y \\in \\{0, 1\\}$ (or soft oracle targets $y \\in [0, 1]$).
        MULTICLASS_CHOICE: Scales categorical choice logits $\\mathbf{z} \\mapsto \\text{softmax}(\\mathbf{z} / T)$
            against integer target indices $y \\in \\{0, \\dots, C - 1\\}$.
    """

    BINARY_NOUL = "binary_noul"
    MULTICLASS_CHOICE = "multiclass_choice"


@dataclass(frozen=True)
class ReliabilityBin:
    """Empirical reliability statistics for a single probability interval $(l_b, u_b]$.

    Attributes:
        bin_index: Zero-based bin index $b \\in \\{0, \\dots, B - 1\\}$.
        lower_bound: Left interval endpoint $l_b \\in [0, 1)$.
        upper_bound: Right interval endpoint $u_b \\in (0, 1]$.
        count: Number of predictions falling into bin $b$.
        mean_confidence: Mean predicted probability / confidence in bin $b$.
        empirical_accuracy: Mean realized accuracy / positive rate in bin $b$.
        absolute_gap: $|\\text{empirical\\_accuracy} - \\text{mean\\_confidence}|$.
    """

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    absolute_gap: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bin statistics to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CalibrationReport:
    """Immutable before/after summary produced by `TemperatureCalibrator`.

    Attributes:
        mode: Calibration mode (`binary_noul` or `multiclass_choice`).
        num_samples: Number of valid calibration predictions evaluated.
        num_bins: Number of equal-width ECE bins (default `15`).
        optimal_temperature: Fitted temperature scalar $T^* > 0$.
        pre_nll: Uncalibrated ($T = 1.0$) Negative Log-Likelihood.
        post_nll: Calibrated ($T = T^*$) Negative Log-Likelihood.
        pre_brier: Uncalibrated ($T = 1.0$) Brier Score.
        post_brier: Calibrated ($T = T^*$) Brier Score.
        pre_ece: Uncalibrated ($T = 1.0$) Expected Calibration Error across `num_bins`.
        post_ece: Calibrated ($T = T^*$) Expected Calibration Error across `num_bins`.
        pre_bins: Reliability diagram bins prior to temperature scaling.
        post_bins: Reliability diagram bins after temperature scaling.
    """

    mode: CalibrationMode
    num_samples: int
    num_bins: int
    optimal_temperature: float
    pre_nll: float
    post_nll: float
    pre_brier: float
    post_brier: float
    pre_ece: float
    post_ece: float
    pre_bins: Tuple[ReliabilityBin, ...]
    post_bins: Tuple[ReliabilityBin, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this calibration report to a JSON-compatible dictionary."""
        return {
            "mode": self.mode.value if isinstance(self.mode, CalibrationMode) else str(self.mode),
            "num_samples": self.num_samples,
            "num_bins": self.num_bins,
            "optimal_temperature": round(self.optimal_temperature, 6),
            "pre_nll": round(self.pre_nll, 6),
            "post_nll": round(self.post_nll, 6),
            "pre_brier": round(self.pre_brier, 6),
            "post_brier": round(self.post_brier, 6),
            "pre_ece": round(self.pre_ece, 6),
            "post_ece": round(self.post_ece, 6),
            "pre_bins": [b.to_dict() for b in self.pre_bins],
            "post_bins": [b.to_dict() for b in self.post_bins],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize this report as a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class TemperatureCalibrator:
    """Post-hoc L-BFGS + Log-Grid NLL Temperature Scaler (`cal` split).

    Fits a single strictly positive scalar temperature $T = \\exp(\\theta) \\in [T_{\\min}, T_{\\max}]$
    minimizing Negative Log-Likelihood on held-out calibration logits without modifying
    relative logit ordering (hence preserving argmax accuracy while calibrating confidence).

     Optimization Strategy:
        1. Evaluates NLL across a dense log-spaced grid (including $T = 1.0$ explicitly)
           to locate a globally reliable basin initialization $\\theta_0 = \\log T_{\\text{grid}}$.
        2. Refines $\\theta$ via PyTorch `torch.optim.LBFGS` with strong Wolfe line search,
           clamping $T \\in [T_{\\min}, T_{\\max}]$ and retaining any iterate strictly improving NLL.
    """

    def __init__(
        self,
        mode: CalibrationMode = CalibrationMode.BINARY_NOUL,
        num_bins: int = 15,
        min_temperature: float = 0.05,
        max_temperature: float = 10.0,
        grid_points: int = 81,
        lbfgs_max_iter: int = 50,
    ) -> None:
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}.")
        if min_temperature <= 0.0 or max_temperature <= min_temperature:
            raise ValueError(
                f"Invalid temperature bounds [{min_temperature}, {max_temperature}]; require 0 < min < max."
            )
        if grid_points < 5:
            raise ValueError("grid_points must be >= 5.")

        self._mode = mode
        self._num_bins = num_bins
        self._min_temp = float(min_temperature)
        self._max_temp = float(max_temperature)
        self._grid_points = int(grid_points)
        self._lbfgs_max_iter = int(lbfgs_max_iter)
        self._temperature: float = 1.0
        self._last_report: Optional[CalibrationReport] = None

    @property
    def mode(self) -> CalibrationMode:
        """Return the calibration mode (`binary_noul` or `multiclass_choice`)."""
        return self._mode

    @property
    def num_bins(self) -> int:
        """Return the number of ECE bins (default `15`)."""
        return self._num_bins

    @property
    def temperature(self) -> float:
        """Return the currently fitted temperature $T^* > 0$ (defaults to `1.0` before `fit`)."""
        return self._temperature

    @property
    def last_report(self) -> Optional[CalibrationReport]:
        """Return the `CalibrationReport` from the most recent `fit` call, if any."""
        return self._last_report

    def _compute_nll_tensor(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Compute differentiable NLL loss for `logits / temperature`."""
        scaled = logits / temperature
        if self._mode == CalibrationMode.BINARY_NOUL:
            return F.binary_cross_entropy_with_logits(scaled, targets, reduction="mean")
        return F.cross_entropy(scaled, targets, reduction="mean")

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: float,
    ) -> Tuple[float, float, float, Tuple[ReliabilityBin, ...]]:
        """Compute `(nll, brier, ece, bins)` at a fixed scalar `temperature`."""
        with torch.no_grad():
            temp_t = torch.tensor(temperature, dtype=torch.float32, device=logits.device)
            nll = float(self._compute_nll_tensor(logits, targets, temp_t).item())
            scaled = logits / temp_t

            if self._mode == CalibrationMode.BINARY_NOUL:
                probs = torch.sigmoid(scaled)
                hard_targets = (targets >= 0.5).to(torch.float32)
                brier = float(torch.mean((probs - hard_targets) ** 2).item())
                confidences = probs
                outcomes = hard_targets
            else:
                probs = F.softmax(scaled, dim=-1)
                num_classes = probs.shape[-1]
                one_hot = F.one_hot(targets.to(torch.long), num_classes=num_classes).to(torch.float32)
                brier = float(torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1)).item())
                max_probs, preds = torch.max(probs, dim=-1)
                confidences = max_probs
                outcomes = (preds == targets.to(torch.long)).to(torch.float32)

            ece, bins = self._compute_ece_and_bins(confidences.cpu(), outcomes.cpu())
        return nll, brier, ece, bins

    def _compute_ece_and_bins(
        self,
        confidences: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> Tuple[float, Tuple[ReliabilityBin, ...]]:
        """Compute Expected Calibration Error across `self._num_bins` equal-width intervals."""
        total_n = int(confidences.numel())
        if total_n == 0:
            raise ValueError("Cannot compute ECE on an empty prediction tensor.")

        boundaries = torch.linspace(0.0, 1.0, self._num_bins + 1, dtype=torch.float32)
        ece_acc = 0.0
        bins_out: List[ReliabilityBin] = []

        for b_idx in range(self._num_bins):
            lo = float(boundaries[b_idx].item())
            hi = float(boundaries[b_idx + 1].item())
            if b_idx == 0:
                in_bin = (confidences >= lo) & (confidences <= hi)
            else:
                in_bin = (confidences > lo) & (confidences <= hi)

            count = int(in_bin.sum().item())
            if count > 0:
                mean_conf = float(confidences[in_bin].mean().item())
                emp_acc = float(outcomes[in_bin].mean().item())
                gap = abs(emp_acc - mean_conf)
                ece_acc += (count / float(total_n)) * gap
            else:
                mean_conf = 0.0
                emp_acc = 0.0
                gap = 0.0

            bins_out.append(
                ReliabilityBin(
                    bin_index=b_idx,
                    lower_bound=round(lo, 6),
                    upper_bound=round(hi, 6),
                    count=count,
                    mean_confidence=round(mean_conf, 6),
                    empirical_accuracy=round(emp_acc, 6),
                    absolute_gap=round(gap, 6),
                )
            )

        return float(ece_acc), tuple(bins_out)

    def _prepare_inputs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate, flatten (for masked binary nouls), and cast calibration inputs to FP32."""
        if self._mode == CalibrationMode.BINARY_NOUL:
            flat_logits = logits.detach().to(dtype=torch.float32, device="cpu")
            flat_targets = targets.detach().to(dtype=torch.float32, device="cpu")
            if mask is not None:
                valid = mask.detach().to(dtype=torch.bool, device="cpu")
                flat_logits = flat_logits[valid]
                flat_targets = flat_targets[valid]
            else:
                flat_logits = flat_logits.reshape(-1)
                flat_targets = flat_targets.reshape(-1)
            # Filter out any sentinel -1e9 masked logits if mask was omitted
            non_sentinel = flat_logits > -1e6
            flat_logits = flat_logits[non_sentinel]
            flat_targets = flat_targets[non_sentinel].clamp(0.0, 1.0)
            if flat_logits.numel() == 0:
                raise ValueError("No valid binary noul logits remain after masking.")
            return flat_logits, flat_targets

        # Multiclass choice logits: shape [N, C], targets: shape [N]
        choice_logits = logits.detach().to(dtype=torch.float32, device="cpu")
        choice_targets = targets.detach().to(dtype=torch.long, device="cpu").reshape(-1)
        if choice_logits.ndim != 2:
            raise ValueError(
                f"MULTICLASS_CHOICE expects 2D logits [N, C], got shape {tuple(choice_logits.shape)}."
            )
        if choice_logits.shape[0] != choice_targets.shape[0] or choice_logits.shape[0] == 0:
            raise ValueError("Logits and targets batch dimensions must match and be non-empty.")
        return choice_logits, choice_targets

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> CalibrationReport:
        """Fit optimal temperature $T^*$ via Log-Grid + L-BFGS on NLL and return `CalibrationReport`.

        Args:
            logits: Raw uncalibrated logits (`[N]` or `[B, K]` for `BINARY_NOUL`;
                `[N, C]` for `MULTICLASS_CHOICE`).
            targets: Ground-truth labels (`[N]` or `[B, K]` float in `[0, 1]` for
                `BINARY_NOUL`; `[N]` integer class indices for `MULTICLASS_CHOICE`).
            mask: Optional boolean validity mask of shape matching `logits` (for `BINARY_NOUL`).

        Returns:
            `CalibrationReport` detailing NLL, Brier Score, and 15-bin ECE before and after
            post-hoc temperature scaling.
        """
        clean_logits, clean_targets = self._prepare_inputs(logits, targets, mask=mask)
        num_samples = int(clean_targets.shape[0])

        pre_nll, pre_brier, pre_ece, pre_bins = self._compute_metrics(
            clean_logits, clean_targets, temperature=1.0
        )

        # Phase A: Global log-space grid search (explicitly including T = 1.0)
        log_min = math.log(self._min_temp)
        log_max = math.log(self._max_temp)
        grid_temps = torch.exp(torch.linspace(log_min, log_max, self._grid_points, dtype=torch.float32))
        candidates = torch.cat([torch.tensor([1.0], dtype=torch.float32), grid_temps])

        best_temp = 1.0
        best_nll = pre_nll
        with torch.no_grad():
            for cand_t in candidates:
                t_val = float(cand_t.clamp(self._min_temp, self._max_temp).item())
                cand_nll = float(
                    self._compute_nll_tensor(
                        clean_logits,
                        clean_targets,
                        torch.tensor(t_val, dtype=torch.float32),
                    ).item()
                )
                if math.isfinite(cand_nll) and cand_nll < best_nll:
                    best_nll = cand_nll
                    best_temp = t_val

        # Phase B: L-BFGS continuous refinement initialized at log(best_temp)
        log_temp_param = torch.nn.Parameter(
            torch.tensor([math.log(best_temp)], dtype=torch.float32)
        )
        optimizer = torch.optim.LBFGS(
            [log_temp_param],
            lr=0.25,
            max_iter=self._lbfgs_max_iter,
            line_search_fn="strong_wolfe",
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            clamped_log_t = log_temp_param.clamp(log_min, log_max)
            loss = self._compute_nll_tensor(clean_logits, clean_targets, torch.exp(clamped_log_t))
            loss.backward()
            return loss

        try:
            optimizer.step(_closure)
            refined_temp = float(
                torch.exp(log_temp_param.detach().clamp(log_min, log_max)).item()
            )
            with torch.no_grad():
                refined_nll = float(
                    self._compute_nll_tensor(
                        clean_logits,
                        clean_targets,
                        torch.tensor(refined_temp, dtype=torch.float32),
                    ).item()
                )
            if math.isfinite(refined_nll) and refined_nll <= best_nll:
                best_temp = refined_temp
        except RuntimeError:
            # Retain guaranteed global-grid optimum if line search encounters flat curvature
            pass

        self._temperature = best_temp
        post_nll, post_brier, post_ece, post_bins = self._compute_metrics(
            clean_logits, clean_targets, temperature=self._temperature
        )
        # Guard against soft-target label-smoothing inflation: never adopt a temperature
        # that simultaneously degrades both proper scoring Brier and 15-bin ECE.
        if post_brier > pre_brier and post_ece > pre_ece:
            best_cal_temp = 1.0
            best_cal_score = pre_brier + pre_ece
            with torch.no_grad():
                for cand_t in candidates:
                    t_val = float(cand_t.clamp(self._min_temp, self._max_temp).item())
                    _, c_brier, c_ece, _ = self._compute_metrics(
                        clean_logits, clean_targets, temperature=t_val
                    )
                    if c_brier <= pre_brier and c_ece <= pre_ece and (c_brier + c_ece) < best_cal_score:
                        best_cal_score = c_brier + c_ece
                        best_cal_temp = t_val
            self._temperature = best_cal_temp
            post_nll, post_brier, post_ece, post_bins = self._compute_metrics(
                clean_logits, clean_targets, temperature=self._temperature
            )

        report = CalibrationReport(
            mode=self._mode,
            num_samples=num_samples,
            num_bins=self._num_bins,
            optimal_temperature=self._temperature,
            pre_nll=pre_nll,
            post_nll=post_nll,
            pre_brier=pre_brier,
            post_brier=post_brier,
            pre_ece=pre_ece,
            post_ece=post_ece,
            pre_bins=pre_bins,
            post_bins=post_bins,
        )
        self._last_report = report
        return report

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide raw logits by the calibrated scalar temperature $T^*$."""
        return logits / self._temperature

    def calibrated_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute calibrated Bernoulli (`sigmoid`) or categorical (`softmax`) probabilities."""
        scaled = self.scale_logits(logits)
        if self._mode == CalibrationMode.BINARY_NOUL:
            return torch.sigmoid(scaled)
        return F.softmax(scaled, dim=-1)

    def save(self, path: str | Path) -> Path:
        """Save fitted temperature and calibration report to a JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "mode": self._mode.value,
            "num_bins": self._num_bins,
            "temperature": self._temperature,
            "report": self._last_report.to_dict() if self._last_report is not None else None,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibrator":
        """Load a fitted `TemperatureCalibrator` from a saved JSON artifact."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        calibrator = cls(
            mode=CalibrationMode(data["mode"]),
            num_bins=int(data.get("num_bins", 15)),
        )
        calibrator._temperature = float(data["temperature"])
        return calibrator
