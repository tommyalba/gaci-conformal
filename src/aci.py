"""Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021, arXiv:2106.00170).

From-scratch implementation. ACI maintains a target miscoverage level alpha_t
updated online by gradient descent on the coverage error:

    alpha_{t+1} = alpha_t + gamma_t * (alpha - err_t)

where err_t = 1{Y_t not in C_t(alpha_t)}. The prediction interval uses the
(1 - alpha_t) quantile of a rolling window of absolute-residual scores.

Variants:
- ACI:    constant step size gamma.
- DtACI:  decaying step size gamma_t = gamma0 / sqrt(t + 1).
- SplitConformal: static baseline, fixed alpha and calibration set.
"""

import numpy as np


def conformity_scores(y, y_pred):
    return np.abs(np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float))


def quantile_halfwidth(scores, alpha_t):
    """Split-conformal quantile with finite-sample correction."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha_t)) / n)
    return float(np.quantile(scores, q_level))


class SplitConformal:
    """Static split-conformal baseline: fixed alpha, fixed calibration set."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.cal_scores = None

    def fit_calibrate(self, y_cal, y_pred_cal):
        self.cal_scores = conformity_scores(y_cal, y_pred_cal)
        return self

    def predict(self, y_pred):
        q = quantile_halfwidth(self.cal_scores, self.alpha)
        return np.asarray(y_pred, dtype=float) - q, np.asarray(y_pred, dtype=float) + q


class ACI:
    """Adaptive Conformal Inference, online.

    Per round t:
        interval_t = y_pred_t +/- Q_{1-alpha_t}(recent scores)
        err_t      = 1{y_true_t outside interval_t}
        alpha_{t+1} = clip(alpha_t + gamma_t * (alpha - err_t), 0, 1)

    Parameters
    ----------
    alpha  : target miscoverage (0.1 -> 90% coverage)
    gamma  : base step size (paper range ~0.005-0.05)
    window : rolling score window for the quantile estimate
    dtaci  : decaying step size if True
    """

    def __init__(self, alpha=0.1, gamma=0.01, window=200, dtaci=False, min_window=10):
        assert 0 < alpha < 1
        self.target_alpha = alpha
        self.gamma0 = gamma
        self.window = window
        self.dtaci = dtaci
        self.min_window = min_window
        self.alpha_t = alpha
        self.t = 0
        self.score_window = []
        self.coverage_history = []
        self.alpha_history = []
        self.width_history = []

    def seed(self, y_cal, y_pred_cal):
        """Initialize score window from a calibration batch."""
        self.score_window.extend(conformity_scores(y_cal, y_pred_cal).tolist())
        return self

    def _gamma_t(self):
        if not self.dtaci:
            return self.gamma0
        return self.gamma0 / np.sqrt(self.t + 1.0)

    def step(self, y_pred_t, y_true_t):
        """One online round. Returns dict with interval bounds and coverage."""
        scores = self.score_window[-self.window:]
        if len(scores) < self.min_window:
            halfwidth = float(np.std(scores)) * 3 if scores else 1.0
            halfwidth = max(halfwidth, 1e-6)
        else:
            halfwidth = quantile_halfwidth(np.asarray(scores), self.alpha_t)
        lo, hi = y_pred_t - halfwidth, y_pred_t + halfwidth
        covered = bool(lo <= y_true_t <= hi)
        err = 0.0 if covered else 1.0

        self.coverage_history.append(covered)
        self.alpha_history.append(self.alpha_t)
        self.width_history.append(hi - lo)
        self.alpha_t = float(np.clip(
            self.alpha_t + self._gamma_t() * (self.target_alpha - err), 0.0, 1.0))
        self.t += 1
        self.score_window.append(abs(y_true_t - y_pred_t))
        # keep only what the quantile slice can use; in-place del preserves
        # object identity for the pooled-window variant's shared list
        if len(self.score_window) > self.window:
            del self.score_window[:-self.window]
        return {"lo": lo, "hi": hi, "covered": covered, "alpha_t": self.alpha_history[-1],
                "halfwidth": halfwidth}

    # ---- batch convenience ----
    def run(self, y_pred_stream, y_true_stream):
        return [self.step(float(p), float(y)) for p, y in zip(y_pred_stream, y_true_stream)]
