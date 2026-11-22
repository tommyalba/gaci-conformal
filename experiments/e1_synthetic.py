"""E1: synthetic distribution-shift experiment (paper's Figure-1 analog).

Data: y_t = x_t @ beta + shift_t + eps_t
  - x_t ~ N(0, I_5)
  - shift_t: piecewise-constant regime shifts (abrupt changes in conditional mean)
  - eps_t ~ N(0, sigma_t^2), sigma_t piecewise-constant (variance drift)
Point predictor: ridge regression fit on the first TRAIN_N points, then frozen
(the paper keeps the black-box predictor fixed; only alpha_t adapts).

Compares: SplitConformal (static) vs ACI vs DtACI on rolling coverage.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aci import ACI, SplitConformal

RNG = np.random.default_rng(42)
N_FEATURES = 5
TRAIN_N = 2000
CAL_N = 500
STREAM_N = 20000
ALPHA = 0.1          # 90% target coverage
GAMMA = 0.01
WINDOW = 500         # ACI score window
EVAL_WIN = 500       # rolling coverage window


def make_data(n, rng):
    X = rng.normal(size=(n, N_FEATURES))
    beta = np.array([1.5, -1.0, 0.5, 0.0, 2.0])
    # regime: shift every ~2500 points, alternating mean-shift and variance-shift
    shift = np.zeros(n)
    sigma = np.ones(n)
    regimes = [(0.0, 1.0), (3.0, 1.0), (0.0, 2.5), (-2.0, 1.0), (0.0, 4.0), (1.0, 1.0)]
    seg = n // len(regimes)
    for i, (s, v) in enumerate(regimes):
        lo, hi = i * seg, (i + 1) * seg if i < len(regimes) - 1 else n
        shift[lo:hi] = s
        sigma[lo:hi] = v
    y = X @ beta + shift + rng.normal(scale=sigma)
    return X, y, beta


def ridge_fit(X, y, lam=1.0):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def rolling_coverage(covered, win=EVAL_WIN):
    c = np.asarray(covered, dtype=float)
    if len(c) < win:
        return c
    kernel = np.ones(win) / win
    return np.convolve(c, kernel, mode="valid")


def main():
    X, y, beta = make_data(TRAIN_N + CAL_N + STREAM_N, RNG)
    Xtr, ytr = X[:TRAIN_N], y[:TRAIN_N]
    Xcal, ycal = X[TRAIN_N:TRAIN_N + CAL_N], y[TRAIN_N:TRAIN_N + CAL_N]
    Xs, ys = X[TRAIN_N + CAL_N:], y[TRAIN_N + CAL_N:]

    w = ridge_fit(Xtr, ytr)
    pred_cal = Xcal @ w
    pred_stream = Xs @ w

    # --- baselines ---
    scp = SplitConformal(alpha=ALPHA).fit_calibrate(ycal, pred_cal)
    scp_lo, scp_hi = scp.predict(pred_stream)
    scp_cov = (ys >= scp_lo) & (ys <= scp_hi)

    aci = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW).seed(ycal, pred_cal)
    aci_res = aci.run(pred_stream, ys)
    aci_cov = np.array([r["covered"] for r in aci_res])
    aci_w = np.array([r["hi"] - r["lo"] for r in aci_res])

    dtaci = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW, dtaci=True).seed(ycal, pred_cal)
    dt_res = dtaci.run(pred_stream, ys)
    dt_cov = np.array([r["covered"] for r in dt_res])
    dt_w = np.array([r["hi"] - r["lo"] for r in dt_res])

    # --- metrics ---
    def summ(cov, width):
        rc = rolling_coverage(cov)
        return {
            "mean_coverage": float(np.mean(cov)),
            "target": 1 - ALPHA,
            "coverage_gap": float(abs(np.mean(cov) - (1 - ALPHA))),
            "mean_width": float(np.mean(width)) if width is not None else None,
            "rolling_coverage_min": float(np.min(rc)) if len(rc) else None,
            "rolling_coverage_max": float(np.max(rc)) if len(rc) else None,
        }

    scp_w = (scp_hi - scp_lo)
    results = {
        "experiment": "E1_synthetic_regime_shift",
        "n_stream": STREAM_N,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "window": WINDOW,
        "scp": summ(scp_cov, scp_w),
        "aci": summ(aci_cov, aci_w),
        "dtaci": summ(dt_cov, dt_w),
    }

    # save time series for plotting
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    np.savez(out / "e1_series.npz",
             ys=ys, pred=pred_stream,
             scp_lo=scp_lo, scp_hi=scp_hi,
             aci_lo=np.array([r["lo"] for r in aci_res]),
             aci_hi=np.array([r["hi"] for r in aci_res]),
             dt_lo=np.array([r["lo"] for r in dt_res]),
             dt_hi=np.array([r["hi"] for r in dt_res]),
             scp_cov=scp_cov, aci_cov=aci_cov, dt_cov=dt_cov,
             aci_alpha=np.array(aci.alpha_history),
             dt_alpha=np.array(dtaci.alpha_history))
    (out / "e1_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
