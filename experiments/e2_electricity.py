"""E2: UCI Individual Household Electric Power Consumption — ACI under natural drift.

Task: forecast next-step Global_active_power (kW) from lagged values and
time-of-day/week features, on 10-minute resampled series (Dec 2006 - Nov 2010).
Natural drift: seasons, changing consumption patterns, economic shifts.

Predictor: ridge regression fit on the earliest TRAIN_FRAC, frozen.
Conformal: SCP on next CAL_FRAC, ACI / DtACI stream the remainder.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aci import ACI, SplitConformal

DATA = Path(__file__).resolve().parent.parent / "data/household_power_consumption.txt"
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

ALPHA = 0.1
GAMMA = 0.01
WINDOW = 2000
TRAIN_FRAC, CAL_FRAC = 0.60, 0.10
LAGS = [1, 2, 3, 6, 12, 24, 144]   # in 10-min steps (144 = 1 day)


def ridge_fit(X, y, lam=10.0):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def rolling_coverage(cov, win=1000):
    c = np.asarray(cov, dtype=float)
    if len(c) < win:
        return c
    return np.convolve(c, np.ones(win) / win, mode="valid")


def main():
    df = pd.read_csv(DATA, sep=";", na_values="?", low_memory=False)
    df["ts"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.set_index("ts").sort_index()
    y = df["Global_active_power"].resample("10min").mean().dropna()
    print(f"series: {len(y)} points, {y.index.min()} .. {y.index.max()}", flush=True)

    # features: lags + calendar harmonics
    feats = pd.DataFrame(index=y.index)
    for l in LAGS:
        feats[f"lag{l}"] = y.shift(l)
    hour = y.index.hour + y.index.minute / 60.0
    dow = y.index.dayofweek
    feats["sin_h"] = np.sin(2 * np.pi * hour / 24)
    feats["cos_h"] = np.cos(2 * np.pi * hour / 24)
    feats["sin_d"] = np.sin(2 * np.pi * dow / 7)
    feats["cos_d"] = np.cos(2 * np.pi * dow / 7)
    feats["roll_mean_1d"] = y.rolling(144).mean().shift(1)
    feats["roll_std_1d"] = y.rolling(144).std().shift(1)

    data = feats.join(y.rename("target")).dropna()
    yv = data["target"].to_numpy()
    X = data.drop(columns=["target"]).to_numpy(dtype=np.float32)
    # standardize on train-slice statistics only (no cal/stream leakage)
    _ntr = int(len(yv) * TRAIN_FRAC)
    mu, sd = X[:_ntr].mean(0), X[:_ntr].std(0) + 1e-9
    X = ((X - mu) / sd).astype(np.float32)

    n = len(yv)
    n_train = int(n * TRAIN_FRAC)
    n_cal = int(n * CAL_FRAC)
    print(f"train={n_train} cal={n_cal} stream={n - n_train - n_cal} feats={X.shape[1]}", flush=True)

    w = ridge_fit(X[:n_train], yv[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    y_cal = yv[n_train:n_train + n_cal]
    y_stream = yv[n_train + n_cal:]
    dates = data.index[n_train + n_cal:].to_numpy()

    scp = SplitConformal(alpha=ALPHA).fit_calibrate(y_cal, pred_cal)
    lo_s, hi_s = scp.predict(pred_stream)
    cov_s = (y_stream >= lo_s) & (y_stream <= hi_s)

    aci = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW).seed(y_cal, pred_cal)
    aci_res = aci.run(pred_stream, y_stream)
    cov_a = np.array([r["covered"] for r in aci_res])
    w_a = np.array([r["hi"] - r["lo"] for r in aci_res])

    dt = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW, dtaci=True).seed(y_cal, pred_cal)
    dt_res = dt.run(pred_stream, y_stream)
    cov_d = np.array([r["covered"] for r in dt_res])
    w_d = np.array([r["hi"] - r["lo"] for r in dt_res])
    w_s = hi_s - lo_s

    def summ(cov, width):
        rc = rolling_coverage(cov)
        return {"mean_coverage": float(np.mean(cov)), "target": 1 - ALPHA,
                "coverage_gap": float(abs(np.mean(cov) - (1 - ALPHA))),
                "mean_width": float(np.mean(width)),
                "rolling_min": float(np.min(rc)), "rolling_max": float(np.max(rc))}

    results = {
        "experiment": "E2_electricity_10min",
        "n": int(n), "span": [str(data.index[0]), str(data.index[-1])],
        "alpha": ALPHA, "gamma": GAMMA, "window": WINDOW,
        "predictor": "ridge(lambda=10) on lags+calendar+rolling features",
        "scp": summ(cov_s, w_s), "aci": summ(cov_a, w_a), "dtaci": summ(cov_d, w_d),
    }
    np.savez(OUT / "e2_series.npz", y=y_stream, pred=pred_stream, dates=dates,
             scp_lo=lo_s, scp_hi=hi_s,
             aci_lo=np.array([r["lo"] for r in aci_res]),
             aci_hi=np.array([r["hi"] for r in aci_res]),
             dt_lo=np.array([r["lo"] for r in dt_res]),
             dt_hi=np.array([r["hi"] for r in dt_res]),
             scp_cov=cov_s, aci_cov=cov_a, dt_cov=cov_d,
             aci_alpha=np.array(aci.alpha_history), dt_alpha=np.array(dt.alpha_history))
    (OUT / "e2_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
