"""E3: Lending Club tape — ACI under real temporal/economic drift.

Task: predict interest rate (int_rate, continuous, set at origination) from
application-time features, ordered by issue_d (origination date). The 2007-2014
tape spans the financial crisis and post-crisis tightening — genuine distribution
shift in both features and the feature->rate relationship.

Predictor: ridge regression fit on the earliest TRAIN_FRAC of loans, frozen.
Conformal: SCP calibrated on the next CAL_FRAC, then ACI / DtACI stream the rest.

Reads data read-only from ~/projects/credit_risk_modeling/data/raw/.
Writes results/e3_*.{npz,json}.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aci import ACI, SplitConformal

RAW = Path.home() / "projects/credit_risk_modeling/data/raw/loan_data_2007_2014.csv"
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

ALPHA = 0.1
GAMMA = 0.01
WINDOW = 5000          # ACI score window (larger: heavy-tailed residuals)
TRAIN_FRAC, CAL_FRAC = 0.55, 0.10
MAX_ROWS = 200_000     # subsample for CPU/memory safety

NUMERIC = ["loan_amnt", "annual_inc", "dti", "revol_util", "open_acc",
           "total_acc", "pub_rec", "delinq_2yrs", "inq_last_6mths",
           "mths_since_last_delinq", "revol_bal"]
CATEGORICAL = {
    "term": None,                    # all levels (2)
    "home_ownership": None,          # all levels (~4)
    "verification_status": None,     # all levels (~3)
    "purpose": 8,                    # top-8 + OTHER
    "emp_length": None,              # bins already strings
}


def build_features(df, fit_rows=None):
    parts = []
    Xn = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    Xn["revol_bal"] = np.log1p(Xn["revol_bal"])
    fit = Xn.iloc[:fit_rows] if fit_rows else Xn
    Xn = Xn.fillna(fit.median())
    # standardize on train-slice statistics only (no cal/stream leakage)
    Xn = (Xn - fit.mean()) / fit.std(ddof=0)
    parts.append(Xn.to_numpy())
    names = list(Xn.columns)

    for col, topk in CATEGORICAL.items():
        s = df[col].astype(str).fillna("missing")
        if topk:
            src = s.iloc[:fit_rows] if fit_rows else s
            keep = src.value_counts().head(topk).index
            s = s.where(s.isin(keep), "OTHER")
        dummies = pd.get_dummies(s, prefix=col, dtype=float)
        parts.append(dummies.to_numpy())
        names += list(dummies.columns)

    X = np.hstack(parts).astype(np.float32)
    return X, names


def ridge_fit(X, y, lam=10.0):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def rolling_coverage(cov, win=2000):
    c = np.asarray(cov, dtype=float)
    if len(c) < win:
        return c
    return np.convolve(c, np.ones(win) / win, mode="valid")


def main():
    usecols = ["issue_d", "int_rate"] + NUMERIC + list(CATEGORICAL)
    df = pd.read_csv(RAW, usecols=lambda c: c in usecols, low_memory=False)
    df = df.dropna(subset=["issue_d", "int_rate"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)
    if len(df) > MAX_ROWS:
        # keep chronological order, take every k-th row to preserve drift
        idx = np.linspace(0, len(df) - 1, MAX_ROWS).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    print(f"rows={len(df)}  span={df.issue_d.min().date()}..{df.issue_d.max().date()}", flush=True)

    y = pd.to_numeric(df["int_rate"], errors="coerce").to_numpy() / 100.0
    mask = ~np.isnan(y)
    df, y = df.loc[mask].reset_index(drop=True), y[mask]

    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    X, feat_names = build_features(df, fit_rows=n_train)
    n_cal = int(n * CAL_FRAC)
    print(f"train={n_train} cal={n_cal} stream={n - n_train - n_cal} feats={X.shape[1]}", flush=True)

    w = ridge_fit(X[:n_train], y[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    y_cal = y[n_train:n_train + n_cal]
    y_stream = y[n_train + n_cal:]
    dates = df["issue_d"].iloc[n_train + n_cal:].to_numpy()

    # baselines
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
        "experiment": "E3_lending_club_int_rate",
        "n": int(n), "span": [str(df.issue_d.min().date()), str(df.issue_d.max().date())],
        "alpha": ALPHA, "gamma": GAMMA, "window": WINDOW,
        "predictor": "ridge(lambda=10) on standardized numeric + one-hot categoricals",
        "scp": summ(cov_s, w_s), "aci": summ(cov_a, w_a), "dtaci": summ(cov_d, w_d),
    }
    np.savez(OUT / "e3_series.npz", y=y_stream, pred=pred_stream, dates=dates,
             scp_lo=lo_s, scp_hi=hi_s,
             aci_lo=np.array([r["lo"] for r in aci_res]),
             aci_hi=np.array([r["hi"] for r in aci_res]),
             dt_lo=np.array([r["lo"] for r in dt_res]),
             dt_hi=np.array([r["hi"] for r in dt_res]),
             scp_cov=cov_s, aci_cov=cov_a, dt_cov=cov_d,
             aci_alpha=np.array(aci.alpha_history), dt_alpha=np.array(dt.alpha_history))
    (OUT / "e3_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
