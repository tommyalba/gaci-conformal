"""E9: unseen-group fallback behavior (Lending Club, grade grouping).

Motivates the manuscript's flagged limitation. Three scenarios on the E4
pipeline, holding out grade G from ALL calibration data (truly unseen):

- cold_gaci : per-group ACI for every group; G's ACI starts with an empty
              score window and cold-starts from its own stream residuals
              (current run_gaci behavior for unseeded groups).
- marginal_fb: unseeded groups route to the shared marginal ACI instead.
- (reference) seeded GACI and marginal ACI numbers come from e4_metrics.

Metrics for G: overall coverage, coverage in first 200/1000 stream
observations (cold-start transient), mean width; plus marginal coverage.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from aci import ACI, SplitConformal  # noqa: E402
from e4_group_aci import (ALPHA, CAL_FRAC, CATEGORICAL, GAMMA, MAX_ROWS,  # noqa: E402
                          MIN_GROUP, NUMERIC, RAW, TRAIN_FRAC, WINDOW,
                          build_features, ridge_fit)

OUT = ROOT / "results"
HOLDOUT = "G"


def prep():
    usecols = ["issue_d", "int_rate", "grade"] + NUMERIC + list(CATEGORICAL)
    df = pd.read_csv(RAW, usecols=lambda c: c in usecols, low_memory=False)
    df = df.dropna(subset=["issue_d", "int_rate", "grade"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)
    if len(df) > MAX_ROWS:
        idx = np.linspace(0, len(df) - 1, MAX_ROWS).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    df["grade"] = df["grade"].astype(str).str.strip().str[0].str.upper()
    df = df[df["grade"].isin(list("ABCDEFG"))].reset_index(drop=True)
    y = pd.to_numeric(df["int_rate"], errors="coerce").to_numpy() / 100.0
    mask = ~np.isnan(y)
    df, y = df.loc[mask].reset_index(drop=True), y[mask]
    groups = df["grade"].to_numpy()
    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    X, _ = build_features(df, fit_rows=n_train)
    n_cal = int(n * CAL_FRAC)
    w = ridge_fit(X[:n_train], y[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    return (y[n_train:n_train + n_cal], y[n_train + n_cal:],
            pred_cal, pred_stream,
            groups[n_train:n_train + n_cal], groups[n_train + n_cal:])


def run_stream(y_cal, y_stream, pred_cal, pred_stream, g_cal, g_stream,
               fallback="cold"):
    """Per-group ACI; groups with <MIN_GROUP cal rows are 'unseen'.
    fallback='cold': unseeded groups get their own empty ACI (cold start).
    fallback='marginal': unseeded groups route to the marginal ACI."""
    uniq = sorted(set(g_cal) | set(g_stream))
    marginal = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW).seed(y_cal, pred_cal)
    acis, unseen = {}, set()
    for g in uniq:
        m = g_cal == g
        if m.sum() >= MIN_GROUP:
            acis[g] = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW).seed(
                y_cal[m], pred_cal[m])
        else:
            unseen.add(g)
            acis[g] = None
    recs = []
    for p, y, g in zip(pred_stream, y_stream, g_stream):
        if g in unseen and fallback == "marginal":
            target = marginal
        else:
            if acis.get(g) is None:
                acis[g] = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW)
            target = acis[g]
        r = target.step(float(p), float(y))
        recs.append({**r, "group": g})
    return recs, unseen


def summarize(recs, unseen):
    df = pd.DataFrame(recs)
    out = {"unseen_groups": sorted(unseen), "marginal_coverage":
           float(df.covered.mean()),
           "mean_width": float((df.hi - df.lo).mean())}
    for g in sorted(unseen):
        sub = df[df.group == g].reset_index(drop=True)
        out[f"group_{g}"] = {
            "n": int(len(sub)),
            "coverage": float(sub.covered.mean()),
            "mean_width": float((sub.hi - sub.lo).mean()),
            "coverage_first200": float(sub.covered.iloc[:200].mean()),
            "coverage_first1000": float(sub.covered.iloc[:1000].mean()),
        }
    return out


def main():
    y_cal, y_stream, pred_cal, pred_stream, g_cal, g_stream = prep()
    # hold out G entirely: it becomes unseen for BOTH marginal and per-group
    keep = g_cal != HOLDOUT
    res = {"experiment": "E9_unseen_groups", "holdout": HOLDOUT,
           "alpha": ALPHA, "gamma": GAMMA, "window": WINDOW,
           "min_group": MIN_GROUP}
    for fb in ("cold", "marginal"):
        recs, unseen = run_stream(y_cal[keep], y_stream, pred_cal[keep],
                                  pred_stream, g_cal[keep], g_stream,
                                  fallback=fb)
        res[fb] = summarize(recs, unseen)
        print(fb, json.dumps(res[fb][f"group_{HOLDOUT}"]), flush=True)
    (OUT / "e9_metrics.json").write_text(json.dumps(res, indent=2))
    print("WROTE", OUT / "e9_metrics.json")


if __name__ == "__main__":
    main()
