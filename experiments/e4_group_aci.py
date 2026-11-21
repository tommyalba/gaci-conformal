"""E4 / Phase 2: Group-conditional ACI under temporal drift (Lending Club).

Delta vs literature: per-group alpha_{g,t} adapted online so EACH segment
(loan grade band) holds its own coverage target under distribution shift —
not just marginal coverage. Groups: Lending Club grade (A-G), assigned at
origination (available at prediction time, no leakage).

Methods compared:
- SCP:        single static quantile from calibration (marginal only)
- ACI:        single adaptive alpha_t (marginal coverage)
- GACI:       one ACI per grade group, routed by group label
- GACI-pooled: shared score window, per-group alpha_t (hybrid)

Metrics: per-group coverage, worst-group coverage, mean width, marginal
coverage — the conditional-vs-marginal trade-off frontier.
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
WINDOW = 5000
TRAIN_FRAC, CAL_FRAC = 0.55, 0.10
MAX_ROWS = 200_000
MIN_GROUP = 200  # min calibration rows for a group's own window

NUMERIC = ["loan_amnt", "annual_inc", "dti", "revol_util", "open_acc",
           "total_acc", "pub_rec", "delinq_2yrs", "inq_last_6mths",
           "mths_since_last_delinq", "revol_bal"]
CATEGORICAL = {"term": None, "home_ownership": None, "verification_status": None,
               "purpose": 8, "emp_length": None}


def build_features(df):
    parts, names = [], []
    Xn = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    Xn["revol_bal"] = np.log1p(Xn["revol_bal"])
    Xn = Xn.fillna(Xn.median())
    Xn = (Xn - Xn.mean()) / Xn.std(ddof=0)
    parts.append(Xn.to_numpy()); names += list(Xn.columns)
    for col, topk in CATEGORICAL.items():
        s = df[col].astype(str).fillna("missing")
        if topk:
            keep = s.value_counts().head(topk).index
            s = s.where(s.isin(keep), "OTHER")
        d = pd.get_dummies(s, prefix=col, dtype=float)
        parts.append(d.to_numpy()); names += list(d.columns)
    return np.hstack(parts).astype(np.float32), names


def ridge_fit(X, y, lam=10.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def run_gaci(pred_cal, y_cal, groups_cal, pred_stream, y_stream, groups_stream,
             pooled=False):
    """Per-group ACI. pooled=True shares ONE score-window list object across
    groups (true pooled variant); each group still keeps its own alpha_t."""
    uniq = sorted(set(groups_cal) | set(groups_stream))
    acis = {}
    shared = list(np.abs(y_cal - pred_cal)) if pooled else None
    for g in uniq:
        aci = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW)
        if pooled:
            aci.score_window = shared          # same object -> pooled scores
        else:
            m = groups_cal == g
            if m.sum() >= MIN_GROUP:
                aci.seed(y_cal[m], pred_cal[m])
        acis[g] = aci
    results = []
    for p, y, g in zip(pred_stream, y_stream, groups_stream):
        r = acis[g].step(float(p), float(y))
        results.append({**r, "group": g})
    return results


def group_stats(res, groups):
    df = pd.DataFrame(res)
    out = {}
    for g in sorted(set(groups)):
        sub = df[df.group == g]
        out[str(g)] = {"n": int(len(sub)),
                       "coverage": float(sub.covered.mean()),
                       "mean_width": float((sub.hi - sub.lo).mean())}
    return out


def main():
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
    groups_all = df["grade"].to_numpy()

    X, _ = build_features(df)
    n = len(y)
    n_train = int(n * TRAIN_FRAC); n_cal = int(n * CAL_FRAC)
    print(f"rows={n} train={n_train} cal={n_cal} stream={n-n_train-n_cal} "
          f"groups={sorted(set(groups_all))}", flush=True)

    w = ridge_fit(X[:n_train], y[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    y_cal, y_stream = y[n_train:n_train + n_cal], y[n_train + n_cal:]
    g_cal, g_stream = groups_all[n_train:n_train + n_cal], groups_all[n_train + n_cal:]

    # SCP + marginal ACI
    scp = SplitConformal(alpha=ALPHA).fit_calibrate(y_cal, pred_cal)
    lo_s, hi_s = scp.predict(pred_stream)
    cov_s = (y_stream >= lo_s) & (y_stream <= hi_s)

    aci = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW).seed(y_cal, pred_cal)
    a_res = aci.run(pred_stream, y_stream)
    cov_a = np.array([r["covered"] for r in a_res])
    w_a = np.array([r["halfwidth"] for r in a_res])

    # group-conditional ACI (own windows)
    g_res = run_gaci(pred_cal, y_cal, g_cal, pred_stream, y_stream, g_stream, pooled=False)
    cov_g = np.array([r["covered"] for r in g_res])
    w_g = np.array([r["hi"] - r["lo"] for r in g_res])

    # pooled-window variant
    gp_res = run_gaci(pred_cal, y_cal, g_cal, pred_stream, y_stream, g_stream, pooled=True)
    cov_gp = np.array([r["covered"] for r in gp_res])
    w_gp = np.array([r["hi"] - r["lo"] for r in gp_res])

    def summ(cov, width):
        return {"marginal_coverage": float(np.mean(cov)),
                "mean_width": float(np.mean(width))}

    results = {
        "experiment": "E4_group_conditional_aci",
        "n": int(n), "alpha": ALPHA, "gamma": GAMMA, "window": WINDOW,
        "groups": "loan grade A-G",
        "scp": summ(cov_s, hi_s - lo_s),
        "aci": summ(cov_a, w_a),
        "gaci": {**summ(cov_g, w_g),
                 "per_group": group_stats(g_res, g_stream),
                 "worst_group_coverage": min(v["coverage"] for v in group_stats(g_res, g_stream).values())},
        "gaci_pooled": {**summ(cov_gp, w_gp),
                        "per_group": group_stats(gp_res, g_stream),
                        "worst_group_coverage": min(v["coverage"] for v in group_stats(gp_res, g_stream).values())},
        # marginal ACI per-group coverage for comparison
        "aci_per_group": {},
    }
    # per-group coverage under marginal ACI
    dfc = pd.DataFrame({"cov": cov_a, "w": w_a, "g": g_stream})
    for g in sorted(set(g_stream)):
        sub = dfc[dfc.g == g]
        results["aci_per_group"][str(g)] = {"n": int(len(sub)),
                                            "coverage": float(sub["cov"].mean()),
                                            "mean_width": float(sub["w"].mean())}

    np.savez(OUT / "e4_series.npz", y=y_stream, pred=pred_stream, groups=g_stream,
             scp_lo=lo_s, scp_hi=hi_s, aci_lo=np.array([r["lo"] for r in a_res]),
             aci_hi=np.array([r["hi"] for r in a_res]),
             gaci_lo=np.array([r["lo"] for r in g_res]),
             gaci_hi=np.array([r["hi"] for r in g_res]),
             scp_cov=cov_s, aci_cov=cov_a, gaci_cov=cov_g)
    (OUT / "e4_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2)[:4000])


if __name__ == "__main__":
    main()
