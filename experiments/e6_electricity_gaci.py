"""E6 / Phase 2 second domain: GACI on UCI Household Electric Power.

Lending Club (E4/E5) is one domain; electricity gives a second, structurally
different one — dense 10-min time series with calendar-driven heterogeneity
instead of risk-tiered groups. Natural groups available at prediction time:
  - hour_band: night(0-6) / morning(6-12) / afternoon(12-18) / evening(18-24)
  - day_type: weekday vs weekend
  - month: 1-12 (seasonality)

Question: does marginal ACI hide subgroup undercoverage here too, and does
GACI fix it at what width cost?
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import e2_electricity as e2
from aci import ACI, SplitConformal

OUT = HERE.parent / "results"


def group_labels(idx):
    hour = idx.hour
    dow = idx.dayofweek
    return {
        "hour_band": np.select(
            [hour < 6, hour < 12, hour < 18],
            ["night", "morning", "afternoon"], default="evening"),
        "day_type": np.where(dow < 5, "weekday", "weekend"),
        "month": idx.month.astype(str).to_numpy(),
    }


def run_gaci(y_cal, p_cal, g_cal, p_st, y_st, g_st, alpha, gamma, window):
    uniq = sorted(set(g_cal) | set(g_st))
    acis = {}
    for g in uniq:
        aci = ACI(alpha=alpha, gamma=gamma, window=window)
        m = g_cal == g
        if m.sum() >= 200:
            aci.seed(y_cal[m], p_cal[m])
        acis[g] = aci
    res = []
    for p, y, g in zip(p_st, y_st, g_st):
        res.append({**acis[g].step(float(p), float(y)), "group": g})
    return res


def group_stats(res):
    df = pd.DataFrame(res)
    out = {}
    for g in sorted(set(df.group)):
        sub = df[df.group == g]
        out[str(g)] = {"n": int(len(sub)),
                       "coverage": float(sub.covered.mean()),
                       "mean_width": float((sub.hi - sub.lo).mean())}
    return out


def evaluate(name, y_cal, p_cal, g_cal, y_st, p_st, g_st, alpha, gamma, window):
    scp = SplitConformal(alpha=alpha).fit_calibrate(y_cal, p_cal)
    lo_s, hi_s = scp.predict(p_st)
    cov_s = (y_st >= lo_s) & (y_st <= hi_s)

    aci = ACI(alpha=alpha, gamma=gamma, window=window).seed(y_cal, p_cal)
    a_res = aci.run(p_st, y_st)
    cov_a = np.array([r["covered"] for r in a_res])
    w_a = np.array([r["hi"] - r["lo"] for r in a_res])

    g_res = run_gaci(y_cal, p_cal, g_cal, p_st, y_st, g_st, alpha, gamma, window)
    cov_g = np.array([r["covered"] for r in g_res])
    w_g = np.array([r["hi"] - r["lo"] for r in g_res])
    pg = group_stats(g_res)

    dfc = pd.DataFrame({"cov": cov_a, "g": g_st})
    aci_pg = {str(g): float(dfc[dfc.g == g]["cov"].mean())
              for g in sorted(set(g_st))}

    return {
        "grouping": name, "k_groups": len(set(g_cal) | set(g_st)),
        "scp": {"marginal_coverage": float(cov_s.mean()),
                "mean_width": float((hi_s - lo_s).mean())},
        "aci": {"marginal_coverage": float(cov_a.mean()),
                "mean_width": float(w_a.mean()),
                "worst_group_coverage": float(min(aci_pg.values())),
                "per_group_coverage": aci_pg},
        "gaci": {"marginal_coverage": float(cov_g.mean()),
                 "mean_width": float(w_g.mean()),
                 "worst_group_coverage": min(v["coverage"] for v in pg.values()),
                 "per_group": pg},
    }


def main():
    # replicate e2's data pipeline
    df = pd.read_csv(e2.DATA, sep=";", na_values="?", low_memory=False)
    df["ts"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                            format="%d/%m/%Y %H:%M:%S")
    df = df.set_index("ts").sort_index()
    y = df["Global_active_power"].resample("10min").mean().dropna()
    print(f"series: {len(y)} points", flush=True)

    feats = pd.DataFrame(index=y.index)
    for l in e2.LAGS:
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
    _ntr = int(len(yv) * e2.TRAIN_FRAC)
    mu, sd = X[:_ntr].mean(0), X[:_ntr].std(0) + 1e-9
    X = ((X - mu) / sd).astype(np.float32)

    n = len(yv)
    n_train = int(n * e2.TRAIN_FRAC)
    n_cal = int(n * e2.CAL_FRAC)
    print(f"train={n_train} cal={n_cal} stream={n-n_train-n_cal}", flush=True)

    w = e2.ridge_fit(X[:n_train], yv[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    y_cal, y_stream = yv[n_train:n_train + n_cal], yv[n_train + n_cal:]
    idx_cal = data.index[n_train:n_train + n_cal]
    idx_st = data.index[n_train + n_cal:]

    groups = group_labels(idx_st)
    groups_cal = {k: group_labels(idx_cal)[k] for k in groups}
    # map stream group arrays aligned to cal via same function
    gc = group_labels(idx_cal)

    alpha, gamma, window = e2.ALPHA, e2.GAMMA, e2.WINDOW
    results = {"experiment": "E6_electricity_gaci", "n": int(n),
               "alpha": alpha, "gamma": gamma, "window": window,
               "domain": "UCI household electric power, 10-min", "groupings": {}}

    for name in groups:
        r = evaluate(name, y_cal, pred_cal, gc[name], y_stream, pred_stream,
                     groups[name], alpha, gamma, window)
        results["groupings"][name] = r
        print(f"[{name}] k={r['k_groups']} "
              f"ACI worst={r['aci']['worst_group_coverage']:.3f} "
              f"GACI worst={r['gaci']['worst_group_coverage']:.3f} "
              f"w={r['gaci']['mean_width']:.3f}", flush=True)

    (OUT / "e6_metrics.json").write_text(json.dumps(results, indent=2))
    print("WROTE", OUT / "e6_metrics.json")


if __name__ == "__main__":
    main()
