"""E7 / pre-writing: seed & subsample variance for E4 and E6.

Reviewers will ask whether the headline gaps (marginal-ACI worst-group
undercoverage vs GACI fix) are artifacts of one deterministic subsample /
calibration split. This script quantifies the spread:

  E4 (Lending Club): two alternate MAX_ROWS subsamples (150k, 250k) —
      different linspace row subsets -> different train/cal/stream
      composition. Reports worst-group coverage for ACI vs GACI per run.
  E6 (electricity):  (a) block-wise uncertainty — 10 contiguous stream
      blocks, worst-group coverage per block per grouping; (b) one
      alternate CAL_FRAC (0.10 -> 0.15) rerun for all three groupings.

Memory-frugal: the Lending Club CSV is read ONCE; per-variant work is
slice -> features -> ridge -> conformal loops (~200k rows max in RAM at
once). Run AFTER the LEANN ai-eng build finishes (needs ~1-1.5GB peak).
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import e2_electricity as e2
import e4_group_aci as e4
import e6_electricity_gaci as e6
from e4_group_aci import (ACI, SplitConformal, NUMERIC, CATEGORICAL,
                          TRAIN_FRAC, CAL_FRAC, build_features, ridge_fit,
                          run_gaci, group_stats, MIN_GROUP)
from e6_electricity_gaci import group_labels, run_gaci as e6_run_gaci

OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

E4_VARIANTS = [150_000, 250_000]          # baseline 200k already in e4_metrics
E6_CAL_FRACS = [0.15]                      # baseline 0.10 already in e6_metrics
N_BLOCKS = 10


def e4_variant(df_full, max_rows):
    """One E4 rerun on a different linspace subsample. Returns summary dict."""
    df = df_full
    if len(df) > max_rows:
        idx = np.linspace(0, len(df) - 1, max_rows).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    df["grade"] = df["grade"].astype(str).str.strip().str[0].str.upper()
    df = df[df["grade"].isin(list("ABCDEFG"))].reset_index(drop=True)

    y = pd.to_numeric(df["int_rate"], errors="coerce").to_numpy() / 100.0
    mask = ~np.isnan(y)
    df, y = df.loc[mask].reset_index(drop=True), y[mask]
    groups_all = df["grade"].to_numpy()

    n = len(y)
    n_train = int(n * TRAIN_FRAC)
    X, _ = build_features(df, fit_rows=n_train)
    n_cal = int(n * CAL_FRAC)
    w = ridge_fit(X[:n_train], y[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    y_cal, y_stream = y[n_train:n_train + n_cal], y[n_train + n_cal:]
    g_cal = groups_all[n_train:n_train + n_cal]
    g_stream = groups_all[n_train + n_cal:]

    aci = ACI(alpha=e4.ALPHA, gamma=e4.GAMMA, window=e4.WINDOW).seed(y_cal, pred_cal)
    a_res = aci.run(pred_stream, y_stream)
    cov_a = np.array([r["covered"] for r in a_res])
    w_a = np.array([r["hi"] - r["lo"] for r in a_res])

    g_res = run_gaci(pred_cal, y_cal, g_cal, pred_stream, y_stream, g_stream,
                     pooled=False)
    cov_g = np.array([r["covered"] for r in g_res])
    w_g = np.array([r["hi"] - r["lo"] for r in g_res])

    aci_pg = {str(g): float(pd.Series(cov_a)[g_stream == g].mean())
              for g in sorted(set(g_stream))}
    gpg = group_stats(g_res, g_stream)
    return {
        "max_rows": int(max_rows), "n": int(n),
        "aci": {"marginal_coverage": float(cov_a.mean()),
                "mean_width": float(w_a.mean()),
                "worst_group_coverage": float(min(aci_pg.values())),
                "per_group_coverage": aci_pg},
        "gaci": {"marginal_coverage": float(cov_g.mean()),
                 "mean_width": float(w_g.mean()),
                 "worst_group_coverage": min(v["coverage"] for v in gpg.values()),
                 "per_group": gpg},
    }


def e6_data():
    """Replicate e6's data pipeline once; return reusable arrays."""
    df = pd.read_csv(e2.DATA, sep=";", na_values="?", low_memory=False)
    df["ts"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                            format="%d/%m/%Y %H:%M:%S")
    df = df.set_index("ts").sort_index()
    y = df["Global_active_power"].resample("10min").mean().dropna()

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
    return yv, X, data.index


def e6_run(yv, X, idx, cal_frac):
    """One E6 evaluate() pass at a given CAL_FRAC; returns per-step covs."""
    n = len(yv)
    n_train = int(n * e2.TRAIN_FRAC)
    n_cal = int(n * cal_frac)
    w = e2.ridge_fit(X[:n_train], yv[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    y_cal = yv[n_train:n_train + n_cal]
    y_st = yv[n_train + n_cal:]
    idx_cal = idx[n_train:n_train + n_cal]
    idx_st = idx[n_train + n_cal:]

    out = {}
    for name in ("hour_band", "day_type", "month"):
        g_st = group_labels(idx_st)[name]
        g_cal = group_labels(idx_cal)[name]
        scp = SplitConformal(alpha=e2.ALPHA).fit_calibrate(y_cal, pred_cal)
        lo_s, hi_s = scp.predict(pred_stream)
        aci = ACI(alpha=e2.ALPHA, gamma=e2.GAMMA,
                  window=e2.WINDOW).seed(y_cal, pred_cal)
        a_res = aci.run(pred_stream, y_st)
        g_res = e6_run_gaci(y_cal, pred_cal, g_cal, pred_stream, y_st, g_st,
                            e2.ALPHA, e2.GAMMA, e2.WINDOW)
        out[name] = {
            "aci_cov": np.array([r["covered"] for r in a_res]),
            "aci_w": np.array([r["hi"] - r["lo"] for r in a_res]),
            "gaci_cov": np.array([r["covered"] for r in g_res]),
            "gaci_w": np.array([r["hi"] - r["lo"] for r in g_res]),
            "gaci_grp": np.array([r["group"] for r in g_res]),
            "aci_grp": g_st,
        }
    return out


def block_worst_group(cov, grp, n_blocks=N_BLOCKS):
    """Worst-group coverage within each contiguous stream block."""
    blocks = np.array_split(np.arange(len(cov)), n_blocks)
    worst = []
    for b in blocks:
        vals = [float(cov[b][grp[b] == g].mean())
                for g in np.unique(grp[b])]
        worst.append(min(vals))
    return {"block_worst": [round(v, 4) for v in worst],
            "mean": float(np.mean(worst)), "min": float(np.min(worst)),
            "max": float(np.max(worst))}


def main():
    results = {"experiment": "E7_seed_subsample_variance", "e4": {}, "e6": {}}

    # ---- E4 alternate subsamples (single CSV read) ----
    usecols = ["issue_d", "int_rate", "grade"] + NUMERIC + list(CATEGORICAL)
    df_full = pd.read_csv(e4.RAW, usecols=lambda c: c in usecols,
                          low_memory=False)
    df_full = df_full.dropna(subset=["issue_d", "int_rate", "grade"])
    df_full["issue_d"] = pd.to_datetime(df_full["issue_d"], format="%b-%y",
                                        errors="coerce")
    df_full = df_full.dropna(subset=["issue_d"]).sort_values(
        "issue_d").reset_index(drop=True)
    print(f"E4 full frame: {len(df_full)} rows", flush=True)

    for mr in E4_VARIANTS:
        r = e4_variant(df_full, mr)
        results["e4"][f"max_rows_{mr}"] = r
        print(f"[E4 {mr}] ACI worst={r['aci']['worst_group_coverage']:.3f} "
              f"GACI worst={r['gaci']['worst_group_coverage']:.3f}", flush=True)
        del r
        gc.collect()
    del df_full
    gc.collect()

    # ---- E6: reuse data once; baseline CAL_FRAC=0.10 block stats + alt ----
    yv, X, idx = e6_data()
    print(f"E6 series: {len(yv)} points", flush=True)

    base = e6_run(yv, X, idx, cal_frac=e2.CAL_FRAC)
    e6_blocks = {}
    for name, d in base.items():
        e6_blocks[name] = {
            "aci_worst_group_by_block": block_worst_group(
                d["aci_cov"], d["aci_grp"]),
            "gaci_worst_group_by_block": block_worst_group(
                d["gaci_cov"], d["gaci_grp"]),
        }
    results["e6"]["baseline_cal0.10_blocks"] = e6_blocks
    del base
    gc.collect()

    for cf in E6_CAL_FRACS:
        r = e6_run(yv, X, idx, cal_frac=cf)
        alt = {}
        for name, d in r.items():
            aci_pg = {}
            for g in np.unique(d["aci_grp"]):
                aci_pg[str(g)] = float(d["aci_cov"][d["aci_grp"] == g].mean())
            gpg = {}
            for g in np.unique(d["gaci_grp"]):
                gpg[str(g)] = float(d["gaci_cov"][d["gaci_grp"] == g].mean())
            alt[name] = {
                "aci_worst_group": float(min(aci_pg.values())),
                "gaci_worst_group": float(min(gpg.values())),
                "aci_marginal": float(d["aci_cov"].mean()),
                "gaci_marginal": float(d["gaci_cov"].mean()),
                "gaci_mean_width": float(d["gaci_w"].mean()),
            }
        results["e6"][f"cal_frac_{cf}"] = alt
        print(f"[E6 cal={cf}] " + "; ".join(
            f"{k}: ACIworst={v['aci_worst_group']:.3f} "
            f"GACIworst={v['gaci_worst_group']:.3f}" for k, v in alt.items()),
            flush=True)
        del r
        gc.collect()

    (OUT / "e7_metrics.json").write_text(json.dumps(results, indent=2))
    print("WROTE", OUT / "e7_metrics.json")


if __name__ == "__main__":
    main()
