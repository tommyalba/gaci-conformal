"""E8 / pre-writing: LightGBM base-model robustness for E4 and E6.

THEORY.md lists "single (weak) ridge predictor" as a limitation; PLAN.md's
TabArena note recommends a stronger base model before any TMLR-level
submission. This reruns the headline comparisons with LightGBM (L2
regression, CPU hist) replacing ridge, holding conformal machinery
(SCP / marginal ACI / GACI) identical.

Question: does the marginal-ACI group undercoverage -> GACI fix pattern
survive a substantially stronger point predictor?

Memory: same CSV reads as e7 (~1-1.5GB peak). Run AFTER ai-eng build.
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
import e4_group_aci as e4
import e6_electricity_gaci as e6
from e4_group_aci import (ACI, SplitConformal, NUMERIC, CATEGORICAL,
                          TRAIN_FRAC, CAL_FRAC, MAX_ROWS, build_features,
                          run_gaci, group_stats)
from e6_electricity_gaci import group_labels, run_gaci as e6_run_gaci

OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

LGB_PARAMS = dict(objective="regression", metric="l2", num_leaves=31,
                  learning_rate=0.05, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=5, min_data_in_leaf=50,
                  num_threads=4, verbose=-1, seed=42)
N_ROUNDS = 400


def lgb_fit_predict(Xtr, ytr, Xev):
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr)
    dte = lgb.Dataset(Xev, reference=dtr)
    booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=N_ROUNDS,
                        valid_sets=[dte], callbacks=[lgb.log_evaluation(0)])
    return booster.predict(Xev)


def e4_lightgbm(df):
    df = df.copy()
    if len(df) > MAX_ROWS:
        idx = np.linspace(0, len(df) - 1, MAX_ROWS).astype(int)
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
    Xtr, Xcal, Xst = X[:n_train], X[n_train:n_train+n_cal], X[n_train+n_cal:]
    ytr, y_cal, y_st = y[:n_train], y[n_train:n_train+n_cal], y[n_train+n_cal:]
    g_cal, g_st = groups_all[n_train:n_train+n_cal], groups_all[n_train+n_cal:]

    pred_cal = lgb_fit_predict(Xtr, ytr, Xcal)
    pred_stream = lgb_fit_predict(Xtr, ytr, Xst)

    aci = ACI(alpha=e4.ALPHA, gamma=e4.GAMMA, window=e4.WINDOW).seed(y_cal, pred_cal)
    a_res = aci.run(pred_stream, y_st)
    cov_a = np.array([r["covered"] for r in a_res])
    w_a = np.array([r["hi"] - r["lo"] for r in a_res])
    g_res = run_gaci(pred_cal, y_cal, g_cal, pred_stream, y_st, g_st, pooled=False)
    cov_g = np.array([r["covered"] for r in g_res])
    w_g = np.array([r["hi"] - r["lo"] for r in g_res])
    aci_pg = {str(g): float(pd.Series(cov_a)[g_st == g].mean())
              for g in sorted(set(g_st))}
    gpg = group_stats(g_res, g_st)
    return {"n": int(n),
            "aci": {"marginal_coverage": float(cov_a.mean()),
                    "mean_width": float(w_a.mean()),
                    "worst_group_coverage": float(min(aci_pg.values())),
                    "per_group_coverage": aci_pg},
            "gaci": {"marginal_coverage": float(cov_g.mean()),
                     "mean_width": float(w_g.mean()),
                     "worst_group_coverage": min(v["coverage"] for v in gpg.values()),
                     "per_group": gpg}}


def e6_data():
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
    return yv, ((X - mu) / sd).astype(np.float32), data.index


def e6_lightgbm(yv, X, idx):
    n = len(yv)
    n_train = int(n * e2.TRAIN_FRAC); n_cal = int(n * e2.CAL_FRAC)
    Xtr = X[:n_train]; Xcal = X[n_train:n_train+n_cal]; Xst = X[n_train+n_cal:]
    ytr = yv[:n_train]; y_cal = yv[n_train:n_train+n_cal]; y_st = yv[n_train+n_cal:]
    idx_cal = idx[n_train:n_train+n_cal]; idx_st = idx[n_train+n_cal:]
    pred_cal = lgb_fit_predict(Xtr, ytr, Xcal)
    pred_stream = lgb_fit_predict(Xtr, ytr, Xst)

    out = {}
    for name in ("hour_band", "day_type", "month"):
        g_st = group_labels(idx_st)[name]
        g_cal = group_labels(idx_cal)[name]
        aci = ACI(alpha=e2.ALPHA, gamma=e2.GAMMA, window=e2.WINDOW).seed(y_cal, pred_cal)
        a_res = aci.run(pred_stream, y_st)
        cov_a = np.array([r["covered"] for r in a_res])
        w_a = np.array([r["hi"] - r["lo"] for r in a_res])
        g_res = e6_run_gaci(y_cal, pred_cal, g_cal, pred_stream, y_st, g_st,
                            e2.ALPHA, e2.GAMMA, e2.WINDOW)
        cov_g = np.array([r["covered"] for r in g_res])
        w_g = np.array([r["hi"] - r["lo"] for r in g_res])
        aci_pg = {str(g): float(pd.Series(cov_a)[g_st == g].mean())
                  for g in sorted(set(g_st))}
        gdf = pd.DataFrame({"cov": cov_g, "g": [r["group"] for r in g_res]})
        gpg = {str(g): float(gdf[gdf.g == g]["cov"].mean())
               for g in sorted(set(gdf.g))}
        out[name] = {
            "aci": {"marginal_coverage": float(cov_a.mean()),
                    "mean_width": float(w_a.mean()),
                    "worst_group_coverage": float(min(aci_pg.values())),
                    "per_group_coverage": aci_pg},
            "gaci": {"marginal_coverage": float(cov_g.mean()),
                     "mean_width": float(w_g.mean()),
                     "worst_group_coverage": float(min(gpg.values())),
                     "per_group_coverage": gpg},
        }
    return out


def main():
    results = {"experiment": "E8_lightgbm_base_model",
               "lgb_params": LGB_PARAMS, "n_rounds": N_ROUNDS,
               "e4": None, "e6": {}}

    usecols = ["issue_d", "int_rate", "grade"] + NUMERIC + list(CATEGORICAL)
    df = pd.read_csv(e4.RAW, usecols=lambda c: c in usecols, low_memory=False)
    df = df.dropna(subset=["issue_d", "int_rate", "grade"])
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%y",
                                   errors="coerce")
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d").reset_index(drop=True)
    print(f"E4 frame: {len(df)} rows", flush=True)
    results["e4"] = e4_lightgbm(df)
    print(f"[E4 lgb] ACI worst={results['e4']['aci']['worst_group_coverage']:.3f} "
          f"GACI worst={results['e4']['gaci']['worst_group_coverage']:.3f}",
          flush=True)
    del df

    yv, X, idx = e6_data()
    print(f"E6 series: {len(yv)}", flush=True)
    results["e6"] = e6_lightgbm(yv, X, idx)
    for name, r in results["e6"].items():
        print(f"[E6 lgb {name}] ACI worst={r['aci']['worst_group_coverage']:.3f} "
              f"GACI worst={r['gaci']['worst_group_coverage']:.3f}", flush=True)

    (OUT / "e8_metrics.json").write_text(json.dumps(results, indent=2))
    print("WROTE", OUT / "e8_metrics.json")


if __name__ == "__main__":
    main()
