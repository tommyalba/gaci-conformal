"""E5 / Phase 2 robustness: GACI across alternate groupings + hyperparameter
sensitivity + block-wise uncertainty.

Builds on E4 (grade-band grouping on Lending Club). Questions:
  A. Does the marginal-ACI-undercoverage / GACI-fix pattern replicate on
     OTHER groupings (purpose, term, home_ownership, issue-year)?
  B. How sensitive is GACI to alpha, gamma, and score-window size?
  C. Block-wise (10 contiguous stream blocks) coverage spread — honest
     uncertainty instead of a single point estimate.

RAM-conscious: reuses E4's ridge predictions; subsamples the stream for the
sensitivity grid; runs everything sequentially.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import e4_group_aci as e4
from e4_group_aci import (ACI, SplitConformal, RAW, OUT, TRAIN_FRAC, CAL_FRAC,
                          MAX_ROWS, build_features, ridge_fit, run_gaci,
                          group_stats)

GRID_STREAM_STEP = 2      # use every 2nd stream point for the grid
N_BLOCKS = 10
ALPHAS = [0.05, 0.1, 0.2]
GAMMAS = [0.005, 0.01, 0.02]
WINDOWS = [2000, 5000, 20000]


def grouping_arrays(df):
    """Return dict name -> group-label array aligned with df rows."""
    g = {}
    g["grade"] = df["grade"].astype(str).str.strip().str[0].str.upper().to_numpy()

    purp = df["purpose"].astype(str).fillna("missing")
    keep = purp.value_counts().head(8).index
    g["purpose"] = purp.where(purp.isin(keep), "OTHER").to_numpy()

    g["term"] = df["term"].astype(str).str.strip().to_numpy()

    home = df["home_ownership"].astype(str).fillna("missing")
    vc = home.value_counts()
    keep = set(vc[vc >= 200].index)          # only groups with real support
    g["home_ownership"] = home.where(home.isin(keep), "OTHER").to_numpy()

    yr = pd.to_datetime(df["issue_d"], errors="coerce").dt.year
    g["issue_year"] = yr.fillna(-1).astype(int).astype(str).to_numpy()
    return g


def evaluate_grouping(name, y_cal, p_cal, g_cal, y_st, p_st, g_st):
    """SCP / marginal ACI / GACI(own windows) / GACI(pooled) for one grouping."""
    # count only groups that arrive in the stream; cal-only sentinel
    # labels (e.g. stream_era's "cal") must not inflate k_groups
    uniq = sorted(set(g_st))
    # drop groups with no stream presence or tiny calibration support handled inside run_gaci
    scp = SplitConformal(alpha=e4.ALPHA).fit_calibrate(y_cal, p_cal)
    lo_s, hi_s = scp.predict(p_st)
    cov_s = (y_st >= lo_s) & (y_st <= hi_s)

    aci = ACI(alpha=e4.ALPHA, gamma=e4.GAMMA, window=e4.WINDOW).seed(y_cal, p_cal)
    a_res = aci.run(p_st, y_st)
    cov_a = np.array([r["covered"] for r in a_res])
    w_a = np.array([r["halfwidth"] for r in a_res])

    g_res = run_gaci(y_cal, p_cal, g_cal, p_st, y_st, g_st, pooled=False)
    cov_g = np.array([r["covered"] for r in g_res])
    w_g = np.array([r["hi"] - r["lo"] for r in g_res])
    pg = group_stats(g_res, g_st)

    gp_res = run_gaci(y_cal, p_cal, g_cal, p_st, y_st, g_st, pooled=True)
    cov_gp = np.array([r["covered"] for r in gp_res])
    w_gp = np.array([r["hi"] - r["lo"] for r in gp_res])
    pgp = group_stats(gp_res, g_st)

    # marginal ACI per-group coverage
    dfc = pd.DataFrame({"cov": cov_a, "g": g_st})
    aci_pg = {str(g): float(dfc[dfc.g == g]["cov"].mean())
              for g in sorted(set(g_st))}

    return {
        "grouping": name, "k_groups": len(uniq),
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
        "gaci_pooled": {"marginal_coverage": float(cov_gp.mean()),
                        "mean_width": float(w_gp.mean()),
                        "worst_group_coverage": min(v["coverage"] for v in pgp.values()),
                        "per_group": pgp},
        # series for block analysis (grade only used downstream)
        "_cov": {"aci": cov_a, "gaci": cov_g, "gaci_pooled": cov_gp,
                 "scp": cov_s},
        "_groups": g_st,
    }


def block_stats(cov, groups, n_blocks=N_BLOCKS):
    """Split stream into contiguous blocks; per-block marginal + worst-group cov."""
    n = len(cov)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    rows = []
    for b in range(n_blocks):
        s = slice(edges[b], edges[b + 1])
        c, g = cov[s], groups[s]
        worst = min(float(c[g == gv].mean()) for gv in set(g))
        rows.append({"block": b, "marginal": float(c.mean()),
                     "worst_group": worst})
    return rows


def sensitivity_grid(y_cal, p_cal, g_cal, y_st, p_st, g_st):
    """alpha x gamma x window grid on subsampled stream, grade grouping."""
    sl = slice(None, None, GRID_STREAM_STEP)
    y_s, p_s, g_s = y_st[sl], p_st[sl], g_st[sl]
    out = []
    for a in ALPHAS:
        for gm in GAMMAS:
            for w in WINDOWS:
                # run_gaci reads e4_group_aci's module globals — mutate those
                e4.ALPHA, e4.GAMMA, e4.WINDOW = a, gm, w
                res = run_gaci(y_cal, p_cal, g_cal, p_s, y_s, g_s, pooled=False)
                cov = np.array([r["covered"] for r in res])
                wid = np.array([r["hi"] - r["lo"] for r in res])
                pg = group_stats(res, g_s)
                out.append({"alpha": a, "gamma": gm, "window": w,
                            "marginal_coverage": float(cov.mean()),
                            "mean_width": float(wid.mean()),
                            "worst_group_coverage": min(
                                v["coverage"] for v in pg.values())})
                print(f"  grid a={a} g={gm} w={w}: cov={cov.mean():.3f} "
                      f"worst={out[-1]['worst_group_coverage']:.3f}", flush=True)
    e4.ALPHA, e4.GAMMA, e4.WINDOW = 0.1, 0.01, 5000  # restore module defaults
    return out


def main():
    usecols = ["issue_d", "int_rate", "grade", "purpose", "term",
               "home_ownership"] + e4.NUMERIC + list(e4.CATEGORICAL)
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

    X, _ = build_features(df)
    n = len(y)
    n_train = int(n * TRAIN_FRAC); n_cal = int(n * CAL_FRAC)
    w = ridge_fit(X[:n_train], y[:n_train])
    pred_cal = X[n_train:n_train + n_cal] @ w
    pred_stream = X[n_train + n_cal:] @ w
    y_cal, y_stream = y[n_train:n_train + n_cal], y[n_train + n_cal:]

    groups = grouping_arrays(df)
    slice_cal = slice(n_train, n_train + n_cal)
    slice_st = slice(n_train + n_cal, n)

    # stream-internal era split: train/cal get own labels; stream split in half
    n_st = n - n_train - n_cal
    era = np.array(["train"] * n_train + ["cal"] * n_cal +
                   ["early" if i < n_st // 2 else "late"
                    for i in range(n_st)])
    groups["stream_era"] = era
    results = {"experiment": "E5_gaci_robustness", "n": int(n),
               "alpha": e4.ALPHA, "gamma": e4.GAMMA, "window": e4.WINDOW,
               "groupings": {}, "sensitivity_grid": [], "block_analysis": {}}

    all_cov = {}
    all_groups = {}
    for name, g_all in groups.items():
        g_cal, g_st = g_all[slice_cal], g_all[slice_st]
        r = evaluate_grouping(name, y_cal, pred_cal, g_cal, y_stream, pred_stream, g_st)
        all_cov[name] = r.pop("_cov"); all_groups[name] = r.pop("_groups")
        results["groupings"][name] = r
        print(f"[{name}] k={r['k_groups']} "
              f"ACI worst-grp={r['aci']['worst_group_coverage']:.3f} "
              f"GACI worst-grp={r['gaci']['worst_group_coverage']:.3f} "
              f"pooled={r['gaci_pooled']['worst_group_coverage']:.3f}", flush=True)

    # block analysis on grade grouping
    grade_cov = all_cov["grade"]
    results["block_analysis"]["grade"] = {
        m: block_stats(grade_cov[m], all_groups["grade"])
        for m in ("aci", "gaci", "gaci_pooled", "scp")}

    # sensitivity grid on grade grouping
    g_all = groups["grade"]
    results["sensitivity_grid"] = sensitivity_grid(
        y_cal, pred_cal, g_all[slice_cal], y_stream, pred_stream, g_all[slice_st])

    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "e5_series.npz",
             **{f"{m}_{k}_cov": v for k, covs in all_cov.items()
                for m, v in covs.items()},
             **{f"{k}_groups": v for k, v in all_groups.items()})
    (OUT / "e5_metrics.json").write_text(json.dumps(results, indent=2))
    print("WROTE", OUT / "e5_metrics.json")


if __name__ == "__main__":
    main()
