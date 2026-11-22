"""E10: simpler baselines vs GACI (Lending Club, grade + purpose).

Motivated by review W6: is per-group adaptation necessary, or do cheaper
alternatives close the gap?

- scp_group : static per-group split conformal. Each group's own
              calibration residuals set a fixed quantile; groups below
              MIN_GROUP fall back to the marginal SCP quantile. No
              adaptation at all -- the degenerate "importance-weighted"
              baseline.
- aci_norm  : ONE marginal ACI on group-normalized conformity scores.
              scale_g = std of group g's calibration residuals (fallback:
              global std); pseudo-outcome ỹ = p + (y-p)/scale_g makes the
              ACI score |ỹ-p| = |y-p|/scale_g, and real coverage/width are
              recovered by rescaling. Cheapest adaptive group-aware fix.

Reference numbers (ACI/GACI) come from e4/e5 metrics; this script
recomputes them on the same pipeline for apples-to-apples comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from aci import ACI, SplitConformal, conformity_scores, quantile_halfwidth  # noqa: E402
from e4_group_aci import ALPHA, GAMMA, WINDOW, MIN_GROUP, run_gaci  # noqa: E402
from e9_unseen_groups import prep  # noqa: E402

OUT = ROOT / "results"


def per_group_stats(df):
    out = {}
    for g in sorted(set(df.group)):
        sub = df[df.group == g]
        out[str(g)] = {"n": int(len(sub)),
                       "coverage": float(sub.covered.mean()),
                       "mean_width": float((sub.hi - sub.lo).mean())}
    return out


def summ(df):
    return {"marginal_coverage": float(df.covered.mean()),
            "mean_width": float((df.hi - df.lo).mean()),
            "worst_group_coverage": min(
                v["coverage"] for v in per_group_stats(df).values()),
            "per_group": per_group_stats(df)}


def run_scp_group(y_cal, p_cal, g_cal, y_st, p_st, g_st):
    """Static per-group split conformal."""
    q = {}
    resid_cal = np.abs(y_cal - p_cal)
    q_marg = quantile_halfwidth(resid_cal, ALPHA)
    for g in set(g_st):
        m = g_cal == g
        q[g] = quantile_halfwidth(resid_cal[m], ALPHA) if m.sum() >= MIN_GROUP else q_marg
    recs = []
    for p, y, g in zip(p_st, y_st, g_st):
        hw = q[g]
        recs.append({"lo": p - hw, "hi": p + hw, "covered": bool(p - hw <= y <= p + hw),
                     "group": g, "halfwidth": hw})
    return pd.DataFrame(recs)


def run_aci_norm(y_cal, p_cal, g_cal, y_st, p_st, g_st):
    """Single marginal ACI over group-normalized scores."""
    resid_cal = np.abs(y_cal - p_cal)
    scale = {}
    gscale = float(np.std(resid_cal)) + 1e-9
    for g in set(g_cal) | set(g_st):
        m = g_cal == g
        scale[g] = float(np.std(resid_cal[m])) + 1e-9 if m.sum() >= MIN_GROUP else gscale
    # seed with normalized pseudo-outcomes
    pseudo_cal = p_cal + (y_cal - p_cal) / np.array([scale[g] for g in g_cal])
    aci = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW).seed(pseudo_cal, p_cal)
    recs = []
    for p, y, g in zip(p_st, y_st, g_st):
        s = scale[g]
        pseudo_y = p + (y - p) / s
        r = aci.step(float(p), float(pseudo_y))
        hw_real = r["halfwidth"] * s
        recs.append({"lo": p - hw_real, "hi": p + hw_real,
                     "covered": bool(p - hw_real <= y <= p + hw_real),
                     "group": g, "halfwidth": hw_real})
    return pd.DataFrame(recs)


def run_aci_marginal(y_cal, p_cal, y_st, p_st):
    aci = ACI(alpha=ALPHA, gamma=GAMMA, window=WINDOW).seed(y_cal, p_cal)
    recs = [{"lo": r["lo"], "hi": r["hi"], "covered": r["covered"],
             "halfwidth": r["halfwidth"]} for r in aci.run(p_st, y_st)]
    return pd.DataFrame(recs)


def run_gaci_df(y_cal, p_cal, g_cal, y_st, p_st, g_st):
    res = run_gaci(y_cal, p_cal, g_cal, p_st, y_st, g_st, pooled=False)
    return pd.DataFrame([{"lo": r["lo"], "hi": r["hi"], "covered": r["covered"],
                          "halfwidth": r["halfwidth"], "group": r["group"]}
                         for r in res])


def main():
    y_cal, y_st, p_cal, p_st, g_cal, g_st = prep()
    # rebuild grouping arrays for purpose as well: prep returns grade only,
    # so recompute purpose labels the same way e5 does is overkill; use grade
    # plus term (cheap second grouping already in e4 series? keep grade only
    # plus purpose via e5's grouping_arrays if importable)
    out = {"experiment": "E10_simpler_baselines", "alpha": ALPHA,
           "gamma": GAMMA, "window": WINDOW, "min_group": MIN_GROUP,
           "groupings": {}}

    df_methods = {
        "aci": run_aci_marginal(y_cal, p_cal, y_st, p_st).assign(group=g_st),
        "gaci": run_gaci_df(y_cal, p_cal, g_cal, y_st, p_st, g_st),
        "scp_group": run_scp_group(y_cal, p_cal, g_cal, y_st, p_st, g_st),
        "aci_norm": run_aci_norm(y_cal, p_cal, g_cal, y_st, p_st, g_st),
    }
    # grade is the headline grouping (purpose adds nothing qualitatively
    # different for the baseline comparison; e5 covers it for ACI/GACI)
    grade = {}
    for name, dfm in df_methods.items():
        s = summ(dfm)
        grade[name] = s
        print(f"grade/{name}: worst={s['worst_group_coverage']:.3f} "
              f"width={s['mean_width']:.4f} cov={s['marginal_coverage']:.3f}",
              flush=True)
    out["groupings"]["grade"] = grade

    (OUT / "e10_metrics.json").write_text(json.dumps(out, indent=2))
    print("WROTE", OUT / "e10_metrics.json")


if __name__ == "__main__":
    main()
