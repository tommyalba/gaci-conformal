"""Block-bootstrap confidence intervals for width/coverage claims (E11).

Addresses review W5/P5: width comparisons were point estimates from a
single stream ordering. We split the E4 stream into 10 contiguous blocks
and bootstrap over blocks (resample with replacement, fixed seed) to get
percentile CIs for:

- mean interval width per method (SCP / marginal ACI / GACI)
- mean width on the hardest group (G) and easiest (A) under each method
- worst-group coverage per method

Outputs results/ci_metrics.json. Run after e4_group_aci.py.
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
N_BLOCKS = 10
N_BOOT = 20000
SEED = 20261123


def block_means(values, groups=None, group_val=None, n_blocks=N_BLOCKS):
    """Mean of `values` per contiguous block (optionally filtered to a
    group). Returns array of per-block means."""
    n = len(values)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    out = []
    for b in range(n_blocks):
        sl = slice(edges[b], edges[b + 1])
        v = values[sl]
        if group_val is not None:
            m = groups[sl] == group_val
            v = v[m]
        out.append(float(v.mean()) if len(v) else np.nan)
    return np.array(out)


def block_worst_group_coverage(cov, groups, n_blocks=N_BLOCKS):
    """Per-block minimum group coverage (matches e5 block_stats)."""
    n = len(cov)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    out = []
    for b in range(n_blocks):
        sl = slice(edges[b], edges[b + 1])
        per_g = [cov[sl][groups[sl] == g].mean() for g in set(groups[sl].tolist())]
        out.append(float(min(per_g)))
    return np.array(out)


def boot_ci(block_vals, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(block_vals), size=(n_boot, len(block_vals)))
    stats = block_vals[idx].mean(axis=1)
    return {"mean": float(block_vals.mean()),
            "ci95": [float(np.percentile(stats, 2.5)),
                     float(np.percentile(stats, 97.5))],
            "block_values": [round(float(v), 5) for v in block_vals]}


def main():
    z = np.load(RES / "e4_series.npz", allow_pickle=True)
    groups = z["groups"]
    methods = {}
    for m in ("scp", "aci", "gaci"):
        lo, hi = z[f"{m}_lo"], z[f"{m}_hi"]
        cov = z[f"{m}_cov"].astype(float)
        w = hi - lo
        entry = {"mean_width": boot_ci(block_means(w)),
                 "worst_group_coverage": boot_ci(
                     block_worst_group_coverage(cov, groups))}
        # per-grade widths for the cost story
        for g in ("A", "G"):
            entry[f"width_grade_{g}"] = boot_ci(block_means(w, groups, g))
            entry[f"coverage_grade_{g}"] = boot_ci(
                block_means(cov, groups, g))
        methods[m] = entry

    out = {"experiment": "E11_block_bootstrap_cis", "n_blocks": N_BLOCKS,
           "n_boot": N_BOOT, "seed": SEED, "methods": methods}
    (RES / "ci_metrics.json").write_text(json.dumps(out, indent=2))
    for m, e in methods.items():
        print(f"{m}: width {e['mean_width']['mean']:.4f} "
              f"CI[{e['mean_width']['ci95'][0]:.4f},{e['mean_width']['ci95'][1]:.4f}] "
              f"| G width {e['width_grade_G']['mean']:.4f} "
              f"CI[{e['width_grade_G']['ci95'][0]:.4f},{e['width_grade_G']['ci95'][1]:.4f}] "
              f"| worst-cov {e['worst_group_coverage']['mean']:.3f}")
    print("WROTE", RES / "ci_metrics.json")


if __name__ == "__main__":
    main()
