"""Extra Phase-1 metrics computed from saved experiment series (no rerun).

Adds per experiment:
- coverage_gap_regret: cumulative sum of |rolling coverage - target| — lower is better
- winkler_interval_score: mean interval score (coverage penalty + width); proper scoring rule
- coverage_by_third: mean coverage in first/middle/last third of stream (drift signature)
"""

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
ALPHA = 0.1


def metrics_from(npz_name):
    d = np.load(RES / npz_name, allow_pickle=True)
    y = d["y"] if "y" in d else d["ys"]
    out = {}
    for m in ["scp", "aci", "dt"]:
        lo, hi = d[f"{m}_lo"], d[f"{m}_hi"]
        cov = d[f"{m}_cov"].astype(float)
        # Winkler interval score at level alpha
        width = hi - lo
        miss = np.maximum(lo - y, 0) + np.maximum(y - hi, 0)
        score = width + (2 / ALPHA) * miss
        # cumulative coverage-gap regret on rolling coverage (win=2000)
        win = min(2000, len(cov) // 10 or 1)
        kern = np.ones(win) / win
        roll = np.convolve(cov, kern, mode="valid")
        regret = float(np.sum(np.abs(roll - (1 - ALPHA))))
        thirds = np.array_split(cov, 3)
        out[m] = {
            "winkler_score": float(np.mean(score)),
            "coverage_gap_regret": regret,
            "coverage_by_third": [float(np.mean(t)) for t in thirds],
        }
    return out


def main():
    allm = {}
    for name, tag in [("e1_series.npz", "E1"), ("e2_series.npz", "E2"), ("e3_series.npz", "E3")]:
        p = RES / name
        if not p.exists():
            continue
        allm[tag] = metrics_from(name)
        # merge into existing metrics file
        mp = RES / name.replace("_series.npz", "_metrics.json")
        if mp.exists():
            cur = json.loads(mp.read_text())
            for m, mm in allm[tag].items():
                cur.setdefault(m, {}).update(mm)
            mp.write_text(json.dumps(cur, indent=2))
    print(json.dumps(allm, indent=2))


if __name__ == "__main__":
    main()
