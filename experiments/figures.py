"""Generate manuscript figures from saved experiment series/metrics.

Fig1: per-grade rolling coverage trajectories (E4, Lending Club grades A-G),
      marginal ACI vs GACI vs SCP — the "money figure" showing ACI's
      grade-G undercoverage (~0.23) vs GACI's equalization (~0.89).
Fig2: coverage-vs-width frontier scatter — worst-group coverage vs mean
      interval width for every method x grouping x experiment.

Run: ~/miniconda3/envs/credit-risk_VENV/bin/python experiments/figures.py
Outputs: results/figures/fig1_coverage_trajectory.png, fig2_frontier.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures"
OUT.mkdir(exist_ok=True)
TARGET = 0.90


def rolling_coverage(cov, window):
    """Trailing rolling mean of a boolean coverage array."""
    c = np.convolve(cov.astype(float), np.ones(window) / window, mode="valid")
    return c


def fig1():
    z = np.load(RES / "e4_series.npz", allow_pickle=True)
    groups = z["groups"]
    series = {"SCP": z["scp_cov"], "ACI": z["aci_cov"], "GACI": z["gaci_cov"]}
    grades = sorted(set(groups.tolist()))

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=False)
    axes = axes.ravel()
    colors = {"SCP": "#9e9e9e", "ACI": "#d62728", "GACI": "#1f77b4"}

    for ax, g in zip(axes, grades):
        mask = groups == g
        n = int(mask.sum())
        # window ~10% of group size, clamped for tiny groups
        w = int(np.clip(n // 8, 100, 2000))
        xs = np.arange(w - 1, n)
        for name, cov in series.items():
            sub = cov[mask]
            if len(sub) < w:
                continue
            r = rolling_coverage(sub, w)
            lw = 1.8 if name == "GACI" else (1.4 if name == "ACI" else 1.0)
            ls = "-" if name != "SCP" else "--"
            ax.plot(xs / max(n - 1, 1), r, color=colors[name], lw=lw,
                    ls=ls, alpha=0.95 if name != "SCP" else 0.6,
                    label=name if g == grades[0] else None)
        ax.axhline(TARGET, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Grade {g}  (n={n:,})", fontsize=10)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="lower right")

    axes[-1].axis("off")
    fig.suptitle("Per-grade coverage over the stream — marginal ACI vs GACI\n"
                 "(E4, Lending Club, rolling window ≈10% of group arrivals)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig1_coverage_trajectory.png", dpi=200)
    plt.close(fig)
    print("fig1 written")


def _methods(block):
    """Yield (method, mean_width, worst_group_coverage) where available."""
    for m in ("aci", "gaci", "gaci_pooled"):
        if m in block and "worst_group_coverage" in block[m]:
            yield m, block[m]["mean_width"], block[m]["worst_group_coverage"]


def fig2():
    pts = []  # (method, width, worst_cov, domain, base_model)

    def add(block, domain, base="ridge"):
        for m, w, c in _methods(block):
            pts.append((m, w, c, domain, base))

    # E5: six groupings on Lending Club (ridge)
    e5 = json.load(open(RES / "e5_metrics.json"))
    for gname, g in e5["groupings"].items():
        add(g, "LendingClub", "ridge")

    # E6: three groupings on Electricity (ridge)
    e6 = json.load(open(RES / "e6_metrics.json"))
    for gname, g in e6["groupings"].items():
        add(g, "Electricity", "ridge")

    # E7: subsample variance, Lending Club (ridge)
    e7 = json.load(open(RES / "e7_metrics.json"))
    for sub, blk in e7["e4"].items():
        add(blk, "LendingClub", "ridge")

    # E8: LightGBM base model
    e8 = json.load(open(RES / "e8_metrics.json"))
    add(e8["e4"], "LendingClub", "lightgbm")
    for gname, g in e8["e6"].items():
        add(g, "Electricity", "lightgbm")

    # SCP reference point: worst-group coverage on grade grouping, from e4_series
    z = np.load(RES / "e4_series.npz", allow_pickle=True)
    groups, scp_cov = z["groups"], z["scp_cov"]
    worst = min(scp_cov[groups == g].mean() for g in set(groups.tolist()))
    e4 = json.load(open(RES / "e4_metrics.json"))
    pts.append(("SCP", e4["scp"]["mean_width"], float(worst),
                "LendingClub", "ridge"))

    method_color = {"SCP": "#9e9e9e", "ACI": "#d62728",
                    "GACI": "#1f77b4", "GACI-pooled": "#ff7f0e"}
    method_marker = {"SCP": "x", "ACI": "o", "GACI": "o", "GACI-pooled": "s"}
    domain_ls = {"ridge": "-", "lightgbm": "--"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    seen = set()
    for m, w, c, dom, base in pts:
        key = (m, dom, base)
        ax.scatter(w, c, s=42,
                   color=method_color.get(m, "k"),
                   marker=method_marker.get(m, "o"),
                   facecolors="none" if m == "GACI-pooled" else method_color.get(m, "k"),
                   edgecolors=method_color.get(m, "k"),
                   linewidths=1.4,
                   linestyle=domain_ls[base],
                   alpha=0.9,
                   label=f"{m} / {dom} / {base}" if key not in seen else None)
        seen.add(key)
    ax.axhline(TARGET, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.99 if False else 0.98,
            TARGET + 0.006, "target 0.90", fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("mean prediction-interval width (log scale)")
    ax.set_ylabel("worst-group coverage")
    ax.set_ylim(0.1, 0.95)
    ax.set_title("Coverage–width frontier: group-conditional adaptation\n"
                 "(points = method × grouping × experiment; "
                 "SCP y = worst-group coverage on grade grouping)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_frontier.png", dpi=200)
    plt.close(fig)
    print(f"fig2 written ({len(pts)} points)")


if __name__ == "__main__":
    fig1()
    fig2()
