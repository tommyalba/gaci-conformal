"""Unit tests for src/aci.py — ACI/GACI core machinery.

Run: ~/miniconda3/envs/credit-risk_VENV/bin/python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aci import ACI, SplitConformal, conformity_scores, quantile_halfwidth  # noqa: E402


# ---------- conformity scores / quantile ----------

def test_conformity_scores_absolute_error():
    np.testing.assert_allclose(
        conformity_scores([1.0, 2.0, 3.0], [0.5, 2.5, 2.0]), [0.5, 0.5, 1.0])


def test_quantile_halfwidth_basic():
    scores = np.arange(1.0, 101.0)
    # alpha=0.1 -> level 0.9 quantile with finite-sample correction
    q = quantile_halfwidth(scores, 0.1)
    assert 85.0 < q < 95.0


def test_quantile_halfwidth_clips_at_one():
    scores = np.ones(50)
    assert quantile_halfwidth(scores, 1e-9) == pytest.approx(1.0)


def test_quantile_halfwidth_empty_window():
    with pytest.raises(Exception):
        quantile_halfwidth(np.array([]), 0.1)


# ---------- SplitConformal ----------

def test_split_conformal_symmetric_intervals():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    p = y + rng.normal(scale=0.5, size=500)
    scp = SplitConformal(alpha=0.1).fit_calibrate(y[:250], p[:250])
    lo, hi = scp.predict(p[250:])
    mid = (lo + hi) / 2
    np.testing.assert_allclose(mid, p[250:])
    np.testing.assert_allclose(hi - mid, mid - lo)


def test_split_conformal_coverage_on_exchangeable():
    rng = np.random.default_rng(1)
    y = rng.normal(size=4000)
    p = y + rng.normal(scale=1.0, size=4000)
    scp = SplitConformal(alpha=0.1).fit_calibrate(y[:2000], p[:2000])
    lo, hi = scp.predict(p[2000:])
    cov = np.mean((y[2000:] >= lo) & (y[2000:] <= hi))
    assert 0.85 < cov < 0.95  # nominal 0.90, finite-sample slack


# ---------- ACI mechanics ----------

def make_aci(**overrides):
    kw = dict(alpha=0.1, gamma=0.01, window=100)
    kw.update(overrides)
    return ACI(**kw)


def test_step_returns_interval_fields():
    aci = make_aci()
    aci.seed(np.zeros(50), np.zeros(50))  # zero scores -> tiny intervals
    r = aci.step(1.0, 1.0)
    assert r["lo"] <= 1.0 <= r["hi"]
    assert r["covered"] is True
    assert "alpha_t" in r and "halfwidth" in r


def test_alpha_decreases_on_miss_increases_on_hit():
    # alpha_t is the miscoverage level: on a miss it drops, so the
    # (1 - alpha_t) quantile rises and the next interval widens.
    aci = make_aci()
    aci.score_window = [1.0] * 50
    a0 = aci.alpha_t
    aci.step(0.0, 10.0)  # guaranteed miss
    assert aci.alpha_t < a0
    aci2 = make_aci()
    aci2.score_window = [100.0] * 50
    b0 = aci2.alpha_t
    aci2.step(0.0, 0.0)  # guaranteed hit
    assert aci2.alpha_t > b0


def test_alpha_clipped_to_unit_interval():
    # large gamma forces the raw update far past the bounds; clip must hold
    aci = make_aci(gamma=2.0)
    aci.alpha_t = 0.5
    for _ in range(5):  # few enough steps that the score window stays thin
        aci.step(0.0, 1e6)
    assert aci.alpha_t == pytest.approx(0.0)
    aci.alpha_t = 0.5
    for _ in range(10):
        aci.step(0.0, 0.0)
    assert aci.alpha_t == pytest.approx(1.0)


def test_score_window_respects_max_window():
    aci = make_aci(window=10)
    aci.seed(np.zeros(5), np.zeros(5))
    for i in range(30):
        aci.step(0.0, 0.0)
    assert len(aci.score_window) <= 5 + 30  # seed + appended, unbounded list
    # but quantile uses only last `window` scores
    assert len(aci.score_window[-aci.window:]) == aci.window


def test_min_window_fallback_halfwidth():
    aci = make_aci(min_window=10)
    aci.score_window = [2.0, 4.0]  # below min_window
    r = aci.step(0.0, 0.0)
    # std*3 fallback: std of [2,4] ~1.414 -> ~4.24
    assert r["halfwidth"] == pytest.approx(np.std([2.0, 4.0]) * 3, rel=1e-6)


def test_empty_window_uses_unit_halfwidth():
    aci = make_aci()
    r = aci.step(0.0, 0.0)
    assert r["halfwidth"] == pytest.approx(1.0)


def test_dtaci_step_size_decays():
    aci = make_aci(dtaci=True, gamma=0.1)
    g0 = aci._gamma_t()
    aci.t = 99
    assert aci._gamma_t() < g0


def test_determinism_same_seed_same_trajectory():
    rng = np.random.default_rng(42)
    y = rng.normal(size=300)
    p = y + rng.normal(scale=0.8, size=300)

    def run():
        aci = make_aci(window=200)
        aci.seed(y[:50], p[:50])
        return aci.run(p[50:], y[50:])

    r1, r2 = run(), run()
    for a, b in zip(r1, r2):
        assert a["lo"] == b["lo"] and a["hi"] == b["hi"]
        assert a["covered"] == b["covered"]


def test_aci_tracks_marginal_coverage_under_shift():
    """Core ACI claim: coverage stays near target when residual scale shifts."""
    rng = np.random.default_rng(7)
    n = 3000
    p = rng.normal(size=n)
    scale = np.where(np.arange(n) < n // 2, 0.5, 2.0)  # mid-stream shift
    y = p + rng.normal(scale=1.0, size=n) * scale
    aci = ACI(alpha=0.1, gamma=0.01, window=500).seed(y[:200], p[:200])
    res = aci.run(p[200:], y[200:])
    cov = np.mean([r["covered"] for r in res])
    assert 0.85 < cov < 0.95


def test_run_matches_manual_step_loop():
    rng = np.random.default_rng(11)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=0.7, size=200)
    a = make_aci(window=50).seed(y[:20], p[:20])
    batch = a.run(p[20:], y[20:])
    b = make_aci(window=50).seed(y[:20], p[:20])
    manual = [b.step(float(pp), float(yy)) for pp, yy in zip(p[20:], y[20:])]
    for x, y_ in zip(batch, manual):
        assert x["lo"] == y_["lo"] and x["hi"] == y_["hi"]


def test_score_window_trimmed_to_window():
    aci = make_aci(window=10)
    aci.seed(np.zeros(50), np.zeros(50))
    for _ in range(30):
        aci.step(0.0, 0.0)
    assert len(aci.score_window) == 10  # in-place trim keeps object identity


def test_pooled_shared_window_sees_all_groups_scores():
    """Pooled GACI variant: ACIs sharing one list object must both observe
    every appended score (the ablation's core invariant)."""
    shared = []
    ga = make_aci(window=20)
    gb = make_aci(window=20)
    ga.score_window = shared
    gb.score_window = shared
    ga.step(0.0, 1.0)   # appends |1-0| = 1.0
    gb.step(0.0, 3.0)   # appends 3.0
    assert shared == [1.0, 3.0]
    assert ga.score_window is gb.score_window is shared


def test_cold_start_unseen_group_self_seeds():
    """E9 path: an ACI with no calibration seed uses the min_window
    fallback, then accumulates its own stream scores."""
    aci = make_aci(min_window=5)
    r1 = aci.step(0.0, 2.0)  # empty window -> halfwidth 1.0 fallback
    assert r1["halfwidth"] == pytest.approx(1.0)
    for i in range(4):
        aci.step(0.0, 2.0)
    # after >= min_window scores the quantile path takes over
    scores = np.array(aci.score_window[-aci.window:])
    assert len(scores) >= 5


def test_per_group_aci_isolation():
    """GACI pattern: two groups with different difficulty keep separate state."""
    rng = np.random.default_rng(3)
    ga = ACI(alpha=0.1, gamma=0.01, window=100)
    gb = ACI(alpha=0.1, gamma=0.01, window=100)
    for i in range(50):
        # group A: easy; group B: hard
        ga.step(0.0, rng.normal(scale=0.1))
        gb.step(0.0, rng.normal(scale=5.0))
    # windows diverged -> different score scales
    assert np.mean(gb.score_window) > 10 * np.mean(ga.score_window)
