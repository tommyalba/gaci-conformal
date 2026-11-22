# Marginal Coverage Can Hide Subgroup Failure: Group-Conditional Adaptive Conformal Inference Under Distribution Shift

**Status:** workshop draft v2 (2026-11-23). All numbers trace to
`results/*.json`; figures in `results/figures/`. AI-assistance disclosure
in Acknowledgments and, for the TMLR version, a first-page footnote.

---

## Abstract

Adaptive conformal inference (ACI) maintains marginal coverage near a
target level under arbitrary distribution shift, which makes it the
default choice for online uncertainty quantification. Marginal coverage,
however, is an average: we show it can conceal severe and stable
undercoverage in identifiable subgroups. On a decade of Lending Club loan
data, marginal ACI holds 90% overall coverage while covering grade-G
loans, the riskiest segment, only 23% of the time. We propose GACI
(group-conditional ACI), which runs one adaptive conformal procedure per
group and restores per-group coverage to target at essentially unchanged
mean interval width, reallocating width toward the groups that need it.
Across two domains (credit-risk tabular drift and
calendar-heterogeneous electricity demand), the phenomenon replicates,
the fix works exactly when groups differ in conditional difficulty, and
fails gracefully, with no coverage gain, when they do not. The practical
takeaway is a diagnostic recipe: measure per-group coverage before
trusting marginal coverage, and pay for group-conditioning only where
coverage diverges. Code, data
slices, and per-step coverage series are publicly available at
github.com/tommyalba/gaci-conformal.

## 1. Introduction

Conformal prediction wraps any point predictor in a prediction interval
with a distribution-free coverage guarantee: if calibration and test data
are exchangeable, the interval covers the true outcome with probability
at least `1 − α`. Adaptive conformal inference (ACI; Gibbs & Candès 2021)
extends this to *non-exchangeable* streams by adjusting the miscoverage
level `α_t` online, keeping marginal coverage near target under arbitrary
distribution shift. ACI and its variants are now standard components of
deployed forecasting and risk systems.

The guarantee, however, is **marginal**: `P(y_t ∈ C_t) ≈ 1 − α` averaged
over all arrivals. Nothing in the guarantee constrains coverage *within*
subgroups. This matters because the groups practitioners care about,
risk grades, customer segments, calendar regimes, are exactly where
conditional difficulty concentrates. A credit-risk interval that covers
98% of grade-A loans and 23% of grade-G loans still reports "90% coverage"
to anyone reading the marginal number.

This paper asks a deliberately narrow question: **when marginal adaptive
coverage hides subgroup failure, can per-group adaptation fix it, and
what does the fix cost?** Our contributions:

1. **The masking phenomenon, demonstrated.** On Lending Club data
   (2007–2014, 200k loans, ridge forecaster frozen before the stream),
   marginal ACI holds 0.900 coverage overall while grade-G coverage sits
   at 0.230, stable across ten stream blocks (0.15–0.31), not a
   single-regime artifact. Split conformal is worse still (0.161 on
   grade G). The same pattern replicates on UCI household electricity
   demand under calendar groupings.

2. **GACI, a simple fix that reallocates width, not multiplies it.**
   Running one ACI per group, each with its own conformity-score window,
   restores worst-group coverage to 0.892 (target 0.90) at essentially
   unchanged mean width (0.107 vs marginal ACI's 0.113); the cost is
   redistributive, grade G's mean width roughly doubles (0.114 → 0.208)
   while easy grades narrow. On electricity, hour-band worst-group
   coverage goes 0.865 → 0.900 at unchanged mean width (3.12 → 3.14).

3. **Boundary conditions, characterized honestly.** The gap is large
   only when groups differ in *conditional difficulty* (loan grade,
   loan purpose). For easy splits, loan term, calendar era, calendar
   month, marginal ACI is already uniform and GACI buys nothing.
   We report these negative results as first-class findings.

4. **Robustness evidence.** The phenomenon and fix replicate across
   stream subsamples (150k/250k rows), a 27-point hyperparameter grid,
   and a stronger LightGBM base model, plus an unseen-group holdout
   (E9), addressing the most likely reviewer objections (subsample luck,
   weak predictor, cold-start groups) before submission.

Our scope is honest: we claim an empirical characterization and a simple
algorithm, not a theorem. Per-group marginal coverage follows directly
from applying the ACI guarantee to each group's subsequence; conditional
coverage given `x` remains impossible under arbitrary shift, and we do
not claim it. The practical message is deliberately small: **measure
per-group coverage before trusting marginal coverage**; adopt
GACI only where the two diverge.

## 2. Related Work

**Adaptive conformal inference.** Vovk et al. introduced inductive
conformal prediction; Gibbs & Candès (2021) introduced ACI, proving
marginal coverage under arbitrary shift via a regret bound against the
best fixed quantile in hindsight. DtACI adds a decaying step size;
follow-ups address delayed feedback (arXiv:2609.07251) and multi-step
forecasting (arXiv:2409.14792). We use ACI as our marginal baseline and
claim no new theory for it.

**Group- and fairness-aware conformal prediction.** Several works target
coverage *parity* across sensitive groups: AFCP (arXiv:2405.15106)
adaptively selects the sensitive attribute in batch settings; C3F
(arXiv:2509.25295) enforces coverage parity post-hoc under covariate
shift via importance weighting; the ICLR 2025 Conformal Fairness
framework controls coverage gaps between groups under exchangeability;
arXiv:2506.05583 adapts to *unknown* subpopulations without labels; MAPIE
pursues feature-conditional coverage at per-point cost. **None maintains
per-group miscoverage levels `α_{t,g}` online under temporal drift with
group labels observed at prediction time**; that is our delta. MACI
(arXiv:2602.01285) uses grouping as a component for multi-LLM settings,
not as the problem.

**Conformal prediction for time series.** Stocker et al. (arXiv:2511.13608)
unify non-exchangeable CP, calibration reweighting, dynamic residuals,
adaptive coverage, with guarantees under weak dependence; our
electricity experiment sits in their adaptive-coverage category.
Sabashvili (arXiv:2601.18509) benchmarks the same method families on
monthly sales data.

**Impossibility results.** Vovk (2012) and Bates et al. (2023) show
distribution-free conditional coverage is impossible in general; this is
why we claim group-marginal, not `x`-conditional, coverage.

## 3. Setup and Method

**Stream protocol.** Data arrive as `(x_t, y_t, g_t)`: features, outcome,
and a group label known at prediction time. A point predictor `ŷ_t =
f(x_t)` is frozen before the stream begins, all adaptation happens in
the conformal layer, not the model.

**Conformity scores and intervals.** With absolute-error scores
`s_t = |y_t − ŷ_t|`, a conformal method maintains a window of recent
scores and emits the interval
`C_t = [ŷ_t − q_{1−α_t}(W_t), ŷ_t + q_{1−α_t}(W_t)]`,
where `q_{1−α_t}` is the empirical `(1−α_t)` quantile of window `W_t`.

**Baselines.** *Split conformal (SCP)* calibrates once on a held-out
calibration set and emits fixed-width intervals. *ACI* updates
`α_{t+1} = α_t + γ(α − err_t)` online (`err_t = 1` if uncovered),
tracking marginal coverage under shift. We use `α = 0.10`,
`γ = 0.01`, window 5000 (Lending Club) / 2000 (electricity).

**GACI (ours).** Run one independent ACI per group `g`: each group sees
only its own subsequence of the stream, maintains its own score window
`W_{t,g}` seeded from calibration rows of that group, and updates its own
`α_{t,g}`. Arrivals route to `ACI_g` by their label; groups below a
minimum seed count (MIN_GROUP = 200) or unseen at test time cold-start
their own ACI from stream residuals, a fallback we quantify in E9. A
*pooled*
variant shares one score window across groups while keeping per-group
`α_t`, our ablation isolates whether per-group windows (not just
per-group step sizes) drive the gains.

**What is guaranteed.** If each group-labeled subsequence satisfies the
ACI assumptions, each group's marginal coverage converges to `1 − α`:
`P(y_t ∈ C_t | g_t = g) → 1 − α`. This is an immediate corollary of the
ACI theorem applied per subsequence, we claim the *phenomenon* and the
*algorithmic fix*, not new theory.

## 4. Experimental Setup

**Domains.** (i) *Lending Club* loan data 2007–2014: 200k-row
stratified slice, ridge regression (λ=10) on 42 application-time features
predicting `int_rate`; groups = loan grade (A–G, the platform's own risk
tiers), purpose, term, home ownership, issue year, stream era.
(ii) *UCI Household Electric Power*: 204,806 ten-minute measurements
(2006–2010), ridge on lag + calendar features; groups = hour band
(night/morning/afternoon/evening), day type (weekday/weekend), month.

**Protocol.** Chronological splits per domain: Lending Club 55% train /
10% calibration / 35% stream; electricity 60% / 10% / 30% (its shorter
stream still gives each calendar group thousands of arrivals). The
predictor is frozen before the stream; all methods see the same stream
in the same order. Numeric features are standardized using
training-partition statistics only (imputation medians and feature
means/stds fit on train, then applied to calibration and stream), so no
calibration or stream information leaks into preprocessing.
Grade G has only ~161 calibration rows, below MIN_GROUP = 200, so it
cold-starts its own ACI in *every* experiment, E9 (§5.5) validates that
fallback explicitly rather than hypothetically. Metrics: marginal
coverage, per-group and worst-group coverage, mean interval width,
Winkler score, coverage-gap regret, and 10-block worst-group spread.

**Sanity checks (E1–E2).** E1 reproduces the ACI paper's claim on
synthetic regime shift (SCP collapses to 0.41 coverage, ACI holds 0.90).
E2 confirms marginal ACI is already calibrated on electricity (SCP 0.900)
, the interesting failures there are group-level, motivating E6.

## 5. Results

**The masking phenomenon (E3–E4).** On Lending Club, split conformal
under-covers marginally (0.865 vs target 0.90) while ACI restores
marginal coverage at a modest width premium (0.113 vs SCP's 0.103), the
adaptive baseline is strictly better here. But marginal validity hides subgroup
failure: ACI's per-grade coverage ranges from 0.972 (grade C) down to
**0.230 (grade G)** and 0.374 (grade F), the riskiest loans are covered
least. Ten-block analysis shows grade-G worst-group coverage stable at
0.15–0.31 across the stream; this is structural, not transient.

**GACI restores per-group validity (E4–E5).** Per-grade GACI lifts
worst-group coverage 0.230 → **0.892** and compresses the per-grade
range to 0.89–0.90, at essentially unchanged mean width (0.107 vs ACI's
0.113; block-bootstrap 95% CIs overlap). The width cost is
redistributive: grade G's mean width rises 0.114 → 0.208 (~1.8×) while
easy grades narrow (grade C: 0.113 → 0.085). Across
six groupings the pattern is selective, exactly as the mechanism
predicts:

| Grouping | ACI worst-group | GACI worst-group | Verdict |
|---|---|---|---|
| grade | 0.230 | 0.892 | large gap, large fix |
| purpose | 0.773 | 0.899 | moderate gap, fixed |
| term | 0.882 | 0.900 | small gap, ~free |
| home_ownership | 0.885 | 0.899 | small gap, ~free |
| issue_year | 0.900 | 0.900 | no gap, no benefit |
| stream_era | 0.900 | 0.900 | no gap, no benefit |

The pooled-window ablation (shared scores, per-group `α_t`) reaches only
0.869 worst-group on grade, **per-group score windows matter**, not
just per-group learning rates.

**Second domain (E6).** On electricity demand, marginal coverage is
already fine; the action is again group-level. Hour-band grouping shows
the same masking: ACI covers morning (0.865) and evening (0.868) poorly
while over-covering night (0.956), marginal ACI *redistributes* coverage
across day-parts. GACI equalizes all four bands to ≈0.900 at unchanged
mean width (3.12 → 3.14; per-band widths diverge, evening widest).
Day-type shows a small gap (0.894 → 0.900). **Month shows
no gap at all** (0.899 → 0.899): when base calibration is already uniform
across groups, group-conditioning buys nothing, our honest
negative result.

**Robustness (E7–E8).** E4 subsample variance: ACI worst-group 0.238 /
0.260 at 150k/250k rows (baseline 0.230); GACI 0.889 / 0.896 (baseline
0.892), the gap is not a subsample artifact. A 27-point grid over
`(α, γ, window)`, run for *both* GACI and marginal ACI on the same
subsampled stream, shows coverage tracking `α` almost exactly and
worst-group coverage varying ≤0.04 at fixed `α` for GACI while marginal
ACI stays at 0.43–0.47 (α=0.05) / ~0.23–0.26 (α=0.10) across the whole
grid, the gap is not a hyperparameter artifact of either method. E8 swaps the ridge predictor for LightGBM:
E4 worst-group goes 0.300 (ACI) → 0.890 (GACI); E6 hour-band 0.862 →
0.900, **the phenomenon and fix survive a substantially stronger base
model**.

**Unseen groups (E9).** We hold grade G out of *all* calibration data so
it is genuinely unseen, then compare fallback strategies. Per-group
cold-start ACI (each unseen group accumulates its own score window from
stream residuals) covers G at **0.892**, statistically indistinguishable
from the seeded 0.892, with 0.885 coverage in its first 200 arrivals;
routing unseen groups to the marginal ACI instead gives 0.869. The
reason is structural: G's own residuals are large, so its personal score
window produces appropriately wide intervals almost immediately, while
the marginal window, dominated by easy grades, stays too narrow. In
practice grade G was already cold-starting in E4–E8 (its calibration
count is below MIN_GROUP), so E9 validates the fallback the main
results already relied on rather than a hypothetical.

**Simpler alternatives (E10).** Two cheaper fixes fail on grade G.
Per-group *static* split conformal (each group's own calibration
quantile, marginal fallback below MIN_GROUP) reaches only 0.157
coverage on G: without online adaptation, static intervals cannot
track drift. A single marginal ACI run on *group-normalized* conformity
scores (each group's calibration residual scale divided out) improves
G coverage to 0.348, better than plain ACI's 0.230 but far short of
GACI's 0.892: normalization shrinks the scale mismatch but cannot
recalibrate each group's miscoverage level independently. We did not
implement importance-weighted split conformal: under temporal drift
within groups it requires a covariate-shift weight model we do not
have, and the label-only alternatives above are the relevant comparison
for our setting.

## 6. Discussion

**When GACI pays.** When (i) group labels are available at prediction
time, (ii) groups differ in conditional difficulty, and (iii) the cost of
within-group undercoverage exceeds the redistributive width cost (width
concentrates on the groups that need it; mean width is roughly
unchanged). Credit-risk
intervals that silently fail on the riskiest segment are precisely the
failure mode fairness-aware uncertainty quantification exists to catch.

**When it doesn't.** When groups are already uniformly calibrated, as
with easy splits or homogeneous domains, GACI adds per-group width
variation for zero coverage gain. Practitioners should diagnose
per-group coverage *before* adopting it; our E5 table is exactly that
diagnostic.

**Limitations.** Group labels must be observed at prediction time; GACI
cannot discover latent subgroups. Unseen groups cold-start per-group
(quantified in E9 for one held-out grade; other unseen regimes, e.g.
groups that are both rare and hard, remain open). Small groups adapt
slowly, grade G's ~566 stream
observations barely fill a window; MIN_GROUP=200 is a heuristic. We
guarantee group-marginal, not `x`-conditional, coverage; residual
heterogeneity *within* groups is untouched. Width/coverage comparisons
carry block-bootstrap 95% CIs (`results/ci_metrics.json`: 10 contiguous
stream blocks, 20k resamples, fixed seed), which quantify uncertainty
over stream/block resampling, not coverage guarantees; the headline
gap (0.230 vs 0.892) is far outside them. We compare against per-group
static SCP and group-normalized-score marginal ACI (E10); both fail,
but importance-weighted split conformal and temporal CP baselines on
electricity remain untested. Our sensitivity grid now covers both GACI
and marginal ACI hyperparameters. All
groups share one `α`;
unequal allocation across groups is future work. Our per-group guarantee
is a corollary of ACI applied to subsequences; we prove nothing about
interactions between groups sharing the predictor. Experiments use a
single frozen predictor per domain (ridge, plus a LightGBM robustness
check); the conformity scores change with the predictor, the mechanism
does not.

## 7. Conclusion

Marginal adaptive coverage is not subgroup coverage. We showed that ACI
can hold 90% overall while covering the riskiest credit segment 23% of
the time. Per-group adaptive conformal inference restores
per-group validity by reallocating interval width toward the groups
that need it, at essentially unchanged mean width, and, just as
importantly, the fix is worthless when groups are already uniform. The practical recipe has two steps: (1) measure per-group coverage
alongside marginal coverage; (2) if per-group coverage diverges, pay
the redistributive width cost of GACI (hard groups get wider intervals,
easy groups narrower). If it does not diverge, keep plain
ACI. Code, data slices, per-step coverage series, and figure-generation
scripts are publicly available at
github.com/tommyalba/gaci-conformal.

---

## Acknowledgments

**Use of AI assistance:** this work used Devin (an AI software-engineering
agent by Cognition) as a research assistant for literature organization,
experiment scripting, figure generation, and manuscript drafting under
the author's direction. All experimental results were produced by code in
this repository; the author verified claims against `results/` artifacts
and takes full responsibility for the content.

## References

- Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift" (arXiv:2106.00170), NeurIPS 2021.
- Vovk, "Conditional validity of inductive conformal predictors" (2012).
- Bates et al., "Distribution-Free Conditional Predictive Inference" (2023).
- AFCP, arXiv:2405.15106; C3F, arXiv:2509.25295; Conformal Fairness (ICLR 2025); unknown-subpopulation CP, arXiv:2506.05583; group-weighted CP, arXiv:2401.17452; Wasserstein-CP, arXiv:2501.13430; MAPIE conditional CP.
- MACI multi-LLM, arXiv:2602.01285; delayed feedback, arXiv:2609.07251; multi-step forecasting, arXiv:2409.14792.
- Stocker, Małgorzewicz, Fontana & Ben Taieb, "A Gentle Introduction to Conformal Time Series Forecasting" (arXiv:2511.13608).
- Sabashvili, "Conformal Prediction Algorithms for Time Series Forecasting" (arXiv:2601.18509).

## Appendix (workshop version: compressed)

- **A.1 Protocol details**: exact split sizes, Lending Club 110,000
  train / 20,000 calibration / 70,000 stream; electricity 122,883 /
  20,480 / 61,443; feature lists; deterministic single run (see
  Limitations; E2–E9 use no RNG).
- **A.2 Full E5 table**: per-group coverage and width for all six
  groupings × three methods.
- **A.3 Sensitivity grid**: 27-run summary statistics for both GACI
  and marginal ACI.
- **A.4 Reproduction**: `conda run -n credit-risk_VENV python
  experiments/e{1..10}_*.py` plus `experiments/ci_analysis.py`;
  figures via `experiments/figures.py`.
