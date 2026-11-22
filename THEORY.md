# GACI — Theory Position and Honest Limitations

This document states precisely what our group-conditional ACI (GACI) does and
does not claim, so the manuscript never overstates. It is the theory section's
raw material.

## What is proven (inherited, not ours)

Gibbs & Candès (2021, arXiv:2106.00170) prove that ACI — online update
`α_{t+1} = α_t + γ(α − err_t)` driving the interval to the `(1−α_t)` quantile
of a rolling conformity-score window — achieves **marginal** coverage
converging to `1−α` under *arbitrary* distribution shift, with a regret bound
relative to the best fixed quantile in hindsight. DtACI (variant with decaying
step size) has analogous guarantees. We verified these empirically (E1–E3);
we claim no new theory for the marginal algorithm.

## What GACI is

Run one independent ACI per group `g ∈ G`, routing each incoming
`(x_t, g_t)` to `ACI_g`. Each group sees its own subsequence of the stream.

**Claim (provable by direct application of the ACI theorem):** if each
group-labeled subsequence satisfies the ACI assumptions, then *each group's
marginal coverage* converges to `1−α`. I.e., GACI delivers
`P(y_t ∈ C_t | g_t = g) → 1−α` for every `g` — **group-marginal coverage**.

This is strictly weaker than (and does not imply) **conditional coverage**
`P(y_t ∈ C_t | x_t) = 1−α`, which remains impossible under arbitrary shift
(Vovk 2012; Bates et al. 2023 impossibility results for distribution-free
conditional coverage). We must state this boundary explicitly — reviewers
will probe it.

## What our contribution actually is

Not a theorem — an **empirical characterization + algorithm**:

1. **The masking phenomenon**: marginal ACI can hold `P(cover)→0.90` while a
   subgroup sits at 0.23 (Lending Club grade G). Marginal validity is not
   subgroup validity. (E4, replicated E5)
2. **The trade-off frontier**: GACI buys per-group validity by
   reallocating interval width — mean width is essentially unchanged
   (0.107 vs 0.113) while the failing group's width roughly doubles
   (grade G: 0.114 → 0.208). Quantifying this redistribution is the
   contribution.
3. **Design insight**: per-group *score windows* matter, not just per-group
   α_t (pooled-window ablation: worst-group 0.869 vs 0.892). (E4)
4. **Boundary conditions** (E5): the gap is large only when groups differ in
   conditional difficulty (grade, purpose) and small/absent for easy splits
   (term, era). GACI is not a universal win — it is a tool for heterogeneous
   risk segments.

## Honest limitations (must appear in the paper)

- **Group labels must be available at prediction time.** GACI cannot discover
  latent subgroups; unknown-group arrivals cold-start their own per-group
  ACI from stream residuals (validated in E9 for one held-out grade;
  rare+hard and late-appearing regimes remain open).
- **Small groups adapt slowly.** Grade G (n≈566 stream rows) needed its own
  window; very small groups may never converge within the horizon. Our
  MIN_GROUP=200 seed threshold is a heuristic, not principled.
- **Group-marginal ≠ within-group conditional.** Two loans in grade G with
  very different `dti` still share one interval-width policy; residual
  heterogeneity inside groups is untouched. True conditional coverage under
  shift is impossible; we do not claim it.
- **Uniform α allocation.** Every group targets the same `1−α`. Whether
  unequal allocation (e.g., tighter coverage for high-stakes groups) is
  better is open — natural extension.
- **No new convergence proof.** The per-group guarantee follows from applying
  the ACI theorem to subsequences; we have not proven anything about the
  *joint* process (e.g., interaction between groups sharing the predictor).
- **Single predictor.** All experiments freeze a ridge regression; results
  may differ with a stronger/weaker base model (the conformity scores change,
  the mechanism does not).

## What would elevate this to a theory paper

- A regret bound for the *vector* of group coverages (simultaneous
  tracking), or
- An impossibility result showing marginal ACI *must* sacrifice some group
  under a stated heterogeneity condition — that would be a real theorem.
  Currently we only demonstrate it empirically.

## Positioning sentence (for abstract/intro)

> "We show that marginal coverage guarantees can conceal severe undercoverage
> in identifiable subgroups under distribution shift, quantify the width cost
> of restoring per-group validity with group-conditional adaptive conformal
> inference, and characterize when the trade-off is worth paying."

This claims exactly what E4–E6 show and nothing more.
