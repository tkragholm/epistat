# epistat-did

Difference-in-differences. Group-time average treatment effects ATT(g,t) for
staggered adoption (Callaway and Sant'Anna), by outcome regression, inverse
probability weighting or the doubly robust estimator, on panels and repeated
cross-sections, with never-treated or not-yet-treated comparison groups,
varying or universal base periods, anticipation and sampling weights. As
`att_gt` of R's `did`, they recode cohorts treated after the last period as
never treated, drop units treated at the start, use the last cohort as
comparison group when no unit is never treated, take base periods from the
observed periods, which need not be consecutive, and balance an unbalanced
panel or estimate it as repeated cross-sections; what they changed is
reported with the estimates. The effects aggregate by event time, cohort,
calendar time or overall, as `aggte` does.

Also the two-period doubly robust estimators of `DRDID`, including the
improved and locally efficient ones; the efficient ATT(g,t) that combines all
baseline periods (Chen, Sant'Anna and Xie); compositional changes in repeated
cross-sections (Sant'Anna and Xu); doubly robust triple differences; and the
average causal response to a continuous treatment by a sieve estimator, whose
standard error comes from the influence function of the whole procedure,
estimated control trend included.

The efficient ATT(g,t) on a panel without covariates uses the sampling
weights, which must be constant within a unit, and indexes every cell's
influence function by unit over the whole panel, so the cells of an
unbalanced panel line up. Its aggregations take their standard errors from the
combined influence function, which counts the covariance between cells that
share units.

Two-way fixed-effects regressions are there as the comparison readers
expect: the static regression and the event study with relative-period
indicators, a chosen reference period and binned end points, as
`fixest::feols(y ~ i(rel, ref = -1) | id + t)`, with `fixest`'s clustered
covariance and influence functions by unit. The unit and period effects are
absorbed by `epistat-glm`'s alternating projections, so the cost is linear in
the rows. The Goodman-Bacon decomposition splits the static estimate into its
2x2 comparisons and their weights as `bacondecomp::bacon` does, with the
within and between components when there are time-varying covariates. Both
are checked against R.

Inference uses influence functions: analytic and cluster-robust standard
errors, and multiplier-bootstrap simultaneous bands, as `did::mboot` does.
Every bootstrap replicate is seeded from the caller's seed, and results do not
depend on the number of threads. A fingerprint test pins the outputs of the
main estimators.

The `honest` feature adds the HonestDiD sensitivity analysis of Rambachan and
Roth: smoothness and relative-magnitude restrictions on violations of
parallel trends, fixed-length and conditional confidence sets, and regions for
the whole post-treatment path. Its second-order cone programs are solved by
[Clarabel](https://crates.io/crates/clarabel), a pure Rust solver and the
only dependency the feature adds; its linear programs run on the crate's own
dense simplex. An own cone solver may replace Clarabel later. An identified
set that no violation path fits, as when the pre-period coefficients have a
second difference larger than a smoothness bound `M`, is empty and reported
as NaN; the confidence sets are computed there as in HonestDiD. The `nightly`
feature enables faer's AVX-512 kernels through epistat-linalg.

The tests compare estimates, standard errors and influence functions with
`did` 2.5.1, `DRDID` 1.3.0, `HonestDiD` 0.2.8, `triplediff` and `contdid`
0.1.2, the continuous treatment's influence function with the
infinitesimal jackknife of refits in R, and the efficient ATT(g,t) and its
aggregations with the estimator written out in R and with the jackknife of the
whole procedure, from reference values written by the R scripts in `tests/`. They run with
`cargo test --features parity-tests`.
