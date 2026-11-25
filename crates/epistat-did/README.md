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

For treatments that go on and off or take several values, such as benefit
spells, `estimate_dcdh` is the event-study estimator of de Chaisemartin and
D'Haultfoeuille as `did_multiplegt_dyn` of R's `DIDmultiplegtDYN` computes
it: the effects of a group's first change of treatment against groups with
the same baseline treatment that have not changed yet, for switchers in and
out, with placebos, the average total effect, normalised effects, controls,
comparisons within strata, weights and clusters, and the package's
influence-function standard errors and joint tests. It follows the rules of
2.4.0 and is tested against 2.2.0, the last release that runs without the R
package polars; the module documentation lists where the two differ and
where this crate departs from both. The work is linear in the cells of the
panel.

The efficient ATT(g,t) on a panel without covariates uses the sampling
weights, which must be constant within a unit, and indexes every cell's
influence function by unit over the whole panel, so the cells of an
unbalanced panel line up. Its aggregations take their standard errors from the
combined influence function, which counts the covariance between cells that
share units.

Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens
and Wager), synthetic control and the plain difference-in-differences
estimator reproduce R's `synthdid` 0.0.9: the unit and time weights by its
Frank-Wolfe solver (in `epistat-optim`) with the penalties it derives from
the noise level, its sparsification and stopping rules, covariates as it
adjusts for them, and its placebo, bootstrap and jackknife standard errors,
with a standard error for each post period. The jackknife without covariates
is computed in closed form, in time linear in the panel. `synthdid` takes a
single adoption date; staggered adoption is estimated cohort by cohort
against the never-treated units and aggregated by treated cells, with the
standard errors replaying the whole procedure.

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
`did` 2.5.1, `DRDID` 1.3.0, `HonestDiD` 0.2.8, `triplediff`, `contdid`
0.1.2, `DIDmultiplegtDYN` 2.2.0 and `synthdid` 0.0.9 (from its authors' repository, as it is not on CRAN), the continuous treatment's influence function with the
infinitesimal jackknife of refits in R, and the efficient ATT(g,t) and its
aggregations with the estimator written out in R and with the jackknife of the
whole procedure, from reference values written by the R scripts in `tests/`. They run with
`cargo test --features parity-tests`.
