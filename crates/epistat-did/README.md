# epistat-did

Difference-in-differences. Group-time average treatment effects ATT(g,t) for
staggered adoption (Callaway and Sant'Anna), by outcome regression, inverse
probability weighting or the doubly robust estimator, on panels and repeated
cross-sections, with never-treated or not-yet-treated comparison groups,
varying or universal base periods, anticipation and sampling weights. The
effects aggregate by event time, cohort, calendar time or overall, as `aggte`
of R's `did` does.

Also the two-period doubly robust estimators of `DRDID`, including the
improved and locally efficient ones; the efficient ATT(g,t) that combines all
baseline periods (Chen, Sant'Anna and Xie); compositional changes in repeated
cross-sections (Sant'Anna and Xu); doubly robust triple differences; and the
average causal response to a continuous treatment by a sieve estimator, whose
standard error comes from the influence function of the whole procedure,
estimated control trend included.

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
0.1.2, and the continuous treatment's influence function with the
infinitesimal jackknife of refits in R, from reference values written by the
R scripts in `tests/`. They run with
`cargo test --features parity-tests`.
