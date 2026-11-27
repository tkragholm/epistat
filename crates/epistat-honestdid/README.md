# epistat-honestdid

How much can parallel trends fail before an event study's conclusion
changes? The sensitivity analysis of Rambachan and Roth (2023), as R's
`HonestDiD` computes it: smoothness and relative-magnitude restrictions on
the violations of parallel trends, with sign and monotonicity variants; the
identified set, fixed-length and conditional confidence sets and their hybrids
for any linear combination of the post-treatment coefficients; the breakdown
value; and confidence regions for the whole post-treatment path.

Also the power of the pre-trends test and the distortion after passing it
(Roth, 2022): from event-study coefficients and their covariance and a
hypothesised violation of parallel trends, the probability that no
pre-period coefficient is significant, the means of the pre- and post-period
coefficients given that the test passes, the Bayes factor and the likelihood
ratio against parallel trends, and the slope of a linear violation that the
test detects with a given power. The rectangle probabilities are computed by
Genz's transformation on a randomly shifted Korobov lattice; every
probability comes with the standard error of its evaluation, and the results
are deterministic.

The entry points are `honest::original` (the interval under exact parallel
trends), `honest::summarize` (identified sets and robust intervals over a grid
of the restriction's parameter, with the breakdown value),
`honest::joint_path` (joint regions over the post-treatment periods, given
directions or the optimisation surface), `honest::smoothness_bound_from_pretrends`
and `pretrends::power`.

They take an `epistat_core::EventStudy`: the coefficients by period relative
to treatment, their covariance and the reference period, from any estimator.
epistat-did's `AttGt::event_study` and its fixed-effects estimators return
one.

The second-order cone programs are solved by
[Clarabel](https://crates.io/crates/clarabel), a pure Rust solver, until
epistat has its own; the linear programs run on epistat-optim's dense
simplex. An identified set that no violation path fits, as when the
pre-period coefficients have a second difference larger than a smoothness
bound `M`, is empty; the confidence sets are computed there as in HonestDiD.
Every simulation is seeded by the caller, and the results do not depend on
the number of threads.

The tests compare the results with `HonestDiD` 0.2.8 and `pretrends` 0.1.0
(from its authors' repository), from reference values written by the R
scripts in `tests/data`.

## License

MIT OR Apache-2.0.
