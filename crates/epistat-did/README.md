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

The package's other options are there too: group-specific linear trends
(`trends_lin`), a continuous baseline treatment through a polynomial in it
(`continuous`), effects over the same switchers at every horizon
(`same_switchers`, `same_switchers_pl`), standard errors that demean the
switchers within their treatment paths (`less_conservative_se`), the weights
of the normalised effects on the treatment's lags and the average number of
periods over which a dose accumulates. Separate functions give the
regressions of the effects on group-level covariates (`predict_het`, with HC2
standard errors), the estimates by subgroup and by treatment path, the
switchers' treatment paths and first-switch dates, and a bootstrap whose
replicates are seeded one by one, so its result does not depend on the
number of threads.

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
absorbed by `epistat-glm`'s fixed-effects solver, so the cost is linear in
the rows. The Goodman-Bacon decomposition splits the static estimate into its
2x2 comparisons and their weights as `bacondecomp::bacon` does, with the
within and between components when there are time-varying covariates. Both
are checked against R.

The interaction-weighted event study of Sun and Abraham fits one coefficient
per cohort and relative period, with the never treated, cohorts not treated
in the panel, named cohorts or the last treated cohort as controls, the
reference periods and the binning of `fixest::sunab`, and averages the
coefficients by their shares of the rows into effects by relative period, by
cohort and overall, with `fixest::aggregate`'s clustered standard errors and
influence functions by unit. It is fitted by the same absorbed regression, so
the cost is linear in the rows. It differs from `fixest` 0.14.2 where that
classifies cohorts by an order-dependent scan that can drop rows and keep
always-treated units as controls; here the always treated are removed. It is
checked against `fixest` 0.14.2.

Wooldridge's extended two-way fixed-effects regression saturates the treated
cohort and period cells, each with its own slopes on the covariates demeaned
within the cohort, beside cohort and period slopes of the covariates, with
cohort or unit effects and not-yet-treated or never-treated controls, as
`etwfe::etwfe` 0.6.2 fits it. The average effects of `etwfe::emfx`, overall,
by cohort, by period and by event time, come with their delta-method standard
errors from the exact Jacobian, where `marginaleffects` uses forward
differences, and with influence functions by unit. The fit is the same
absorbed regression, so the cost is linear in the rows. The Poisson and logit
models of `etwfe(family = "poisson" / "logit")` are fitted as `fixest::feglm`
fits them, with an intercept and cohort and period dummies in place of the
absorbed effects, by `epistat-glm`'s iteratively reweighted least squares on
rows formed block by block, so no design matrix is held. Their average effects
are means of differences of predicted means, with the delta method on the
exact Jacobian of the mean, the iid, robust or clustered covariance of
`fixest` and influence functions by unit.

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

The imputation estimator of Borusyak, Jaravel and Spiess and the two-stage
estimator of Gardner fit unit and period effects and covariates on the
untreated rows, impute the untreated outcomes of the treated rows and average
the imputed effects, overall, by horizon or with the caller's weights, with
the pre-trend regression of the imputation estimator. The effects are
absorbed by the fixed-effects solver of `epistat-glm`, and the variances need
fitted values of the first stage's auxiliary systems, which the same
solver finds from their level sums, so the cost is linear
in the rows and no dummies are formed. They
follow `didimputation` 0.5.1 and `did2s` 1.2.1, except that a row whose unit
and period effects are not jointly identified is dropped rather than imputed,
and that with varying prior weights the imputation variance is the derivative
of the estimate, which the package's is not.

When the pre-treatment outcome is missing for some units, missing at random
given the covariates and treatment, or given those and the post-treatment
outcome, `att_missing_pre` estimates the two-period ATT by the efficient
estimators of Testa, Kennedy and Reimherr, including the nested regression
that the outcome-dependent case needs. It fits logistic models for the
propensity and the probability of an observed baseline and least squares for
the outcome regressions, on all units or cross-fitted over folds dealt from a
seed (by cluster when there are clusters). `att_missing_pre_from_predictions`
takes out-of-fold predictions from any learner instead, which is how random
forests or other flexible models are used. Probabilities near 0 or 1 are
reported and can be refused at a floor of the caller's; they are not clipped.
With every baseline observed the estimator is the panel doubly robust one with
the control weights divided by the number treated rather than by their sum.
It is checked against the authors' Python code on their simulation design,
with correctly and wrongly specified models.

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

The `pretrends` module gives the power of the pre-trends test and the
distortion after passing it (Roth, 2022). From event-study coefficients and
their covariance, from any estimator, and a hypothesised violation of parallel
trends, it returns the probability that no pre-period coefficient is
significant, the means of the pre- and post-period coefficients given that the
test passes, the Bayes factor and the likelihood ratio against parallel
trends, and the slope of a linear violation that the test detects with a given
power. The rectangle probabilities are computed by Genz's transformation on a
fixed, randomly shifted Korobov lattice; every probability comes with the
standard error of its evaluation, and the results are deterministic. The cost
depends on the number of pre-period coefficients, not on the number of rows.

The tests compare estimates, standard errors and influence functions with
`did` 2.5.1, `DRDID` 1.3.0, `HonestDiD` 0.2.8, `pretrends` 0.1.0 (from its authors' repository), `triplediff`, `contdid`
0.1.2, `DIDmultiplegtDYN` 2.2.0, `synthdid` 0.0.9 (from its authors' repository, as it is not on CRAN), `didimputation` 0.5.1 and `did2s` 1.2.1, the continuous treatment's influence function with the
infinitesimal jackknife of refits in R, and the efficient ATT(g,t) and its
aggregations with the estimator written out in R and with the jackknife of the
whole procedure, from reference values written by the R scripts in `tests/`;
the estimator for a missing baseline is compared with the Python replication
code of Testa, Kennedy and Reimherr through a Python script there. They run with
`cargo test --features parity-tests`.
