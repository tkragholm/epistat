# epistat-survival

Survival analysis for follow-up data in R's vocabulary, checked against `survival`,
`mstate`, `cmprsk`, `riskRegression` and other R packages.

- Cox regression on a design matrix (Breslow, Efron and exact ties, delayed
  entry, strata, case weights, offsets, cluster-robust variance), its tests,
  residuals, predictions and time-transform covariates, and `clogit`, with a count of the matched sets and
  strata that carry information about a covariate.
- The Fine-Gray model, Aalen-Johansen cumulative incidence, multi-state
  occupation probabilities and Cox models for multi-state processes, each with
  infinitesimal jackknife standard errors, and clustered bootstraps for the
  first and the third.
- Inverse probability weights for treatment (average effect, on the treated or
  the controls, overlap, matching and optimal-subset weights), for censoring and
  for time-varying treatment, with weighted Cox and Kaplan-Meier fits whose
  variance accounts for the estimated weights.
- The risk by a time with right censoring under each treatment by augmented
  inverse probability weighting, with a Cox model of the event, a logistic model
  of the treatment and a Kaplan-Meier or Cox model of the censoring, as
  `riskRegression::ate`, with the influence-function standard error.
- Target trial emulation: the weights of treatment switches and of censoring for
  sequential trials with the weighted logistic outcome model and cumulative
  incidence of `TrialEmulation`, fitted on the rows `TrialEmulation` chooses,
  with case weights for a bootstrap over persons, truncation of the weights as
  its `"p99"` and `"weight_limits"`, and a competing event in the
  outcome model and the cumulative incidence. The whole chain from
  `epistat-matching`'s expansion is checked against `TrialEmulation`.
  Clone-censor-weight for strategies with a window of starting, a minimum
  duration of treatment and dynamic starts set off by a covariate, with a
  bootstrap over persons. Weighted Kaplan-Meier curves give the risk difference
  of two arms with its standard error.
- Flexible parametric survival models of Royston and Parmar on the log cumulative
  hazard, log cumulative odds and probit scales, with a restricted cubic spline
  baseline in log time, time-varying effects, delayed entry, case weights and
  cluster-robust variance, and survivor, hazard and cumulative hazard curves and
  contrasts with delta-method limits, checked against `survreg`,
  `flexsurv::flexsurvspline` and `rstpm2::stpm2`.
- Relative survival: net survival by Pohar Perme and by Ederer II with
  variances and limits, by group, exactly at every entry and exit time or on a
  grid of intervals, from the reference hazards of a rate table, and excess
  hazard models, piecewise constant on split or aggregated follow-up and
  flexible parametric with the expected hazard at each exit, checked against
  `relsurv::rs.surv`, `popEpi::relpois` and `rstpm2::stpm2`. Also the expected
  survival of a cohort by Ederer I, Hakulinen and the conditional method
  (`survival::survexp`), relative survival by Ederer I and Hakulinen,
  age-standardised net survival with the ICSS weights, the crude probabilities
  of death from the disease and from other causes (`relsurv::cmp.rel`) and the
  test of equal net survival of Grafféo et al. (`relsurv::rs.diff`).
- Case-cohort analysis by the five methods of `survival::cch` (Prentice, Self-Prentice,
  Lin-Ying with Barlow's robust variance, and Borgan's two methods for a stratified
  subcohort), with the variance of the sampling of the subcohort.
- Nested case-control data with Samuelsen's inverse probability weights, computed in
  `O(N log N)` with the risk-set sizes, the weighted Cox fit with a robust variance and
  the absolute risk from its baseline hazard, checked against `multipleNCC`.
- Shared gamma frailty Cox models by penalised partial likelihood with Breslow or
  Efron ties, strata, entry times, offsets and case weights, theta from the
  profile marginal likelihood or by `coxph`'s own search, the variance from the
  whole information and from its diagonal form, and EM as a second route, for
  hundreds of thousands of clusters without a dense information matrix, checked
  against `coxph` with `frailty()`.
- The Titman-Putter test of the Markov property.
- The self-controlled case series (conditional Poisson likelihood, robust
  variance, relative incidences with intervals) with the cutting of follow-up
  into pieces by risk windows of several exposures, age bands, seasons and
  calendar periods, counts of exposed and informative cases, the
  semiparametric fit, the fit for exposures the event censors and the sample
  size; the windows of the case-crossover, case-time-control and
  case-case-time-control designs with unidirectional, bidirectional and
  time-stratified referents in calendar or fixed strata, the sampling of
  control persons at each case's date and the exposure in each window; and
  the attributable fraction function from standardised Cox risks. Checked
  against `SCCS` (`standardsccs`, `semisccs`, `eventdepenexp`, `samplesize`),
  `rSPARCS::CXover.data`, `season::casecross`, `survival::clogit` and `AF`.
- Kaplan-Meier and Nelson-Aalen curves, restricted means, log-rank tests,
  pseudo-observations, concordance, Brier scores and calibration, standardised
  risks, accelerated failure time models, person-time tables, rate tables
  (built from long rows, with or without R's extended ends) with expected
  counts for standardised incidence ratios, and the data tools
  `survSplit`, `tmerge` and `survcondense`.

Most estimators read a `SurvData`, the checked columns of a follow-up table.
A bootstrap takes a root seed from the caller, replicate `r` drawing from
`epistat_rng::replicate_seed(seed, r)`, and results do not depend on the number
of threads. With the `arrow` feature a
`SurvData` is built from Apache Arrow arrays without copying the columns.
