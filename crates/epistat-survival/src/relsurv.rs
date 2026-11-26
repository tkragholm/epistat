//! Relative and net survival: the survival of a cohort after the deaths that
//! the general population would have had are taken out, as `relsurv` and
//! `popEpi`, the expected survival of a cohort, as `survival::survexp`, the
//! crude probabilities of death and the test of equal net survival, as
//! `relsurv::cmp.rel` and `relsurv::rs.diff`, and excess hazard regression,
//! as `popEpi::relpois` and `rstpm2::stpm2` with `bhazard`.
//!
//! The observed hazard of a person is the sum of the hazard `lambda*_i(t)` of
//! the reference population, by age, sex and calendar year from a
//! [`RateTable`](crate::ratetable::RateTable), and an excess hazard
//! `lambda_E(t)`. The net survival is `R(t) = exp(-Lambda(t))` with
//! `Lambda = integral of lambda_E`, the survival if the disease were the only
//! cause of death. [`ExpectedHazard`] holds the reference hazard of each row
//! of a cohort, [`net_survival`] estimates `Lambda` without a model, and
//! [`excess_hazard`] fits `lambda_E` as a piecewise constant function of the
//! follow-up time and of covariates. The flexible parametric excess hazard
//! model of Nelson et al. is [`crate::flexpar::fit`] with
//! [`crate::flexpar::FlexParOptions::bhazard`].
//!
//! # Expected hazard
//!
//! The hazard of a row is constant between the cuts of the rate table, so
//! its cumulative hazard `H*_i(t)` is piecewise linear in the time since the
//! origin, from 0 and not from the entry time, and the expected survival is
//! `S*_i(t) = exp(-H*_i(t))`. Follow-up off the table is an error, by the
//! rule of [`crate::ratetable`]. Beyond the follow-up of a row the hazard
//! continues along the row's walk through the table, and is undefined where
//! that walk leaves it.
//!
//! # Estimators
//!
//! With `Y_i(t)` the indicator of being at risk (entry before `t`, exit at
//! or after it) and `dN_i(t)` that of dying at `t`, the estimators of the
//! cumulative excess hazard are
//!
//! ```text
//! Ederer II:    d Lambda(t) = sum_i dN_i / sum_i Y_i - sum_i Y_i lambda*_i(t) dt / sum_i Y_i
//! Pohar Perme:  d Lambda(t) = sum_i w_i dN_i / sum_i w_i Y_i - sum_i w_i Y_i lambda*_i(t) dt / sum_i w_i Y_i
//!               w_i(t) = 1 / S*_i(t)
//! variance:     Ederer II     sum_{s <= t} sum_i dN_i / (sum_i Y_i)^2
//!               Pohar Perme   sum_{s <= t} sum_i w_i^2 dN_i / (sum_i w_i Y_i)^2
//! ```
//!
//! The variances are those of `relsurv::rs.surv` (its `std.err` squared, in
//! the source of version 2.3.3: `cumsum(dnisisq / yisi^2)` and
//! `cumsum(dni / yi^2)`) and of Pohar Perme et al. (2012), which treat the
//! weights as known. The weights of the deaths at `t` and of the denominator
//! are taken at `t`. The net survival is `exp(-Lambda)`, as `relsurv::rs.surv`
//! gives with `type = "fleming-harrington"`, and the limits are normal on the
//! scale of [`Transform`], as `rs.surv` forms them.
//!
//! # Exact evaluation
//!
//! Between two consecutive time points at which someone enters or leaves, the
//! risk set is fixed, and `w_i(u) lambda*_i(u) = w_i'(u)`. The integral of the
//! Pohar Perme estimator over the interval `(a, b]` is therefore
//! `ln W(b) - ln W(a)` with `W(u) = sum_i w_i(u)` over the persons at risk
//! in it, computed as `ln_1p(sum_i w_i(a) expm1(H*_i(b) - H*_i(a)) / W(a))` so
//! that the small increment keeps its relative precision, and that of Ederer
//! II is the mean of `H*_i(b) - H*_i(a)`. With [`Grid::EventTimes`] the
//! estimate has no discretisation error and is evaluated at every distinct
//! entry and exit time. A person entering at `t` is at risk after `t`, not at
//! it. The evaluation is serial, visits every person at risk at every time
//! point, and is refused above 3e8 such pairs.
//!
//! `relsurv::rs.surv` instead integrates the weights by the trapezoidal rule in
//! steps of `precision` days between consecutive exit times, with the hazard
//! of each step taken at its start, counts a person entering between two exit
//! times as at risk from the earlier one, and counts a person entering at an
//! exit time as at risk at it. Its estimate equals this one up to the
//! trapezoidal error when the steps do not straddle a cut of the table and no
//! one enters at a time at which someone dies.
//!
//! # Intervals
//!
//! With [`Grid::Width`] follow-up is cut into intervals `(a_k, b_k]` of the
//! width given, and the person `i` has in interval `k` the person-time
//! `y_ik`, the event indicator `d_ik` and the expected events
//! `h*_ik = H*_i(hi) - H*_i(lo)` over the overlap `(lo, hi]` of its
//! follow-up with the interval. The weight is `exp(H*_i(m))` at the midpoint
//! `m` of the overlap for Pohar Perme and 1 for Ederer II, and
//!
//! ```text
//! Lambda_k - Lambda_{k-1} = (b_k - a_k) sum_i w_ik (d_ik - h*_ik) / sum_i w_ik y_ik
//! variance increment      = (b_k - a_k)^2 sum_i w_ik^2 d_ik / (sum_i w_ik y_ik)^2
//! ```
//!
//! The last interval ends at the largest exit time. This is the estimator
//! with person-time denominators, an approximation of the continuous one. Its
//! error has two parts: the weights and expected hazards of an interval, which
//! go to the continuous ones as the width goes to 0, and the person-time of
//! those who exit inside an interval, which makes the denominator smaller than
//! the risk set at the exit. The second does not vanish with the width unless
//! the cuts come to fall on the exit times; it is of the order of the deaths
//! over the square of the number at risk, so it vanishes as the cohort grows.
//! Persons are summed in blocks of 1024 and the blocks added in order, so the
//! result does not depend on the number of threads, and the memory is the
//! number of intervals, not the number of persons times it. An interval in
//! which no one is at risk adds nothing, as `survival::survfit` treats a time
//! with an empty risk set.
//!
//! `popEpi::survtab` forms similar sums with weights from its own
//! approximation of `S*_i` in the middle of the interval. In popEpi 0.5.0 a
//! person with a single record in an interval `(a, b]`, from time `s`, gets
//! the weight `1 / (S*_i(s) exp(lambda* (b - s) / 2))`, below 1, where a
//! person with several records gets the inverse of an expected survival, above
//! 1: the sign of the exponent is reversed. Its Pohar Perme estimate is
//! therefore not used as a reference.
//!
//! # Scale
//!
//! [`ExpectedHazard`] keeps 8 bytes for each time dimension, 8 for the
//! factors, 8 for the end of the follow-up and 1 for whether it is used, for
//! every row, 33 bytes for a table by age, sex and calendar year. The pieces of a row, 24 bytes each and
//! about two a year of follow-up for such a table, are formed when an
//! estimator reads the row, by walking it through the table, the same walk
//! each time. The intervals of [`Grid::Width`] cost `O(n g)` time for `g`
//! intervals at risk per person and `O(g)` memory beyond the rows;
//! [`excess_hazard`] streams its records, walking each row in every pass, and
//! costs `O(n k p^2)` per iteration for `k` intervals per person and `p`
//! parameters, or that of its cells with [`ExcessOptions::aggregate`].
//! [`Grid::EventTimes`] holds the pieces of every person who has entered the
//! risk set.
//!
//! # Excess hazard regression
//!
//! [`excess_hazard`] cuts every row at the interval cut points and at the cuts
//! of the rate table, so that the reference hazard `lambda*` is constant over
//! each record, with expected events `d* = y lambda*`; follow-up beyond the
//! last interval cut is not used, as in `popEpi::relpois`. For a cell with
//! person-time `y`, events `d` and expected events `d*`, the events are
//! Poisson with
//!
//! ```text
//! mu = d* + y exp(eta),  eta = alpha_k + x'beta,
//! l = sum (d ln mu - mu),
//! score_j = sum (d / mu - 1) y exp(eta) z_j,
//! I_jl    = sum (y exp(eta))^2 / mu z_j z_l,
//! ```
//!
//! with `z` the indicator of the interval and `x`. A cell is one record, as
//! `popEpi::relpois` fits data split by `lexpand` or `splitMulti` at the same
//! cuts; then `ln mu = ln y + ln(lambda*(t) + lambda_E)` for a death at `t`,
//! and the log likelihood is that of Esteve et al. (1990) for a piecewise
//! constant excess hazard, up to a constant. Records split more coarsely, as
//! at the interval cuts alone, would average `lambda*` over a record and
//! change the estimate. With [`ExcessOptions::aggregate`] a cell is the sum of
//! the records of one interval and one set of covariate values, the grouped
//! data of Dickman et al. (2004).
//!
//! The model is the generalised linear model with the link `log(mu - d*)` and
//! the offset `ln y`. The link is not the canonical one, so Fisher scoring,
//! which `glm` uses, converges only linearly; the fit takes Newton steps with
//! the observed information
//!
//! ```text
//! J_jl = sum [ d (y exp(eta) / mu)^2 - (d / mu - 1) y exp(eta) ] z_j z_l
//! ```
//!
//! where it is positive definite, and Fisher steps elsewhere, with step
//! halving, from the interval rates `max(d - d*, 1/2) / y`. The covariance is
//! the inverse of the Fisher information at the estimate, as `glm` gives it.
//!
//! The log likelihood increases without bound as the log excess hazard of an
//! interval without events goes to minus infinity, so such an interval is
//! fixed at an excess hazard of 0 and reported in [`ExcessFit::at_bound`]
//! with the coefficient minus infinity, and the other parameters are fitted
//! given it. An interval with events can also have its maximum there, when
//! its deaths are few for its expected deaths; it is fixed when its fitted
//! excess events fall below `1e-10` of its events.
//!
//! # Expected survival and relative survival
//!
//! [`expected_survival`] gives the expected survival of a cohort by Ederer I,
//! Hakulinen and the conditional method, as `survival::survexp`, exactly
//! between the points at which its set of rows changes; its documentation has
//! the formulas. Ederer I and Hakulinen follow rows beyond their own
//! follow-up through the table, as [`ExpectedHazard::cumulative`] does.
//! [`Method::EdererI`] and [`Method::Hakulinen`] of [`net_survival`] are the
//! relative survival estimators of `relsurv::rs.surv` with `type =
//! "fleming-harrington"`: the observed survival `exp(-Lambda_O)` of Nelson and
//! Aalen divided by the expected survival of the method, so that the reported
//! cumulative excess hazard is `Lambda_O + ln S_e`, with the variance
//! `sum d / n^2` of `Lambda_O`, as `rs.surv` gives it. `rs.surv` forms
//! Hakulinen's hazard with the weights at the end of each step,
//! `sum S*_i(b) dH*_i / sum S*_i(b)`, which differs from the exact integral
//! by less than the largest squared increment of the step. Neither takes
//! delayed entry.
//!
//! # Age standardisation
//!
//! [`standardise_net`] weights the curves of age groups, such as the strata
//! of [`net_survival`] by age group, with [`ICSS`] or other weights:
//! `R = sum w_k R_k` with variance `sum w_k^2 R_k^2 V_k`, the traditional
//! approach of Corazziari et al. (2004).
//!
//! # Crude probabilities of death
//!
//! [`crude_mortality`] gives the probabilities of dying of the disease and of
//! other causes in the presence of each other, `integral S(u-) dLambda_E(u)`
//! and `integral S(u-) lambda_P(u) du` with the Kaplan-Meier survival `S`
//! (Cronin and Feuer 2000), with the variances of `relsurv::cmp.rel`.
//!
//! # Test of equal net survival
//!
//! [`net_survival_test`] is the log-rank type test of Grafféo et al. (2016),
//! the observed minus expected excess deaths of each group weighted by the
//! inverse expected survival, as `relsurv::rs.diff`, on its grid.
//!
//! `relsurv::cmp.rel` and `relsurv::rs.diff` (relsurv 2.3-3) call `exp_prep`
//! without the entry times, so a row with delayed entry counts as at risk
//! from time 0 there; [`crude_mortality`] and [`net_survival_test`] count it
//! from its entry, and the tests check them with delayed entry against the
//! definitions written out in R and without it against relsurv.
//!
//! # Not covered
//!
//! The period approach (restricting follow-up to a calendar window) is made
//! by clipping the entry and exit times of each row before the call. Age
//! standardisation by individual weights inside the Pohar Perme estimator
//! (Brenner's alternative) and the estimators of relative survival by the
//! actuarial method are not implemented.
//!
//! # References
//!
//! 1. Ederer F, Heise H. Instructions to IBM 650 programmers in processing
//!    survival computations. Methodological note no. 10, End Results Evaluation
//!    Section, National Cancer Institute; 1959.
//! 2. Dickman PW, Sloggett A, Hills M, Hakulinen T. Regression models for
//!    relative survival. Stat Med. 2004;23(1):51-64.
//! 3. Pohar Perme M, Stare J, Esteve J. On estimation in relative survival.
//!    Biometrics. 2012;68(1):113-120.
//! 4. Pohar Perme M, Pavlic K. Nonparametric relative survival analysis with
//!    the R package relsurv. J Stat Softw. 2018;87(8):1-27.
//! 5. Nelson CP, Lambert PC, Squire IB, Jones DR. Flexible parametric models
//!    for relative survival, with application in coronary heart disease. Stat
//!    Med. 2007;26(30):5486-98.
//! 6. Esteve J, Benhamou E, Croasdale M, Raymond L. Relative survival and the
//!    estimation of net survival: elements for further discussion. Stat Med.
//!    1990;9(5):529-538.
//! 7. Ederer F, Axtell LM, Cutler SJ. The relative survival rate: a
//!    statistical methodology. Natl Cancer Inst Monogr. 1961;6:101-121.
//! 8. Hakulinen T. Cancer survival corrected for heterogeneity in patient
//!    withdrawal. Biometrics. 1982;38(4):933-942.
//! 9. Cronin KA, Feuer EJ. Cumulative cause-specific mortality for cancer
//!    patients in the presence of other causes: a crude analogue of relative
//!    survival. Stat Med. 2000;19(13):1729-1740.
//! 10. Graffeo N, Castell F, Belot A, Giorgi R. A log-rank-type test to
//!     compare net survival distributions. Biometrics. 2016;72(3):760-769.
//! 11. Corazziari I, Quinn M, Capocaccia R. Standard cancer patient population
//!     for age standardising survival ratios. Eur J Cancer.
//!     2004;40(15):2307-2316.

mod cohort;
mod crude;
mod difference;
mod estimator;
mod excess;
mod expected;
mod standardise;

pub use cohort::{ExpectedCurve, ExpectedMethod, expected_survival};
pub use crude::{CrudeMortality, crude_mortality};
pub use difference::{NetSurvivalTest, net_survival_test};
pub use estimator::{Grid, Method, NetOptions, NetSurvival, Transform, net_survival};
pub use excess::{ExcessFit, ExcessOptions, excess_hazard};
pub use expected::ExpectedHazard;
pub use standardise::{ICSS, StandardisedNet, standardise_net};
