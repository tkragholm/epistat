//! Cox regression on a design matrix, with Breslow, Efron or exact ties,
//! delayed entry, strata, case weights and a cluster-robust variance.
//!
//! The fit follows `survival::coxph` step by step: the same standardisation of
//! the covariates, the same factorisation of the information matrix, the same
//! step halving and the same stopping rule, so that the iterations, and not
//! only the limit, can be compared with R's. It takes an offset, starting
//! coefficients and the limits of `coxph.control`, and reports the Wald, score
//! and likelihood-ratio tests of `summary.coxph`.
//!
//! # The data and the entry points
//!
//! [`fit`] and [`score_residuals`] take a [`SurvData`], which holds the exit
//! time, the optional entry time, the status, the case weight, the strata and
//! the clusters, a design matrix with one row per row of the data, and a
//! [`CoxOptions`]. The data type checks its columns once when it is built. The
//! entry points check the design matrix, which must have a column and finite
//! values.
//!
//! The times are first passed through the rule of `survival::aeqSurv`, as
//! `coxph` does: times that differ by no more than `sqrt(eps)`, absolutely or
//! relative to their mean, are made equal. `CoxOptions::timefix` sets the rule,
//! and `TimeFix::Exact` compares the times as they are. A missing entry time
//! means that every row is at risk from the beginning.
//!
//! The event is the status `CoxOptions::cause`. With one status code that is 1
//! and the option may stay `None`. With several codes the option names the one
//! to model, the other codes are censored, and the fit is cause-specific. A
//! status code is never read as a transition between states; that is a model
//! of its own.
//!
//! Rows of weight zero are left out of the fit, as before. `n_rows` of the
//! result counts every row of the data, as `n` of a `coxph` fit does, and
//! `n_used` the rows with a weight above zero. Strata and clusters are the
//! dense codes of the data, which are in the order of the codes the caller
//! gave, so the strata are summed in that order. A cluster whose rows all have
//! weight zero adds a row of zeros to the sum of cluster scores, as a code that
//! no row carries did for the deprecated entry points. Cluster codes that skip
//! values can change the order of the additions of that sum, and with it the
//! last bits of the robust variance, against the deprecated entry points.
//!
//! [`fit_cox`], [`cox_score_residuals`] and [`CoxDesign`] are deprecated. They
//! take the same columns as slices, compare times exactly, and give the bits
//! they gave before the data type. Their `n_rows` counts the rows fitted, so it
//! equals `n_used`.
//!
//! # Algorithm
//!
//! Row `i` has a covariate vector `x_i` of length `p`, a case weight `w_i > 0`,
//! an entry time `a_i`, an exit time `t_i` and an event indicator `delta_i`.
//! Rows are in strata, and a stratum has its own baseline hazard. At an event
//! time `t` the risk set `R(t)` of a stratum is the rows with
//! `a_j < t <= t_j`. With `theta_j = exp(beta' x_j)` the weighted sums over the
//! risk set are
//!
//! ```text
//! S_0(t) = sum over R(t) of w_j theta_j,
//! S_1(t) = sum over R(t) of w_j theta_j x_j,
//! S_2(t) = sum over R(t) of w_j theta_j x_j x_j'.
//! ```
//!
//! Let `D(t)` be the events at `t`, `d` their number and `w_bar` the mean of
//! their weights, and let `E_0, E_1, E_2` be the sums above taken over `D(t)`
//! only. Breslow's approximation \[1\] gives every tied event the whole risk set
//! as its denominator. Efron's approximation \[2\] gives the `k`-th tied event
//! the risk set less `k / d` of the tied events, `k = 0, ..., d - 1`:
//!
//! ```text
//! Breslow:  one term,   weight sum of w over D(t),  denominators S_r
//! Efron:    d terms,    weight w_bar each,          denominators S_r - (k / d) E_r
//! ```
//!
//! For a term with weight `u` and denominators `D_0, D_1, D_2`, the log
//! partial likelihood \[3\] gains `u (beta' x - log D_0)` over the covariates of
//! the events, so
//!
//! ```text
//! l(beta) = sum over t of [ sum over D(t) of w beta' x  -  sum over terms of u log D_0 ]
//! U(beta) = sum over t of [ sum over D(t) of w x  -  sum over terms of u D_1 / D_0 ]
//! I(beta) = sum over t, terms of u [ D_2 / D_0  -  (D_1 / D_0)(D_1 / D_0)' ].
//! ```
//!
//! Strata add their terms. A stratum with no event adds nothing.
//!
//! # Exact ties
//!
//! `Ties::Exact` is the exact partial likelihood of `coxph(ties = "exact")`,
//! the discrete-time conditional logistic likelihood \[6\]. At an event time
//! the `d` tied events are taken to be one set of `d` rows out of the risk set
//! `R(t)` of size `n`, each set `S` having the probability proportional to
//! the product of its risk scores. With `r_j = exp(beta' x_j + o_j)`:
//!
//! ```text
//! D_0 = sum over sets S of size d in R(t) of  prod over S of r_j
//! D_1 = sum over sets S                       (sum over S of x_j) prod over S of r_j
//! D_2 = sum over sets S                       (sum over S of x_j)(sum over S of x_j)' prod over S of r_j
//! l(beta) = sum over t of [ sum over D(t) of beta' x + o  -  log D_0 ]
//! U(beta) = sum over t of [ sum over D(t) of x  -  D_1 / D_0 ]
//! I(beta) = sum over t of [ D_2 / D_0  -  (D_1 / D_0)(D_1 / D_0)' ].
//! ```
//!
//! The sums over sets are not enumerated. They follow the recursion of Gail,
//! Lubin and Rubinstein \[6\] over the rows of the risk set `j = 1, ..., n`,
//! with `D_r(m, j)` the sum over the sets of size `m` among the first `j` rows,
//! `D_0(0, j) = 1` and `D_1(0, j) = D_2(0, j) = 0`:
//!
//! ```text
//! D_0(m, j) = D_0(m, j-1) + r_j D_0(m-1, j-1)
//! D_1(m, j) = D_1(m, j-1) + r_j [ D_1(m-1, j-1) + x_j D_0(m-1, j-1) ]
//! D_2(m, j) = D_2(m, j-1) + r_j [ D_2(m-1, j-1) + x_j D_1(m-1, j-1)'
//!                                 + D_1(m-1, j-1) x_j' + x_j x_j' D_0(m-1, j-1) ]
//! ```
//!
//! The risk set is scanned once for each such time, and only the `m` of the
//! last row of the recursion is kept, so the memory is `d p^2` and the time
//! `d n p^2` for a time with `d` tied events among `n` at risk and `p`
//! columns. A time with one event costs the same as in a Breslow fit and is
//! computed as one. The cost of a fit with exact ties is therefore the sum over
//! the event times with ties of `d n p^2`, which is small for matched sets
//! and large for a stratum of many thousand rows with ties of many events. R
//! refuses only when its working array overflows an `int`, which this
//! algorithm does not need, so no limit is imposed here. For data with entry
//! times R does not use the recursion but lists every subset of the risk set,
//! which is of the order of `n` choose `d` per time, and so cannot fit a time
//! with ten events among thirty at risk in reasonable time. A caller who
//! expects large tied sets should expect the time, which is not reported.
//!
//! `coxph` supports exact ties for data with and without entry times, strata
//! and offsets. It does not support case weights other than 1, which is an
//! error here, a cluster-robust variance, which needs score residuals, or score
//! and Schoenfeld residuals, all of which are errors here. The iteration
//! follows `coxexact.fit` for data without entry times and `agexact.fit` for
//! data with them, which differ from `coxph.fit` and `agreg.fit`: the
//! halving goes half way back each time, no convergence is accepted during
//! halving, and on running out of iterations `coxexact.fit` goes back to the
//! last accepted point where `agexact.fit` keeps the last trial point. Both
//! flag a coefficient as possibly infinite when the Newton step at the end
//! exceeds both `eps` and `sqrt(eps)` times the coefficient, on the scaled
//! covariates here, on the scale of the columns in R for data with entry times,
//! so the flag can differ in a borderline case.
//!
//! # Newton-Raphson with step halving
//!//!
//! The covariates are centred at their weighted mean and divided by their
//! weighted mean absolute deviation, as `coxph` does, and the iteration runs on
//! these. A column that is constant over the rows is set to zero. Starting at
//! the coefficients `beta` of `CoxOptions::init`, zero by default, with `l_0`
//! the log partial likelihood there:
//!
//! 1. The trial point is `beta + I^{-1} U`, with `I` and `U` taken at the last
//!    accepted `beta`. The first trial is the first Newton step.
//! 2. The log partial likelihood `l` at the trial point is computed. When
//!    `|1 - l_prev / l| <= eps` the fit has converged at the trial point, with
//!    `eps` of `CoxOptions::eps`, `1e-9` by default as in `coxph.control`.
//! 3. Otherwise, when `l` is below the last accepted value, or is not finite,
//!    the trial point moves back towards the last accepted `beta`, and step 2
//!    is repeated. The `h`-th halving in a row sets the trial point to
//!    `(trial + h beta) / (h + 1)`, so that the first goes half way, the second
//!    two thirds of the way and so on. When `l` is not below the last accepted
//!    value, the trial point is accepted, and the next trial is a Newton step
//!    from it.
//! 4. After `CoxOptions::iter_max` trial points without convergence, 20 by
//!    default as `iter.max` of `coxph.control`, the fit stops, with
//!    `converged` false. `iterations` counts the trial points.
//!
//! With `iter_max` 0 no trial point is computed: the model is the one at the
//! starting coefficients, its coefficients are those, and its covariance and
//! tests are taken there. `coxph` fits that way to get a score test.
//!
//! `coxph` has two fitting routines, `coxph.fit` for data without entry times
//! and `agreg.fit` for data with them, and the steps above follow each in what
//! it does when it meets the limit. In both, the halving is as in step 3.
//! `agreg.fit` also does not accept convergence during halving, and treats a
//! change of the rank of the information as a failed step. When it runs out of
//! iterations it keeps the last trial point unless the likelihood there is
//! lower than at the last accepted point by more than `eps`, relative, in
//! which case it goes back to the last accepted point. `coxph.fit` accepts
//! convergence also during halving, and when it runs out of iterations it goes
//! back to the last accepted point if the last trial was not an improvement.
//! `coxph.fit` reports one trial point more than the limit when it runs out of
//! iterations, and also returns a variance matrix that is not that of the point
//! it reports. `iterations` is the number of trial points, and the variance is
//! that of the point reported. The deprecated entry points halve half way
//! each time, accept no convergence during halving and keep the last trial
//! point, as they always did.
//!
//! The offset enters the linear predictor of every row, `beta' x_j + o_j`, and
//! therefore the risk score and the log partial likelihood, but not the score
//! or the information in any other way. The mean of the offsets is taken out
//! before the fit, as `coxph` does, to keep the exponential in range. A constant
//! added to every offset does not change the partial likelihood, so this changes
//! no result.
//!
//! The Newton step solves `I s = U` by the factorisation `I = L D L'` of
//! `coxph`. A pivot at most `1.8e-12` (`.Machine$double.eps^0.75`, `toler.chol`)
//! of the largest diagonal element of `I` is treated as zero, and its column is
//! removed from the solve. A column removed this way is aliased with the
//! columns before it, as when `coxph` reports `NA`, and its coefficient and
//! standard error are `None`. The standard errors come from the diagonal of
//! the inverse of `I` over the other columns, with `1 / sqrt` of the scale
//! undone, so they are on the scale of the columns given.
//!
//! # Covariance matrices
//!
//! `covariance` is the whole inverse `V`, not only its diagonal, on the scale
//! of the columns given: entry `(a, b)` is `V_ab f_a f_b` with `f` the factor
//! that undoes the scaling of the columns. Row-major, `p * p` entries, and
//! `None` in every entry whose row or column is an aliased coefficient, as the
//! standard errors are. It is exactly symmetric. The standard errors are
//! computed as `sqrt(V_aa) f_a`, which differs from the square root of the
//! diagonal entry in the last bits.
//!
//! # Cluster-robust variance
//!
//! With cluster codes the robust variance is the grouped sandwich estimate of
//! Lin and Wei \[4\]:
//!
//! ```text
//! V_robust = V (sum over clusters g of U_g U_g') V,    V = I(beta_hat)^{-1}
//! ```
//!
//! with no small-sample factor, as `coxph` computes it. `U_g` is the sum within
//! cluster `g` of the score residuals, which are defined below. The sum of the
//! cross-products `U_g U_g'` is the meat of the sandwich. It is formed by
//! `epistat_inference::sandwich::meat` from the matrix with one row `U_g` per
//! cluster, and `V M V` is then formed here, in a fixed order and averaged with
//! its transpose, since `V` has zero rows for aliased columns and is not the
//! inverse of a matrix that `epistat-inference` could factorise. The
//! covariance of `epistat_inference::influence` is not used: it multiplies by
//! `G / (G - 1)` and divides by `n^2`, which `coxph` does not. With a
//! cluster code that no row carries, the cluster has no row and adds nothing.
//! `robust_covariance` is the whole matrix `V M V` on the scale of the columns
//! given, laid out as `covariance`.
//!
//! # Score residuals
//!
//! [`cox_score_residuals`] returns, for every row, the derivative of the log
//! partial likelihood with respect to each coefficient through the case weight
//! of the row, on the scale of the columns given. A row with an event
//! contributes `w_j (x_j - m)` with `m` the mean over the terms at its time of
//! `D_1 / D_0`. Every row contributes `-w_j theta_j sum_t (x_j h_t - c_t)` over
//! the event times it is at risk at, where `h_t` is the sum of `u / D_0` over
//! the terms at `t` and `c_t` the sum of `u D_1 / D_0^2`. The sums over the
//! event times are differences of running sums over the times of the stratum.
//! Under Efron's approximation a tied event is at risk in only part of the
//! terms of its own time, and a final correction adds that back \[5\]. The
//! residuals add up to the score, so at the estimate their sum is zero up to
//! the stopping rule. They contain the case weight. `residuals(fit, "score")`
//! of `survival` leaves it out by default, and `weighted = TRUE` puts it in,
//! so that column divided by the weights of the rows is R's default. The
//! residual of a row whose weight is zero, and of a row in a stratum without
//! events, is zero.
//!
//! The residuals are a separate function and not a field of the fit because
//! they are `rows * p` numbers, which at `10^7` rows is `80` MB per column,
//! and most fits do not need them. The robust variance does not hold them
//! either: it sums them by cluster as they are produced. The function repeats
//! the sort of the rows by stratum and exit time, which costs a sort of the
//! rows, and takes the coefficients as an argument, so it also gives the
//! residuals at a point other than the estimate.
//!
//! # Tests of the coefficients
//!
//! [`CoxTests`] in `CoxModelFit::tests` holds the three tests of
//! `summary.coxph` that all coefficients equal their starting values, which are
//! zero unless `CoxOptions::init` says otherwise. With `d` the distance of the
//! estimate from the starting values, `U_0` and `I_0` the score and information
//! there, `l_0` and `l` the log partial likelihoods at the starting values and
//! at the estimate, and `V` the covariance of the estimate:
//!
//! ```text
//! likelihood ratio:  2 (l - l_0)
//! Wald:              d' V^-1 d
//! score (log-rank):  U_0' I_0^-1 U_0
//! ```
//!
//! each with the rank of the fit as degrees of freedom, compared with the
//! chi-square distribution. The Wald test is also given with the robust
//! covariance (`robust_wald`, which `summary.coxph` reports as its Wald test
//! when clusters are given). The robust score test replaces `I_0` by the sum of
//! the cross-products of the cluster score sums at the starting values
//! (`robust_score`, `rscore` of the `coxph` fit). `coxph` leaves the offset
//! out of the residuals of that test, so its value is not that of the model with
//! the offset. These tests use the offset.
//!
//! The model-based Wald statistic is computed as `d' I d` over the columns that
//! are not aliased, which is the same quantity. The solves with `I_0` and with
//! the robust covariance use the pivot rule described above, on the scaled
//! covariates, and a test is `None` when the matrix does not have the rank of
//! the fit, when the statistic is negative, or when it is not finite. The score
//! tests do not need the estimate, so they are given also when a coefficient
//! diverges. The others are `None` then, as for no estimate. The robust tests
//! are `None` without clusters and with fewer than two.
//!
//! The tests are a field of the fit and not a function of it because the score
//! tests need the score and information at the starting values, which the fit
//! has computed already and which a function of the fit alone would have to
//! compute again with a pass over the rows. Adding the field changes no
//! existing field.
//!
//! # Results that are not numbers
//!
//! The estimate is `None`, with zero iterations and `converged` false, when no
//! row has a positive weight, no row has an event, every column is aliased, the
//! information matrix is not positive semi-definite, or the log partial
//! likelihood is not finite at `beta = 0`.
//!
//! The maximum of the partial likelihood does not exist when a column
//! separates the events, for example when every row with `x = 1` is censored.
//! The likelihood then rises towards a limit, the coefficient grows without
//! bound, and `coxph` stops when the likelihood stops changing and warns that
//! the coefficient may be infinite. It warns for column `j` when, at the end,
//! the Newton step `|(I^{-1} U)_j|` on the scaled covariates exceeds
//! `1e-9 + sqrt(1e-9)`. The fit does the same test, lists those columns in
//! `monotone_columns`, and reports every estimate and standard error as `None`.
//! The other coefficients are conditional on a value of the diverging one that
//! only the stopping rule chose, so they are not estimates either. The log
//! partial likelihood is `None` as well, and `converged` is false. The
//! iteration count is that of the run, which `coxph` also reports.
//!
//! The robust standard errors are `None` with fewer than two clusters, where
//! the sum of residuals within the one cluster is the score and the variance
//! is zero by construction.
//!
//! # Invalid input
//!
//! A design matrix whose number of rows differs from the data's, a design
//! without columns, a covariate that is not finite, an event code that is
//! missing for a status with several codes or outside the codes, and an
//! interval that the tie rule collapses to length zero are errors. The data
//! type refuses the rest when it is built: columns of different lengths, a
//! time or weight that is not finite, a negative weight, an entry time that is
//! not before the exit time, and more rows than a `u32` addresses.
//!
//! The deprecated entry points refuse columns of different lengths, a
//! covariate, weight or exit time that is not finite, a negative weight, an
//! entry time that is NaN or not before the exit time on a row of positive
//! weight, a design without columns, and more rows than a `u32` addresses.
//! Without delayed entry they take negative infinity as every entry time.
//!
//! # Agreement with R
//!
//! The tests run [`fit`] and [`score_residuals`] with the default tie rule, as
//! `coxph` has it, and compare coefficients, both standard errors, both covariance
//! matrices, the score residuals with and without the case weight, the log
//! partial likelihood and the iteration count with `survival::coxph` for
//! Breslow and Efron ties, delayed entry, strata, weights, clusters, aliased
//! columns and a diverging coefficient. A second set of fits adds offsets,
//! starting values near and far from the estimate, `iter.max` of 0 to 3 and
//! other values of `eps`, and checks the three tests, with the robust ones, on
//! all of them. Two implementations of one Newton
//! iteration that take the same steps differ by the error of the solve, which
//! is about the condition number of the information matrix on the scaled
//! covariates times the machine epsilon. The tolerance is 8 times that for a
//! coefficient and a standard error. The covariance tolerance is `p` times
//! that, since each entry of a row of an inverse carries it, and the robust
//! covariance twice as much again, since the error of `V` enters twice. The
//! tolerance of a residual adds `n` times the machine epsilon, the error of a
//! running sum over at most `n` event times, to `p` times the 8 above.
//!
//! # References
//!
//! 1. Breslow N. Covariance analysis of censored survival data. Biometrics
//!    1974;30(1):89-99. doi:10.2307/2529620
//! 2. Efron B. The efficiency of Cox's likelihood function for censored data.
//!    J Am Stat Assoc 1977;72(359):557-65. doi:10.1080/01621459.1977.10480613
//! 3. Cox DR. Regression models and life-tables. J R Stat Soc Series B
//!    1972;34(2):187-220. doi:10.1111/j.2517-6161.1972.tb00899.x
//! 4. Lin DY, Wei LJ. The robust inference for the Cox proportional hazards
//!    model. J Am Stat Assoc 1989;84(408):1074-8.
//!    doi:10.1080/01621459.1989.10478874
//! 5. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.
//! 6. Gail MH, Lubin JH, Rubinstein LV. Likelihood calculations for matched
//!    case-control studies and survival studies with tied death times.
//!    Biometrika 1981;68(3):703-7. doi:10.1093/biomet/68.3.703

// Rows and clusters are counted in `u32`; the entry point refuses more rows
// than that.
#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry point checks that the row count fits in u32"
)]

use std::borrow::Cow;

use epistat_inference::sandwich::{meat, sandwich_with_inverse};
use epistat_linalg::{Mat, MatRef};
use epistat_stats::gamma::chi_square_sf;

use super::Ties;
use crate::check::{check_lengths, check_row_count, refuse_non_finite};
use crate::error::{Error, Result};
use crate::ldl;
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// Largest number of trial points, `iter.max` of `coxph.control`.
const MAX_ITERATIONS: u32 = 20;
/// Relative change of the log partial likelihood at convergence, `eps`.
const EPS: f64 = 1e-9;
/// Smallest pivot of the information matrix relative to its largest diagonal
/// element, `.Machine$double.eps^0.75`, which is `2^-39`.
const TOLER_CHOL: f64 = 1.818_989_403_545_856_5e-12;

/// The rows of a Cox fit on a design matrix. One entry per row, all of the same
/// length.
///
/// Superseded by [`SurvData`] and [`fit`], which take the same
/// columns and apply the tie rule of `survival`.
#[deprecated(
    since = "0.3.0",
    note = "build a `SurvData` and call `cox::fit` with a `CoxOptions`"
)]
#[derive(Debug, Clone, Copy)]
pub struct CoxDesign<'a> {
    /// The covariates, one row per observation and one column per covariate.
    /// There is no intercept column: a column that is constant over the rows is
    /// aliased.
    pub x: MatRef<'a, f64>,
    /// Whether the row ends in the event of interest.
    pub is_event: &'a [bool],
    /// Entry time. Minus infinity when there is no delayed entry.
    pub start: &'a [f64],
    /// Exit time.
    pub stop: &'a [f64],
    /// Case weight. Rows with a weight of zero are dropped.
    pub weight: &'a [f64],
    /// Stratum codes. Strata are summed in code order.
    pub stratum: &'a [u32],
    /// How tied events enter the partial likelihood.
    pub ties: Ties,
    /// Cluster codes for the robust variance, or `None` for the model-based
    /// variance alone.
    pub cluster: Option<&'a [u32]>,
}

/// One Cox fit on a design matrix. See [`fit_cox`] for when entries are `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxModelFit {
    /// One coefficient per column of the design, `None` for an aliased column
    /// and for every column when the fit is degenerate or a coefficient
    /// diverges.
    pub coefficients: Vec<Option<f64>>,
    /// The model-based standard errors, `None` where the coefficient is.
    pub standard_errors: Vec<Option<f64>>,
    /// The model-based covariance matrix of the coefficients, row-major with
    /// `p * p` entries on the scale of the columns given. An entry is `None`
    /// where its row or its column is a coefficient that is `None`.
    pub covariance: Vec<Option<f64>>,
    /// Standard errors from the cluster-robust variance. `None` when no
    /// cluster was given or there are fewer than two clusters.
    pub robust_standard_errors: Option<Vec<Option<f64>>>,
    /// The cluster-robust covariance matrix, laid out as `covariance`. `None`
    /// exactly when `robust_standard_errors` is, and with every entry `None`
    /// when clusters were given and the fit has no robust variance.
    pub robust_covariance: Option<Vec<Option<f64>>>,
    /// The log partial likelihood at the estimate, `None` when there is no
    /// estimate.
    pub log_partial_likelihood: Option<f64>,
    /// The log partial likelihood at zero, `None` when it is not finite or
    /// there are no events.
    pub null_log_partial_likelihood: Option<f64>,
    /// The tests that the coefficients equal the starting coefficients, zero
    /// unless `CoxOptions::init` says otherwise.
    pub tests: CoxTests,
    /// The summed weight of the events among the rows fitted.
    pub n_events: f64,
    /// The rows of the input. [`fit`] counts every row, as `n` of a `coxph`
    /// fit does. The deprecated [`fit_cox`] counts only the rows it fitted,
    /// after dropping rows whose weight is zero, so there it equals `n_used`.
    pub n_rows: usize,
    /// The rows fitted: those with a weight above zero.
    pub n_used: usize,
    /// The columns that are not aliased, 0 when there is no estimate.
    pub rank: usize,
    /// The trial points computed, as `iter` in a `coxph` fit.
    pub iterations: u32,
    /// Whether the log partial likelihood stopped changing within the
    /// iteration limit.
    pub converged: bool,
    /// The columns whose coefficient diverges, which `coxph` warns about as
    /// possibly infinite. Empty when none do.
    pub monotone_columns: Vec<usize>,
}

/// A chi-square test: the statistic, its degrees of freedom and the upper tail
/// probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquareTest {
    pub statistic: f64,
    /// The rank of the fit.
    pub df: usize,
    pub p_value: f64,
}

impl ChiSquareTest {
    /// The test of a statistic with `df` degrees of freedom, or `None` when
    /// the statistic is negative or not finite.
    pub(crate) fn new(statistic: f64, df: usize) -> Option<Self> {
        (statistic.is_finite() && statistic >= 0.0).then(|| Self {
            statistic,
            df,
            p_value: chi_square_sf(statistic, df as f64),
        })
    }
}

/// The tests of `summary.coxph` that all coefficients equal their starting
/// values, which are zero by default. Each is `None` where it is undefined.
///
/// The tests are a field of the fit and not a function of it because the
/// score test needs the score and the information at the starting
/// coefficients. The fit has computed them already, and a function of the
/// fit would have to read the data again, which is a pass over the rows. The
/// other tests could be computed from the rest of the fit, but they are given
/// here so that the four are in one place.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CoxTests {
    /// The Wald test with the model-based covariance matrix.
    pub wald: Option<ChiSquareTest>,
    /// The Wald test with the cluster-robust covariance matrix. `None` without
    /// clusters, and when the robust matrix does not have the rank of the fit.
    /// This is the test `summary.coxph` reports as the Wald test when clusters
    /// are given.
    pub robust_wald: Option<ChiSquareTest>,
    /// The score (log-rank) test, defined also when a coefficient diverges.
    pub score: Option<ChiSquareTest>,
    /// The score test with the sum of the cross-products of the cluster score
    /// sums in place of the information, `rscore` of a `coxph` fit.
    pub robust_score: Option<ChiSquareTest>,
    /// The likelihood-ratio test, twice the gain in log partial likelihood
    /// from the starting coefficients to the estimate.
    pub likelihood_ratio: Option<ChiSquareTest>,
}

impl CoxModelFit {
    fn degenerate(p: usize, n_events: f64, n_rows: usize, with_clusters: bool) -> Self {
        Self {
            coefficients: vec![None; p],
            standard_errors: vec![None; p],
            covariance: vec![None; p * p],
            robust_standard_errors: with_clusters.then(|| vec![None; p]),
            robust_covariance: with_clusters.then(|| vec![None; p * p]),
            log_partial_likelihood: None,
            null_log_partial_likelihood: None,
            tests: CoxTests::default(),
            n_events,
            n_rows,
            n_used: n_rows,
            rank: 0,
            iterations: 0,
            converged: false,
            monotone_columns: Vec::new(),
        }
    }
}

/// The columns the fit works on, after the checks of either entry point.
struct Input<'a> {
    x: MatRef<'a, f64>,
    is_event: &'a [bool],
    start: &'a [f64],
    stop: &'a [f64],
    weight: &'a [f64],
    stratum: &'a [u32],
    ties: Ties,
    cluster: Option<&'a [u32]>,
    /// One offset per row of the data.
    offset: Option<&'a [f64]>,
    /// Starting coefficients on the scale of the columns given.
    init: Option<&'a [f64]>,
    iter_max: u32,
    eps: f64,
    stepping: Stepping,
}

/// Which of the fitting routines of `survival` the iteration follows. They
/// differ in what they do when the likelihood stops changing during step
/// halving, and when they run out of iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stepping {
    /// `coxph.fit`, which `coxph` calls for data without entry times.
    RightCensored,
    /// `agreg.fit`, which `coxph` calls for data with entry times.
    Counting,
    /// `coxexact.fit`, which `coxph` calls for exact ties without entry times:
    /// half way back at each halving, no convergence during halving, and the
    /// last accepted point on running out of iterations after the first.
    /// `agexact.fit`, for data with entry times, is the legacy rule.
    ExactRight,
    /// The rule of the deprecated entry points: half way back at each halving,
    /// the last trial point kept on running out of iterations, and no
    /// convergence during halving.
    Legacy,
}

#[expect(deprecated, reason = "the wrapper reads the deprecated input type")]
impl<'a> From<&CoxDesign<'a>> for Input<'a> {
    fn from(data: &CoxDesign<'a>) -> Self {
        Self {
            x: data.x,
            is_event: data.is_event,
            start: data.start,
            stop: data.stop,
            weight: data.weight,
            stratum: data.stratum,
            ties: data.ties,
            cluster: data.cluster,
            offset: None,
            init: None,
            iter_max: MAX_ITERATIONS,
            eps: EPS,
            stepping: Stepping::Legacy,
        }
    }
}

/// What [`fit`] and [`score_residuals`] take besides the data.
///
/// The offset, the starting coefficients and the two iteration limits are read
/// by [`fit`] only. [`score_residuals`] reads the offset, which is part of the
/// model, and takes its coefficients as an argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoxOptions<'a> {
    /// How tied events enter the partial likelihood. `Efron`, as in `coxph`.
    pub ties: Ties,
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data before the fit. The default is the rule of
    /// `survival::aeqSurv`, which `coxph` applies.
    pub timefix: TimeFix,
    /// The event code to model. With more than one status code the rows with
    /// this status have the event and the others are censored, so the model is
    /// cause-specific, and the code is required. With one status code it is
    /// `None` or `Some(1)`.
    pub cause: Option<u32>,
    /// A term added to the linear predictor of every row, one value per row
    /// of the data, as `offset()` in a `coxph` formula. It has no coefficient.
    /// The default is no offset.
    pub offset: Option<&'a [f64]>,
    /// The coefficients the iteration starts from, one per column of the
    /// design on the scale of the columns given, as `init` of `coxph`. A value
    /// for a column without variation is ignored. The default is zero.
    pub init: Option<&'a [f64]>,
    /// The largest number of trial points, `iter.max` of `coxph.control`. With
    /// 0 the fit is the model at `init`: its coefficients are `init`, except
    /// that an aliased column is `None`, and its covariance and tests are those
    /// at that point.
    pub iter_max: u32,
    /// The relative change of the log partial likelihood at which the
    /// iteration stops, `eps` of `coxph.control`. It must be positive.
    pub eps: f64,
}

impl Default for CoxOptions<'_> {
    fn default() -> Self {
        Self {
            ties: Ties::Efron,
            timefix: TimeFix::default(),
            cause: None,
            offset: None,
            init: None,
            iter_max: MAX_ITERATIONS,
            eps: EPS,
        }
    }
}

/// Check `x` against `data`, apply the options to the data, and call `f` on the
/// columns the fit reads. The vectors that `f` borrows live only for the call.
fn with_input<R>(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    opts: &CoxOptions<'_>,
    f: impl FnOnce(&Input<'_>) -> R,
) -> Result<R> {
    check_lengths(&[("data", data.n_rows()), ("x", x.nrows())])?;
    check_finite_matrix(x)?;
    check_control(opts, data.n_rows(), x.ncols())?;
    if opts.ties == Ties::Exact {
        check_exact(data)?;
    }
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    let code = match (opts.cause, surv.n_codes()) {
        (None, 1) => 1,
        (Some(code), _) => code,
        (None, _) => {
            return Err(Error::input(
                "cause: the status has several codes, so the event code is required",
            ));
        }
    };
    let is_event: Vec<bool> = data.cause(code)?.iter().collect();
    let rows = data.n_rows();
    // A missing start is at risk from the beginning, which is what an entry
    // time of minus infinity says.
    let start = surv
        .start()
        .map_or_else(|| Cow::Owned(vec![f64::NEG_INFINITY; rows]), Cow::Borrowed);
    let weight = data
        .weight()
        .map_or_else(|| Cow::Owned(vec![1.0; rows]), Cow::Borrowed);
    let stratum = data
        .stratum()
        .map_or_else(|| Cow::Owned(vec![0; rows]), Cow::Borrowed);
    Ok(f(&Input {
        x,
        is_event: &is_event,
        start: &start,
        stop: surv.stop(),
        weight: &weight,
        stratum: &stratum,
        ties: opts.ties,
        cluster: data.cluster(),
        offset: opts.offset,
        init: opts.init,
        iter_max: opts.iter_max,
        eps: opts.eps,
        stepping: match (surv.start().is_some(), opts.ties) {
            (true, Ties::Exact) => Stepping::Legacy,
            (false, Ties::Exact) => Stepping::ExactRight,
            (true, _) => Stepping::Counting,
            (false, _) => Stepping::RightCensored,
        },
    }))
}

/// What `coxph` refuses with exact ties: case weights other than 1 on the rows
/// that are used, and clusters, which need score residuals.
fn check_exact(data: &SurvData<'_>) -> Result<()> {
    if data.cluster().is_some() {
        return Err(Error::input(
            "ties: the exact method has no cluster-robust variance",
        ));
    }
    if let Some(weight) = data.weight()
        && weight.iter().any(|&w| w > 0.0 && w - 1.0 != 0.0)
    {
        return Err(Error::input(
            "weight: the exact method supports case weights of 1 only",
        ));
    }
    Ok(())
}

/// The checks on the offset, the starting values and the convergence limit.
fn check_control(opts: &CoxOptions<'_>, rows: usize, columns: usize) -> Result<()> {
    if !(opts.eps.is_finite() && opts.eps > 0.0) {
        return Err(Error::input("eps: must be finite and positive"));
    }
    if let Some(offset) = opts.offset {
        check_lengths(&[("data", rows), ("offset", offset.len())])?;
        if offset
            .iter()
            .any(|o| !o.is_finite() || !o.exp().is_finite())
        {
            return Err(Error::input(
                "offset: a value is not finite or gives a risk score that is not finite",
            ));
        }
    }
    if let Some(init) = opts.init {
        check_lengths(&[("x columns", columns), ("init", init.len())])?;
        refuse_non_finite(init, "init")?;
    }
    Ok(())
}

/// Newton-Raphson with step halving on the weighted partial likelihood of a
/// design matrix, Breslow or Efron, as `survival::coxph` fits it.
///
/// `data` supplies the exit and entry times, the status, the weights, the
/// strata and the clusters, and `x` the covariates, one row per row of `data`.
/// The rule `opts.timefix` is applied to the times first, and rows of weight
/// zero are left out of the fit. Strata are summed in the order of their
/// codes. See the module documentation for the method, the robust variance,
/// aliased columns and diverging coefficients.
///
/// A design with no column is the model without covariates, `coxph(y ~ 1)`, or
/// `coxph(y ~ offset(o))` with an offset. It has no coefficients, an empty
/// covariance, rank 0 and no tests, since there is nothing to test. Its log
/// partial likelihood is the null one, at the offset, and it takes no
/// iteration. Strata, weights, entry times, ties and clusters are honoured, so
/// the likelihood is right in each case. The predictions of
/// [`basehaz`](crate::cox::basehaz), [`predict`](fn@crate::cox::predict) and
/// [`survfit`](crate::cox::survfit) accept such a fit.
///
/// # Errors
///
/// [`Error::Input`] when `x` has a number of rows other than
/// `data`, or holds a value that is not finite, when `opts.cause` is missing
/// for a status with several codes or outside the codes, when the rule for
/// the times finds an interval of length zero, when `opts.offset` has another
/// length than `data` or a value whose exponential is not finite, when
/// `opts.init` has another length than `x` has columns, has a value that is not
/// finite or gives a risk score that is not finite, and when `opts.eps` is not
/// positive and finite.
pub fn fit(data: &SurvData<'_>, x: MatRef<'_, f64>, opts: &CoxOptions<'_>) -> Result<CoxModelFit> {
    let mut fit = with_input(data, x, opts, fit_checked)??;
    fit.n_rows = data.n_rows();
    fit.n_used = data.n_used();
    Ok(fit)
}

/// Newton-Raphson with step halving on the weighted partial likelihood of a
/// design matrix, Breslow or Efron, as `survival::coxph` fits it.
///
/// This is the form that takes no tie rule: times are compared exactly, which
/// gives the numbers of 0.2. Use [`fit`].
///
/// See the module documentation for the method, the robust variance, aliased
/// columns and diverging coefficients.
///
/// # Errors
///
/// [`Error::Input`] when the columns differ in length, the design has no
/// column, a covariate, weight or exit time is not finite, a weight is
/// negative, an entry time is NaN or not before its exit time, or there are
/// more rows than a `u32` addresses.
#[deprecated(
    since = "0.3.0",
    note = "build a `SurvData` and call `cox::fit` with a `CoxOptions`"
)]
#[expect(deprecated, reason = "the wrapper reads the deprecated input type")]
pub fn fit_cox(data: &CoxDesign<'_>) -> Result<CoxModelFit> {
    check_design(data)?;
    fit_checked(&data.into())
}

/// The checks every deprecated entry point makes on a design.
#[expect(deprecated, reason = "the wrapper reads the deprecated input type")]
fn check_design(data: &CoxDesign<'_>) -> Result<()> {
    let rows = data.x.nrows();
    check_lengths(&[
        ("x", rows),
        ("is_event", data.is_event.len()),
        ("start", data.start.len()),
        ("stop", data.stop.len()),
        ("weight", data.weight.len()),
        ("stratum", data.stratum.len()),
        ("cluster", data.cluster.map_or(rows, <[u32]>::len)),
    ])?;
    check_row_count(rows)?;
    if data.ties == Ties::Exact {
        return Err(Error::input("ties: exact ties are fitted by `cox::fit`"));
    }
    if data.x.ncols() == 0 {
        return Err(Error::input("x has no columns"));
    }
    refuse_non_finite(data.stop, "stop")?;
    refuse_non_finite(data.weight, "weight")?;
    if let Some(index) = data.weight.iter().position(|&w| w < 0.0) {
        return Err(Error::input(format!("weight: row {index} is negative")));
    }
    check_finite_matrix(data.x)?;
    for i in 0..rows {
        if data.weight[i] > 0.0 && (data.start[i].is_nan() || data.start[i] >= data.stop[i]) {
            return Err(Error::input(format!(
                "start: row {i} has an entry time that is NaN or not before its exit time"
            )));
        }
    }
    Ok(())
}

fn check_finite_matrix(x: MatRef<'_, f64>) -> Result<()> {
    let non_finite = (0..x.nrows())
        .flat_map(|i| (0..x.ncols()).map(move |j| (i, j)))
        .filter(|&(i, j)| !x[(i, j)].is_finite())
        .count();
    if non_finite > 0 {
        return Err(Error::input(format!(
            "x: {non_finite} non-finite values; drop or resolve them before estimating"
        )));
    }
    Ok(())
}

/// One stratum's rows in descending exit order.
struct Stratum {
    /// Standardised covariates, `p` per row, in descending exit order.
    x: Vec<f64>,
    weight: Vec<f64>,
    /// The offset of each row less its mean, empty without an offset.
    offset: Vec<f64>,
    stop: Vec<f64>,
    start: Vec<f64>,
    is_event: Vec<bool>,
    /// Index in the input of each row.
    row: Vec<u32>,
    /// Cluster code of each row, empty without clusters.
    cluster: Vec<u32>,
    /// Positions in descending entry order, and the entry times in that order.
    by_start: Vec<u32>,
    start_sorted: Vec<f64>,
    /// Distinct event times, descending.
    event_times: Vec<f64>,
    /// The positions of the rows with their event at each time, in the order
    /// of `event_times`: `dying_rows[dying_offset[k]..dying_offset[k + 1]]`.
    dying_rows: Vec<u32>,
    dying_offset: Vec<usize>,
}

/// The sums `sum c x^r` for `r = 0, 1, 2`, with the lower triangle of `r = 2`.
struct Moments {
    m0: f64,
    m1: Vec<f64>,
    m2: Vec<f64>,
}

impl Moments {
    fn new(p: usize) -> Self {
        Self {
            m0: 0.0,
            m1: vec![0.0; p],
            m2: vec![0.0; p * p],
        }
    }

    fn clear(&mut self) {
        self.m0 = 0.0;
        self.m1.fill(0.0);
        self.m2.fill(0.0);
    }

    fn add(&mut self, c: f64, x: &[f64]) {
        let p = x.len();
        self.m0 += c;
        for a in 0..p {
            let cx = c * x[a];
            self.m1[a] += cx;
            for (b, xb) in x.iter().enumerate().take(a + 1) {
                self.m2[a * p + b] += cx * xb;
            }
        }
    }
}

/// The risk-set sums and the sums over the tied events at one event time.
struct Sums {
    risk: Moments,
    dying: Moments,
}

impl Stratum {
    /// Visit every event time in descending order with the sums at the
    /// per-row weights `risk`, which are `w exp(beta' x)`.
    fn sweep(
        &self,
        p: usize,
        risk: &[f64],
        active: &mut Vec<bool>,
        sums: &mut Sums,
        mut visit: impl FnMut(usize, &Sums),
    ) {
        let size = self.stop.len();
        active.clear();
        active.resize(size, false);
        sums.risk.clear();
        let (mut entering, mut leaving) = (0usize, 0usize);
        for (position, &time) in self.event_times.iter().enumerate() {
            while entering < size && self.stop[entering] >= time {
                sums.risk
                    .add(risk[entering], &self.x[entering * p..(entering + 1) * p]);
                active[entering] = true;
                entering += 1;
            }
            while leaving < size && self.start_sorted[leaving] >= time {
                let row = self.by_start[leaving] as usize;
                if active[row] {
                    sums.risk.add(-risk[row], &self.x[row * p..(row + 1) * p]);
                    active[row] = false;
                }
                leaving += 1;
            }
            sums.dying.clear();
            for &row in
                &self.dying_rows[self.dying_offset[position]..self.dying_offset[position + 1]]
            {
                let row = row as usize;
                sums.dying.add(risk[row], &self.x[row * p..(row + 1) * p]);
            }
            visit(position, sums);
        }
    }

    /// The terms of the log partial likelihood at event time `position`, as
    /// `(weight, fraction)`: the term has denominator `S - fraction * E` and
    /// is multiplied by `weight`.
    fn terms(&self, position: usize, ties: Ties) -> impl Iterator<Item = (f64, f64)> + use<> {
        let range = self.dying_offset[position]..self.dying_offset[position + 1];
        let count = range.len();
        let total: f64 = self.dying_rows[range]
            .iter()
            .map(|&row| self.weight[row as usize])
            .sum();
        let (terms, weight, spread) = match ties {
            // A single exact event has the Breslow term; sets of more are
            // computed by `ExactScratch` before the terms are asked for.
            Ties::Breslow | Ties::Exact => (1, total, 0.0),
            Ties::Efron => (count, total / count as f64, 1.0 / count as f64),
        };
        (0..terms).map(move |k| (weight, k as f64 * spread))
    }
}

/// Working arrays of the recursion for the exact partial likelihood.
#[derive(Default)]
struct ExactScratch {
    /// `D_0(m)` for `m = 0..=d`.
    d0: Vec<f64>,
    /// `D_1(m)`, `p` per `m`.
    d1: Vec<f64>,
    /// `D_2(m)`, the lower triangle of `p` by `p` per `m`.
    d2: Vec<f64>,
}

impl ExactScratch {
    /// The sums `D_0`, `D_1` and `D_2` of the module documentation over the
    /// sets of each size up to `d` among `rows`, with risk scores `risk` and
    /// covariates `x` (`p` per row). The sums for size `m` are updated from
    /// size `m - 1`, downwards in `m`, so one pass over the rows needs the
    /// arrays of one `j` only.
    fn run(
        &mut self,
        d: usize,
        p: usize,
        rows: impl Iterator<Item = usize>,
        risk: &[f64],
        x: &[f64],
    ) {
        self.d0.clear();
        self.d0.resize(d + 1, 0.0);
        self.d0[0] = 1.0;
        self.d1.clear();
        self.d1.resize((d + 1) * p, 0.0);
        self.d2.clear();
        self.d2.resize((d + 1) * p * p, 0.0);
        let mut seen = 0usize;
        for j in rows {
            seen += 1;
            let r = risk[j];
            let xj = &x[j * p..(j + 1) * p];
            for m in (1..=seen.min(d)).rev() {
                let (lo, hi) = (m - 1, m);
                for a in 0..p {
                    for b in 0..=a {
                        let t = self.d2[lo * p * p + a * p + b]
                            + xj[a] * self.d1[lo * p + b]
                            + xj[b] * self.d1[lo * p + a]
                            + xj[a] * xj[b] * self.d0[lo];
                        self.d2[hi * p * p + a * p + b] += r * t;
                    }
                }
                for (a, &xa) in xj.iter().enumerate() {
                    let t = self.d1[lo * p + a] + xa * self.d0[lo];
                    self.d1[hi * p + a] += r * t;
                }
                self.d0[hi] += r * self.d0[lo];
            }
        }
    }
}

/// The log partial likelihood, the score and the information at one beta.
struct Evaluation {
    loglik: f64,
    score: Vec<f64>,
    /// `p` by `p`, symmetric.
    information: Vec<f64>,
}

struct Model {
    p: usize,
    ties: Ties,
    strata: Vec<Stratum>,
    n_clusters: usize,
}

impl Model {
    fn row_risk(stratum: &Stratum, p: usize, beta: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let size = stratum.stop.len();
        let mut eta = Vec::with_capacity(size);
        let mut risk = Vec::with_capacity(size);
        for row in 0..size {
            let mut value: f64 = stratum.x[row * p..(row + 1) * p]
                .iter()
                .zip(beta)
                .map(|(x, b)| x * b)
                .sum();
            if !stratum.offset.is_empty() {
                value += stratum.offset[row];
            }
            let weighted = stratum.weight[row] * value.exp();
            if !weighted.is_finite() {
                return None;
            }
            eta.push(value);
            risk.push(weighted);
        }
        Some((eta, risk))
    }

    /// The log partial likelihood, score and information at `beta`, or `None`
    /// when any of them is not finite.
    fn evaluate(&self, beta: &[f64]) -> Option<Evaluation> {
        let p = self.p;
        let mut loglik = 0.0f64;
        let mut score = vec![0.0f64; p];
        let mut information = vec![0.0f64; p * p];
        let mut active = Vec::new();
        let mut sums = Sums {
            risk: Moments::new(p),
            dying: Moments::new(p),
        };
        let mut d1 = vec![0.0f64; p];
        let mut scratch = ExactScratch::default();
        for stratum in &self.strata {
            let (eta, risk) = Self::row_risk(stratum, p, beta)?;
            let mut failed = false;
            stratum.sweep(p, &risk, &mut active, &mut sums, |position, at| {
                let range = stratum.dying_offset[position]..stratum.dying_offset[position + 1];
                let tied = range.len();
                for &row in &stratum.dying_rows[range] {
                    let row = row as usize;
                    loglik += stratum.weight[row] * eta[row];
                    for (total, x) in score.iter_mut().zip(&stratum.x[row * p..(row + 1) * p]) {
                        *total += stratum.weight[row] * x;
                    }
                }
                if self.ties == Ties::Exact && tied >= 2 {
                    // The risk set is the rows that have not left before this
                    // time, which are a prefix of the descending exit order,
                    // and have entered before it.
                    let time = stratum.event_times[position];
                    let end = stratum.stop.partition_point(|&s| s >= time);
                    scratch.run(
                        tied,
                        p,
                        (0..end).filter(|&j| stratum.start[j] < time),
                        &risk,
                        &stratum.x,
                    );
                    let d0 = scratch.d0[tied];
                    if d0 <= 0.0 || !d0.is_finite() {
                        failed = true;
                        return;
                    }
                    loglik -= d0.ln();
                    for a in 0..p {
                        d1[a] = scratch.d1[tied * p + a] / d0;
                        score[a] -= d1[a];
                    }
                    for a in 0..p {
                        for b in 0..=a {
                            let d2 = scratch.d2[tied * p * p + a * p + b];
                            information[a * p + b] += d2 / d0 - d1[a] * d1[b];
                        }
                    }
                    return;
                }
                for (u, fraction) in stratum.terms(position, self.ties) {
                    let d0 = at.risk.m0 - fraction * at.dying.m0;
                    if d0 <= 0.0 || !d0.is_finite() {
                        failed = true;
                        return;
                    }
                    loglik -= u * d0.ln();
                    for a in 0..p {
                        d1[a] = at.risk.m1[a] - fraction * at.dying.m1[a];
                        score[a] -= u * d1[a] / d0;
                    }
                    for a in 0..p {
                        for b in 0..=a {
                            let d2 = at.risk.m2[a * p + b] - fraction * at.dying.m2[a * p + b];
                            information[a * p + b] += u * (d2 / d0 - (d1[a] / d0) * (d1[b] / d0));
                        }
                    }
                }
            });
            if failed {
                return None;
            }
        }
        for a in 0..p {
            for b in 0..a {
                information[b * p + a] = information[a * p + b];
            }
        }
        let finite = loglik.is_finite()
            && score.iter().all(|v| v.is_finite())
            && information.iter().all(|v| v.is_finite());
        finite.then_some(Evaluation {
            loglik,
            score,
            information,
        })
    }

    /// The score residuals summed within cluster, `p` per cluster.
    fn cluster_scores(&self, beta: &[f64]) -> Option<Vec<f64>> {
        let p = self.p;
        let mut totals = vec![0.0f64; self.n_clusters * p];
        self.score_rows(beta, |stratum, j, residual| {
            let cluster = stratum.cluster[j] as usize;
            for (total, r) in totals[cluster * p..(cluster + 1) * p]
                .iter_mut()
                .zip(residual)
            {
                *total += r;
            }
        })?;
        Some(totals)
    }

    /// Hand the score residual of every row to `emit`, as the stratum, the
    /// position in it and the `p` values, in stratum order and then in the
    /// order of the stratum. `None` when a risk score or a denominator is not
    /// finite or not positive.
    fn score_rows(
        &self,
        beta: &[f64],
        mut emit: impl FnMut(&Stratum, usize, &[f64]),
    ) -> Option<()> {
        let p = self.p;
        let mut residual = vec![0.0f64; p];
        let mut active = Vec::new();
        let mut sums = Sums {
            risk: Moments::new(p),
            dying: Moments::new(p),
        };
        for stratum in &self.strata {
            let (_, risk) = Self::row_risk(stratum, p, beta)?;
            let count = stratum.event_times.len();
            // Per event time: `hazard` is the sum of weight / d0 over the
            // terms, `centre` the sum of weight * d1 / d0^2, `late_hazard` and
            // `late_centre` the same two sums with the fraction as a factor,
            // and `mean` the mean of d1 / d0.
            let mut hazard = vec![0.0f64; count];
            let mut late_hazard = vec![0.0f64; count];
            let mut centre = vec![0.0f64; count * p];
            let mut late_centre = vec![0.0f64; count * p];
            let mut mean = vec![0.0f64; count * p];
            let mut failed = false;
            stratum.sweep(p, &risk, &mut active, &mut sums, |position, at| {
                let mut terms = 0.0f64;
                for (u, fraction) in stratum.terms(position, self.ties) {
                    let d0 = at.risk.m0 - fraction * at.dying.m0;
                    if d0 <= 0.0 || !d0.is_finite() {
                        failed = true;
                        return;
                    }
                    hazard[position] += u / d0;
                    late_hazard[position] += u * fraction / d0;
                    for a in 0..p {
                        let d1 = at.risk.m1[a] - fraction * at.dying.m1[a];
                        centre[position * p + a] += u * d1 / (d0 * d0);
                        late_centre[position * p + a] += u * fraction * d1 / (d0 * d0);
                        mean[position * p + a] += d1 / d0;
                    }
                    terms += 1.0;
                }
                for a in 0..p {
                    mean[position * p + a] /= terms;
                }
            });
            if failed {
                return None;
            }
            // Prefix sums over the descending event times.
            let mut hazard_sum = vec![0.0f64; count + 1];
            let mut centre_sum = vec![0.0f64; (count + 1) * p];
            for k in 0..count {
                hazard_sum[k + 1] = hazard_sum[k] + hazard[k];
                for a in 0..p {
                    centre_sum[(k + 1) * p + a] = centre_sum[k * p + a] + centre[k * p + a];
                }
            }
            for (j, &risk_j) in risk.iter().enumerate() {
                let x = &stratum.x[j * p..(j + 1) * p];
                // Event times t with start < t <= stop.
                let first = stratum
                    .event_times
                    .partition_point(|&t| t > stratum.stop[j]);
                let last = stratum
                    .event_times
                    .partition_point(|&t| t > stratum.start[j]);
                let span = hazard_sum[last] - hazard_sum[first];
                for a in 0..p {
                    let around = centre_sum[last * p + a] - centre_sum[first * p + a];
                    let mut value = -risk_j * (x[a] * span - around);
                    if stratum.is_event[j] {
                        // The row has its event at `stop`, an event time of the
                        // stratum.
                        value += stratum.weight[j] * (x[a] - mean[first * p + a]);
                        value += risk_j * (x[a] * late_hazard[first] - late_centre[first * p + a]);
                    }
                    residual[a] = value;
                }
                emit(stratum, j, &residual);
            }
        }
        Some(())
    }
}

/// The `L D L'` factorisation of `coxph` with its tolerance. `None` when a
/// pivot is not finite or is clearly negative, otherwise the number of pivots
/// kept.
fn factorise(a: &mut [f64], p: usize) -> Option<usize> {
    let pivots = ldl::factorise(a, p, TOLER_CHOL);
    pivots.clean.then_some(pivots.rank)
}

/// The model on the scaled covariates, and the factor that scales each
/// column: a column's coefficient on the scale given is its coefficient on the
/// scaled column times the factor, and a column without variation has factor 0.
struct Prepared {
    model: Model,
    factor: Vec<f64>,
}

/// Centre and scale the covariates of the rows `kept`, which are not empty,
/// and sort them into strata.
fn prepare(input: &Input<'_>, kept: &[usize]) -> Prepared {
    let p = input.x.ncols();
    // Centre and scale as coxph does.
    let total_weight: f64 = kept.iter().map(|&i| input.weight[i]).sum();
    let mut centre = vec![0.0f64; p];
    let mut factor = vec![0.0f64; p];
    for a in 0..p {
        let first = input.x[(kept[0], a)];
        let constant = kept.iter().all(|&i| input.x[(i, a)] - first == 0.0);
        if constant {
            continue;
        }
        let mean: f64 = kept
            .iter()
            .map(|&i| input.weight[i] * input.x[(i, a)])
            .sum::<f64>()
            / total_weight;
        let deviation: f64 = kept
            .iter()
            .map(|&i| input.weight[i] * (input.x[(i, a)] - mean).abs())
            .sum::<f64>()
            / total_weight;
        centre[a] = mean;
        factor[a] = if deviation > 0.0 {
            1.0 / deviation
        } else {
            1.0
        };
    }

    // The partial likelihood does not change when a constant is added to every
    // offset. The mean is taken out, as `coxph` does, to keep `exp` in range.
    let shift = input
        .offset
        .map(|offset| kept.iter().map(|&i| offset[i]).sum::<f64>() / kept.len() as f64);
    let mut order = kept.to_vec();
    order.sort_by_key(|&i| input.stratum[i]);
    let mut strata = Vec::new();
    let mut begin = 0;
    while begin < order.len() {
        let code = input.stratum[order[begin]];
        let end = begin
            + order[begin..]
                .iter()
                .take_while(|&&i| input.stratum[i] == code)
                .count();
        let members = &mut order[begin..end];
        begin = end;
        if !members.iter().any(|&i| input.is_event[i]) {
            // A stratum without events has an unidentified baseline hazard and
            // adds nothing.
            continue;
        }
        // Stable, so rows with the same exit keep their input order.
        members.sort_by(|&a, &b| input.stop[b].total_cmp(&input.stop[a]));
        let size = members.len();
        let mut by_start: Vec<u32> = (0..size as u32).collect();
        by_start.sort_by(|&a, &b| {
            input.start[members[b as usize]].total_cmp(&input.start[members[a as usize]])
        });
        let start_sorted = by_start
            .iter()
            .map(|&row| input.start[members[row as usize]])
            .collect();
        let mut event_times = Vec::new();
        let mut dying_rows = Vec::new();
        let mut dying_offset = vec![0usize];
        for (position, &i) in members.iter().enumerate() {
            if !input.is_event[i] {
                continue;
            }
            if event_times.last() != Some(&input.stop[i]) {
                if !event_times.is_empty() {
                    dying_offset.push(dying_rows.len());
                }
                event_times.push(input.stop[i]);
            }
            dying_rows.push(position as u32);
        }
        dying_offset.push(dying_rows.len());
        let mut x = Vec::with_capacity(size * p);
        for &i in members.iter() {
            for a in 0..p {
                x.push(if factor[a] == 0.0 {
                    0.0
                } else {
                    (input.x[(i, a)] - centre[a]) * factor[a]
                });
            }
        }
        strata.push(Stratum {
            x,
            row: members.iter().map(|&i| i as u32).collect(),
            weight: members.iter().map(|&i| input.weight[i]).collect(),
            offset: match (input.offset, shift) {
                (Some(offset), Some(shift)) => members.iter().map(|&i| offset[i] - shift).collect(),
                _ => Vec::new(),
            },
            stop: members.iter().map(|&i| input.stop[i]).collect(),
            start: members.iter().map(|&i| input.start[i]).collect(),
            is_event: members.iter().map(|&i| input.is_event[i]).collect(),
            cluster: input
                .cluster
                .map(|cluster| members.iter().map(|&i| cluster[i]).collect())
                .unwrap_or_default(),
            by_start,
            start_sorted,
            event_times,
            dying_rows,
            dying_offset,
        });
    }
    let n_clusters = input.cluster.map_or(0, |cluster| {
        kept.iter()
            .map(|&i| cluster[i] as usize + 1)
            .max()
            .unwrap_or(0)
    });
    Prepared {
        model: Model {
            p,
            ties: input.ties,
            strata,
            n_clusters,
        },
        factor,
    }
}

/// The sum of the cross-products of the cluster score sums at `beta`, or
/// `None` with fewer than two clusters or when a score is not finite.
fn cluster_meat(
    model: &Model,
    input: &Input<'_>,
    kept: &[usize],
    beta: &[f64],
) -> Option<Mat<f64>> {
    let cluster = input.cluster?;
    let p = model.p;
    let mut present = vec![false; model.n_clusters];
    for &i in kept {
        present[cluster[i] as usize] = true;
    }
    if present.iter().filter(|&&seen| seen).count() < 2 {
        return None;
    }
    let totals = model.cluster_scores(beta)?;
    let sums = Mat::from_fn(model.n_clusters, p, |g, a| totals[g * p + a]);
    // Every cluster is a row of `sums` already, so the meat is the plain
    // cross-product of the rows, `sum_g U_g U_g'`.
    Some(
        meat(sums.as_ref(), |_| 1.0, None)
            .expect("the meat of rows without clusters has no error to report"),
    )
}

/// `y' A^- y` for the symmetric `p` by `p` matrix `a` with the pivot rule of
/// `coxph`, and the rank of that rule, or `None` when `a` is not positive
/// semi-definite.
fn quadratic_form(a: &[f64], p: usize, y: &[f64]) -> Option<(f64, usize)> {
    let mut factor = a.to_vec();
    let rank = factorise(&mut factor, p)?;
    let mut z = y.to_vec();
    ldl::solve(&factor, p, &mut z);
    Some((y.iter().zip(&z).map(|(a, b)| a * b).sum(), rank))
}

/// A test statistic with `df` degrees of freedom, or `None` when the degrees
/// of freedom are not those of the fit, the statistic is negative or it is not
/// finite.
fn chi_square(statistic: f64, df: usize, expected_df: usize) -> Option<ChiSquareTest> {
    if df == expected_df && df > 0 {
        ChiSquareTest::new(statistic, df)
    } else {
        None
    }
}

/// The score tests at the starting coefficients `beta0`, scaled. `score` and
/// `solved` are the score and the solution of `I z = U` there, and `rank` the
/// rank of the information.
fn score_tests(
    model: &Model,
    input: &Input<'_>,
    kept: &[usize],
    beta0: &[f64],
    score: &[f64],
    solved: &[f64],
    rank: usize,
) -> (Option<ChiSquareTest>, Option<ChiSquareTest>) {
    let p = model.p;
    let plain = chi_square(
        score.iter().zip(solved).map(|(a, b)| a * b).sum(),
        rank,
        rank,
    );
    let robust = cluster_meat(model, input, kept, beta0).and_then(|meat| {
        let flat: Vec<f64> = (0..p * p).map(|k| meat[(k / p, k % p)]).collect();
        let (statistic, df) = quadratic_form(&flat, p, score)?;
        chi_square(statistic, df, rank)
    });
    (plain, robust)
}

fn fit_checked(input: &Input<'_>) -> Result<CoxModelFit> {
    let p = input.x.ncols();
    let with_clusters = input.cluster.is_some();
    let kept: Vec<usize> = (0..input.x.nrows())
        .filter(|&i| input.weight[i] > 0.0)
        .collect();
    if kept.is_empty() {
        return Ok(CoxModelFit::degenerate(p, 0.0, 0, with_clusters));
    }
    let n_rows = kept.len();
    let n_events: f64 = kept
        .iter()
        .filter(|&&i| input.is_event[i])
        .map(|&i| input.weight[i])
        .sum();
    if !kept.iter().any(|&i| input.is_event[i]) {
        return Ok(CoxModelFit::degenerate(p, n_events, n_rows, with_clusters));
    }

    let Prepared { model, factor } = prepare(input, &kept);
    let undefined = || CoxModelFit::degenerate(p, n_events, n_rows, with_clusters);

    // The starting coefficients on the scaled covariates.
    let mut beta: Vec<f64> = input.init.map_or_else(
        || vec![0.0f64; p],
        |init| {
            init.iter()
                .zip(&factor)
                .map(|(b, &f)| if f == 0.0 { 0.0 } else { b / f })
                .collect()
        },
    );
    let beta0 = beta.clone();
    let Some(start) = model.evaluate(&beta) else {
        if input.init.is_some() {
            return Err(Error::input(
                "init: the starting coefficients give a risk score or a partial \
                 likelihood that is not finite",
            ));
        }
        return Ok(undefined());
    };
    let null_loglik = if input.init.is_some() {
        model.evaluate(&vec![0.0f64; p]).map(|e| e.loglik)
    } else {
        Some(start.loglik)
    }
    .filter(|v| v.is_finite());
    if p == 0 {
        // The model without covariates is its null model: nothing is iterated
        // or tested.
        return Ok(CoxModelFit {
            coefficients: Vec::new(),
            standard_errors: Vec::new(),
            covariance: Vec::new(),
            robust_standard_errors: with_clusters.then(Vec::new),
            robust_covariance: with_clusters.then(Vec::new),
            log_partial_likelihood: Some(start.loglik),
            null_log_partial_likelihood: null_loglik,
            tests: CoxTests::default(),
            n_events,
            n_rows,
            n_used: n_rows,
            rank: 0,
            iterations: 0,
            converged: true,
            monotone_columns: Vec::new(),
        });
    }
    let initial_loglik = start.loglik;
    let mut accepted_loglik = start.loglik;
    let mut matrix = start.information.clone();
    let Some(initial_rank) = factorise(&mut matrix, p) else {
        return Ok(undefined());
    };
    if initial_rank == 0 {
        return Ok(undefined());
    }
    let mut step = start.score.clone();
    ldl::solve(&matrix, p, &mut step);
    let (score_test, robust_score_test) = score_tests(
        &model,
        input,
        &kept,
        &beta0,
        &start.score,
        &step,
        initial_rank,
    );
    let mut trial: Vec<f64> = beta.iter().zip(&step).map(|(b, s)| b + s).collect();

    let stepping = input.stepping;
    let mut halvings = 0u32;
    let mut iterations = 0u32;
    let mut converged = false;
    let mut last: Option<Evaluation> = None;
    if input.iter_max == 0 {
        // The model at the starting coefficients.
        last = Some(start);
        trial.clone_from(&beta);
    }
    for count in 1..=input.iter_max {
        iterations = count;
        let evaluation = model.evaluate(&trial);
        // Counting-process data also treat a change of the rank of the
        // information as a failed step.
        let rank_changed = stepping == Stepping::Counting
            && evaluation.as_ref().is_some_and(|e| {
                let mut m = e.information.clone();
                factorise(&mut m, p) != Some(initial_rank)
            });
        let halving = halvings > 0 && stepping != Stepping::RightCensored;
        if let Some(e) = &evaluation
            && !halving
            && !rank_changed
            && (1.0 - accepted_loglik / e.loglik).abs() <= input.eps
        {
            converged = true;
            last = evaluation;
            break;
        }
        let better = !rank_changed && matches!(&evaluation, Some(e) if e.loglik >= accepted_loglik);
        if count == input.iter_max {
            // Out of iterations.
            let keep_trial = match (&evaluation, stepping) {
                (None, _) => false,
                (Some(_), Stepping::Legacy) => true,
                (Some(_), Stepping::ExactRight) => input.iter_max == 1,
                (Some(e), Stepping::Counting) => {
                    input.iter_max == 1
                        || (e.loglik - accepted_loglik) / accepted_loglik.abs() >= -input.eps
                }
                (Some(_), Stepping::RightCensored) => better,
            };
            if keep_trial || stepping == Stepping::Legacy {
                last = evaluation;
            } else {
                trial.clone_from(&beta);
                last = model.evaluate(&beta);
            }
            break;
        }
        if let (true, Some(e)) = (better, evaluation) {
            halvings = 0;
            accepted_loglik = e.loglik;
            beta.clone_from(&trial);
            matrix = e.information;
            if factorise(&mut matrix, p).is_none() {
                let mut fit = undefined();
                fit.iterations = count;
                return Ok(fit);
            }
            step = e.score;
            ldl::solve(&matrix, p, &mut step);
            for (t, s) in trial.iter_mut().zip(&step) {
                *t += s;
            }
        } else {
            // Step halving. Each halving in a row takes the trial point a
            // larger share of the way back to the last accepted point, as
            // `coxph` does. The first fits kept half way back each time.
            halvings += 1;
            let h = if matches!(stepping, Stepping::Legacy | Stepping::ExactRight) {
                1.0
            } else {
                f64::from(halvings)
            };
            for (t, b) in trial.iter_mut().zip(&beta) {
                *t = (*t + h * b) / (h + 1.0);
            }
        }
    }
    let Some(last) = last else {
        let mut fit = undefined();
        fit.iterations = iterations;
        return Ok(fit);
    };

    let mut final_matrix = last.information.clone();
    let Some(rank) = factorise(&mut final_matrix, p) else {
        let mut fit = undefined();
        fit.iterations = iterations;
        return Ok(fit);
    };
    if rank == 0 {
        let mut fit = undefined();
        fit.iterations = iterations;
        return Ok(fit);
    }
    let aliased = |a: usize| final_matrix[a * p + a] == 0.0;

    // The coefficient of a column whose Newton step is still large when the
    // likelihood has stopped changing is not finite.
    let mut newton = last.score.clone();
    ldl::solve(&final_matrix, p, &mut newton);
    let monotone_columns: Vec<usize> = if converged && input.ties == Ties::Exact {
        // The rule of `coxexact.fit` and `agexact.fit`, with `toler.inf` its
        // default `sqrt(eps)`, and no check for `iter.max` of 1 or less.
        (0..p)
            .filter(|&a| {
                let step = newton[a].abs();
                input.iter_max > 1
                    && !aliased(a)
                    && step > input.eps
                    && step > input.eps.sqrt() * trial[a].abs()
            })
            .collect()
    } else if converged {
        (0..p)
            .filter(|&a| !aliased(a) && newton[a].abs() > input.eps + input.eps.sqrt())
            .collect()
    } else {
        Vec::new()
    };
    if !monotone_columns.is_empty() {
        let mut fit = undefined();
        fit.iterations = iterations;
        fit.null_log_partial_likelihood = null_loglik;
        fit.monotone_columns = monotone_columns;
        fit.tests.score = score_test;
        fit.tests.robust_score = robust_score_test;
        return Ok(fit);
    }

    let variance = ldl::inverse(&final_matrix, p);
    let coefficients: Vec<Option<f64>> = (0..p)
        .map(|a| (!aliased(a)).then(|| trial[a] * factor[a]))
        .collect();
    let standard_errors: Vec<Option<f64>> = (0..p)
        .map(|a| (!aliased(a)).then(|| variance[a * p + a].sqrt() * factor[a]))
        .collect();
    let covariance = on_input_scale(&variance, &factor, &aliased);

    let robust = cluster_meat(&model, input, &kept, &trial).map(|meat| {
        let sandwich = sandwich(&variance, &meat, p);
        (
            (0..p)
                .map(|a| (!aliased(a)).then(|| sandwich[a * p + a].sqrt() * factor[a]))
                .collect::<Vec<_>>(),
            on_input_scale(&sandwich, &factor, &aliased),
            sandwich,
        )
    });

    // The Wald statistics are those of the distance from the starting
    // coefficients, on the scaled covariates, over the columns that are not
    // aliased. The model-based statistic is `d' I d` since `V` is the inverse
    // of `I` on those columns.
    let distance: Vec<f64> = (0..p)
        .map(|a| if aliased(a) { 0.0 } else { trial[a] - beta0[a] })
        .collect();
    let mut wald = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            if !aliased(a) && !aliased(b) {
                wald += distance[a] * last.information[a * p + b] * distance[b];
            }
        }
    }
    let robust_wald = robust.as_ref().and_then(|(_, _, sandwich)| {
        let (statistic, df) = quadratic_form(sandwich, p, &distance)?;
        chi_square(statistic, df, rank)
    });
    let tests = CoxTests {
        wald: chi_square(wald, rank, rank),
        robust_wald,
        score: score_test,
        robust_score: robust_score_test,
        likelihood_ratio: chi_square(2.0 * (last.loglik - initial_loglik), rank, rank),
    };

    let (robust_standard_errors, robust_covariance) = match (with_clusters, robust) {
        (true, None) => (Some(vec![None; p]), Some(vec![None; p * p])),
        (_, Some((se, covariance, _))) => (Some(se), Some(covariance)),
        (false, None) => (None, None),
    };

    Ok(CoxModelFit {
        coefficients,
        standard_errors,
        covariance,
        robust_standard_errors,
        robust_covariance,
        log_partial_likelihood: Some(last.loglik),
        null_log_partial_likelihood: null_loglik,
        tests,
        n_events,
        n_rows,
        n_used: n_rows,
        rank,
        iterations,
        converged,
        monotone_columns,
    })
}

/// `V M V'` for the symmetric `p` by `p` matrices `variance` and `meat`, as a
/// row-major vector.
fn sandwich(variance: &[f64], meat: &Mat<f64>, p: usize) -> Vec<f64> {
    let inverse = Mat::from_fn(p, p, |a, b| variance[a * p + b]);
    let v = sandwich_with_inverse(inverse.as_ref(), meat.as_ref());
    (0..p * p).map(|k| v[(k / p, k % p)]).collect()
}

/// A covariance matrix of the scaled coefficients on the scale of the columns
/// given, `None` in the rows and columns of an aliased column.
fn on_input_scale(
    scaled: &[f64],
    factor: &[f64],
    aliased: &impl Fn(usize) -> bool,
) -> Vec<Option<f64>> {
    let p = factor.len();
    (0..p * p)
        .map(|k| {
            let (a, b) = (k / p, k % p);
            (!aliased(a) && !aliased(b)).then(|| scaled[k] * (factor[a] * factor[b]))
        })
        .collect()
}

/// The score residual of every row at `coefficients`, one vector of `rows`
/// values per column, as `residuals(fit, type = "score")` of `survival`.
///
/// Row `i` of column `j` is the derivative of the log partial likelihood with
/// respect to the coefficient of column `j` through the case weight of row
/// `i`, on the scale of the columns given. The rows are in the order of the
/// input. A row whose weight is zero, and a row in a stratum without events,
/// has residuals of zero. A column whose coefficient is `None`, such as one
/// that is aliased, is `None`.
///
/// `coefficients` is normally `CoxModelFit::coefficients`. Any other value is
/// the residual at that point, with a column left out where it is `None`. The
/// clusters of `data` are not used. `data`, `x` and `opts` are read as in
/// [`fit`], so the residuals are those of the model that `fit` estimates.
///
/// The residuals contain the case weight; see the module documentation for
/// how they relate to `residuals(fit, "score")` of `survival`.
///
/// # Errors
///
/// [`Error::Input`] for the invalid input of [`fit`], for `coefficients` of
/// another length than `x` has columns, for a coefficient that is not finite,
/// and for coefficients at which a risk score or the denominator of a
/// partial-likelihood term is not finite and positive.
pub fn score_residuals(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    coefficients: &[Option<f64>],
    opts: &CoxOptions<'_>,
) -> Result<Vec<Option<Vec<f64>>>> {
    if opts.ties == Ties::Exact {
        return Err(Error::input(
            "ties: score residuals are not available for the exact method",
        ));
    }
    with_input(data, x, opts, |input| {
        score_residuals_checked(input, coefficients)
    })?
}

/// The score residual of every row at `coefficients`, one vector of `rows`
/// values per column, as `residuals(fit, type = "score")` of `survival`.
///
/// Row `i` of column `j` is the derivative of the log partial likelihood with
/// respect to the coefficient of column `j` through the case weight of row
/// `i`, on the scale of the columns given. The rows are in the order of the
/// input. A row whose weight is zero, and a row in a stratum without events,
/// has residuals of zero. A column whose coefficient is `None`, such as one
/// that is aliased, is `None`.
///
/// `coefficients` is normally `CoxModelFit::coefficients`. Any other value is
/// the residual at that point, with a column left out where it is `None`. The
/// `cluster` of `data` is not used.
///
/// The residuals contain the case weight; see the module documentation for
/// how they relate to `residuals(fit, "score")` of `survival`. Times are
/// compared exactly, which gives the numbers of 0.2; [`score_residuals`] is the
/// form that applies the tie rule.
///
/// # Errors
///
/// [`Error::Input`] for the invalid input of [`fit_cox`], for `coefficients`
/// of another length than the design has columns, for a coefficient that is
/// not finite, and for coefficients at which a risk score or the denominator
/// of a partial-likelihood term is not finite and positive.
#[deprecated(
    since = "0.3.0",
    note = "build a `SurvData` and call `cox::score_residuals` with a `CoxOptions`"
)]
#[expect(deprecated, reason = "the wrapper reads the deprecated input type")]
pub fn cox_score_residuals(
    data: &CoxDesign<'_>,
    coefficients: &[Option<f64>],
) -> Result<Vec<Option<Vec<f64>>>> {
    check_design(data)?;
    score_residuals_checked(&data.into(), coefficients)
}

fn score_residuals_checked(
    data: &Input<'_>,
    coefficients: &[Option<f64>],
) -> Result<Vec<Option<Vec<f64>>>> {
    let (rows, p) = (data.x.nrows(), data.x.ncols());
    if coefficients.len() != p {
        return Err(Error::input(format!(
            "coefficients: {} values for {p} columns",
            coefficients.len()
        )));
    }
    if coefficients.iter().flatten().any(|c| !c.is_finite()) {
        return Err(Error::input("coefficients: a value is not finite"));
    }
    let mut columns: Vec<Option<Vec<f64>>> = coefficients
        .iter()
        .map(|c| c.map(|_| vec![0.0f64; rows]))
        .collect();
    let kept: Vec<usize> = (0..rows).filter(|&i| data.weight[i] > 0.0).collect();
    if kept.is_empty() {
        return Ok(columns);
    }
    let Prepared { model, factor } = prepare(data, &kept);
    // A column without variation has a scaled column of zeros and a residual
    // of zero whatever its coefficient.
    let beta: Vec<f64> = coefficients
        .iter()
        .zip(&factor)
        .map(|(c, &f)| if f == 0.0 { 0.0 } else { c.unwrap_or(0.0) / f })
        .collect();
    model
        .score_rows(&beta, |stratum, j, residual| {
            let row = stratum.row[j] as usize;
            for (a, column) in columns.iter_mut().enumerate() {
                if let Some(column) = column {
                    column[row] = if factor[a] == 0.0 {
                        0.0
                    } else {
                        residual[a] / factor[a]
                    };
                }
            }
        })
        .ok_or_else(|| {
            Error::input(
                "coefficients: a risk score or a denominator of the partial likelihood \
                 is not finite and positive at these values",
            )
        })?;
    Ok(columns)
}

#[cfg(test)]
#[expect(deprecated, reason = "the tests of the deprecated entry points")]
mod tests {
    use epistat_linalg::Mat;

    use super::{CoxDesign, cox_score_residuals, fit_cox};
    use crate::cox::Ties;

    const STOP: [f64; 8] = [1., 2., 2., 3., 4., 5., 6., 7.];
    const EVENT: [bool; 8] = [true, true, true, false, true, true, false, true];

    fn design(x: &Mat<f64>, cluster: Option<&[u32]>) -> super::Result<super::CoxModelFit> {
        fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Efron,
            cluster,
        })
    }

    fn column(values: [f64; 8]) -> Mat<f64> {
        Mat::from_fn(8, 1, |i, _| values[i])
    }

    #[test]
    fn one_cluster_has_no_robust_standard_error() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = design(&x, Some(&[0; 8])).unwrap();
        assert!(fit.converged && fit.coefficients[0].is_some());
        assert_eq!(fit.robust_standard_errors, Some(vec![None]));
    }

    #[test]
    fn a_column_without_variation_is_aliased() {
        let x = Mat::from_fn(8, 2, |i, j| {
            if j == 0 {
                [0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3][i]
            } else {
                0.1
            }
        });
        let fit = design(&x, None).unwrap();
        assert_eq!(fit.rank, 1);
        assert!(fit.coefficients[0].is_some() && fit.coefficients[1].is_none());
        assert!(fit.standard_errors[1].is_none());
    }

    #[test]
    fn no_events_gives_no_estimate() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &[false; 8],
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Breslow,
            cluster: None,
        })
        .unwrap();
        assert_eq!(fit.coefficients, vec![None]);
        assert!(!fit.converged && fit.iterations == 0);
    }

    #[test]
    fn invalid_input_is_an_error() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, f64::NAN]);
        assert!(
            design(&x, None)
                .unwrap_err()
                .to_string()
                .starts_with("x: 1 non-finite")
        );
        let none = Mat::<f64>::zeros(8, 0);
        assert_eq!(
            design(&none, None).unwrap_err().to_string(),
            "x has no columns"
        );
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let mut weight = [1.0; 8];
        weight[3] = -1.0;
        let error = fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &weight,
            stratum: &[0; 8],
            ties: Ties::Breslow,
            cluster: None,
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "weight: row 3 is negative");
        let error = fit_cox(&CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[0., 0., 2., 0., 0., 0., 0., 0.],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Breslow,
            cluster: None,
        })
        .unwrap_err();
        assert!(error.to_string().starts_with("start: row 2"));
    }

    #[test]
    fn covariance_matrices_are_symmetric_and_hold_the_standard_errors() {
        let x = Mat::from_fn(8, 3, |i, j| match j {
            0 => [0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3][i],
            1 => [1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 0.5][i],
            _ => 0.1,
        });
        let fit = design(&x, Some(&[0, 0, 1, 1, 2, 2, 3, 3])).unwrap();
        let robust = fit.robust_covariance.as_ref().unwrap();
        let robust_se = fit.robust_standard_errors.as_ref().unwrap();
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(fit.covariance[a * 3 + b], fit.covariance[b * 3 + a]);
                assert_eq!(robust[a * 3 + b], robust[b * 3 + a]);
                assert_eq!(
                    fit.covariance[a * 3 + b].is_none(),
                    a == 2 || b == 2,
                    "the constant column is aliased"
                );
            }
            if let (Some(variance), Some(se)) = (fit.covariance[a * 3 + a], fit.standard_errors[a])
            {
                assert!((variance.sqrt() - se).abs() <= 4.0 * f64::EPSILON * se);
            }
            if let (Some(variance), Some(se)) = (robust[a * 3 + a], robust_se[a]) {
                assert!((variance.sqrt() - se).abs() <= 4.0 * f64::EPSILON * se);
            }
        }
    }

    #[test]
    fn one_cluster_has_no_robust_covariance() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = design(&x, Some(&[0; 8])).unwrap();
        assert_eq!(fit.robust_covariance, Some(vec![None]));
        assert!(design(&x, None).unwrap().robust_covariance.is_none());
    }

    #[test]
    fn score_residuals_add_up_to_the_score() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let fit = design(&x, None).unwrap();
        // At the estimate the score is zero, and at zero it is the sum of the
        // residuals there, which is the first Newton step times the information.
        let at_estimate = cox_score_residuals(&data(&x), &fit.coefficients).unwrap();
        let sum: f64 = at_estimate[0].as_ref().unwrap().iter().sum();
        assert!(sum.abs() < 1e-8, "{sum}");
        let at_zero = cox_score_residuals(&data(&x), &[Some(0.0)]).unwrap();
        let sum: f64 = at_zero[0].as_ref().unwrap().iter().sum();
        assert!(sum.abs() > 0.1);
    }

    #[test]
    fn score_residuals_of_dropped_columns_and_rows() {
        let x = Mat::from_fn(8, 2, |i, j| {
            if j == 0 {
                [0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3][i]
            } else {
                0.1
            }
        });
        let fit = design(&x, None).unwrap();
        let mut weight = [1.0; 8];
        weight[4] = 0.0;
        let residuals = cox_score_residuals(
            &CoxDesign {
                weight: &weight,
                ..data(&x)
            },
            &fit.coefficients,
        )
        .unwrap();
        assert!(residuals[1].is_none());
        assert_eq!(
            residuals[0].as_ref().unwrap()[4].to_bits(),
            0.0f64.to_bits()
        );
    }

    #[test]
    fn score_residuals_refuse_invalid_coefficients() {
        let x = column([0.5, 1.5, -1.0, 0.2, 2.0, -0.7, 1.1, 0.3]);
        let wrong = cox_score_residuals(&data(&x), &[Some(0.1), Some(0.2)]).unwrap_err();
        assert_eq!(wrong.to_string(), "coefficients: 2 values for 1 columns");
        let nan = cox_score_residuals(&data(&x), &[Some(f64::NAN)]).unwrap_err();
        assert_eq!(nan.to_string(), "coefficients: a value is not finite");
        let huge = cox_score_residuals(&data(&x), &[Some(1e4)]).unwrap_err();
        assert!(huge.to_string().starts_with("coefficients: a risk score"));
        let none = cox_score_residuals(&data(&x), &[None]).unwrap();
        assert_eq!(none, vec![None]);
    }

    fn data(x: &Mat<f64>) -> CoxDesign<'_> {
        CoxDesign {
            x: x.as_ref(),
            is_event: &EVENT,
            start: &[f64::NEG_INFINITY; 8],
            stop: &STOP,
            weight: &[1.0; 8],
            stratum: &[0; 8],
            ties: Ties::Efron,
            cluster: None,
        }
    }
}
