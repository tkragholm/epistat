//! Shared gamma frailty Cox models, as `survival::coxph` with
//! `frailty(cluster, distribution = "gamma")`.
//!
//! # The model
//!
//! Row `j` of cluster `i` has the hazard `z_i h_0(t) exp(x_j' beta + o_j)`,
//! where the frailty `z_i` is gamma distributed with mean 1 and variance
//! `theta`, shared by the rows of the cluster \[1\] \[2\]. The coefficients are
//! conditional on the cluster. `theta` measures the dependence within clusters,
//! and Kendall's tau of two members of a cluster is `theta / (theta + 2)`
//! \[6\]. Clusters are the cluster column of the [`SurvData`], which is
//! required. Strata give each stratum its own `h_0` with the frailties shared
//! across strata, entry times make rows (start, stop] intervals of a counting
//! process, and case weights and an offset enter as in [`crate::cox::fit`].
//!
//! # Penalised partial likelihood
//!
//! With `w_i = ln z_i` as parameters and `nu = 1 / theta`, the fit maximises
//!
//! ```text
//! PPL(beta, w) = l(beta, w) - nu sum_i (e^{w_i} - w_i),
//! ```
//!
//! where `l` is the Cox partial log likelihood (Breslow or Efron) with linear
//! predictor `x_j' beta + o_j + w_{i(j)}` \[4\] \[5\]. `l` does not change when
//! the same constant is added to every `w_i`, and the penalty is smallest when
//! the frailties have mean 1, so after every step the `w_i` are shifted to
//! `mean(e^w) = 1`, which is also where the maximum lies. At the maximum the
//! score of `w_i` gives `e^{w_i} = (nu + D_i) / (nu + Lambda_i)`, with `D_i` the
//! weighted events of the cluster and `Lambda_i` its Breslow cumulative hazard
//! without the frailty, which is the E-step of the EM algorithm below, so for
//! Breslow ties the two fits have the same solution \[5\].
//!
//! # Score and information
//!
//! At an event time `t` of a stratum with `d` events, Efron's approximation has
//! terms `k = 0, ..., d - 1` with weight `u` (the mean case weight of the events)
//! and fraction `f_k = k / d`, Breslow's one term with `u` the summed weight and
//! `f = 0`. With `r_j = c_j exp(eta_j)` and the sums `S_0`, `S_1`, `S_2` over the
//! risk set and `E_0`, `E_1`, `E_2` over the events at `t`, a term has the
//! denominators `D_r = S_r - f E_r`, as in [`crate::cox`]. The coefficient
//! block of the information is that of the Cox model. For the frailties, form at
//! each event time
//!
//! ```text
//! h = sum_k u / D_0          g = sum_k u f_k / D_0
//! k1 = sum_k u / D_0^2       k2 = sum_k u f_k / D_0^2     k3 = sum_k u f_k^2 / D_0^2
//! m1 = sum_k u D_1 / D_0^2   m2 = sum_k u f_k D_1 / D_0^2
//! ```
//!
//! and their sums `H`, `M_1` and `K_1` over the event times of the stratum. A row
//! is in the risk set of every term at a time except its own event time, where
//! Efron's `k`-th term counts it with weight `1 - f_k`. With `dH_j` the sum of `h`
//! over the times the row is at risk less `g` at its own event time, and `dM_j`
//! likewise from `m1` and `m2`, both differences of the running sums:
//!
//! ```text
//! U_{w_i}         = sum_{j in i} (c_j delta_j - r_j dH_j) - nu (e^{w_i} - 1)
//! I_{beta, w_i}   = sum_{j in i} r_j (x_j dH_j - dM_j)
//! I_{w_i, w_i}    = sum_{j in i} r_j dH_j - sum_t (k1 R_i^2 - 2 k2 R_i E_i + k3 E_i^2) + nu e^{w_i}
//! I_{w_i, w_m}    = - sum_t sum_k u (R_i - f_k E_i)(R_m - f_k E_m) / D_0^2
//! ```
//!
//! with `R_i(t)` the risk-set total of cluster `i` and `E_i(t)` its total over
//! the events at `t`. `R_i` is constant between the entry and exit times of the
//! cluster's own rows, so the sum over `t` of `k1 R_i^2` is a sum over those
//! segments of `R_i^2` times the increase of `K_1`, a pass over the cluster's
//! rows. The frailty block is dense, but its product with a vector `v` needs
//! only one pass: `q(t) = sum` over the risk set of `r_j v_{i(j)}` is a running
//! sum, and the product for cluster `i` is a difference of running sums of
//! `q k1` and `q k2` over the times its rows are at risk. Nothing of size `G^2`
//! is formed.
//!
//! # Algorithm
//!
//! Newton's method on `PPL` with the whole information `H`. The step `H^{-1} U`
//! is solved by conjugate gradients on the `p + G` parameters, preconditioned by
//! the information with its frailty block replaced by its diagonal `Delta`,
//! which is solved exactly through the Schur complement
//! `A - B' Delta^{-1} B` of the coefficient block `A` and the cross block `B`.
//! The off-diagonal part of the frailty block is small when clusters are small
//! against the risk sets, so few iterations are needed; the solve stops at a
//! relative residual of `1e-12` in the norm of the preconditioner. A step that
//! lowers `PPL` is halved, up to 40 times. The iteration has converged when the
//! decrement `U' H^{-1} U` at the last accepted point is at most
//! `FrailtyOptions::tolerance`; the step it predicts is then taken. The
//! decrement is the squared length of the step in standard errors. At theta = 0
//! the model is the Cox model, fitted by Newton's method on the coefficients
//! alone with the frailties at 0. Each fit of a search starts from the fit at
//! the nearest theta already fitted.
//!
//! `coxph` iterates with the diagonal form of the frailty block (`sparse = TRUE`,
//! its default for more than five clusters) or with the whole information as
//! ordinary penalised columns (`sparse = FALSE`). Both solve the same score
//! equations, so the estimates are the same; the first converges linearly, and
//! the second needs `G^2` memory.
//!
//! # Covariance matrices
//!
//! `covariance` is the coefficient block of `H^{-1}`, found by `p` solves of the
//! conjugate gradient method, `var` of `coxph(sparse = FALSE)`, and
//! `covariance_sandwich` is that of `H^{-1} (H - P) H^{-1}` with `P` the
//! penalty's curvature `nu e^{w_i}`, its `var2`. `diagonal_covariance` is
//! `(A - B' Delta^{-1} B)^{-1}`, the variance of `coxph(sparse = TRUE)`, with its
//! own sandwich and the frailty variances `1 / Delta_i + b_i' V b_i / Delta_i^2`
//! (`fvar`). The two forms differ by a few percent \[5\]; on `kidney` the diagonal
//! form's standard errors are 2 percent larger. Both condition on theta and
//! ignore its uncertainty. When a conjugate gradient solve does not meet its
//! tolerance in 500 iterations the exact covariances are `None`.
//!
//! # The profile likelihood of theta
//!
//! For Breslow ties and case weights of 1 the marginal likelihood, with the
//! frailties integrated out and the baseline hazard profiled out, is
//!
//! ```text
//! sum over events (x'beta + o + ln dH_0) + sum_i [ln Gamma(nu + D_i) - ln Gamma(nu)
//!     + D_i ln theta - (nu + D_i) ln(1 + theta Lambda_i)]
//! ```
//!
//! \[2\] \[3\] \[6\]. At the maximum over `beta` and `h_0` it equals the corrected
//! likelihood of `coxph`,
//!
//! ```text
//! c(theta) = l + nu sum_i w_i + sum_i [D_i - nu ln(1 + D_i / nu) + sum_{m=1}^{D_i} ln(1 - m / (nu + D_i))],
//! ```
//!
//! (`frailty.gammacon`) with `D_i` the unweighted events, up to the constant
//! `sum_k (d_k ln d_k - d_k)` over the event times, which `c` leaves out so that
//! `c(0)` is the Cox partial log likelihood. `marginal_log_likelihood` is `c`, on
//! that scale. The two forms of the correction are computed without
//! cancellation for large `nu`. With Efron ties `c` is the same expression with
//! Efron's `l`, as in `coxph`, and is not a likelihood. With case weights other
//! than 1 there is no marginal likelihood: `marginal_log_likelihood` is `None`,
//! and theta can only be fixed.
//!
//! [`Theta::Profile`] maximises `c` over `ln theta`, bracketing the maximum
//! from theta = 0 and 1 by steps of factors of 4, down to the last power of 4
//! above `1e-6`, about `4e-6`, and up to `1e4`. A fit that fails at a trial
//! theta ends the search with [`Error::Frailty`], which names that theta and
//! holds the failure. When `c` still rises at `1e4` the fit is at the last
//! theta tried and is not an estimate (`ThetaStatus::Unbounded`).
//!
//! With Breslow ties `c` is a profile likelihood, and its derivative in theta
//! is, by the envelope theorem, the partial derivative of the marginal log
//! likelihood above at the fit's coefficients and baseline hazard \[2\] \[5\],
//!
//! ```text
//! dc/dtheta = sum_i [ -nu^2 (psi(nu + D_i) - psi(nu)) + D_i / theta
//!                     + nu^2 ln(1 + theta Lambda_i) - (nu + D_i) Lambda_i / (1 + theta Lambda_i) ],
//! ```
//!
//! with `Lambda_i` from the fit at theta, computed in a form without
//! cancellation as theta goes to 0, where it is the score of Commenges and
//! Andersen, `(1/2) sum_i [(D_i - Lambda_i)^2 - D_i]` from the Cox fit \[10\]
//! (`score_at_zero`). The bracketing follows the sign of this score from
//! theta = 1, and the estimate is its root on `ln theta` by the ITP method
//! ([`epistat_optim::root::itp`]). The estimate is 0 (`ThetaStatus::Boundary`)
//! when the score at 0 is at most 0 and `c` is not higher at the smallest
//! theta of the bracketing than at 0; when the score at 0 is positive and the
//! score is negative down to that theta, the root lies between 0 and it and is
//! found on theta itself. A score at 0 that is not positive with `c` higher at
//! the smallest theta is an error. The profile-likelihood interval
//! (`theta_interval`) is the two roots of `c(theta) - c(theta_hat) + q / 2`,
//! `q` the chi-square quantile on one degree of freedom at
//! `FrailtyOptions::level`, found by the same root finder after the same
//! steps, with the lower limit 0 when `c(0)` is above the cut. The standard
//! error of `ln theta` (`ln_theta_se`) is from the second derivative of `c`
//! in `ln theta` by a central difference of its score with steps of `1e-3`,
//! whose truncation error is of the order of `1e-6` relative.
//!
//! With Efron ties `c` is not a likelihood and the score above does not
//! apply. The search fits theta = 0, 1 and 1/4, steps by factors of 4 until
//! `c` is bracketed and runs Brent's minimiser ([`epistat_optim::brent`]) on
//! the bracket; when `c` still rises towards 0 at about `4e-6`, the estimate
//! is 0, so a maximum below that is reported as 0. Neither interval nor
//! standard error is given.
//!
//! When every cluster has one row the data do not identify theta, and an
//! estimate other than the boundary is an error.
//!
//! `coxph`'s default search stops before the maximum: its estimate is the last
//! theta it tried, on `kidney` with Efron ties 0.4122 where the maximum is at
//! 0.4078. To reproduce a `coxph` fit, pass its reported theta as
//! [`Theta::Fixed`]; every other output then agrees with `coxph`. The
//! likelihood ratio test of theta = 0 compares `c` with the Cox model on the
//! boundary of the parameter space, with the 50:50 mixture of 0 and chi-square
//! on one degree of freedom of Self and Liang \[8\].
//!
//! # EM
//!
//! [`FrailtyMethod::Em`] alternates a Cox fit of [`crate::cox::fit`] with the offset
//! `o_j + ln z_i` (the M-step for `beta` and the Breslow `h_0`) and the E-step
//! `z_i = (nu + D_i) / (nu + Lambda_i)` \[2\] \[3\], and computes the marginal
//! likelihood above from the gamma integral, on the scale of `c`. It stops when
//! the marginal likelihood changes by less than its tolerance, relative. It
//! converges linearly, often in tens of steps, each a Cox fit, and is offered as
//! an independent route to the same estimate and likelihood, for Breslow ties
//! and case weights of 1 only. Its covariance is that of the penalised fit at
//! the EM solution.
//!
//! # Entry times
//!
//! Entry times are the start of the rows' intervals at risk, as in `coxph`. The
//! gamma distribution of the frailties is then that of the clusters as they are
//! observed. This is right for follow-up split into intervals and for
//! time-varying covariates; it is not right when clusters enter late and are
//! selected by having survived to entry, whose frailties are then not gamma
//! with mean 1 \[7\]. The likelihood corrected for such selection is not built.
//!
//! # Results that are not numbers, and invalid input
//!
//! An aliased column, constant or a linear combination of those before it over
//! the rows used (by the `L D L'` factorisation of their weighted correlation
//! matrix with a pivot floor of `1e-10`), has `None` for its coefficient and its
//! covariances. A cluster whose rows all have weight zero has no frailty. A
//! missing cluster column, exact ties, a theta that is negative or not finite,
//! an estimated theta or EM with case weights other than 1, EM with Efron ties,
//! no event, and an information that is not positive definite at a trial point
//! are errors.
//!
//! # Scale
//!
//! An evaluation is one sweep over the rows in each stratum, `O(n p^2)`, a pass
//! over the rows for the frailty terms, `O(n p)`, and a pass over the clusters'
//! rows in time order. A conjugate gradient iteration is `O(n + G p)`. The
//! memory is `O(n p + G p + T p)` with `T` event times. Rows are sorted once.
//! The work is serial and in a fixed order, so the result does not depend on the
//! number of threads. At `10^5` rows in `2.5 10^4` clusters a fit at a fixed
//! theta takes about 0.15 s on an M-series Mac and the profile search 1.3 s.
//!
//! # Agreement with R
//!
//! The tests compare with `coxph` from `survival` 3.8.12 at fixed theta with
//! both variance forms, Breslow and Efron ties, strata, entry times, offsets and
//! case weights, up to 5,000 clusters, and with `optimize` on its corrected
//! likelihood. Coefficients and frailties agree
//! to well within bounds derived from R's stopping rule, about `1e-12` standard
//! errors against the dense fits. Two defects of `coxph`'s C code are avoided
//! there: with case weights the diagonal form of `coxph(sparse = TRUE)` leaves
//! out the case weights of the events from the diagonal of the frailty block,
//! so its standard errors are wrong, and with case weights, Efron ties and
//! entry times `coxph` sums the squared weights into Efron's correction and does
//! not converge; that case is compared with a transcription of the penalised
//! likelihood on `coxph`'s own unpenalised fit.
//!
//! # Not built
//!
//! The log-normal frailty of `coxme`, the standard error and the profile
//! interval of theta for Efron ties, the observed information of theta of
//! Andersen et al. \[9\], the exact variances of the frailties, and the correction
//! for late entry.
//!
//! # References
//!
//! 1. Vaupel JW, Manton KG, Stallard E. The impact of heterogeneity in individual
//!    frailty on the dynamics of mortality. Demography. 1979;16(3):439-54.
//! 2. Nielsen GG, Gill RD, Andersen PK, Sorensen TIA. A counting process approach
//!    to maximum likelihood estimation in frailty models. Scand J Stat.
//!    1992;19(1):25-43.
//! 3. Klein JP. Semiparametric estimation of random effects using the Cox model
//!    based on the EM algorithm. Biometrics. 1992;48(3):795-806.
//! 4. Ripatti S, Palmgren J. Estimation of multivariate frailty models using
//!    penalized partial likelihood. Biometrics. 2000;56(4):1016-22.
//! 5. Therneau TM, Grambsch PM, Pankratz VS. Penalized survival models and
//!    frailty. J Comput Graph Stat. 2003;12(1):156-75.
//! 6. Hougaard P. Analysis of Multivariate Survival Data. New York: Springer;
//!    2000.
//! 7. van den Berg GJ, Drepper B. Inference for shared-frailty survival models
//!    with left-truncated data. Econom Rev. 2016;35(6):1075-98.
//! 8. Self SG, Liang KY. Asymptotic properties of maximum likelihood estimators
//!    and likelihood ratio tests under nonstandard conditions. J Am Stat Assoc.
//!    1987;82(398):605-10.
//! 9. Andersen PK, Klein JP, Knudsen KM, Palacios RT. Estimation of variance
//!    in Cox's regression model with shared gamma frailties. Biometrics.
//!    1997;53(4):1475-84.
//! 10. Commenges D, Andersen PK. Score test of homogeneity for survival data.
//!     Lifetime Data Anal. 1995;1(2):145-56.

mod em;
mod engine;

use std::borrow::Cow;

use epistat_core::{Covariance, Interval, Level};
use epistat_linalg::{Mat, MatRef};
use epistat_optim::{brent, root};
use epistat_stats::dist::chi_square::{quantile as chi_square_quantile, sf as chi_square_sf};

use self::em::{EmInput, fit_em};
use self::engine::{Eval, Failure, Layout, Penalty, Rows, State};
use crate::check::{check_lengths, refuse_non_finite_matrix};
use crate::covariance;
use crate::cox::Ties;
use crate::error::{Error, Result};
use crate::ldl;
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// The profile search reports theta as 0 when the profile still rises at the
/// last power of 4 above this.
const THETA_MIN: f64 = 1e-6;
/// Largest theta the profile search tries.
const THETA_MAX: f64 = 1e4;
/// Pivot floor of the check for aliased columns, on the correlation matrix.
const ALIAS_TOLERANCE: f64 = 1e-10;

/// How the variance of the frailty is chosen.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Theta {
    /// A given variance, finite and not negative. Zero is the Cox model.
    Fixed(f64),
    /// The maximum of the profile marginal log likelihood on `ln theta`, to
    /// within `FrailtyOptions::theta_tolerance`: the root of its score for
    /// Breslow ties, Brent's minimiser for Efron ties.
    Profile,
}

/// How the model is fitted at a given theta.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum FrailtyMethod {
    /// Newton's method on the penalised partial likelihood with the whole
    /// information, `coxph`'s model.
    Penalised,
    /// The EM algorithm on the marginal likelihood, Breslow ties and case
    /// weights of 1 only. It reaches the same estimate as `Penalised` with
    /// Breslow ties, slowly, and computes the marginal likelihood from the
    /// gamma integral rather than from the penalised fit.
    Em {
        /// Most EM steps at one theta.
        max_iterations: usize,
        /// Relative change of the marginal log likelihood at which EM stops.
        tolerance: f64,
    },
}

/// What [`fit`] takes besides the data and the design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrailtyOptions {
    /// Breslow or Efron. Exact ties are refused. `Efron`, as in `coxph`.
    pub ties: Ties,
    /// The rule that makes nearly equal times equal, as in [`crate::cox::fit`].
    pub timefix: TimeFix,
    /// The event code to model, as in [`crate::cox::CoxOptions`].
    pub cause: Option<u32>,
    /// How theta is chosen. [`Theta::Profile`] by default.
    pub theta: Theta,
    /// How the model is fitted at each theta.
    pub method: FrailtyMethod,
    /// The Newton iteration has converged when the decrement
    /// `U' H^-1 U` is at most this. It must be positive; `1e-10` by default.
    pub tolerance: f64,
    /// Most Newton trial points at one theta, 50 by default.
    pub max_iterations: usize,
    /// The tolerance on `ln theta` of the root finder (Breslow ties) or
    /// of Brent's method (Efron ties) for [`Theta::Profile`], `1e-6` by
    /// default.
    pub theta_tolerance: f64,
    /// The level of the profile-likelihood interval of theta, which
    /// [`Theta::Profile`] gives with Breslow ties; `None` leaves the interval
    /// out and saves its fits, about ten. A 95% level by default.
    pub level: Option<Level>,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl Default for FrailtyOptions {
    fn default() -> Self {
        Self {
            ties: Ties::Efron,
            timefix: TimeFix::default(),
            cause: None,
            theta: Theta::Profile,
            method: FrailtyMethod::Penalised,
            tolerance: 1e-10,
            max_iterations: 50,
            theta_tolerance: 1e-6,
            level: Some(Level::default()),
            _extensible: epistat_core::Extensible::NEW,
        }
    }
}

/// How the reported theta was reached.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaStatus {
    /// Given by the caller.
    Fixed,
    /// A maximum of the profile inside the range searched.
    Interior,
    /// Theta is reported as 0: the model is the Cox model and every log
    /// frailty is 0. With Breslow ties the score of theta at 0 is at most 0
    /// and the profile is not higher at the smallest theta of the bracketing,
    /// about `4e-6`; with Efron ties the profile rises towards 0 down to that
    /// theta.
    Boundary,
    /// The profile was still rising at the largest theta tried, `1e4`; the
    /// fit is at that theta and is not an estimate.
    Unbounded,
}

/// One theta of a search.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ThetaStep {
    /// The variance of the frailty.
    pub theta: f64,
    /// The partial log likelihood at the solution for this theta.
    pub log_partial_likelihood: f64,
    /// The profile marginal log likelihood, `c.loglik` of `coxph`.
    pub marginal_log_likelihood: f64,
    /// Whether the fit at this theta converged.
    pub converged: bool,
}

/// The likelihood-ratio test of theta = 0, on the boundary of the parameter
/// space.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BoundaryTest {
    /// Twice the gain of the profile marginal log likelihood over the Cox model.
    pub statistic: f64,
    /// Half the upper tail of the chi-square with one degree of freedom, the
    /// 50:50 mixture of 0 and chi-square 1 of Self and Liang.
    pub p_value: f64,
}

/// A shared gamma frailty Cox model. See the module documentation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FrailtyFit {
    /// One coefficient per column of the design, `None` for an aliased column.
    pub coefficients: Vec<Option<f64>>,
    /// Standard errors from `covariance`.
    pub se: Vec<Option<f64>>,
    /// The coefficient block of the inverse of the whole penalised
    /// information, `p` by `p`, `var` of `coxph(sparse = FALSE)`.
    /// `None` in an entry of an aliased column, and everywhere when the
    /// iterative solve did not meet its tolerance.
    pub covariance: Covariance,
    /// The covariance with the penalty's curvature taken out,
    /// `H^-1 (H - P) H^-1`, `var2` of `coxph(sparse = FALSE)`.
    pub covariance_sandwich: Covariance,
    /// Standard errors from `diagonal_covariance`.
    pub diagonal_se: Vec<Option<f64>>,
    /// The covariance with the frailty block of the information replaced by
    /// its diagonal, `var` of `coxph(sparse = TRUE)`, the default for more
    /// than five clusters.
    pub diagonal_covariance: Covariance,
    /// `var2` of `coxph(sparse = TRUE)`.
    pub diagonal_covariance_sandwich: Covariance,
    /// The log frailty of each cluster of the data, `frail` of `coxph`,
    /// centred so that the frailties have mean 1. `None` for a cluster whose
    /// rows all have weight zero.
    pub frailties: Vec<Option<f64>>,
    /// The variance of each log frailty from the diagonal form, `fvar` of
    /// `coxph(sparse = TRUE)`.
    pub frailty_variances: Vec<Option<f64>>,
    /// The variance of the frailty.
    pub theta: f64,
    /// How `theta` was reached.
    pub theta_status: ThetaStatus,
    /// Kendall's tau of two members of a cluster, `theta / (theta + 2)`.
    pub kendall_tau: f64,
    /// The partial log likelihood at the estimate, `loglik[2]` of `coxph`.
    pub log_partial_likelihood: f64,
    /// The penalised log likelihood that the fit maximises.
    pub penalised_log_likelihood: f64,
    /// The profile marginal log likelihood at theta, `c.loglik` of `coxph`.
    /// `None` when a case weight is not 1.
    pub marginal_log_likelihood: Option<f64>,
    /// The partial log likelihood of the Cox model without frailty, which is
    /// the marginal log likelihood at theta = 0. `None` when a case weight is
    /// not 1.
    pub cox_log_likelihood: Option<f64>,
    /// The test of theta = 0, given when theta is estimated.
    pub likelihood_ratio: Option<BoundaryTest>,
    /// The score of the profile marginal log likelihood in theta at theta =
    /// 0, `(1/2) sum_i [(D_i - Lambda_i)^2 - D_i]` from the Cox fit. Given
    /// for Breslow ties and case weights of 1 when the Cox model is fitted,
    /// which it is unless theta is fixed at a positive value with other
    /// weights.
    pub score_at_zero: Option<f64>,
    /// The score of the profile in theta at the reported theta, for Breslow
    /// ties and case weights of 1; near 0 at an interior estimate.
    pub theta_score: Option<f64>,
    /// The profile-likelihood interval of theta at `FrailtyOptions::level`:
    /// the thetas where the profile is `q / 2` below its maximum, `q` the
    /// chi-square quantile on one degree of freedom. The lower limit is 0
    /// when the profile at 0 is above that cut, and the upper limit `None`
    /// when the profile is still above it at `1e4`. Given by
    /// [`Theta::Profile`] with Breslow ties for an interior or boundary
    /// estimate.
    pub theta_interval: Option<Interval>,
    /// The standard error of `ln theta`, `(-c''(ln theta))^(-1/2)` with the
    /// second derivative of the profile from a central difference of its
    /// score, steps of `1e-3` on `ln theta`. Given by [`Theta::Profile`]
    /// with Breslow ties for an interior estimate where the second
    /// derivative is negative.
    pub ln_theta_se: Option<f64>,
    /// Every theta fitted, in order.
    pub history: Vec<ThetaStep>,
    /// Trial points (Newton) or EM steps of the final fit.
    pub iterations: usize,
    /// Whether the final fit converged.
    pub converged: bool,
    /// The summed weight of the events.
    pub n_events: f64,
    /// Every row of the data.
    pub n_rows: usize,
    /// The rows with a weight above zero.
    pub n_used: usize,
    /// Clusters with a row of positive weight.
    pub n_clusters: usize,
}

/// The columns of the design that are not constant and not linear
/// combinations of those before them, over the rows of positive weight.
fn kept_columns(x: MatRef<'_, f64>, used: &[usize], weight: &[f64]) -> Vec<usize> {
    let p = x.ncols();
    let total: f64 = used.iter().map(|&i| weight[i]).sum();
    let mut mean = vec![0.0; p];
    for &i in used {
        for (c, m) in mean.iter_mut().enumerate() {
            *m += weight[i] * x[(i, c)];
        }
    }
    for m in &mut mean {
        *m /= total;
    }
    let mut gram = vec![0.0; p * p];
    for &i in used {
        for r in 0..p {
            let dr = x[(i, r)] - mean[r];
            for c in 0..=r {
                gram[r * p + c] += weight[i] * dr * (x[(i, c)] - mean[c]);
            }
        }
    }
    let scale: Vec<f64> = (0..p).map(|c| gram[c * p + c].sqrt()).collect();
    let varying: Vec<usize> = (0..p).filter(|&c| scale[c] > 0.0).collect();
    let q = varying.len();
    let mut corr = vec![0.0; q * q];
    for (a, &r) in varying.iter().enumerate() {
        for (b, &c) in varying.iter().enumerate().take(a + 1) {
            let v = gram[r * p + c] / (scale[r] * scale[c]);
            corr[a * q + b] = v;
            corr[b * q + a] = v;
        }
    }
    ldl::factorise(&mut corr, q, ALIAS_TOLERANCE);
    varying
        .iter()
        .enumerate()
        .filter(|&(a, _)| corr[a * q + a] != 0.0)
        .map(|(_, &c)| c)
        .collect()
}

/// Whether every cluster has exactly one row of positive weight; `cluster`
/// holds the dense index of each row, `u32::MAX` for a row left out.
fn singletons(cluster: &[u32], g: usize) -> bool {
    let mut size = vec![0_u32; g];
    for &c in cluster {
        if c != u32::MAX {
            size[c as usize] += 1;
        }
    }
    size.iter().all(|&n| n == 1)
}

/// `x - ln(1 + x)` for `x >= 0` without cancellation near 0.
fn x_minus_ln_1p(x: f64) -> f64 {
    if x < 1e-2 {
        let mut term = x * x;
        let mut sum = 0.0;
        for k in 2..=14 {
            let t = term / f64::from(k);
            sum += if k % 2 == 0 { t } else { -t };
            term *= x;
        }
        sum
    } else {
        x - x.ln_1p()
    }
}

/// The correction that turns the penalised log likelihood into the profile
/// marginal log likelihood, `frailty.gammacon` of `survival`, for clusters
/// with `events` events each and `nu = 1 / theta`.
fn gamma_correction(events: &[u32], nu: f64) -> f64 {
    let mut total = 0.0;
    for &d in events {
        if d == 0 {
            continue;
        }
        let df = f64::from(d);
        let mut term = nu * x_minus_ln_1p(df / nu);
        let denominator = nu + df;
        for m in 1..=d {
            term += (-f64::from(m) / denominator).ln_1p();
        }
        total += term;
    }
    total
}

/// The fit at one theta.
struct AtTheta {
    theta: f64,
    state: State,
    eval: Eval,
    marginal: Option<f64>,
    iterations: usize,
    converged: bool,
}

struct Problem<'a> {
    layout: Layout,
    tolerance: f64,
    max_iterations: usize,
    method: FrailtyMethod,
    em: Option<EmInput<'a>>,
}

fn failure(f: Failure) -> Error {
    match f {
        Failure::NotFinite => Error::undefined(
            "risk score",
            "a risk score or a risk-set total is not finite",
        ),
        Failure::Information => Error::not_positive_definite(
            "information",
            "the penalised information is not positive definite at a trial point",
        ),
    }
}

impl Problem<'_> {
    fn penalty(theta: f64) -> Penalty {
        if theta == 0.0 {
            Penalty::None
        } else {
            Penalty::Gamma { nu: 1.0 / theta }
        }
    }

    fn at(&self, theta: f64, start: &State) -> Result<AtTheta> {
        let layout = &self.layout;
        let penalty = Self::penalty(theta);
        let (state, eval, iterations, converged, em_marginal) = match (self.method, &self.em) {
            (FrailtyMethod::Em { .. }, Some(input)) if theta > 0.0 => {
                let em = fit_em(layout, input, theta, start)?;
                let eval = layout
                    .evaluate(&em.state, penalty)
                    .ok_or_else(|| failure(Failure::NotFinite))?;
                (
                    em.state,
                    eval,
                    em.iterations,
                    em.converged,
                    Some(em.marginal),
                )
            }
            _ => {
                let inner = layout
                    .maximise(start, penalty, self.tolerance, self.max_iterations)
                    .map_err(failure)?;
                (
                    inner.state,
                    inner.eval,
                    inner.iterations,
                    inner.converged,
                    None,
                )
            }
        };
        let marginal = if !layout.unit_weights {
            None
        } else if let Some(m) = em_marginal {
            Some(m)
        } else if theta == 0.0 {
            Some(eval.loglik)
        } else {
            let nu = 1.0 / theta;
            let sum_w: f64 = state.w.iter().sum();
            Some(eval.loglik + nu * sum_w + gamma_correction(&layout.events, nu))
        };
        Ok(AtTheta {
            theta,
            state,
            eval,
            marginal,
            iterations,
            converged,
        })
    }
}

/// The searches over theta keep every fit, to start each new one from the
/// fit at the nearest theta.
struct Search<'p, 'a> {
    problem: &'p Problem<'a>,
    fits: Vec<AtTheta>,
}

impl Search<'_, '_> {
    fn nearest(&self, theta: f64) -> State {
        let key = |t: f64| if t > 0.0 { t.ln() } else { f64::NEG_INFINITY };
        let target = key(theta);
        self.fits
            .iter()
            .filter(|f| f.theta > 0.0 || theta == 0.0)
            .min_by(|a, b| {
                (key(a.theta) - target)
                    .abs()
                    .total_cmp(&(key(b.theta) - target).abs())
            })
            .map_or_else(
                || {
                    let base = self.fits.first().map(|f| f.state.beta.clone());
                    let mut s = State::zero(self.problem.layout.p, self.problem.layout.g);
                    if let Some(beta) = base {
                        s.beta = beta;
                    }
                    s
                },
                |f| f.state.clone(),
            )
    }

    fn run(&mut self, theta: f64) -> Result<f64> {
        let start = self.nearest(theta);
        let fit = self.problem.at(theta, &start)?;
        let m = fit.marginal.expect("searches require case weights of 1");
        self.fits.push(fit);
        Ok(m)
    }

    /// The index of the fit at `theta`, which is fitted unless it is there.
    /// A failure is [`Error::Frailty`] at `theta`.
    fn index(&mut self, theta: f64) -> Result<usize> {
        if let Some(i) = self
            .fits
            .iter()
            .position(|f| f.theta.to_bits() == theta.to_bits())
        {
            return Ok(i);
        }
        self.run(theta).map_err(at_theta(theta))?;
        Ok(self.fits.len() - 1)
    }

    /// The profile marginal log likelihood `c` at `theta`.
    fn marginal(&mut self, theta: f64) -> Result<f64> {
        let i = self.index(theta)?;
        Ok(self.fits[i]
            .marginal
            .expect("searches require case weights of 1"))
    }

    /// The derivative in theta of `c` at `theta`, for Breslow ties.
    fn score(&mut self, theta: f64) -> Result<f64> {
        let i = self.index(theta)?;
        let layout = &self.problem.layout;
        let (lambda, _) = layout
            .breslow_cluster_hazards(&self.fits[i].state)
            .ok_or_else(|| at_theta(theta)(failure(Failure::NotFinite)))?;
        Ok(profile_score(&layout.events, &lambda, theta))
    }

    fn history(&self) -> Vec<ThetaStep> {
        self.fits
            .iter()
            .map(|f| ThetaStep {
                theta: f.theta,
                log_partial_likelihood: f.eval.loglik,
                marginal_log_likelihood: f.marginal.unwrap_or(f64::NAN),
                converged: f.converged,
            })
            .collect()
    }
}

/// A closure that wraps the failure of the fit at `theta`.
fn at_theta(theta: f64) -> impl FnOnce(Error) -> Error {
    move |source| match source {
        Error::Frailty { .. } => source,
        source => Error::Frailty {
            theta,
            source: Box::new(source),
        },
    }
}

/// `(ln(1 + x) - x / (1 + x)) / x^2` for `x >= 0`, without cancellation near
/// 0, where it is 1/2.
fn log_excess_ratio(x: f64) -> f64 {
    if x < 0.1 {
        // sum_{k >= 2} (-1)^k (k - 1) / k x^(k - 2); the terms after k = 21
        // are below 0.1^20 of the first.
        let mut sum = 0.0;
        let mut power = 1.0;
        for k in 2..=21_u32 {
            let t = power * f64::from(k - 1) / f64::from(k);
            sum += if k % 2 == 0 { t } else { -t };
            power *= x;
        }
        sum
    } else {
        (x.ln_1p() - x / (1.0 + x)) / (x * x)
    }
}

/// The derivative in theta of the profile marginal log likelihood for
/// Breslow ties, from the events `D_i` and the cumulative hazards `Lambda_i`
/// of the clusters without the frailty at the fit at `theta`, which may be 0.
/// By the envelope theorem it is the partial derivative of the marginal log
/// likelihood at fixed coefficients and baseline hazard,
///
/// ```text
/// -nu^2 (psi(nu + D) - psi(nu)) + D / theta + nu^2 ln(1 + theta Lambda) - (nu + D) Lambda / (1 + theta Lambda),
/// ```
///
/// summed over the clusters, written as
/// `sum_{m < D} m / (1 + m theta) + Lambda^2 (ln(1 + x) - x / (1 + x)) / x^2 - D Lambda / (1 + x)`
/// with `x = theta Lambda`, which has no cancellation as theta goes to 0 and
/// is `(1/2) [(D - Lambda)^2 - D]` there.
fn profile_score(events: &[u32], lambda: &[f64], theta: f64) -> f64 {
    let mut total = 0.0;
    for (&d, &l) in events.iter().zip(lambda) {
        let mut term = 0.0;
        for m in 1..d {
            let m = f64::from(m);
            term += m / (1.0 + m * theta);
        }
        let x = theta * l;
        term += l * l * log_excess_ratio(x);
        term -= f64::from(d) * l / (1.0 + x);
        total += term;
    }
    total
}

/// A root of `f` in `[lower, upper]` by the ITP method, where `f` may fail; the
/// first failure is returned.
fn find_root<'p, 'a>(
    search: &mut Search<'p, 'a>,
    mut f: impl FnMut(&mut Search<'p, 'a>, f64) -> Result<f64>,
    lower: f64,
    upper: f64,
    tolerance: f64,
) -> Result<f64> {
    let mut failed = None;
    let options = root::Itp {
        tolerance,
        ..root::Itp::default()
    };
    let found = root::itp(
        |x| match f(search, x) {
            Ok(v) => v,
            Err(e) => {
                failed.get_or_insert(e);
                f64::NAN
            }
        },
        lower,
        upper,
        &options,
    );
    if let Some(e) = failed {
        return Err(e);
    }
    Ok(found?.x)
}

/// What the profile by the score found.
struct ScoreProfile {
    chosen: usize,
    status: ThetaStatus,
    interval: Option<Interval>,
    ln_theta_se: Option<f64>,
}

/// Step on `ln theta` of the central difference of the score.
const SE_STEP: f64 = 1e-3;

/// The maximum of the profile on `ln theta` for Breslow ties, as the root of
/// its score, with the boundary rule, the profile-likelihood interval and the
/// standard error of `ln theta`.
fn profile_by_score(
    search: &mut Search<'_, '_>,
    tolerance: f64,
    level: Option<Level>,
) -> Result<ScoreProfile> {
    let c0 = search.marginal(0.0)?;
    let u0 = search.score(0.0)?;
    let step = 4.0_f64.ln();
    let (t_min, t_max) = (THETA_MIN.ln(), THETA_MAX.ln());
    // The score in ln theta, which has the sign of the score in theta.
    let slope = |s: &mut Search<'_, '_>, t: f64| -> Result<f64> {
        let theta = t.exp();
        Ok(theta * s.score(theta)?)
    };
    let mut t = 0.0;
    let first = slope(search, t)?;
    // A bracket on ln theta, or None when the score is negative down to the
    // smallest theta of the bracketing, e^t.
    let bracket = if first < 0.0 {
        loop {
            let next = t - step;
            if next < t_min {
                break None;
            }
            if slope(search, next)? >= 0.0 {
                break Some((next, t));
            }
            t = next;
        }
    } else if first > 0.0 {
        loop {
            let next = t + step;
            if next > t_max {
                let last = search.index(t.exp())?;
                return Ok(ScoreProfile {
                    chosen: last,
                    status: ThetaStatus::Unbounded,
                    interval: None,
                    ln_theta_se: None,
                });
            }
            if slope(search, next)? <= 0.0 {
                break Some((t, next));
            }
            t = next;
        }
    } else {
        // The score is 0 at theta = 1: a bracket of the two points about it.
        Some((-step, step))
    };
    let smallest = t.exp();
    let theta_hat = match bracket {
        Some((a, b)) => Some(find_root(search, slope, a, b, tolerance)?.exp()),
        None if u0 > 0.0 => Some(find_root(
            search,
            Search::score,
            0.0,
            smallest,
            tolerance * smallest,
        )?),
        None if search.marginal(smallest)? <= c0 => None,
        None => {
            return Err(Error::undefined(
                "theta",
                format!(
                    "theta: the score of the profile at theta = 0 is not positive, but the profile \
                 is higher at theta = {smallest:e} than at 0, so its maximum near 0 is not \
                 resolved"
                ),
            ));
        }
    };
    let (chosen, status) = match theta_hat {
        Some(theta) => (search.index(theta)?, ThetaStatus::Interior),
        None => (search.index(0.0)?, ThetaStatus::Boundary),
    };
    let ln_theta_se = match theta_hat {
        Some(theta) => {
            let t_hat = theta.ln();
            let up = slope(search, t_hat + SE_STEP)?;
            let down = slope(search, t_hat - SE_STEP)?;
            let curvature = (up - down) / (2.0 * SE_STEP);
            (curvature < 0.0).then(|| (-1.0 / curvature).sqrt())
        }
        None => None,
    };
    let interval = match level {
        Some(level) => Some(profile_interval(
            search, theta_hat, smallest, c0, level, tolerance,
        )?),
        None => None,
    };
    Ok(ScoreProfile {
        chosen,
        status,
        interval,
        ln_theta_se,
    })
}

/// The profile-likelihood interval of theta around the estimate `theta_hat`
/// (`None` on the boundary, where `smallest` is the smallest theta of the
/// bracketing).
fn profile_interval(
    search: &mut Search<'_, '_>,
    theta_hat: Option<f64>,
    smallest: f64,
    c0: f64,
    level: Level,
    tolerance: f64,
) -> Result<Interval> {
    let step = 4.0_f64.ln();
    let (t_min, t_max) = (THETA_MIN.ln(), THETA_MAX.ln());
    let top = match theta_hat {
        Some(theta) => search.marginal(theta)?,
        None => c0,
    };
    let cut = top - 0.5 * chi_square_quantile(level.confidence_value(), 1.0);
    let gap = |s: &mut Search<'_, '_>, t: f64| -> Result<f64> { Ok(s.marginal(t.exp())? - cut) };
    let linear_gap =
        |s: &mut Search<'_, '_>, theta: f64| -> Result<f64> { Ok(s.marginal(theta)? - cut) };
    let base = theta_hat.unwrap_or(smallest).ln();
    // The upper limit, walking up from the estimate.
    let upper = if gap(search, base)? < 0.0 {
        Some(find_root(
            search,
            linear_gap,
            0.0,
            smallest,
            tolerance * smallest,
        )?)
    } else {
        let mut t = base;
        loop {
            let next = t + step;
            if next > t_max {
                break None;
            }
            if gap(search, next)? < 0.0 {
                break Some(find_root(search, gap, t, next, tolerance)?.exp());
            }
            t = next;
        }
    };
    // The lower limit, walking down from the estimate.
    let lower = match theta_hat {
        None => 0.0,
        Some(_) if c0 >= cut => 0.0,
        Some(theta) => {
            let mut t = theta.ln();
            loop {
                let next = t - step;
                if next < t_min {
                    let last = t.exp();
                    break find_root(search, linear_gap, 0.0, last, tolerance * last)?;
                }
                if gap(search, next)? < 0.0 {
                    break find_root(search, gap, next, t, tolerance)?.exp();
                }
                t = next;
            }
        }
    };
    Ok(Interval::new(Some(lower), upper, level))
}

/// The maximum of the profile marginal log likelihood on `ln theta` by Brent's
/// minimiser, for Efron ties. Returns the index of the chosen fit and its
/// status.
fn profile(search: &mut Search<'_, '_>, tolerance: f64) -> Result<(usize, ThetaStatus)> {
    let at = at_theta;
    search.run(0.0).map_err(at(0.0))?;
    let step = 4.0_f64.ln();
    let (t_min, t_max) = (THETA_MIN.ln(), THETA_MAX.ln());
    let f = |s: &mut Search<'_, '_>, t: f64| -> Result<f64> {
        let theta = t.exp();
        Ok(-s.run(theta).map_err(at(theta))?)
    };
    let (mut t0, mut f0) = (0.0, f(search, 0.0)?);
    let (mut t1, mut f1) = (-step, f(search, -step)?);
    let bracket = if f1 <= f0 {
        loop {
            let t2 = t1 - step;
            if t2 < t_min {
                return Ok((0, ThetaStatus::Boundary));
            }
            let f2 = f(search, t2)?;
            if f2 > f1 {
                break (t2, t0);
            }
            (t0, f0, t1, f1) = (t1, f1, t2, f2);
        }
    } else {
        (t0, f0, t1, f1) = (t1, f1, t0, f0);
        loop {
            let t2 = t1 + step;
            if t2 > t_max {
                return Ok((search.fits.len() - 1, ThetaStatus::Unbounded));
            }
            let f2 = f(search, t2)?;
            if f2 > f1 {
                break (t0, t2);
            }
            (t0, f0, t1, f1) = (t1, f1, t2, f2);
        }
    };
    let _ = (f0, f1);
    let options = brent::Brent {
        tolerance,
        ..brent::Brent::default()
    };
    let found = brent::minimize(|t| f(search, t), bracket.0, bracket.1, &options)?;
    let best = search
        .fits
        .iter()
        .enumerate()
        .filter(|(_, fit)| fit.theta > 0.0)
        .min_by(|a, b| {
            (a.1.theta.ln() - found.x)
                .abs()
                .total_cmp(&(b.1.theta.ln() - found.x).abs())
        })
        .map(|(i, _)| i)
        .expect("Brent evaluates at least one point");
    Ok((best, ThetaStatus::Interior))
}

fn check_options(opts: &FrailtyOptions, data: &SurvData<'_>) -> Result<()> {
    if opts.ties == Ties::Exact {
        return Err(Error::unsupported(
            "ties",
            "ties: the frailty model supports Breslow and Efron ties only",
        ));
    }
    if !(opts.tolerance.is_finite() && opts.tolerance > 0.0) {
        return Err(Error::options("tolerance: must be finite and positive"));
    }
    if opts.max_iterations == 0 {
        return Err(Error::options("max_iterations: must be at least 1"));
    }
    if !(opts.theta_tolerance.is_finite() && opts.theta_tolerance > 0.0) {
        return Err(Error::options(
            "theta_tolerance: must be finite and positive",
        ));
    }
    match opts.theta {
        Theta::Fixed(t) if !(t.is_finite() && t >= 0.0) => {
            return Err(Error::options(
                "theta: a fixed theta must be finite and not negative",
            ));
        }
        _ => {}
    }
    if let FrailtyMethod::Em {
        max_iterations,
        tolerance,
    } = opts.method
    {
        if opts.ties != Ties::Breslow {
            return Err(Error::options(
                "method: EM gives the maximum of the marginal likelihood for Breslow ties only",
            ));
        }
        if max_iterations == 0 || !(tolerance.is_finite() && tolerance > 0.0) {
            return Err(Error::options(
                "method: EM needs at least one step and a positive tolerance",
            ));
        }
    }
    if let Some(offset) = data.offset()
        && offset
            .iter()
            .any(|o| !o.is_finite() || !o.exp().is_finite())
    {
        return Err(Error::value(
            "offset: a value is not finite or gives a risk score that is not finite",
        ));
    }
    Ok(())
}

/// Fit a shared gamma frailty Cox model. See the module documentation for
/// the model, the algorithm and what each field means.
///
/// # Errors
///
/// An input error when `x` has another number of rows than `data` or a
/// value that is not finite, when the data has no cluster column, when the
/// ties are exact, when an option is out of its range, when theta is
/// estimated or EM is asked for and a case weight is not 1, when EM is asked
/// for with Efron ties, and when a fit meets a risk score that is not finite
/// or an information that is not positive definite. Under [`Theta::Profile`]
/// such a failure at a trial theta is [`Error::Frailty`], which names the
/// theta.
///
/// # Panics
///
/// Only on a broken internal invariant.
#[expect(
    clippy::too_many_lines,
    reason = "the checks, the search and the assembly of the result in order"
)]
pub fn fit(data: &SurvData<'_>, x: MatRef<'_, f64>, opts: &FrailtyOptions) -> Result<FrailtyFit> {
    check_lengths(&[("data", data.n_rows()), ("x", x.nrows())])?;
    refuse_non_finite_matrix(x, "x")?;
    check_options(opts, data)?;
    if data.cluster().is_none() {
        return Err(Error::missing(
            "cluster: the frailty model needs a cluster column",
        ));
    }
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup();
    let code = match (opts.cause, surv.n_codes()) {
        (None, 1) => 1,
        (Some(code), _) => code,
        (None, _) => {
            return Err(Error::missing(
                "cause: the status has several codes, so the event code is required",
            ));
        }
    };
    let is_event: Vec<bool> = data.cause(code)?.iter().collect();
    let rows = data.n_rows();
    let start = surv
        .start()
        .map_or_else(|| Cow::Owned(vec![f64::NEG_INFINITY; rows]), Cow::Borrowed);
    let weight = data
        .weight()
        .map_or_else(|| Cow::Owned(vec![1.0; rows]), Cow::Borrowed);
    let stratum = data
        .stratum()
        .map_or_else(|| Cow::Owned(vec![0; rows]), Cow::Borrowed);
    let cluster_codes = data.cluster().expect("checked above");
    let used: Vec<usize> = (0..rows).filter(|&i| weight[i] > 0.0).collect();
    let mut dense = vec![u32::MAX; data.n_clusters()];
    for &i in &used {
        dense[cluster_codes[i] as usize] = 0;
    }
    let mut g = 0u32;
    for d in &mut dense {
        if *d == 0 {
            *d = g;
            g += 1;
        }
    }
    let cluster: Vec<u32> = (0..rows)
        .map(|i| {
            if weight[i] > 0.0 {
                dense[cluster_codes[i] as usize]
            } else {
                u32::MAX
            }
        })
        .collect();
    let kept = if used.is_empty() {
        Vec::new()
    } else {
        kept_columns(x, &used, &weight)
    };
    let p = kept.len();
    let mut xk = Vec::with_capacity(rows * p);
    for i in 0..rows {
        for &c in &kept {
            xk.push(x[(i, c)]);
        }
    }
    let layout = Layout::new(&Rows {
        x: &xk,
        p,
        start: &start,
        stop: surv.stop(),
        event: &is_event,
        weight: &weight,
        stratum: &stratum,
        n_strata: data.n_strata(),
        cluster: &cluster,
        g: g as usize,
        offset: data.offset(),
        efron: opts.ties == Ties::Efron,
    });
    let estimated = !matches!(opts.theta, Theta::Fixed(_));
    let em = matches!(opts.method, FrailtyMethod::Em { .. });
    if (estimated || em) && !layout.unit_weights {
        return Err(Error::not_applicable(
            "weight: estimating theta and EM need case weights of 1, for which the marginal \
             likelihood is defined",
        ));
    }
    let n_events: f64 = used
        .iter()
        .filter(|&&i| is_event[i])
        .map(|&i| weight[i])
        .sum();
    if n_events == 0.0 {
        return Err(Error::no_information(
            "status: there is no event among the rows used",
        ));
    }
    let zero_offset = vec![0.0; rows];
    let x_kept = Mat::from_fn(rows, p, |i, c| xk[i * p + c]);
    let em_input = match opts.method {
        FrailtyMethod::Em {
            max_iterations,
            tolerance,
        } => Some(EmInput {
            data: &data,
            x: x_kept.as_ref(),
            cause: opts.cause,
            offset: data.offset().unwrap_or(&zero_offset),
            cluster: &cluster,
            max_iterations,
            tolerance,
        }),
        FrailtyMethod::Penalised => None,
    };
    let problem = Problem {
        layout,
        tolerance: opts.tolerance,
        max_iterations: opts.max_iterations,
        method: opts.method,
        em: em_input,
    };
    let mut search = Search {
        problem: &problem,
        fits: Vec::new(),
    };
    let mut theta_interval = None;
    let mut ln_theta_se = None;
    let (chosen, status) = match opts.theta {
        Theta::Fixed(theta) => {
            if problem.layout.unit_weights && theta > 0.0 {
                search.run(0.0)?;
                let start = search.nearest(theta);
                let fit = problem.at(theta, &start)?;
                search.fits.push(fit);
            } else {
                let start = State::zero(p, g as usize);
                let fit = problem.at(theta, &start)?;
                search.fits.push(fit);
            }
            (search.fits.len() - 1, ThetaStatus::Fixed)
        }
        Theta::Profile if opts.ties == Ties::Breslow => {
            let found = profile_by_score(&mut search, opts.theta_tolerance, opts.level)?;
            theta_interval = found.interval;
            ln_theta_se = found.ln_theta_se;
            (found.chosen, found.status)
        }
        Theta::Profile => profile(&mut search, opts.theta_tolerance)?,
    };
    if estimated && status != ThetaStatus::Boundary && singletons(&cluster, g as usize) {
        return Err(Error::not_identified(
            "theta",
            "theta: every cluster has one row, so the data do not identify theta, and the \
             profile is not on the boundary",
        ));
    }
    let score_at_zero = if opts.ties == Ties::Breslow && problem.layout.unit_weights {
        if search.fits.iter().any(|f| f.theta == 0.0) {
            Some(search.score(0.0)?)
        } else {
            None
        }
    } else {
        None
    };
    let theta_score = if opts.ties == Ties::Breslow && problem.layout.unit_weights {
        Some(search.score(search.fits[chosen].theta)?)
    } else {
        None
    };
    let history = if estimated {
        search.history()
    } else {
        Vec::new()
    };
    let cox_log_likelihood = search
        .fits
        .iter()
        .find(|f| f.theta == 0.0)
        .and_then(|f| f.marginal);
    let fits = std::mem::take(&mut search.fits);
    let chosen = fits.into_iter().nth(chosen).expect("a chosen fit");
    let penalty = Problem::penalty(chosen.theta);
    let layout = &problem.layout;
    let covariances = layout
        .covariances(&chosen.eval, penalty)
        .ok_or_else(|| failure(Failure::Information))?;
    let full = |values: Option<&[f64]>| -> Vec<Option<f64>> {
        let width = x.ncols();
        let mut out = vec![None; width * width];
        if let Some(values) = values {
            for (a, &ra) in kept.iter().enumerate() {
                for (b, &rb) in kept.iter().enumerate() {
                    out[ra * width + rb] = Some(values[a * p + b]);
                }
            }
        }
        out
    };
    let diagonal = |m: &[Option<f64>]| -> Vec<Option<f64>> {
        let width = x.ncols();
        (0..width)
            .map(|c| m[c * width + c].map(f64::sqrt))
            .collect()
    };
    let mut coefficients = vec![None; x.ncols()];
    for (a, &c) in kept.iter().enumerate() {
        coefficients[c] = Some(chosen.state.beta[a]);
    }
    let covariance = full(covariances.exact.as_deref());
    let covariance_sandwich = full(covariances.exact_sandwich.as_deref());
    let diagonal_covariance = full(Some(&covariances.diagonal));
    let diagonal_covariance_sandwich = full(Some(&covariances.diagonal_sandwich));
    let frail = chosen.theta > 0.0;
    let frailties = dense
        .iter()
        .map(|&d| {
            (d != u32::MAX).then(|| {
                if frail {
                    chosen.state.w[d as usize]
                } else {
                    0.0
                }
            })
        })
        .collect();
    let frailty_variances = dense
        .iter()
        .map(|&d| (d != u32::MAX).then(|| covariances.frailty_diagonal[d as usize]))
        .collect();
    let likelihood_ratio = if estimated {
        match (chosen.marginal, cox_log_likelihood) {
            (Some(m), Some(m0)) => {
                let statistic = (2.0 * (m - m0)).max(0.0);
                Some(BoundaryTest {
                    statistic,
                    p_value: if statistic > 0.0 {
                        0.5 * chi_square_sf(statistic, 1.0)
                    } else {
                        1.0
                    },
                })
            }
            _ => None,
        }
    } else {
        None
    };
    let width = x.ncols();
    Ok(FrailtyFit {
        se: diagonal(&covariance),
        diagonal_se: diagonal(&diagonal_covariance),
        coefficients,
        covariance: covariance::from_options(width, covariance)?,
        covariance_sandwich: covariance::from_options(width, covariance_sandwich)?,
        diagonal_covariance: covariance::from_options(width, diagonal_covariance)?,
        diagonal_covariance_sandwich: covariance::from_options(
            width,
            diagonal_covariance_sandwich,
        )?,
        frailties,
        frailty_variances,
        theta: chosen.theta,
        theta_status: status,
        kendall_tau: chosen.theta / (chosen.theta + 2.0),
        log_partial_likelihood: chosen.eval.loglik,
        penalised_log_likelihood: chosen.eval.ppl,
        marginal_log_likelihood: chosen.marginal,
        cox_log_likelihood,
        likelihood_ratio,
        score_at_zero,
        theta_score,
        theta_interval,
        ln_theta_se,
        history,
        iterations: chosen.iterations,
        converged: chosen.converged,
        n_events,
        n_rows: rows,
        n_used: used.len(),
        n_clusters: g as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_equals_the_log_gamma_form() {
        let events = [0u32, 1, 2, 3, 7];
        for nu in [0.3, 1.0, 2.5, 40.0] {
            let direct: f64 = events
                .iter()
                .map(|&d| {
                    let d = f64::from(d);
                    d + nu * (nu / (nu + d)).ln() + epistat_stats::dist::gamma::ln_gamma(nu + d)
                        - epistat_stats::dist::gamma::ln_gamma(nu)
                        - d * (nu + d).ln()
                })
                .sum();
            let got = gamma_correction(&events, nu);
            assert!(
                (got - direct).abs() <= 1e-12 * (1.0 + direct.abs()),
                "{nu}: {got} {direct}"
            );
        }
    }

    /// The series and the direct form meet at `x = 0.1`: the direct form loses
    /// a factor of about `ln(1.1) / 0.0044`, 22, to cancellation there, and the
    /// series' truncation is below `0.1^20`, so they agree to `64 eps`.
    #[test]
    fn the_two_forms_of_the_log_excess_ratio_meet() {
        let below = log_excess_ratio(0.1_f64.next_down());
        let x = 0.1_f64;
        let direct = (x.ln_1p() - x / (1.0 + x)) / (x * x);
        assert!(
            (below - direct).abs() <= 64.0 * f64::EPSILON * direct,
            "{below} {direct}"
        );
        assert_eq!(log_excess_ratio(0.0).to_bits(), 0.5_f64.to_bits());
    }

    /// At theta = 0 the score is `(1/2) [(D - Lambda)^2 - D]`, in which every
    /// term is exact for these small integers and halves.
    #[test]
    fn the_score_at_zero_is_the_homogeneity_score() {
        let events = [0_u32, 1, 2, 5];
        let lambda = [0.5, 1.5, 2.0, 3.0];
        let want: f64 = events
            .iter()
            .zip(&lambda)
            .map(|(&d, &l)| 0.5 * ((f64::from(d) - l).powi(2) - f64::from(d)))
            .sum();
        assert_eq!(profile_score(&events, &lambda, 0.0), want);
    }

    #[test]
    fn correction_vanishes_as_theta_goes_to_zero() {
        // sum over clusters of D^2 / (2 nu) - D (D + 1) / (2 nu) = -D / (2 nu) to first order.
        let events = [1u32, 2, 3];
        let nu = 1e9;
        let got = gamma_correction(&events, nu);
        let first_order = -(1.0 + 2.0 + 3.0) / (2.0 * nu);
        assert!(
            (got - first_order).abs() <= 1e-6 * first_order.abs(),
            "{got} {first_order}"
        );
    }
}
