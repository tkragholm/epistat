//! Augmented inverse probability weighting for the risk of an event by a time
//! with right-censored data, as `ate` of riskRegression with `estimator =
//! "AIPTW"` and `"IPTW"`.
//!
//! # The estimator
//!
//! A person has a binary treatment `A`, covariates `X`, an exit time `T` and
//! status `D` (1 for the event, 0 for censoring). Three models are fitted: a Cox
//! model of the event with `S(u | a, x) = exp(-H_0(u) exp(x(a)' b))`, a logistic
//! model of the treatment with `pi_a(x) = P(A = a | X = x)`, and a model of the
//! censoring hazard with survival `G(u | a, x)`, a Kaplan-Meier curve (by
//! groups of the caller) or a Cox model. With `F_a(t | x) = 1 - S(t | a, x)`,
//! `Y_i(t) = 1(T_i <= t, D_i = 1)`, the censoring counting process
//! `N^C_i(u) = 1(T_i <= u, D_i = 0)`, its intensity
//! `Y*_i(u) dLambda^C_i(u)` with `Y*_i(u) = 1(T_i >= u)`, and the martingale
//! `dM^C_i(u) = dN^C_i(u) - Y*_i(u) dLambda^C_i(u)`, the contribution of
//! person `i` to the risk had everyone been given `a` is
//!
//! ```text
//! phi_i,a(t) = F_a(t | x_i) + 1(A_i = a) / pi_a(x_i) *
//!              ( Y_i(t) / G(T_i- | a, x_i) - F_a(t | x_i) + aug_i(t) )
//! aug_i(t)   = F_a(t | x_i) * integral_0^(t ^ T_i) dM^C_i(u) / (G(u- ) S(u-))
//!              - integral_0^(t ^ T_i) F_a(u | x_i) dM^C_i(u) / (G(u-) S(u-))
//! ```
//!
//! and the estimate `F_a^(t) = sum phi_i,a(t) / n`. Without censoring `G = 1`,
//! the integral is 0 and this is the augmented estimator of a binary outcome. The
//! estimator of [`Estimator::Iptw`] is `sum 1(A_i = a) Y_i(t) / (pi_a G(T_i-)) /
//! n`. It is consistent if the model of the treatment and that of the censoring
//! are right, or if the models of the event and of the censoring are right, for
//! [`Estimator::Aiptw`]. Both are the forms of `ate`, which calls
//! the first "AIPTW,IPCW": the weight of the event is the inverse of the
//! censoring survival and the augmentation is the integral above (Ozenne et al.
//! 2020, after Bai, Tsiatis and O'Brien 2013).
//!
//! # The standard error
//!
//! The influence function of `F_a^(t)` for the nuisance functions fixed at their
//! estimates is `(phi_i,a(t) - F_a^(t)) / n`, and the standard error is the
//! square root of the sum of its squares, the covariance of the two levels the
//! sum of the products. This is `ate` with `known.nuisance = TRUE`. For the
//! augmented estimator it is the variance of the efficient influence function and
//! it is also the asymptotic variance with the nuisance models estimated, when
//! they are all right. The variance of the estimator that is right when only some
//! models are right, which would add the influence of the estimated nuisance
//! parameters, is not given. A person bootstrap, which refits everything, is the
//! way to get it. The contrasts and their limits are those of
//! [`Standardised::contrast`].
//!
//! # Cost: the integral by moment sums
//!
//! The integral is a sum over the distinct censoring times `u_j` up to
//! `T_i ^ t`. Write `theta_i = exp(x_i(a)' b)`, `x_j = H_0(u_j-)`, `h_j =
//! H_0(u_j)` and `dx_j = h_j - x_j`. The jump of the person's own censoring is
//! added directly. Of the compensator, the part `dLambda^C_i(u_j) / G(u_j-)` is
//! `w_j`, the Kaplan-Meier hazard increment over the survival of the person's
//! group, or `s_i dLambda^C_0(u_j) exp(s_i y_j)` for a Cox model of the
//! censoring with `s_i = exp(z_i' c)` and `y_j = Lambda^C_0(u_j-)`, and
//! `1 / S(u_j-) = exp(theta_i x_j)`. The two sums of the augmentation are
//! then
//!
//! ```text
//! A_i(P) = sum_{j < P} w_j exp(theta_i x_j + s_i y_j)
//! B_i(P) = sum_{j < P} w_j exp(s_i y_j) (exp(theta_i x_j) - exp(-theta_i dx_j))
//! ```
//!
//! (with `s_i y_j = 0` for Kaplan-Meier), and a term depends on the person
//! through `theta_i` and `s_i`, so neither is a running sum over persons.
//! Summed person by person they cost `O(n J)`, with `J` the number of censoring
//! times, which grows with `n` for continuous times.
//!
//! The persons are therefore put in bins by `theta` (and by `s` within a
//! `theta` strip for a Cox censoring model) such that `(theta - theta_0) X <=
//! r_theta` and `(s - s_0) Y <= r_s` for every person of a bin, with
//! `(theta_0, s_0)` the lowest values of the bin, `X` and `Y` the largest `h_j`
//! and `y_j` reached, and `r_theta + r_s = 2`. With `delta = theta - theta_0`
//! and `epsilon = s - s_0`,
//!
//! ```text
//! exp(theta x + s y) = exp(theta_0 x + s_0 y) sum_{m, k} (delta x)^m (epsilon y)^k / (m! k!)
//! ```
//!
//! so `A` and `B` are polynomials in `delta` and `epsilon` whose coefficients
//! are running sums over the censoring times, the same for every person of the
//! bin: one pass over the censoring times per bin, and the value of a person
//! at each time asked for is read from the running sums as the pass reaches
//! it. The series stops at the total degree `M` for which `r^(M+1) e^r / (M+1)!
//! <= 2^-57`, `r` the largest `delta X + epsilon Y` of the bin, which is at most
//! 25 for `r = 2`. The truncation is then below a sixteenth of the rounding
//! unit of every term. The terms of `A` and of the coefficients of `B` without
//! tied event and censoring times are positive, so the sums have the rounding
//! error of a sum of positive terms, which is that of the sum person by person:
//! the two agree to `(2 P + 4 M^2) eps` relative to the sum of the absolute
//! terms. The cost is `O(n log n + B M^2 J)` with `B` the number of bins, which
//! depends on the spread of `theta H_0(t)` and `s Lambda^C_0(t)` over the persons
//! and not on their number. A bin is summed person by person where that costs
//! less, as for a bin of a few persons.
//!
//! The bins, the chunks of 1024 censoring times of a bin and the persons are
//! processed in parallel. The running sums at a censoring time are the sums of
//! the chunks before it added in order plus the censoring times of its own
//! chunk added in order, and the sums over persons are formed in blocks of
//! fixed size added in order, so the result does not depend on the number of
//! threads.
//!
//! # The grid
//!
//! With `grid = Some(K)` the censoring times are pooled into `K` cells of about
//! equal numbers of times, cut also at every time asked for, and a person costs
//! `O(K)`. A cell is used from the first censoring time in it. The compensator
//! `integral of h(u) dLambda^C_i(u)`, with `h = F_a / (G S)` or `1 / (G S)`, over
//! a cell is the increment of `Lambda^C_i` over the cell, up to the exit of the
//! person, times the mean of `h` at the first and the last censoring time of the
//! cell (the trapezoid rule), and the jump of the person's own censoring is added
//! at its own time. With a cell for every censoring time it is the exact sum.
//! The error is that of the trapezoid rule and depends on how smooth the
//! integrand is over the censoring times. With 6000 persons and 1500 censoring
//! times the largest error of a risk and of its standard error is `7e-4` with 8
//! cells, `6e-5` with 64 and `1e-5` with 512. The grid is not the default for
//! that reason.
//!
//! # Where this differs from riskRegression
//!
//! - Only one event type is handled, no competing risks.
//! - Entry times, case weights, strata and clusters are refused.
//! - The Cox models use the Breslow baseline hazard, as `predictCox` does, with
//!   the coefficients of the partial likelihood for the ties asked for.
//! - The hazard of a Kaplan-Meier censoring curve, which enters the martingale, is
//!   the number censored over the number at risk at each censoring time.
//! - The standard error is the first-order one of the formulas above.
//!
//! # References
//!
//! 1. Bai X, Tsiatis AA, O'Brien SM. Doubly-robust estimators of treatment-specific
//!    survival distributions in observational studies with stratified sampling.
//!    Biometrics 2013;69(4):830-9.
//! 2. Ozenne BMH, Scheike TH, Staerk L, Gerds TA. On the estimation of average
//!    treatment effects with right-censored time to event outcome and competing
//!    risks. Biom J 2020;62(3):751-63. doi:10.1002/bimj.201800298
//! 3. Zhang M, Schaubel DE. Double-robust semiparametric estimator for
//!    differences in restricted mean lifetimes in observational studies.
//!    Biometrics 2012;68(4):999-1009.

use epistat_glm::{Family, Glm, Link};
use epistat_linalg::MatRef;
use rayon::prelude::*;

use crate::cox::{
    BaselineOptions, Centring, CoxOptions, FittedModel, HazardType, Ties, Variance, basehaz,
    fit as cox_fit,
};
use crate::error::{Error, Result};
use crate::par;
use crate::standardise::{Counterfactual, Standardised};
use crate::step::Steps;
use crate::surv::{SurvBuf, SurvData};
use crate::timefix::TimeFix;
use crate::weighting::WeightSummary;

/// The rows per block of a sum over the persons.
const BLOCK: usize = 128;
/// The blocks computed at once.
const WAVE: usize = 32;

/// Which estimator of the risk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Estimator {
    /// Augmented inverse probability weighting, `estimator = "AIPTW"`.
    #[default]
    Aiptw,
    /// Inverse probability of treatment and censoring weighting, `estimator =
    /// "IPTW"` with a censoring model.
    Iptw,
}

/// The model of the censoring hazard.
#[derive(Debug, Clone, Copy)]
pub enum Censoring<'a> {
    /// No censoring before the times asked for: the censoring survival is 1. Rows
    /// with status 0 are an error.
    None,
    /// The Kaplan-Meier curve (see `AipwSurvOptions::product_limit`) of the censoring times, one curve for each group of
    /// `groups` (for example the treatment), or one curve without groups.
    KaplanMeier {
        /// A group for each row.
        groups: Option<&'a [u32]>,
    },
    /// A Cox model of the censoring hazard on the columns of `x`, one row per row
    /// of the data.
    Cox {
        /// The design of the censoring model.
        x: MatRef<'a, f64>,
    },
}

/// The models and the treatment of [`aipw_cox`].
#[derive(Debug, Clone, Copy)]
pub struct AipwSurvSpec<'a> {
    /// The treatment of each row.
    pub treatment: &'a [bool],
    /// The design the Cox model of the event is fitted on, with the treatment
    /// among its columns.
    pub outcome_x: MatRef<'a, f64>,
    /// The covariates of the event model with the treatment set to 0 and to 1.
    pub levels: [Counterfactual<'a>; 2],
    /// The design of the logistic model of the treatment, with an intercept.
    pub propensity_x: MatRef<'a, f64>,
    /// The model of the censoring hazard.
    pub censoring: Censoring<'a>,
}

/// The options of [`aipw_cox`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AipwSurvOptions {
    /// The estimator.
    pub estimator: Estimator,
    /// The ties of the Cox models.
    pub ties: Ties,
    /// The rule that makes times equal when they differ by rounding error.
    pub timefix: TimeFix,
    /// The confidence level of the limits, in `(0, 1)`.
    pub conf_level: f64,
    /// Whether the weight of an event, `1 / G(T-)`, takes the censoring survival
    /// of a Kaplan-Meier model as the product limit or, as `ate` of riskRegression
    /// does by default, as the exponential of minus the sum of the hazard
    /// increments. The integral of the augmentation always has the product limit
    /// in `ate`, and so here. Cox models of the censoring give the exponential of
    /// minus the cumulative hazard.
    pub product_limit: bool,
    /// The number of cells the censoring times are pooled into for the integral of
    /// the augmentation, `None` for the exact sum over every censoring time. See
    /// the module documentation.
    pub grid: Option<usize>,
    /// Limits `(lower, upper)` within `(0, 1)` to which the propensity is
    /// truncated, which changes the estimand. `None` is no truncation.
    pub truncation: Option<(f64, f64)>,
}

impl Default for AipwSurvOptions {
    fn default() -> Self {
        Self {
            estimator: Estimator::Aiptw,
            ties: Ties::Efron,
            timefix: TimeFix::default(),
            conf_level: 0.95,
            product_limit: false,
            grid: None,
            truncation: None,
        }
    }
}

/// The risks of the two treatments at the times asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct AipwSurv {
    /// The risks and their standard errors and covariances. Level 0 is the
    /// risk had everyone been untreated and level 1 treated, so that
    /// [`Standardised::contrast`]`(1, 0)` is the effect of treatment.
    pub risks: Standardised,
    /// The times.
    pub times: Vec<f64>,
    /// The influence of each person on the risk of level 0 and of level 1, `n`
    /// rows of one value per time, as the estimate is the sum of the values
    /// added to nothing: the influence function of the estimator, `(phi - est) / n`.
    /// `None` after the last follow-up time.
    pub influence: [Vec<f64>; 2],
    /// The number of propensities set to a limit of the truncation.
    pub n_trimmed: usize,
    /// The summary of the weights `1 / pi_a` of the treatment of each person.
    pub weights: Option<WeightSummary>,
}

/// The risk by each time under each treatment from a Cox model of the event, a
/// logistic model of the treatment and a model of the censoring, by the
/// estimator in `opts`.
///
/// `data` has one row per person, with status 0 for censoring and 1 for the
/// event, and no entry times, case weights or strata. The models are fitted
/// inside: the Cox model of the event on `spec.outcome_x`, the logistic model on
/// `spec.propensity_x` and the censoring model of `spec.censoring`. A time after
/// the last exit has no risk.
///
/// # Errors
///
/// [`Error::Input`] for data that is not right-censored or has an entry time, a
/// case weight or strata, a status outside 0 and 1, columns of another length
/// than the data, times that are not finite, positive and increasing, a
/// treatment that is the same for every row, a model that has an aliased column
/// or does not converge, a propensity of exactly 0 or 1, a censoring survival of 0
/// before an event, censoring with `Censoring::None`, truncation limits that
/// are not ordered probabilities, and a `conf_level` outside `(0, 1)`.
pub fn aipw_cox(
    data: &SurvData<'_>,
    spec: &AipwSurvSpec<'_>,
    times: &[f64],
    opts: &AipwSurvOptions,
) -> Result<AipwSurv> {
    let model = Model::fit(data, spec, times, opts)?;
    model.estimate(opts, Route::ByCost)
}

/// A one-dimensional table of a step function over the censoring times.
struct Tables {
    /// The distinct censoring times.
    jump: Vec<f64>,
    /// The baseline cumulative hazard of the event just before and at each jump.
    h_before: Vec<f64>,
    h_at: Vec<f64>,
    cens: CensTables,
}

enum CensTables {
    None,
    /// Per group: the survival just before each jump and the increment of the
    /// hazard at it, and the curve itself for `G(T-)`.
    Km {
        group: Vec<u32>,
        g_before: Vec<Vec<f64>>,
        d_lambda: Vec<Vec<f64>>,
        /// The cumulative hazard just before and at each jump.
        lam_before: Vec<Vec<f64>>,
        lam_at: Vec<Vec<f64>>,
        curve: Vec<Steps>,
    },
    /// The risk score of each person, the baseline cumulative hazard just before
    /// each jump and its increment, and the baseline.
    Cox {
        score: Vec<f64>,
        h_before: Vec<f64>,
        h_at: Vec<f64>,
        d_h: Vec<f64>,
        base: Steps,
    },
}

struct Model {
    n: usize,
    k: usize,
    times: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    treated: Vec<bool>,
    /// The linear predictor of the event model at each level.
    lp: [Vec<f64>; 2],
    /// The baseline cumulative hazard of the event model.
    base: Steps,
    /// The probability of the treatment actually received.
    pi: Vec<f64>,
    n_trimmed: usize,
    tables: Tables,
    /// The cells of censoring times of the grid, as index ranges into the jumps.
    cells: Option<Vec<(usize, usize)>>,
    max_stop: f64,
}

fn expit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

fn linear_predictor(x: MatRef<'_, f64>, cf: Counterfactual<'_>, beta: &[f64], row: usize) -> f64 {
    let p = x.ncols();
    let mut eta = 0.0;
    match cf {
        Counterfactual::Design(m) => {
            for j in 0..p {
                eta += m[(row, j)] * beta[j];
            }
        }
        Counterfactual::Columns(set) => {
            for j in 0..p {
                let v = set
                    .iter()
                    .find(|&&(c, _)| c == j)
                    .map_or(x[(row, j)], |&(_, v)| v);
                eta += v * beta[j];
            }
        }
    }
    eta
}

fn coefficients(fit: &crate::cox::CoxModelFit, what: &str) -> Result<Vec<f64>> {
    if !fit.converged {
        return Err(Error::input(format!("the {what} model did not converge")));
    }
    fit.coefficients
        .iter()
        .enumerate()
        .map(|(j, c)| {
            c.ok_or_else(|| Error::input(format!("the {what} model has an aliased column {j}")))
        })
        .collect()
}

/// Fit a Cox model and give its coefficients and the Breslow baseline cumulative
/// hazard at zero covariates.
fn cox_model(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    ties: Ties,
    what: &str,
) -> Result<(Vec<f64>, Steps)> {
    let options = CoxOptions {
        ties,
        timefix: TimeFix::Exact,
        ..CoxOptions::default()
    };
    let fit = cox_fit(data, x, &options)?;
    let beta = coefficients(&fit, what)?;
    let zero = vec![0.0; x.ncols()];
    let base = basehaz(
        &FittedModel {
            data,
            x,
            fit: &fit,
            options: &options,
        },
        &BaselineOptions {
            centring: Centring::Given(&zero),
            hazard: Some(HazardType::Breslow),
            variance: Variance::Model,
        },
    )?;
    let steps = base.strata.first().map_or(
        Steps {
            time: Vec::new(),
            first: 0.0,
            value: Vec::new(),
        },
        |s| Steps {
            time: s.time.clone(),
            first: 0.0,
            value: s.cumulative_hazard.clone(),
        },
    );
    Ok((beta, steps))
}

fn check_design(name: &str, x: MatRef<'_, f64>, n: usize) -> Result<()> {
    if x.nrows() != n {
        return Err(Error::input(format!(
            "{name}: {} rows for {n} rows of data",
            x.nrows()
        )));
    }
    if x.ncols() == 0 {
        return Err(Error::input(format!("{name}: no columns")));
    }
    if !epistat_linalg::is_finite(x) {
        return Err(Error::input(format!("{name}: a value is not finite")));
    }
    Ok(())
}

impl Model {
    #[expect(
        clippy::too_many_lines,
        reason = "one pass of checks and fits that share their locals"
    )]
    fn fit(
        data: &SurvData<'_>,
        spec: &AipwSurvSpec<'_>,
        times: &[f64],
        opts: &AipwSurvOptions,
    ) -> Result<Self> {
        if !(opts.conf_level > 0.0 && opts.conf_level < 1.0) {
            return Err(Error::input("conf_level must be in (0, 1)"));
        }
        if let Some((lo, hi)) = opts.truncation
            && !(0.0 < lo && lo < hi && hi < 1.0)
        {
            return Err(Error::input(format!(
                "truncation: ({lo}, {hi}) are not probabilities in increasing order"
            )));
        }
        if times.is_empty()
            || times.iter().any(|t| !(t.is_finite() && *t > 0.0))
            || times.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::input(
                "times: need at least one, finite, positive and increasing",
            ));
        }
        let fixed = data.clone().with_timefix(opts.timefix)?;
        let surv = fixed.followup()?;
        if surv.start().is_some()
            || fixed.weight().is_some()
            || fixed.stratum().is_some()
            || fixed.cluster().is_some()
        {
            return Err(Error::input(
                "data: entry times, case weights, strata and clusters are not supported",
            ));
        }
        let n = fixed.n_rows();
        if spec.treatment.len() != n {
            return Err(Error::input(format!(
                "treatment: {} values for {n} rows",
                spec.treatment.len()
            )));
        }
        let n_treated = spec.treatment.iter().filter(|&&a| a).count();
        if n_treated == 0 || n_treated == n {
            return Err(Error::input(
                "treatment: both treated and untreated rows are needed",
            ));
        }
        let stop = surv.stop().to_vec();
        let status = surv.status().to_vec();
        if status.iter().any(|&s| s != 0 && s != 1) {
            return Err(Error::input("data: the status must be 0 or 1"));
        }
        check_design("outcome_x", spec.outcome_x, n)?;
        check_design("propensity_x", spec.propensity_x, n)?;
        for (l, cf) in spec.levels.iter().enumerate() {
            if let Counterfactual::Design(m) = cf {
                check_design(&format!("levels[{l}]"), *m, n)?;
                if m.ncols() != spec.outcome_x.ncols() {
                    return Err(Error::input(format!(
                        "levels[{l}]: {} columns for the {} of the model",
                        m.ncols(),
                        spec.outcome_x.ncols()
                    )));
                }
            }
            if let Counterfactual::Columns(set) = cf
                && set.iter().any(|&(c, _)| c >= spec.outcome_x.ncols())
            {
                return Err(Error::input(format!(
                    "levels[{l}]: a column is outside the design"
                )));
            }
        }
        // The event model, on the data with the times as fixed.
        let buf = SurvBuf::builder(stop.clone(), status.clone()).build()?;
        let process = buf.as_data();
        let (beta, base) = cox_model(&process, spec.outcome_x, opts.ties, "event")?;
        let mut lp = [vec![0.0; n], vec![0.0; n]];
        for (l, v) in lp.iter_mut().enumerate() {
            for (i, slot) in v.iter_mut().enumerate() {
                let eta = linear_predictor(spec.outcome_x, spec.levels[l], &beta, i);
                *slot = eta;
            }
        }
        // The model of the treatment.
        let y: Vec<f64> = spec
            .treatment
            .iter()
            .map(|&a| f64::from(u8::from(a)))
            .collect();
        let glm = Glm::new(Family::Binomial, Link::Logit)
            .fit(spec.propensity_x, &y, None, None, None)
            .map_err(|e| Error::input(format!("treatment model: {e}")))?;
        if !glm.converged {
            return Err(Error::input("treatment model: no convergence"));
        }
        let mut gamma = Vec::with_capacity(glm.coefficients.len());
        for (j, c) in glm.coefficients.iter().enumerate() {
            gamma.push(c.ok_or_else(|| {
                Error::input(format!("treatment model: design column {j} is aliased"))
            })?);
        }
        let mut n_trimmed = 0;
        let mut pi = Vec::with_capacity(n);
        for i in 0..n {
            let eta: f64 = (0..gamma.len())
                .map(|j| spec.propensity_x[(i, j)] * gamma[j])
                .sum();
            let mut e = expit(eta);
            if !(e > 0.0 && e < 1.0) {
                return Err(Error::input(format!(
                    "treatment model: row {i} has fitted probability {e}"
                )));
            }
            if let Some((lo, hi)) = opts.truncation {
                let clamped = e.clamp(lo, hi);
                if clamped.to_bits() != e.to_bits() {
                    n_trimmed += 1;
                }
                e = clamped;
            }
            pi.push(if spec.treatment[i] { e } else { 1.0 - e });
        }
        // The censoring times and the tables over them.
        let mut jump: Vec<f64> = (0..n)
            .filter(|&i| status[i] == 0)
            .map(|i| stop[i])
            .collect();
        jump.sort_by(f64::total_cmp);
        jump.dedup();
        let (h_before, h_at): (Vec<f64>, Vec<f64>) =
            jump.iter().map(|&u| (base.before(u), base.at(u))).unzip();
        let cens = match spec.censoring {
            Censoring::None => {
                if !jump.is_empty() {
                    return Err(Error::input(
                        "data: rows with status 0 need a censoring model",
                    ));
                }
                CensTables::None
            }
            Censoring::KaplanMeier { groups } => {
                let group: Vec<u32> = match groups {
                    Some(g) if g.len() == n => g.to_vec(),
                    Some(g) => {
                        return Err(Error::input(format!(
                            "groups: {} values for {n} rows",
                            g.len()
                        )));
                    }
                    None => vec![0; n],
                };
                km_tables(&stop, &status, &group, &jump, opts.product_limit)
            }
            Censoring::Cox { x } => {
                check_design("censoring x", x, n)?;
                let flipped: Vec<i32> = status.iter().map(|&s| 1 - s).collect();
                let cbuf = SurvBuf::builder(stop.clone(), flipped).build()?;
                let cdata = cbuf.as_data();
                let (c, hc) = cox_model(&cdata, x, opts.ties, "censoring")?;
                let score: Vec<f64> = (0..n)
                    .map(|i| {
                        let eta: f64 = (0..x.ncols()).map(|j| x[(i, j)] * c[j]).sum();
                        eta.exp()
                    })
                    .collect();
                let hb: Vec<f64> = jump.iter().map(|&u| hc.before(u)).collect();
                let ha: Vec<f64> = jump.iter().map(|&u| hc.at(u)).collect();
                let dh: Vec<f64> = jump.iter().map(|&u| hc.at(u) - hc.before(u)).collect();
                CensTables::Cox {
                    score,
                    h_before: hb,
                    h_at: ha,
                    d_h: dh,
                    base: hc,
                }
            }
        };
        let max_stop = stop.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let cells = opts.grid.map(|k| cells_of(&jump, times, k)).transpose()?;
        Ok(Self {
            n,
            k: times.len(),
            times: times.to_vec(),
            stop,
            status,
            treated: spec.treatment.to_vec(),
            lp,
            base,
            pi,
            n_trimmed,
            tables: Tables {
                jump,
                h_before,
                h_at,
                cens,
            },
            cells,
            max_stop,
        })
    }

    /// The censoring survival just before `t` of person `i`.
    fn g_before(&self, i: usize, t: f64) -> f64 {
        match &self.tables.cens {
            CensTables::None => 1.0,
            CensTables::Km { group, curve, .. } => curve[group[i] as usize].before(t).exp(),
            CensTables::Cox { score, base, .. } => (-score[i] * base.before(t)).exp(),
        }
    }

    /// The row of the person `i`, its values at each time, for the level of the
    /// treatment received. `aug` is the augmentation of the person at each
    /// time when it is formed for all persons at once.
    fn person(
        &self,
        i: usize,
        aiptw: bool,
        aug: Option<&[f64]>,
        out: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<()> {
        let k = self.k;
        let a = usize::from(self.treated[i]);
        let w = 1.0 / self.pi[i];
        let (t_i, d_i) = (self.stop[i], self.status[i]);
        // The risk at each time under each level.
        for l in 0..2 {
            let theta = self.lp[l][i].exp();
            for (s, &tau) in self.times.iter().enumerate() {
                scratch.f[l][s] = 1.0 - (-theta * self.base.at(tau)).exp();
            }
        }
        // The weight of the event at each time.
        let g_t = if d_i == 1 { self.g_before(i, t_i) } else { 1.0 };
        if d_i == 1 && (g_t.is_nan() || g_t <= 0.0) {
            return Err(Error::input(format!(
                "the censoring survival is 0 before the event of row {i}"
            )));
        }
        for (s, &tau) in self.times.iter().enumerate() {
            let y = if t_i <= tau && d_i == 1 {
                1.0 / g_t
            } else {
                0.0
            };
            scratch.ipcw[s] = y;
        }
        scratch.aug.iter_mut().for_each(|v| *v = 0.0);
        if aiptw && !matches!(self.tables.cens, CensTables::None) {
            match (&self.cells, aug) {
                (Some(cells), _) => self.augmentation_cells(i, a, cells, scratch),
                (None, Some(row)) => scratch.aug.copy_from_slice(row),
                (None, None) => {
                    let Scratch { f, aug, .. } = scratch;
                    self.augmentation(i, a, &f[a], aug);
                }
            }
        }
        for l in 0..2 {
            for s in 0..k {
                let f = scratch.f[l][s];
                out[l * k + s] = if aiptw {
                    if l == a {
                        f + w * (scratch.ipcw[s] - f + scratch.aug[s])
                    } else {
                        f
                    }
                } else if l == a {
                    w * scratch.ipcw[s]
                } else {
                    0.0
                };
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error::input(format!(
                "row {i} has a value that is not finite"
            )));
        }
        Ok(())
    }

    /// The augmentation of person `i` at each time, from the level `a` received
    /// with the risk `f` at each time, summed over the censoring times one by
    /// one.
    fn augmentation(&self, i: usize, a: usize, f: &[f64], aug: &mut [f64]) {
        let theta = self.lp[a][i].exp();
        let t_i = self.stop[i];
        let last = *self.times.last().unwrap_or(&0.0);
        let reach = t_i.min(last);
        let j_max = self.tables.jump.partition_point(|&u| u <= reach);
        let (mut sum_a, mut sum_b) = (0.0, 0.0);
        let mut s = 0;
        let censored_here = self.status[i] == 0;
        for j in 0..j_max {
            let u = self.tables.jump[j];
            while s < self.k && self.times[s] < u {
                aug[s] = f[s] * sum_a - sum_b;
                s += 1;
            }
            let (g, d_lambda) = match &self.tables.cens {
                CensTables::None => (1.0, 0.0),
                CensTables::Km {
                    group,
                    g_before,
                    d_lambda,
                    ..
                } => (
                    g_before[group[i] as usize][j],
                    d_lambda[group[i] as usize][j],
                ),
                CensTables::Cox {
                    score,
                    h_before,
                    d_h,
                    ..
                } => ((-score[i] * h_before[j]).exp(), score[i] * d_h[j]),
            };
            let surv = (-theta * self.tables.h_before[j]).exp();
            let mut d_m = -d_lambda;
            if censored_here && u.total_cmp(&t_i).is_eq() {
                d_m += 1.0;
            }
            let term = d_m / (g * surv);
            let f_u = 1.0 - (-theta * self.tables.h_at[j]).exp();
            sum_a += term;
            sum_b += f_u * term;
        }
        while s < self.k {
            aug[s] = f[s] * sum_a - sum_b;
            s += 1;
        }
    }

    /// The augmentation of person `i` with the censoring times pooled into
    /// cells. A cell is used from the person's first censoring time in it. The
    /// integral of the compensator over the cell is the increment of the
    /// censoring hazard over the jumps of the cell up to the person's exit times
    /// the mean of the integrand at the first and the last of them, and the jump
    /// of the person's own censoring is added at its own time.
    fn augmentation_cells(
        &self,
        i: usize,
        a: usize,
        cells: &[(usize, usize)],
        scratch: &mut Scratch,
    ) {
        let theta = self.lp[a][i].exp();
        let t_i = self.stop[i];
        let last = *self.times.last().unwrap_or(&0.0);
        let reach = t_i.min(last);
        let jump = &self.tables.jump;
        let n_cells = cells.partition_point(|&(j0, _)| jump[j0] <= reach);
        let up_to_exit = jump.partition_point(|&u| u <= t_i);
        let own = jump.partition_point(|&u| u < t_i);
        let censored_here = self.status[i] == 0;
        let (mut sum_a, mut sum_b) = (0.0, 0.0);
        let mut s = 0;
        for &(j0, j1) in &cells[..n_cells] {
            let u = jump[j0];
            while s < self.k && self.times[s] < u {
                scratch.aug[s] = scratch.f[a][s] * sum_a - sum_b;
                s += 1;
            }
            let j_e = (j1 - 1).min(up_to_exit - 1);
            // The survival of the censoring just before jump `j` and the
            // increment of its cumulative hazard over the cell.
            let g_at = |j: usize| match &self.tables.cens {
                CensTables::None => 1.0,
                CensTables::Km {
                    group, g_before, ..
                } => g_before[group[i] as usize][j],
                CensTables::Cox {
                    score, h_before, ..
                } => (-score[i] * h_before[j]).exp(),
            };
            let d_lambda = match &self.tables.cens {
                CensTables::None => 0.0,
                CensTables::Km {
                    group,
                    lam_before,
                    lam_at,
                    ..
                } => {
                    let gr = group[i] as usize;
                    lam_at[gr][j_e] - lam_before[gr][j0]
                }
                CensTables::Cox {
                    score,
                    h_before,
                    h_at,
                    ..
                } => score[i] * (h_at[j_e] - h_before[j0]),
            };
            let inv = |j: usize| 1.0 / (g_at(j) * (-theta * self.tables.h_before[j]).exp());
            let f_at = |j: usize| 1.0 - (-theta * self.tables.h_at[j]).exp();
            // The compensator by the trapezoid rule over the cell, the jump of
            // the person's own censoring at its own time.
            let (i0, ie) = (inv(j0), inv(j_e));
            let comp = -d_lambda;
            sum_a += comp * 0.5 * (i0 + ie);
            sum_b += comp * 0.5 * (f_at(j0) * i0 + f_at(j_e) * ie);
            if censored_here && j0 <= own && own < j1 {
                let w = inv(own);
                sum_a += w;
                sum_b += f_at(own) * w;
            }
        }
        while s < self.k {
            scratch.aug[s] = scratch.f[a][s] * sum_a - sum_b;
            s += 1;
        }
    }

    /// The augmentation of every person at each time, `n` rows of `k`, from the
    /// level received, by the moment sums of the module documentation.
    fn augmentations(&self, route: Route) -> Vec<f64> {
        let (n, k) = (self.n, self.k);
        let mut out = vec![0.0; n * k];
        let jump = &self.tables.jump;
        let last = self.times[k - 1];
        let reach = jump.partition_point(|&u| u <= last);
        if reach == 0 || matches!(self.tables.cens, CensTables::None) {
            return out;
        }
        let theta: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| self.lp[usize::from(self.treated[i])][i].exp())
            .collect();
        let j_max: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| jump.partition_point(|&u| u <= self.stop[i]).min(reach))
            .collect();
        let x_top = self.tables.h_at[reach - 1];
        let (group, s, y_top): (Vec<usize>, Option<&[f64]>, f64) = match &self.tables.cens {
            CensTables::Km { group, .. } => {
                (group.iter().map(|&g| g as usize).collect(), None, 0.0)
            }
            CensTables::Cox {
                score, h_before, ..
            } => (vec![0; n], Some(score), h_before[reach - 1]),
            CensTables::None => return out,
        };
        let r_s = if s.is_some() && y_top > 0.0 {
            RADIUS / 2.0
        } else {
            0.0
        };
        let r_theta = RADIUS - r_s;
        // The bins: persons who reach a censoring time, by group, in strips of
        // theta and, within a strip, in bins of s. The orders are total, so the
        // bins do not depend on the sort.
        let n_groups = group.iter().copied().max().map_or(0, |g| g + 1);
        let mut by_group: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
        for i in (0..n).filter(|&i| j_max[i] > 0) {
            by_group[group[i]].push(i);
        }
        let mut bins: Vec<Bin> = Vec::new();
        for (g, mut rows) in by_group.into_iter().enumerate() {
            rows.par_sort_unstable_by(|&p, &q| theta[p].total_cmp(&theta[q]).then(p.cmp(&q)));
            for strip in runs(&rows, |i| theta[i], r_theta / x_top) {
                match s {
                    None => bins.push(Bin::new(g, strip.to_vec(), &theta, None)),
                    Some(s) => {
                        let mut strip = strip.to_vec();
                        strip.par_sort_unstable_by(|&p, &q| s[p].total_cmp(&s[q]).then(p.cmp(&q)));
                        for part in runs(&strip, |i| s[i], r_s / y_top) {
                            bins.push(Bin::new(g, part.to_vec(), &theta, Some(s)));
                        }
                    }
                }
            }
        }
        let h_tau: Vec<f64> = self.times.iter().map(|&tau| self.base.at(tau)).collect();
        let tau_idx: Vec<usize> = self
            .times
            .iter()
            .map(|&tau| jump.partition_point(|&u| u <= tau))
            .collect();
        let ctx = BinContext {
            theta: &theta,
            s,
            j_max: &j_max,
            h_tau: &h_tau,
            tau_idx: &tau_idx,
            x_top,
            y_top,
        };
        let values: Vec<Vec<f64>> = bins
            .par_iter()
            .map(|bin| self.bin_augmentation(bin, &ctx, route))
            .collect();
        // Where each person's row is in the values of its bin.
        let mut at: Vec<Option<(usize, usize)>> = vec![None; n];
        for (b, bin) in bins.iter().enumerate() {
            for (r, &i) in bin.rows.iter().enumerate() {
                at[i] = Some((b, r));
            }
        }
        out.par_chunks_mut(k).zip(&at).for_each(|(row, at)| {
            if let Some((b, r)) = *at {
                row.copy_from_slice(&values[b][r * k..(r + 1) * k]);
            }
        });
        out
    }

    /// The augmentation of the persons of a bin at each time, a row of `k` for
    /// each person in the order of the bin.
    ///
    /// The running sums at `P` censoring times are those at the start of its
    /// chunk of [`CHUNK`] censoring times, themselves the sums of the chunks
    /// before added in order, plus the censoring times of the chunk before `P`
    /// added in order. They are the same whichever thread forms them, so the
    /// chunks and the persons are processed in parallel.
    fn bin_augmentation(&self, bin: &Bin, ctx: &BinContext<'_>, route: Route) -> Vec<f64> {
        let k = self.k;
        let rows = &bin.rows;
        let span = rows.iter().map(|&i| ctx.j_max[i]).max().unwrap_or(0);
        let series = Series::new(self, bin, ctx);
        let reached: usize = rows.iter().map(|&i| ctx.j_max[i]).sum();
        let direct = match route {
            Route::Direct => true,
            Route::Moments => false,
            // A censoring time costs a person about three exponentials summed
            // one by one, and the bin about four exponentials and two products
            // per moment.
            Route::ByCost => 32 * reached < (64 + 2 * series.moments()) * span,
        };
        let mut out = vec![0.0; rows.len() * k];
        if direct {
            out.par_chunks_mut(k).zip(rows).for_each(|(row, &i)| {
                let a = usize::from(self.treated[i]);
                let f: Vec<f64> = ctx
                    .h_tau
                    .iter()
                    .map(|&h| 1.0 - (-ctx.theta[i] * h).exp())
                    .collect();
                self.augmentation(i, a, &f, row);
            });
            return out;
        }
        // The sums at the start of each chunk.
        let n_chunks = span.div_ceil(CHUNK);
        let totals: Vec<Sums> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let (mut sums, mut powers) = (series.zero(), series.powers());
                for j in c * CHUNK..((c + 1) * CHUNK).min(span) {
                    series.add(j, &mut sums, &mut powers);
                }
                sums
            })
            .collect();
        let mut starts: Vec<Sums> = Vec::with_capacity(n_chunks + 1);
        starts.push(series.zero());
        for t in &totals {
            let mut next = starts[starts.len() - 1].clone();
            next.add_sums(t);
            starts.push(next);
        }
        drop(totals);
        // The queries, each person at each time as an index into `out`, by the
        // number of censoring times included: a counting sort in row order.
        let p_of = |r: usize, t: usize| ctx.tau_idx[t].min(ctx.j_max[rows[r]]);
        let mut first = vec![0_usize; span + 2];
        for r in 0..rows.len() {
            for t in 0..k {
                first[p_of(r, t) + 1] += 1;
            }
        }
        for p in 0..=span {
            first[p + 1] += first[p];
        }
        let mut fill = first.clone();
        let mut queries = vec![0_usize; rows.len() * k];
        for r in 0..rows.len() {
            for t in 0..k {
                let p = p_of(r, t);
                queries[fill[p]] = r * k + t;
                fill[p] += 1;
            }
        }
        drop(fill);
        // The values a person needs at every time: its offsets in the bin and
        // the jump of its own censoring.
        let person: Vec<Person> = rows
            .par_iter()
            .map(|&i| {
                let th = ctx.theta[i];
                let own = (self.status[i] == 0).then(|| {
                    let own = self.tables.jump.partition_point(|&u| u < self.stop[i]);
                    let (w, f) = self.own_censoring(i, th, own);
                    (own, w, f)
                });
                Person {
                    theta: th,
                    t: if bin.d_theta > 0.0 {
                        (th - bin.theta0) / bin.d_theta
                    } else {
                        0.0
                    },
                    scale: ctx.s.map_or(1.0, |s| s[i]),
                    v: match ctx.s {
                        Some(s) if bin.d_s > 0.0 => (s[i] - bin.s0) / bin.d_s,
                        _ => 0.0,
                    },
                    own,
                }
            })
            .collect();
        let values: Vec<Vec<f64>> = (0..queries.len().div_ceil(QUERIES))
            .into_par_iter()
            .map(|b| {
                let (q0, q1) = (b * QUERIES, ((b + 1) * QUERIES).min(queries.len()));
                let mut p = first.partition_point(|&f| f <= q0) - 1;
                let mut j = (p / CHUNK) * CHUNK;
                let (mut sums, mut powers) = (starts[j / CHUNK].clone(), series.powers());
                let mut v = Vec::with_capacity(q1 - q0);
                let mut at = None;
                for (q, &query) in (q0..q1).zip(&queries[q0..q1]) {
                    while first[p + 1] <= q {
                        p += 1;
                    }
                    if at != Some(p) {
                        while j < p {
                            if j.is_multiple_of(CHUNK) {
                                sums.clone_from(&starts[j / CHUNK]);
                            }
                            series.add(j, &mut sums, &mut powers);
                            j += 1;
                        }
                        if p.is_multiple_of(CHUNK) {
                            sums.clone_from(&starts[p / CHUNK]);
                        }
                        at = Some(p);
                    }
                    let (r, t) = (query / k, query % k);
                    let who = &person[r];
                    let (ca, cb) = series.value(&sums, who.t, who.v);
                    let (mut sa, mut sb) = (-who.scale * ca, -who.scale * cb);
                    if let Some((own, w, f_own)) = who.own
                        && own < p
                    {
                        sa += w;
                        sb += f_own * w;
                    }
                    let f = 1.0 - (-who.theta * ctx.h_tau[t]).exp();
                    v.push(f * sa - sb);
                }
                v
            })
            .collect();
        for (b, v) in values.iter().enumerate() {
            for (q, &x) in v.iter().enumerate() {
                out[queries[b * QUERIES + q]] = x;
            }
        }
        out
    }

    /// The weight `1 / (G(T_i-) S(T_i-))` of the censoring of person `i` at its
    /// own time, the censoring time `own`, and the risk `F(T_i)` there, as the
    /// sum person by person forms them.
    fn own_censoring(&self, i: usize, theta: f64, own: usize) -> (f64, f64) {
        let g = match &self.tables.cens {
            CensTables::None => 1.0,
            CensTables::Km {
                group, g_before, ..
            } => g_before[group[i] as usize][own],
            CensTables::Cox {
                score, h_before, ..
            } => (-score[i] * h_before[own]).exp(),
        };
        let surv = (-theta * self.tables.h_before[own]).exp();
        let f = 1.0 - (-theta * self.tables.h_at[own]).exp();
        (1.0 / (g * surv), f)
    }

    fn estimate(&self, opts: &AipwSurvOptions, route: Route) -> Result<AipwSurv> {
        let (n, k) = (self.n, self.k);
        let aiptw = opts.estimator == Estimator::Aiptw;
        let aug = (aiptw && self.cells.is_none()).then(|| self.augmentations(route));
        // The values phi_i for both levels and every time.
        let mut phi = vec![0.0; n * 2 * k];
        phi.par_chunks_mut(2 * k * BLOCK)
            .enumerate()
            .try_for_each(|(b, chunk)| {
                let mut scratch = Scratch::new(k);
                for (r, row) in chunk.chunks_mut(2 * k).enumerate() {
                    let i = b * BLOCK + r;
                    let own = aug.as_ref().map(|v| &v[i * k..(i + 1) * k]);
                    self.person(i, aiptw, own, row, &mut scratch)?;
                }
                Ok::<(), Error>(())
            })?;
        let sums = par::ordered_reduce(
            n,
            BLOCK,
            WAVE,
            |range| {
                let mut s = vec![0.0; 2 * k];
                for i in range {
                    for (v, p) in s.iter_mut().zip(&phi[i * 2 * k..(i + 1) * 2 * k]) {
                        *v += p;
                    }
                }
                s
            },
            |a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            },
        );
        let nf = n as f64;
        let est: Vec<f64> = sums.iter().map(|s| s / nf).collect();
        // The influence function and the covariances.
        let mut influence = [vec![0.0; n * k], vec![0.0; n * k]];
        let cov = par::ordered_reduce(
            n,
            BLOCK,
            WAVE,
            |range| {
                let mut c = vec![0.0; 3 * k];
                for i in range {
                    for s in 0..k {
                        let a = (phi[i * 2 * k + s] - est[s]) / nf;
                        let b = (phi[i * 2 * k + k + s] - est[k + s]) / nf;
                        c[s] += a * a;
                        c[k + s] += b * b;
                        c[2 * k + s] += a * b;
                    }
                }
                c
            },
            |a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            },
        );
        for i in 0..n {
            for s in 0..k {
                influence[0][i * k + s] = (phi[i * 2 * k + s] - est[s]) / nf;
                influence[1][i * k + s] = (phi[i * 2 * k + k + s] - est[k + s]) / nf;
            }
        }
        let z = epistat_stats::dist::normal::two_sided_z(opts.conf_level);
        let mut risk = vec![None; 2 * k];
        let mut std_err = vec![None; 2 * k];
        let mut lower = vec![None; 2 * k];
        let mut upper = vec![None; 2 * k];
        let mut covariance = vec![None; k * 4];
        for s in 0..k {
            if self.times[s] > self.max_stop {
                continue;
            }
            for l in 0..2 {
                let e = est[l * k + s];
                let v = cov[l * k + s];
                let se = v.max(0.0).sqrt();
                risk[l * k + s] = Some(e);
                std_err[l * k + s] = Some(se);
                lower[l * k + s] = Some(e - z * se);
                upper[l * k + s] = Some(e + z * se);
            }
            covariance[s * 4] = Some(cov[s]);
            covariance[s * 4 + 3] = Some(cov[k + s]);
            covariance[s * 4 + 1] = Some(cov[2 * k + s]);
            covariance[s * 4 + 2] = Some(cov[2 * k + s]);
        }
        let weights = WeightSummary::new((0..n).map(|i| 1.0 / self.pi[i]));
        Ok(AipwSurv {
            risks: Standardised {
                n_levels: 2,
                n_times: k,
                n_rows: n,
                n_used: n,
                risk,
                std_err: Some(std_err),
                lower: Some(lower),
                upper: Some(upper),
                covariance: Some(covariance),
                conf_level: opts.conf_level,
            },
            times: self.times.clone(),
            influence,
            n_trimmed: self.n_trimmed,
            weights,
        })
    }
}

struct Scratch {
    f: [Vec<f64>; 2],
    ipcw: Vec<f64>,
    aug: Vec<f64>,
}

impl Scratch {
    fn new(k: usize) -> Self {
        Self {
            f: [vec![0.0; k], vec![0.0; k]],
            ipcw: vec![0.0; k],
            aug: vec![0.0; k],
        }
    }
}

/// The largest `delta X + epsilon Y` of a bin of the moment sums.
const RADIUS: f64 = 2.0;
/// The bound on the relative truncation of the series of a bin, `2^-57`.
const TRUNCATION: f64 = f64::EPSILON / 32.0;

/// How the augmentation of a bin is summed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// By moment sums or person by person, whichever costs less.
    ByCost,
    /// By moment sums.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the tests compare the two routes")
    )]
    Moments,
    /// Person by person.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the tests compare the two routes")
    )]
    Direct,
}

/// The censoring times per chunk of the running sums of a bin.
const CHUNK: usize = 1024;
/// The queries per parallel task of a bin.
const QUERIES: usize = 8192;

/// The running sums of a bin: the coefficients of `A` and of `B`, row `m` of
/// the powers of `delta` and column `k` of the powers of `epsilon`.
#[derive(Clone)]
struct Sums {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl Sums {
    fn add_sums(&mut self, other: &Self) {
        for (x, y) in self.a.iter_mut().zip(&other.a) {
            *x += y;
        }
        for (x, y) in self.b.iter_mut().zip(&other.b) {
            *x += y;
        }
    }
}

/// The scaled powers `xi^m / m!`, `(-zeta)^m / m!` and `eta^k / k!` of one
/// censoring time.
struct Powers {
    xi: Vec<f64>,
    zeta: Vec<f64>,
    eta: Vec<f64>,
}

/// What a person of a bin needs at each time.
struct Person {
    theta: f64,
    /// The offsets `delta / d_theta` and `epsilon / d_s` in `[0, 1]`.
    t: f64,
    v: f64,
    /// The censoring score of a Cox model of the censoring, 1 for Kaplan-Meier.
    scale: f64,
    /// The person's own censoring time, its weight and the risk there.
    own: Option<(usize, f64, f64)>,
}

/// The series of a bin: its degree in each offset and the terms each censoring
/// time adds to the running sums.
struct Series<'a> {
    model: &'a Model,
    group: usize,
    theta0: f64,
    d_theta: f64,
    s0: f64,
    d_s: f64,
    degree: usize,
    m_theta: usize,
    m_s: usize,
}

impl<'a> Series<'a> {
    fn new(model: &'a Model, bin: &Bin, ctx: &BinContext<'_>) -> Self {
        let degree = degree(bin.d_theta * ctx.x_top + bin.d_s * ctx.y_top);
        Self {
            model,
            group: bin.group,
            theta0: bin.theta0,
            d_theta: bin.d_theta,
            s0: bin.s0,
            d_s: bin.d_s,
            degree,
            m_theta: if bin.d_theta * ctx.x_top > 0.0 {
                degree
            } else {
                0
            },
            m_s: if bin.d_s * ctx.y_top > 0.0 { degree } else { 0 },
        }
    }

    /// The powers of `epsilon` in row `m`.
    fn len(&self, m: usize) -> usize {
        (self.degree - m).min(self.m_s) + 1
    }

    fn width(&self) -> usize {
        self.m_s + 1
    }

    fn moments(&self) -> usize {
        (0..=self.m_theta).map(|m| self.len(m)).sum()
    }

    fn zero(&self) -> Sums {
        let size = (self.m_theta + 1) * self.width();
        Sums {
            a: vec![0.0; size],
            b: vec![0.0; size],
        }
    }

    fn powers(&self) -> Powers {
        Powers {
            xi: vec![0.0; self.m_theta + 1],
            zeta: vec![0.0; self.m_theta + 1],
            eta: vec![0.0; self.width()],
        }
    }

    /// Adds the terms of censoring time `j`.
    fn add(&self, j: usize, sums: &mut Sums, p: &mut Powers) {
        let tables = &self.model.tables;
        let (w, y) = match &tables.cens {
            CensTables::Km {
                g_before, d_lambda, ..
            } => (d_lambda[self.group][j] / g_before[self.group][j], 0.0),
            CensTables::Cox { d_h, h_before, .. } => (d_h[j], h_before[j]),
            CensTables::None => (0.0, 0.0),
        };
        if w == 0.0 {
            return;
        }
        let (x, ha) = (tables.h_before[j], tables.h_at[j]);
        let dx = ha - x;
        let e_s = if y > 0.0 { (self.s0 * y).exp() } else { 1.0 };
        let e_x = (self.theta0 * x).exp();
        let e_tie = if dx > 0.0 {
            (-self.theta0 * dx).exp()
        } else {
            1.0
        };
        let (xi, zeta, eta) = (self.d_theta * x, -(self.d_theta * dx), self.d_s * y);
        p.xi[0] = 1.0;
        p.zeta[0] = 1.0;
        for m in 1..=self.m_theta {
            p.xi[m] = p.xi[m - 1] * xi / m as f64;
            p.zeta[m] = p.zeta[m - 1] * zeta / m as f64;
        }
        p.eta[0] = 1.0;
        for kk in 1..self.width() {
            p.eta[kk] = p.eta[kk - 1] * eta / kk as f64;
        }
        let wa = w * e_s * e_x;
        let wb = w * e_s;
        let width = self.width();
        for m in 0..=self.m_theta {
            let coef_b = if m == 0 {
                e_x * -(-self.theta0 * ha).exp_m1()
            } else if dx > 0.0 {
                e_x * p.xi[m] - e_tie * p.zeta[m]
            } else {
                e_x * p.xi[m]
            };
            let (ca, cb) = (wa * p.xi[m], wb * coef_b);
            let len = self.len(m);
            let row_a = &mut sums.a[m * width..m * width + len];
            for (v, q) in row_a.iter_mut().zip(&p.eta[..len]) {
                *v += ca * q;
            }
            let row_b = &mut sums.b[m * width..m * width + len];
            for (v, q) in row_b.iter_mut().zip(&p.eta[..len]) {
                *v += cb * q;
            }
        }
    }

    /// The sums `A` and `B` without the factor of the censoring score, at the
    /// offsets `t` and `v` from the running sums.
    fn value(&self, sums: &Sums, t: f64, v: f64) -> (f64, f64) {
        let width = self.width();
        let (mut ca, mut cb, mut tp) = (0.0, 0.0, 1.0);
        for m in 0..=self.m_theta {
            let (mut ra, mut rb, mut vp) = (0.0, 0.0, 1.0);
            for kk in 0..self.len(m) {
                ra += vp * sums.a[m * width + kk];
                rb += vp * sums.b[m * width + kk];
                vp *= v;
            }
            ca += tp * ra;
            cb += tp * rb;
            tp *= t;
        }
        (ca, cb)
    }
}

/// Persons whose augmentation is formed from the same moment sums.
struct Bin {
    /// The group of the Kaplan-Meier censoring curve, 0 for a Cox model.
    group: usize,
    rows: Vec<usize>,
    /// The lowest `theta` and the spread of `theta` above it.
    theta0: f64,
    d_theta: f64,
    /// The lowest censoring score and the spread above it, 0 and 0 without a
    /// Cox model of the censoring.
    s0: f64,
    d_s: f64,
}

impl Bin {
    fn new(group: usize, rows: Vec<usize>, theta: &[f64], s: Option<&[f64]>) -> Self {
        let range = |v: &[f64]| {
            let lo = rows.iter().map(|&i| v[i]).fold(f64::INFINITY, f64::min);
            let hi = rows.iter().map(|&i| v[i]).fold(f64::NEG_INFINITY, f64::max);
            (lo, hi - lo)
        };
        let (theta0, d_theta) = range(theta);
        let (s0, d_s) = s.map_or((0.0, 0.0), range);
        Self {
            group,
            rows,
            theta0,
            d_theta,
            s0,
            d_s,
        }
    }
}

/// The values shared by the bins of [`Model::augmentations`].
struct BinContext<'a> {
    theta: &'a [f64],
    /// The censoring score of each person for a Cox model of the censoring.
    s: Option<&'a [f64]>,
    /// The censoring times each person reaches.
    j_max: &'a [usize],
    /// The baseline cumulative hazard of the event at each time asked for.
    h_tau: &'a [f64],
    /// The censoring times up to each time asked for.
    tau_idx: &'a [usize],
    /// The largest `H_0(u_j)` and `Lambda^C_0(u_j-)` reached.
    x_top: f64,
    y_top: f64,
}

/// The total degree at which the series of `exp(z)` for `0 <= z <= r` stops
/// with a relative truncation below [`TRUNCATION`]: the remainder after degree
/// `M` is at most `r^(M+1) e^r / (M+1)!` of the value.
fn degree(r: f64) -> usize {
    if r <= 0.0 || !r.is_finite() {
        return 0;
    }
    let mut bound = r.exp();
    let mut m = 0;
    loop {
        bound *= r / (m + 1) as f64;
        if bound <= TRUNCATION || m == 64 {
            return m;
        }
        m += 1;
    }
}

/// Splits `rows`, sorted by `value`, into runs whose values lie within `width`
/// of the first of the run. A width that is not finite gives one run.
fn runs(rows: &[usize], value: impl Fn(usize) -> f64, width: f64) -> Vec<&[usize]> {
    if !width.is_finite() {
        return if rows.is_empty() {
            Vec::new()
        } else {
            vec![rows]
        };
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let lo = value(rows[start]);
        let mut end = start + 1;
        while end < rows.len() && value(rows[end]) - lo <= width {
            end += 1;
        }
        out.push(&rows[start..end]);
        start = end;
    }
    out
}

/// The cells of `k` about equal numbers of censoring times, cut also before the
/// first censoring time after each of the `times`, so that no cell holds times
/// on both sides of a time asked for.
fn cells_of(jump: &[f64], times: &[f64], k: usize) -> Result<Vec<(usize, usize)>> {
    if k == 0 {
        return Err(Error::input("grid: at least one cell is needed"));
    }
    let j = jump.len();
    let mut cuts: Vec<usize> = (0..=k).map(|c| c * j / k).collect();
    cuts.extend(times.iter().map(|&t| jump.partition_point(|&u| u <= t)));
    cuts.push(j);
    cuts.sort_unstable();
    cuts.dedup();
    Ok(cuts
        .windows(2)
        .filter(|w| w[0] < w[1])
        .map(|w| (w[0], w[1]))
        .collect())
}

/// The Kaplan-Meier curves of the censoring times by group, on the log scale for
/// `G(T-)`, with the survival before each jump and the hazard increment there.
fn km_tables(
    stop: &[f64],
    status: &[i32],
    group: &[u32],
    jump: &[f64],
    product_limit: bool,
) -> CensTables {
    let n_groups = group.iter().copied().max().map_or(0, |g| g as usize + 1);
    let mut by_group: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (i, &g) in group.iter().enumerate() {
        by_group[g as usize].push(i);
    }
    let mut g_before = Vec::with_capacity(n_groups);
    let mut d_lambda = Vec::with_capacity(n_groups);
    let mut lam_before = Vec::with_capacity(n_groups);
    let mut lam_at = Vec::with_capacity(n_groups);
    let mut curve = Vec::with_capacity(n_groups);
    for rows in &by_group {
        let mut times: Vec<f64> = rows.iter().map(|&i| stop[i]).collect();
        times.sort_by(f64::total_cmp);
        let mut censored: Vec<f64> = rows
            .iter()
            .filter(|&&i| status[i] == 0)
            .map(|&i| stop[i])
            .collect();
        censored.sort_by(f64::total_cmp);
        let mut steps = Steps {
            time: Vec::new(),
            first: 0.0,
            value: Vec::new(),
        };
        let mut dl = vec![0.0; jump.len()];
        let mut gb = vec![1.0; jump.len()];
        let mut lb = vec![0.0; jump.len()];
        let mut la = vec![0.0; jump.len()];
        let mut surv = 1.0_f64;
        let mut cum = 0.0_f64;
        let mut c = 0;
        for (j, &u) in jump.iter().enumerate() {
            gb[j] = surv;
            lb[j] = cum;
            let at_risk = times.len() - times.partition_point(|&t| t < u);
            let mut d = 0;
            while c < censored.len() && censored[c] <= u {
                d += usize::from(censored[c].total_cmp(&u).is_eq());
                c += 1;
            }
            if d > 0 {
                let h = d as f64 / at_risk as f64;
                dl[j] = h;
                cum += h;
                surv *= 1.0 - h;
                steps.time.push(u);
                steps
                    .value
                    .push(if product_limit { surv.ln() } else { -cum });
            }
            la[j] = cum;
        }
        g_before.push(gb);
        d_lambda.push(dl);
        lam_before.push(lb);
        lam_at.push(la);
        curve.push(steps);
    }
    CensTables::Km {
        group: group.to_vec(),
        g_before,
        d_lambda,
        lam_before,
        lam_at,
        curve,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::Mat;
    use epistat_rng::Stream;

    struct Cohort {
        time: Vec<f64>,
        status: Vec<i32>,
        treated: Vec<bool>,
        x: Mat<f64>,
        z: Mat<f64>,
        group: Vec<u32>,
    }

    /// Confounded treatment and censoring that depends on the covariates, with
    /// times rounded to a quarter for ties of events and censorings, or not.
    fn cohort(n: usize, seed: u64, rounded: bool) -> Cohort {
        let mut rng = Stream::seeded(seed);
        let (mut time, mut status, mut treated, mut cols) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let x1 = rng.normal();
            let x2 = f64::from(u8::from(rng.uniform() < 0.4));
            let a = rng.uniform() < expit(0.5 * x1 - 0.4 * x2);
            let af = f64::from(u8::from(a));
            let te = -rng.uniform().ln() / (0.12 * (0.5 * x1 - 0.4 * x2 - 0.6 * af).exp());
            let tc = (-rng.uniform().ln() / (0.06 * (0.3 * x1 + 0.3 * af).exp())).min(12.0);
            let t = te.min(tc);
            time.push(if rounded { (t * 4.0).ceil() / 4.0 } else { t });
            status.push(i32::from(te <= tc));
            treated.push(a);
            cols.push([af, x1, x2]);
        }
        Cohort {
            time,
            status,
            group: treated.iter().map(|&a| u32::from(a)).collect(),
            treated,
            x: Mat::from_fn(n, 3, |i, j| cols[i][j]),
            z: Mat::from_fn(n, 3, |i, j| [1.0, cols[i][1], cols[i][2]][j]),
        }
    }

    fn model(c: &Cohort, km: bool, times: &[f64]) -> Model {
        let data = SurvData::builder(&c.time, &c.status).build().unwrap();
        let (c0, c1) = ([(0usize, 0.0)], [(0usize, 1.0)]);
        let spec = AipwSurvSpec {
            treatment: &c.treated,
            outcome_x: c.x.as_ref(),
            levels: [Counterfactual::Columns(&c0), Counterfactual::Columns(&c1)],
            propensity_x: c.z.as_ref(),
            censoring: if km {
                Censoring::KaplanMeier {
                    groups: Some(&c.group),
                }
            } else {
                Censoring::Cox { x: c.x.as_ref() }
            },
        };
        let opts = AipwSurvOptions {
            ties: Ties::Breslow,
            ..AipwSurvOptions::default()
        };
        Model::fit(&data, &spec, times, &opts).unwrap()
    }

    /// The sum of the absolute terms of the integral of person `i` over the
    /// first `p` censoring times, the scale of the rounding error of both sums.
    fn absolute_terms(m: &Model, i: usize, p: usize) -> f64 {
        let theta = m.lp[usize::from(m.treated[i])][i].exp();
        (0..p)
            .map(|j| {
                let (g, d_lambda) = match &m.tables.cens {
                    CensTables::Km {
                        group,
                        g_before,
                        d_lambda,
                        ..
                    } => (
                        g_before[group[i] as usize][j],
                        d_lambda[group[i] as usize][j],
                    ),
                    CensTables::Cox {
                        score,
                        h_before,
                        d_h,
                        ..
                    } => ((-score[i] * h_before[j]).exp(), score[i] * d_h[j]),
                    CensTables::None => (1.0, 0.0),
                };
                let own = m.status[i] == 0 && m.tables.jump[j].total_cmp(&m.stop[i]).is_eq();
                (d_lambda + f64::from(u8::from(own))) / (g * (-theta * m.tables.h_before[j]).exp())
            })
            .sum()
    }

    // The moment sums and the sum person by person each have the rounding
    // error of a sum of positive terms, at most `(P + 3 M + moments + 8) eps`
    // of the sum of the absolute terms for `P` censoring times, degree `M <=
    // 25` and at most 351 moments, below `(P + 4 M^2) eps`, and a truncation
    // below `eps / 32`. The sum `B` has terms below twice those of `A`, and
    // the augmentation `F A - B` adds `(F + 2)` times the error of `A` and its
    // own rounding.
    #[test]
    fn the_moment_sums_agree_with_the_sum_person_by_person() {
        let times = [2.0, 5.0, 8.0];
        // The last cohort has several chunks of censoring times and several
        // tasks of queries in its largest bins.
        for (n, seed, rounded) in [(3000, 7, true), (3000, 11, false), (12_000, 13, false)] {
            let c = cohort(n, seed, rounded);
            for km in [true, false] {
                let m = model(&c, km, &times);
                let (by_moments, direct) = (
                    m.augmentations(Route::Moments),
                    m.augmentations(Route::Direct),
                );
                let reach = m.tables.jump.partition_point(|&u| u <= 8.0);
                let mut worst: f64 = 0.0;
                for i in 0..m.n {
                    let j_max = m
                        .tables
                        .jump
                        .partition_point(|&u| u <= m.stop[i])
                        .min(reach);
                    let theta = m.lp[usize::from(m.treated[i])][i].exp();
                    for (t, &tau) in times.iter().enumerate() {
                        let p = m.tables.jump.partition_point(|&u| u <= tau).min(j_max);
                        let f = 1.0 - (-theta * m.base.at(tau)).exp();
                        let scale = absolute_terms(&m, i, p);
                        let tol = (f + 2.0) * (2.0 * p as f64 + 2520.0) * f64::EPSILON * scale;
                        let (a, b) = (by_moments[i * 3 + t], direct[i * 3 + t]);
                        assert!(
                            (a - b).abs() <= tol,
                            "km {km} rounded {rounded} row {i} time {tau}: {a} {b} tol {tol}"
                        );
                        if scale > 0.0 {
                            worst = worst.max((a - b).abs() / (scale * f64::EPSILON));
                        }
                    }
                }
                println!(
                    "n {n} km {km}: {} censoring times, largest difference {worst:.1} eps of the terms",
                    m.tables.jump.len()
                );
            }
        }
    }

    // The estimate is the mean of `phi`, which differs from one route to the
    // other by `w_i` times the difference of the augmentation, so the
    // estimates differ by at most the mean of `w_i` times its bound above, and
    // the standard errors, norms of `(phi - est) / n`, by at most the norm of
    // the difference of those vectors.
    #[test]
    fn the_estimates_agree_between_the_routes() {
        let times = [2.0, 5.0, 8.0];
        let c = cohort(3000, 11, false);
        for km in [true, false] {
            let m = model(&c, km, &times);
            let opts = AipwSurvOptions::default();
            let (a, b) = (
                m.estimate(&opts, Route::Moments).unwrap(),
                m.estimate(&opts, Route::Direct).unwrap(),
            );
            let aug = m.augmentations(Route::Direct);
            let reach = m.tables.jump.partition_point(|&u| u <= 8.0);
            let nf = m.n as f64;
            for (t, &tau) in times.iter().enumerate() {
                let mut bound = vec![0.0; m.n];
                for (i, slot) in bound.iter_mut().enumerate() {
                    let j_max = m
                        .tables
                        .jump
                        .partition_point(|&u| u <= m.stop[i])
                        .min(reach);
                    let p = m.tables.jump.partition_point(|&u| u <= tau).min(j_max);
                    let theta = m.lp[usize::from(m.treated[i])][i].exp();
                    let f = 1.0 - (-theta * m.base.at(tau)).exp();
                    let tol = (f + 2.0) * (2.0 * p as f64 + 2520.0) * f64::EPSILON;
                    // The rounding of `phi` itself, which both routes share up
                    // to the augmentation.
                    let phi = f64::EPSILON * 4.0 * (1.0 + (f + aug[i * 3 + t].abs()) / m.pi[i]);
                    *slot = tol * absolute_terms(&m, i, p) / m.pi[i] + phi;
                }
                let d_est = bound.iter().sum::<f64>() / nf;
                let d_se = bound
                    .iter()
                    .map(|v| (v + d_est) * (v + d_est))
                    .sum::<f64>()
                    .sqrt()
                    / nf;
                for l in 0..2 {
                    let idx = l * 3 + t;
                    let (ea, eb) = (a.risks.risk[idx].unwrap(), b.risks.risk[idx].unwrap());
                    let se = |r: &AipwSurv| r.risks.std_err.as_ref().unwrap()[idx].unwrap();
                    assert!(
                        (ea - eb).abs() <= d_est,
                        "km {km} level {l} time {tau}: {ea} {eb}"
                    );
                    assert!(
                        (se(&a) - se(&b)).abs() <= d_se * 1.01,
                        "km {km} level {l} time {tau}: {} {}",
                        se(&a),
                        se(&b)
                    );
                    println!(
                        "km {km} level {l} time {tau}: risk {:.1e} of {d_est:.1e}, se {:.1e} of {d_se:.1e}",
                        (ea - eb).abs(),
                        (se(&a) - se(&b)).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn the_moment_sums_are_the_same_on_every_pool() {
        // Several chunks of censoring times and several tasks of queries.
        let c = cohort(12_000, 13, false);
        for km in [true, false] {
            let m = model(&c, km, &[2.0, 5.0, 8.0]);
            epistat_testkit::assert_same_on_pools(&epistat_testkit::POOLS, || {
                m.augmentations(Route::Moments)
            });
        }
    }

    #[test]
    fn the_degree_bounds_the_truncation() {
        assert_eq!(degree(0.0), 0);
        let m = degree(RADIUS);
        assert!(m <= 25, "{m}");
        let mut bound = RADIUS.exp();
        for d in 1..=m + 1 {
            bound *= RADIUS / d as f64;
        }
        assert!(bound <= TRUNCATION);
    }
}
