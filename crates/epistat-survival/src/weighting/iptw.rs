//! Inverse probability of treatment weights from a logistic propensity model.
//!
//! # Algorithm
//!
//! The probability of treatment given the covariates, `p_i = P(A_i = 1 | z_i)`,
//! is the fit of a logistic regression of the treatment on the design `z` by
//! iteratively reweighted least squares, as `glm(family = binomial)`. The
//! weight of subject `i` is, for the estimand
//!
//! ```text
//! ATE   a_i / p_i + (1 - a_i) / (1 - p_i)
//! ATT   a_i + (1 - a_i) p_i / (1 - p_i)
//! ATC   (1 - a_i) + a_i (1 - p_i) / p_i
//! ```
//!
//! which are the weights of `WeightIt::weightit(method = "glm")`. The
//! stabilised weight of the average treatment effect multiplies the treated
//! weights by `pi` and the control weights by `1 - pi`, with `pi` the
//! proportion treated, the fit of the logistic model with an intercept only
//! (Robins, Hernan and Brumback 2000). `WeightIt` stabilises only for the
//! average treatment effect, and so does this module.
//!
//! The summary of the weights per treatment group is the range, mean,
//! coefficient of variation and Kish's effective sample size.
//!
//! # Edge cases
//!
//! A design column that is a combination of the earlier ones (an aliased
//! column), a model that does not converge, and a fitted probability of
//! exactly 0 or 1 are errors, since the weights would be undefined or
//! infinite. A weight that is very large without the probability reaching 0
//! or 1, which is what near-separation looks like, is not an error and shows
//! in the summary.

use epistat_glm::{Family, Glm, Link};
use faer::MatRef;

use super::summary::WeightSummary;
use crate::error::{Error, Result};

/// The population the weights standardise to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Estimand {
    /// The whole population, `estimand = "ATE"`.
    #[default]
    Ate,
    /// The treated, `estimand = "ATT"`: controls are weighted to look like the
    /// treated.
    Att,
    /// The controls, `estimand = "ATC"`.
    Atc,
}

/// What [`fit`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IptwOptions {
    /// The population the weights standardise to. The default is `ATE`, as in
    /// `weightit`.
    pub estimand: Estimand,
    /// Multiply the weights by the marginal probability of the treatment each
    /// subject received, `stabilize = TRUE` of `weightit`. Only for the average
    /// treatment effect. The default is unstabilised weights.
    pub stabilise: bool,
}

/// The summary of the weights in each treatment group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IptwSummary {
    /// The treated, `None` when there are none.
    pub treated: Option<WeightSummary>,
    /// The controls.
    pub control: Option<WeightSummary>,
}

/// The weights of a propensity model, and what the variance of an outcome fitted
/// with them needs.
///
/// It borrows the treatment and the design, which the stacked variance reads
/// again.
#[derive(Debug, Clone)]
pub struct Iptw<'a> {
    /// The estimand of the weights.
    pub estimand: Estimand,
    /// Whether the weights are stabilised.
    pub stabilised: bool,
    /// The weight of each subject.
    pub weights: Vec<f64>,
    /// The fitted probability of treatment of each subject.
    pub propensity: Vec<f64>,
    /// The coefficients of the propensity model, one per column of the design.
    pub coefficients: Vec<f64>,
    /// The proportion treated, used by stabilised weights, otherwise `None`.
    pub marginal: Option<f64>,
    /// The summary of the weights by treatment group.
    pub summary: IptwSummary,
    pub(super) treatment: &'a [bool],
    pub(super) design: MatRef<'a, f64>,
}

impl Iptw<'_> {
    /// The number of subjects.
    #[must_use]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether there are no subjects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// Fit the propensity model of `treatment` on `design` and weight the subjects.
///
/// `design` holds one row per subject and includes the intercept column, as
/// the design of `glm`. `treatment` is true for the treated.
///
/// # Errors
///
/// [`Error::Input`] when `design` has another number of rows than `treatment`,
/// no column, or a value that is not finite, when everyone or no one is
/// treated, when `stabilise` is set for an estimand other than `Ate`, when the
/// propensity model has an aliased column or does not converge, and when a
/// fitted probability is exactly 0 or 1.
pub fn fit<'a>(
    treatment: &'a [bool],
    design: MatRef<'a, f64>,
    opts: &IptwOptions,
) -> Result<Iptw<'a>> {
    let n = treatment.len();
    if design.nrows() != n {
        return Err(Error::input(format!(
            "design has {} rows for {n} subjects",
            design.nrows()
        )));
    }
    if design.ncols() == 0 {
        return Err(Error::input("design has no columns"));
    }
    if !epistat_linalg::is_finite(design) {
        return Err(Error::input("design: a value is not finite"));
    }
    let n_treated = treatment.iter().filter(|&&a| a).count();
    if n_treated == 0 || n_treated == n {
        return Err(Error::input("treatment: both groups need subjects"));
    }
    if opts.stabilise && opts.estimand != Estimand::Ate {
        return Err(Error::input(
            "stabilise: weights are stabilised for the average treatment effect only",
        ));
    }
    let y: Vec<f64> = treatment.iter().map(|&a| f64::from(u8::from(a))).collect();
    let model = Glm::new(Family::Binomial, Link::Logit)
        .fit(design, &y, None, None, None)
        .map_err(|e| Error::input(format!("propensity model: {e}")))?;
    if !model.converged {
        return Err(Error::input("propensity model: no convergence"));
    }
    let mut coefficients = Vec::with_capacity(model.coefficients.len());
    for (j, c) in model.coefficients.iter().enumerate() {
        coefficients.push(c.ok_or_else(|| {
            Error::input(format!("propensity model: design column {j} is aliased"))
        })?);
    }
    let propensity = model.fitted;
    if let Some(i) = propensity.iter().position(|&p| !(p > 0.0 && p < 1.0)) {
        return Err(Error::input(format!(
            "propensity model: subject {i} has fitted probability {}",
            propensity[i]
        )));
    }
    let marginal = opts.stabilise.then(|| n_treated as f64 / n as f64);
    let weights: Vec<f64> = treatment
        .iter()
        .zip(&propensity)
        .map(|(&a, &p)| {
            let base = match (opts.estimand, a) {
                (Estimand::Ate, true) => 1.0 / p,
                (Estimand::Ate, false) => 1.0 / (1.0 - p),
                (Estimand::Att, true) | (Estimand::Atc, false) => 1.0,
                (Estimand::Att, false) => p / (1.0 - p),
                (Estimand::Atc, true) => (1.0 - p) / p,
            };
            match marginal {
                Some(pi) if a => pi * base,
                Some(pi) => (1.0 - pi) * base,
                None => base,
            }
        })
        .collect();
    let group = |treated: bool| {
        WeightSummary::new(
            weights
                .iter()
                .zip(treatment)
                .filter(|&(_, &a)| a == treated)
                .map(|(&w, _)| w),
        )
    };
    let summary = IptwSummary {
        treated: group(true),
        control: group(false),
    };
    Ok(Iptw {
        estimand: opts.estimand,
        stabilised: opts.stabilise,
        weights,
        propensity,
        coefficients,
        marginal,
        summary,
        treatment,
        design,
    })
}
