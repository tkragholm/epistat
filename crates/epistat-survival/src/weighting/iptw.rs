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
//! ATO   a_i (1 - p_i) + (1 - a_i) p_i
//! ATM   min(p_i, 1 - p_i) / (a_i p_i + (1 - a_i)(1 - p_i))
//! ATOS  (a_i / p_i + (1 - a_i) / (1 - p_i)) 1(alpha <= p_i <= 1 - alpha)
//! ```
//!
//! which are the weights of `WeightIt::weightit(method = "glm")`. The
//! stabilised weight of the average treatment effect multiplies the treated
//! weights by `pi` and the control weights by `1 - pi`, with `pi` the
//! proportion treated, the fit of the logistic model with an intercept only
//! (Robins, Hernan and Brumback 2000). `WeightIt` stabilises only for the
//! average treatment effect, and so does this module.
//!
//! The overlap weights (ATO) target the population with weight `p (1 - p)`, the
//! one with the most overlap between the groups, and balance every covariate of
//! the logistic model exactly: with an intercept, the weighted sums of the
//! covariates in the two groups are equal, `sum a_i (1 - p_i) z_i = sum (1 - a_i)
//! p_i z_i`, because the score equation of the model is `sum (a_i - p_i) z_i = 0`
//! (Li, Morgan and Zaslavsky 2018). The matching weights (ATM) weight each
//! subject by the smaller of the two probabilities over the probability of the
//! group it is in, the population that one-to-one matching on the propensity
//! score would give (Li and Greene 2013). The optimal subset (ATOS) weights as
//! the average treatment effect and gives weight 0 outside `[alpha, 1 - alpha]`.
//! The `alpha` of Crump, Hotz, Imbens and Mitnik (2009) is, over the candidates
//! `alpha` in increasing order among the sorted values of `p` and `1 - p` below
//! one half, the first with
//!
//! ```text
//! 2 alpha (1 - alpha) * mean(1 / (p_i (1 - p_i)) over p_i (1 - p_i) >= alpha (1 - alpha)) >= 1
//! ```
//!
//! as `WeightIt` computes it, 0 if none qualifies. Candidates that differ from
//! the previous by less than `1e-8` are skipped, as there. [`fit_atos`] takes an
//! `alpha` of the caller instead.
//!
//! The variance that accounts for the estimated propensity model (see
//! [`weighting`](super)) takes the weights as functions of the coefficients and
//! the cut-off `alpha` of ATOS, and the 0.5 of ATM, as fixed.
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
//! in the summary. The weights of ATOS are 0 for the subjects outside the
//! optimal subset, and the summary includes them.
//!
//! # References
//!
//! 1. Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score
//!    weighting. J Am Stat Assoc. 2018;113(521):390-400.
//! 2. Li L, Greene T. A weighting analogue to pair matching in propensity score
//!    analysis. Int J Biostat. 2013;9(2):215-34.
//! 3. Crump RK, Hotz VJ, Imbens GW, Mitnik OA. Dealing with limited overlap in
//!    estimation of average treatment effects. Biometrika. 2009;96(1):187-99.
//! 4. Greifer N. WeightIt: weighting for covariate balance in observational
//!    studies. R package.
//! 5. Zhou T, Tong G, Li F, Thomas LE, Li F. PSweight: an R package for
//!    propensity score weighting analysis. R J. 2022;14(1):282-300.

use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{Mat, MatRef};

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
    /// The population with the most overlap, weight `p (1 - p)`,
    /// `estimand = "ATO"`: treated subjects are weighted by `1 - p` and controls
    /// by `p`.
    Ato,
    /// The population of matching on the propensity score, `estimand = "ATM"`.
    Atm,
    /// The optimal subset of Crump et al., `estimand = "ATOS"`: the weights of the
    /// average treatment effect on the subjects with propensity in `[alpha, 1 -
    /// alpha]` and 0 outside.
    Atos,
}

/// What [`fit`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct IptwOptions {
    /// The population the weights standardise to. The default is `ATE`, as in
    /// `weightit`.
    pub estimand: Estimand,
    /// Multiply the weights by the marginal probability of the treatment each
    /// subject received, `stabilize = TRUE` of `weightit`. Only for the average
    /// treatment effect. The default is unstabilised weights.
    pub stabilise: bool,
    /// The cut-off `alpha` of the optimal subset for [`Estimand::Atos`], in
    /// `[0, 0.5)`, or `None` for the one of Crump et al. as `WeightIt` finds
    /// it. Ignored for the other estimands.
    pub alpha: Option<f64>,
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
/// It keeps the treatment and the design, which the stacked variance reads
/// again.
#[derive(Debug, Clone)]
pub struct Iptw {
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
    /// The cut-off `alpha` of the optimal subset, for `Atos` only.
    pub alpha: Option<f64>,
    /// The summary of the weights by treatment group.
    pub summary: IptwSummary,
    pub(super) treatment: Vec<bool>,
    pub(super) design: Mat<f64>,
}

impl Iptw {
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
/// the design of `glm`. `treatment` is true for the treated. For
/// [`Estimand::Atos`] the weights are those of the average treatment effect
/// inside the optimal subset between `alpha` and `1 - alpha` and 0 outside,
/// and [`Iptw::alpha`] holds the cut-off used.
///
/// # Errors
///
/// An input error when `design` has another number of rows than `treatment`,
/// no column, or a value that is not finite, when everyone or no one is
/// treated, when `stabilise` is set for an estimand other than `Ate`, when
/// `alpha` is not in `[0, 0.5)`, and when a fitted probability is exactly 0
/// or 1; [`Error::Model`] when the propensity model has an aliased column or
/// does not converge.
pub fn fit(treatment: &[bool], design: MatRef<'_, f64>, opts: &IptwOptions) -> Result<Iptw> {
    if opts.estimand == Estimand::Atos
        && let Some(a) = opts.alpha
        && !(0.0..0.5).contains(&a)
    {
        return Err(Error::invalid(format!("alpha: {a} is not in [0, 0.5)")));
    }
    let alpha = if opts.estimand == Estimand::Atos {
        opts.alpha
    } else {
        None
    };
    fit_with(treatment, design, *opts, alpha)
}

/// The cut-off of Crump et al. as `WeightIt` finds it, 0 if none qualifies.
fn crump_alpha(propensity: &[f64]) -> f64 {
    let n = propensity.len();
    let mut sorted: Vec<f64> = propensity
        .iter()
        .copied()
        .chain(propensity.iter().map(|&p| 1.0 - p))
        .collect();
    sorted.sort_by(f64::total_cmp);
    // `z` ascending with the suffix sums of `1 / z`.
    let mut z: Vec<f64> = propensity.iter().map(|&p| p * (1.0 - p)).collect();
    z.sort_by(f64::total_cmp);
    let mut suffix = vec![0.0; n + 1];
    for i in (0..n).rev() {
        suffix[i] = suffix[i + 1] + 1.0 / z[i];
    }
    let below = propensity.iter().filter(|&&p| p < 0.5).count();
    for i in 0..below {
        if i > 0 && (sorted[i] - sorted[i - 1]).abs() < 1e-8 {
            continue;
        }
        let alpha = sorted[i];
        let a = alpha * (1.0 - alpha);
        let first = z.partition_point(|&v| v < a);
        let count = (n - first) as f64;
        if count > 0.0 && 2.0 * a * suffix[first] / count >= 1.0 {
            return alpha;
        }
    }
    0.0
}

fn fit_with(
    treatment: &[bool],
    design: MatRef<'_, f64>,
    opts: IptwOptions,
    alpha: Option<f64>,
) -> Result<Iptw> {
    let n = treatment.len();
    if design.nrows() != n {
        return Err(Error::invalid(format!(
            "design has {} rows for {n} subjects",
            design.nrows()
        )));
    }
    if design.ncols() == 0 {
        return Err(Error::invalid("design has no columns"));
    }
    if !epistat_linalg::is_finite(design) {
        return Err(Error::invalid("design: a value is not finite"));
    }
    let n_treated = treatment.iter().filter(|&&a| a).count();
    if n_treated == 0 || n_treated == n {
        return Err(Error::invalid("treatment: both groups need subjects"));
    }
    if opts.stabilise && opts.estimand != Estimand::Ate {
        return Err(Error::invalid(
            "stabilise: weights are stabilised for the average treatment effect only",
        ));
    }
    let y: Vec<f64> = treatment.iter().map(|&a| f64::from(u8::from(a))).collect();
    let model = Glm::new(Family::Binomial, Link::Logit)
        .fit(&crate::logistic::data(design, &y, None, None, None)?)
        .map_err(Error::model("propensity model"))?;
    let coefficients = model
        .full_rank()
        .map_err(Error::model("propensity model"))?;
    let propensity = model.fitted;
    if let Some(i) = propensity.iter().position(|&p| !(p > 0.0 && p < 1.0)) {
        return Err(Error::invalid(format!(
            "propensity model: subject {i} has fitted probability {}",
            propensity[i]
        )));
    }
    let marginal = opts.stabilise.then(|| n_treated as f64 / n as f64);
    let alpha = (opts.estimand == Estimand::Atos)
        .then(|| alpha.unwrap_or_else(|| crump_alpha(&propensity)));
    let weights: Vec<f64> = treatment
        .iter()
        .zip(&propensity)
        .map(|(&a, &p)| {
            // The optimal subset weights as the average treatment effect inside it.
            let estimand = if opts.estimand == Estimand::Atos {
                Estimand::Ate
            } else {
                opts.estimand
            };
            let outside = alpha.is_some_and(|al| p < al || p > 1.0 - al);
            let base = match (estimand, a) {
                _ if outside => 0.0,
                (Estimand::Ate | Estimand::Atos, true) => 1.0 / p,
                (Estimand::Ate | Estimand::Atos, false) => 1.0 / (1.0 - p),
                (Estimand::Att, false) => p / (1.0 - p),
                (Estimand::Atc, true) => (1.0 - p) / p,
                (Estimand::Ato, true) => 1.0 - p,
                (Estimand::Ato, false) => p,
                (Estimand::Atm, true) if p > 0.5 => (1.0 - p) / p,
                (Estimand::Atm, false) if p < 0.5 => p / (1.0 - p),
                (Estimand::Att, true) | (Estimand::Atc, false) | (Estimand::Atm, _) => 1.0,
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
        alpha,
        summary,
        treatment: treatment.to_vec(),
        design: design.to_owned(),
    })
}
