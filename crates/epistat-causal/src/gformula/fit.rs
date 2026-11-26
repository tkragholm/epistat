//! The covariate, competing-event and outcome models, fitted to the observed
//! rows with each person's prior weight.

use epistat_glm::{Family, Glm, Link, Multinom};
use epistat_linalg::Mat;

use super::history::{History, Term};
use super::prepare::Prepared;
use super::{Censoring, Covariate, CovariateKind, FittedModel, Models, Outcome, Restriction};
use crate::error::{Error, Result};

/// A fitted linear predictor and how it maps to a mean; for a multinomial
/// model, `classes - 1` linear predictors one after the other in `beta`.
pub(super) struct Linear {
    pub(super) beta: Vec<f64>,
    pub(super) link: Link,
    pub(super) model: FittedModel,
    /// The number of classes of a multinomial model, 0 otherwise.
    pub(super) classes: usize,
}

impl Linear {
    /// The mean of the model on the history.
    pub(super) fn mean(&self, terms: &[Term], h: &History<'_>) -> f64 {
        let mut eta = 0.0;
        for (t, b) in terms.iter().zip(&self.beta) {
            eta += t.eval(h) * b;
        }
        self.link.inverse(eta)
    }

    /// The class of a multinomial model whose cumulative probability, in the
    /// order of the classes, first exceeds `u`.
    pub(super) fn class(&self, terms: &[Term], h: &History<'_>, u: f64) -> f64 {
        let q = terms.len();
        let x: Vec<f64> = terms.iter().map(|t| t.eval(h)).collect();
        let exps: Vec<f64> = (1..self.classes)
            .map(|c| {
                let mut eta = 0.0;
                for (xj, b) in x.iter().zip(&self.beta[(c - 1) * q..c * q]) {
                    eta += xj * b;
                }
                eta.exp()
            })
            .collect();
        let total = 1.0 + exps.iter().sum::<f64>();
        let mut cumulative = 1.0 / total;
        if u < cumulative {
            return 0.0;
        }
        for (c, e) in exps.iter().enumerate() {
            cumulative += e / total;
            if u < cumulative {
                return (c + 1) as f64;
            }
        }
        (self.classes - 1) as f64
    }
}

pub(super) struct Fitted {
    pub(super) covariates: Vec<Option<Linear>>,
    pub(super) outcome: Linear,
    pub(super) competing: Option<Linear>,
}

impl Fitted {
    pub(super) fn report(&self) -> Models {
        Models {
            covariates: self
                .covariates
                .iter()
                .map(|c| c.as_ref().map(|l| l.model.clone()))
                .collect(),
            outcome: self.outcome.model.clone(),
            competing: self.competing.as_ref().map(|l| l.model.clone()),
            censoring: None,
        }
    }
}

/// Whether the model of covariate `j` applies to the history.
pub(super) fn applies(restriction: Option<&Restriction>, j: usize, h: &History<'_>) -> bool {
    match restriction {
        None => true,
        Some(Restriction::Absorbing) => h.lag(j, 1) == 0.0,
        Some(Restriction::When { applies, .. }) => applies(h),
    }
}

/// Fits every model with person weights `weights`.
pub(super) fn fit_all(
    prepared: &Prepared<'_>,
    covariates: &[Covariate],
    outcome: &Outcome<'_>,
    weights: &[f64],
) -> Result<Fitted> {
    let w = prepared.width;
    let fitted_covariates = covariates
        .iter()
        .enumerate()
        .map(|(j, c)| {
            if prepared.periods < 2 {
                return Ok(None);
            }
            let (family, link) = match c.kind {
                CovariateKind::Binary => (Family::Binomial, Link::Logit),
                CovariateKind::Normal => (Family::Gaussian, Link::Identity),
                CovariateKind::Categorical(levels) => {
                    return fit_multinomial(
                        prepared,
                        &c.terms,
                        levels,
                        weights,
                        &format!("the model of variable {j}"),
                        |p, r, k| {
                            if k == 0 {
                                return None;
                            }
                            let h = prepared.history(p, k, j);
                            applies(c.restriction.as_ref(), j, &h)
                                .then(|| (prepared.values[r * w + j], j))
                        },
                    )
                    .map(Some);
                }
            };
            fit_model(
                prepared,
                &c.terms,
                family,
                link,
                weights,
                &format!("the model of variable {j}"),
                |p, r, k| {
                    if k == 0 {
                        return None;
                    }
                    let h = prepared.history(p, k, j);
                    applies(c.restriction.as_ref(), j, &h).then(|| (prepared.values[r * w + j], j))
                },
            )
            .map(Some)
        })
        .collect::<Result<Vec<_>>>()?;
    let (outcome, competing) = match outcome {
        Outcome::Survival {
            event,
            terms,
            competing,
            censoring,
        } => {
            let censored = |r: usize| censoring.as_ref().is_some_and(|c| c.censored[r]);
            let fitted_outcome = fit_model(
                prepared,
                terms,
                Family::Binomial,
                Link::Logit,
                weights,
                "the outcome model",
                |_, r, _| {
                    let d = competing.as_ref().is_some_and(|c| c.event[r]);
                    (!d && !censored(r)).then(|| (f64::from(u8::from(event[r])), w))
                },
            )?;
            let fitted_competing = competing
                .as_ref()
                .map(|c| {
                    fit_model(
                        prepared,
                        &c.terms,
                        Family::Binomial,
                        Link::Logit,
                        weights,
                        "the competing-event model",
                        |_, r, _| (!censored(r)).then(|| (f64::from(u8::from(c.event[r])), w)),
                    )
                })
                .transpose()?;
            (fitted_outcome, fitted_competing)
        }
        Outcome::BinaryEnd { value, terms } | Outcome::ContinuousEnd { value, terms } => {
            let (family, link) = if matches!(outcome, Outcome::BinaryEnd { .. }) {
                (Family::Binomial, Link::Logit)
            } else {
                (Family::Gaussian, Link::Identity)
            };
            let last = prepared.last_period;
            let fitted_outcome = fit_model(
                prepared,
                terms,
                family,
                link,
                weights,
                "the outcome model",
                |_, r, k| (k == last && !value[r].is_nan()).then(|| (value[r], w)),
            )?;
            (fitted_outcome, None)
        }
    };
    Ok(Fitted {
        covariates: fitted_covariates,
        outcome,
        competing,
    })
}

/// Fits the censoring model on every row, with the whole history of the
/// period visible.
pub(super) fn fit_censoring(
    prepared: &Prepared<'_>,
    censoring: &Censoring<'_>,
    weights: &[f64],
) -> Result<Linear> {
    fit_model(
        prepared,
        &censoring.terms,
        Family::Binomial,
        Link::Logit,
        weights,
        "the censoring model",
        |_, r, _| Some((f64::from(u8::from(censoring.censored[r])), prepared.width)),
    )
}

/// Fits one model. `row(p, r, k)` gives, for row `r` of person `p` at period
/// `k`, the response and the number of variables visible at period `k`, or
/// `None` when the row does not enter.
fn fit_model(
    prepared: &Prepared<'_>,
    terms: &[Term],
    family: Family,
    link: Link,
    weights: &[f64],
    what: &str,
    row: impl Fn(usize, usize, usize) -> Option<(f64, usize)>,
) -> Result<Linear> {
    let q = terms.len();
    let mut design = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for (p, &weight) in weights.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        for (k, r) in prepared.rows_of(p).enumerate() {
            let Some((response, visible)) = row(p, r, k) else {
                continue;
            };
            let h = prepared.history(p, k, visible);
            design.extend(terms.iter().map(|t| t.eval(&h)));
            y.push(response);
            w.push(weight);
        }
    }
    let n = y.len();
    if n == 0 {
        return Err(Error::input(format!("{what}: no rows to fit")));
    }
    let x = Mat::from_fn(n, q, |i, j| design[i * q + j]);
    let glm = Glm::new(family, link);
    let fit = glm
        .fit(x.as_ref(), &y, Some(&w), None, None)
        .map_err(|e| Error::input(format!("{what}: {e}")))?;
    if !fit.converged {
        return Err(Error::NoConvergence(format!("{what}: no convergence")));
    }
    let beta = fit
        .coefficients
        .iter()
        .enumerate()
        .map(|(j, c)| c.ok_or_else(|| Error::input(format!("{what}: term {j} is aliased"))))
        .collect::<Result<Vec<_>>>()?;
    let rmse = (family == Family::Gaussian).then(|| {
        let (mut ss, mut total) = (0.0, 0.0);
        for i in 0..n {
            let e = y[i] - fit.fitted[i];
            ss += w[i] * e * e;
            total += w[i];
        }
        (ss / total).sqrt()
    });
    Ok(Linear {
        beta: beta.clone(),
        link,
        model: FittedModel {
            coefficients: beta,
            rmse,
            rows: n,
        },
        classes: 0,
    })
}

/// Fits the multinomial model of a categorical covariate with `levels`
/// classes, on the rows `row` gives as in [`fit_model`].
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the classes are checked to be whole numbers below the number of classes"
)]
fn fit_multinomial(
    prepared: &Prepared<'_>,
    terms: &[Term],
    levels: usize,
    weights: &[f64],
    what: &str,
    row: impl Fn(usize, usize, usize) -> Option<(f64, usize)>,
) -> Result<Linear> {
    let q = terms.len();
    let mut design = Vec::new();
    let mut y = Vec::new();
    let mut w = Vec::new();
    for (p, &weight) in weights.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        for (k, r) in prepared.rows_of(p).enumerate() {
            let Some((response, visible)) = row(p, r, k) else {
                continue;
            };
            let h = prepared.history(p, k, visible);
            design.extend(terms.iter().map(|t| t.eval(&h)));
            y.push(response as usize);
            w.push(weight);
        }
    }
    let n = y.len();
    if n == 0 {
        return Err(Error::input(format!("{what}: no rows to fit")));
    }
    let x = Mat::from_fn(n, q, |i, j| design[i * q + j]);
    let fit = Multinom::default()
        .fit(x.as_ref(), &y, levels, Some(&w))
        .map_err(|e| Error::input(format!("{what}: {e}")))?;
    if !fit.converged {
        return Err(Error::NoConvergence(format!("{what}: no convergence")));
    }
    Ok(Linear {
        beta: fit.coefficients.clone(),
        link: Link::Logit,
        model: FittedModel {
            coefficients: fit.coefficients,
            rmse: None,
            rows: n,
        },
        classes: levels,
    })
}
