//! Poisson models for rates on split follow-up.
//!
//! The pieces of a Lexis split, or the cells they are summed into, have an
//! exposure `y` (person-time) and a count of events `d`. The rate model is
//! the Poisson regression
//!
//! ```text
//! d ~ Poisson(mu),   ln mu = ln y + x beta,
//! ```
//!
//! with the log of the person-time as offset, which is
//! `glm(d ~ x, family = poisson, offset = log(y))` in R. The design `x` is
//! built by the caller, one row per piece or cell and one column per
//! parameter, intercept included if wanted: R builds the formula, and this
//! takes its columns. The fit is [`epistat_glm::Glm`] with the Poisson family
//! and the log link, so its iterations are those of `glm.fit`.
//!
//! # Rates and limits
//!
//! The rate of a log-linear prediction with contrast row `l` is
//! `exp(l beta)`. [`Rates::ci_lin`] returns `l beta`, its standard error
//! `sqrt(l V l')` with `V` the covariance of the fit, the limits
//! `l beta -+ z se` with `z` the normal quantile of `1 - alpha / 2`, and the
//! two-sided Wald p-value of `l beta = 0`, as `Epi::ci.lin`. [`Rates::ci_exp`]
//! returns the exponentials of the estimate and the limits, as `Epi::ci.exp`:
//! rates, or rate ratios for contrasts of differences, with the limits of
//! the Wald interval on the log scale. The unit of a rate is that of the
//! person-time. A contrast with a nonzero weight on an aliased coefficient is
//! not estimable and gives `None`, where R drops the coefficient silently.
//!
//! # Pieces with no exposure
//!
//! A piece of length 0, which the split makes of a row of duration 0, has
//! `ln y = -inf` and contributes nothing to the likelihood when it has no
//! event. Such pieces are left out and counted in [`Rates::n_zero_length`].
//! R would fail on them. A piece of length 0 with an event is an error: the
//! rate is not defined.
//!
//! # Cells and pieces
//!
//! The pieces with the same design row have the likelihood of one cell with
//! the sums of their `y` and `d`, see [`crate::lexis::cells`], so the
//! estimates and the covariance are those of the pieces. The deviance of the
//! fit differs between the two by the saturated terms, which are not the same
//! sum, and the iterations may differ, since the starting means `d + 0.1`
//! are not sums.
//!
//! # References
//!
//! 1. Carstensen B, Plummer M, Laara E, Hills M. Epi: A Package for
//!    Statistical Analysis in Epidemiology. R package, help pages of `ci.lin`
//!    and `ci.exp`.
//! 2. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, chapter 4, Poisson regression for rates.
//! 3. Clayton D, Hills M. Statistical Models in Epidemiology. Oxford 1993,
//!    chapters 22 to 24.

#![expect(
    deprecated,
    reason = "the fits take their data as separate arguments, the epistat-glm 0.2 form"
)]

use epistat_glm::{Family, Fit, Glm, Link};
use epistat_linalg::Mat;
use epistat_stats::dist::normal;

use crate::error::{Error, Result};

/// The fitting controls of [`fit_rates`], as `glm.control`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateOptions {
    /// R's `epsilon`, the relative change of the deviance that stops the
    /// iterations. Positive and finite.
    pub tolerance: f64,
    /// R's `maxit`.
    pub max_iterations: usize,
}

impl Default for RateOptions {
    /// `glm.control`'s defaults: `1e-8` and 25 iterations.
    fn default() -> Self {
        let glm = Glm::new(Family::Poisson, Link::Log);
        Self {
            tolerance: glm.tolerance,
            max_iterations: glm.max_iterations,
        }
    }
}

/// A fitted rate model.
#[derive(Debug, Clone)]
pub struct Rates {
    /// The Poisson fit, over the pieces with exposure only: the coefficients
    /// with `None` for an aliased column, and the covariance.
    pub fit: Fit,
    /// The number of pieces or cells given.
    pub n_input: usize,
    /// The number used: those with positive exposure.
    pub n_used: usize,
    /// The number left out for having no exposure.
    pub n_zero_length: usize,
    /// The sum of the events of the pieces used.
    pub events: f64,
    /// The sum of the exposure of the pieces used.
    pub person_time: f64,
}

/// A log-linear prediction with its Wald limits, as a row of `ci.lin`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinCi {
    /// `l beta`, `None` when the contrast is not estimable.
    pub estimate: Option<f64>,
    /// Its standard error.
    pub se: Option<f64>,
    /// The lower limit.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
    /// The two-sided Wald p-value of `l beta = 0`.
    pub p_value: Option<f64>,
}

/// The exponential of a prediction and its limits, as a row of `ci.exp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpCi {
    /// `exp(l beta)`, `None` when the contrast is not estimable.
    pub estimate: Option<f64>,
    /// The lower limit.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
}

/// Fit the rate model of the events on the columns of the design, with the
/// log of `length` as offset.
///
/// `columns` are the columns of the design, each as long as `length`.
///
/// # Errors
///
/// [`Error::Input`] when a column or `events` has the wrong length, a value is
/// not finite, a length or event count is negative, a piece with length 0 has
/// events, there is no column, no piece has exposure, or the fit fails, see
/// [`epistat_glm::GlmError`].
pub fn fit_rates(
    length: &[f64],
    events: &[f64],
    columns: &[&[f64]],
    options: &RateOptions,
) -> Result<Rates> {
    let n = length.len();
    if events.len() != n {
        return Err(Error::input(format!(
            "events has {} values against {n} lengths",
            events.len()
        )));
    }
    if columns.is_empty() {
        return Err(Error::input("the design has no columns"));
    }
    for (j, c) in columns.iter().enumerate() {
        if c.len() != n {
            return Err(Error::input(format!(
                "column {j} has {} values against {n} pieces",
                c.len()
            )));
        }
        if let Some(i) = c.iter().position(|v| !v.is_finite()) {
            return Err(Error::input(format!(
                "column {j}: the value of piece {i} is not finite"
            )));
        }
    }
    let mut keep = Vec::with_capacity(n);
    for i in 0..n {
        let (y, d) = (length[i], events[i]);
        if !(y.is_finite() && y >= 0.0) {
            return Err(Error::input(format!(
                "the length of piece {i} is not finite and non-negative"
            )));
        }
        if !(d.is_finite() && d >= 0.0) {
            return Err(Error::input(format!(
                "the events of piece {i} are not finite and non-negative"
            )));
        }
        if y > 0.0 {
            keep.push(i);
        } else if d > 0.0 {
            return Err(Error::input(format!(
                "piece {i} has events and no exposure, so its rate is not defined"
            )));
        }
    }
    if keep.is_empty() {
        return Err(Error::input("no piece has exposure"));
    }
    let x = Mat::from_fn(keep.len(), columns.len(), |r, j| columns[j][keep[r]]);
    let y: Vec<f64> = keep.iter().map(|&i| events[i]).collect();
    let offset: Vec<f64> = keep.iter().map(|&i| length[i].ln()).collect();
    let mut glm = Glm::new(Family::Poisson, Link::Log);
    glm.tolerance = options.tolerance;
    glm.max_iterations = options.max_iterations;
    let fit = glm
        .fit_slices(x.as_ref(), &y, None, Some(&offset), None)
        .map_err(|e| Error::input(format!("the Poisson fit failed: {e}")))?;
    Ok(Rates {
        fit,
        n_input: n,
        n_used: keep.len(),
        n_zero_length: n - keep.len(),
        events: y.iter().sum(),
        person_time: keep.iter().map(|&i| length[i]).sum(),
    })
}

impl Rates {
    /// The linear predictions `l beta` of the contrast rows, with standard
    /// errors, `1 - alpha` limits and p-values, as `Epi::ci.lin`.
    ///
    /// Each contrast has one weight per coefficient. `alpha` is R's usual
    /// 0.05 in `ci.lin`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when a contrast has the wrong length or a weight that
    /// is not finite, or `alpha` is not in `(0, 1)`.
    pub fn ci_lin(&self, contrasts: &[&[f64]], alpha: f64) -> Result<Vec<LinCi>> {
        let p = self.fit.coefficients.len();
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(Error::input("alpha must be in (0, 1)"));
        }
        let z = normal::quantile(1.0 - alpha / 2.0);
        let mut out = Vec::with_capacity(contrasts.len());
        for (r, l) in contrasts.iter().enumerate() {
            if l.len() != p {
                return Err(Error::input(format!(
                    "contrast {r} has {} weights against {p} coefficients",
                    l.len()
                )));
            }
            if l.iter().any(|w| !w.is_finite()) {
                return Err(Error::input(format!(
                    "contrast {r} has a weight that is not finite"
                )));
            }
            let estimable = (0..p).all(|j| l[j] == 0.0 || self.fit.coefficients[j].is_some());
            if !estimable {
                out.push(LinCi {
                    estimate: None,
                    se: None,
                    lower: None,
                    upper: None,
                    p_value: None,
                });
                continue;
            }
            let mut estimate = 0.0;
            for j in 0..p {
                if l[j] != 0.0 {
                    estimate += l[j] * self.fit.coefficients[j].unwrap_or(0.0);
                }
            }
            let mut variance = Some(0.0);
            'sum: for a in 0..p {
                if l[a] == 0.0 {
                    continue;
                }
                for b in 0..p {
                    if l[b] == 0.0 {
                        continue;
                    }
                    if let (Some(v), Some(acc)) = (self.fit.covariance.get(a, b), variance) {
                        variance = Some(acc + l[a] * l[b] * v);
                    } else {
                        variance = None;
                        break 'sum;
                    }
                }
            }
            let se = variance.map(|v| v.max(0.0).sqrt());
            out.push(LinCi {
                estimate: Some(estimate),
                se,
                lower: se.map(|s| estimate - z * s),
                upper: se.map(|s| estimate + z * s),
                p_value: se
                    .and_then(|s| (s > 0.0).then(|| 2.0 * normal::cdf(-(estimate / s).abs()))),
            });
        }
        Ok(out)
    }

    /// The exponentials of [`ci_lin`](Self::ci_lin): rates, or rate ratios,
    /// with the limits of the Wald interval on the log scale, as
    /// `Epi::ci.exp`.
    ///
    /// # Errors
    ///
    /// As [`ci_lin`](Self::ci_lin).
    pub fn ci_exp(&self, contrasts: &[&[f64]], alpha: f64) -> Result<Vec<ExpCi>> {
        Ok(self
            .ci_lin(contrasts, alpha)?
            .into_iter()
            .map(|c| ExpCi {
                estimate: c.estimate.map(f64::exp),
                lower: c.lower.map(f64::exp),
                upper: c.upper.map(f64::exp),
            })
            .collect())
    }
}
