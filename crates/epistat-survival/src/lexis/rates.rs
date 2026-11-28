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
//! `l beta -+ z se` with `z` the normal quantile of the upper probability of the level, and the
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
//! # Groups without events
//!
//! When a column of the design is 0 on every piece with events and has one
//! sign, not all 0, on the pieces without, the likelihood rises without a
//! finite maximum as its coefficient goes to minus infinity (for a column not
//! below 0) or to plus infinity (not above 0): the pieces where it is not 0
//! get a fitted rate that tends to 0, and the pieces where it is 0 are not
//! affected. The indicator of a group without events, a factor level coded as
//! a column of its own, is such a column. R's `glm` stops at a coefficient of
//! large magnitude, with a rate ratio like `1e-9` or `1e9` and the limits 0
//! and infinity, which are not estimates.
//!
//! [`fit_rates`] finds these columns and reports them in [`Rates::separated`].
//! The coefficient of such a column is `None`, and so is every contrast of
//! [`Rates::ci_lin`] and [`Rates::ci_exp`] with a nonzero weight on it, with
//! the reason in [`LinCi::unestimable`]. The other coefficients are those of
//! the limit of the likelihood as the coefficient goes to infinity: the fit of
//! the pieces on which the column is 0, which R's coefficients approach to
//! within its convergence tolerance. The pieces left out are counted in
//! [`Rates::n_separated`]. The search is repeated on the pieces that remain,
//! since leaving pieces out can give another column one sign.
//!
//! The check is exact for a column of its own. A separation that only a
//! combination of columns shows, such as a reference level without events
//! under an intercept and the indicators of the other levels, is not found,
//! and its coefficients are those `glm` gives; coding the factor with one
//! indicator per level and no intercept makes it a column of its own. Data
//! without any event leave nothing to estimate and are an error.
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

use epistat_core::{InputError, Level, check};
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
#[non_exhaustive]
pub struct Rates {
    /// The Poisson fit, over the pieces used only: the coefficients with
    /// `None` for an aliased or a separated column, and the covariance.
    pub fit: Fit,
    /// The number of pieces or cells given.
    pub n_input: usize,
    /// The number used: those with positive exposure that no separated column
    /// leaves out.
    pub n_used: usize,
    /// The number left out for having no exposure.
    pub n_zero_length: usize,
    /// The number left out because a separated column is not 0 on them.
    pub n_separated: usize,
    /// The columns that are 0 on every piece with events and of one sign on
    /// the others, in ascending order, whose coefficients have no finite
    /// estimate; see the module documentation.
    pub separated: Vec<usize>,
    /// The sum of the events of the pieces used, which is that of all
    /// pieces.
    pub events: f64,
    /// The sum of the exposure of the pieces used.
    pub person_time: f64,
}

/// Why a coefficient or a contrast of a rate model has no estimate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unestimable {
    /// The column is a combination of other columns on the pieces used, and
    /// the contrast puts weight on it.
    Aliased {
        /// The first such column with a nonzero weight.
        column: usize,
    },
    /// The column is separated: the likelihood has no finite maximum in its
    /// coefficient, and the contrast puts weight on it.
    Separated {
        /// The first such column with a nonzero weight.
        column: usize,
    },
}

/// A log-linear prediction with its Wald limits, as a row of `ci.lin`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
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
    /// Why the contrast has no estimate, `None` when it has one.
    pub unestimable: Option<Unestimable>,
}

/// The exponential of a prediction and its limits, as a row of `ci.exp`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ExpCi {
    /// `exp(l beta)`, `None` when the contrast is not estimable.
    pub estimate: Option<f64>,
    /// The lower limit.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
    /// Why the contrast has no estimate, `None` when it has one.
    pub unestimable: Option<Unestimable>,
}

/// The refusal of `value`, at row `row` of `what`, where a count or a length
/// must be finite and not negative.
fn not_a_count(what: &'static str, value: f64, row: usize) -> InputError {
    if value.is_finite() {
        InputError::Domain {
            what,
            value,
            domain: "[0, inf)",
            row: Some(row),
        }
    } else {
        InputError::NotFinite {
            what,
            row,
            column: None,
        }
    }
}

/// Fit the rate model of the events on the columns of the design, with the
/// log of `length` as offset.
///
/// `columns` are the columns of the design, each as long as `length`. A
/// column that is 0 on every piece with events and of one sign on the others
/// is reported in [`Rates::separated`] and the other coefficients are fitted
/// without the pieces on which it is not 0, see the module documentation.
///
/// # Errors
///
/// An input error when a column or `events` has the wrong length, a value is
/// not finite, a length or event count is negative, a piece with length 0 has
/// events, there is no column, no piece has exposure, no piece has an event,
/// or the fit fails, see [`epistat_glm::GlmError`].
pub fn fit_rates(
    length: &[f64],
    events: &[f64],
    columns: &[&[f64]],
    options: &RateOptions,
) -> Result<Rates> {
    let n = length.len();
    check::same_length("events", n, events.len())?;
    check::non_empty("columns", columns.len())?;
    for (j, c) in columns.iter().enumerate() {
        if c.len() != n {
            return Err(InputError::Length {
                what: "columns",
                expected: n,
                found: c.len(),
            }
            .into());
        }
        check::finite_column("columns", j, c)?;
    }
    let mut keep = Vec::with_capacity(n);
    for i in 0..n {
        let (y, d) = (length[i], events[i]);
        if !(y.is_finite() && y >= 0.0) {
            return Err(not_a_count("length", y, i).into());
        }
        if !(d.is_finite() && d >= 0.0) {
            return Err(not_a_count("events", d, i).into());
        }
        if y > 0.0 {
            keep.push(i);
        } else if d > 0.0 {
            return Err(Error::value(format!(
                "piece {i} has events and no exposure, so its rate is not defined"
            )));
        }
    }
    if keep.is_empty() {
        return Err(Error::no_information("no piece has exposure"));
    }
    let n_exposed = keep.len();
    let separated = separate(columns, events, &mut keep);
    if keep.is_empty() {
        return Err(Error::no_information(
            "no piece has an event, so no rate has an estimate",
        ));
    }
    let x = Mat::from_fn(keep.len(), columns.len(), |r, j| columns[j][keep[r]]);
    let y: Vec<f64> = keep.iter().map(|&i| events[i]).collect();
    let offset: Vec<f64> = keep.iter().map(|&i| length[i].ln()).collect();
    let mut glm = Glm::new(Family::Poisson, Link::Log);
    glm.tolerance = options.tolerance;
    glm.max_iterations = options.max_iterations;
    let fit = glm
        .fit(&crate::logistic::data(
            x.as_ref(),
            &y,
            None,
            Some(&offset),
            None,
        )?)
        .map_err(Error::model("the Poisson model"))?;
    // A separated column is 0 on every piece used, and glm aliases a column
    // of zeros, so its coefficient and covariance are None.
    Ok(Rates {
        fit,
        n_input: n,
        n_used: keep.len(),
        n_zero_length: n - n_exposed,
        n_separated: n_exposed - keep.len(),
        separated,
        events: y.iter().sum(),
        person_time: keep.iter().map(|&i| length[i]).sum(),
    })
}

/// The columns that are 0 on every piece of `keep` with events and of one
/// sign, not all 0, on the others, in ascending order. Each one found takes
/// the pieces on which it is not 0 out of `keep`, and the search repeats on
/// the pieces that remain until a pass finds none.
fn separate(columns: &[&[f64]], events: &[f64], keep: &mut Vec<usize>) -> Vec<usize> {
    let mut separated = Vec::new();
    loop {
        let mut found = false;
        for (j, column) in columns.iter().enumerate() {
            if separated.contains(&j) {
                continue;
            }
            let mut positive: Option<bool> = None;
            let one_sided = keep.iter().all(|&i| {
                let v = column[i];
                if v == 0.0 {
                    return true;
                }
                if events[i] > 0.0 {
                    return false;
                }
                *positive.get_or_insert(v > 0.0) == (v > 0.0)
            });
            if one_sided && positive.is_some() {
                separated.push(j);
                keep.retain(|&i| column[i] == 0.0);
                found = true;
            }
        }
        if !found {
            break;
        }
    }
    separated.sort_unstable();
    separated
}

impl Rates {
    /// Why coefficient `column` has no estimate: [`Unestimable::Separated`]
    /// for a column in [`Rates::separated`], [`Unestimable::Aliased`] for
    /// another column whose coefficient is `None`, and `None` when it has an
    /// estimate or there is no such column.
    #[must_use]
    pub fn unestimable(&self, column: usize) -> Option<Unestimable> {
        if self.separated.contains(&column) {
            Some(Unestimable::Separated { column })
        } else if self
            .fit
            .coefficients
            .get(column)
            .is_some_and(Option::is_none)
        {
            Some(Unestimable::Aliased { column })
        } else {
            None
        }
    }

    /// The linear predictions `l beta` of the contrast rows, with standard
    /// errors, limits at `level` and p-values, as `Epi::ci.lin`.
    ///
    /// Each contrast has one weight per coefficient. `level` mirrors the `alpha` of `ci.lin`.
    ///
    /// # Errors
    ///
    /// An input error when a contrast has the wrong length or a weight that
    /// is not finite.
    pub fn ci_lin(&self, contrasts: &[&[f64]], level: Level) -> Result<Vec<LinCi>> {
        let p = self.fit.coefficients.len();
        let z = normal::quantile(level.upper_probability());
        let mut out = Vec::with_capacity(contrasts.len());
        for (r, l) in contrasts.iter().enumerate() {
            if l.len() != p {
                return Err(Error::mismatch(format!(
                    "contrast {r} has {} weights against {p} coefficients",
                    l.len()
                )));
            }
            if let Some(j) = l.iter().position(|w| !w.is_finite()) {
                return Err(InputError::NotFinite {
                    what: "contrasts",
                    row: r,
                    column: Some(j),
                }
                .into());
            }
            let weighted = || (0..p).filter(|&j| l[j] != 0.0);
            let unestimable = weighted()
                .find(|j| self.separated.contains(j))
                .or_else(|| weighted().find(|&j| self.fit.coefficients[j].is_none()))
                .and_then(|j| self.unestimable(j));
            if unestimable.is_some() {
                out.push(LinCi {
                    estimate: None,
                    se: None,
                    lower: None,
                    upper: None,
                    p_value: None,
                    unestimable,
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
                unestimable: None,
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
    pub fn ci_exp(&self, contrasts: &[&[f64]], level: Level) -> Result<Vec<ExpCi>> {
        Ok(self
            .ci_lin(contrasts, level)?
            .into_iter()
            .map(|c| ExpCi {
                estimate: c.estimate.map(f64::exp),
                lower: c.lower.map(f64::exp),
                upper: c.upper.map(f64::exp),
                unestimable: c.unestimable,
            })
            .collect())
    }
}
