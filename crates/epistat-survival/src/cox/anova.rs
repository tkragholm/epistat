//! Likelihood-ratio tests between nested Cox models, as `anova.coxph` and
//! `anova.coxphlist` of `survival`.
//!
//! # Method
//!
//! For models `1, ..., m` fitted to the same rows, with maximised log partial
//! likelihoods `l_k` and ranks `r_k` (the number of coefficients that are not
//! aliased), row `k > 1` of the table tests model `k` against model `k - 1`:
//!
//! ```text
//! chisq_k = 2 (l_k - l_{k-1}),   df_k = r_k - r_{k-1},   p_k = P(chi2_{df_k} > chisq_k).
//! ```
//!
//! The first row has the log likelihood only. This is the large-sample test of
//! the model with the extra terms against the one without \[1\]. [`anova`]
//! takes the fits and tests them in the order given. [`anova_terms`] is the form
//! of `anova.coxph` for one model: it fits the design with its term groups added
//! one at a time, in order, with [`fit`], and the first row is the
//! null model, the log partial likelihood at zero coefficients.
//!
//! # What is checked
//!
//! A fit carries no copy of its data, so what [`anova`] can check from the fits
//! is that they agree in the number of rows, the number of rows used and the
//! summed weight of the events, that every one has an estimate and has
//! converged, that the ranks strictly increase, and that the log likelihood
//! does not fall by more than the iteration's own tolerance, a relative change
//! of `1e-9`, the stopping rule of `coxph`. A fall beyond that means the models
//! are not nested or one fit is not at its maximum. What cannot be checked is
//! that the rows are the same rows in the same order, that the strata, weights,
//! offset and tie rule are the same, and that the columns of one model span
//! those of the previous. `survival` checks the number of rows and events too,
//! and warns of the rest in the same way. The caller is responsible for them.
//! [`anova_terms`] fits every model itself on one data set and design, so it
//! has none of these gaps.
//!
//! `survival` takes the absolute value of the differences and so accepts the
//! models in any order. Here the models must be given from the smallest to the
//! largest, because the sign of a difference is what shows that the models are
//! not nested.
//!
//! # References
//!
//! 1. Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox
//!    Model. Springer, 2000, section 3.4.

use epistat_core::InputError;
use epistat_linalg::{Mat, MatRef};
use epistat_stats::dist::chi_square::sf as chi_square_sf;

use super::{CoxModelFit, CoxOptions, fit};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// The relative change of the log likelihood at which `coxph` stops.
const EPS: f64 = 1e-9;

/// One row of an analysis of deviance table.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AnovaRow {
    /// The maximised log partial likelihood, `None` when the model has none.
    pub log_likelihood: Option<f64>,
    /// The rank of the model, the number of coefficients that are not aliased.
    pub rank: usize,
    /// `2 (l_k - l_{k-1})`, `None` in the first row and where either log
    /// likelihood is `None`.
    pub chisq: Option<f64>,
    /// `r_k - r_{k-1}`, `None` in the first row.
    pub df: Option<usize>,
    /// The upper tail of the chi-square distribution at the statistic, `None`
    /// where the statistic is or the degrees of freedom are 0.
    pub p_value: Option<f64>,
}

/// An analysis of deviance table, the result of [`anova`] and [`anova_terms`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AnovaTable {
    /// One row per model. For [`anova_terms`] the first row is the null model
    /// and row `k` adds term `k - 1`.
    pub rows: Vec<AnovaRow>,
    /// The rows of the data.
    pub n_rows: usize,
    /// The rows with a weight above zero.
    pub n_used: usize,
    /// The summed weight of the events.
    pub n_events: f64,
}

fn tolerance(a: f64, b: f64) -> f64 {
    EPS * a.abs().max(b.abs())
}

/// The statistic, the degrees of freedom and the p-value between two models.
fn compare(smaller: (f64, usize), larger: (f64, usize)) -> Result<(f64, usize, Option<f64>)> {
    let diff = larger.0 - smaller.0;
    if diff < -tolerance(larger.0, smaller.0) {
        return Err(Error::inconsistent(
            "a model has a lower log likelihood than the smaller model it contains, so the \
             models are not nested or one fit is not at its maximum",
        ));
    }
    let chisq = 2.0 * diff.max(0.0);
    let df = larger.1 - smaller.1;
    let p = (df > 0).then(|| chi_square_sf(chisq, df as f64));
    Ok((chisq, df, p))
}

/// The likelihood-ratio tests of each model against the one before it, as
/// `anova(fit1, fit2, ...)`.
///
/// The fits must come from the smallest model to the largest. See the module
/// documentation for what is checked and what is left to the caller.
///
/// # Errors
///
/// An input error when fewer than two fits are given, when the fits differ in
/// `n_rows`, `n_used` or `n_events`, when a fit has no log likelihood or has not
/// converged, when the ranks do not strictly increase, or when a log likelihood
/// falls below that of the previous model by more than the tolerance.
#[expect(
    clippy::float_cmp,
    reason = "fits of the same rows sum the same weights in the same order, so the totals are equal bit for bit"
)]
pub fn anova(fits: &[&CoxModelFit]) -> Result<AnovaTable> {
    if fits.len() < 2 {
        return Err(Error::no_information("anova needs at least two fits"));
    }
    let first = fits[0];
    for (k, f) in fits.iter().enumerate() {
        if f.n_rows != first.n_rows || f.n_used != first.n_used || f.n_events != first.n_events {
            return Err(Error::fit(format!(
                "fit {k} has another number of rows or events than fit 0, so it was not fitted to \
                 the same data"
            )));
        }
        if f.log_partial_likelihood.is_none() {
            return Err(Error::fit(format!("fit {k} has no log likelihood")));
        }
        if !f.converged {
            return Err(Error::NotConverged {
                iterations: f.iterations,
            });
        }
    }
    let mut rows = Vec::with_capacity(fits.len());
    for (k, f) in fits.iter().enumerate() {
        let l = f.log_partial_likelihood.unwrap_or(f64::NAN);
        let mut row = AnovaRow {
            log_likelihood: Some(l),
            rank: f.rank,
            chisq: None,
            df: None,
            p_value: None,
        };
        if k > 0 {
            let before = fits[k - 1];
            if f.rank <= before.rank {
                return Err(Error::order(format!(
                    "fit {k} does not have a larger rank than fit {}: give the models from the \
                     smallest to the largest",
                    k - 1
                )));
            }
            let (chisq, df, p) = compare(
                (
                    before.log_partial_likelihood.unwrap_or(f64::NAN),
                    before.rank,
                ),
                (l, f.rank),
            )?;
            row.chisq = Some(chisq);
            row.df = Some(df);
            row.p_value = p;
        }
        rows.push(row);
    }
    Ok(AnovaTable {
        rows,
        n_rows: first.n_rows,
        n_used: first.n_used,
        n_events: first.n_events,
    })
}

/// The sequential tests of one model, as `anova(fit)`: the null model, then the
/// model with the first group of columns, with the first two, and so on.
///
/// `groups` lists the columns of each term, in the order they are added, and
/// every column of `x` that is in no group is left out. Each model is fitted
/// with [`fit`] on the data and `opts`. The log likelihood of the first row is
/// that of the null model, taken from the fit of the first group, and so
/// includes the offset of `opts`. A group whose columns are all aliased adds no
/// degrees of freedom and has no p-value.
///
/// # Errors
///
/// An input error when there are no groups, a group is empty, a column is not
/// in `x` or is in two groups, when `opts.start` is given (starting values of
/// the full model do not apply to its sub-models), when a fit has no log
/// likelihood or has not converged, and as [`fit`] does.
pub fn anova_terms(
    data: &SurvData<'_>,
    x: MatRef<'_, f64>,
    groups: &[&[usize]],
    opts: &CoxOptions,
) -> Result<AnovaTable> {
    if groups.is_empty() {
        return Err(InputError::Empty { what: "groups" }.into());
    }
    if opts.start.is_some() {
        return Err(Error::options(
            "start gives starting values for the full design and does not apply to its prefixes",
        ));
    }
    let mut seen = vec![false; x.ncols()];
    let mut columns: Vec<usize> = Vec::new();
    let mut fits: Vec<CoxModelFit> = Vec::with_capacity(groups.len());
    for (g, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(Error::no_information(format!("term {g} has no columns")));
        }
        for &c in *group {
            if c >= x.ncols() {
                return Err(Error::out_of_range(format!(
                    "term {g}: column {c} is not in x"
                )));
            }
            if std::mem::replace(&mut seen[c], true) {
                return Err(Error::inconsistent(format!(
                    "column {c} is in more than one term"
                )));
            }
            columns.push(c);
        }
        let sub = Mat::from_fn(x.nrows(), columns.len(), |i, j| x[(i, columns[j])]);
        let f = fit(data, sub.as_ref(), opts)?;
        if f.log_partial_likelihood.is_none() || !f.converged {
            return Err(Error::fit(format!(
                "the model with terms 0 to {g} has no log likelihood or did not converge"
            )));
        }
        fits.push(f);
    }
    let first = &fits[0];
    let mut rows = vec![AnovaRow {
        log_likelihood: first.null_log_partial_likelihood,
        rank: 0,
        chisq: None,
        df: None,
        p_value: None,
    }];
    let mut before = first.null_log_partial_likelihood.map(|l| (l, 0usize));
    for f in &fits {
        let l = f.log_partial_likelihood.unwrap_or(f64::NAN);
        let mut row = AnovaRow {
            log_likelihood: Some(l),
            rank: f.rank,
            chisq: None,
            df: None,
            p_value: None,
        };
        if let Some(b) = before {
            let (chisq, df, p) = compare(b, (l, f.rank))?;
            row.chisq = Some(chisq);
            row.df = Some(df);
            row.p_value = p;
        } else {
            row.df = Some(f.rank - rows.last().map_or(0, |r| r.rank));
        }
        before = Some((l, f.rank));
        rows.push(row);
    }
    Ok(AnovaTable {
        rows,
        n_rows: first.n_rows,
        n_used: first.n_used,
        n_events: first.n_events,
    })
}
