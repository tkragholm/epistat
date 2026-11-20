//! Observed against predicted risk in groups of the predictions, as the
//! quantile groups of `riskRegression::plotCalibration`.
//!
//! [`estimate`] takes the predicted risk of the event by one time `t` for every
//! row, sorts the rows into groups by the predicted risk, and for each group
//! returns the mean predicted risk and the observed risk by `t`: one minus the
//! Kaplan-Meier estimate in the group for single-event data, and the
//! Aalen-Johansen cumulative incidence of the cause when there is a competing
//! event. A well-calibrated prediction has the two equal in every group. The
//! observed risk of a group is the estimate of that group's own follow-up, so
//! censoring is allowed for by the estimator within the group and no weights
//! are needed. It is the same as weighting by the inverse probability of
//! censoring when the censoring distribution is estimated separately in each
//! group.
//!
//! # Groups
//!
//! [`Groups::Quantiles`] with `q` groups cuts at the type 7 quantiles of the
//! predicted risks of the used rows at `0, 1/q, ..., 1`, as `quantile(p, seq(0,
//! 1, 1/q))` does, drops repeated cut points, and puts a row with risk `r` in
//! the group `(b_{k-1}, b_k]`, the first group also containing `b_0`, as
//! `cut(include.lowest = TRUE)` does. [`Groups::Breaks`] takes the cut points
//! from the caller. When the predictions take no more distinct values than
//! there are cut points, each distinct value is a group of its own.
//!
//! # What is left out
//!
//! The smoothed calibration curve by nearest neighbours (`method = "nne"`), the
//! calibration by pseudo-values and by a regression of pseudo-values, the
//! confidence limits of the observed risk and the calibration slope and
//! intercept are not computed here.
//!
//! # Differences from `riskRegression`
//!
//! - `plotCalibration` with few distinct predictions builds its groups from
//!   `factor(sort(p))`, which orders the predictions separately from the rows, so
//!   the observed risks are those of the wrong rows. The groups here are
//!   assigned to the rows they belong to. The R script of the test shows the
//!   misassignment.
//! - A group with no row is left out. `plotCalibration` returns a row of
//!   missing values.
//! - The predicted risks must lie in `[0, 1]`.
//! - `prodlim` returns a missing value for the observed risk of a group when
//!   the time is after the last time in the group, and so does this function,
//!   as `None`.
//!
//! # Errors of the computed numbers
//!
//! The mean predicted risk is a sum of at most `n` terms in `[0, 1]`, so it has
//! the relative error `n eps` in each implementation. The observed risk is a
//! product-limit estimate of at most `n` factors, each with the relative error
//! of a division of two counts, so its absolute error is at most
//! `(n + 2) eps` in each. Quantile cut points interpolate between two risks
//! with an error of a few `eps`, and a row exactly on a cut point is therefore
//! the one place where two implementations may disagree on the group: the test
//! data has none.
//!
//! # References
//!
//! 1. Gerds TA, Kattan MW. Medical Risk Prediction Models: With Ties to
//!    Machine Learning. Chapman and Hall/CRC, 2021. Chapter on calibration.
//! 2. Ozenne B, Sorensen AL, Scheike T, Torp-Pedersen C, Gerds TA.
//!    riskRegression: predicting the risk of an event using Cox regression
//!    models. R J 2017;9(2):440-60. doi:10.32614/RJ-2017-062
//! 3. Austin PC, Harrell FE, van Klaveren D. Graphical calibration curves and
//!    the integrated calibration index (ICI) for survival models. Stat Med
//!    2020;39(21):2714-42. doi:10.1002/sim.8570

use super::marginal::marginal_risk;
use super::prepare::sort_rows;
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// How the rows are divided into groups.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Groups<'a> {
    /// This many groups of about equal size, from the quantiles of the
    /// predictions. At least 1.
    Quantiles(usize),
    /// The cut points, which must cover every prediction.
    Breaks(&'a [f64]),
}

/// What [`estimate`] takes besides the data, the predictions and the time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationOptions<'a> {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. `Exact`: `plotCalibration` applies none.
    pub timefix: TimeFix,
    /// The event of interest. 1, the first cause as in `Score`.
    pub cause: u32,
    /// How the groups are formed. Without a default: the number of groups has
    /// no principled value.
    pub groups: Groups<'a>,
}

impl<'a> CalibrationOptions<'a> {
    /// The options with the given groups and the defaults of the others.
    #[must_use]
    pub fn new(groups: Groups<'a>) -> Self {
        Self {
            timefix: TimeFix::Exact,
            cause: 1,
            groups,
        }
    }
}

/// One group of the predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskGroup {
    /// The lower cut point: the group holds risks above it, or at it for the
    /// first group. For a group of one distinct value, that value.
    pub lower: f64,
    /// The upper cut point, included. For a group of one distinct value, that
    /// value.
    pub upper: f64,
    /// The number of rows.
    pub n: usize,
    /// The mean predicted risk.
    pub predicted: f64,
    /// The observed risk by the time, `None` when the time is after the last
    /// time in the group.
    pub observed: Option<f64>,
}

/// The result of [`estimate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// The time the risks are for.
    pub time: f64,
    /// The cut points, ascending. Empty when each distinct prediction is a
    /// group.
    pub breaks: Vec<f64>,
    /// The groups with at least one row, by ascending risk.
    pub groups: Vec<RiskGroup>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// Predicted and observed risk by `time` in groups of `risk`.
///
/// `risk` has one value per row of the data: the predicted probability of the
/// event of interest by `time`.
///
/// # Errors
///
/// [`Error::Input`] when the data is interval-censored, has a start column or
/// a weight other than 0 and 1, has no row of positive weight, or `cause` is
/// outside the event codes; when `time` is not finite; when `risk` has another
/// length than the data or a value outside `[0, 1]`; when the number of
/// groups is 0, or the breaks are not finite or do not cover every prediction.
pub fn estimate(
    data: &SurvData<'_>,
    risk: &[f64],
    time: f64,
    opts: &CalibrationOptions<'_>,
) -> Result<Calibration> {
    let sorted = sort_rows(data, opts.timefix, opts.cause)?;
    if !time.is_finite() {
        return Err(Error::input("time is not finite"));
    }
    if risk.len() != sorted.n_rows {
        return Err(Error::input(format!(
            "risk has {} values for {} rows",
            risk.len(),
            sorted.n_rows
        )));
    }
    if let Some(position) = risk.iter().position(|p| !(0.0..=1.0).contains(p)) {
        return Err(Error::input(format!(
            "risk: row {position} is {}, which is not a probability",
            risk[position]
        )));
    }
    let n = sorted.n();
    let p: Vec<f64> = sorted.row.iter().map(|&r| risk[r as usize]).collect();

    let mut values = p.clone();
    values.sort_by(f64::total_cmp);
    let mut distinct = values.clone();
    distinct.dedup();

    let mut breaks: Vec<f64> = match opts.groups {
        Groups::Quantiles(q) => {
            if q == 0 {
                return Err(Error::input("the number of groups must be at least 1"));
            }
            (0..=q)
                .map(|k| epistat_stats::quantile::type7_ratio(&values, k, q))
                .collect::<Option<Vec<f64>>>()
                .ok_or_else(|| Error::input("no predicted risk to cut into groups"))?
        }
        Groups::Breaks(b) => {
            if let Some(position) = b.iter().position(|v| !v.is_finite()) {
                return Err(Error::input(format!(
                    "breaks: value {position} is not finite"
                )));
            }
            b.to_vec()
        }
    };
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    if let Groups::Breaks(_) = opts.groups {
        if breaks.first().is_none_or(|&b| b > values[0])
            || breaks.last().is_none_or(|&b| b < values[n - 1])
        {
            return Err(Error::input(
                "the breaks must cover every predicted risk, from the smallest to the largest",
            ));
        }
    }

    let by_value = distinct.len() <= breaks.len();
    let n_groups = if by_value {
        distinct.len()
    } else {
        breaks.len() - 1
    };
    let group_of = |value: f64| -> usize {
        if by_value {
            distinct.partition_point(|&d| d < value)
        } else {
            // The first group holds its lower cut point; the others are open
            // below and closed above.
            let above = breaks.partition_point(|&b| b < value);
            above.saturating_sub(1)
        }
    };
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (pos, &value) in p.iter().enumerate() {
        members[group_of(value)].push(pos);
    }
    let mut groups = Vec::new();
    for (g, rows) in members.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let predicted = rows.iter().map(|&pos| p[pos]).sum::<f64>() / rows.len() as f64;
        let time_g: Vec<f64> = rows.iter().map(|&pos| sorted.time[pos]).collect();
        let kind_g: Vec<_> = rows.iter().map(|&pos| sorted.kind[pos]).collect();
        let observed =
            (time <= time_g[time_g.len() - 1]).then(|| marginal_risk(&time_g, &kind_g, &[time])[0]);
        let (lower, upper) = if by_value {
            (distinct[g], distinct[g])
        } else {
            (breaks[g], breaks[g + 1])
        };
        groups.push(RiskGroup {
            lower,
            upper,
            n: rows.len(),
            predicted,
            observed,
        });
    }
    Ok(Calibration {
        time,
        breaks: if by_value { Vec::new() } else { breaks },
        groups,
        n_rows: sorted.n_rows,
        n_used: n,
    })
}
