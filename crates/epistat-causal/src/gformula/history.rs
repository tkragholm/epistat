//! A person's history and the terms of the models, evaluated on it.

use crate::error::{Error, Result};

/// One term of a model's linear predictor, evaluated on a person's history at
/// a period `k`.
///
/// Time-varying variables are numbered by their position in
/// [`PersonPeriods::time_varying`](super::PersonPeriods::time_varying), which
/// is also the order in which they are simulated within a period; baseline
/// variables by their position in
/// [`PersonPeriods::baseline`](super::PersonPeriods::baseline). Lags reach back
/// before period 0 as [`GFormulaOptions::baseline_lags`](super::GFormulaOptions::baseline_lags)
/// says: 0 by default, the value at period 0 with the option.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// The constant 1.
    Intercept,
    /// The period `k` as a number.
    Period,
    /// 1 at period `j`, 0 at other periods, for time as a categorical variable.
    PeriodIs(usize),
    /// A baseline variable.
    Baseline(usize),
    /// A time-varying variable at period `k`.
    Current(usize),
    /// A time-varying variable at period `k - lag`, `lag >= 1` (gfoRmula's
    /// `lag<lag>_<name>`).
    Lag(usize, usize),
    /// The mean of a time-varying variable over periods `0..=k` (gfoRmula's
    /// `cumavg_<name>`).
    CumulativeAverage(usize),
    /// The mean over periods `0..=k - lag`, `lag >= 1` (gfoRmula's
    /// `lag_cumavg<lag>_<name>`).
    LaggedCumulativeAverage(usize, usize),
    /// The product of two terms, for interactions and powers.
    Product(Box<Term>, Box<Term>),
}

impl Term {
    /// The product of `a` and `b`.
    #[must_use]
    pub fn product(a: Self, b: Self) -> Self {
        Self::Product(Box::new(a), Box::new(b))
    }

    pub(crate) fn eval(&self, h: &History<'_>) -> f64 {
        match self {
            Self::Intercept => 1.0,
            Self::Period => h.period as f64,
            Self::PeriodIs(j) => f64::from(u8::from(h.period == *j)),
            Self::Baseline(b) => h.baseline(*b),
            Self::Current(v) => h.current(*v),
            Self::Lag(v, lag) => h.lag(*v, *lag),
            Self::CumulativeAverage(v) => h.cumulative_average(*v),
            Self::LaggedCumulativeAverage(v, lag) => h.lagged_cumulative_average(*v, *lag),
            Self::Product(a, b) => a.eval(h) * b.eval(h),
        }
    }

    /// Checks that the term names existing variables, that its lags are at
    /// least 1, and that it reads the current period only of variables before
    /// `current` in the simulation order.
    fn check(&self, baseline: usize, width: usize, current: usize, what: &str) -> Result<()> {
        let bad = |message: String| Err(Error::input(format!("{what}: {message}")));
        match self {
            Self::Baseline(b) if *b >= baseline => {
                bad(format!("baseline variable {b} does not exist"))
            }
            Self::Intercept | Self::Period | Self::PeriodIs(_) | Self::Baseline(_) => Ok(()),
            Self::Current(v) | Self::CumulativeAverage(v) => {
                if *v >= width {
                    bad(format!("time-varying variable {v} does not exist"))
                } else if *v >= current {
                    bad(format!(
                        "reads time-varying variable {v} at the current period, \
                         which is simulated after it"
                    ))
                } else {
                    Ok(())
                }
            }
            Self::Lag(v, lag) | Self::LaggedCumulativeAverage(v, lag) => {
                if *v >= width {
                    bad(format!("time-varying variable {v} does not exist"))
                } else if *lag == 0 {
                    bad("a lag must be at least 1".into())
                } else {
                    Ok(())
                }
            }
            Self::Product(a, b) => {
                a.check(baseline, width, current, what)?;
                b.check(baseline, width, current, what)
            }
        }
    }
}

/// Checks every term of a model; see [`Term`].
pub(crate) fn check_terms(
    terms: &[Term],
    baseline: usize,
    width: usize,
    current: usize,
    what: &str,
) -> Result<()> {
    if terms.is_empty() {
        return Err(Error::input(format!("{what}: the model has no terms")));
    }
    terms
        .iter()
        .try_for_each(|t| t.check(baseline, width, current, what))
}

/// A read-only view of one person's history at one period, observed or
/// simulated.
///
/// At the current period only the variables before the one being simulated
/// are visible; reading another gives NaN. A restriction or a dynamic
/// intervention of variable `v` therefore sees the current values of variables
/// `0..v` and every earlier period. The same view is used to fit the models to
/// the observed data and to simulate, so a condition reads the same values in
/// both.
#[derive(Debug, Clone, Copy)]
pub struct History<'a> {
    /// `periods x width` values, row by row.
    pub(crate) values: &'a [f64],
    /// Their running sums over periods.
    pub(crate) sums: &'a [f64],
    pub(crate) baseline: &'a [f64],
    pub(crate) width: usize,
    pub(crate) period: usize,
    pub(crate) visible: usize,
    pub(crate) baseline_lags: bool,
}

impl History<'_> {
    /// The current period.
    #[must_use]
    pub fn period(&self) -> usize {
        self.period
    }

    /// Baseline variable `b`.
    ///
    /// # Panics
    ///
    /// If `b` is not a baseline variable.
    #[must_use]
    pub fn baseline(&self, b: usize) -> f64 {
        self.baseline[b]
    }

    /// Time-varying variable `v` at period `period`, NaN for a later period
    /// or for a variable not yet simulated at the current one.
    ///
    /// # Panics
    ///
    /// If `v` is not a time-varying variable.
    #[must_use]
    pub fn value(&self, v: usize, period: usize) -> f64 {
        assert!(v < self.width, "time-varying variable {v} does not exist");
        if period > self.period || (period == self.period && v >= self.visible) {
            f64::NAN
        } else {
            self.values[period * self.width + v]
        }
    }

    /// Time-varying variable `v` at the current period, as [`History::value`].
    #[must_use]
    pub fn current(&self, v: usize) -> f64 {
        self.value(v, self.period)
    }

    /// Time-varying variable `v` at period `k - lag`, or before period 0 the
    /// convention of [`Term`].
    #[must_use]
    pub fn lag(&self, v: usize, lag: usize) -> f64 {
        match self.period.checked_sub(lag) {
            Some(k) => self.value(v, k),
            None if self.baseline_lags => self.value(v, 0),
            None => 0.0,
        }
    }

    /// The mean of time-varying variable `v` over periods `0..=k`.
    #[must_use]
    pub fn cumulative_average(&self, v: usize) -> f64 {
        self.average_through(v, self.period)
    }

    /// The mean of time-varying variable `v` over periods `0..=k - lag`, or
    /// before period 0 the convention of [`Term`].
    #[must_use]
    pub fn lagged_cumulative_average(&self, v: usize, lag: usize) -> f64 {
        match self.period.checked_sub(lag) {
            Some(k) => self.average_through(v, k),
            None if self.baseline_lags => self.value(v, 0),
            None => 0.0,
        }
    }

    fn average_through(&self, v: usize, k: usize) -> f64 {
        if self.value(v, k).is_nan() {
            return f64::NAN;
        }
        self.sums[k * self.width + v] / (k + 1) as f64
    }
}
