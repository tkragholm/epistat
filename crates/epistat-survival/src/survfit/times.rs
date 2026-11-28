//! Curves read at chosen times, and the starting point of a curve.
//!
//! [`CurveIndex`] finds where chosen times fall among the times of a step
//! curve, and reads any column of the curve there. It works on slices, so the
//! curves of [`aalen_johansen`](crate::aalen_johansen) and
//! [`multistate`](crate::multistate), whose times and values are plain
//! vectors, can be read with it too: cut the vectors of one group, put the
//! starting point first, and pass the times. The rules are in the module
//! documentation of [`survfit`](crate::survfit).

use std::ops::AddAssign;

use epistat_core::{InputError, check};

use super::{GroupCurve, Survfit};
use crate::check::check_ascending;
use crate::error::{Error, Result};

/// How a read behaves where `summary.survfit` has arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadOptions {
    /// Keep chosen times after the last time of the curve, `extend` in
    /// `summary.survfit`. The default drops them.
    pub extend: bool,
    /// Sum events and censorings between chosen times, `dosum`. `None` sums
    /// when the chosen times are strictly increasing and looks up otherwise,
    /// which is `summary.survfit`'s default. `Some(true)` is an error for
    /// times that are not strictly increasing.
    pub sum: Option<bool>,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

/// Where chosen times fall among the times of a curve.
#[derive(Debug, Clone, PartialEq)]
pub struct CurveIndex {
    n: usize,
    time: Vec<f64>,
    position: Vec<usize>,
    last: Vec<usize>,
    next: Vec<usize>,
    sum: bool,
}

impl CurveIndex {
    /// Locate `times` among the ascending times `curve_time` of a curve.
    ///
    /// The values read later must have one entry per curve time, so
    /// `curve_time` includes the starting point if they do.
    ///
    /// # Errors
    ///
    /// An input error when `curve_time` is empty or not ascending, `times`
    /// is empty or holds a NaN, or `sum` is `Some(true)` and `times` is not
    /// strictly increasing.
    pub fn new(curve_time: &[f64], times: &[f64], opts: &ReadOptions) -> Result<Self> {
        check::non_empty("curve_time", curve_time.len())?;
        check_ascending(curve_time, "curve_time", true)?;
        check::non_empty("times", times.len())?;
        if let Some(row) = times.iter().position(|t| t.is_nan()) {
            return Err(InputError::NotFinite {
                what: "times",
                row,
                column: None,
            }
            .into());
        }
        let increasing = times.windows(2).all(|w| w[0] < w[1]);
        let sum = match opts.sum {
            None => increasing,
            Some(true) if !increasing => {
                return Err(Error::options(
                    "summing between times needs strictly increasing times",
                ));
            }
            Some(sum) => sum,
        };
        let end = curve_time[curve_time.len() - 1];
        let mut out = Self {
            n: curve_time.len(),
            time: Vec::new(),
            position: Vec::new(),
            last: Vec::new(),
            next: Vec::new(),
            sum,
        };
        for (i, &t) in times.iter().enumerate() {
            if !opts.extend && t > end {
                continue;
            }
            out.time.push(t);
            out.position.push(i);
            out.last.push(curve_time.partition_point(|&c| c <= t));
            out.next.push(curve_time.partition_point(|&c| c < t));
        }
        Ok(out)
    }

    /// The chosen times that were kept, in the order given.
    #[must_use]
    pub fn time(&self) -> &[f64] {
        &self.time
    }

    /// The position in the chosen times of each kept time.
    #[must_use]
    pub fn position(&self) -> &[usize] {
        &self.position
    }

    fn check<T>(&self, values: &[T]) -> Result<()> {
        Ok(check::same_length("values", self.n, values.len())?)
    }

    /// The value at each kept time: that of the last curve time at or before it,
    /// and of the first row before the first time.
    ///
    /// # Errors
    ///
    /// An input error when `values` does not have one entry per curve time.
    pub fn pick<T: Copy>(&self, values: &[T]) -> Result<Vec<T>> {
        self.check(values)?;
        Ok(self.last.iter().map(|&k| values[k.max(1) - 1]).collect())
    }

    /// The number at risk at each kept time: the value at the first curve time
    /// at or after it, and zero after the last.
    ///
    /// # Errors
    ///
    /// An input error when `values` does not have one entry per curve time.
    pub fn at_risk<T: Copy + Default>(&self, values: &[T]) -> Result<Vec<T>> {
        self.check(values)?;
        Ok(self
            .next
            .iter()
            .map(|&k| values.get(k).copied().unwrap_or_default())
            .collect())
    }

    /// The counts at each kept time: summed over the curve times since the
    /// previous kept time when the times are summed, and picked otherwise.
    ///
    /// # Errors
    ///
    /// An input error when `values` does not have one entry per curve time.
    pub fn counts<T: Copy + Default + AddAssign>(&self, values: &[T]) -> Result<Vec<T>> {
        if !self.sum {
            return self.pick(values);
        }
        self.check(values)?;
        let mut out = Vec::with_capacity(self.last.len());
        let mut done = 0;
        for &k in &self.last {
            let mut total = T::default();
            for &v in &values[done..k] {
                total += v;
            }
            done = k;
            out.push(total);
        }
        Ok(out)
    }
}

/// A curve read at chosen times, one entry per kept time in each vector.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveAt {
    /// The caller's stratum code.
    pub group: u32,
    /// The chosen times that were kept.
    pub time: Vec<f64>,
    /// The weight at risk.
    pub n_risk: Vec<f64>,
    /// The weight of events since the previous chosen time.
    pub n_event: Vec<f64>,
    /// The weight censored since the previous chosen time.
    pub n_censor: Vec<f64>,
    /// The number of rows at risk.
    pub count_risk: Vec<u32>,
    /// The number of events since the previous chosen time.
    pub count_event: Vec<u32>,
    /// The number of censorings since the previous chosen time.
    pub count_censor: Vec<u32>,
    /// The survival probability.
    pub surv: Vec<f64>,
    /// Its standard error, `None` where undefined.
    pub se: Vec<Option<f64>>,
    /// The lower confidence limit. Empty with [`ConfType::None`](super::ConfType::None).
    pub lower: Vec<Option<f64>>,
    /// The upper confidence limit. Empty with [`ConfType::None`](super::ConfType::None).
    pub upper: Vec<Option<f64>>,
    /// The cumulative hazard.
    pub cumhaz: Vec<f64>,
    /// The standard error of the cumulative hazard.
    pub se_cumhaz: Vec<f64>,
}

impl GroupCurve {
    /// The curve with its starting point, as `survfit0`.
    ///
    /// When the first time is the origin nothing is added. A curve with the
    /// starting point is returned unchanged.
    #[must_use]
    pub fn with_start(&self) -> GroupCurve {
        let mut c = self.clone();
        if c.time.first() == Some(&c.origin) || c.time.is_empty() {
            return c;
        }
        c.time.insert(0, c.origin);
        c.n_risk.insert(0, self.n_risk[0]);
        c.n_event.insert(0, 0.0);
        c.n_censor.insert(0, 0.0);
        c.count_risk.insert(0, self.count_risk[0]);
        c.count_event.insert(0, 0);
        c.count_censor.insert(0, 0);
        c.surv.insert(0, 1.0);
        c.se.insert(0, Some(0.0));
        if !c.lower.is_empty() {
            c.lower.insert(0, Some(1.0));
            c.upper.insert(0, Some(1.0));
        }
        c.cumhaz.insert(0, 0.0);
        c.se_cumhaz.insert(0, 0.0);
        c
    }

    /// The curve at chosen times, as `summary.survfit(times = )`.
    ///
    /// See the rules in the documentation of [`survfit`](crate::survfit).
    ///
    /// # Errors
    ///
    /// An input error as [`CurveIndex::new`].
    pub fn at(&self, times: &[f64], opts: &ReadOptions) -> Result<CurveAt> {
        let c = self.with_start();
        let index = CurveIndex::new(&c.time, times, opts)?;
        let limits = |v: &[Option<f64>]| -> Result<Vec<Option<f64>>> {
            if v.is_empty() {
                Ok(Vec::new())
            } else {
                index.pick(v)
            }
        };
        Ok(CurveAt {
            group: c.group,
            time: index.time().to_vec(),
            n_risk: index.at_risk(&c.n_risk)?,
            n_event: index.counts(&c.n_event)?,
            n_censor: index.counts(&c.n_censor)?,
            count_risk: index.at_risk(&c.count_risk)?,
            count_event: index.counts(&c.count_event)?,
            count_censor: index.counts(&c.count_censor)?,
            surv: index.pick(&c.surv)?,
            se: index.pick(&c.se)?,
            lower: limits(&c.lower)?,
            upper: limits(&c.upper)?,
            cumhaz: index.pick(&c.cumhaz)?,
            se_cumhaz: index.pick(&c.se_cumhaz)?,
        })
    }
}

impl Survfit {
    /// Every group's curve read at chosen times, in the order of
    /// [`groups`](Self::groups).
    ///
    /// # Errors
    ///
    /// An input error as [`CurveIndex::new`].
    pub fn at(&self, times: &[f64], opts: &ReadOptions) -> Result<Vec<CurveAt>> {
        self.groups.iter().map(|g| g.at(times, opts)).collect()
    }

    /// Every group's curve with its starting point, as `survfit0`.
    #[must_use]
    pub fn with_start(&self) -> Survfit {
        let mut s = self.clone();
        s.groups = self.groups.iter().map(GroupCurve::with_start).collect();
        s
    }
}

#[cfg(test)]
mod r_tests;
