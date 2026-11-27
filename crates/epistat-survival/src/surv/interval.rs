//! Interval-censored outcomes: exact, right-, left- and interval-censored
//! times in one column pair.

use std::borrow::Cow;

use super::checks;
use super::dense::Dense;
use super::{Outcome, SurvData};
use crate::check::{check_lengths, check_row_count};
use crate::error::Result;

/// How one row of an [`IntervalSurv`] was observed.
///
/// The codes are those of the status of R's `Surv(type = "interval")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalKind {
    /// The event had not happened by the lower bound: status 0.
    Right,
    /// The event happened at exactly the lower bound, which equals the upper:
    /// status 1.
    Exact,
    /// The event had happened by the upper bound: status 2.
    Left,
    /// The event happened between the bounds, which differ: status 3.
    Interval,
}

impl IntervalKind {
    /// The status code of `Surv(type = "interval")`: 0 right, 1 exact, 2 left,
    /// 3 interval.
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            Self::Right => 0,
            Self::Exact => 1,
            Self::Left => 2,
            Self::Interval => 3,
        }
    }
}

/// Times known exactly, to lie above a bound, below a bound or between two.
///
/// Every row has a lower and an upper bound of the time of the event, `L` and
/// `U`, as R's `Surv(type = "interval2")` writes them, with an infinity where
/// a bound is missing:
///
/// | kind | bounds | meaning |
/// |---|---|---|
/// | [`Exact`](IntervalKind::Exact) | `L == U`, finite | the event is at `L` |
/// | [`Right`](IntervalKind::Right) | `L` finite, `U = +inf` | the event is after `L` |
/// | [`Left`](IntervalKind::Left) | `L = -inf`, `U` finite | the event is at or before `U` |
/// | [`Interval`](IntervalKind::Interval) | `L < U`, both finite | the event is in `(L, U]` |
///
/// The kind is a function of the two bounds and is stored beside them, so a
/// caller reads it and need not infer it. `Surv()` writes the same rows with a
/// status of 0 to 3 and a single time for the first three kinds, which
/// [`SurvData::interval_status_builder`] reads.
///
/// An outcome of this kind has no entry time, no event codes and no states.
/// It is read by the accelerated failure time models and refused by every
/// follow-up estimator, since [`SurvData::followup`] is an error for it.
#[derive(Debug, Clone)]
pub struct IntervalSurv<'a> {
    lower: Cow<'a, [f64]>,
    upper: Cow<'a, [f64]>,
    kind: Vec<IntervalKind>,
}

impl IntervalSurv<'_> {
    /// The lower bound of each row, `-inf` for a left-censored row.
    #[must_use]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// The upper bound of each row, `+inf` for a right-censored row.
    #[must_use]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// The kind of each row.
    #[must_use]
    pub fn kind(&self) -> &[IntervalKind] {
        &self.kind
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }
}

#[expect(
    clippy::float_cmp,
    reason = "an exact time is a row whose two bounds are the same number"
)]
fn kind_of(lower: f64, upper: f64) -> IntervalKind {
    if lower == upper {
        IntervalKind::Exact
    } else if upper == f64::INFINITY {
        IntervalKind::Right
    } else if lower == f64::NEG_INFINITY {
        IntervalKind::Left
    } else {
        IntervalKind::Interval
    }
}

/// Where the bounds come from.
#[derive(Debug, Clone, Copy)]
enum Source<'a> {
    Bounds {
        lower: &'a [f64],
        upper: &'a [f64],
    },
    Status {
        time: &'a [f64],
        time2: &'a [f64],
        status: &'a [i32],
    },
}

/// Collects an interval-censored outcome and the columns around it, and
/// checks them in [`build`](Self::build).
#[derive(Debug, Clone)]
#[must_use]
pub struct IntervalDataBuilder<'a> {
    source: Source<'a>,
    weight: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    stratum: Option<&'a [u32]>,
    cluster: Option<&'a [u32]>,
    id: Option<&'a [u32]>,
}

impl<'a> SurvData<'a> {
    /// Start an interval-censored outcome from its two bounds, written as in
    /// `Surv(type = "interval2")`: `-inf` for a missing lower bound and
    /// `+inf` for a missing upper one. See [`IntervalSurv`] for the kinds.
    /// Nothing is copied.
    pub fn interval_builder(lower: &'a [f64], upper: &'a [f64]) -> IntervalDataBuilder<'a> {
        IntervalDataBuilder {
            source: Source::Bounds { lower, upper },
            weight: None,
            offset: None,
            stratum: None,
            cluster: None,
            id: None,
        }
    }

    /// Start an interval-censored outcome from the columns of
    /// `Surv(time, time2, status, type = "interval")`: `status` is 0 for a
    /// right-censored time, 1 for an exact one, 2 for a left-censored one and
    /// 3 for an interval from `time` to `time2`. `time2` is read on the rows
    /// of status 3 only. The bounds are built as owned columns, 16 bytes per
    /// row.
    pub fn interval_status_builder(
        time: &'a [f64],
        time2: &'a [f64],
        status: &'a [i32],
    ) -> IntervalDataBuilder<'a> {
        IntervalDataBuilder {
            source: Source::Status {
                time,
                time2,
                status,
            },
            weight: None,
            offset: None,
            stratum: None,
            cluster: None,
            id: None,
        }
    }
}

impl<'a> IntervalDataBuilder<'a> {
    /// Case weights. Leave them out for weight 1 on every row.
    pub fn weight(mut self, weight: &'a [f64]) -> Self {
        self.weight = Some(weight);
        self
    }

    /// A term added to the linear predictor of every row, as `offset()` in an
    /// R formula. Leave it out for no offset.
    pub fn offset(mut self, offset: &'a [f64]) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Stratum codes, any `u32` values.
    pub fn stratum(mut self, stratum: &'a [u32]) -> Self {
        self.stratum = Some(stratum);
        self
    }

    /// Cluster codes, any `u32` values.
    pub fn cluster(mut self, cluster: &'a [u32]) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// Subject codes, any `u32` values.
    pub fn id(mut self, id: &'a [u32]) -> Self {
        self.id = Some(id);
        self
    }

    /// Check the columns and recode stratum, cluster and id.
    ///
    /// The rules are those of [`SurvDataBuilder::build`](super::SurvDataBuilder::build)
    /// for the columns they share, and for the bounds: no NaN, `-inf` only as
    /// a lower bound and `+inf` only as an upper one, not both infinite on one
    /// row, and the lower bound not above the upper. For
    /// [`interval_status_builder`](SurvData::interval_status_builder) the
    /// status is 0 to 3, the times are finite, and on a row of status 3 `time2`
    /// is finite and above `time`.
    ///
    /// # Errors
    ///
    /// An input error for the first rule that fails.
    pub fn build(self) -> Result<SurvData<'a>> {
        let n_rows = match self.source {
            Source::Bounds { lower, .. } => lower.len(),
            Source::Status { time, .. } => time.len(),
        };
        let mut lengths = match self.source {
            Source::Bounds { lower, upper } => vec![("lower", lower.len()), ("upper", upper.len())],
            Source::Status {
                time,
                time2,
                status,
            } => vec![
                ("time", time.len()),
                ("time2", time2.len()),
                ("status", status.len()),
            ],
        };
        let optional = [
            ("weight", self.weight.map(<[f64]>::len)),
            ("offset", self.offset.map(<[f64]>::len)),
            ("stratum", self.stratum.map(<[u32]>::len)),
            ("cluster", self.cluster.map(<[u32]>::len)),
            ("id", self.id.map(<[u32]>::len)),
        ];
        lengths.extend(
            optional
                .into_iter()
                .filter_map(|(name, len)| Some((name, len?))),
        );
        check_lengths(&lengths)?;
        check_row_count(n_rows)?;
        let (lower, upper): (Cow<'a, [f64]>, Cow<'a, [f64]>) = match self.source {
            Source::Bounds { lower, upper } => {
                checks::check_interval_bounds(lower, upper)?;
                (Cow::Borrowed(lower), Cow::Borrowed(upper))
            }
            Source::Status {
                time,
                time2,
                status,
            } => {
                checks::check_status(status, 3)?;
                checks::check_interval_times(time, time2, status)?;
                let mut lower = Vec::with_capacity(n_rows);
                let mut upper = Vec::with_capacity(n_rows);
                for row in 0..n_rows {
                    let (l, u) = match status[row] {
                        0 => (time[row], f64::INFINITY),
                        1 => (time[row], time[row]),
                        2 => (f64::NEG_INFINITY, time[row]),
                        _ => (time[row], time2[row]),
                    };
                    lower.push(l);
                    upper.push(u);
                }
                (Cow::Owned(lower), Cow::Owned(upper))
            }
        };
        if let Some(weight) = self.weight {
            checks::check_finite(weight, "weight")?;
            checks::check_weights(weight)?;
        }
        if let Some(offset) = self.offset {
            checks::check_finite(offset, "offset")?;
        }
        let kind = lower
            .iter()
            .zip(upper.iter())
            .map(|(&l, &u)| kind_of(l, u))
            .collect();
        let n_used = self
            .weight
            .map_or(n_rows, |weight| weight.iter().filter(|&&w| w > 0.0).count());
        Ok(SurvData {
            outcome: Outcome::Interval(IntervalSurv { lower, upper, kind }),
            weight: self.weight,
            offset: self.offset,
            stratum: self.stratum.map(Dense::new),
            cluster: self.cluster.map(Dense::new),
            id: self.id.map(Dense::new),
            istate: None,
            n_rows,
            n_used,
        })
    }
}
