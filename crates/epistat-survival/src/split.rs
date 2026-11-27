//! Splitting follow-up at cut points: the row logic of R's
//! `survival::survSplit`.
//!
//! [`split`] cuts every follow-up interval at the given times and returns the
//! pieces as columns, with the row of the input each piece came from, so a
//! caller carries any other column along with one indexed copy. The caller
//! owns the rows of its own data frame, and nothing here knows about it.
//!
//! # What a split does
//!
//! Row `i` is the interval `(start, stop]`, or `(zero, stop]` when the data
//! has no start column. With the cut points sorted and unique, the row is cut
//! at each cut `c` with `start < c < stop`. A cut equal to the start or to the
//! stop of the row cuts nothing, a cut outside the row cuts nothing, and a row
//! with no cut inside it comes out unchanged. A row cut at `m` points gives
//! `m + 1` pieces, in time order.
//!
//! The status of the row is on its last piece. Every earlier piece ends at a
//! cut and has status 0, as a piece that is censored there.
//!
//! The episode of a piece is one more than the number of cut points at or
//! before its start. The first episode is the time before the first cut, so a
//! piece starting at a cut belongs to the episode that begins at that cut.
//! [`Split::ends_at_cut`] is true for the pieces that end at a cut, that is,
//! every piece but the last of a row. R names this variable `added`.
//!
//! # Time without a start column
//!
//! Without a start column, every row starts at [`SplitOptions::zero`]. Every
//! stop must then be larger than `zero`, as `survSplit` requires.
//!
//! # Ties
//!
//! Times and cut points that differ by rounding error are made equal by the
//! tie rule of [`crate::timefix`] before the split, with the cut points in the
//! same pool as the start and stop times, as `survSplit` does with its default
//! `timefix = TRUE`. The cut points are put in order and made unique after the
//! rule has been applied. R keeps two cuts that the rule made equal, which
//! gives a piece of length zero, and this function keeps one. A start or stop
//! that the rule moves is returned moved. [`TimeFix::Exact`] compares exactly.
//!
//! # Rows and weights
//!
//! Every row of the data is split, whatever its weight, because the output is
//! data and a row with weight 0 may be wanted later. The weights, strata and
//! clusters of the data are not read. A caller that carries them uses
//! [`Split::row`].
//!
//! # Algorithm
//!
//! The cuts are sorted once. For a row, two binary searches find the number
//! of cuts at or before the start and the number before the stop, and their
//! difference is the number of cuts inside the row. One pass sizes the output
//! exactly, a second fills it, so the memory is the output and nothing more.
//! The result does not depend on the number of threads, since nothing is
//! parallel.
//!
//! # References
//!
//! Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox
//! Model. Springer, 2000, section 6.2 on time-dependent covariates by
//! splitting follow-up. The `survSplit` help page of the `survival` package.

use crate::error::{Error, Result};
use crate::surv::{Surv, SurvData};
use crate::timefix::{TimeFix, fix_times};

/// Settings of [`split`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitOptions {
    /// The start of every row when the data has no start column, finite and
    /// below every stop. Not used when the data has a start column.
    pub zero: f64,
    /// The tie rule applied to the times and the cut points together.
    pub timefix: TimeFix,
}

impl Default for SplitOptions {
    /// `survSplit`'s defaults: `zero = 0` and the `aeqSurv` tie rule.
    fn default() -> Self {
        Self {
            zero: 0.0,
            timefix: TimeFix::default(),
        }
    }
}

/// The pieces of a split, as columns of equal length.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// The start of each piece.
    pub start: Vec<f64>,
    /// The stop of each piece.
    pub stop: Vec<f64>,
    /// The status of each piece: the status of the row on its last piece and
    /// 0 on the others.
    pub status: Vec<i32>,
    /// The row of the input each piece came from, counted from 0.
    pub row: Vec<u32>,
    /// The episode of each piece, counted from 1, see the module
    /// documentation.
    pub episode: Vec<u32>,
    /// Whether each piece ends at a cut point and not at the stop of its row.
    pub ends_at_cut: Vec<bool>,
    /// The cut points used: sorted, unique, after the tie rule.
    pub cuts: Vec<f64>,
}

impl Split {
    /// The value of `column`, one value per row of the input, at the input row
    /// of each row here.
    ///
    /// # Errors
    ///
    /// An input error when `column` has no value at some input row.
    pub fn gather<T: Copy>(&self, column: &[T]) -> Result<Vec<T>> {
        crate::expanded::gather(&self.row, column, "column")
    }

    /// The rows as follow-up data, with the largest status as the number of
    /// codes, at least 1, for a fit on them.
    ///
    /// # Errors
    ///
    /// The errors of building a [`SurvBuf`](crate::SurvBuf) from the columns.
    pub fn to_surv_buf(&self) -> Result<crate::SurvBuf> {
        crate::expanded::surv_buf(Some(&self.start), &self.stop, &self.status, None)
    }
}

impl Split {
    /// The number of pieces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stop.len()
    }

    /// Whether there are no pieces, which is the case when the input has no
    /// rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stop.is_empty()
    }
}

/// Cut the follow-up of `data` at the times in `cuts`.
///
/// The cut points may be in any order and may repeat. An empty `cuts` returns
/// the rows unchanged.
///
/// # Errors
///
/// An input error when a cut point is not finite, when the tie rule fails
/// (see [`crate::timefix::fix_times`]), when `zero` is not finite, or when the
/// data has no start column and a stop is not above `zero`.
pub fn split(data: &SurvData<'_>, cuts: &[f64], options: &SplitOptions) -> Result<Split> {
    let data = data.followup()?;
    if let Some(bad) = cuts.iter().find(|c| !c.is_finite()) {
        return Err(Error::invalid(format!(
            "cut points must be finite, found {bad}"
        )));
    }
    if !options.zero.is_finite() {
        return Err(Error::invalid("zero must be finite"));
    }
    let n = data.len();
    let mut cut = cuts.to_vec();
    cut.sort_unstable_by(f64::total_cmp);
    cut.dedup();

    let mut start: Vec<f64>;
    let stop: Vec<f64>;
    let fixed = pooled_fix(data, &cut, options.timefix)?;
    if let Some((cut_fixed, start_fixed, stop_fixed)) = fixed {
        cut = cut_fixed;
        cut.sort_unstable_by(f64::total_cmp);
        cut.dedup();
        start = start_fixed;
        stop = stop_fixed;
    } else {
        start = data.start().map(<[f64]>::to_vec).unwrap_or_default();
        stop = data.stop().to_vec();
    }
    if data.start().is_none() {
        if let Some(i) = stop.iter().position(|&s| s <= options.zero) {
            return Err(Error::invalid(format!(
                "zero must be less than every stop, but row {i} stops at {}",
                stop[i]
            )));
        }
        start = vec![options.zero; n];
    }
    let status = data.status();

    // Cuts at or before the start, cuts before the stop.
    let before = |t: f64, strict: bool| {
        if strict {
            cut.partition_point(|&c| c < t)
        } else {
            cut.partition_point(|&c| c <= t)
        }
    };
    let mut total = n;
    for i in 0..n {
        total += before(stop[i], true) - before(start[i], false);
    }
    let mut out = Split {
        start: Vec::with_capacity(total),
        stop: Vec::with_capacity(total),
        status: Vec::with_capacity(total),
        row: Vec::with_capacity(total),
        episode: Vec::with_capacity(total),
        ends_at_cut: Vec::with_capacity(total),
        cuts: Vec::new(),
    };
    for i in 0..n {
        let lo = before(start[i], false);
        let hi = before(stop[i], true);
        let row = u32::try_from(i).map_err(|_| Error::invalid("more rows than u32::MAX"))?;
        for p in lo..=hi {
            let last = p == hi;
            out.start.push(if p == lo { start[i] } else { cut[p - 1] });
            out.stop.push(if last { stop[i] } else { cut[p] });
            out.status.push(if last { status[i] } else { 0 });
            out.row.push(row);
            // `p` cuts are at or before the start of this piece.
            out.episode.push(
                u32::try_from(p + 1)
                    .map_err(|_| Error::invalid("more cut points than u32::MAX"))?,
            );
            out.ends_at_cut.push(!last);
        }
    }
    debug_assert_eq!(out.stop.len(), total);
    out.cuts = cut;
    Ok(out)
}

type Fixed = (Vec<f64>, Vec<f64>, Vec<f64>);

/// The tie rule on start, stop and cuts pooled. Returns the cuts, start and
/// stop after the rule, or `None` when nothing is tied. The start is empty for
/// data without a start column.
fn pooled_fix(data: &Surv<'_>, cuts: &[f64], policy: TimeFix) -> Result<Option<Fixed>> {
    let m = cuts.len();
    let n = data.len();
    if m == 0 || n == 0 {
        return Ok(None);
    }
    let (pool_start, mut pool_stop) = match data.start() {
        Some(start) => {
            let first = start.iter().copied().fold(f64::INFINITY, f64::min);
            let mut s = vec![first; m];
            s.extend_from_slice(start);
            (Some(s), cuts.to_vec())
        }
        None => (None, cuts.to_vec()),
    };
    pool_stop.extend_from_slice(data.stop());
    let Some(fixed) = fix_times(pool_start.as_deref(), &pool_stop, policy)? else {
        return Ok(None);
    };
    let cut = fixed.stop[..m].to_vec();
    let stop = fixed.stop[m..].to_vec();
    let start = fixed.start.map(|s| s[m..].to_vec()).unwrap_or_default();
    Ok(Some((cut, start, stop)))
}
