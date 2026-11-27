//! The survival data every estimator takes: follow-up intervals with an
//! event status, and optional weights, offsets, strata, clusters, subject ids
//! and initial states.
//!
//! [`SurvData`] is the Rust counterpart of R's `Surv()` object plus the
//! columns a model formula adds around it (`strata()`, `cluster()`, `id`,
//! `offset()`, case weights). It borrows its columns, so building one copies nothing
//! unless a code column has to be recoded. [`SurvBuf`] owns the same columns
//! for callers that keep data between calls.
//!
//! # What a row means
//!
//! Row `i` says that subject `id[i]` was at risk during the interval
//! `(start[i], stop[i]]`. At `stop[i]` the status is `status[i]`: 0 means the
//! row ended without an event, and a code `k` in `1..=n_codes` means that
//! event, cause or state `k` happened. With `start` absent, every row is at
//! risk from the beginning of follow-up, with no delayed entry.
//!
//! # Mapping to R
//!
//! | `Surv()` type | `start` | `n_codes` | `id`, `istate` |
//! |---|---|---|---|
//! | `right` | absent | 1 | not used |
//! | `counting` | present | 1 | `id` optional |
//! | `mright` | absent | number of states or causes | `istate` optional |
//! | `mcounting` | present | number of states or causes | `id` and `istate` usual |
//!
//! Competing risks are `n_codes > 1` with the cause as status. R's factor
//! status for `mright` has the censoring level first and the events after it,
//! which is the same numbering. A binary status of 0 and 1 is `n_codes = 1`.
//! Left and interval censoring are the other kind of outcome,
//! [`Outcome::Interval`], built by [`SurvData::interval_builder`] and
//! [`SurvData::interval_status_builder`] and described in [`IntervalSurv`].
//! The follow-up case is read through [`SurvData::followup`], which refuses
//! interval-censored data, so an estimator of follow-up data cannot take it.
//!
//! # Checks at build time
//!
//! All checks run once, when the builder's `build` is called, so an estimator
//! and the loops around it never repeat them. Each is an error, never a
//! repair, and names the count of offending rows and the first row number
//! (from 0).
//!
//! - Columns of different lengths, since rows would not line up.
//! - More rows than `u32::MAX`, since row indices are `u32` downstream.
//! - `n_codes` of 0, which leaves no code for an event.
//! - A start, stop, weight or offset that is NaN or infinite. An infinite stop has no
//!   use in an estimator, and a missing start is expressed by leaving the
//!   column out, so neither needs an infinity. Missing values must be dropped
//!   or resolved by the caller.
//! - A negative weight. Weight 0 is allowed: the row stays in
//!   [`n_rows`](SurvData::n_rows) and not in [`n_used`](SurvData::n_used), and
//!   a fit drops it.
//! - A status below 0 or above `n_codes`.
//! - An initial state above `n_codes`. 0 means unknown.
//! - A row with `start >= stop`. Such an interval has no time at risk.
//!   Whether two stops count as the same time is a question for the tie
//!   policy and is not decided here.
//!
//! # What is not checked here
//!
//! Relations between rows are not checked: overlapping intervals of one id,
//! gaps, a state that does not continue from the previous row's status, or an
//! initial state that contradicts the row before. Those are the structural
//! checks of R's `survcheck()` and come later, with the tie policy of
//! `aeqSurv()`. Times are used as given.
//!
//! # Strata, clusters and ids
//!
//! Stratum, cluster and id take any `u32` codes. At build time each is
//! recoded to dense codes `0..k` in sorted order of the original codes, so a
//! sum over levels runs in code order whatever the caller's numbering. A
//! column that is already exactly `0..k` is used as given, detected in one
//! pass. A column that is not is recoded into an owned `Vec<u32>`, 4 bytes
//! per row, which is 40 MB per column at 10^7 rows. While recoding, a lookup
//! table of up to as many entries is held, so the peak is about 80 MB per
//! sparse column. [`SurvData::n_strata`] and its two siblings give `k`, and 0
//! when the column is absent. The recoded codes are what the accessors
//! return, so the original codes are read back through
//! [`SurvData::stratum_levels`] and its two siblings, which give the caller's
//! code of each level ([`Levels`]). A recoded column keeps one `u32` per
//! level, in sorted order, so level `k` is the `k`-th smallest code given. A
//! column that was already dense keeps no map, since a level is its own code.
//! `istate` is a state, not a grouping, and is not recoded.
//!
//! # Ties
//!
//! Times are used as given until [`SurvData::with_timefix`] applies the tie
//! policy of `survival::aeqSurv`. The follow-up times are held as
//! copy-on-write slices, so the policy replaces start and stop, and only
//! them, when it finds a tie. Status, weights and codes stay borrowed.
//! [`SurvData::timefix`] says which policy was applied, which is what makes
//! applying the same policy twice a no-op: the rule is not provably a fixed
//! point, because the mean absolute time it compares against changes when
//! values are merged, so repeating it is skipped rather than recomputed.
//! Applying `TimeFix::Exact` changes nothing and does not undo an earlier
//! policy.
//!
//! # Rows used
//!
//! Rows of weight 0 stay in the data and are dropped from a fit
//! ([`n_rows`](SurvData::n_rows) against [`n_used`](SurvData::n_used)).
//! [`SurvData::used_rows`] iterates over the rows with positive weight
//! without allocating, and [`SurvData::used_index`] collects them when a fit
//! needs a slice, which costs 4 bytes per used row (40 MB at 10^7) and is
//! `None` when every row is used. Neither is cached, so a fit that needs the
//! index more than once keeps its own.
//!
//! # One cause as the event
//!
//! [`SurvData::cause`] reads status `k` as the event and every other code as
//! censoring or, for [`Cause::is_competing`], as a competing event, without
//! copying the status column.

use std::borrow::Cow;

use crate::check::{check_lengths, check_row_count};
use crate::error::{Error, Result};
use crate::timefix::{TimeFix, fix_times};

mod cause;
mod checks;
mod dense;
mod interval;
mod levels;
mod rows;

pub use cause::Cause;
use dense::{Dense, RecodedParts};
pub use interval::{IntervalDataBuilder, IntervalKind, IntervalSurv};
pub use levels::Levels;
pub use rows::UsedRows;

/// Follow-up intervals with an event status.
///
/// Built only through [`SurvData`], which checks the columns. See the
/// [module documentation](self) for what the columns mean.
#[derive(Debug, Clone)]
pub struct Surv<'a> {
    start: Option<Cow<'a, [f64]>>,
    stop: Cow<'a, [f64]>,
    status: &'a [i32],
    n_codes: u32,
    timefix: TimeFix,
}

impl<'a> Surv<'a> {
    /// Start of each risk interval, or `None` when every row is at risk from
    /// the beginning of follow-up.
    #[must_use]
    pub fn start(&self) -> Option<&[f64]> {
        self.start.as_deref()
    }

    /// End of each risk interval, where the status applies.
    #[must_use]
    pub fn stop(&self) -> &[f64] {
        &self.stop
    }

    /// 0 for censored, `k` in `1..=n_codes` for the event, cause or state
    /// entered.
    #[must_use]
    pub fn status(&self) -> &'a [i32] {
        self.status
    }

    /// The number of event codes. 1 is a binary status.
    #[must_use]
    pub fn n_codes(&self) -> u32 {
        self.n_codes
    }

    /// The tie policy applied to the times, [`TimeFix::Exact`] until
    /// [`SurvData::with_timefix`] applies another.
    #[must_use]
    pub fn timefix(&self) -> TimeFix {
        self.timefix
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stop.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stop.is_empty()
    }

    /// Apply a tie policy, unless it changes nothing or was applied already.
    fn apply_timefix(&mut self, policy: TimeFix) -> Result<()> {
        if matches!(policy, TimeFix::Exact | TimeFix::Aeq { tolerance: 0.0 }) {
            return Ok(());
        }
        if self.timefix == policy {
            return Ok(());
        }
        if let Some(fixed) = fix_times(self.start.as_deref(), &self.stop, policy)? {
            self.stop = Cow::Owned(fixed.stop);
            self.start = fixed.start.map(Cow::Owned);
        }
        self.timefix = policy;
        Ok(())
    }
}
/// The outcome of a survival dataset.
///
/// The enum is non-exhaustive so that a further kind of outcome can be added
/// without breaking a caller's `match`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Outcome<'a> {
    /// Risk intervals with an event status, as R's `right`, `counting`,
    /// `mright` and `mcounting` types.
    Followup(Surv<'a>),
    /// Exact, right-, left- and interval-censored times, as R's `interval`
    /// and `interval2` types.
    Interval(IntervalSurv<'a>),
}

/// Survival data: an outcome and the columns that go with it.
///
/// Build one with [`SurvData::builder`]. The columns are borrowed for `'a`.
#[derive(Debug, Clone)]
pub struct SurvData<'a> {
    outcome: Outcome<'a>,
    weight: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    stratum: Option<Dense<'a>>,
    cluster: Option<Dense<'a>>,
    id: Option<Dense<'a>>,
    istate: Option<&'a [u32]>,
    n_rows: usize,
    n_used: usize,
}

impl<'a> SurvData<'a> {
    /// Start a builder from the two required columns.
    pub fn builder(stop: &'a [f64], status: &'a [i32]) -> SurvDataBuilder<'a> {
        SurvDataBuilder {
            columns: Columns {
                start: None,
                stop,
                status,
                n_codes: 1,
                weight: None,
                offset: None,
                stratum: None,
                cluster: None,
                id: None,
                istate: None,
            },
        }
    }

    /// The outcome, for code that handles every kind.
    #[must_use]
    pub fn outcome(&self) -> &Outcome<'a> {
        &self.outcome
    }

    /// The follow-up outcome.
    ///
    /// # Errors
    ///
    /// An input error when the outcome is interval-censored data, which a
    /// follow-up estimator has no use for.
    pub fn followup(&self) -> Result<&Surv<'a>> {
        match &self.outcome {
            Outcome::Followup(surv) => Ok(surv),
            Outcome::Interval(_) => Err(Error::invalid(
                "the outcome is interval-censored; this estimator needs follow-up data \
                 (stop and status)"
                    .to_owned(),
            )),
        }
    }

    /// The interval-censored outcome.
    ///
    /// # Errors
    ///
    /// An input error when the outcome is follow-up data.
    pub fn interval(&self) -> Result<&IntervalSurv<'a>> {
        match &self.outcome {
            Outcome::Interval(surv) => Ok(surv),
            Outcome::Followup(_) => Err(Error::invalid(
                "the outcome is follow-up data; this estimator needs interval-censored data"
                    .to_owned(),
            )),
        }
    }

    /// Case weights, or `None` when every row has weight 1.
    #[must_use]
    pub fn weight(&self) -> Option<&[f64]> {
        self.weight
    }

    /// A term added to the linear predictor of every row, or `None` for no
    /// offset. A fit that has a linear predictor reads it as `offset()` in an
    /// R formula: it has no coefficient.
    #[must_use]
    pub fn offset(&self) -> Option<&[f64]> {
        self.offset
    }

    /// Dense stratum codes `0..n_strata`, in sorted order of the given codes.
    #[must_use]
    pub fn stratum(&self) -> Option<&[u32]> {
        self.stratum.as_ref().map(|dense| &*dense.codes)
    }

    /// Dense cluster codes `0..n_clusters`, in sorted order of the given codes.
    #[must_use]
    pub fn cluster(&self) -> Option<&[u32]> {
        self.cluster.as_ref().map(|dense| &*dense.codes)
    }

    /// Dense subject codes `0..n_ids`, in sorted order of the given ids.
    #[must_use]
    pub fn id(&self) -> Option<&[u32]> {
        self.id.as_ref().map(|dense| &*dense.codes)
    }

    /// The state at the start of each row, `1..=n_codes`, 0 for unknown.
    #[must_use]
    pub fn istate(&self) -> Option<&[u32]> {
        self.istate
    }

    /// The caller's code of each stratum level, or `None` without a stratum
    /// column. See [`Levels`].
    #[must_use]
    pub fn stratum_levels(&self) -> Option<Levels<'_>> {
        self.stratum.as_ref().map(Dense::levels_view)
    }

    /// The caller's code of each cluster level, or `None` without a cluster
    /// column. See [`Levels`].
    #[must_use]
    pub fn cluster_levels(&self) -> Option<Levels<'_>> {
        self.cluster.as_ref().map(Dense::levels_view)
    }

    /// The caller's code of each subject level, or `None` without an id
    /// column. See [`Levels`].
    #[must_use]
    pub fn id_levels(&self) -> Option<Levels<'_>> {
        self.id.as_ref().map(Dense::levels_view)
    }

    /// The number of strata, 0 when there is no stratum column.
    #[must_use]
    pub fn n_strata(&self) -> usize {
        self.stratum.as_ref().map_or(0, |dense| dense.levels)
    }

    /// The number of clusters, 0 when there is no cluster column.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.cluster.as_ref().map_or(0, |dense| dense.levels)
    }

    /// The number of distinct ids, 0 when there is no id column.
    #[must_use]
    pub fn n_ids(&self) -> usize {
        self.id.as_ref().map_or(0, |dense| dense.levels)
    }

    /// The number of rows, including rows of weight 0.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// The number of rows with positive weight, which is all rows when there
    /// are no weights.
    #[must_use]
    pub fn n_used(&self) -> usize {
        self.n_used
    }
}

impl<'a> SurvData<'a> {
    /// Apply the tie policy to the follow-up times and return the data.
    ///
    /// The policy is that of `survival::aeqSurv`, see
    /// [`fix_times`]. Only start and stop can change. When no tie is found
    /// they stay borrowed, and when one is found they become owned, which
    /// costs 8 bytes per row for each of the two columns (80 MB each at
    /// 10^7) and a transient pooled copy for the search of up to 16 bytes
    /// per row. Status, weights, the grouping codes and their level maps are
    /// moved through untouched. The data is taken by value so that nothing
    /// is cloned. Clone first to keep the original.
    ///
    /// [`TimeFix::Exact`] and a tolerance of zero return the data unchanged,
    /// also for interval-censored data, to which no other policy applies.
    /// The policy applied is recorded, see [`timefix`](Self::timefix), and
    /// the same policy applied again returns at once without searching. A
    /// different policy is applied to the times as they now stand.
    ///
    /// The result is the same on any number of threads.
    ///
    /// # Errors
    ///
    /// An input error as for [`fix_times`]: a negative or non-finite
    /// tolerance, a sum of absolute times that overflows, or a row whose
    /// start and stop become equal. Also when the outcome is
    /// interval-censored and the policy is not [`TimeFix::Exact`]. The data
    /// is consumed whether or not the call succeeds.
    pub fn with_timefix(mut self, policy: TimeFix) -> Result<Self> {
        match &mut self.outcome {
            Outcome::Followup(surv) => surv.apply_timefix(policy)?,
            Outcome::Interval(_) => {
                if !matches!(policy, TimeFix::Exact | TimeFix::Aeq { tolerance: 0.0 }) {
                    return Err(Error::invalid(
                        "a tie policy applies to follow-up times, not to interval-censored data"
                            .to_owned(),
                    ));
                }
            }
        }
        Ok(self)
    }

    /// The same data with `offset` in place of its own, for a fit that
    /// recomputes the offset between fits.
    ///
    /// # Errors
    ///
    /// An input error when `offset` has another length than the data or a
    /// value that is not finite.
    pub(crate) fn with_offset<'b>(&self, offset: &'b [f64]) -> Result<SurvData<'b>>
    where
        'a: 'b,
    {
        check_lengths(&[("data", self.n_rows), ("offset", offset.len())])?;
        checks::check_finite(offset, "offset")?;
        let mut data: SurvData<'b> = self.clone();
        data.offset = Some(offset);
        Ok(data)
    }

    /// The tie policy applied so far, [`TimeFix::Exact`] when none.
    ///
    /// A result that depends on how ties were treated can report this, and a
    /// caller that needs exact ties can check it.
    ///
    /// # Errors
    ///
    /// As for [`followup`](Self::followup).
    pub fn timefix(&self) -> Result<TimeFix> {
        Ok(self.followup()?.timefix())
    }

    /// The rows with positive weight, ascending, which are the rows a fit
    /// uses. All rows when there are no weights.
    ///
    /// Allocates nothing and reads the weights lazily, one pass in total,
    /// so the order and the rows do not depend on threads. The iterator knows
    /// its length, [`n_used`](Self::n_used), and can be walked from the back.
    #[must_use]
    pub fn used_rows(&self) -> UsedRows<'a> {
        UsedRows::new(self.weight, self.n_rows, self.n_used)
    }

    /// The rows with positive weight as a vector, or `None` when every row
    /// is used and the index would only list `0..n_rows`.
    ///
    /// Built on each call in one pass: 4 bytes per used row, 40 MB at 10^7.
    #[must_use]
    pub fn used_index(&self) -> Option<Vec<u32>> {
        (self.n_used < self.n_rows).then(|| self.used_rows().collect())
    }

    /// Status code `code` read as the event, for a cause-specific analysis.
    ///
    /// Copies nothing and costs nothing to make.
    ///
    /// # Errors
    ///
    /// An input error when `code` is 0, which is censoring, or above
    /// `n_codes`.
    pub fn cause(&self, code: u32) -> Result<Cause<'a>> {
        let surv = self.followup()?;
        match i32::try_from(code) {
            Ok(signed) if code != 0 && code <= surv.n_codes => Ok(Cause {
                status: surv.status,
                weight: self.weight,
                code: signed,
            }),
            _ => Err(Error::invalid(format!(
                "cause {code} is outside 1..={}; 0 is censoring",
                surv.n_codes
            ))),
        }
    }
}

/// The columns as given, before any check.
#[derive(Clone, Copy)]
struct Columns<'a> {
    start: Option<&'a [f64]>,
    stop: &'a [f64],
    status: &'a [i32],
    n_codes: u32,
    weight: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    stratum: Option<&'a [u32]>,
    cluster: Option<&'a [u32]>,
    id: Option<&'a [u32]>,
    istate: Option<&'a [u32]>,
}

/// What a successful check leaves behind.
struct Checked<'a> {
    n_rows: usize,
    n_used: usize,
    stratum: Option<Dense<'a>>,
    cluster: Option<Dense<'a>>,
    id: Option<Dense<'a>>,
}

impl<'a> Columns<'a> {
    /// Run every rule once and recode the grouping columns.
    ///
    /// Cheap structural rules (lengths, row count, `n_codes`) run first, so
    /// the later rules can index every column by row.
    fn check(&self) -> Result<Checked<'a>> {
        let mut lengths = vec![("stop", self.stop.len()), ("status", self.status.len())];
        let optional = [
            ("start", self.start.map(<[f64]>::len)),
            ("weight", self.weight.map(<[f64]>::len)),
            ("offset", self.offset.map(<[f64]>::len)),
            ("stratum", self.stratum.map(<[u32]>::len)),
            ("cluster", self.cluster.map(<[u32]>::len)),
            ("id", self.id.map(<[u32]>::len)),
            ("istate", self.istate.map(<[u32]>::len)),
        ];
        lengths.extend(
            optional
                .into_iter()
                .filter_map(|(name, len)| Some((name, len?))),
        );
        check_lengths(&lengths)?;
        check_row_count(self.stop.len())?;
        checks::check_n_codes(self.n_codes)?;
        if let Some(start) = self.start {
            checks::check_finite(start, "start")?;
        }
        checks::check_finite(self.stop, "stop")?;
        if let Some(weight) = self.weight {
            checks::check_finite(weight, "weight")?;
            checks::check_weights(weight)?;
        }
        if let Some(offset) = self.offset {
            checks::check_finite(offset, "offset")?;
        }
        checks::check_status(self.status, self.n_codes)?;
        if let Some(istate) = self.istate {
            checks::check_istate(istate, self.n_codes)?;
        }
        if let Some(start) = self.start {
            checks::check_intervals(start, self.stop)?;
        }
        let n_rows = self.stop.len();
        let n_used = self
            .weight
            .map_or(n_rows, |weight| weight.iter().filter(|&&w| w > 0.0).count());
        Ok(Checked {
            n_rows,
            n_used,
            stratum: self.stratum.map(Dense::new),
            cluster: self.cluster.map(Dense::new),
            id: self.id.map(Dense::new),
        })
    }
}

/// Collects the columns of a [`SurvData`] and checks them in
/// [`build`](Self::build).
#[derive(Debug, Clone)]
#[must_use]
pub struct SurvDataBuilder<'a> {
    columns: Columns<'a>,
}

impl std::fmt::Debug for Columns<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Columns")
            .field("rows", &self.stop.len())
            .field("n_codes", &self.n_codes)
            .finish_non_exhaustive()
    }
}

impl<'a> SurvDataBuilder<'a> {
    /// The start of each risk interval. Leave it out for no delayed entry.
    pub fn start(mut self, start: &'a [f64]) -> Self {
        self.columns.start = Some(start);
        self
    }

    /// The number of event codes in status. The default is 1.
    pub fn n_codes(mut self, n_codes: u32) -> Self {
        self.columns.n_codes = n_codes;
        self
    }

    /// Case weights. Leave them out for weight 1 on every row.
    pub fn weight(mut self, weight: &'a [f64]) -> Self {
        self.columns.weight = Some(weight);
        self
    }

    /// A term added to the linear predictor of every row, as `offset()` in an
    /// R formula. Leave it out for no offset.
    pub fn offset(mut self, offset: &'a [f64]) -> Self {
        self.columns.offset = Some(offset);
        self
    }

    /// Stratum codes, any `u32` values.
    pub fn stratum(mut self, stratum: &'a [u32]) -> Self {
        self.columns.stratum = Some(stratum);
        self
    }

    /// Cluster codes, any `u32` values.
    pub fn cluster(mut self, cluster: &'a [u32]) -> Self {
        self.columns.cluster = Some(cluster);
        self
    }

    /// Subject codes, any `u32` values.
    pub fn id(mut self, id: &'a [u32]) -> Self {
        self.columns.id = Some(id);
        self
    }

    /// The state at the start of each row, `1..=n_codes`, 0 for unknown.
    pub fn istate(mut self, istate: &'a [u32]) -> Self {
        self.columns.istate = Some(istate);
        self
    }

    /// Check the columns and recode stratum, cluster and id.
    ///
    /// # Errors
    ///
    /// An input error for the first rule that fails, in the order of the
    /// list in the [module documentation](self).
    pub fn build(self) -> Result<SurvData<'a>> {
        let c = self.columns;
        let checked = c.check()?;
        Ok(SurvData {
            outcome: Outcome::Followup(Surv {
                start: c.start.map(Cow::Borrowed),
                stop: Cow::Borrowed(c.stop),
                status: c.status,
                n_codes: c.n_codes,
                timefix: TimeFix::Exact,
            }),
            weight: c.weight,
            offset: c.offset,
            stratum: checked.stratum,
            cluster: checked.cluster,
            id: checked.id,
            istate: c.istate,
            n_rows: checked.n_rows,
            n_used: checked.n_used,
        })
    }
}

/// Survival data that owns its columns.
///
/// Built with [`SurvBuf::builder`], which runs the same checks as
/// [`SurvData::builder`], so [`as_data`](Self::as_data) cannot fail and costs
/// no copy or check. The grouping columns are stored already recoded.
#[derive(Debug, Clone)]
pub struct SurvBuf {
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    status: Vec<i32>,
    n_codes: u32,
    weight: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    stratum: Option<OwnedDense>,
    cluster: Option<OwnedDense>,
    id: Option<OwnedDense>,
    istate: Option<Vec<u32>>,
    n_used: usize,
}

impl SurvBuf {
    /// Start a builder from the two required columns.
    pub fn builder(stop: Vec<f64>, status: Vec<i32>) -> SurvBufBuilder {
        SurvBufBuilder {
            start: None,
            stop,
            status,
            n_codes: 1,
            weight: None,
            offset: None,
            stratum: None,
            cluster: None,
            id: None,
            istate: None,
        }
    }

    /// A borrowed view of the data. The checks ran when the buffer was built.
    #[must_use]
    pub fn as_data(&self) -> SurvData<'_> {
        SurvData {
            outcome: Outcome::Followup(Surv {
                start: self.start.as_deref().map(Cow::Borrowed),
                stop: Cow::Borrowed(&self.stop),
                status: &self.status,
                n_codes: self.n_codes,
                timefix: TimeFix::Exact,
            }),
            weight: self.weight.as_deref(),
            offset: self.offset.as_deref(),
            stratum: borrow_dense(self.stratum.as_ref()),
            cluster: borrow_dense(self.cluster.as_ref()),
            id: borrow_dense(self.id.as_ref()),
            istate: self.istate.as_deref(),
            n_rows: self.stop.len(),
            n_used: self.n_used,
        }
    }
}

/// A grouping column kept by a [`SurvBuf`]: dense codes, the number of
/// levels and, when the column was recoded, the caller's code of each level.
#[derive(Debug, Clone)]
struct OwnedDense {
    codes: Vec<u32>,
    levels: usize,
    labels: Option<Vec<u32>>,
}

fn borrow_dense(column: Option<&OwnedDense>) -> Option<Dense<'_>> {
    column.map(|owned| Dense {
        codes: Cow::Borrowed(owned.codes.as_slice()),
        levels: owned.levels,
        labels: owned.labels.as_deref().map(Cow::Borrowed),
    })
}

/// Collects the columns of a [`SurvBuf`] and checks them in
/// [`build`](Self::build).
#[derive(Debug, Clone)]
#[must_use]
pub struct SurvBufBuilder {
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    status: Vec<i32>,
    n_codes: u32,
    weight: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    stratum: Option<Vec<u32>>,
    cluster: Option<Vec<u32>>,
    id: Option<Vec<u32>>,
    istate: Option<Vec<u32>>,
}

impl SurvBufBuilder {
    /// The start of each risk interval. Leave it out for no delayed entry.
    pub fn start(mut self, start: Vec<f64>) -> Self {
        self.start = Some(start);
        self
    }

    /// The number of event codes in status. The default is 1.
    pub fn n_codes(mut self, n_codes: u32) -> Self {
        self.n_codes = n_codes;
        self
    }

    /// Case weights. Leave them out for weight 1 on every row.
    pub fn weight(mut self, weight: Vec<f64>) -> Self {
        self.weight = Some(weight);
        self
    }

    /// A term added to the linear predictor of every row, as `offset()` in an
    /// R formula. Leave it out for no offset.
    pub fn offset(mut self, offset: Vec<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Stratum codes, any `u32` values.
    pub fn stratum(mut self, stratum: Vec<u32>) -> Self {
        self.stratum = Some(stratum);
        self
    }

    /// Cluster codes, any `u32` values.
    pub fn cluster(mut self, cluster: Vec<u32>) -> Self {
        self.cluster = Some(cluster);
        self
    }

    /// Subject codes, any `u32` values.
    pub fn id(mut self, id: Vec<u32>) -> Self {
        self.id = Some(id);
        self
    }

    /// The state at the start of each row, `1..=n_codes`, 0 for unknown.
    pub fn istate(mut self, istate: Vec<u32>) -> Self {
        self.istate = Some(istate);
        self
    }

    /// Check the columns, recode stratum, cluster and id, and keep them.
    ///
    /// A grouping column that was already dense is kept as given, without a
    /// copy.
    ///
    /// # Errors
    ///
    /// An input error for the first rule that fails, as for
    /// [`SurvDataBuilder::build`].
    pub fn build(self) -> Result<SurvBuf> {
        let columns = Columns {
            start: self.start.as_deref(),
            stop: &self.stop,
            status: &self.status,
            n_codes: self.n_codes,
            weight: self.weight.as_deref(),
            offset: self.offset.as_deref(),
            stratum: self.stratum.as_deref(),
            cluster: self.cluster.as_deref(),
            id: self.id.as_deref(),
            istate: self.istate.as_deref(),
        };
        let checked = columns.check()?;
        let n_used = checked.n_used;
        // The recoded parts own their data, so the borrow of the raw columns
        // ends here and they can be moved below.
        let recode = |dense: Option<Dense<'_>>| dense.map(Dense::into_recoded);
        let (stratum, cluster, id) = (
            recode(checked.stratum),
            recode(checked.cluster),
            recode(checked.id),
        );
        let keep = |recoded: Option<RecodedParts>, raw: Option<Vec<u32>>| {
            let (recoded, levels, labels) = recoded?;
            // A column without a recoding was dense and is kept as given.
            let codes = recoded.or(raw)?;
            Some(OwnedDense {
                codes,
                levels,
                labels,
            })
        };
        Ok(SurvBuf {
            stratum: keep(stratum, self.stratum),
            cluster: keep(cluster, self.cluster),
            id: keep(id, self.id),
            start: self.start,
            stop: self.stop,
            status: self.status,
            n_codes: self.n_codes,
            weight: self.weight,
            offset: self.offset,
            istate: self.istate,
            n_used,
        })
    }
}

#[cfg(test)]
mod tests;
