//! One event code read as the event, for cause-specific analysis.

/// A view of the status column in which one code is the event.
///
/// For cause `k`, a row is an event when its status is `k`. A row with
/// status 0 is censored, and a row with another nonzero status is a competing
/// event, which a cause-specific analysis treats as censoring at that time.
/// Made by [`SurvData::cause`](super::SurvData::cause). The view borrows the
/// status and weight columns and copies nothing.
#[derive(Debug, Clone, Copy)]
pub struct Cause<'a> {
    pub(super) status: &'a [i32],
    pub(super) weight: Option<&'a [f64]>,
    pub(super) code: i32,
}

impl<'a> Cause<'a> {
    /// The event code, in `1..=n_codes`.
    #[must_use]
    pub fn code(&self) -> u32 {
        self.code.unsigned_abs()
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.status.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.status.is_empty()
    }

    /// Whether `row` ends in this event.
    ///
    /// # Panics
    ///
    /// When `row` is not below [`len`](Self::len), like indexing a slice.
    #[must_use]
    pub fn is_event(&self, row: usize) -> bool {
        self.status[row] == self.code
    }

    /// Whether `row` ends in another event than this one.
    ///
    /// # Panics
    ///
    /// When `row` is not below [`len`](Self::len).
    #[must_use]
    pub fn is_competing(&self, row: usize) -> bool {
        let status = self.status[row];
        status != 0 && status != self.code
    }

    /// Whether each row ends in this event, in row order.
    #[must_use = "iterators are lazy"]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + 'a {
        let code = self.code;
        self.status.iter().map(move |&status| status == code)
    }

    /// The number of events among the rows with positive weight, which is
    /// the count a fit reports. One pass over the status, and over the weights
    /// when there are any.
    #[must_use]
    pub fn n_events(&self) -> usize {
        match self.weight {
            None => self.iter().filter(|&event| event).count(),
            Some(weight) => self
                .status
                .iter()
                .zip(weight)
                .filter(|&(&status, &w)| status == self.code && w > 0.0)
                .count(),
        }
    }
}
