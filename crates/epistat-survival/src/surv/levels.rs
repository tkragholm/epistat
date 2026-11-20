//! The caller's codes behind the dense levels of a grouping column.

/// The original code of each dense level of a stratum, cluster or id column.
///
/// Level `k` is the `k`-th smallest of the codes the caller gave, so
/// [`code`](Self::code) is the label an estimator puts on the result it
/// computed for level `k`. A column that was already exactly `0..n` keeps no
/// map, and each level is its own code.
///
/// The view borrows from the [`SurvData`](super::SurvData) it came from and
/// costs nothing to make. The map itself holds one `u32` per level, not per
/// row, so it is 40 MB only when every row of 10^7 is its own level.
#[derive(Debug, Clone, Copy)]
pub struct Levels<'a> {
    pub(super) labels: Option<&'a [u32]>,
    pub(super) levels: usize,
}

impl<'a> Levels<'a> {
    /// The number of levels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels
    }

    /// Whether there are no levels, which happens only with no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels == 0
    }

    /// Whether each level is its own code, because the column was already
    /// `0..n`.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.labels.is_none()
    }

    /// The caller's code for a dense level, or `None` when `level` is not
    /// below [`len`](Self::len).
    #[must_use]
    pub fn code(&self, level: usize) -> Option<u32> {
        if level >= self.levels {
            return None;
        }
        match self.labels {
            Some(labels) => labels.get(level).copied(),
            None => u32::try_from(level).ok(),
        }
    }

    /// The caller's codes in level order, which is ascending.
    #[must_use = "iterators are lazy"]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a level of a column without a map is below the number of distinct u32 codes"
    )]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = u32> + DoubleEndedIterator + 'a {
        let labels = self.labels;
        (0..self.levels).map(move |level| match labels {
            Some(labels) => labels[level],
            None => level as u32,
        })
    }

    /// The caller's codes as a vector, in level order.
    #[must_use]
    pub fn to_vec(&self) -> Vec<u32> {
        self.iter().collect()
    }
}
