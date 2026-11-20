//! The rows with positive weight.

/// The rows of a [`SurvData`](super::SurvData) with positive weight, in
/// ascending order, as `u32` row indices.
///
/// Made by [`SurvData::used_rows`](super::SurvData::used_rows). It holds a
/// reference to the weights and two cursors and allocates nothing.
#[derive(Debug, Clone)]
pub struct UsedRows<'a> {
    weight: Option<&'a [f64]>,
    front: usize,
    back: usize,
    remaining: usize,
}

impl<'a> UsedRows<'a> {
    pub(super) fn new(weight: Option<&'a [f64]>, n_rows: usize, n_used: usize) -> Self {
        Self {
            weight,
            front: 0,
            back: n_rows,
            remaining: n_used,
        }
    }

    fn used(&self, row: usize) -> bool {
        self.weight.is_none_or(|weight| weight[row] > 0.0)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the row count was checked to fit in u32 when the data was built"
)]
impl Iterator for UsedRows<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        while self.front < self.back {
            let row = self.front;
            self.front += 1;
            if self.used(row) {
                self.remaining -= 1;
                return Some(row as u32);
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the row count was checked to fit in u32 when the data was built"
)]
impl DoubleEndedIterator for UsedRows<'_> {
    fn next_back(&mut self) -> Option<u32> {
        while self.front < self.back {
            self.back -= 1;
            let row = self.back;
            if self.used(row) {
                self.remaining -= 1;
                return Some(row as u32);
            }
        }
        None
    }
}

impl ExactSizeIterator for UsedRows<'_> {}
