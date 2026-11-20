//! Dense recoding of stratum, cluster and id codes.

use std::borrow::Cow;

/// Codes `0..levels`, each level present, in the order of the caller's codes.
///
/// `labels` holds the caller's code of each level, ascending, and is absent
/// when the codes were already `0..levels`, where a level is its own code.
#[derive(Debug, Clone)]
pub(super) struct Dense<'a> {
    pub(super) codes: Cow<'a, [u32]>,
    pub(super) levels: usize,
    pub(super) labels: Option<Cow<'a, [u32]>>,
}

impl<'a> Dense<'a> {
    /// Recode `codes` to `0..k` in sorted code order, where `k` is the number of
    /// distinct codes.
    ///
    /// Codes that are already exactly `0..k` are borrowed without a copy. The
    /// test is one pass over a table of `max + 1` entries, which is used only
    /// when `max` is below the row count. Larger codes are sparse by necessity
    /// and go through a sorted copy of the codes and a binary search.
    ///
    /// The number of rows must not exceed `u32::MAX`.
    pub(super) fn new(codes: &'a [u32]) -> Self {
        let Some(&max) = codes.iter().max() else {
            return Self {
                codes: Cow::Borrowed(codes),
                levels: 0,
                labels: None,
            };
        };
        let width = max as usize + 1;
        if width <= codes.len() {
            Self::by_table(codes, width)
        } else {
            Self::by_sorting(codes)
        }
    }

    fn by_table(codes: &'a [u32], width: usize) -> Self {
        let mut rank = vec![0_u32; width];
        for &code in codes {
            rank[code as usize] = 1;
        }
        let levels = rank.iter().filter(|&&slot| slot == 1).count();
        if levels == width {
            return Self {
                codes: Cow::Borrowed(codes),
                levels,
                labels: None,
            };
        }
        let mut labels = Vec::with_capacity(levels);
        for (code, slot) in rank.iter_mut().enumerate() {
            if *slot == 1 {
                *slot = u32::try_from(labels.len()).expect("at most u32::MAX distinct codes");
                labels.push(u32::try_from(code).expect("the table index is a u32 code"));
            }
        }
        let mapped = codes.iter().map(|&code| rank[code as usize]).collect();
        Self {
            codes: Cow::Owned(mapped),
            levels,
            labels: Some(Cow::Owned(labels)),
        }
    }

    fn by_sorting(codes: &'a [u32]) -> Self {
        let distinct = crate::order::dense_of_present(codes.iter().copied());
        let mapped = codes
            .iter()
            .map(|code| {
                let position = distinct
                    .binary_search(code)
                    .expect("every code is among the distinct codes");
                u32::try_from(position).expect("at most u32::MAX distinct codes")
            })
            .collect();
        Self {
            codes: Cow::Owned(mapped),
            levels: distinct.len(),
            labels: Some(Cow::Owned(distinct)),
        }
    }

    /// The caller's codes behind the levels.
    pub(super) fn levels_view(&self) -> super::Levels<'_> {
        super::Levels {
            labels: self.labels.as_deref(),
            levels: self.levels,
        }
    }

    /// The owned parts, when recoding had to allocate them: the recoded
    /// codes, the number of levels and the original code of each level. A
    /// column that was already dense has neither codes nor labels.
    pub(super) fn into_recoded(self) -> RecodedParts {
        match (self.codes, self.labels) {
            (Cow::Owned(codes), Some(labels)) => {
                (Some(codes), self.levels, Some(labels.into_owned()))
            }
            _ => (None, self.levels, None),
        }
    }
}

/// See [`Dense::into_recoded`].
pub(super) type RecodedParts = (Option<Vec<u32>>, usize, Option<Vec<u32>>);
