//! Dense recoding of stratum, cluster and id codes.

use std::borrow::Cow;

/// Codes `0..levels`, each level present, in the order of the caller's codes.
#[derive(Debug, Clone)]
pub(super) struct Dense<'a> {
    pub(super) codes: Cow<'a, [u32]>,
    pub(super) levels: usize,
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
        let mut levels = 0_u32;
        for slot in &mut rank {
            if *slot == 1 {
                *slot = levels;
                levels += 1;
            }
        }
        let levels = levels as usize;
        if levels == width {
            return Self {
                codes: Cow::Borrowed(codes),
                levels,
            };
        }
        let mapped = codes.iter().map(|&code| rank[code as usize]).collect();
        Self {
            codes: Cow::Owned(mapped),
            levels,
        }
    }

    fn by_sorting(codes: &'a [u32]) -> Self {
        let mut distinct = codes.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
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
        }
    }

    /// The recoded codes, when recoding had to allocate them.
    pub(super) fn into_recoded(self) -> (Option<Vec<u32>>, usize) {
        match self.codes {
            Cow::Owned(codes) => (Some(codes), self.levels),
            Cow::Borrowed(_) => (None, self.levels),
        }
    }
}
