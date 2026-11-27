//! The score interval where two groups' score ranges overlap.

use crate::types::CommonSupport;

/// The interval from the larger of two groups' lowest scores to the smaller of
/// their highest, with both ends inside.
///
/// This is MatchIt's `discard` rule: a unit is outside when its score is not
/// finite or lies beyond the other group's range, so a unit equal to an end of
/// that range is kept. The multigroup rule in [`crate::multigroup`] is a
/// different one: it keeps only the units strictly inside the intersection of
/// every group's range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Overlap {
    lower: f64,
    upper: f64,
}

impl Overlap {
    /// The overlap of two `(lowest, highest)` ranges, or `None` when they do
    /// not meet.
    pub(crate) fn from_bounds(left: (f64, f64), right: (f64, f64)) -> Option<Self> {
        let lower = left.0.max(right.0);
        let upper = left.1.min(right.1);
        (lower <= upper).then_some(Self { lower, upper })
    }

    /// The interval spanned by one `(lowest, highest)` range, as produced by
    /// [`finite_bounds`].
    pub(crate) fn of_range(range: (f64, f64)) -> Self {
        Self {
            lower: range.0,
            upper: range.1,
        }
    }

    /// Whether `score` is finite and inside the interval.
    pub(crate) fn contains(self, score: f64) -> bool {
        score.is_finite() && score >= self.lower && score <= self.upper
    }

    /// The `(lower, upper)` ends.
    pub(crate) fn bounds(self) -> (f64, f64) {
        (self.lower, self.upper)
    }
}

/// The lowest and highest finite score among `scores`, or `None` when none is
/// finite.
pub(crate) fn finite_bounds(scores: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    scores
        .into_iter()
        .filter(|score| score.is_finite())
        .fold(None, |bounds, score| match bounds {
            None => Some((score, score)),
            Some((low, high)) => Some((low.min(score), high.max(score))),
        })
}

impl CommonSupport {
    /// Whether the policy discards cases, or cases, outside the overlap.
    pub(crate) fn trims_anchors(self) -> bool {
        matches!(self, Self::Treated | Self::Both)
    }

    /// Whether the policy discards controls outside the overlap.
    pub(crate) fn trims_candidates(self) -> bool {
        matches!(self, Self::Control | Self::Both)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_bounds(scores: &[f64]) -> Option<(f64, f64)> {
        let mut min_score = f64::INFINITY;
        let mut max_score = f64::NEG_INFINITY;
        let mut found = false;
        for &score in scores {
            if !score.is_finite() {
                continue;
            }
            min_score = min_score.min(score);
            max_score = max_score.max(score);
            found = true;
        }
        found.then_some((min_score, max_score))
    }

    fn reference_interval(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
        let lower = left.0.max(right.0);
        let upper = left.1.min(right.1);
        (lower <= upper).then_some((lower, upper))
    }

    fn reference_inside(score: f64, interval: (f64, f64)) -> bool {
        score.is_finite() && score >= interval.0 && score <= interval.1
    }

    #[test]
    fn matches_direct_loop_evaluations() {
        let pool = [
            0.0,
            -0.0,
            0.25,
            0.5,
            0.5,
            0.75,
            1.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut draw = |n: usize| {
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    pool[(state >> 33) as usize % pool.len()]
                })
                .collect::<Vec<f64>>()
        };
        for round in 0..2000 {
            let left = draw(round % 5);
            let right = draw(round % 7);
            let left_bounds = finite_bounds(left.iter().copied());
            let right_bounds = finite_bounds(right.iter().copied());
            assert_eq!(
                left_bounds.map(|b| (b.0.to_bits(), b.1.to_bits())),
                reference_bounds(&left).map(|b| (b.0.to_bits(), b.1.to_bits()))
            );
            let (Some(a), Some(b)) = (left_bounds, right_bounds) else {
                continue;
            };
            let overlap = Overlap::from_bounds(a, b);
            let expected = reference_interval(a, b);
            assert_eq!(overlap.map(Overlap::bounds), expected);
            if let (Some(overlap), Some(expected)) = (overlap, expected) {
                for &score in left.iter().chain(&right) {
                    assert_eq!(overlap.contains(score), reference_inside(score, expected));
                }
            }
        }
    }

    #[test]
    fn policy_names_the_groups_it_trims() {
        assert!(CommonSupport::Treated.trims_anchors());
        assert!(!CommonSupport::Treated.trims_candidates());
        assert!(!CommonSupport::Control.trims_anchors());
        assert!(CommonSupport::Control.trims_candidates());
        assert!(CommonSupport::Both.trims_anchors() && CommonSupport::Both.trims_candidates());
    }
}
