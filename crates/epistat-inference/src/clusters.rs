//! Cluster labels reduced to the numbers `0..G`.
//!
//! [`Clusters::from_labels`] numbers the distinct labels in the order in which
//! they first appear. A hash map looks the labels up, but the numbers come
//! from a counter and never from the map's iteration order, so the numbering
//! does not depend on the hasher or its random state. Every later sum over
//! clusters runs in this order, which fixes the order of floating-point
//! additions and makes the results reproducible. [`Clusters::singletons`] puts
//! each observation in its own cluster. [`Clusters::from_sorted_labels`]
//! numbers the labels in increasing order instead, which is the order R
//! sums clusters in.

use std::collections::HashMap;
use std::hash::Hash;

/// The cluster of each observation, numbered from 0 in order of first
/// appearance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clusters {
    of: Vec<usize>,
    count: usize,
}

impl Clusters {
    /// Numbers the distinct labels in the order they first appear.
    #[must_use]
    pub fn from_labels<L: Eq + Hash>(labels: &[L]) -> Self {
        let mut number: HashMap<&L, usize> = HashMap::new();
        let of = labels
            .iter()
            .map(|label| {
                let next = number.len();
                *number.entry(label).or_insert(next)
            })
            .collect();
        Self {
            of,
            count: number.len(),
        }
    }

    /// Numbers the distinct labels in increasing order, as R's `rowsum`
    /// with `reorder = TRUE` and `sort(unique(x))` do.
    #[must_use]
    pub fn from_sorted_labels<L: Ord>(labels: &[L]) -> Self {
        let mut distinct: Vec<&L> = labels.iter().collect();
        distinct.sort_unstable();
        distinct.dedup();
        let of = labels
            .iter()
            .map(|label| {
                // Every label is among the distinct ones, so this is `Ok`.
                distinct.binary_search(&label).unwrap_or_else(|at| at)
            })
            .collect();
        Self {
            of,
            count: distinct.len(),
        }
    }

    /// Every observation its own cluster.
    #[must_use]
    pub fn singletons(n: usize) -> Self {
        Self {
            of: (0..n).collect(),
            count: n,
        }
    }

    /// The cluster of each observation, from 0.
    #[must_use]
    pub fn of(&self) -> &[usize] {
        &self.of
    }

    /// The number of distinct clusters.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// The number of observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.of.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.of.is_empty()
    }

    /// The sum of `values` within each cluster, adding in observation order.
    pub(crate) fn sums(&self, values: impl IntoIterator<Item = f64>) -> Vec<f64> {
        let mut sums = vec![0.0; self.count];
        for (value, &cluster) in values.into_iter().zip(&self.of) {
            sums[cluster] += value;
        }
        sums
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusters_are_numbered_by_first_appearance() {
        let clusters = Clusters::from_labels(&["b", "a", "b", "c", "a"]);
        assert_eq!(clusters.of(), [0, 1, 0, 2, 1]);
        assert_eq!((clusters.count(), clusters.len()), (3, 5));
        assert_eq!(clusters.sums([1.0, 2.0, 3.0, 4.0, 5.0]), [4.0, 7.0, 4.0]);
    }

    #[test]
    fn sorted_labels_are_numbered_in_increasing_order() {
        let clusters = Clusters::from_sorted_labels(&[30, 10, 30, 20, 10]);
        assert_eq!(clusters.of(), [2, 0, 2, 1, 0]);
        assert_eq!((clusters.count(), clusters.len()), (3, 5));
    }
}
