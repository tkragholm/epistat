//! Cluster labels reduced to the numbers `0..G`.
//!
//! [`Clusters::from_codes`] numbers the distinct `u32` codes in increasing
//! order, which for codes from `epistat_core::Codes` is the sorted order of
//! the labels. [`Clusters::per_unit`] takes a unit column and a cluster
//! column of person-period data, checks that each unit stays in one
//! cluster, and gives one entry per unit; an influence function over a
//! subset of those units looks its clusters up by unit code
//! ([`Clusters::for_units`]). [`Clusters::from_labels`] numbers the distinct
//! labels in the order in which they first appear. A hash map looks the labels up, but the numbers come
//! from a counter and never from the map's iteration order, so the numbering
//! does not depend on the hasher or its random state. Every later sum over
//! clusters runs in this order, which fixes the order of floating-point
//! additions and makes the results reproducible. [`Clusters::singletons`] puts
//! each observation in its own cluster. [`Clusters::from_sorted_labels`]
//! numbers the labels in increasing order instead, which is the order R
//! sums clusters in.

use std::collections::HashMap;
use std::hash::Hash;

use epistat_core::{InputError, check};

use crate::Error;

/// The cluster of each observation, numbered from 0.
///
/// The numbering is the order of first appearance for
/// [`Clusters::from_labels`] and increasing order of label or code for the
/// other constructors. Built by [`Clusters::per_unit`], it also knows the unit
/// code of each entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clusters {
    of: Vec<usize>,
    count: usize,
    units: Option<Vec<u32>>,
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
            units: None,
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
            units: None,
        }
    }

    /// Numbers the distinct codes in increasing order, so clusters are summed
    /// in code order. Codes from `epistat_core::Codes::from_labels` are in
    /// sorted label order, and then so are the clusters.
    #[must_use]
    pub fn from_codes(codes: &[u32]) -> Self {
        Self::from_sorted_labels(codes)
    }

    /// The cluster of each unit of person-period data, from the unit and
    /// cluster code of each row.
    ///
    /// The result has one entry per distinct unit, in increasing unit code,
    /// and numbers the clusters in increasing cluster code. It remembers the
    /// unit codes, so that [`crate::Influence::variance`] can look up the
    /// cluster of each unit an estimator kept.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for no rows, [`InputError::Length`] when the two
    /// columns differ in length, and [`InputError::NotConstant`] for a unit
    /// that appears in two clusters.
    pub fn per_unit(unit: &[u32], cluster: &[u32]) -> Result<Self, Error> {
        check::non_empty("unit", unit.len())?;
        check::same_length("cluster", unit.len(), cluster.len())?;
        check::constant_within("cluster", unit, cluster)?;
        let mut pairs: Vec<(u32, u32)> =
            unit.iter().copied().zip(cluster.iter().copied()).collect();
        pairs.sort_unstable();
        pairs.dedup_by_key(|pair| pair.0);
        let (units, codes): (Vec<u32>, Vec<u32>) = pairs.into_iter().unzip();
        let numbered = Self::from_codes(&codes);
        Ok(Self {
            units: Some(units),
            ..numbered
        })
    }

    /// The clusters of `units`, looked up by unit code in clusters built by
    /// [`Clusters::per_unit`], and numbered again in increasing order among
    /// the clusters those units reach. A cluster none of them is in does not
    /// count.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for clusters without unit codes, and
    /// [`InputError::Domain`] for a unit the clusters do not cover.
    pub fn for_units(&self, units: &[u32]) -> Result<Self, Error> {
        let Some(known) = &self.units else {
            return Err(InputError::Empty {
                what: "unit codes of the clusters",
            }
            .into());
        };
        let mut numbers = Vec::with_capacity(units.len());
        for &unit in units {
            let at = known.binary_search(&unit).map_err(|_| InputError::Domain {
                what: "unit",
                value: f64::from(unit),
                domain: "the units of the clusters",
                row: None,
            })?;
            numbers.push(self.of[at]);
        }
        let numbered = Self::from_sorted_labels(&numbers);
        Ok(Self {
            units: Some(units.to_vec()),
            ..numbered
        })
    }

    /// Every observation its own cluster.
    #[must_use]
    pub fn singletons(n: usize) -> Self {
        Self {
            of: (0..n).collect(),
            count: n,
            units: None,
        }
    }

    /// The unit code of each entry, for clusters built by
    /// [`Clusters::per_unit`] or [`Clusters::for_units`].
    #[must_use]
    pub fn units(&self) -> Option<&[u32]> {
        self.units.as_deref()
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

    /// Whether there are no observations.
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
    fn per_unit_clusters_follow_the_units_an_estimator_kept() {
        let unit = [4, 4, 1, 1, 9, 2, 2];
        let cluster = [30, 30, 10, 10, 30, 20, 20];
        let clusters = Clusters::per_unit(&unit, &cluster).unwrap();
        assert_eq!(clusters.units(), Some(&[1, 2, 4, 9][..]));
        assert_eq!(clusters.of(), [0, 1, 2, 2]);
        let kept = clusters.for_units(&[9, 1, 4]).unwrap();
        assert_eq!((kept.of(), kept.count()), (&[1, 0, 1][..], 2));
        assert!(matches!(
            clusters.for_units(&[5]),
            Err(Error::Input(InputError::Domain { .. }))
        ));
        assert!(matches!(
            Clusters::per_unit(&[1, 1], &[1, 2]),
            Err(Error::Input(InputError::NotConstant { unit: 1, .. }))
        ));
        assert!(Clusters::from_codes(&[3]).for_units(&[3]).is_err());
    }

    #[test]
    fn codes_are_numbered_in_increasing_order() {
        assert_eq!(Clusters::from_codes(&[7, 2, 7, 5]).of(), [2, 0, 2, 1]);
    }

    #[test]
    fn sorted_labels_are_numbered_in_increasing_order() {
        let clusters = Clusters::from_sorted_labels(&[30, 10, 30, 20, 10]);
        assert_eq!(clusters.of(), [2, 0, 2, 1, 0]);
        assert_eq!((clusters.count(), clusters.len()), (3, 5));
    }
}
