//! Cluster labels reduced to the numbers `0..G`.
//!
//! [`Clusters::from_codes`] numbers the distinct `u32` codes in increasing
//! order, which for codes from `epistat_core::Codes` is the sorted order of
//! the labels. [`Clusters::per_unit`] takes a unit column and a cluster
//! column of person-period data, checks that each unit stays in one
//! cluster, and gives one entry per unit, keyed by unit code;
//! [`Clusters::per_row`] gives the cluster of each row of an estimator's
//! input, keyed by row. An influence function over a subset of those keys
//! looks its clusters up by key ([`Clusters::for_keys`]), and only when it is
//! keyed the same way ([`crate::KeyedBy`]).
//!
//! [`Clusters::from_labels_first_seen`] numbers the distinct labels in the
//! order in which they first appear. A hash map looks the labels up, but the
//! numbers come from a counter and never from the map's iteration order, so
//! the numbering does not depend on the hasher or its random state.
//! [`Clusters::from_sorted_labels`] numbers the labels in increasing order
//! instead, which is the order R sums clusters in and the order of
//! `epistat_core::Codes::from_labels`. Every later sum over clusters runs in
//! the numbering order, which fixes the order of floating-point additions and
//! makes the results reproducible, so the two numberings of one column agree
//! to rounding but not always to the bit. [`Clusters::singletons`] puts each
//! observation in its own cluster.

use std::collections::HashMap;
use std::hash::Hash;

use epistat_core::{InputError, check};

use crate::{Error, KeyedBy};

/// The cluster of each observation, numbered from 0.
///
/// The numbering is the order of first appearance for
/// [`Clusters::from_labels_first_seen`] and increasing order of label or code
/// for the other constructors. Built by [`Clusters::per_unit`] or
/// [`Clusters::per_row`], it also knows the key of each entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clusters {
    of: Vec<usize>,
    count: usize,
    keys: Option<(KeyedBy, Vec<u32>)>,
}

impl Clusters {
    /// Numbers the distinct labels in the order they first appear.
    #[deprecated(
        since = "0.4.0",
        note = "use Clusters::from_labels_first_seen, the same numbering under a name that \
                says its order, or Clusters::from_sorted_labels for the order of Codes::from_labels"
    )]
    #[must_use]
    pub fn from_labels<L: Eq + Hash>(labels: &[L]) -> Self {
        Self::from_labels_first_seen(labels)
    }

    /// Numbers the distinct labels in the order they first appear.
    #[must_use]
    pub fn from_labels_first_seen<L: Eq + Hash>(labels: &[L]) -> Self {
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
            keys: None,
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
            keys: None,
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
    /// and numbers the clusters in increasing cluster code. It is keyed by
    /// the unit codes, so that [`crate::Influence::variance`] can look up the
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
            keys: Some((KeyedBy::Units, units)),
            ..numbered
        })
    }

    /// The cluster of each row of an estimator's input, from its cluster
    /// code, keyed by row, for an influence function keyed by rows
    /// ([`crate::Influence::over_rows`]) over all rows or a subset of them.
    /// The clusters are numbered in increasing cluster code.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for no rows and [`InputError::TooLarge`] for more
    /// rows than `u32` numbers.
    pub fn per_row(cluster: &[u32]) -> Result<Self, Error> {
        check::non_empty("cluster", cluster.len())?;
        let n = u32::try_from(cluster.len()).map_err(|_| InputError::TooLarge {
            what: "cluster",
            limit: u64::from(u32::MAX),
            found: cluster.len() as u64,
        })?;
        Ok(Self {
            keys: Some((KeyedBy::Rows, (0..n).collect())),
            ..Self::from_codes(cluster)
        })
    }

    /// The clusters of `keys`, looked up in clusters built by
    /// [`Clusters::per_unit`] or [`Clusters::per_row`], keyed as these are,
    /// and numbered again in increasing order among the clusters those keys
    /// reach. A cluster none of them is in does not count.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for clusters without keys, and
    /// [`InputError::Domain`] for a key the clusters do not cover.
    pub fn for_keys(&self, keys: &[u32]) -> Result<Self, Error> {
        let Some((keyed_by, known)) = &self.keys else {
            return Err(InputError::Empty {
                what: "keys of the clusters",
            }
            .into());
        };
        let mut numbers = Vec::with_capacity(keys.len());
        for &key in keys {
            let at = known.binary_search(&key).map_err(|_| InputError::Domain {
                what: "key",
                value: f64::from(key),
                domain: "the keys of the clusters",
                row: None,
            })?;
            numbers.push(self.of[at]);
        }
        let numbered = Self::from_sorted_labels(&numbers);
        Ok(Self {
            keys: Some((*keyed_by, keys.to_vec())),
            ..numbered
        })
    }

    /// Every observation its own cluster.
    #[must_use]
    pub fn singletons(n: usize) -> Self {
        Self {
            of: (0..n).collect(),
            count: n,
            keys: None,
        }
    }

    /// What the keys of the entries are, for clusters built by
    /// [`Clusters::per_unit`], [`Clusters::per_row`] or [`Clusters::for_keys`].
    #[must_use]
    pub fn keyed_by(&self) -> Option<KeyedBy> {
        self.keys.as_ref().map(|(keyed_by, _)| *keyed_by)
    }

    /// The key of each entry, for clusters that have keys.
    #[must_use]
    pub fn keys(&self) -> Option<&[u32]> {
        self.keys.as_ref().map(|(_, keys)| keys.as_slice())
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
        let clusters = Clusters::from_labels_first_seen(&["b", "a", "b", "c", "a"]);
        assert_eq!(clusters.of(), [0, 1, 0, 2, 1]);
        assert_eq!((clusters.count(), clusters.len()), (3, 5));
        assert_eq!(clusters.sums([1.0, 2.0, 3.0, 4.0, 5.0]), [4.0, 7.0, 4.0]);
    }

    #[test]
    fn per_unit_clusters_follow_the_units_an_estimator_kept() {
        let unit = [4, 4, 1, 1, 9, 2, 2];
        let cluster = [30, 30, 10, 10, 30, 20, 20];
        let clusters = Clusters::per_unit(&unit, &cluster).unwrap();
        assert_eq!(clusters.keys(), Some(&[1, 2, 4, 9][..]));
        assert_eq!(clusters.keyed_by(), Some(KeyedBy::Units));
        assert_eq!(clusters.of(), [0, 1, 2, 2]);
        let kept = clusters.for_keys(&[9, 1, 4]).unwrap();
        assert_eq!((kept.of(), kept.count()), (&[1, 0, 1][..], 2));
        assert_eq!(kept.keyed_by(), Some(KeyedBy::Units));
        assert!(matches!(
            clusters.for_keys(&[5]),
            Err(Error::Input(InputError::Domain { .. }))
        ));
        assert!(matches!(
            Clusters::per_unit(&[1, 1], &[1, 2]),
            Err(Error::Input(InputError::NotConstant { unit: 1, .. }))
        ));
        assert!(Clusters::from_codes(&[3]).for_keys(&[3]).is_err());
    }

    #[test]
    fn row_clusters_number_as_unit_clusters_over_the_row_numbers() {
        let cluster = [30, 10, 30, 20];
        let by_row = Clusters::per_row(&cluster).unwrap();
        let by_unit = Clusters::per_unit(&[0, 1, 2, 3], &cluster).unwrap();
        assert_eq!((by_row.of(), by_row.keys()), (by_unit.of(), by_unit.keys()));
        assert_eq!(by_row.keyed_by(), Some(KeyedBy::Rows));
        let kept = by_row.for_keys(&[0, 2, 3]).unwrap();
        assert_eq!(
            (kept.of(), kept.keyed_by()),
            (&[1, 1, 0][..], Some(KeyedBy::Rows))
        );
        assert!(Clusters::per_row(&[]).is_err());
    }

    #[test]
    #[expect(deprecated, reason = "the 0.3 name keeps its numbering")]
    fn the_deprecated_name_numbers_by_first_appearance() {
        let labels = ["b", "a", "b"];
        assert_eq!(
            Clusters::from_labels(&labels),
            Clusters::from_labels_first_seen(&labels)
        );
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
