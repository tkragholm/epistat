//! The records of a cohort in the order of their ids, computed once per job
//! (once per panel for sequential trials) and read by every step that orders
//! records by id: the validation of the cohort, the order of the cases, the
//! pools of the scan and the strata of the sweep.
//!
//! # Algorithm
//!
//! Each record is keyed by the first eight bytes of its id as a big-endian
//! integer and by its index. The keys are sorted, reading the ids themselves
//! only when two prefixes are equal, so the order is that of `(id, index)`
//! under `str::cmp`. The rank of a record is its position in that order.
//! Comparing ranks therefore compares ids, and records with equal ids by
//! index, which is the order a stable sort by id leaves them in.
//!
//! The order of a subset of the records, listed in increasing index, is the
//! order of the whole restricted to the subset, read off in one pass over it.
//!
//! # Complexity
//!
//! `O(n log n)` comparisons of integers for `n` records, plus one comparison
//! of ids for each comparison of equal prefixes; `2 n` words. A subset of `m`
//! records of a cohort of `n` costs `O(n)` time and `n + 2 m` words.

use crate::matching::{MatchingRecord, id_prefix};
use rayon::slice::ParallelSliceMut;

/// Below this many records the keys are sorted on one thread.
const PARALLEL_SORT: usize = 1 << 16;

/// The records of a cohort in the order of `(id, index)`.
#[derive(Debug, Clone)]
pub(crate) struct IdOrder {
    /// Record indices in the order of `(id, index)`.
    by_id: Vec<u32>,
    /// `rank[idx]`: the position of record `idx` in `by_id`.
    rank: Vec<u32>,
    /// Whether every id occurs once.
    distinct: bool,
}

impl IdOrder {
    /// The order of `records`, or `None` when there are more than `u32::MAX`
    /// of them.
    pub(crate) fn of<R: MatchingRecord + Sync>(records: &[R]) -> Option<Self> {
        Self::of_ids(records.len(), |idx| records[idx].id())
    }

    /// The order of `n` records whose ids `id` gives, or `None` when there
    /// are more than `u32::MAX` of them.
    pub(crate) fn of_ids<'r>(n: usize, id: impl Fn(usize) -> &'r str + Sync) -> Option<Self> {
        let n = u32::try_from(n).ok()?;
        let mut keyed: Vec<(u64, u32)> = (0..n)
            .map(|idx| (id_prefix(id(idx as usize)), idx))
            .collect();
        let compare = |left: &(u64, u32), right: &(u64, u32)| {
            left.0
                .cmp(&right.0)
                .then_with(|| id(left.1 as usize).cmp(id(right.1 as usize)))
                .then_with(|| left.1.cmp(&right.1))
        };
        // A total order, so the parallel sort gives the same order as the
        // sequential one.
        if keyed.len() < PARALLEL_SORT {
            keyed.sort_unstable_by(compare);
        } else {
            keyed.par_sort_unstable_by(compare);
        }
        let by_id: Vec<u32> = keyed.iter().map(|&(_, idx)| idx).collect();
        let distinct = keyed
            .windows(2)
            .all(|pair| pair[0].0 != pair[1].0 || id(pair[0].1 as usize) != id(pair[1].1 as usize));
        Some(Self::from_sorted(by_id, distinct))
    }

    fn from_sorted(by_id: Vec<u32>, distinct: bool) -> Self {
        let mut rank = vec![0_u32; by_id.len()];
        for (position, &idx) in by_id.iter().enumerate() {
            rank[idx as usize] = u32::try_from(position).expect("at most u32::MAX records");
        }
        Self {
            by_id,
            rank,
            distinct,
        }
    }

    /// The order of the records `members[0], members[1], ...` of this
    /// cohort, which are distinct and in increasing order, as a cohort of
    /// their own in which `members[k]` is record `k`.
    pub(crate) fn subset(&self, members: &[u32]) -> Self {
        debug_assert!(members.windows(2).all(|pair| pair[0] < pair[1]));
        let mut position = vec![u32::MAX; self.by_id.len()];
        for (k, &member) in members.iter().enumerate() {
            position[member as usize] = u32::try_from(k).expect("at most u32::MAX records");
        }
        let by_id = self
            .by_id
            .iter()
            .map(|&idx| position[idx as usize])
            .filter(|&k| k != u32::MAX)
            .collect();
        // Distinct ids stay distinct in a subset. Otherwise the subset is
        // marked as possibly repeating, which [`Self::first_repeat`] settles.
        Self::from_sorted(by_id, self.distinct)
    }

    /// The number of records.
    pub(crate) fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Record indices in the order of `(id, index)`.
    pub(crate) fn by_id(&self) -> &[u32] {
        &self.by_id
    }

    /// The position of each record in [`Self::by_id`].
    pub(crate) fn ranks(&self) -> &[u32] {
        &self.rank
    }

    /// The position of record `idx` in [`Self::by_id`].
    pub(crate) fn rank(&self, idx: usize) -> u32 {
        self.rank[idx]
    }

    /// Whether every id occurs once.
    pub(crate) const fn distinct(&self) -> bool {
        self.distinct
    }

    /// The first record, in index order, whose id an earlier record has.
    pub(crate) fn first_repeat<R: MatchingRecord>(&self, records: &[R]) -> Option<usize> {
        if self.distinct {
            return None;
        }
        // Within a run of equal ids the indices increase, so the second entry
        // of a run is its first repeat.
        self.by_id
            .windows(2)
            .filter(|pair| records[pair[0] as usize].id() == records[pair[1] as usize].id())
            .map(|pair| pair[1] as usize)
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn records(ids: &[&str]) -> Vec<BaseRecord> {
        let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
        ids.iter().map(|id| BaseRecord::new(*id, birth)).collect()
    }

    /// Ids of every length around the eight bytes of the prefix, with shared
    /// prefixes, zero bytes, repeats and bytes above ASCII.
    fn awkward_ids(seed: u64, n: usize) -> Vec<String> {
        let alphabet: [&str; 7] = ["a", "b", "\0", "é", "ab", "aaaaaaaa", "aaaaaaab"];
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let parts = (state >> 60) as usize % 4 + 1;
                (0..parts)
                    .map(|k| alphabet[usize::try_from((state >> (8 * k + 8)) % 7).expect("small")])
                    .collect()
            })
            .collect()
    }

    /// The order is that of a stable sort by id, the ranks invert it, and the
    /// first repeat is the first record whose id was seen before.
    #[test]
    fn the_order_is_a_stable_sort_by_id() {
        for (seed, n) in [(1, 0), (2, 1), (3, 40), (4, 500), (5, 70_000)] {
            let ids = awkward_ids(seed, n);
            let people = records(&ids.iter().map(String::as_str).collect::<Vec<_>>());
            let order = IdOrder::of(&people).expect("fits");
            let mut stable: Vec<usize> = (0..n).collect();
            stable.sort_by(|&a, &b| ids[a].cmp(&ids[b]));
            let got: Vec<usize> = order.by_id().iter().map(|&idx| idx as usize).collect();
            assert_eq!(got, stable, "seed {seed}");
            for (position, &idx) in order.by_id().iter().enumerate() {
                assert_eq!(order.rank(idx as usize) as usize, position);
            }
            let mut seen = std::collections::HashSet::new();
            let repeat = ids.iter().position(|id| !seen.insert(id.clone()));
            assert_eq!(order.first_repeat(&people), repeat, "seed {seed}");
            assert_eq!(order.distinct(), repeat.is_none());
        }
    }

    #[test]
    fn a_subset_is_ordered_as_its_own_cohort() {
        let ids: Vec<String> = (0..300).map(|i| format!("{}", (i * 7919) % 1000)).collect();
        let people = records(&ids.iter().map(String::as_str).collect::<Vec<_>>());
        let order = IdOrder::of(&people).expect("fits");
        let members: Vec<u32> = (0..300).filter(|i| i % 3 != 1).collect();
        let sub: Vec<BaseRecord> = members
            .iter()
            .map(|&i| people[i as usize].clone())
            .collect();
        let direct = IdOrder::of(&sub).expect("fits");
        let derived = order.subset(&members);
        assert_eq!(derived.by_id(), direct.by_id());
        assert_eq!(derived.ranks(), direct.ranks());
        assert_eq!(derived.len(), sub.len());
    }
}
