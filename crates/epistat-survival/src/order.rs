//! Orderings by a float key with ties broken by position, and the dense
//! recoding of integer codes by the sorted codes present.
//!
//! The order of every sort here is strict, so an unstable or parallel sort
//! gives the same permutation as a stable sequential one. Dense recoding by
//! sorted code differs from `epistat_inference::Clusters`, which numbers
//! groups in order of first appearance: the order of a sum over groups
//! follows the numbering, so the two are not interchangeable.

use rayon::prelude::*;

/// Sorts `items` by `key(item)` under `f64::total_cmp`, ties by the item.
pub(crate) fn sort_by_key<T: Ord + Copy>(items: &mut [T], key: impl Fn(T) -> f64) {
    items.sort_unstable_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
}

/// [`sort_by_key`] on the rayon pool.
pub(crate) fn par_sort_by_key<T: Ord + Copy + Send>(
    items: &mut [T],
    key: impl Fn(T) -> f64 + Sync,
) {
    items.par_sort_unstable_by(|&a, &b| key(a).total_cmp(&key(b)).then(a.cmp(&b)));
}

/// The row indices `0..keys.len()` ordered by `keys`, ties by index.
pub(crate) fn by_key(keys: &[f64]) -> Vec<u32> {
    let mut index: Vec<u32> = (0..u32::try_from(keys.len()).expect("rows fit u32")).collect();
    sort_by_key(&mut index, |row| keys[row as usize]);
    index
}

/// The distinct values of `codes`, ascending. A code's dense number is its
/// position here, which [`dense_index`] finds.
pub(crate) fn dense_of_present(codes: impl IntoIterator<Item = u32>) -> Vec<u32> {
    let mut present: Vec<u32> = codes.into_iter().collect();
    present.sort_unstable();
    present.dedup();
    present
}

/// The position of `code` among the ascending `present` codes, or the number
/// of codes below it when it is absent.
pub(crate) fn dense_index(present: &[u32], code: u32) -> usize {
    present.partition_point(|&p| p < code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_keep_the_order_of_the_rows_and_nan_sorts_last() {
        assert_eq!(by_key(&[2.0, 1.0, 2.0, f64::NAN, 1.0]), [1, 4, 0, 2, 3]);
        let mut items = [3_usize, 0, 2, 1];
        let key = [1.0, 0.0, 0.0, 0.0];
        sort_by_key(&mut items, |i| key[i]);
        assert_eq!(items, [1, 2, 3, 0]);
        let mut parallel = [3_usize, 0, 2, 1];
        par_sort_by_key(&mut parallel, |i| key[i]);
        assert_eq!(parallel, items);
    }

    #[test]
    fn dense_numbers_follow_the_sorted_codes() {
        let present = dense_of_present([30, 7, 30, 12]);
        assert_eq!(present, [7, 12, 30]);
        assert_eq!(dense_index(&present, 12), 1);
        assert_eq!(dense_index(&present, 30), 2);
    }
}
