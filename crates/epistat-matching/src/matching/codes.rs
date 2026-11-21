//! Dense numbers for keys, taken once per job.
//!
//! Exact, coarsened exact and quick matching stratify units on the values of
//! several variables. The numbers here replace a key built per unit, such as a
//! list of owned strings, with one `u32` per unit.
//!
//! # Algorithm
//!
//! [`first_appearance_codes`] numbers the distinct keys of a slice from 0 in
//! the order in which they first appear. The slice is cut into chunks of a
//! fixed length. Each chunk is numbered on its own, in parallel, giving its
//! distinct keys in order of first appearance and a local code per element.
//! The distinct keys of the chunks are then merged in chunk order into one
//! table, which gives each its global number, and the local codes are mapped to
//! global numbers in parallel. A key's global number is that of the earliest
//! chunk that holds it, and within a chunk the order is the order of first
//! appearance, so the numbering is that of a single pass over the slice. The
//! chunk length does not depend on the number of threads, so the result does
//! not either.
//!
//! [`combine_codes`] turns one code column per variable into one number per
//! unit by joining the columns two at a time. The pair of a number so far and
//! the next code is itself numbered by first appearance, so after the last
//! column the numbers are those of first appearance of the whole combination of
//! values.
//!
//! # Complexity
//!
//! With `n` keys and `d` distinct keys, `O(n)` hashing work in parallel and
//! `O(d * c)` in the sequential merge, where `c` is the number of chunks that
//! hold a given key, at most `n / CHUNK`. Memory is `O(n)` for the codes and
//! `O(d)` per chunk for the local tables.

use rapidhash::RapidHashMap;
use rayon::prelude::*;
use std::hash::Hash;

/// Elements numbered together by one task. Fixed, so the result is the same on
/// any number of threads.
const CHUNK: usize = 16_384;

/// The number of each key and the count of distinct keys, numbering the keys
/// from 0 in order of first appearance.
///
/// # Panics
///
/// When the slice holds more than `u32::MAX` distinct keys.
pub(super) fn first_appearance_codes<K>(keys: &[K]) -> (Vec<u32>, usize)
where
    K: Hash + Eq + Copy + Send + Sync,
{
    let chunks: Vec<(Vec<K>, Vec<u32>)> = keys
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut local: RapidHashMap<K, u32> = RapidHashMap::default();
            let mut distinct: Vec<K> = Vec::new();
            let codes: Vec<u32> = chunk
                .iter()
                .map(|key| {
                    *local.entry(*key).or_insert_with(|| {
                        distinct.push(*key);
                        u32::try_from(distinct.len() - 1).expect("fewer than 2^32 keys in a chunk")
                    })
                })
                .collect();
            (distinct, codes)
        })
        .collect();
    let mut global: RapidHashMap<K, u32> = RapidHashMap::default();
    let remaps: Vec<Vec<u32>> = chunks
        .iter()
        .map(|(distinct, _)| {
            distinct
                .iter()
                .map(|key| {
                    let next = u32::try_from(global.len()).expect("fewer than 2^32 distinct keys");
                    *global.entry(*key).or_insert(next)
                })
                .collect()
        })
        .collect();
    let mut out = vec![0_u32; keys.len()];
    out.par_chunks_mut(CHUNK)
        .zip(chunks.par_iter().zip(remaps.par_iter()))
        .for_each(|(slot, ((_, codes), remap))| {
            for (slot, code) in slot.iter_mut().zip(codes) {
                *slot = remap[*code as usize];
            }
        });
    (out, global.len())
}

/// One number per unit from one code column per variable, numbered from 0 in
/// order of first appearance of the combination of codes, with the count of
/// distinct combinations.
///
/// Every column has the length of the units. Codes need not be dense or in any
/// order. With no column the result is empty.
pub(super) fn combine_codes(columns: &[Vec<u32>]) -> (Vec<u32>, usize) {
    let Some((first, rest)) = columns.split_first() else {
        return (Vec::new(), 0);
    };
    let (mut numbers, mut count) = first_appearance_codes(first);
    for column in rest {
        let pairs: Vec<u64> = numbers
            .par_iter()
            .zip(column.par_iter())
            .map(|(number, code)| (u64::from(*number) << 32) | u64::from(*code))
            .collect();
        (numbers, count) = first_appearance_codes(&pairs);
    }
    (numbers, count)
}

/// The first position where `slot` is `None`.
pub(super) fn first_missing<T: Sync>(slot: &[Option<T>]) -> Option<usize> {
    slot.par_iter().position_first(Option::is_none)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference<K: Hash + Eq + Copy>(keys: &[K]) -> (Vec<u32>, usize) {
        let mut seen: std::collections::HashMap<K, u32> = std::collections::HashMap::new();
        let codes = keys
            .iter()
            .map(|key| {
                let next = u32::try_from(seen.len()).unwrap();
                *seen.entry(*key).or_insert(next)
            })
            .collect();
        (codes, seen.len())
    }

    #[test]
    fn numbers_by_first_appearance_across_chunks() {
        for (n, modulus) in [
            (0_usize, 3_u64),
            (5, 3),
            (CHUNK, 7),
            (3 * CHUNK + 11, 1000),
            (50_000, 49_999),
        ] {
            let keys: Vec<u64> = (0..n as u64)
                .map(|i| (i * 2_654_435_761) % modulus)
                .collect();
            assert_eq!(first_appearance_codes(&keys), reference(&keys), "n {n}");
        }
    }

    #[test]
    fn combination_is_first_appearance_of_the_tuple() {
        let n = 40_000_u64;
        let a: Vec<u32> = (0..n).map(|i| ((i * 7919) % 13) as u32 + 100).collect();
        let b: Vec<u32> = (0..n).map(|i| ((i * 104_729) % 5) as u32).collect();
        let c: Vec<u32> = (0..n).map(|i| (i % 2) as u32 * 9).collect();
        let tuples: Vec<(u32, u32, u32)> = (0..usize::try_from(n).unwrap())
            .map(|i| (a[i], b[i], c[i]))
            .collect();
        assert_eq!(
            combine_codes(&[a.clone(), b.clone(), c]),
            reference(&tuples)
        );
        let single: Vec<(u32,)> = a.iter().map(|x| (*x,)).collect();
        assert_eq!(combine_codes(&[a]), reference(&single));
        assert_eq!(combine_codes(&[]), (Vec::new(), 0));
    }
}
