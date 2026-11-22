//! Sorting of (key, index) pairs by least-significant-digit radix sort.
//!
//! # Algorithm
//!
//! A stable LSD radix sort on `DIGIT_BITS`-bit digits orders `n` keys of `b`
//! bits in `ceil(b / DIGIT_BITS)` passes over the data, each a count of the
//! digits, a prefix sum of the counts and a scatter. Keys are first shifted
//! down by the smallest key, so `b` is the width of their range. A pass in
//! which every key has the same digit changes nothing and is skipped. Below
//! `DAYS_FROM` or `KEYED_FROM` entries the comparison sort is faster and is used instead.
//!
//! A (day, index) pair is packed into one word, the shifted day above the
//! index, so that the word order is the pair order and one `u64` moves per
//! entry. A (key, index) pair with a 64-bit key is sorted on the key with the
//! index as payload; the sort is stable, so entries with equal keys keep their
//! input order, and runs of equal keys are then put in index order, which
//! makes the result that of sorting the pairs.
//!
//! Every function gives exactly the order of `sort_unstable` on the pairs.
//!
//! # References
//!
//! - Knuth, D. E. (1998). The Art of Computer Programming, vol. 3: Sorting
//!   and Searching, 2nd ed., section 5.2.5.

/// Bits of the key sorted in one pass.
const DIGIT_BITS: u32 = 11;

/// Fewer (day, index) pairs than this are sorted by comparison. Measured on
/// an M3 Pro, the radix sort is as fast at about 1,500 and 1.7 times as fast
/// at 10^6.
const DAYS_FROM: usize = 1 << 11;

/// Fewer (key, index) pairs than this are sorted by comparison. The radix
/// sort is as fast at about 4,000 and 1.8 times as fast at 10^5.
const KEYED_FROM: usize = 1 << 12;

const RADIX: usize = 1 << DIGIT_BITS;

/// The number of bits needed to write `value`.
const fn width(value: u64) -> u32 {
    u64::BITS - value.leading_zeros()
}

/// The start of each digit's block of output, from the digit counts.
fn offsets(counts: &mut [usize; RADIX]) {
    let mut total = 0;
    for count in counts.iter_mut() {
        let here = *count;
        *count = total;
        total += here;
    }
}

/// `words` sorted ascending, which are known to be below `2^bits`.
fn radix_words(words: &mut Vec<u64>, bits: u32) {
    let mut scratch = vec![0_u64; words.len()];
    for pass in 0..bits.div_ceil(DIGIT_BITS) {
        let shift = pass * DIGIT_BITS;
        let digit = |word: u64| ((word >> shift) as usize) & (RADIX - 1);
        let mut counts = [0_usize; RADIX];
        for &word in words.iter() {
            counts[digit(word)] += 1;
        }
        if counts.contains(&words.len()) {
            continue;
        }
        offsets(&mut counts);
        for &word in words.iter() {
            let slot = &mut counts[digit(word)];
            scratch[*slot] = word;
            *slot += 1;
        }
        std::mem::swap(words, &mut scratch);
    }
}

/// Sort `entries` by day, then by index.
pub(crate) fn sort_days(entries: &mut Vec<(i32, u32)>) {
    if entries.len() < DAYS_FROM {
        entries.sort_unstable();
        return;
    }
    let (min, max) = entries
        .iter()
        .fold((i32::MAX, i32::MIN), |(min, max), &(day, _)| {
            (min.min(day), max.max(day))
        });
    let index_bits = width(u64::from(entries.iter().map(|e| e.1).max().unwrap_or(0)));
    let day_bits = width(i64::from(max).abs_diff(i64::from(min)));
    let mut words: Vec<u64> = entries
        .iter()
        .map(|&(day, index)| {
            (i64::from(day).abs_diff(i64::from(min)) << index_bits) | u64::from(index)
        })
        .collect();
    radix_words(&mut words, day_bits + index_bits);
    let mask = (1_u64 << index_bits) - 1;
    for (entry, word) in entries.iter_mut().zip(words) {
        let day = i64::from(min) + i64::try_from(word >> index_bits).expect("a day range");
        *entry = (
            i32::try_from(day).expect("a day between the extremes"),
            (word & mask) as u32,
        );
    }
}

/// [`sort_days`] for indices of type `usize`.
pub(crate) fn sort_days_usize(entries: &mut Vec<(i32, usize)>) {
    let narrow: Option<Vec<(i32, u32)>> = (entries.len() >= DAYS_FROM)
        .then(|| {
            entries
                .iter()
                .map(|&(day, index)| u32::try_from(index).ok().map(|index| (day, index)))
                .collect()
        })
        .flatten();
    let Some(mut narrow) = narrow else {
        entries.sort_unstable();
        return;
    };
    sort_days(&mut narrow);
    for (entry, (day, index)) in entries.iter_mut().zip(narrow) {
        *entry = (day, index as usize);
    }
}

/// Sort `entries` by key, then by index.
pub(crate) fn sort_keyed(entries: &mut Vec<(u64, u32)>) {
    if entries.len() < KEYED_FROM {
        entries.sort_unstable();
        return;
    }
    let (min, max) = entries.iter().fold((u64::MAX, 0), |(min, max), &(key, _)| {
        (min.min(key), max.max(key))
    });
    let mut scratch = vec![(0_u64, 0_u32); entries.len()];
    for pass in 0..width(max - min).div_ceil(DIGIT_BITS) {
        let shift = pass * DIGIT_BITS;
        let digit = |key: u64| (((key - min) >> shift) as usize) & (RADIX - 1);
        let mut counts = [0_usize; RADIX];
        for &(key, _) in entries.iter() {
            counts[digit(key)] += 1;
        }
        if counts.contains(&entries.len()) {
            continue;
        }
        offsets(&mut counts);
        for &entry in entries.iter() {
            let slot = &mut counts[digit(entry.0)];
            scratch[*slot] = entry;
            *slot += 1;
        }
        std::mem::swap(entries, &mut scratch);
    }
    let mut start = 0;
    while start < entries.len() {
        let key = entries[start].0;
        let mut end = start + 1;
        while end < entries.len() && entries[end].0 == key {
            end += 1;
        }
        if end - start > 1 {
            entries[start..end].sort_unstable_by_key(|entry| entry.1);
        }
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mix(u64);

    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    const SIZES: [usize; 10] = [0, 1, 2, 100, 2047, 2048, 4095, 4096, 4097, 20_011];

    #[test]
    fn days_sort_as_pairs_do() {
        let mut rng = Mix(1);
        for n in SIZES {
            for spread in [1_u64, 40, 11_000, u64::from(u32::MAX)] {
                let mut pairs: Vec<(i32, u32)> = (0..n)
                    .map(|i| {
                        let day = (rng.next() % spread) as i64 - i64::from(i32::MAX / 2);
                        (
                            day as i32,
                            (rng.next() % (n as u64 + 1)) as u32 + i as u32 % 3,
                        )
                    })
                    .collect();
                let mut expected = pairs.clone();
                expected.sort_unstable();
                sort_days(&mut pairs);
                assert_eq!(pairs, expected, "n {n}, spread {spread}");
            }
        }
    }

    #[test]
    fn days_with_the_extreme_values_sort() {
        let mut pairs: Vec<(i32, u32)> = (0..DAYS_FROM as u32 + 5)
            .map(|i| match i % 3 {
                0 => (i32::MIN, i),
                1 => (i32::MAX, u32::MAX - i),
                _ => (0, i),
            })
            .collect();
        let mut expected = pairs.clone();
        expected.sort_unstable();
        sort_days(&mut pairs);
        assert_eq!(pairs, expected);
    }

    #[test]
    fn days_with_usize_indices_sort_as_pairs_do() {
        let mut rng = Mix(2);
        for n in SIZES {
            let mut pairs: Vec<(i32, usize)> = (0..n)
                .map(|i| ((rng.next() % 500) as i32 + 700_000, i / 2))
                .collect();
            let mut expected = pairs.clone();
            expected.sort_unstable();
            sort_days_usize(&mut pairs);
            assert_eq!(pairs, expected);
        }
    }

    #[test]
    fn keyed_pairs_sort_as_pairs_do() {
        let mut rng = Mix(3);
        for n in SIZES {
            for distinct in [3_u64, 1000, u64::MAX] {
                let mut pairs: Vec<(u64, u32)> = (0..n)
                    .map(|_| {
                        let key = (rng.next() % distinct).wrapping_add(u64::MAX - 5_000);
                        (key, (rng.next() % 50) as u32)
                    })
                    .collect();
                let mut expected = pairs.clone();
                expected.sort_unstable();
                sort_keyed(&mut pairs);
                assert_eq!(pairs, expected, "n {n}, distinct {distinct}");
            }
        }
    }
}
