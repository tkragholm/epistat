//! Counting and order statistics over a fixed set of points that are switched
//! on and off.
//!
//! # Algorithm
//!
//! [`Fenwick`] is a binary indexed tree (Fenwick 1994) of counts over the
//! positions `0..n`. Entry `i` (1-based) holds the sum of the `i & -i`
//! positions that end at `i`, so a prefix sum and a point update each visit
//! at most `log2 n + 1` entries. The `k`-th active position is found by
//! descending the implicit tree from its largest power of two.
//!
//! [`WaveletMatrix`] answers "the `k`-th smallest value among the active
//! points whose position is in `l..r`" for points with distinct values in
//! `0..n`, with the wavelet matrix of Claude, Navarro and Ordóñez (2015) over
//! the values and a Fenwick tree of active points on each of its levels.
//! Level `t` sorts the points stably by the top `t` bits of
//! their values, most significant first: points whose next bit is 0 go first,
//! in the order they had, then those whose bit is 1. Each level stores, for
//! every prefix of its order, how many points have a 0 bit there, which maps a
//! range at one level to the two ranges its points occupy at the next. The
//! counts come from a bit vector with a cumulative count per 64 bits and a
//! popcount. A
//! Fenwick tree on each level counts the active points of that level's order.
//! A query descends one level per bit: it counts the active points of the
//! range whose bit is 0, goes to the zeros if `k` is below that count, and to
//! the ones with `k` reduced by it otherwise.
//!
//! # Complexity
//!
//! For `n` points and `L = ceil(log2 n)` levels: building is `O(n L)` time and
//! about `(n + 1) L` 32-bit words for the Fenwick trees and `3 n L / 64` for
//! the bit vectors; switching a point on or off and the `k`-th
//! query are `O(L log n)`. [`Fenwick`] builds in `O(n)` and answers in
//! `O(log n)`.
//!
//! # References
//!
//! - Fenwick, P. M. (1994). A new data structure for cumulative frequency
//!   tables. Software: Practice and Experience 24(3), 327-336.
//! - Claude, F., Navarro, G. and Ordóñez, A. (2015). The wavelet matrix: an
//!   efficient wavelet tree for large alphabets. Information Systems 47,
//!   15-32.

/// Counts over the positions `0..n`, with point updates and prefix sums.
pub(crate) struct Fenwick {
    /// 1-based; entry 0 is unused.
    tree: Vec<u32>,
}

impl Fenwick {
    /// A tree of `n` positions with the given counts, built in `O(n)`.
    pub(crate) fn from_counts(counts: impl ExactSizeIterator<Item = u32>) -> Self {
        let mut tree = Vec::with_capacity(counts.len() + 1);
        tree.push(0);
        tree.extend(counts);
        let n = tree.len();
        for i in 1..n {
            let parent = i + (i & i.wrapping_neg());
            if parent < n {
                tree[parent] = tree[parent].wrapping_add(tree[i]);
            }
        }
        Self { tree }
    }

    /// A tree of `n` positions, all 0.
    pub(crate) fn zeros(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }

    /// Add `delta` (1 or -1) at `position`.
    pub(crate) fn add(&mut self, position: usize, delta: i32) {
        let mut i = position + 1;
        while i < self.tree.len() {
            self.tree[i] = self.tree[i].wrapping_add_signed(delta);
            i += i & i.wrapping_neg();
        }
    }

    /// The sum over the positions `0..end`.
    pub(crate) fn prefix(&self, end: usize) -> u32 {
        let mut i = end;
        let mut sum = 0_u32;
        while i > 0 {
            sum = sum.wrapping_add(self.tree[i]);
            i &= i - 1;
        }
        sum
    }

    /// The sum over the positions `start..end`.
    pub(crate) fn range(&self, start: usize, end: usize) -> u32 {
        self.prefix(end).wrapping_sub(self.prefix(start))
    }

    /// The position of the active point of rank `k` (0-based) when every
    /// count is 0 or 1. `k` must be below the total.
    pub(crate) fn kth(&self, k: u32) -> usize {
        let n = self.tree.len() - 1;
        let mut position = 0_usize;
        let mut remaining = k;
        let mut step = if n == 0 { 0 } else { 1_usize << n.ilog2() };
        while step > 0 {
            let next = position + step;
            if next <= n && self.tree[next] <= remaining {
                position = next;
                remaining -= self.tree[next];
            }
            step >>= 1;
        }
        position
    }
}

/// Counts of several kinds over the positions `0..n`, kept side by side in one
/// tree, so that an update or a range sum walks the tree once for all kinds.
///
/// Kind 0 counts the points in play and every other kind a subset of them, so
/// a point counts under kind 0 and under at most one other kind, its
/// [`Tally`]. Each entry holds [`LANES`] kinds, in groups when there are more,
/// and a walk adds a whole group at each step.
pub(crate) struct CountTree {
    groups: usize,
    /// 1-based entries, `groups` of them per position; the first is unused.
    tree: Vec<[u32; LANES]>,
}

/// The kinds counted side by side in one group of a [`CountTree`].
pub(crate) const LANES: usize = 4;

/// The groups of a [`CountTree`] whose sums a query keeps on the stack.
const STACK_GROUPS: usize = 4;

/// How a point counts: `None` when it is not in play, `Some(None)` when it
/// counts under kind 0 only, and `Some(Some(kind))` when it counts under kind
/// 0 and under `kind`, which is at least 1.
pub(crate) type Tally = Option<Option<usize>>;

/// Whether a point with this tally counts under `kind`.
fn counts_under(tally: Tally, kind: usize) -> i32 {
    match tally {
        None => 0,
        Some(other) => i32::from(kind == 0 || other == Some(kind)),
    }
}

impl CountTree {
    /// A tree of `n` positions, all counts 0, with `kinds` kinds.
    pub(crate) fn zeros(n: usize, kinds: usize) -> Self {
        let groups = kinds.div_ceil(LANES).max(1);
        Self {
            groups,
            tree: vec![[0; LANES]; (n + 1) * groups],
        }
    }

    /// Move the point at `position` from the tally `old` to the tally `new`.
    pub(crate) fn change(&mut self, position: usize, old: Tally, new: Tally) {
        let groups = self.groups;
        let n = self.tree.len() / groups;
        let delta = |group: usize| -> [i32; LANES] {
            std::array::from_fn(|lane| {
                let kind = group * LANES + lane;
                counts_under(new, kind) - counts_under(old, kind)
            })
        };
        let mut i = position + 1;
        if groups == 1 {
            let delta = delta(0);
            while i < n {
                let cell = &mut self.tree[i];
                for lane in 0..LANES {
                    cell[lane] = cell[lane].wrapping_add_signed(delta[lane]);
                }
                i += i & i.wrapping_neg();
            }
        } else {
            let deltas: Vec<[i32; LANES]> = (0..groups).map(delta).collect();
            while i < n {
                let cells = &mut self.tree[i * groups..(i + 1) * groups];
                for (cell, delta) in cells.iter_mut().zip(&deltas) {
                    for lane in 0..LANES {
                        cell[lane] = cell[lane].wrapping_add_signed(delta[lane]);
                    }
                }
                i += i & i.wrapping_neg();
            }
        }
    }

    /// Write the sums over the positions `start..end`, one per kind, to
    /// `out`, which has one entry per kind. The two prefix walks stop where
    /// they meet.
    pub(crate) fn range_into(&self, start: usize, end: usize, out: &mut [u32]) {
        let groups = self.groups;
        if groups == 1 {
            let mut sum = [0_u32; LANES];
            let (mut high, mut low) = (end, start);
            while high > low {
                let cell = &self.tree[high];
                for lane in 0..LANES {
                    sum[lane] = sum[lane].wrapping_add(cell[lane]);
                }
                high &= high - 1;
            }
            while low > high {
                let cell = &self.tree[low];
                for lane in 0..LANES {
                    sum[lane] = sum[lane].wrapping_sub(cell[lane]);
                }
                low &= low - 1;
            }
            for (slot, &value) in out.iter_mut().zip(&sum) {
                *slot = value;
            }
            return;
        }
        let mut stack = [[0_u32; LANES]; STACK_GROUPS];
        let mut heap;
        let sums: &mut [[u32; LANES]] = if groups <= STACK_GROUPS {
            &mut stack[..groups]
        } else {
            heap = vec![[0_u32; LANES]; groups];
            &mut heap
        };
        let (mut high, mut low) = (end, start);
        while high > low {
            for (sum, cell) in sums.iter_mut().zip(&self.tree[high * groups..]) {
                for lane in 0..LANES {
                    sum[lane] = sum[lane].wrapping_add(cell[lane]);
                }
            }
            high &= high - 1;
        }
        while low > high {
            for (sum, cell) in sums.iter_mut().zip(&self.tree[low * groups..]) {
                for lane in 0..LANES {
                    sum[lane] = sum[lane].wrapping_sub(cell[lane]);
                }
            }
            low &= low - 1;
        }
        for (slot, &sum) in out.iter_mut().zip(sums.iter().flatten()) {
            *slot = sum;
        }
    }
}

/// A bit vector that counts its set bits in a prefix with a popcount: one
/// cumulative count per 64-bit word and a popcount of the bits of the word
/// before the position, which is 1.5 bits per position and `O(1)` per query.
struct RankBits {
    words: Vec<u64>,
    /// `before[w]`: the set bits in the words before word `w`.
    before: Vec<u32>,
    total: u32,
}

impl RankBits {
    /// The vector of `bits`.
    fn new(bits: impl Iterator<Item = bool>) -> Self {
        let mut words = Vec::new();
        let mut word = 0_u64;
        let mut length = 0_usize;
        for bit in bits {
            word |= u64::from(bit) << (length % 64);
            length += 1;
            if length.is_multiple_of(64) {
                words.push(word);
                word = 0;
            }
        }
        if !length.is_multiple_of(64) {
            words.push(word);
        }
        let mut before = Vec::with_capacity(words.len());
        let mut total = 0_u32;
        for &word in &words {
            before.push(total);
            total += word.count_ones();
        }
        Self {
            words,
            before,
            total,
        }
    }

    /// The set bits among the first `position` bits, for a position up to
    /// and including the length.
    fn rank(&self, position: usize) -> u32 {
        let (word, bit) = (position / 64, position % 64);
        if bit == 0 {
            return self.before.get(word).copied().unwrap_or(self.total);
        }
        self.before[word] + (self.words[word] & ((1_u64 << bit) - 1)).count_ones()
    }

    fn get(&self, position: usize) -> bool {
        self.words[position / 64] >> (position % 64) & 1 == 1
    }
}

/// One level of a [`WaveletMatrix`].
struct Level {
    /// The points of this level's order whose bit is 0, as set bits.
    zeros: RankBits,
    /// Active points in the order of the next level.
    active_below: Fenwick,
}

/// The `k`-th smallest value among the active points of a position range.
pub(crate) struct WaveletMatrix {
    levels: Vec<Level>,
}

impl WaveletMatrix {
    /// The matrix of the points with `values[p]` at position `p`, values
    /// distinct and below `values.len()`, with point `p` active when
    /// `active(p)`.
    pub(crate) fn new(values: &[u32], active: impl Fn(usize) -> bool) -> Self {
        let n = values.len();
        let bits = if n <= 1 {
            1
        } else {
            usize::BITS - (n - 1).leading_zeros()
        };
        // (value, original position) in the order of the current level.
        let mut order: Vec<(u32, u32)> = values
            .iter()
            .enumerate()
            .map(|(p, &v)| (v, u32::try_from(p).expect("position fits u32")))
            .collect();
        let mut levels = Vec::with_capacity(bits as usize);
        let mut next: Vec<(u32, u32)> = Vec::with_capacity(n);
        for level in 0..bits {
            let bit = bits - 1 - level;
            let zeros = RankBits::new(order.iter().map(|&(v, _)| (v >> bit) & 1 == 0));
            next.clear();
            next.extend(order.iter().filter(|(v, _)| (v >> bit) & 1 == 0));
            next.extend(order.iter().filter(|(v, _)| (v >> bit) & 1 == 1));
            let active_below =
                Fenwick::from_counts(next.iter().map(|&(_, p)| u32::from(active(p as usize))));
            levels.push(Level {
                zeros,
                active_below,
            });
            std::mem::swap(&mut order, &mut next);
        }
        Self { levels }
    }

    /// Where the point at `position` of a level goes in the next level.
    fn descend(level: &Level, position: usize) -> usize {
        let zeros_before = level.zeros.rank(position) as usize;
        if level.zeros.get(position) {
            zeros_before
        } else {
            level.zeros.total as usize + position - zeros_before
        }
    }

    /// Add `delta` (1 or -1) to the point at `position`.
    pub(crate) fn add(&mut self, position: usize, delta: i32) {
        let mut position = position;
        for level in &mut self.levels {
            position = Self::descend(level, position);
            level.active_below.add(position, delta);
        }
    }

    /// The value of the active point of rank `k` (0-based) in value order
    /// among those at positions `start..end`. `k` must be below their number.
    pub(crate) fn kth(&self, start: usize, end: usize, k: u32) -> u32 {
        let (mut start, mut end, mut k) = (start, end, k);
        let mut value = 0_u32;
        for level in &self.levels {
            let total_zeros = level.zeros.total as usize;
            let zeros_start = level.zeros.rank(start) as usize;
            let zeros_end = level.zeros.rank(end) as usize;
            let active_zeros = level.active_below.range(zeros_start, zeros_end);
            value <<= 1;
            if k < active_zeros {
                start = zeros_start;
                end = zeros_end;
            } else {
                k -= active_zeros;
                start = total_zeros + start - zeros_start;
                end = total_zeros + end - zeros_end;
                value |= 1;
            }
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) % bound as u64) as usize
        }
    }

    #[test]
    fn fenwick_sums_and_ranks_agree_with_counting() {
        let mut rng = Lcg(5);
        for n in [0_usize, 1, 2, 3, 7, 8, 9, 64, 100] {
            let mut flags: Vec<bool> = (0..n).map(|_| rng.next(2) == 1).collect();
            let mut tree = Fenwick::from_counts(flags.iter().map(|&f| u32::from(f)));
            for _ in 0..200 {
                if n > 0 {
                    let p = rng.next(n);
                    tree.add(p, if flags[p] { -1 } else { 1 });
                    flags[p] = !flags[p];
                }
                let start = if n == 0 { 0 } else { rng.next(n + 1) };
                let end = start
                    + if n == start {
                        0
                    } else {
                        rng.next(n - start + 1)
                    };
                let expected = flags[start..end].iter().filter(|&&f| f).count();
                assert_eq!(tree.range(start, end) as usize, expected);
                let active: Vec<usize> = (0..n).filter(|&p| flags[p]).collect();
                for (k, &p) in active.iter().enumerate() {
                    assert_eq!(tree.kth(u32::try_from(k).unwrap()), p);
                }
            }
        }
    }

    #[test]
    fn count_tree_agrees_with_one_fenwick_tree_per_kind() {
        let mut rng = Lcg(23);
        for kinds in [1_usize, 2, 4, 5, 8, 9] {
            for n in [0_usize, 1, 2, 5, 8, 33, 100] {
                let mut tree = CountTree::zeros(n, kinds);
                let mut reference: Vec<Fenwick> = (0..kinds).map(|_| Fenwick::zeros(n)).collect();
                let mut tallies: Vec<Tally> = vec![None; n];
                for _ in 0..200 {
                    if n > 0 {
                        let p = rng.next(n);
                        let new: Tally = match rng.next(4) {
                            0 => None,
                            1 => Some(None),
                            _ if kinds > 1 => Some(Some(1 + rng.next(kinds - 1))),
                            _ => Some(None),
                        };
                        tree.change(p, tallies[p], new);
                        for (kind, fenwick) in reference.iter_mut().enumerate() {
                            let d = counts_under(new, kind) - counts_under(tallies[p], kind);
                            fenwick.add(p, d);
                        }
                        tallies[p] = new;
                    }
                    let start = rng.next(n + 1);
                    let end = start + rng.next(n - start + 1);
                    let mut got = vec![0; kinds];
                    tree.range_into(start, end, &mut got);
                    let expected: Vec<u32> =
                        reference.iter().map(|f| f.range(start, end)).collect();
                    assert_eq!(got, expected, "kinds {kinds}, n {n}");
                }
            }
        }
    }

    #[test]
    fn rank_bits_count_as_a_prefix_sum_does() {
        let mut rng = Lcg(31);
        for n in [0_usize, 1, 63, 64, 65, 127, 128, 129, 1000] {
            for density in [0_usize, 1, 2, 3] {
                let bits: Vec<bool> = (0..n)
                    .map(|_| match density {
                        0 => false,
                        1 => rng.next(8) == 0,
                        2 => rng.next(2) == 0,
                        _ => true,
                    })
                    .collect();
                let ranks = RankBits::new(bits.iter().copied());
                let mut prefix = vec![0_u32];
                for &bit in &bits {
                    prefix.push(prefix[prefix.len() - 1] + u32::from(bit));
                }
                for (position, &expected) in prefix.iter().enumerate() {
                    assert_eq!(ranks.rank(position), expected, "n {n}, at {position}");
                }
                for (position, &bit) in bits.iter().enumerate() {
                    assert_eq!(ranks.get(position), bit);
                }
                assert_eq!(ranks.total, prefix[n]);
            }
        }
    }

    #[test]
    fn wavelet_order_statistics_agree_with_sorting() {
        let mut rng = Lcg(11);
        for n in [1_usize, 2, 3, 5, 8, 16, 17, 100, 257] {
            let mut values: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
            for i in (1..n).rev() {
                values.swap(i, rng.next(i + 1));
            }
            let mut flags: Vec<bool> = (0..n).map(|_| rng.next(3) > 0).collect();
            let mut matrix = WaveletMatrix::new(&values, |p| flags[p]);
            for _ in 0..300 {
                let p = rng.next(n);
                matrix.add(p, if flags[p] { -1 } else { 1 });
                flags[p] = !flags[p];
                let start = rng.next(n + 1);
                let end = start + rng.next(n - start + 1);
                let mut inside: Vec<u32> = (start..end)
                    .filter(|&q| flags[q])
                    .map(|q| values[q])
                    .collect();
                inside.sort_unstable();
                for (k, &v) in inside.iter().enumerate() {
                    assert_eq!(matrix.kth(start, end, u32::try_from(k).unwrap()), v);
                }
            }
        }
    }
}
