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
//! range at one level to the two ranges its points occupy at the next. A
//! Fenwick tree on each level counts the active points of that level's order.
//! A query descends one level per bit: it counts the active points of the
//! range whose bit is 0, goes to the zeros if `k` is below that count, and to
//! the ones with `k` reduced by it otherwise.
//!
//! # Complexity
//!
//! For `n` points and `L = ceil(log2 n)` levels: building is `O(n L)` time and
//! `2 (n + 1) L` 32-bit words; switching a point on or off and the `k`-th
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

/// One level of a [`WaveletMatrix`].
struct Level {
    /// `zeros[i]`: points among the first `i` of this level's order whose bit
    /// is 0.
    zeros: Vec<u32>,
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
            let mut zeros = Vec::with_capacity(n + 1);
            let mut count = 0_u32;
            zeros.push(0);
            for &(v, _) in &order {
                if (v >> bit) & 1 == 0 {
                    count += 1;
                }
                zeros.push(count);
            }
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
        let zeros_before = level.zeros[position] as usize;
        if level.zeros[position + 1] as usize > zeros_before {
            zeros_before
        } else {
            *level.zeros.last().expect("one entry per prefix") as usize + position - zeros_before
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
            let total_zeros = *level.zeros.last().expect("one entry per prefix") as usize;
            let zeros_start = level.zeros[start] as usize;
            let zeros_end = level.zeros[end] as usize;
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
