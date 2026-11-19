//! Optimal-matching distances between sequences, and k-medoids over them.
//!
//! The optimal-matching distance of two sequences is the cost of the cheapest
//! alignment that turns one into the other by substituting a state for another,
//! at a cost from a substitution matrix, or inserting or deleting one, at a
//! fixed indel cost \[1\]. It does not factorise over positions, so a fit
//! cannot re-derive a distance from the sequences and has to read them all.
//! [`CondensedMatrix`] holds every pairwise distance, computed once, and the
//! fit is a method on it.
//!
//! # Algorithm
//!
//! For sequences `x` of length `m` and `y` of length `n`, with substitution
//! cost `s(a, b)` and indel cost `g`, the Needleman-Wunsch recursion \[2\] is
//!
//! ```text
//! D(0, 0) = 0
//! D(i, 0) = i * g,    D(0, j) = j * g
//! D(i, j) = min( D(i-1, j-1) + s(x_i, y_j),
//!                D(i-1, j)   + g,
//!                D(i, j-1)   + g )
//! d(x, y) = D(m, n)
//! ```
//!
//! Two rolling rows hold the table, since only the cost is wanted. The
//! arithmetic is in `f32`, in the order above. The distance is symmetric when
//! the substitution matrix is, and the triangle stores one value per pair.
//!
//! # The triangle
//!
//! The matrix is the strict upper triangle of the distance table in row-major
//! order, as `f32`. For `n` sequences it holds `n (n - 1) / 2` values, and the
//! distance of rows `i < j` is at `start(i) + j - i - 1`, where `start(i)` is
//! the sum of `n - r - 1` over the rows before `i`. Single precision halves
//! the memory against `f64`, and the values are small sums of costs, so
//! k-medoids, which takes minima, and the silhouette, which takes means, do not
//! compare two distances closer than an `f32` resolves. The triangle is built
//! in parallel over rows. Each row writes its own slice of one buffer, so the
//! triangle is held once.
//!
//! # Cluster distance sums
//!
//! The silhouette needs, for each sequence `i` and cluster `c`, the weighted
//! sum `sum_{j in c} w_j d(i, j)`. It is quadratic here, since every term is a
//! lookup. Each row's terms are added with the columns `j < i` in ascending `j`
//! first and then the columns `j > i` in ascending `j`, in `f64`, and no term
//! is fused with its multiplication. The pass is tiled over blocks of
//! 128 rows so that the `j < i` half, which is strided in the triangle, reads
//! each cache line once. The tiling changes which accumulator is touched when,
//! not the order of additions into any one accumulator, so the sums are the
//! same as the one-row-at-a-time loop to the bit. The medoid update reads only
//! the sum over a sequence's own cluster, which is the diagonal of the full
//! result bit for bit and is computed without the other columns.
//!
//! # k-medoids
//!
//! The fit is that described in the crate documentation, with the distance
//! read from the triangle. The caller's order of sequences, not the triangle's,
//! is the order of the seeding and of tie-breaks, and `offsets` maps one to the
//! other. The offsets may name fewer sequences than the triangle holds.
//!
//! # Edge cases
//!
//! - Fewer than two sequences give an empty triangle.
//! - Sequences of length 0 are at distance `g * length` of the other.
//! - An offset outside the triangle, a symbol outside the alphabet or a
//!   substitution matrix that is not `alphabet * alphabet` is refused.
//! - A cluster with no member has no medoid, and keeps its previous one.
//!
//! # References
//!
//! 1. Abbott A, Tsay A. Sequence analysis and optimal matching methods in
//!    sociology: review and prospect. Sociol Methods Res. 2000;29(1):3-33.
//!    doi:10.1177/0049124100029001001
//! 2. Needleman SB, Wunsch CD. A general method applicable to the search for
//!    similarities in the amino acid sequence of two proteins. J Mol Biol.
//!    1970;48(3):443-53. doi:10.1016/0022-2836(70)90057-4
//! 3. Studer M, Ritschard G. What matters in differences between life
//!    trajectories: a comparative review of sequence dissimilarity measures.
//!    J R Stat Soc Ser A. 2016;179(2):481-511. doi:10.1111/rssa.12125

#![expect(
    clippy::cast_possible_truncation,
    reason = "sequence indices are bounded by the u32 limit checked when a panel or triangle is built, symbol ranks by the alphabet, and cluster ranks by the number of labels"
)]

use epistat_rng::Stream;
use rayon::prelude::*;

use crate::silhouette::{rank_labels, weighted_average};
use crate::{Attempt, SequenceError, best_attempt};

/// Rows of the triangle one worker takes at a time in the cluster sums.
///
/// A 64-byte cache line of the triangle holds sixteen `f32`, and a tile this
/// size reads each line once and uses all of it. The tile's `TILE * k`
/// accumulators stay in the L1 cache for any `k` a typology uses.
const TILE: usize = 128;

/// Needleman-Wunsch over two rank-encoded sequences.
///
/// Two rolling rows hold the table, since the alignment itself is never wanted,
/// only its cost.
// The two `usize as f32` casts are gap counts, bounded by the sequence length.
// `f32` represents every integer exactly to 2^24, so clippy's precision warning
// describes a case that sequences shorter than 16 million states cannot reach.
fn align(left: &[u8], right: &[u8], substitution: &[f32], size: usize, indel: f32) -> f32 {
    let mut previous: Vec<f32> = (0..=right.len()).map(|c| c as f32 * indel).collect();
    let mut current: Vec<f32> = vec![0.0; right.len() + 1];
    for (row, &first) in left.iter().enumerate() {
        current[0] = (row + 1) as f32 * indel;
        let costs = &substitution[first as usize * size..(first as usize + 1) * size];
        for (column, &second) in right.iter().enumerate() {
            let diagonal = previous[column] + costs[second as usize];
            let above = previous[column + 1] + indel;
            let beside = current[column] + indel;
            current[column + 1] = diagonal.min(above).min(beside);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// All pairwise distances over one fixed list of sequences.
///
/// Stored as the strict upper triangle in row-major order, as single-precision
/// floats, which halves the size twice over against the full square in double
/// precision. It is immutable once built, so a fit can borrow it across
/// threads.
pub struct CondensedMatrix {
    /// Number of sequences, not of distances.
    size: usize,
    values: Vec<f32>,
    /// Where each row of the triangle starts, so the quadratic pass walks a
    /// contiguous slice for `j > i` instead of recomputing the offset per term.
    starts: Vec<usize>,
}

impl CondensedMatrix {
    fn new(size: usize, values: Vec<f32>) -> Self {
        let mut starts = Vec::with_capacity(size);
        let mut running = 0usize;
        for row in 0..size {
            starts.push(running);
            running += size - row - 1;
        }
        Self {
            size,
            values,
            starts,
        }
    }
}

impl CondensedMatrix {
    /// The number of sequences the triangle describes.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.size
    }

    /// The number of distances, which is `n * (n - 1) / 2`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the triangle holds no distance.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// One distance by its position in the triangle, or `None` when the
    /// position is not below [`len`](CondensedMatrix::len).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }

    /// Every distance, the strict upper triangle in row-major order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Every restart of a k-medoids fit over this triangle.
    ///
    /// The counterpart of [`positionwise::kmedoids`](crate::positionwise::kmedoids).
    /// It is a method rather than a free function because the distance does
    /// not factorise, so a fit must read the triangle.
    ///
    /// `offsets` maps the caller's position to this triangle's. The two orders
    /// need not agree, and the fit works in the caller's, because that is the
    /// order the seeding walks and the order a tie between two equally central
    /// members resolves in.
    ///
    /// Restart `r` seeds from `Stream::seeded(seed + 7919 r)` and the restart
    /// with the least objective is kept, the earliest on a tie. A `restarts`
    /// of 0 runs one.
    ///
    /// # Errors
    ///
    /// [`SequenceError::Input`] when the weights do not match the offsets, an
    /// offset is outside the triangle, or `k` is zero or more than the
    /// sequences.
    pub fn kmedoids(
        &self,
        offsets: &[u32],
        weights: &[f64],
        k: usize,
        seed: u64,
        restarts: usize,
        max_iterations: usize,
    ) -> Result<Attempt, SequenceError> {
        let count = offsets.len();
        if weights.len() != count {
            return Err(SequenceError::input(format!(
                "{count} offsets but {} weights; they must correspond one to one",
                weights.len()
            )));
        }
        if let Some(bad) = offsets.iter().find(|&&o| o as usize >= self.size) {
            return Err(SequenceError::input(format!(
                "offset {bad} is outside the {} sequences this matrix covers",
                self.size
            )));
        }
        if k == 0 || k > count {
            return Err(SequenceError::input(format!(
                "asked for {k} types from {count} distinct sequences"
            )));
        }

        let attempts: Vec<Attempt> = (0..restarts.max(1))
            .into_par_iter()
            .map(|attempt| {
                let mut stream =
                    Stream::seeded(seed.wrapping_add((attempt as u64).wrapping_mul(7919)));
                self.one_fit(offsets, weights, k, max_iterations, &mut stream)
            })
            .collect();
        Ok(best_attempt(attempts))
    }

    /// Weighted sum of distances from every sequence to every cluster.
    ///
    /// For each sequence `i` and cluster `c`, `sum_{j in c} w_j d(i, j)`. For
    /// the position-wise distances that sum factorises over positions and is
    /// linear in time. Here it is quadratic, and every term is a lookup into
    /// the triangle.
    ///
    /// Returns `n * k` sums, row-major by sequence. They are sums and not
    /// means, so the caller divides by whichever cluster weights it holds and
    /// there is one definition of the denominator.
    ///
    /// The terms of each row are added in the order described in the module
    /// explainer. Using a fused multiply-add here would give a slightly
    /// different number in the last bits, so none is used.
    ///
    /// # Errors
    ///
    /// [`SequenceError::Input`] when the weights or labels do not cover the
    /// triangle's sequences, or a label is not below `k`.
    pub fn cluster_distance_sums(
        &self,
        weights: &[f64],
        labels: &[u32],
        k: usize,
    ) -> Result<Vec<f64>, SequenceError> {
        let n = self.size;
        if weights.len() != n || labels.len() != n {
            return Err(SequenceError::input(format!(
                "{n} sequences but {} weights and {} labels; all three describe the \
                 same sequences",
                weights.len(),
                labels.len()
            )));
        }
        if let Some(bad) = labels.iter().find(|&&label| label as usize >= k) {
            return Err(SequenceError::input(format!(
                "cluster label {bad} is outside the {k} clusters declared"
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        // `n * k` values, not `n^2 / 2`, so the one copy out is megabytes.
        Ok(self.cluster_sums(weights, labels, k))
    }

    /// The weighted average silhouette width of a partition over this triangle.
    ///
    /// `offsets` maps the caller's position to this triangle's, as for
    /// [`kmedoids`](CondensedMatrix::kmedoids), and may name fewer sequences
    /// than the triangle covers. `labels` are the caller's cluster numbers,
    /// any integers, ranked in ascending order. The weighting is described in
    /// [`silhouette`](crate::silhouette). Returns 0.0 for fewer than two
    /// distinct labels or when no sequence can be scored.
    ///
    /// # Errors
    ///
    /// [`SequenceError::Input`] when the weights or labels do not match the
    /// offsets, or an offset is outside the triangle.
    pub fn silhouette(
        &self,
        offsets: &[u32],
        weights: &[f64],
        labels: &[i64],
    ) -> Result<f64, SequenceError> {
        let count = offsets.len();
        if weights.len() != count || labels.len() != count {
            return Err(SequenceError::input(format!(
                "{count} offsets but {} weights and {} labels; all three describe the \
                 same sequences",
                weights.len(),
                labels.len()
            )));
        }
        if let Some(bad) = offsets.iter().find(|&&o| o as usize >= self.size) {
            return Err(SequenceError::input(format!(
                "offset {bad} is outside the {} sequences this matrix covers",
                self.size
            )));
        }
        Ok(self.silhouette_by_offset(offsets, weights, labels))
    }

    /// The silhouette of [`silhouette`](CondensedMatrix::silhouette), unchecked.
    ///
    /// The sums come from the full pass over the triangle by offset. An offset
    /// the caller did not name keeps rank 0 and weight 0, so it adds exactly
    /// zero to cluster 0's sums and is never scored.
    fn silhouette_by_offset(&self, offsets: &[u32], weights: &[f64], labels: &[i64]) -> f64 {
        let (ranks, k) = rank_labels(labels);
        if k < 2 {
            return 0.0;
        }
        let mut rank_by_offset = vec![0u32; self.size];
        let mut weight_by_offset = vec![0.0f64; self.size];
        for ((&offset, &rank), &weight) in offsets.iter().zip(&ranks).zip(weights) {
            rank_by_offset[offset as usize] = rank;
            weight_by_offset[offset as usize] = weight;
        }
        let sums = self.cluster_sums(&weight_by_offset, &rank_by_offset, k);
        weighted_average(&sums, k, |index| offsets[index] as usize, weights, &ranks)
    }

    /// One distance, by the two sequences' offsets into the triangle.
    fn distance(&self, first: usize, second: usize) -> f64 {
        if first == second {
            return 0.0;
        }
        let (low, high) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        f64::from(self.values[self.starts[low] + high - low - 1])
    }

    /// Rank of the closest medoid and the distance to it.
    ///
    /// `offsets` maps the caller's position to this triangle's, because the two
    /// orders need not agree and the fit works in the caller's: that is the
    /// order the seeding walks and the order a medoid tie is broken in.
    fn nearest(&self, offsets: &[u32], index: usize, medoids: &[u32]) -> (usize, f64) {
        let mine = offsets[index] as usize;
        let mut best = (0usize, f64::INFINITY);
        for (rank, &medoid) in medoids.iter().enumerate() {
            let candidate = self.distance(mine, offsets[medoid as usize] as usize);
            if candidate < best.1 {
                best = (rank, candidate);
            }
        }
        best
    }

    /// `sum_{j in c} w_j d(i, j)` for every sequence and cluster, by offset.
    ///
    /// Both halves of each row are added in the order the triangle stores
    /// them, so the sums are the same bit for bit as the one-row-at-a-time loop
    /// and do not depend on the tiling or the thread count.
    ///
    /// # Why the rows are tiled
    ///
    /// A row of the triangle is `n - i - 1` values wide and starts at
    /// `starts[i]`, so the `j > i` half of row `i` is one contiguous slice. The
    /// `j < i` half is `values[starts[j] + i - j - 1]` for ascending `j`, which
    /// steps by `n - j - 2` and lands on a different cache line every time. For
    /// a fixed `j`, consecutive `i` are adjacent, so a tile of rows reads each
    /// of those lines once and uses all of it. The loop is written with `j`
    /// outer and `i` inner within the tile. Every row still accumulates its
    /// `j < i` terms in ascending `j` and then its `j > i` terms in ascending
    /// `j`, the same additions into each accumulator.
    ///
    /// A tile also balances the work. Tile `t` does about `TILE * t` reads down
    /// columns and `TILE * (n - t)` along rows, which is `TILE * n` wherever it
    /// sits.
    // Not `mul_add`: it rounds once where `d * w` then `+=` rounds twice, and
    // the sums are specified as the latter.
    fn cluster_sums(&self, weights: &[f64], labels: &[u32], k: usize) -> Vec<f64> {
        let n = self.size;
        let mut flat = vec![0.0f64; n * k];
        flat.par_chunks_mut(k * TILE)
            .enumerate()
            .for_each(|(tile, rows)| {
                let first = tile * TILE;
                let last = first + rows.len() / k;

                // `j < i`, ascending `j`. Contiguous in `i` for a fixed `j`,
                // which is the whole point of the tile; the writes scatter
                // instead, over `TILE * k` accumulators that stay in L1.
                for j in 0..last.saturating_sub(1) {
                    let (label, weight, base) = (labels[j] as usize, weights[j], self.starts[j]);
                    for i in first.max(j + 1)..last {
                        let d = f64::from(self.values[base + i - j - 1]);
                        rows[(i - first) * k + label] += d * weight;
                    }
                }

                // `j > i`, ascending `j`. Already contiguous, and unchanged.
                for i in first..last {
                    let sums = &mut rows[(i - first) * k..(i - first + 1) * k];
                    let row = &self.values[self.starts[i]..self.starts[i] + (n - i - 1)];
                    for (offset, &d) in row.iter().enumerate() {
                        let j = i + 1 + offset;
                        sums[labels[j] as usize] += f64::from(d) * weights[j];
                    }
                }
            });
        flat
    }

    /// `sum_{j in cluster(i)} w_j d(i, j)` for every sequence, by offset.
    ///
    /// The one column of the full pass that the medoid update reads. That
    /// update takes an argmin within a cluster, so the `k - 1` other columns
    /// are never looked at.
    ///
    /// This is bit for bit the diagonal of the full pass. An accumulator
    /// depends on the values added to it and their order, and dropping the
    /// terms that went to other clusters changes neither: still every `j < i`
    /// of the same cluster ascending, then every `j > i` of the same cluster
    /// ascending.
    ///
    /// The arithmetic is `sum_c |C_c|^2` against `n^2` for the full pass, a
    /// factor of `k` less. The members of a cluster are scattered through the
    /// triangle, so a cache line yields about `16 / k` useful values and most
    /// of the saving in arithmetic is given back, but the pass is still faster
    /// than the full one. Clusters are taken one at a time, each in parallel
    /// tiles, because they differ in size and the work is quadratic in it.
    fn own_cluster_sums(&self, weights: &[f64], labels: &[u32], k: usize) -> Vec<f64> {
        let mut members: Vec<Vec<u32>> = (0..k).map(|_| Vec::new()).collect();
        for (offset, &label) in labels.iter().enumerate() {
            members[label as usize].push(offset as u32);
        }

        let mut owned = vec![0.0f64; self.size];
        for group in &members {
            // One tile of members per worker, one cluster at a time.
            //
            // Clusters are not taken in parallel with each other: they differ
            // in size and the work is quadratic in it, so `k` tasks would leave
            // the largest cluster running alone. Nor are their tiles pooled into
            // one task list, which is the obvious answer to the barrier this
            // leaves between clusters and was tried: a cluster is about `n / k`
            // members, so at `k = 10` and 17,273 sequences that is fourteen
            // tiles over twelve threads and two of them run a second tile while
            // ten wait, ten times per update. Flattening removes exactly that
            // and measured SLOWER, 0.94 s of fit against 0.73 s, presumably
            // because the restarts above already keep every thread fed and what
            // the pooled list costs in locality it cannot win back in balance.
            let mut sums = vec![0.0f64; group.len()];
            sums.par_chunks_mut(TILE)
                .enumerate()
                .for_each(|(tile, slots)| {
                    let first = tile * TILE;
                    let last = first + slots.len();

                    // `j < i`, ascending `j`, as in `cluster_sums` and for the
                    // same reason: `i` inner keeps the reads moving forwards.
                    for (at, &member) in group[..last.saturating_sub(1)].iter().enumerate() {
                        let j = member as usize;
                        let (weight, base) = (weights[j], self.starts[j]);
                        for position in first.max(at + 1)..last {
                            let i = group[position] as usize;
                            slots[position - first] +=
                                f64::from(self.values[base + i - j - 1]) * weight;
                        }
                    }

                    // `j > i`, ascending `j`. Row-major here, because for a
                    // fixed `j` the `j > i` offsets stride by a whole row.
                    for position in first..last {
                        let i = group[position] as usize;
                        let base = self.starts[i];
                        let mut total = slots[position - first];
                        for &member in &group[position + 1..] {
                            let j = member as usize;
                            total += f64::from(self.values[base + j - i - 1]) * weights[j];
                        }
                        slots[position - first] = total;
                    }
                });
            for (&offset, &value) in group.iter().zip(&sums) {
                owned[offset as usize] = value;
            }
        }
        owned
    }

    /// The most central member of every cluster, as caller positions.
    ///
    /// The argmin is strictly less, scanning the caller's order, so two
    /// exactly equally central members resolve to the earlier one. A cluster
    /// with no member is `None`.
    fn medoids_from(&self, offsets: &[u32], weights: &[f64], labels: &[u32]) -> Vec<Option<u32>> {
        let k = labels
            .iter()
            .map(|&label| label as usize + 1)
            .max()
            .unwrap_or(0);
        // Offsets the caller did not name keep label 0 and weight 0, so they
        // join cluster 0 and contribute `d * 0.0` to it. Exactly zero, added to
        // a running sum that is never negative zero, so they change no total.
        let mut by_offset = vec![0u32; self.size];
        let mut weight_by_offset = vec![0.0f64; self.size];
        for (index, &offset) in offsets.iter().enumerate() {
            by_offset[offset as usize] = labels[index];
            weight_by_offset[offset as usize] = weights[index];
        }
        let owned = self.own_cluster_sums(&weight_by_offset, &by_offset, k);

        let mut best: Vec<Option<u32>> = vec![None; k];
        let mut best_cost: Vec<f64> = vec![f64::INFINITY; k];
        for (index, (&offset, &label)) in offsets.iter().zip(labels).enumerate() {
            let cost = owned[offset as usize];
            if best[label as usize].is_none() || cost < best_cost[label as usize] {
                best[label as usize] = Some(index as u32);
                best_cost[label as usize] = cost;
            }
        }
        best
    }
}

impl CondensedMatrix {
    /// One seeding and the alternating loop that follows it, over the triangle.
    fn one_fit(
        &self,
        offsets: &[u32],
        weights: &[f64],
        k: usize,
        max_iterations: usize,
        stream: &mut Stream,
    ) -> Attempt {
        let count = offsets.len();

        // k-means++ over the triangle, as in the position-wise fit: a
        // degenerate step pads in index order and takes no draw, and the total
        // and the running sum accumulate left to right.
        let mut medoids: Vec<u32> = vec![0];
        let mut chosen = vec![false; count];
        chosen[0] = true;
        while medoids.len() < k {
            let scores: Vec<f64> = (0..count)
                .into_par_iter()
                .map(|index| {
                    let distance = self.nearest(offsets, index, &medoids).1;
                    distance * distance * weights[index]
                })
                .collect();
            let total: f64 = scores.iter().sum();
            if total <= 0.0 {
                match chosen.iter().position(|taken| !taken) {
                    Some(next) => {
                        medoids.push(next as u32);
                        chosen[next] = true;
                    }
                    None => break,
                }
                continue;
            }
            let target = stream.uniform() * total;
            let mut running = 0.0;
            for (index, contribution) in scores.iter().enumerate() {
                running += contribution;
                if running >= target {
                    medoids.push(index as u32);
                    chosen[index] = true;
                    break;
                }
            }
        }

        let mut iterations = 0usize;
        let mut converged = false;
        for iteration in 1..=max_iterations {
            iterations = iteration;
            let labels: Vec<u32> = (0..count)
                .into_par_iter()
                .map(|index| self.nearest(offsets, index, &medoids).0 as u32)
                .collect();
            let mut moved = false;
            for (cluster, choice) in self
                .medoids_from(offsets, weights, &labels)
                .iter()
                .enumerate()
            {
                if let Some(best) = choice.filter(|&best| best != medoids[cluster]) {
                    medoids[cluster] = best;
                    moved = true;
                }
            }
            if !moved {
                converged = true;
                break;
            }
        }

        let assigned: Vec<(u32, f64)> = (0..count)
            .into_par_iter()
            .map(|index| {
                let (rank, distance) = self.nearest(offsets, index, &medoids);
                (rank as u32, distance)
            })
            .collect();
        let objective = assigned
            .iter()
            .zip(weights.iter())
            .fold(0.0f64, |total, ((_, distance), weight)| {
                total + distance * weight
            });
        let labels: Vec<u32> = assigned.into_iter().map(|(rank, _)| rank).collect();
        Attempt {
            medoids,
            labels,
            iterations,
            converged,
            objective,
        }
    }
}

/// Every pairwise distance between sequences of one alphabet.
///
/// `encoded` is every sequence's symbol ranks, concatenated, and `lengths`
/// how many belong to each, so sequences of different lengths are one slice.
/// `substitution` is the dense `alphabet_size * alphabet_size` cost matrix,
/// row-major, and `indel` the cost of an insertion or deletion.
///
/// Parallel over rows of the triangle. Rows are unequal in length, the first
/// being `n - 1` pairs and the last one, so the work is chunked by rayon's
/// dynamic scheduler. Each row is handed a disjoint sub-slice of the one
/// output buffer, so the threads write into their final positions and the
/// triangle is held once.
///
/// # Errors
///
/// [`SequenceError::Input`] when the substitution matrix is not dense and
/// square, the lengths do not add up to the symbols given, or a symbol is
/// outside the alphabet.
pub fn condensed(
    encoded: &[u8],
    lengths: &[u32],
    substitution: &[f32],
    alphabet_size: usize,
    indel: f32,
) -> Result<CondensedMatrix, SequenceError> {
    if substitution.len() != alphabet_size * alphabet_size {
        return Err(SequenceError::input(format!(
            "substitution matrix has {} entries for an alphabet of {}; it must be \
             dense and square, or a symbol pair would silently take another pair's cost",
            substitution.len(),
            alphabet_size
        )));
    }
    let declared: usize = lengths.iter().map(|&length| length as usize).sum();
    if declared != encoded.len() {
        return Err(SequenceError::input(format!(
            "the lengths add up to {declared} symbols but {} were sent; each \
             sequence would be read from the wrong place",
            encoded.len()
        )));
    }
    if let Some(bad) = encoded.iter().find(|&&rank| rank as usize >= alphabet_size) {
        return Err(SequenceError::input(format!(
            "symbol rank {bad} is outside an alphabet of {alphabet_size}; the caller \
             encoded against a different alphabet from the one it sent"
        )));
    }
    let mut sequences: Vec<&[u8]> = Vec::with_capacity(lengths.len());
    let mut rest = encoded;
    for &length in lengths {
        let (head, tail) = rest.split_at(length as usize);
        sequences.push(head);
        rest = tail;
    }
    let n = sequences.len();
    if n < 2 {
        return Ok(CondensedMatrix::new(n, Vec::new()));
    }

    let condensed = {
        let mut condensed = vec![0.0f32; n * (n - 1) / 2];

        // One disjoint sub-slice per row of the triangle, in row order, so a row
        // index is all a worker needs to know where it writes. `split_at_mut`
        // rather than arithmetic on a pointer: the compiler then owns the proof
        // that no two rows overlap, which is the only thing making the parallel
        // write sound.
        let mut rows: Vec<&mut [f32]> = Vec::with_capacity(n - 1);
        let mut rest: &mut [f32] = &mut condensed;
        for first in 0..n - 1 {
            let (head, tail) = rest.split_at_mut(n - first - 1);
            rows.push(head);
            rest = tail;
        }

        rows.into_par_iter().enumerate().for_each(|(first, row)| {
            let left = sequences[first];
            for (offset, slot) in row.iter_mut().enumerate() {
                *slot = align(
                    left,
                    sequences[first + 1 + offset],
                    substitution,
                    alphabet_size,
                    indel,
                );
            }
        });
        condensed
    };

    Ok(CondensedMatrix::new(n, condensed))
}

#[cfg(test)]
mod tests {
    use super::{CondensedMatrix, TILE, align, condensed};
    use epistat_rng::Stream;

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// A triangle of distinct values, so a misplaced index cannot pass by luck.
    fn triangle(n: usize) -> CondensedMatrix {
        let mut stream = Stream::seeded(20_260_810);
        let values = (0..n * (n - 1) / 2)
            .map(|_| stream.uniform() as f32)
            .collect();
        CondensedMatrix::new(n, values)
    }

    /// `cluster_sums` as the obvious loop nest: one row at a time, both halves
    /// in the order the triangle stores them. The tiled version has to agree
    /// with this to the bit.
    fn per_row(matrix: &CondensedMatrix, weights: &[f64], labels: &[u32], k: usize) -> Vec<f64> {
        let n = matrix.size;
        let mut flat = vec![0.0f64; n * k];
        for i in 0..n {
            let sums = &mut flat[i * k..(i + 1) * k];
            for j in 0..i {
                let d = f64::from(matrix.values[matrix.starts[j] + i - j - 1]);
                sums[labels[j] as usize] += d * weights[j];
            }
            for offset in 0..n - i - 1 {
                let j = i + 1 + offset;
                let d = f64::from(matrix.values[matrix.starts[i] + offset]);
                sums[labels[j] as usize] += d * weights[j];
            }
        }
        flat
    }

    fn fixture(n: usize, k: usize) -> (CondensedMatrix, Vec<f64>, Vec<u32>) {
        let matrix = triangle(n);
        let mut stream = Stream::seeded(7);
        let weights = (0..n).map(|_| 1.0 + stream.uniform()).collect();
        let labels = (0..n).map(|_| stream.below(k) as u32).collect();
        (matrix, weights, labels)
    }

    #[test]
    fn tiling_the_rows_changes_no_bit() {
        // Sizes either side of a tile boundary, and one that is not a multiple
        // of it, because the last tile is short and is where an off-by-one goes.
        for n in [2, 3, 17, TILE, TILE + 1, 2 * TILE + 5, 1000] {
            let k = 4.min(n);
            let (matrix, weights, labels) = fixture(n, k);
            assert_eq!(
                bits(&matrix.cluster_sums(&weights, &labels, k)),
                bits(&per_row(&matrix, &weights, &labels, k)),
                "tiled cluster_sums differs from the per-row nest at n = {n}"
            );
        }
    }

    #[test]
    fn the_own_cluster_pass_is_the_diagonal_of_the_full_one() {
        for n in [2, 17, TILE + 1, 2 * TILE + 5, 1000] {
            for k in [1usize, 3, 10] {
                if k > n {
                    continue;
                }
                let (matrix, weights, labels) = fixture(n, k);
                let full = matrix.cluster_sums(&weights, &labels, k);
                let owned = matrix.own_cluster_sums(&weights, &labels, k);
                for (i, &label) in labels.iter().enumerate() {
                    assert_eq!(
                        owned[i].to_bits(),
                        full[i * k + label as usize].to_bits(),
                        "own-cluster sum differs at sequence {i} of {n}, k = {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_empty_cluster_leaves_its_members_alone() {
        // Label 2 is unused, and every sequence still gets its own-cluster sum.
        let n = 300;
        let matrix = triangle(n);
        let mut stream = Stream::seeded(1);
        let weights: Vec<f64> = (0..n).map(|_| 1.0 + stream.uniform()).collect();
        let labels: Vec<u32> = (0..n).map(|i| u32::from(i % 2 != 0)).collect();
        let owned = matrix.own_cluster_sums(&weights, &labels, 3);
        assert_eq!(owned.len(), n);
        assert!(owned.iter().all(|&value| value > 0.0));
    }

    /// Twelve sequences with weights, and the partition `wcKMedoids` of
    /// WeightedCluster returned for them.
    const SEQUENCES: [&str; 12] = [
        "GGGEEEEE", "GGVVEEEE", "GGGGTTTE", "NNNNNNNN", "GGGNNNNN", "GGEEEEEE", "GGVVVEEE",
        "CGGGEEEE", "GNNNEEEE", "GGGGGTTT", "NNNGGGGG", "EEEEEEEE",
    ];
    const WEIGHTS: [f64; 12] = [
        4000.0, 3500.0, 40.0, 30.0, 8.0, 6.0, 5.0, 3.0, 2.0, 1.0, 1.0, 1.0,
    ];
    const PARTITION: [i64; 12] = [1, 2, 1, 4, 4, 1, 2, 1, 1, 1, 4, 1];

    /// A triangle of Hamming distances over `sequences`, in their order.
    // `usize as f32` on a count of at most eight positions.
    fn hamming_triangle(sequences: &[&str]) -> CondensedMatrix {
        let n = sequences.len();
        let mut values = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for (i, left) in sequences.iter().enumerate() {
            for right in &sequences[i + 1..] {
                let differing = left
                    .bytes()
                    .zip(right.bytes())
                    .filter(|(a, b)| a != b)
                    .count();
                values.push(differing as f32);
            }
        }
        CondensedMatrix::new(n, values)
    }

    #[test]
    fn the_silhouette_is_weightedclusters_asww() {
        // `wcClusterQuality(d, cl, weights = w)$stats[["ASWw"]]`, WeightedCluster
        // 2.0, TraMineR 2.2.13, R 4.6.1, on Hamming distances.
        let matrix = hamming_triangle(&SEQUENCES);
        let offsets: Vec<u32> = (0..12).collect();
        let value = matrix.silhouette_by_offset(&offsets, &WEIGHTS, &PARTITION);
        assert!((value - 0.981_074_279_353).abs() < 1e-9, "{value}");
    }

    #[test]
    fn the_silhouette_reads_through_offsets_into_a_wider_triangle() {
        // The triangle in reverse order with two sequences no one asks about.
        // Offsets map the caller's order onto it, and the unnamed rows must
        // change nothing.
        let mut wide: Vec<&str> = SEQUENCES.iter().rev().copied().collect();
        wide.extend(["VVVVVVVV", "TTTTTTTT"]);
        let matrix = hamming_triangle(&wide);
        let offsets: Vec<u32> = (0..12).map(|i| 11 - i).collect();
        let value = matrix.silhouette_by_offset(&offsets, &WEIGHTS, &PARTITION);
        assert!((value - 0.981_074_279_353).abs() < 1e-9, "{value}");
    }

    #[test]
    fn one_label_has_no_silhouette() {
        let matrix = hamming_triangle(&SEQUENCES);
        let offsets: Vec<u32> = (0..12).collect();
        assert!(
            matrix
                .silhouette_by_offset(&offsets, &WEIGHTS, &[0; 12])
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn a_delay_costs_two_indels_per_position() {
        // `seqdist(method = "OM", sm = CONSTANT 2, indel = 1)` in TraMineR
        // 2.2.13: an on-time path is 2 from itself delayed a year and 4 from
        // itself delayed two, and 12 from an unrelated path.
        let rank = |s: &str| -> Vec<u8> {
            s.bytes()
                .map(|b| match b {
                    b'G' => 0,
                    b'V' => 1,
                    b'E' => 2,
                    b'T' => 3,
                    _ => 4,
                })
                .collect()
        };
        let mut substitution = vec![2.0f32; 25];
        for symbol in 0..5 {
            substitution[symbol * 5 + symbol] = 0.0;
        }
        let on_time = rank("GGVVEETT");
        let distance = |other: &str| align(&on_time, &rank(other), &substitution, 5, 1.0);
        assert_eq!(distance("GGGVVEET").to_bits(), 2.0_f32.to_bits());
        assert_eq!(distance("GGGGVVEE").to_bits(), 4.0_f32.to_bits());
        assert_eq!(distance("GGNNNNNN").to_bits(), 12.0_f32.to_bits());
        assert_eq!(distance("GGVVEETT").to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn the_triangle_holds_each_pair_once_in_row_order() {
        // Sequences "0", "01" and "011" under unit costs and indel 1.
        let encoded = [0u8, 0, 1, 0, 1, 1];
        let substitution = [0.0f32, 1.0, 1.0, 0.0];
        let matrix = condensed(&encoded, &[1, 2, 3], &substitution, 2, 1.0).unwrap();
        assert_eq!((matrix.n(), matrix.len()), (3, 3));
        assert!(!matrix.is_empty());
        let values: Vec<u32> = matrix
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected: Vec<u32> = [1.0f32, 2.0, 1.0]
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(values, expected);
        assert_eq!(matrix.get(2).map(f32::to_bits), Some(1.0f32.to_bits()));
        assert!(matrix.get(3).is_none());
        let single = condensed(&[0], &[1], &substitution, 2, 1.0).unwrap();
        assert!(single.is_empty());
    }

    #[test]
    fn invalid_input_is_refused() {
        let substitution = [0.0f32, 1.0, 1.0, 0.0];
        assert!(condensed(&[0, 1], &[2], &substitution[..3], 2, 1.0).is_err());
        assert!(condensed(&[0, 1], &[1], &substitution, 2, 1.0).is_err());
        assert!(condensed(&[0, 2], &[2], &substitution, 2, 1.0).is_err());
        let matrix = hamming_triangle(&SEQUENCES);
        let offsets: Vec<u32> = (0..12).collect();
        assert!(matrix.kmedoids(&offsets, &WEIGHTS, 0, 1, 1, 5).is_err());
        assert!(
            matrix
                .kmedoids(&offsets, &WEIGHTS[1..], 2, 1, 1, 5)
                .is_err()
        );
        assert!(matrix.kmedoids(&[0, 99], &[1.0, 1.0], 1, 1, 1, 5).is_err());
        assert!(matrix.silhouette(&offsets, &WEIGHTS, &[0; 11]).is_err());
        assert!(matrix.cluster_distance_sums(&WEIGHTS, &[5; 12], 2).is_err());
    }

    #[test]
    fn the_weighted_fit_separates_the_reference_groups() {
        let matrix = hamming_triangle(&SEQUENCES);
        let offsets: Vec<u32> = (0..12).collect();
        let fit = matrix.kmedoids(&offsets, &WEIGHTS, 3, 1, 6, 30).unwrap();
        assert!(fit.converged);
        // The two heavy sequences differ at two positions and belong to
        // different types, and the all-N sequences share one.
        assert_ne!(fit.labels[0], fit.labels[1]);
        assert_eq!(fit.labels[3], fit.labels[4]);
        let again = matrix.kmedoids(&offsets, &WEIGHTS, 3, 1, 6, 30).unwrap();
        assert_eq!(fit, again);
    }
}
