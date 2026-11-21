//! Exact nearest-neighbour search on a k-d tree, with deletion.
//!
//! Nearest neighbour matching on a vector distance asks, for each anchor, for
//! the `k` nearest candidates that are still available and that the
//! constraints admit. A scan answers that by computing the distance to every
//! candidate. This index answers it with the same result while computing the
//! distance to far fewer.
//!
//! # Algorithm
//!
//! The tree is the k-d tree of Friedman, Bentley and Finkel \[1\]. The root
//! holds every point. A node with more than [`LEAF`] points is split at the
//! median of the coordinate with the largest spread in the node (the first
//! such coordinate on a tie), points equal on that coordinate ordered by
//! their key, and each half becomes a child. A node stores the bounding box
//! of its points: the smallest and largest value of each coordinate among
//! them, and the number of its points that are still live.
//!
//! A search is depth-first branch and bound. It keeps the best `k` points
//! found so far, ordered by value and then by key. At a node it computes a
//! lower bound of the value of any point in the node's box, and it skips the
//! node when no point in it is live, when the bound is above the search
//! radius, or when it already holds `k` points and the bound is above the
//! value of the last. It visits the child with the smaller bound first, as
//! Arya and Mount \[2\] do for exact search, and tests the other child again
//! after the first returns. In a leaf each live point's value is computed and
//! the point is offered to the caller's admission test only when it would
//! enter the best `k`. A refused point is passed over and the search goes on,
//! so the result is the `k` best of the admitted points, whatever the
//! admission test refuses.
//!
//! Deletion marks a point dead and lowers the live count of every node above
//! it, so a subtree whose points are all gone is skipped in one step.
//!
//! # Why the result equals the scan's
//!
//! The values of points are computed as the scans compute them, coordinate by
//! coordinate in the same order, so they are the same numbers, and the tie
//! rule is the same (value, then key). Pruning never removes a point the
//! scan would keep, because the bound of a box is never above the computed
//! value of a point in it. That holds for the computed numbers, not only for
//! exact arithmetic: the gap between the query and a box on one coordinate is
//! `low - q` or `q - high` rounded, and a point in the box differs from the
//! query by at least as much in floating point too, because rounding is
//! monotone. Squares, products with a weight that is not negative and sums
//! taken in the same order keep that order, and so does the square root. A
//! node is skipped only when its bound is strictly above the value it is
//! compared with, so every point at a value equal to the last of the best `k`
//! is still examined and the tie rule decides.
//!
//! # Cost
//!
//! Building costs `O(d n log n)` work for `n` points in `d` coordinates: each
//! of the `log n` levels computes the boxes and the median selection of all
//! `n` points. The points are moved in place, a row of coordinates with its
//! key at a time, so that the points of a node are contiguous, and the
//! selection is a quickselect, linear in expectation, on the order of the coordinate and then the
//! key, which is total because keys are distinct. The two children of a node
//! with more than 2^15 points are built on separate rayon tasks over disjoint
//! halves of the rows, so the span is `O(d n)` plus the sequential top
//! levels. The tree does not depend on the thread count: the points of each
//! node are fixed by the order, and the rows of a leaf are sorted by key.
//! Nodes and boxes are stored by index as `u32`, a node's box beside the
//! others in one array, and the live flags and counts are kept apart from
//! the shape so that a copy of the tree for a search that removes points
//! copies only them.
//!
//! Deletion costs `O(log n)`. A search costs `O(d)` per node visited. For
//! points in general position in few dimensions, Friedman, Bentley and
//! Finkel give an expected `O(log n)` nodes per search for a fixed `k`; the
//! worst case, and the case where the admission test refuses most near
//! points, visits every node, `O(d n)`, as a scan does. The advantage shrinks
//! as `d` grows, which is why the callers fall back to the scan in high
//! dimension and for small sets of points, by the measured rule of
//! [`worth_a_tree`].
//!
//! # References
//!
//! 1. Friedman JH, Bentley JL, Finkel RA. An algorithm for finding best
//!    matches in logarithmic expected time. ACM Trans Math Softw.
//!    1977;3(3):209-26.
//! 2. Arya S, Mount DM. Algorithms for fast vector quantization. In:
//!    Proceedings of the Data Compression Conference. IEEE; 1993. p. 381-90.

use std::sync::Arc;

/// The most points a leaf holds.
pub(crate) const LEAF: usize = 8;

/// The fewest points in up to five coordinates searched on a tree.
const MIN_POINTS: usize = 1024;

/// Points with more coordinates than this are searched by a scan.
const MAX_DIMENSION: usize = 12;

/// Whether a set of `points` points in `dimension` coordinates is searched on
/// a tree rather than by a scan: when there are at least [`MIN_POINTS`] of
/// them, twice as many for every two coordinates beyond four, and at most
/// [`MAX_DIMENSION`] coordinates.
///
/// The rule comes from timing both on one thread, with one query per eight
/// points and each query's nearest point removed, the tree's build included,
/// on points whose coordinates are sums of three uniform draws. The tree
/// broke even with the scan at about 1,500 points in three to five
/// coordinates, 4,000 in eight, 8,000 in ten and 25,000 in twelve, and was
/// slower than the scan at every size up to 131,072 points in sixteen. It was
/// 55 times faster at 131,072 points in three coordinates and 10 times in
/// eight. The scan of a long pool runs on several threads and the tree on
/// one, which the rule does not credit to the scan.
pub(crate) const fn worth_a_tree(points: usize, dimension: usize) -> bool {
    if dimension == 0 || dimension > MAX_DIMENSION {
        return false;
    }
    points >= MIN_POINTS << (dimension.saturating_sub(4) / 2)
}

/// Marks a key that has no point in the tree, and a missing node.
const NONE: u32 = u32::MAX;

/// A subtree with more points than this is built with its two children on
/// separate tasks.
const PARALLEL_BUILD: usize = 1 << 15;

/// How the value of a point, the number searches rank by, is computed, and
/// a lower bound of it over a box.
pub(crate) trait Form {
    /// The value of the point, or `None` when it has none and is never a
    /// match.
    fn value(&self, query: &[f64], point: &[f64]) -> Option<f64>;

    /// A number not above the [`Self::value`] of any point whose
    /// coordinates lie between `low` and `high`, computed so that this holds
    /// for the rounded numbers. A NaN bound prunes nothing.
    fn bound(&self, query: &[f64], low: &[f64], high: &[f64]) -> f64;
}

/// The distance from the query to the nearest point of the box on one
/// coordinate, zero inside.
fn gap(q: f64, low: f64, high: f64) -> f64 {
    if q < low {
        low - q
    } else if q > high {
        q - high
    } else {
        0.0
    }
}

/// The Euclidean distance: the square root of the sum of squared
/// differences, summed from the first coordinate, and no value when the sum
/// is not finite. The difference is query minus point.
pub(crate) struct Euclidean;

impl Form for Euclidean {
    fn value(&self, query: &[f64], point: &[f64]) -> Option<f64> {
        let mut sum = 0.0_f64;
        for (q, p) in query.iter().zip(point) {
            let delta = q - p;
            sum += delta * delta;
        }
        sum.is_finite().then(|| sum.sqrt())
    }

    fn bound(&self, query: &[f64], low: &[f64], high: &[f64]) -> f64 {
        let mut sum = 0.0_f64;
        for ((q, l), h) in query.iter().zip(low).zip(high) {
            let g = gap(*q, *l, *h);
            sum += g * g;
        }
        sum.sqrt()
    }
}

/// A weighted sum of squared differences, `sum_k w_k (q_k - p_k)^2` summed
/// from the first coordinate, with weights that are finite and not negative.
/// It has a value for every point; the caller makes sure the sums are
/// finite.
pub(crate) struct WeightedSquares<'w>(pub &'w [f64]);

impl Form for WeightedSquares<'_> {
    fn value(&self, query: &[f64], point: &[f64]) -> Option<f64> {
        let mut sum = 0.0_f64;
        for ((q, p), w) in query.iter().zip(point).zip(self.0) {
            let delta = q - p;
            sum += w * (delta * delta);
        }
        Some(sum)
    }

    fn bound(&self, query: &[f64], low: &[f64], high: &[f64]) -> f64 {
        let mut sum = 0.0_f64;
        for (((q, l), h), w) in query.iter().zip(low).zip(high).zip(self.0) {
            let g = gap(*q, *l, *h);
            sum += w * (g * g);
        }
        sum
    }
}

/// One node: its points are the slots `start..end`. Indices are `u32`, with
/// [`NONE`] for a missing child or parent. The node's bounding box is stored
/// beside the nodes, at the same index, so a visit reads both from one area.
#[derive(Debug, Clone, Copy)]
struct Node {
    start: u32,
    end: u32,
    left: u32,
    right: u32,
    parent: u32,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.left == NONE
    }
}

/// Nodes and boxes of a (sub)tree under construction. The box of node `i`
/// is `boxes[2 * i * d..(2 * i + 1) * d]` for the lows and the next `d`
/// values for the highs.
#[derive(Default)]
struct Part {
    nodes: Vec<Node>,
    boxes: Vec<f64>,
}

/// The fixed shape of a tree: points, nodes and boxes. It is never changed
/// after the build, so searches and clones share it.
#[derive(Debug)]
struct Shape {
    dimension: usize,
    /// Coordinates by slot, `dimension` each.
    coords: Vec<f64>,
    /// The key of each slot.
    keys: Vec<u32>,
    /// The slot of each key, `NONE` for a key with no point.
    slot_of: Vec<u32>,
    /// The leaf holding each slot.
    leaf_of: Vec<u32>,
    nodes: Vec<Node>,
    /// For each node `d` lows then `d` highs.
    boxes: Vec<f64>,
}

/// Which points are live, and how many points of each node are.
#[derive(Debug, Clone)]
struct Live {
    alive: Vec<bool>,
    counts: Vec<u32>,
}

/// A k-d tree over points with a key each. See the module documentation.
///
/// Cloning shares the shape and copies only the live flags and counts.
#[derive(Debug, Clone)]
pub(crate) struct KdTree {
    shape: Arc<Shape>,
    live: Live,
}

/// The best points of a search, by value and then key.
type Best = Vec<(f64, usize)>;

fn before(left: (f64, usize), right: (f64, usize)) -> bool {
    match left.0.partial_cmp(&right.0) {
        Some(std::cmp::Ordering::Less) => true,
        Some(std::cmp::Ordering::Equal) => left.1 < right.1,
        _ => false,
    }
}

/// Whether row `i` comes before row `j` in the order of one coordinate and
/// then the key. Keys are distinct, so the order is total.
fn row_before(coords: &[f64], keys: &[u32], d: usize, axis: usize, i: usize, j: usize) -> bool {
    coords[i * d + axis]
        .total_cmp(&coords[j * d + axis])
        .then(keys[i].cmp(&keys[j]))
        .is_lt()
}

fn swap_rows(coords: &mut [f64], keys: &mut [u32], d: usize, i: usize, j: usize) {
    if i == j {
        return;
    }
    keys.swap(i, j);
    let (a, b) = (i.min(j), i.max(j));
    let (head, tail) = coords.split_at_mut(b * d);
    head[a * d..(a + 1) * d].swap_with_slice(&mut tail[..d]);
}

/// Rearrange the rows so that row `nth` is where it would be in the order
/// of `axis` and then key, with no later row before it and no earlier row
/// after it. Quickselect with a median-of-three pivot, expected linear. An
/// input that defeats the pivot rule cannot make it quadratic: after 64
/// partitions the rest of the range is sorted, which bounds the worst case
/// by a sort and leaves the same row at `nth`, since the order is total.
fn select_row(coords: &mut [f64], keys: &mut [u32], d: usize, axis: usize, nth: usize) {
    let mut lo = 0;
    let mut hi = keys.len() - 1;
    let mut rounds = 0_u32;
    while lo < hi {
        rounds += 1;
        if rounds > 64 {
            sort_rows_by_axis(
                &mut coords[lo * d..(hi + 1) * d],
                &mut keys[lo..=hi],
                d,
                axis,
            );
            return;
        }
        let mid = lo + (hi - lo) / 2;
        // Move the median of the three to `hi` as the pivot.
        if row_before(coords, keys, d, axis, mid, lo) {
            swap_rows(coords, keys, d, mid, lo);
        }
        if row_before(coords, keys, d, axis, hi, lo) {
            swap_rows(coords, keys, d, hi, lo);
        }
        if row_before(coords, keys, d, axis, mid, hi) {
            swap_rows(coords, keys, d, mid, hi);
        }
        let mut store = lo;
        for i in lo..hi {
            if row_before(coords, keys, d, axis, i, hi) {
                swap_rows(coords, keys, d, i, store);
                store += 1;
            }
        }
        swap_rows(coords, keys, d, store, hi);
        match store.cmp(&nth) {
            std::cmp::Ordering::Equal => return,
            std::cmp::Ordering::Greater => hi = store - 1,
            std::cmp::Ordering::Less => lo = store + 1,
        }
    }
}

/// Order the rows by `axis` and then key.
fn sort_rows_by_axis(coords: &mut [f64], keys: &mut [u32], d: usize, axis: usize) {
    let m = keys.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by(|&i, &j| {
        coords[i * d + axis]
            .total_cmp(&coords[j * d + axis])
            .then(keys[i].cmp(&keys[j]))
    });
    let new_coords: Vec<f64> = order
        .iter()
        .flat_map(|&i| coords[i * d..(i + 1) * d].iter().copied())
        .collect();
    let new_keys: Vec<u32> = order.iter().map(|&i| keys[i]).collect();
    coords.copy_from_slice(&new_coords);
    keys.copy_from_slice(&new_keys);
}

/// Order the rows by key.
fn sort_rows_by_key(coords: &mut [f64], keys: &mut [u32], d: usize) {
    let m = keys.len();
    if m < 2 {
        return;
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by_key(|&i| keys[i]);
    let new_coords: Vec<f64> = order
        .iter()
        .flat_map(|&i| coords[i * d..(i + 1) * d].iter().copied())
        .collect();
    let new_keys: Vec<u32> = order.iter().map(|&i| keys[i]).collect();
    coords.copy_from_slice(&new_coords);
    keys.copy_from_slice(&new_keys);
}

/// Build the node over these rows, whose first slot is `start`, and its
/// descendants into `part`, and return the node's index in `part`.
fn build(
    part: &mut Part,
    coords: &mut [f64],
    keys: &mut [u32],
    d: usize,
    start: usize,
    parent: u32,
) -> u32 {
    let m = keys.len();
    let index = u32::try_from(part.nodes.len()).expect("fewer than 2^32 nodes");
    let mut low = vec![f64::INFINITY; d];
    let mut high = vec![f64::NEG_INFINITY; d];
    for row in coords.chunks_exact(d) {
        for (j, value) in row.iter().enumerate() {
            low[j] = low[j].min(*value);
            high[j] = high[j].max(*value);
        }
    }
    let mut axis = 0;
    let mut spread = f64::NEG_INFINITY;
    for (j, (h, l)) in high.iter().zip(&low).enumerate() {
        if h - l > spread {
            spread = h - l;
            axis = j;
        }
    }
    part.boxes.extend_from_slice(&low);
    part.boxes.extend_from_slice(&high);
    part.nodes.push(Node {
        start: u32::try_from(start).expect("fewer than 2^32 points"),
        end: u32::try_from(start + m).expect("fewer than 2^32 points"),
        left: NONE,
        right: NONE,
        parent,
    });
    if m <= LEAF || spread <= 0.0 {
        sort_rows_by_key(coords, keys, d);
        return index;
    }
    let mid = m / 2;
    select_row(coords, keys, d, axis, mid);
    let (left_coords, right_coords) = coords.split_at_mut(mid * d);
    let (left_keys, right_keys) = keys.split_at_mut(mid);
    if m > PARALLEL_BUILD {
        let (mut left_part, mut right_part) = (Part::default(), Part::default());
        rayon::join(
            || build(&mut left_part, left_coords, left_keys, d, start, NONE),
            || {
                build(
                    &mut right_part,
                    right_coords,
                    right_keys,
                    d,
                    start + mid,
                    NONE,
                )
            },
        );
        let left = append(part, &left_part, index);
        let right = append(part, &right_part, index);
        part.nodes[index as usize].left = left;
        part.nodes[index as usize].right = right;
    } else {
        let left = build(part, left_coords, left_keys, d, start, index);
        let right = build(part, right_coords, right_keys, d, start + mid, index);
        part.nodes[index as usize].left = left;
        part.nodes[index as usize].right = right;
    }
    index
}

/// Append a built subtree to `part`, with its indices moved up and its
/// root's parent set, and return the root's new index.
fn append(part: &mut Part, sub: &Part, parent: u32) -> u32 {
    let base = u32::try_from(part.nodes.len()).expect("fewer than 2^32 nodes");
    let shift = |index: u32| if index == NONE { NONE } else { index + base };
    part.nodes.extend(sub.nodes.iter().map(|node| Node {
        start: node.start,
        end: node.end,
        left: shift(node.left),
        right: shift(node.right),
        parent: shift(node.parent),
    }));
    part.nodes[base as usize].parent = parent;
    part.boxes.extend_from_slice(&sub.boxes);
    base
}

impl KdTree {
    /// A tree over the points `coords`, `dimension` values each and every
    /// value finite, with the keys `keys`, distinct and below `key_space`.
    /// The points and the key space must number fewer than 2^32.
    pub(crate) fn new(dimension: usize, coords: &[f64], keys: &[usize], key_space: usize) -> Self {
        let n = keys.len();
        debug_assert_eq!(coords.len(), n * dimension);
        assert!(
            key_space < NONE as usize && n < NONE as usize,
            "a k-d tree holds fewer than 2^32 points"
        );
        let mut coords = coords.to_vec();
        let mut keys: Vec<u32> = keys
            .iter()
            .map(|&key| u32::try_from(key).expect("a key below the key space"))
            .collect();
        let mut part = Part::default();
        if n > 0 {
            build(&mut part, &mut coords, &mut keys, dimension, 0, NONE);
        }
        let mut slot_of = vec![NONE; key_space];
        for (slot, &key) in keys.iter().enumerate() {
            slot_of[key as usize] = u32::try_from(slot).expect("fewer than 2^32 points");
        }
        let mut leaf_of = vec![0; n];
        for (index, node) in part.nodes.iter().enumerate() {
            if node.is_leaf() {
                leaf_of[node.start as usize..node.end as usize]
                    .fill(u32::try_from(index).expect("fewer than 2^32 nodes"));
            }
        }
        let counts = part
            .nodes
            .iter()
            .map(|node| node.end - node.start)
            .collect();
        Self {
            shape: Arc::new(Shape {
                dimension,
                coords,
                keys,
                slot_of,
                leaf_of,
                nodes: part.nodes,
                boxes: part.boxes,
            }),
            live: Live {
                alive: vec![true; n],
                counts,
            },
        }
    }

    /// Mark the point with this key dead. A key with no point, or a point
    /// already dead, is left alone.
    pub(crate) fn remove(&mut self, key: usize) {
        let shape = &*self.shape;
        let Some(&slot) = shape.slot_of.get(key) else {
            return;
        };
        let slot = slot as usize;
        if slot == NONE as usize || !self.live.alive[slot] {
            return;
        }
        self.live.alive[slot] = false;
        let mut node = shape.leaf_of[slot];
        while node != NONE {
            self.live.counts[node as usize] -= 1;
            node = shape.nodes[node as usize].parent;
        }
    }

    /// The up to `k` live points with the smallest values that `admits`
    /// accepts, with value at most `radius` when one is given, ordered by
    /// value and then key, as (value, key).
    pub(crate) fn nearest<F: Form>(
        &self,
        form: &F,
        query: &[f64],
        k: usize,
        radius: Option<f64>,
        admits: &mut dyn FnMut(usize) -> bool,
    ) -> Vec<(f64, usize)> {
        let mut best: Best = Vec::with_capacity(k + 1);
        if k == 0 || self.shape.nodes.is_empty() {
            return best;
        }
        let mut search = Search {
            shape: &self.shape,
            live: &self.live,
            form,
            query,
            k,
            radius,
            admits,
            best: &mut best,
        };
        let bound = self.shape.bound_of(form, query, 0);
        search.visit(0, bound);
        best
    }

    /// The keys of every live point with a value at most `limit`, in no
    /// particular order.
    pub(crate) fn within<F: Form>(&self, form: &F, query: &[f64], limit: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let shape = &*self.shape;
        if shape.nodes.is_empty() {
            return out;
        }
        let d = shape.dimension;
        let mut stack = vec![0_u32];
        while let Some(node) = stack.pop() {
            let entry = &shape.nodes[node as usize];
            if self.live.counts[node as usize] == 0
                || shape.bound_of(form, query, node as usize) > limit
            {
                continue;
            }
            if entry.is_leaf() {
                for slot in entry.start as usize..entry.end as usize {
                    if !self.live.alive[slot] {
                        continue;
                    }
                    let point = &shape.coords[slot * d..(slot + 1) * d];
                    if form.value(query, point).is_some_and(|value| value <= limit) {
                        out.push(shape.keys[slot] as usize);
                    }
                }
            } else {
                stack.push(entry.right);
                stack.push(entry.left);
            }
        }
        out
    }
}

impl Shape {
    fn bound_of<F: Form>(&self, form: &F, query: &[f64], node: usize) -> f64 {
        let d = self.dimension;
        let at = 2 * d * node;
        form.bound(
            query,
            &self.boxes[at..at + d],
            &self.boxes[at + d..at + 2 * d],
        )
    }
}

/// One k-nearest search in progress.
struct Search<'s, F> {
    shape: &'s Shape,
    live: &'s Live,
    form: &'s F,
    query: &'s [f64],
    k: usize,
    radius: Option<f64>,
    admits: &'s mut dyn FnMut(usize) -> bool,
    best: &'s mut Best,
}

impl<F: Form> Search<'_, F> {
    /// Whether a node with this bound can hold a point that enters the best.
    fn open(&self, bound: f64) -> bool {
        if self.radius.is_some_and(|radius| bound > radius) {
            return false;
        }
        !(self.best.len() == self.k && bound > self.best[self.k - 1].0)
    }

    fn visit(&mut self, node: u32, bound: f64) {
        let shape = self.shape;
        let entry = &shape.nodes[node as usize];
        if self.live.counts[node as usize] == 0 || !self.open(bound) {
            return;
        }
        if !entry.is_leaf() {
            let (left, right) = (entry.left, entry.right);
            let left_bound = shape.bound_of(self.form, self.query, left as usize);
            let right_bound = shape.bound_of(self.form, self.query, right as usize);
            // A NaN bound sorts last and prunes nothing.
            let (first, first_bound, second, second_bound) = if right_bound < left_bound {
                (right, right_bound, left, left_bound)
            } else {
                (left, left_bound, right, right_bound)
            };
            self.visit(first, first_bound);
            self.visit(second, second_bound);
            return;
        }
        let d = shape.dimension;
        for slot in entry.start as usize..entry.end as usize {
            if !self.live.alive[slot] {
                continue;
            }
            let point = &shape.coords[slot * d..(slot + 1) * d];
            let Some(value) = self.form.value(self.query, point) else {
                continue;
            };
            if self.radius.is_some_and(|radius| value > radius) {
                continue;
            }
            let key = shape.keys[slot] as usize;
            if self.best.len() == self.k && !before((value, key), self.best[self.k - 1]) {
                continue;
            }
            if !(self.admits)(key) {
                continue;
            }
            let at = self
                .best
                .partition_point(|entry| before(*entry, (value, key)));
            self.best.insert(at, (value, key));
            self.best.truncate(self.k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator for test points.
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

    /// The selection puts the same row at `nth` as a full sort, with the
    /// rows before it no later and the rows after it no earlier, on inputs
    /// with heavy ties, already sorted, reversed, and organ-pipe shaped,
    /// the last two being the shapes that defeat a median-of-three pivot
    /// and send the selection to its sorting fallback.
    #[test]
    fn selection_agrees_with_a_sort_on_every_input_shape() {
        let d = 2;
        let mut mix = Mix(17);
        for n in [1_usize, 2, 3, 7, 64, 1000, 4097] {
            let shapes: Vec<Vec<f64>> = vec![
                (0..n)
                    .map(|_| f64::from(u32::try_from(mix.next() % 5).unwrap()))
                    .collect(),
                (0..n).map(|i| i as f64).collect(),
                (0..n).rev().map(|i| i as f64).collect(),
                (0..n).map(|i| (i.min(n - 1 - i)) as f64).collect(),
            ];
            for axis_values in shapes {
                for nth in [0, n / 3, n / 2, n - 1] {
                    let mut coords: Vec<f64> = axis_values.iter().flat_map(|&v| [v, -v]).collect();
                    let mut keys: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
                    let (mut sc, mut sk) = (coords.clone(), keys.clone());
                    sort_rows_by_axis(&mut sc, &mut sk, d, 0);
                    select_row(&mut coords, &mut keys, d, 0, nth);
                    assert_eq!(keys[nth], sk[nth], "n {n}, nth {nth}");
                    for i in 0..n {
                        let before = row_before(&coords, &keys, d, 0, i, nth);
                        assert_eq!(before, i < nth, "n {n}, nth {nth}, row {i}");
                    }
                }
            }
        }
    }

    /// The scan the tree must agree with.
    fn scan<F: Form>(
        form: &F,
        coords: &[f64],
        keys: &[usize],
        live: &[bool],
        d: usize,
        query: &[f64],
        k: usize,
        radius: Option<f64>,
        admits: &dyn Fn(usize) -> bool,
    ) -> Vec<(f64, usize)> {
        let mut all: Vec<(f64, usize)> = (0..keys.len())
            .filter(|&i| live[i] && admits(keys[i]))
            .filter_map(|i| {
                form.value(query, &coords[i * d..(i + 1) * d])
                    .map(|value| (value, keys[i]))
            })
            .filter(|(value, _)| radius.is_none_or(|r| *value <= r))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        all.truncate(k);
        all
    }

    /// The tree returns the scan's points, ties included, under deletion,
    /// a radius, an admission test that refuses a third of the keys, and
    /// weights, in one to six coordinates with many exactly equal values.
    #[test]
    fn search_equals_the_scan() {
        let mut mix = Mix(11);
        for d in 1..=6 {
            for &n in &[1_usize, 7, 40, 300] {
                let coords: Vec<f64> = (0..n * d).map(|_| (mix.next() % 5) as f64).collect();
                // Keys in a scrambled order, with gaps in the key space.
                let mut keys: Vec<usize> = (0..n).map(|i| 3 * i + 1).collect();
                for i in (1..n).rev() {
                    let j = usize::try_from(mix.next() % (i as u64 + 1)).expect("small");
                    keys.swap(i, j);
                }
                let mut tree = KdTree::new(d, &coords, &keys, 3 * n + 2);
                let mut live = vec![true; n];
                let weights: Vec<f64> = (0..d).map(|j| [1.0, 0.0, 2.5][j % 3]).collect();
                for round in 0..3 {
                    for _ in 0..20 {
                        let query: Vec<f64> = (0..d)
                            .map(|_| (mix.next() % 11) as f64 / 2.0 - 0.5)
                            .collect();
                        let k = 1 + usize::try_from(mix.next() % 4).expect("small");
                        let radius = (round == 1).then_some(1.5);
                        let refuse = |key: usize| key % 3 == 1 && key % 9 == 4;
                        let expected = scan(
                            &Euclidean,
                            &coords,
                            &keys,
                            &live,
                            d,
                            &query,
                            k,
                            radius,
                            &|key| !refuse(key),
                        );
                        let found =
                            tree.nearest(&Euclidean, &query, k, radius, &mut |key| !refuse(key));
                        assert_eq!(found, expected, "d {d}, n {n}");
                        let form = WeightedSquares(&weights);
                        let expected =
                            scan(&form, &coords, &keys, &live, d, &query, k, None, &|_| true);
                        assert_eq!(
                            tree.nearest(&form, &query, k, None, &mut |_| true),
                            expected
                        );
                        let limit = 2.0;
                        let mut inside = tree.within(&Euclidean, &query, limit);
                        inside.sort_unstable();
                        let mut expected: Vec<usize> = (0..n)
                            .filter(|&i| live[i])
                            .filter(|&i| {
                                Euclidean
                                    .value(&query, &coords[i * d..(i + 1) * d])
                                    .is_some_and(|v| v <= limit)
                            })
                            .map(|i| keys[i])
                            .collect();
                        expected.sort_unstable();
                        assert_eq!(inside, expected);
                    }
                    // Delete about a third of what is left.
                    for i in 0..n {
                        if live[i] && mix.next() % 3 == 0 {
                            live[i] = false;
                            tree.remove(keys[i]);
                        }
                    }
                }
            }
        }
    }

    /// A tree large enough to be built in parallel has the same points,
    /// nodes and boxes on 1, 2, 4 and 12 threads, and its searches equal the
    /// scan's.
    #[test]
    fn a_parallel_build_is_the_same_on_every_pool_and_searches_like_the_scan() {
        let d = 3;
        let n = 3 * PARALLEL_BUILD / 2 + 11;
        let mut mix = Mix(5);
        let coords: Vec<f64> = (0..n * d).map(|_| (mix.next() % 40) as f64 / 4.0).collect();
        let keys: Vec<usize> = (0..n).map(|i| (i * 7919) % n).collect();
        epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
            let tree = KdTree::new(d, &coords, &keys, n);
            let shape = &tree.shape;
            let mut out: Vec<f64> = shape.coords.clone();
            out.extend(shape.keys.iter().map(|&k| f64::from(k)));
            out.extend(&shape.boxes);
            for node in &shape.nodes {
                out.extend(
                    [node.start, node.end, node.left, node.right, node.parent].map(f64::from),
                );
            }
            out
        });
        let tree = KdTree::new(d, &coords, &keys, n);
        let live = vec![true; n];
        for _ in 0..30 {
            let query: Vec<f64> = (0..d)
                .map(|_| (mix.next() % 100) as f64 / 8.0 - 1.0)
                .collect();
            let expected = scan(
                &Euclidean,
                &coords,
                &keys,
                &live,
                d,
                &query,
                5,
                None,
                &|_| true,
            );
            assert_eq!(
                tree.nearest(&Euclidean, &query, 5, None, &mut |_| true),
                expected
            );
        }
    }

    /// A clone shares the shape: removals in the clone do not reach the
    /// original.
    #[test]
    fn a_clone_removes_independently() {
        let tree = KdTree::new(1, &[0.0, 1.0, 2.0], &[0, 1, 2], 3);
        let mut copy = tree.clone();
        copy.remove(0);
        let first = |t: &KdTree| t.nearest(&Euclidean, &[0.0], 1, None, &mut |_| true)[0].1;
        assert_eq!(first(&tree), 0);
        assert_eq!(first(&copy), 1);
        assert!(Arc::ptr_eq(&tree.shape, &copy.shape));
    }

    #[test]
    fn removing_a_missing_or_dead_key_changes_nothing() {
        let mut tree = KdTree::new(1, &[0.0, 1.0], &[0, 2], 3);
        tree.remove(1);
        tree.remove(7);
        tree.remove(2);
        tree.remove(2);
        assert_eq!(tree.live.counts[0], 1);
        assert_eq!(
            tree.nearest(&Euclidean, &[1.0], 2, None, &mut |_| true),
            vec![(1.0, 0)]
        );
    }
}
