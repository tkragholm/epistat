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
//! Building costs `O(d n log n)` for `n` points in `d` coordinates: each of
//! the `log n` levels computes the boxes and medians of all `n` points.
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

/// Marks a key that has no point in the tree.
const NO_SLOT: usize = usize::MAX;

/// Marks the root's parent.
const NO_NODE: usize = usize::MAX;

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

#[derive(Debug, Clone)]
struct Node {
    /// The node's points are the slots `start..end`.
    start: usize,
    end: usize,
    /// The two children, or `None` for a leaf.
    children: Option<(usize, usize)>,
    parent: usize,
    live: usize,
}

/// A k-d tree over points with a key each. See the module documentation.
#[derive(Debug, Clone)]
pub(crate) struct KdTree {
    dimension: usize,
    /// Coordinates by slot, `dimension` each.
    coords: Vec<f64>,
    /// The key of each slot.
    keys: Vec<usize>,
    live: Vec<bool>,
    /// The slot of each key, `NO_SLOT` for a key with no point.
    slot_of: Vec<usize>,
    /// The leaf holding each slot.
    leaf_of: Vec<usize>,
    nodes: Vec<Node>,
    /// The bounding box of each node, `dimension` values each.
    low: Vec<f64>,
    high: Vec<f64>,
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

impl KdTree {
    /// A tree over the points `coords`, `dimension` values each and every
    /// value finite, with the keys `keys`, distinct and below `key_space`.
    pub(crate) fn new(dimension: usize, coords: &[f64], keys: &[usize], key_space: usize) -> Self {
        let n = keys.len();
        debug_assert_eq!(coords.len(), n * dimension);
        let mut order: Vec<usize> = (0..n).collect();
        let mut tree = Self {
            dimension,
            coords: Vec::with_capacity(n * dimension),
            keys: Vec::with_capacity(n),
            live: vec![true; n],
            slot_of: vec![NO_SLOT; key_space],
            leaf_of: vec![0; n],
            nodes: Vec::with_capacity(2 * n / LEAF + 1),
            low: Vec::new(),
            high: Vec::new(),
        };
        if n > 0 {
            tree.split(coords, keys, &mut order, 0, n, NO_NODE);
        }
        for (slot, &row) in order.iter().enumerate() {
            tree.coords
                .extend_from_slice(&coords[row * dimension..(row + 1) * dimension]);
            tree.keys.push(keys[row]);
            tree.slot_of[keys[row]] = slot;
        }
        tree
    }

    /// Build the node over `order[start..end]` and its descendants, and
    /// return its index.
    fn split(
        &mut self,
        coords: &[f64],
        keys: &[usize],
        order: &mut [usize],
        start: usize,
        end: usize,
        parent: usize,
    ) -> usize {
        let d = self.dimension;
        let index = self.nodes.len();
        let mut low = vec![f64::INFINITY; d];
        let mut high = vec![f64::NEG_INFINITY; d];
        for &row in &order[start..end] {
            for (j, value) in coords[row * d..(row + 1) * d].iter().enumerate() {
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
        self.low.extend_from_slice(&low);
        self.high.extend_from_slice(&high);
        self.nodes.push(Node {
            start,
            end,
            children: None,
            parent,
            live: end - start,
        });
        if end - start <= LEAF || spread <= 0.0 {
            self.leaf_of[start..end].fill(index);
            return index;
        }
        let mid = start + (end - start) / 2;
        order[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            coords[a * d + axis]
                .total_cmp(&coords[b * d + axis])
                .then(keys[a].cmp(&keys[b]))
        });
        let left = self.split(coords, keys, order, start, mid, index);
        let right = self.split(coords, keys, order, mid, end, index);
        self.nodes[index].children = Some((left, right));
        index
    }

    /// Mark the point with this key dead. A key with no point, or a point
    /// already dead, is left alone.
    pub(crate) fn remove(&mut self, key: usize) {
        let Some(&slot) = self.slot_of.get(key) else {
            return;
        };
        if slot == NO_SLOT || !self.live[slot] {
            return;
        }
        self.live[slot] = false;
        let mut node = self.leaf_of[slot];
        while node != NO_NODE {
            self.nodes[node].live -= 1;
            node = self.nodes[node].parent;
        }
    }

    fn bound_of<F: Form>(&self, form: &F, query: &[f64], node: usize) -> f64 {
        let d = self.dimension;
        form.bound(
            query,
            &self.low[node * d..(node + 1) * d],
            &self.high[node * d..(node + 1) * d],
        )
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
        if k == 0 || self.nodes.is_empty() {
            return best;
        }
        let mut search = Search {
            tree: self,
            form,
            query,
            k,
            radius,
            admits,
            best: &mut best,
        };
        let bound = self.bound_of(form, query, 0);
        search.visit(0, bound);
        best
    }

    /// The keys of every live point with a value at most `limit`, in no
    /// particular order.
    pub(crate) fn within<F: Form>(&self, form: &F, query: &[f64], limit: f64) -> Vec<usize> {
        let mut out = Vec::new();
        if self.nodes.is_empty() {
            return out;
        }
        let d = self.dimension;
        let mut stack = vec![0_usize];
        while let Some(node) = stack.pop() {
            let entry = &self.nodes[node];
            if entry.live == 0 || self.bound_of(form, query, node) > limit {
                continue;
            }
            match entry.children {
                Some((left, right)) => {
                    stack.push(right);
                    stack.push(left);
                }
                None => {
                    for slot in entry.start..entry.end {
                        if !self.live[slot] {
                            continue;
                        }
                        let point = &self.coords[slot * d..(slot + 1) * d];
                        if form.value(query, point).is_some_and(|value| value <= limit) {
                            out.push(self.keys[slot]);
                        }
                    }
                }
            }
        }
        out
    }
}

/// One k-nearest search in progress.
struct Search<'s, F> {
    tree: &'s KdTree,
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

    fn visit(&mut self, node: usize, bound: f64) {
        let tree = self.tree;
        let entry = &tree.nodes[node];
        if entry.live == 0 || !self.open(bound) {
            return;
        }
        if let Some((left, right)) = entry.children {
            let left_bound = tree.bound_of(self.form, self.query, left);
            let right_bound = tree.bound_of(self.form, self.query, right);
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
        let d = tree.dimension;
        for slot in entry.start..entry.end {
            if !tree.live[slot] {
                continue;
            }
            let point = &tree.coords[slot * d..(slot + 1) * d];
            let Some(value) = self.form.value(self.query, point) else {
                continue;
            };
            if self.radius.is_some_and(|radius| value > radius) {
                continue;
            }
            let key = tree.keys[slot];
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

    #[test]
    fn removing_a_missing_or_dead_key_changes_nothing() {
        let mut tree = KdTree::new(1, &[0.0, 1.0], &[0, 2], 3);
        tree.remove(1);
        tree.remove(7);
        tree.remove(2);
        tree.remove(2);
        assert_eq!(tree.nodes[0].live, 1);
        assert_eq!(
            tree.nearest(&Euclidean, &[1.0], 2, None, &mut |_| true),
            vec![(1.0, 0)]
        );
    }
}
