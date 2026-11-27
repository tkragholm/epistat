//! Generalised full matching, MatchIt's `method = "quick"`.
//!
//! The method is that of Sävje, Higgins and Sekhon (2021), "Generalized full
//! matching", Political Analysis 29(4). It partitions the units into groups so
//! that every group holds at least a stated number of units of each treatment
//! group, and so that the largest distance between two units of the same group
//! is within a factor of four of the smallest possible. The group is the
//! subclass.
//!
//! # Algorithm
//!
//! 1. Build a directed graph on the units. From each unit an arc points to its
//!    nearest neighbours: the nearest `t_k` units of each group `k`, counting
//!    the unit itself in its own group, and then the nearest remaining units
//!    until the unit and its neighbours number the size constraint. This is the
//!    graph with the smallest arc lengths among those in which every unit and
//!    its out-neighbours satisfy the constraints.
//! 2. Choose seeds. A seed is a unit whose closed neighbourhood (itself and its
//!    out-neighbours) shares no unit with that of another seed. Seeds are added
//!    until no unit can be added.
//! 3. Each seed and its neighbours form a subclass. Every neighbourhood meets
//!    the constraints, so every subclass does.
//! 4. A unit in no subclass joins the subclass of the nearest seed.
//!
//! The largest arc of the graph is at most the largest within-group distance
//! of an optimal partition, because the groups of an optimal partition give a
//! graph with arcs no longer. A unit left over after step 3 points to a unit
//! in some neighbourhood, so it is within twice that arc of its nearest seed,
//! and two units of one subclass are within four times it of each other.
//!
//! # Seed order
//!
//! The bound holds for any set of seeds. The order in which units are tried as
//! seeds changes the number of subclasses, and with it the quality in
//! practice. [`SeedMethod::Lexical`] tries units in the order of the data.
//! [`SeedMethod::InwardsOrder`] tries them by the number of arcs pointing at
//! them, fewest first, and by position among units with equal counts.
//! [`SeedMethod::InwardsUpdating`], the default and the one `quickmatch` uses,
//! starts from that order and sweeps through it once. A unit that the sweep
//! passes without making it a seed, and every neighbour of a new seed, can no
//! longer be a seed, so the arcs they send are taken out of the counts of the
//! units still ahead that can be seeds. A unit whose count falls moves to the
//! end of the run of its new count, which changes the order among equal counts.
//!
//! The partition depends on the order of the units, and for the updating method
//! on the order of the arcs out of a unit. Units are taken anchors first, then
//! candidates, each in the order given. The arcs out of a unit go to the
//! nearest candidates, then the nearest anchors, then the nearest of either to
//! fill the size constraint, each nearest first. With the treated units as
//! anchors and the data ordered treated first this is the order in which
//! `quickmatch` meets them under `matchit()`, where the factor level of the
//! controls comes before that of the treated. The tie order among units at
//! exactly equal distance is not reproduced: `quickmatch` leaves it to the
//! nearest neighbour library.
//!
//! # Caliper
//!
//! A caliper `c` bounds the distance between any two units of a subclass. The
//! radius `c / 2` limits the arcs of the graph: a unit with an arc longer than
//! the radius cannot be a seed and has no neighbourhood, and a unit left over
//! joins its nearest seed only if that seed is within the radius. A unit that
//! fits no subclass is dropped.
//!
//! # Nearest neighbour search and cost
//!
//! For one coordinate the units are sorted once and a query walks outwards from
//! its position, so each neighbour costs constant time after a binary search.
//! For several coordinates a k-d tree (median splits on the widest axis,
//! leaves of at most 16 units) answers each query in `O(log n + k)` expected
//! time when the dimension is small, and its cost grows quickly with the
//! dimension. A Mahalanobis distance is Euclidean after a Cholesky factor of
//! the inverse covariance is applied, so it uses the same search.
//!
//! Sorting and tree building take `O(n log n)`. Building the graph takes
//! `O(n (log n + k))` for size constraint `k`. Choosing seeds takes `O(n k)`
//! for each seed method, since the updating sweep touches each arc a constant
//! number of times and moves a unit in constant time. Assigning the leftover
//! units takes `O(n log n)`. Queries run in parallel, and every choice is made
//! on a total order of the distance and the position of the unit, so the result
//! does not depend on the number of threads.
//!
//! The weights are those of full matching and subclassification: for the effect
//! in the anchors an anchor weighs 1 and a candidate weighs the anchor share of
//! its subclass over the candidate share, and likewise for the other effects.
//! MatchIt rescales the weights of each group to sum to the number of its kept
//! units.

use super::exact::{SubclassedUnits, stratum_numbers, unit_ids};
use super::records::MatchingRecord;
use super::subclassification::build_subclass_summaries;
use crate::Error;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

const NONE: u32 = u32::MAX;
const LEAF: usize = 16;
const PARALLEL_BUILD: usize = 8192;

/// The order in which units are tried as seeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedMethod {
    /// Fewest arcs pointing at the unit first, counted again after every seed
    /// without the arcs of units that can no longer be seeds. The default of
    /// `quickmatch`.
    #[default]
    InwardsUpdating,
    /// Fewest arcs pointing at the unit first, counted once.
    InwardsOrder,
    /// The order of the data.
    Lexical,
}

/// The constraints of generalised full matching.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QuickMatchOptions {
    /// Fewest cases in a subclass, at least 1. `min_anchors` in 0.2.
    #[serde(alias = "min_anchors")]
    pub min_cases: usize,
    /// Fewest controls in a subclass, at least 1. `min_candidates` in 0.2.
    #[serde(alias = "min_candidates")]
    pub min_controls: usize,
    /// Fewest units in a subclass, `size_constraint` of `quickmatch()`. The
    /// default is `min_cases + min_controls`.
    pub size: Option<usize>,
    /// Widest distance between two units of a subclass, `caliper` of
    /// `quickmatch()` in the units of the distance.
    pub caliper: Option<f64>,
    /// How seeds are chosen.
    pub seed_method: SeedMethod,
}

impl Default for QuickMatchOptions {
    fn default() -> Self {
        Self {
            min_cases: 1,
            min_controls: 1,
            size: None,
            caliper: None,
            seed_method: SeedMethod::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Nearest neighbour search
// ---------------------------------------------------------------------------

/// A neighbour: squared distance and the unit.
type Hit = (f64, u32);

fn hit_order(a: &Hit, b: &Hit) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// Units sorted by one coordinate.
struct Sorted1D {
    values: Vec<f64>,
    ids: Vec<u32>,
}

impl Sorted1D {
    /// `order` lists unit ids by value, then id.
    fn new(points: &[f64], order: impl Iterator<Item = u32>) -> Self {
        let ids: Vec<u32> = order.collect();
        let values = ids.iter().map(|id| points[*id as usize]).collect();
        Self { values, ids }
    }

    /// The `k` units nearest `query`, excluding `exclude`, appended to `out` in
    /// order of distance. Among equally distant units the one earlier in the
    /// sorted order comes first.
    fn nearest(&self, query: f64, k: usize, exclude: u32, out: &mut Vec<Hit>) {
        let len = self.values.len();
        let pos = self.values.partition_point(|value| *value < query);
        let (mut left, mut right) = (pos, pos);
        let start = out.len();
        while out.len() - start < k {
            while left > 0 && self.ids[left - 1] == exclude {
                left -= 1;
            }
            while right < len && self.ids[right] == exclude {
                right += 1;
            }
            let from_left = (left > 0).then(|| query - self.values[left - 1]);
            let from_right = (right < len).then(|| self.values[right] - query);
            match (from_left, from_right) {
                (None, None) => break,
                (Some(d), None) => {
                    left -= 1;
                    out.push((d * d, self.ids[left]));
                }
                (None, Some(d)) => {
                    out.push((d * d, self.ids[right]));
                    right += 1;
                }
                (Some(dl), Some(dr)) => {
                    if dl <= dr {
                        left -= 1;
                        out.push((dl * dl, self.ids[left]));
                    } else {
                        out.push((dr * dr, self.ids[right]));
                        right += 1;
                    }
                }
            }
        }
    }
}

/// A k-d tree over some of the units. The unit at the middle of a slice of
/// `order` splits it on `axis`.
struct KdTree<'a> {
    points: &'a [f64],
    dim: usize,
    order: Vec<u32>,
    axis: Vec<u8>,
}

impl<'a> KdTree<'a> {
    fn new(points: &'a [f64], dim: usize, mut order: Vec<u32>) -> Self {
        let mut axis = vec![0_u8; order.len()];
        Self::build(points, dim, &mut order, &mut axis);
        Self {
            points,
            dim,
            order,
            axis,
        }
    }

    fn build(points: &[f64], dim: usize, order: &mut [u32], axis: &mut [u8]) {
        let len = order.len();
        if len <= LEAF {
            return;
        }
        let mut best_axis = 0;
        let mut best_spread = -1.0_f64;
        for k in 0..dim {
            let mut low = f64::INFINITY;
            let mut high = f64::NEG_INFINITY;
            for id in order.iter() {
                let value = points[*id as usize * dim + k];
                low = low.min(value);
                high = high.max(value);
            }
            if high - low > best_spread {
                best_spread = high - low;
                best_axis = k;
            }
        }
        let mid = len / 2;
        order.select_nth_unstable_by(mid, |a, b| {
            points[*a as usize * dim + best_axis]
                .total_cmp(&points[*b as usize * dim + best_axis])
                .then(a.cmp(b))
        });
        // The axis is below 256: the dimension is checked on entry.
        axis[mid] = u8::try_from(best_axis).unwrap_or(u8::MAX);
        let (order_low, order_rest) = order.split_at_mut(mid);
        let (axis_low, axis_rest) = axis.split_at_mut(mid);
        let order_high = &mut order_rest[1..];
        let axis_high = &mut axis_rest[1..];
        if len >= PARALLEL_BUILD {
            rayon::join(
                || Self::build(points, dim, order_low, axis_low),
                || Self::build(points, dim, order_high, axis_high),
            );
        } else {
            Self::build(points, dim, order_low, axis_low);
            Self::build(points, dim, order_high, axis_high);
        }
    }

    fn consider(&self, id: u32, query: &[f64], k: usize, exclude: u32, best: &mut Vec<Hit>) {
        if id == exclude {
            return;
        }
        let row = &self.points[id as usize * self.dim..][..self.dim];
        let mut sum = 0.0_f64;
        for (a, b) in row.iter().zip(query) {
            let delta = a - b;
            sum += delta * delta;
        }
        let hit = (sum, id);
        if best.len() == k && hit_order(&hit, &best[k - 1]) != Ordering::Less {
            return;
        }
        let at = best.partition_point(|other| hit_order(other, &hit) == Ordering::Less);
        best.insert(at, hit);
        best.truncate(k);
    }

    fn search(
        &self,
        low: usize,
        high: usize,
        query: &[f64],
        k: usize,
        exclude: u32,
        best: &mut Vec<Hit>,
    ) {
        let len = high - low;
        if len <= LEAF {
            for id in &self.order[low..high] {
                self.consider(*id, query, k, exclude, best);
            }
            return;
        }
        let mid = low + len / 2;
        let pivot = self.order[mid];
        self.consider(pivot, query, k, exclude, best);
        let axis = usize::from(self.axis[mid]);
        let diff = query[axis] - self.points[pivot as usize * self.dim + axis];
        let (near, far) = if diff < 0.0 {
            ((low, mid), (mid + 1, high))
        } else {
            ((mid + 1, high), (low, mid))
        };
        self.search(near.0, near.1, query, k, exclude, best);
        // Equal distances count: a unit across the split can tie and have the
        // lower position.
        if best.len() < k || diff * diff <= best[k - 1].0 {
            self.search(far.0, far.1, query, k, exclude, best);
        }
    }

    fn nearest(&self, query: &[f64], k: usize, exclude: u32, out: &mut Vec<Hit>) {
        let mut best = Vec::with_capacity(k + 1);
        if k > 0 {
            self.search(0, self.order.len(), query, k, exclude, &mut best);
        }
        out.extend(best);
    }
}

enum Index<'a> {
    Line(Sorted1D),
    Tree(KdTree<'a>),
}

impl<'a> Index<'a> {
    /// Index the units of `ids`, given in increasing order of unit.
    fn new(points: &'a [f64], dim: usize, sorted_line: Option<&[u32]>, ids: Vec<u32>) -> Self {
        match sorted_line {
            Some(line) => Self::Line(Sorted1D::new(points, line.iter().copied())),
            None => Self::Tree(KdTree::new(points, dim, ids)),
        }
    }

    fn nearest(&self, query: &[f64], k: usize, exclude: u32, out: &mut Vec<Hit>) {
        match self {
            Self::Line(line) => line.nearest(query[0], k, exclude, out),
            Self::Tree(tree) => tree.nearest(query, k, exclude, out),
        }
    }
}

// ---------------------------------------------------------------------------
// The graph and the seeds
// ---------------------------------------------------------------------------

struct Graph {
    stride: usize,
    arcs: Vec<u32>,
    eligible: Vec<bool>,
}

impl Graph {
    fn out(&self, unit: usize) -> impl Iterator<Item = u32> + '_ {
        self.arcs[unit * self.stride..][..self.stride]
            .iter()
            .copied()
            .take_while(|arc| *arc != NONE)
    }
}

#[derive(Default)]
struct Scratch {
    chosen: Vec<Hit>,
    buffer: Vec<Hit>,
}

struct Constraints {
    minimum: [usize; 2],
    size: usize,
    radius: Option<f64>,
}

fn group_of(anchor: bool) -> usize {
    usize::from(!anchor)
}

/// Arcs needed from a unit of `group`: the neighbours of each group it still
/// needs, or enough to reach the size constraint.
fn arcs_needed(group: usize, constraints: &Constraints) -> usize {
    let mut quota = 0;
    for (k, minimum) in constraints.minimum.iter().enumerate() {
        quota += minimum.saturating_sub(usize::from(k == group));
    }
    quota.max(constraints.size - 1)
}

fn build_graph(
    points: &[f64],
    dim: usize,
    anchor: &[bool],
    constraints: &Constraints,
    all: &Index<'_>,
    by_group: &[Index<'_>; 2],
) -> Graph {
    let n = anchor.len();
    let stride = arcs_needed(0, constraints).max(arcs_needed(1, constraints));
    let mut arcs = vec![NONE; n * stride];
    let mut eligible = vec![true; n];
    arcs.par_chunks_mut(stride)
        .zip(eligible.par_iter_mut())
        .enumerate()
        .for_each_init(Scratch::default, |scratch, (unit, (slot, ok))| {
            let group = group_of(anchor[unit]);
            let query = &points[unit * dim..][..dim];
            // A unit count fits u32: checked on entry.
            let me = u32::try_from(unit).unwrap_or(NONE);
            scratch.chosen.clear();
            // Candidates before anchors: the order of the arcs fixes the order in
            // which the updating seed method visits units.
            for k in [1, 0] {
                let quota = constraints.minimum[k].saturating_sub(usize::from(k == group));
                if quota > 0 {
                    by_group[k].nearest(query, quota, me, &mut scratch.chosen);
                }
            }
            let need = arcs_needed(group, constraints);
            if scratch.chosen.len() < need {
                scratch.buffer.clear();
                all.nearest(query, need, me, &mut scratch.buffer);
                for hit in &scratch.buffer {
                    if scratch.chosen.len() == need {
                        break;
                    }
                    if !scratch.chosen.iter().any(|other| other.1 == hit.1) {
                        scratch.chosen.push(*hit);
                    }
                }
            }
            if let Some(radius) = constraints.radius
                && scratch.chosen.iter().any(|hit| hit.0.sqrt() > radius)
            {
                *ok = false;
                return;
            }
            for (place, hit) in slot.iter_mut().zip(&scratch.chosen) {
                *place = hit.1;
            }
        });
    Graph {
        stride,
        arcs,
        eligible,
    }
}

/// Units ordered by the number of arcs pointing at them, fewest first and by
/// position among equals, with the start of each count's run.
struct ByInwards {
    count: Vec<u32>,
    sorted: Vec<u32>,
    position: Vec<usize>,
    run_start: Vec<usize>,
}

impl ByInwards {
    fn new(graph: &Graph, n: usize) -> Self {
        let mut count = vec![0_u32; n];
        for unit in 0..n {
            for arc in graph.out(unit) {
                count[arc as usize] += 1;
            }
        }
        let top = count.iter().copied().max().unwrap_or(0) as usize;
        let mut run_start = vec![0_usize; top + 2];
        for unit in 0..n {
            run_start[count[unit] as usize + 1] += 1;
        }
        for c in 1..run_start.len() {
            run_start[c] += run_start[c - 1];
        }
        let mut next = run_start.clone();
        let mut sorted = vec![NONE; n];
        let mut position = vec![0_usize; n];
        for unit in 0..n {
            let slot = &mut next[count[unit] as usize];
            sorted[*slot] = u32::try_from(unit).unwrap_or(NONE);
            position[unit] = *slot;
            *slot += 1;
        }
        run_start.truncate(top + 1);
        Self {
            count,
            sorted,
            position,
            run_start,
        }
    }

    /// Take one arc out of the count of `unit`, which lies after `current` in
    /// the order. The unit moves to the end of the run of its new count, but
    /// not before the unit after `current`: the swap exchanges it with the
    /// first unit of its old run.
    fn decrease(&mut self, unit: usize, current: usize) {
        let old = self.count[unit] as usize;
        let from = self.position[unit];
        let mut to = self.run_start[old];
        if to <= current {
            to = current + 1;
            self.run_start[old - 1] = to;
        }
        self.run_start[old] = to + 1;
        self.count[unit] -= 1;
        let other = self.sorted[to];
        self.sorted[from] = other;
        self.sorted[to] = u32::try_from(unit).unwrap_or(NONE);
        self.position[other as usize] = from;
        self.position[unit] = to;
    }
}

/// The seeds, and the subclass of each unit in a seed's closed neighbourhood.
fn choose_seeds(graph: &Graph, n: usize, method: SeedMethod) -> (Vec<u32>, Vec<u32>) {
    let mut cluster = vec![NONE; n];
    let mut seeds: Vec<u32> = Vec::new();
    let take = |unit: usize, cluster: &mut Vec<u32>, seeds: &mut Vec<u32>| {
        let label = u32::try_from(seeds.len()).unwrap_or(NONE);
        seeds.push(u32::try_from(unit).unwrap_or(NONE));
        cluster[unit] = label;
        for arc in graph.out(unit) {
            cluster[arc as usize] = label;
        }
    };
    let free = |unit: usize, cluster: &[u32]| {
        graph.eligible[unit]
            && cluster[unit] == NONE
            && graph.out(unit).all(|arc| cluster[arc as usize] == NONE)
    };

    match method {
        SeedMethod::Lexical => {
            for unit in 0..n {
                if free(unit, &cluster) {
                    take(unit, &mut cluster, &mut seeds);
                }
            }
        }
        SeedMethod::InwardsOrder => {
            let order = ByInwards::new(graph, n);
            for unit in &order.sorted {
                if free(*unit as usize, &cluster) {
                    take(*unit as usize, &mut cluster, &mut seeds);
                }
            }
        }
        SeedMethod::InwardsUpdating => {
            // One sweep through the order. A unit that is passed without
            // becoming a seed, and the neighbours of a seed, can no longer
            // be a seed, so the arcs they send are taken out of the counts of
            // the units still ahead of the sweep that can be seeds.
            let mut order = ByInwards::new(graph, n);
            for current in 0..n {
                let unit = order.sorted[current] as usize;
                if free(unit, &cluster) {
                    take(unit, &mut cluster, &mut seeds);
                    let neighbours: Vec<u32> = graph.out(unit).collect();
                    for neighbour in neighbours {
                        if current < order.position[neighbour as usize] {
                            for target in graph.out(neighbour as usize) {
                                let target = target as usize;
                                if cluster[target] == NONE
                                    && current < order.position[target]
                                    && graph.eligible[target]
                                {
                                    order.decrease(target, current);
                                }
                            }
                        }
                    }
                } else if cluster[unit] == NONE {
                    for target in graph.out(unit) {
                        let target = target as usize;
                        if cluster[target] == NONE
                            && current < order.position[target]
                            && graph.eligible[target]
                        {
                            order.decrease(target, current);
                        }
                    }
                }
            }
        }
    }
    (seeds, cluster)
}

/// The subclass of each unit, numbered from 0 in the order of the first unit
/// each holds, or `None` for a unit that fits no subclass.
///
/// `points` holds `dim` coordinates per unit in unit order and `anchor` says
/// which units are anchors. Distances are Euclidean.
///
/// # Errors
///
/// A [`Error`] when the input is malformed, when a group has too few
/// units, or when the caliper leaves no subclass.
pub fn quick_clusters(
    points: &[f64],
    dim: usize,
    anchor: &[bool],
    options: &QuickMatchOptions,
) -> Result<Vec<Option<usize>>, Error> {
    let n = anchor.len();
    if dim == 0 || dim > 255 || points.len() != n * dim {
        return Err(Error::Coordinates);
    }
    let Ok(n32) = u32::try_from(n) else {
        return Err(Error::TooManyUnits);
    };
    if let Some(bad) = points.iter().position(|value| !value.is_finite()) {
        return Err(Error::NonFiniteCoordinate { unit: bad / dim });
    }
    if options.min_cases == 0 || options.min_controls == 0 {
        return Err(Error::SubclassMinimum);
    }
    let minimum_total = options.min_cases + options.min_controls;
    let size = options.size.unwrap_or(minimum_total);
    if size < minimum_total {
        return Err(Error::SizeBelowMinimums);
    }
    if let Some(caliper) = options.caliper
        && !(caliper.is_finite() && caliper > 0.0)
    {
        return Err(Error::InvalidCaliper);
    }
    let anchors = anchor.iter().filter(|flag| **flag).count();
    if anchors < options.min_cases || n - anchors < options.min_controls || n < size {
        return Err(Error::TooFewUnits { stratum: None });
    }
    let constraints = Constraints {
        minimum: [options.min_cases, options.min_controls],
        size,
        radius: options.caliper.map(|caliper| caliper / 2.0),
    };

    // Indices: all units, and each group. On a line the units are sorted once
    // and each group is a filtered copy of that order.
    let line: Option<Vec<u32>> = (dim == 1).then(|| {
        let mut ids: Vec<u32> = (0..n32).collect();
        ids.par_sort_unstable_by(|a, b| {
            points[*a as usize]
                .total_cmp(&points[*b as usize])
                .then(a.cmp(b))
        });
        ids
    });
    let group_ids = |group: usize| -> Vec<u32> {
        (0..n32)
            .filter(|unit| group_of(anchor[*unit as usize]) == group)
            .collect()
    };
    let group_line = |group: usize| -> Option<Vec<u32>> {
        line.as_ref().map(|all| {
            all.iter()
                .copied()
                .filter(|unit| group_of(anchor[*unit as usize]) == group)
                .collect()
        })
    };
    let all = Index::new(points, dim, line.as_deref(), (0..n32).collect());
    let line_a = group_line(0);
    let line_c = group_line(1);
    let by_group = [
        Index::new(points, dim, line_a.as_deref(), group_ids(0)),
        Index::new(points, dim, line_c.as_deref(), group_ids(1)),
    ];

    let graph = build_graph(points, dim, anchor, &constraints, &all, &by_group);
    let (seeds, cluster) = choose_seeds(&graph, n, options.seed_method);
    if seeds.is_empty() {
        return Err(Error::NoSubclass);
    }

    // Units in no neighbourhood join the nearest seed.
    let seed_line: Option<Vec<u32>> = line.as_ref().map(|_| {
        let mut ids = seeds.clone();
        ids.par_sort_unstable_by(|a, b| {
            points[*a as usize]
                .total_cmp(&points[*b as usize])
                .then(a.cmp(b))
        });
        ids
    });
    let seed_index = Index::new(points, dim, seed_line.as_deref(), {
        let mut ids = seeds.clone();
        ids.sort_unstable();
        ids
    });
    let mut label: Vec<u32> = cluster;
    let mut seed_cluster = vec![NONE; n];
    for (position, seed) in seeds.iter().enumerate() {
        seed_cluster[*seed as usize] = u32::try_from(position).unwrap_or(NONE);
    }
    label.par_iter_mut().enumerate().for_each_init(
        Vec::new,
        |hits: &mut Vec<Hit>, (unit, slot)| {
            if *slot != NONE {
                return;
            }
            hits.clear();
            seed_index.nearest(&points[unit * dim..][..dim], 1, NONE, hits);
            if let Some((d2, seed)) = hits.first() {
                let within = constraints.radius.is_none_or(|radius| d2.sqrt() <= radius);
                if within {
                    *slot = seed_cluster[*seed as usize];
                }
            }
        },
    );

    // Number the subclasses by the first unit each holds.
    let mut number = vec![NONE; seeds.len()];
    let mut next = 0_u32;
    Ok(label
        .iter()
        .map(|cluster| {
            if *cluster == NONE {
                return None;
            }
            let slot = &mut number[*cluster as usize];
            if *slot == NONE {
                *slot = next;
                next += 1;
            }
            Some(*slot as usize)
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Units with identifiers
// ---------------------------------------------------------------------------

/// What one stratum of the units gave.
enum StratumLabels {
    /// The stratum lacks one of the groups.
    Absent,
    /// The labels of its units, in position order.
    Labels(Vec<Option<usize>>),
    /// No subclass can be made in the stratum, for the reason given: too few
    /// units, or no unit that fits a subclass under the caliper.
    Lost(Error),
}

/// Generalised full matching of anchors and candidates.
///
/// `coordinates` gives the point of a unit: one number for a score, or the
/// coordinates of a covariate distance made Euclidean, all of length
/// `dimension`, which is at most 255. Units that share a value of every `exact`
/// variable are matched among themselves, as `matchit()` does with `exact`, and
/// a stratum with one group only is dropped. Subclasses are numbered over all strata.
///
/// The strata are independent and are matched in parallel. The stratum of a
/// unit comes from dictionary codes of its values, taken once, and the outcome
/// is a table by unit position, see [`SubclassedUnits`].
///
/// # Errors
///
/// A [`Error`] when a unit lacks coordinates or an exact variable, or
/// when nothing can be matched. A stratum that is too small for the constraints,
/// or in which no unit fits a subclass under the caliper, loses its units, which
/// are dropped; the call fails only when no stratum has a subclass, with the
/// error of the first stratum lost in order of first appearance.
pub fn quick_match_subclasses<'c, R, V, F>(
    anchors: &[R],
    candidates: &[R],
    coordinates: F,
    dimension: usize,
    exact: &[V],
    options: &QuickMatchOptions,
) -> Result<SubclassedUnits, Error>
where
    R: MatchingRecord,
    V: AsRef<str>,
    F: Fn(&R) -> Option<&'c [f64]> + Sync,
{
    let unit_count = anchors.len() + candidates.len();
    let unit = |position: usize| -> &R {
        if position < anchors.len() {
            &anchors[position]
        } else {
            &candidates[position - anchors.len()]
        }
    };
    let is_case = |position: usize| position < anchors.len();

    let (strata, stratum_count): (Vec<u32>, usize) = if exact.is_empty() {
        (vec![0; unit_count], 1)
    } else {
        stratum_numbers(anchors, candidates, exact).map_err(|(position, variable)| {
            Error::MissingValue {
                id: unit(position).id().to_string(),
                variable: exact[variable].as_ref().to_string(),
            }
        })?
    };

    // Positions of each stratum in ascending order: a counting sort.
    let mut start = vec![0_usize; stratum_count + 1];
    for stratum in &strata {
        start[*stratum as usize + 1] += 1;
    }
    for stratum in 0..stratum_count {
        start[stratum + 1] += start[stratum];
    }
    let mut order = vec![0_usize; unit_count];
    let mut fill = start.clone();
    for (position, stratum) in strata.iter().enumerate() {
        order[fill[*stratum as usize]] = position;
        fill[*stratum as usize] += 1;
    }

    let stratified = !exact.is_empty();
    let results: Vec<Result<StratumLabels, Error>> = (0..stratum_count)
        .into_par_iter()
        .map(|stratum| {
            let positions = &order[start[stratum]..start[stratum + 1]];
            // Anchors come first in position order.
            let both = positions.first().is_some_and(|p| is_case(*p))
                && positions.last().is_some_and(|p| !is_case(*p));
            if !both {
                return Ok(StratumLabels::Absent);
            }
            let mut points = Vec::with_capacity(positions.len() * dimension);
            for position in positions {
                let row = unit(*position);
                let point = coordinates(row)
                    .filter(|point| point.len() == dimension)
                    .ok_or_else(|| Error::MissingCoordinates {
                        id: row.id().to_string(),
                    })?;
                points.extend_from_slice(point);
            }
            let flags: Vec<bool> = positions.iter().map(|p| is_case(*p)).collect();
            match quick_clusters(&points, dimension, &flags, options) {
                Ok(labels) => Ok(StratumLabels::Labels(labels)),
                // With several strata, one with too few units or in which no
                // unit fits a subclass under the caliper loses its units, which
                // are dropped; the run fails only when no stratum has a subclass.
                Err(Error::NoSubclass) if stratified => Ok(StratumLabels::Lost(Error::NoSubclass)),
                Err(Error::TooFewUnits { .. }) if stratified => {
                    Ok(StratumLabels::Lost(Error::TooFewUnits {
                        stratum: Some(stratum.to_string()),
                    }))
                }
                Err(error) => Err(error),
            }
        })
        .collect();

    // A label of a stratum is below its number of units, so a label and the
    // first position of its stratum in `order` name a subclass uniquely.
    let mut global = vec![NONE; unit_count];
    let mut shared = false;
    let mut first_lost: Option<Error> = None;
    for (stratum, result) in results.into_iter().enumerate() {
        match result? {
            StratumLabels::Absent => {}
            StratumLabels::Lost(error) => {
                shared = true;
                first_lost.get_or_insert(error);
            }
            StratumLabels::Labels(labels) => {
                shared = true;
                let positions = &order[start[stratum]..start[stratum + 1]];
                for (position, label) in positions.iter().zip(labels) {
                    if let Some(label) = label {
                        global[*position] = u32::try_from(start[stratum] + label)
                            .map_err(|_| Error::TooManyUnits)?;
                    }
                }
            }
        }
    }
    if !shared {
        return Err(Error::NoSharedStratum);
    }

    let mut number = vec![NONE; unit_count];
    let mut anchor_counts: Vec<usize> = Vec::new();
    let mut candidate_counts: Vec<usize> = Vec::new();
    let mut subclass: Vec<Option<usize>> = Vec::with_capacity(unit_count);
    for (position, key) in global.iter().enumerate() {
        if *key == NONE {
            subclass.push(None);
            continue;
        }
        let slot = &mut number[*key as usize];
        if *slot == NONE {
            *slot = u32::try_from(anchor_counts.len()).map_err(|_| Error::TooManyUnits)?;
            anchor_counts.push(0);
            candidate_counts.push(0);
        }
        let numbered = *slot as usize;
        if is_case(position) {
            anchor_counts[numbered] += 1;
        } else {
            candidate_counts[numbered] += 1;
        }
        subclass.push(Some(numbered));
    }
    if anchor_counts.is_empty() {
        return Err(first_lost.unwrap_or(Error::NoSubclass));
    }
    Ok(SubclassedUnits {
        ids: unit_ids(anchors, candidates),
        case_count: anchors.len(),
        subclass,
        subclasses: build_subclass_summaries(&anchor_counts, &candidate_counts),
        trimmed: crate::TrimmedUnits::default(),
    })
}

/// `L^T x` for a lower triangular `L`.
#[must_use]
pub fn apply_transposed(lower: &[f64], vector: &[f64], dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|j| {
            let mut sum = 0.0_f64;
            for i in j..dimension {
                sum += lower[i * dimension + j] * vector[i];
            }
            sum
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::cholesky::{Pivot, cholesky_lower};
    use std::collections::HashMap;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
        }
    }

    fn sample(n: usize, dim: usize, seed: u64) -> (Vec<f64>, Vec<bool>) {
        let mut rng = Lcg(seed);
        let points = (0..n * dim).map(|_| rng.next()).collect();
        let anchor = (0..n).map(|_| rng.next() < 0.3).collect();
        (points, anchor)
    }

    fn check_constraints(labels: &[Option<usize>], anchor: &[bool], options: &QuickMatchOptions) {
        let count = labels.iter().flatten().max().map_or(0, |m| m + 1);
        let mut anchors = vec![0; count];
        let mut candidates = vec![0; count];
        for (label, is_case) in labels.iter().zip(anchor) {
            if let Some(label) = label {
                if *is_case {
                    anchors[*label] += 1;
                } else {
                    candidates[*label] += 1;
                }
            }
        }
        let size = options
            .size
            .unwrap_or(options.min_cases + options.min_controls);
        for subclass in 0..count {
            assert!(anchors[subclass] >= options.min_cases);
            assert!(candidates[subclass] >= options.min_controls);
            assert!(anchors[subclass] + candidates[subclass] >= size);
        }
    }

    /// The method as it was written with one vector of strings per unit as the
    /// stratum key and four maps by id as the outcome, kept to check the
    /// columnar one against.
    mod reference {
        use super::*;
        use crate::matching::exact::reference::Maps;
        use crate::matching::subclassification::subclass_weight_for_estimand;
        use crate::types::Estimand;
        use std::collections::HashMap;

        /// Doc comments of the original are not repeated.
        pub fn quick_by_maps<'c, R, V, F>(
            anchors: &[R],
            candidates: &[R],
            coordinates: F,
            dimension: usize,
            exact: &[V],
            options: &QuickMatchOptions,
        ) -> Result<Maps, Error>
        where
            R: MatchingRecord,
            V: AsRef<str>,
            F: Fn(&R) -> Option<&'c [f64]>,
        {
            // Units in order: anchors, then candidates.
            let units: Vec<(&R, bool)> = anchors
                .iter()
                .map(|row| (row, true))
                .chain(candidates.iter().map(|row| (row, false)))
                .collect();

            let mut strata: Vec<usize> = vec![0; units.len()];
            let mut stratum_count = 1;
            if !exact.is_empty() {
                let mut seen: HashMap<Vec<&str>, usize> = HashMap::new();
                for (position, (row, _)) in units.iter().enumerate() {
                    let key: Vec<&str> = exact
                        .iter()
                        .map(|variable| {
                            row.stratum(variable.as_ref())
                                .ok_or_else(|| Error::MissingValue {
                                    id: row.id().to_string(),
                                    variable: variable.as_ref().to_string(),
                                })
                        })
                        .collect::<Result<_, _>>()?;
                    let next = seen.len();
                    strata[position] = *seen.entry(key).or_insert(next);
                }
                stratum_count = seen.len();
            }

            let mut members: Vec<Vec<usize>> = vec![Vec::new(); stratum_count];
            for (position, stratum) in strata.iter().enumerate() {
                members[*stratum].push(position);
            }

            // Subclass of each unit, numbered over strata by first appearance.
            let mut global: Vec<Option<(usize, usize)>> = vec![None; units.len()];
            let mut shared = false;
            let mut first_lost: Option<Error> = None;
            for (stratum, positions) in members.iter().enumerate() {
                let has_anchor = positions.iter().any(|p| units[*p].1);
                let has_candidate = positions.iter().any(|p| !units[*p].1);
                if !(has_anchor && has_candidate) {
                    continue;
                }
                shared = true;
                let mut points = Vec::with_capacity(positions.len() * dimension);
                for position in positions {
                    let row = units[*position].0;
                    let point = coordinates(row)
                        .filter(|point| point.len() == dimension)
                        .ok_or_else(|| Error::MissingCoordinates {
                            id: row.id().to_string(),
                        })?;
                    points.extend_from_slice(point);
                }
                let flags: Vec<bool> = positions.iter().map(|p| units[*p].1).collect();
                let labels = match quick_clusters(&points, dimension, &flags, options) {
                    Ok(labels) => labels,
                    // With several strata, one that is too small or in which no unit
                    // fits a subclass under the caliper loses its units, which are
                    // dropped; the run fails only when no stratum has a subclass.
                    Err(Error::NoSubclass) if !exact.is_empty() => {
                        first_lost.get_or_insert(Error::NoSubclass);
                        continue;
                    }
                    Err(Error::TooFewUnits { .. }) if !exact.is_empty() => {
                        first_lost.get_or_insert(Error::TooFewUnits {
                            stratum: Some(stratum.to_string()),
                        });
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                for (position, label) in positions.iter().zip(labels) {
                    global[*position] = label.map(|label| (stratum, label));
                }
            }
            if !shared {
                return Err(Error::NoSharedStratum);
            }

            let mut number: HashMap<(usize, usize), usize> = HashMap::new();
            let mut anchor_counts: Vec<usize> = Vec::new();
            let mut candidate_counts: Vec<usize> = Vec::new();
            let mut subclass_of: Vec<Option<usize>> = Vec::with_capacity(units.len());
            for (key, (_, is_case)) in global.iter().zip(&units) {
                let subclass = key.map(|key| {
                    let next = anchor_counts.len();
                    let subclass = *number.entry(key).or_insert(next);
                    if subclass == next {
                        anchor_counts.push(0);
                        candidate_counts.push(0);
                    }
                    if *is_case {
                        anchor_counts[subclass] += 1;
                    } else {
                        candidate_counts[subclass] += 1;
                    }
                    subclass
                });
                subclass_of.push(subclass);
            }
            if anchor_counts.is_empty() {
                return Err(first_lost.unwrap_or(Error::NoSubclass));
            }
            let subclasses = build_subclass_summaries(&anchor_counts, &candidate_counts);
            let mut outcome = Maps {
                subclasses,
                ..Maps::default()
            };
            for ((row, is_case), subclass) in units.iter().zip(&subclass_of) {
                let id = row.id().to_string();
                let Some(subclass) = subclass else {
                    outcome.dropped_ids.push(id);
                    continue;
                };
                // Both minimums are at least 1, so both groups are present.
                let Some(share) = outcome.subclasses[*subclass].subclass_propensity else {
                    unreachable!("a kept subclass has units");
                };
                for (weights, estimand) in [
                    (&mut outcome.att_weights, Estimand::Att),
                    (&mut outcome.atc_weights, Estimand::Atc),
                    (&mut outcome.ate_weights, Estimand::Ate),
                ] {
                    weights.insert(
                        id.clone(),
                        subclass_weight_for_estimand(estimand, share, *is_case),
                    );
                }
                outcome.assignments.insert(id, *subclass);
            }
            Ok(outcome)
        }
    }

    fn points_of(
        anchors: &[crate::types::BaseRecord],
        candidates: &[crate::types::BaseRecord],
        dim: usize,
        seed: u64,
        holes: bool,
    ) -> HashMap<String, Vec<f64>> {
        let mut rng = Lcg(seed);
        anchors
            .iter()
            .chain(candidates)
            .filter_map(|row| {
                let vector: Vec<f64> = (0..dim)
                    // Values from a coarse grid, so that distances tie exactly.
                    .map(|_| (rng.next() * 40.0).floor() / 4.0)
                    .collect();
                if holes && rng.next() < 0.01 {
                    return None;
                }
                Some((row.id().to_string(), vector))
            })
            .collect()
    }

    #[test]
    fn the_columns_equal_the_string_keyed_version() {
        use crate::matching::exact::reference as data;
        let mut agreed = 0;
        for (seed, anchors, candidates, areas, dim, holes, exact) in [
            (1_u64, 60_usize, 140_usize, 3_u64, 1_usize, false, vec![]),
            (2, 400, 900, 4, 2, false, vec!["sex"]),
            (3, 2000, 3000, 6, 1, false, vec!["sex", "area"]),
            (4, 3000, 2500, 5, 3, false, vec!["region"]),
            (5, 500, 800, 4, 2, true, vec!["sex"]),
            (6, 20_000, 30_000, 7, 1, false, vec!["area", "sex"]),
            (7, 300, 400, 3, 2, false, vec!["sex", "gone"]),
        ] {
            let (a, c) = data::records(seed, anchors, candidates, areas, false);
            let vectors = points_of(&a, &c, dim, seed + 100, holes);
            for options in [
                QuickMatchOptions::default(),
                QuickMatchOptions {
                    min_cases: 2,
                    min_controls: 1,
                    caliper: Some(1.5),
                    seed_method: SeedMethod::Lexical,
                    ..QuickMatchOptions::default()
                },
                QuickMatchOptions {
                    seed_method: SeedMethod::InwardsOrder,
                    ..QuickMatchOptions::default()
                },
            ] {
                let context = format!("seed {seed} {exact:?} {options:?}");
                let lookup =
                    |row: &crate::types::BaseRecord| vectors.get(row.id()).map(Vec::as_slice);
                let new = quick_match_subclasses(&a, &c, lookup, dim, &exact, &options);
                let old = reference::quick_by_maps(&a, &c, lookup, dim, &exact, &options);
                match (new, old) {
                    (Ok(new), Ok(old)) => {
                        agreed += 1;
                        data::assert_same(&new, &old, &context);
                    }
                    (Err(new), Err(old)) => assert_eq!(new, old, "{context}"),
                    (new, old) => panic!("{context}: {new:?} against {old:?}"),
                }
            }
        }
        assert!(agreed >= 12, "only {agreed} runs produced an outcome");
    }

    /// Records with a stratum "g" and one coordinate each, anchors first.
    fn grouped(
        groups: &[(&str, usize, usize)],
    ) -> (
        Vec<crate::types::BaseRecord>,
        Vec<crate::types::BaseRecord>,
        HashMap<String, Vec<f64>>,
    ) {
        let date = chrono::NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid test date");
        let (mut anchors, mut candidates) = (Vec::new(), Vec::new());
        let mut points = HashMap::new();
        for (group, case_count, control_count) in groups {
            for (prefix, count, list) in [
                ("a", *case_count, &mut anchors),
                ("c", *control_count, &mut candidates),
            ] {
                for i in 0..count {
                    let id = format!("{prefix}{group}{i}");
                    let mut record = crate::types::BaseRecord::new(id.clone(), date);
                    record.strata.insert("g".into(), (*group).to_string());
                    points.insert(id, vec![i as f64]);
                    list.push(record);
                }
            }
        }
        (anchors, candidates, points)
    }

    #[test]
    fn a_stratum_with_too_few_units_is_dropped() {
        let (anchors, candidates, points) = grouped(&[("x", 5, 5), ("y", 1, 0), ("z", 0, 1)]);
        let lookup = |row: &crate::types::BaseRecord| points.get(row.id()).map(Vec::as_slice);
        let options = QuickMatchOptions {
            min_cases: 2,
            ..QuickMatchOptions::default()
        };
        let out = quick_match_subclasses(&anchors, &candidates, lookup, 1, &["g"], &options)
            .expect("the stratum that can be matched is matched");
        assert!(out.subclass.iter().any(Option::is_some));

        let (anchors, candidates, points) = grouped(&[("x", 1, 1), ("y", 1, 1)]);
        let lookup = |row: &crate::types::BaseRecord| points.get(row.id()).map(Vec::as_slice);
        assert!(matches!(
            quick_match_subclasses(&anchors, &candidates, lookup, 1, &["g"], &options),
            Err(Error::TooFewUnits { stratum: Some(_) })
        ));
    }

    #[test]
    fn the_outcome_is_the_same_on_any_number_of_threads() {
        use crate::matching::exact::reference as data;
        let (a, c) = data::records(21, 30_000, 40_000, 40, false);
        let vectors = points_of(&a, &c, 2, 5, false);
        epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
            let out = quick_match_subclasses(
                &a,
                &c,
                |row: &crate::types::BaseRecord| vectors.get(row.id()).map(Vec::as_slice),
                2,
                &["sex", "area"],
                &QuickMatchOptions::default(),
            )
            .unwrap();
            data::numbers(&out)
        });
    }

    #[test]
    fn subclasses_meet_the_constraints() {
        for dim in [1, 2, 3] {
            for method in [
                SeedMethod::InwardsUpdating,
                SeedMethod::InwardsOrder,
                SeedMethod::Lexical,
            ] {
                let (points, anchor) = sample(400, dim, 7 + dim as u64);
                for (min_cases, min_controls, size) in [(1, 1, None), (1, 2, None), (2, 3, Some(7))]
                {
                    let options = QuickMatchOptions {
                        min_cases,
                        min_controls,
                        size,
                        caliper: None,
                        seed_method: method,
                    };
                    let labels = quick_clusters(&points, dim, &anchor, &options).unwrap();
                    assert!(labels.iter().all(Option::is_some));
                    check_constraints(&labels, &anchor, &options);
                }
            }
        }
    }

    #[test]
    fn a_line_and_a_plane_with_a_flat_axis_agree() {
        let (points, anchor) = sample(500, 1, 11);
        let flat: Vec<f64> = points.iter().flat_map(|value| [*value, 0.0]).collect();
        for method in [SeedMethod::Lexical, SeedMethod::InwardsUpdating] {
            let options = QuickMatchOptions {
                seed_method: method,
                ..QuickMatchOptions::default()
            };
            assert_eq!(
                quick_clusters(&points, 1, &anchor, &options).unwrap(),
                quick_clusters(&flat, 2, &anchor, &options).unwrap()
            );
        }
    }

    #[test]
    fn the_tree_finds_the_nearest_units() {
        let (points, _) = sample(300, 3, 3);
        let ids: Vec<u32> = (0..300).collect();
        let tree = KdTree::new(&points, 3, ids);
        for unit in [0_usize, 17, 150, 299] {
            let query = &points[unit * 3..][..3];
            let mut found = Vec::new();
            tree.nearest(query, 5, u32::try_from(unit).unwrap(), &mut found);
            let mut brute: Vec<Hit> = (0..300_u32)
                .filter(|other| *other as usize != unit)
                .map(|other| {
                    let row = &points[other as usize * 3..][..3];
                    let sum: f64 = row.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                    (sum, other)
                })
                .collect();
            brute.sort_by(hit_order);
            brute.truncate(5);
            assert_eq!(found, brute);
        }
    }

    #[test]
    fn a_caliper_bounds_the_distance_within_a_subclass() {
        let (points, anchor) = sample(600, 1, 5);
        let caliper = 0.006;
        let options = QuickMatchOptions {
            caliper: Some(caliper),
            ..QuickMatchOptions::default()
        };
        let labels = quick_clusters(&points, 1, &anchor, &options).unwrap();
        assert!(labels.iter().any(Option::is_none));
        let count = labels.iter().flatten().max().unwrap() + 1;
        let mut low = vec![f64::INFINITY; count];
        let mut high = vec![f64::NEG_INFINITY; count];
        for (label, value) in labels.iter().zip(&points) {
            if let Some(label) = label {
                low[*label] = low[*label].min(*value);
                high[*label] = high[*label].max(*value);
            }
        }
        for subclass in 0..count {
            assert!(high[subclass] - low[subclass] <= caliper);
        }
        check_constraints(&labels, &anchor, &options);
    }

    #[test]
    fn refuses_what_cannot_be_matched() {
        let points = [0.0, 1.0, 2.0];
        let options = QuickMatchOptions::default();
        assert_eq!(
            quick_clusters(&points, 1, &[true, true, true], &options).unwrap_err(),
            Error::TooFewUnits { stratum: None }
        );
        let zero = QuickMatchOptions {
            min_controls: 0,
            ..options
        };
        assert_eq!(
            quick_clusters(&points, 1, &[true, false, false], &zero).unwrap_err(),
            Error::SubclassMinimum
        );
        assert_eq!(
            quick_clusters(&[0.0, f64::NAN], 1, &[true, false], &options).unwrap_err(),
            Error::NonFiniteCoordinate { unit: 1 }
        );
        let tight = QuickMatchOptions {
            caliper: Some(0.1),
            ..options
        };
        assert_eq!(
            quick_clusters(&points, 1, &[true, false, true], &tight).unwrap_err(),
            Error::NoSubclass
        );
    }

    #[test]
    fn mahalanobis_coordinates_give_the_mahalanobis_distance() {
        let inverse = [2.0, 0.5, 0.5, 1.0];
        let factor = cholesky_lower(&inverse, 2, Pivot::Strict).unwrap();
        let a = apply_transposed(&factor, &[1.0, 2.0], 2);
        let b = apply_transposed(&factor, &[0.0, 4.0], 2);
        let direct = {
            let d = [1.0, -2.0];
            2.0 * d[0] * d[0] + 2.0 * 0.5 * d[0] * d[1] + d[1] * d[1]
        };
        let embedded: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((embedded - direct).abs() < 1e-12);
        assert!(cholesky_lower(&[1.0, 2.0, 2.0, 1.0], 2, Pivot::Strict).is_none());
    }
}
