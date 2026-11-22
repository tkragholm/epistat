//! Fine, near-fine and refined balance in optimal pair matching.
//!
//! The pairs minimise the total distance among those whose matched candidates
//! have the anchors' distribution of a nominal variable. With `ratio` candidates
//! per anchor, the candidates of a category should number `ratio` times the
//! anchors in it. This is fine balance (Rosenbaum, Ross and Silber \[1\]). When
//! that is impossible, the least total deviation from it is wanted first and the
//! least distance second, which is near-fine balance (Yang, Small, Silber and
//! Rosenbaum \[2\]). A list of balance variables, each nested in the one before
//! and with decreasing priority, is refined balance (Pimentel, Yoon, Keele
//! \[3\]).
//!
//! # Network
//!
//! The network is that of pair matching, with the flow running from the sink to
//! the anchors. The sink supplies `ratio` times the anchors. A category node has
//! two arcs from the sink, or from its category one level up. The first has room
//! for the number of units the category is meant to receive and costs nothing.
//! The second has room for all units and costs a penalty per unit. The leaf
//! category node has an arc to each of its candidates, and a candidate has an
//! arc to each anchor it may be matched with, which carries one unit when the
//! pair is matched and costs the distance. Each anchor demands `ratio` units.
//! Candidates beyond the target of their category use the second arc, so the
//! penalties add up to the number of candidates above target. Since the targets
//! and the number of candidates both sum to `ratio` times the anchors, that
//! equals the number below target, and the total deviation is twice the
//! penalised units.
//!
//! The penalties are lexicographic. A unit above target in the finest level
//! costs more than the largest possible total distance, and a unit at each
//! coarser level costs more than every unit at the finer levels together. The
//! optimum therefore minimises the deviation of the first (coarsest) level,
//! then of the second, and so on, and the total distance last. This is the
//! ordering the papers aim for; `rcbalance` uses a fixed base of 3 for its
//! penalties, which only approximates it.
//!
//! The balance is global. When the request has exact strata, pairs cannot cross
//! them, but the targets are those of all the anchors, so the balance is over
//! the whole match, not within each stratum.
//!
//! # Costs
//!
//! Distances are scaled to integers as in [`super::optimal`], to `2^b` for the
//! largest `b` at most 40 for which the penalties of all levels stay inside the
//! solver's range. The largest cost is about `n^levels` times the largest
//! scaled distance for `n` pairs, so deep hierarchies on large problems get a
//! coarser distance. An integer-cost problem with `b` below 10 is refused.
//!
//! # Solving
//!
//! The network has an arc for every admissible pair, which for a few hundred
//! units per stratum is hundreds of arcs per anchor. The solver starts from far
//! fewer and adds the rest only if they can improve the result. The strata
//! touch only through the category nodes, so each stratum is solved as a block
//! of its own and the blocks are coupled through those nodes.
//!
//! 1. The pairs are listed one stratum at a time and never held together. In
//!    the first pass, the pairs of each anchor with its `3 * ratio` nearest
//!    candidates in each category of the finest level become the first arcs,
//!    found by selection, and for each anchor and category with pairs left
//!    out the cost of the last pair kept is recorded. No pair left out costs
//!    less.
//! 2. Every block is solved on its own as a flow from a root through one entry
//!    per finest-level category, with the arc into the entry of category `k`
//!    costing `price[k]`. The prices are moved (resilient propagation, each
//!    price with its own step) until the number of candidates taken per
//!    category is near its target, each round starting from the final tree of
//!    the round before. The blocks keep the flow of the round nearest the
//!    targets.
//! 3. The arcs between category nodes are filled or left empty by the sign of
//!    their reduced costs against potentials that put each finest-level
//!    category at its price. Since each block is optimal at those prices, no
//!    cycle of the network has a negative cost, and the flow left over at
//!    category nodes is moved by successive shortest paths \[4, ch. 9\]
//!    between category nodes. A path through a block moves one matched
//!    candidate from one category to another along the block's cheapest
//!    exchange, so the shortest paths are found on the category nodes alone,
//!    with each pair of categories joined by the cheapest exchange over the
//!    blocks. Each block keeps its exchange costs, from one shortest-path
//!    search per entry, and only the blocks a path crosses are searched again.
//!    Augmenting along shortest paths keeps every cycle at a non-negative
//!    cost, so the flow ends optimal for the pairs kept.
//! 4. Potentials for the whole network are the shortest distances from the
//!    category nodes, and every pair left out is priced against them. A pair
//!    whose reduced cost is negative could improve the flow. Only the anchors
//!    for which the recorded costs of step 1 do not already show that none of
//!    their pairs left out prices negatively have their pairs listed again.
//!    The pairs that do price negatively are
//!    added, their blocks are solved again from their flow at prices that are
//!    the potentials of the category nodes, which keeps every cycle at a
//!    non-negative cost, and step 3 continues. When no pair prices negatively
//!    the potentials are feasible for the full network, so the flow is optimal
//!    for it by complementary slackness: the optimal value, and so the
//!    deviation at every level and the total distance, is that of the network
//!    with all pairs. If the arcs so far leave some anchor without enough
//!    candidates, the lists are widened fourfold until it is feasible.
//!
//! The pairs returned can differ from those of the full network only among
//! matches of equal cost. Nothing depends on the number of threads: the blocks
//! are solved independently and combined in stratum order, and the shortest
//! paths are found on one thread with ties settled by node number.
//!
//! The work is one listing of the admissible pairs, linear in their number,
//! and the sweeps, which list the pairs of the few anchors not ruled out,
//! plus the price rounds, each a warm-started solve of every block, plus for
//! each unit of flow moved in step 3 a shortest-path search on the category
//! nodes and `L` searches of each block crossed, `L` being its number of
//! entries. Memory is the arcs kept, about 50 bytes each, and the blocks'
//! flows; the admissible pairs are not held.
//!
//! # References
//!
//! 1. Rosenbaum, P. R., Ross, R. N. and Silber, J. H. (2007). Minimum distance
//!    matched sampling with fine balance in an observational study of
//!    treatment for ovarian cancer. *Journal of the American Statistical
//!    Association* 102, 75-83.
//! 2. Yang, D., Small, D. S., Silber, J. H. and Rosenbaum, P. R. (2012). Optimal
//!    matching with minimal deviation from fine balance in a study of obesity
//!    and surgical outcomes. *Biometrics* 68, 628-636.
//! 3. Pimentel, S. D., Kelz, R. R., Silber, J. H. and Rosenbaum, P. R. (2015).
//!    Large, sparse optimal matching with refined covariate balance in an
//!    observational study of the health outcomes produced by new surgeons.
//!    *Journal of the American Statistical Association* 110, 515-527.
//! 4. Ahuja, R. K., Magnanti, T. L. and Orlin, J. B. (1993). *Network Flows:
//!    Theory, Algorithms, and Applications.* Prentice Hall. Chapter 9.

use super::constraints::ConstraintGroup;
use super::optimal::{
    Edge, OptimalError, OptimalRequest, Stratum, StratumPairs, pair_outcome, with_strata_pairs,
};
use super::records::MatchingRecord;
use crate::types::MatchOutcome;
use epistat_optim::min_cost_flow::{Basis, FlowError, FlowSolution, MinCostFlow};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Whether the balance must be exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BalanceMode {
    /// Minimise the deviation from fine balance first and the distance second.
    #[default]
    Near,
    /// Require fine balance on every level. When it cannot be met the result is
    /// [`OptimalError::Infeasible`], and its message gives the least deviation.
    /// Fine balance matches one set of anchors, so the stratum in that error is
    /// always 0.
    Exact,
}

/// The balance variables of a match.
///
/// Each level is a list of keys of the units' `strata()` maps. A level with
/// several keys is the interaction of them, one category per combination. A
/// unit without a key has its own category for that key, as if the value were
/// a level named "missing".
///
/// The first level has the highest priority. Every later level must be nested
/// in the one before: units that share a category of a level must share the
/// category of the level before. Repeating the earlier keys in each later level
/// ensures it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FineBalance {
    /// The keys of each level, in decreasing order of priority.
    pub levels: Vec<Vec<String>>,
    /// Exact or near-fine balance.
    pub mode: BalanceMode,
}

impl FineBalance {
    /// Near-fine balance on the interaction of `keys`.
    #[must_use]
    pub fn on_keys<S: Into<String>>(keys: impl IntoIterator<Item = S>) -> Self {
        Self {
            levels: vec![keys.into_iter().map(Into::into).collect()],
            mode: BalanceMode::Near,
        }
    }

    /// Require exact fine balance.
    #[must_use]
    pub const fn exact(mut self) -> Self {
        self.mode = BalanceMode::Exact;
        self
    }
}

/// One category of a balance level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BalanceCategory {
    /// The value of each key of the level, in the order of the keys. `None` for
    /// a unit without the key.
    pub values: Vec<Option<String>>,
    /// The number of candidates the category should have: `ratio` times the
    /// anchors in it.
    pub target: usize,
    /// The number of candidates matched in it.
    pub matched: usize,
}

/// The balance of one level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BalanceLevel {
    /// The keys of the level.
    pub keys: Vec<String>,
    /// Its categories, in order of their values.
    pub categories: Vec<BalanceCategory>,
}

impl BalanceLevel {
    /// The sum over categories of the absolute difference between the matched
    /// candidates and the target. It is twice the number of candidates above
    /// target, and 0 for fine balance.
    #[must_use]
    pub fn total_deviation(&self) -> usize {
        self.categories
            .iter()
            .map(|category| category.matched.abs_diff(category.target))
            .sum()
    }
}

/// Pairs matched with balance, and the balance achieved.
#[derive(Debug, Clone)]
pub struct FineBalanceOutcome {
    /// The pairs and their diagnostics.
    pub outcome: MatchOutcome,
    /// The balance of each level, in the order of the levels.
    pub levels: Vec<BalanceLevel>,
}

type Key = Vec<Option<String>>;

struct LevelData {
    keys: Vec<String>,
    /// Categories in order of their values.
    categories: Vec<Key>,
    /// The category of each unit, anchors first and then candidates.
    of_unit: Vec<usize>,
    /// The category one level up of each category, for levels after the first.
    parent: Vec<usize>,
    /// Anchors in each category.
    anchors: Vec<usize>,
}

fn option_error(message: impl Into<String>) -> OptimalError {
    OptimalError::Option {
        option: "fine_balance",
        message: message.into(),
    }
}

/// The rank of each value among the distinct values, from 0 in order.
fn ranks(mut values: Vec<usize>) -> Vec<usize> {
    let mut distinct = values.clone();
    distinct.sort_unstable();
    distinct.dedup();
    for value in &mut values {
        *value = distinct.binary_search(value).unwrap_or(0);
    }
    values
}

fn level_data<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    keys: &[String],
    above: Option<&LevelData>,
) -> Result<LevelData, OptimalError> {
    let units: Vec<&R> = anchors.iter().chain(candidates.iter()).collect();
    // The rank of each unit's key among the keys present, built one key at a
    // time: a rank over the first keys is joined with the rank of the next
    // key's value and ranked again. Values are ranked in order, with a unit
    // that lacks a value before every unit that has one, so the ranks follow
    // the order of the keys as lists.
    let mut of_unit = vec![0_usize; units.len()];
    for key in keys {
        let values: Vec<Option<&str>> = units.iter().map(|record| record.stratum(key)).collect();
        let mut distinct = values.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let width = distinct.len();
        let joined: Vec<usize> = values
            .iter()
            .zip(&of_unit)
            .map(|(value, rank)| rank * width + distinct.binary_search(value).unwrap_or(0))
            .collect();
        of_unit = ranks(joined);
    }
    let count = of_unit.iter().max().map_or(0, |&last| last + 1);
    let mut first_unit = vec![usize::MAX; count];
    for (unit, &category) in of_unit.iter().enumerate() {
        if first_unit[category] == usize::MAX {
            first_unit[category] = unit;
        }
    }
    let categories: Vec<Key> = first_unit
        .iter()
        .map(|&unit| {
            keys.iter()
                .map(|key| units[unit].stratum(key).map(str::to_owned))
                .collect()
        })
        .collect();
    let mut parent = vec![usize::MAX; categories.len()];
    if let Some(above) = above {
        for (unit, category) in of_unit.iter().enumerate() {
            let up = above.of_unit[unit];
            if parent[*category] == usize::MAX {
                parent[*category] = up;
            } else if parent[*category] != up {
                return Err(option_error(format!(
                    "the level with keys {keys:?} is not nested in the level before it: \
                     units of one category {:?} fall in different categories above. \
                     Add the keys of the earlier levels to this one.",
                    categories[*category]
                )));
            }
        }
    }
    let mut anchor_counts = vec![0_usize; categories.len()];
    for category in &of_unit[..anchors.len()] {
        anchor_counts[*category] += 1;
    }
    Ok(LevelData {
        keys: keys.to_vec(),
        categories,
        of_unit,
        parent,
        anchors: anchor_counts,
    })
}

fn to_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Cost scale and penalties, with the largest cost within the solver's range.
struct Costs {
    scale: f64,
    /// Penalty per unit above target, by level.
    penalty: Vec<i64>,
}

fn costs(max_distance: f64, n: usize, levels: usize, nodes: usize) -> Option<Costs> {
    let n = u128::try_from(n).ok()?;
    let nodes = u128::try_from(nodes).ok()?;
    for bits in (10..=40_i32).rev() {
        let scale = if max_distance > 0.0 && max_distance.is_finite() {
            let exponent = f64::from(bits) - max_distance.log2();
            // The exponent is a small whole number once floored.
            #[expect(clippy::cast_possible_truncation, reason = "range is clamped")]
            let power = exponent.floor().clamp(-1000.0, 1000.0) as i32;
            2.0_f64.powi(power)
        } else {
            1.0
        };
        // The scaled distance of any pair is at most 2^bits plus rounding.
        let largest = (1_u128 << bits) + 1;
        let mut penalty = vec![0_u128; levels];
        let mut previous = n.saturating_mul(largest).saturating_add(1);
        for slot in penalty.iter_mut().rev() {
            *slot = previous;
            previous = n.saturating_mul(previous).saturating_add(1);
        }
        let top = penalty.first().copied().unwrap_or(1);
        // The solver refuses 4 (nodes + 2)(largest cost + 1) above i64::MAX / 4.
        if top.saturating_add(1).saturating_mul(4 * (nodes + 2)) <= (1_u128 << 61) {
            return Some(Costs {
                scale,
                penalty: penalty
                    .into_iter()
                    .map(|value| i64::try_from(value).unwrap_or(i64::MAX))
                    .collect(),
            });
        }
    }
    None
}

/// Which pairs enter the network.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arcs {
    /// The nearest candidates of each anchor in each category of the finest
    /// level, then every pair that prices negatively against the potentials of
    /// the solution. The optimum is that of the network with every pair.
    Pruned,
    /// Every admissible pair.
    #[cfg(test)]
    All,
}

/// Candidates kept per anchor and finest-level category at the start, as a
/// multiple of the match ratio.
const FIRST_PER_CATEGORY: usize = 3;

/// Rounds of price adjustment for the start, at most.
const ROUNDS: usize = 40;

/// The price rounds stop when this many rounds in a row come no nearer the
/// targets than the best so far.
const PATIENCE: usize = 8;

const NONE: u32 = u32::MAX;

/// The admissible pairs, listed one stratum at a time when they are needed
/// rather than held. The pairs are numbered stratum by stratum, in the order
/// of each stratum's list, so pair `k` is pair `k - offset[s]` of stratum `s`.
struct Pairs<'a> {
    strata: &'a [Stratum],
    list: &'a StratumPairs<'a>,
    /// The number of the first pair of each stratum, and the total at the end.
    offset: Vec<usize>,
    /// For each stratum, the position in its list of the first pair of each
    /// row, and the number of pairs at the end.
    row_start: Vec<Vec<u32>>,
}

impl Pairs<'_> {
    fn total(&self) -> usize {
        self.offset[self.strata.len()]
    }

    /// The pairs of the rows `rows` (increasing) of stratum `s` as (number,
    /// anchor, candidate, distance), numbered as in the list of all its pairs.
    fn of_rows(&self, s: usize, rows: &[usize]) -> Vec<(usize, usize, usize, f64)> {
        let stratum = &self.strata[s];
        let starts = &self.row_start[s];
        let mut row = usize::MAX;
        let mut next = 0;
        (self.list)(stratum, Some(rows))
            .into_iter()
            .map(|edge| {
                if edge.row != row {
                    row = edge.row;
                    next = starts[row] as usize;
                }
                let k = self.offset[s] + next;
                next += 1;
                (
                    k,
                    stratum.anchors[edge.row],
                    stratum.candidates[edge.col],
                    edge.distance,
                )
            })
            .collect()
    }

    /// The pairs of stratum `s` as (anchor, candidate, distance), by position
    /// in the anchor and candidate slices.
    fn of(&self, s: usize) -> Vec<(usize, usize, f64)> {
        let stratum = &self.strata[s];
        (self.list)(stratum, None)
            .into_iter()
            .map(|edge| {
                (
                    stratum.anchors[edge.row],
                    stratum.candidates[edge.col],
                    edge.distance,
                )
            })
            .collect()
    }
}

/// A pair in the network: its number among all pairs, its units by position in
/// the slices, and its integer cost.
#[derive(Clone, Copy)]
struct Kept {
    k: usize,
    anchor: usize,
    candidate: usize,
    cost: i64,
}

/// A set of pair numbers.
struct PairSet(Vec<u64>);

impl PairSet {
    fn new(total: usize) -> Self {
        Self(vec![0; total.div_ceil(64)])
    }
    fn insert(&mut self, k: usize) {
        self.0[k / 64] |= 1 << (k % 64);
    }
    fn contains(&self, k: usize) -> bool {
        self.0[k / 64] >> (k % 64) & 1 == 1
    }
}

/// Lists of numbers, one per key, stored end to end.
struct Lists {
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Lists {
    /// The items of `(key, item)` grouped by key, each list in input order.
    fn new(keys: usize, entries: &[(usize, u32)]) -> Self {
        let mut start = vec![0_u32; keys + 1];
        for &(key, _) in entries {
            start[key + 1] += 1;
        }
        for key in 0..keys {
            start[key + 1] += start[key];
        }
        let mut fill: Vec<u32> = start[..keys].to_vec();
        let mut items = vec![0; entries.len()];
        for &(key, item) in entries {
            items[fill[key] as usize] = item;
            fill[key] += 1;
        }
        Self { start, items }
    }

    fn of(&self, key: usize) -> &[u32] {
        &self.items[self.start[key] as usize..self.start[key + 1] as usize]
    }
}

fn index(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(NONE)
}

/// A shortest-path search inside a block.
struct Search {
    /// Reduced distance of each node, `i64::MAX` when not reached.
    dist: Vec<i64>,
    /// The node before each node on its path, `NONE` at a seed.
    pred: Vec<u32>,
    /// The pair whose arc reached the node, `NONE` for an arc between an
    /// entry and a candidate.
    via: Vec<u32>,
}

/// One stratum's part of the network, with local node numbers: an entry for
/// each finest-level category of its candidates, then the candidates, then
/// the anchors. The entry of category `k` is joined to the category's node by
/// an arc of unbounded capacity and no cost; the entry has an arc to each of
/// its candidates, and a candidate has an arc to each anchor it is kept with.
struct Block {
    /// The finest-level category of each entry, increasing.
    leaf: Vec<usize>,
    /// Anchors by position in the anchor slice, increasing.
    anchors: Vec<usize>,
    /// Candidates by position in the candidate slice, increasing.
    candidates: Vec<usize>,
    /// The entry of each candidate.
    entry_of: Vec<u32>,
    /// The pairs, in order of their numbers.
    arcs: Vec<Kept>,
    /// The local candidate and anchor of each pair.
    from: Vec<u32>,
    to: Vec<u32>,
    /// Whether each pair is matched.
    on: Vec<bool>,
    /// Whether each candidate is matched.
    taken: Vec<bool>,
    /// Matched candidates of each entry.
    count: Vec<i64>,
    /// Node potentials for which every arc of the block with room for flow has
    /// a non-negative reduced cost.
    pi: Vec<i64>,
    entry_candidates: Lists,
    candidate_arcs: Lists,
    anchor_arcs: Lists,
}

impl Block {
    /// A block of the pairs `arcs`, sorted by number, with no flow.
    fn new(arcs: Vec<Kept>, leaf_of: &[usize]) -> Self {
        let mut anchors: Vec<usize> = arcs.iter().map(|pair| pair.anchor).collect();
        anchors.sort_unstable();
        anchors.dedup();
        let mut candidates: Vec<usize> = arcs.iter().map(|pair| pair.candidate).collect();
        candidates.sort_unstable();
        candidates.dedup();
        let mut leaf: Vec<usize> = candidates.iter().map(|&c| leaf_of[c]).collect();
        leaf.sort_unstable();
        leaf.dedup();
        let entry_of: Vec<u32> = candidates
            .iter()
            .map(|&c| index(leaf.binary_search(&leaf_of[c]).unwrap_or(0)))
            .collect();
        let from: Vec<u32> = arcs
            .iter()
            .map(|pair| index(candidates.binary_search(&pair.candidate).unwrap_or(0)))
            .collect();
        let to: Vec<u32> = arcs
            .iter()
            .map(|pair| index(anchors.binary_search(&pair.anchor).unwrap_or(0)))
            .collect();
        let entry_candidates = Lists::new(
            leaf.len(),
            &entry_of
                .iter()
                .enumerate()
                .map(|(c, &e)| (e as usize, index(c)))
                .collect::<Vec<_>>(),
        );
        let candidate_arcs = Lists::new(
            candidates.len(),
            &from
                .iter()
                .enumerate()
                .map(|(i, &c)| (c as usize, index(i)))
                .collect::<Vec<_>>(),
        );
        let anchor_arcs = Lists::new(
            anchors.len(),
            &to.iter()
                .enumerate()
                .map(|(i, &a)| (a as usize, index(i)))
                .collect::<Vec<_>>(),
        );
        let nodes = leaf.len() + candidates.len() + anchors.len();
        Self {
            on: vec![false; arcs.len()],
            taken: vec![false; candidates.len()],
            count: vec![0; leaf.len()],
            pi: vec![0; nodes],
            leaf,
            anchors,
            candidates,
            entry_of,
            arcs,
            from,
            to,
            entry_candidates,
            candidate_arcs,
            anchor_arcs,
        }
    }

    /// This block with the pairs `more` added, sorted by number, and the flow
    /// of this block.
    fn with(&self, more: &[Kept], leaf_of: &[usize]) -> Self {
        let mut arcs = Vec::with_capacity(self.arcs.len() + more.len());
        let (mut i, mut j) = (0, 0);
        while i < self.arcs.len() || j < more.len() {
            if j == more.len() || (i < self.arcs.len() && self.arcs[i].k < more[j].k) {
                arcs.push(self.arcs[i]);
                i += 1;
            } else {
                arcs.push(more[j]);
                j += 1;
            }
        }
        let mut block = Self::new(arcs, leaf_of);
        let on: std::collections::HashSet<usize> = self
            .arcs
            .iter()
            .zip(&self.on)
            .filter(|(_, on)| **on)
            .map(|(pair, _)| pair.k)
            .collect();
        for (slot, pair) in block.arcs.iter().enumerate() {
            block.on[slot] = on.contains(&pair.k);
        }
        for (c, &candidate) in self.candidates.iter().enumerate() {
            if self.taken[c] {
                let local = block.candidates.binary_search(&candidate).unwrap_or(0);
                block.taken[local] = true;
                block.count[block.entry_of[local] as usize] += 1;
            }
        }
        block
    }

    fn entries(&self) -> usize {
        self.leaf.len()
    }

    fn nodes(&self) -> usize {
        self.pi.len()
    }

    fn candidate_node(&self, c: usize) -> usize {
        self.entries() + c
    }

    fn anchor_node(&self, a: usize) -> usize {
        self.entries() + self.candidates.len() + a
    }

    /// The block as a flow problem of its own, with a root that supplies
    /// every anchor through the entries. The arc from the root to the entry of
    /// category `k` costs `price[k]`, less the least price of the block's
    /// entries, and has room for more than all the flow. The arcs are those
    /// to the entries, then those to the candidates, then the pairs.
    fn problem(&self, ratio: usize, price: &[i64]) -> MinCostFlow {
        let (e, c, a) = (self.entries(), self.candidates.len(), self.anchors.len());
        let root = e + c + a;
        let mut flow = MinCostFlow::new(root + 1);
        flow.reserve_arcs(e + c + self.arcs.len());
        flow.set_supply(root, to_i64(ratio * a));
        for anchor in 0..a {
            flow.set_supply(e + c + anchor, -to_i64(ratio));
        }
        let least = self.leaf.iter().map(|&k| price[k]).min().unwrap_or(0);
        for (entry, &k) in self.leaf.iter().enumerate() {
            flow.add_arc(root, entry, to_i64(ratio * a + 1), price[k] - least);
        }
        for (candidate, &entry) in self.entry_of.iter().enumerate() {
            flow.add_arc(entry as usize, e + candidate, 1, 0);
        }
        for (slot, pair) in self.arcs.iter().enumerate() {
            flow.add_arc(
                e + self.from[slot] as usize,
                e + c + self.to[slot] as usize,
                1,
                pair.cost,
            );
        }
        flow
    }

    /// The flow of the block on the arcs of [`Self::problem`].
    fn flow_values(&self) -> Vec<i64> {
        let mut values = self.count.clone();
        values.extend(self.taken.iter().map(|&t| i64::from(t)));
        values.extend(self.on.iter().map(|&o| i64::from(o)));
        values
    }

    /// Take the flow and the potentials of a solution of [`Self::problem`].
    fn adopt(&mut self, solution: &FlowSolution) {
        let (e, c) = (self.entries(), self.candidates.len());
        self.count.copy_from_slice(&solution.flow[..e]);
        for (slot, taken) in self.taken.iter_mut().enumerate() {
            *taken = solution.flow[e + slot] == 1;
        }
        for (slot, on) in self.on.iter_mut().enumerate() {
            *on = solution.flow[e + c + slot] == 1;
        }
        let nodes = self.nodes();
        self.pi.copy_from_slice(&solution.potentials[..nodes]);
    }

    /// Calls `f(head, cost, via)` for every arc out of `node` with room for
    /// flow, `via` being the pair of the arc or `NONE`.
    fn each_arc(&self, node: usize, mut f: impl FnMut(usize, i64, u32)) {
        let (e, c) = (self.entries(), self.candidates.len());
        if node < e {
            for &candidate in self.entry_candidates.of(node) {
                if !self.taken[candidate as usize] {
                    f(e + candidate as usize, 0, NONE);
                }
            }
        } else if node < e + c {
            let candidate = node - e;
            if self.taken[candidate] {
                f(self.entry_of[candidate] as usize, 0, NONE);
            }
            for &slot in self.candidate_arcs.of(candidate) {
                if !self.on[slot as usize] {
                    let anchor = self.to[slot as usize] as usize;
                    f(e + c + anchor, self.arcs[slot as usize].cost, slot);
                }
            }
        } else {
            for &slot in self.anchor_arcs.of(node - e - c) {
                if self.on[slot as usize] {
                    let candidate = self.from[slot as usize] as usize;
                    f(e + candidate, -self.arcs[slot as usize].cost, slot);
                }
            }
        }
    }

    /// Dijkstra's method on the reduced costs, from `seeds` with their
    /// starting distances. Ties are settled by node number.
    fn search(&self, seeds: &[(usize, i64)]) -> Search {
        let nodes = self.nodes();
        let mut dist = vec![i64::MAX; nodes];
        let mut pred = vec![NONE; nodes];
        let mut via = vec![NONE; nodes];
        let mut heap = std::collections::BinaryHeap::new();
        for &(node, start) in seeds {
            if start < dist[node] {
                dist[node] = start;
                heap.push(std::cmp::Reverse((start, node)));
            }
        }
        while let Some(std::cmp::Reverse((d, u))) = heap.pop() {
            if d > dist[u] {
                continue;
            }
            self.each_arc(u, |v, cost, arc| {
                let reduced = cost + self.pi[u] - self.pi[v];
                let next = d + reduced;
                if next < dist[v] {
                    dist[v] = next;
                    pred[v] = index(u);
                    via[v] = arc;
                    heap.push(std::cmp::Reverse((next, v)));
                }
            });
        }
        Search { dist, pred, via }
    }

    /// The path of `search` to `node`, from its seed: each node with the pair
    /// that reached it.
    fn path(search: &Search, node: usize) -> Vec<(usize, u32)> {
        let mut path = vec![(node, search.via[node])];
        let mut at = node;
        while search.pred[at] != NONE {
            at = search.pred[at] as usize;
            path.push((at, search.via[at]));
        }
        path.reverse();
        path
    }

    /// The least cost of moving one matched candidate from category `j` to
    /// category `k` in this block, for every pair of entries: one unit enters
    /// at the entry of `k` and leaves at the entry of `j`. The result is
    /// `(k, j, cost)` by finest-level category.
    fn exchanges(&self) -> Vec<(usize, usize, i64)> {
        let mut out = Vec::new();
        for entry in 0..self.entries() {
            let search = self.search(&[(entry, 0)]);
            for exit in 0..self.entries() {
                if exit != entry && search.dist[exit] != i64::MAX {
                    out.push((
                        self.leaf[entry],
                        self.leaf[exit],
                        search.dist[exit] - self.pi[entry] + self.pi[exit],
                    ));
                }
            }
        }
        out
    }

    /// Whether every arc of the block with room for flow has a non-negative
    /// reduced cost.
    fn reduced_costs_hold(&self) -> bool {
        let mut hold = true;
        for u in 0..self.nodes() {
            self.each_arc(u, |v, cost, _| hold &= cost + self.pi[u] - self.pi[v] >= 0);
        }
        hold
    }

    /// The anchors of the block, by position in the anchor slice, that may
    /// have a pair left out with a negative reduced cost against the
    /// potentials `d` of [`Self::network_potentials`]. A pair of an anchor
    /// and category left out costs at least the bound of that anchor and
    /// category, every candidate of the category has at least the least
    /// potential among the block's candidates of the category, its entry, or
    /// (without an entry) the category node `hub[k]`, and the anchors and
    /// categories without a bound have no pair left out.
    fn wanting(&self, bounds: &[(usize, usize, i64)], d: &[i64], hub: &[i64]) -> Vec<usize> {
        let mut least: Vec<i64> = d[..self.entries()].to_vec();
        for (c, &entry) in self.entry_of.iter().enumerate() {
            let entry = entry as usize;
            least[entry] = least[entry].min(d[self.candidate_node(c)]);
        }
        let mut out: Vec<usize> = bounds
            .iter()
            .filter(|&&(anchor, k, bound)| {
                let lowest = match self.leaf.binary_search(&k) {
                    Ok(entry) => least[entry],
                    Err(_) => hub[k],
                };
                self.anchors.binary_search(&anchor).ok().is_none_or(|a| {
                    i128::from(bound) + i128::from(lowest) - i128::from(d[self.anchor_node(a)]) < 0
                })
            })
            .map(|&(anchor, _, _)| anchor)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Restore non-negative reduced costs by label correction from the
    /// current potentials. The block has no cycle of negative cost, so it
    /// ends.
    fn repair(&mut self) {
        let nodes = self.nodes();
        let mut queue: std::collections::VecDeque<usize> = (0..nodes).collect();
        let mut queued = vec![true; nodes];
        while let Some(u) = queue.pop_front() {
            queued[u] = false;
            let mut lowered = Vec::new();
            self.each_arc(u, |v, cost, _| {
                if self.pi[u] + cost < self.pi[v] {
                    lowered.push((v, self.pi[u] + cost));
                }
            });
            for (v, value) in lowered {
                if value < self.pi[v] {
                    self.pi[v] = value;
                    if !queued[v] {
                        queued[v] = true;
                        queue.push_back(v);
                    }
                }
            }
        }
    }

    /// Potentials for the whole network on the nodes of this block: the
    /// shortest distance from the category nodes, with the category node of
    /// `k` at `hub[k]`. A node no category reaches gets its own potential
    /// shifted by the least amount that keeps the arcs out of it
    /// non-negative.
    fn network_potentials(&self, hub: &[i64]) -> Vec<i64> {
        let seeds: Vec<(usize, i64)> = self
            .leaf
            .iter()
            .enumerate()
            .map(|(entry, &k)| (entry, hub[k] - self.pi[entry]))
            .collect();
        let search = self.search(&seeds);
        let mut out: Vec<i64> = (0..self.nodes())
            .map(|v| {
                if search.dist[v] == i64::MAX {
                    i64::MIN
                } else {
                    search.dist[v] + self.pi[v]
                }
            })
            .collect();
        let mut shift: Option<i64> = None;
        for u in 0..self.nodes() {
            if search.dist[u] != i64::MAX {
                continue;
            }
            self.each_arc(u, |v, cost, _| {
                if search.dist[v] != i64::MAX {
                    let need = out[v] - cost - self.pi[u];
                    shift = Some(shift.map_or(need, |s| s.max(need)));
                }
            });
        }
        for (u, slot) in out.iter_mut().enumerate() {
            if search.dist[u] == i64::MAX {
                *slot = self.pi[u] + shift.unwrap_or(0);
            }
        }
        out
    }
}

/// An arc between category nodes, or from the sink to a category.
struct HubArc {
    from: usize,
    to: usize,
    cap: i64,
    cost: i64,
    flow: i64,
}

/// A step of a shortest path between category nodes.
#[derive(Clone, Copy)]
enum Step {
    None,
    /// Along a hub arc, forward or against its flow.
    Hub {
        arc: usize,
        forward: bool,
    },
    /// Through a block, from the entry of one finest-level category to that of
    /// another.
    Exchange {
        block: usize,
        from: usize,
        to: usize,
    },
}

/// A node of the whole network on an augmenting path.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Node {
    Hub(usize),
    Local(usize, usize),
}

/// How a path reaches a node.
#[derive(Clone, Copy)]
enum Move {
    Start,
    Hub {
        arc: usize,
        forward: bool,
    },
    /// From a category node into the entry of a block.
    Enter {
        block: usize,
        entry: usize,
    },
    /// From the entry of a block to its category node.
    Leave {
        block: usize,
        entry: usize,
    },
    /// Along an arc of a block.
    Inside {
        block: usize,
        from: usize,
        to: usize,
        via: u32,
    },
}

#[cfg(test)]
fn always_repair() -> bool {
    tests::always_repair()
}

#[cfg(not(test))]
const fn always_repair() -> bool {
    false
}

/// The state of the decomposed solve: the blocks, the arcs between the
/// category nodes, and for each pair of finest-level categories the cost of
/// the cheapest exchange in each block.
struct Coupled {
    blocks: Vec<Block>,
    /// Hub node of the first category of each level; node 0 is the sink.
    first: Vec<usize>,
    hub: Vec<HubArc>,
    /// Hub arcs at each hub node, with whether the node is the arc's tail.
    adjacent: Vec<Vec<(usize, bool)>>,
    excess: Vec<i64>,
    /// `(k, j)` to the exchange costs and blocks, cheapest first.
    exchange: BTreeMap<(usize, usize), BTreeSet<(i64, usize)>>,
    /// The entries of `exchange` from each block.
    block_exchange: Vec<Vec<(usize, usize, i64)>>,
}

impl Coupled {
    fn hub_nodes(&self) -> usize {
        self.excess.len()
    }

    fn leaf_node(&self, k: usize) -> usize {
        self.first[self.first.len() - 1] + k
    }

    /// The blocks' exchanges into the table, replacing their old entries.
    fn refresh(&mut self, touched: &[usize]) {
        let fresh: Vec<Vec<(usize, usize, i64)>> = touched
            .par_iter()
            .map(|&b| self.blocks[b].exchanges())
            .collect();
        for (&b, entries) in touched.iter().zip(fresh) {
            for &(k, j, cost) in &self.block_exchange[b] {
                let key = (k, j);
                if let Some(set) = self.exchange.get_mut(&key) {
                    set.remove(&(cost, b));
                    if set.is_empty() {
                        self.exchange.remove(&key);
                    }
                }
            }
            for &(k, j, cost) in &entries {
                self.exchange.entry((k, j)).or_default().insert((cost, b));
            }
            self.block_exchange[b] = entries;
        }
    }

    /// Calls `f(head, cost, step)` for every arc out of hub node `u` with room
    /// for flow, the exchanges taking the cheapest block.
    fn each_hub_arc(&self, u: usize, mut f: impl FnMut(usize, i64, Step)) {
        for &(arc, tail) in &self.adjacent[u] {
            let a = &self.hub[arc];
            if tail && a.flow < a.cap {
                f(a.to, a.cost, Step::Hub { arc, forward: true });
            } else if !tail && a.flow > 0 {
                f(
                    a.from,
                    -a.cost,
                    Step::Hub {
                        arc,
                        forward: false,
                    },
                );
            }
        }
        let leaves = self.first[self.first.len() - 1];
        if u >= leaves {
            let k = u - leaves;
            for (&(_, j), set) in self.exchange.range((k, 0)..(k + 1, 0)) {
                if let Some(&(cost, block)) = set.first() {
                    f(
                        leaves + j,
                        cost,
                        Step::Exchange {
                            block,
                            from: k,
                            to: j,
                        },
                    );
                }
            }
        }
    }

    /// Shortest distances between hub nodes by label correction from the
    /// labels `dist`, with the step into each node. There is no cycle of
    /// negative cost, so it ends.
    fn label(&self, mut dist: Vec<i64>) -> (Vec<i64>, Vec<Step>) {
        let nodes = self.hub_nodes();
        let mut pred = vec![Step::None; nodes];
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let mut queued = vec![false; nodes];
        for v in 0..nodes {
            if dist[v] != i64::MAX {
                queue.push_back(v);
                queued[v] = true;
            }
        }
        while let Some(u) = queue.pop_front() {
            queued[u] = false;
            let du = dist[u];
            let mut lowered = Vec::new();
            self.each_hub_arc(u, |v, cost, step| {
                if du + cost < dist[v] {
                    lowered.push((v, du + cost, step));
                }
            });
            for (v, value, step) in lowered {
                if value < dist[v] {
                    dist[v] = value;
                    pred[v] = step;
                    if !queued[v] {
                        queued[v] = true;
                        queue.push_back(v);
                    }
                }
            }
        }
        (dist, pred)
    }

    fn tail_of(&self, step: Step) -> usize {
        match step {
            Step::Hub { arc, forward } => {
                if forward {
                    self.hub[arc].from
                } else {
                    self.hub[arc].to
                }
            }
            Step::Exchange { from, .. } => self.leaf_node(from),
            Step::None => usize::MAX,
        }
    }

    /// Successive shortest paths: while a hub node has flow to spare, send it
    /// along a cheapest path to a hub node short of flow.
    fn balance(&mut self) -> Result<(), OptimalError> {
        let nodes = self.hub_nodes();
        while let Some(source) = (0..nodes).find(|&v| self.excess[v] > 0) {
            let mut start = vec![i64::MAX; nodes];
            start[source] = 0;
            let (dist, pred) = self.label(start);
            let Some(target) = (0..nodes)
                .filter(|&v| self.excess[v] < 0 && dist[v] != i64::MAX)
                .min_by_key(|&v| (dist[v], v))
            else {
                return Err(OptimalError::Infeasible {
                    stratum: 0,
                    message: "the categories cannot be given their candidates".into(),
                });
            };
            let mut steps = Vec::new();
            let mut at = target;
            while at != source {
                let step = pred[at];
                steps.push(step);
                at = self.tail_of(step);
            }
            steps.reverse();
            self.augment(source, target, &steps);
        }
        Ok(())
    }

    /// Send flow along the hub path `steps` from `source` to `target`, each
    /// exchange taken along the shortest path of its block.
    fn augment(&mut self, source: usize, target: usize, steps: &[Step]) {
        let mut path: Vec<(Node, Move)> = vec![(Node::Hub(source), Move::Start)];
        let mut at: std::collections::HashMap<Node, usize> = std::collections::HashMap::new();
        at.insert(Node::Hub(source), 0);
        let mut cut = false;
        let mut segments: Vec<(usize, Search, usize)> = Vec::new();
        let mut push = |path: &mut Vec<(Node, Move)>,
                        at: &mut std::collections::HashMap<Node, usize>,
                        node: Node,
                        how: Move| {
            if let Some(&position) = at.get(&node) {
                // A node met twice closes a loop of zero cost; drop it.
                for (dropped, _) in path.drain(position + 1..) {
                    at.remove(&dropped);
                }
                cut = true;
            } else {
                at.insert(node, path.len());
                path.push((node, how));
            }
        };
        for &step in steps {
            match step {
                Step::Hub { arc, forward } => {
                    let head = if forward {
                        self.hub[arc].to
                    } else {
                        self.hub[arc].from
                    };
                    push(
                        &mut path,
                        &mut at,
                        Node::Hub(head),
                        Move::Hub { arc, forward },
                    );
                }
                Step::Exchange { block, from, to } => {
                    let b = &self.blocks[block];
                    let entry = b.leaf.binary_search(&from).unwrap_or(0);
                    let exit = b.leaf.binary_search(&to).unwrap_or(0);
                    let search = b.search(&[(entry, 0)]);
                    let inner = Block::path(&search, exit);
                    push(
                        &mut path,
                        &mut at,
                        Node::Local(block, entry),
                        Move::Enter { block, entry },
                    );
                    for pair in inner.windows(2) {
                        let ((u, _), (v, via)) = (pair[0], pair[1]);
                        push(
                            &mut path,
                            &mut at,
                            Node::Local(block, v),
                            Move::Inside {
                                block,
                                from: u,
                                to: v,
                                via,
                            },
                        );
                    }
                    push(
                        &mut path,
                        &mut at,
                        Node::Hub(self.leaf_node(to)),
                        Move::Leave { block, entry: exit },
                    );
                    segments.push((block, search, exit));
                }
                Step::None => {}
            }
        }
        let mut amount = self.excess[source].min(-self.excess[target]);
        for &(_, how) in &path {
            match how {
                Move::Hub { arc, forward } => {
                    let a = &self.hub[arc];
                    amount = amount.min(if forward { a.cap - a.flow } else { a.flow });
                }
                Move::Leave { block, entry } => {
                    amount = amount.min(self.blocks[block].count[entry]).min(1);
                }
                Move::Inside { .. } | Move::Enter { .. } => amount = amount.min(1),
                Move::Start => {}
            }
        }
        let mut touched: Vec<usize> = Vec::new();
        for &(_, how) in &path {
            match how {
                Move::Hub { arc, forward } => {
                    self.hub[arc].flow += if forward { amount } else { -amount };
                }
                Move::Enter { block, entry } => {
                    self.blocks[block].count[entry] += amount;
                    touched.push(block);
                }
                Move::Leave { block, entry } => {
                    self.blocks[block].count[entry] -= amount;
                    touched.push(block);
                }
                Move::Inside {
                    block,
                    from,
                    to,
                    via,
                } => {
                    let b = &mut self.blocks[block];
                    let (e, c) = (b.entries(), b.candidates.len());
                    if via != NONE {
                        b.on[via as usize] = from < e + c;
                    } else if from < e {
                        b.taken[to - e] = true;
                    } else {
                        b.taken[from - e] = false;
                    }
                    touched.push(block);
                }
                Move::Start => {}
            }
        }
        self.excess[source] -= amount;
        self.excess[target] += amount;
        touched.sort_unstable();
        touched.dedup();
        for &block in &touched {
            let mine: Vec<&(usize, Search, usize)> =
                segments.iter().filter(|s| s.0 == block).collect();
            let b = &mut self.blocks[block];
            if !cut && mine.len() == 1 && !always_repair() {
                // The segment is a shortest path from its entry, so these
                // potentials keep every reduced cost non-negative.
                let (_, search, exit) = mine[0];
                let cap = search.dist[*exit];
                for (pi, &d) in b.pi.iter_mut().zip(&search.dist) {
                    *pi += d.min(cap);
                }
            } else {
                b.repair();
            }
            debug_assert!(b.reduced_costs_hold(), "potentials of block {block}");
        }
        self.refresh(&touched);
    }

    /// Potentials of the hub nodes that keep every reduced cost of the
    /// network non-negative.
    fn hub_potentials(&self) -> Vec<i64> {
        self.label(vec![0; self.hub_nodes()]).0
    }

    /// The total cost of the flow, pairs and penalties.
    fn cost(&self) -> i128 {
        let pairs: i128 = self
            .blocks
            .iter()
            .flat_map(|b| b.arcs.iter().zip(&b.on))
            .filter(|(_, on)| **on)
            .map(|(pair, _)| i128::from(pair.cost))
            .sum();
        let hub: i128 = self
            .hub
            .iter()
            .map(|a| i128::from(a.cost) * i128::from(a.flow))
            .sum();
        pairs + hub
    }
}

/// The pairs of a stratum kept at the start, as (position in the stratum's
/// list, anchor, candidate, distance), and for every anchor and finest-level category with pairs left
/// out, the distance of the last pair kept, which no pair left out is below.
struct Nearest {
    pairs: Vec<(usize, usize, usize, f64)>,
    bounds: Vec<(usize, usize, f64)>,
}

/// For anchors and finest-level categories with pairs left out: (anchor,
/// category, least cost of a pair left out).
type Bounds = Vec<(usize, usize, i64)>;

/// [`Nearest`] with integer costs.
struct Near {
    pairs: Vec<Kept>,
    bounds: Bounds,
}

/// The `per` nearest pairs of every anchor and finest-level category among
/// the pairs of a stratum, ties by candidate, in the order of the stratum's
/// list. The list is grouped by anchor, so each anchor's pairs are split by
/// category and the nearest of each category found by selection, in time
/// linear in the pairs.
fn nearest(pairs: &[(usize, usize, f64)], per: usize, leaf_of: &[usize]) -> Nearest {
    let mut keep = Vec::new();
    let mut bounds = Vec::new();
    let order = |x: &(usize, usize, usize, f64), y: &(usize, usize, usize, f64)| {
        x.3.total_cmp(&y.3).then(x.2.cmp(&y.2))
    };
    let mut group: Vec<(usize, usize, usize, f64)> = Vec::new();
    let mut start = 0;
    while start < pairs.len() {
        let anchor = pairs[start].0;
        let mut stop = start;
        while stop < pairs.len() && pairs[stop].0 == anchor {
            stop += 1;
        }
        group.clear();
        group.extend((start..stop).map(|i| (leaf_of[pairs[i].1], i, pairs[i].1, pairs[i].2)));
        group.sort_unstable_by_key(|pair| pair.0);
        let mut from = 0;
        while from < group.len() {
            let leaf = group[from].0;
            let mut to = from;
            while to < group.len() && group[to].0 == leaf {
                to += 1;
            }
            let run = &mut group[from..to];
            if run.len() > per {
                run.select_nth_unstable_by(per - 1, order);
                let chosen = &mut run[..per];
                chosen.sort_unstable_by(order);
                bounds.push((anchor, leaf, chosen[per - 1].3));
            }
            for pair in &run[..run.len().min(per)] {
                keep.push((pair.1, anchor, pair.2, pair.3));
            }
            from = to;
        }
        start = stop;
    }
    keep.sort_unstable_by_key(|pair| pair.0);
    Nearest {
        pairs: keep,
        bounds,
    }
}

/// The fixed parts of the flow network.
struct Network<'a> {
    na: usize,
    ratio: usize,
    n: usize,
    levels: &'a [LevelData],
    costs: &'a Costs,
}

impl Network<'_> {
    fn pair_cost(&self, distance: f64) -> i64 {
        // The value is a non-negative whole number within the solver's range.
        #[expect(clippy::cast_possible_truncation, reason = "scaled distance fits i64")]
        let cost = (distance.max(0.0) * self.costs.scale).round_ties_even() as i64;
        cost
    }

    fn leaf(&self) -> &LevelData {
        &self.levels[self.levels.len() - 1]
    }

    /// The finest-level category of each candidate.
    fn leaf_of(&self) -> &[usize] {
        &self.leaf().of_unit[self.na..]
    }

    /// The pairs of stratum `s` with their numbers and costs.
    #[cfg(test)]
    fn kept(&self, pairs: &Pairs<'_>, s: usize) -> Vec<Kept> {
        let first = pairs.offset[s];
        pairs
            .of(s)
            .into_iter()
            .enumerate()
            .map(|(i, (anchor, candidate, distance))| Kept {
                k: first + i,
                anchor,
                candidate,
                cost: self.pair_cost(distance),
            })
            .collect()
    }

    /// [`nearest`] for every stratum, with the pairs' costs.
    fn nearest(&self, pairs: &Pairs<'_>, per: usize) -> Vec<Near> {
        let leaf_of = self.leaf_of();
        (0..pairs.strata.len())
            .into_par_iter()
            .map(|s| self.priced(nearest(&pairs.of(s), per, leaf_of), pairs.offset[s]))
            .collect()
    }

    /// The pairs and bounds of `near`, a stratum whose first pair is number
    /// `first`, with integer costs. Rounding keeps the order of the
    /// distances, so a bound stays at or below the cost of every pair left
    /// out.
    fn priced(&self, near: Nearest, first: usize) -> Near {
        Near {
            pairs: near
                .pairs
                .into_iter()
                .map(|(i, anchor, candidate, distance)| Kept {
                    k: first + i,
                    anchor,
                    candidate,
                    cost: self.pair_cost(distance),
                })
                .collect(),
            bounds: near
                .bounds
                .into_iter()
                .map(|(anchor, leaf, distance)| (anchor, leaf, self.pair_cost(distance)))
                .collect(),
        }
    }

    /// Every pair of each stratum.
    #[cfg(test)]
    fn every(&self, pairs: &Pairs<'_>) -> Vec<Near> {
        (0..pairs.strata.len())
            .into_par_iter()
            .map(|s| Near {
                pairs: self.kept(pairs, s),
                bounds: Vec::new(),
            })
            .collect()
    }

    /// The prices of the start and the blocks' flows at those prices. Each
    /// round solves every block on its own with the candidates of category
    /// `k` costing `price[k]` more, from the final tree of the round before,
    /// and moves the prices (resilient propagation, each price with its own
    /// step) until the numbers taken per category meet their targets or
    /// [`PATIENCE`] rounds in a row bring them no nearer. The blocks are left
    /// with the flows of the round nearest the targets. `None` when a block
    /// has no flow.
    fn start(&self, blocks: &mut [Block]) -> Result<Option<Vec<i64>>, OptimalError> {
        let leaf = self.leaf();
        let categories = leaf.categories.len();
        let target: Vec<usize> = leaf.anchors.iter().map(|a| self.ratio * a).collect();
        let unit = {
            let (sum, count) = blocks
                .iter()
                .flat_map(|b| b.arcs.iter())
                .fold((0.0_f64, 0_usize), |(s, c), pair| {
                    (s + pair.cost as f64, c + 1)
                });
            (sum / count.max(1) as f64).max(1.0)
        };
        // Prices stay well inside the penalties, so that they choose only the
        // start.
        let limit = self.costs.penalty[self.levels.len() - 1] / 8;
        let mut price = vec![0_f64; categories];
        let mut step = vec![unit * 1e-4; categories];
        let mut last = vec![0_i8; categories];
        let mut bases: Vec<Option<Basis>> = vec![None; blocks.len()];
        let mut best: Option<(usize, Vec<i64>)> = None;
        let mut since_best = 0;
        for _ in 0..ROUNDS {
            #[expect(clippy::cast_possible_truncation, reason = "the price is clamped")]
            let whole: Vec<i64> = price
                .iter()
                .map(|p| (p.clamp(-1e18, 1e18) as i64).clamp(-limit, limit))
                .collect();
            let solved: Vec<Result<FlowSolution, FlowError>> = blocks
                .par_iter()
                .zip(bases.par_iter())
                .map(|(block, basis)| {
                    let problem = block.problem(self.ratio, &whole);
                    match basis {
                        Some(tree) => problem.solve_from(tree),
                        None => problem.solve(),
                    }
                })
                .collect();
            let mut solutions = Vec::with_capacity(blocks.len());
            for result in solved {
                match result {
                    Ok(solution) => solutions.push(solution),
                    Err(FlowError::Infeasible) => return Ok(None),
                    Err(FlowError::Overflow | FlowError::InvalidInput) => {
                        return Err(OptimalError::TooLarge { stratum: 0 });
                    }
                }
            }
            let mut got = vec![0_usize; categories];
            for (block, solution) in blocks.iter().zip(&solutions) {
                for (entry, &k) in block.leaf.iter().enumerate() {
                    got[k] += usize::try_from(solution.flow[entry]).unwrap_or(0);
                }
            }
            let gap: usize = got.iter().zip(&target).map(|(a, b)| a.abs_diff(*b)).sum();
            let improved = best.as_ref().is_none_or(|b| gap < b.0);
            // The blocks take the flow of the round nearest the targets as it
            // is found, and the next round starts from this round's basis, so
            // nothing of a round is kept beyond those.
            for ((block, basis), solution) in blocks.iter_mut().zip(bases.iter_mut()).zip(solutions)
            {
                if improved {
                    block.adopt(&solution);
                }
                *basis = Some(solution.basis);
            }
            if improved {
                best = Some((gap, whole));
                since_best = 0;
            } else {
                since_best += 1;
            }
            if gap == 0 || since_best == PATIENCE {
                break;
            }
            // Resilient propagation: each price moves by its own step in the
            // direction that reduces its surplus, and the step grows while the
            // direction holds and shrinks when it reverses.
            for k in 0..categories {
                let direction = match got[k].cmp(&target[k]) {
                    std::cmp::Ordering::Greater => 1_i8,
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                };
                if direction != 0 && last[k] != 0 {
                    step[k] *= if direction == last[k] { 1.6 } else { 0.4 };
                }
                price[k] += step[k] * f64::from(direction);
                if direction != 0 {
                    last[k] = direction;
                }
            }
        }
        Ok(Some(
            best.map_or_else(|| vec![0; categories], |(_, whole)| whole),
        ))
    }

    /// The category nodes and their arcs for blocks solved at the prices
    /// `price`. The flow on each hub arc is set from the sign of its reduced
    /// cost against potentials that put each finest-level category at its
    /// price and every coarser one at the least of the categories below it:
    /// full when negative, and for an arc of the target when zero. With the
    /// blocks optimal at those prices, no cycle of the network then has a
    /// negative cost, and what is left is flow to move between category
    /// nodes.
    fn couple(&self, blocks: Vec<Block>, price: &[i64]) -> Coupled {
        let depth = self.levels.len();
        let mut first = Vec::with_capacity(depth);
        let mut next = 1;
        for level in self.levels {
            first.push(next);
            next += level.categories.len();
        }
        let nodes = next;
        let mut potential = vec![i64::MAX; nodes];
        for (k, &value) in price.iter().enumerate() {
            potential[first[depth - 1] + k] = value;
        }
        for d in (1..depth).rev() {
            for c in 0..self.levels[d].categories.len() {
                let up = first[d - 1] + self.levels[d].parent[c];
                potential[up] = potential[up].min(potential[first[d] + c]);
            }
        }
        for c in 0..self.levels[0].categories.len() {
            potential[0] = potential[0].min(potential[first[0] + c]);
        }
        let mut hub = Vec::new();
        for (d, level) in self.levels.iter().enumerate() {
            for c in 0..level.categories.len() {
                let from = if d == 0 {
                    0
                } else {
                    first[d - 1] + level.parent[c]
                };
                let to = first[d] + c;
                let mut add = |cap: i64, cost: i64, fill_when_zero: bool| {
                    let reduced = cost + potential[from] - potential[to];
                    let flow = if reduced < 0 || (reduced == 0 && fill_when_zero) {
                        cap
                    } else {
                        0
                    };
                    hub.push(HubArc {
                        from,
                        to,
                        cap,
                        cost,
                        flow,
                    });
                };
                let target = self.ratio * level.anchors[c];
                if target > 0 {
                    add(to_i64(target), 0, true);
                }
                add(to_i64(self.n), self.costs.penalty[d], false);
            }
        }
        let mut adjacent = vec![Vec::new(); nodes];
        let mut excess = vec![0_i64; nodes];
        excess[0] = to_i64(self.n);
        for (slot, arc) in hub.iter().enumerate() {
            adjacent[arc.from].push((slot, true));
            adjacent[arc.to].push((slot, false));
            excess[arc.from] -= arc.flow;
            excess[arc.to] += arc.flow;
        }
        for block in &blocks {
            for (entry, &k) in block.leaf.iter().enumerate() {
                excess[first[depth - 1] + k] -= block.count[entry];
            }
        }
        let all: Vec<usize> = (0..blocks.len()).collect();
        let mut coupled = Coupled {
            block_exchange: vec![Vec::new(); blocks.len()],
            blocks,
            first,
            hub,
            adjacent,
            excess,
            exchange: BTreeMap::new(),
        };
        coupled.refresh(&all);
        coupled
    }

    /// The pairs left out of the network whose reduced cost against
    /// potentials of the whole network is negative, by stratum in order of
    /// their numbers. `hub` gives the potential of each hub node.
    fn violating(
        &self,
        pairs: &Pairs<'_>,
        selected: &PairSet,
        coupled: &Coupled,
        block_of: &[usize],
        bounds: &[Bounds],
        hub: &[i64],
    ) -> Vec<Vec<Kept>> {
        let leaf_of = self.leaf_of();
        let leaf_hub: Vec<i64> = (0..self.leaf().categories.len())
            .map(|k| hub[coupled.leaf_node(k)])
            .collect();
        let potentials: Vec<Vec<i64>> = coupled
            .blocks
            .par_iter()
            .map(|block| block.network_potentials(&leaf_hub))
            .collect();
        (0..pairs.strata.len())
            .into_par_iter()
            .map(|s| {
                let Some(&b) = block_of.get(s).filter(|&&b| b != usize::MAX) else {
                    return Vec::new();
                };
                let (block, d) = (&coupled.blocks[b], &potentials[b]);
                let stratum = &pairs.strata[s];
                // Rows of the stratum, increasing.
                let rows: Vec<usize> = block
                    .wanting(&bounds[s], d, &leaf_hub)
                    .into_iter()
                    .filter_map(|anchor| stratum.anchors.binary_search(&anchor).ok())
                    .collect();
                let price = |&(k, anchor, candidate, distance): &(usize, usize, usize, f64)| {
                    if selected.contains(k) {
                        return None;
                    }
                    // A candidate outside the block hangs from the entry of
                    // its category, or from the category node.
                    let at_candidate = match block.candidates.binary_search(&candidate) {
                        Ok(c) => d[block.candidate_node(c)],
                        Err(_) => match block.leaf.binary_search(&leaf_of[candidate]) {
                            Ok(entry) => d[entry],
                            Err(_) => leaf_hub[leaf_of[candidate]],
                        },
                    };
                    let a = block.anchors.binary_search(&anchor).ok()?;
                    let at_anchor = d[block.anchor_node(a)];
                    let cost = self.pair_cost(distance);
                    (i128::from(cost) + i128::from(at_candidate) - i128::from(at_anchor) < 0)
                        .then_some(Kept {
                            k,
                            anchor,
                            candidate,
                            cost,
                        })
                };
                let found: Vec<Kept> = if rows.is_empty() {
                    Vec::new()
                } else {
                    pairs.of_rows(s, &rows).iter().filter_map(price).collect()
                };
                if cfg!(debug_assertions) {
                    let every: Vec<usize> = (0..stratum.anchors.len()).collect();
                    let all: Vec<usize> = pairs
                        .of_rows(s, &every)
                        .iter()
                        .filter_map(price)
                        .map(|pair| pair.k)
                        .collect();
                    let ours: Vec<usize> = found.iter().map(|pair| pair.k).collect();
                    debug_assert_eq!(
                        ours, all,
                        "stratum {s}: pairs to add outside the rows listed"
                    );
                }
                found
            })
            .collect()
    }

    /// The matched pairs and the optimal cost.
    fn solve(
        &self,
        pairs: &Pairs<'_>,
        start: Vec<Nearest>,
        arcs: Arcs,
    ) -> Result<(Vec<(usize, usize)>, i128), OptimalError> {
        let total = pairs.total();
        let leaf_of = self.leaf_of();
        let mut per = FIRST_PER_CATEGORY * self.ratio;
        let mut selected = PairSet::new(total);
        let near: Vec<Near> = match arcs {
            Arcs::Pruned => start
                .into_iter()
                .zip(&pairs.offset)
                .map(|(near, &first)| self.priced(near, first))
                .collect(),
            #[cfg(test)]
            Arcs::All => {
                drop(start);
                self.every(pairs)
            }
        };
        let (mut kept, mut bounds): (Vec<Vec<Kept>>, Vec<Bounds>) = near
            .into_iter()
            .map(|near| (near.pairs, near.bounds))
            .unzip();
        let mut in_network = 0;
        for pair in kept.iter().flatten() {
            selected.insert(pair.k);
            in_network += 1;
        }
        loop {
            let everything = in_network == total;
            let mut block_of = vec![usize::MAX; kept.len()];
            let mut lists = Vec::new();
            for (s, list) in kept.iter_mut().enumerate() {
                if !list.is_empty() {
                    block_of[s] = lists.len();
                    lists.push(std::mem::take(list));
                }
            }
            let mut blocks: Vec<Block> = lists
                .into_par_iter()
                .map(|list| Block::new(list, leaf_of))
                .collect();
            let Some(price) = self.start(&mut blocks)? else {
                if everything {
                    return Err(OptimalError::Infeasible {
                        stratum: 0,
                        message: format!(
                            "the anchors cannot all be given {} candidates within the \
                             admissible pairs",
                            self.ratio
                        ),
                    });
                }
                // Too few pairs for the anchors to get their candidates.
                for (list, &b) in kept.iter_mut().zip(&block_of) {
                    if b != usize::MAX {
                        *list = std::mem::take(&mut blocks[b].arcs);
                    }
                }
                per = per.saturating_mul(4);
                for ((list, bound), more) in kept
                    .iter_mut()
                    .zip(bounds.iter_mut())
                    .zip(self.nearest(pairs, per))
                {
                    *bound = more.bounds;
                    for pair in more.pairs {
                        if !selected.contains(pair.k) {
                            selected.insert(pair.k);
                            list.push(pair);
                            in_network += 1;
                        }
                    }
                    list.sort_unstable_by_key(|pair| pair.k);
                }
                continue;
            };
            let mut coupled = self.couple(blocks, &price);
            loop {
                coupled.balance()?;
                if everything {
                    break;
                }
                let hub = coupled.hub_potentials();
                let violating =
                    self.violating(pairs, &selected, &coupled, &block_of, &bounds, &hub);
                if violating.iter().all(Vec::is_empty) {
                    break;
                }
                // The blocks with new pairs are solved again from their flow at
                // prices that are the potentials of the category nodes, which
                // keeps every cycle of the network at a non-negative cost.
                let price: Vec<i64> = (0..self.leaf().categories.len())
                    .map(|k| hub[coupled.leaf_node(k)])
                    .collect();
                let touched: Vec<(usize, Vec<Kept>)> = violating
                    .into_iter()
                    .enumerate()
                    .filter(|(_, more)| !more.is_empty())
                    .map(|(s, more)| (block_of[s], more))
                    .collect();
                for (_, more) in &touched {
                    for pair in more {
                        selected.insert(pair.k);
                    }
                }
                let solved: Vec<Result<Block, FlowError>> = touched
                    .par_iter()
                    .map(|(b, more)| {
                        let mut block = coupled.blocks[*b].with(more, leaf_of);
                        let solution = block
                            .problem(self.ratio, &price)
                            .solve_from_flow(&block.flow_values())?;
                        block.adopt(&solution);
                        Ok(block)
                    })
                    .collect();
                let mut changed = Vec::with_capacity(touched.len());
                for ((b, _), result) in touched.iter().zip(solved) {
                    let block = result.map_err(|_| OptimalError::TooLarge { stratum: 0 })?;
                    let old = std::mem::replace(&mut coupled.blocks[*b], block);
                    for (entry, &k) in old.leaf.iter().enumerate() {
                        let node = coupled.leaf_node(k);
                        coupled.excess[node] += old.count[entry];
                    }
                    for entry in 0..coupled.blocks[*b].entries() {
                        let k = coupled.blocks[*b].leaf[entry];
                        let node = coupled.leaf_node(k);
                        coupled.excess[node] -= coupled.blocks[*b].count[entry];
                    }
                    changed.push(*b);
                }
                coupled.refresh(&changed);
            }
            let matched = coupled
                .blocks
                .iter()
                .flat_map(|b| b.arcs.iter().zip(&b.on))
                .filter(|(_, on)| **on)
                .map(|(pair, _)| (pair.anchor, pair.candidate))
                .collect();
            return Ok((matched, coupled.cost()));
        }
    }
}

/// Optimal pair matching with fine, near-fine or refined balance.
///
/// The anchors are the focal group, as in [`super::match_optimal`]. Every anchor
/// gets `ratio` candidates: where the admissible pairs or the number of
/// candidates do not allow it, the result is [`OptimalError::Infeasible`]
/// (`rcbalance`'s `exclude.treated` is not available). The targets are those of
/// the anchors, so the balance is that of the anchors' distribution of each
/// level, whatever the estimand.
///
/// Among the matches with the least deviation, the one with the least total
/// distance is returned. See the module documentation for the order of the
/// levels.
///
/// # Errors
///
/// [`OptimalError`] for invalid criteria, replacement, a unique key, no levels,
/// a level without keys or not nested in the one before, a match that is not
/// feasible (or, in exact mode, not exactly balanced), a problem too large for
/// the integer arithmetic of the solver, or when no pair is admissible.
pub fn match_fine_balance<R, G>(
    anchors: &[R],
    candidates: &[R],
    request: &OptimalRequest<'_, G>,
    balance: &FineBalance,
) -> Result<FineBalanceOutcome, OptimalError>
where
    R: MatchingRecord,
    G: ConstraintGroup<R> + ?Sized,
{
    run(anchors, candidates, request, balance, Arcs::Pruned).map(|(outcome, _)| outcome)
}

/// The match and the optimal value of the flow, with the pairs entering the
/// network as `arcs` says.
fn run<R, G>(
    anchors: &[R],
    candidates: &[R],
    request: &OptimalRequest<'_, G>,
    balance: &FineBalance,
    arcs: Arcs,
) -> Result<(FineBalanceOutcome, i128), OptimalError>
where
    R: MatchingRecord,
    G: ConstraintGroup<R> + ?Sized,
{
    if balance.levels.is_empty() {
        return Err(option_error("name at least one level of keys"));
    }
    if balance.levels.iter().any(Vec::is_empty) {
        return Err(option_error("every level needs at least one key"));
    }
    let ratio = request.criteria.match_ratio;
    with_strata_pairs(anchors, candidates, request, |strata, list| {
        let mut levels: Vec<LevelData> = Vec::with_capacity(balance.levels.len());
        for keys in &balance.levels {
            let data = level_data(anchors, candidates, keys, levels.last())?;
            levels.push(data);
        }
        let (na, nc) = (anchors.len(), candidates.len());

        // One pass over the pairs of each stratum for their number, the largest
        // distance, the anchors that have a pair and the nearest candidates.
        let per = FIRST_PER_CATEGORY * ratio;
        let leaf_of = &levels[levels.len() - 1].of_unit[na..];
        let mut summaries: Vec<Summary> = strata
            .par_iter()
            .map(|stratum| Summary::of(stratum, &list(stratum, None), per, leaf_of))
            .collect();
        let mut offset = Vec::with_capacity(strata.len() + 1);
        offset.push(0);
        for summary in &summaries {
            offset.push(offset[offset.len() - 1] + summary.pairs);
        }
        let row_start = summaries
            .iter_mut()
            .map(|summary| std::mem::take(&mut summary.row_start))
            .collect();
        let pairs = Pairs {
            strata: &strata,
            list,
            offset,
            row_start,
        };
        if pairs.total() == 0 {
            return Err(OptimalError::NoMatches);
        }
        let n = ratio * na;
        let mut admissible = vec![false; na];
        for summary in &summaries {
            for &anchor in &summary.anchors {
                admissible[anchor] = true;
            }
        }
        if let Some(anchor) = admissible.iter().position(|ok| !ok) {
            return Err(OptimalError::Infeasible {
                stratum: 0,
                message: format!(
                    "anchor {} has no admissible candidate, and every anchor needs {ratio}",
                    anchors[anchor].id()
                ),
            });
        }
        if n > nc {
            return Err(OptimalError::Infeasible {
                stratum: 0,
                message: format!("{na} anchors need {n} candidates and there are {nc}"),
            });
        }

        // The full network has a node per anchor, candidate and category, and
        // the sink; the penalties are set for that many nodes.
        let nodes = na + nc + levels.iter().map(|l| l.categories.len()).sum::<usize>() + 1;
        let max_distance = summaries
            .iter()
            .map(|summary| summary.max_distance)
            .fold(0.0_f64, f64::max);
        let Some(costs) = costs(max_distance, n, levels.len(), nodes) else {
            return Err(OptimalError::TooLarge { stratum: 0 });
        };
        let start: Vec<Nearest> = summaries.into_iter().map(|summary| summary.near).collect();

        let network = Network {
            na,
            ratio,
            n,
            levels: &levels,
            costs: &costs,
        };
        let (matched, optimum) = network.solve(&pairs, start, arcs)?;
        let table = balance_table(&levels, &matched, na, ratio);
        if balance.mode == BalanceMode::Exact
            && let Some((depth, level)) = table
                .iter()
                .enumerate()
                .find(|(_, level)| level.total_deviation() > 0)
        {
            return Err(OptimalError::Infeasible {
                stratum: 0,
                message: format!(
                    "fine balance on level {depth} (keys {:?}) is not possible with these \
                         pairs; the least total deviation is {}",
                    level.keys,
                    level.total_deviation()
                ),
            });
        }
        let outcome = pair_outcome(
            anchors,
            candidates,
            ratio,
            request.criteria.estimand,
            vec![matched],
        );
        Ok((
            FineBalanceOutcome {
                outcome,
                levels: table,
            },
            optimum,
        ))
    })
}

/// What the first pass over a stratum's pairs records.
struct Summary {
    pairs: usize,
    max_distance: f64,
    /// Anchors with a pair, by position in the anchor slice.
    anchors: Vec<usize>,
    near: Nearest,
    /// The position of the first pair of each row, and the count at the end.
    row_start: Vec<u32>,
}

impl Summary {
    fn of(stratum: &Stratum, edges: &[Edge], per: usize, leaf_of: &[usize]) -> Self {
        let mut anchor_seen = vec![false; stratum.anchors.len()];
        let mut anchors = Vec::new();
        let mut max_distance = 0.0_f64;
        let mut row_start = vec![0_u32; stratum.anchors.len() + 1];
        for edge in edges {
            if !anchor_seen[edge.row] {
                anchor_seen[edge.row] = true;
                anchors.push(stratum.anchors[edge.row]);
            }
            row_start[edge.row + 1] += 1;
            max_distance = max_distance.max(edge.distance);
        }
        for row in 0..stratum.anchors.len() {
            row_start[row + 1] += row_start[row];
        }
        let near = nearest(
            &edges
                .iter()
                .map(|edge| {
                    (
                        stratum.anchors[edge.row],
                        stratum.candidates[edge.col],
                        edge.distance,
                    )
                })
                .collect::<Vec<_>>(),
            per,
            leaf_of,
        );
        Self {
            pairs: edges.len(),
            max_distance,
            anchors,
            near,
            row_start,
        }
    }
}

/// The balance of each level for the matched pairs.
fn balance_table(
    levels: &[LevelData],
    matched: &[(usize, usize)],
    na: usize,
    ratio: usize,
) -> Vec<BalanceLevel> {
    let mut table: Vec<BalanceLevel> = Vec::with_capacity(levels.len());
    for level in levels {
        let mut counts = vec![0_usize; level.categories.len()];
        for (_, candidate) in matched {
            counts[level.of_unit[na + candidate]] += 1;
        }
        let mut categories: BTreeMap<&Key, BalanceCategory> = BTreeMap::new();
        for (index, values) in level.categories.iter().enumerate() {
            categories.insert(
                values,
                BalanceCategory {
                    values: values.clone(),
                    target: ratio * level.anchors[index],
                    matched: counts[index],
                },
            );
        }
        table.push(BalanceLevel {
            keys: level.keys.clone(),
            categories: categories.into_values().collect(),
        });
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RapidHashMap;
    use crate::prelude::*;
    use chrono::NaiveDate;

    thread_local! {
        static ALWAYS_REPAIR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Whether the potentials of a block crossed by a path are always found by
    /// label correction, so that tests reach that branch.
    pub(super) fn always_repair() -> bool {
        ALWAYS_REPAIR.with(std::cell::Cell::get)
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            usize::try_from(self.next()).expect("31 bits") % n
        }
    }

    struct Case {
        anchors: Vec<BaseRecord>,
        candidates: Vec<BaseRecord>,
        scores: RapidHashMap<String, f64>,
    }

    /// Anchors and candidates in a few exact strata, each with a coarse and a
    /// fine category. The scores are multiples of `1 / grid`, so that many
    /// distances are exactly equal.
    fn case(rng: &mut Lcg, anchors: usize, candidates: usize, strata: usize, grid: usize) -> Case {
        let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("date");
        let mut scores: RapidHashMap<String, f64> = RapidHashMap::default();
        let mut make = |id: String, rng: &mut Lcg, tilt: usize| {
            let coarse = rng.below(2 + tilt);
            let fine = rng.below(2);
            let mut record = BaseRecord::new(id.clone(), birth);
            record
                .strata
                .insert("s".into(), rng.below(strata).to_string());
            record.strata.insert("a".into(), coarse.to_string());
            record.strata.insert("b".into(), format!("{coarse}{fine}"));
            scores.insert(id, rng.below(grid + 1) as f64 / grid as f64);
            record
        };
        let anchors: Vec<BaseRecord> = (0..anchors)
            .map(|i| make(format!("a{i}"), rng, 0))
            .collect();
        let candidates: Vec<BaseRecord> = (0..candidates)
            .map(|i| make(format!("c{i}"), rng, 1))
            .collect();
        Case {
            anchors,
            candidates,
            scores,
        }
    }

    /// The categories of a level by a key of strings per unit, which the
    /// ranks of `level_data` replace.
    fn level_data_by_strings<R: MatchingRecord>(
        anchors: &[R],
        candidates: &[R],
        keys: &[String],
        above: Option<&LevelData>,
    ) -> Result<LevelData, OptimalError> {
        let key_of = |record: &R| -> Key {
            keys.iter()
                .map(|key| record.stratum(key).map(str::to_owned))
                .collect()
        };
        let unit_keys: Vec<Key> = anchors
            .iter()
            .chain(candidates.iter())
            .map(key_of)
            .collect();
        let mut categories: Vec<Key> = unit_keys.clone();
        categories.sort();
        categories.dedup();
        let of_unit: Vec<usize> = unit_keys
            .iter()
            .map(|key| categories.binary_search(key).unwrap_or(0))
            .collect();
        let mut parent = vec![usize::MAX; categories.len()];
        if let Some(above) = above {
            for (unit, category) in of_unit.iter().enumerate() {
                let up = above.of_unit[unit];
                if parent[*category] == usize::MAX {
                    parent[*category] = up;
                } else if parent[*category] != up {
                    return Err(option_error("not nested"));
                }
            }
        }
        let mut anchor_counts = vec![0_usize; categories.len()];
        for category in &of_unit[..anchors.len()] {
            anchor_counts[*category] += 1;
        }
        Ok(LevelData {
            keys: keys.to_vec(),
            categories,
            of_unit,
            parent,
            anchors: anchor_counts,
        })
    }

    #[test]
    fn level_data_by_ranks_equals_the_keys_of_strings() {
        let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("date");
        let mut rng = Lcg(7);
        for round in 0..40 {
            let units = 1 + rng.below(60);
            let mut records = Vec::new();
            for i in 0..units {
                let mut record = BaseRecord::new(format!("u{i}"), birth);
                // Values of different lengths and prefixes, and some missing.
                for (key, width) in [("a", 3), ("b", 4), ("c", 2)] {
                    let pick = rng.below(width + 1);
                    if pick < width {
                        record
                            .strata
                            .insert(key.into(), "xy".chars().cycle().take(pick + 1).collect());
                    }
                }
                records.push(record);
            }
            let split = rng.below(units + 1);
            let (anchors, candidates) = records.split_at(split);
            let first_keys = vec!["a".to_string()];
            let both = vec!["a".to_string(), "b".to_string()];
            let all = vec!["c".to_string(), "a".to_string(), "b".to_string()];
            let old_first = level_data_by_strings(anchors, candidates, &first_keys, None).unwrap();
            let new_first = level_data(anchors, candidates, &first_keys, None).unwrap();
            assert_eq!(new_first.keys, old_first.keys, "round {round}");
            assert_eq!(new_first.categories, old_first.categories, "round {round}");
            assert_eq!(new_first.of_unit, old_first.of_unit, "round {round}");
            assert_eq!(new_first.parent, old_first.parent, "round {round}");
            assert_eq!(new_first.anchors, old_first.anchors, "round {round}");
            let old_second =
                level_data_by_strings(anchors, candidates, &both, Some(&old_first)).unwrap();
            let new_second = level_data(anchors, candidates, &both, Some(&new_first)).unwrap();
            assert_eq!(
                new_second.categories, old_second.categories,
                "round {round}"
            );
            assert_eq!(new_second.of_unit, old_second.of_unit, "round {round}");
            assert_eq!(new_second.parent, old_second.parent, "round {round}");
            assert_eq!(new_second.anchors, old_second.anchors, "round {round}");
            // Not nested: both functions refuse.
            let old_bad = level_data_by_strings(anchors, candidates, &all, Some(&old_second));
            let new_bad = level_data(anchors, candidates, &all, Some(&new_second));
            assert_eq!(old_bad.is_err(), new_bad.is_err(), "round {round}");
            let old_up = level_data_by_strings(anchors, candidates, &first_keys, Some(&old_second));
            let new_up = level_data(anchors, candidates, &first_keys, Some(&new_second));
            assert_eq!(old_up.is_err(), new_up.is_err(), "round {round}");
        }
    }

    fn matched_pairs(outcome: &FineBalanceOutcome) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = outcome
            .outcome
            .pairs
            .iter()
            .map(|p| (p.case_id.clone(), p.control_id.clone()))
            .collect();
        pairs.sort();
        pairs
    }

    /// The network with the nearest pairs and the pairs that price negatively
    /// has the optimum of the network with every pair: the same total cost and
    /// the same deviation from balance at every level. The pairs agree up to
    /// equal-cost optima.
    /// The decomposed solve, on the pruned pairs and on every pair, has the
    /// optimum of the network with every pair solved as one problem by the
    /// network simplex: the same optimal value and the same deviation from
    /// balance at every level. The pairs agree up to equal-cost optima, so the
    /// total distance is compared instead. The problems have exact ties
    /// (coarse score grids), one to four strata, one and two levels, ratios 1
    /// and 2, and some a caliper.
    #[test]
    fn decomposed_solve_has_the_optimum_of_the_full_network() {
        compare_with_full_network();
    }

    /// As above, with the potentials of every block an augmenting path
    /// crosses found by label correction rather than from the path's search.
    #[test]
    fn label_correction_keeps_the_optimum() {
        ALWAYS_REPAIR.with(|flag| flag.set(true));
        compare_with_full_network();
    }

    fn compare_with_full_network() {
        let mut rng = Lcg(20_261_121);
        let mut compared = 0;
        for round in 0..72 {
            let strata = 1 + round % 4;
            let ratio = 1 + (round / 4) % 2;
            let anchors = 8 + rng.below(40);
            let candidates = ratio * anchors + 5 + rng.below(60);
            let grid = [4, 16, 1000][round % 3];
            let data = case(&mut rng, anchors, candidates, strata, grid);
            let caliper = (round % 5 == 4).then(|| DistanceCaliper::new(0.3).expect("caliper"));
            let distance = DistanceConfig::propensity_score_map(data.scores.clone(), caliper);
            let criteria = MatchingCriteria::builder()
                .match_ratio(ratio)
                .required_strata(vec!["s".to_string()])
                .build();
            let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
            for levels in [
                vec![vec!["a".to_string()]],
                vec![vec!["a".to_string()], vec!["b".to_string()]],
            ] {
                let balance = FineBalance {
                    levels,
                    ..FineBalance::default()
                };
                let full = full_network::run(&data.anchors, &data.candidates, &request, &balance);
                for arcs in [Arcs::Pruned, Arcs::All] {
                    let ours = run(&data.anchors, &data.candidates, &request, &balance, arcs);
                    match (&full, ours) {
                        (Ok((full, full_cost)), Ok((ours, our_cost))) => {
                            compared += 1;
                            assert_eq!(our_cost, *full_cost, "round {round}: optimal value");
                            // Categories that have no target can share their
                            // surplus in different ways at equal cost, so the
                            // deviation of each level is compared, not the
                            // counts.
                            let deviation = |o: &FineBalanceOutcome| -> Vec<usize> {
                                o.levels.iter().map(BalanceLevel::total_deviation).collect()
                            };
                            assert_eq!(deviation(&ours), deviation(full), "round {round}: balance");
                            let (full_pairs, our_pairs) =
                                (matched_pairs(full), matched_pairs(&ours));
                            assert_eq!(full_pairs.len(), our_pairs.len());
                            let key = |id: &str| data.scores[id];
                            let total = |pairs: &[(String, String)]| -> f64 {
                                pairs.iter().map(|(a, c)| (key(a) - key(c)).abs()).sum()
                            };
                            // Equal optimal values mean equal total distance up
                            // to the rounding of the integer costs.
                            assert!(
                                (total(&full_pairs) - total(&our_pairs)).abs() < 1e-6,
                                "round {round}: total distance"
                            );
                        }
                        (Err(a), Err(b)) => {
                            assert_eq!(a.to_string(), b.to_string(), "round {round}");
                        }
                        (a, b) => {
                            panic!("round {round}: full {:?}, ours {:?}", a.is_ok(), b.is_ok())
                        }
                    }
                }
            }
        }
        assert!(compared > 200, "only {compared} problems compared");
    }

    #[test]
    fn decomposed_solve_is_the_same_on_every_thread_count() {
        let mut rng = Lcg(99);
        let data = case(&mut rng, 60, 200, 5, 16);
        let distance = DistanceConfig::propensity_score_map(data.scores.clone(), None);
        let criteria = MatchingCriteria::builder()
            .required_strata(vec!["s".to_string()])
            .build();
        let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
        let balance = FineBalance {
            levels: vec![vec!["a".to_string()], vec!["b".to_string()]],
            ..FineBalance::default()
        };
        let index: RapidHashMap<&str, f64> = data
            .anchors
            .iter()
            .chain(&data.candidates)
            .enumerate()
            .map(|(i, record)| (record.id.as_str(), i as f64))
            .collect();
        epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
            let got = match_fine_balance(&data.anchors, &data.candidates, &request, &balance)
                .expect("feasible");
            matched_pairs(&got)
                .iter()
                .flat_map(|(a, c)| [index[a.as_str()], index[c.as_str()]])
                .collect()
        });
    }

    /// The network with every admissible pair solved as one problem by the
    /// network simplex, from the start of price rounds, as a reference for the
    /// optimum.
    mod full_network {
        use super::super::{
            BalanceCategory, BalanceLevel, BalanceMode, Costs, FineBalance, FineBalanceOutcome,
            Key, LevelData, costs, level_data, option_error, to_i64,
        };
        use crate::matching::constraints::ConstraintGroup;
        use crate::matching::optimal::{
            OptimalError, OptimalRequest, StrataEdges, pair_outcome, strata_edges,
        };
        use crate::matching::records::MatchingRecord;
        use epistat_optim::min_cost_flow::{Basis, FlowError, MinCostFlow};
        use rayon::prelude::*;
        use std::collections::BTreeMap;

        /// Candidates kept per anchor and finest-level category at the start, as a
        /// multiple of the match ratio.
        const FIRST_PER_CATEGORY: usize = 3;

        /// Arcs per block of the solver's pricing search. The pivots of this network
        /// are mostly cheap, so a short block pays.
        const PRICING_BLOCK: usize = 32;

        /// Rounds of price adjustment for the start.
        const ROUNDS: usize = 40;

        /// One stratum's pairs as an assignment problem, with local numbering.
        struct Local {
            /// Positions in the pair list.
            ks: Vec<usize>,
            anchors: usize,
            candidates: usize,
            from: Vec<usize>,
            to: Vec<usize>,
            /// Finest-level category of each local candidate.
            category: Vec<usize>,
        }

        impl Local {
            /// The positions of the pairs of the cheapest assignment and the number of
            /// candidates it takes in each category (indexed as the prices are).
            fn assign(
                &self,
                ratio: usize,
                costs: &[i64],
                price: &[i64],
            ) -> Option<(Vec<usize>, Vec<usize>)> {
                let (a, c) = (self.anchors, self.candidates);
                let mut flow = MinCostFlow::new(a + c + 1);
                for node in 0..a {
                    flow.set_supply(node, to_i64(ratio));
                }
                flow.set_supply(a + c, -to_i64(ratio * a));
                for (slot, &k) in self.ks.iter().enumerate() {
                    let cost = costs[k].saturating_add(price[self.category[self.to[slot]]]);
                    flow.add_arc(self.from[slot], a + self.to[slot], 1, cost);
                }
                for node in 0..c {
                    flow.add_arc(a + node, a + c, 1, 0);
                }
                let solution = flow.solve().ok()?;
                let mut chosen = Vec::new();
                let mut counts = vec![0; price.len()];
                for (slot, &k) in self.ks.iter().enumerate() {
                    if solution.flow[slot] == 1 {
                        chosen.push(k);
                        counts[self.category[self.to[slot]]] += 1;
                    }
                }
                Some((chosen, counts))
            }
        }

        /// Node numbers for the anchors and candidates, in the order of the strata, so
        /// that the pairs of one stratum touch nearby nodes.
        struct Ids {
            anchor: Vec<usize>,
            candidate: Vec<usize>,
            /// Candidates in node order.
            candidate_order: Vec<usize>,
        }

        impl Ids {
            fn new(
                na: usize,
                nc: usize,
                pairs: &[(usize, usize, f64)],
                runs: &[std::ops::Range<usize>],
            ) -> Self {
                let mut anchor = vec![usize::MAX; na];
                let mut candidate = vec![usize::MAX; nc];
                let (mut next_anchor, mut next_candidate) = (0, na);
                for run in runs {
                    for k in run.clone() {
                        let (a, c, _) = pairs[k];
                        if anchor[a] == usize::MAX {
                            anchor[a] = next_anchor;
                            next_anchor += 1;
                        }
                        if candidate[c] == usize::MAX {
                            candidate[c] = next_candidate;
                            next_candidate += 1;
                        }
                    }
                }
                for slot in &mut anchor {
                    if *slot == usize::MAX {
                        *slot = next_anchor;
                        next_anchor += 1;
                    }
                }
                for slot in &mut candidate {
                    if *slot == usize::MAX {
                        *slot = next_candidate;
                        next_candidate += 1;
                    }
                }
                let mut candidate_order = vec![0; nc];
                for (c, id) in candidate.iter().enumerate() {
                    candidate_order[id - na] = c;
                }
                Self {
                    anchor,
                    candidate,
                    candidate_order,
                }
            }
        }

        /// The fixed parts of the flow network.
        struct Network<'a> {
            na: usize,
            nc: usize,
            ratio: usize,
            n: usize,
            sink: usize,
            first_category: &'a [usize],
            levels: &'a [LevelData],
            costs: &'a Costs,
        }

        impl Network<'_> {
            fn pair_cost(&self, distance: f64) -> i64 {
                // The value is a non-negative whole number within the solver's range.
                #[expect(clippy::cast_possible_truncation, reason = "scaled distance fits i64")]
                let cost = (distance.max(0.0) * self.costs.scale).round_ties_even() as i64;
                cost
            }

            /// For each stratum run of `pairs`, the positions of the `per` cheapest
            /// pairs of every anchor and finest-level category, ties by candidate.
            fn nearest(
                &self,
                pairs: &[(usize, usize, f64)],
                costs: &[i64],
                runs: &[std::ops::Range<usize>],
                per: usize,
            ) -> Vec<usize> {
                let leaf = &self.levels[self.levels.len() - 1];
                let chosen: Vec<Vec<usize>> = runs
                    .par_iter()
                    .map(|run| {
                        let mut order: Vec<usize> = run.clone().collect();
                        order.sort_unstable_by_key(|&k| {
                            let (anchor, candidate, _) = pairs[k];
                            (
                                anchor,
                                leaf.of_unit[self.na + candidate],
                                costs[k],
                                candidate,
                            )
                        });
                        let mut keep = Vec::new();
                        let mut group = (usize::MAX, usize::MAX);
                        let mut taken = 0;
                        for k in order {
                            let (anchor, candidate, _) = pairs[k];
                            let this = (anchor, leaf.of_unit[self.na + candidate]);
                            if this != group {
                                group = this;
                                taken = 0;
                            }
                            if taken < per {
                                keep.push(k);
                                taken += 1;
                            }
                        }
                        keep
                    })
                    .collect();
                chosen.into_iter().flatten().collect()
            }

            /// The network with the arcs of the categories, then the pairs at `used`
            /// in that order. Pairs added later go at the end, as
            /// [`MinCostFlow::solve_from`] needs.
            fn flow(
                &self,
                pairs: &[(usize, usize, f64)],
                costs: &[i64],
                used: &[usize],
                ids: &Ids,
            ) -> MinCostFlow {
                let (na, nc) = (self.na, self.nc);
                let mut flow = MinCostFlow::new(self.sink + 1);
                flow.set_pricing_block(PRICING_BLOCK);
                flow.reserve_arcs(
                    used.len()
                        + nc
                        + 2 * self
                            .levels
                            .iter()
                            .map(|l| l.categories.len())
                            .sum::<usize>(),
                );
                for &node in &ids.anchor {
                    flow.set_supply(node, -to_i64(self.ratio));
                }
                flow.set_supply(self.sink, to_i64(self.n));
                let depth = self.levels.len();
                let leaf = &self.levels[depth - 1];
                for &candidate in &ids.candidate_order {
                    let category = leaf.of_unit[na + candidate];
                    flow.add_arc(
                        self.first_category[depth - 1] + category,
                        ids.candidate[candidate],
                        1,
                        0,
                    );
                }
                for (depth, level) in self.levels.iter().enumerate() {
                    for category in 0..level.categories.len() {
                        let to = if depth == 0 {
                            self.sink
                        } else {
                            self.first_category[depth - 1] + level.parent[category]
                        };
                        let from = self.first_category[depth] + category;
                        let target = self.ratio * level.anchors[category];
                        if target > 0 {
                            flow.add_arc(to, from, to_i64(target), 0);
                        }
                        flow.add_arc(to, from, to_i64(self.n), self.costs.penalty[depth]);
                    }
                }
                for &k in used {
                    let (anchor, candidate, _) = pairs[k];
                    flow.add_arc(ids.candidate[candidate], ids.anchor[anchor], 1, costs[k]);
                }
                flow
            }

            /// The pairs at `selected` of one stratum as a small assignment problem.
            fn local(
                &self,
                pairs: &[(usize, usize, f64)],
                run: &std::ops::Range<usize>,
                selected: &[bool],
            ) -> Local {
                let leaf = &self.levels[self.levels.len() - 1];
                let ks: Vec<usize> = run.clone().filter(|&k| selected[k]).collect();
                let mut anchors: Vec<usize> = ks.iter().map(|&k| pairs[k].0).collect();
                anchors.sort_unstable();
                anchors.dedup();
                let mut candidates: Vec<usize> = ks.iter().map(|&k| pairs[k].1).collect();
                candidates.sort_unstable();
                candidates.dedup();
                let from = ks
                    .iter()
                    .map(|&k| anchors.binary_search(&pairs[k].0).unwrap_or(0))
                    .collect();
                let to = ks
                    .iter()
                    .map(|&k| candidates.binary_search(&pairs[k].1).unwrap_or(0))
                    .collect();
                let category = candidates
                    .iter()
                    .map(|&c| leaf.of_unit[self.na + c])
                    .collect();
                Local {
                    ks,
                    anchors: anchors.len(),
                    candidates: candidates.len(),
                    from,
                    to,
                    category,
                }
            }

            /// Pairs for the start: the cheapest assignment in each stratum when the
            /// candidates of finest-level category `k` cost `price[k]` more, with the
            /// prices moved until the number taken per category is near its target.
            /// Which pairs the start uses does not change the optimum. `None` when some
            /// stratum has no assignment.
            fn stratum_start(
                &self,
                pairs: &[(usize, usize, f64)],
                costs: &[i64],
                runs: &[std::ops::Range<usize>],
                selected: &[bool],
            ) -> Option<Vec<usize>> {
                let leaf = &self.levels[self.levels.len() - 1];
                let categories = leaf.categories.len();
                let locals: Vec<Local> = runs
                    .par_iter()
                    .map(|run| self.local(pairs, run, selected))
                    .collect();
                let target: Vec<usize> = leaf.anchors.iter().map(|a| self.ratio * a).collect();
                let unit = {
                    let (sum, count) = locals
                        .iter()
                        .flat_map(|l| l.ks.iter())
                        .fold((0.0_f64, 0_usize), |(s, c), &k| {
                            (s + costs[k] as f64, c + 1)
                        });
                    (sum / count.max(1) as f64).max(1.0)
                };
                let mut price = vec![0_f64; categories];
                let mut step = vec![unit * 1e-4; categories];
                let mut last = vec![0_i8; categories];
                let mut best: Option<(usize, Vec<usize>)> = None;
                for _ in 0..ROUNDS {
                    #[expect(clippy::cast_possible_truncation, reason = "a price is a few costs")]
                    let whole: Vec<i64> = price.iter().map(|p| *p as i64).collect();
                    let parts: Vec<Option<(Vec<usize>, Vec<usize>)>> = locals
                        .par_iter()
                        .map(|l| l.assign(self.ratio, costs, &whole))
                        .collect();
                    let mut chosen = Vec::new();
                    let mut got = vec![0_usize; categories];
                    for part in parts {
                        let (ks, counts) = part?;
                        chosen.extend(ks);
                        for (total, add) in got.iter_mut().zip(counts) {
                            *total += add;
                        }
                    }
                    let gap: usize = got.iter().zip(&target).map(|(a, b)| a.abs_diff(*b)).sum();
                    if best.as_ref().is_none_or(|b| gap < b.0) {
                        best = Some((gap, chosen));
                    }
                    if gap == 0 {
                        break;
                    }
                    // Resilient propagation: each price moves by its own step in the
                    // direction that reduces its surplus, and the step grows while the
                    // direction holds and shrinks when it reverses.
                    for k in 0..categories {
                        let direction = match got[k].cmp(&target[k]) {
                            std::cmp::Ordering::Greater => 1_i8,
                            std::cmp::Ordering::Less => -1,
                            std::cmp::Ordering::Equal => 0,
                        };
                        if direction != 0 && last[k] != 0 {
                            step[k] *= if direction == last[k] { 1.6 } else { 0.4 };
                        }
                        price[k] += step[k] * f64::from(direction);
                        if direction != 0 {
                            last[k] = direction;
                        }
                    }
                }
                best.map(|b| b.1)
            }

            /// The flow on each arc of [`Self::flow`] for the pairs at `chosen`, with
            /// the category arcs filled in from the counts.
            fn start_values(
                &self,
                pairs: &[(usize, usize, f64)],
                chosen: &[usize],
                used: &[usize],
                ids: &Ids,
            ) -> Vec<i64> {
                let na = self.na;
                let mut taken = vec![false; self.nc];
                for &k in chosen {
                    taken[pairs[k].1] = true;
                }
                let depth = self.levels.len();
                let leaf = &self.levels[depth - 1];
                let mut values: Vec<i64> = ids
                    .candidate_order
                    .iter()
                    .map(|&c| i64::from(taken[c]))
                    .collect();
                let mut count: Vec<Vec<usize>> = self
                    .levels
                    .iter()
                    .map(|level| vec![0; level.categories.len()])
                    .collect();
                for (candidate, t) in taken.iter().enumerate() {
                    if *t {
                        count[depth - 1][leaf.of_unit[na + candidate]] += 1;
                    }
                }
                for level in (1..depth).rev() {
                    for category in 0..self.levels[level].categories.len() {
                        let up = self.levels[level].parent[category];
                        let moved = count[level][category];
                        count[level - 1][up] += moved;
                    }
                }
                for (level, data) in self.levels.iter().enumerate() {
                    for (category, &here) in count[level].iter().enumerate() {
                        let target = self.ratio * data.anchors[category];
                        let cheap = here.min(target);
                        if target > 0 {
                            values.push(to_i64(cheap));
                        }
                        values.push(to_i64(here - cheap));
                    }
                }
                let on: std::collections::HashSet<usize> = chosen.iter().copied().collect();
                values.extend(used.iter().map(|k| i64::from(on.contains(k))));
                values
            }

            /// The matched pairs and the optimal cost.
            fn solve(
                &self,
                pairs: &[(usize, usize, f64)],
                runs: &[std::ops::Range<usize>],
            ) -> Result<(Vec<(usize, usize)>, i128), OptimalError> {
                let costs: Vec<i64> = pairs
                    .par_iter()
                    .map(|pair| self.pair_cost(pair.2))
                    .collect();
                let mut per = FIRST_PER_CATEGORY * self.ratio;
                let mut selected = vec![false; pairs.len()];
                let ids = Ids::new(self.na, self.nc, pairs, runs);
                let mut basis: Option<Basis> = None;
                let mut used: Vec<usize> = (0..pairs.len()).collect();
                used.sort_unstable();
                for &k in &used {
                    selected[k] = true;
                }
                loop {
                    let everything = used.len() == pairs.len();
                    let problem = self.flow(pairs, &costs, &used, &ids);
                    let offset = problem.arcs() - used.len();
                    let solved = match &basis {
                        Some(basis) => problem.solve_from(basis),
                        None => match self.stratum_start(pairs, &costs, runs, &selected) {
                            Some(chosen) => problem
                                .solve_from_flow(&self.start_values(pairs, &chosen, &used, &ids)),
                            _ => problem.solve(),
                        },
                    };
                    let solution = match solved {
                        Ok(solution) => solution,
                        Err(FlowError::Infeasible) if !everything => {
                            // Too few pairs for the anchors to get their candidates.
                            per = per.saturating_mul(4);
                            let mut more = self.nearest(pairs, &costs, runs, per);
                            more.retain(|&k| !selected[k]);
                            more.sort_unstable();
                            for &k in &more {
                                selected[k] = true;
                            }
                            used.extend(more);
                            basis = None;
                            continue;
                        }
                        Err(FlowError::Infeasible) => {
                            return Err(OptimalError::Infeasible {
                                stratum: 0,
                                message: format!(
                                    "the anchors cannot all be given {} candidates within the \
                                     admissible pairs",
                                    self.ratio
                                ),
                            });
                        }
                        Err(FlowError::Overflow | FlowError::InvalidInput) => {
                            return Err(OptimalError::TooLarge { stratum: 0 });
                        }
                    };
                    // A pair left out has no flow. Its reduced cost against the
                    // potentials is what pricing the pair would give, and a negative
                    // one means the pair could improve the flow.
                    let violating: Vec<usize> = if everything {
                        Vec::new()
                    } else {
                        (0..pairs.len())
                            .into_par_iter()
                            .filter(|&k| {
                                let (anchor, candidate, _) = pairs[k];
                                !selected[k]
                                    && i128::from(costs[k])
                                        + i128::from(solution.potentials[ids.candidate[candidate]])
                                        - i128::from(solution.potentials[ids.anchor[anchor]])
                                        < 0
                            })
                            .collect()
                    };
                    if violating.is_empty() {
                        let matched = used
                            .iter()
                            .enumerate()
                            .filter(|&(arc, _)| solution.flow[offset + arc] == 1)
                            .map(|(_, &k)| (pairs[k].0, pairs[k].1))
                            .collect();
                        return Ok((matched, solution.cost));
                    }
                    for &k in &violating {
                        selected[k] = true;
                    }
                    used.extend(violating);
                    basis = Some(solution.basis);
                }
            }
        }

        pub(super) fn run<R, G>(
            anchors: &[R],
            candidates: &[R],
            request: &OptimalRequest<'_, G>,
            balance: &FineBalance,
        ) -> Result<(FineBalanceOutcome, i128), OptimalError>
        where
            R: MatchingRecord,
            G: ConstraintGroup<R> + ?Sized,
        {
            if balance.levels.is_empty() {
                return Err(option_error("name at least one level of keys"));
            }
            if balance.levels.iter().any(Vec::is_empty) {
                return Err(option_error("every level needs at least one key"));
            }
            let ratio = request.criteria.match_ratio;
            let StrataEdges { strata, edges } = strata_edges(anchors, candidates, request)?;

            let mut levels: Vec<LevelData> = Vec::with_capacity(balance.levels.len());
            for keys in &balance.levels {
                let data = level_data(anchors, candidates, keys, levels.last())?;
                levels.push(data);
            }

            // Admissible pairs by position in the anchor and candidate slices, with
            // the start of each stratum's run of pairs.
            let mut pairs: Vec<(usize, usize, f64)> =
                Vec::with_capacity(edges.iter().map(Vec::len).sum());
            let mut runs: Vec<std::ops::Range<usize>> = Vec::with_capacity(strata.len());
            for (stratum, stratum_edges) in strata.iter().zip(edges) {
                let first = pairs.len();
                for edge in stratum_edges {
                    pairs.push((
                        stratum.anchors[edge.row],
                        stratum.candidates[edge.col],
                        edge.distance,
                    ));
                }
                runs.push(first..pairs.len());
            }
            drop(strata);
            if pairs.is_empty() {
                return Err(OptimalError::NoMatches);
            }
            let (na, nc) = (anchors.len(), candidates.len());
            let n = ratio * na;
            let mut admissible = vec![false; na];
            for (anchor, _, _) in &pairs {
                admissible[*anchor] = true;
            }
            if let Some(anchor) = admissible.iter().position(|ok| !ok) {
                return Err(OptimalError::Infeasible {
                    stratum: 0,
                    message: format!(
                        "anchor {} has no admissible candidate, and every anchor needs {ratio}",
                        anchors[anchor].id()
                    ),
                });
            }
            if n > nc {
                return Err(OptimalError::Infeasible {
                    stratum: 0,
                    message: format!("{na} anchors need {n} candidates and there are {nc}"),
                });
            }

            // Nodes: anchors, candidates, the categories of each level, the sink.
            let mut first_category = Vec::with_capacity(levels.len());
            let mut next = na + nc;
            for level in &levels {
                first_category.push(next);
                next += level.categories.len();
            }
            let sink = next;
            let nodes = next + 1;
            let max_distance = pairs.iter().map(|pair| pair.2).fold(0.0_f64, f64::max);
            let Some(costs) = costs(max_distance, n, levels.len(), nodes) else {
                return Err(OptimalError::TooLarge { stratum: 0 });
            };

            let network = Network {
                na,
                nc,
                ratio,
                n,
                sink,
                first_category: &first_category,
                levels: &levels,
                costs: &costs,
            };
            let (matched, optimum) = network.solve(&pairs, &runs)?;
            let mut table: Vec<BalanceLevel> = Vec::with_capacity(levels.len());
            for level in &levels {
                let mut counts = vec![0_usize; level.categories.len()];
                for (_, candidate) in &matched {
                    counts[level.of_unit[na + candidate]] += 1;
                }
                let mut categories: BTreeMap<&Key, BalanceCategory> = BTreeMap::new();
                for (index, values) in level.categories.iter().enumerate() {
                    categories.insert(
                        values,
                        BalanceCategory {
                            values: values.clone(),
                            target: ratio * level.anchors[index],
                            matched: counts[index],
                        },
                    );
                }
                table.push(BalanceLevel {
                    keys: level.keys.clone(),
                    categories: categories.into_values().collect(),
                });
            }
            if balance.mode == BalanceMode::Exact
                && let Some((depth, level)) = table
                    .iter()
                    .enumerate()
                    .find(|(_, level)| level.total_deviation() > 0)
            {
                return Err(OptimalError::Infeasible {
                    stratum: 0,
                    message: format!(
                        "fine balance on level {depth} (keys {:?}) is not possible with these \
                             pairs; the least total deviation is {}",
                        level.keys,
                        level.total_deviation()
                    ),
                });
            }
            let outcome = pair_outcome(
                anchors,
                candidates,
                ratio,
                request.criteria.estimand,
                vec![matched],
            );
            Ok((
                FineBalanceOutcome {
                    outcome,
                    levels: table,
                },
                optimum,
            ))
        }
    }
}
