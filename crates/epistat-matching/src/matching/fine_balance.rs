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
//! fewer and adds the rest only if they can improve the result.
//!
//! 1. For each anchor, the pairs with its `3 * ratio` nearest candidates in
//!    each category of the finest level are the first arcs.
//! 2. A start flow is found stratum by stratum, as a plain assignment problem
//!    on those arcs. The candidates of finest-level category `k` cost `price[k]`
//!    more, and the prices are moved (resilient propagation, each price with
//!    its own step) until the number of candidates taken per category is near
//!    its target. This only chooses where the solver starts, so it cannot
//!    change the optimum, and a start with little to fix by moving candidates
//!    between categories is what keeps the solver's pivots cheap.
//! 3. [`MinCostFlow::solve_from_flow`] solves the network from that start.
//! 4. Every pair left out is priced against the potentials of the solution.
//!    A pair whose reduced cost is negative could improve the flow, so those
//!    pairs are added, at the end of the arc list, and the network is solved
//!    again from the final tree of the last solve
//!    ([`MinCostFlow::solve_from`]). When no pair prices negatively the
//!    potentials are feasible for the full network, so the flow is optimal for
//!    it by complementary slackness: the optimal value, and so the deviation at
//!    every level and the total distance, is that of the network with all pairs.
//!    If the arcs so far leave some anchor without enough candidates, the lists
//!    are widened fourfold until it is feasible.
//!
//! The pairs returned can differ from those of the full network only among
//! matches of equal cost. The prices of step 2 and the order of arcs do not
//! depend on the number of threads, since the strata are solved independently
//! and combined in stratum order.
//!
//! The work is a sort of the pairs of each stratum by distance, the sweeps that
//! price the pairs left out (each `O(pairs)`, and there are a few), the price
//! rounds of step 2, which solve each stratum's small assignment problem, and
//! the solver on the pairs kept. Memory is the admissible pairs, 24 bytes each
//! plus 8 for the integer cost, and the network of the pairs kept. The solver's
//! block of arcs searched per pivot is short, since most pivots of this network
//! touch a small subtree.
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

use super::constraints::ConstraintGroup;
use super::optimal::{OptimalError, OptimalRequest, StrataEdges, pair_outcome, strata_edges};
use super::records::MatchingRecord;
use crate::types::MatchOutcome;
use epistat_optim::min_cost_flow::{Basis, FlowError, MinCostFlow};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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

fn level_data<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    keys: &[String],
    above: Option<&LevelData>,
) -> Result<LevelData, OptimalError> {
    let key_of = |record: &R| -> Key {
        keys.iter()
            .map(|key| record.strata().get(key).cloned())
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
        arcs: Arcs,
    ) -> Result<(Vec<(usize, usize)>, i128), OptimalError> {
        let costs: Vec<i64> = pairs
            .par_iter()
            .map(|pair| self.pair_cost(pair.2))
            .collect();
        let mut per = FIRST_PER_CATEGORY * self.ratio;
        let mut selected = vec![false; pairs.len()];
        let ids = Ids::new(self.na, self.nc, pairs, runs);
        let mut basis: Option<Basis> = None;
        let mut used: Vec<usize> = match arcs {
            Arcs::Pruned => self.nearest(pairs, &costs, runs, per),
            #[cfg(test)]
            Arcs::All => (0..pairs.len()).collect(),
        };
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
                    Some(chosen) => {
                        problem.solve_from_flow(&self.start_values(pairs, &chosen, &used, &ids))
                    }
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
    let StrataEdges { strata, edges } = strata_edges(anchors, candidates, request)?;

    let mut levels: Vec<LevelData> = Vec::with_capacity(balance.levels.len());
    for keys in &balance.levels {
        let data = level_data(anchors, candidates, keys, levels.last())?;
        levels.push(data);
    }

    // Admissible pairs by position in the anchor and candidate slices, with
    // the start of each stratum's run of pairs.
    let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.iter().map(Vec::len).sum());
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
    let (matched, optimum) = network.solve(&pairs, &runs, arcs)?;
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
    if balance.mode == BalanceMode::Exact {
        if let Some((depth, level)) = table
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RapidHashMap;
    use crate::prelude::*;
    use chrono::NaiveDate;

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
    #[test]
    fn pruned_network_has_the_optimum_of_the_full_network() {
        let mut rng = Lcg(20_261_121);
        let mut compared = 0;
        for round in 0..60 {
            let strata = 1 + round % 4;
            let ratio = 1 + round % 2;
            let anchors = 8 + rng.below(40);
            let candidates = ratio * anchors + 5 + rng.below(60);
            let grid = [4, 16, 1000][round % 3];
            let data = case(&mut rng, anchors, candidates, strata, grid);
            let distance = DistanceConfig::propensity_score_map(data.scores.clone(), None);
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
                let full = run(
                    &data.anchors,
                    &data.candidates,
                    &request,
                    &balance,
                    Arcs::All,
                );
                let pruned = run(
                    &data.anchors,
                    &data.candidates,
                    &request,
                    &balance,
                    Arcs::Pruned,
                );
                match (full, pruned) {
                    (Ok((full, full_cost)), Ok((pruned, pruned_cost))) => {
                        compared += 1;
                        assert_eq!(pruned_cost, full_cost, "round {round}: optimal value");
                        // Categories that have no target can share their surplus
                        // in different ways at equal cost, so the deviation of
                        // each level is compared, not the counts.
                        let deviation = |o: &FineBalanceOutcome| -> Vec<usize> {
                            o.levels.iter().map(BalanceLevel::total_deviation).collect()
                        };
                        assert_eq!(
                            deviation(&pruned),
                            deviation(&full),
                            "round {round}: balance"
                        );
                        let (full_pairs, pruned_pairs) =
                            (matched_pairs(&full), matched_pairs(&pruned));
                        assert_eq!(full_pairs.len(), pruned_pairs.len());
                        let key = |id: &str| data.scores[id];
                        let total = |pairs: &[(String, String)]| -> f64 {
                            pairs.iter().map(|(a, c)| (key(a) - key(c)).abs()).sum()
                        };
                        // Equal optimal values mean equal total distance up to
                        // the rounding of the integer costs.
                        assert!(
                            (total(&full_pairs) - total(&pruned_pairs)).abs() < 1e-6,
                            "round {round}: total distance"
                        );
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "round {round}"),
                    (a, b) => panic!(
                        "round {round}: full {:?}, pruned {:?}",
                        a.is_ok(),
                        b.is_ok()
                    ),
                }
            }
        }
        assert!(compared > 80, "only {compared} problems compared");
    }

    #[test]
    fn pruned_network_is_the_same_on_every_thread_count() {
        let mut rng = Lcg(99);
        let data = case(&mut rng, 60, 200, 5, 16);
        let distance = DistanceConfig::propensity_score_map(data.scores.clone(), None);
        let criteria = MatchingCriteria::builder()
            .required_strata(vec!["s".to_string()])
            .build();
        let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
        let balance = FineBalance::on_keys(["a"]);
        let mut results = Vec::new();
        for threads in [1, 2, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let got = pool
                .install(|| match_fine_balance(&data.anchors, &data.candidates, &request, &balance))
                .expect("feasible");
            results.push(matched_pairs(&got));
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
    }
}
