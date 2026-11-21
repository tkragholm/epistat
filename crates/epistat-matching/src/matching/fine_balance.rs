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
//! The network is that of pair matching: each anchor supplies `ratio` units, an
//! arc from an anchor to a candidate carries one unit when the pair is matched
//! and costs the distance. A candidate passes its unit to the node of its
//! category. A category node has two arcs towards the sink, or towards its
//! category one level up. The first has room for the number of units the
//! category is meant to receive and costs nothing. The second has room for all
//! units and costs a penalty per unit. Candidates beyond the target of their
//! category use the second arc, so the penalties add up to the number of
//! candidates above target. Since the targets and the number of candidates both
//! sum to `ratio` times the anchors, that equals the number below target, and
//! the total deviation is twice the penalised units.
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
use epistat_optim::min_cost_flow::{FlowError, MinCostFlow};
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

    // Admissible pairs by position in the anchor and candidate slices.
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (stratum, stratum_edges) in strata.iter().zip(&edges) {
        for edge in stratum_edges {
            pairs.push((
                stratum.anchors[edge.row],
                stratum.candidates[edge.col],
                edge.distance,
            ));
        }
    }
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

    let mut flow = MinCostFlow::new(nodes);
    flow.reserve_arcs(
        pairs.len() + nc + 2 * levels.iter().map(|l| l.categories.len()).sum::<usize>(),
    );
    for anchor in 0..na {
        flow.set_supply(anchor, to_i64(ratio));
    }
    flow.set_supply(sink, -to_i64(n));
    for (anchor, candidate, distance) in &pairs {
        // The value is a non-negative whole number within the solver's range.
        #[expect(clippy::cast_possible_truncation, reason = "scaled distance fits i64")]
        let cost = (distance.max(0.0) * costs.scale).round_ties_even() as i64;
        flow.add_arc(*anchor, na + *candidate, 1, cost);
    }
    for candidate in 0..nc {
        let leaf = &levels[levels.len() - 1];
        let category = leaf.of_unit[na + candidate];
        flow.add_arc(
            na + candidate,
            first_category[levels.len() - 1] + category,
            1,
            0,
        );
    }
    for (depth, level) in levels.iter().enumerate() {
        for category in 0..level.categories.len() {
            let to = if depth == 0 {
                sink
            } else {
                first_category[depth - 1] + level.parent[category]
            };
            let from = first_category[depth] + category;
            let target = ratio * level.anchors[category];
            if target > 0 {
                flow.add_arc(from, to, to_i64(target), 0);
            }
            flow.add_arc(from, to, to_i64(n), costs.penalty[depth]);
        }
    }
    let solution = flow.solve().map_err(|error| match error {
        FlowError::Infeasible => OptimalError::Infeasible {
            stratum: 0,
            message: format!(
                "the anchors cannot all be given {ratio} candidates within the admissible pairs"
            ),
        },
        FlowError::Overflow | FlowError::InvalidInput => OptimalError::TooLarge { stratum: 0 },
    })?;

    let mut matched: Vec<(usize, usize)> = Vec::with_capacity(n);
    for (arc, (anchor, candidate, _)) in pairs.iter().enumerate() {
        if solution.flow[arc] == 1 {
            matched.push((*anchor, *candidate));
        }
    }
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
    Ok(FineBalanceOutcome {
        outcome,
        levels: table,
    })
}
