//! Optimal pair matching and optimal full matching by minimum-cost flow.
//!
//! These are `matchit(method = "optimal")` and `matchit(method = "full")`.
//! Both minimise the sum of the distances of the matched pairs. They differ in
//! what a matched set may look like. In pair matching every anchor gets
//! `ratio` candidates (or fewer where the candidates run out), and no candidate
//! is used twice. In full matching every unit is in a subclass of one anchor
//! with several candidates or one candidate with several anchors, within
//! limits on the number of candidates per anchor.
//!
//! # Network
//!
//! A stratum of anchors (rows) and candidates (columns) with the admissible
//! pairs as arcs is solved as a minimum-cost flow on the network of
//! Rosenbaum \[1\] and Hansen and Klopfer \[2\]. Each row node supplies `mxc`
//! units, and each column node can pass flow on to a sink. An arc from a row to
//! a column carries 1 unit when the pair is matched. A row sends the part of its
//! supply it does not use on matches to an end node, through an arc with room
//! for `mxc - mnc` units, which is what lets a row have between `mnc` and `mxc`
//! columns. A column has an arc to the sink for its own unit and an arc to the
//! end node with room for `mxr - 1` units, which is what lets a column have up
//! to `mxr` rows. The sink demands the number of columns to match and the end
//! node the rest of the supply. With positive costs the optimum is a set of
//! stars (one row with several columns, or one column with several rows), and
//! each star is a subclass.
//!
//! The limits are those of `optmatch::fullmatch` \[3\]. From `min.controls`,
//! `max.controls` and the number of rows and columns, `mxc` is the larger of 1
//! and the smaller of `max.controls` and the number of columns, rounded up,
//! `mnc` is the larger of 1 and `min.controls`, rounded down, and `mxr` is the
//! reciprocal of the larger of `min.controls` and `1 / rows`, rounded up. When
//! `mnc` exceeds 1, `mxr` is 1. A positive omitted fraction, or a mean number
//! of controls, sets the number of columns to match. A negative omitted
//! fraction, or a maximum below one half, solves the problem with rows and
//! columns exchanged.
//!
//! # Costs
//!
//! The solver takes integer costs. A distance `d` becomes
//! `round(d * 2^k) + 1` for one power of two `2^k` per stratum, chosen so that
//! the largest distance is about `2^40` (less when the stratum has more than
//! 2^16 units). The error in one cost is at most `0.5 / 2^k + 1 / 2^k`, so the
//! total distance of the result is within `1.5 n / 2^k` of the optimum for `n`
//! matched pairs, and `2^k` is at least `2^40 / max d`. The added 1 keeps every
//! cost positive, which the network needs, and makes ties between solutions of
//! equal total distance go to the one with fewer pairs, so more subclasses.
//! optmatch rounds to a coarser grid (a tolerance of 0.001 times the number of
//! units in total), so where two solutions are closer than that it may pick the
//! other one.
//!
//! # Ties
//!
//! Among solutions of the same cost the solver returns one that depends only
//! on the order of the units: rows and columns in the order of the slices, arcs
//! in the order of row and then column. See
//! [`epistat_optim::min_cost_flow`]. The result does not depend on the number
//! of threads, since each exact stratum is solved on its own and the strata are
//! put together in order of their first anchor.
//!
//! # Cost of solving
//!
//! The network has one node per unit and one arc per admissible pair, and the
//! solver is the network simplex method. A stratum of 2,000 anchors and 8,000
//! candidates within a caliper that leaves a few hundred candidates per anchor
//! has a few hundred thousand arcs. A stratum without a caliper has one arc for
//! every pair of an anchor and a candidate, which takes memory in proportion to
//! their product (about 100 bytes per pair).
//!
//! # References
//!
//! 1. Rosenbaum, P. R. (1989). Optimal matching for observational studies.
//!    *Journal of the American Statistical Association* 84, 1024-1032. And
//!    Rosenbaum, P. R. (1991). A characterization of optimal designs for
//!    observational studies. *Journal of the Royal Statistical Society B* 53,
//!    597-610.
//! 2. Hansen, B. B. and Klopfer, S. O. (2006). Optimal full matching and
//!    related designs via network flows. *Journal of Computational and
//!    Graphical Statistics* 15, 609-627.
//! 3. The documentation of `optmatch::fullmatch` (Hansen), version 0.10.8.

use super::constraints::{ConstraintContext, ConstraintGroup, UsedControlsVec};
use super::distance::{DistanceConfig, DistanceMetric};
use super::engine::{build_outcome, finalize_estimand_diagnostics, to_f64};
use super::index::StrataValues;
use super::records::MatchingRecord;
use super::subclassification::{
    SubclassSummary, build_subclass_summaries, subclass_weight_for_estimand,
};
use super::weights::{UnitRole, UnitWeightSet};
use crate::output::{MatchData, Units};
use crate::types::{
    ControlIdx, CriteriaValidationError, Estimand, MatchDiagnostics, MatchOutcome, MatchedPair,
    MatchingCriteria,
};
use epistat_optim::min_cost_flow::{FlowError, MinCostFlow};
use rapidhash::RapidHashMap;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Why optimal matching could not run.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimalError {
    /// The criteria are invalid.
    Criteria(CriteriaValidationError),
    /// One option is out of range or does not fit the others.
    Option {
        /// The name of the option.
        option: &'static str,
        /// What is wrong.
        message: String,
    },
    /// The limits on the sizes of the matched sets cannot be met in an exact
    /// stratum, or the pairs the constraints admit are too few for them.
    Infeasible {
        /// The stratum, numbered from 0 in order of the first anchor in it.
        stratum: usize,
        /// What cannot be met.
        message: String,
    },
    /// No stratum has an admissible pair of an anchor and a candidate.
    NoMatches,
    /// A stratum is too large for the integer arithmetic of the solver.
    TooLarge {
        /// The stratum, numbered as in [`OptimalError::Infeasible`].
        stratum: usize,
    },
}

impl fmt::Display for OptimalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Criteria(error) => write!(f, "{error}"),
            Self::Option { option, message } => write!(f, "option `{option}`: {message}"),
            Self::Infeasible { stratum, message } => {
                write!(
                    f,
                    "no matching satisfies the limits in stratum {stratum}: {message}"
                )
            }
            Self::NoMatches => f.write_str("no anchor and candidate form an admissible pair"),
            Self::TooLarge { stratum } => {
                write!(f, "stratum {stratum} is too large to solve")
            }
        }
    }
}

impl std::error::Error for OptimalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Criteria(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CriteriaValidationError> for OptimalError {
    fn from(error: CriteriaValidationError) -> Self {
        Self::Criteria(error)
    }
}

/// How many of the candidates a full match uses.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum Retain {
    /// Every candidate that has an admissible anchor.
    #[default]
    All,
    /// Leave out up to this fraction of the candidates, a number in `[0, 1]`.
    /// A number in `[-1, 0)` leaves out that fraction of the anchors.
    OmitFraction(f64),
    /// Use this many candidates per anchor on average (`mean.controls`), or
    /// all the candidates that have an admissible anchor when there are fewer.
    MeanControls(f64),
}

/// The limits of a full match, those of `optmatch::fullmatch`.
///
/// The defaults are `fullmatch`'s: any number of candidates per anchor and any
/// number of anchors per candidate, and every candidate used.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FullMatchConfig {
    /// Fewest candidates per anchor in a subclass (`min.controls`). A number
    /// below 1 allows subclasses of several anchors with one candidate, down to
    /// one candidate per `1 / min_controls` anchors. 0 allows any number.
    pub min_controls: f64,
    /// Most candidates per anchor in a subclass (`max.controls`). Infinity
    /// allows any number. A number below 1 limits the anchors per candidate to
    /// the reciprocal.
    pub max_controls: f64,
    /// How many candidates are used.
    pub retain: Retain,
}

impl Default for FullMatchConfig {
    fn default() -> Self {
        Self {
            min_controls: 0.0,
            max_controls: f64::INFINITY,
            retain: Retain::All,
        }
    }
}

/// What optimal matching reads besides the units.
#[derive(Clone, Copy)]
pub struct OptimalRequest<'a, G: ?Sized> {
    /// The criteria. The exact strata are `required_strata`, the ratio is
    /// `match_ratio`, and without a distance the birth date window is the
    /// distance caliper. Replacement and unique keys are not available.
    pub criteria: &'a MatchingCriteria,
    /// Further constraints on a pair, which a forbidden pair does not pass.
    pub constraints: &'a G,
    /// The distance. Birth date, with the criteria's window as the caliper,
    /// when absent.
    pub distance_config: Option<&'a DistanceConfig>,
}

impl<'a, G: ?Sized> OptimalRequest<'a, G> {
    /// A request without a distance of its own.
    #[must_use]
    pub const fn new(criteria: &'a MatchingCriteria, constraints: &'a G) -> Self {
        Self {
            criteria,
            constraints,
            distance_config: None,
        }
    }

    crate::impl_with_distance_config!();
}

/// An admissible pair in a stratum, by position within the stratum.
#[derive(Debug, Clone, Copy)]
struct Edge {
    row: usize,
    col: usize,
    distance: f64,
}

/// A matched set: its pairs, by position within the stratum.
struct Group {
    edges: Vec<(usize, usize)>,
}

enum StratumError {
    Option(&'static str, String),
    Infeasible(String),
    TooLarge,
}

fn infeasible<T>(message: impl Into<String>) -> Result<T, StratumError> {
    Err(StratumError::Infeasible(message.into()))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is a small whole number or is range-checked by the caller"
)]
fn to_i64(value: f64) -> i64 {
    value as i64
}

fn round_half_even(value: f64) -> i64 {
    to_i64(value.round_ties_even())
}

/// The limits and the number of candidates to use, before orientation.
#[derive(Clone, Copy)]
struct Restrictions {
    min_controls: f64,
    max_controls: f64,
    retain: Retain,
}

impl Restrictions {
    fn validate(&self, rows: usize, cols: usize) -> Result<(), StratumError> {
        let Self {
            min_controls,
            max_controls,
            retain,
        } = *self;
        if !(min_controls.is_finite() && min_controls >= 0.0) {
            return Err(StratumError::Option(
                "min_controls",
                "use a finite number of 0 or more".into(),
            ));
        }
        if max_controls.is_nan() || max_controls <= 0.0 {
            return Err(StratumError::Option(
                "max_controls",
                "use a number above 0, or infinity".into(),
            ));
        }
        if min_controls > max_controls {
            return Err(StratumError::Option(
                "min_controls",
                format!("the minimum {min_controls} is above the maximum {max_controls}"),
            ));
        }
        match retain {
            Retain::All => {}
            Retain::MeanControls(mean) => {
                let tolerance = f64::EPSILON.sqrt();
                if !(mean.is_finite() && mean > 0.0) {
                    return Err(StratumError::Option(
                        "mean_controls",
                        "use a number above 0".into(),
                    ));
                }
                if mean + tolerance < min_controls {
                    return Err(StratumError::Option(
                        "mean_controls",
                        format!("the mean {mean} is below the minimum {min_controls}"),
                    ));
                }
                if mean - tolerance > max_controls {
                    return Err(StratumError::Option(
                        "mean_controls",
                        format!("the mean {mean} is above the maximum {max_controls}"),
                    ));
                }
                if mean > to_f64(cols) / to_f64(rows) {
                    return Err(StratumError::Option(
                        "mean_controls",
                        format!(
                            "the mean {mean} is above the {cols} candidates per {rows} anchors in the stratum"
                        ),
                    ));
                }
            }
            Retain::OmitFraction(fraction) => {
                if !(fraction.is_finite() && fraction.abs() <= 1.0) {
                    return Err(StratumError::Option(
                        "omit_fraction",
                        "use a number from -1 to 1".into(),
                    ));
                }
                if fraction > 0.0 && max_controls <= 0.5 {
                    return Err(StratumError::Option(
                        "omit_fraction",
                        "a positive fraction needs max_controls above 0.5".into(),
                    ));
                }
                if fraction < 0.0 && min_controls >= 2.0 {
                    return Err(StratumError::Option(
                        "omit_fraction",
                        "a negative fraction needs min_controls below 2".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The number of candidates to match, before the stratum's admissible units
/// are known.
#[derive(Clone, Copy)]
enum Target {
    All,
    /// Omit this fraction of the columns (non-negative).
    Omit(f64),
    /// This many columns per matchable row, on average.
    Mean(f64),
}

fn transpose(edges: &[Edge]) -> Vec<Edge> {
    let mut flipped: Vec<Edge> = edges
        .iter()
        .map(|edge| Edge {
            row: edge.col,
            col: edge.row,
            distance: edge.distance,
        })
        .collect();
    flipped.sort_unstable_by_key(|edge| (edge.row, edge.col));
    flipped
}

/// The largest power of two by which the distances can be scaled so that the
/// largest is at most about `2^bits`.
fn cost_scale(max_distance: f64, nodes: usize) -> f64 {
    let node_bits = i32::try_from(usize::BITS - (nodes + 2).leading_zeros()).unwrap_or(64);
    let bits = (56 - node_bits).clamp(20, 40);
    if max_distance <= 0.0 || !max_distance.is_finite() {
        return 1.0;
    }
    let exponent = f64::from(bits) - max_distance.log2();
    2.0_f64.powi(
        to_i64(exponent.floor())
            .clamp(-1000, 1000)
            .try_into()
            .unwrap_or(0),
    )
}

/// Full matching of one stratum under the limits, as `optmatch::fullmatch`.
///
/// `edges` hold every admissible pair of the stratum, sorted by row and then
/// column. Rows and columns that have none are not matched.
fn full_match_stratum(
    rows: usize,
    cols: usize,
    edges: &[Edge],
    restrictions: Restrictions,
) -> Result<Vec<Group>, StratumError> {
    if rows == 0 || cols == 0 || edges.is_empty() {
        return Ok(Vec::new());
    }
    restrictions.validate(rows, cols)?;
    let Restrictions {
        min_controls,
        max_controls,
        retain,
    } = restrictions;

    let (flipped, target) = match retain {
        Retain::All => (max_controls <= 0.5, Target::All),
        Retain::OmitFraction(fraction) => (fraction < 0.0, Target::Omit(fraction.abs())),
        Retain::MeanControls(mean) => (false, Target::Mean(mean)),
    };
    let (rows, cols, edges_oriented): (usize, usize, std::borrow::Cow<'_, [Edge]>) = if flipped {
        (cols, rows, std::borrow::Cow::Owned(transpose(edges)))
    } else {
        (rows, cols, std::borrow::Cow::Borrowed(edges))
    };
    let (max_per_row, min_per_row) = if flipped {
        (
            (1.0 / min_controls).min(to_f64(cols)),
            (1.0 / max_controls).max(1.0 / to_f64(rows)),
        )
    } else {
        (
            max_controls.min(to_f64(cols)),
            min_controls.max(1.0 / to_f64(rows)),
        )
    };

    // Rows and columns with an admissible pair, renumbered.
    let mut row_id = vec![usize::MAX; rows];
    let mut col_id = vec![usize::MAX; cols];
    for edge in edges_oriented.iter() {
        row_id[edge.row] = 0;
        col_id[edge.col] = 0;
    }
    let mut feasible_rows = Vec::new();
    for (row, slot) in row_id.iter_mut().enumerate() {
        if *slot == 0 {
            *slot = feasible_rows.len();
            feasible_rows.push(row);
        }
    }
    let mut feasible_cols = Vec::new();
    for (col, slot) in col_id.iter_mut().enumerate() {
        if *slot == 0 {
            *slot = feasible_cols.len();
            feasible_cols.push(col);
        }
    }
    let (nt, nc) = (feasible_rows.len(), feasible_cols.len());

    let omit = match target {
        Target::Omit(fraction) if fraction > 0.0 && nc < cols => {
            let adjusted = (fraction * to_f64(cols) - to_f64(cols - nc)) / to_f64(nc);
            (adjusted > 0.0).then_some(adjusted)
        }
        Target::Omit(fraction) if fraction > 0.0 => Some(fraction),
        _ => None,
    };
    let to_match = match target {
        Target::All | Target::Omit(_) => {
            round_half_even(to_f64(nc) * omit.map_or(1.0, |fraction| 1.0 - fraction))
        }
        Target::Mean(mean) => {
            round_half_even(mean * to_f64(nt)).min(i64::try_from(nc).unwrap_or(i64::MAX))
        }
    };

    if min_per_row.floor() > max_per_row.ceil()
        || (1.0 / min_per_row).ceil() < (1.0 / max_per_row).floor()
    {
        return infeasible(format!(
            "min_controls {min_controls} and max_controls {max_controls} leave no size for a subclass"
        ));
    }
    let max_cols = to_i64(max_per_row.ceil());
    let min_cols = to_i64(min_per_row.floor()).max(1);
    let max_rows = if min_cols > 1 {
        1
    } else {
        to_i64(min_per_row.recip().ceil())
    };
    if max_cols < min_cols {
        return infeasible("min_controls is above max_controls once rounded");
    }
    let rows_count = i64::try_from(nt).map_err(|_| StratumError::TooLarge)?;
    let cols_count = i64::try_from(nc).map_err(|_| StratumError::TooLarge)?;
    let too_few_rows = if max_rows > 1 {
        to_f64(nt) / to_f64(usize::try_from(max_rows).unwrap_or(usize::MAX))
            > to_f64(usize::try_from(to_match).unwrap_or(0))
    } else {
        rows_count * min_cols > to_match
    };
    if too_few_rows || rows_count * max_cols < to_match || to_match > cols_count {
        return infeasible(format!(
            "{nt} anchors with an admissible candidate cannot be matched to {to_match} of {nc} candidates within the limits"
        ));
    }

    let max_distance = edges_oriented
        .iter()
        .map(|edge| edge.distance)
        .fold(0.0_f64, f64::max);
    let scale = cost_scale(max_distance, nt + nc + 2);

    // Nodes: rows, columns, the end node, the sink.
    let end = nt + nc;
    let sink = end + 1;
    let mut flow = MinCostFlow::new(nt + nc + 2);
    flow.reserve_arcs(edges_oriented.len() + nt + 2 * nc);
    for row in 0..nt {
        flow.set_supply(row, max_cols);
    }
    flow.set_supply(end, -(max_cols * rows_count - to_match));
    flow.set_supply(sink, -to_match);
    for edge in edges_oriented.iter() {
        let cost = round_half_even(edge.distance.max(0.0) * scale) + 1;
        flow.add_arc(row_id[edge.row], nt + col_id[edge.col], 1, cost);
    }
    if max_cols > min_cols {
        for row in 0..nt {
            flow.add_arc(row, end, max_cols - min_cols, 0);
        }
    }
    if max_rows > 1 {
        for col in 0..nc {
            flow.add_arc(nt + col, end, max_rows - 1, 0);
        }
    }
    for col in 0..nc {
        flow.add_arc(nt + col, sink, 1, 0);
    }
    let solution = flow.solve().map_err(|error| match error {
        FlowError::Infeasible => StratumError::Infeasible(format!(
            "{nt} anchors with an admissible candidate cannot be matched to {to_match} of {nc} candidates within the limits and the admissible pairs"
        )),
        FlowError::Overflow | FlowError::InvalidInput => StratumError::TooLarge,
    })?;

    // Matched pairs, in the stratum's own numbering and orientation.
    let mut matched: Vec<(usize, usize)> = Vec::new();
    for (arc, edge) in edges_oriented.iter().enumerate() {
        if solution.flow[arc] == 1 {
            matched.push((edge.row, edge.col));
        }
    }
    let groups = connected_groups(&matched, rows, cols);
    Ok(if flipped {
        groups
            .into_iter()
            .map(|group| {
                let mut edges: Vec<(usize, usize)> = group
                    .edges
                    .into_iter()
                    .map(|(row, col)| (col, row))
                    .collect();
                edges.sort_unstable();
                Group { edges }
            })
            .collect()
    } else {
        groups
    })
}

/// The connected components of the matched pairs. Each is a matched set.
fn connected_groups(matched: &[(usize, usize)], rows: usize, cols: usize) -> Vec<Group> {
    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    let mut parent: Vec<usize> = (0..rows + cols).collect();
    for (row, col) in matched {
        let a = find(&mut parent, *row);
        let b = find(&mut parent, rows + *col);
        if a != b {
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut index_of_root: FxHashMap<usize, usize> = FxHashMap::default();
    let mut groups: Vec<Group> = Vec::new();
    for (row, col) in matched {
        let root = find(&mut parent, *row);
        let position = *index_of_root.entry(root).or_insert_with(|| {
            groups.push(Group { edges: Vec::new() });
            groups.len() - 1
        });
        groups[position].edges.push((*row, *col));
    }
    for group in &mut groups {
        group.edges.sort_unstable();
    }
    groups.sort_unstable_by_key(|group| group.edges[0]);
    groups
}

/// Pair matching of one stratum with `ratio` candidates per anchor, as
/// MatchIt arranges `optmatch::fullmatch` for `method = "optimal"`.
///
/// With fewer candidates than anchors the problem is solved with the groups
/// exchanged, so each candidate gets one anchor. With more than 1 and fewer
/// than `ratio` candidates per anchor, every candidate is used and an anchor
/// gets between the whole number of candidates per anchor, rounded down, and
/// `ratio`.
fn pair_match_stratum(
    rows: usize,
    cols: usize,
    edges: &[Edge],
    ratio: usize,
) -> Result<Vec<Group>, StratumError> {
    if rows == 0 || cols == 0 || edges.is_empty() {
        return Ok(Vec::new());
    }
    let ratio_f = to_f64(ratio);
    let available = to_f64(cols) / to_f64(rows);
    if available < 1.0 {
        let groups = full_match_stratum(
            cols,
            rows,
            &transpose(edges),
            Restrictions {
                min_controls: 1.0,
                max_controls: 1.0,
                retain: Retain::MeanControls(1.0),
            },
        )?;
        return Ok(groups
            .into_iter()
            .map(|group| {
                let mut edges: Vec<(usize, usize)> = group
                    .edges
                    .into_iter()
                    .map(|(row, col)| (col, row))
                    .collect();
                edges.sort_unstable();
                Group { edges }
            })
            .collect());
    }
    let restrictions = if available < ratio_f {
        Restrictions {
            min_controls: available.floor().min(ratio_f),
            max_controls: ratio_f,
            retain: Retain::MeanControls(available),
        }
    } else {
        Restrictions {
            min_controls: ratio_f,
            max_controls: ratio_f,
            retain: Retain::MeanControls(ratio_f),
        }
    };
    full_match_stratum(rows, cols, edges, restrictions)
}

/// The groups of the stratum with their units in the order of the slices.
struct Stratum {
    /// Positions in the anchor slice, in slice order.
    anchors: Vec<usize>,
    /// Positions in the candidate slice, in slice order.
    candidates: Vec<usize>,
}

/// What building the pairs of a stratum reads.
struct PairBuilder<'a, R: MatchingRecord, G: ?Sized> {
    anchors: &'a [R],
    candidates: &'a [R],
    criteria: &'a MatchingCriteria,
    constraints: &'a G,
    config: &'a DistanceConfig,
    caliper: Option<f64>,
    anchor_strata: Option<StrataValues<'a>>,
    candidate_strata: Option<StrataValues<'a>>,
}

impl<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized> PairBuilder<'_, R, G> {
    /// The distance of an anchor and a candidate, by their positions in the
    /// anchor and candidate slices, or `None` when the pair has none. Every
    /// cost of the network is read here.
    fn pair_distance(
        &self,
        anchor_idx: usize,
        candidate_idx: usize,
        metric: &impl DistanceMetric<R>,
    ) -> Option<f64> {
        metric.distance(&self.anchors[anchor_idx], &self.candidates[candidate_idx])
    }

    /// Every admissible pair of the stratum, sorted by row and then column.
    fn edges(&self, stratum: &Stratum) -> Vec<Edge> {
        let used = UsedControlsVec::with_capacity(0);
        let used_unique = FxHashSet::default();
        let interner: RapidHashMap<String, crate::types::UniqueValueId> = RapidHashMap::default();
        let metric = self.config.metric();
        let lookup = self.config.scalar();

        // With one number per record and a caliper, each anchor reads only the
        // candidates inside the window.
        let sorted: Option<(Vec<f64>, Vec<usize>)> = match (lookup, self.caliper) {
            (Some(lookup), Some(_)) => {
                let mut entries: Vec<(f64, usize)> = stratum
                    .candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(col, idx)| {
                        lookup
                            .value(&self.candidates[*idx])
                            .map(|value| (value, col))
                    })
                    .collect();
                entries.sort_unstable_by(|left, right| {
                    left.0.total_cmp(&right.0).then(left.1.cmp(&right.1))
                });
                Some(entries.into_iter().unzip())
            }
            _ => None,
        };

        let mut edges = Vec::new();
        let mut row_cols: Vec<usize> = Vec::new();
        for (row, anchor_idx) in stratum.anchors.iter().enumerate() {
            let anchor = &self.anchors[*anchor_idx];
            row_cols.clear();
            match (&sorted, lookup, self.caliper) {
                (Some((keys, cols)), Some(lookup), Some(width)) => {
                    let Some(x) = lookup.value(anchor) else {
                        continue;
                    };
                    let slack = 8.0 * f64::EPSILON * (x.abs() + width.abs() + 1.0);
                    let start = keys.partition_point(|key| *key < x - width - slack);
                    let stop = keys.partition_point(|key| *key <= x + width + slack);
                    row_cols.extend_from_slice(&cols[start..stop]);
                    row_cols.sort_unstable();
                }
                _ => row_cols.extend(0..stratum.candidates.len()),
            }
            for col in &row_cols {
                let candidate_idx = stratum.candidates[*col];
                let candidate = &self.candidates[candidate_idx];
                if anchor.id() == candidate.id() {
                    continue;
                }
                let context = ConstraintContext {
                    criteria: self.criteria,
                    used_controls: &used,
                    used_unique: &used_unique,
                    unique_interner: &interner,
                    control_idx: ControlIdx::new(candidate_idx),
                    case_strata_values: self
                        .anchor_strata
                        .as_ref()
                        .and_then(|values| values.row(*anchor_idx)),
                    control_strata_values: self
                        .candidate_strata
                        .as_ref()
                        .and_then(|values| values.row(candidate_idx)),
                };
                if self
                    .constraints
                    .first_blocking_reason(anchor, candidate, &context)
                    .is_some()
                {
                    continue;
                }
                let Some(distance) = self.pair_distance(*anchor_idx, candidate_idx, &metric) else {
                    continue;
                };
                if !distance.is_finite() || self.caliper.is_some_and(|width| distance > width) {
                    continue;
                }
                edges.push(Edge {
                    row,
                    col: *col,
                    distance,
                });
            }
        }
        edges
    }
}

/// Group the units into exact strata, numbered in order of the first anchor.
/// Strata without both groups are left out.
fn exact_strata<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    anchor_strata: &StrataValues<'_>,
    candidate_strata: &StrataValues<'_>,
) -> Vec<Stratum> {
    let mut index: FxHashMap<Vec<Option<&str>>, usize> = FxHashMap::default();
    let mut strata: Vec<Stratum> = Vec::new();
    for anchor in 0..anchors.len() {
        let key: Vec<Option<&str>> = anchor_strata.row(anchor).unwrap_or(&[]).to_vec();
        let position = *index.entry(key).or_insert_with(|| {
            strata.push(Stratum {
                anchors: Vec::new(),
                candidates: Vec::new(),
            });
            strata.len() - 1
        });
        strata[position].anchors.push(anchor);
    }
    for candidate in 0..candidates.len() {
        let key: Vec<Option<&str>> = candidate_strata.row(candidate).unwrap_or(&[]).to_vec();
        if let Some(position) = index.get(&key) {
            strata[*position].candidates.push(candidate);
        }
    }
    strata.retain(|stratum| !stratum.candidates.is_empty());
    strata
}

/// Solves one stratum from its numbers of anchors and candidates and its pairs.
type StratumSolver<'a> =
    dyn Fn(usize, usize, &[Edge]) -> Result<Vec<Group>, StratumError> + Sync + 'a;

/// A solved stratum: its matched sets by position in the anchor and candidate
/// slices.
type Solved = Vec<Vec<(usize, usize)>>;

fn solve_strata<R, G>(
    anchors: &[R],
    candidates: &[R],
    request: &OptimalRequest<'_, G>,
    solve: &StratumSolver<'_>,
) -> Result<Solved, OptimalError>
where
    R: MatchingRecord,
    G: ConstraintGroup<R> + ?Sized,
{
    let criteria = request.criteria;
    let validated = criteria.validate()?;
    let criteria = validated.as_ref();
    if criteria.allow_replacement {
        return Err(OptimalError::Option {
            option: "replace",
            message: "optimal matching uses each candidate once; drop replacement".into(),
        });
    }
    if criteria.unique_by_key.is_some() {
        return Err(OptimalError::Option {
            option: "unique_by_key",
            message: "optimal matching does not take a unique key; drop it".into(),
        });
    }
    let date_default = DistanceConfig::date(criteria.typed_birth_date_caliper());
    let config = request.distance_config.unwrap_or(&date_default);
    let required = &criteria.required_strata;
    let anchor_strata = StrataValues::of_controls(anchors, required);
    let candidate_strata = StrataValues::of_controls(candidates, required);
    let strata = exact_strata(anchors, candidates, &anchor_strata, &candidate_strata);
    let builder = PairBuilder {
        anchors,
        candidates,
        criteria,
        constraints: request.constraints,
        config,
        caliper: config.caliper(),
        anchor_strata: (!required.is_empty()).then_some(anchor_strata),
        candidate_strata: (!required.is_empty()).then_some(candidate_strata),
    };

    let results: Vec<Result<Vec<Group>, StratumError>> = strata
        .par_iter()
        .map(|stratum| {
            let edges = builder.edges(stratum);
            solve(stratum.anchors.len(), stratum.candidates.len(), &edges)
        })
        .collect();

    let mut solved: Solved = Vec::new();
    for (number, (stratum, result)) in strata.iter().zip(results).enumerate() {
        match result {
            Ok(groups) => {
                for group in groups {
                    solved.push(
                        group
                            .edges
                            .into_iter()
                            .map(|(row, col)| (stratum.anchors[row], stratum.candidates[col]))
                            .collect(),
                    );
                }
            }
            Err(StratumError::Option(option, message)) => {
                return Err(OptimalError::Option { option, message });
            }
            Err(StratumError::Infeasible(message)) => {
                return Err(OptimalError::Infeasible {
                    stratum: number,
                    message,
                });
            }
            Err(StratumError::TooLarge) => return Err(OptimalError::TooLarge { stratum: number }),
        }
    }
    if solved.is_empty() {
        return Err(OptimalError::NoMatches);
    }
    // Sets in the order of their first anchor, and then of their first
    // candidate, which does not depend on how the strata were scheduled.
    solved.sort_unstable_by_key(|set| set.iter().copied().min());
    Ok(solved)
}

/// Optimal pair matching: the assignment of `ratio` candidates to each anchor
/// that minimises the sum of the distances of the pairs, as
/// `matchit(method = "optimal")`.
///
/// The anchors are the focal group, as in the nearest neighbour method, so a
/// match for the effect in the controls is made with the controls as anchors.
/// The pairs are in the order of the anchors, and an anchor's candidates in the
/// order of the candidate slice. An anchor gets `ratio` candidates unless there
/// are fewer than `ratio` candidates per anchor in its exact stratum, and then
/// the candidates are shared out as evenly as the limits allow. With fewer
/// candidates than anchors in a stratum, each candidate gets one anchor
/// instead, so some anchors have none.
///
/// Pairs outside the caliper or blocked by a constraint are forbidden. Where
/// they leave a stratum with no assignment that gives every anchor with an
/// admissible candidate its candidates, the result is
/// [`OptimalError::Infeasible`]. `matchit(method = "optimal")` itself ignores
/// calipers.
///
/// # Errors
///
/// [`OptimalError`] for invalid criteria, replacement, a unique key, a stratum
/// without a feasible assignment, or when no stratum has an admissible pair.
pub fn match_optimal<R, G>(
    anchors: &[R],
    candidates: &[R],
    request: &OptimalRequest<'_, G>,
) -> Result<MatchOutcome, OptimalError>
where
    R: MatchingRecord,
    G: ConstraintGroup<R> + ?Sized,
{
    let ratio = request.criteria.match_ratio;
    let sets = solve_strata(anchors, candidates, request, &|rows, cols, edges| {
        pair_match_stratum(rows, cols, edges, ratio)
    })?;

    let mut positions: Vec<(usize, usize)> = sets.into_iter().flatten().collect();
    positions.sort_unstable();
    let mut matched_anchors = 0_usize;
    let mut below_ratio = 0_usize;
    let mut previous: Option<usize> = None;
    let mut run = 0_usize;
    let mut pairs = Vec::with_capacity(positions.len());
    for (anchor, candidate) in &positions {
        pairs.push(MatchedPair::new(
            anchors[*anchor].id(),
            candidates[*candidate].id(),
        ));
        if previous == Some(*anchor) {
            run += 1;
        } else {
            if previous.is_some() && run < ratio {
                below_ratio += 1;
            }
            matched_anchors += 1;
            previous = Some(*anchor);
            run = 1;
        }
    }
    if previous.is_some() && run < ratio {
        below_ratio += 1;
    }
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchors.len(),
        requested_estimand: request.criteria.estimand,
        realized_estimand: request.criteria.estimand,
        matched_anchors,
        pairs_selected: pairs.len(),
        anchors_with_no_candidates: anchors.len() - matched_anchors,
        anchors_below_required_ratio: below_ratio,
        ..MatchDiagnostics::default()
    };
    finalize_estimand_diagnostics(
        &mut diagnostics,
        request.criteria.estimand,
        anchors.len(),
        None,
    );
    let used = pairs.len();
    Ok(build_outcome(
        pairs,
        matched_anchors,
        anchors.len(),
        used,
        diagnostics,
    ))
}

/// Subclasses of an optimal full match, with the weights of each estimand.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FullMatchOutcome {
    /// Subclass of each matched unit. Subclasses are numbered from 0 in the
    /// order of their first anchor in the anchor slice.
    pub assignments: HashMap<String, usize>,
    /// Ids of the units in no subclass, anchors first and then candidates, each
    /// in the order of its slice.
    pub dropped_ids: Vec<String>,
    /// Counts and anchor share of each subclass.
    pub subclasses: Vec<SubclassSummary>,
    /// Subclass weights for the effect in the anchor group: 1 for an anchor
    /// and the number of anchors over the number of candidates for a candidate.
    pub att_weights: HashMap<String, f64>,
    /// Subclass weights for the effect in the candidate group.
    pub atc_weights: HashMap<String, f64>,
    /// Subclass weights for the effect in the whole population.
    pub ate_weights: HashMap<String, f64>,
}

impl FullMatchOutcome {
    /// Number of matched units.
    #[must_use]
    pub fn retained_units(&self) -> usize {
        self.assignments.len()
    }

    /// The weights for an estimand, or `None` for the matched-sample effect,
    /// which full matching does not define separately.
    #[must_use]
    pub const fn weights(&self, estimand: Estimand) -> Option<&HashMap<String, f64>> {
        match estimand {
            Estimand::Att => Some(&self.att_weights),
            Estimand::Atc => Some(&self.atc_weights),
            Estimand::Ate => Some(&self.ate_weights),
            Estimand::Atm => None,
        }
    }

    /// MatchIt's `match_data()` of the full match: the units with their weight
    /// and subclass, in the order of `units`.
    ///
    /// `weights` is one of the outcome's weight maps. With `normalize` the
    /// weights of each group are rescaled to a mean of 1 over its matched
    /// units, as `matchit()` does. The subclass is numbered from 1. A unit with
    /// no weight has weight 0 and no subclass, and is left out with
    /// `drop_unmatched`.
    #[must_use]
    pub fn match_data(
        &self,
        units: &Units,
        weights: &HashMap<String, f64>,
        normalize: bool,
        drop_unmatched: bool,
    ) -> MatchData {
        let mut set = UnitWeightSet::default();
        for (id, is_anchor) in units.ids().iter().zip(units.anchor_flags()) {
            let Some(weight) = weights.get(id) else {
                continue;
            };
            if *is_anchor {
                set.anchor.insert(id.clone(), *weight);
            } else {
                set.candidate.insert(id.clone(), *weight);
            }
        }
        let set = if normalize { set.normalized() } else { set };
        let mut out = MatchData {
            id: Vec::new(),
            role: Vec::new(),
            weight: Vec::new(),
            subclass: Vec::new(),
        };
        for (id, is_anchor) in units.ids().iter().zip(units.anchor_flags()) {
            let group = if *is_anchor {
                &set.anchor
            } else {
                &set.candidate
            };
            let weight = group.get(id).copied();
            if weight.is_none() && drop_unmatched {
                continue;
            }
            out.id.push(id.clone());
            out.role.push(if *is_anchor {
                UnitRole::Anchor
            } else {
                UnitRole::Candidate
            });
            out.weight.push(weight.unwrap_or(0.0));
            out.subclass.push(
                weight
                    .and_then(|_| self.assignments.get(id))
                    .map(|subclass| subclass + 1),
            );
        }
        out
    }
}

/// Optimal full matching: subclasses of one anchor with several candidates or
/// one candidate with several anchors that minimise the sum of the distances
/// of the matched pairs, within the limits of `config`, as
/// `matchit(method = "full")`.
///
/// The anchors are the focal group. The limits count candidates per anchor, so
/// a match for the effect in the controls is made with the controls as anchors.
/// A unit with no admissible pair is in no subclass. The weights are the
/// subclass weights of [`FullMatchOutcome`], before MatchIt's rescaling, which
/// [`FullMatchOutcome::match_data`] applies on request.
///
/// A caliper, an exact stratum or a constraint removes pairs from the problem.
/// `matchit(method = "full")` also takes calipers, but only when `min.controls`
/// is 0.
///
/// # Errors
///
/// [`OptimalError`] for invalid criteria, replacement, a unique key, limits
/// that are out of range, a stratum where the limits cannot be met, or when no
/// stratum has an admissible pair.
pub fn match_full<R, G>(
    anchors: &[R],
    candidates: &[R],
    request: &OptimalRequest<'_, G>,
    config: &FullMatchConfig,
) -> Result<FullMatchOutcome, OptimalError>
where
    R: MatchingRecord,
    G: ConstraintGroup<R> + ?Sized,
{
    let restrictions = Restrictions {
        min_controls: config.min_controls,
        max_controls: config.max_controls,
        retain: config.retain,
    };
    let sets = solve_strata(anchors, candidates, request, &|rows, cols, edges| {
        full_match_stratum(rows, cols, edges, restrictions)
    })?;

    let mut assignments: HashMap<String, usize> = HashMap::new();
    let mut anchor_counts = Vec::with_capacity(sets.len());
    let mut candidate_counts = Vec::with_capacity(sets.len());
    let mut anchor_flags: Vec<(String, bool, usize)> = Vec::new();
    for (subclass, set) in sets.iter().enumerate() {
        let mut anchor_positions: Vec<usize> = set.iter().map(|pair| pair.0).collect();
        anchor_positions.sort_unstable();
        anchor_positions.dedup();
        let mut candidate_positions: Vec<usize> = set.iter().map(|pair| pair.1).collect();
        candidate_positions.sort_unstable();
        candidate_positions.dedup();
        anchor_counts.push(anchor_positions.len());
        candidate_counts.push(candidate_positions.len());
        for position in anchor_positions {
            anchor_flags.push((anchors[position].id().to_string(), true, subclass));
        }
        for position in candidate_positions {
            anchor_flags.push((candidates[position].id().to_string(), false, subclass));
        }
    }
    let subclasses = build_subclass_summaries(&anchor_counts, &candidate_counts);
    let mut anchor_effect = HashMap::new();
    let mut candidate_effect = HashMap::new();
    let mut population_effect = HashMap::new();
    for (id, is_anchor, subclass) in anchor_flags {
        let Some(share) = subclasses[subclass].subclass_propensity else {
            continue;
        };
        anchor_effect.insert(
            id.clone(),
            subclass_weight_for_estimand(Estimand::Att, share, is_anchor),
        );
        candidate_effect.insert(
            id.clone(),
            subclass_weight_for_estimand(Estimand::Atc, share, is_anchor),
        );
        population_effect.insert(
            id.clone(),
            subclass_weight_for_estimand(Estimand::Ate, share, is_anchor),
        );
        assignments.insert(id, subclass);
    }
    let mut dropped_ids = Vec::new();
    for rows in [anchors, candidates] {
        for row in rows {
            if !assignments.contains_key(row.id()) {
                dropped_ids.push(row.id().to_string());
            }
        }
    }
    Ok(FullMatchOutcome {
        assignments,
        dropped_ids,
        subclasses,
        att_weights: anchor_effect,
        atc_weights: candidate_effect,
        ate_weights: population_effect,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(row: usize, col: usize, distance: f64) -> Edge {
        Edge { row, col, distance }
    }

    fn dense(costs: &[&[f64]]) -> (usize, usize, Vec<Edge>) {
        let mut edges = Vec::new();
        for (row, line) in costs.iter().enumerate() {
            for (col, distance) in line.iter().enumerate() {
                edges.push(edge(row, col, *distance));
            }
        }
        (costs.len(), costs[0].len(), edges)
    }

    fn sets(groups: &[Group]) -> Vec<Vec<(usize, usize)>> {
        groups.iter().map(|group| group.edges.clone()).collect()
    }

    fn solve(
        rows: usize,
        cols: usize,
        edges: &[Edge],
        min: f64,
        max: f64,
        retain: Retain,
    ) -> Vec<Vec<(usize, usize)>> {
        let restrictions = Restrictions {
            min_controls: min,
            max_controls: max,
            retain,
        };
        match full_match_stratum(rows, cols, edges, restrictions) {
            Ok(groups) => sets(&groups),
            Err(StratumError::Infeasible(message)) => panic!("infeasible: {message}"),
            Err(StratumError::Option(option, message)) => panic!("{option}: {message}"),
            Err(StratumError::TooLarge) => panic!("too large"),
        }
    }

    #[test]
    fn a_two_by_two_assignment_takes_the_cheaper_pairing() {
        // Costs [[4, 1], [2, 3]]: the cross pairing costs 3, the diagonal 7.
        let (rows, cols, edges) = dense(&[&[4.0, 1.0], &[2.0, 3.0]]);
        let got = solve(rows, cols, &edges, 1.0, 1.0, Retain::MeanControls(1.0));
        assert_eq!(got, vec![vec![(0, 1)], vec![(1, 0)]]);
    }

    #[test]
    fn full_matching_attaches_the_spare_candidate_to_its_nearest_anchor() {
        // Two anchors, three candidates. Candidate 2 is cheapest with anchor 1.
        let (rows, cols, edges) = dense(&[&[1.0, 5.0, 9.0], &[6.0, 2.0, 3.0]]);
        let got = solve(rows, cols, &edges, 0.0, f64::INFINITY, Retain::All);
        assert_eq!(got, vec![vec![(0, 0)], vec![(1, 1), (1, 2)]]);
    }

    #[test]
    fn full_matching_can_give_one_candidate_several_anchors() {
        // Three anchors, one candidate: every anchor joins it.
        let (rows, cols, edges) = dense(&[&[1.0], &[2.0], &[3.0]]);
        let got = solve(rows, cols, &edges, 0.0, f64::INFINITY, Retain::All);
        assert_eq!(got, vec![vec![(0, 0), (1, 0), (2, 0)]]);
    }

    #[test]
    fn max_controls_one_uses_as_many_candidates_as_there_are_anchors() {
        // Two anchors, three candidates, no more than one candidate per anchor:
        // the two cheapest of the pairings, leaving candidate 2 out. Without a
        // mean the limit cannot be met, since every candidate is to be used.
        let (rows, cols, edges) = dense(&[&[1.0, 5.0, 9.0], &[6.0, 2.0, 3.0]]);
        let got = solve(rows, cols, &edges, 0.0, 1.0, Retain::MeanControls(1.0));
        assert_eq!(got, vec![vec![(0, 0)], vec![(1, 1)]]);
    }

    #[test]
    fn a_ratio_of_two_gives_each_anchor_two_candidates() {
        let (rows, cols, edges) = dense(&[&[1.0, 2.0, 9.0, 9.0, 8.0], &[9.0, 9.0, 1.0, 3.0, 9.0]]);
        let groups =
            pair_match_stratum(rows, cols, &edges, 2).unwrap_or_else(|_| panic!("feasible"));
        assert_eq!(
            sets(&groups),
            vec![vec![(0, 0), (0, 1)], vec![(1, 2), (1, 3)]]
        );
    }

    #[test]
    fn with_fewer_candidates_than_anchors_each_candidate_gets_one_anchor() {
        let (rows, cols, edges) = dense(&[&[1.0, 9.0], &[2.0, 9.0], &[9.0, 4.0]]);
        let groups =
            pair_match_stratum(rows, cols, &edges, 1).unwrap_or_else(|_| panic!("feasible"));
        assert_eq!(sets(&groups), vec![vec![(0, 0)], vec![(2, 1)]]);
    }

    #[test]
    fn omitting_a_fraction_of_the_candidates_drops_the_dearest() {
        // Two anchors, four candidates, half omitted: the two cheapest pairs.
        let (rows, cols, edges) = dense(&[&[1.0, 7.0, 8.0, 9.0], &[7.0, 2.0, 9.0, 8.0]]);
        let got = solve(
            rows,
            cols,
            &edges,
            0.0,
            f64::INFINITY,
            Retain::OmitFraction(0.5),
        );
        assert_eq!(got, vec![vec![(0, 0)], vec![(1, 1)]]);
    }

    #[test]
    fn a_limit_that_cannot_be_met_is_infeasible() {
        // Two anchors, one candidate, no more than one anchor per candidate.
        let (rows, cols, edges) = dense(&[&[1.0], &[2.0]]);
        let restrictions = Restrictions {
            min_controls: 1.0,
            max_controls: 1.0,
            retain: Retain::All,
        };
        assert!(matches!(
            full_match_stratum(rows, cols, &edges, restrictions),
            Err(StratumError::Infeasible(_))
        ));
    }

    #[test]
    fn forbidden_pairs_can_make_a_stratum_infeasible() {
        // Both anchors admit only candidate 0, so they cannot each have one.
        let edges = vec![edge(0, 0, 1.0), edge(1, 0, 1.0)];
        let restrictions = Restrictions {
            min_controls: 1.0,
            max_controls: 1.0,
            retain: Retain::MeanControls(1.0),
        };
        assert!(matches!(
            full_match_stratum(2, 2, &edges, restrictions),
            Err(StratumError::Infeasible(_))
        ));
    }

    /// The cheapest pair matching of every row to one column, by trying every
    /// injection.
    fn brute_force(costs: &[Vec<f64>]) -> f64 {
        fn go(row: usize, costs: &[Vec<f64>], used: &mut Vec<bool>) -> f64 {
            if row == costs.len() {
                return 0.0;
            }
            let mut best = f64::INFINITY;
            for col in 0..used.len() {
                if !used[col] {
                    used[col] = true;
                    best = best.min(costs[row][col] + go(row + 1, costs, used));
                    used[col] = false;
                }
            }
            best
        }
        go(0, costs, &mut vec![false; costs[0].len()])
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "small non-negative draws"
    )]
    fn pair_matching_agrees_with_brute_force_on_random_costs() {
        let mut state = 17_u64;
        let mut draw = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(state >> 40).unwrap_or(0)) / f64::from(1_u32 << 24)
        };
        for _ in 0..200 {
            let rows = 1 + (draw() * 4.0) as usize;
            let cols = rows + (draw() * 3.0) as usize;
            let costs: Vec<Vec<f64>> = (0..rows)
                .map(|_| (0..cols).map(|_| draw()).collect())
                .collect();
            let mut edges = Vec::new();
            for (row, line) in costs.iter().enumerate() {
                for (col, distance) in line.iter().enumerate() {
                    edges.push(edge(row, col, *distance));
                }
            }
            let groups =
                pair_match_stratum(rows, cols, &edges, 1).unwrap_or_else(|_| panic!("feasible"));
            let total: f64 = groups
                .iter()
                .flat_map(|group| group.edges.iter())
                .map(|(row, col)| costs[*row][*col])
                .sum();
            assert_eq!(groups.len(), rows);
            assert!(
                (total - brute_force(&costs)).abs() < 1e-9,
                "total {total} against {}",
                brute_force(&costs)
            );
        }
    }
}
