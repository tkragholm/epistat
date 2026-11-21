//! Coarsened exact matching.
//!
//! Numeric variables are cut into bins, categorical variables may be merged
//! into groups, and units are matched exactly on the result, as
//! `matchit(method = "cem")` does. This is not a call into the `cem` package:
//! the binning, the strata and the weights follow the documented behaviour of
//! the MatchIt method.
//!
//! # Rules
//!
//! * **Binning.** A numeric variable is binned by one of
//!   - a rule, [`CemRule`], that gives a number of bins from the values of
//!     all anchors and candidates (Sturges, Freedman-Diaconis, Scott), which
//!     is the default (Sturges);
//!   - a number of bins, which cuts the range from the smallest to the largest
//!     value into bins of equal width;
//!   - the cut points themselves;
//!   - a number of quantile groups, whose cut points are the quantiles of
//!     all values (type 7);
//!   - nothing, in which case units must share the raw value.
//!
//!   A value that lies on a cut point goes to the bin above it. The bins
//!   extend to minus and plus infinity, so the smallest and largest values
//!   belong to the first and last bin. One bin leaves the variable out of
//!   the matching.
//! * **Grouping.** A categorical variable is read from the unit's strata. Each
//!   group lists values that count as the first of them. Groups are applied in
//!   order, so a value that the first group renamed can be renamed by the
//!   second.
//! * **Strata.** A stratum is a combination of bins and groups. Units in a
//!   stratum that lacks one of the groups are dropped. The strata that remain
//!   are subclasses with the weights of the exact method, see
//!   [`exact_match_subclasses`](super::exact_match_subclasses).
//! * **k:k matching.** [`cem_k2k_pairs`] pairs the units of each stratum one to
//!   one, by a distance, in the order of the anchors, [`MatchOrder::Data`], or
//!   by the distance of the pair, [`MatchOrder::Closest`] and
//!   [`MatchOrder::Farthest`], with the same ties and replacement rules as
//!   [`NearestSelection`](super::NearestSelection): without replacement, ties
//!   to the candidate that comes first. Without a distance, a stratum's
//!   candidates are taken in an order drawn from the seed.
//! * **Missing values.** A missing or non-finite value is an error, as the
//!   bins of such a unit are undefined.
//!
//! # Cost
//!
//! Binning sorts each numeric variable once. The k:k pairing computes the
//! distance of every anchor to every candidate of its stratum, so a stratum
//! of `a` anchors and `c` candidates costs `a * c` distances.

use super::distance::DistanceConfig;
use super::distance::DistanceMetric;
use super::engine::{build_outcome, finalize_estimand_diagnostics};
use super::exact::{ExactMatchError, ExactMatchOutcome, subclasses_of_strata};
use super::nearest::MatchOrder;
use super::records::MatchingRecord;
use crate::types::{Estimand, MatchDiagnostics, MatchOutcome, MatchedPair};
use epistat_rng::Stream;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

/// A rule that gives a number of bins from the values of a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CemRule {
    /// Sturges: the logarithm of the number of values to base 2, plus 1,
    /// rounded up.
    #[default]
    Sturges,
    /// Freedman and Diaconis: bins of width twice the interquartile range
    /// over the cube root of the number of values. The values are first
    /// rounded to five significant digits. A zero range is widened to the
    /// range between lower and upper quantiles closer to the ends, and then to
    /// 3.5 standard deviations.
    FreedmanDiaconis,
    /// Scott: bins of width 3.5 standard deviations over the cube root of the
    /// number of values.
    Scott,
}

/// How a numeric variable is binned.
#[derive(Debug, Clone, PartialEq)]
pub enum CemBinning {
    /// A number of bins from a rule.
    Rule(CemRule),
    /// This many bins of equal width between the smallest and largest value.
    Bins(usize),
    /// Cut points, in any order. A value on a cut point goes to the bin above.
    Cutpoints(Vec<f64>),
    /// This many groups of equal share, cut at the quantiles of all values.
    Quantiles(usize),
    /// No binning: units must share the value.
    Exact,
}

impl Default for CemBinning {
    fn default() -> Self {
        Self::Rule(CemRule::default())
    }
}

/// What a variable is, and how it is coarsened.
#[derive(Debug, Clone, PartialEq)]
pub enum CemKind {
    /// A numeric field of the unit, binned.
    Numeric(CemBinning),
    /// A strata field of the unit, whose values are merged into groups. Each
    /// group counts as its first value. No groups leaves the values as they
    /// are.
    Categorical(Vec<Vec<String>>),
}

/// A variable that units are matched on.
#[derive(Debug, Clone, PartialEq)]
pub struct CemVariable {
    /// The numeric field or strata key.
    pub name: String,
    /// How it is coarsened.
    pub kind: CemKind,
}

impl CemVariable {
    /// A numeric variable with its binning.
    #[must_use]
    pub fn numeric(name: impl Into<String>, binning: CemBinning) -> Self {
        Self {
            name: name.into(),
            kind: CemKind::Numeric(binning),
        }
    }

    /// A categorical variable with its groups of values.
    #[must_use]
    pub fn categorical(name: impl Into<String>, groups: Vec<Vec<String>>) -> Self {
        Self {
            name: name.into(),
            kind: CemKind::Categorical(groups),
        }
    }
}

/// Why coarsened exact matching could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CemError {
    /// No variable was given.
    NoVariables,
    /// A unit has no value, or no finite value, for a variable.
    MissingValue {
        /// Id of the unit.
        id: String,
        /// Name of the variable.
        variable: String,
    },
    /// The binning of a variable is out of range.
    Binning {
        /// Name of the variable.
        variable: String,
        /// What is wrong.
        message: String,
    },
    /// No stratum holds units of both groups.
    NoSharedStratum,
    /// The order of the anchors needs a score, which a pair distance lacks.
    Order,
}

impl fmt::Display for CemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVariables => write!(f, "no variable to coarsen and match on was given"),
            Self::MissingValue { id, variable } => write!(
                f,
                "unit {id} has no finite value for the variable {variable}"
            ),
            Self::Binning { variable, message } => {
                write!(f, "the binning of {variable} is invalid: {message}")
            }
            Self::NoSharedStratum => write!(
                f,
                "no stratum holds units of both groups; coarsen the variables further \
                 or use fewer of them"
            ),
            Self::Order => write!(
                f,
                "the largest and smallest orders need a score; use data, random, \
                 closest or farthest"
            ),
        }
    }
}

impl std::error::Error for CemError {}

impl From<ExactMatchError> for CemError {
    fn from(error: ExactMatchError) -> Self {
        match error {
            ExactMatchError::NoVariables => Self::NoVariables,
            ExactMatchError::MissingValue { id, variable } => Self::MissingValue { id, variable },
            ExactMatchError::NoSharedStratum => Self::NoSharedStratum,
        }
    }
}

/// The value of a unit on one variable after coarsening.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Cell {
    Bin(usize),
    Raw(u64),
    Text(String),
}

/// The stratum of every anchor and candidate, or `None` where a unit is in a
/// stratum that lacks one of the groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CemStrata {
    /// Stratum of each anchor, in the order of the anchors.
    pub anchors: Vec<Option<usize>>,
    /// Stratum of each candidate, in the order of the candidates.
    pub candidates: Vec<Option<usize>>,
    /// Number of strata that hold both groups. They are numbered from 0 in
    /// order of first appearance among the anchors.
    pub count: usize,
}

fn binning_error(variable: &str, message: impl Into<String>) -> CemError {
    CemError::Binning {
        variable: variable.to_string(),
        message: message.into(),
    }
}

/// `x` to `digits` significant digits, rounding halves to even.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the decimal exponent of a finite double is within 324"
)]
fn signif(x: f64, digits: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let magnitude = x.abs();
    let e10 = digits - 1 - magnitude.log10().floor() as i32;
    if e10 > 300 {
        return x;
    }
    let sign = x.signum();
    if e10 > 0 {
        let scale = 10_f64.powi(e10);
        sign * ((magnitude * scale).round_ties_even() / scale)
    } else {
        let scale = 10_f64.powi(-e10);
        sign * ((magnitude / scale).round_ties_even() * scale)
    }
}

/// The type 7 quantile of sorted values: linear interpolation between order
/// statistics at position `(n - 1) p`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "the position lies between 0 and n - 1, and equal neighbours are \
              returned as they are rather than interpolated"
)]
fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * p;
    let lower = index.floor();
    let upper = index.ceil();
    let low = sorted[lower as usize];
    let high = sorted[upper as usize];
    if index > lower && high != low {
        let h = index - lower;
        (1.0 - h) * low + h * high
    } else {
        low
    }
}

fn sample_variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let squares: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    squares / (n - 1.0)
}

/// The number of bins a rule gives for these values.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the count is a positive whole number from a ceiling"
)]
fn rule_bins(rule: CemRule, values: &[f64]) -> usize {
    let n = values.len() as f64;
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = maximum - minimum;
    let count = match rule {
        CemRule::Sturges => (n.log2() + 1.0).ceil(),
        CemRule::Scott => {
            let h = 3.5 * sample_variance(values).sqrt() * n.powf(-1.0 / 3.0);
            if h > 0.0 {
                (range / h).ceil().max(1.0)
            } else {
                1.0
            }
        }
        CemRule::FreedmanDiaconis => {
            let mut rounded: Vec<f64> = values.iter().map(|v| signif(*v, 5)).collect();
            rounded.sort_by(f64::total_cmp);
            let mut h = 2.0 * (quantile7(&rounded, 0.75) - quantile7(&rounded, 0.25));
            if h == 0.0 {
                let mut tail = 0.25;
                loop {
                    tail /= 2.0;
                    if tail < 1.0 / 512.0 {
                        break;
                    }
                    h = (quantile7(&rounded, 1.0 - tail) - quantile7(&rounded, tail))
                        / (1.0 - 2.0 * tail);
                    if h != 0.0 {
                        break;
                    }
                }
            }
            if h == 0.0 {
                h = 3.5 * sample_variance(values).sqrt();
            }
            if h > 0.0 {
                (range / h * n.powf(1.0 / 3.0)).ceil()
            } else {
                1.0
            }
        }
    };
    count.max(1.0) as usize
}

/// The interior cut points of a variable: a value goes to the bin numbered by
/// how many of them are at or below it.
fn cutpoints(
    variable: &str,
    binning: &CemBinning,
    values: &[f64],
) -> Result<Option<Vec<f64>>, CemError> {
    let equal_width = |bins: usize| -> Vec<f64> {
        let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let step = (maximum - minimum) / bins as f64;
        (1..bins).map(|i| minimum + i as f64 * step).collect()
    };
    Ok(Some(match binning {
        CemBinning::Exact => return Ok(None),
        CemBinning::Rule(rule) => equal_width(rule_bins(*rule, values)),
        CemBinning::Bins(0) => {
            return Err(binning_error(
                variable,
                "0 bins; use 1 or more, or the exact binning for the raw values",
            ));
        }
        CemBinning::Bins(bins) => equal_width(*bins),
        CemBinning::Cutpoints(points) => {
            if points.is_empty() || points.iter().any(|p| !p.is_finite()) {
                return Err(binning_error(
                    variable,
                    "cut points must be a non-empty list of finite numbers",
                ));
            }
            let mut points = points.clone();
            points.sort_by(f64::total_cmp);
            points.dedup();
            points
        }
        CemBinning::Quantiles(groups) => {
            if *groups < 2 {
                return Err(binning_error(
                    variable,
                    "fewer than 2 quantile groups; use 2 or more",
                ));
            }
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            let step = 1.0 / *groups as f64;
            // Probabilities step / 1, ..., 1 - step in steps of `step`, as the
            // sequence from `step` to `1 - step` runs, with the last held at
            // its end.
            let last = 1.0 - step;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a small non-negative count from a floor"
            )]
            let count = ((last - step) / step + 1e-10).floor() as usize;
            (0..=count)
                .map(|i| quantile7(&sorted, (step + i as f64 * step).min(last)))
                .collect()
        }
    }))
}

/// Coarsen the units and number the strata that hold both groups.
///
/// # Errors
///
/// [`CemError`] when no variable is given, a unit has no value for a
/// variable, a binning is out of range, or no stratum holds both groups.
pub fn cem_strata<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    variables: &[CemVariable],
) -> Result<CemStrata, CemError> {
    coarsen(anchors, candidates, variables).map(|(strata, _)| strata)
}

/// Coarsened exact matching into subclasses with weights.
///
/// The strata are those of [`cem_strata`] and the weights those of
/// [`exact_match_subclasses`](super::exact_match_subclasses). The numbers of
/// the subclasses follow the first appearance among the anchors.
///
/// # Errors
///
/// As [`cem_strata`].
pub fn cem_subclasses<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    variables: &[CemVariable],
) -> Result<ExactMatchOutcome, CemError> {
    coarsen(anchors, candidates, variables).map(|(_, outcome)| outcome)
}

fn coarsen<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    variables: &[CemVariable],
) -> Result<(CemStrata, ExactMatchOutcome), CemError> {
    if variables.is_empty() {
        return Err(CemError::NoVariables);
    }
    let units: Vec<&R> = anchors.iter().chain(candidates.iter()).collect();
    let mut cells: Vec<Vec<Cell>> = vec![Vec::with_capacity(variables.len()); units.len()];
    for variable in variables {
        let missing = |unit: &R| CemError::MissingValue {
            id: unit.id().to_string(),
            variable: variable.name.clone(),
        };
        match &variable.kind {
            CemKind::Numeric(binning) => {
                let mut values = Vec::with_capacity(units.len());
                for unit in &units {
                    values.push(
                        unit.numeric(&variable.name)
                            .filter(|value| value.is_finite())
                            .ok_or_else(|| missing(unit))?,
                    );
                }
                match cutpoints(&variable.name, binning, &values)? {
                    Some(points) => {
                        for (row, value) in cells.iter_mut().zip(&values) {
                            row.push(Cell::Bin(points.partition_point(|p| p <= value)));
                        }
                    }
                    None => {
                        for (row, value) in cells.iter_mut().zip(&values) {
                            // Zero has two representations that are equal.
                            let value = if *value == 0.0 { 0.0 } else { *value };
                            row.push(Cell::Raw(value.to_bits()));
                        }
                    }
                }
            }
            CemKind::Categorical(groups) => {
                for (row, unit) in cells.iter_mut().zip(&units) {
                    let mut value = unit
                        .stratum(&variable.name)
                        .map(str::to_owned)
                        .ok_or_else(|| missing(unit))?;
                    for group in groups {
                        match group.first() {
                            Some(first) if group.contains(&value) => value.clone_from(first),
                            _ => {}
                        }
                    }
                    row.push(Cell::Text(value));
                }
            }
        }
    }

    // Stratum numbers in order of first appearance, anchors first.
    let mut stratum_of: HashMap<&[Cell], usize> = HashMap::new();
    let mut numbers = Vec::with_capacity(units.len());
    for row in &cells {
        let next = stratum_of.len();
        numbers.push(*stratum_of.entry(row.as_slice()).or_insert(next));
    }
    let (anchor_numbers, candidate_numbers) = numbers.split_at(anchors.len());
    let outcome = subclasses_of_strata(
        anchors,
        candidates,
        anchor_numbers,
        candidate_numbers,
        stratum_of.len(),
    )?;
    // The subclass of a unit is the one the weights code gave it.
    let subclass = |unit: &R| outcome.assignments.get(unit.id()).copied();
    let strata = CemStrata {
        anchors: anchors.iter().map(subclass).collect(),
        candidates: candidates.iter().map(subclass).collect(),
        count: outcome.subclasses.len(),
    };
    Ok((strata, outcome))
}

/// The nearest available candidate of an anchor: the smallest distance, and
/// among equal distances the smallest tie rank.
fn nearest_available<R: MatchingRecord>(
    anchor: &R,
    candidates: &[R],
    pool: &[usize],
    available: &[bool],
    metric: Option<&super::distance::ConfiguredMetric<'_>>,
    tie_rank: &[usize],
) -> Option<(usize, f64)> {
    let mut best: Option<(f64, usize, usize)> = None;
    for &candidate in pool {
        if !available[candidate] {
            continue;
        }
        let distance = match metric {
            Some(metric) => {
                match DistanceMetric::<R>::distance(metric, anchor, &candidates[candidate]) {
                    Some(distance) if distance.is_finite() => distance,
                    _ => continue,
                }
            }
            None => 0.0,
        };
        let better = best.is_none_or(|(d, _, rank)| match distance.total_cmp(&d) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Equal => tie_rank[candidate] < rank,
            std::cmp::Ordering::Greater => false,
        });
        if better {
            best = Some((distance, candidate, tie_rank[candidate]));
        }
    }
    best.map(|(distance, candidate, _)| (candidate, distance))
}

/// Pair the units of each stratum one to one, as `k2k = TRUE` does in
/// `matchit(method = "cem")`.
///
/// Each anchor takes the nearest candidate of its stratum by the distance, no
/// candidate twice. The anchors are taken in the order `order`: the order of
/// the slice, an order drawn from `seed`, or by the distance of the pairs,
/// nearest first for [`MatchOrder::Closest`] and farthest first for
/// [`MatchOrder::Farthest`], where an anchor whose candidate has gone takes the
/// next nearest and its place in the order by the new distance. Candidates at
/// the same distance rank by their place in the slice. With no `distance`
/// every candidate of a stratum is equally near, and the candidates rank in an
/// order drawn from `seed`. An anchor with no candidate left stays unmatched.
///
/// A pair distance with no value for a pair, as a forbidden entry of a
/// distance matrix, rules the pair out.
///
/// # Errors
///
/// [`CemError::Order`] for the largest and smallest orders, which need a
/// score.
pub fn cem_k2k_pairs<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    strata: &CemStrata,
    distance: Option<&DistanceConfig>,
    order: MatchOrder,
    seed: u64,
    estimand: Estimand,
) -> Result<MatchOutcome, CemError> {
    if matches!(order, MatchOrder::Largest | MatchOrder::Smallest) {
        return Err(CemError::Order);
    }
    let metric = distance.map(DistanceConfig::metric);

    let mut anchors_of: Vec<Vec<usize>> = vec![Vec::new(); strata.count];
    let mut candidates_of: Vec<Vec<usize>> = vec![Vec::new(); strata.count];
    for (index, stratum) in strata.anchors.iter().enumerate() {
        if let Some(stratum) = stratum {
            anchors_of[*stratum].push(index);
        }
    }
    for (index, stratum) in strata.candidates.iter().enumerate() {
        if let Some(stratum) = stratum {
            candidates_of[*stratum].push(index);
        }
    }

    // A tie goes to the candidate that comes first in the slice, or, with no
    // distance, in the drawn order.
    let mut tie_rank: Vec<usize> = (0..candidates.len()).collect();
    if metric.is_none() {
        Stream::seeded(seed).shuffle(&mut tie_rank);
    }
    let anchor_rank: Vec<usize> = if order == MatchOrder::Random {
        let mut ids: Vec<usize> = (0..anchors.len()).collect();
        Stream::seeded(seed.wrapping_add(1)).shuffle(&mut ids);
        let mut rank = vec![0; anchors.len()];
        for (position, index) in ids.into_iter().enumerate() {
            rank[index] = position;
        }
        rank
    } else {
        (0..anchors.len()).collect()
    };

    let mut available = vec![true; candidates.len()];
    let mut partner: Vec<Option<usize>> = vec![None; anchors.len()];
    for stratum in 0..strata.count {
        let mut slots = anchors_of[stratum].clone();
        if order == MatchOrder::Random {
            slots.sort_by_key(|index| anchor_rank[*index]);
        }
        let pool = &candidates_of[stratum];
        let find = |anchor: usize, available: &[bool]| {
            nearest_available(
                &anchors[anchor],
                candidates,
                pool,
                available,
                metric.as_ref(),
                &tie_rank,
            )
        };
        match order {
            MatchOrder::Data | MatchOrder::Random => {
                for anchor in slots {
                    if let Some((candidate, _)) = find(anchor, &available) {
                        available[candidate] = false;
                        partner[anchor] = Some(candidate);
                    }
                }
            }
            MatchOrder::Closest | MatchOrder::Farthest => {
                let closest = order == MatchOrder::Closest;
                let key = |distance: f64, entry: usize| {
                    (
                        OrderedFloat(if closest { -distance } else { distance }),
                        Reverse(entry),
                    )
                };
                let mut entries: Vec<(usize, usize)> = Vec::new();
                let mut queue = BinaryHeap::new();
                for &anchor in &slots {
                    if let Some((candidate, distance)) = find(anchor, &available) {
                        queue.push(key(distance, entries.len()));
                        entries.push((anchor, candidate));
                    }
                }
                while let Some((_, Reverse(entry))) = queue.pop() {
                    let (anchor, candidate) = entries[entry];
                    if available[candidate] {
                        available[candidate] = false;
                        partner[anchor] = Some(candidate);
                    } else if let Some((next, distance)) = find(anchor, &available) {
                        entries[entry] = (anchor, next);
                        queue.push(key(distance, entry));
                    }
                }
            }
            MatchOrder::Largest | MatchOrder::Smallest => unreachable!("refused above"),
        }
    }

    let mut pairs = Vec::new();
    let mut used = 0;
    for (anchor, candidate) in partner.iter().enumerate() {
        if let Some(candidate) = candidate {
            pairs.push(MatchedPair::new(
                anchors[anchor].id(),
                candidates[*candidate].id(),
            ));
            used += 1;
        }
    }
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchors.len(),
        anchors_with_no_candidates: anchors.len() - pairs.len(),
        matched_anchors: pairs.len(),
        pairs_selected: pairs.len(),
        ..MatchDiagnostics::default()
    };
    finalize_estimand_diagnostics(&mut diagnostics, estimand, anchors.len(), None);
    let matched = pairs.len();
    Ok(build_outcome(
        pairs,
        matched,
        anchors.len(),
        used,
        diagnostics,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn unit(id: &str, x: f64, group: &str) -> BaseRecord {
        let mut record = BaseRecord::new(
            id,
            NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid test date"),
        )
        .with_numeric("x", x);
        record.strata.insert("g".into(), group.into());
        record
    }

    #[test]
    fn a_value_on_a_cut_point_goes_up() {
        let anchors = vec![unit("a1", 1.0, "p"), unit("a2", 0.999, "p")];
        let candidates = vec![unit("c1", 1.5, "p"), unit("c2", 0.5, "p")];
        let variables = [CemVariable::numeric("x", CemBinning::Cutpoints(vec![1.0]))];
        let strata = cem_strata(&anchors, &candidates, &variables).unwrap();
        assert_eq!(strata.anchors, vec![Some(0), Some(1)]);
        assert_eq!(strata.candidates, vec![Some(0), Some(1)]);
    }

    #[test]
    fn groups_are_applied_in_order() {
        let anchors = vec![unit("a1", 0.0, "q")];
        let candidates = vec![unit("c1", 0.0, "r"), unit("c2", 0.0, "z")];
        let groups = vec![
            vec!["p".to_string(), "q".to_string()],
            vec!["r".to_string(), "p".to_string()],
        ];
        let variables = [CemVariable::categorical("g", groups)];
        let strata = cem_strata(&anchors, &candidates, &variables).unwrap();
        // q becomes p, which the second group turns into r.
        assert_eq!(strata.anchors, vec![Some(0)]);
        assert_eq!(strata.candidates, vec![Some(0), None]);
    }

    #[test]
    fn refuses_what_cannot_be_coarsened() {
        let anchors = vec![unit("a1", 1.0, "p")];
        let candidates = vec![unit("c1", 2.0, "p")];
        assert_eq!(
            cem_strata(&anchors, &candidates, &[]).unwrap_err(),
            CemError::NoVariables
        );
        assert_eq!(
            cem_strata(
                &anchors,
                &candidates,
                &[CemVariable::numeric("w", CemBinning::default())]
            )
            .unwrap_err(),
            CemError::MissingValue {
                id: "a1".into(),
                variable: "w".into()
            }
        );
        assert!(matches!(
            cem_strata(
                &anchors,
                &candidates,
                &[CemVariable::numeric("x", CemBinning::Bins(0))]
            )
            .unwrap_err(),
            CemError::Binning { .. }
        ));
        assert_eq!(
            cem_strata(
                &anchors,
                &candidates,
                &[CemVariable::numeric("x", CemBinning::Exact)]
            )
            .unwrap_err(),
            CemError::NoSharedStratum
        );
    }

    #[test]
    fn signif_rounds_to_digits() {
        assert!((signif(123_456.0, 5) - 123_460.0).abs() < 1e-9);
        assert!((signif(0.000_123_456, 5) - 0.000_123_46).abs() < 1e-15);
        assert!((signif(-2.5, 1) + 2.0).abs() < 1e-12);
    }
}
