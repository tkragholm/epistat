//! Matching with more than two treatment groups: the generalised propensity
//! score and vector matching.
//!
//! MatchIt has no counterpart, so this is built from parts that are each
//! checked against R, and the procedure as a whole is checked by those parts
//! only (see the end of this page).
//!
//! # Generalised propensity score
//!
//! [`generalised_propensity_score`] fits a multinomial logistic model of the
//! group on the encoded covariates, [`epistat_glm::Multinom`], and returns for
//! every unit the probability of each group, the generalised propensity score
//! of Imbens \[1\]. It is checked against `nnet::multinom()` in
//! `epistat-glm`.
//!
//! # Vector matching
//!
//! [`vector_match`] is the algorithm of Lopez and Gutman \[2\] for the effect
//! in a focal group, the reference treatment `t`, on `Z` groups:
//!
//! 1. Fit the generalised propensity score `r(l, x)` of every group `l`.
//! 2. With `common_support` set, drop the units outside the rectangular common
//!    support. For each group `l`, `low(l)` is the largest over the groups of
//!    the smallest `r(l, x)` among the units of that group, and `high(l)` the
//!    smallest of the largest. A unit is dropped when some `r(l, x)` is not
//!    strictly between `low(l)` and `high(l)`. With `refit` set, the model is
//!    fitted again once on the units that remain.
//! 3. For each group `t'` other than `t`: cluster all remaining units by
//!    k-means on `logit r(l, x)` for the `Z - 2` groups `l` other than `t` and
//!    `t'` ([`epistat_stats::cluster`]; with two groups there is nothing to
//!    cluster on and one cluster holds everyone). Within each cluster, match
//!    each unit of `t` to the unit of `t'` nearest in `logit r(t, x)`, with
//!    replacement, and refuse a match further apart than `caliper` standard
//!    deviations of that score (0.25 in the paper), the deviation, with divisor
//!    `n`, taken over the units of `t` and `t'` in the cluster. This is
//!    [`match_within_clusters`].
//! 4. The final cohort is the units of `t` matched in every comparison, with
//!    their matches.
//!
//! The number of clusters is the caller's: the paper gives no rule.
//! The k-means start is deterministic and not R's random one: the first
//! centre is the point with the smallest first coordinate, and each further
//! centre is the point farthest from the centres chosen so far (ties by
//! position). The algorithm is Hartigan and Wong's by default.
//!
//! [`balance_by_group`] gives the balance report of the focal group against
//! each other group on the final cohort, from the balance functions of this
//! crate.
//!
//! # What is checked
//!
//! Not checked against R as a whole. The parts are: the multinomial fit
//! against `nnet::multinom()` (`epistat-glm`), the k-means step against
//! `stats::kmeans()` from the same start centres (`epistat-stats`), and the
//! matching within clusters against `Matching::Match()` on the same scores and
//! clusters (`tests/r_vector_matching.rs`). The common-support rule, the choice
//! of start centres and the assembly of the final cohort have no R reference
//! and are covered by tests of their properties.
//!
//! # References
//!
//! 1. Imbens GW. The role of the propensity score in estimating dose-response
//!    functions. Biometrika. 2000;87(3):706-10.
//! 2. Lopez MJ, Gutman R. Estimation of causal effects with multiple
//!    treatments: a review and new ideas. Stat Sci. 2017;32(3):432-54.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use chrono::NaiveDate;
use epistat_glm::{Multinom, MultinomError};
use epistat_linalg::Mat;
use epistat_stats::cluster::{KMeansAlgorithm, KMeansError, kmeans};

use crate::matching::{
    CovariateEncodingConfig, CovariateRecord, EstimationError, MatchingRecord, encode_covariates,
};
use crate::types::{
    BalanceRecord, DistanceCaliper, MatchDiagnostics, MatchOutcome, MatchRatio, MatchedPair,
};
use crate::{
    BalanceInputError, BalanceReport, BalanceReportOptions, BalanceWeights, DistanceConfig,
    Estimand, MatchJob, RapidHashMap, balance_report_on_weights,
};

/// Why a multi-group match could not be made.
#[derive(Debug, Clone)]
pub enum MultigroupError {
    /// Fewer than two groups, a label outside them, or a focal group outside
    /// them.
    Groups {
        /// The number of groups.
        groups: usize,
    },
    /// An input list has the wrong length.
    Length {
        /// The input.
        what: &'static str,
        /// The length it needed.
        expected: usize,
        /// The length it had.
        found: usize,
    },
    /// The caliper is not a positive finite number of standard deviations.
    Caliper,
    /// The number of clusters is zero.
    Clusters,
    /// A generalised propensity score is 0 or 1 to machine precision, so its
    /// logit is not finite: the groups are separated by the covariates.
    Separated,
    /// No unit of the group is inside the common support.
    EmptyGroup {
        /// The group.
        group: usize,
    },
    /// The covariates could not be encoded.
    Estimation(EstimationError),
    /// The multinomial model could not be fitted.
    Fit(MultinomError),
    /// The k-means clustering failed.
    Cluster(KMeansError),
}

impl Display for MultigroupError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Groups { groups } => write!(
                f,
                "the group labels must lie in 0..{groups} with at least two groups, \
                 and the focal group must be one of them"
            ),
            Self::Length {
                what,
                expected,
                found,
            } => write!(
                f,
                "{what} has {found} entries where {expected} were expected"
            ),
            Self::Caliper => {
                f.write_str("the caliper must be a positive finite number of standard deviations")
            }
            Self::Clusters => f.write_str("the number of clusters must be at least 1"),
            Self::Separated => f.write_str(
                "a generalised propensity score is 0 or 1 to machine precision; \
                 the groups are separated by the covariates",
            ),
            Self::EmptyGroup { group } => {
                write!(f, "no unit of group {group} lies inside the common support")
            }
            Self::Estimation(e) => write!(f, "{e}"),
            Self::Fit(e) => write!(f, "{e}"),
            Self::Cluster(e) => write!(f, "{e}"),
        }
    }
}

impl Error for MultigroupError {}

impl From<EstimationError> for MultigroupError {
    fn from(e: EstimationError) -> Self {
        Self::Estimation(e)
    }
}

impl From<MultinomError> for MultigroupError {
    fn from(e: MultinomError) -> Self {
        Self::Fit(e)
    }
}

impl From<KMeansError> for MultigroupError {
    fn from(e: KMeansError) -> Self {
        Self::Cluster(e)
    }
}

/// How the generalised propensity score is fitted.
#[derive(Debug, Clone)]
pub struct GpsConfig {
    /// Covariate encoding. The default has an intercept.
    pub encoding: CovariateEncodingConfig,
    /// The multinomial fit controls.
    pub fit: Multinom,
}

impl Default for GpsConfig {
    fn default() -> Self {
        Self {
            encoding: CovariateEncodingConfig::builder()
                .include_intercept(true)
                .build(),
            fit: Multinom::default(),
        }
    }
}

/// The generalised propensity score of a set of units.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralisedPropensityScore {
    /// The number of groups.
    pub groups: usize,
    /// The ids of the units, in the order of the probabilities.
    pub ids: Vec<String>,
    /// The probability of each group for each unit, row by row, `ids.len() *
    /// groups` values.
    pub probabilities: Vec<f64>,
    /// The coefficients of the groups after the first, group by group.
    pub coefficients: Vec<f64>,
    /// The names of the design columns.
    pub column_names: Vec<String>,
    /// Whether the fit converged.
    pub converged: bool,
    /// Whether a fitted probability is within ten machine epsilons of 0 or 1.
    pub fitted_at_boundary: bool,
}

impl GeneralisedPropensityScore {
    /// The probability of `group` for the unit in row `row`.
    #[must_use]
    pub fn probability(&self, row: usize, group: usize) -> f64 {
        self.probabilities[row * self.groups + group]
    }

    /// The logit of that probability, `None` when it is not finite.
    #[must_use]
    pub fn logit(&self, row: usize, group: usize) -> Option<f64> {
        let p = self.probability(row, group);
        let value = (p / (1.0 - p)).ln();
        value.is_finite().then_some(value)
    }
}

fn check_groups(groups: &[usize], n_groups: usize, n: usize) -> Result<(), MultigroupError> {
    if n_groups < 2 || groups.iter().any(|&g| g >= n_groups) {
        return Err(MultigroupError::Groups { groups: n_groups });
    }
    if groups.len() != n {
        return Err(MultigroupError::Length {
            what: "groups",
            expected: n,
            found: groups.len(),
        });
    }
    Ok(())
}

/// Fits the generalised propensity score of `records`, whose groups are
/// `groups` (labels in `0..n_groups`).
///
/// # Errors
///
/// See [`MultigroupError`].
pub fn generalised_propensity_score<R: CovariateRecord>(
    records: &[R],
    groups: &[usize],
    n_groups: usize,
    config: &GpsConfig,
) -> Result<GeneralisedPropensityScore, MultigroupError> {
    check_groups(groups, n_groups, records.len())?;
    let matrix = encode_covariates(records, &config.encoding)?;
    let x = Mat::from_fn(matrix.n_rows, matrix.n_cols, |i, j| {
        matrix.values[i * matrix.n_cols + j]
    });
    let fit = config.fit.fit(x.as_ref(), groups, n_groups, None)?;
    Ok(GeneralisedPropensityScore {
        groups: n_groups,
        ids: matrix.row_ids,
        probabilities: fit.probabilities,
        coefficients: fit.coefficients,
        column_names: matrix.column_names,
        converged: fit.converged,
        fitted_at_boundary: fit.fitted_at_boundary,
    })
}

fn dummy_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).expect("a valid date")
}

/// Matches each focal unit to the nearest other unit in its cluster by score,
/// with replacement, and returns the pairs.
///
/// `is_focal` marks the units to match, the rest are the candidates. A focal
/// unit is matched to the candidate of its cluster with the nearest `score`,
/// the lowest position on a tie, unless that candidate is further than
/// `caliper` standard deviations away. The standard deviation, with divisor
/// `n`, is that of the scores of all units in the cluster, as
/// `Matching::Match()` takes it over the units it is given. (`matchit()`
/// divides by `n - 1`.) A focal unit with
/// no candidate in its cluster or within the caliper is not matched. Pairs are
/// listed by cluster, in order of the cluster label, then in the order the
/// matching assigns them.
///
/// The matching is the nearest neighbour matching of this crate, checked
/// against `matchit()`, and the whole is checked against `Matching::Match()`
/// on the same scores and clusters.
///
/// # Errors
///
/// [`MultigroupError::Length`] for lists of different lengths and
/// [`MultigroupError::Caliper`] for a caliper that is not positive and finite.
///
/// # Panics
///
/// Never: the ratio of 1 it asks the matcher for is valid.
pub fn match_within_clusters(
    ids: &[String],
    score: &[f64],
    is_focal: &[bool],
    cluster: &[usize],
    caliper: Option<f64>,
) -> Result<Vec<MatchedPair>, MultigroupError> {
    let n = ids.len();
    for (what, found) in [
        ("score", score.len()),
        ("is_focal", is_focal.len()),
        ("cluster", cluster.len()),
    ] {
        if found != n {
            return Err(MultigroupError::Length {
                what,
                expected: n,
                found,
            });
        }
    }
    if caliper.is_some_and(|c| !(c.is_finite() && c > 0.0)) {
        return Err(MultigroupError::Caliper);
    }
    let mut by_cluster: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &c) in cluster.iter().enumerate() {
        by_cluster.entry(c).or_default().push(i);
    }
    let mut pairs = Vec::new();
    for members in by_cluster.values() {
        let anchors: Vec<BalanceRecord> = members
            .iter()
            .filter(|&&i| is_focal[i])
            .map(|&i| BalanceRecord::new(ids[i].clone(), dummy_date()))
            .collect();
        let candidates: Vec<BalanceRecord> = members
            .iter()
            .filter(|&&i| !is_focal[i])
            .map(|&i| BalanceRecord::new(ids[i].clone(), dummy_date()))
            .collect();
        if anchors.is_empty() || candidates.is_empty() {
            continue;
        }
        let scores: RapidHashMap<String, f64> = members
            .iter()
            .map(|&i| (ids[i].clone(), score[i]))
            .collect();
        // The width is a multiple of the population standard deviation of the
        // scores in the cluster, as Matching::Match() takes it. A cluster
        // whose scores are all equal has no caliper to apply.
        let width = caliper.and_then(|c| {
            let values: Vec<f64> = members.iter().map(|&i| score[i]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
            DistanceCaliper::new(c * variance.sqrt())
        });
        let outcome = MatchJob::new_standard(&anchors, &candidates, 0)
            .with_nearest()
            .with_ratio(MatchRatio::new(1).expect("a ratio of 1"))
            .with_replacement(true)
            .with_distance_config(DistanceConfig::propensity_score_map(scores, width))
            .run();
        pairs.extend(outcome.pairs);
    }
    Ok(pairs)
}

/// The choices of a vector match.
#[derive(Debug, Clone)]
pub struct VectorMatchSpec {
    /// The focal group, whose effect is estimated.
    pub focal: usize,
    /// The number of k-means clusters.
    pub clusters: usize,
    /// The caliper in standard deviations of the score, 0.25 in the paper;
    /// `None` for no caliper.
    pub caliper: Option<f64>,
    /// Drop the units outside the rectangular common support.
    pub common_support: bool,
    /// Fit the generalised propensity score again on the units that remain.
    pub refit: bool,
    /// The k-means algorithm.
    pub algorithm: KMeansAlgorithm,
    /// The generalised propensity score model.
    pub gps: GpsConfig,
}

impl VectorMatchSpec {
    /// The choices of the paper: a caliper of 0.25, common support and one
    /// refit, with `clusters` clusters.
    #[must_use]
    pub fn new(focal: usize, clusters: usize) -> Self {
        Self {
            focal,
            clusters,
            caliper: Some(0.25),
            common_support: true,
            refit: true,
            algorithm: KMeansAlgorithm::HartiganWong,
            gps: GpsConfig::default(),
        }
    }
}

/// The match of the focal group to one other group.
#[derive(Debug, Clone)]
pub struct GroupMatch {
    /// The group matched to the focal group.
    pub group: usize,
    /// The cluster of each unit of [`VectorMatchOutcome::kept`], in that
    /// order.
    pub cluster: Vec<usize>,
    /// The pairs of the final cohort, focal unit first. The match is with
    /// replacement, so a candidate can be in several pairs.
    pub outcome: MatchOutcome,
}

/// The result of a vector match.
#[derive(Debug, Clone)]
pub struct VectorMatchOutcome {
    /// The score the matching used: the refit one when refitting.
    pub gps: GeneralisedPropensityScore,
    /// The units outside the common support.
    pub discarded: Vec<String>,
    /// The units inside it, in the order of the input.
    pub kept: Vec<String>,
    /// The match to each group other than the focal one, in order of group.
    pub comparisons: Vec<GroupMatch>,
    /// The focal units matched in every comparison.
    pub matched_focal: Vec<String>,
}

/// The first centre is the point with the smallest first coordinate, and each
/// further one the point farthest from the nearest centre so far, the lowest
/// position on a tie.
fn farthest_point_start(points: &[f64], d: usize, k: usize) -> Vec<f64> {
    let n = points.len() / d;
    let row = |i: usize| &points[i * d..(i + 1) * d];
    let mut first = 0;
    for i in 1..n {
        if points[i * d] < points[first * d] {
            first = i;
        }
    }
    let mut chosen = vec![first];
    let mut nearest = vec![f64::INFINITY; n];
    while chosen.len() < k {
        let last = row(chosen[chosen.len() - 1]);
        let mut best = 0;
        let mut best_distance = -1.0;
        for (i, near) in nearest.iter_mut().enumerate() {
            let mut dist = 0.0;
            for (a, b) in row(i).iter().zip(last) {
                dist += (a - b) * (a - b);
            }
            *near = near.min(dist);
            if *near > best_distance {
                best_distance = *near;
                best = i;
            }
        }
        chosen.push(best);
    }
    chosen.iter().flat_map(|&i| row(i).to_vec()).collect()
}

/// The units inside the rectangular common support of the probabilities.
fn inside_support(gps: &GeneralisedPropensityScore, groups: &[usize]) -> Vec<bool> {
    let n = groups.len();
    let k = gps.groups;
    let mut inside = vec![true; n];
    for l in 0..k {
        let mut low = f64::NEG_INFINITY;
        let mut high = f64::INFINITY;
        for g in 0..k {
            let values = (0..n)
                .filter(|&i| groups[i] == g)
                .map(|i| gps.probability(i, l));
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for v in values {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            low = low.max(lo);
            high = high.min(hi);
        }
        for (i, flag) in inside.iter_mut().enumerate() {
            let p = gps.probability(i, l);
            if !(p > low && p < high) {
                *flag = false;
            }
        }
    }
    inside
}

/// Matches the focal group to every other group by vector matching.
///
/// # Errors
///
/// See [`MultigroupError`].
pub fn vector_match<R: CovariateRecord>(
    records: &[R],
    groups: &[usize],
    n_groups: usize,
    spec: &VectorMatchSpec,
) -> Result<VectorMatchOutcome, MultigroupError> {
    check_groups(groups, n_groups, records.len())?;
    if spec.focal >= n_groups {
        return Err(MultigroupError::Groups { groups: n_groups });
    }
    if spec.clusters == 0 {
        return Err(MultigroupError::Clusters);
    }
    if spec.caliper.is_some_and(|c| !(c.is_finite() && c > 0.0)) {
        return Err(MultigroupError::Caliper);
    }
    let mut gps = generalised_propensity_score(records, groups, n_groups, &spec.gps)?;
    let mut keep: Vec<usize> = (0..records.len()).collect();
    if spec.common_support {
        let inside = inside_support(&gps, groups);
        keep = (0..records.len()).filter(|&i| inside[i]).collect();
        for group in 0..n_groups {
            if !keep.iter().any(|&i| groups[i] == group) {
                return Err(MultigroupError::EmptyGroup { group });
            }
        }
        if spec.refit && keep.len() < records.len() {
            let kept_records: Vec<&R> = keep.iter().map(|&i| &records[i]).collect();
            let kept_groups: Vec<usize> = keep.iter().map(|&i| groups[i]).collect();
            gps = generalised_propensity_score(&kept_records, &kept_groups, n_groups, &spec.gps)?;
        } else {
            // The units that remain keep the scores of the one fit.
            let mut probabilities = Vec::with_capacity(keep.len() * n_groups);
            for &i in &keep {
                probabilities
                    .extend_from_slice(&gps.probabilities[i * n_groups..(i + 1) * n_groups]);
            }
            gps.ids = keep.iter().map(|&i| gps.ids[i].clone()).collect();
            gps.probabilities = probabilities;
        }
    }
    let kept_ids: Vec<String> = keep.iter().map(|&i| records[i].id().to_owned()).collect();
    let mut kept = vec![false; records.len()];
    for &i in &keep {
        kept[i] = true;
    }
    let discarded: Vec<String> = (0..records.len())
        .filter(|&i| !kept[i])
        .map(|i| records[i].id().to_owned())
        .collect();
    let kept_groups: Vec<usize> = keep.iter().map(|&i| groups[i]).collect();
    let n = keep.len();
    let logit = |row: usize, group: usize| gps.logit(row, group).ok_or(MultigroupError::Separated);

    let mut comparisons = Vec::new();
    for other in (0..n_groups).filter(|&g| g != spec.focal) {
        let dims: Vec<usize> = (0..n_groups)
            .filter(|&l| l != spec.focal && l != other)
            .collect();
        let cluster = if dims.is_empty() {
            vec![0; n]
        } else {
            let mut points = Vec::with_capacity(n * dims.len());
            for row in 0..n {
                for &l in &dims {
                    points.push(logit(row, l)?);
                }
            }
            let start = farthest_point_start(&points, dims.len(), spec.clusters.min(n));
            kmeans(&points, dims.len(), &start, spec.algorithm, 1000)?.cluster
        };
        let in_pair: Vec<usize> = (0..n)
            .filter(|&i| kept_groups[i] == spec.focal || kept_groups[i] == other)
            .collect();
        let mut scores = Vec::with_capacity(in_pair.len());
        for &i in &in_pair {
            scores.push(logit(i, spec.focal)?);
        }
        let pairs = match_within_clusters(
            &in_pair
                .iter()
                .map(|&i| kept_ids[i].clone())
                .collect::<Vec<_>>(),
            &scores,
            &in_pair
                .iter()
                .map(|&i| kept_groups[i] == spec.focal)
                .collect::<Vec<_>>(),
            &in_pair.iter().map(|&i| cluster[i]).collect::<Vec<_>>(),
            spec.caliper,
        )?;
        comparisons.push(GroupMatch {
            group: other,
            cluster,
            outcome: outcome_of(pairs, 0),
        });
    }

    // The focal units matched against every other group.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for comparison in &comparisons {
        let matched: HashSet<&str> = comparison
            .outcome
            .pairs
            .iter()
            .map(MatchedPair::anchor_id)
            .collect();
        for id in matched {
            *counts.entry(id).or_default() += 1;
        }
    }
    let matched_focal: Vec<String> = (0..n)
        .filter(|&i| kept_groups[i] == spec.focal)
        .map(|i| kept_ids[i].clone())
        .filter(|id| counts.get(id.as_str()) == Some(&comparisons.len()))
        .collect();
    let final_set: HashSet<&str> = matched_focal.iter().map(String::as_str).collect();
    let focal_total = kept_groups.iter().filter(|&&g| g == spec.focal).count();
    for comparison in &mut comparisons {
        let pairs: Vec<MatchedPair> = comparison
            .outcome
            .pairs
            .iter()
            .filter(|p| final_set.contains(p.anchor_id()))
            .cloned()
            .collect();
        comparison.outcome = outcome_of(pairs, focal_total);
    }
    Ok(VectorMatchOutcome {
        gps,
        discarded,
        kept: kept_ids,
        comparisons,
        matched_focal,
    })
}

/// An outcome of with-replacement pairs of `focal_total` anchors, 1:1.
fn outcome_of(pairs: Vec<MatchedPair>, focal_total: usize) -> MatchOutcome {
    let matched: HashSet<&str> = pairs.iter().map(MatchedPair::anchor_id).collect();
    let matched_cases = matched.len();
    let diagnostics = MatchDiagnostics {
        realized_estimand: Estimand::Att,
        ..MatchDiagnostics::default()
    };
    MatchOutcome {
        unmatched_cases: focal_total.saturating_sub(matched_cases),
        used_controls: 0,
        matched_cases,
        avg_controls_per_case: if matched_cases == 0 { 0.0 } else { 1.0 },
        diagnostics,
        pairs,
    }
}

/// The balance of the focal group against each other group on the final
/// cohort.
///
/// For each comparison, the report is that of
/// [`balance_report_on_weights`] for the focal units and the units of the
/// other group inside the common support, with the matching weights of the
/// pairs of the final cohort. A unit outside the cohort has weight 0.
///
/// # Errors
///
/// [`BalanceInputError`] for a weight that is not valid.
pub fn balance_by_group<R: CovariateRecord>(
    records: &[R],
    groups: &[usize],
    focal: usize,
    outcome: &VectorMatchOutcome,
    options: &BalanceReportOptions,
) -> Result<Vec<(usize, BalanceReport)>, BalanceInputError> {
    let kept: HashSet<&str> = outcome.kept.iter().map(String::as_str).collect();
    let focal_records: Vec<&R> = records
        .iter()
        .zip(groups)
        .filter(|(r, g)| **g == focal && kept.contains(r.id()))
        .map(|(r, _)| r)
        .collect();
    outcome
        .comparisons
        .iter()
        .map(|comparison| {
            let others: Vec<&R> = records
                .iter()
                .zip(groups)
                .filter(|(r, g)| **g == comparison.group && kept.contains(r.id()))
                .map(|(r, _)| r)
                .collect();
            let weights = BalanceWeights::from_outcome(&comparison.outcome);
            balance_report_on_weights(&focal_records, &others, &weights, options)
                .map(|report| (comparison.group, report))
        })
        .collect()
}
