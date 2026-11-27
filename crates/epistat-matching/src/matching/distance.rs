//! Distances between an anchor and a candidate, and the tables they read.
//!
//! A distance is a number for a pair, smaller for a closer match, or `None`
//! for a pair that has none. [`DistanceMetric`] is the trait the matchers use.
//! [`DistanceConfig`] names a distance as plain data that does not depend on
//! the record type, so that a preparation step can build it once and any
//! matcher can read it.
//!
//! # Distances
//!
//! The birth-date distance is the absolute difference of the two birth dates
//! in days. The propensity score distance is `|s_a - s_c|` of the two scores,
//! and a pair with a score that is missing or not finite has none. The
//! Euclidean distance of two vectors `x` and `y` of `p` values is
//!
//! ```text
//! d(x, y) = sqrt( sum_j (x_j - y_j)^2 )
//! ```
//!
//! with the squares added in column order by a plain multiply and add. The
//! Mahalanobis distance with the inverse covariance matrix `M` is
//!
//! ```text
//! d(x, y) = sqrt( (x - y)' M (x - y) )
//! ```
//!
//! Rubin \[1\] used it for matching. When `M` is positive definite with lower
//! Cholesky factor `L` (`M = L L'`), the form equals `|L' (x - y)|^2`, so every
//! vector is transformed once and a pair costs a pass over two transformed
//! vectors. A matrix that is not positive definite keeps the plain form, summed
//! in a fixed order, and a form that is negative beyond rounding gives the
//! pair no distance. The constructors refuse a matrix with a negative
//! eigenvalue, since its form is not a distance.
//!
//! A [`PairDistanceMatrix`] holds distances that the caller computed, one for
//! each anchor and candidate. An infinite entry forbids the pair.
//!
//! Every sum runs in a fixed order and without fused multiply-add, so a
//! distance has the same bits on any CPU and any number of threads.
//!
//! # Tables
//!
//! A [`ScoreTable`] holds one score for each unit and a [`VectorTable`] one
//! row of values for each unit, both in the order the units were listed in.
//! A matcher given the same units in the same order reads them by position. A
//! matcher given other units finds each id once through an index that is built
//! on first use.
//!
//! # References
//!
//! 1. Rubin DB. Bias reduction using Mahalanobis-metric matching. Biometrics
//!    1980;36:293-298.
//! 2. Stuart EA. Matching methods for causal inference: a review and a look
//!    forward. Statistical Science 2010;25:1-21.

use super::records::MatchingRecord;
use crate::Error;
use crate::types::DistanceCaliper;
use epistat_linalg::{Attempt, Factor, Mat, Method};
use ordered_float::OrderedFloat;
use rapidhash::RapidHashMap;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

/// Distance metric used to compare an anchor and a candidate.
///
/// Returning `None` indicates the pair is ineligible for this metric.
pub trait DistanceMetric<R: MatchingRecord> {
    /// Stable metric channel name used in diagnostics.
    fn channel(&self) -> &'static str;

    /// Compute pair distance. Smaller values indicate closer matches.
    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64>;

    /// Optional candidate pre-filtering by caliper using an internal index.
    /// Returns `None` if this metric does not support indexed queries.
    #[must_use]
    fn candidate_indices(&self, _anchor: &R, _caliper: f64) -> Option<Vec<usize>> {
        None
    }
}

/// Reusable named distance channel with optional caliper.
pub struct DistanceChannel<'a, R: MatchingRecord, D: DistanceMetric<R>> {
    metric: &'a D,
    caliper: Option<DistanceCaliper>,
    reason: &'static str,
    marker: PhantomData<R>,
}

impl<R: MatchingRecord, D: DistanceMetric<R>> Copy for DistanceChannel<'_, R, D> {}

impl<R: MatchingRecord, D: DistanceMetric<R>> Clone for DistanceChannel<'_, R, D> {
    fn clone(&self) -> Self {
        *self
    }
}

/// One score for each unit, held by the unit's position.
///
/// The units are listed once, in the order the scores were made in: for a
/// propensity score, the anchors and then the candidates. A matcher that is
/// given the same units in the same order reads the scores by position and
/// never looks an id up. A matcher given other units, or the same ones in
/// another order, finds each id once. Ids and scores are shared between
/// clones.
#[derive(Debug, Clone)]
pub struct ScoreTable {
    ids: Arc<[String]>,
    values: Arc<[f64]>,
    index: Arc<OnceLock<RapidHashMap<String, usize>>>,
}

impl ScoreTable {
    /// The scores `values` of the units `ids`, aligned by position.
    ///
    /// # Panics
    ///
    /// Panics when the two have different lengths.
    #[must_use]
    pub fn new(ids: Arc<[String]>, values: Arc<[f64]>) -> Self {
        assert_eq!(ids.len(), values.len(), "one score for each id");
        Self {
            ids,
            values,
            index: Arc::new(OnceLock::new()),
        }
    }

    /// As [`Self::new`], sharing the lookup of ids that `index` holds.
    pub(crate) fn with_shared_index(
        ids: Arc<[String]>,
        values: Arc<[f64]>,
        index: Arc<OnceLock<RapidHashMap<String, usize>>>,
    ) -> Self {
        Self { ids, values, index }
    }

    /// The scores of a map keyed by unit id, in the order of the map.
    #[must_use]
    pub fn from_map(scores: &RapidHashMap<String, f64>) -> Self {
        let mut ids = Vec::with_capacity(scores.len());
        let mut values = Vec::with_capacity(scores.len());
        for (id, score) in scores {
            ids.push(id.clone());
            values.push(*score);
        }
        Self::new(ids.into(), values.into())
    }

    /// The scores as a map keyed by unit id.
    #[must_use]
    pub fn to_map(&self) -> RapidHashMap<String, f64> {
        self.ids
            .iter()
            .cloned()
            .zip(self.values.iter().copied())
            .collect()
    }

    /// The unit ids, in order.
    #[must_use]
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// The scores, in the order of [`Self::ids`].
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The number of units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn index(&self) -> &RapidHashMap<String, usize> {
        self.index.get_or_init(|| {
            self.ids
                .iter()
                .enumerate()
                .map(|(position, id)| (id.clone(), position))
                .collect()
        })
    }

    /// The position of the unit `id`, `None` when it has no score. The
    /// first call builds a lookup of every id.
    #[must_use]
    pub fn position(&self, id: &str) -> Option<usize> {
        self.index().get(id).copied()
    }

    /// The score of the unit `id`, `None` when it has none. The first call
    /// builds a lookup of every id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&f64> {
        self.position(id).map(|position| &self.values[position])
    }

    /// The scores of `units`, by their position in the slice. A unit without
    /// a score, or whose score is not finite, has `NaN`.
    ///
    /// The scores are borrowed when `units` are exactly the units of the
    /// table, in order, and looked up by id otherwise.
    pub(crate) fn of_units<R: MatchingRecord>(
        &self,
        first: &[R],
        second: &[R],
    ) -> (std::borrow::Cow<'_, [f64]>, std::borrow::Cow<'_, [f64]>) {
        let split = first.len();
        let aligned = self.len() == first.len() + second.len()
            && first
                .iter()
                .chain(second)
                .zip(self.ids.iter())
                .all(|(record, id)| record.id() == id.as_str());
        if aligned {
            return (
                std::borrow::Cow::Borrowed(&self.values[..split]),
                std::borrow::Cow::Borrowed(&self.values[split..]),
            );
        }
        let read = |units: &[R]| {
            units
                .iter()
                .map(|record| self.get(record.id()).copied().unwrap_or(f64::NAN))
                .collect::<Vec<f64>>()
        };
        (
            std::borrow::Cow::Owned(read(first)),
            std::borrow::Cow::Owned(read(second)),
        )
    }
}

/// The vectors of a distance by unit position: one row of `dimension` values
/// for each unit id, in one array. Ids and values are shared between clones, so
/// a preparation and the distance made from it hold the vectors once. An id is
/// looked up through an index built on the first lookup; a matcher that reads
/// by position builds none.
#[derive(Debug, Clone)]
pub struct VectorTable {
    ids: Arc<[String]>,
    dimension: usize,
    values: Arc<[f64]>,
    index: Arc<OnceLock<RapidHashMap<String, usize>>>,
}

impl VectorTable {
    /// The vectors `values` of the units `ids`, row by row.
    ///
    /// # Errors
    ///
    /// [`Error::ZeroDimension`] when `dimension` is zero, and
    /// [`Error::VectorLength`] when `values` does not hold
    /// `dimension` values for each id.
    pub fn new(ids: Arc<[String]>, dimension: usize, values: Arc<[f64]>) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::ZeroDimension);
        }
        if values.len() != ids.len().saturating_mul(dimension) {
            return Err(Error::VectorLength);
        }
        Ok(Self {
            ids,
            dimension,
            values,
            index: Arc::new(OnceLock::new()),
        })
    }

    /// The vectors of a map keyed by unit id, in the order of the ids. A vector
    /// that is not `dimension` long is left out, as it would have no distance.
    #[must_use]
    pub fn from_map(vectors: &RapidHashMap<String, Vec<f64>>, dimension: usize) -> Self {
        let mut ids: Vec<&String> = vectors
            .iter()
            .filter(|(_, vector)| vector.len() == dimension)
            .map(|(id, _)| id)
            .collect();
        ids.sort_unstable();
        let mut values = Vec::with_capacity(ids.len() * dimension);
        for id in &ids {
            values.extend_from_slice(&vectors[*id]);
        }
        Self {
            ids: ids.into_iter().cloned().collect(),
            dimension,
            values: values.into(),
            index: Arc::new(OnceLock::new()),
        }
    }

    /// The vectors as a map keyed by unit id.
    #[must_use]
    pub fn to_map(&self) -> RapidHashMap<String, Vec<f64>> {
        (0..self.len())
            .map(|position| (self.ids[position].clone(), self.row(position).to_vec()))
            .collect()
    }

    /// The unit ids, in order.
    #[must_use]
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// The length of a vector.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// The values of every vector, row by row, in the order of [`Self::ids`].
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The number of units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The vector of the unit at `position`.
    ///
    /// # Panics
    ///
    /// When `position` is not below [`Self::len`].
    #[must_use]
    pub fn row(&self, position: usize) -> &[f64] {
        &self.values[position * self.dimension..(position + 1) * self.dimension]
    }

    /// The position of the unit `id`, `None` when it has no vector. The first
    /// call builds a lookup of every id. Of an id that occurs twice, the
    /// later position is found.
    #[must_use]
    pub fn position(&self, id: &str) -> Option<usize> {
        self.index
            .get_or_init(|| {
                self.ids
                    .iter()
                    .enumerate()
                    .map(|(position, id)| (id.clone(), position))
                    .collect()
            })
            .get(id)
            .copied()
    }

    /// The vector of the unit `id`, `None` when it has none.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&[f64]> {
        self.position(id).map(|position| self.row(position))
    }
}

impl From<RapidHashMap<String, f64>> for ScoreTable {
    fn from(scores: RapidHashMap<String, f64>) -> Self {
        Self::from_map(&scores)
    }
}

/// Named and reusable distance setup independent of concrete record type.
#[derive(Debug, Clone)]
pub enum DistanceConfig {
    /// The birth-date distance in days.
    Date {
        /// The largest allowed distance, if any.
        caliper: Option<DistanceCaliper>,
        /// The reason recorded when the caliper refuses a pair.
        reason: &'static str,
    },
    /// The absolute difference of propensity scores.
    PropensityScoreMap {
        /// The score of each unit.
        scores: ScoreTable,
        /// The largest allowed distance, if any.
        caliper: Option<DistanceCaliper>,
        /// The reason recorded when the caliper refuses a pair.
        reason: &'static str,
    },
    /// The Mahalanobis distance of covariate vectors.
    MahalanobisMap {
        /// The covariate vector of each unit.
        vectors: VectorTable,
        /// The inverse covariance matrix, row by row.
        inverse_covariance: Vec<f64>,
        /// The largest allowed distance, if any.
        caliper: Option<DistanceCaliper>,
        /// The reason recorded when the caliper refuses a pair.
        reason: &'static str,
    },
    /// The Euclidean distance of covariate vectors.
    EuclideanMap {
        /// The covariate vector of each unit.
        vectors: VectorTable,
        /// The largest allowed distance, if any.
        caliper: Option<DistanceCaliper>,
        /// The reason recorded when the caliper refuses a pair.
        reason: &'static str,
    },
    /// Distances between every anchor and every candidate, given by the
    /// caller.
    Matrix {
        /// The distances.
        matrix: PairDistanceMatrix,
        /// The largest allowed distance, if any.
        caliper: Option<DistanceCaliper>,
        /// The reason recorded when the caliper refuses a pair.
        reason: &'static str,
    },
}

impl DistanceConfig {
    /// A birth-date distance with an optional caliper in days.
    #[must_use]
    pub const fn date(caliper: Option<DistanceCaliper>) -> Self {
        Self::Date {
            caliper,
            reason: "date_caliper",
        }
    }

    /// A propensity score distance on scores keyed by unit id.
    #[must_use]
    pub fn propensity_score_map(
        scores: RapidHashMap<String, f64>,
        caliper: Option<DistanceCaliper>,
    ) -> Self {
        Self::propensity_score_table(ScoreTable::from(scores), caliper)
    }

    /// A score distance on the scores of `scores`, which a matcher reads by
    /// position when it is given the units in the order of the table.
    #[must_use]
    pub const fn propensity_score_table(
        scores: ScoreTable,
        caliper: Option<DistanceCaliper>,
    ) -> Self {
        Self::PropensityScoreMap {
            scores,
            caliper,
            reason: "propensity_score_caliper",
        }
    }

    /// Build Mahalanobis config from id-indexed vectors and inverse covariance.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when `dimension` is zero, the matrix shape
    /// is invalid or the matrix is not positive semi-definite.
    #[expect(clippy::needless_pass_by_value, reason = "the signature is public")]
    pub fn mahalanobis_map(
        vectors: RapidHashMap<String, Vec<f64>>,
        inverse_covariance: Vec<f64>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
    ) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::ZeroDimension);
        }
        Self::mahalanobis_table(
            VectorTable::from_map(&vectors, dimension),
            inverse_covariance,
            caliper,
        )
    }

    /// Build Mahalanobis config from vectors by position and inverse
    /// covariance.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when the matrix is not `dimension` squared
    /// long for the dimension of the vectors or is not positive semi-definite.
    pub fn mahalanobis_table(
        vectors: VectorTable,
        inverse_covariance: Vec<f64>,
        caliper: Option<DistanceCaliper>,
    ) -> Result<Self, Error> {
        let dimension = vectors.dimension();
        if dimension == 0 {
            return Err(Error::ZeroDimension);
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return Err(Error::InverseCovarianceLength);
        }
        check_positive_semi_definite(&inverse_covariance, dimension)?;

        Ok(Self::MahalanobisMap {
            vectors,
            inverse_covariance,
            caliper,
            reason: "mahalanobis_caliper",
        })
    }

    /// Build a Euclidean config from id-indexed vectors, already scaled as the
    /// caller wants them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ZeroDimension`] when `dimension` is zero.
    #[expect(clippy::needless_pass_by_value, reason = "the signature is public")]
    pub fn euclidean_map(
        vectors: RapidHashMap<String, Vec<f64>>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
    ) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::ZeroDimension);
        }
        Self::euclidean_table(VectorTable::from_map(&vectors, dimension), caliper)
    }

    /// Build a Euclidean config from vectors by position, already scaled as
    /// the caller wants them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ZeroDimension`] when the vectors have no
    /// values.
    pub fn euclidean_table(
        vectors: VectorTable,
        caliper: Option<DistanceCaliper>,
    ) -> Result<Self, Error> {
        if vectors.dimension() == 0 {
            return Err(Error::ZeroDimension);
        }
        Ok(Self::EuclideanMap {
            vectors,
            caliper,
            reason: "euclidean_caliper",
        })
    }

    /// A distance given as a table of anchor-by-candidate values, as a matrix
    /// given as `distance` to `matchit()`.
    #[must_use]
    pub const fn matrix(matrix: PairDistanceMatrix, caliper: Option<DistanceCaliper>) -> Self {
        Self::Matrix {
            matrix,
            caliper,
            reason: "distance_matrix_caliper",
        }
    }

    /// The same configuration with another reason recorded for the caliper.
    #[must_use]
    pub const fn with_reason(mut self, reason: &'static str) -> Self {
        match &mut self {
            Self::Date { reason: slot, .. }
            | Self::PropensityScoreMap { reason: slot, .. }
            | Self::MahalanobisMap { reason: slot, .. }
            | Self::EuclideanMap { reason: slot, .. }
            | Self::Matrix { reason: slot, .. } => *slot = reason,
        }
        self
    }

    /// The table of a matrix configuration.
    #[must_use]
    pub const fn pair_matrix(&self) -> Option<&PairDistanceMatrix> {
        match self {
            Self::Matrix { matrix, .. } => Some(matrix),
            Self::Date { .. }
            | Self::PropensityScoreMap { .. }
            | Self::MahalanobisMap { .. }
            | Self::EuclideanMap { .. } => None,
        }
    }

    /// The distance of the anchor in row `anchor` and the candidate in column
    /// `candidate` of a matrix configuration, in the order of the matrix.
    ///
    /// `None` when this is not a matrix configuration, when a position is out
    /// of range, or when the pair is forbidden by an infinite entry.
    /// [`PairDistanceMatrix::positions_of`] maps the units of a job to these
    /// positions.
    #[must_use]
    pub fn pair_distance_by_position(&self, anchor: usize, candidate: usize) -> Option<f64> {
        self.pair_matrix()?.at(anchor, candidate)
    }

    /// The largest allowed distance, if any.
    #[must_use]
    pub const fn caliper(&self) -> Option<f64> {
        match self {
            Self::Date { caliper, .. }
            | Self::PropensityScoreMap { caliper, .. }
            | Self::MahalanobisMap { caliper, .. }
            | Self::EuclideanMap { caliper, .. }
            | Self::Matrix { caliper, .. } => match caliper {
                Some(caliper) => Some(caliper.get()),
                None => None,
            },
        }
    }

    /// The largest allowed distance as its validated type, if any.
    #[must_use]
    pub const fn typed_caliper(&self) -> Option<DistanceCaliper> {
        match self {
            Self::Date { caliper, .. }
            | Self::PropensityScoreMap { caliper, .. }
            | Self::MahalanobisMap { caliper, .. }
            | Self::EuclideanMap { caliper, .. }
            | Self::Matrix { caliper, .. } => *caliper,
        }
    }

    /// The reason recorded when the caliper refuses a pair.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Date { reason, .. }
            | Self::PropensityScoreMap { reason, .. }
            | Self::MahalanobisMap { reason, .. }
            | Self::EuclideanMap { reason, .. }
            | Self::Matrix { reason, .. } => reason,
        }
    }
}

impl DistanceConfig {
    /// The metric this configuration describes, borrowing its maps.
    pub(crate) fn metric(&self) -> ConfiguredMetric<'_> {
        match self {
            Self::Date { .. } => ConfiguredMetric::Date(DateDistance),
            Self::PropensityScoreMap { scores, .. } => {
                ConfiguredMetric::PropensityScore(IdMapPropensityScoreDistance::new(scores))
            }
            Self::MahalanobisMap {
                vectors,
                inverse_covariance,
                ..
            } => ConfiguredMetric::Mahalanobis(IdMapMahalanobisDistance::new(
                vectors,
                inverse_covariance,
            )),
            Self::EuclideanMap { vectors, .. } => {
                ConfiguredMetric::Euclidean(IdMapEuclideanDistance::new(vectors))
            }
            Self::Matrix { matrix, .. } => ConfiguredMetric::Matrix(MatrixDistance::new(matrix)),
        }
    }

    /// The scores common-support trimming needs, which only a propensity
    /// configuration has.
    pub(crate) const fn propensity_scores(&self) -> Option<&ScoreTable> {
        match self {
            Self::PropensityScoreMap { scores, .. } => Some(scores),
            Self::Date { .. }
            | Self::MahalanobisMap { .. }
            | Self::EuclideanMap { .. }
            | Self::Matrix { .. } => None,
        }
    }
}

/// A distance that is the absolute difference of one number per record.
///
/// Records are ranked on such a distance by sorting them on that number, so a
/// nearest neighbour is found without comparing every pair.
#[derive(Clone, Copy)]
pub(crate) enum ScalarLookup<'a> {
    /// The birth date as a day number.
    BirthDate,
    /// A score per record id.
    Scores(&'a ScoreTable),
}

impl ScalarLookup<'_> {
    /// The record's number, or `None` when it has none or it is not finite.
    pub(crate) fn value<R: MatchingRecord>(self, record: &R) -> Option<f64> {
        use chrono::Datelike;
        match self {
            Self::BirthDate => Some(f64::from(record.birth_date().num_days_from_ce())),
            Self::Scores(scores) => scores
                .get(record.id())
                .copied()
                .filter(|score| score.is_finite()),
        }
    }
}

impl<'a> ScalarLookup<'a> {
    /// The numbers of two slices of records, by position in each. A record
    /// without a number, or with one that is not finite, has `NaN`.
    pub(crate) fn of_units<R: MatchingRecord>(
        self,
        first: &[R],
        second: &[R],
    ) -> (std::borrow::Cow<'a, [f64]>, std::borrow::Cow<'a, [f64]>) {
        use chrono::Datelike;
        match self {
            Self::BirthDate => {
                let days = |units: &[R]| {
                    units
                        .iter()
                        .map(|record| f64::from(record.birth_date().num_days_from_ce()))
                        .collect::<Vec<f64>>()
                };
                (
                    std::borrow::Cow::Owned(days(first)),
                    std::borrow::Cow::Owned(days(second)),
                )
            }
            Self::Scores(scores) => scores.of_units(first, second),
        }
    }
}

impl DistanceConfig {
    /// The one-number-per-record form of this distance, when it has one.
    pub(crate) const fn scalar(&self) -> Option<ScalarLookup<'_>> {
        match self {
            Self::Date { .. } => Some(ScalarLookup::BirthDate),
            Self::PropensityScoreMap { scores, .. } => Some(ScalarLookup::Scores(scores)),
            Self::MahalanobisMap { .. } | Self::EuclideanMap { .. } | Self::Matrix { .. } => None,
        }
    }
}

/// One of the metrics a [`DistanceConfig`] can name, so the matchers take a
/// single metric type whichever variant the caller configured.
pub(crate) enum ConfiguredMetric<'a> {
    Date(DateDistance),
    PropensityScore(IdMapPropensityScoreDistance<'a>),
    Mahalanobis(IdMapMahalanobisDistance<'a>),
    Euclidean(IdMapEuclideanDistance<'a>),
    Matrix(MatrixDistance<'a>),
}

impl<R: MatchingRecord> DistanceMetric<R> for ConfiguredMetric<'_> {
    fn channel(&self) -> &'static str {
        match self {
            Self::Date(metric) => DistanceMetric::<R>::channel(metric),
            Self::PropensityScore(metric) => DistanceMetric::<R>::channel(metric),
            Self::Mahalanobis(metric) => DistanceMetric::<R>::channel(metric),
            Self::Euclidean(metric) => DistanceMetric::<R>::channel(metric),
            Self::Matrix(metric) => DistanceMetric::<R>::channel(metric),
        }
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        match self {
            Self::Date(metric) => metric.distance(anchor, candidate),
            Self::PropensityScore(metric) => metric.distance(anchor, candidate),
            Self::Mahalanobis(metric) => metric.distance(anchor, candidate),
            Self::Euclidean(metric) => metric.distance(anchor, candidate),
            Self::Matrix(metric) => metric.distance(anchor, candidate),
        }
    }

    fn candidate_indices(&self, anchor: &R, caliper: f64) -> Option<Vec<usize>> {
        match self {
            Self::Date(metric) => metric.candidate_indices(anchor, caliper),
            Self::PropensityScore(metric) => metric.candidate_indices(anchor, caliper),
            Self::Mahalanobis(metric) => metric.candidate_indices(anchor, caliper),
            Self::Euclidean(metric) => metric.candidate_indices(anchor, caliper),
            Self::Matrix(metric) => metric.candidate_indices(anchor, caliper),
        }
    }
}

/// Distances between every anchor and every candidate, as the matrix a caller
/// gives as `distance` to `matchit()`.
///
/// Rows are anchors and columns are candidates, named by their ids. An
/// infinite value forbids the pair, as in `matchit()`. A pair whose anchor or
/// candidate is not in the table has no distance and is never matched.
#[derive(Debug, Clone)]
pub struct PairDistanceMatrix {
    case_ids: Vec<String>,
    control_ids: Vec<String>,
    values: Vec<f64>,
    case_position: RapidHashMap<String, usize>,
    control_position: RapidHashMap<String, usize>,
}

impl PairDistanceMatrix {
    /// Build the table from row-major values, one row per anchor.
    ///
    /// # Errors
    ///
    /// [`Error`] when a group is empty, the values do not
    /// fill the table, an id repeats, a value is not a number, or a value is negative.
    pub fn new(
        case_ids: Vec<String>,
        control_ids: Vec<String>,
        values: Vec<f64>,
    ) -> Result<Self, Error> {
        if case_ids.is_empty() || control_ids.is_empty() {
            return Err(Error::empty("distance matrix ids"));
        }
        let expected = case_ids.len().saturating_mul(control_ids.len());
        if values.len() != expected {
            return Err(Error::length(
                "distance matrix values",
                expected,
                values.len(),
            ));
        }
        if let Some(at) = values.iter().position(|value| value.is_nan()) {
            return Err(Error::DistanceNotANumber {
                case: at / control_ids.len(),
                control: at % control_ids.len(),
            });
        }
        if let Some(at) = values.iter().position(|value| *value < 0.0) {
            return Err(Error::NegativeDistance {
                case: at / control_ids.len(),
                control: at % control_ids.len(),
            });
        }
        let positions = |ids: &[String]| -> Result<RapidHashMap<String, usize>, _> {
            let mut map = RapidHashMap::default();
            for (position, id) in ids.iter().enumerate() {
                if map.insert(id.clone(), position).is_some() {
                    return Err(Error::DuplicateId { id: id.clone() });
                }
            }
            Ok(map)
        };
        Ok(Self {
            case_position: positions(&case_ids)?,
            control_position: positions(&control_ids)?,
            case_ids,
            control_ids,
            values,
        })
    }

    /// Ids of the rows.
    #[must_use]
    pub fn case_ids(&self) -> &[String] {
        &self.case_ids
    }

    /// Ids of the columns.
    #[must_use]
    pub fn control_ids(&self) -> &[String] {
        &self.control_ids
    }

    /// Row of an anchor.
    #[must_use]
    pub fn case_position(&self, id: &str) -> Option<usize> {
        self.case_position.get(id).copied()
    }

    /// Column of a candidate.
    #[must_use]
    pub fn control_position(&self, id: &str) -> Option<usize> {
        self.control_position.get(id).copied()
    }

    /// The value at a row and column, or `None` outside the table or where the
    /// pair is forbidden by an infinite value.
    #[must_use]
    pub fn at(&self, anchor: usize, candidate: usize) -> Option<f64> {
        if anchor >= self.case_ids.len() || candidate >= self.control_ids.len() {
            return None;
        }
        let value = self.values[anchor * self.control_ids.len() + candidate];
        value.is_finite().then_some(value)
    }

    /// The row of each anchor and the column of each candidate of a job, by
    /// id. A unit not in the table has no position.
    #[must_use]
    pub fn positions_of<R: MatchingRecord>(
        &self,
        anchors: &[R],
        candidates: &[R],
    ) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
        (
            anchors
                .iter()
                .map(|record| self.case_position(record.id()))
                .collect(),
            candidates
                .iter()
                .map(|record| self.control_position(record.id()))
                .collect(),
        )
    }
}

/// The distance of a [`PairDistanceMatrix`], looked up by the ids of the two
/// records.
#[derive(Debug, Clone, Copy)]
pub struct MatrixDistance<'a> {
    matrix: &'a PairDistanceMatrix,
}

impl<'a> MatrixDistance<'a> {
    /// The distance of `matrix`.
    #[must_use]
    pub const fn new(matrix: &'a PairDistanceMatrix) -> Self {
        Self { matrix }
    }
}

impl<R: MatchingRecord> DistanceMetric<R> for MatrixDistance<'_> {
    fn channel(&self) -> &'static str {
        "distance_matrix_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        self.matrix.at(
            self.matrix.case_position(anchor.id())?,
            self.matrix.control_position(candidate.id())?,
        )
    }
}

/// Propensity score distance keyed by [`MatchingRecord::id`].
#[derive(Debug, Clone)]
pub struct IdMapPropensityScoreDistance<'a> {
    scores: &'a ScoreTable,
    index: Option<BTreeMap<OrderedFloat<f64>, Vec<usize>>>,
}

impl<'a> IdMapPropensityScoreDistance<'a> {
    /// The score distance on `scores`, with no index built.
    #[must_use]
    pub const fn new(scores: &'a ScoreTable) -> Self {
        Self {
            scores,
            index: None,
        }
    }

    /// Build a sorted propensity index for faster caliper queries.
    #[must_use]
    pub fn with_index<R: MatchingRecord>(mut self, controls: &[R]) -> Self {
        let mut index: BTreeMap<OrderedFloat<f64>, Vec<usize>> = BTreeMap::new();
        for (idx, record) in controls.iter().enumerate() {
            if let Some(score) = self
                .scores
                .get(record.id())
                .copied()
                .filter(|s| s.is_finite())
            {
                index.entry(OrderedFloat(score)).or_default().push(idx);
            }
        }
        self.index = Some(index);
        self
    }

    /// Query indices within a propensity score caliper.
    #[must_use]
    pub fn query_caliper(&self, center: f64, caliper: f64) -> Option<Vec<usize>> {
        let index = self.index.as_ref()?;
        let lower = OrderedFloat(center - caliper);
        let upper = OrderedFloat(center + caliper);
        Some(
            index
                .range(lower..=upper)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect(),
        )
    }

    /// The scores the distance reads.
    #[must_use]
    pub const fn scores(&self) -> &'a ScoreTable {
        self.scores
    }
}

impl<R: MatchingRecord> DistanceMetric<R> for IdMapPropensityScoreDistance<'_> {
    fn channel(&self) -> &'static str {
        "propensity_score_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = *self.scores.get(anchor.id())?;
        let right = *self.scores.get(candidate.id())?;
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        Some((left - right).abs())
    }

    fn candidate_indices(&self, anchor: &R, caliper: f64) -> Option<Vec<usize>> {
        let center = *self.scores.get(anchor.id())?;
        if !center.is_finite() {
            return Some(Vec::new());
        }
        self.query_caliper(center, caliper)
    }
}

/// The quadratic form `d' M d` of a symmetric matrix `M`, evaluated without a
/// matrix product per pair.
///
/// When `M` is positive definite its lower Cholesky factor `L` gives
/// `d' M d = |L' d|^2`, so a vector can be transformed once and distances are
/// squared Euclidean distances of the transformed vectors. A matrix that is
/// not positive definite keeps the plain form, summed in a fixed order, and a
/// pair whose form is negative beyond rounding has no distance. The public
/// constructors refuse a matrix with a negative eigenvalue, since its form is
/// not a distance.
#[derive(Debug, Clone)]
struct QuadraticForm {
    dimension: usize,
    /// Row-major `L`, when `M` is positive definite.
    lower: Option<Vec<f64>>,
    /// Row-major `M`, when it is not.
    matrix: Vec<f64>,
}

impl QuadraticForm {
    /// `None` when `inverse_covariance` is not `dimension` squared long.
    fn new(inverse_covariance: &[f64], dimension: usize) -> Option<Self> {
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return None;
        }
        let square = Mat::from_fn(dimension, dimension, |row, col| {
            inverse_covariance[row * dimension + col]
        });
        let lower = Factor::new(square.as_ref(), &[Attempt::plain(Method::Cholesky)])
            .ok()
            .and_then(|factor| {
                factor.cholesky_factor().map(|lower| {
                    let mut rows = Vec::with_capacity(dimension * dimension);
                    for row in 0..dimension {
                        for col in 0..dimension {
                            rows.push(lower[(row, col)]);
                        }
                    }
                    rows
                })
            });
        let matrix = if lower.is_some() {
            Vec::new()
        } else {
            inverse_covariance.to_vec()
        };
        Some(Self {
            dimension,
            lower,
            matrix,
        })
    }

    /// `L' x`, summed in column order, when `M` is positive definite.
    fn whiten_into(&self, lower: &[f64], vector: &[f64], out: &mut [f64]) {
        let n = self.dimension;
        for (j, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0_f64;
            for i in j..n {
                sum += lower[i * n + j] * vector[i];
            }
            *slot = sum;
        }
    }

    /// The distance of two vectors of this form's dimension.
    fn pair(&self, left: &[f64], right: &[f64]) -> Option<f64> {
        let n = self.dimension;
        if left.len() != n || right.len() != n {
            return None;
        }
        let mut stack = [0.0_f64; 2 * STACK_DIMENSION];
        let mut heap = Vec::new();
        let scratch = if n <= STACK_DIMENSION {
            &mut stack[..2 * n]
        } else {
            heap.resize(2 * n, 0.0);
            heap.as_mut_slice()
        };
        let (delta, transformed) = scratch.split_at_mut(n);
        for ((slot, left), right) in delta.iter_mut().zip(left).zip(right) {
            *slot = left - right;
        }
        let distance_sq = if let Some(lower) = &self.lower {
            self.whiten_into(lower, delta, transformed);
            let mut sum = 0.0_f64;
            for value in transformed.iter() {
                sum += value * value;
            }
            sum
        } else {
            let mut sum = 0.0_f64;
            for (i, matrix_row) in self.matrix.chunks_exact(n).enumerate() {
                let mut row = 0.0_f64;
                for (m, d) in matrix_row.iter().zip(delta.iter()) {
                    row += m * d;
                }
                sum += delta[i] * row;
            }
            sum
        };
        // The whitened form is a sum of squares and cannot be negative. The
        // plain form of a singular matrix can come out below zero by rounding,
        // by no more than (2n + 1) eps times the sum of the absolute terms,
        // which bounds the error of a quadratic form summed in this order. A
        // value below that is the form of an indefinite matrix and no distance.
        let rounding_bound = if self.lower.is_some() {
            0.0
        } else {
            let mut absolute = 0.0_f64;
            for (i, matrix_row) in self.matrix.chunks_exact(n).enumerate() {
                for (m, d) in matrix_row.iter().zip(delta.iter()) {
                    absolute += m.abs() * delta[i].abs() * d.abs();
                }
            }
            (2 * n + 1) as f64 * f64::EPSILON * absolute
        };
        if !distance_sq.is_finite() || distance_sq < -rounding_bound {
            return None;
        }
        Some(distance_sq.max(0.0).sqrt())
    }
}

/// Whether the symmetric part of `matrix`, of order `dimension`, has no
/// eigenvalue below minus `dimension * eps` times the largest in absolute value,
/// the size of the error of the eigenvalue computation, and every value is
/// finite.
fn check_positive_semi_definite(matrix: &[f64], dimension: usize) -> Result<(), Error> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(Error::NotPositiveSemiDefinite);
    }
    let symmetric: Vec<f64> = (0..dimension * dimension)
        .map(|index| {
            let (row, col) = (index / dimension, index % dimension);
            f64::midpoint(matrix[row * dimension + col], matrix[col * dimension + row])
        })
        .collect();
    let (values, _) = epistat_linalg::eigen::symmetric_eigen_jacobi(&symmetric, dimension);
    let largest = values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let floor = -(dimension as f64) * f64::EPSILON * largest;
    if values.iter().any(|value| *value < floor) {
        return Err(Error::NotPositiveSemiDefinite);
    }
    Ok(())
}

/// Dimensions up to this size keep the scratch vectors on the stack.
const STACK_DIMENSION: usize = 32;

/// Mahalanobis distance keyed by [`MatchingRecord::id`].
///
/// With a positive definite inverse covariance matrix every vector is
/// transformed once, on construction, and a pair costs one pass over the two
/// transformed vectors. Sums run in a fixed order, so the value does not
/// depend on the CPU or the number of threads.
#[derive(Debug, Clone)]
pub struct IdMapMahalanobisDistance<'a> {
    vectors: &'a VectorTable,
    form: Option<QuadraticForm>,
    /// The transformed vector of each row of `vectors`, when the matrix is
    /// positive definite.
    whitened: Vec<f64>,
}

impl<'a> IdMapMahalanobisDistance<'a> {
    /// The distance of the vectors in `vectors` under the inverse covariance
    /// matrix, which is as long as the dimension of the vectors squared.
    #[must_use]
    pub fn new(vectors: &'a VectorTable, inverse_covariance: &'a [f64]) -> Self {
        let dimension = vectors.dimension();
        let form = QuadraticForm::new(inverse_covariance, dimension);
        let mut whitened = Vec::new();
        if let Some((form, lower)) = form
            .as_ref()
            .and_then(|form| form.lower.as_deref().map(|lower| (form, lower)))
        {
            whitened.resize(vectors.len() * dimension, 0.0);
            for position in 0..vectors.len() {
                let out = &mut whitened[position * dimension..(position + 1) * dimension];
                form.whiten_into(lower, vectors.row(position), out);
            }
        }
        Self {
            vectors,
            form,
            whitened,
        }
    }
}

impl IdMapMahalanobisDistance<'_> {
    /// The row of the record's transformed vector, when the matrix is positive
    /// definite and the record has a vector of the right length.
    pub(crate) fn row(&self, id: &str) -> Option<usize> {
        self.dimension()?;
        self.vectors.position(id)
    }

    /// The length of a whitened vector, when the matrix is positive definite.
    pub(crate) fn dimension(&self) -> Option<usize> {
        let form = self.form.as_ref()?;
        form.lower.as_ref().map(|_| form.dimension)
    }

    /// The record's whitened vector, in which the Euclidean distance is this
    /// distance, when [`Self::row`] has a row for it.
    pub(crate) fn vector(&self, id: &str) -> Option<&[f64]> {
        let n = self.dimension()?;
        let row = self.row(id)?;
        Some(&self.whitened[row * n..(row + 1) * n])
    }

    /// The distance of two rows given by [`Self::row`].
    pub(crate) fn row_distance(&self, left: usize, right: usize) -> Option<f64> {
        let n = self.form.as_ref()?.dimension;
        let left = &self.whitened[left * n..(left + 1) * n];
        let right = &self.whitened[right * n..(right + 1) * n];
        let mut sum = 0.0_f64;
        for (a, b) in left.iter().zip(right) {
            let delta = a - b;
            sum += delta * delta;
        }
        sum.is_finite().then(|| sum.sqrt())
    }
}

impl<R: MatchingRecord> DistanceMetric<R> for IdMapMahalanobisDistance<'_> {
    fn channel(&self) -> &'static str {
        "mahalanobis_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let form = self.form.as_ref()?;
        if form.lower.is_none() {
            let left = self.vectors.get(anchor.id())?;
            let right = self.vectors.get(candidate.id())?;
            return form.pair(left, right);
        }
        self.row_distance(self.row(anchor.id())?, self.row(candidate.id())?)
    }
}

/// Euclidean distance keyed by [`MatchingRecord::id`].
///
/// The squares are summed in column order with a plain multiply and add, so
/// the value does not depend on the CPU or the number of threads.
#[derive(Debug, Clone)]
pub struct IdMapEuclideanDistance<'a> {
    vectors: &'a VectorTable,
}

impl<'a> IdMapEuclideanDistance<'a> {
    /// The Euclidean distance on `vectors`.
    #[must_use]
    pub const fn new(vectors: &'a VectorTable) -> Self {
        Self { vectors }
    }
}

impl IdMapEuclideanDistance<'_> {
    /// The length of the vectors.
    pub(crate) const fn dimension(&self) -> usize {
        self.vectors.dimension()
    }

    /// The record's vector, when it has one.
    pub(crate) fn vector(&self, id: &str) -> Option<&[f64]> {
        self.vectors.get(id)
    }
}

impl<R: MatchingRecord> DistanceMetric<R> for IdMapEuclideanDistance<'_> {
    fn channel(&self) -> &'static str {
        "euclidean_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = self.vectors.get(anchor.id())?;
        let right = self.vectors.get(candidate.id())?;
        let mut sum = 0.0_f64;
        for (a, b) in left.iter().zip(right) {
            let delta = a - b;
            sum += delta * delta;
        }
        sum.is_finite().then(|| sum.sqrt())
    }
}

impl<'a, R: MatchingRecord, D: DistanceMetric<R>> DistanceChannel<'a, R, D> {
    /// Create a channel using the metric's default diagnostics channel name.
    #[must_use]
    pub fn new(metric: &'a D, caliper: Option<DistanceCaliper>) -> Self {
        Self {
            metric,
            caliper,
            reason: metric.channel(),
            marker: PhantomData,
        }
    }

    crate::impl_with_reason!();

    /// The distance that is measured.
    #[must_use]
    pub const fn metric(&self) -> &'a D {
        self.metric
    }

    /// The largest allowed distance, if any.
    #[must_use]
    pub const fn caliper(&self) -> Option<f64> {
        match self.caliper {
            Some(caliper) => Some(caliper.get()),
            None => None,
        }
    }

    /// The largest allowed distance as its validated type, if any.
    #[must_use]
    pub const fn typed_caliper(&self) -> Option<DistanceCaliper> {
        self.caliper
    }

    /// The reason recorded when the caliper refuses a pair.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        self.reason
    }
}

/// Absolute date distance in days based on [`MatchingRecord::birth_date`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DateDistance;

impl<R: MatchingRecord> DistanceMetric<R> for DateDistance {
    fn channel(&self) -> &'static str {
        "date_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let diff_days = (candidate.birth_date() - anchor.birth_date()).num_days();
        let abs_days = diff_days.unsigned_abs();
        let capped = u32::try_from(abs_days).unwrap_or(u32::MAX);
        Some(f64::from(capped))
    }
}

/// Distance on scalar propensity scores.
///
/// The supplied accessor should return a finite score for each record.
#[derive(Debug, Clone, Copy)]
pub struct PropensityScoreDistance<F> {
    accessor: F,
}

impl<F> PropensityScoreDistance<F> {
    /// The score distance of the values `accessor` returns.
    #[must_use]
    pub const fn new(accessor: F) -> Self {
        Self { accessor }
    }
}

impl<R: MatchingRecord, F> DistanceMetric<R> for PropensityScoreDistance<F>
where
    F: Fn(&R) -> Option<f64>,
{
    fn channel(&self) -> &'static str {
        "propensity_score_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = (self.accessor)(anchor)?;
        let right = (self.accessor)(candidate)?;
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        Some((left - right).abs())
    }
}

/// Mahalanobis distance using a precomputed inverse covariance matrix.
///
/// The inverse covariance matrix is supplied in row-major format.
#[derive(Debug, Clone)]
pub struct MahalanobisDistance<F> {
    accessor: F,
    form: QuadraticForm,
}

impl<F> MahalanobisDistance<F> {
    /// Construct a Mahalanobis metric from an inverse covariance matrix.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when `dimension` is zero, the matrix length
    /// does not equal `dimension * dimension` or the matrix is not positive
    /// semi-definite.
    #[expect(clippy::needless_pass_by_value, reason = "the signature is public")]
    pub fn new(accessor: F, inverse_covariance: Vec<f64>, dimension: usize) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::ZeroDimension);
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return Err(Error::InverseCovarianceLength);
        }
        check_positive_semi_definite(&inverse_covariance, dimension)?;
        let form = QuadraticForm::new(&inverse_covariance, dimension)
            .ok_or(Error::InverseCovarianceLength)?;

        Ok(Self { accessor, form })
    }
}

impl<R: MatchingRecord, F> DistanceMetric<R> for MahalanobisDistance<F>
where
    F: Fn(&R) -> Option<Vec<f64>>,
{
    fn channel(&self) -> &'static str {
        "mahalanobis_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = (self.accessor)(anchor)?;
        let right = (self.accessor)(candidate)?;
        self.form.pair(&left, &right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    #[derive(Clone)]
    struct DemoRecord {
        base: BaseRecord,
        numeric: Vec<f64>,
    }

    impl DemoRecord {
        fn new(id: &str, birth_date: NaiveDate, numeric: Vec<f64>) -> Self {
            Self {
                base: BaseRecord::new(id, birth_date),
                numeric,
            }
        }
    }

    impl crate::MatchingRecord for DemoRecord {
        crate::delegate_matching_record!(base);
    }

    #[test]
    fn date_distance_returns_absolute_day_difference() {
        let anchor = BaseRecord::new("a", date(2010, 1, 1));
        let candidate = BaseRecord::new("c", date(2010, 1, 6));
        let metric = DateDistance;
        let distance = metric
            .distance(&anchor, &candidate)
            .expect("date distance should exist");
        assert!((distance - 5.0).abs() < 1e-12);
    }

    #[test]
    fn propensity_score_distance_handles_missing_values() {
        let anchor = BaseRecord::new("a", date(2010, 1, 1));
        let candidate = BaseRecord::new("c", date(2010, 1, 2));
        let scores: RapidHashMap<String, f64> =
            std::iter::once(("a".to_string(), 0.25_f64)).collect();
        let metric =
            PropensityScoreDistance::new(|row: &BaseRecord| scores.get(row.id.as_str()).copied());
        assert!(metric.distance(&anchor, &candidate).is_none());
    }

    #[test]
    fn distance_channel_defaults_to_metric_channel_and_allows_override() {
        let metric = DateDistance;
        let default_channel: DistanceChannel<'_, BaseRecord, _> = DistanceChannel::new(
            &metric,
            Some(DistanceCaliper::new(3.0).expect("valid positive caliper")),
        );
        assert_eq!(default_channel.reason(), "date_caliper");
        assert_eq!(default_channel.caliper(), Some(3.0));

        let custom = default_channel.with_reason("custom_date_band");
        assert_eq!(custom.reason(), "custom_date_band");
    }

    #[test]
    fn distance_config_constructors_and_reason_override() {
        let date = DistanceConfig::date(Some(
            DistanceCaliper::new(10.0).expect("valid positive date caliper"),
        ))
        .with_reason("date_band");
        assert_eq!(date.reason(), "date_band");
        assert_eq!(date.caliper(), Some(10.0));

        let ps = DistanceConfig::propensity_score_map(
            std::iter::once(("a".to_string(), 0.2_f64)).collect(),
            Some(DistanceCaliper::new(0.05).expect("valid positive ps caliper")),
        );
        assert_eq!(ps.reason(), "propensity_score_caliper");

        let maha = DistanceConfig::mahalanobis_map(
            RapidHashMap::default(),
            vec![1.0],
            1,
            Some(DistanceCaliper::new(3.0).expect("valid positive mahalanobis caliper")),
        )
        .expect("valid mahalanobis config");
        assert_eq!(maha.reason(), "mahalanobis_caliper");
        assert_eq!(maha.caliper(), Some(3.0));
    }

    #[test]
    fn mahalanobis_distance_matches_identity_matrix_expectation() {
        let anchor = DemoRecord::new("a", date(2010, 1, 1), vec![1.0, 2.0]);
        let candidate = DemoRecord::new("c", date(2010, 1, 2), vec![4.0, 6.0]);
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let metric =
            MahalanobisDistance::new(|row: &DemoRecord| Some(row.numeric.clone()), identity, 2)
                .expect("valid identity matrix");

        let distance = metric
            .distance(&anchor, &candidate)
            .expect("distance should be computable");
        assert!((distance - 5.0).abs() < 1e-12);
    }

    #[test]
    fn mahalanobis_distance_rejects_invalid_matrix_length() {
        let result =
            MahalanobisDistance::new(|_row: &DemoRecord| None::<Vec<f64>>, vec![1.0, 0.0, 0.0], 2);
        assert!(matches!(result, Err(Error::InverseCovarianceLength)));
    }

    #[test]
    fn id_map_mahalanobis_distance_handles_invalid_matrix_length() {
        let vectors: RapidHashMap<String, Vec<f64>> = [
            ("a".to_string(), vec![1.0, 2.0]),
            ("b".to_string(), vec![3.0, 4.0]),
        ]
        .into_iter()
        .collect();
        let table = VectorTable::from_map(&vectors, 2);
        let metric = IdMapMahalanobisDistance::new(&table, &[1.0, 0.0, 0.0]);
        let anchor = BaseRecord::new("a", date(2010, 1, 1));
        let candidate = BaseRecord::new("b", date(2010, 1, 2));

        assert!(metric.distance(&anchor, &candidate).is_none());
    }

    #[test]
    fn mahalanobis_distance_rejects_a_non_finite_matrix() {
        let invalid_inverse = vec![f64::NAN, 0.0, 0.0, 1.0];
        let result = MahalanobisDistance::new(
            |row: &DemoRecord| Some(row.numeric.clone()),
            invalid_inverse,
            2,
        );
        assert!(matches!(result, Err(Error::NotPositiveSemiDefinite)));
    }

    /// The per-pair form: two faer matrix products per distance.
    fn mahalanobis_distance_faer_reference(
        left: &[f64],
        right: &[f64],
        inverse_covariance_matrix: &Mat<f64>,
    ) -> Option<f64> {
        let dimension = inverse_covariance_matrix.nrows();
        let delta = Mat::from_fn(dimension, 1, |row, _| left[row] - right[row]);
        let transformed = inverse_covariance_matrix * &delta;
        let distance_sq = (delta.transpose() * transformed)[(0, 0)];
        if !distance_sq.is_finite() || distance_sq < 0.0 {
            return None;
        }
        Some(distance_sq.sqrt())
    }

    struct Xorshift(u64);

    impl Xorshift {
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    /// A positive definite matrix, row-major: `A' A + I` for a random `A`.
    fn random_positive_definite(rng: &mut Xorshift, n: usize) -> Vec<f64> {
        let a: Vec<f64> = (0..n * n).map(|_| rng.unit() - 0.5).collect();
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = if i == j { 1.0 } else { 0.0 };
                for k in 0..n {
                    sum += a[k * n + i] * a[k * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    /// Distances through the transformed vectors agree with the matrix
    /// products. Both sides round differently, so the tolerance is a small
    /// multiple of the machine epsilon times the dimension, relative to the
    /// distance or to 1 when it is smaller. The covariates are of order 1, so
    /// the cancellation in a difference is of the order of 1 as well.
    #[test]
    fn whitened_distance_matches_the_matrix_form() {
        let mut rng = Xorshift(0x2545_f491_4f6c_dd1d);
        for dimension in [1_usize, 2, 3, 7, 12, 31, 33] {
            let inverse = random_positive_definite(&mut rng, dimension);
            let matrix = Mat::from_fn(dimension, dimension, |r, c| inverse[r * dimension + c]);
            let vectors: RapidHashMap<String, Vec<f64>> = (0..30)
                .map(|i| {
                    (
                        format!("r{i:02}"),
                        (0..dimension).map(|_| 4.0 * rng.unit() - 2.0).collect(),
                    )
                })
                .collect();
            let table = VectorTable::from_map(&vectors, dimension);
            let metric = IdMapMahalanobisDistance::new(&table, &inverse);
            for (a, left) in &vectors {
                for (b, right) in &vectors {
                    let got = metric
                        .distance(
                            &BaseRecord::new(a, date(2010, 1, 1)),
                            &BaseRecord::new(b, date(2010, 1, 1)),
                        )
                        .expect("distance");
                    let want = mahalanobis_distance_faer_reference(left, right, &matrix)
                        .expect("distance");
                    let tolerance = 64.0 * dimension as f64 * f64::EPSILON * want.max(1.0);
                    assert!(
                        (got - want).abs() <= tolerance,
                        "dimension {dimension}: {got} against {want}"
                    );
                }
            }
        }
    }

    /// The nearest candidate of every anchor is the same under the matrix
    /// form and the whitened form of the distance, on data with many exact ties
    /// (integer covariates, repeated rows). Exact duplicates have identical transformed vectors, so their
    /// distances are equal in both forms and a first-index tie-break agrees.
    /// Other near ties are skipped when the gap is within rounding.
    #[test]
    fn nearest_candidates_agree_with_the_matrix_form() {
        for seed in 1..=6_u64 {
            let mut rng = Xorshift(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0x1234_5679));
            let dimension = 4;
            let inverse = random_positive_definite(&mut rng, dimension);
            let matrix = Mat::from_fn(dimension, dimension, |r, c| inverse[r * dimension + c]);
            let rows: Vec<Vec<f64>> = (0..200)
                .map(|_| (0..dimension).map(|_| (rng.unit() * 3.0).floor()).collect())
                .collect();
            let vectors: RapidHashMap<String, Vec<f64>> = rows
                .iter()
                .enumerate()
                .map(|(i, row)| (format!("r{i:03}"), row.clone()))
                .collect();
            let table = VectorTable::from_map(&vectors, dimension);
            let metric = IdMapMahalanobisDistance::new(&table, &inverse);
            let record = |i: usize| BaseRecord::new(format!("r{i:03}"), date(2010, 1, 1));
            for anchor in 0..100 {
                let new: Vec<f64> = (100..200)
                    .map(|c| {
                        metric
                            .distance(&record(anchor), &record(c))
                            .expect("distance")
                    })
                    .collect();
                let old: Vec<f64> = (100..200)
                    .map(|c| {
                        mahalanobis_distance_faer_reference(&rows[anchor], &rows[c], &matrix)
                            .expect("distance")
                    })
                    .collect();
                let first = |d: &[f64]| {
                    let best = d.iter().copied().fold(f64::INFINITY, f64::min);
                    (
                        best,
                        d.iter().position(|v| *v <= best + 1e-9 * best.max(1.0)),
                    )
                };
                let (best_new, pick_new) = first(&new);
                let (best_old, pick_old) = first(&old);
                assert!((best_new - best_old).abs() <= 1e-9 * best_old.max(1.0));
                assert_eq!(pick_new, pick_old, "seed {seed}, anchor {anchor}");
            }
        }
    }

    /// A matrix that is not positive definite is refused by the public
    /// constructors.
    #[test]
    fn constructors_refuse_an_indefinite_matrix() {
        let inverse = vec![1.0, 2.0, 2.0, 1.0];
        assert!(matches!(
            MahalanobisDistance::new(|_row: &BaseRecord| None::<Vec<f64>>, inverse.clone(), 2),
            Err(Error::NotPositiveSemiDefinite)
        ));
        assert!(matches!(
            DistanceConfig::mahalanobis_map(RapidHashMap::default(), inverse, 2, None),
            Err(Error::NotPositiveSemiDefinite)
        ));
        // A singular positive semi-definite matrix is a distance.
        assert!(
            MahalanobisDistance::new(
                |_row: &BaseRecord| None::<Vec<f64>>,
                vec![1.0, 1.0, 1.0, 1.0],
                2
            )
            .is_ok()
        );
    }

    /// A matrix that is not positive definite keeps the plain quadratic form.
    #[test]
    fn indefinite_matrix_uses_the_plain_form() {
        let inverse = vec![1.0, 2.0, 2.0, 1.0];
        let vectors: RapidHashMap<String, Vec<f64>> = [
            ("a".to_string(), vec![0.0, 0.0]),
            ("b".to_string(), vec![1.0, 1.0]),
            ("c".to_string(), vec![1.0, -1.0]),
        ]
        .into_iter()
        .collect();
        let table = VectorTable::from_map(&vectors, 2);
        let metric = IdMapMahalanobisDistance::new(&table, &inverse);
        let a = BaseRecord::new("a", date(2010, 1, 1));
        let b = BaseRecord::new("b", date(2010, 1, 1));
        let c = BaseRecord::new("c", date(2010, 1, 1));
        let ab = metric.distance(&a, &b).expect("positive form");
        assert!((ab - 6.0_f64.sqrt()).abs() < 1e-12);
        let scalar = mahalanobis_distance_scalar_reference(&[0.0, 0.0], &[1.0, 1.0], &inverse, 2)
            .expect("positive form");
        assert!((ab - scalar).abs() < 1e-12);
        assert!(metric.distance(&a, &c).is_none());
    }

    fn mahalanobis_distance_scalar_reference(
        left: &[f64],
        right: &[f64],
        inverse_covariance: &[f64],
        dimension: usize,
    ) -> Option<f64> {
        if left.len() != dimension || right.len() != dimension {
            return None;
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return None;
        }

        let mut distance_sq = 0.0_f64;
        for row in 0..dimension {
            let mut row_dot = 0.0_f64;
            for col in 0..dimension {
                let idx = row * dimension + col;
                let delta_col = left[col] - right[col];
                row_dot += inverse_covariance[idx] * delta_col;
            }
            let delta_row = left[row] - right[row];
            distance_sq += delta_row * row_dot;
        }

        if !distance_sq.is_finite() || distance_sq < 0.0 {
            return None;
        }
        Some(distance_sq.sqrt())
    }

    #[test]
    fn a_vector_table_holds_the_vectors_of_its_map_and_drops_the_wrong_length() {
        let map: RapidHashMap<String, Vec<f64>> = [
            ("b".to_string(), vec![3.0, 4.0]),
            ("a".to_string(), vec![1.0, 2.0]),
            ("short".to_string(), vec![9.0]),
        ]
        .into_iter()
        .collect();
        let table = VectorTable::from_map(&map, 2);
        assert_eq!(table.ids(), ["a", "b"]);
        assert_eq!(table.values(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(table.get("b"), Some([3.0, 4.0].as_slice()));
        assert_eq!(table.get("short"), None);
        assert_eq!(table.position("a"), Some(0));
        let mut expected = map;
        expected.remove("short");
        assert_eq!(table.to_map(), expected);
        assert_eq!(
            VectorTable::new(table.ids().into(), 3, table.values().into()).unwrap_err(),
            Error::VectorLength
        );
        assert_eq!(
            VectorTable::new(table.ids().into(), 0, Vec::new().into()).unwrap_err(),
            Error::ZeroDimension
        );
        let euclidean = IdMapEuclideanDistance::new(&table);
        let a = BaseRecord::new("a", date(2010, 1, 1));
        let b = BaseRecord::new("b", date(2010, 1, 1));
        let gone = BaseRecord::new("short", date(2010, 1, 1));
        assert_eq!(euclidean.distance(&a, &b), Some(8.0_f64.sqrt()));
        assert_eq!(euclidean.distance(&a, &gone), None);
    }
}
