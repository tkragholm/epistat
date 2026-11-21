use super::records::MatchingRecord;
use crate::types::DistanceCaliper;
use epistat_linalg::{Attempt, Factor, Mat, Method};
use ordered_float::OrderedFloat;
use rapidhash::RapidHashMap;
use std::collections::BTreeMap;
use std::marker::PhantomData;

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

/// Named and reusable distance setup independent of concrete record type.
#[derive(Debug, Clone)]
pub enum DistanceConfig {
    Date {
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
    PropensityScoreMap {
        scores: RapidHashMap<String, f64>,
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
    MahalanobisMap {
        vectors: RapidHashMap<String, Vec<f64>>,
        inverse_covariance: Vec<f64>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
    EuclideanMap {
        vectors: RapidHashMap<String, Vec<f64>>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
    /// Distances between every anchor and every candidate, given by the
    /// caller.
    Matrix {
        matrix: PairDistanceMatrix,
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
}

impl DistanceConfig {
    #[must_use]
    pub const fn date(caliper: Option<DistanceCaliper>) -> Self {
        Self::Date {
            caliper,
            reason: "date_caliper",
        }
    }

    #[must_use]
    pub const fn propensity_score_map(
        scores: RapidHashMap<String, f64>,
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
    /// Returns [`MahalanobisError`] when `dimension` is zero or matrix shape is invalid.
    pub fn mahalanobis_map(
        vectors: RapidHashMap<String, Vec<f64>>,
        inverse_covariance: Vec<f64>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
    ) -> Result<Self, MahalanobisError> {
        if dimension == 0 {
            return Err(MahalanobisError::ZeroDimension);
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return Err(MahalanobisError::InvalidInverseCovarianceLength);
        }

        Ok(Self::MahalanobisMap {
            vectors,
            inverse_covariance,
            dimension,
            caliper,
            reason: "mahalanobis_caliper",
        })
    }

    /// Build a Euclidean config from id-indexed vectors, already scaled as the
    /// caller wants them.
    ///
    /// # Errors
    ///
    /// Returns [`MahalanobisError::ZeroDimension`] when `dimension` is zero.
    pub fn euclidean_map(
        vectors: RapidHashMap<String, Vec<f64>>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
    ) -> Result<Self, MahalanobisError> {
        if dimension == 0 {
            return Err(MahalanobisError::ZeroDimension);
        }
        Ok(Self::EuclideanMap {
            vectors,
            dimension,
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
                dimension,
                ..
            } => ConfiguredMetric::Mahalanobis(IdMapMahalanobisDistance::new(
                vectors,
                inverse_covariance,
                *dimension,
            )),
            Self::EuclideanMap {
                vectors, dimension, ..
            } => ConfiguredMetric::Euclidean(IdMapEuclideanDistance::new(vectors, *dimension)),
            Self::Matrix { matrix, .. } => ConfiguredMetric::Matrix(MatrixDistance::new(matrix)),
        }
    }

    /// The scores common-support trimming needs, which only a propensity
    /// configuration has.
    pub(crate) const fn propensity_scores(&self) -> Option<&RapidHashMap<String, f64>> {
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
    Scores(&'a RapidHashMap<String, f64>),
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

/// Why a table of pair distances is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairDistanceMatrixError {
    /// There are no anchors or no candidates.
    Empty,
    /// The number of values is not the number of anchors times candidates.
    Shape {
        /// Anchors times candidates.
        expected: usize,
        /// Values given.
        found: usize,
    },
    /// An id appears twice among the anchors or among the candidates.
    DuplicateId(String),
    /// A value is not a number. An infinite value is allowed and forbids the
    /// pair.
    NotANumber {
        /// Row of the value.
        anchor: usize,
        /// Column of the value.
        candidate: usize,
    },
}

impl std::fmt::Display for PairDistanceMatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("the distance matrix has no anchors or no candidates"),
            Self::Shape { expected, found } => write!(
                f,
                "the distance matrix needs {expected} values, one for each anchor and \
                 candidate, and has {found}"
            ),
            Self::DuplicateId(id) => write!(f, "the id {id} appears twice in the distance matrix"),
            Self::NotANumber { anchor, candidate } => write!(
                f,
                "the distance matrix has no number at row {anchor}, column {candidate}; \
                 use an infinite value to forbid a pair"
            ),
        }
    }
}

impl std::error::Error for PairDistanceMatrixError {}

/// Distances between every anchor and every candidate, as the matrix a caller
/// gives as `distance` to `matchit()`.
///
/// Rows are anchors and columns are candidates, named by their ids. An
/// infinite value forbids the pair, as in `matchit()`. A pair whose anchor or
/// candidate is not in the table has no distance and is never matched.
#[derive(Debug, Clone)]
pub struct PairDistanceMatrix {
    anchor_ids: Vec<String>,
    candidate_ids: Vec<String>,
    values: Vec<f64>,
    anchor_position: RapidHashMap<String, usize>,
    candidate_position: RapidHashMap<String, usize>,
}

impl PairDistanceMatrix {
    /// Build the table from row-major values, one row per anchor.
    ///
    /// # Errors
    ///
    /// [`PairDistanceMatrixError`] when a group is empty, the values do not
    /// fill the table, an id repeats, or a value is not a number.
    pub fn new(
        anchor_ids: Vec<String>,
        candidate_ids: Vec<String>,
        values: Vec<f64>,
    ) -> Result<Self, PairDistanceMatrixError> {
        if anchor_ids.is_empty() || candidate_ids.is_empty() {
            return Err(PairDistanceMatrixError::Empty);
        }
        let expected = anchor_ids.len().saturating_mul(candidate_ids.len());
        if values.len() != expected {
            return Err(PairDistanceMatrixError::Shape {
                expected,
                found: values.len(),
            });
        }
        if let Some(at) = values.iter().position(|value| value.is_nan()) {
            return Err(PairDistanceMatrixError::NotANumber {
                anchor: at / candidate_ids.len(),
                candidate: at % candidate_ids.len(),
            });
        }
        let positions = |ids: &[String]| -> Result<RapidHashMap<String, usize>, _> {
            let mut map = RapidHashMap::default();
            for (position, id) in ids.iter().enumerate() {
                if map.insert(id.clone(), position).is_some() {
                    return Err(PairDistanceMatrixError::DuplicateId(id.clone()));
                }
            }
            Ok(map)
        };
        Ok(Self {
            anchor_position: positions(&anchor_ids)?,
            candidate_position: positions(&candidate_ids)?,
            anchor_ids,
            candidate_ids,
            values,
        })
    }

    /// Ids of the rows.
    #[must_use]
    pub fn anchor_ids(&self) -> &[String] {
        &self.anchor_ids
    }

    /// Ids of the columns.
    #[must_use]
    pub fn candidate_ids(&self) -> &[String] {
        &self.candidate_ids
    }

    /// Row of an anchor.
    #[must_use]
    pub fn anchor_position(&self, id: &str) -> Option<usize> {
        self.anchor_position.get(id).copied()
    }

    /// Column of a candidate.
    #[must_use]
    pub fn candidate_position(&self, id: &str) -> Option<usize> {
        self.candidate_position.get(id).copied()
    }

    /// The value at a row and column, or `None` outside the table or where the
    /// pair is forbidden by an infinite value.
    #[must_use]
    pub fn at(&self, anchor: usize, candidate: usize) -> Option<f64> {
        if anchor >= self.anchor_ids.len() || candidate >= self.candidate_ids.len() {
            return None;
        }
        let value = self.values[anchor * self.candidate_ids.len() + candidate];
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
                .map(|record| self.anchor_position(record.id()))
                .collect(),
            candidates
                .iter()
                .map(|record| self.candidate_position(record.id()))
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
            self.matrix.anchor_position(anchor.id())?,
            self.matrix.candidate_position(candidate.id())?,
        )
    }
}

/// Propensity score distance keyed by [`MatchingRecord::id`].
#[derive(Debug, Clone)]
pub struct IdMapPropensityScoreDistance<'a> {
    scores: &'a RapidHashMap<String, f64>,
    index: Option<BTreeMap<OrderedFloat<f64>, Vec<usize>>>,
}

impl<'a> IdMapPropensityScoreDistance<'a> {
    #[must_use]
    pub const fn new(scores: &'a RapidHashMap<String, f64>) -> Self {
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

    #[must_use]
    pub const fn scores(&self) -> &'a RapidHashMap<String, f64> {
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
/// pair whose form is clearly negative has no distance.
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
        if !distance_sq.is_finite() || distance_sq < -1e-12 {
            return None;
        }
        Some(distance_sq.max(0.0).sqrt())
    }
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
    vectors: &'a RapidHashMap<String, Vec<f64>>,
    form: Option<QuadraticForm>,
    /// Position of each id's transformed vector in `whitened`, for the ids
    /// whose vector has the right length.
    slots: RapidHashMap<&'a str, usize>,
    whitened: Vec<f64>,
}

impl<'a> IdMapMahalanobisDistance<'a> {
    #[must_use]
    pub fn new(
        vectors: &'a RapidHashMap<String, Vec<f64>>,
        inverse_covariance: &'a [f64],
        dimension: usize,
    ) -> Self {
        let form = QuadraticForm::new(inverse_covariance, dimension);
        let mut slots = RapidHashMap::default();
        let mut whitened = Vec::new();
        if let Some((form, lower)) = form
            .as_ref()
            .and_then(|form| form.lower.as_deref().map(|lower| (form, lower)))
        {
            // Hash order would not matter for the values, but a sorted pass
            // keeps the buffer layout reproducible.
            let mut ids: Vec<&'a String> = vectors.keys().collect();
            ids.sort_unstable();
            slots.reserve(ids.len());
            whitened.resize(ids.len() * dimension, 0.0);
            let mut position = 0;
            for id in ids {
                let vector = &vectors[id];
                if vector.len() != dimension {
                    continue;
                }
                let out = &mut whitened[position * dimension..(position + 1) * dimension];
                form.whiten_into(lower, vector, out);
                slots.insert(id.as_str(), position);
                position += 1;
            }
            whitened.truncate(position * dimension);
        }
        Self {
            vectors,
            form,
            slots,
            whitened,
        }
    }
}

impl IdMapMahalanobisDistance<'_> {
    /// The row of the record's transformed vector, when the matrix is positive
    /// definite and the record has a vector of the right length.
    pub(crate) fn row(&self, id: &str) -> Option<usize> {
        self.slots.get(id).copied()
    }

    /// The length of a whitened vector, when the matrix is positive definite.
    pub(crate) fn dimension(&self) -> Option<usize> {
        let form = self.form.as_ref()?;
        form.lower.as_ref().map(|_| form.dimension)
    }

    /// The whitened vector of a row given by [`Self::row`].
    pub(crate) fn whitened_row(&self, row: usize, dimension: usize) -> &[f64] {
        &self.whitened[row * dimension..(row + 1) * dimension]
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
    vectors: &'a RapidHashMap<String, Vec<f64>>,
    dimension: usize,
}

impl<'a> IdMapEuclideanDistance<'a> {
    #[must_use]
    pub const fn new(vectors: &'a RapidHashMap<String, Vec<f64>>, dimension: usize) -> Self {
        Self { vectors, dimension }
    }
}

impl<R: MatchingRecord> DistanceMetric<R> for IdMapEuclideanDistance<'_> {
    fn channel(&self) -> &'static str {
        "euclidean_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = self.vectors.get(anchor.id())?;
        let right = self.vectors.get(candidate.id())?;
        if left.len() != self.dimension || right.len() != self.dimension {
            return None;
        }
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

    #[must_use]
    pub const fn metric(&self) -> &'a D {
        self.metric
    }

    #[must_use]
    pub const fn caliper(&self) -> Option<f64> {
        match self.caliper {
            Some(caliper) => Some(caliper.get()),
            None => None,
        }
    }

    #[must_use]
    pub const fn typed_caliper(&self) -> Option<DistanceCaliper> {
        self.caliper
    }

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

/// Errors constructing [`MahalanobisDistance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MahalanobisError {
    ZeroDimension,
    InvalidInverseCovarianceLength,
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
    /// Returns [`MahalanobisError`] when `dimension` is zero or the matrix length
    /// does not equal `dimension * dimension`.
    #[expect(clippy::needless_pass_by_value, reason = "the signature is public")]
    pub fn new(
        accessor: F,
        inverse_covariance: Vec<f64>,
        dimension: usize,
    ) -> Result<Self, MahalanobisError> {
        if dimension == 0 {
            return Err(MahalanobisError::ZeroDimension);
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return Err(MahalanobisError::InvalidInverseCovarianceLength);
        }
        let form = QuadraticForm::new(&inverse_covariance, dimension)
            .ok_or(MahalanobisError::InvalidInverseCovarianceLength)?;

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
        assert!(matches!(
            result,
            Err(MahalanobisError::InvalidInverseCovarianceLength)
        ));
    }

    #[test]
    fn id_map_mahalanobis_distance_handles_invalid_matrix_length() {
        let vectors: RapidHashMap<String, Vec<f64>> = [
            ("a".to_string(), vec![1.0, 2.0]),
            ("b".to_string(), vec![3.0, 4.0]),
        ]
        .into_iter()
        .collect();
        let metric = IdMapMahalanobisDistance::new(&vectors, &[1.0, 0.0, 0.0], 2);
        let anchor = BaseRecord::new("a", date(2010, 1, 1));
        let candidate = BaseRecord::new("b", date(2010, 1, 2));

        assert!(metric.distance(&anchor, &candidate).is_none());
    }

    #[test]
    fn mahalanobis_distance_rejects_non_finite_result() {
        let anchor = DemoRecord::new("a", date(2010, 1, 1), vec![1.0, 2.0]);
        let candidate = DemoRecord::new("c", date(2010, 1, 2), vec![4.0, 6.0]);
        let invalid_inverse = vec![f64::NAN, 0.0, 0.0, 1.0];
        let metric = MahalanobisDistance::new(
            |row: &DemoRecord| Some(row.numeric.clone()),
            invalid_inverse,
            2,
        )
        .expect("matrix shape is valid");

        assert!(metric.distance(&anchor, &candidate).is_none());
    }

    /// The old per-pair form: two faer matrix products per distance.
    fn mahalanobis_distance_faer_reference(
        left: &[f64],
        right: &[f64],
        inverse_covariance_matrix: &Mat<f64>,
    ) -> Option<f64> {
        let dimension = inverse_covariance_matrix.nrows();
        let delta = Mat::from_fn(dimension, 1, |row, _| left[row] - right[row]);
        let transformed = inverse_covariance_matrix * &delta;
        let distance_sq = (delta.transpose() * transformed)[(0, 0)];
        if !distance_sq.is_finite() || distance_sq < -1e-12 {
            return None;
        }
        Some(distance_sq.max(0.0).sqrt())
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
            let metric = IdMapMahalanobisDistance::new(&vectors, &inverse, dimension);
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

    /// The nearest candidate of every anchor is the same under the pairwise
    /// distance and the matrix form, on data with many exact ties (integer covariates, repeated
    /// rows). Exact duplicates have identical transformed vectors, so their
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
            let metric = IdMapMahalanobisDistance::new(&vectors, &inverse, dimension);
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
        let metric = IdMapMahalanobisDistance::new(&vectors, &inverse, 2);
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

        if !distance_sq.is_finite() || distance_sq < -1e-12 {
            return None;
        }
        Some(distance_sq.max(0.0).sqrt())
    }
}
