//! Variances of an estimator from its influence function.
//!
//! An estimator `theta_hat` is asymptotically linear when
//! `theta_hat - theta = sum(psi_i) / n` plus a negligible remainder, where
//! `psi_i` is the influence function of unit `i` \[1\]. Its variance is then
//! estimated from the sample values of `psi_i`. [`Influence`] holds those
//! values, for one estimate or several, with the unit code of each row and
//! the cluster rule its estimator reports standard errors under.
//!
//! # Scale
//!
//! Every epistat estimator gives its influence function on this one scale,
//! per unit, with
//!
//! ```text
//! theta_hat - theta ~ sum_i psi_i / n,       var(theta_hat) ~ sum_i psi_i^2 / n^2
//! ```
//!
//! Some references report the dfbeta scale instead, `D_i = psi_i / n`, whose
//! squares sum to the variance (R's `survival` and `riskRegression` among
//! them). [`Influence::from_dfbeta`] multiplies such values by `n`.
//!
//! # Method
//!
//! [`Influence::variance`] sums the values within each of `G` clusters and
//! centres the sums. With `S_ga` the sum of column `a` over the `n_g` units
//! of cluster `g` and `psi_bar_a` the mean of column `a` over all `n` units,
//! entry `(a, b)` of the covariance is
//!
//! ```text
//! d_ga   = S_ga - n_g psi_bar_a
//! V_ab   = c sum_g d_ga d_gb / n^2
//! ```
//!
//! the cluster-robust variance of \[2\] and \[3\]. Without clusters every unit
//! is its own cluster, so the clustered and the unclustered variance agree
//! when the clusters are singletons. The factor `c` is the [`ClusterRule`]:
//!
//! ```text
//! SmallSample     c = G / (G - 1)
//! NoSmallSample   c = 1
//! ```
//!
//! With singleton clusters, `SmallSample` gives `sd(psi)^2 / n` with the
//! `n - 1` divisor, as R's `tmle` package reports, and `NoSmallSample` gives
//! the `n` divisor, as R's `did`, `survival`'s robust variances and
//! `riskRegression::ate` do. An influence function evaluated at the estimate
//! has mean zero, so centring changes nothing there but the rounding.
//!
//! [`Influence::combine`] forms the influence function of a linear
//! combination of the estimates, `psi_i = sum_j w_j psi_ij`, the influence
//! function of `sum_j w_j theta_j`.
//!
//! # Keys
//!
//! Each value of an influence function has a key. An estimator whose data
//! carry a unit column keys its values by the caller's unit codes
//! ([`KeyedBy::Units`]), the same codes the caller clusters by. An estimator
//! whose data have no unit column keys them by the rows of its input
//! ([`KeyedBy::Rows`]). Clusters looked up by key carry the same mark, and
//! [`Influence::variance`] refuses clusters keyed by unit codes on an
//! influence function keyed by rows, and the other way round: a row number is
//! also a valid unit code, so the lookup would otherwise succeed and pair
//! each row with another unit's cluster.
//!
//! # Determinism
//!
//! Cluster sums are formed in row order and added in cluster order, which is
//! increasing cluster code for clusters from [`Clusters::from_codes`],
//! [`Clusters::per_unit`] and [`Clusters::per_row`]. The covariance computes the variance of column `a`
//! with the same sums and the same products in the same order, so its diagonal
//! entry equals the variance of that column alone to the bit. Nothing depends
//! on the number of threads.
//!
//! # Edge cases
//!
//! - No keys, no columns, values of the wrong count, a value that is not
//!   finite and a key that repeats are [`InputError`]s when the
//!   [`Influence`] is built.
//! - Positional clusters that cover another number of keys, and clusters
//!   looked up by key that do not cover a key, are [`InputError`]s. Clusters
//!   keyed otherwise than the influence function are
//!   [`Error::KeyMismatch`].
//! - Fewer than two clusters leave every entry of the covariance `None`: the
//!   factor of `SmallSample` does not exist, and one cluster carries no
//!   information about the variance under either rule.
//! - A covariance entry that is not finite is [`Error::NotFinite`].
//!
//! # Earlier functions
//!
//! [`standard_error`], [`clustered_variance`], [`clustered_standard_error`]
//! and [`clustered_covariance`] are the 0.3 functions, kept with their bits
//! for one release. The first centres and has no factor, the others have the
//! factor and do not centre, so they disagree with each other on singleton
//! clusters. They are deprecated in favour of [`Influence::variance`].
//!
//! # References
//!
//! 1. Hampel FR. The influence curve and its role in robust estimation. J Am
//!    Stat Assoc. 1974;69(346):383-93.
//! 2. Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
//!    models. Biometrika. 1986;73(1):13-22.
//! 3. Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for
//!    inference with clustered errors. Rev Econ Stat. 2008;90(3):414-27.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use epistat_core::{Covariance, InputError, check};
use epistat_stats::sum::lanes;

use crate::{Clusters, Error, check_length, empty};

/// The small-sample factor of a cluster-robust variance from an influence
/// function.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ClusterRule {
    /// `G / (G - 1)` for `G` clusters, R's `sandwich::vcovCL` with
    /// `cadjust = TRUE` and, without clusters, the `n - 1` divisor of R's
    /// `tmle` package.
    #[default]
    SmallSample,
    /// No factor, for ports whose R reference has none: R's `did`, the
    /// robust variances of R's `survival`, and `riskRegression::ate`, whose
    /// variance without clusters has the `n` divisor.
    NoSmallSample,
}

/// What the keys of an influence function, or of clusters looked up by key,
/// are.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyedBy {
    /// The caller's unit codes, such as the codes `epistat_core::Codes` gives
    /// person ids. A unit has the same key in every estimator and in every
    /// cluster table built from the same codes.
    Units,
    /// Row positions in the input of the estimator, for estimators whose data
    /// carry no unit column. A row key means nothing outside that input.
    Rows,
}

impl fmt::Display for KeyedBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Units => "unit codes",
            Self::Rows => "rows",
        })
    }
}

/// The influence function of one estimate or several, per unit, on the scale
/// `theta_hat - theta ~ sum(psi_i) / n`.
///
/// Column `j` holds the values of estimate `j` over the `n` keys that
/// [`Influence::keys`] gives. The keys are the caller's unit codes
/// ([`Influence::new`]) or, for an estimator whose data have no unit column,
/// the rows of its input ([`Influence::over_rows`]); [`Influence::keyed_by`]
/// says which. Clusters looked up by key must be keyed the same way, so unit
/// clusters are never matched to row positions. The rule is the one the
/// estimator's own standard errors used, so [`Influence::variance`]
/// reproduces them.
#[derive(Clone, Debug, PartialEq)]
pub struct Influence {
    values: Vec<f64>,
    keys: Arc<[u32]>,
    keyed_by: KeyedBy,
    columns: usize,
    rule: ClusterRule,
}

impl Influence {
    /// The influence function with `columns` columns of `units.len()` values
    /// each, column after column, on the per-unit scale, keyed by the
    /// caller's unit codes `units`.
    ///
    /// Units in increasing code are checked in one pass; units in another
    /// order are accepted and checked through a hash set.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for no units or no columns,
    /// [`InputError::Length`] unless there are `columns * units.len()` values,
    /// [`InputError::NotFinite`] for a value that is not finite, and
    /// [`InputError::Duplicate`] for a unit code that repeats.
    pub fn new(
        values: Vec<f64>,
        units: Vec<u32>,
        columns: usize,
        rule: ClusterRule,
    ) -> Result<Self, Error> {
        Self::build(values, units, KeyedBy::Units, columns, rule)
    }

    /// [`Influence::new`] keyed by `rows`, positions in the estimator's input,
    /// for an estimator whose data have no unit column. Its variance takes
    /// positional clusters or clusters from [`Clusters::per_row`], and refuses
    /// clusters keyed by unit codes.
    ///
    /// # Errors
    ///
    /// As [`Influence::new`].
    pub fn over_rows(
        values: Vec<f64>,
        rows: Vec<u32>,
        columns: usize,
        rule: ClusterRule,
    ) -> Result<Self, Error> {
        Self::build(values, rows, KeyedBy::Rows, columns, rule)
    }

    /// The influence function from values on the dfbeta scale, whose squares
    /// sum to the variance, multiplied by the number of units.
    ///
    /// # Errors
    ///
    /// As [`Influence::new`].
    pub fn from_dfbeta(
        values: Vec<f64>,
        units: Vec<u32>,
        columns: usize,
        rule: ClusterRule,
    ) -> Result<Self, Error> {
        let scaled = scale_dfbeta(values, units.len());
        Self::new(scaled, units, columns, rule)
    }

    /// [`Influence::from_dfbeta`] keyed by rows, as [`Influence::over_rows`].
    ///
    /// # Errors
    ///
    /// As [`Influence::new`].
    pub fn from_dfbeta_over_rows(
        values: Vec<f64>,
        rows: Vec<u32>,
        columns: usize,
        rule: ClusterRule,
    ) -> Result<Self, Error> {
        let scaled = scale_dfbeta(values, rows.len());
        Self::over_rows(scaled, rows, columns, rule)
    }

    fn build(
        values: Vec<f64>,
        keys: Vec<u32>,
        keyed_by: KeyedBy,
        columns: usize,
        rule: ClusterRule,
    ) -> Result<Self, Error> {
        let what = match keyed_by {
            KeyedBy::Units => "influence units",
            KeyedBy::Rows => "influence rows",
        };
        check::non_empty(what, keys.len())?;
        check_values(&values, keys.len(), columns)?;
        distinct(what, &keys)?;
        Ok(Self {
            values,
            keys: keys.into(),
            keyed_by,
            columns,
            rule,
        })
    }

    /// The values of estimate `j`, one per key.
    ///
    /// # Panics
    ///
    /// If `j` is not below [`Influence::n_columns`].
    #[must_use]
    pub fn column(&self, j: usize) -> &[f64] {
        let n = self.keys.len();
        &self.values[j * n..(j + 1) * n]
    }

    /// What the keys are: unit codes or rows.
    #[must_use]
    pub const fn keyed_by(&self) -> KeyedBy {
        self.keyed_by
    }

    /// The key of each value, a unit code or a row as [`Influence::keyed_by`]
    /// says.
    #[must_use]
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }

    /// The unit code of each value, `None` for an influence function keyed by
    /// rows.
    #[must_use]
    pub fn units(&self) -> Option<&[u32]> {
        (self.keyed_by == KeyedBy::Units).then_some(&*self.keys)
    }

    /// The row of each value, `None` for an influence function keyed by unit
    /// codes.
    #[must_use]
    pub fn rows(&self) -> Option<&[u32]> {
        (self.keyed_by == KeyedBy::Rows).then_some(&*self.keys)
    }

    /// The number of keys, the length of each column.
    #[must_use]
    pub fn n_units(&self) -> usize {
        self.keys.len()
    }

    /// The number of estimates.
    #[must_use]
    pub const fn n_columns(&self) -> usize {
        self.columns
    }

    /// The rule the estimator's standard errors used.
    #[must_use]
    pub const fn rule(&self) -> ClusterRule {
        self.rule
    }

    /// The influence function of `sum_j weights[j] theta_j`, one column, with
    /// the keys and rule of this one. Each value is the sum of
    /// `weights[j] * psi_ij` in increasing `j`.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] unless there is one weight per column, and
    /// [`InputError::NotFinite`] for a weight, or a combined value, that is
    /// not finite.
    pub fn combine(&self, weights: &[f64]) -> Result<Self, Error> {
        check::same_length("weights", self.columns, weights.len())?;
        check::finite("weights", weights)?;
        let n = self.keys.len();
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let mut terms = weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * self.values[j * n + i]);
                let first = terms.next().unwrap_or(0.0);
                terms.fold(first, |acc, term| acc + term)
            })
            .collect();
        self.with_values(values, 1, self.rule)
    }

    /// The influence function of the estimates `columns`, in that order, with
    /// the keys and rule of this one.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for no columns and [`InputError::NotBelow`] for
    /// a column that is not below [`Influence::n_columns`].
    pub fn select(&self, columns: &[usize]) -> Result<Self, Error> {
        check::non_empty("influence columns", columns.len())?;
        let mut values = Vec::with_capacity(columns.len() * self.keys.len());
        for &j in columns {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the column is reported, not computed with"
            )]
            check::below("influence column", j as f64, self.columns as f64)?;
            values.extend_from_slice(self.column(j));
        }
        Ok(Self {
            values,
            keys: Arc::clone(&self.keys),
            keyed_by: self.keyed_by,
            columns: columns.len(),
            rule: self.rule,
        })
    }

    /// An influence function of `columns` other estimates over the keys of
    /// this one, under `rule`, with the values laid out as in
    /// [`Influence::new`]. The keys are not checked again.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for no columns, [`InputError::Length`] unless
    /// there are `columns` values per key, and [`InputError::NotFinite`] for a
    /// value that is not finite.
    pub fn with_values(
        &self,
        values: Vec<f64>,
        columns: usize,
        rule: ClusterRule,
    ) -> Result<Self, Error> {
        check_values(&values, self.keys.len(), columns)?;
        Ok(Self {
            values,
            keys: Arc::clone(&self.keys),
            keyed_by: self.keyed_by,
            columns,
            rule,
        })
    }

    /// The covariance of the estimates under the influence function's own
    /// rule, clustered by `clusters` or with every unit its own cluster.
    ///
    /// Clusters from [`Clusters::per_unit`] or [`Clusters::per_row`] are
    /// looked up by key, so they may cover more keys than this influence
    /// function, and they must be keyed as it is. Other clusters are matched
    /// by position and must cover exactly its keys. See the module for the
    /// formula.
    ///
    /// # Errors
    ///
    /// [`Error::KeyMismatch`] for clusters keyed by unit codes on an influence
    /// function keyed by rows, or the other way round,
    /// [`InputError::Length`] for positional clusters of another length,
    /// [`InputError::Domain`] for a key the clusters do not cover, and
    /// [`Error::NotFinite`] for an entry that is not finite.
    pub fn variance(&self, clusters: Option<&Clusters>) -> Result<Covariance, Error> {
        self.variance_with(clusters, self.rule)
    }

    /// [`Influence::variance`] under `rule` instead of the influence
    /// function's own.
    ///
    /// # Errors
    ///
    /// As [`Influence::variance`].
    pub fn variance_with(
        &self,
        clusters: Option<&Clusters>,
        rule: ClusterRule,
    ) -> Result<Covariance, Error> {
        let n = self.keys.len();
        let k = self.columns;
        let aligned;
        let clusters = match clusters {
            None => None,
            Some(clusters) => match (clusters.keyed_by(), clusters.keys()) {
                (Some(keyed_by), _) if keyed_by != self.keyed_by => {
                    return Err(Error::KeyMismatch {
                        influence: self.keyed_by,
                        clusters: keyed_by,
                    });
                }
                (_, Some(keys)) if keys != &*self.keys => {
                    aligned = clusters.for_keys(&self.keys)?;
                    Some(&aligned)
                }
                _ => {
                    check_length("clusters", n, clusters.len())?;
                    Some(clusters)
                }
            },
        };
        let groups = clusters.map_or(n, Clusters::count);
        if groups < 2 {
            return Ok(Covariance::new(k, vec![None; k * k])?);
        }
        let sizes: Option<Vec<f64>> = clusters.map(|c| {
            let mut sizes = vec![0.0; c.count()];
            for &g in c.of() {
                sizes[g as usize] += 1.0;
            }
            sizes
        });
        let nf = n as f64;
        let centred: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                let column = self.column(j);
                let mean = lanes::sum(column) / nf;
                match (clusters, &sizes) {
                    (Some(c), Some(sizes)) => c
                        .sums(column.iter().copied())
                        .iter()
                        .zip(sizes)
                        .map(|(s, size)| s - size * mean)
                        .collect(),
                    _ => column.iter().map(|v| v - mean).collect(),
                }
            })
            .collect();
        let g = groups as f64;
        let factor = match rule {
            ClusterRule::SmallSample => g / (g - 1.0),
            ClusterRule::NoSmallSample => 1.0,
        };
        let mut values = vec![None; k * k];
        for a in 0..k {
            for b in a..k {
                let value = factor * lanes::dot(&centred[a], &centred[b]) / (nf * nf);
                if !value.is_finite() {
                    return Err(Error::NotFinite);
                }
                values[a * k + b] = Some(value);
                values[b * k + a] = Some(value);
            }
        }
        Ok(Covariance::new(k, values)?)
    }
}

/// Values on the dfbeta scale multiplied by the number of keys `n`.
fn scale_dfbeta(values: Vec<f64>, n: usize) -> Vec<f64> {
    let n = n as f64;
    values.into_iter().map(|d| d * n).collect()
}

/// `columns` columns of `n` finite values.
fn check_values(values: &[f64], n: usize, columns: usize) -> Result<(), Error> {
    check::non_empty("influence columns", columns)?;
    let size = n.checked_mul(columns).ok_or(InputError::TooLarge {
        what: "influence function",
        limit: usize::MAX as u64,
        found: u64::MAX,
    })?;
    check::same_length("influence function", size, values.len())?;
    check::finite("influence function", values)?;
    Ok(())
}

/// No key repeats: one comparison per key when the keys increase, as every
/// estimator gives them, and a hash set otherwise.
fn distinct(what: &'static str, keys: &[u32]) -> Result<(), InputError> {
    if keys.windows(2).all(|pair| pair[0] < pair[1]) {
        return Ok(());
    }
    let mut seen = HashSet::with_capacity(keys.len());
    match keys.iter().position(|key| !seen.insert(*key)) {
        Some(row) => Err(InputError::Duplicate { what, row }),
        None => Ok(()),
    }
}

/// `sqrt(mean((psi - mean(psi))^2)) / sqrt(n)`.
///
/// # Errors
///
/// [`InputError::Empty`] for no values.
#[deprecated(
    since = "0.4.0",
    note = "use Influence::variance with ClusterRule::NoSmallSample"
)]
pub fn standard_error(psi: &[f64]) -> Result<f64, Error> {
    if psi.is_empty() {
        return Err(empty("influence function"));
    }
    let n = psi.len() as f64;
    let mean = lanes::sum(psi) / n;
    let mean_square = lanes::centred_squares(psi, mean) / n;
    Ok(mean_square.sqrt() / n.sqrt())
}

/// The cluster-robust variance `G / (G - 1) * sum_g (sum_{i in g} psi_i)^2 / n^2`,
/// without centring. It is `None` with fewer than two clusters.
///
/// Cluster sums are formed in observation order and added in cluster order.
///
/// # Errors
///
/// [`InputError::Empty`] for no values, [`InputError::Length`] when
/// `clusters` covers another number of observations, and
/// [`Error::NotFinite`].
#[deprecated(since = "0.4.0", note = "use Influence::variance, which centres")]
pub fn clustered_variance(psi: &[f64], clusters: &Clusters) -> Result<Option<f64>, Error> {
    if psi.is_empty() {
        return Err(empty("influence function"));
    }
    check_length("clusters", psi.len(), clusters.len())?;
    if clusters.count() < 2 {
        return Ok(None);
    }
    let n = psi.len() as f64;
    let g = clusters.count() as f64;
    let squares = lanes::sum_squares(&clusters.sums(psi.iter().copied()));
    let variance = g / (g - 1.0) * squares / (n * n);
    if variance.is_finite() {
        Ok(Some(variance.max(0.0)))
    } else {
        Err(Error::NotFinite)
    }
}

/// The square root of [`clustered_variance`], `None` where that is.
///
/// # Errors
///
/// As [`clustered_variance`].
#[deprecated(since = "0.4.0", note = "use Influence::variance, which centres")]
pub fn clustered_standard_error(psi: &[f64], clusters: &Clusters) -> Result<Option<f64>, Error> {
    #[expect(deprecated, reason = "the wrapper of a deprecated function")]
    clustered_variance(psi, clusters).map(|variance| variance.map(f64::sqrt))
}

/// The covariance matrix of several estimators from their influence
/// functions, `columns[k][i]` for estimator `k` and observation `i`, clustered
/// as in [`clustered_variance`]. Returned row-major, `k x k`, or `None` with
/// fewer than two clusters.
///
/// # Errors
///
/// As [`clustered_variance`], and [`InputError::Length`] when the columns
/// differ in length.
#[deprecated(since = "0.4.0", note = "use Influence::variance, which centres")]
pub fn clustered_covariance(
    columns: &[&[f64]],
    clusters: &Clusters,
) -> Result<Option<Vec<f64>>, Error> {
    let n = clusters.len();
    if n == 0 || columns.is_empty() {
        return Err(empty("influence function"));
    }
    for column in columns {
        check_length("influence column", n, column.len())?;
    }
    if clusters.count() < 2 {
        return Ok(None);
    }
    let k = columns.len();
    let (nf, g) = (n as f64, clusters.count() as f64);
    let sums: Vec<Vec<f64>> = columns
        .iter()
        .map(|c| clusters.sums(c.iter().copied()))
        .collect();
    let scale = g / (g - 1.0);
    let mut covariance = vec![0.0; k * k];
    for a in 0..k {
        for b in a..k {
            let cross = lanes::dot(&sums[a], &sums[b]);
            let value = scale * cross / (nf * nf);
            if !value.is_finite() {
                return Err(Error::NotFinite);
            }
            covariance[a * k + b] = value;
            covariance[b * k + a] = value;
        }
    }
    Ok(Some(covariance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::Classify;
    use epistat_testkit::Tolerance;

    fn one(values: &[f64], rule: ClusterRule) -> Influence {
        let units = (0..u32::try_from(values.len()).unwrap()).collect();
        Influence::new(values.to_vec(), units, 1, rule).unwrap()
    }

    /// A relative tolerance for sums of a handful of exactly representable
    /// terms: each operation rounds once, by at most half an ulp, so a few
    /// ulps bound the difference between two orders of evaluation.
    const CLOSE: Tolerance = Tolerance::exact().rel(16.0 * f64::EPSILON);

    #[test]
    fn singleton_clusters_give_the_unclustered_variance() {
        let psi = [1.0, -2.0, 0.5, 0.5, 3.0];
        for rule in [ClusterRule::SmallSample, ClusterRule::NoSmallSample] {
            let influence = one(&psi, rule);
            let plain = influence.variance(None).unwrap();
            let singletons = influence
                .variance(Some(&Clusters::singletons(5).unwrap()))
                .unwrap();
            assert_eq!(plain, singletons);
        }
    }

    #[test]
    fn the_rules_are_the_two_divisors_of_the_sample_variance() {
        let psi = [1.0, -2.0, 0.5, 0.5, 3.0];
        let n = 5.0;
        let mean = psi.iter().sum::<f64>() / n;
        let ss: f64 = psi.iter().map(|v| (v - mean) * (v - mean)).sum();
        let small = one(&psi, ClusterRule::SmallSample).variance(None).unwrap();
        let none = one(&psi, ClusterRule::NoSmallSample)
            .variance(None)
            .unwrap();
        assert!(CLOSE.accepts(small.get(0, 0).unwrap(), ss / (n - 1.0) / n));
        assert!(CLOSE.accepts(none.get(0, 0).unwrap(), ss / n / n));
        let rule = ClusterRule::SmallSample;
        assert!(
            CLOSE.accepts(
                one(&psi, rule)
                    .variance_with(None, ClusterRule::NoSmallSample)
                    .unwrap()
                    .get(0, 0)
                    .unwrap(),
                ss / n / n
            )
        );
    }

    #[test]
    fn clusters_sum_centred_values_in_code_order() {
        let psi = [1.0, -2.0, 0.5, 0.5, 3.0];
        let clusters = Clusters::from_codes(&[7, 7, 2, 9, 9]);
        let v = one(&psi, ClusterRule::SmallSample)
            .variance(Some(&clusters))
            .unwrap()
            .get(0, 0)
            .unwrap();
        let mean = 3.0 / 5.0;
        let d = [0.5 - mean, 3.5 - 2.0 * mean, -1.0 - 2.0 * mean];
        let want = 1.5 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / 25.0;
        assert!(CLOSE.accepts(v, want));
    }

    #[test]
    fn per_unit_clusters_are_looked_up_by_unit() {
        let values = vec![1.0, -2.0, 0.5, 3.0];
        let influence =
            Influence::new(values, vec![9, 4, 1, 2], 1, ClusterRule::SmallSample).unwrap();
        // Unit 5 is in the data but not kept by the estimator.
        let clusters = Clusters::per_unit(&[1, 2, 4, 5, 9], &[0, 0, 1, 1, 2]).unwrap();
        let by_unit = influence.variance(Some(&clusters)).unwrap();
        let by_position = influence
            .variance(Some(&Clusters::from_codes(&[2, 1, 0, 0])))
            .unwrap();
        assert_eq!(by_unit, by_position);
    }

    #[test]
    fn unit_clusters_on_an_influence_keyed_by_rows_are_refused() {
        // Rows 0..4 of an estimator's input whose persons have the unit codes
        // 3, 0, 2, 1. Every row number is also a unit code, so a lookup by
        // code would succeed and pair each row with another person's cluster.
        let values = vec![1.0, -2.0, 0.5, 3.0];
        let rows = Influence::over_rows(
            values.clone(),
            vec![0, 1, 2, 3],
            1,
            ClusterRule::SmallSample,
        )
        .unwrap();
        assert_eq!(rows.keyed_by(), KeyedBy::Rows);
        assert_eq!((rows.units(), rows.rows()), (None, Some(&[0, 1, 2, 3][..])));
        let person = [3, 0, 2, 1];
        let family = [0, 0, 1, 1];
        let by_person = Clusters::per_unit(&person, &family).unwrap();
        assert_eq!(
            rows.variance(Some(&by_person)),
            Err(Error::KeyMismatch {
                influence: KeyedBy::Rows,
                clusters: KeyedBy::Units
            })
        );
        // The same clusters given per row, or by position, are accepted and
        // agree.
        let family_by_row: Vec<u32> = person.iter().map(|&p| family[p as usize]).collect();
        let per_row = rows
            .variance(Some(&Clusters::per_row(&family_by_row).unwrap()))
            .unwrap();
        let positional = rows
            .variance(Some(&Clusters::from_codes(&family_by_row)))
            .unwrap();
        assert_eq!(per_row, positional);
        // And row clusters are refused on an influence function keyed by
        // units.
        let units = Influence::new(values, vec![0, 1, 2, 3], 1, ClusterRule::SmallSample).unwrap();
        let error = units
            .variance(Some(&Clusters::per_row(&family_by_row).unwrap()))
            .unwrap_err();
        assert_eq!(error.code(), "inference.key_mismatch");
        assert_eq!(error.kind(), epistat_core::ErrorKind::Input);
    }

    #[test]
    fn row_clusters_are_looked_up_by_row() {
        let rows = Influence::over_rows(
            vec![1.0, -2.0, 0.5],
            vec![1, 3, 4],
            1,
            ClusterRule::SmallSample,
        )
        .unwrap();
        let clusters = Clusters::per_row(&[5, 5, 6, 6, 7]).unwrap();
        let by_row = rows.variance(Some(&clusters)).unwrap();
        let by_position = rows
            .variance(Some(&Clusters::from_codes(&[5, 6, 7])))
            .unwrap();
        assert_eq!(by_row, by_position);
    }

    #[test]
    fn units_in_any_order_are_checked_for_repeats() {
        let rule = ClusterRule::NoSmallSample;
        assert!(Influence::new(vec![1.0, 2.0, 3.0], vec![5, 1, 3], 1, rule).is_ok());
        assert_eq!(
            Influence::new(vec![1.0, 2.0, 3.0], vec![5, 1, 5], 1, rule),
            Err(Error::Input(InputError::Duplicate {
                what: "influence units",
                row: 2
            }))
        );
        assert_eq!(
            Influence::over_rows(vec![1.0, 2.0], vec![4, 4], 1, rule),
            Err(Error::Input(InputError::Duplicate {
                what: "influence rows",
                row: 1
            }))
        );
    }

    #[test]
    fn selected_and_replaced_columns_keep_the_keys() {
        let a = [1.0, -2.0, 0.5];
        let b = [0.0, 1.0, -1.0];
        let c = [2.0, 0.25, 4.0];
        let all = Influence::over_rows(
            [a, b, c].concat(),
            vec![2, 5, 9],
            3,
            ClusterRule::NoSmallSample,
        )
        .unwrap();
        let picked = all.select(&[2, 0]).unwrap();
        let direct = Influence::over_rows(
            [c, a].concat(),
            vec![2, 5, 9],
            2,
            ClusterRule::NoSmallSample,
        )
        .unwrap();
        assert_eq!(picked, direct);
        assert_eq!(picked.variance(None), direct.variance(None));
        assert!(matches!(
            all.select(&[3]),
            Err(Error::Input(InputError::NotBelow { .. }))
        ));
        assert!(matches!(
            all.select(&[]),
            Err(Error::Input(InputError::Empty { .. }))
        ));
        let other = all
            .with_values(b.to_vec(), 1, ClusterRule::SmallSample)
            .unwrap();
        assert_eq!(
            (other.keyed_by(), other.keys()),
            (KeyedBy::Rows, &[2, 5, 9][..])
        );
        assert_eq!(other.rule(), ClusterRule::SmallSample);
        assert!(matches!(
            all.with_values(vec![1.0, f64::NAN, 0.0], 1, ClusterRule::SmallSample),
            Err(Error::Input(InputError::NotFinite { row: 1, .. }))
        ));
    }

    #[test]
    fn the_covariance_diagonal_is_the_variance_of_each_column() {
        let a = [1.0, -2.0, 0.5, 0.5, 3.0];
        let b = [0.0, 1.0, -1.0, 2.0, 0.5];
        let both = Influence::new(
            [a, b].concat(),
            (0..5).collect(),
            2,
            ClusterRule::SmallSample,
        )
        .unwrap();
        let clusters = Clusters::from_codes(&[1, 1, 2, 3, 3]);
        let covariance = both.variance(Some(&clusters)).unwrap();
        for (j, column) in [a, b].iter().enumerate() {
            let alone = one(column, ClusterRule::SmallSample)
                .variance(Some(&clusters))
                .unwrap();
            assert_eq!(
                covariance.get(j, j).map(f64::to_bits),
                alone.get(0, 0).map(f64::to_bits)
            );
        }
        assert_eq!(covariance.get(0, 1), covariance.get(1, 0));
    }

    #[test]
    fn a_combination_has_the_variance_of_the_quadratic_form() {
        let a = [1.0, -2.0, 0.5, 0.5, 3.0];
        let b = [0.0, 1.0, -1.0, 2.0, 0.5];
        let both = Influence::new(
            [a, b].concat(),
            (0..5).collect(),
            2,
            ClusterRule::SmallSample,
        )
        .unwrap();
        let w = [0.5, -1.5];
        let combined = both.combine(&w).unwrap();
        assert_eq!(combined.n_columns(), 1);
        assert_eq!(combined.rule(), ClusterRule::SmallSample);
        assert_eq!(
            combined.column(0)[1].to_bits(),
            (0.5 * -2.0 + -1.5 * 1.0_f64).to_bits()
        );
        let direct = combined.variance(None).unwrap().get(0, 0).unwrap();
        let quadratic = both.variance(None).unwrap().quadratic(&w).unwrap();
        assert!(CLOSE.accepts(direct, quadratic));
    }

    #[test]
    fn dfbeta_values_are_multiplied_by_n() {
        let influence =
            Influence::from_dfbeta(vec![0.25, -0.5], vec![0, 1], 1, ClusterRule::NoSmallSample)
                .unwrap();
        assert_eq!(influence.column(0), [0.5, -1.0]);
    }

    #[test]
    fn fewer_than_two_clusters_leave_the_covariance_undefined() {
        let influence = one(&[1.0, -1.0], ClusterRule::NoSmallSample);
        let covariance = influence
            .variance(Some(&Clusters::from_codes(&[3, 3])))
            .unwrap();
        assert_eq!(covariance.get(0, 0), None);
        let single = one(&[1.0], ClusterRule::SmallSample);
        assert_eq!(single.variance(None).unwrap().get(0, 0), None);
    }

    #[test]
    fn invalid_input_is_an_input_error() {
        let rule = ClusterRule::SmallSample;
        assert!(matches!(
            Influence::new(vec![], vec![], 1, rule),
            Err(Error::Input(InputError::Empty { .. }))
        ));
        assert!(matches!(
            Influence::new(vec![1.0], vec![0, 1], 1, rule),
            Err(Error::Input(InputError::Length { .. }))
        ));
        assert!(matches!(
            Influence::new(vec![1.0, f64::NAN], vec![0, 1], 1, rule),
            Err(Error::Input(InputError::NotFinite { row: 1, .. }))
        ));
        assert!(matches!(
            Influence::new(vec![1.0, 2.0], vec![3, 3], 1, rule),
            Err(Error::Input(InputError::Duplicate { row: 1, .. }))
        ));
        let influence = one(&[1.0, 2.0], rule);
        assert!(matches!(
            influence.variance(Some(&Clusters::singletons(3).unwrap())),
            Err(Error::Input(InputError::Length { .. }))
        ));
        assert!(matches!(
            influence.combine(&[1.0, 2.0]),
            Err(Error::Input(InputError::Length { .. }))
        ));
    }

    #[test]
    #[expect(deprecated, reason = "the 0.3 functions keep their bits")]
    fn the_earlier_functions_keep_their_rules() {
        let psi = [1.0, -2.0, 0.5, 0.5];
        let variance = clustered_variance(&psi, &Clusters::singletons(4).unwrap())
            .unwrap()
            .unwrap();
        assert!((variance - 4.0 / 3.0 * 5.5 / 16.0).abs() < 1e-15);
        let one_cluster = Clusters::from_labels_first_seen(&[7, 7]).unwrap();
        assert_eq!(clustered_variance(&psi[..2], &one_cluster), Ok(None));
        assert_eq!(clustered_standard_error(&psi[..2], &one_cluster), Ok(None));
        assert_eq!(
            clustered_covariance(&[&psi[..2], &psi[..2]], &one_cluster),
            Ok(None)
        );
        let a = [1.0, -2.0, 0.5, 0.5, 3.0];
        let b = [0.0, 1.0, -1.0, 2.0, 0.5];
        let clusters = Clusters::from_labels_first_seen(&[1, 1, 2, 3, 3]).unwrap();
        let covariance = clustered_covariance(&[&a, &b], &clusters).unwrap().unwrap();
        assert_eq!(
            covariance[0].to_bits(),
            clustered_variance(&a, &clusters)
                .unwrap()
                .unwrap()
                .to_bits()
        );
        assert!(matches!(
            standard_error(&[]),
            Err(Error::Input(InputError::Empty { .. }))
        ));
        assert!(matches!(
            clustered_variance(&[1.0], &Clusters::singletons(2).unwrap()),
            Err(Error::Input(InputError::Length { .. }))
        ));
    }
}
