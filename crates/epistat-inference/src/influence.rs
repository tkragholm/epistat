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
//! `n - 1` divisor, as `riskRegression` reports, and `NoSmallSample` gives
//! the `n` divisor, as R's `did` and `survival`'s robust variances do. An
//! influence function evaluated at the estimate has mean zero, so centring
//! changes nothing there but the rounding.
//!
//! [`Influence::combine`] forms the influence function of a linear
//! combination of the estimates, `psi_i = sum_j w_j psi_ij`, the influence
//! function of `sum_j w_j theta_j`.
//!
//! # Determinism
//!
//! Cluster sums are formed in row order and added in cluster order, which is
//! increasing cluster code for clusters from [`Clusters::from_codes`] and
//! [`Clusters::per_unit`]. The covariance computes the variance of column `a`
//! with the same sums and the same products in the same order, so its diagonal
//! entry equals the variance of that column alone to the bit. Nothing depends
//! on the number of threads.
//!
//! # Edge cases
//!
//! - No units, no columns, values of the wrong count, a value that is not
//!   finite and a unit code that repeats are [`InputError`]s when the
//!   [`Influence`] is built.
//! - Clusters that cover another number of units, or per-unit clusters that
//!   do not cover a unit, are [`InputError`]s.
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

use epistat_core::{Covariance, InputError, check};
use epistat_stats::lanes;

use crate::{Clusters, Error, check_length, empty};

/// The small-sample factor of a cluster-robust variance from an influence
/// function.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ClusterRule {
    /// `G / (G - 1)` for `G` clusters, R's `sandwich::vcovCL` with
    /// `cadjust = TRUE` and, without clusters, `riskRegression`'s `n - 1`
    /// divisor.
    #[default]
    SmallSample,
    /// No factor, for ports whose R reference has none: R's `did` and the
    /// robust variances of R's `survival`.
    NoSmallSample,
}

/// The influence function of one estimate or several, per unit, on the scale
/// `theta_hat - theta ~ sum(psi_i) / n`.
///
/// Column `j` holds the values of estimate `j` over the `n` units, whose codes
/// [`Influence::units`] gives. The rule is the one the estimator's own
/// standard errors used, so [`Influence::variance`] reproduces them.
#[derive(Clone, Debug, PartialEq)]
pub struct Influence {
    values: Vec<f64>,
    units: Vec<u32>,
    columns: usize,
    rule: ClusterRule,
}

impl Influence {
    /// The influence function with `columns` columns of `units.len()` values
    /// each, column after column, on the per-unit scale.
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
        check::non_empty("influence units", units.len())?;
        check::non_empty("influence columns", columns)?;
        let size = units
            .len()
            .checked_mul(columns)
            .ok_or(InputError::TooLarge {
                what: "influence function",
                limit: usize::MAX as u64,
                found: u64::MAX,
            })?;
        check::same_length("influence function", size, values.len())?;
        check::finite("influence function", &values)?;
        let mut seen = HashSet::with_capacity(units.len());
        if let Some(row) = units.iter().position(|unit| !seen.insert(*unit)) {
            return Err(InputError::Duplicate {
                what: "influence units",
                row,
            }
            .into());
        }
        Ok(Self {
            values,
            units,
            columns,
            rule,
        })
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
        let n = units.len() as f64;
        let scaled = values.into_iter().map(|d| d * n).collect();
        Self::new(scaled, units, columns, rule)
    }

    /// The values of estimate `j`, one per unit.
    ///
    /// # Panics
    ///
    /// If `j` is not below [`Influence::n_columns`].
    #[must_use]
    pub fn column(&self, j: usize) -> &[f64] {
        let n = self.units.len();
        &self.values[j * n..(j + 1) * n]
    }

    /// The unit code of each row.
    #[must_use]
    pub fn units(&self) -> &[u32] {
        &self.units
    }

    /// The number of units.
    #[must_use]
    pub fn n_units(&self) -> usize {
        self.units.len()
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
    /// the units and rule of this one. Each value is the sum of
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
        let n = self.units.len();
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
        Self::new(values, self.units.clone(), 1, self.rule)
    }

    /// The covariance of the estimates under the influence function's own
    /// rule, clustered by `clusters` or with every unit its own cluster.
    ///
    /// Clusters from [`Clusters::per_unit`] are looked up by unit code, so
    /// they may cover more units than this influence function; other clusters
    /// are matched by position and must cover exactly its units. See the
    /// module for the formula.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] for positional clusters of another length,
    /// [`InputError::Domain`] for a unit per-unit clusters do not cover, and
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
        let n = self.units.len();
        let k = self.columns;
        let aligned;
        let clusters = match clusters {
            None => None,
            Some(clusters) if clusters.units().is_none_or(|u| u == self.units.as_slice()) => {
                check_length("clusters", n, clusters.len())?;
                Some(clusters)
            }
            Some(clusters) => {
                aligned = clusters.for_units(&self.units)?;
                Some(&aligned)
            }
        };
        let groups = clusters.map_or(n, Clusters::count);
        if groups < 2 {
            return Ok(Covariance::new(k, vec![None; k * k])?);
        }
        let sizes: Option<Vec<f64>> = clusters.map(|c| {
            let mut sizes = vec![0.0; c.count()];
            for &g in c.of() {
                sizes[g] += 1.0;
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

    fn one(values: &[f64], rule: ClusterRule) -> Influence {
        let units = (0..u32::try_from(values.len()).unwrap()).collect();
        Influence::new(values.to_vec(), units, 1, rule).unwrap()
    }

    /// A relative tolerance for sums of a handful of exactly representable
    /// terms: each operation rounds once, by at most half an ulp, so a few
    /// ulps bound the difference between two orders of evaluation.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 16.0 * f64::EPSILON * b.abs()
    }

    #[test]
    fn singleton_clusters_give_the_unclustered_variance() {
        let psi = [1.0, -2.0, 0.5, 0.5, 3.0];
        for rule in [ClusterRule::SmallSample, ClusterRule::NoSmallSample] {
            let influence = one(&psi, rule);
            let plain = influence.variance(None).unwrap();
            let singletons = influence.variance(Some(&Clusters::singletons(5))).unwrap();
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
        assert!(close(small.get(0, 0).unwrap(), ss / (n - 1.0) / n));
        assert!(close(none.get(0, 0).unwrap(), ss / n / n));
        let rule = ClusterRule::SmallSample;
        assert!(close(
            one(&psi, rule)
                .variance_with(None, ClusterRule::NoSmallSample)
                .unwrap()
                .get(0, 0)
                .unwrap(),
            ss / n / n
        ));
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
        assert!(close(v, want));
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
        assert!(close(direct, quadratic));
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
            influence.variance(Some(&Clusters::singletons(3))),
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
        let variance = clustered_variance(&psi, &Clusters::singletons(4))
            .unwrap()
            .unwrap();
        assert!((variance - 4.0 / 3.0 * 5.5 / 16.0).abs() < 1e-15);
        let one_cluster = Clusters::from_labels(&[7, 7]);
        assert_eq!(clustered_variance(&psi[..2], &one_cluster), Ok(None));
        assert_eq!(clustered_standard_error(&psi[..2], &one_cluster), Ok(None));
        assert_eq!(
            clustered_covariance(&[&psi[..2], &psi[..2]], &one_cluster),
            Ok(None)
        );
        let a = [1.0, -2.0, 0.5, 0.5, 3.0];
        let b = [0.0, 1.0, -1.0, 2.0, 0.5];
        let clusters = Clusters::from_labels(&[1, 1, 2, 3, 3]);
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
            clustered_variance(&[1.0], &Clusters::singletons(2)),
            Err(Error::Input(InputError::Length { .. }))
        ));
    }
}
