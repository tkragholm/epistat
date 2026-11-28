//! Critical values for simultaneous bands: the sup-t method.
//!
//! A band that covers `k` estimates at once, with probability `1 - alpha`
//! for a [`Level`] of confidence `1 - alpha`,
//! is `estimate_j +- c se_j` for a `c` larger than the pointwise normal
//! critical value. The sup-t band takes `c` as the `1 - alpha` quantile of the
//! maximum absolute standardised coordinate, `max_j |T_j|`, which keeps the
//! correlation between the estimates \[1\]. Two functions give `c`.
//!
//! # From a covariance matrix
//!
//! [`sup_t_from_covariance`] takes the covariance matrix `V` of a normally
//! distributed estimator. It standardises `V` to the correlation matrix `R`,
//! `R_ij = V_ij / (sd_i sd_j)` with entries clamped to `[-1, 1]` and a unit
//! diagonal, factors `R = L L'` with a lower Cholesky factor, and draws
//!
//! ```text
//! T = L z,      z ~ N(0, I),      c = quantile_(1 - alpha) of max_j |T_j|
//! ```
//!
//! over the replicates of a [`Bootstrap`]. The quantile is R's type 7, which is what R's
//! `quantile()` and the HonestDiD package return. A positive semidefinite
//! `R` is accepted: a pivot below `1e-12` leaves its column of `L` at zero.
//! The normal draws are pairs from the Box-Muller transform, taken from the
//! stream seeded with `replicate_seed(seed, r)` for replicate `r`, and
//! `L z` is summed in column order without fused multiply-add.
//!
//! # Multiplier bootstrap bands
//!
//! [`multiplier_bands`] is the band calculation of R's `did::mboot`, which
//! `att_gt` and `aggte` call for their uniform bands \[2\]. Its input is one
//! influence function per estimate, over `n` units, scaled so that
//! `se = sqrt(sum psi^2) / n`. With `G` clusters (`G = n` without clustering)
//! the influence functions are first summed within clusters. For each
//! replicate there is one Rademacher multiplier `s_g = +-1` per cluster, shared
//! by all estimates, and the draw for estimate `j` is
//!
//! ```text
//! b_j = sqrt(G) * (sum_g s_g psi_jg) / G
//! ```
//!
//! The multipliers are the bits of `SplitMix64` outputs, 64 per output, lowest
//! bit first, a set bit meaning `+1`, taken from the stream seeded with
//! `replicate_seed(seed, r)`. Clusters are numbered by the caller, and `did`
//! numbers them in increasing order of label ([`Clusters::from_sorted_labels`]).
//!
//! The scale of `b_j` is not the sample standard deviation. It is
//!
//! ```text
//! sigma_j = (b_j[ceil(0.75 B)] - b_j[ceil(0.25 B)]) / (qnorm(0.75) - qnorm(0.25))
//! ```
//!
//! from the `B` sorted draws, so a few extreme draws do not move it. The
//! maximum over estimates is `T_r = max_j |b_rj / sigma_j|`, and the critical
//! value is the `1 - alpha` quantile of the `T_r` by R's type 1, the inverse
//! empirical distribution function. The standard error of estimate `j` is
//! `sigma_j sqrt(G) / n`.
//!
//! As in `did`, an estimate whose draws contain a NaN, or whose sum of squared
//! draws is not above `10 sqrt(eps)`, takes no part in the maximum and has no
//! standard error. A replicate whose maximum is not finite is dropped before
//! the quantile. `did`'s callers raise a critical value below the pointwise
//! normal one, or at 7 and above, to the pointwise one, and
//! [`MultiplierBands::at_least_pointwise`] is the first of these.
//!
//! The quantile probability `1 - alpha` is the level's
//! [`Level::confidence_value`] and the pointwise one is its
//! [`Level::upper_probability`], so a level given as an alpha keeps the
//! arithmetic of `1 - alpha` and `1 - alpha / 2`.
//!
//! # Determinism
//!
//! Replicates run in parallel, each from its own seed and into its own slot,
//! and every sum runs in a fixed order. The result is the same, to the bit, on
//! any number of threads.
//!
//! # Agreement with R
//!
//! The tests apply `did::mboot`'s own post-processing to a given matrix of
//! draws and require the scale, the critical value and the standard errors to
//! match to the bit. R draws its multipliers from its own generator, so
//! complete runs on one influence matrix agree to Monte Carlo error, which the
//! tests derive from the draws. [`sup_t_from_covariance`] is compared with
//! `mvtnorm::qmvnorm`, which integrates the same probability numerically.
//!
//! # References
//!
//! 1. Montiel Olea JL, Plagborg-Moller M. Simultaneous confidence bands:
//!    theory, implementation, and an application to SVARs. J Appl Econom.
//!    2019;34(1):1-17.
//! 2. Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//!    periods. J Econom. 2021;225(2):200-30.

use epistat_core::{Bootstrap, Level};
use epistat_linalg::MatRef;
use epistat_linalg::cholesky::{Pivot, cholesky_lower};
use epistat_rng::{Stream, replicate_seed};
use epistat_stats::dist::normal;
use epistat_stats::quantile;
use epistat_stats::sum::lanes;
use rayon::prelude::*;

use crate::{Clusters, Error, check_length, empty};

/// A covariance pivot below this leaves its column of the Cholesky factor zero.
const PIVOT_FLOOR: f64 = 1e-12;
/// A diagonal sum below minus this is not positive semidefinite.
const NEGATIVE_FLOOR: f64 = 1e-10;

/// The sup-t critical value of a normal estimator with covariance
/// `covariance`: the quantile (type 7) at the confidence of `level` of the
/// maximum absolute standardised coordinate over the simulated replicates of
/// `draws`. Replicate `r` draws from `replicate_seed(seed, r)`.
///
/// For one coordinate the value approaches the pointwise `normal` critical
/// value as the number of replicates grows.
///
/// # Errors
///
/// [`InputError::Empty`](epistat_core::InputError::Empty) for a covariance
/// with no rows or no replicates,
/// [`InputError::Length`](epistat_core::InputError::Length) when it is not
/// square, and [`Error::NotFinite`] for a variance that is not positive and
/// finite, or a correlation matrix that is not positive semidefinite.
pub fn sup_t_from_covariance(
    covariance: MatRef<'_, f64>,
    level: Level,
    draws: Bootstrap,
) -> Result<f64, Error> {
    let Bootstrap {
        replicates: draws,
        seed,
    } = draws;
    let dim = covariance.nrows();
    check_length("covariance columns", dim, covariance.ncols())?;
    if dim == 0 {
        return Err(empty("covariance"));
    }
    if draws == 0 {
        return Err(empty("replicates"));
    }
    let lower = lower_cholesky(&correlation(covariance)?, dim)?;
    let mut maxima: Vec<f64> = (0..draws as u64)
        .into_par_iter()
        .map_init(
            || (vec![0.0; dim], vec![0.0; dim]),
            |(z, t), r| {
                let mut stream = Stream::seeded(replicate_seed(seed, r));
                stream.fill_normals(z);
                lower_times(&lower, dim, z, t);
                t.iter().map(|v| v.abs()).fold(0.0, f64::max)
            },
        )
        .collect();
    quantile::type7_unsorted(&mut maxima, level.confidence_value()).ok_or(Error::NotFinite)
}

/// The correlation matrix of the symmetrised covariance, row-major.
pub(crate) fn correlation(covariance: MatRef<'_, f64>) -> Result<Vec<f64>, Error> {
    let dim = covariance.nrows();
    let symmetric = |i: usize, j: usize| covariance[(i, j)].midpoint(covariance[(j, i)]);
    let sd: Vec<f64> = (0..dim).map(|i| symmetric(i, i).sqrt()).collect();
    if sd.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err(Error::NotFinite);
    }
    let mut corr = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            corr[i * dim + j] = if i == j {
                1.0
            } else {
                (symmetric(i, j) / (sd[i] * sd[j])).clamp(-1.0, 1.0)
            };
        }
    }
    Ok(corr)
}

/// The lower Cholesky factor of a symmetric positive semidefinite matrix,
/// row-major, with sums in column order.
pub(crate) fn lower_cholesky(matrix: &[f64], n: usize) -> Result<Vec<f64>, Error> {
    cholesky_lower(
        matrix,
        n,
        Pivot::Semidefinite {
            negative_floor: NEGATIVE_FLOOR,
            pivot_floor: PIVOT_FLOOR,
        },
    )
    .ok_or(Error::NotFinite)
}

/// `out = L z` for a row-major lower triangular `L`.
fn lower_times(lower: &[f64], dim: usize, z: &[f64], out: &mut [f64]) {
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = lanes::dot(&lower[i * dim..=i * dim + i], &z[..=i]);
    }
}

/// The result of [`multiplier_bands`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct MultiplierBands {
    /// The simultaneous critical value, a multiplier of the standard errors.
    pub critical: f64,
    /// The bootstrap standard error of each estimate, `None` for an estimate
    /// whose draws are degenerate.
    pub se: Vec<Option<f64>>,
    /// The level of the bands.
    pub level: Level,
}

impl MultiplierBands {
    /// The critical value, or the pointwise normal one at the bands' level if
    /// that is larger. A band can then not be narrower than the pointwise
    /// interval.
    #[must_use]
    pub fn at_least_pointwise(&self) -> f64 {
        self.critical
            .max(normal::quantile(self.level.upper_probability()))
    }
}

/// Simultaneous bands by the multiplier bootstrap, as `did::mboot` computes
/// them.
///
/// `influence` is `n x k`, one column per estimate over the same `n` units.
/// `clusters` assigns the units to clusters, one multiplier per cluster, or is
/// `None` for one per unit; the clusters are matched to the units by
/// position. Replicate `r` draws its multipliers from
/// `replicate_seed(seed, r)`. The band of estimate `j` is
/// `estimate_j +- critical * se_j`.
///
/// # Errors
///
/// [`InputError::Empty`](epistat_core::InputError::Empty) for no estimates,
/// no units or no replicates,
/// [`InputError::Length`](epistat_core::InputError::Length) when the clusters
/// are not one per unit,
/// [`InputError::NotFinite`](epistat_core::InputError::NotFinite) for an
/// influence value that is not finite, and [`Error::NotFinite`] when no
/// estimate gives a finite critical value.
pub fn multiplier_bands(
    influence: MatRef<'_, f64>,
    clusters: Option<&Clusters>,
    level: Level,
    bootstrap: Bootstrap,
) -> Result<MultiplierBands, Error> {
    let Bootstrap { replicates, seed } = bootstrap;
    if influence.ncols() == 0 {
        return Err(empty("influence columns"));
    }
    let n = influence.nrows();
    let columns: Vec<Vec<f64>> = (0..influence.ncols())
        .map(|j| influence.col(j).iter().copied().collect())
        .collect();
    for (j, column) in columns.iter().enumerate() {
        epistat_core::check::finite_column("influence", j, column)?;
    }
    if n == 0 {
        return Err(empty("influence column"));
    }
    if replicates == 0 {
        return Err(empty("replicates"));
    }
    let width = columns.len();
    let (sums, groups): (Vec<Vec<f64>>, usize) = match clusters {
        None => (columns, n),
        Some(clusters) => {
            check_length("clusters", n, clusters.len())?;
            (
                columns
                    .iter()
                    .map(|c| clusters.sums(c.iter().copied()))
                    .collect(),
                clusters.count(),
            )
        }
    };
    let groups_f = groups as f64;
    let mut draws = vec![0.0; replicates * width];
    draws.par_chunks_mut(width).enumerate().for_each_init(
        || vec![0.0; groups],
        |signs, (r, row)| {
            let mut stream = Stream::seeded(replicate_seed(seed, r as u64));
            for chunk in signs.chunks_mut(64) {
                let mut bits = stream.next_u64();
                for sign in chunk {
                    *sign = if bits & 1 == 1 { 1.0 } else { -1.0 };
                    bits >>= 1;
                }
            }
            for (slot, column) in row.iter_mut().zip(&sums) {
                let mean = lanes::dot(column, signs) / groups_f;
                *slot = groups_f.sqrt() * mean;
            }
        },
    );
    from_draws(&draws, width, groups, n, level)
}

/// The scale, standard errors and critical value from replicate-major draws
/// `b`, `width` per replicate.
fn from_draws(
    draws: &[f64],
    width: usize,
    groups: usize,
    n: usize,
    level: Level,
) -> Result<MultiplierBands, Error> {
    let replicates = draws.len() / width;
    let column = |j: usize| (0..replicates).map(move |r| draws[r * width + j]);
    let keep: Vec<bool> = (0..width)
        .map(|j| {
            let total: f64 = column(j).fold(0.0, |a, b| a + b);
            let squares: f64 = column(j).fold(0.0, |a, b| a + b * b);
            !total.is_nan() && squares > f64::EPSILON.sqrt() * 10.0
        })
        .collect();
    let iqr_normal = normal::quantile(0.75) - normal::quantile(0.25);
    let scale: Vec<Option<f64>> = (0..width)
        .map(|j| {
            if !keep[j] {
                return None;
            }
            let mut sorted: Vec<f64> = column(j).collect();
            sorted.sort_by(f64::total_cmp);
            let upper = quantile::type1(&sorted, 0.75)?;
            let lower = quantile::type1(&sorted, 0.25)?;
            Some((upper - lower) / iqr_normal)
        })
        .collect();
    let mut maxima: Vec<f64> = (0..replicates)
        .map(|r| {
            (0..width)
                .filter_map(|j| scale[j].map(|s| (draws[r * width + j] / s).abs()))
                .map(|t| if t.is_nan() { f64::NEG_INFINITY } else { t })
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .filter(|t| t.is_finite())
        .collect();
    maxima.sort_by(f64::total_cmp);
    let critical = quantile::type1(&maxima, level.confidence_value()).ok_or(Error::NotFinite)?;
    let root = (groups as f64).sqrt();
    let se = scale
        .iter()
        .map(|s| s.map(|s| s * root / n as f64))
        .collect();
    Ok(MultiplierBands {
        critical,
        se,
        level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::InputError;
    use epistat_linalg::Mat;
    use epistat_testkit::{assert_same_on_threads, load_json};
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Run {
        crit: f64,
        se: Vec<f64>,
    }

    #[derive(Deserialize)]
    struct Post {
        bres: Vec<Vec<f64>>,
        n: usize,
        n_clusters: usize,
        alpha: f64,
        ndg: Vec<bool>,
        b_sigma: Vec<f64>,
        crit: f64,
        se: Vec<Option<f64>>,
        iqr_norm: f64,
    }

    #[derive(Deserialize)]
    struct SupT {
        cov: Vec<Vec<f64>>,
        q95: f64,
        q90: f64,
    }

    #[derive(Deserialize)]
    struct Reference {
        influence: Vec<Vec<f64>>,
        clusters: Vec<i64>,
        alpha: f64,
        biters: usize,
        plain: Run,
        clustered: Run,
        post: Post,
        supt: SupT,
    }

    fn alpha(a: f64) -> Level {
        Level::alpha(a).unwrap()
    }

    fn reference() -> Reference {
        load_json(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/simultaneous.json"
        ))
    }

    fn columns(values: &[Vec<f64>]) -> Vec<&[f64]> {
        values.iter().map(Vec::as_slice).collect()
    }

    /// The units-by-estimates matrix of one influence column per estimate.
    fn by_unit(columns: &[&[f64]]) -> Mat<f64> {
        Mat::from_fn(columns[0].len(), columns.len(), |i, j| columns[j][i])
    }

    /// The Monte Carlo standard error of the `1 - alpha` quantile of `maxima`,
    /// `sqrt(p (1 - p) / B) / f`, with the density `f` from the quantiles at
    /// `p +- 0.01`.
    fn quantile_se(maxima: &mut [f64], alpha: f64) -> f64 {
        let p = 1.0 - alpha;
        let b = maxima.len() as f64;
        maxima.sort_by(f64::total_cmp);
        let high = quantile::type7(maxima, p + 0.01).unwrap();
        let low = quantile::type7(maxima, p - 0.01).unwrap();
        (p * (1.0 - p) / b).sqrt() * (high - low) / 0.02
    }

    /// The maxima that `multiplier_bands` takes the quantile of, rebuilt from
    /// the definition.
    fn bootstrap_maxima(
        influence: &[&[f64]],
        clusters: Option<&Clusters>,
        replicates: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = influence[0].len();
        let sums: Vec<Vec<f64>> = influence
            .iter()
            .map(|c| match clusters {
                Some(g) => g.sums(c.iter().copied()),
                None => c.to_vec(),
            })
            .collect();
        let groups = sums[0].len();
        let mut draws = Vec::new();
        for r in 0..replicates {
            let mut stream = Stream::seeded(replicate_seed(seed, r as u64));
            let mut signs = Vec::new();
            while signs.len() < groups {
                let bits = stream.next_u64();
                for bit in 0..64.min(groups - signs.len()) {
                    signs.push(if (bits >> bit) & 1 == 1 { 1.0 } else { -1.0 });
                }
            }
            for column in &sums {
                let total = lanes::dot(column, &signs);
                draws.push((groups as f64).sqrt() * (total / groups as f64));
            }
        }
        let bands = from_draws(&draws, influence.len(), groups, n, alpha(0.05)).unwrap();
        let sigma = bands.se.iter().map(|s| s.unwrap()).collect();
        (draws, sigma)
    }

    #[test]
    fn r_post_processing_of_given_draws_matches_to_the_bit() {
        let post = reference().post;
        let width = post.bres.len();
        let replicates = post.bres[0].len();
        let mut draws = vec![0.0; replicates * width];
        for (j, column) in post.bres.iter().enumerate() {
            for (r, value) in column.iter().enumerate() {
                draws[r * width + j] = *value;
            }
        }
        assert_eq!(
            (normal::quantile(0.75) - normal::quantile(0.25)).to_bits(),
            post.iqr_norm.to_bits()
        );
        let bands = from_draws(&draws, width, post.n_clusters, post.n, alpha(post.alpha)).unwrap();
        assert_eq!(bands.critical.to_bits(), post.crit.to_bits());
        assert_eq!(bands.se.len(), post.se.len());
        let mut scales = post.b_sigma.iter();
        for ((got, want), kept) in bands.se.iter().zip(&post.se).zip(&post.ndg) {
            assert_eq!(got.map(f64::to_bits), want.map(f64::to_bits));
            assert_eq!(got.is_some(), *kept);
            if *kept {
                let scale = scales.next().unwrap();
                let expected = scale * (post.n_clusters as f64).sqrt() / post.n as f64;
                assert_eq!(got.unwrap().to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn bands_follow_their_definition_to_the_bit() {
        let reference = reference();
        let influence = columns(&reference.influence);
        let clusters = Clusters::from_sorted_labels(&reference.clusters);
        for clusters in [None, Some(&clusters)] {
            let got = multiplier_bands(
                by_unit(&influence).as_ref(),
                clusters,
                alpha(0.05),
                Bootstrap::new(700, 11),
            )
            .unwrap();
            let (draws, sigma) = bootstrap_maxima(&influence, clusters, 700, 11);
            let groups = clusters.map_or(influence[0].len(), Clusters::count);
            let expected = from_draws(
                &draws,
                influence.len(),
                groups,
                influence[0].len(),
                alpha(0.05),
            )
            .unwrap();
            assert_eq!(got.critical.to_bits(), expected.critical.to_bits());
            for (a, b) in got.se.iter().zip(&sigma) {
                assert_eq!(a.unwrap().to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn complete_runs_agree_with_r_within_monte_carlo_error() {
        let reference = reference();
        let influence = columns(&reference.influence);
        let units = influence[0].len();
        let n = units as f64;
        let clusters = Clusters::from_sorted_labels(&reference.clusters);
        let runs = [
            (None, &reference.plain),
            (Some(&clusters), &reference.clustered),
        ];
        for (clusters, r) in runs {
            let ours = multiplier_bands(
                by_unit(&influence).as_ref(),
                clusters,
                alpha(reference.alpha),
                Bootstrap::new(reference.biters, 3),
            )
            .unwrap();
            // The critical value: both runs have the standard error of the
            // quantile, estimated from our maxima.
            let groups = clusters.map_or(units, Clusters::count);
            let (draws, _) = bootstrap_maxima(&influence, clusters, reference.biters, 3);
            let width = influence.len();
            let mut maxima: Vec<f64> = (0..reference.biters)
                .map(|row| {
                    (0..width)
                        .map(|j| {
                            (draws[row * width + j]
                                / (ours.se[j].unwrap() * n / (groups as f64).sqrt()))
                            .abs()
                        })
                        .fold(0.0, f64::max)
                })
                .collect();
            let se = quantile_se(&mut maxima, reference.alpha);
            assert!(
                (ours.critical - r.crit).abs() < 5.0 * 2.0_f64.sqrt() * se,
                "critical {} against R's {} with standard error {se}",
                ours.critical,
                r.crit
            );
            // An IQR-based scale from B normal draws has relative standard
            // error about 1.17 / sqrt(B).
            let relative = 1.17 / (reference.biters as f64).sqrt();
            for (got, want) in ours.se.iter().zip(&r.se) {
                let got = got.unwrap();
                assert!(
                    (got - want).abs() < 5.0 * 2.0_f64.sqrt() * relative * want,
                    "se {got} against R's {want}"
                );
            }
        }
    }

    #[test]
    fn bands_do_not_depend_on_threads() {
        let reference = reference();
        let influence = columns(&reference.influence);
        let clusters = Clusters::from_sorted_labels(&reference.clusters);
        assert_same_on_threads(|| {
            let mut out = Vec::new();
            for clusters in [None, Some(&clusters)] {
                let bands = multiplier_bands(
                    by_unit(&influence).as_ref(),
                    clusters,
                    alpha(0.05),
                    Bootstrap::new(900, 5),
                )
                .unwrap();
                out.push(bands.critical);
                out.extend(bands.se.iter().map(|s| s.unwrap()));
            }
            out
        });
    }

    #[test]
    fn a_degenerate_estimate_has_no_standard_error() {
        let flat = vec![0.0; 50];
        let varied: Vec<f64> = (0..50).map(|i| f64::from(i % 7) - 3.0).collect();
        let bands = multiplier_bands(
            by_unit(&[&flat, &varied]).as_ref(),
            None,
            alpha(0.05),
            Bootstrap::new(400, 1),
        )
        .unwrap();
        assert_eq!(bands.se[0], None);
        assert!(bands.se[1].is_some());
        assert!(bands.critical.is_finite());
        assert!(bands.at_least_pointwise() >= normal::quantile(0.975));
        assert_eq!(bands.level, alpha(0.05));
    }

    #[test]
    fn invalid_input_is_an_error() {
        let column = [1.0, -1.0, 2.0];
        let ten = Bootstrap::new(10, 1);
        let level = alpha(0.05);
        assert!(matches!(
            multiplier_bands(Mat::<f64>::zeros(3, 0).as_ref(), None, level, ten),
            Err(Error::Input(InputError::Empty { .. }))
        ));
        assert!(matches!(
            multiplier_bands(
                by_unit(&[&column]).as_ref(),
                None,
                level,
                Bootstrap::new(0, 1)
            ),
            Err(Error::Input(InputError::Empty { .. }))
        ));
        assert!(matches!(
            multiplier_bands(
                by_unit(&[&column]).as_ref(),
                Some(&Clusters::singletons(2)),
                level,
                ten
            ),
            Err(Error::Input(InputError::Length { .. }))
        ));
        assert!(matches!(
            multiplier_bands(by_unit(&[&[1.0, f64::NAN, 0.0]]).as_ref(), None, level, ten),
            Err(Error::Input(InputError::NotFinite { row: 1, .. }))
        ));
    }

    fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
        Mat::from_fn(rows.len(), rows.len(), |i, j| rows[i][j])
    }

    /// The maxima of `sup_t_from_covariance`, rebuilt from the definition.
    fn supt_maxima(cov: &Mat<f64>, draws: usize, seed: u64) -> Vec<f64> {
        let dim = cov.nrows();
        let lower = lower_cholesky(&correlation(cov.as_ref()).unwrap(), dim).unwrap();
        (0..draws as u64)
            .map(|r| {
                let mut stream = Stream::seeded(replicate_seed(seed, r));
                let mut z = vec![0.0; dim];
                stream.fill_normals(&mut z);
                let mut best = 0.0_f64;
                for i in 0..dim {
                    let sum = lanes::dot(&lower[i * dim..=i * dim + i], &z[..=i]);
                    best = best.max(sum.abs());
                }
                best
            })
            .collect()
    }

    /// Box-Muller pairs by hand: the cosine, then the sine, of one transform,
    /// and an odd last entry takes the cosine of its own.
    fn reference_normals(stream: &mut Stream, out: &mut [f64]) {
        use std::f64::consts::PI;
        let transform = |stream: &mut Stream| {
            let u1 = stream.uniform().max(f64::MIN_POSITIVE);
            let u2 = stream.uniform();
            ((-2.0 * u1.ln()).sqrt(), 2.0 * PI * u2)
        };
        let (pairs, rest) = out.as_chunks_mut::<2>();
        for pair in pairs {
            let (radius, theta) = transform(stream);
            pair[0] = radius * theta.cos();
            pair[1] = radius * std::hint::black_box(theta).sin();
        }
        if let [last] = rest {
            let (radius, theta) = transform(stream);
            *last = radius * theta.cos();
        }
    }

    #[test]
    fn fill_normals_matches_the_hand_written_pairs_for_odd_and_even_lengths() {
        for len in 0..20 {
            for seed in 0..50 {
                let mut a = vec![0.0; len];
                let mut b = vec![0.0; len];
                let mut sa = Stream::seeded(replicate_seed(7, seed));
                let mut sb = Stream::seeded(replicate_seed(7, seed));
                sa.fill_normals(&mut a);
                reference_normals(&mut sb, &mut b);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a), bits(&b), "length {len}");
                assert_eq!(sa.uniform().to_bits(), sb.uniform().to_bits());
            }
        }
    }

    #[test]
    fn sup_t_follows_its_definition_to_the_bit() {
        let cov = matrix(&reference().supt.cov);
        let got =
            sup_t_from_covariance(cov.as_ref(), alpha(0.05), Bootstrap::new(4_001, 9)).unwrap();
        let mut maxima = supt_maxima(&cov, 4_001, 9);
        let expected = quantile::type7_unsorted(&mut maxima, 1.0 - 0.05).unwrap();
        assert_eq!(got.to_bits(), expected.to_bits());
    }

    #[test]
    fn sup_t_agrees_with_qmvnorm_within_monte_carlo_error() {
        let supt = reference().supt;
        let cov = matrix(&supt.cov);
        let draws = 400_000;
        for (alpha, exact) in [(0.05, supt.q95), (0.10, supt.q90)] {
            let got = sup_t_from_covariance(
                cov.as_ref(),
                Level::alpha(alpha).unwrap(),
                Bootstrap::new(draws, 21),
            )
            .unwrap();
            let mut maxima = supt_maxima(&cov, draws, 21);
            let se = quantile_se(&mut maxima, alpha);
            assert!(
                (got - exact).abs() < 5.0 * se + 1e-5,
                "{got} against qmvnorm {exact} with standard error {se}"
            );
        }
    }

    #[test]
    fn sup_t_of_one_coordinate_is_the_pointwise_value() {
        let cov = Mat::from_fn(1, 1, |_, _| 7.5);
        let got =
            sup_t_from_covariance(cov.as_ref(), alpha(0.05), Bootstrap::new(400_000, 4)).unwrap();
        assert!((got - normal::quantile(0.975)).abs() < 0.01);
    }

    #[test]
    fn sup_t_does_not_depend_on_threads() {
        let cov = matrix(&reference().supt.cov);
        assert_same_on_threads(|| {
            vec![
                sup_t_from_covariance(cov.as_ref(), alpha(0.05), Bootstrap::new(5_000, 2)).unwrap(),
            ]
        });
    }

    #[test]
    fn sup_t_rejects_what_it_cannot_standardise() {
        let zero_variance = Mat::from_fn(2, 2, |i, j| if i == j && i == 0 { 0.0 } else { 1.0 });
        assert_eq!(
            sup_t_from_covariance(zero_variance.as_ref(), alpha(0.05), Bootstrap::new(10, 1)),
            Err(Error::NotFinite)
        );
        let above_one = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 2.0 });
        assert!(
            sup_t_from_covariance(above_one.as_ref(), alpha(0.05), Bootstrap::new(10, 1)).is_ok()
        );
        let indefinite = Mat::from_fn(3, 3, |i, j| match (i, j) {
            _ if i == j => 1.0,
            (0, 2) | (2, 0) => -0.9,
            _ => 0.9,
        });
        assert_eq!(
            sup_t_from_covariance(indefinite.as_ref(), alpha(0.05), Bootstrap::new(10, 1)),
            Err(Error::NotFinite)
        );
        let wide = Mat::<f64>::zeros(2, 3);
        assert!(matches!(
            sup_t_from_covariance(wide.as_ref(), alpha(0.05), Bootstrap::new(10, 1)),
            Err(Error::Input(InputError::Length { .. }))
        ));
        let empty = Mat::<f64>::zeros(0, 0);
        assert!(matches!(
            sup_t_from_covariance(empty.as_ref(), alpha(0.05), Bootstrap::new(10, 1)),
            Err(Error::Input(InputError::Empty { .. }))
        ));
    }
}
