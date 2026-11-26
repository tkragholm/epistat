//! k-means clustering from given starting centres.
//!
//! [`kmeans`] partitions `n` points in `d` dimensions into `k` clusters that
//! make the within-cluster sum of squared distances to the cluster means
//! small. It starts from centres the caller gives, so the result is a
//! function of the data and the starts alone and needs no random numbers. Two
//! algorithms are offered, as in R's `stats::kmeans()`.
//!
//! # Lloyd
//!
//! Each point goes to the nearest centre, the first on a tie, then every
//! centre becomes the mean of its points, and the two steps repeat until no
//! point changes cluster. This is Lloyd's algorithm \[1\] in the form of
//! Forgy's \[2\].
//!
//! # Hartigan-Wong
//!
//! The default of R, Algorithm AS 136 of Hartigan and Wong \[3\]. Every point
//! is given its nearest and second nearest centre, the centres become the
//! means of their points, and then two stages alternate. In the optimal
//! transfer stage a point `i` in a cluster of `n1 > 1` points with mean `c1`
//! moves to the cluster `l` of `n_l` points with mean `c_l` that minimises
//! `n_l / (n_l + 1) * |x_i - c_l|^2`, if that is below
//! `n1 / (n1 - 1) * |x_i - c1|^2`. Both sides are the change in the
//! within-cluster sum of squares that the move would cause, so a move lowers
//! it. A point is tested only against the clusters that changed since it was
//! last tested. The quick transfer stage then moves points between their own
//! two nearest clusters until none moves. The algorithm ends when a whole pass
//! of the optimal transfer stage moves no point, or after one pass of both
//! stages when `k = 2`. The means are updated after every move. The paper
//! fixes the order in which points are visited and clusters tried, and this
//! implementation follows it, so from the same start it reaches the same
//! local minimum as AS 136.
//!
//! A cluster that loses all its points is an error, as in R, where
//! `kmeans()` stops with "empty cluster". Sums run in the order of the points.
//!
//! The partitions and centres of both algorithms are checked against
//! `stats::kmeans()` from the same starts.
//!
//! # References
//!
//! 1. Lloyd SP. Least squares quantization in PCM. IEEE Trans Inf Theory.
//!    1982;28(2):129-37.
//! 2. Forgy EW. Cluster analysis of multivariate data: efficiency versus
//!    interpretability of classifications. Biometrics. 1965;21:768-9.
//! 3. Hartigan JA, Wong MA. Algorithm AS 136: a k-means clustering algorithm.
//!    J R Stat Soc Ser C. 1979;28(1):100-8.

/// The algorithm that moves points between clusters.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KMeansAlgorithm {
    /// Lloyd and Forgy.
    Lloyd,
    /// Hartigan and Wong, with single-point transfers.
    #[default]
    HartiganWong,
}

/// The error of [`kmeans`] before 0.3.0.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::Error`")]
pub type KMeansError = Error;

/// A partition into clusters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct KMeans {
    /// The cluster of each point, in `0..k`.
    pub cluster: Vec<usize>,
    /// The cluster means, `k * d` values, cluster by cluster.
    pub centres: Vec<f64>,
    /// The sum of squared distances to the mean within each cluster.
    pub within_ss: Vec<f64>,
    /// The number of points in each cluster.
    pub size: Vec<usize>,
    /// The number of passes.
    pub iterations: usize,
    /// Whether the iteration ended before `max_iterations`.
    pub converged: bool,
}

use epistat_core::InputError;
use rayon::prelude::*;

use crate::Error;

/// The error for a count `what` that must be at least 1.
fn at_least_one(what: &'static str, value: usize) -> Error {
    InputError::Domain {
        what,
        value: value as f64,
        domain: "[1, inf)",
    }
    .into()
}

/// Clusters the rows of `points` (`n * d` values, row by row) from the rows of
/// `start` (`k * d` values), with at most `max_iterations` passes.
///
/// # Errors
///
/// [`Error::Input`] for no points, a dimension or `max_iterations` of 0, a
/// length of `points` or `start` that is not a multiple of `d`, a value that
/// is not finite, and no centres or more centres than points;
/// [`Error::EmptyCluster`] when a cluster loses all its points.
pub fn kmeans(
    points: &[f64],
    d: usize,
    start: &[f64],
    algorithm: KMeansAlgorithm,
    max_iterations: usize,
) -> Result<KMeans, Error> {
    if points.is_empty() {
        return Err(InputError::Empty { what: "points" }.into());
    }
    if d == 0 {
        return Err(at_least_one("d", 0));
    }
    if max_iterations == 0 {
        return Err(at_least_one("max_iterations", 0));
    }
    for (what, values) in [("points", points), ("start", start)] {
        if !values.len().is_multiple_of(d) {
            return Err(InputError::Domain {
                what: if what == "points" {
                    "length of points"
                } else {
                    "length of start"
                },
                value: values.len() as f64,
                domain: "a multiple of d",
            }
            .into());
        }
    }
    for (what, values) in [("points", points), ("start", start)] {
        if let Some(row) = values.iter().position(|v| !v.is_finite()) {
            return Err(InputError::NotFinite { what, row }.into());
        }
    }
    let (n, k) = (points.len() / d, start.len() / d);
    if k == 0 || k > n {
        return Err(InputError::Domain {
            what: "number of centres",
            value: k as f64,
            domain: "1 to the number of points",
        }
        .into());
    }
    let mut centres = start.to_vec();
    let mut cluster = vec![usize::MAX; n];
    let mut iterations = 0;
    let mut converged = false;

    match algorithm {
        KMeansAlgorithm::Lloyd => {
            for iteration in 1..=max_iterations {
                iterations = iteration;
                if assign(points, d, &centres, &mut cluster) == 0 {
                    converged = true;
                    break;
                }
                centres = means(points, d, &cluster, k)?;
            }
        }
        KMeansAlgorithm::HartiganWong => {
            (converged, iterations) =
                hartigan_wong(points, d, &mut centres, &mut cluster, k, max_iterations)?;
        }
    }
    let mut size = vec![0usize; k];
    for &c in &cluster {
        size[c] += 1;
    }
    if let Some(empty) = size.iter().position(|&s| s == 0) {
        return Err(Error::EmptyCluster { cluster: empty });
    }
    centres = means(points, d, &cluster, k)?;
    let mut within_ss = vec![0.0; k];
    for i in 0..n {
        let c = cluster[i];
        within_ss[c] += squared_distance(&points[i * d..(i + 1) * d], &centres[c * d..(c + 1) * d]);
    }
    Ok(KMeans {
        cluster,
        centres,
        within_ss,
        size,
        iterations,
        converged,
    })
}

/// Gives every point its nearest centre and returns how many changed.
///
/// The points are independent, so the nearest centres are found in parallel for
/// many points, and the count is the same whatever the number of threads.
fn assign(points: &[f64], d: usize, centres: &[f64], cluster: &mut [usize]) -> usize {
    let nearest = |i: usize| nearest_centre(&points[i * d..(i + 1) * d], centres, d);
    let found: Vec<usize> = if cluster.len() >= PARALLEL_FROM {
        (0..cluster.len()).into_par_iter().map(nearest).collect()
    } else {
        (0..cluster.len()).map(nearest).collect()
    };
    let mut changed = 0;
    for (slot, nearest) in cluster.iter_mut().zip(found) {
        if *slot != nearest {
            *slot = nearest;
            changed += 1;
        }
    }
    changed
}

/// Points from which the work on each point runs in parallel.
const PARALLEL_FROM: usize = 4096;

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += (x - y) * (x - y);
    }
    s
}

fn nearest_centre(x: &[f64], centres: &[f64], d: usize) -> usize {
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (c, centre) in centres.chunks_exact(d).enumerate() {
        let distance = squared_distance(x, centre);
        if distance < best_distance {
            best_distance = distance;
            best = c;
        }
    }
    best
}

fn means(points: &[f64], d: usize, cluster: &[usize], k: usize) -> Result<Vec<f64>, Error> {
    let mut sums = vec![0.0; k * d];
    let mut size = vec![0usize; k];
    for (i, &c) in cluster.iter().enumerate() {
        size[c] += 1;
        for j in 0..d {
            sums[c * d + j] += points[i * d + j];
        }
    }
    if let Some(empty) = size.iter().position(|&s| s == 0) {
        return Err(Error::EmptyCluster { cluster: empty });
    }
    for c in 0..k {
        for j in 0..d {
            sums[c * d + j] /= size[c] as f64;
        }
    }
    Ok(sums)
}

/// A count as the signed step numbers of AS 136, which go below zero.
fn signed(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// The factor `n / (n - 1)` stands for when a cluster is left with one point.
const BIG: f64 = 1e30;

/// The state of AS 136. Steps and positions are counted from 1 as in the
/// paper, so `live` and `ncp` compare with them directly.
struct Transfers<'a> {
    points: &'a [f64],
    d: usize,
    k: usize,
    m: i64,
    centres: &'a mut [f64],
    ic1: Vec<usize>,
    ic2: Vec<usize>,
    size: Vec<usize>,
    an1: Vec<f64>,
    an2: Vec<f64>,
    ncp: Vec<i64>,
    dist: Vec<f64>,
    itran: Vec<bool>,
    live: Vec<i64>,
    index: i64,
}

impl Transfers<'_> {
    fn distance(&self, i: usize, cluster: usize) -> f64 {
        squared_distance(
            &self.points[i * self.d..(i + 1) * self.d],
            &self.centres[cluster * self.d..(cluster + 1) * self.d],
        )
    }

    /// Moves point `i` from cluster `l1` to `l2` and updates the centres and
    /// the factors `n / (n - 1)` and `n / (n + 1)`.
    fn move_point(&mut self, i: usize, l1: usize, l2: usize) {
        let d = self.d;
        let al1 = self.size[l1] as f64;
        let alw = al1 - 1.0;
        let al2 = self.size[l2] as f64;
        let alt = al2 + 1.0;
        for j in 0..d {
            let x = self.points[i * d + j];
            self.centres[l1 * d + j] = (self.centres[l1 * d + j] * al1 - x) / alw;
            self.centres[l2 * d + j] = (self.centres[l2 * d + j] * al2 + x) / alt;
        }
        self.size[l1] -= 1;
        self.size[l2] += 1;
        self.an2[l1] = alw / al1;
        self.an1[l1] = if alw > 1.0 { alw / (alw - 1.0) } else { BIG };
        self.an1[l2] = alt / al2;
        self.an2[l2] = alt / (alt + 1.0);
        self.ic1[i] = l2;
        self.ic2[i] = l1;
    }

    /// The optimal transfer stage. Returns true when a whole pass found no
    /// transfer.
    fn optimal_transfer(&mut self) -> bool {
        let m = self.m;
        for l in 0..self.k {
            if self.itran[l] {
                self.live[l] = m;
            }
        }
        for i in 0..self.ic1.len() {
            let step = signed(i) + 1;
            self.index += 1;
            let l1 = self.ic1[i];
            if self.size[l1] != 1 {
                if self.ncp[l1] != 0 {
                    self.dist[i] = self.distance(i, l1) * self.an1[l1];
                }
                let mut l2 = self.ic2[i];
                let ll = l2;
                let mut r2 = self.distance(i, l2) * self.an2[l2];
                for l in 0..self.k {
                    if (step >= self.live[l1] && step >= self.live[l]) || l == l1 || l == ll {
                        continue;
                    }
                    let rr = r2 / self.an2[l];
                    let dc = self.distance(i, l);
                    if dc >= rr {
                        continue;
                    }
                    r2 = dc * self.an2[l];
                    l2 = l;
                }
                if r2 >= self.dist[i] {
                    self.ic2[i] = l2;
                } else {
                    self.index = 0;
                    self.live[l1] = m + step;
                    self.live[l2] = m + step;
                    self.ncp[l1] = step;
                    self.ncp[l2] = step;
                    self.move_point(i, l1, l2);
                }
            }
            if self.index == m {
                return true;
            }
        }
        for l in 0..self.k {
            self.itran[l] = false;
            self.live[l] -= m;
        }
        false
    }

    /// The quick transfer stage, which moves points between their two nearest
    /// clusters until a whole pass moves none. Returns false when it exceeds
    /// `max_steps`.
    fn quick_transfer(&mut self, max_steps: i64) -> bool {
        let m = self.m;
        let mut count = 0;
        let mut step = 0;
        loop {
            for i in 0..self.ic1.len() {
                count += 1;
                step += 1;
                if step >= max_steps {
                    return false;
                }
                let l1 = self.ic1[i];
                let l2 = self.ic2[i];
                if self.size[l1] != 1 {
                    if step <= self.ncp[l1] {
                        self.dist[i] = self.distance(i, l1) * self.an1[l1];
                    }
                    if step < self.ncp[l1] || step < self.ncp[l2] {
                        let r2 = self.dist[i] / self.an2[l2];
                        if self.distance(i, l2) < r2 {
                            count = 0;
                            self.index = 0;
                            self.itran[l1] = true;
                            self.itran[l2] = true;
                            self.ncp[l1] = step + m;
                            self.ncp[l2] = step + m;
                            self.move_point(i, l1, l2);
                        }
                    }
                }
                if count == m {
                    return true;
                }
            }
        }
    }
}

/// Algorithm AS 136 from the starting `centres`. Returns whether it ended
/// within `max_passes` passes, and the number of passes.
fn hartigan_wong(
    points: &[f64],
    d: usize,
    centres: &mut [f64],
    cluster: &mut [usize],
    k: usize,
    max_passes: usize,
) -> Result<(bool, usize), Error> {
    let n = cluster.len();
    // The nearest and second nearest centre of every point. With one centre
    // the second does not exist and the clustering is trivial.
    let two_nearest = |i: usize| {
        let x = &points[i * d..(i + 1) * d];
        let dist = |l: usize| squared_distance(x, &centres[l * d..(l + 1) * d]);
        let (mut a, mut b) = (0, 1.min(k - 1));
        let (mut da, mut db) = (dist(a), dist(b));
        if k > 1 && da > db {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut da, &mut db);
        }
        for l in 2..k {
            let dl = dist(l);
            if dl < db {
                if dl < da {
                    db = da;
                    b = a;
                    da = dl;
                    a = l;
                } else {
                    db = dl;
                    b = l;
                }
            }
        }
        (a, b)
    };
    let (ic1, ic2): (Vec<usize>, Vec<usize>) = if n >= PARALLEL_FROM {
        (0..n).into_par_iter().map(two_nearest).unzip()
    } else {
        (0..n).map(two_nearest).unzip()
    };
    centres.copy_from_slice(&means(points, d, &ic1, k)?);
    let mut size = vec![0usize; k];
    for &c in &ic1 {
        size[c] += 1;
    }
    let an1 = size
        .iter()
        .map(|&s| s as f64 / (s as f64 - 1.0))
        .collect::<Vec<_>>();
    let an2 = size
        .iter()
        .map(|&s| s as f64 / (s as f64 + 1.0))
        .collect::<Vec<_>>();
    let mut state = Transfers {
        points,
        d,
        k,
        m: signed(n),
        centres,
        ic1,
        ic2,
        size,
        an1,
        an2,
        ncp: vec![-1; k],
        dist: vec![0.0; n],
        itran: vec![true; k],
        live: vec![0; k],
        index: 0,
    };
    let max_quick = 50 * signed(n);
    let mut converged = false;
    let mut passes = 0;
    for pass in 1..=max_passes {
        passes = pass;
        if state.optimal_transfer() {
            converged = true;
            break;
        }
        if !state.quick_transfer(max_quick) {
            break;
        }
        if k == 2 {
            converged = true;
            break;
        }
        state.ncp.fill(0);
    }
    cluster.copy_from_slice(&state.ic1);
    Ok((converged, passes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs() -> (Vec<f64>, Vec<f64>) {
        let points = vec![
            0.0, 0.1, 0.2, 0.15, 5.0, 5.2, 5.1, 4.9, 10.0, 10.1, 9.9, 10.2,
        ];
        (points, vec![0.0, 5.0, 10.0])
    }

    #[test]
    fn well_separated_blobs_are_found_by_both_algorithms() {
        let (points, start) = blobs();
        for algorithm in [KMeansAlgorithm::Lloyd, KMeansAlgorithm::HartiganWong] {
            let fit = kmeans(&points, 1, &start, algorithm, 100).unwrap();
            assert_eq!(fit.cluster, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
            assert!(fit.converged);
            assert_eq!(fit.size, [4, 4, 4]);
        }
    }

    #[test]
    fn transfers_do_not_end_above_the_sum_of_squares_of_lloyd() {
        let points = [0.0, 1.0, 2.9, 4.0];
        let start = [0.0, 4.0];
        let lloyd = kmeans(&points, 1, &start, KMeansAlgorithm::Lloyd, 100).unwrap();
        let hw = kmeans(&points, 1, &start, KMeansAlgorithm::HartiganWong, 100).unwrap();
        let total = |f: &KMeans| f.within_ss.iter().sum::<f64>();
        assert!(total(&hw) <= total(&lloyd) + 1e-12);
    }

    #[test]
    fn bad_input_is_an_error() {
        assert_eq!(
            kmeans(&[], 1, &[0.0], KMeansAlgorithm::Lloyd, 10),
            Err(Error::Input(InputError::Empty { what: "points" }))
        );
        assert!(kmeans(&[1.0, 2.0, 3.0], 2, &[1.0, 2.0], KMeansAlgorithm::Lloyd, 10).is_err());
        assert!(kmeans(&[1.0, f64::NAN], 1, &[1.0], KMeansAlgorithm::Lloyd, 10).is_err());
        assert_eq!(
            kmeans(&[1.0, 2.0], 1, &[1.0, 2.0, 3.0], KMeansAlgorithm::Lloyd, 10),
            Err(Error::Input(InputError::Domain {
                what: "number of centres",
                value: 3.0,
                domain: "1 to the number of points"
            }))
        );
        // A start far from every point leaves its cluster empty.
        assert_eq!(
            kmeans(
                &[1.0, 2.0, 3.0],
                1,
                &[1.5, 100.0],
                KMeansAlgorithm::Lloyd,
                10
            ),
            Err(Error::EmptyCluster { cluster: 1 })
        );
    }

    #[test]
    fn zero_iterations_is_an_error() {
        for algorithm in [KMeansAlgorithm::Lloyd, KMeansAlgorithm::HartiganWong] {
            assert_eq!(
                kmeans(&[0.0, 1.0, 10.0, 11.0], 1, &[0.0, 10.0], algorithm, 0),
                Err(Error::Input(InputError::Domain {
                    what: "max_iterations",
                    value: 0.0,
                    domain: "[1, inf)"
                }))
            );
        }
    }

    fn scattered(n: usize) -> Vec<f64> {
        let mut state = 99_u64;
        (0..n * 2)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1_u64 << 53) as f64
            })
            .collect()
    }

    /// The parallel assignment gives each point the centre a plain loop gives
    /// it and counts the same changes.
    #[test]
    fn assignment_equals_a_plain_loop() {
        let n = 6000;
        let points = scattered(n);
        let centres = [0.1, 0.2, 0.8, 0.3, 0.5, 0.9, 0.2, 0.7];
        let mut cluster = vec![usize::MAX; n];
        let changed = assign(&points, 2, &centres, &mut cluster);
        assert_eq!(changed, n);
        for (i, &c) in cluster.iter().enumerate() {
            assert_eq!(c, nearest_centre(&points[i * 2..i * 2 + 2], &centres, 2));
        }
        assert_eq!(assign(&points, 2, &centres, &mut cluster), 0);
    }

    /// Both algorithms give the same clustering on pools of 1, 2, 4 and 12
    /// threads.
    #[test]
    fn clustering_is_the_same_on_every_pool_size() {
        let points = scattered(6000);
        let start = [0.1, 0.2, 0.8, 0.3, 0.5, 0.9, 0.2, 0.7];
        for algorithm in [KMeansAlgorithm::Lloyd, KMeansAlgorithm::HartiganWong] {
            epistat_testkit::assert_same_on_pools(&[1, 2, 4, 12], || {
                let fit = kmeans(&points, 2, &start, algorithm, 100).unwrap();
                fit.cluster
                    .iter()
                    .map(|&c| c as f64)
                    .chain(fit.centres)
                    .chain(fit.within_ss)
                    .collect()
            });
        }
    }
}
