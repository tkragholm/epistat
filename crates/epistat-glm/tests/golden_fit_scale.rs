//! The bits of `Glm::fit` on problems of tens of thousands of rows, pinned by
//! a hash of every output field, on one thread and on several. The rows span
//! many blocks of the parallel loops, and the problems take the paths of a fit:
//! an aliased column, prior weights with zeros, an offset, starting
//! coefficients, step halving, an estimated dispersion and a design that is not
//! column-major. The hashes were recorded before the fit was parallelised over
//! rows, so they show that the parallel fit takes the same floating-point
//! steps.
//!
//! The bits depend on the platform's `ln` and `exp`, so the hashes are
//! asserted only on the target they were recorded on, aarch64 macOS. Elsewhere
//! the test computes them and, when `EPISTAT_HASH_OUT` is set, appends them to
//! that file.

use epistat_glm::{Family, Fit, Glm, Link};
use epistat_linalg::Mat;
use epistat_testkit::{Fingerprint, SplitMix64, on_threads};

mod common;
use common::*;

fn normal(rng: &mut SplitMix64) -> f64 {
    let (u, v) = (1.0 - rng.unit(), rng.unit());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn hash_options(h: Fingerprint, values: &[Option<f64>]) -> Fingerprint {
    values
        .iter()
        .fold(h.u64(values.len() as u64), |h, &v| h.option_f64(v))
}

fn hash_fit(fit: &Fit) -> u64 {
    let mut h = hash_options(Fingerprint::new(), &fit.coefficients);
    h = hash_options(h, fit.covariance.as_slice());
    h = h
        .u64(fit.rank as u64)
        .u64(fit.residual_df as u64)
        .f64s(&fit.fitted)
        .f64s(&fit.linear_predictor)
        .f64(fit.deviance)
        .u64(fit.iterations as u64)
        .u64(u64::from(fit.converged))
        .u64(u64::from(fit.boundary))
        .f64s(&fit.working_weights)
        .option_f64(fit.dispersion);
    h.finish()
}

/// An intercept and `p - 1` columns, of which column 3 is twice column 1 when
/// `p > 3`, so that it is aliased.
fn design(n: usize, p: usize, rng: &mut SplitMix64) -> Mat<f64> {
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        for j in 1..p {
            x[(i, j)] = if j == 3 {
                2.0 * x[(i, 1)]
            } else if j.is_multiple_of(2) {
                f64::from(u8::from(rng.unit() < 0.4))
            } else {
                normal(rng)
            };
        }
    }
    x
}

fn eta(x: &Mat<f64>, i: usize, base: f64) -> f64 {
    (1..x.ncols()).fold(base, |e, j| e + 0.15 * x[(i, j)] / j as f64)
}

/// The hashes of the fits, in a fixed order.
fn fits() -> Vec<u64> {
    let n = 30_011;
    let mut rng = SplitMix64::new(20_261_124);
    let mut hashes = Vec::new();

    // Logistic, with an aliased column.
    let x = design(n, 8, &mut rng);
    let y: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(u8::from(
                rng.unit() < 1.0 / (1.0 + (-eta(&x, i, -0.8)).exp()),
            ))
        })
        .collect();
    let logit = Glm::new(Family::Binomial, Link::Logit);
    let fit = logit.fit5(x.as_ref(), &y, None, None, None).unwrap();
    assert_eq!(fit.rank, 7);
    hashes.push(hash_fit(&fit));

    // The same on a row-major design, which the fit copies.
    let rows = Mat::from_fn(x.ncols(), n, |j, i| x[(i, j)]);
    hashes.push(hash_fit(
        &logit
            .fit5(rows.as_ref().transpose(), &y, None, None, None)
            .unwrap(),
    ));

    // Poisson with prior weights, some zero, and an offset.
    let x = design(n, 6, &mut rng);
    let offset: Vec<f64> = (0..n).map(|_| 0.2 * normal(&mut rng)).collect();
    let weights: Vec<f64> = (0..n)
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => 2.5,
            _ => 1.0,
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let mean = (eta(&x, i, 0.3) + offset[i]).exp();
            // Inversion of the Poisson distribution function.
            let (mut k, mut p, u) = (0.0, (-mean).exp(), rng.unit());
            let mut cdf = p;
            while u > cdf && k < 60.0 {
                k += 1.0;
                p *= mean / k;
                cdf += p;
            }
            k
        })
        .collect();
    let poisson = Glm::new(Family::Poisson, Link::Log);
    hashes.push(hash_fit(
        &poisson
            .fit5(x.as_ref(), &y, Some(&weights), Some(&offset), None)
            .unwrap(),
    ));

    // Gamma with a log link from starting coefficients, dispersion estimated.
    let x = design(n, 5, &mut rng);
    let y: Vec<f64> = (0..n)
        .map(|i| eta(&x, i, 1.0).exp() * (0.2 + 1.6 * rng.unit()))
        .collect();
    let start = [0.5, 0.0, 0.0, 0.0, 0.0];
    let gamma = Glm::new(Family::Gamma, Link::Log);
    hashes.push(hash_fit(
        &gamma
            .fit5(x.as_ref(), &y, None, None, Some(&start))
            .unwrap(),
    ));

    // A log-binomial fit, whose steps leave the unit interval and are
    // halved, with a ridge.
    let y: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(u8::from(
                rng.unit() < (eta(&x, i, -0.4) + 0.4 * x[(i, 1)]).exp().min(0.97),
            ))
        })
        .collect();
    let low = [-1.5, 0.0, 0.0, 0.0, 0.0];
    let ridged = Glm {
        ridge: 0.5,
        ..Glm::new(Family::Binomial, Link::Log)
    };
    let fit = ridged.fit5(x.as_ref(), &y, None, None, Some(&low)).unwrap();
    assert!(fit.boundary);
    hashes.push(hash_fit(&fit));
    hashes
}

#[test]
fn the_fits_keep_their_bits_on_every_pool() {
    let expected: [u64; 5] = [
        10_970_117_131_332_751_219,
        10_970_117_131_332_751_219,
        15_433_776_158_932_545_206,
        2_265_443_942_968_990_635,
        1_716_175_949_192_639_765,
    ];
    let results = on_threads(&[1, 3, 8], fits);
    for (i, hash) in results[0].iter().enumerate() {
        epistat_testkit::hash_out(&format!("fit/{i}"), *hash);
    }
    // Off the recording target the pools still have to agree with each other.
    let want: Vec<u64> = if epistat_testkit::RECORDED_PLATFORM {
        expected.to_vec()
    } else {
        results[0].clone()
    };
    for (threads, got) in [1, 3, 8].into_iter().zip(results) {
        assert_eq!(got, want, "{threads} threads");
    }
}
