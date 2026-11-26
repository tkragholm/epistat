//! The distribution draws: moments against the known values within the Monte
//! Carlo error, and bit identity with the copies the consumers hold.

use std::f64::consts::PI;

use epistat_core::{Classify, ErrorKind, InputError};
use epistat_rng::{Error, Stream, dirichlet_scale, dirichlet_weights};
use epistat_testkit::assert_within_standard_errors;

const DRAWS: usize = 200_000;
const K: f64 = 5.0;

/// The mean and variance of a sample, each with its standard error from the
/// sample's own second and fourth central moments.
fn moments(sample: &[f64]) -> (f64, f64, f64, f64) {
    let n = sample.len() as f64;
    let mean = sample.iter().sum::<f64>() / n;
    let m2 = sample.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4 = sample.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    let variance = m2 * n / (n - 1.0);
    (mean, (m2 / n).sqrt(), variance, ((m4 - m2 * m2) / n).sqrt())
}

fn check(sample: &[f64], mean: f64, variance: f64) {
    let (m, se_m, v, se_v) = moments(sample);
    assert_within_standard_errors(m, se_m, mean, 0.0, K);
    assert_within_standard_errors(v, se_v, variance, 0.0, K);
}

#[test]
fn normal_has_mean_zero_and_variance_one() {
    let mut stream = Stream::seeded(1);
    let sample: Vec<f64> = (0..DRAWS).map(|_| stream.normal()).collect();
    check(&sample, 0.0, 1.0);
    let mut sample = vec![0.0; DRAWS + 1];
    Stream::seeded(3).fill_normals(&mut sample);
    check(&sample, 0.0, 1.0);
}

#[test]
fn gamma_chi_square_and_beta_have_their_moments() {
    for (shape, scale) in [(0.5, 2.0), (1.0, 1.0), (3.0, 0.5), (12.5, 2.0)] {
        let mut stream = Stream::seeded(4);
        let sample: Vec<f64> = (0..DRAWS)
            .map(|_| stream.gamma(shape, scale).unwrap())
            .collect();
        check(&sample, shape * scale, shape * scale * scale);
    }
    for df in [1.0, 3.0, 10.0] {
        let mut stream = Stream::seeded(5);
        let sample: Vec<f64> = (0..DRAWS).map(|_| stream.chi_square(df).unwrap()).collect();
        check(&sample, df, 2.0 * df);
    }
    for (a, b) in [(2.0, 2.0), (0.5, 1.5), (5.0, 1.0)] {
        let mut stream = Stream::seeded(6);
        let sample: Vec<f64> = (0..DRAWS).map(|_| stream.beta(a, b).unwrap()).collect();
        let s = a + b;
        check(&sample, a / s, a * b / (s * s * (s + 1.0)));
    }
}

/// Lower Cholesky factor of a small positive definite matrix, row-major.
const FACTOR: [f64; 9] = [2.0, 0.0, 0.0, 0.5, 1.5, 0.0, -0.3, 0.4, 1.0];

fn scale_matrix() -> [f64; 9] {
    let mut s = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                s[i * 3 + j] += FACTOR[i * 3 + k] * FACTOR[j * 3 + k];
            }
        }
    }
    s
}

fn mean_of_matrices(draw: impl Fn(&mut Stream) -> Vec<f64>, entry: usize) -> Vec<f64> {
    let mut stream = Stream::seeded(7);
    (0..DRAWS / 4).map(|_| draw(&mut stream)[entry]).collect()
}

#[test]
fn wishart_and_inverse_wishart_have_their_means() {
    let scale = scale_matrix();
    let df = 7.0;
    for (entry, &mean) in scale.iter().enumerate() {
        let sample = mean_of_matrices(|s| s.wishart(df, &FACTOR, 3).unwrap(), entry);
        let (m, se, _, _) = moments(&sample);
        assert_within_standard_errors(m, se, df * mean, 0.0, K);
        let sample = mean_of_matrices(|s| s.inverse_wishart(df, &FACTOR, 3).unwrap(), entry);
        let (m, se, _, _) = moments(&sample);
        assert_within_standard_errors(m, se, mean / (df - 3.0 - 1.0), 0.0, K);
    }
}

#[test]
fn wishart_is_exactly_symmetric_and_the_one_by_one_case_is_a_scaled_chi_square() {
    let draw = Stream::seeded(8).wishart(5.0, &FACTOR, 3).unwrap();
    for i in 0..3 {
        for j in 0..i {
            assert_eq!(draw[i * 3 + j].to_bits(), draw[j * 3 + i].to_bits());
        }
    }
    let one = Stream::seeded(9).wishart(4.0, &[1.5], 1).unwrap()[0];
    let chi = Stream::seeded(9).chi_square(4.0).unwrap();
    assert_eq!(
        one.to_bits(),
        (1.5 * chi.sqrt() * (1.5 * chi.sqrt())).to_bits()
    );
}

#[test]
fn inverse_wishart_inverts_a_wishart_draw_of_the_inverse_scale() {
    // For one dimension the inverse Wishart is scale / chi-square.
    let one = Stream::seeded(10).inverse_wishart(6.0, &[2.0], 1).unwrap()[0];
    let chi = Stream::seeded(10).chi_square(6.0).unwrap();
    assert!((one - 4.0 / chi).abs() <= 4.0 / chi * 1e-15);
}

#[test]
fn invalid_parameters_are_errors_and_draw_nothing() {
    let mut stream = Stream::seeded(11);
    let before = stream.clone();
    let invalid: [Result<f64, Error>; 6] = [
        stream.gamma(0.0, 1.0),
        stream.gamma(1.0, f64::NAN),
        stream.gamma(f64::INFINITY, 1.0),
        stream.chi_square(-1.0),
        stream.beta(0.0, 1.0),
        stream.beta(1.0, f64::NAN),
    ];
    assert!(invalid.iter().all(Result::is_err));
    assert!(stream.wishart(2.0, &FACTOR, 3).is_err());
    assert!(stream.wishart(5.0, &FACTOR[..8], 3).is_err());
    assert!(stream.wishart(5.0, &[], 0).is_err());
    assert!(
        stream
            .inverse_wishart(5.0, &[1.0, 0.0, 0.0, 0.0], 2)
            .is_err()
    );
    assert_eq!(stream, before);
}

#[test]
fn a_refusal_names_the_parameter_and_is_an_input_error() {
    let mut stream = Stream::seeded(11);
    let error = stream.gamma(-2.0, 1.0).unwrap_err();
    assert_eq!(
        error,
        Error::Input(InputError::Domain {
            what: "shape",
            value: -2.0,
            domain: "(0, inf)"
        })
    );
    assert_eq!(
        (error.kind(), error.code()),
        (ErrorKind::Input, "input.domain")
    );
    assert_eq!(error.to_string(), "shape is -2, outside (0, inf)");
    assert_eq!(
        stream.wishart(5.0, &FACTOR[..8], 3),
        Err(Error::Input(InputError::Length {
            what: "scale_cholesky",
            expected: 9,
            found: 8
        }))
    );
    assert!(matches!(
        stream.wishart(2.0, &FACTOR, 3),
        Err(Error::Input(InputError::Domain { what: "df", .. }))
    ));
    assert!(matches!(
        stream.inverse_wishart(5.0, &[1.0, 0.0, 0.0, 0.0], 2),
        Err(Error::Input(InputError::Domain {
            what: "diagonal of scale_cholesky",
            ..
        }))
    ));
}

// The copies below are the consumers' functions as they are written there.

/// did_methods `inference/sensitivity/linear_algebra.rs`, `draw_standard_normal`.
fn did_methods_normal(rng: &mut Stream) -> f64 {
    let u1 = rng.uniform().max(f64::MIN_POSITIVE);
    let u2 = rng.uniform();
    (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * PI * u2).cos()
}

/// did_methods `draw_standard_normal_vec_into_cached`.
fn did_methods_cached(rng: &mut Stream, out: &mut [f64]) {
    let mut index = 0usize;
    while index + 1 < out.len() {
        let u1 = rng.uniform().max(f64::MIN_POSITIVE);
        let u2 = rng.uniform();
        let radius = (-2.0_f64 * u1.ln()).sqrt();
        let theta = 2.0_f64 * PI * u2;
        out[index] = radius * theta.cos();
        out[index + 1] = radius * theta.sin();
        index += 2;
    }
    if index < out.len() {
        out[index] = did_methods_normal(rng);
    }
}

/// did_methods `inference/mod.rs`, `fill_rademacher_signs`.
fn did_methods_signs(stream: &mut Stream, signs: &mut [f64]) {
    for chunk in signs.chunks_mut(64) {
        let mut bits = stream.next_u64();
        for sign in chunk {
            *sign = if bits & 1 == 1 { 1.0 } else { -1.0 };
            bits >>= 1;
        }
    }
}

/// kernels `bootstrap.rs`, `cluster_weight_scale`.
fn kernels_cluster_weight_scale(n_clusters: usize, seed: u64) -> f64 {
    let mut stream = Stream::seeded(seed);
    let mut sum = 0.0f64;
    for _ in 0..n_clusters {
        sum += stream.exp1();
    }
    if sum > 0.0 {
        n_clusters as f64 / sum
    } else {
        1.0
    }
}

#[test]
fn normals_are_the_consumers_draws_to_the_bit() {
    for seed in [0, 1, 99, u64::MAX] {
        let mut a = Stream::seeded(seed);
        let mut b = Stream::seeded(seed);
        for _ in 0..500 {
            assert_eq!(a.normal().to_bits(), did_methods_normal(&mut b).to_bits());
        }
        assert_eq!(a, b);
        for len in [0, 1, 2, 7, 64, 65] {
            let mut x = vec![0.0; len];
            let mut y = vec![0.0; len];
            let mut a = Stream::seeded(seed);
            let mut b = Stream::seeded(seed);
            a.fill_normals(&mut x);
            did_methods_cached(&mut b, &mut y);
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(a, b);
        }
    }
}

#[test]
fn signs_are_the_consumers_signs_to_the_bit() {
    for len in [0, 1, 63, 64, 65, 200] {
        let mut x = vec![0.0; len];
        let mut y = vec![0.0; len];
        let mut a = Stream::seeded(5);
        let mut b = Stream::seeded(5);
        a.fill_signs(&mut x);
        did_methods_signs(&mut b, &mut y);
        assert_eq!(
            x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(a, b);
    }
}

#[test]
fn the_dirichlet_scale_is_the_kernels_copy_and_the_weights_scale() {
    for (n, seed) in [(0, 1), (1, 1), (10, 2), (1_000, 3)] {
        assert_eq!(
            dirichlet_scale(n, seed).to_bits(),
            kernels_cluster_weight_scale(n, seed).to_bits()
        );
        let mut weights = Vec::new();
        dirichlet_weights(n, seed, &mut weights);
        let mut stream = Stream::seeded(seed);
        let scale = dirichlet_scale(n, seed);
        for &weight in &weights {
            assert_eq!(weight.to_bits(), (stream.exp1() * scale).to_bits());
        }
    }
}

/// epistat-stats `bias.rs`, `binomial`, as written there.
fn stats_bias_binomial(stream: &mut Stream, mut n: u64, mut p: f64) -> Option<u64> {
    if !(0.0..=1.0).contains(&p) {
        return None;
    }
    let mut k = 0_u64;
    while n > 10 {
        let half = 1 + n / 2;
        let x = stream.beta(half as f64, (n + 1 - half) as f64).ok()?;
        if x >= p {
            n = half - 1;
            p /= x;
        } else {
            k += half;
            n -= half;
            p = (p - x) / (1.0 - x);
        }
    }
    for _ in 0..n {
        if stream.uniform() < p {
            k += 1;
        }
    }
    Some(k)
}

#[test]
fn binomial_draws_are_the_bias_analysis_draws() {
    let mut ours = Stream::seeded(23);
    let mut theirs = Stream::seeded(23);
    for (n, p) in [
        (0_u64, 0.5),
        (5, 0.0),
        (5, 1.0),
        (7, 0.4),
        (30, 0.1),
        (1_000, 0.3),
        (1_000_000, 0.02),
        (u64::from(u32::MAX), 0.999),
    ] {
        for _ in 0..200 {
            assert_eq!(
                Some(ours.binomial(n, p).unwrap()),
                stats_bias_binomial(&mut theirs, n, p)
            );
        }
        assert_eq!(ours, theirs);
    }
    let before = ours.clone();
    for p in [-0.1, 1.5, f64::NAN] {
        assert!(matches!(
            ours.binomial(5, p),
            Err(Error::Input(InputError::Domain { what: "p", .. }))
        ));
    }
    assert_eq!(ours, before);
}

#[test]
fn binomial_has_its_mean_and_variance() {
    let mut stream = Stream::seeded(7);
    for (n, p) in [(7_u64, 0.4), (30, 0.1), (1_000, 0.3), (1_000_000, 0.02)] {
        let sample: Vec<f64> = (0..20_000)
            .map(|_| stream.binomial(n, p).unwrap() as f64)
            .collect();
        assert!(sample.iter().all(|&k| k <= n as f64));
        check(&sample, n as f64 * p, n as f64 * p * (1.0 - p));
    }
}
