//! Fits must be bit-identical across repeated calls and across thread counts.
//!
//! The robust covariance of a poorly identified model amplifies any change in
//! summation order, so a reduction whose order depends on thread scheduling or
//! on hash-map iteration order shows up as standard errors that differ between
//! runs. These tests fit each model several times, in the calling thread and in
//! rayon pools of 1, 2 and 8 threads, and compare every coefficient and every
//! standard error bit for bit.

use epistat_core::CaseWeights;
use epistat_linalg::Mat;
use epistat_semicontinuous::lognormal::LogNormalOptions;
use epistat_semicontinuous::tweedie::TweedieOptions;
use epistat_semicontinuous::two_part::TwoPartOptions;
use epistat_semicontinuous::{CostData, Variance, lognormal, tweedie, two_part};
use epistat_testkit::on_threads;

const N_ROWS: usize = 12_000;
const N_COLS: usize = 8;
const REPEATS: usize = 2;
const THREAD_COUNTS: [usize; 3] = [1, 2, 8];

/// SplitMix64, so the fixture does not depend on an RNG crate's stream.
struct Stream(u64);

impl Stream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// A cost-shaped fixture: intercept, a binary exposure, continuous covariates,
/// about 40% zeros, right-skewed positives, positive weights, and family-sized
/// clusters of one to four rows.
struct Fixture {
    x: Mat<f64>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    clusters: Vec<u32>,
}

impl Fixture {
    fn data(&self) -> CostData<'_> {
        let mut data = CostData::builder(self.x.as_ref(), &self.y).clusters(&self.clusters);
        if let Some(weights) = &self.weights {
            data = data.weights(CaseWeights::new(weights).expect("weights"));
        }
        data.build().expect("data")
    }
}

fn fixture(weighted: bool) -> Fixture {
    let mut stream = Stream(20_261_001);
    let mut x = Mat::<f64>::zeros(N_ROWS, N_COLS);
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = if stream.uniform() < 0.3 { 1.0 } else { 0.0 };
        for j in 2..N_COLS {
            x[(i, j)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        }
    }
    let mut y = vec![0.0; N_ROWS];
    for i in 0..N_ROWS {
        let eta = 0.4f64.mul_add(x[(i, 1)], 0.2 * x[(i, 2)] - 0.1 * x[(i, 3)]);
        if stream.uniform() < 0.4 {
            continue;
        }
        let noise = -stream.uniform().max(1e-12).ln();
        y[i] = 1_000.0 * (8.0f64 + eta).exp() / 3_000.0 * noise;
    }
    let mut clusters = Vec::with_capacity(N_ROWS);
    let mut cluster = 0_u32;
    while clusters.len() < N_ROWS {
        let size = 1 + (stream.next_u64() % 4) as usize;
        for _ in 0..size.min(N_ROWS - clusters.len()) {
            clusters.push(cluster);
        }
        cluster += 1;
    }
    let weights = weighted.then(|| (0..N_ROWS).map(|_| 0.5 + stream.uniform()).collect());
    Fixture {
        x,
        y,
        weights,
        clusters,
    }
}

/// The bits of the coefficients and of the standard errors, each defined.
fn bits(coefficients: &[&[f64]], se: &[&[Option<f64>]]) -> Vec<u64> {
    coefficients
        .iter()
        .flat_map(|values| values.iter().map(|v| v.to_bits()))
        .chain(
            se.iter()
                .flat_map(|values| values.iter().map(|v| v.expect("standard error").to_bits())),
        )
        .collect()
}

fn two_part_bits(data: &CostData<'_>) -> Vec<u64> {
    let options = TwoPartOptions {
        variance: Variance::Sandwich,
        ..TwoPartOptions::default()
    };
    let model = two_part::fit(data, &options, None).expect("two-part fit");
    bits(
        &[&model.logit.coefficients, &model.gamma.coefficients],
        &[&model.logit.se, &model.gamma.se],
    )
}

fn tweedie_bits(data: &CostData<'_>, power: f64) -> Vec<u64> {
    let options = TweedieOptions {
        variance: Variance::Sandwich,
        ..TweedieOptions::default()
    };
    let model = tweedie::fit(data, power, &options, None, None).expect("tweedie fit");
    bits(&[&model.coefficients], &[&model.se])
}

fn lognormal_bits(data: &CostData<'_>) -> Vec<u64> {
    let options = LogNormalOptions {
        variance: Variance::Sandwich,
        ..LogNormalOptions::default()
    };
    let model = lognormal::fit(data, &options).expect("lognormal fit");
    bits(&[&model.coefficients], &[&model.se])
}

fn assert_reproducible(label: &str, fit: impl Fn() -> Vec<u64> + Sync) {
    let reference = fit();
    for repeat in 0..REPEATS {
        assert_eq!(fit(), reference, "{label}: repeat {repeat} differs");
    }
    let runs = on_threads(&THREAD_COUNTS, &fit);
    for (run, threads) in runs.iter().zip(THREAD_COUNTS) {
        assert_eq!(*run, reference, "{label}: {threads} threads differs");
    }
}

#[test]
fn two_part_fits_are_bit_identical() {
    for weighted in [false, true] {
        let fixture = fixture(weighted);
        let clustered = fixture.data();
        let plain = clustered.without_clusters();
        assert_reproducible("two-part clustered", || two_part_bits(&clustered));
        assert_reproducible("two-part", || two_part_bits(&plain));
    }
}

#[test]
fn tweedie_fits_are_bit_identical() {
    for weighted in [false, true] {
        let fixture = fixture(weighted);
        let clustered = fixture.data();
        let plain = clustered.without_clusters();
        for power in [1.5, 2.0] {
            assert_reproducible("tweedie clustered", || tweedie_bits(&clustered, power));
            assert_reproducible("tweedie", || tweedie_bits(&plain, power));
        }
    }
}

#[test]
fn lognormal_fits_are_bit_identical() {
    for weighted in [false, true] {
        let fixture = fixture(weighted);
        let clustered = fixture.data();
        let plain = clustered.without_clusters();
        assert_reproducible("lognormal clustered", || lognormal_bits(&clustered));
        assert_reproducible("lognormal", || lognormal_bits(&plain));
    }
}
