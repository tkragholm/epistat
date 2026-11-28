//! The fits whose row blocks run on rayon give the same bits on any number of
//! threads.
//!
//! The results are pinned as bit patterns and must not depend on the rayon thread count
//! or on `RAYON_NUM_THREADS`.

use epistat_core::CaseWeights;
use epistat_linalg::Mat;
use epistat_semicontinuous::tweedie::TweedieOptions;
use epistat_semicontinuous::two_part::{TwoPartOptions, TwoPartWarmStart};
use epistat_semicontinuous::{CostData, Model, Variance, tweedie, two_part};
use epistat_testkit::{Fingerprint, on_threads};
use std::process::Command;

const N_ROWS: usize = 20_000;
const N_COLS: usize = 12;
const THREAD_COUNTS: [usize; 4] = [1, 2, 3, 12];

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

/// The columns of a fixture.
struct Fixture {
    x: Mat<f64>,
    y: Vec<f64>,
    weights: Vec<f64>,
    clusters: Vec<u32>,
}

impl Fixture {
    fn data(&self) -> CostData<'_> {
        CostData::builder(self.x.as_ref(), &self.y)
            .weights(CaseWeights::new(&self.weights).expect("weights"))
            .clusters(&self.clusters)
            .build()
            .expect("data")
    }
}

/// Intercept, eight dummies, three continuous columns; about 40% zeros when `zeros`
/// is set; positive weights and family-sized clusters, each family a code of its own.
fn fixture(zeros: bool) -> Fixture {
    let mut stream = Stream(20_261_029);
    let mut x = Mat::<f64>::zeros(N_ROWS, N_COLS);
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        for j in 1..9 {
            let prevalence = 0.05 + 0.05 * j as f64;
            x[(i, j)] = f64::from(stream.uniform() < prevalence);
        }
        for j in 9..N_COLS {
            x[(i, j)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        }
    }
    let mut y = vec![0.0; N_ROWS];
    for i in 0..N_ROWS {
        let eta = 0.4f64.mul_add(
            x[(i, 1)],
            0.2 * x[(i, 2)] - 0.1 * x[(i, 9)] + 0.15 * x[(i, 10)],
        );
        let is_zero = zeros && stream.uniform() < 0.4 - 0.1 * x[(i, 3)];
        let noise = -stream.uniform().max(1e-12).ln() - stream.uniform().max(1e-12).ln();
        if !is_zero {
            y[i] = (8.0f64 + eta).exp() / 2.0 * noise + 1.0;
        }
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
    let weights = (0..N_ROWS).map(|_| 0.5 + stream.uniform()).collect();
    Fixture {
        x,
        y,
        weights,
        clusters,
    }
}

/// Coefficients, standard errors and predictions of one fit, in that order.
struct Fit {
    name: &'static str,
    coefficients: Vec<f64>,
    standard_errors: Vec<f64>,
    predictions: Vec<f64>,
}

/// The standard errors, each defined.
fn defined(se: &[Option<f64>]) -> impl Iterator<Item = f64> + '_ {
    se.iter().map(|se| se.expect("se"))
}

fn collect() -> Vec<Fit> {
    let zeros = fixture(true);
    let positive = fixture(false);
    let mut out = Vec::new();

    let options = TwoPartOptions {
        variance: Variance::Sandwich,
        ..TwoPartOptions::default()
    };
    let model = two_part::fit(&zeros.data(), &options, None).expect("two-part");
    let prediction = model.predict(zeros.x.as_ref()).expect("predict");
    out.push(Fit {
        name: "two_part/robust/clustered",
        coefficients: [model.logit.coefficients.as_slice(), &model.gamma.coefficients].concat(),
        standard_errors: defined(&model.logit.se).chain(defined(&model.gamma.se)).collect(),
        predictions: prediction.mean,
    });
    let warm = TwoPartWarmStart::from(&model);

    // The data carry clusters, so the model-based covariance is asked for.
    let options = TwoPartOptions {
        variance: Variance::ModelBased,
        ..TwoPartOptions::default()
    };
    let model = two_part::fit(&zeros.data(), &options, None).expect("two-part");
    let prediction = model.predict(zeros.x.as_ref()).expect("predict");
    out.push(Fit {
        name: "two_part/model_se",
        coefficients: [model.logit.coefficients.as_slice(), &model.gamma.coefficients].concat(),
        standard_errors: defined(&model.logit.se).chain(defined(&model.gamma.se)).collect(),
        predictions: prediction.mean,
    });

    let options = TwoPartOptions {
        variance: Variance::None,
        ..TwoPartOptions::stable_defaults()
    };
    let model = two_part::fit(&zeros.data(), &options, Some(&warm)).expect("two-part");
    let prediction = model.predict(zeros.x.as_ref()).expect("predict");
    out.push(Fit {
        name: "two_part/warm/no_cov",
        coefficients: [model.logit.coefficients.as_slice(), &model.gamma.coefficients].concat(),
        standard_errors: Vec::new(),
        predictions: prediction.mean,
    });

    for (name, power, input) in [
        ("tweedie2/robust/clustered", 2.0, &positive),
        ("tweedie1.5/robust/clustered", 1.5, &zeros),
    ] {
        let options = TweedieOptions {
            variance: Variance::Sandwich,
            ..TweedieOptions::default()
        };
        let model = tweedie::fit(&input.data(), power, &options, None, None).expect(name);
        out.push(Fit {
            name,
            standard_errors: defined(&model.se).collect(),
            predictions: model.predict(input.x.as_ref()).expect("predict").mean,
            coefficients: model.coefficients,
        });
    }
    out
}

fn hash(fits: &[Fit]) -> Vec<(&'static str, u64)> {
    fits.iter()
        .map(|fit| {
            let hash = Fingerprint::new()
                .f64s(&fit.coefficients)
                .f64s(&fit.standard_errors)
                .f64s(&fit.predictions);
            (fit.name, hash.finish())
        })
        .collect()
}

#[test]
fn serial_fits_do_not_depend_on_the_rayon_thread_count() {
    let reference = hash(&collect());
    let runs = on_threads(&THREAD_COUNTS, || hash(&collect()));
    for (hashes, threads) in runs.iter().zip(THREAD_COUNTS) {
        assert_eq!(*hashes, reference, "{threads} threads differs");
    }
}

/// Prints the hashes; the parent test runs it as a child process under different
/// `RAYON_NUM_THREADS` values.
#[test]
#[ignore = "run by serial_fits_do_not_depend_on_rayon_num_threads"]
fn serial_hashes_child() {
    for (name, hash) in hash(&collect()) {
        println!("HASH {name} {hash:#018x}");
    }
}

#[test]
fn serial_fits_do_not_depend_on_rayon_num_threads() {
    let exe = std::env::current_exe().expect("test binary");
    let run = |threads: &str| -> String {
        let output = Command::new(&exe)
            .args(["--ignored", "--exact", "serial_hashes_child", "--nocapture"])
            .env("RAYON_NUM_THREADS", threads)
            .output()
            .expect("child test");
        assert!(output.status.success(), "child failed at {threads} threads");
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| line.starts_with("HASH "))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let reference = run("1");
    assert!(reference.lines().count() >= 5, "child printed no hashes");
    for threads in ["2", "5", "12"] {
        assert_eq!(
            run(threads),
            reference,
            "RAYON_NUM_THREADS={threads} differs"
        );
    }
    let in_process = hash(&collect())
        .iter()
        .map(|(name, h)| format!("HASH {name} {h:#018x}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(in_process, reference, "child and parent differ");
}

#[test]
fn serial_fits_match_the_recorded_bit_patterns() {
    const GOLDEN: &[(&str, u64)] = &[
        ("two_part/robust/clustered", 0x286c_106e_eaa6_2b85),
        ("two_part/model_se", 0xccd5_d65d_7055_f000),
        ("two_part/warm/no_cov", 0xa99b_e55f_83d3_9adf),
        ("tweedie2/robust/clustered", 0x342c_d209_c1b1_55a0),
        ("tweedie1.5/robust/clustered", 0x28a2_0d9c_3320_38af),
    ];
    let actual = hash(&collect());
    if std::env::var_os("GOLDEN_PRINT").is_some() {
        for (name, hash) in &actual {
            println!("    (\"{name}\", {hash:#018x}),");
        }
    }
    for (name, hash) in &actual {
        epistat_testkit::hash_out(name, *hash);
    }
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
    assert_eq!(actual.len(), GOLDEN.len(), "golden table is out of date");
    for ((name, hash), (golden_name, golden_hash)) in actual.iter().zip(GOLDEN) {
        assert_eq!(name, golden_name);
        assert_eq!(*hash, *golden_hash, "{name}: output bits changed");
    }
}
