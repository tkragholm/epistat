//! Golden outputs of the fitters, pinned as exact f64 bit patterns.
//!
//! Each entry is an FNV-1a hash over the bit patterns of every coefficient,
//! standard error, covariance entry or prediction a fit returns, plus the
//! iteration counts. Any change to the order of floating-point operations in
//! the fitters changes a hash.
//!
//! The hashes were recorded on aarch64 macOS. The libm `exp` and `ln` and the
//! gemm kernels faer selects differ between CPUs, so the test asserts them only
//! on that target. Elsewhere it computes them and, when `EPISTAT_HASH_OUT` is
//! set, appends them to that file. Regenerate with
//! `GOLDEN_PRINT=1 cargo test --test fit_golden -- --nocapture` after a change
//! that is meant to alter results.
use epistat_core::{CaseWeights, Covariance};
use epistat_linalg::Mat;
use epistat_semicontinuous::lognormal::LogNormalOptions;
use epistat_semicontinuous::tweedie::TweedieOptions;
use epistat_semicontinuous::two_part::{TwoPartOptions, TwoPartWarmStart};
use epistat_semicontinuous::{CostData, Model, Variance, lognormal, tweedie, two_part};
use epistat_testkit::Fingerprint;

const N_ROWS: usize = 10_000;
const N_COLS: usize = 12;

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
    weights: Option<Vec<f64>>,
    clusters: Option<Vec<u32>>,
}

impl Fixture {
    fn data(&self) -> CostData<'_> {
        let mut data = CostData::builder(self.x.as_ref(), &self.y);
        if let Some(weights) = &self.weights {
            data = data.weights(CaseWeights::new(weights).expect("weights"));
        }
        if let Some(clusters) = &self.clusters {
            data = data.clusters(clusters);
        }
        data.build().expect("data")
    }
}

/// Intercept, eight dummies, three continuous columns; about 40% zeros when
/// `zeros` is set; positive weights and family-sized clusters, each family a
/// code of its own.
fn fixture(zeros: bool, weighted: bool, clustered: bool) -> Fixture {
    let mut stream = Stream(20_261_028);
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
    let mut input = Fixture {
        x,
        y,
        weights: None,
        clusters: None,
    };
    if clustered {
        let mut clusters = Vec::with_capacity(N_ROWS);
        let mut cluster = 0_u32;
        while clusters.len() < N_ROWS {
            let size = 1 + (stream.next_u64() % 4) as usize;
            for _ in 0..size.min(N_ROWS - clusters.len()) {
                clusters.push(cluster);
            }
            cluster += 1;
        }
        input.clusters = Some(clusters);
    }
    if weighted {
        input.weights = Some((0..N_ROWS).map(|_| 0.5 + stream.uniform()).collect());
    }
    input
}

/// Accumulates a testkit fingerprint through `&mut` calls and adds the matrix
/// layouts the fits return.
struct Hasher(Fingerprint);

impl Hasher {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, word: u64) {
        self.0 = std::mem::take(&mut self.0).u64(word);
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }

    fn mat(&mut self, m: &Mat<f64>) {
        self.word(m.nrows() as u64);
        self.word(m.ncols() as u64);
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                self.word(m[(i, j)].to_bits());
            }
        }
    }

    /// A vector, hashed as the `n x 1` matrix it was.
    fn vec(&mut self, v: &[f64]) {
        self.word(v.len() as u64);
        self.word(1);
        for &value in v {
            self.word(value.to_bits());
        }
    }

    /// A covariance, hashed as the matrix it was, column by column; `None`
    /// as a missing matrix.
    fn cov(&mut self, c: Option<&Covariance>) {
        let Some(c) = c else {
            self.word(u64::MAX);
            return;
        };
        self.word(c.dim() as u64);
        self.word(c.dim() as u64);
        for j in 0..c.dim() {
            for i in 0..c.dim() {
                self.word(c.get(i, j).unwrap_or(f64::NAN).to_bits());
            }
        }
    }

    /// Standard errors, hashed as the `n x 1` matrix they were, or as a
    /// missing matrix when the fit computed no covariance.
    fn se(&mut self, se: &[Option<f64>], covariance: Option<&Covariance>) {
        if covariance.is_none() {
            self.word(u64::MAX);
            return;
        }
        let se: Vec<f64> = se.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
        self.vec(&se);
    }
}

type Golden = Vec<(&'static str, u64)>;

fn two_part(
    out: &mut Golden,
    name: &'static str,
    input: &Fixture,
    options: TwoPartOptions,
    warm: Option<&TwoPartWarmStart>,
) -> TwoPartWarmStart {
    let model = two_part::fit(&input.data(), &options, warm).expect(name);
    let (logit, gamma) = (&model.logit, &model.gamma);
    let mut h = Hasher::new();
    h.vec(&logit.coefficients);
    h.vec(&gamma.coefficients);
    h.se(&logit.se, logit.covariance.as_ref());
    h.se(&gamma.se, gamma.covariance.as_ref());
    h.cov(logit.covariance.as_ref());
    h.cov(gamma.covariance.as_ref());
    h.word(logit.meta.iterations as u64);
    h.word(gamma.meta.iterations as u64);
    h.word(model.report.meta.fallback_attempts as u64);
    let prediction = model.predict(input.x.as_ref()).expect("predict");
    h.vec(&prediction.prob_positive);
    h.vec(&prediction.mean_positive);
    h.vec(&prediction.mean);
    out.push((name, h.finish()));
    TwoPartWarmStart::from(&model)
}

fn tweedie(
    out: &mut Golden,
    name: &'static str,
    input: &Fixture,
    power: f64,
    options: TweedieOptions,
    warm: Option<&[f64]>,
) -> Vec<f64> {
    let model = tweedie::fit(&input.data(), power, &options, warm, None).expect(name);
    let report = &model.report;
    let mut h = Hasher::new();
    h.vec(&model.coefficients);
    h.se(&model.se, model.covariance.as_ref());
    h.cov(model.covariance.as_ref());
    h.word(report.meta.iterations as u64);
    h.word(report.meta.fallback_attempts as u64);
    h.vec(&model.predict(input.x.as_ref()).expect("predict").mean);
    out.push((name, h.finish()));
    model.coefficients
}

fn collect() -> Golden {
    let mut out = Golden::new();

    // Two-part: robust clustered SEs, weighted and unweighted, plain model-based SEs,
    // ridge with the relaxed strategy, and bootstrap-style warm starts.
    let zeros_unweighted = fixture(true, false, true);
    let zeros_weighted = fixture(true, true, true);
    let zeros_plain = fixture(true, true, false);
    let model_based = TwoPartOptions {
        variance: Variance::ModelBased,
        ..TwoPartOptions::default()
    };
    two_part(
        &mut out,
        "two_part/unweighted/robust/clustered",
        &zeros_unweighted,
        TwoPartOptions {
            variance: Variance::Sandwich,
            ..TwoPartOptions::default()
        },
        None,
    );
    two_part(
        &mut out,
        "two_part/unweighted/model_se",
        &fixture(true, false, false),
        model_based,
        None,
    );
    two_part(
        &mut out,
        "two_part/weighted/model_se",
        &zeros_plain,
        model_based,
        None,
    );
    let warm = two_part(
        &mut out,
        "two_part/weighted/robust/clustered",
        &zeros_weighted,
        TwoPartOptions {
            variance: Variance::Sandwich,
            ..TwoPartOptions::default()
        },
        None,
    );
    let stable = TwoPartOptions {
        variance: Variance::ModelBased,
        ..TwoPartOptions::stable_defaults()
    };
    two_part(
        &mut out,
        "two_part/weighted/stable",
        &zeros_weighted,
        stable,
        None,
    );
    let replicate_options = TwoPartOptions {
        variance: Variance::None,
        ..stable
    };
    two_part(
        &mut out,
        "two_part/weighted/warm/no_cov",
        &zeros_plain,
        replicate_options,
        Some(&warm),
    );
    two_part(
        &mut out,
        "two_part/weighted/warm/model_se",
        &zeros_plain,
        model_based,
        Some(&warm),
    );

    // Tweedie: gamma (power 2) with cached factor, power 1.5 with per-iteration
    // weights, robust and not, warm and cold.
    let positive_weighted = fixture(false, true, true);
    let positive_plain = fixture(false, false, false);
    let robust = TweedieOptions {
        variance: Variance::Sandwich,
        ..TweedieOptions::default()
    };
    let plain = TweedieOptions {
        variance: Variance::None,
        ..TweedieOptions::default()
    };
    let cold = tweedie(
        &mut out,
        "tweedie2/weighted/robust/clustered",
        &positive_weighted,
        2.0,
        robust,
        None,
    );
    tweedie(
        &mut out,
        "tweedie2/unweighted/plain",
        &positive_plain,
        2.0,
        plain,
        None,
    );
    tweedie(
        &mut out,
        "tweedie2/weighted/warm",
        &positive_weighted,
        2.0,
        plain,
        Some(cold.as_slice()),
    );
    tweedie(
        &mut out,
        "tweedie2/weighted/ridge/warm/robust",
        &positive_weighted,
        2.0,
        TweedieOptions {
            l2_penalty: 1e-4,
            variance: Variance::Sandwich,
            ..TweedieOptions::default()
        },
        Some(cold.as_slice()),
    );
    tweedie(
        &mut out,
        "tweedie1.5/weighted/robust/clustered",
        &fixture(true, true, true),
        1.5,
        robust,
        None,
    );
    tweedie(
        &mut out,
        "tweedie1.5/unweighted/plain",
        &fixture(true, false, false),
        1.5,
        plain,
        None,
    );

    // Log-normal, which shares the Gram and matrix-vector helpers. It has no
    // prior weights, so it is fitted to the data without the fixture's.
    let model = lognormal::fit(
        &fixture(true, true, true).data().without_weights(),
        &LogNormalOptions {
            variance: Variance::Sandwich,
            ..LogNormalOptions::default()
        },
    )
    .expect("lognormal");
    let mut h = Hasher::new();
    h.vec(&model.coefficients);
    h.se(&model.se, model.covariance.as_ref());
    h.cov(model.covariance.as_ref());
    out.push(("lognormal/weighted/robust/clustered", h.finish()));

    // The plain-API wrappers used elsewhere.
    let model =
        two_part::fit(&zeros_plain.data(), &TwoPartOptions::default(), None).expect("two-part");
    let mut h = Hasher::new();
    h.vec(&model.logit.coefficients);
    h.vec(&model.gamma.coefficients);
    out.push(("two_part/input/default", h.finish()));
    let model = tweedie::fit(
        &positive_plain.data(),
        2.0,
        &TweedieOptions::default(),
        None,
        None,
    )
    .expect("tweedie");
    let mut h = Hasher::new();
    h.vec(&model.coefficients);
    out.push(("tweedie2/input/default", h.finish()));

    out
}

const GOLDEN: &[(&str, u64)] = &[
    (
        "two_part/unweighted/robust/clustered",
        0x005f_ca98_46ef_4ac6,
    ),
    ("two_part/unweighted/model_se", 0x7238_922a_7178_7077),
    ("two_part/weighted/model_se", 0x288f_d34e_b86d_1456),
    ("two_part/weighted/robust/clustered", 0x67d2_4eae_4970_d5b5),
    ("two_part/weighted/stable", 0x18cb_2aa0_4df1_f58e),
    ("two_part/weighted/warm/no_cov", 0xc925_fead_5cba_18f0),
    ("two_part/weighted/warm/model_se", 0xa901_12d7_dbe6_f2b5),
    ("tweedie2/weighted/robust/clustered", 0x00c1_506a_4a88_5bb7),
    ("tweedie2/unweighted/plain", 0x66ca_3ead_887a_50d8),
    ("tweedie2/weighted/warm", 0x1699_4634_4253_d936),
    ("tweedie2/weighted/ridge/warm/robust", 0xd152_0258_92ed_d8f1),
    (
        "tweedie1.5/weighted/robust/clustered",
        0x6c27_f84b_95f4_274a,
    ),
    ("tweedie1.5/unweighted/plain", 0x7366_d249_674b_0c28),
    ("lognormal/weighted/robust/clustered", 0xb9ba_cb98_7d7e_837d),
    ("two_part/input/default", 0x5044_3c25_6527_acf3),
    ("tweedie2/input/default", 0xfeca_4ccc_d30d_521c),
];

#[test]
fn fits_match_the_recorded_bit_patterns() {
    let actual = collect();
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
