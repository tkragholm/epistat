//! Fingerprints of the estimators that `fit_golden` does not cover: the bootstrap and
//! its intervals, model comparison and selection with cross-validation and the Park
//! test, the elastic net, the compact designs, the clustered and prevalidated entry
//! points, and the MTP sampler with its convergence and posterior predictive summaries.
//!
//! Each entry is a testkit fingerprint over the bit patterns of every number the call
//! returns. The hashes were recorded on aarch64 macOS, for the reason `fit_golden`
//! gives, and the test asserts them only on that target. Elsewhere it computes them
//! and, when `EPISTAT_HASH_OUT` is set, appends them to that file. Print them with
//! `GOLDEN_PRINT=1 cargo test --test estimator_fingerprints -- --nocapture`.
use epistat_core::{Bootstrap, Interval, Level};
use epistat_linalg::Mat;
use epistat_rng::Stream as Draws;
use epistat_semicontinuous::compare::{CompareOptions, ModelComparison, ModelFitMetrics};
use epistat_semicontinuous::lognormal::LogNormalOptions;
use epistat_semicontinuous::mtp::{
    FamilyRandomEffects, MtpOptions, MtpPosteriorSummary, ParameterSummary, autocorrelation,
    effective_sample_size, posterior_predictive_summary,
};
use epistat_semicontinuous::prepared::{
    PreparedGammaDesign, PreparedPredictionDesign, PreparedScratch, PreparedTwoPartDesign,
};
use epistat_semicontinuous::tweedie::{TweedieOptions, TweedieScratch};
use epistat_semicontinuous::two_part::{
    Regularization, TwoPartModel, TwoPartOptions, TwoPartWarmStart,
};
use epistat_semicontinuous::{
    CostData, LongitudinalData, compare, lognormal, mtp, tweedie, two_part,
};
use epistat_testkit::Fingerprint;

const N_ROWS: usize = 2_000;
const N_COLS: usize = 6;

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
#[derive(Clone)]
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
            data = data.weights(weights);
        }
        if let Some(clusters) = &self.clusters {
            data = data.clusters(clusters);
        }
        data.build().expect("data")
    }
}

/// An intercept, three dummies and two continuous columns, a gamma-like cost with
/// about 35% zeros, positive weights and clusters of one to four rows, each a code
/// of its own.
fn fixture(seed: u64) -> Fixture {
    let mut stream = Stream(seed);
    let mut x = Mat::<f64>::zeros(N_ROWS, N_COLS);
    let mut y = vec![0.0; N_ROWS];
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        for j in 1..4 {
            x[(i, j)] = f64::from(stream.uniform() < 0.15 * j as f64);
        }
        for j in 4..N_COLS {
            x[(i, j)] = 2.0 * stream.uniform() - 1.0;
        }
        let eta = 0.3 * x[(i, 1)] - 0.2 * x[(i, 2)] + 0.25 * x[(i, 4)] - 0.1 * x[(i, 5)];
        let zero = stream.uniform() < 0.35 - 0.1 * x[(i, 3)];
        let noise = -stream.uniform().max(1e-12).ln() - stream.uniform().max(1e-12).ln();
        if !zero {
            y[i] = (6.0 + eta).exp() / 2.0 * noise + 1.0;
        }
    }
    let mut clusters = Vec::with_capacity(N_ROWS);
    let mut cluster = 0_u32;
    while clusters.len() < N_ROWS {
        let size = 1 + usize::try_from(stream.next_u64() % 4).expect("small");
        for _ in 0..size.min(N_ROWS - clusters.len()) {
            clusters.push(cluster);
        }
        cluster += 1;
    }
    let weights = (0..N_ROWS).map(|_| 0.5 + stream.uniform()).collect();
    Fixture {
        x,
        y,
        weights: Some(weights),
        clusters: Some(clusters),
    }
}

fn unweighted(input: &Fixture) -> Fixture {
    Fixture {
        weights: None,
        clusters: None,
        ..input.clone()
    }
}

fn positive_rows(input: &Fixture) -> Fixture {
    let rows: Vec<usize> = (0..input.y.len()).filter(|&i| input.y[i] > 0.0).collect();
    let x = Mat::from_fn(rows.len(), input.x.ncols(), |i, j| input.x[(rows[i], j)]);
    let y = rows.iter().map(|&i| input.y[i]).collect();
    Fixture {
        x,
        y,
        weights: None,
        clusters: None,
    }
}

/// A panel and its subject codes, to build the data from.
struct Panel {
    outcome: Vec<f64>,
    design: Mat<f64>,
    ids: Vec<u32>,
    time: Vec<f64>,
}

impl Panel {
    fn data(&self) -> LongitudinalData<'_> {
        LongitudinalData::builder(
            &self.outcome,
            self.design.as_ref(),
            self.design.as_ref(),
            &self.ids,
            &self.time,
        )
        .build()
        .expect("data")
    }
}

/// A panel of subjects over periods, a logistic zero part and a log-normal positive
/// part with subject intercepts.
fn panel(subjects: usize, periods: usize, seed: u64) -> Panel {
    let rows = subjects * periods;
    let mut draws = Draws::seeded(seed);
    let mut outcome = Vec::with_capacity(rows);
    let mut design = Vec::with_capacity(rows * 2);
    let mut ids = Vec::with_capacity(rows);
    let mut time = Vec::with_capacity(rows);
    for subject in 0..subjects {
        let level = 0.6 * draws.normal();
        for period in 0..periods {
            let t = 2.0 * period as f64 / (periods - 1) as f64 - 1.0;
            let probability = 1.0 / (1.0 + (0.35 - 0.8 * t).exp());
            let y = if draws.uniform() < probability {
                (1.2 + 0.55 * t + level + 0.6 * draws.normal()).exp()
            } else {
                0.0
            };
            outcome.push(y);
            design.extend_from_slice(&[1.0, t]);
            ids.push(u32::try_from(subject + 1).expect("small"));
            time.push(period as f64);
        }
    }
    Panel {
        outcome,
        design: Mat::from_fn(rows, 2, |i, j| design[2 * i + j]),
        ids,
        time,
    }
}

struct Hasher(Fingerprint);

impl Hasher {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, word: u64) {
        self.0 = std::mem::take(&mut self.0).u64(word);
    }

    fn f(&mut self, value: f64) {
        self.word(value.to_bits());
    }

    fn n(&mut self, value: usize) {
        self.word(value as u64);
    }

    fn of(&mut self, value: Option<f64>) {
        match value {
            Some(value) => self.f(value),
            None => self.word(u64::MAX),
        }
    }

    fn mat(&mut self, m: &Mat<f64>) {
        self.n(m.nrows());
        self.n(m.ncols());
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                self.f(m[(i, j)]);
            }
        }
    }

    fn opt(&mut self, m: Option<&Mat<f64>>) {
        match m {
            Some(m) => self.mat(m),
            None => self.word(u64::MAX),
        }
    }

    /// A limit that is `None` hashes as NaN, its value before it was undefined.
    fn intervals(&mut self, intervals: &[Interval]) {
        self.n(intervals.len());
        for interval in intervals {
            self.f(interval.lower.unwrap_or(f64::NAN));
            self.f(interval.upper.unwrap_or(f64::NAN));
        }
    }

    fn metrics(&mut self, metrics: &ModelFitMetrics) {
        for value in [
            metrics.rmse,
            metrics.mae,
            metrics.rmsle,
            metrics.r2,
            metrics.deviance,
        ] {
            self.f(value);
        }
    }

    fn two_part(&mut self, model: &TwoPartModel) {
        let report = &model.report;
        self.mat(&model.beta_logit);
        self.mat(&model.beta_gamma);
        self.opt(report.se_logit.as_ref());
        self.opt(report.se_gamma.as_ref());
        self.opt(report.cov_logit.as_ref());
        self.opt(report.cov_gamma.as_ref());
        self.n(report.iterations_logit);
        self.n(report.iterations_gamma);
        self.n(report.meta.fallback_attempts);
    }

    fn comparison(&mut self, comparison: &ModelComparison) {
        for table in [
            &comparison.in_sample,
            &comparison.cv_summary,
            &comparison.cv_ranking,
        ] {
            self.n(table.len());
            for score in table {
                self.word(score.candidate.label().len() as u64);
                self.metrics(&score.metrics);
            }
        }
        for table in [&comparison.information_criteria, &comparison.tweedie_ic] {
            self.n(table.len());
            for row in table {
                self.f(row.loglik);
                self.f(row.aic);
                self.f(row.bic);
            }
        }
        for row in &comparison.tweedie_cv_ranking {
            self.f(row.power);
            self.metrics(&row.metrics);
            self.f(row.aic.unwrap_or(f64::NAN));
            self.f(row.bic.unwrap_or(f64::NAN));
        }
    }

    fn parameter(&mut self, parameter: &ParameterSummary) {
        for value in [
            parameter.mean,
            parameter.std_dev,
            parameter.q025,
            parameter.q50,
            parameter.q975,
        ] {
            // Every summary of these fits is defined.
            self.f(value.expect("a defined summary"));
        }
    }

    fn posterior(&mut self, summary: Option<&MtpPosteriorSummary>) {
        let Some(summary) = summary else {
            self.word(u64::MAX);
            return;
        };
        for parameter in summary.alpha.iter().chain(&summary.beta) {
            self.parameter(parameter);
        }
        for parameter in [summary.kappa.as_ref(), summary.omega_sq.as_ref()] {
            match parameter {
                Some(parameter) => self.parameter(parameter),
                None => self.word(u64::MAX),
            }
        }
        self.n(summary.draw_count);
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

type Golden = Vec<(&'static str, u64)>;

fn bootstrap(out: &mut Golden, input: &Fixture) {
    let result = two_part::bootstrap(
        &input.data(),
        &TwoPartOptions::default(),
        Bootstrap::new(25, 7),
        Some(50),
    )
    .expect("bootstrap");
    let mut h = Hasher::new();
    for beta in result.betas_logit.iter().chain(&result.betas_gamma) {
        h.mat(beta);
    }
    h.n(result.failures);
    out.push(("bootstrap/draws", h.finish()));

    let mut h = Hasher::new();
    for betas in [&result.betas_logit, &result.betas_gamma] {
        let summary = two_part::summary(betas, Level::alpha(0.05).expect("level"));
        h.mat(&summary.mean);
        h.mat(&summary.se);
        h.intervals(&summary.intervals);
        h.intervals(&two_part::percentile_intervals(
            betas,
            Level::alpha(0.1).expect("level"),
        ));
    }
    out.push(("bootstrap/summary", h.finish()));
}

fn two_part_variants(out: &mut Golden, input: &Fixture) -> TwoPartWarmStart {
    let data = input.data();
    let robust = TwoPartOptions {
        robust_se: true,
        ..TwoPartOptions::default()
    };
    let model = two_part::fit(&data, &robust, None).expect("clustered");
    let mut h = Hasher::new();
    h.two_part(&model);
    let cov = model.report.cov_gamma.as_ref().expect("covariance");
    h.intervals(
        &two_part::wald_intervals(&model.beta_gamma, cov, Level::alpha(0.05).expect("level"))
            .expect("intervals"),
    );
    out.push(("two_part/clustered/robust", h.finish()));

    let elastic = TwoPartOptions {
        regularization: Regularization::ElasticNet {
            lambda: 0.02,
            alpha: 0.5,
            exclude_intercept: true,
        },
        ..TwoPartOptions::default()
    };
    let model = two_part::fit(&unweighted(input).data(), &elastic, None).expect("elastic net");
    let mut h = Hasher::new();
    h.two_part(&model);
    out.push(("two_part/elastic_net", h.finish()));

    let stable = TwoPartOptions::stable_defaults();
    let model = two_part::fit(&data, &stable, None).expect("stable");
    let iterations_logit = model.report.iterations_logit;
    let warm = TwoPartWarmStart {
        beta_logit: model.beta_logit.clone(),
        beta_gamma: model.beta_gamma.clone(),
    };
    let replicate = TwoPartOptions {
        compute_covariance: false,
        ..stable
    };
    let model = two_part::fit(&data, &replicate, Some(&warm)).expect("prevalidated");
    let mut h = Hasher::new();
    h.n(iterations_logit);
    h.two_part(&model);
    out.push(("two_part/prevalidated/warm", h.finish()));
    warm
}

fn compact(out: &mut Golden, input: &Fixture, warm: &TwoPartWarmStart) {
    let x = &input.x;
    let y = &input.y;
    let mut stream = Stream(99);
    let weights: Vec<f64> = (0..N_ROWS)
        .map(|_| -stream.uniform().max(1e-12).ln())
        .collect();
    let mut scratch = PreparedScratch::default();

    let positive = positive_rows(input);
    let gamma_options = TweedieOptions::default();
    let positive_data = positive.data();
    let gamma = tweedie::fit(&positive_data, 2.0, &gamma_options, None, None).expect("gamma");
    let design = PreparedGammaDesign::new(positive.x.as_ref(), &positive.y).expect("gamma design");
    let positive_weights = &weights[..positive.y.len()];
    let fit = design
        .fit(positive_weights, &gamma.beta, &gamma_options, &mut scratch)
        .expect("compact gamma");
    let mut h = Hasher::new();
    h.mat(&fit.beta);
    h.n(fit.iterations);
    let mut tweedie_scratch = TweedieScratch::default();
    let weighted = positive_data
        .with_weights(positive_weights)
        .expect("weights");
    let dense = tweedie::fit(
        &weighted,
        2.0,
        &gamma_options,
        Some(&gamma.beta),
        Some(&mut tweedie_scratch),
    )
    .expect("dense gamma");
    h.mat(&dense.beta);
    let prediction = PreparedPredictionDesign::new(positive.x.as_ref()).expect("prediction design");
    h.f(prediction.mean_gamma(&fit.beta, positive_weights));
    out.push(("compact/gamma", h.finish()));

    let options = TwoPartOptions {
        compute_covariance: false,
        ..TwoPartOptions::default()
    };
    let design = PreparedTwoPartDesign::new(x.as_ref(), y).expect("two-part design");
    let fit = design
        .fit(&weights, warm, &options, &mut scratch)
        .expect("compact two-part");
    let mut h = Hasher::new();
    h.mat(&fit.beta_logit);
    h.mat(&fit.beta_gamma);
    let prediction = PreparedPredictionDesign::new(x.as_ref()).expect("prediction design");
    h.f(prediction.mean_two_part(&fit.beta_logit, &fit.beta_gamma, &weights));
    out.push(("compact/two_part", h.finish()));
}

fn selection(out: &mut Golden, input: &Fixture) {
    let plain = unweighted(input);
    let plain = plain.data();
    let comparison = compare::compare(&plain, &CompareOptions::new(42)).expect("comparison");
    let mut h = Hasher::new();
    h.comparison(&comparison);
    out.push(("comparison/default", h.finish()));

    let options = CompareOptions {
        two_part_elastic_net: Some(TwoPartOptions {
            regularization: Regularization::ElasticNet {
                lambda: 0.01,
                alpha: 0.8,
                exclude_intercept: true,
            },
            ..TwoPartOptions::default()
        }),
        ..CompareOptions::new(42)
    };
    let comparison = compare::compare(&plain, &options).expect("comparison");
    let mut h = Hasher::new();
    h.comparison(&comparison);
    out.push(("comparison/elastic_net", h.finish()));

    let selected = compare::select(&plain, &[1.2, 1.5, 1.8]).expect("selection");
    let mut h = Hasher::new();
    // A Park test that fails hashes as intercept 0 and slope 2, the values it
    // was replaced by before it was undefined.
    h.f(selected.park_test.map_or(0.0, |park| park.intercept));
    h.f(selected.park_test.map_or(2.0, |park| park.slope));
    for candidate in &selected.tweedie_candidates {
        h.mat(&candidate.model.beta);
        h.metrics(&candidate.metrics);
        h.f(candidate.information_criteria.loglik);
        h.f(candidate.information_criteria.aic);
        h.f(candidate.information_criteria.bic);
    }
    if let Some(candidate) = &selected.lognormal_candidate {
        h.mat(&candidate.model.beta);
        h.f(candidate.model.smearing_factor);
        h.metrics(&candidate.metrics);
    }
    out.push(("selection", h.finish()));

    let model = lognormal::fit(
        &input.data().without_weights(),
        &LogNormalOptions::default(),
    )
    .expect("ln");
    let report = &model.report;
    let mut h = Hasher::new();
    h.mat(&model.beta);
    h.f(model.smearing_factor);
    h.opt(report.se.as_ref());
    out.push(("lognormal/weighted", h.finish()));
}

fn mtp(out: &mut Golden) {
    let panel = panel(80, 4, 42);
    let input = panel.data();
    let options = MtpOptions {
        iterations: 800,
        burn_in: 200,
        thin: 2,
        family_random_effects: FamilyRandomEffects::Disabled,
        chains: 3,
        keep_draws: true,
        ..MtpOptions::new(11)
    };
    let model = mtp::fit(&input, &options).expect("mtp");
    let chains = model.draws.as_ref().expect("draws");
    let mut h = Hasher::new();
    for chain in chains {
        h.n(chain.draws.len());
        for draw in &chain.draws {
            for &value in draw.alpha.iter().chain(&draw.beta) {
                h.f(value);
            }
            h.f(draw.kappa);
            h.f(draw.omega_sq);
        }
    }
    out.push(("mtp/draws", h.finish()));

    let mut h = Hasher::new();
    let convergence = model.convergence.as_ref().expect("convergence");
    h.n(convergence.chain_count);
    h.n(convergence.draws_per_chain_used);
    for &value in convergence
        .alpha_split_rhat
        .iter()
        .chain(&convergence.beta_split_rhat)
    {
        h.of(value);
    }
    h.of(convergence.kappa_split_rhat);
    h.of(convergence.omega_sq_split_rhat);
    h.of(convergence.max_split_rhat);
    h.posterior(model.posterior_summary.as_ref());
    for chain in &model.chains {
        h.n(chain.diagnostics.iterations_completed);
        h.n(chain.diagnostics.retained_draws);
        if let Some(rates) = chain.diagnostics.acceptance_rates {
            for value in [
                rates.alpha,
                rates.beta,
                rates.random_effects,
                rates.family_effects,
                rates.kappa,
                rates.omega_sq,
            ] {
                // A rate without proposals hashes as zero, its value before it was undefined.
                h.f(value.unwrap_or(0.0));
            }
        }
        h.posterior(chain.posterior_summary.as_ref());
    }
    out.push(("mtp/convergence", h.finish()));

    let summary = posterior_predictive_summary(&chains[0], &input, 5).expect("predictive");
    let mut h = Hasher::new();
    h.f(summary.observed_zero_rate);
    h.f(summary.observed_mean);
    for interval in [
        summary.predicted_zero_rate,
        summary.predicted_mean,
        summary.brier_score,
    ] {
        for value in [interval.mean, interval.q025, interval.q50, interval.q975] {
            // Every summary of this fit is defined.
            h.f(value.expect("a defined summary"));
        }
    }
    for bin in &summary.calibration_bins {
        h.n(bin.count);
        h.f(bin.lower);
        h.f(bin.upper);
        h.f(bin.mean_predicted_probability);
        h.f(bin.observed_positive_rate);
    }
    out.push(("mtp/posterior_predictive", h.finish()));

    let series: Vec<f64> = chains[0].draws.iter().map(|draw| draw.beta[1]).collect();
    let mut h = Hasher::new();
    for lag in 1..6 {
        h.of(autocorrelation(&series, lag));
    }
    h.of(effective_sample_size(&series));
    out.push(("mtp/autocorrelation", h.finish()));
}

fn collect() -> Golden {
    let mut out = Golden::new();
    let input = fixture(20_261_122);
    bootstrap(&mut out, &unweighted(&input));
    let warm = two_part_variants(&mut out, &input);
    compact(&mut out, &input, &warm);
    selection(&mut out, &input);
    mtp(&mut out);
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("bootstrap/draws", 0x1665_116b_57fa_c17c),
    ("bootstrap/summary", 0xda0e_70c4_2c70_b7ca),
    ("two_part/clustered/robust", 0x6a61_5d52_a469_116e),
    ("two_part/elastic_net", 0x7aa9_a058_c6c7_73e0),
    ("two_part/prevalidated/warm", 0x45cb_4451_1f4c_3630),
    ("compact/gamma", 0xfbca_6c2d_3c43_fbf8),
    ("compact/two_part", 0xbc06_d03e_59d9_f18a),
    ("comparison/default", 0xa7aa_bf74_1b04_d9aa),
    ("comparison/elastic_net", 0xa7bb_6818_a8e4_6428),
    ("selection", 0x3359_3df6_d30b_6785),
    ("lognormal/weighted", 0x22f8_bede_7fbf_6966),
    ("mtp/draws", 0xf090_2e04_a16e_a2fd),
    ("mtp/convergence", 0x062b_db42_0fa8_775c),
    ("mtp/posterior_predictive", 0x2a81_8f18_ff31_a831),
    ("mtp/autocorrelation", 0xe7e8_46cf_e5a1_e143),
];

#[test]
fn estimators_match_the_recorded_bit_patterns() {
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
