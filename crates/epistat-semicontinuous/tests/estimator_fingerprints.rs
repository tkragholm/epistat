//! Fingerprints of the estimators that `fit_golden` does not cover: the bootstrap and
//! its intervals, model comparison and selection with cross-validation and the Park
//! test, the elastic net, the compact designs, the clustered and prevalidated entry
//! points, and the MTP sampler with its convergence and posterior predictive summaries.
//!
//! Each entry is a testkit fingerprint over the bit patterns of every number the call
//! returns. The hashes were recorded on aarch64 macOS, for the reason `fit_golden`
//! gives, and the test is compiled only on that target. Print them with
//! `GOLDEN_PRINT=1 cargo test --test estimator_fingerprints -- --nocapture`.
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use epistat_rng::Stream as Draws;
use epistat_testkit::Fingerprint;
use faer::Mat;
use epistat_semicontinuous::{
    BootstrapOptions, ConfidenceInterval, FamilyRandomEffects, FitOptions,
    LongitudinalModelInput, ModelComparison, ModelComparisonOptions, ModelFitMetrics,
    ModelInput, MtpFitOptions, MtpMultiChainOptions, MtpPosteriorSummary, MtpSamplerConfig,
    ParameterSummary, PreparedGammaDesign, PreparedPredictionDesign, PreparedScratch,
    PreparedTwoPartDesign, Regularization, TweedieOptions, TweedieScratch, TwoPartModel,
    TwoPartReport, TwoPartWarmStart, autocorrelation, bootstrap_percentile_ci,
    bootstrap_summary, bootstrap_two_part, coefficient_confidence_intervals,
    compare_models_input, effective_sample_size, fit_lognormal_smearing_input,
    fit_mtp_input_multi_chain_with_posterior_config, fit_tweedie_input,
    fit_tweedie_ref_warm_prevalidated_with_scratch, fit_two_part_clustered_input,
    fit_two_part_input, fit_two_part_ref_warm_prevalidated, posterior_predictive_summary,
    select_models_input,
};

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

/// An intercept, three dummies and two continuous columns, a gamma-like cost with
/// about 35% zeros, positive weights and clusters of one to four rows.
fn fixture(seed: u64) -> ModelInput {
    let mut stream = Stream(seed);
    let mut x = Mat::<f64>::zeros(N_ROWS, N_COLS);
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
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
            y[(i, 0)] = (6.0 + eta).exp() / 2.0 * noise + 1.0;
        }
    }
    let mut clusters = Vec::with_capacity(N_ROWS);
    let mut cluster = 0_u64;
    while clusters.len() < N_ROWS {
        let size = 1 + usize::try_from(stream.next_u64() % 4).expect("small");
        for _ in 0..size.min(N_ROWS - clusters.len()) {
            clusters.push(cluster * 7_919 + 13);
        }
        cluster += 1;
    }
    let weights = Mat::from_fn(N_ROWS, 1, |_, _| 0.5 + stream.uniform());
    ModelInput::new(x, y)
        .with_cluster_ids(clusters)
        .with_sample_weights(weights)
}

fn unweighted(input: &ModelInput) -> ModelInput {
    ModelInput::new(input.design_matrix.clone(), input.outcome.clone())
}

fn positive_rows(input: &ModelInput) -> ModelInput {
    let rows: Vec<usize> = (0..input.outcome.nrows())
        .filter(|&i| input.outcome[(i, 0)] > 0.0)
        .collect();
    let x = Mat::from_fn(rows.len(), input.design_matrix.ncols(), |i, j| {
        input.design_matrix[(rows[i], j)]
    });
    let y = Mat::from_fn(rows.len(), 1, |i, _| input.outcome[(rows[i], 0)]);
    ModelInput::new(x, y)
}

/// A panel of subjects over periods, a logistic zero part and a log-normal positive
/// part with subject intercepts.
fn panel(subjects: usize, periods: usize, seed: u64) -> LongitudinalModelInput {
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
            ids.push(subject as u64 + 1);
            time.push(period as f64);
        }
    }
    LongitudinalModelInput::new(
        Mat::from_fn(rows, 1, |i, _| outcome[i]),
        Mat::from_fn(rows, 2, |i, j| design[2 * i + j]),
        Mat::from_fn(rows, 2, |i, j| design[2 * i + j]),
        ids,
        time,
    )
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

    fn intervals(&mut self, intervals: &[ConfidenceInterval]) {
        self.n(intervals.len());
        for interval in intervals {
            self.f(interval.lower);
            self.f(interval.upper);
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

    fn two_part(&mut self, model: &TwoPartModel, report: &TwoPartReport) {
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
                self.word(score.name.len() as u64);
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
            self.f(row.aic);
            self.f(row.bic);
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
            self.f(value);
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

fn bootstrap(out: &mut Golden, input: &ModelInput) {
    let options = BootstrapOptions::builder().iterations(25).seed(7).build();
    let result = bootstrap_two_part(
        &input.design_matrix,
        &input.outcome,
        FitOptions::default(),
        options,
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
        let summary = bootstrap_summary(betas, 0.05);
        h.mat(&summary.mean);
        h.mat(&summary.se);
        h.intervals(&summary.ci);
        h.intervals(&bootstrap_percentile_ci(betas, 0.1));
    }
    out.push(("bootstrap/summary", h.finish()));
}

fn two_part_variants(out: &mut Golden, input: &ModelInput) -> TwoPartWarmStart {
    let robust = FitOptions::builder().robust_se(true).build();
    let (model, report) = fit_two_part_clustered_input(input, robust).expect("clustered");
    let mut h = Hasher::new();
    h.two_part(&model, &report);
    let cov = report.cov_gamma.as_ref().expect("covariance");
    h.intervals(&coefficient_confidence_intervals(&model.beta_gamma, cov, 0.05));
    out.push(("two_part/clustered/robust", h.finish()));

    let elastic = FitOptions::builder()
        .regularization(Regularization::ElasticNet {
            lambda: 0.02,
            alpha: 0.5,
            exclude_intercept: true,
        })
        .build();
    let (model, report) = fit_two_part_input(&unweighted(input), elastic).expect("elastic net");
    let mut h = Hasher::new();
    h.two_part(&model, &report);
    out.push(("two_part/elastic_net", h.finish()));

    let stable = FitOptions::stable_defaults();
    let (model, report) = fit_two_part_input(input, stable).expect("stable");
    let warm = TwoPartWarmStart {
        beta_logit: model.beta_logit.clone(),
        beta_gamma: model.beta_gamma.clone(),
    };
    let replicate = FitOptions {
        compute_covariance: false,
        ..stable
    };
    let (model, report2) =
        fit_two_part_ref_warm_prevalidated(input.as_ref(), replicate, Some(&warm))
            .expect("prevalidated");
    let mut h = Hasher::new();
    h.n(report.iterations_logit);
    h.two_part(&model, &report2);
    out.push(("two_part/prevalidated/warm", h.finish()));
    warm
}

fn compact(out: &mut Golden, input: &ModelInput, warm: &TwoPartWarmStart) {
    let x = &input.design_matrix;
    let y = input.outcome.col_as_slice(0);
    let mut stream = Stream(99);
    let weights: Vec<f64> = (0..N_ROWS)
        .map(|_| -stream.uniform().max(1e-12).ln())
        .collect();
    let mut scratch = PreparedScratch::default();

    let positive = positive_rows(input);
    let gamma_options = TweedieOptions {
        inner_parallel: false,
        ..TweedieOptions::default()
    };
    let (gamma, _) = fit_tweedie_input(&positive, 2.0, gamma_options).expect("gamma");
    let design = PreparedGammaDesign::new(
        positive.design_matrix.as_ref(),
        positive.outcome.col_as_slice(0),
    )
    .expect("gamma design");
    let positive_weights = &weights[..positive.outcome.nrows()];
    let fit = design
        .fit(positive_weights, &gamma.beta, gamma_options, &mut scratch)
        .expect("compact gamma");
    let mut h = Hasher::new();
    h.mat(&fit.beta);
    h.n(fit.iterations);
    let mut tweedie_scratch = TweedieScratch::default();
    let weighted = positive
        .clone()
        .with_sample_weights(Mat::from_fn(positive_weights.len(), 1, |i, _| {
            positive_weights[i]
        }));
    let (dense, _) = fit_tweedie_ref_warm_prevalidated_with_scratch(
        weighted.as_ref(),
        2.0,
        gamma_options,
        Some(&gamma.beta),
        &mut tweedie_scratch,
    )
    .expect("dense gamma");
    h.mat(&dense.beta);
    let prediction = PreparedPredictionDesign::new(positive.design_matrix.as_ref())
        .expect("prediction design");
    h.f(prediction.mean_gamma(&fit.beta, positive_weights));
    out.push(("compact/gamma", h.finish()));

    let options = FitOptions {
        compute_covariance: false,
        inner_parallel: false,
        ..FitOptions::default()
    };
    let design = PreparedTwoPartDesign::new(x.as_ref(), y).expect("two-part design");
    let fit = design
        .fit(&weights, warm, options, &mut scratch)
        .expect("compact two-part");
    let mut h = Hasher::new();
    h.mat(&fit.beta_logit);
    h.mat(&fit.beta_gamma);
    let prediction = PreparedPredictionDesign::new(x.as_ref()).expect("prediction design");
    h.f(prediction.mean_two_part(&fit.beta_logit, &fit.beta_gamma, &weights));
    out.push(("compact/two_part", h.finish()));
}

fn selection(out: &mut Golden, input: &ModelInput) {
    let plain = unweighted(input);
    let comparison =
        compare_models_input(&plain, &ModelComparisonOptions::default()).expect("comparison");
    let mut h = Hasher::new();
    h.comparison(&comparison);
    out.push(("comparison/default", h.finish()));

    let options = ModelComparisonOptions {
        two_part_elastic_net_options: Some(
            FitOptions::builder()
                .regularization(Regularization::ElasticNet {
                    lambda: 0.01,
                    alpha: 0.8,
                    exclude_intercept: true,
                })
                .build(),
        ),
        ..ModelComparisonOptions::default()
    };
    let comparison = compare_models_input(&plain, &options).expect("comparison");
    let mut h = Hasher::new();
    h.comparison(&comparison);
    out.push(("comparison/elastic_net", h.finish()));

    let selected = select_models_input(&plain, &[1.2, 1.5, 1.8]);
    let mut h = Hasher::new();
    h.f(selected.park_test.intercept);
    h.f(selected.park_test.slope);
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

    let (model, report) = fit_lognormal_smearing_input(input, Default::default()).expect("ln");
    let mut h = Hasher::new();
    h.mat(&model.beta);
    h.f(model.smearing_factor);
    h.opt(report.se.as_ref());
    out.push(("lognormal/weighted", h.finish()));
}

fn mtp(out: &mut Golden) {
    let input = panel(80, 4, 42);
    let config = MtpSamplerConfig {
        fit_options: MtpFitOptions::builder()
            .iterations(800)
            .burn_in(200)
            .thin(2)
            .seed(11)
            .build(),
        family_random_effects: FamilyRandomEffects::Disabled,
        ..MtpSamplerConfig::default()
    };
    let (_model, report, chains) = fit_mtp_input_multi_chain_with_posterior_config(
        &input,
        config,
        MtpMultiChainOptions { chains: 3 },
    )
    .expect("mtp");
    let mut h = Hasher::new();
    for chain in &chains {
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
    let convergence = &report.convergence;
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
    h.posterior(report.pooled_posterior_summary.as_ref());
    for chain in &report.chain_reports {
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
                h.f(value);
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
            h.f(value);
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
    ("bootstrap/draws", 0x1665116b57fac17c),
    ("bootstrap/summary", 0xda0e70c42c70b7ca),
    ("two_part/clustered/robust", 0x6a615d52a469116e),
    ("two_part/elastic_net", 0x7aa9a058c6c773e0),
    ("two_part/prevalidated/warm", 0x45cb44511f4c3630),
    ("compact/gamma", 0xfbca6c2d3c43fbf8),
    ("compact/two_part", 0xbc06d03e59d9f18a),
    ("comparison/default", 0xa7aabf741b04d9aa),
    ("comparison/elastic_net", 0xa7bb6818a8e46428),
    ("selection", 0x33593df6d30b6785),
    ("lognormal/weighted", 0x22f8bede7fbf6966),
    ("mtp/draws", 0xf0902e04a16ea2fd),
    ("mtp/convergence", 0x062bdb420fa8775c),
    ("mtp/posterior_predictive", 0x2a818f18ff31a831),
    ("mtp/autocorrelation", 0xe7e846cfe5a1e143),
];

#[test]
fn estimators_match_the_recorded_bit_patterns() {
    let actual = collect();
    if std::env::var_os("GOLDEN_PRINT").is_some() {
        for (name, hash) in &actual {
            println!("    (\"{name}\", {hash:#018x}),");
        }
    }
    assert_eq!(actual.len(), GOLDEN.len(), "golden table is out of date");
    for ((name, hash), (golden_name, golden_hash)) in actual.iter().zip(GOLDEN) {
        assert_eq!(name, golden_name);
        assert_eq!(*hash, *golden_hash, "{name}: output bits changed");
    }
}
