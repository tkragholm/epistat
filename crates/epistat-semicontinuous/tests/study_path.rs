//! The cost models of Study II, hashed to the bit.
//!
//! Each case makes the calls of PHD's `models::cost_model` and
//! `models::bootstrap::driver` with the options they pass:
//!
//! - one-part gamma: `fit_tweedie_input` at power 2 with robust standard
//!   errors and the relaxed strategy (fallback ridge 1e-3, three retries, warm
//!   start), the contrast of the means predicted with the exposure set to 1
//!   and to 0, and fractional-random-weight replicates refitted by
//!   `fit_tweedie_ref_warm_prevalidated_with_scratch` from the full-sample
//!   coefficients with the inner loops serial;
//! - Tweedie robustness: the same fit at power 1.5;
//! - log-normal robustness: `fit_lognormal_smearing_input` with robust
//!   standard errors;
//! - two-part: `fit_two_part_clustered_input` with `FitOptions::stable_defaults`,
//!   replicates by `fit_two_part_ref_warm_prevalidated` without covariance and
//!   with the inner loops serial, warm-started from the full-sample fit, and
//!   the replicate coefficients through `bootstrap_percentile_ci` and
//!   `bootstrap_summary` at alpha 0.05;
//! - the compact designs of the comparator-only contrast, refitted under
//!   replicate weights with the same options;
//! - model comparison with `ModelComparisonOptions::default`;
//! - the exploratory MTP model, four chains with the default sampler and
//!   family random effects off, on the design with the subject and the
//!   follow-up year as covariates, as `longitudinal_input_from_model_input`
//!   builds it.
//!
//! Replicate `r` reweights each family by the Dirichlet weight that
//! `epistat_rng::dirichlet_weights` draws from `replicate_seed(seed, r)`. A
//! change that moves a hash moves Study II, so a port of these calls to new
//! names must reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS and are asserted only there.
//! Elsewhere the test computes them and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.

use epistat_linalg::Mat;
use epistat_rng::{dirichlet_weights, replicate_seed};
use epistat_semicontinuous::models::FitStrategy;
use epistat_semicontinuous::{
    ConfidenceInterval, FamilyRandomEffects, FitOptions, LogNormalOptions, LongitudinalModelInput,
    ModelComparison, ModelComparisonOptions, ModelFitMetrics, ModelInput, ModelInputRef,
    MtpFitOptions, MtpMultiChainOptions, MtpSamplerConfig, PreparedGammaDesign,
    PreparedPredictionDesign, PreparedScratch, PreparedTwoPartDesign, TweedieOptions,
    TweedieScratch, TwoPartModel, TwoPartReport, TwoPartWarmStart, bootstrap_percentile_ci,
    bootstrap_summary, compare_models_input, fit_lognormal_smearing_input,
    fit_mtp_input_multi_chain_with_config, fit_tweedie_input,
    fit_tweedie_ref_warm_prevalidated_with_scratch, fit_two_part_clustered_input,
    fit_two_part_ref_warm_prevalidated,
};
use epistat_testkit::{Fingerprint, SplitMix64};

const N_ROWS: usize = 3_000;
const EXPOSURE: usize = 1;
const SEED: u64 = 20_261_126;
const REPLICATES: u64 = 12;

/// A fingerprint that hashes NaN and `None` alike, so a value that becomes
/// `Option` without changing leaves the hash alone.
struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, word: u64) {
        self.0 = std::mem::take(&mut self.0).u64(word);
    }

    fn o(&mut self, value: Option<f64>) {
        let value = value.filter(|v| !v.is_nan());
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn f(&mut self, value: f64) {
        self.o(Some(value));
    }

    fn n(&mut self, value: usize) {
        self.word(value as u64);
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
            None => self.o(None),
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
        self.n(report.iterations_logit);
        self.n(report.iterations_gamma);
        self.word(u64::from(report.clustered));
        self.o(report.cluster_count.map(|count| count as f64));
    }

    fn comparison(&mut self, comparison: &ModelComparison) {
        for table in [
            &comparison.in_sample,
            &comparison.cv_summary,
            &comparison.cv_ranking,
        ] {
            self.n(table.len());
            for score in table {
                self.n(score.name.len());
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

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

/// A Study II slice: per person-year, an exposure indicator, sex, two age
/// bands and the index year, costs with about 35% zeros, families of one to
/// four rows and match weights of a quarter to one.
struct Slice {
    input: ModelInput,
    families: Vec<u64>,
    family_of_row: Vec<usize>,
}

fn slice(floor_zeros: bool) -> Slice {
    let mut stream = SplitMix64::new(SEED);
    let columns = 5;
    let mut x = Mat::<f64>::zeros(N_ROWS, columns);
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
    let mut weights = Mat::<f64>::zeros(N_ROWS, 1);
    let mut family_ids = Vec::with_capacity(N_ROWS);
    let mut family_of_row = Vec::with_capacity(N_ROWS);
    let mut families = Vec::new();
    while family_ids.len() < N_ROWS {
        let id = 1_000 + 7_919 * families.len() as u64;
        families.push(id);
        let size = 1 + usize::try_from(stream.below(4)).expect("small");
        let exposed = stream.unit() < 0.2;
        let weight = 0.25 * f64::from(1 + u32::try_from(stream.below(4)).expect("small"));
        for _ in 0..size.min(N_ROWS - family_ids.len()) {
            let i = family_ids.len();
            x[(i, 0)] = f64::from(exposed);
            x[(i, 1)] = f64::from(stream.unit() < 0.5);
            let age = stream.below(3);
            x[(i, 2)] = f64::from(age == 1);
            x[(i, 3)] = f64::from(age == 2);
            x[(i, 4)] = (stream.below(12) as f64) - 6.0;
            let eta = 0.45 * x[(i, 0)] + 0.1 * x[(i, 1)] - 0.2 * x[(i, 2)]
                + 0.15 * x[(i, 3)]
                + 0.02 * x[(i, 4)];
            let zero = stream.unit() < 0.35 - 0.1 * x[(i, 0)];
            let noise = -(1.0 - stream.unit()).ln() - (1.0 - stream.unit()).ln();
            y[(i, 0)] = if zero {
                if floor_zeros { 1.0 } else { 0.0 }
            } else {
                (8.0 + eta).exp() / 2.0 * noise + 1.0
            };
            weights[(i, 0)] = weight;
            family_ids.push(id);
            family_of_row.push(families.len() - 1);
        }
    }
    let input = ModelInput::new(x, y)
        .with_cluster_ids(family_ids)
        .with_column_labels(
            ["exposed", "sex", "age_1", "age_2", "index_year"]
                .map(String::from)
                .to_vec(),
        )
        .with_intercept()
        .with_sample_weights(weights);
    Slice {
        input,
        families,
        family_of_row,
    }
}

fn counterfactual(design: &Mat<f64>, value: f64) -> Mat<f64> {
    Mat::from_fn(design.nrows(), design.ncols(), |i, j| {
        if j == EXPOSURE { value } else { design[(i, j)] }
    })
}

/// The row weights of replicate `r`: the match weight times the family's
/// Dirichlet weight.
fn replicate_weights(slice: &Slice, replicate: u64) -> (Vec<f64>, Mat<f64>) {
    let mut family_weights = Vec::new();
    dirichlet_weights(
        slice.families.len(),
        replicate_seed(SEED, replicate),
        &mut family_weights,
    );
    let row: Vec<f64> = slice
        .family_of_row
        .iter()
        .map(|&family| family_weights[family])
        .collect();
    let base = slice.input.sample_weights.as_ref().expect("weights");
    let combined = Mat::from_fn(row.len(), 1, |i, _| base[(i, 0)] * row[i]);
    (row, combined)
}

fn relaxed() -> FitStrategy {
    FitStrategy::Relaxed {
        fallback_lambda: 1e-3,
        max_retries: 3,
        warm_start: true,
        time_budget: None,
    }
}

/// The one-part gamma model on costs whose zeros are floored at 1, Study
/// II's `positive-floor` policy, with its replicates; or the Tweedie
/// robustness model on the costs as they are.
fn tweedie(out: &mut Vec<(&'static str, u64)>, slice: &Slice, gamma: bool) {
    let treated = counterfactual(&slice.input.design_matrix, 1.0);
    let control = counterfactual(&slice.input.design_matrix, 0.0);
    let (name, power) = if gamma {
        ("one_part_gamma", 2.0)
    } else {
        ("tweedie_robustness", 1.5)
    };
    {
        let options = TweedieOptions::builder()
            .robust_se(true)
            .strategy(relaxed())
            .build();
        let (model, report) = fit_tweedie_input(&slice.input, power, options).expect("fit");
        let mut h = Hash::new();
        h.mat(&model.beta);
        h.f(model.power);
        h.opt(report.se.as_ref());
        h.n(report.meta.iterations);
        h.word(u64::from(report.clustered));
        h.o(report.cluster_count.map(|count| count as f64));
        h.f(model.predict_mean(&treated, None));
        h.f(model.predict_mean(&control, None));
        if gamma {
            let replicate_options = TweedieOptions {
                inner_parallel: false,
                ..TweedieOptions::default()
            };
            let mut scratch = TweedieScratch::default();
            for replicate in 0..REPLICATES {
                let (_, weights) = replicate_weights(slice, replicate);
                let input = ModelInputRef::new(
                    slice.input.design_matrix.as_ref(),
                    slice.input.outcome.as_ref(),
                )
                .with_sample_weights(weights.as_ref());
                let (boot, _) = fit_tweedie_ref_warm_prevalidated_with_scratch(
                    input,
                    2.0,
                    replicate_options,
                    Some(&model.beta),
                    &mut scratch,
                )
                .expect("replicate");
                h.mat(&boot.beta);
                h.f(boot.predict_mean(&treated, None) - boot.predict_mean(&control, None));
            }
        }
        out.push((name, h.finish()));
    }
}

fn lognormal(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let options = LogNormalOptions::builder().robust_se(true).build();
    let (model, report) = fit_lognormal_smearing_input(&slice.input, options).expect("fit");
    let mut h = Hash::new();
    h.mat(&model.beta);
    h.f(model.smearing_factor);
    h.opt(report.se.as_ref());
    h.n(report.meta.iterations);
    h.f(model.predict_mean(&counterfactual(&slice.input.design_matrix, 1.0), None));
    h.f(model.predict_mean(&counterfactual(&slice.input.design_matrix, 0.0), None));
    out.push(("log_normal_robustness", h.finish()));
}

fn two_part(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let treated = counterfactual(&slice.input.design_matrix, 1.0);
    let control = counterfactual(&slice.input.design_matrix, 0.0);
    let fit_options = FitOptions::stable_defaults();
    let (model, report) = fit_two_part_clustered_input(&slice.input, fit_options).expect("fit");
    let mut h = Hash::new();
    h.two_part(&model, &report);
    h.f(model.predict_mean(&treated, None));
    h.f(model.predict_mean(&control, None));
    out.push(("two_part", h.finish()));

    let warm = TwoPartWarmStart {
        beta_logit: model.beta_logit.clone(),
        beta_gamma: model.beta_gamma.clone(),
    };
    let replicate_options = FitOptions {
        compute_covariance: false,
        inner_parallel: false,
        ..fit_options
    };
    let mut h = Hash::new();
    let (mut betas_logit, mut betas_gamma) = (Vec::new(), Vec::new());
    for replicate in 0..REPLICATES {
        let (_, weights) = replicate_weights(slice, replicate);
        let input = ModelInputRef::new(
            slice.input.design_matrix.as_ref(),
            slice.input.outcome.as_ref(),
        )
        .with_sample_weights(weights.as_ref());
        let (boot, _) = fit_two_part_ref_warm_prevalidated(input, replicate_options, Some(&warm))
            .expect("replicate");
        h.f(boot.predict_mean(&treated, None) - boot.predict_mean(&control, None));
        betas_logit.push(boot.beta_logit);
        betas_gamma.push(boot.beta_gamma);
    }
    for betas in [&betas_logit, &betas_gamma] {
        for beta in betas {
            h.mat(beta);
        }
        h.intervals(&bootstrap_percentile_ci(betas, 0.05));
        let summary = bootstrap_summary(betas, 0.05);
        h.mat(&summary.mean);
        h.mat(&summary.se);
    }
    out.push(("two_part/replicates", h.finish()));
}

/// The comparator-only contrast: fit on the unexposed rows without the
/// exposure column, predict for the exposed rows.
fn compact(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let design = &slice.input.design_matrix;
    let rows = |exposed: f64| -> Vec<usize> {
        (0..design.nrows())
            .filter(|&i| design[(i, EXPOSURE)].to_bits() == exposed.to_bits())
            .collect()
    };
    let (comparators, exposed) = (rows(0.0), rows(1.0));
    let gather = |rows: &[usize]| {
        Mat::from_fn(rows.len(), design.ncols() - 1, |i, j| {
            design[(rows[i], if j < EXPOSURE { j } else { j + 1 })]
        })
    };
    let (fit_design, exposed_design) = (gather(&comparators), gather(&exposed));
    let fit_outcome: Vec<f64> = comparators
        .iter()
        .map(|&i| slice.input.outcome[(i, 0)])
        .collect();
    let prediction = PreparedPredictionDesign::new(exposed_design.as_ref()).expect("prediction");
    let mut scratch = PreparedScratch::default();

    let gamma_options = TweedieOptions {
        inner_parallel: false,
        ..TweedieOptions::default()
    };
    let positive: Vec<f64> = fit_outcome.iter().map(|&y| y.max(1.0)).collect();
    let full = ModelInput::new(
        fit_design.clone(),
        Mat::from_fn(positive.len(), 1, |i, _| positive[i]),
    );
    let (gamma, _) = fit_tweedie_input(&full, 2.0, gamma_options).expect("gamma");
    let gamma_design =
        PreparedGammaDesign::new(fit_design.as_ref(), &positive).expect("gamma design");
    let two_part_design =
        PreparedTwoPartDesign::new(fit_design.as_ref(), &fit_outcome).expect("two-part design");
    let (two_part, _) = fit_two_part_clustered_input(
        &ModelInput::new(
            fit_design.clone(),
            Mat::from_fn(fit_outcome.len(), 1, |i, _| fit_outcome[i]),
        )
        .with_cluster_ids(
            comparators
                .iter()
                .map(|&i| slice.input.cluster_ids.as_ref().expect("clusters")[i])
                .collect(),
        ),
        FitOptions::stable_defaults(),
    )
    .expect("two-part");
    let warm = TwoPartWarmStart {
        beta_logit: two_part.beta_logit.clone(),
        beta_gamma: two_part.beta_gamma.clone(),
    };
    let two_part_options = FitOptions {
        compute_covariance: false,
        inner_parallel: false,
        ..FitOptions::stable_defaults()
    };

    let mut h = Hash::new();
    for replicate in 0..REPLICATES {
        let (row, combined) = replicate_weights(slice, replicate);
        let comparator_weights: Vec<f64> = comparators.iter().map(|&i| combined[(i, 0)]).collect();
        let exposed_weights: Vec<f64> = exposed.iter().map(|&i| row[i]).collect();
        let fit = gamma_design
            .fit(
                &comparator_weights,
                &gamma.beta,
                gamma_options,
                &mut scratch,
            )
            .expect("compact gamma");
        h.mat(&fit.beta);
        h.n(fit.iterations);
        h.f(prediction.mean_gamma(&fit.beta, &exposed_weights));
        let fit = two_part_design
            .fit(&comparator_weights, &warm, two_part_options, &mut scratch)
            .expect("compact two-part");
        h.mat(&fit.beta_logit);
        h.mat(&fit.beta_gamma);
        h.f(prediction.mean_two_part(&fit.beta_logit, &fit.beta_gamma, &exposed_weights));
    }
    out.push(("compact", h.finish()));
}

fn comparison(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let comparison =
        compare_models_input(&slice.input, &ModelComparisonOptions::default()).expect("compare");
    let mut h = Hash::new();
    h.comparison(&comparison);
    out.push(("model_comparison", h.finish()));
}

/// Subjects followed for four years, with the subject and the year among the
/// covariates, families of one or two subjects.
fn mtp(out: &mut Vec<(&'static str, u64)>) {
    let mut stream = SplitMix64::new(SEED + 1);
    let (subjects, years) = (60_usize, 4_usize);
    let rows = subjects * years;
    let mut design = Mat::<f64>::zeros(rows, 4);
    let mut outcome = Mat::<f64>::zeros(rows, 1);
    let mut subject_ids = Vec::with_capacity(rows);
    let mut time = Vec::with_capacity(rows);
    let mut families = Vec::with_capacity(rows);
    for subject in 0..subjects {
        let exposed = stream.unit() < 0.3;
        let level = 0.5 * (stream.unit() - 0.5);
        for year in 0..years {
            let i = subject * years + year;
            design[(i, 0)] = 1.0;
            design[(i, 1)] = f64::from(exposed);
            design[(i, 2)] = year as f64;
            design[(i, 3)] = (subject + 1) as f64;
            let p = 0.55 + 0.15 * f64::from(exposed);
            outcome[(i, 0)] = if stream.unit() < p {
                (7.5 + 0.4 * f64::from(exposed) + level + 0.6 * (stream.unit() - 0.5)).exp()
            } else {
                0.0
            };
            subject_ids.push(subject as u64 + 1);
            time.push(year as f64);
            families.push(subject as u64 / 2 + 1);
        }
    }
    let input = LongitudinalModelInput::new(outcome, design.clone(), design, subject_ids, time)
        .with_family_ids(families);
    let config = MtpSamplerConfig {
        fit_options: MtpFitOptions::builder().seed(SEED).build(),
        family_random_effects: FamilyRandomEffects::Disabled,
        ..MtpSamplerConfig::default()
    };
    let (_model, report) =
        fit_mtp_input_multi_chain_with_config(&input, config, MtpMultiChainOptions { chains: 4 })
            .expect("mtp");
    let mut h = Hash::new();
    let convergence = &report.convergence;
    h.n(convergence.chain_count);
    h.n(convergence.draws_per_chain_used);
    for &value in convergence
        .alpha_split_rhat
        .iter()
        .chain(&convergence.beta_split_rhat)
    {
        h.o(value);
    }
    h.o(convergence.kappa_split_rhat);
    h.o(convergence.omega_sq_split_rhat);
    h.o(convergence.max_split_rhat);
    for chain in &report.chain_reports {
        h.n(chain.diagnostics.iterations_completed);
        h.n(chain.diagnostics.retained_draws);
    }
    if let Some(summary) = &report.pooled_posterior_summary {
        for parameter in summary.alpha.iter().chain(&summary.beta) {
            for value in [
                parameter.mean,
                parameter.std_dev,
                parameter.q025,
                parameter.q50,
                parameter.q975,
            ] {
                h.f(value);
            }
        }
    }
    out.push(("mtp_exploratory", h.finish()));
}

fn cases() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    tweedie(&mut out, &slice(true), true);
    let with_zeros = slice(false);
    tweedie(&mut out, &with_zeros, false);
    lognormal(&mut out, &with_zeros);
    two_part(&mut out, &with_zeros);
    compact(&mut out, &with_zeros);
    comparison(&mut out, &with_zeros);
    mtp(&mut out);
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("one_part_gamma", 0x28b5_3b87_16fe_0c4f),
    ("tweedie_robustness", 0x1893_ee15_d0b0_2a56),
    ("log_normal_robustness", 0xc6d5_c3c3_d706_d50d),
    ("two_part", 0x5460_cf7a_0fff_999b),
    ("two_part/replicates", 0x2861_9d3d_f785_02b3),
    ("compact", 0x9821_db6e_9e38_d9bb),
    ("model_comparison", 0x967c_945c_0343_c076),
    ("mtp_exploratory", 0xe265_4caa_2aeb_af60),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    for (name, hash) in &current {
        epistat_testkit::hash_out(name, *hash);
    }
    if std::env::var_os("STUDY_PATH_PRINT").is_some() {
        for (name, hash) in &current {
            println!(
                "    (\"{name}\", 0x{:04x}_{:04x}_{:04x}_{:04x}),",
                hash >> 48,
                (hash >> 32) & 0xffff,
                (hash >> 16) & 0xffff,
                hash & 0xffff
            );
        }
    }
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(current.as_slice(), GOLDEN);
    }
}
