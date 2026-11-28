//! The cost models of Study II, hashed to the bit.
//!
//! Each case makes the calls of PHD's `models::cost_model` and
//! `models::bootstrap::driver` with the options they pass:
//!
//! - one-part gamma: `tweedie::fit` at power 2 with robust standard
//!   errors and the relaxed strategy (fallback ridge 1e-3, three retries), on
//!   the weighted, clustered data, the contrast of the means predicted with
//!   the exposure set to 1 and to 0, and fractional-random-weight replicates
//!   refitted by `tweedie::fit` on `with_weights` views without clusters, from
//!   the full-sample coefficients with one scratch;
//! - Tweedie robustness: the same fit at power 1.5;
//! - log-normal robustness: `lognormal::fit` with robust standard errors on
//!   the data without their weights, which the log-normal model does not use;
//! - two-part: `two_part::fit` with `TwoPartOptions::stable_defaults` on the
//!   weighted, clustered data, replicates without covariance on `with_weights`
//!   views without clusters, warm-started from the full-sample fit, and the
//!   replicate coefficients through `two_part::percentile_intervals` and
//!   `two_part::summary` at alpha 0.05;
//! - the compact designs of the comparator-only contrast, refitted under
//!   replicate weights with the same options;
//! - model comparison with `CompareOptions::new(42)`;
//! - the exploratory MTP model, four chains with the default sampler and
//!   family random effects off, on the design with the subject and the
//!   follow-up year as covariates.
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

use epistat_core::{Interval, Level};
use epistat_linalg::Mat;
use epistat_rng::{dirichlet_weights, replicate_seed};
use epistat_semicontinuous::compare::{CompareOptions, ModelComparison, ModelFitMetrics};
use epistat_semicontinuous::lognormal::LogNormalOptions;
use epistat_semicontinuous::mtp::{FamilyRandomEffects, MtpOptions};
use epistat_semicontinuous::prepared::{
    PreparedGammaDesign, PreparedPredictionDesign, PreparedScratch, PreparedTwoPartDesign,
};
use epistat_semicontinuous::tweedie::{TweedieOptions, TweedieScratch};
use epistat_semicontinuous::two_part::{TwoPartModel, TwoPartOptions, TwoPartWarmStart};
use epistat_semicontinuous::{
    CostData, FitStrategy, LongitudinalData, compare, lognormal, mtp, tweedie, two_part,
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

    fn intervals(&mut self, intervals: &[Interval]) {
        self.n(intervals.len());
        for interval in intervals {
            self.o(interval.lower);
            self.o(interval.upper);
        }
    }

    fn metrics(&mut self, metrics: &ModelFitMetrics) {
        self.f(metrics.rmse);
        self.f(metrics.mae);
        self.f(metrics.rmsle);
        self.o(metrics.r2);
        self.f(metrics.deviance);
    }

    fn two_part(&mut self, model: &TwoPartModel) {
        let report = &model.report;
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
                self.n(score.candidate.label().len());
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
            self.o(row.aic);
            self.o(row.bic);
        }
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

/// A Study II slice: per person-year, an exposure indicator, sex, two age
/// bands and the index year, costs with about 35% zeros, families of one to
/// four rows and match weights of a quarter to one.
///
/// The design holds an intercept in column 0 before the five covariates, and
/// the cluster of a row is its family's code.
struct Slice {
    x: Mat<f64>,
    y: Vec<f64>,
    weights: Vec<f64>,
    clusters: Vec<u32>,
    families: usize,
    family_of_row: Vec<usize>,
}

impl Slice {
    /// The weighted, clustered data of the slice.
    fn data(&self) -> CostData<'_> {
        CostData::builder(self.x.as_ref(), &self.y)
            .weights(&self.weights)
            .clusters(&self.clusters)
            .build()
            .expect("data")
    }
}

fn slice(floor_zeros: bool) -> Slice {
    let mut stream = SplitMix64::new(SEED);
    let columns = 5;
    let mut x = Mat::<f64>::zeros(N_ROWS, columns + 1);
    let mut y = vec![0.0; N_ROWS];
    let mut weights = vec![0.0; N_ROWS];
    let mut clusters = Vec::with_capacity(N_ROWS);
    let mut family_of_row = Vec::with_capacity(N_ROWS);
    let mut families = 0_usize;
    while clusters.len() < N_ROWS {
        let code = u32::try_from(families).expect("small");
        families += 1;
        let size = 1 + usize::try_from(stream.below(4)).expect("small");
        let exposed = stream.unit() < 0.2;
        let weight = 0.25 * f64::from(1 + u32::try_from(stream.below(4)).expect("small"));
        for _ in 0..size.min(N_ROWS - clusters.len()) {
            let i = clusters.len();
            // Columns: intercept, exposed, sex, age_1, age_2, index_year.
            x[(i, 0)] = 1.0;
            x[(i, 1)] = f64::from(exposed);
            x[(i, 2)] = f64::from(stream.unit() < 0.5);
            let age = stream.below(3);
            x[(i, 3)] = f64::from(age == 1);
            x[(i, 4)] = f64::from(age == 2);
            x[(i, 5)] = (stream.below(12) as f64) - 6.0;
            let eta = 0.45 * x[(i, 1)] + 0.1 * x[(i, 2)] - 0.2 * x[(i, 3)]
                + 0.15 * x[(i, 4)]
                + 0.02 * x[(i, 5)];
            let zero = stream.unit() < 0.35 - 0.1 * x[(i, 1)];
            let noise = -(1.0 - stream.unit()).ln() - (1.0 - stream.unit()).ln();
            y[i] = if zero {
                if floor_zeros { 1.0 } else { 0.0 }
            } else {
                (8.0 + eta).exp() / 2.0 * noise + 1.0
            };
            weights[i] = weight;
            clusters.push(code);
            family_of_row.push(families - 1);
        }
    }
    Slice {
        x,
        y,
        weights,
        clusters,
        families,
        family_of_row,
    }
}

fn counterfactual(design: &Mat<f64>, value: f64) -> Mat<f64> {
    Mat::from_fn(design.nrows(), design.ncols(), |i, j| {
        if j == EXPOSURE { value } else { design[(i, j)] }
    })
}

/// The family's Dirichlet weight of each row in replicate `r`, and the row
/// weights: the match weight times that.
fn replicate_weights(slice: &Slice, replicate: u64) -> (Vec<f64>, Vec<f64>) {
    let mut family_weights = Vec::new();
    dirichlet_weights(
        slice.families,
        replicate_seed(SEED, replicate),
        &mut family_weights,
    );
    let row: Vec<f64> = slice
        .family_of_row
        .iter()
        .map(|&family| family_weights[family])
        .collect();
    let combined = (0..row.len()).map(|i| slice.weights[i] * row[i]).collect();
    (row, combined)
}

fn relaxed() -> FitStrategy {
    FitStrategy::Relaxed {
        fallback_lambda: 1e-3,
        max_retries: 3,
    }
}

/// The one-part gamma model on costs whose zeros are floored at 1, Study
/// II's `positive-floor` policy, with its replicates; or the Tweedie
/// robustness model on the costs as they are.
fn tweedie(out: &mut Vec<(&'static str, u64)>, slice: &Slice, gamma: bool) {
    let treated = counterfactual(&slice.x, 1.0);
    let control = counterfactual(&slice.x, 0.0);
    let data = slice.data();
    let (name, power) = if gamma {
        ("one_part_gamma", 2.0)
    } else {
        ("tweedie_robustness", 1.5)
    };
    {
        let options = TweedieOptions {
            robust_se: true,
            strategy: relaxed(),
            ..TweedieOptions::default()
        };
        let model = tweedie::fit(&data, power, &options, None, None).expect("fit");
        let report = &model.report;
        let mut h = Hash::new();
        h.mat(&model.beta);
        h.f(model.power);
        h.opt(report.se.as_ref());
        h.n(report.meta.iterations);
        h.word(u64::from(report.clustered));
        h.o(report.cluster_count.map(|count| count as f64));
        let predict = |m: &tweedie::TweedieModel, x: &Mat<f64>| {
            m.predict_mean(x.as_ref(), None).expect("mean")
        };
        h.f(predict(&model, &treated));
        h.f(predict(&model, &control));
        if gamma {
            let replicate_options = TweedieOptions::default();
            let mut scratch = TweedieScratch::default();
            let unclustered = data.without_clusters();
            for replicate in 0..REPLICATES {
                let (_, weights) = replicate_weights(slice, replicate);
                let input = unclustered.with_weights(&weights).expect("weights");
                let boot = tweedie::fit(
                    &input,
                    2.0,
                    &replicate_options,
                    Some(&model.beta),
                    Some(&mut scratch),
                )
                .expect("replicate");
                h.mat(&boot.beta);
                h.f(predict(&boot, &treated) - predict(&boot, &control));
            }
        }
        out.push((name, h.finish()));
    }
}

fn lognormal(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let options = LogNormalOptions {
        robust_se: true,
        ..LogNormalOptions::default()
    };
    let model = lognormal::fit(&slice.data().without_weights(), &options).expect("fit");
    let report = &model.report;
    let mut h = Hash::new();
    h.mat(&model.beta);
    h.f(model.smearing_factor);
    h.opt(report.se.as_ref());
    h.n(report.meta.iterations);
    for value in [1.0, 0.0] {
        let x = counterfactual(&slice.x, value);
        h.f(model.predict_mean(x.as_ref(), None).expect("mean"));
    }
    out.push(("log_normal_robustness", h.finish()));
}

fn two_part(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let treated = counterfactual(&slice.x, 1.0);
    let control = counterfactual(&slice.x, 0.0);
    let predict = |m: &TwoPartModel, x: &Mat<f64>| m.predict_mean(x.as_ref(), None).expect("mean");
    let data = slice.data();
    let fit_options = TwoPartOptions::stable_defaults();
    let model = two_part::fit(&data, &fit_options, None).expect("fit");
    let mut h = Hash::new();
    h.two_part(&model);
    h.f(predict(&model, &treated));
    h.f(predict(&model, &control));
    out.push(("two_part", h.finish()));

    let warm = TwoPartWarmStart {
        beta_logit: model.beta_logit.clone(),
        beta_gamma: model.beta_gamma.clone(),
    };
    let replicate_options = TwoPartOptions {
        compute_covariance: false,
        ..fit_options
    };
    let unclustered = data.without_clusters();
    let mut h = Hash::new();
    let (mut betas_logit, mut betas_gamma) = (Vec::new(), Vec::new());
    for replicate in 0..REPLICATES {
        let (_, weights) = replicate_weights(slice, replicate);
        let input = unclustered.with_weights(&weights).expect("weights");
        let boot = two_part::fit(&input, &replicate_options, Some(&warm)).expect("replicate");
        h.f(predict(&boot, &treated) - predict(&boot, &control));
        betas_logit.push(boot.beta_logit);
        betas_gamma.push(boot.beta_gamma);
    }
    for betas in [&betas_logit, &betas_gamma] {
        for beta in betas {
            h.mat(beta);
        }
        let level = Level::alpha(0.05).expect("level");
        h.intervals(&two_part::percentile_intervals(betas, level));
        let summary = two_part::summary(betas, level);
        h.mat(&summary.mean);
        h.mat(&summary.se);
    }
    out.push(("two_part/replicates", h.finish()));
}

/// The comparator-only contrast: fit on the unexposed rows without the
/// exposure column, predict for the exposed rows.
fn compact(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let design = &slice.x;
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
    let fit_outcome: Vec<f64> = comparators.iter().map(|&i| slice.y[i]).collect();
    let prediction = PreparedPredictionDesign::new(exposed_design.as_ref()).expect("prediction");
    let mut scratch = PreparedScratch::default();

    let gamma_options = TweedieOptions::default();
    let positive: Vec<f64> = fit_outcome.iter().map(|&y| y.max(1.0)).collect();
    let full = CostData::builder(fit_design.as_ref(), &positive)
        .build()
        .expect("data");
    let gamma = tweedie::fit(&full, 2.0, &gamma_options, None, None).expect("gamma");
    let gamma_design =
        PreparedGammaDesign::new(fit_design.as_ref(), &positive).expect("gamma design");
    let two_part_design =
        PreparedTwoPartDesign::new(fit_design.as_ref(), &fit_outcome).expect("two-part design");
    let comparator_clusters: Vec<u32> = comparators.iter().map(|&i| slice.clusters[i]).collect();
    let two_part = two_part::fit(
        &CostData::builder(fit_design.as_ref(), &fit_outcome)
            .clusters(&comparator_clusters)
            .build()
            .expect("data"),
        &TwoPartOptions::stable_defaults(),
        None,
    )
    .expect("two-part");
    let warm = TwoPartWarmStart {
        beta_logit: two_part.beta_logit.clone(),
        beta_gamma: two_part.beta_gamma.clone(),
    };
    let two_part_options = TwoPartOptions {
        compute_covariance: false,
        ..TwoPartOptions::stable_defaults()
    };

    let mut h = Hash::new();
    for replicate in 0..REPLICATES {
        let (row, combined) = replicate_weights(slice, replicate);
        let comparator_weights: Vec<f64> = comparators.iter().map(|&i| combined[i]).collect();
        let exposed_weights: Vec<f64> = exposed.iter().map(|&i| row[i]).collect();
        let fit = gamma_design
            .fit(
                &comparator_weights,
                &gamma.beta,
                &gamma_options,
                &mut scratch,
            )
            .expect("compact gamma");
        h.mat(&fit.beta);
        h.n(fit.iterations);
        h.f(prediction.mean_gamma(&fit.beta, &exposed_weights));
        let fit = two_part_design
            .fit(&comparator_weights, &warm, &two_part_options, &mut scratch)
            .expect("compact two-part");
        h.mat(&fit.beta_logit);
        h.mat(&fit.beta_gamma);
        h.f(prediction.mean_two_part(&fit.beta_logit, &fit.beta_gamma, &exposed_weights));
    }
    out.push(("compact", h.finish()));
}

fn comparison(out: &mut Vec<(&'static str, u64)>, slice: &Slice) {
    let comparison = compare::compare(&slice.data(), &CompareOptions::new(42)).expect("compare");
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
    let mut outcome = vec![0.0; rows];
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
            outcome[i] = if stream.unit() < p {
                (7.5 + 0.4 * f64::from(exposed) + level + 0.6 * (stream.unit() - 0.5)).exp()
            } else {
                0.0
            };
            subject_ids.push(u32::try_from(subject + 1).expect("small"));
            time.push(year as f64);
            families.push(u32::try_from(subject / 2 + 1).expect("small"));
        }
    }
    let input = LongitudinalData::builder(
        &outcome,
        design.as_ref(),
        design.as_ref(),
        &subject_ids,
        &time,
    )
    .families(&families)
    .build()
    .expect("data");
    let options = MtpOptions {
        family_random_effects: FamilyRandomEffects::Disabled,
        chains: 4,
        ..MtpOptions::new(SEED)
    };
    let model = mtp::fit(&input, &options).expect("mtp");
    let mut h = Hash::new();
    let convergence = model.convergence.as_ref().expect("convergence");
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
    for chain in &model.chains {
        h.n(chain.diagnostics.iterations_completed);
        h.n(chain.diagnostics.retained_draws);
    }
    if let Some(summary) = &model.posterior_summary {
        for parameter in summary.alpha.iter().chain(&summary.beta) {
            for value in [
                parameter.mean,
                parameter.std_dev,
                parameter.q025,
                parameter.q50,
                parameter.q975,
            ] {
                // Every summary of these fits is defined.
                h.f(value.expect("a defined summary"));
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
