//! The Cox and Fine-Gray fits of the edu_labour_trajectories analysis in PHD
//! (Study III), hashed to the bit.
//!
//! `production_handlers/analysis_primary/hazards.py` fits the log hazard ratio
//! of one binary exposure through `dansk_register.estimation.cox_fit` and
//! `fine_gray_fit`, which reach this crate through `kernels::survival`. That
//! module builds a [`SurvData`] with the case weight, the cluster and the
//! stratum, gives `cox::fit` a one-column design and `CoxOptions` that differ
//! from the default only in the tie rule, and gives `fine_gray::fit` the same
//! with `FineGrayOptions::new(cause)`. Every fit uses Efron's approximation.
//! No entry time, offset, starting values or iteration limits are set, so the
//! defaults of `coxph` apply: at most 20 trial points, a relative tolerance of
//! 1e-9 and the `aeqSurv` rule for times.
//!
//! Each case mirrors one call of that file:
//!
//! - cause-specific hazard: `fit_binary_cox` with the match weight, the family
//!   as cluster and the outcome as the event, death as censoring;
//! - the same without a cluster, for an event frame that has no `family_id`;
//! - subdistribution hazard: `fit_fine_gray` with the match weight and the
//!   family as cluster, death as the competing event;
//! - the same without a cluster;
//! - sibling fixed effects: `fit_binary_cox` on the families that have both an
//!   exposed and an unexposed member, stratified and clustered on the family.
//!
//! The data have the shape of the analysis: thousands of rows in families of
//! one to six, a binary exposure that is shared within most families, match
//! weights of a quarter to one, follow-up in whole days so that events tie, an
//! outcome, a competing death and censoring. Each case hashes everything the
//! fit returns: coefficients, standard errors, both covariance matrices, the
//! log partial likelihoods, the five tests, the counts, the iterations and
//! whether it converged.
//!
//! A change that moves a hash moves the hazard ratios of Study III. The hashes
//! were recorded on aarch64 macOS and are asserted only there. Elsewhere the
//! test computes them and, when `EPISTAT_HASH_OUT` is set, appends them to that
//! file. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.

use epistat_linalg::MatRef;
use epistat_survival::cox::{self, ChiSquareTest, CoxModelFit, CoxOptions, Ties};
use epistat_survival::fine_gray::{self, FineGrayOptions};
use epistat_survival::{Covariance, SurvData};
use epistat_testkit::{Fingerprint, SplitMix64};

const N_ROWS: usize = 6_000;
const SEED: u64 = 20_261_127;
const OUTCOME: i32 = 1;
const DEATH: i32 = 2;

/// An event frame of the analysis: one row per person.
struct Frame {
    exposed: Vec<f64>,
    /// 0 censored, 1 outcome, 2 death.
    event: Vec<i32>,
    time: Vec<f64>,
    weight: Vec<f64>,
    family: Vec<u32>,
}

fn frame() -> Frame {
    let mut stream = SplitMix64::new(SEED);
    let mut f = Frame {
        exposed: Vec::new(),
        event: Vec::new(),
        time: Vec::new(),
        weight: Vec::new(),
        family: Vec::new(),
    };
    let mut family = 0_u32;
    while f.exposed.len() < N_ROWS {
        family += 1;
        let size = 1 + usize::try_from(stream.below(6)).expect("small");
        let family_exposed = stream.unit() < 0.3;
        let mixed = stream.unit() < 0.45;
        let weight = 0.25 * f64::from(1 + u32::try_from(stream.below(4)).expect("small"));
        for member in 0..size.min(N_ROWS - f.exposed.len()) {
            let exposed = if mixed && member % 2 == 1 {
                !family_exposed
            } else {
                family_exposed
            };
            // The earlier of two uniform draws puts more of the follow-up
            // early, as a cohort with an end of follow-up does. The day is
            // whole, so many rows share a time.
            let day = 1 + stream.below(900).min(stream.below(900));
            let u = stream.below(1000);
            let outcome_below = 120 + 60 * u64::from(exposed);
            let event = if u < outcome_below {
                OUTCOME
            } else if u < outcome_below + 70 {
                DEATH
            } else {
                0
            };
            let time = day;
            f.exposed.push(f64::from(exposed));
            f.event.push(event);
            f.time.push(time as f64);
            f.weight.push(weight);
            f.family.push(family);
        }
    }
    f
}

/// A fingerprint that hashes NaN and `None` alike.
struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, word: u64) {
        self.0 = std::mem::take(&mut self.0).u64(word);
    }

    fn n(&mut self, value: usize) {
        self.word(value as u64);
    }

    fn o(&mut self, value: Option<f64>) {
        let value = value.filter(|v| !v.is_nan());
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn options(&mut self, values: &[Option<f64>]) {
        self.n(values.len());
        for value in values {
            self.o(*value);
        }
    }

    fn covariance(&mut self, covariance: &Covariance) {
        self.options(covariance.as_slice());
    }

    fn test(&mut self, test: Option<ChiSquareTest>) {
        match test {
            Some(t) => {
                self.o(Some(t.statistic));
                self.n(t.df);
                self.o(Some(t.p_value));
            }
            None => self.o(None),
        }
    }

    fn fit(&mut self, fit: &CoxModelFit) {
        self.options(&fit.coefficients);
        self.options(&fit.se);
        self.covariance(&fit.covariance);
        match (&fit.robust_se, &fit.robust_covariance) {
            (Some(se), Some(covariance)) => {
                self.word(1);
                self.options(se);
                self.covariance(covariance);
            }
            _ => self.word(0),
        }
        self.o(fit.log_partial_likelihood);
        self.o(fit.null_log_partial_likelihood);
        let tests = &fit.tests;
        for test in [
            tests.wald,
            tests.robust_wald,
            tests.score,
            tests.robust_score,
            tests.likelihood_ratio,
        ] {
            self.test(test);
        }
        self.o(Some(fit.n_events));
        self.n(fit.n_rows);
        self.n(fit.n_used);
        self.n(fit.rank);
        self.n(fit.iterations);
        self.word(u64::from(fit.converged));
        self.n(fit.monotone_columns.len());
        for column in &fit.monotone_columns {
            self.n(*column);
        }
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

fn column(x: &[f64]) -> MatRef<'_, f64> {
    MatRef::from_column_major_slice(x, x.len(), 1)
}

/// `fit_binary_cox` on the rows `keep`: the outcome is the event, death and
/// censoring are censoring.
fn cause_specific(f: &Frame, keep: &[usize], clustered: bool, stratified: bool) -> CoxModelFit {
    let x: Vec<f64> = keep.iter().map(|&i| f.exposed[i]).collect();
    let status: Vec<i32> = keep
        .iter()
        .map(|&i| i32::from(f.event[i] == OUTCOME))
        .collect();
    let stop: Vec<f64> = keep.iter().map(|&i| f.time[i]).collect();
    let weight: Vec<f64> = keep.iter().map(|&i| f.weight[i]).collect();
    let family: Vec<u32> = keep.iter().map(|&i| f.family[i]).collect();
    let mut builder = SurvData::builder(&stop, &status).weight(&weight);
    if stratified {
        builder = builder.stratum(&family);
    }
    if clustered {
        builder = builder.cluster(&family);
    }
    cox::fit(
        &builder.build().expect("valid data"),
        column(&x),
        &CoxOptions {
            ties: Ties::Efron,
            ..CoxOptions::default()
        },
    )
    .expect("the fit succeeds")
}

/// `fit_fine_gray` with the outcome as the cause and death as the competing
/// event.
fn subdistribution(f: &Frame, clustered: bool) -> CoxModelFit {
    let mut builder = SurvData::builder(&f.time, &f.event)
        .n_codes(2)
        .weight(&f.weight);
    if clustered {
        builder = builder.cluster(&f.family);
    }
    fine_gray::fit(
        &builder.build().expect("valid data"),
        column(&f.exposed),
        &FineGrayOptions {
            ties: Ties::Efron,
            ..FineGrayOptions::new(OUTCOME.unsigned_abs())
        },
    )
    .expect("the fit succeeds")
}

/// The rows of the families with an exposed and an unexposed member.
fn discordant(f: &Frame) -> Vec<usize> {
    let mut levels = std::collections::BTreeMap::<u32, (bool, bool)>::new();
    for (&family, &exposed) in f.family.iter().zip(&f.exposed) {
        let entry = levels.entry(family).or_default();
        if exposed > 0.5 {
            entry.0 = true;
        } else {
            entry.1 = true;
        }
    }
    (0..f.family.len())
        .filter(|&i| levels[&f.family[i]] == (true, true))
        .collect()
}

fn cases() -> Vec<(&'static str, u64)> {
    let f = frame();
    let all: Vec<usize> = (0..f.exposed.len()).collect();
    let within = discordant(&f);
    let mut out = Vec::new();
    let mut record = |name: &'static str, fit: &CoxModelFit| {
        assert!(fit.converged, "{name} converges");
        let mut h = Hash::new();
        h.fit(fit);
        out.push((name, h.finish()));
    };
    record(
        "cause_specific_efron_weighted_clustered",
        &cause_specific(&f, &all, true, false),
    );
    record(
        "cause_specific_efron_weighted",
        &cause_specific(&f, &all, false, false),
    );
    record(
        "subdistribution_efron_weighted_clustered",
        &subdistribution(&f, true),
    );
    record(
        "subdistribution_efron_weighted",
        &subdistribution(&f, false),
    );
    record(
        "sibling_fixed_effects_stratified_clustered",
        &cause_specific(&f, &within, true, true),
    );
    out
}

const GOLDEN: &[(&str, u64)] = &[
    (
        "cause_specific_efron_weighted_clustered",
        0x8507_0140_798d_45ff,
    ),
    ("cause_specific_efron_weighted", 0x4932_0890_175a_c63e),
    (
        "subdistribution_efron_weighted_clustered",
        0x874c_3990_4179_be2f,
    ),
    ("subdistribution_efron_weighted", 0x5b27_aaf9_414d_72d4),
    (
        "sibling_fixed_effects_stratified_clustered",
        0xade7_9267_e287_5988,
    ),
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
