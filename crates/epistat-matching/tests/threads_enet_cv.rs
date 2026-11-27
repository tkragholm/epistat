//! The cross-validated elastic net gives bit-identical output in rayon pools of
//! 1, 2, 3, 4 and 8 threads.
//!
//! The paths of the folds and of all rows are fitted at the same time, so the
//! comparison covers the penalty path, the cross-validated deviance and its
//! standard error, the chosen penalty, the coefficients and the fitted
//! probabilities. The cohort is simulated from a fixed seed, with a column of
//! large mean and small spread and one that is nearly collinear with another.

use chrono::NaiveDate;
use epistat_matching::{
    BalanceRecord, CovariateValue, ElasticNetCvConfig, PropensityEstimator, PropensityScoreConfig,
    prepare_propensity_distance_config,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

/// A small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

fn cohort(seed: u64, n: usize) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut rng = Lcg(seed);
    let (mut anchors, mut candidates) = (Vec::new(), Vec::new());
    for idx in 0..n {
        let a = rng.next() - 0.5;
        let b = rng.next() - 0.5;
        let c = 2000.0 + 8.0 * (rng.next() - 0.5);
        let d = a + 0.05 * (rng.next() - 0.5);
        let e = f64::from(u8::from(rng.next() < 0.3));
        let score = 0.9 * a - 0.6 * b + 0.05 * (c - 2000.0) + 0.5 * e - 1.0;
        let treated = rng.next() < 1.0 / (1.0 + (-score).exp());
        let mut record = BalanceRecord::new(format!("u{idx}"), birth);
        for (name, value) in [("a", a), ("b", b), ("c", c), ("d", d), ("e", e)] {
            record
                .covariates
                .insert(name.to_string(), CovariateValue::Numeric(value));
        }
        if treated {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (anchors, candidates)
}

fn fit(alpha: f64, standardize: bool) -> Vec<f64> {
    let (anchors, candidates) = cohort(11, 500);
    let config = PropensityScoreConfig {
        estimator: PropensityEstimator::ElasticNetCv(
            ElasticNetCvConfig::builder()
                .seed(5)
                .alpha(alpha)
                .nfolds(10)
                .standardize(standardize)
                .build(),
        ),
        ..PropensityScoreConfig::default()
    };
    let prepared = prepare_propensity_distance_config(&anchors, &candidates, &config, None)
        .expect("cross-validation succeeds");
    let estimate = &prepared.estimate;
    let cv = estimate.cross_validation.as_ref().expect("ran");
    assert_eq!(cv.unconverged_fits, 0);
    let mut out = Vec::new();
    out.extend(&cv.lambda);
    out.extend(&cv.deviance);
    out.extend(&cv.deviance_se);
    out.push(cv.index_chosen as f64);
    out.extend(estimate.coefficients.iter().map(|c| c.expect("fitted")));
    out.extend(&estimate.probabilities);
    out
}

#[test]
fn lasso_does_not_depend_on_the_pool() {
    assert_same_on_pools(&POOLS, || fit(1.0, true));
}

#[test]
fn elastic_net_without_standardising_does_not_depend_on_the_pool() {
    assert_same_on_pools(&POOLS, || fit(0.5, false));
}
