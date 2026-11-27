//! The bits of linear and generalised linear mixed model fits on seeded data,
//! as an FNV-1a digest of the `to_bits` of each output with its length. To
//! record again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste the
//! lines.
//!
//! The bits depend on the platform's `ln`, `exp` and `sqrt` and on faer's
//! kernels, so the test asserts the recorded values only on the target it was
//! recorded on, aarch64 macOS. Thread-count independence is checked everywhere
//! by `threads`. Elsewhere it computes the hashes and, when `EPISTAT_HASH_OUT`
//! is set, appends them to that file.

mod common;

use common::simulate;
use epistat_glm::{Family, Link};
use epistat_mixed::{Criterion, Glmm, Lmm};
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("lmm/reml/intercept/beta", 3, 0xc2b9_b554_003d_477e),
    (
        "lmm/reml/intercept/beta_covariance",
        9,
        0x92f5_3550_c546_4df6,
    ),
    ("lmm/reml/intercept/theta", 1, 0xd3a8_6023_a5bc_72a5),
    (
        "lmm/reml/intercept/sigma criterion",
        2,
        0x9b38_f816_8bd6_0f43,
    ),
    ("lmm/reml/intercept/modes", 700, 0x11bb_1b73_a6bf_05a0),
    (
        "lmm/reml/intercept/mode_covariances",
        700,
        0x6625_4ac4_50a9_d508,
    ),
    ("lmm/reml/slope/beta", 3, 0xe29f_cacf_1fe9_0e6b),
    ("lmm/reml/slope/beta_covariance", 9, 0x2227_6633_1a90_a921),
    ("lmm/reml/slope/theta", 3, 0x862f_14e0_0808_3747),
    ("lmm/reml/slope/sigma criterion", 2, 0x2df5_093c_4517_c665),
    ("lmm/reml/slope/modes", 1400, 0xb2c5_e011_7d57_d50a),
    (
        "lmm/reml/slope/mode_covariances",
        2800,
        0xc660_bc80_c239_6ff1,
    ),
    ("lmm/ml/intercept/beta", 3, 0x69c8_e4dc_8229_afde),
    ("lmm/ml/intercept/beta_covariance", 9, 0x8e5f_fee8_8ac2_9af0),
    ("lmm/ml/intercept/theta", 1, 0x9256_52bc_8c92_0031),
    ("lmm/ml/intercept/sigma criterion", 2, 0xb09c_06c5_fa1c_1b2d),
    ("lmm/ml/intercept/modes", 700, 0x69ab_a83f_deb1_fe33),
    (
        "lmm/ml/intercept/mode_covariances",
        700,
        0x1dab_df4f_3300_5135,
    ),
    ("lmm/ml/slope/beta", 3, 0x9e26_9c3f_5453_e098),
    ("lmm/ml/slope/beta_covariance", 9, 0xe3ee_3ded_57a2_4f77),
    ("lmm/ml/slope/theta", 3, 0x296d_5ebb_974d_213e),
    ("lmm/ml/slope/sigma criterion", 2, 0x1111_8eff_1fcb_e517),
    ("lmm/ml/slope/modes", 1400, 0x732a_938a_23a8_cd0b),
    ("lmm/ml/slope/mode_covariances", 2800, 0x9bb5_4591_4258_c951),
    ("glmm/binomial/1/beta", 3, 0x5b04_e3a1_7541_a036),
    ("glmm/binomial/1/beta_covariance", 9, 0x8af6_05fe_f5b6_f80c),
    ("glmm/binomial/1/theta se loglik", 3, 0x438e_eafb_6805_0502),
    ("glmm/binomial/1/modes", 700, 0x072a_8029_e5fc_8c7c),
    ("glmm/binomial/1/mode_variances", 700, 0x53d4_828b_594a_ee0a),
    ("glmm/binomial/9/beta", 3, 0x4bd7_e5fc_96c9_654b),
    ("glmm/binomial/9/beta_covariance", 9, 0x2a75_cbe7_5f56_6e5a),
    ("glmm/binomial/9/theta se loglik", 3, 0x65ae_a9ff_de37_d819),
    ("glmm/binomial/9/modes", 700, 0xc08a_dfe1_70c9_e6af),
    ("glmm/binomial/9/mode_variances", 700, 0x08ce_69c0_2b33_c5ad),
    ("glmm/poisson/1/beta", 3, 0x5288_c8d1_1313_19b6),
    ("glmm/poisson/1/beta_covariance", 9, 0x16a4_67b2_74ac_a7a4),
    ("glmm/poisson/1/theta se loglik", 3, 0xa55b_fcae_2582_e8f0),
    ("glmm/poisson/1/modes", 700, 0x09bd_8bb0_9459_527c),
    ("glmm/poisson/1/mode_variances", 700, 0xd4c0_93f0_5aeb_8a17),
    ("glmm/poisson/9/beta", 3, 0xca74_a23d_4d36_07bf),
    ("glmm/poisson/9/beta_covariance", 9, 0xa202_dfd2_74c7_9326),
    ("glmm/poisson/9/theta se loglik", 3, 0x8d9c_1634_4ec3_4d94),
    ("glmm/poisson/9/modes", 700, 0x4b3b_c2b1_99db_2a81),
    ("glmm/poisson/9/mode_variances", 700, 0x2b70_6b81_fd97_f60b),
];

#[track_caller]
fn check(name: &str, values: &[f64]) {
    let got = (values.len(), Fingerprint::new().f64s(values).finish());
    epistat_testkit::hash_out(name, got.1);
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        let hex = format!("{:016x}", got.1);
        let hex: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
        println!("    (\"{name}\", {}, 0x{}),", got.0, hex.join("_"));
        return;
    }
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
    let want = EXPECTED
        .iter()
        .find(|entry| entry.0 == name)
        .unwrap_or_else(|| panic!("{name} has no recorded value"));
    assert_eq!(
        got,
        (want.1, want.2),
        "{name}: the bits of the output changed"
    );
}

#[test]
fn mixed_model_bits_are_recorded() {
    let d = simulate(700, "gaussian", 21);
    for (label, criterion) in [("reml", Criterion::Reml), ("ml", Criterion::Ml)] {
        for (shape, slope) in [("intercept", None), ("slope", Some(&d.slope[..]))] {
            let fit = Lmm::new(criterion)
                .fit(
                    d.x.as_ref(),
                    &d.y,
                    &d.cluster,
                    slope,
                    Some(&d.weights),
                    Some(&d.offset),
                )
                .unwrap();
            let name = format!("lmm/{label}/{shape}");
            check(&format!("{name}/beta"), &fit.beta);
            check(&format!("{name}/beta_covariance"), &fit.beta_covariance);
            check(&format!("{name}/theta"), &fit.theta);
            check(
                &format!("{name}/sigma criterion"),
                &[fit.sigma, fit.criterion],
            );
            check(&format!("{name}/modes"), &fit.modes);
            check(&format!("{name}/mode_covariances"), &fit.mode_covariances);
        }
    }
    for (kind, family, link) in [
        ("binomial", Family::Binomial, Link::Logit),
        ("poisson", Family::Poisson, Link::Log),
    ] {
        let d = simulate(700, kind, 22);
        for nodes in [1, 9] {
            let fit = Glmm::new(family, link, nodes)
                .fit(
                    d.x.as_ref(),
                    &d.y,
                    &d.cluster,
                    Some(&d.weights),
                    Some(&d.offset),
                )
                .unwrap();
            let name = format!("glmm/{kind}/{nodes}");
            check(&format!("{name}/beta"), &fit.beta);
            check(
                &format!("{name}/beta_covariance"),
                fit.beta_covariance.as_ref().unwrap(),
            );
            check(
                &format!("{name}/theta se loglik"),
                &[fit.theta, fit.theta_se.unwrap(), fit.log_likelihood],
            );
            check(&format!("{name}/modes"), &fit.modes);
            check(&format!("{name}/mode_variances"), &fit.mode_variances);
        }
    }
}
