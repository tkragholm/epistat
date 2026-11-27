//! The bits of the attributable fraction of a logistic model on seeded data,
//! cross-sectional and case-control, with and without weights and clusters, as
//! an FNV-1a digest of the `to_bits` of each output with its length. To record
//! again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on the target it was recorded on, aarch64 macOS.
//! Thread-count independence is checked everywhere by `threads_af`. Elsewhere
//! it computes the hashes and, when `EPISTAT_HASH_OUT` is set, appends them to
//! that file.

use epistat_glm::attributable::{AfOptions, Design, IntervalScale, attributable_logistic};
use epistat_glm::{Family, Glm, Link};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::{Fingerprint, SplitMix64};

mod common;
use common::*;

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("cs/plain/identity", 4, 0xebb0_6121_dcb3_eefd),
    ("cs/plain/log", 4, 0xe428_05f7_7d60_38ce),
    ("cs/cluster/identity", 4, 0x4a56_b1a1_b212_c484),
    ("cs/cluster/log", 4, 0x9331_1535_15e0_757e),
    ("cs/weighted/identity", 4, 0x6e35_71df_a647_2368),
    ("cs/weighted/log", 4, 0xc16a_b46e_b42b_9284),
    ("cs/weighted cluster/identity", 4, 0x2a25_fafa_6e57_fdf1),
    ("cs/weighted cluster/log", 4, 0x6cd5_0dca_e90c_725d),
    ("cc/plain/identity", 4, 0x92f8_7007_016a_f4f4),
    ("cc/plain/log", 4, 0x2e9a_cfe3_dabc_eb5e),
    ("cc/cluster/identity", 4, 0xdfd2_05f6_4b5b_659f),
    ("cc/cluster/log", 4, 0x140a_b447_703d_6590),
    ("cc/weighted/identity", 4, 0x9863_3dcb_3a90_7d6f),
    ("cc/weighted/log", 4, 0x3a85_9495_102d_b00d),
    ("cc/weighted cluster/identity", 4, 0x75d2_3425_539a_1806),
    ("cc/weighted cluster/log", 4, 0xb2fb_2d23_3560_4fa9),
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
fn attributable_fraction_bits_are_recorded() {
    let mut rng = SplitMix64::new(68);
    let n = 600_usize;
    let z: Vec<f64> = (0..n)
        .map(|_| {
            let (a, b) = (1.0 - rng.unit(), rng.unit());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect();
    let e: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(rng.unit() < 1.0 / (1.0 + (-0.4 * z[i]).exp()))))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let eta = -1.0 + 0.8 * e[i] + 0.5 * z[i] + 0.3 * e[i] * z[i];
            f64::from(u8::from(rng.unit() < 1.0 / (1.0 + (-eta).exp())))
        })
        .collect();
    let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.unit()).collect();
    let cluster: Vec<u32> = (0..n).map(|i| u32::try_from(i / 3).unwrap()).collect();
    let x = Mat::from_fn(n, 4, |i, j| [1.0, e[i], z[i], e[i] * z[i]][j]);
    let x0 = Mat::from_fn(n, 4, |i, j| if j == 1 || j == 3 { 0.0 } else { x[(i, j)] });
    let model = Glm::new(Family::Binomial, Link::Logit);
    let plain = model.fit5(x.as_ref(), &y, None, None, None).unwrap();
    let weighted = model.fit5(x.as_ref(), &y, Some(&w), None, None).unwrap();
    for (name, design) in [("cs", Design::CrossSectional), ("cc", Design::CaseControl)] {
        let clusters = Clusters::from_labels_first_seen(&cluster);
        let unweighted = glm_data(x.as_ref(), &y, None, None, None).unwrap();
        let with_weights = glm_data(x.as_ref(), &y, Some(&w), None, None).unwrap();
        for (variant, fit, data, cl) in [
            ("plain", &plain, &unweighted, None),
            ("cluster", &plain, &unweighted, Some(&clusters)),
            ("weighted", &weighted, &with_weights, None),
            (
                "weighted cluster",
                &weighted,
                &with_weights,
                Some(&clusters),
            ),
        ] {
            for (scale_name, scale) in [
                ("identity", IntervalScale::Identity),
                ("log", IntervalScale::LogComplement),
            ] {
                let a = attributable_logistic(
                    fit,
                    data,
                    x0.as_ref(),
                    cl,
                    &AfOptions {
                        design,
                        scale,
                        ..AfOptions::default()
                    },
                )
                .unwrap()
                .af
                .unwrap();
                let v: Vec<f64> = [Some(a.estimate), a.se, a.lower(), a.upper()]
                    .map(Option::unwrap)
                    .to_vec();
                check(&format!("{name}/{variant}/{scale_name}"), &v);
            }
        }
    }
}
