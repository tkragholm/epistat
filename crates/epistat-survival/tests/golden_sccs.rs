//! The bits of the pieces and of the fit of the self-controlled case series on
//! seeded data.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output
//! field, with its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test runs only on
//! the target it was recorded on, aarch64 macOS. Thread-count independence is
//! checked everywhere by `threads_sccs`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_rng::Stream;
use epistat_survival::sccs::{
    ExposureSeries, PieceSpec, SccsOptions, WindowSpec, build_pieces, fit,
};
use epistat_testkit::{Fingerprint, absent_all};

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("pieces/start", 2943, 0x9621_71e3_2832_b448),
    ("pieces/end", 2943, 0x5fc6_0584_21ce_2d85),
    ("pieces/events", 2943, 0x9872_8861_7f2b_9cf8),
    ("fit/coefficients", 6, 0x4dca_56a9_d141_90dd),
    ("fit/covariance", 36, 0xcff8_7b09_7253_be10),
    ("fit/robust", 36, 0x620b_36f7_eff1_bd1e),
    ("fit/log_likelihood", 1, 0xf4f7_cd1b_91ca_7aad),
    ("fit/null_log_likelihood", 1, 0xac7a_ed60_35bd_b6a4),
    ("fit/iterations", 1, 0xa891_5832_27eb_9a75),
];

fn digest(values: &[f64]) -> u64 {
    Fingerprint::new().f64s(values).finish()
}

#[track_caller]
fn check(name: &str, values: &[f64]) {
    let got = (values.len(), digest(values));
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        let hex = format!("{:016x}", got.1);
        let hex: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
        println!("    (\"{name}\", {}, 0x{}),", got.0, hex.join("_"));
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
fn sccs_bits_are_recorded() {
    let mut rng = Stream::seeded(62);
    let n = 400_usize;
    let (mut start, mut end) = (vec![], vec![]);
    let (mut event_offsets, mut events) = (vec![0_usize], vec![]);
    let (mut exposure_offsets, mut exposures) = (vec![0_usize], vec![]);
    for _ in 0..n {
        let a = rng.below(100) as f64;
        let length = 300 + rng.below(400);
        let b = a + length as f64;
        start.push(a);
        end.push(b);
        for _ in 0..rng.below(3) {
            exposures.push(a + rng.below(300) as f64);
        }
        exposure_offsets.push(exposures.len());
        for _ in 0..=rng.below(3) {
            events.push(a + rng.below(length) as f64);
        }
        event_offsets.push(events.len());
    }
    let windows = [
        WindowSpec {
            from: -10.0,
            to: 0.0,
        },
        WindowSpec {
            from: 0.0,
            to: 14.0,
        },
        WindowSpec {
            from: 14.0,
            to: 60.0,
        },
    ];
    let spec = PieceSpec {
        start: &start,
        end: &end,
        event_offsets: &event_offsets,
        events: &events,
        exposures: &[ExposureSeries {
            offsets: &exposure_offsets,
            dates: &exposures,
            windows: &windows,
        }],
        age_cuts: &[150.0, 300.0, 450.0],
        calendar: None,
    };
    let pieces = build_pieces(&spec).unwrap();
    check("pieces/start", &pieces.start);
    check("pieces/end", &pieces.end);
    check("pieces/events", &pieces.events);
    let length = pieces.length();
    let x = pieces.design();
    let opts = SccsOptions {
        robust: true,
        ..SccsOptions::default()
    };
    let f = fit(&pieces.data(&length), x.as_ref(), &opts).unwrap();
    check("fit/coefficients", &absent_all(&f.coefficients));
    check("fit/covariance", &absent_all(&f.covariance));
    check(
        "fit/robust",
        &absent_all(f.robust_covariance.as_ref().unwrap()),
    );
    check("fit/log_likelihood", &[f.log_likelihood.unwrap()]);
    check("fit/null_log_likelihood", &[f.null_log_likelihood.unwrap()]);
    check("fit/iterations", &[f.iterations as f64]);
}
