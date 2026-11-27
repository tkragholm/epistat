//! The windows and exposures of the case-crossover construction on seeded
//! data, as an FNV-1a digest of the `to_bits` of each output with its length.
//! They are integers, so the digest does not depend on the platform. To record
//! again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "the data are seeded days and counts of a few thousand, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::crossover::{
    ExposureIntervals, Measure, Observation, Referent, StratumUnit, build_windows,
    exposure_in_windows,
};
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("month/set", 2276, 0x7c5f_333c_2198_b5d0),
    ("month/start", 2276, 0x4433_a9ec_3192_3050),
    ("month/end", 2276, 0xb0c4_c79e_7224_54f1),
    ("month/counts", 4, 0x3c34_e5ec_52f5_36aa),
    ("month/days", 2276, 0x0fb3_3970_9862_38b8),
    ("year/set", 7704, 0xd33c_68d8_6b36_d3eb),
    ("year/start", 7704, 0xc24c_1bb0_7c45_c1da),
    ("year/end", 7704, 0xc24c_1bb0_7c45_c1da),
    ("year/counts", 4, 0x9b14_8f0e_9b4a_48bf),
    ("year/days", 7704, 0x5074_5d85_ada9_3638),
    ("uni/set", 1966, 0x5d4d_6caf_529b_1d54),
    ("uni/start", 1966, 0x8955_d1f2_e0ff_b23c),
    ("uni/end", 1966, 0x129d_b2ec_c413_0a45),
    ("uni/counts", 4, 0x0221_d767_46ea_5778),
    ("uni/days", 1966, 0x1475_8e8c_8043_a148),
    ("bi/set", 2262, 0x32c9_5023_dccf_8cb1),
    ("bi/start", 2262, 0x1822_ae84_c637_2684),
    ("bi/end", 2262, 0xad9b_f349_7088_f0fa),
    ("bi/counts", 4, 0xfc5d_ecdb_509e_cc18),
    ("bi/days", 2262, 0x0147_1c9c_ab7e_9a20),
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
    assert_eq!(got, (want.1, want.2), "{name}: the output changed");
}

fn as_f64(values: &[i64]) -> Vec<f64> {
    values.iter().map(|&v| v as f64).collect()
}

#[test]
fn crossover_outputs_are_recorded() {
    let mut rng = Stream::seeded(64);
    let n = 600_usize;
    let day = |rng: &mut Stream, a: i64, b: i64| a + rng.below((b - a) as usize) as i64;
    let event: Vec<i64> = (0..n).map(|_| day(&mut rng, 17_532, 18_993)).collect();
    let person: Vec<u32> = (0..n).map(|_| rng.below(80) as u32).collect();
    let start: Vec<i64> = event.iter().map(|&t| t - rng.below(100) as i64).collect();
    let end: Vec<i64> = event.iter().map(|&t| t + rng.below(100) as i64).collect();
    let m = 300_usize;
    let ex_person: Vec<u32> = (0..m).map(|_| rng.below(80) as u32).collect();
    let ex_start: Vec<i64> = (0..m).map(|_| day(&mut rng, 17_500, 19_000)).collect();
    let ex_end: Vec<i64> = ex_start.iter().map(|&s| s + rng.below(40) as i64).collect();
    let exposure = ExposureIntervals {
        person: &ex_person,
        start: &ex_start,
        end: &ex_end,
    };
    let observation = Observation {
        start: &start,
        end: &end,
    };
    for (name, referent, h, lag) in [
        (
            "month",
            Referent::TimeStratified {
                unit: StratumUnit::Month,
            },
            2,
            1,
        ),
        (
            "year",
            Referent::TimeStratified {
                unit: StratumUnit::Year,
            },
            1,
            0,
        ),
        ("uni", Referent::Unidirectional { n: 3, step: 7 }, 3, 2),
        ("bi", Referent::Bidirectional { n: 2, step: 14 }, 7, 0),
    ] {
        let w = build_windows(&event, &person, h, lag, referent, Some(observation)).unwrap();
        check(
            &format!("{name}/set"),
            &w.set.iter().map(|&s| f64::from(s)).collect::<Vec<_>>(),
        );
        check(&format!("{name}/start"), &as_f64(&w.start));
        check(&format!("{name}/end"), &as_f64(&w.end));
        check(
            &format!("{name}/counts"),
            &[
                w.n_referents_outside_observation as f64,
                w.n_referents_on_event_day as f64,
                w.sets_without_referent.len() as f64,
                w.sets_case_outside_observation.len() as f64,
            ],
        );
        check(
            &format!("{name}/days"),
            &exposure_in_windows(&w, &exposure, Measure::Days).unwrap(),
        );
    }
}
