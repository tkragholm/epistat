//! The bits of the expected hazards and of the estimators that read them, on a
//! cohort with strata, rows of weight 0 and delayed entry, including the
//! values at the cuts of the rate table.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output,
//! with its length, taken from the expected hazards held as pieces of every
//! row, before they were formed one row at a time. They must not change. To
//! record again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste the
//! lines. The lookups were recorded again when `cumulative` and `hazard`
//! came to be defined beyond the follow-up of a row and `None` for a row of
//! weight 0; the lookups within the follow-up of the used rows kept their
//! bits.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on aarch64 macOS, where it was recorded. Elsewhere it
//! computes the hashes and, when `EPISTAT_HASH_OUT` is set, appends them to
//! that file.

mod relsurv_sim;

use epistat_survival::Level;
use epistat_survival::SurvData;
use epistat_survival::relsurv::{
    ExcessOptions, ExpectedHazard, Grid, NetMethod, NetOptions, NetTransform, excess_hazard,
    net_survival,
};
use epistat_testkit::Fingerprint;
use relsurv_sim::{cohort, table};

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("lookups", 150_000, 0xa22a_ad99_9d04_788a),
    ("pohar perme exact", 29_242, 0x9a0d_d0da_8bc0_33d7),
    ("pohar perme quarterly", 474, 0x1d17_933c_73ac_d470),
    ("ederer2 exact", 29_242, 0xc260_bf74_8882_b2bf),
    ("ederer2 quarterly", 474, 0x20e8_629c_56e4_710f),
    ("excess records", 58, 0xb276_f344_73d3_9e07),
    ("excess aggregated", 46, 0x27fd_379a_82c6_543f),
];

fn outputs() -> Vec<(String, Vec<f64>)> {
    let t = table();
    let c = cohort(6000, 20_261_124, true);
    let n = c.time.len();
    // Every seventh row has weight 0 and so no pieces.
    let weight: Vec<f64> = (0..n).map(|i| if i % 7 == 3 { 0.0 } else { 1.0 }).collect();
    let data = SurvData::builder(&c.time, &c.status)
        .start(&c.start)
        .stratum(&c.group)
        .weight(&weight)
        .build()
        .unwrap();
    let expected = ExpectedHazard::from_table(&t, &data, &[&c.age0, &c.date0], &[&c.sex]).unwrap();
    let mut out = Vec::new();

    let mut lookups = Vec::new();
    for i in 0..n {
        let h = expected.horizon(i);
        lookups.push(h);
        // The first cut of age and of the calendar after the origin, where
        // the hazard is that before the cut.
        let age_cut = 365.0 - c.age0[i] % 365.0;
        let year_cut = 365.0 - (c.date0[i] - relsurv_sim::DAY0) % 365.0;
        for u in [
            -1.0,
            0.0,
            0.5,
            age_cut,
            year_cut,
            age_cut + 365.0,
            h / 3.0,
            h / 2.0,
            h - 0.25,
            h,
            h + 1.0,
            f64::NAN,
        ] {
            lookups.push(expected.cumulative(i, u).unwrap_or(f64::NAN));
            lookups.push(expected.hazard(i, u).unwrap_or(f64::NAN));
        }
    }
    out.push(("lookups".to_string(), lookups));

    for (name, method) in [
        ("pohar perme", NetMethod::PoharPerme),
        ("ederer2", NetMethod::EdererII),
    ] {
        for (label, grid) in [
            ("exact", Grid::EventTimes),
            ("quarterly", Grid::Width(91.3125)),
        ] {
            let opts = NetOptions {
                method,
                grid,
                level: Level::default(),
                transform: NetTransform::Log,
                potential: None,
            };
            let mut v = Vec::new();
            for s in net_survival(&data, &expected, &opts).unwrap() {
                v.push(s.stratum as f64);
                v.extend(&s.times);
                v.extend(&s.cumulative_excess_hazard);
                v.extend(&s.variance);
                v.extend(&s.survival);
                v.extend(
                    s.lower
                        .iter()
                        .chain(&s.upper)
                        .map(|x| x.unwrap_or(f64::NAN)),
                );
                v.extend(&s.n_at_risk);
                v.extend(&s.events);
            }
            out.push((format!("{name} {label}"), v));
        }
    }

    let plain = SurvData::builder(&c.time, &c.status)
        .start(&c.start)
        .weight(&weight)
        .build()
        .unwrap();
    let cuts = [0.0, 180.0, 365.0, 1095.0, 2000.0];
    for (label, aggregate) in [("records", false), ("aggregated", true)] {
        let opts = ExcessOptions {
            aggregate,
            ..ExcessOptions::new(&cuts)
        };
        let x: Vec<&[f64]> = if aggregate {
            vec![&c.x2]
        } else {
            vec![&c.x1, &c.x2]
        };
        let f = excess_hazard(&plain, &x, &expected, &opts).unwrap();
        let mut v: Vec<f64> = f
            .coefficients
            .iter()
            .map(|x| x.unwrap_or(f64::NAN))
            .collect();
        v.extend(
            f.covariance
                .iter()
                .flat_map(epistat_survival::Covariance::as_slice)
                .map(|x| x.unwrap_or(f64::NAN)),
        );
        v.extend([
            f.log_likelihood,
            f.deviance,
            f.iterations as f64,
            f.n_cells as f64,
        ]);
        v.extend(
            f.events
                .iter()
                .chain(&f.expected_events)
                .chain(&f.person_time),
        );
        out.push((format!("excess {label}"), v));
    }
    out
}

#[test]
fn bits_are_unchanged() {
    let got: Vec<(String, usize, u64)> = outputs()
        .into_iter()
        .map(|(name, v)| {
            let digest = Fingerprint::new().u64(v.len() as u64).f64s(&v).finish();
            (name, v.len(), digest)
        })
        .collect();
    for (name, _, digest) in &got {
        epistat_testkit::hash_out(name, *digest);
    }
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        for (name, len, digest) in &got {
            println!("    ({name:?}, {len}, {digest:#020x}),");
        }
        return;
    }
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
    let expected: Vec<(String, usize, u64)> = EXPECTED
        .iter()
        .map(|&(n, l, d)| (n.to_string(), l, d))
        .collect();
    assert_eq!(got, expected);
}
