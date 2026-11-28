//! The bits of net survival, the piecewise excess hazard model and the
//! flexible excess hazard model on a simulated cohort.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on aarch64 macOS, where it was recorded. Thread-count
//! independence is checked everywhere by `threads_relsurv`. Elsewhere it
//! computes the hashes and, when `EPISTAT_HASH_OUT` is set, appends them to
//! that file.

mod relsurv_sim;

use epistat_linalg::Mat;
use epistat_survival::Level;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{self, FlexParOptions, FlexparScale};
use epistat_survival::relsurv::{
    ExcessOptions, Grid, NetMethod, NetOptions, NetTransform, excess_hazard, net_survival,
};
use epistat_testkit::Fingerprint;
use relsurv_sim::{cohort, table};

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("pohar perme exact", 10832, 0x4a00_0e18_5129_bfac),
    ("pohar perme monthly", 1384, 0x5c90_bc31_7e1b_2c36),
    ("ederer2 exact", 10832, 0xeeb1_27f9_f735_7f5e),
    ("ederer2 monthly", 1384, 0x108a_88b7_27c5_b49d),
    ("excess records", 75, 0xf1a8_3522_a4c5_5334),
    ("excess aggregated", 61, 0x8c56_3818_4c99_432f),
    ("flexpar bhazard", 58, 0xc716_5c6d_b16f_2144),
];

fn outputs() -> Vec<(String, Vec<f64>)> {
    let t = table();
    let c = cohort(1500, 31, true);
    let data = c.data(true);
    let expected = c.expected(&t, &data);
    let mut out = Vec::new();
    for (name, method) in [
        ("pohar perme", NetMethod::PoharPerme),
        ("ederer2", NetMethod::EdererII),
    ] {
        for (label, grid) in [
            ("exact", Grid::EventTimes),
            ("monthly", Grid::Width(30.4375)),
        ] {
            let opts = NetOptions {
                method,
                grid,
                level: Level::confidence(0.9).unwrap(),
                transform: NetTransform::LogLog,
                ..Default::default()
            };
            let mut v = Vec::new();
            for s in net_survival(&data, &expected, &opts).unwrap() {
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
    let plain = c.data(false);
    let cuts = [0.0, 90.0, 365.0, 730.0, 1460.0, 2700.0];
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
    let n = c.time.len();
    let bh: Vec<f64> = (0..n)
        .map(|i| expected.hazard(i, c.time[i]).unwrap() * 365.25)
        .collect();
    let time: Vec<f64> = c.time.iter().map(|v| v / 365.25).collect();
    let start: Vec<f64> = c.start.iter().map(|v| v / 365.25).collect();
    let years = SurvData::builder(&time, &c.status)
        .start(&start)
        .build()
        .unwrap();
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { c.x1[i] } else { c.x2[i] });
    let opts = FlexParOptions::new(FlexparScale::Hazard, 4);
    let f = flexpar::fit(&years, x.as_ref(), Some(&bh), &opts).unwrap();
    let mut v = f.gamma.clone();
    v.extend(f.beta.iter().map(|b| b.unwrap_or(f64::NAN)));
    v.extend(
        f.covariance
            .iter()
            .flat_map(epistat_survival::Covariance::as_slice)
            .map(|x| x.unwrap_or(f64::NAN)),
    );
    v.extend([
        f.log_likelihood,
        f64::from(u32::try_from(f.iterations).unwrap()),
    ]);
    out.push(("flexpar bhazard".to_string(), v));
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
            println!("    (\"{name}\", {len}, {digest:#018x}),");
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
