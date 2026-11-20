//! The Fine-Gray entry points give identical bits in rayon pools of 1, 2, 3,
//! 4 and 8 threads.
//!
//! The rows and the fit are sequential loops, so the parallel code in the
//! path is the tie rule applied to the times. The times are written as `k *
//! 0.1` and `k / 10.0`, which differ in the last bit for some `k`, so the rule
//! has ties to merge. The sample has 2,003 rows, which no pool divides evenly,
//! weights including zeros, three causes and clusters.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::Ties;
use epistat_survival::fine_gray::{
    FineGrayOptions, Rows, RowsOptions, fit, float_codes, risk_rows, rows,
};
use epistat_survival::timefix::TimeFix;
use epistat_testkit::assert_same_on_pools;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

/// A value standing for `None`, so that an absent number differs from any
/// present one.
const ABSENT: u64 = 0x7ff8_0000_dead_beef;

fn opt(value: Option<f64>) -> f64 {
    value.unwrap_or_else(|| f64::from_bits(ABSENT))
}

struct Sample {
    time: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    x: Vec<f64>,
    cluster: Vec<u32>,
}

fn sample(n: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        time: Vec::new(),
        status: Vec::new(),
        weight: Vec::new(),
        x: Vec::new(),
        cluster: Vec::new(),
    };
    for _ in 0..n {
        let k = 1 + rng.below(60);
        s.time.push(if rng.below(2) == 0 {
            k as f64 * 0.1
        } else {
            k as f64 / 10.0
        });
        s.status.push(match rng.below(10) {
            0..=3 => 1,
            4..=5 => 2,
            6 => 3,
            _ => 0,
        });
        s.weight.push(if rng.below(20) == 0 {
            0.0
        } else {
            0.5 + rng.uniform() * 2.0
        });
        s.x.push(rng.below(5) as f64 * 0.5);
        s.cluster.push(100 + rng.below(97) as u32);
    }
    s
}

fn rows_bits(rows: &Rows) -> Vec<f64> {
    let mut out: Vec<f64> = rows.row.iter().map(|&r| f64::from(r)).collect();
    out.extend(&rows.start);
    out.extend(&rows.stop);
    out.extend(rows.event.iter().map(|&e| f64::from(e)));
    out.extend(&rows.weight);
    out
}

fn data(s: &Sample, clustered: bool) -> SurvData<'_> {
    let builder = SurvData::builder(&s.time, &s.status)
        .n_codes(3)
        .weight(&s.weight);
    if clustered {
        builder.cluster(&s.cluster).build().unwrap()
    } else {
        builder.build().unwrap()
    }
}

/// The expansion and the collapsed rows, under the default rule and exact
/// times.
#[test]
fn fine_gray_rows() {
    let s = sample(2003, 41);
    let levels = float_codes(&s.x);
    let d = data(&s, false);
    for timefix in [TimeFix::default(), TimeFix::Exact] {
        let opts = RowsOptions { timefix };
        for cause in 1..=3 {
            assert_same_on_pools(&POOLS, || {
                let mut out = rows_bits(&rows(&d, cause, &opts).unwrap());
                out.extend(rows_bits(&risk_rows(&d, cause, &levels, &opts).unwrap()));
                out
            });
        }
    }
}

/// The fit on the collapsed rows, and on the expanded rows with clusters.
#[test]
fn fine_gray_fit_all_options() {
    let s = sample(2003, 42);
    for ties in [Ties::Breslow, Ties::Efron] {
        for clustered in [false, true] {
            let d = data(&s, clustered);
            assert_same_on_pools(&POOLS, || {
                let fit = fit(
                    &d,
                    &s.x,
                    &FineGrayOptions {
                        cause: 1,
                        ties,
                        timefix: TimeFix::default(),
                    },
                )
                .unwrap();
                assert!(fit.converged && fit.log_hazard_ratio.is_some());
                vec![
                    opt(fit.log_hazard_ratio),
                    opt(fit.standard_error),
                    fit.n_events,
                    fit.n_rows as f64,
                    f64::from(fit.iterations),
                    f64::from(u8::from(fit.converged)),
                    opt(fit.robust_standard_error),
                ]
            });
        }
    }
}
