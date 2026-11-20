//! The Aalen-Johansen estimate and its bootstrap give identical bits in rayon
//! pools of 1, 2, 3, 4 and 8 threads.
//!
//! The inputs are drawn from a seeded stream on a grid of tenths, so that
//! times tie. The number of rows is odd and above rayon's cut-off for a
//! parallel sort, there are seven strata of unequal size, and 37 replicates
//! are not divisible by any pool size.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::aalen_johansen::{
    AalenJohansenOptions, BootstrapOptions, Draw, RowWindows, bootstrap, estimate,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

struct Sample {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    cluster: Vec<u32>,
}

fn sample(n: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        start: Vec::new(),
        stop: Vec::new(),
        status: Vec::new(),
        weight: Vec::new(),
        stratum: Vec::new(),
        cluster: Vec::new(),
    };
    for _ in 0..n {
        let start = (rng.uniform() * 30.0).floor() / 10.0;
        s.start.push(start);
        s.stop
            .push(start + 0.1 + (rng.exp1() * 40.0).floor() / 10.0);
        s.status.push(match rng.below(10) {
            0..=3 => 1,
            4..=5 => 2,
            _ => 0,
        });
        // Some rows have weight 0 and are not used.
        s.weight.push(if rng.below(25) == 0 {
            0.0
        } else {
            0.5 + rng.uniform() * 2.0
        });
        s.stratum.push(5 * rng.below(7) as u32 + 3);
        s.cluster.push(rng.below(600) as u32 * 2);
    }
    s
}

fn push(out: &mut Vec<f64>, column: &[f64]) {
    out.extend_from_slice(column);
}

#[test]
fn estimate_does_not_depend_on_the_number_of_threads() {
    let s = sample(3001, 21);
    for start in [None, Some(s.start.as_slice())] {
        let mut builder = SurvData::builder(&s.stop, &s.status)
            .n_codes(3)
            .weight(&s.weight)
            .stratum(&s.stratum);
        if let Some(start) = start {
            builder = builder.start(start);
        }
        let data = builder.build().unwrap();
        assert_same_on_pools(&POOLS, || {
            let c = estimate(&data, &AalenJohansenOptions::default()).unwrap();
            let mut out: Vec<f64> = c.causes.iter().map(|&k| f64::from(k)).collect();
            out.extend(c.group.iter().map(|&g| f64::from(g)));
            out.extend([c.n_rows as f64, c.n_used as f64]);
            for column in [&c.time, &c.n_risk, &c.n_at_risk, &c.n_event, &c.event_free] {
                push(&mut out, column);
            }
            for column in &c.cif {
                push(&mut out, column);
            }
            out
        });
    }
}

/// Seven cells of unequal size over a frame of 2401 rows, overlapping. Cell 5
/// holds only exposed rows and is not estimable.
#[test]
fn bootstrap_does_not_depend_on_the_number_of_threads() {
    let n = 2401;
    let s = sample(n, 22);
    let mut rng = Stream::seeded(23);
    let exposed: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(2)
        .weight(&s.weight)
        .cluster(&s.cluster)
        .build()
        .unwrap();
    let mut rows = Vec::new();
    let mut offsets = vec![0u32];
    for (cell, len) in [900usize, 31, 407, 1500, 9, 120, 61]
        .into_iter()
        .enumerate()
    {
        let start = cell * 137 % (n - len);
        for (row, &is_exposed) in exposed.iter().enumerate().skip(start).take(len) {
            if cell != 5 || is_exposed {
                rows.push(row as u32);
            }
        }
        offsets.push(rows.len() as u32);
    }
    let cells = RowWindows::new(rows, offsets, n, "cell").unwrap();
    let seeds: Vec<u64> = (0..cells.len() as u64).map(|c| 1000 + 7919 * c).collect();
    let ages = [3.0, 0.5, 1.0, 2.2, 4.4, 6.0];
    for draw in [Draw::PerCell, Draw::Shared] {
        let opts = BootstrapOptions::new(37, draw);
        assert_same_on_pools(&POOLS, || {
            let (a, b, estimable) =
                bootstrap(&data, &exposed, &cells, &seeds, &ages, 1, &opts).unwrap();
            assert!(estimable[0] && !estimable[5]);
            assert!(a.iter().any(|&v| v > 0.0));
            let mut out = a;
            out.extend(b);
            out.extend(estimable.iter().map(|&e| f64::from(u8::from(e))));
            out
        });
    }
}
