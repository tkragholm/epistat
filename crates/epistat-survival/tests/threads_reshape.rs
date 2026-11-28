//! The data tools give identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads. Condensing sorts in parallel, the others are sequential, and all
//! three are run on random data on a coarse grid so that times tie.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::condense::{CondenseOptions, condense};
use epistat_survival::split::{SplitOptions, split};
use epistat_survival::tmerge::{Observations, Spans, Tmerge, TmergeOptions};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn split_and_condense_do_not_depend_on_the_number_of_threads() {
    let n = 100_003;
    let mut rng = Stream::seeded(7);
    let mut start = Vec::with_capacity(n);
    let mut stop = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    let mut key = Vec::with_capacity(n);
    for _ in 0..n {
        let begin = rng.below(30) as f64;
        start.push(begin);
        stop.push(begin + 1.0 + rng.below(4) as f64);
        status.push(rng.below(2) as i32);
        id.push(1000 * rng.below(5_011) as u32 + 7);
        key.push(rng.below(2) as u64);
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .n_codes(1)
        .build()
        .unwrap();
    let cuts = [3.0, 9.5, 10.0, 21.0];
    assert_same_on_pools(&POOLS, || {
        let s = split(&data, &cuts, &SplitOptions::default()).unwrap();
        let mut out: Vec<f64> = Vec::new();
        out.extend(&s.start);
        out.extend(&s.stop);
        out.extend(s.row.iter().map(|&r| f64::from(r)));
        out.extend(s.episode.iter().map(|&r| f64::from(r)));
        for merge in [true, false] {
            let c = condense(
                &data,
                &key,
                &CondenseOptions {
                    merge_over_events: merge,
                    ..Default::default()
                },
            )
            .unwrap();
            out.extend(&c.start);
            out.extend(&c.stop);
            out.extend(c.row.iter().map(|&r| f64::from(r)));
            assert!(c.row.len() < n);
        }
        out
    });
}

#[test]
fn tmerge_does_not_depend_on_the_number_of_threads() {
    let n = 20_011;
    let mut rng = Stream::seeded(8);
    let base: Vec<u32> = (0..n as u32).map(|i| 3 * i + 1).collect();
    let tstop: Vec<f64> = (0..n).map(|_| 5.0 + rng.below(30) as f64).collect();
    let m = 60_001;
    let oid: Vec<u32> = (0..m).map(|_| 3 * rng.below(n) as u32 + 1).collect();
    let time: Vec<Option<f64>> = (0..m).map(|_| Some(rng.below(40) as f64 / 2.0)).collect();
    let value: Vec<Option<f64>> = (0..m).map(|_| Some(rng.below(5) as f64)).collect();
    assert_same_on_pools(&POOLS, || {
        let mut tm = Tmerge::new(
            &base,
            &Spans {
                id: &base,
                tstart: None,
                tstop: &tstop,
            },
            TmergeOptions::default(),
        )
        .unwrap();
        let obs = Observations {
            id: &oid,
            time: (&time).into(),
            value: Some((&value).into()),
        };
        tm.tdc("x", &obs, None).unwrap();
        tm.cumtdc("c", &obs, None).unwrap();
        tm.event("e", &obs).unwrap();
        tm.cumevent("k", &obs).unwrap();
        let mut out: Vec<f64> = Vec::new();
        out.extend(tm.start());
        out.extend(tm.stop());
        out.extend(tm.row().iter().map(|&r| f64::from(r)));
        for v in tm.variables() {
            out.extend(v.values.iter().map(|x| x.unwrap_or(f64::NAN)));
        }
        assert!(tm.len() > n);
        out
    });
}
