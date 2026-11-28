//! `aalen_johansen::variance` gives identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads.
//!
//! Three groups of competing-risks data with delayed entry, weights and
//! clusters, 700 or more clusters in the largest group so that the work is
//! split into at least three blocks of 256 clusters and pools of different
//! sizes group them into waves differently. The result holds the standard
//! errors, the horizons and the influence at chosen times.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the times in the data are exact integers and the sizes are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::aalen_johansen::{
    AalenJohansenOptions, AalenJohansenVarianceOptions, variance,
};
use epistat_survival::timefix::TimeFix;
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn variance_does_not_depend_on_the_number_of_threads() {
    let n = 4200usize;
    let mut rng = Stream::seeded(77);
    let mut stop = Vec::new();
    let mut start = Vec::new();
    let mut status = Vec::new();
    let mut weight = Vec::new();
    let mut cluster = Vec::new();
    let mut stratum = Vec::new();
    for i in 0..n {
        let g = (rng.below(7) / 3) as u32;
        let entry = if rng.below(3) == 0 {
            rng.below(5) as f64
        } else {
            0.0
        };
        stop.push(entry + 1.0 + rng.below(20) as f64);
        start.push(entry);
        status.push(rng.below(4) as i32);
        weight.push(0.5 + rng.uniform());
        cluster.push(i as u32 / 3);
        stratum.push(g);
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(3)
        .weight(&weight)
        .cluster(&cluster)
        .build()
        .unwrap();
    let opts = AalenJohansenOptions {
        timefix: TimeFix::Exact,
    };
    let v = AalenJohansenVarianceOptions {
        horizons: vec![0.5, 3.0, 20.0, 40.0],
        influence_times: vec![0.5, 4.0, 9.0, 30.0],
    };
    assert_same_on_pools(&POOLS, || {
        let got = variance(&data, Some(&stratum), &opts, &v).unwrap();
        assert!(got.groups.iter().any(|g| g.clusters.len() > 512));
        let mut out = got.time.clone();
        out.extend(&got.event_free);
        out.extend(&got.std_event_free);
        out.extend(&got.std_time_event_free);
        for c in 0..3 {
            out.extend(&got.cif[c]);
            out.extend(&got.std_cif[c]);
            out.extend(&got.cumhaz[c]);
            out.extend(&got.std_cumhaz[c]);
            out.extend(&got.std_time[c]);
        }
        for g in &got.groups {
            out.extend(&g.std_rmean);
            let inf = g.influence.as_ref().unwrap();
            out.extend((0..inf.n_columns()).flat_map(|j| inf.column(j).to_vec()));
        }
        out
    });
}
