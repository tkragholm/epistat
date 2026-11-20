//! `multistate::variance` gives identical bits in rayon pools of 1, 2, 3, 4 and
//! 8 threads.
//!
//! The data are a seeded illness-death process with delayed entry, weights and
//! clusters, 700 clusters so that the work is split into three blocks of 256
//! clusters and a pool of 3 or 8 threads groups them into waves differently
//! from a pool of 1. The result holds the standard errors, the horizons and the
//! influence at chosen times.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::multistate::{Options, VarianceOptions, variance};
use epistat_survival::timefix::TimeFix;
use epistat_testkit::assert_same_on_pools;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

struct Process {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    istate: Vec<u32>,
    person: Vec<u32>,
    cluster: Vec<u32>,
    weight: Vec<f64>,
}

fn process(people: usize, seed: u64) -> Process {
    let mut rng = Stream::seeded(seed);
    let mut p = Process {
        start: Vec::new(),
        stop: Vec::new(),
        status: Vec::new(),
        istate: Vec::new(),
        person: Vec::new(),
        cluster: Vec::new(),
        weight: Vec::new(),
    };
    for id in 0..people {
        let mut state = rng.below(2) as u32;
        let mut clock = if rng.below(3) == 0 {
            1.0 + rng.below(30) as f64 / 10.0
        } else {
            0.0
        };
        let weight = 0.5 + rng.uniform();
        loop {
            let stop = clock + 0.1 + (rng.exp1() * 30.0).floor() / 10.0;
            let next = match (state, rng.below(10)) {
                (_, 0) => 2,
                (0, _) => 1,
                _ => 0,
            };
            let last = next == 2 || stop > 12.0 || rng.below(6) == 0;
            p.start.push(clock);
            p.stop.push(stop);
            p.istate.push(state + 1);
            p.status.push(if last && next != 2 {
                0
            } else {
                next as i32 + 1
            });
            p.person.push(id as u32);
            p.cluster.push(id as u32 / 2);
            p.weight.push(weight);
            if last {
                break;
            }
            state = next;
            clock = stop;
        }
    }
    p
}

fn data(p: &Process) -> SurvData<'_> {
    SurvData::builder(&p.stop, &p.status)
        .start(&p.start)
        .n_codes(3)
        .id(&p.person)
        .istate(&p.istate)
        .cluster(&p.cluster)
        .weight(&p.weight)
        .build()
        .unwrap()
}

fn event_times(p: &Process) -> Vec<f64> {
    let mut times = p.stop.clone();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

// The times are on a grid of tenths and carry rounding noise, so the tie rule
// is the exact one: the comparison is between thread counts, not with R.
#[test]
fn variance_does_not_depend_on_the_number_of_threads() {
    let p = process(1400, 33);
    let data = data(&p);
    let times = event_times(&p);
    let opts = Options {
        timefix: TimeFix::Exact,
        ..Options::default()
    };
    let mut v = VarianceOptions::new(0.0);
    v.horizons = vec![0.5, 3.0, 20.0];
    v.influence_times = vec![0.0, 1.5, 6.0, 30.0];
    assert_same_on_pools(&POOLS, || {
        let got = variance(&data, &times, &opts, &v).unwrap();
        assert_eq!(got.n_clusters, 700);
        assert!(got.std_err.iter().any(|&x| x > 0.0));
        let mut out = got.probabilities.clone();
        out.extend(&got.std_err);
        out.extend(&got.cumhaz);
        out.extend(&got.std_cumhaz);
        out.extend(&got.std_time);
        out.extend(&got.std_initial);
        out.extend(&got.std_rmean);
        out.extend(&got.influence);
        out
    });
}
