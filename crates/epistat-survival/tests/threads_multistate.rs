//! `multistate::estimate` and `multistate::bootstrap` give identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! The data are a seeded illness-death process on a grid of tenths, so stops
//! tie, with delayed entry, weights, clusters and censoring. With ascending
//! times the accumulation walks the precomputed plan, which is sequential in
//! the rows. With descending times there is no plan and it splits over the
//! times in parallel, one risk row and transition block per time, inside the
//! parallel loop over the replicates of the bootstrap. There are 41
//! replicates and about 120 times, and no pool divides either evenly.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_core::Bootstrap;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::multistate::{Options, bootstrap, estimate, integrate};
use epistat_survival::timefix::TimeFix;
use epistat_testkit::{POOLS, assert_same_on_pools};

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
fn options() -> Options {
    Options {
        timefix: TimeFix::Exact,
        ..Options::default()
    }
}

#[test]
fn estimate_does_not_depend_on_the_number_of_threads() {
    let p = process(500, 21);
    let data = data(&p);
    let ascending = event_times(&p);
    let mut descending = ascending.clone();
    descending.reverse();
    for times in [&ascending, &descending] {
        assert_same_on_pools(&POOLS, || {
            let got = estimate(&data, times, &options()).unwrap();
            assert!(got.at_risk.iter().any(|&v| v > 0.0));
            let mut out = got.probabilities.clone();
            out.extend(&got.at_risk);
            out.extend(&got.initial);
            out.extend([got.n_rows as f64, got.n_used as f64]);
            for (horizon, origin) in [(5.0, 0.0), (20.0, 1.3), (0.5, 0.0)] {
                out.extend(integrate(
                    &got.probabilities,
                    times,
                    &got.initial,
                    3,
                    horizon,
                    origin,
                ));
            }
            out
        });
    }
}

#[test]
fn bootstrap_does_not_depend_on_the_number_of_threads() {
    let p = process(400, 22);
    let data = data(&p);
    let ascending = event_times(&p);
    let mut descending = ascending.clone();
    descending.reverse();
    let seeds = Bootstrap::new(41, 500);
    for times in [&ascending, &descending] {
        for horizons in [vec![], vec![2.0, 7.5, 30.0]] {
            assert_same_on_pools(&POOLS, || {
                let got = bootstrap(&data, times, &horizons, 0.5, seeds, &options()).unwrap();
                assert_eq!(got.n_replicates, 41);
                let mut out = got.values;
                out.extend([got.n_rows as f64, got.n_used as f64]);
                out
            });
        }
    }
}
