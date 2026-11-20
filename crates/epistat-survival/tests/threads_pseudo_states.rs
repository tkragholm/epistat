//! Pseudo-values of a multi-state process give identical bits in rayon pools of
//! 1, 2, 3, 4 and 8 threads, for the three targets, with two strata, delayed
//! entry, case weights, subjects of several rows and more subjects than a block
//! of the influence recursion holds, with and without a bound on the influence
//! held at once.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::pseudo::{StateOptions, StatePseudo, StateTarget, pseudo_states};
use epistat_testkit::{POOLS, assert_same_on_pools};

struct Sample {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    id: Vec<u32>,
    istate: Vec<u32>,
}

/// Subjects that start in state 1, move to 2 or 3 and from 2 to 3, with
/// tied times on a grid of tenths and censoring.
fn sample(subjects: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        start: Vec::new(),
        stop: Vec::new(),
        status: Vec::new(),
        weight: Vec::new(),
        stratum: Vec::new(),
        id: Vec::new(),
        istate: Vec::new(),
    };
    for subject in 0..subjects {
        let (stratum, weight) = (
            5 * rng.below(2) as u32 + 3,
            if rng.below(30) == 0 {
                0.0
            } else {
                0.5 + rng.uniform() * 2.0
            },
        );
        let mut time = 0.0f64;
        let mut state = 1u32;
        loop {
            let next = time + 0.1 + (rng.exp1() * 30.0).floor() / 10.0;
            let (status, done) = if rng.below(8) == 0 {
                (0, true)
            } else if state == 1 {
                (if rng.below(3) == 0 { 3 } else { 2 }, false)
            } else {
                (3, true)
            };
            s.start.push(time);
            s.stop.push(next);
            s.status.push(status);
            s.weight.push(weight);
            s.stratum.push(stratum);
            s.id.push(subject as u32 * 3 + 1);
            s.istate.push(state);
            if done || status == 3 {
                break;
            }
            time = next;
            state = status as u32;
        }
    }
    s
}

fn flatten(p: &StatePseudo, out: &mut Vec<f64>) {
    out.extend([p.n_rows as f64, p.n_used as f64, p.width as f64]);
    out.extend(
        p.transitions
            .iter()
            .flat_map(|&(a, b)| [f64::from(a), f64::from(b)]),
    );
    out.extend(p.group.iter().map(|&g| f64::from(g)));
    out.extend(p.subject.iter().map(|&g| f64::from(g)));
    out.extend_from_slice(&p.values);
    for g in &p.groups {
        out.push(f64::from(g.group));
        out.extend_from_slice(&g.estimate);
        out.extend(g.beyond.iter().map(|&b| f64::from(u8::from(b))));
    }
}

#[test]
fn multistate_pseudo_does_not_depend_on_the_number_of_threads() {
    let s = sample(700, 83);
    let times = [0.5, 1.5, 3.0, 6.0, 12.0];
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(3)
        .start(&s.start)
        .weight(&s.weight)
        .stratum(&s.stratum)
        .id(&s.id)
        .istate(&s.istate)
        .build()
        .unwrap();
    for target in [
        StateTarget::Occupation,
        StateTarget::TimeInState,
        StateTarget::CumulativeHazard,
    ] {
        for bound in [None, Some(7000)] {
            let mut opts = StateOptions::new(target, 0.0);
            opts.max_influence_values = bound;
            assert_same_on_pools(&POOLS, || {
                let p = pseudo_states(&data, &times, &opts).unwrap();
                let mut out = Vec::new();
                flatten(&p, &mut out);
                assert!(out.len() > 5000);
                out
            });
        }
    }
}
