//! `markov::test` gives identical bits in rayon pools of 1, 2, 3, 4 and 8
//! threads.
//!
//! The data are a seeded illness-death process on a grid of tenths, so exits
//! tie. The test is parallel over 7 landmarks, then over 14 (landmark, state)
//! slots that each write a row of the centred scores, then over 33
//! wild-bootstrap replicates reduced by an integer sum.

#![expect(
    clippy::cast_possible_wrap,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_core::{Bootstrap, Seed};
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::markov::{MarkovOptions, test};
use epistat_survival::timefix::TimeFix;
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

#[test]
fn markov_test_does_not_depend_on_the_number_of_threads() {
    let mut rng = Stream::seeded(41);
    let (mut start, mut stop, mut status, mut istate, mut person) =
        (vec![], vec![], vec![], vec![], vec![]);
    for id in 0..350_u32 {
        let mut state = 0_u32;
        let mut clock = 0.0_f64;
        loop {
            let end = clock + 0.1 + (rng.exp1() * 30.0).floor() / 10.0;
            let next = match (state, rng.below(10)) {
                (_, 0) => 2,
                (0, _) => 1,
                _ => 0,
            };
            let last = next == 2 || end > 12.0 || rng.below(6) == 0;
            start.push(clock);
            stop.push(end);
            istate.push(state + 1);
            status.push(if last && next != 2 {
                0
            } else {
                next as i32 + 1
            });
            person.push(id);
            if last {
                break;
            }
            state = next;
            clock = end;
        }
    }
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(3)
        .id(&person)
        .istate(&istate)
        .build()
        .unwrap();
    let grid = [0.3, 0.9, 1.6, 2.4, 3.5, 4.8, 6.1];
    let seeds = Bootstrap::new(33, Seed(77));
    // Exact ties: the times carry rounding noise, and the comparison is
    // between thread counts.
    let opts = MarkovOptions {
        timefix: TimeFix::Exact,
    };
    assert_same_on_pools(&POOLS, || {
        let t = test(&data, &[1, 2], 2, &grid, seeds, &opts).unwrap();
        assert!(t.p_value.iter().all(Option::is_some));
        assert!(t.variance.iter().any(|&v| v > 0.0));
        let mut out = t.trace.clone();
        for column in [&t.variance, &t.n_occupied, &t.n_events_after] {
            out.extend(column.iter().copied());
        }
        out.extend(t.statistic.iter().copied().map(absent));
        out.extend(t.p_value.iter().copied().map(absent));
        out.extend([t.n_rows as f64, t.n_used as f64]);
        out
    });
}
