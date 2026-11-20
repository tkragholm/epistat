//! `markov::test` gives the bits of the deprecated `markov_test` on the same
//! stays, and refuses what it should.
//!
//! The process has three states and the status codes 1 to 3. The deprecated
//! function reads the index of each stay's state among the qualifying states,
//! so the same stays are given to it with the states mapped through the list.

#![expect(
    deprecated,
    reason = "the deprecated function is the reference for the bits"
)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::markov::{Options, Stays, markov_test, test};

struct Process {
    start: Vec<f64>,
    stop: Vec<f64>,
    from: Vec<u32>,
    to: Vec<u32>,
    person: Vec<u32>,
    weight: Vec<f64>,
}

fn process(people: usize, seed: u64) -> Process {
    let mut rng = Stream::seeded(seed);
    let mut p = Process {
        start: vec![],
        stop: vec![],
        from: vec![],
        to: vec![],
        person: vec![],
        weight: vec![],
    };
    for id in 0..people {
        let mut state = rng.below(2) as u32;
        let mut clock = if rng.below(3) == 0 {
            1.0 + rng.below(30) as f64 / 10.0
        } else {
            0.0
        };
        let weight = f64::from(u8::from(rng.below(9) != 0));
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
            p.from.push(state);
            p.to.push(if last && next != 2 { 9 } else { next });
            p.person.push(id as u32);
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

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn test_gives_the_bits_of_the_deprecated_function() {
    let p = process(150, 41);
    let status: Vec<i32> =
        p.to.iter()
            .map(|&t| if t < 3 { t as i32 + 1 } else { 0 })
            .collect();
    let istate: Vec<u32> = p.from.iter().map(|&f| f + 1).collect();
    let data = SurvData::builder(&p.stop, &status)
        .start(&p.start)
        .n_codes(3)
        .id(&p.person)
        .istate(&istate)
        .weight(&p.weight)
        .build()
        .unwrap();
    let grid = [0.3, 0.9, 1.6, 2.4, 3.5, 4.8, 6.1];
    let seeds: Vec<u64> = (0..25).map(|r| 77 + r * 7919).collect();
    let opts = Options {
        timefix: epistat_survival::timefix::TimeFix::Exact,
    };
    // (qualifying codes, target code)
    for (qualifying, target) in [
        (vec![1_u32], 2_u32),
        (vec![1, 2], 2),
        (vec![1, 2, 3], 2),
        (vec![2, 1], 1),
        (vec![2, 3], 3),
        (vec![3, 1], 2),
    ] {
        let new = test(&data, &qualifying, target, &grid, &seeds, &opts).unwrap();
        // The rows of weight 0 are left out, and the rest map through the list.
        let keep: Vec<usize> = (0..p.start.len()).filter(|&r| p.weight[r] > 0.0).collect();
        let pick = |v: &[f64]| keep.iter().map(|&r| v[r]).collect::<Vec<f64>>();
        let person: Vec<u32> = keep.iter().map(|&r| p.person[r]).collect();
        let state: Vec<u32> = keep
            .iter()
            .map(|&r| {
                qualifying
                    .iter()
                    .position(|&q| q == p.from[r] + 1)
                    .map_or(99, |i| i as u32)
            })
            .collect();
        let event: Vec<f64> = keep
            .iter()
            .map(|&r| f64::from(u8::from(status[r] == target as i32)))
            .collect();
        let (start, stop) = (pick(&p.start), pick(&p.stop));
        let stays = Stays::new(&person, &start, &stop, &state, &event).unwrap();
        let old = markov_test(&stays, &grid, qualifying.len(), &seeds).unwrap();
        assert_eq!(bits(&new.trace), bits(&old.trace), "{qualifying:?}");
        assert_eq!(bits(&new.variance), bits(&old.variance));
        assert_eq!(bits(&new.n_occupied), bits(&old.n_occupied));
        assert_eq!(bits(&new.n_events_after), bits(&old.n_events_after));
        assert_eq!(bits(&new.statistic), bits(&old.statistic));
        assert_eq!(new.p_value, old.p_value);
        assert_eq!(new.qualifying, qualifying);
        assert_eq!(new.n_rows, p.start.len());
        assert_eq!(new.n_used, keep.len());
    }
}

fn build<'a>(
    cols: (&'a [f64], &'a [f64], &'a [i32], &'a [u32]),
    weight: Option<&'a [f64]>,
    istate: Option<&'a [u32]>,
) -> SurvData<'a> {
    let (stop, start, status, id) = cols;
    let mut b = SurvData::builder(stop, status)
        .start(start)
        .n_codes(2)
        .id(id);
    if let Some(w) = weight {
        b = b.weight(w);
    }
    if let Some(i) = istate {
        b = b.istate(i);
    }
    b.build().unwrap()
}

#[test]
fn the_arguments_and_data_are_checked() {
    let stop = [2.0, 3.0, 2.0];
    let start = [0.0, 2.0, 0.0];
    let status = [2, 0, 0];
    let istate = [1, 2, 1];
    let id = [0, 0, 1];
    let cols = (&stop[..], &start[..], &status[..], &id[..]);
    let good = build(cols, None, Some(&istate));
    let run = |d: &SurvData<'_>, q: &[u32], t: u32, g: &[f64]| {
        test(d, q, t, g, &[], &Options::default()).map(|_| ())
    };
    assert!(run(&good, &[1, 2], 2, &[1.0]).is_ok());
    assert!(run(&good, &[], 2, &[1.0]).is_err());
    assert!(run(&good, &[1, 2], 2, &[]).is_err());
    assert!(run(&good, &[1, 1], 2, &[1.0]).is_err());
    assert!(run(&good, &[0, 1], 2, &[1.0]).is_err());
    assert!(run(&good, &[1, 3], 2, &[1.0]).is_err());
    assert!(run(&good, &[1, 2], 0, &[1.0]).is_err());
    assert!(run(&good, &[1, 2], 3, &[1.0]).is_err());
    assert!(run(&build(cols, None, None), &[1, 2], 2, &[1.0]).is_err());
    assert!(
        run(
            &build(cols, Some(&[1.0, 2.0, 1.0]), Some(&istate)),
            &[1, 2],
            2,
            &[1.0]
        )
        .is_err()
    );
    assert!(
        run(
            &build(cols, Some(&[0.0; 3]), Some(&istate)),
            &[1, 2],
            2,
            &[1.0]
        )
        .is_err()
    );
    assert!(
        run(
            &build(cols, Some(&[1.0, 1.0, 0.0]), Some(&istate)),
            &[1, 2],
            2,
            &[1.0]
        )
        .is_ok()
    );
}
