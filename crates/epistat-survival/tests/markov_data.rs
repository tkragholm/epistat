//! `markov::test` reports the rows it used, gives no statistic to a state
//! without variance, and refuses what it should.
//!
//! The process has three states and the status codes 1 to 3.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_core::Bootstrap;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::markov::{MarkovOptions, test};

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

#[test]
fn test_reports_its_rows_and_leaves_a_state_without_variance_untested() {
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
    let seeds = Bootstrap::new(25, 77);
    let opts = MarkovOptions {
        timefix: epistat_survival::timefix::TimeFix::Exact,
    };
    let mut undefined = 0;
    // (qualifying codes, target code)
    for (qualifying, target) in [
        (vec![1_u32], 2_u32),
        (vec![1, 2], 2),
        (vec![1, 2, 3], 2),
        (vec![2, 1], 1),
        (vec![2, 3], 3),
        (vec![3, 1], 2),
    ] {
        let new = test(&data, &qualifying, target, &grid, seeds, &opts).unwrap();
        let width = qualifying.len();
        assert_eq!(new.trace.len(), grid.len() * width);
        assert_eq!(new.variance.len(), grid.len() * width);
        assert_eq!(new.n_occupied.len(), grid.len() * width);
        assert_eq!(new.n_events_after.len(), grid.len());
        for j in 0..width {
            let defined = (0..grid.len()).any(|l| new.variance[l * width + j] > 0.0);
            assert_eq!(new.statistic[j].is_some(), defined);
            assert_eq!(new.p_value[j].is_some(), defined);
            undefined += usize::from(!defined);
        }
        assert_eq!(new.qualifying, qualifying);
        assert_eq!(new.n_rows, p.start.len());
        assert_eq!(new.n_used, p.weight.iter().filter(|&&w| w > 0.0).count());
    }
    assert!(undefined > 0, "the fixture has a state without a test");
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
        test(d, q, t, g, Bootstrap::new(0, 0), &MarkovOptions::default()).map(|_| ())
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
