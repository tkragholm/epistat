//! `multistate::estimate` and `multistate::bootstrap` report the rows they
//! used, form the initial distribution from the opening rows, leave rows of
//! weight 0 out and refuse what they should.
//!
//! The process has three states and the status codes 1 to 3. A move to a state
//! that is left out of the analysis is status 0.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::multistate::{Options, bootstrap, estimate};
use epistat_survival::timefix::TimeFix;
use epistat_testkit::bits;

struct Process {
    start: Vec<f64>,
    stop: Vec<f64>,
    from: Vec<u32>,
    to: Vec<u32>,
    person: Vec<u32>,
    weight: Vec<f64>,
    /// The row that opens each person's follow-up.
    first: Vec<u32>,
}

/// People with an illness-death process with recovery. A person may enter late,
/// may be unobserved for a while, and may be censored or move to a state that
/// is left out, which ends the follow-up. With `integer` the times are whole
/// numbers, otherwise tenths.
fn process(people: usize, seed: u64, integer: bool, zeros: bool) -> Process {
    let mut rng = Stream::seeded(seed);
    let mut p = Process {
        start: Vec::new(),
        stop: Vec::new(),
        from: Vec::new(),
        to: Vec::new(),
        person: Vec::new(),
        weight: Vec::new(),
        first: Vec::new(),
    };
    let width = |rng: &mut Stream| {
        if integer {
            1.0 + (rng.exp1() * 2.0).floor()
        } else {
            0.1 + (rng.exp1() * 30.0).floor() / 10.0
        }
    };
    for id in 0..people {
        let mut state = rng.below(2) as u32;
        let mut clock = if rng.below(3) == 0 {
            width(&mut rng)
        } else {
            0.0
        };
        p.first.push(p.start.len() as u32);
        let weight = if zeros && rng.below(7) == 0 {
            0.0
        } else {
            0.5 + rng.uniform()
        };
        loop {
            let stop = clock + width(&mut rng);
            let next = match (state, rng.below(10)) {
                (_, 0) => 2,
                (0, _) => 1,
                _ => 0,
            };
            let (to, last) = match rng.below(12) {
                0 => (9, true),
                1 => (5, true),
                _ if next == 2 || stop > 12.0 => (next, true),
                _ => (next, false),
            };
            p.start.push(clock);
            p.stop.push(stop);
            p.from.push(state);
            p.to.push(to);
            p.person.push(id as u32);
            p.weight.push(weight);
            if last {
                break;
            }
            state = next;
            // A gap with the state unchanged.
            clock = if rng.below(8) == 0 {
                stop + width(&mut rng)
            } else {
                stop
            };
        }
    }
    p
}

fn event_times(p: &Process) -> Vec<f64> {
    let mut times = p.stop.clone();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

struct Columns {
    status: Vec<i32>,
    istate: Vec<u32>,
    cluster: Vec<u32>,
}

fn columns(p: &Process) -> Columns {
    Columns {
        status: p
            .to
            .iter()
            .map(|&to| if to < 3 { to as i32 + 1 } else { 0 })
            .collect(),
        istate: p.from.iter().map(|&from| from + 1).collect(),
        cluster: p.person.iter().map(|&person| person / 3).collect(),
    }
}

fn data<'a>(p: &'a Process, c: &'a Columns, weighted: bool) -> SurvData<'a> {
    let builder = SurvData::builder(&p.stop, &c.status)
        .start(&p.start)
        .n_codes(3)
        .id(&p.person)
        .istate(&c.istate)
        .cluster(&c.cluster);
    if weighted {
        builder.weight(&p.weight)
    } else {
        builder
    }
    .build()
    .unwrap()
}

#[test]
fn estimate_reports_its_rows_and_takes_the_distribution_it_formed() {
    for (seed, integer, zeros, weighted) in [
        (1, false, false, false),
        (2, false, true, true),
        (3, true, true, true),
        (4, true, false, false),
    ] {
        let p = process(70, seed, integer, zeros);
        let c = columns(&p);
        let data = data(&p, &c, weighted);
        let mut times = event_times(&p);
        let mut descending = times.clone();
        descending.reverse();
        times.insert(0, 0.0);
        for times in [&times, &descending] {
            // Exact ties for the tenths, which carry rounding noise that the
            // default rule would merge, and the default rule for whole numbers.
            let timefix = if integer {
                TimeFix::default()
            } else {
                TimeFix::Exact
            };
            let derived = estimate(
                &data,
                times,
                &Options {
                    timefix,
                    ..Options::default()
                },
            )
            .unwrap();
            assert_eq!(derived.n_rows, p.start.len());
            assert_eq!(
                derived.n_used,
                p.weight.iter().filter(|&&w| !weighted || w > 0.0).count()
            );
            assert!((derived.initial.iter().sum::<f64>() - 1.0).abs() < 1e-14);

            // The distribution formed from the data, given, changes nothing.
            let given = Options {
                timefix,
                initial: Some(derived.initial.clone()),
                ..Options::default()
            };
            let again = estimate(&data, times, &given).unwrap();
            assert_eq!(bits(&derived.probabilities), bits(&again.probabilities));
            assert_eq!(bits(&derived.at_risk), bits(&again.at_risk));
        }
    }
}

#[test]
fn the_initial_distribution_is_the_weighted_opening_states() {
    let p = process(70, 6, true, true);
    let c = columns(&p);
    let data = data(&p, &c, true);
    let got = estimate(&data, &event_times(&p), &Options::default()).unwrap();
    let mut expected = [0.0f64; 3];
    for &row in &p.first {
        expected[p.from[row as usize] as usize] += p.weight[row as usize];
    }
    let total: f64 = expected.iter().sum();
    for (state, weight) in expected.iter().enumerate() {
        assert!((got.initial[state] - weight / total).abs() < 1e-14);
    }
}

#[test]
fn bootstrap_reports_its_shape() {
    for (seed, integer, zeros, weighted) in [
        (11, false, true, true),
        (12, true, false, false),
        (13, true, true, true),
    ] {
        let p = process(80, seed, integer, zeros);
        let c = columns(&p);
        let data = data(&p, &c, weighted);
        let ascending = event_times(&p);
        let mut descending = ascending.clone();
        descending.reverse();
        let seeds: Vec<u64> = (0..6).map(|r| 500 + r * 7919).collect();
        let options = Options {
            timefix: if integer {
                TimeFix::default()
            } else {
                TimeFix::Exact
            },
            ..Options::default()
        };
        for (horizons, width) in [(vec![], 3), (vec![2.0, 7.5, 30.0], 3)] {
            for times in [&ascending, &descending] {
                let out = bootstrap(&data, times, &horizons, 0.5, &seeds, &options).unwrap();
                assert_eq!(out.n_replicates, seeds.len());
                assert_eq!(out.n_states, 3);
                assert_eq!(out.n_rows, p.start.len());
                let per_replicate = if horizons.is_empty() {
                    times.len() * width
                } else {
                    horizons.len() * width
                };
                assert_eq!(out.values.len(), seeds.len() * per_replicate);
            }
        }
    }
}

#[test]
fn rows_of_weight_zero_are_not_used_and_change_nothing() {
    let p = process(60, 21, true, false);
    let c = columns(&p);
    let times = event_times(&p);
    let all = data(&p, &c, false);
    let base = estimate(&all, &times, &Options::default()).unwrap();

    // A copy of the process in which every person appears twice, the second
    // copy with weight 0.
    let n = p.start.len();
    let doubled = |column: &[f64]| [column, column].concat();
    let start = doubled(&p.start);
    let stop = doubled(&p.stop);
    let weight: Vec<f64> = (0..2 * n).map(|row| f64::from(u8::from(row < n))).collect();
    let status = [c.status.clone(), c.status.clone()].concat();
    let istate = [c.istate.clone(), c.istate.clone()].concat();
    let id: Vec<u32> = [
        p.person.clone(),
        p.person.iter().map(|x| x + 1000).collect(),
    ]
    .concat();
    let padded = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(3)
        .id(&id)
        .istate(&istate)
        .weight(&weight)
        .build()
        .unwrap();
    let got = estimate(&padded, &times, &Options::default()).unwrap();
    assert_eq!(got.n_rows, 2 * n);
    assert_eq!(got.n_used, n);
    assert_eq!(bits(&got.probabilities), bits(&base.probabilities));
    assert_eq!(bits(&got.at_risk), bits(&base.at_risk));
    assert_eq!(bits(&got.initial), bits(&base.initial));
}

struct Cols<'a> {
    stop: &'a [f64],
    status: &'a [i32],
    cluster: &'a [u32],
}
fn build<'a>(
    cols: &Cols<'a>,
    start: &'a [f64],
    istate: Option<&'a [u32]>,
    id: Option<&'a [u32]>,
    weight: &'a [f64],
) -> SurvData<'a> {
    let mut builder = SurvData::builder(cols.stop, cols.status)
        .start(start)
        .n_codes(2)
        .cluster(cols.cluster)
        .weight(weight);
    if let Some(istate) = istate {
        builder = builder.istate(istate);
    }
    if let Some(id) = id {
        builder = builder.id(id);
    }
    builder.build().unwrap()
}

#[test]
fn the_data_are_checked_before_estimating() {
    let times = [1.0, 2.0, 3.0];
    let stop = [2.0, 3.0, 2.0];
    let start = [0.0, 2.0, 0.0];
    let status = [2, 0, 0];
    let istate = [1, 2, 1];
    let id = [0, 0, 1];
    let cluster = [0, 0, 1];
    let cols = Cols {
        stop: &stop,
        status: &status,
        cluster: &cluster,
    };
    let ones = [1.0; 3];
    let options = Options::default();
    // Clean data: a gap is allowed, so is nothing else to report.
    let good = build(&cols, &start, Some(&istate), Some(&id), &ones);
    assert!(estimate(&good, &times, &options).is_ok());
    assert!(bootstrap(&good, &times, &[], 0.0, &[1, 2], &options).is_ok());

    let message = |result: Result<_, epistat_survival::Error>| match result {
        Ok(()) => "ok".to_owned(),
        Err(error) => error.to_string(),
    };
    let estimate_of = |data: &SurvData<'_>, options: &Options| {
        message(estimate(data, &times, options).map(|_| ()))
    };

    // No id column.
    let data = build(&cols, &start, Some(&istate), None, &ones);
    assert!(estimate_of(&data, &options).contains("id column"));
    // No starting state on the first row.
    let data = build(&cols, &start, None, Some(&id), &ones);
    assert!(estimate_of(&data, &options).contains("no starting state"));
    // Overlapping rows of one subject.
    let overlapping = [0.0, 1.0, 0.0];
    let data = build(&cols, &overlapping, Some(&istate), Some(&id), &ones);
    assert!(estimate_of(&data, &options).contains("overlap"));
    // A state that the earlier rows do not lead to.
    let jumping = [1, 1, 1];
    let data = build(&cols, &start, Some(&jumping), Some(&id), &ones);
    assert!(estimate_of(&data, &options).contains("teleport"));
    // No row of positive weight.
    let data = build(&cols, &start, Some(&istate), Some(&id), &[0.0; 3]);
    assert!(estimate_of(&data, &options).contains("positive weight"));
    // A distribution that is not one.
    for initial in [
        vec![0.5, 0.4],
        vec![1.0],
        vec![1.5, -0.5],
        vec![f64::NAN, 1.0],
    ] {
        let given = Options {
            initial: Some(initial),
            ..Options::default()
        };
        assert_ne!(estimate_of(&good, &given), "ok");
    }
    // A rate tolerance that is negative or not finite has no meaning.
    for tolerance in [-1e-12, f64::NAN, f64::INFINITY] {
        let given = Options {
            rate_tolerance: Some(tolerance),
            ..Options::default()
        };
        assert!(estimate_of(&good, &given).contains("rate_tolerance"));
    }
    let zero = Options {
        rate_tolerance: Some(0.0),
        ..Options::default()
    };
    assert_eq!(estimate_of(&good, &zero), "ok");
    // The bootstrap needs clusters and forms the distribution itself.
    let given = Options {
        initial: Some(vec![0.5, 0.5]),
        ..Options::default()
    };
    assert!(
        message(bootstrap(&good, &times, &[], 0.0, &[1], &given).map(|_| ()))
            .contains("in each replicate")
    );
    let no_cluster = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(2)
        .id(&id)
        .istate(&istate)
        .build()
        .unwrap();
    assert!(
        message(bootstrap(&no_cluster, &times, &[], 0.0, &[1], &options).map(|_| ()))
            .contains("cluster column")
    );
}
