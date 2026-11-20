//! Every public estimator gives identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads.
//!
//! The inputs are built from a seeded stream and are large enough that each
//! parallel loop has more items than the largest pool has threads, with sizes
//! that no pool divides evenly. Each test names the path it exercises.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::aalen_johansen::{
    ArmColumns, Draw, RowWindows, advance, bootstrap_arm_incidence, curves,
};
use epistat_survival::cox::{CoxData, CoxDesign, Ties, fit_cox, fit_cox_single_covariate};
use epistat_survival::fine_gray::{FineGrayData, Rows, fine_gray_fit, float_codes};
use epistat_survival::markov::{Stays, markov_test};
use epistat_survival::multistate::{
    Intervals, bootstrap_state_occupation, integrate, state_occupation_curve,
};
use epistat_testkit::assert_same_on_pools;
use faer::Mat;

const POOLS: [usize; 5] = [1, 2, 3, 4, 8];

/// A value standing for `None`, so that an absent number differs from any
/// present one.
const ABSENT: u64 = 0x7ff8_0000_dead_beef;

fn opt(value: Option<f64>) -> f64 {
    value.unwrap_or_else(|| f64::from_bits(ABSENT))
}

fn opts(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().copied().map(opt).collect()
}

/// Rows of a competing-risks sample: exit time on a grid of tenths so that
/// times tie, cause 0 to 2, weights, groups and entry times.
struct Sample {
    time: Vec<f64>,
    event: Vec<i32>,
    weight: Vec<f64>,
    group: Vec<u32>,
    entry: Vec<f64>,
    x: Vec<f64>,
    cluster: Vec<u32>,
}

fn sample(n: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        time: Vec::new(),
        event: Vec::new(),
        weight: Vec::new(),
        group: Vec::new(),
        entry: Vec::new(),
        x: Vec::new(),
        cluster: Vec::new(),
    };
    for _ in 0..n {
        let entry = (rng.uniform() * 30.0).floor() / 10.0;
        let time = entry + 0.1 + (rng.exp1() * 40.0).floor() / 10.0;
        let event = match rng.below(10) {
            0..=3 => 1,
            4..=5 => 2,
            6..=9 => 0,
            _ => unreachable!(),
        };
        s.entry.push(entry);
        s.time.push(time);
        s.event.push(event);
        s.weight.push(0.5 + rng.uniform() * 2.0);
        s.group.push(rng.below(7) as u32);
        // Few distinct values, so levels repeat.
        s.x.push(rng.below(5) as f64 * 0.5);
        s.cluster.push(rng.below(97) as u32);
    }
    s
}

/// Cox fit on one covariate. The sweep is sequential, so this checks that no
/// hidden global state or pool-dependent library path enters the fit. Strata
/// of unequal size, delayed entry, ties on a tenth grid and 97 clusters.
#[test]
fn cox_single_covariate() {
    let s = sample(3001, 11);
    let is_event: Vec<bool> = s.event.iter().map(|&e| e == 1).collect();
    let stratum: Vec<u32> = s.group.iter().map(|&g| g % 3).collect();
    for ties in [Ties::Breslow, Ties::Efron] {
        for clustered in [false, true] {
            assert_same_on_pools(&POOLS, || {
                let fit = fit_cox_single_covariate(&CoxData {
                    x: &s.x,
                    is_event: &is_event,
                    start: &s.entry,
                    stop: &s.time,
                    weight: &s.weight,
                    stratum: &stratum,
                    ties,
                    cluster: clustered.then_some(s.cluster.as_slice()),
                })
                .unwrap();
                assert!(fit.converged && fit.log_hazard_ratio.is_some());
                vec![
                    opt(fit.log_hazard_ratio),
                    opt(fit.standard_error),
                    fit.n_events,
                    fit.n_rows as f64,
                    f64::from(fit.iterations),
                    f64::from(u8::from(fit.converged)),
                    opt(fit.robust_standard_error),
                ]
            });
        }
    }
}

/// Cox fit on a design matrix of three columns, one of them a duplicate so
/// that a column is aliased, with strata, delayed entry and clusters.
#[test]
fn cox_design_matrix() {
    let s = sample(2503, 12);
    let n = s.time.len();
    let is_event: Vec<bool> = s.event.iter().map(|&e| e == 1).collect();
    let stratum: Vec<u32> = s.group.iter().map(|&g| g % 3).collect();
    let mut rng = Stream::seeded(99);
    let noise: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let design = Mat::from_fn(n, 4, |i, j| match j {
        1 => noise[i],
        2 => s.x[i] + noise[i] * 0.5,
        _ => s.x[i],
    });
    for ties in [Ties::Breslow, Ties::Efron] {
        for clustered in [false, true] {
            assert_same_on_pools(&POOLS, || {
                let fit = fit_cox(&CoxDesign {
                    x: design.as_ref(),
                    is_event: &is_event,
                    start: &s.entry,
                    stop: &s.time,
                    weight: &s.weight,
                    stratum: &stratum,
                    ties,
                    cluster: clustered.then_some(s.cluster.as_slice()),
                })
                .unwrap();
                assert!(fit.converged && fit.coefficients[0].is_some());
                assert!(fit.coefficients[3].is_none(), "the duplicate is aliased");
                let mut out = opts(&fit.coefficients);
                out.extend(opts(&fit.standard_errors));
                if let Some(robust) = &fit.robust_standard_errors {
                    out.extend(opts(robust));
                }
                out.extend([
                    opt(fit.log_partial_likelihood),
                    opt(fit.null_log_partial_likelihood),
                    fit.n_events,
                    fit.n_rows as f64,
                    fit.rank as f64,
                    f64::from(fit.iterations),
                    f64::from(u8::from(fit.converged)),
                ]);
                out.extend(fit.monotone_columns.iter().map(|&c| c as f64));
                out
            });
        }
    }
}

/// Aalen-Johansen curves. Exercises the parallel sort of the row order
/// (3001 rows, above rayon's sequential cut-off for sorts) and the parallel map
/// over groups (7 groups of unequal size, with delayed entry and ties).
#[test]
fn aalen_johansen_curves() {
    let s = sample(3001, 13);
    for entry in [None, Some(s.entry.as_slice())] {
        assert_same_on_pools(&POOLS, || {
            let c = curves(&s.time, &s.event, &s.weight, &s.group, entry).unwrap();
            let mut out: Vec<f64> = c.causes.iter().map(|&k| f64::from(k)).collect();
            out.extend(c.group.iter().map(|&g| f64::from(g)));
            for column in [&c.time, &c.n_risk, &c.n_at_risk, &c.n_event, &c.event_free] {
                out.extend(column.iter().copied());
            }
            for column in &c.cif {
                out.extend(column.iter().copied());
            }
            out
        });
    }
}

/// `advance` is a single sequential step, and every estimator is built from
/// it. A long chain of steps shows it carries no state between calls.
#[test]
fn aalen_johansen_advance() {
    let mut rng = Stream::seeded(14);
    let steps: Vec<(f64, f64, [f64; 2])> = (0..5000)
        .map(|_| {
            let at_risk = 1.0 + rng.uniform() * 100.0;
            let d1 = rng.uniform() * 0.4;
            let d2 = rng.uniform() * 0.4;
            (at_risk, d1 + d2, [d1, d2])
        })
        .collect();
    assert_same_on_pools(&POOLS, || {
        let mut survival = 1.0;
        let mut incidence = [0.0; 2];
        for (at_risk, d_any, d_causes) in &steps {
            advance(&mut survival, *at_risk, *d_any, d_causes, &mut incidence);
        }
        vec![survival, incidence[0], incidence[1]]
    });
}

/// Frame of `n` rows in 600 families, and seven cells of unequal size. Cell 5
/// holds only exposed rows and is not estimable. Cells overlap.
fn arm_frame(n: usize) -> (ArmColumns, RowWindows) {
    let s = sample(n, 15);
    let mut rng = Stream::seeded(16);
    let exposed: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
    let family: Vec<u32> = (0..n).map(|_| rng.below(600) as u32).collect();
    let columns = ArmColumns::new(s.time, s.event, s.weight, family, exposed.clone()).unwrap();
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
    (columns, RowWindows::new(rows, offsets, n, "cell").unwrap())
}

/// Bootstrap of two-arm incidence. Per-cell mode parallelises over cells,
/// then over 37 replicates inside each cell, and writes each replicate with
/// its own seed. Shared mode parallelises over the replicates and walks every
/// cell inside one. The cells hold 9 to 1500 rows and 37 does not divide by
/// any pool size.
#[test]
fn aalen_johansen_bootstrap() {
    let (columns, cells) = arm_frame(2400);
    let seeds: Vec<u64> = (0..cells.len() as u64).map(|c| 1000 + 7919 * c).collect();
    let ages = vec![3.0, 0.5, 1.0, 2.2, 4.4, 6.0];
    for draw in [Draw::PerCell, Draw::Shared] {
        assert_same_on_pools(&POOLS, || {
            let (exposed, comparator, estimable) =
                bootstrap_arm_incidence(&columns, &cells, &seeds, ages.clone(), 1, 37, draw)
                    .unwrap();
            assert!(estimable[0] && !estimable[5]);
            assert!(exposed.iter().any(|&v| v > 0.0));
            let mut out = exposed;
            out.extend(comparator);
            out.extend(estimable.iter().map(|&e| f64::from(u8::from(e))));
            out
        });
    }
}

/// A multi-state process on states 0, 1, 2 with `people` individuals. Each
/// moves between 0 and 1 a random number of times and may end in 2. Times
/// are on a grid of tenths, so stops tie.
struct Process {
    starts: Vec<f64>,
    stops: Vec<f64>,
    from: Vec<u32>,
    to: Vec<u32>,
    person: Vec<u32>,
    weight: Vec<f64>,
    first: Vec<u32>,
}

fn process(people: usize, seed: u64) -> Process {
    let mut rng = Stream::seeded(seed);
    let mut p = Process {
        starts: Vec::new(),
        stops: Vec::new(),
        from: Vec::new(),
        to: Vec::new(),
        person: Vec::new(),
        weight: Vec::new(),
        first: Vec::new(),
    };
    for id in 0..people {
        let mut state = 0u32;
        let mut clock = 0.0f64;
        p.first.push(p.starts.len() as u32);
        loop {
            let stop = clock + 0.1 + (rng.exp1() * 30.0).floor() / 10.0;
            let next = match (state, rng.below(10)) {
                (_, 0) => 2,
                (0, _) => 1,
                _ => 0,
            };
            let last = next == 2 || stop > 12.0 || rng.below(6) == 0;
            p.starts.push(clock);
            p.stops.push(stop);
            p.from.push(state);
            p.to.push(if last && next != 2 { 9 } else { next });
            p.person.push(id as u32);
            p.weight.push(0.5 + rng.uniform());
            if last {
                break;
            }
            state = next;
            clock = stop;
        }
    }
    p
}

fn event_times(p: &Process) -> Vec<f64> {
    let mut times = p.stops.clone();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

/// Occupation curve and its integral. With ascending times the accumulation
/// walks the precomputed plan, which is sequential in the intervals. With
/// descending times there is no plan, and the accumulation splits over the
/// times in parallel, one risk row and transition block per time (about 120
/// times, not a multiple of any pool size).
#[test]
fn state_occupation_and_integral() {
    let p = process(500, 21);
    let ascending = event_times(&p);
    let mut descending = ascending.clone();
    descending.reverse();
    let initial = [1.0, 0.0, 0.0];
    for times in [&ascending, &descending] {
        assert_same_on_pools(&POOLS, || {
            let intervals = Intervals::new(&p.starts, &p.stops, &p.from, &p.to).unwrap();
            let occupation =
                state_occupation_curve(intervals, &p.weight, times, &initial, 3, 1e-9).unwrap();
            assert!(occupation.at_risk.iter().any(|&v| v > 0.0));
            let mut out = occupation.probabilities.clone();
            out.extend(&occupation.at_risk);
            for (horizon, origin) in [(5.0, 0.0), (20.0, 1.3), (0.5, 0.0)] {
                out.extend(integrate(
                    &occupation.probabilities,
                    times,
                    &initial,
                    3,
                    horizon,
                    origin,
                ));
            }
            out
        });
    }
}

/// Clustered bootstrap of the occupation probabilities and of the restricted
/// times. Parallel over 41 replicates, each drawing its weights from its own
/// seed; with descending times each replicate also splits its accumulation
/// over the times inside the outer loop.
#[test]
fn state_occupation_bootstrap() {
    let p = process(400, 22);
    let ascending = event_times(&p);
    let mut descending = ascending.clone();
    descending.reverse();
    let seeds: Vec<u64> = (0..41).map(|r| 500 + r * 7919).collect();
    let first_states: Vec<i32> = p.first.iter().map(|&i| p.from[i as usize] as i32).collect();
    for times in [&ascending, &descending] {
        for horizons in [vec![], vec![2.0, 7.5, 30.0]] {
            assert_same_on_pools(&POOLS, || {
                let intervals = Intervals::new(&p.starts, &p.stops, &p.from, &p.to).unwrap();
                bootstrap_state_occupation(
                    intervals,
                    &p.weight,
                    &p.person,
                    &p.first,
                    &first_states,
                    times,
                    &horizons,
                    0.5,
                    &seeds,
                    400,
                    3,
                    1e-9,
                )
                .unwrap()
            });
        }
    }
}

fn rows_bits(rows: &Rows) -> Vec<f64> {
    let mut out: Vec<f64> = rows.row.iter().map(|&r| f64::from(r)).collect();
    out.extend(&rows.start);
    out.extend(&rows.stop);
    out.extend(rows.event.iter().map(|&e| f64::from(e)));
    out.extend(&rows.weight);
    out
}

/// Fine-Gray row construction. Both builders are sequential loops over the
/// individuals and the censoring times; the check covers the Kaplan-Meier
/// estimate of the censoring distribution that they share, at 2000 rows with
/// tied times.
#[test]
fn fine_gray_rows() {
    let s = sample(2000, 31);
    let levels = float_codes(&s.x);
    assert_same_on_pools(&POOLS, || {
        let data = FineGrayData::new(&s.time, &s.event, &s.weight, 1).unwrap();
        let mut out = rows_bits(&data.expand());
        out.extend(rows_bits(&data.risk_rows(&levels).unwrap()));
        out
    });
}

/// Fine-Gray fit on the collapsed rows, and on the expanded rows with
/// clusters.
#[test]
fn fine_gray_fit_all_options() {
    let s = sample(1500, 32);
    for ties in [Ties::Breslow, Ties::Efron] {
        for clustered in [false, true] {
            assert_same_on_pools(&POOLS, || {
                let fit = fine_gray_fit(
                    &s.time,
                    &s.event,
                    &s.weight,
                    &s.x,
                    1,
                    ties,
                    clustered.then_some(s.cluster.as_slice()),
                )
                .unwrap();
                assert!(fit.converged && fit.log_hazard_ratio.is_some());
                vec![
                    opt(fit.log_hazard_ratio),
                    opt(fit.standard_error),
                    fit.n_events,
                    fit.n_rows as f64,
                    f64::from(fit.iterations),
                    f64::from(u8::from(fit.converged)),
                    opt(fit.robust_standard_error),
                ]
            });
        }
    }
}

/// Titman-Putter test. Parallel over 7 landmarks, then over 14
/// (landmark, state) slots that each write a row of the centred scores, then
/// over 33 wild-bootstrap replicates reduced by an integer sum.
#[test]
fn markov_test_all_paths() {
    let p = process(350, 41);
    let state: Vec<u32> = p.from.clone();
    let event: Vec<f64> =
        p.to.iter()
            .zip(&p.from)
            .map(|(&t, &f)| f64::from(u8::from(f == 0 && t == 1)))
            .collect();
    let grid = [0.3, 0.9, 1.6, 2.4, 3.5, 4.8, 6.1];
    let seeds: Vec<u64> = (0..33).map(|r| 77 + r * 7919).collect();
    assert_same_on_pools(&POOLS, || {
        let stays = Stays::new(&p.person, &p.starts, &p.stops, &state, &event).unwrap();
        let t = markov_test(&stays, &grid, 2, &seeds).unwrap();
        assert!(t.p_value.iter().all(Option::is_some));
        assert!(t.variance.iter().any(|&v| v > 0.0));
        let mut out = t.trace.clone();
        for column in [&t.variance, &t.n_occupied, &t.n_events_after, &t.statistic] {
            out.extend(column.iter().copied());
        }
        out.extend(opts(&t.p_value));
        out
    });
}
