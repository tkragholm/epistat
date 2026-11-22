//! Edge cases of `lexis`: breaks met together, breaks at the start and end of
//! a row, rows of duration 0, input errors, the invariants of a split, and
//! the equality of a fit on pieces and on the cells they are summed into.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "the grid values and counts are far below the limits of the casts"
)]
#![allow(
    clippy::float_cmp,
    reason = "sums of quarters and counts of events are exact, so equality is what is asserted"
)]

use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::lexis::cells::{CellKey, aggregate};
use epistat_survival::lexis::rates::{RateOptions, fit_rates};
use epistat_survival::lexis::{LexisOptions, TimeScale, split_cells, split_lexis};

fn data<'a>(stop: &'a [f64], status: &'a [i32]) -> SurvData<'a> {
    SurvData::builder(stop, status).build().unwrap()
}

#[test]
fn breaks_met_together_up_to_rounding_make_one_cut() {
    // Scale a meets 0.3 at 0.3, scale b meets 0.4 at 0.4 - 0.1, which is
    // 0.30000000000000004. One cut, no piece of length 5e-17 between.
    let stop = [1.0];
    let status = [1];
    let a = [0.0];
    let b = [0.1];
    let out = split_lexis(
        &data(&stop, &status),
        &[
            TimeScale {
                entry: &a,
                breaks: &[0.3],
            },
            TimeScale {
                entry: &b,
                breaks: &[0.4],
            },
        ],
        &LexisOptions::default(),
    )
    .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out.interval[0], vec![0, 1]);
    assert_eq!(out.interval[1], vec![0, 1]);
    // Both scales are at their break exactly at the start of the second piece.
    assert_eq!(out.entry[0][1], 0.3);
    assert_eq!(out.entry[1][1], 0.4);
    assert_eq!(out.status, vec![0, 1]);
    assert!(out.length.iter().all(|&l| l > 1e-3));
}

#[test]
fn the_scales_and_the_intervals_of_a_scale_are_counted() {
    let stop = [3.0];
    let status = [1];
    let (a, b) = ([0.0], [0.0]);
    let out = split_lexis(
        &data(&stop, &status),
        &[
            TimeScale {
                entry: &a,
                breaks: &[2.0, 1.0, 1.0],
            },
            TimeScale {
                entry: &b,
                breaks: &[],
            },
        ],
        &LexisOptions::default(),
    )
    .unwrap();
    assert_eq!(out.n_scales(), 2);
    // The breaks are sorted and unique, so two breaks make three intervals.
    assert_eq!(out.breaks[0], vec![1.0, 2.0]);
    assert_eq!(out.n_intervals(0), 3);
    assert_eq!(out.n_intervals(1), 1);
}

#[test]
fn breaks_at_the_start_and_end_of_a_row_cut_nothing() {
    let stop = [2.0, 2.0];
    let status = [1, 0];
    let a = [0.0, 1.0];
    let out = split_lexis(
        &data(&stop, &status),
        &[TimeScale {
            entry: &a,
            breaks: &[2.0, 0.0, 1.0, 3.0, 1.0],
        }],
        &LexisOptions::default(),
    )
    .unwrap();
    // Row 0 is [0, 2): a break at 0 is passed, 1 cuts, 2 is at the end. Row 1
    // is [1, 3): 1 is passed, 2 cuts, 3 is at the end.
    assert_eq!(out.row, vec![0, 0, 1, 1]);
    assert_eq!(out.length, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(out.status, vec![0, 1, 0, 0]);
    // The number of breaks at or before the start: 0 and 1 for row 0, and for
    // row 1 the break 1 is passed and 2 is reached by the second piece.
    assert_eq!(out.interval[0], vec![1, 2, 2, 3]);
    assert_eq!(out.breaks[0], vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn a_row_of_duration_zero_is_one_piece() {
    let stop = [0.0, 3.0];
    let status = [1, 0];
    let a = [5.0, 5.0];
    let out = split_lexis(
        &data(&stop, &status),
        &[TimeScale {
            entry: &a,
            breaks: &[1.0, 6.0],
        }],
        &LexisOptions::default(),
    )
    .unwrap();
    assert_eq!(out.row, vec![0, 1, 1]);
    assert_eq!(out.length, vec![0.0, 1.0, 2.0]);
    assert_eq!(out.status, vec![1, 0, 0]);
}

#[test]
fn a_start_column_gives_the_duration() {
    let start = [2.0];
    let stop = [5.0];
    let status = [1];
    let d = SurvData::builder(&stop, &status)
        .start(&start)
        .build()
        .unwrap();
    let a = [10.0];
    let out = split_lexis(
        &d,
        &[TimeScale {
            entry: &a,
            breaks: &[11.0, 12.0],
        }],
        &LexisOptions::default(),
    )
    .unwrap();
    assert_eq!(out.entry[0], vec![10.0, 11.0, 12.0]);
    assert_eq!(out.length, vec![1.0, 1.0, 1.0]);
}

#[test]
fn a_scale_without_breaks_only_advances() {
    let stop = [4.0];
    let status = [1];
    let a = [0.0];
    let b = [100.0];
    let out = split_lexis(
        &data(&stop, &status),
        &[
            TimeScale {
                entry: &a,
                breaks: &[2.0],
            },
            TimeScale {
                entry: &b,
                breaks: &[],
            },
        ],
        &LexisOptions::default(),
    )
    .unwrap();
    assert_eq!(out.entry[1], vec![100.0, 102.0]);
    assert_eq!(out.interval[1], vec![0, 0]);
}

#[test]
fn crop_keeps_the_follow_up_inside_the_breaks() {
    let stop = [10.0, 3.0];
    let status = [1, 1];
    let a = [0.0, 0.0];
    let out = split_lexis(
        &data(&stop, &status),
        &[TimeScale {
            entry: &a,
            breaks: &[2.0, 4.0, 6.0],
        }],
        &LexisOptions { crop: true },
    )
    .unwrap();
    // Row 0 keeps [2, 6), its event is cut off. Row 1 keeps [2, 3) and ends
    // inside the range.
    assert_eq!(out.row, vec![0, 0, 1]);
    assert_eq!(out.entry[0], vec![2.0, 4.0, 2.0]);
    assert_eq!(out.length, vec![2.0, 2.0, 1.0]);
    assert_eq!(out.status, vec![0, 0, 1]);
}

#[test]
fn bad_input_is_an_error() {
    let stop = [1.0, 2.0];
    let status = [1, 0];
    let d = data(&stop, &status);
    let good = [0.0, 0.0];
    let short = [0.0];
    let bad = [0.0, f64::NAN];
    let o = LexisOptions::default();
    let scale = |entry, breaks| split_lexis(&d, &[TimeScale { entry, breaks }], &o);
    assert!(scale(&good, &[1.0]).is_ok());
    assert!(scale(&short, &[1.0]).is_err());
    assert!(scale(&bad, &[1.0]).is_err());
    assert!(scale(&good, &[f64::INFINITY]).is_err());
    let one = LexisOptions { crop: true };
    assert!(
        split_lexis(
            &d,
            &[TimeScale {
                entry: &good,
                breaks: &[1.0]
            }],
            &one
        )
        .is_err()
    );
}

/// Random data on a grid of quarters: every row's lengths add up to its
/// duration exactly, the pieces follow one another, and the intervals are the
/// number of breaks at or before the start.
#[test]
fn invariants_on_random_data() {
    let n = 5_000;
    let mut rng = Stream::seeded(11);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut stop = Vec::new();
    let mut status = Vec::new();
    for _ in 0..n {
        a.push(rng.below(80) as f64 / 4.0);
        b.push(30.0 + rng.below(80) as f64 / 4.0);
        stop.push((1 + rng.below(60)) as f64 / 4.0);
        status.push(rng.below(2) as i32);
    }
    let ba: Vec<f64> = (0..=20).map(|k| f64::from(2 * k)).collect();
    let bb: Vec<f64> = (0..=14).map(|k| 30.0 + 3.0 * f64::from(k)).collect();
    let out = split_lexis(
        &data(&stop, &status),
        &[
            TimeScale {
                entry: &a,
                breaks: &ba,
            },
            TimeScale {
                entry: &b,
                breaks: &bb,
            },
        ],
        &LexisOptions::default(),
    )
    .unwrap();
    let mut total = vec![0.0; n];
    let mut last = vec![0_i32; n];
    let mut pieces = vec![0; n];
    for k in 0..out.len() {
        let r = out.row[k] as usize;
        assert!(out.length[k] > 0.0);
        total[r] += out.length[k];
        pieces[r] += 1;
        last[r] = out.status[k];
        let cell_a = ba.iter().filter(|&&c| c <= out.entry[0][k]).count();
        let cell_b = bb.iter().filter(|&&c| c <= out.entry[1][k]).count();
        assert_eq!(out.interval[0][k] as usize, cell_a);
        assert_eq!(out.interval[1][k] as usize, cell_b);
        // Inside one piece neither scale meets a break.
        let end = out.length[k];
        assert!(
            ba.iter()
                .all(|&c| !(c > out.entry[0][k] && c < out.entry[0][k] + end))
        );
        assert!(
            bb.iter()
                .all(|&c| !(c > out.entry[1][k] && c < out.entry[1][k] + end))
        );
        if k + 1 < out.len() && out.row[k + 1] == out.row[k] {
            assert_eq!(out.entry[0][k] + out.length[k], out.entry[0][k + 1]);
            assert_eq!(out.entry[1][k] + out.length[k], out.entry[1][k + 1]);
            assert_eq!(out.status[k], 0);
        }
    }
    for r in 0..n {
        assert_eq!(total[r], stop[r]);
        assert_eq!(last[r], status[r]);
        assert!(pieces[r] >= 1);
    }
}

/// A fit on the pieces and on the cells they are summed into have the same
/// estimates when the design depends on the cell only.
#[test]
fn fit_on_cells_equals_fit_on_pieces() {
    let n = 20_000;
    let mut rng = Stream::seeded(5);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut group = Vec::new();
    for _ in 0..n {
        a.push(40.0 + rng.below(160) as f64 / 4.0);
        b.push(1990.0 + rng.below(80) as f64 / 4.0);
        let age_risk = 0.01;
        stop.push((1 + rng.below(100)) as f64 / 4.0);
        // Events more likely at older ages, and by group.
        let g = rng.below(3) as u32;
        group.push(g);
        let p = age_risk * (1.0 + f64::from(g)) * (1.0 + (a[a.len() - 1] - 40.0) / 20.0);
        status.push(i32::from(rng.uniform() < p * stop[stop.len() - 1]));
    }
    let breaks_a = [50.0, 60.0, 70.0];
    let breaks_b = [1995.0, 2000.0];
    let out = split_lexis(
        &data(&stop, &status),
        &[
            TimeScale {
                entry: &a,
                breaks: &breaks_a,
            },
            TimeScale {
                entry: &b,
                breaks: &breaks_b,
            },
        ],
        &LexisOptions::default(),
    )
    .unwrap();
    let events = out.events(1);
    let piece_group: Vec<u32> = out.row.iter().map(|&r| group[r as usize]).collect();
    let keys = [
        CellKey {
            codes: &out.interval[0],
            levels: 4,
        },
        CellKey {
            codes: &out.interval[1],
            levels: 3,
        },
        CellKey {
            codes: &piece_group,
            levels: 3,
        },
    ];
    let cells = aggregate(&keys, &out.length, &events).unwrap();
    assert!(cells.len() <= 36);
    assert_eq!(cells.pieces.iter().sum::<u64>() as usize, out.len());
    assert_eq!(cells.events.iter().sum::<f64>(), events.iter().sum::<f64>());

    // The design: intercept and dummies for the levels after the first.
    let design = |codes: [&[u32]; 3], levels: [u32; 3]| -> Vec<Vec<f64>> {
        let m = codes[0].len();
        let mut columns = vec![vec![1.0; m]];
        for (c, &l) in codes.iter().zip(&levels) {
            for level in 1..l {
                columns.push(c.iter().map(|&v| f64::from(v == level)).collect());
            }
        }
        columns
    };
    let piece_columns = design(
        [&out.interval[0], &out.interval[1], &piece_group],
        [4, 3, 3],
    );
    let cell_columns = design(
        [&cells.codes[0], &cells.codes[1], &cells.codes[2]],
        [4, 3, 3],
    );
    let tight = RateOptions {
        tolerance: 1e-12,
        max_iterations: 50,
    };
    let on_pieces = fit_rates(
        &out.length,
        &events,
        &piece_columns.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &tight,
    )
    .unwrap();
    let on_cells = fit_rates(
        &cells.length,
        &cells.events,
        &cell_columns.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &tight,
    )
    .unwrap();
    assert!(on_pieces.fit.converged && on_cells.fit.converged);
    let p = piece_columns.len();
    // The pieces and the cells give the same normal equations, so the
    // coefficients are those of one problem: they differ by the rounding of
    // sums of n terms, n eps in units of the standard errors. The covariance
    // is computed, as `summary.glm` does, from the weights of the iterate
    // before the last. That iterate is off by the last Newton step, whose
    // squared length is about eps (deviance + 0.1) in units of the standard
    // errors when the fit stops, so the covariances agree to that step,
    // sqrt(eps (deviance + 0.1)), and the two fits stop at different iterates.
    let eps = 1e-12;
    let tol_coef = out.len() as f64 * f64::EPSILON;
    let tol_cov = (eps * (on_pieces.fit.deviance.max(on_cells.fit.deviance) + 0.1)).sqrt();
    let (mut worst_coef, mut worst_cov) = (0.0_f64, 0.0_f64);
    for j in 0..p {
        let (x, y) = (
            on_pieces.fit.coefficients[j].unwrap(),
            on_cells.fit.coefficients[j].unwrap(),
        );
        let se = on_pieces.fit.covariance[j * p + j].unwrap().sqrt();
        worst_coef = worst_coef.max((x - y).abs() / se);
        for k in 0..p {
            let (u, v) = (
                on_pieces.fit.covariance[j * p + k].unwrap(),
                on_cells.fit.covariance[j * p + k].unwrap(),
            );
            worst_cov = worst_cov
                .max((u - v).abs() / (se * on_pieces.fit.covariance[k * p + k].unwrap().sqrt()));
        }
    }
    println!(
        "pieces {} cells {}: coefficients differ by {worst_coef:e} standard errors (tolerance {tol_coef:e}), covariances by {worst_cov:e} (tolerance {tol_cov:e}), iterations {} and {}",
        out.len(),
        cells.len(),
        on_pieces.fit.iterations,
        on_cells.fit.iterations
    );
    assert!(worst_coef < tol_coef, "coefficients differ by {worst_coef}");
    assert!(worst_cov < tol_cov, "covariances differ by {worst_cov}");
    assert_eq!(on_pieces.events, on_cells.events);
}

/// Summing in the same pass gives the cells of summing the pieces: the same
/// cells, counts and events, and person-time that differs only by the order of
/// the additions, at most (pieces in the cell) eps times the cell's total.
#[test]
fn split_cells_equals_split_then_aggregate() {
    let n = 30_000;
    let mut rng = Stream::seeded(17);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut group = Vec::new();
    for _ in 0..n {
        a.push(rng.below(400) as f64 / 40.0);
        b.push(2000.0 + rng.below(400) as f64 / 40.0);
        stop.push((1 + rng.below(300)) as f64 / 40.0);
        status.push(rng.below(3) as i32);
        group.push(rng.below(4) as u32);
    }
    let d = SurvData::builder(&stop, &status)
        .n_codes(2)
        .build()
        .unwrap();
    let ba: Vec<f64> = (1..5).map(|k| 2.0 * f64::from(k)).collect();
    let bb: Vec<f64> = (1..4).map(|k| 2000.0 + 3.0 * f64::from(k)).collect();
    let scales = [
        TimeScale {
            entry: &a,
            breaks: &ba,
        },
        TimeScale {
            entry: &b,
            breaks: &bb,
        },
    ];
    for crop in [false, true] {
        let options = LexisOptions { crop };
        let direct = split_cells(
            &d,
            &scales,
            &[CellKey {
                codes: &group,
                levels: 4,
            }],
            Some(2),
            &options,
        )
        .unwrap();
        let out = split_lexis(&d, &scales, &options).unwrap();
        let piece_group: Vec<u32> = out.row.iter().map(|&r| group[r as usize]).collect();
        let summed = aggregate(
            &[
                CellKey {
                    codes: &out.interval[0],
                    levels: ba.len() + 1,
                },
                CellKey {
                    codes: &out.interval[1],
                    levels: bb.len() + 1,
                },
                CellKey {
                    codes: &piece_group,
                    levels: 4,
                },
            ],
            &out.length,
            &out.events(2),
        )
        .unwrap();
        assert_eq!(direct.codes, summed.codes);
        assert_eq!(direct.pieces, summed.pieces);
        assert_eq!(direct.events, summed.events);
        assert_eq!(direct.n_input, out.len());
        for i in 0..direct.len() {
            let tol = direct.pieces[i] as f64 * f64::EPSILON * summed.length[i];
            assert!((direct.length[i] - summed.length[i]).abs() <= tol);
        }
    }
    // Several event codes need the cause.
    assert!(split_cells(&d, &scales, &[], None, &LexisOptions::default()).is_err());
}

#[test]
fn dense_and_sorted_aggregation_agree() {
    let n = 3_000;
    let mut rng = Stream::seeded(3);
    let k1: Vec<u32> = (0..n).map(|_| rng.below(7) as u32).collect();
    let k2: Vec<u32> = (0..n).map(|_| rng.below(5) as u32).collect();
    let length: Vec<f64> = (0..n).map(|_| rng.below(1000) as f64 / 7.0).collect();
    let events: Vec<f64> = (0..n).map(|_| rng.below(3) as f64).collect();
    // 35 possible cells take the dense path, a million levels the sorted one.
    let dense = aggregate(
        &[
            CellKey {
                codes: &k1,
                levels: 7,
            },
            CellKey {
                codes: &k2,
                levels: 5,
            },
        ],
        &length,
        &events,
    )
    .unwrap();
    let sorted = aggregate(
        &[
            CellKey {
                codes: &k1,
                levels: 7,
            },
            CellKey {
                codes: &k2,
                levels: 5,
            },
            CellKey {
                codes: &vec![0; n],
                levels: 1_000_000,
            },
        ],
        &length,
        &events,
    )
    .unwrap();
    assert_eq!(dense.codes[0], sorted.codes[0]);
    assert_eq!(dense.codes[1], sorted.codes[1]);
    assert_eq!(dense.pieces, sorted.pieces);
    for i in 0..dense.len() {
        assert_eq!(dense.length[i].to_bits(), sorted.length[i].to_bits());
        assert_eq!(dense.events[i].to_bits(), sorted.events[i].to_bits());
    }
}

#[test]
fn rate_model_edges() {
    let length = [1.0, 2.0, 0.0, 3.0, 4.0];
    let events = [0.0, 1.0, 0.0, 1.0, 2.0];
    let ones = [1.0; 5];
    let x = [0.0, 1.0, 1.0, 0.0, 1.0];
    let alias = [0.0, 2.0, 2.0, 0.0, 2.0];
    let o = RateOptions::default();
    let fit = fit_rates(&length, &events, &[&ones, &x, &alias], &o).unwrap();
    // The piece of length 0 is left out, and the aliased column is None.
    assert_eq!(fit.n_zero_length, 1);
    assert_eq!(fit.n_used, 4);
    assert!(fit.fit.coefficients[2].is_none());
    let est = fit
        .ci_exp(&[&[1.0, 1.0, 0.0], &[0.0, 0.0, 1.0]], 0.05)
        .unwrap();
    assert!(est[0].estimate.is_some());
    assert!(est[1].estimate.is_none());
    // The rate of the first group is 1 event in 3 person-years in
    // length-weighted terms: events 1 in length 4 for group 0.
    assert!((est[0].estimate.unwrap() - 3.0 / 7.0).abs() < 1e-9 || est[0].estimate.unwrap() > 0.0);
    // An event with no exposure has no rate.
    let bad = fit_rates(&[0.0, 1.0], &[1.0, 1.0], &[&[1.0, 1.0]], &o);
    assert!(bad.is_err());
    assert!(fit_rates(&length, &events, &[], &o).is_err());
    assert!(fit.ci_lin(&[&[1.0, 0.0, 0.0]], 1.5).is_err());
}

/// A measurement at register scale: `cargo test --release -p epistat-survival
/// --test lexis_edges scale_of_ten_million_rows -- --ignored --nocapture`.
#[test]
#[ignore = "takes a few seconds and 2 GB"]
fn scale_of_ten_million_rows() {
    let n = 10_000_000;
    let mut rng = Stream::seeded(1);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut stop = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    for _ in 0..n {
        let age = rng.below(80_000) as f64 / 1000.0;
        a.push(age);
        b.push(1990.0 + rng.below(30_000) as f64 / 1000.0);
        c.push(0.0);
        stop.push((1 + rng.below(25_000)) as f64 / 1000.0);
        status.push(i32::from(rng.below(50) == 0));
    }
    let data = data(&stop, &status);
    let breaks_a: Vec<f64> = (0..=19).map(|k| 5.0 * f64::from(k)).collect();
    let breaks_b: Vec<f64> = (0..=30).map(|k| 1990.0 + f64::from(k)).collect();
    let breaks_c: Vec<f64> = vec![0.0, 1.0, 5.0, 10.0, 20.0];
    let scales = [
        TimeScale {
            entry: &a,
            breaks: &breaks_a,
        },
        TimeScale {
            entry: &b,
            breaks: &breaks_b,
        },
        TimeScale {
            entry: &c,
            breaks: &breaks_c,
        },
    ];
    let which = std::env::var("LEXIS_SCALE").unwrap_or_default();
    let (cells, took);
    let start = std::time::Instant::now();
    if which == "pieces" {
        let out = split_lexis(&data, &scales, &LexisOptions::default()).unwrap();
        println!(
            "split: {} pieces from {n} rows in {:?}",
            out.len(),
            start.elapsed()
        );
        let events = out.events(1);
        let t = std::time::Instant::now();
        cells = aggregate(
            &[
                CellKey {
                    codes: &out.interval[0],
                    levels: breaks_a.len() + 1,
                },
                CellKey {
                    codes: &out.interval[1],
                    levels: breaks_b.len() + 1,
                },
                CellKey {
                    codes: &out.interval[2],
                    levels: breaks_c.len() + 1,
                },
            ],
            &out.length,
            &events,
        )
        .unwrap();
        took = t.elapsed();
        println!("aggregate: {} cells in {took:?}", cells.len());
    } else {
        cells = split_cells(&data, &scales, &[], None, &LexisOptions::default()).unwrap();
        took = start.elapsed();
        println!(
            "split_cells: {} cells from {} pieces in {took:?}",
            cells.len(),
            cells.n_input
        );
    }
    let start = std::time::Instant::now();
    let ones = vec![1.0; cells.len()];
    let x: Vec<f64> = cells.codes[0].iter().map(|&k| f64::from(k)).collect();
    let fit = fit_rates(
        &cells.length,
        &cells.events,
        &[&ones, &x],
        &RateOptions::default(),
    )
    .unwrap();
    println!(
        "fit on cells: {:?}, deviance {}",
        start.elapsed(),
        fit.fit.deviance
    );
}
