//! Properties of `dcdh::estimate` that hold without a reference: exact
//! effects on outcomes without noise, also with group-specific linear trends
//! under `trends_lin`, the same switchers at every horizon under
//! `same_switchers`, `by` with one level, independence of the thread count,
//! clusters of one group, and the errors.
//!
//! On noiseless outcomes every estimate is a ratio of sums of exact
//! effects, so it matches the effect up to the rounding of sums over at most
//! `n` cells: `TOL = 16 n eps` relative to the effect.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test data: small whole numbers"
)]

#[path = "common/switching.rs"]
mod switching;

use epistat_core::{Classify, InputError};
use epistat_did::{Dcdh, DcdhOptions, DcdhSwitchers, Error, SwitchingPanel, dcdh};
use switching::{Columns, Row};

const EPS: f64 = f64::EPSILON;

/// A small deterministic generator, so the tests need no random crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn row(group: i64, time: i64, d: f64, y: f64) -> Row {
    Row::new(group, time, d, y)
}

fn estimate(rows: &[Row], options: &DcdhOptions) -> Result<Dcdh, Error> {
    dcdh::estimate(&Columns::new(rows).panel(), options)
}

/// The value of an estimate.
fn value(e: &epistat_did::DcdhTerm) -> Option<f64> {
    e.estimate.map(|e| e.estimate)
}

/// Absorbing treatment first taken in period 3, 5 or 7 (or never) by 60
/// groups over 8 periods; the outcome is a group effect, a period effect and
/// `tau(e) = 1 + 0.5 e` from the `e`-th treated period on, with `e` from 0.
fn absorbing() -> Vec<Row> {
    let mut rng = Lcg(7);
    let mut rows = Vec::new();
    for g in 0..60 {
        let first = [3, 5, 7, 99][g % 4];
        let alpha = rng.next();
        for t in 1..=8 {
            let lambda = 0.3 * t as f64 + (t * t) as f64 * 0.01;
            let (d, effect) = if t >= first {
                (1.0, 1.0 + 0.5 * (t - first) as f64)
            } else {
                (0.0, 0.0)
            };
            rows.push(row(g as i64, t, d, alpha + lambda + effect));
        }
    }
    rows
}

#[test]
fn noiseless_absorbing_effects_are_exact() {
    let rows = absorbing();
    let tol = 16.0 * rows.len() as f64 * EPS;
    let r = estimate(
        &rows,
        &DcdhOptions {
            effects: 4,
            placebo: 2,
            ..DcdhOptions::default()
        },
    )
    .expect("estimate");
    assert_eq!(r.effects.len(), 4);
    for (l, e) in r.effects.iter().enumerate() {
        let want = 1.0 + 0.5 * l as f64;
        let got = value(e).expect("effect");
        assert!((got - want).abs() <= tol * want, "effect {}: {got}", l + 1);
    }
    for p in &r.placebos {
        assert!(value(p).expect("placebo").abs() <= tol, "placebo");
    }
}

/// With a linear trend of its own for every group, the effects of
/// `trends_lin` are still exact: effect `l` sums `l` exact differences, so
/// its tolerance is `l TOL`, relative to the largest outcome change.
#[test]
fn trends_lin_removes_group_trends() {
    let mut rows = absorbing();
    for r in &mut rows {
        let slope = 0.4 * (r.group % 4) as f64 + 0.1 * ((r.group * 7) % 3) as f64;
        *r.outcome.as_mut().expect("outcome") += slope * r.time as f64;
    }
    let tol = 16.0 * rows.len() as f64 * EPS * 8.0;
    let r = estimate(
        &rows,
        &DcdhOptions {
            effects: 4,
            placebo: 2,
            trends_lin: true,
            ..DcdhOptions::default()
        },
    )
    .expect("estimate");
    assert_eq!(r.effects.len(), 4);
    for (l, e) in r.effects.iter().enumerate() {
        let want = 1.0 + 0.5 * l as f64;
        let got = value(e).expect("effect");
        assert!(
            (got - want).abs() <= (l + 1) as f64 * tol * want,
            "effect {}: {got}",
            l + 1
        );
    }
    for p in &r.placebos {
        assert!(value(p).expect("placebo").abs() <= 2.0 * tol, "placebo");
    }
    assert!(r.average_total_effect.estimate.is_none());
    // Without the option the trends, which differ by cohort, bias the
    // effects.
    let plain = estimate(
        &rows,
        &DcdhOptions {
            effects: 4,
            ..DcdhOptions::default()
        },
    )
    .expect("estimate");
    assert!((value(&plain.effects[3]).expect("effect") - 2.5).abs() > 0.01);
}

#[test]
fn same_switchers_are_the_same_at_every_horizon() {
    let rows = moving(2, 1.0, 11);
    let r = estimate(
        &rows,
        &DcdhOptions {
            effects: 3,
            placebo: 2,
            same_switchers: true,
            same_switchers_pl: true,
            ..DcdhOptions::default()
        },
    )
    .expect("estimate");
    // The placebos keep those of the effects' switchers that also allow
    // every placebo.
    let (n, n_pl) = (r.effects[0].n_switchers, r.placebos[0].n_switchers);
    assert!(n > 0.0 && n_pl > 0.0 && n_pl <= n);
    for e in &r.effects {
        assert_eq!(e.n_switchers, n, "switchers at {}", e.relative_period);
    }
    for e in &r.placebos {
        assert_eq!(e.n_switchers, n_pl, "switchers at {}", e.relative_period);
    }
}

#[test]
fn by_with_one_level_is_the_whole_panel() {
    let rows = moving(3, 0.7, 5);
    let config = DcdhOptions {
        effects: 2,
        placebo: 1,
        ..DcdhOptions::default()
    };
    let whole = estimate(&rows, &config).expect("estimate");
    let columns = Columns::new(&rows).with_subgroups(&vec![4; rows.len()]);
    let by = dcdh::by(&columns.panel(), &config).expect("by");
    assert_eq!(by.len(), 1);
    assert_eq!(
        columns.subgroups_of.as_ref().unwrap().label(by[0].level),
        Some(&4)
    );
    assert_eq!(by[0].result.as_ref().expect("estimate"), &whole);
}

/// A treatment that moves among `levels` after a random first change, and
/// an outcome with the contemporaneous effect `beta D`.
fn moving(levels: usize, beta: f64, seed: u64) -> Vec<Row> {
    let mut rng = Lcg(seed);
    let mut rows = Vec::new();
    let top = (levels - 1) as f64;
    for g in 0..90 {
        let mut d = (rng.next() * levels as f64).floor();
        let first = 2 + (rng.next() * 9.0) as i64;
        let alpha = rng.next();
        for t in 1..=8 {
            if t == first {
                d = if d == 0.0 { 1.0 } else { d - 1.0 };
            } else if t > first && rng.next() < 0.3 {
                d = (d + if rng.next() < 0.5 { -1.0 } else { 1.0 }).clamp(0.0, top);
            }
            let y = alpha + 0.2 * t as f64 + beta * d;
            rows.push(row(g, t, d, y));
        }
    }
    rows
}

#[test]
fn average_total_effect_is_the_contemporaneous_effect() {
    // With Y = a_g + b_t + beta D, each DID_l is beta times the switchers'
    // mean change of treatment, so the average total effect, the ratio of
    // their weighted sums, is beta.
    let mut ran = 0;
    for levels in [2, 3] {
        let rows = moving(levels, 1.7, 11 + levels as u64);
        let tol = 16.0 * rows.len() as f64 * EPS;
        for switchers in [DcdhSwitchers::Both, DcdhSwitchers::In, DcdhSwitchers::Out] {
            let r = estimate(
                &rows,
                &DcdhOptions {
                    effects: 3,
                    switchers,
                    dont_drop_larger_lower: true,
                    ..DcdhOptions::default()
                },
            );
            let Ok(r) = r else {
                assert!(matches!(r, Err(Error::NoEffect)));
                continue;
            };
            ran += 1;
            let got = value(&r.average_total_effect).expect("average");
            assert!(
                (got - 1.7).abs() <= tol * 1.7,
                "{levels} levels, {switchers:?}: {got}"
            );
        }
    }
    assert_eq!(ran, 6, "every design estimates");
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let rows = moving(3, 0.8, 5);
    let config = DcdhOptions {
        effects: 3,
        placebo: 2,
        ..DcdhOptions::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| estimate(&rows, &config).expect("estimate"))
    };
    let one = run(1);
    for threads in [2, 4, 12] {
        assert_eq!(run(threads), one, "{threads} threads");
    }
}

#[test]
fn clusters_of_one_group_give_the_unclustered_inference() {
    let rows = moving(2, 0.8, 9);
    let config = DcdhOptions {
        effects: 2,
        placebo: 1,
        ..DcdhOptions::default()
    };
    let plain = estimate(&rows, &config).expect("estimate");
    let clustered_rows: Vec<Row> = rows
        .iter()
        .map(|r| Row {
            cluster: Some(r.group),
            ..r.clone()
        })
        .collect();
    let clustered = estimate(&clustered_rows, &config).expect("estimate");
    for (a, b) in plain.effects.iter().zip(&clustered.effects) {
        assert_eq!(a.estimate, b.estimate);
    }
    assert_eq!(plain.covariance, clustered.covariance);
    assert_eq!(plain.influence, clustered.influence);
}

/// The reported variance, `sum_g psi_g^2 / G^2`, does not centre the
/// values; the influence function's variance does, so without clusters the
/// two differ by the square of the values' sum: `V = V_c + (sum psi)^2 /
/// G^3`. Each side is a sum of `G` terms, so the identity holds to
/// `TOL = 16 n eps` of the variance.
#[test]
fn the_influence_function_gives_the_standard_errors() {
    let mut rows = moving(3, 0.8, 5);
    let mut rng = Lcg(3);
    for r in &mut rows {
        *r.outcome.as_mut().expect("outcome") += rng.next() - 0.5;
    }
    let tol = 16.0 * rows.len() as f64 * EPS;
    let r = estimate(
        &rows,
        &DcdhOptions {
            effects: 3,
            placebo: 2,
            ..DcdhOptions::default()
        },
    )
    .expect("estimate");
    let influence = r.influence.as_ref().expect("influence");
    assert_eq!(influence.units(), Some(&r.units[..]));
    let centred = influence.variance(None).expect("variance");
    let g = r.n_units as f64;
    let terms: Vec<_> = r.effects.iter().chain(&r.placebos).collect();
    for (j, t) in terms.iter().enumerate() {
        let se = t.estimate.and_then(|e| e.se).expect("se");
        let psi = influence.column(j);
        assert_eq!(r.influence_of(j), Some(psi));
        let sum: f64 = psi.iter().sum();
        let got = centred.get(j, j).expect("variance") + sum * sum / (g * g * g);
        let want = se * se;
        assert!(
            (got - want).abs() <= tol * want,
            "{}: {got} against {want}",
            t.relative_period
        );
    }
}

#[test]
fn invalid_input_is_refused() {
    let rows = absorbing();
    let config = DcdhOptions::default();
    let input = |r: Result<Dcdh, Error>| match r {
        Err(Error::Input(e)) => e,
        other => panic!("expected an input error, got {other:?}"),
    };
    let empty = SwitchingPanel::builder(&[], &[], &[], &[]).build();
    assert!(matches!(empty, Err(Error::Input(InputError::Empty { .. }))));
    let error = input(estimate(
        &rows,
        &DcdhOptions {
            effects: 0,
            ..config
        },
    ));
    assert_eq!(error.code(), "input.domain");
    let error = estimate(
        &rows,
        &DcdhOptions {
            bootstrap: Some(epistat_core::Bootstrap::new(1, epistat_core::Seed(3))),
            ..config
        },
    );
    assert_eq!(error.map(|_| ()).unwrap_err().code(), "input.domain");
    assert!(matches!(
        estimate(
            &rows,
            &DcdhOptions {
                same_switchers_pl: true,
                ..config
            }
        ),
        Err(Error::Unsupported { .. })
    ));
    let (unit, period) = ([0, 0, 1, 1], [1, 2, 1, 2]);
    let y = [Some(0.0); 4];
    let d = [Some(0.0), Some(1.0), Some(0.0), Some(0.0)];
    let nested = SwitchingPanel::builder(&unit, &period, &y, &d)
        .clusters(&[0, 1, 2, 2])
        .build();
    assert_eq!(
        nested.map(|p| p.len()),
        Err(Error::Input(InputError::NotConstant {
            what: "clusters",
            unit: 0
        }))
    );
    // Every group switches at the same date: no comparison group.
    let same: Vec<Row> = (0..10)
        .flat_map(|g| (1..=5).map(move |t| row(g, t, f64::from(u8::from(t >= 3)), 0.0)))
        .collect();
    assert_eq!(estimate(&same, &config), Err(Error::NoEffect));
}

/// Every entry point refuses bad columns as input errors.
#[test]
fn bad_columns_are_input_errors() {
    use epistat_testkit::{BadInput, assert_rejects_bad_input};
    let rows = absorbing();
    let columns = Columns::new(&rows);
    let panel = columns.panel();
    let unit = panel.unit().to_vec();
    let period = panel.period().to_vec();
    let d = panel.treatment().to_vec();
    let y: Vec<f64> = panel.outcome().iter().map(|v| v.unwrap()).collect();
    let reversed: Vec<i64> = period.iter().rev().copied().collect();
    let options = DcdhOptions::default();
    assert_rejects_bad_input("dcdh::estimate", |bad| {
        let y: Vec<Option<f64>> = bad.column(&y).into_iter().map(Some).collect();
        let built = match bad {
            BadInput::Unsorted => SwitchingPanel::builder(&unit, &reversed, &y, &d).build(),
            BadInput::Empty => SwitchingPanel::builder(&[], &[], &[], &[]).build(),
            _ => SwitchingPanel::builder(&unit, &period, &y, &d).build(),
        };
        Some(built.and_then(|panel| dcdh::estimate(&panel, &options)))
    });
}
