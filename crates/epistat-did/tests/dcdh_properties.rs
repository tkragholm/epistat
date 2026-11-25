//! Properties of `estimate_dcdh` that hold without a reference: exact
//! effects on outcomes without noise, independence of the thread count,
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

use epistat_did::{DcdhConfig, DcdhError, DcdhObservation, DcdhSwitchers, estimate_dcdh};

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

fn row(group: i64, time: i64, d: f64, y: f64) -> DcdhObservation {
    DcdhObservation {
        group,
        time,
        outcome: Some(y),
        treatment: Some(d),
        weight: 1.0,
        controls: Vec::new(),
        stratum: 0,
        cluster: None,
    }
}

/// Absorbing treatment first taken in period 3, 5 or 7 (or never) by 60
/// groups over 8 periods; the outcome is a group effect, a period effect and
/// `tau(e) = 1 + 0.5 e` from the `e`-th treated period on, with `e` from 0.
fn absorbing() -> Vec<DcdhObservation> {
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
    let r = estimate_dcdh(
        &rows,
        &DcdhConfig {
            effects: 4,
            placebo: 2,
            ..DcdhConfig::default()
        },
    )
    .expect("estimate");
    assert_eq!(r.effects.len(), 4);
    for (l, e) in r.effects.iter().enumerate() {
        let want = 1.0 + 0.5 * l as f64;
        let got = e.estimate.expect("effect");
        assert!((got - want).abs() <= tol * want, "effect {}: {got}", l + 1);
    }
    for p in &r.placebos {
        assert!(p.estimate.expect("placebo").abs() <= tol, "placebo");
    }
}

/// A treatment that moves among `levels` after a random first change, and
/// an outcome with the contemporaneous effect `beta D`.
fn moving(levels: usize, beta: f64, seed: u64) -> Vec<DcdhObservation> {
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
            let r = estimate_dcdh(
                &rows,
                &DcdhConfig {
                    effects: 3,
                    switchers,
                    dont_drop_larger_lower: true,
                    ..DcdhConfig::default()
                },
            );
            let Ok(r) = r else {
                assert!(matches!(r, Err(DcdhError::NoEffect)));
                continue;
            };
            ran += 1;
            let got = r.average_total_effect.estimate.expect("average");
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
    let config = DcdhConfig {
        effects: 3,
        placebo: 2,
        ..DcdhConfig::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| estimate_dcdh(&rows, &config).expect("estimate"))
    };
    let one = run(1);
    for threads in [2, 4, 12] {
        assert_eq!(run(threads), one, "{threads} threads");
    }
}

#[test]
fn clusters_of_one_group_give_the_unclustered_inference() {
    let rows = moving(2, 0.8, 9);
    let config = DcdhConfig {
        effects: 2,
        placebo: 1,
        ..DcdhConfig::default()
    };
    let plain = estimate_dcdh(&rows, &config).expect("estimate");
    let clustered_rows: Vec<DcdhObservation> = rows
        .iter()
        .map(|r| DcdhObservation {
            cluster: Some(r.group),
            ..r.clone()
        })
        .collect();
    let clustered = estimate_dcdh(&clustered_rows, &config).expect("estimate");
    for (a, b) in plain.effects.iter().zip(&clustered.effects) {
        assert_eq!(a.estimate, b.estimate);
        assert_eq!(a.se, b.se);
    }
    assert_eq!(plain.covariance, clustered.covariance);
}

#[test]
fn invalid_input_is_refused() {
    let rows = absorbing();
    let config = DcdhConfig::default();
    assert_eq!(estimate_dcdh(&[], &config), Err(DcdhError::Empty));
    assert!(matches!(
        estimate_dcdh(
            &rows,
            &DcdhConfig {
                effects: 0,
                ..config
            }
        ),
        Err(DcdhError::InvalidConfig(_))
    ));
    let mut dup = rows.clone();
    dup.push(rows[0].clone());
    assert!(matches!(
        estimate_dcdh(&dup, &config),
        Err(DcdhError::DuplicateCell { .. })
    ));
    let mut strata = rows.clone();
    strata[1].stratum = 1;
    assert!(matches!(
        estimate_dcdh(&strata, &config),
        Err(DcdhError::VaryingStratum { .. })
    ));
    let mut nested: Vec<DcdhObservation> = rows
        .iter()
        .map(|r| DcdhObservation {
            cluster: Some(r.group / 3),
            ..r.clone()
        })
        .collect();
    nested[1].cluster = Some(1000);
    assert!(matches!(
        estimate_dcdh(&nested, &config),
        Err(DcdhError::ClusterNotNested { .. })
    ));
    let mut missing = rows.clone();
    missing[2].outcome = Some(f64::NAN);
    assert!(matches!(
        estimate_dcdh(&missing, &config),
        Err(DcdhError::InvalidRow { .. })
    ));
    // Every group switches at the same date: no comparison group.
    let same: Vec<DcdhObservation> = (0..10)
        .flat_map(|g| (1..=5).map(move |t| row(g, t, f64::from(u8::from(t >= 3)), 0.0)))
        .collect();
    assert_eq!(estimate_dcdh(&same, &config), Err(DcdhError::NoEffect));
}
