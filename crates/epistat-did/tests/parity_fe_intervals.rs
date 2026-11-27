//! The reference distributions of the intervals of the fixed-effects family,
//! checked against fixest 0.14.2, etwfe 0.6.2 with marginaleffects 1.0.0,
//! did2s 1.2.1 and didimputation 0.5.1 (`tests/data/
//! generate_fe_intervals_reference.R`). For each model R records
//! `q = (upper - lower) / (2 se)` of its 95% intervals, which names the
//! distribution and its degrees of freedom: Student t with `G - 1` for
//! fixest's clustered covariances, `n - K` for its others and for did2s, and
//! the normal for fixest's Poisson models, for marginaleffects and, rounded
//! to 1.96, for didimputation.
//!
//! # Tolerance
//!
//! An interval `b +- q s` is rounded to `eps (|b| + q s)` at each end, so the
//! width over `2 s` carries `eps (|b| / s + q)` on each side, and as much
//! again in R's own width. The quantiles of `epistat_stats::dist` agree with
//! R's to about `1e-13` relative. The bound is twice the sum of the two.

use epistat_core::Estimate;
use epistat_did::{
    EtwfeFamily, EtwfeOptions, EtwfeVcov, ImputationOptions, ImputationTarget, StaggeredPanel,
    SunabOptions, TwfeOptions, TwoStageOptions, TwoStageSpec, etwfe, imputation, sunab, twfe,
    two_stage,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    unit: Vec<u32>,
    time: Vec<i64>,
    first_treated: Vec<Option<i64>>,
    clusters: Vec<u32>,
    outcome: Vec<f64>,
    count: Vec<f64>,
    twfe: f64,
    sunab: f64,
    sunab_aggregate: f64,
    etwfe_iid: f64,
    etwfe_iid_effects: f64,
    etwfe_cluster: f64,
    etwfe_cluster_effects: f64,
    etwfe_poisson: f64,
    etwfe_poisson_effects: f64,
    did2s_static: f64,
    did2s_event: f64,
    imputation: f64,
    imputation_horizons: f64,
    normal: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/fe_intervals_ref.json"
    ))
}

impl Reference {
    fn panel<'a>(&'a self, outcome: &'a [f64]) -> StaggeredPanel<'a> {
        StaggeredPanel::builder(&self.unit, &self.time, &self.first_treated, outcome)
            .clusters(&self.clusters)
            .build()
            .expect("a valid panel")
    }
}

/// Every estimate with a positive standard error has the quantile `q`.
fn assert_quantile<'a>(what: &str, estimates: impl Iterator<Item = &'a Estimate>, q: f64) {
    let mut checked = 0;
    for e in estimates {
        let Some(s) = e.se.filter(|&s| s > 0.0) else {
            continue;
        };
        let (lower, upper) = (e.lower().unwrap(), e.upper().unwrap());
        let mine = (upper - lower) / (2.0 * s);
        let tol = 2.0 * (2.0 * f64::EPSILON * (e.estimate.abs() / s + q) + 1e-13 * q);
        assert!(
            (mine - q).abs() <= tol,
            "{what}: {mine} against R's {q}, {tol:e}"
        );
        checked += 1;
    }
    assert!(checked > 0, "{what}: no interval");
}

#[test]
fn fixest_intervals_use_t_with_its_degrees_of_freedom() {
    let r = reference();
    let panel = r.panel(&r.outcome);
    let fit = twfe::fit(&panel, &TwfeOptions::default()).unwrap();
    assert_quantile(
        "twfe",
        fit.terms.iter().filter_map(|t| t.estimate.as_ref()),
        r.twfe,
    );
    let fit = sunab::fit(&panel, &SunabOptions::default()).unwrap();
    assert_quantile(
        "sunab cells",
        fit.cells.iter().filter_map(|c| c.estimate.as_ref()),
        r.sunab,
    );
    assert_quantile(
        "sunab aggregates",
        fit.aggregates.iter().map(|a| &a.estimate),
        r.sunab_aggregate,
    );
}

#[test]
fn etwfe_coefficients_follow_fixest_and_effects_marginaleffects() {
    let r = reference();
    let panel = r.panel(&r.outcome);
    for (vcov, q, q_effects, what) in [
        (EtwfeVcov::Iid, r.etwfe_iid, r.etwfe_iid_effects, "iid"),
        (
            EtwfeVcov::Clustered,
            r.etwfe_cluster,
            r.etwfe_cluster_effects,
            "clustered",
        ),
    ] {
        let fit = etwfe::fit(
            &panel,
            &EtwfeOptions {
                vcov,
                ..EtwfeOptions::default()
            },
        )
        .unwrap();
        assert_quantile(
            &format!("etwfe {what}"),
            fit.coefficients.iter().filter_map(|c| c.estimate.as_ref()),
            q,
        );
        assert_quantile(
            &format!("etwfe {what} effects"),
            fit.effects.iter().map(|e| &e.estimate),
            q_effects,
        );
    }
    let counts = r.panel(&r.count);
    let fit = etwfe::fit(
        &counts,
        &EtwfeOptions {
            family: EtwfeFamily::Poisson,
            ..EtwfeOptions::default()
        },
    )
    .unwrap();
    assert_quantile(
        "etwfe poisson",
        fit.coefficients.iter().filter_map(|c| c.estimate.as_ref()),
        r.etwfe_poisson,
    );
    assert_quantile(
        "etwfe poisson effects",
        fit.effects.iter().map(|e| &e.estimate),
        r.etwfe_poisson_effects,
    );
}

#[test]
fn did2s_intervals_use_t_with_the_residual_degrees_of_freedom() {
    let r = reference();
    let panel = r.panel(&r.outcome);
    for (spec, q, what) in [
        (TwoStageSpec::Static, r.did2s_static, "static"),
        (
            TwoStageSpec::EventStudy {
                references: vec![-1],
            },
            r.did2s_event,
            "event study",
        ),
    ] {
        let fit = two_stage::estimate(
            &panel,
            &TwoStageOptions {
                spec,
                ..TwoStageOptions::default()
            },
        )
        .unwrap();
        assert_quantile(
            &format!("did2s {what}"),
            fit.terms.iter().filter_map(|t| t.estimate.as_ref()),
            q,
        );
    }
}

/// didimputation forms `estimate +- 1.96 se`; the estimator takes the normal
/// quantile at its level, which at 95% is 1.959964.
#[test]
fn imputation_intervals_use_the_normal_quantile() {
    let r = reference();
    assert_eq!((r.imputation, r.imputation_horizons), (1.96, 1.96));
    let panel = r.panel(&r.outcome);
    for target in [ImputationTarget::Static, ImputationTarget::Horizons(None)] {
        let fit = imputation::estimate(
            &panel,
            &ImputationOptions {
                target,
                ..ImputationOptions::default()
            },
        )
        .unwrap();
        assert_quantile(
            "imputation",
            fit.effects.iter().filter_map(|e| e.estimate.as_ref()),
            r.normal,
        );
    }
}
