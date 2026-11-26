//! `sccs::build_pieces` and `sccs::fit` checked against R: `SCCS::standardsccs`
//! (SCCS 1.8), `glm` Poisson with a dummy for each person, `survival::clogit`
//! and an explicit R loop for the pieces. `tests/data/sccs.R` makes the data
//! and the references and records the versions.
//!
//! How the references map. The pieces are compared exactly: the cut points are
//! sums of the integers in the data, so every start, end and count is exact in
//! `f64`. The coefficients and covariance of `glm` with person dummies equal
//! the conditional ones (the profile likelihood of the person intercepts), and
//! the conditional log likelihood is evaluated by hand at them. `SCCS` and
//! `clogit` give the same slopes by the stacked format, with the exception
//! that `standardsccs` drops a repeated event day of a person.
//!
//! Tolerances, in standard errors `s` of the coefficient. The fit stops when
//! the log likelihood changes by at most `eps |ln L|`, so the last Newton step
//! has information norm at most `sqrt(2 eps |ln L|)` and the error after it,
//! which is quadratic in the step, is of the order `2 eps |ln L|` in the same
//! norm. The test fits at `eps = 1e-12` with `|ln L|` below 2,000, which is
//! `4e-9 s`. `glm` is run at epsilon 1e-13 on the deviance, `clogit` at 1e-12,
//! the same order, and `standardsccs` at the default of `clogit`, 1e-9, which
//! is `4e-6 s`. Each bound is ten times the figure. The covariance of `glm` is
//! formed from the weights of its last iteration, before the last update, so
//! its error relative to `sqrt(V_jj V_kk)` is first order in that step, whose
//! information norm is at most `sqrt(2e-13 D)` for a deviance `D`, about
//! `1e-5` here; the bound is `1e-6`, which the data meet by two orders. The
//! sandwich is checked against the Richardson derivative of `numDeriv`, whose
//! error is about `1e-9`, at `1e-6`. The log likelihoods are compared
//! relative.

use epistat_linalg::Mat;
use epistat_survival::sccs::{
    ExposureSeries, PieceSpec, Pieces, SccsData, SccsOptions, WindowSpec, build_pieces, fit,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    #[serde(rename = "SCCS")]
    sccs: String,
    survival: String,
}

#[derive(Deserialize)]
struct Case {
    label: String,
    spec: Spec,
    extra: Option<Vec<f64>>,
    pieces: Ref,
    n_windows: usize,
    n_age_bands: usize,
    glm: Option<Glm>,
    sccs: Option<Sccs>,
    clogit: Option<Clogit>,
    aliased_extra: Option<bool>,
    separation: Option<bool>,
}

#[derive(Deserialize)]
struct Spec {
    start: Vec<f64>,
    end: Vec<f64>,
    event_offsets: Vec<usize>,
    events: Vec<f64>,
    exposure_offsets: Vec<usize>,
    exposures: Vec<f64>,
    age_cuts: Vec<f64>,
    windows: Vec<Win>,
}

#[derive(Deserialize)]
struct Win {
    from: f64,
    to: f64,
}

#[derive(Deserialize)]
struct Ref {
    person: Vec<u32>,
    start: Vec<f64>,
    end: Vec<f64>,
    events: Vec<f64>,
    age_band: Vec<u32>,
    window: Vec<i64>,
}

#[derive(Deserialize)]
struct Glm {
    beta: Vec<f64>,
    cov: Vec<Vec<f64>>,
    robust: Vec<Vec<f64>>,
    loglik: f64,
    null_loglik: f64,
    n_used: usize,
    n_events: f64,
}

#[derive(Deserialize)]
struct Sccs {
    coef: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    loglik: Vec<f64>,
}

#[derive(Deserialize)]
struct Clogit {
    coef: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    loglik: Vec<f64>,
}

fn build(spec: &Spec) -> Pieces {
    let windows: Vec<WindowSpec> = spec
        .windows
        .iter()
        .map(|w| WindowSpec {
            from: w.from,
            to: w.to,
        })
        .collect();
    build_pieces(&PieceSpec {
        start: &spec.start,
        end: &spec.end,
        event_offsets: &spec.event_offsets,
        events: &spec.events,
        exposures: &[ExposureSeries {
            offsets: &spec.exposure_offsets,
            dates: &spec.exposures,
            windows: &windows,
        }],
        age_cuts: &spec.age_cuts,
        calendar: None,
    })
    .unwrap()
}

fn design(case: &Case, pieces: &Pieces) -> Mat<f64> {
    let base = pieces.design();
    match &case.extra {
        None => base,
        Some(extra) => Mat::from_fn(base.nrows(), base.ncols() + 1, |i, j| {
            if j < base.ncols() {
                base[(i, j)]
            } else {
                extra[pieces.person[i] as usize]
            }
        }),
    }
}

const EPS: f64 = 1e-12;

fn options(robust: bool) -> SccsOptions {
    SccsOptions {
        tolerance: EPS,
        robust,
        ..SccsOptions::default()
    }
}

/// The largest deviation of the coefficients in standard errors and of the
/// covariance relative to `sqrt(V_jj V_kk)`.
fn deviations(
    beta: &[Option<f64>],
    cov: &[Option<f64>],
    want_beta: &[f64],
    want_cov: &[Vec<f64>],
) -> (f64, f64) {
    let p = want_beta.len();
    let (mut db, mut dc) = (0.0f64, 0.0f64);
    for j in 0..p {
        let s = want_cov[j][j].sqrt();
        db = db.max((beta[j].unwrap() - want_beta[j]).abs() / s);
        for k in 0..p {
            let scale = (want_cov[j][j] * want_cov[k][k]).sqrt();
            dc = dc.max((cov[j * p + k].unwrap() - want_cov[j][k]).abs() / scale);
        }
    }
    (db, dc)
}

#[test]
fn pieces_equal_the_explicit_loop() {
    let reference: Reference = load_json("tests/data/sccs.json");
    assert!(reference.versions.r.starts_with("R version 4.6"));
    for case in &reference.cases {
        let got = build(&case.spec);
        let want = &case.pieces;
        assert_eq!(got.person, want.person, "{}", case.label);
        assert_eq!(got.start, want.start, "{}", case.label);
        assert_eq!(got.end, want.end, "{}", case.label);
        assert_eq!(got.events, want.events, "{}", case.label);
        assert_eq!(got.age_band, want.age_band, "{}", case.label);
        let window: Vec<i64> = got.window[0]
            .iter()
            .map(|w| w.map_or(-1, i64::from))
            .collect();
        assert_eq!(window, want.window, "{}", case.label);
        assert_eq!(got.n_windows, vec![case.n_windows]);
        assert_eq!(got.n_age_bands, case.n_age_bands);
    }
}

#[test]
fn fits_match_glm_with_person_dummies_and_scca() {
    let reference: Reference = load_json("tests/data/sccs.json");
    assert_eq!(reference.versions.sccs, "1.8");
    assert_eq!(reference.versions.survival, "3.8.12");
    let mut worst = [0.0f64; 6];
    for case in &reference.cases {
        let Some(g) = &case.glm else { continue };
        let pieces = build(&case.spec);
        let length = pieces.length();
        let x = design(case, &pieces);
        let data: SccsData<'_> = pieces.data(&length);
        let f = fit(&data, x.as_ref(), &options(true)).unwrap();
        assert!(f.converged, "{}", case.label);
        assert_eq!(f.n_persons_used, g.n_used, "{}", case.label);
        assert!((f.n_events - g.n_events).abs() < 0.5, "{}", case.label);
        let p = g.beta.len();
        let aliased = case.aliased_extra == Some(true);
        if aliased {
            assert_eq!(f.coefficients.len(), p + 1);
            assert!(f.coefficients[p].is_none());
            assert_eq!(f.rank, p);
        }
        let n_cols = f.coefficients.len();
        // Compare on the columns of the reference, drop the aliased one.
        let coef: Vec<Option<f64>> = f.coefficients[..p].to_vec();
        let cov: Vec<Option<f64>> = (0..p)
            .flat_map(|j| (0..p).map(move |k| (j, k)))
            .map(|(j, k)| f.covariance[j * n_cols + k])
            .collect();
        let rob: Vec<Option<f64>> = (0..p)
            .flat_map(|j| (0..p).map(move |k| (j, k)))
            .map(|(j, k)| f.robust_covariance.as_ref().unwrap()[j * n_cols + k])
            .collect();
        let (db, dc) = deviations(&coef, &cov, &g.beta, &g.cov);
        assert!(db <= 4e-8, "{}: coefficients {db} s", case.label);
        assert!(dc <= 1e-6, "{}: covariance {dc}", case.label);
        // The sandwich from numerical gradients: the derivative error of
        // numDeriv's Richardson scheme with r = 6 is about 1e-9 relative.
        let (_, dr) = deviations(&coef, &rob, &g.beta, &g.robust);
        assert!(dr <= 1e-6, "{}: robust covariance {dr}", case.label);
        let ll = f.log_likelihood.unwrap();
        let ll_err = (ll - g.loglik).abs() / g.loglik.abs();
        assert!(ll_err <= 1e-10, "{}: log likelihood", case.label);
        let null_err = (f.null_log_likelihood.unwrap() - g.null_loglik).abs() / g.null_loglik.abs();
        assert!(null_err <= 1e-12, "{}: null log likelihood", case.label);
        worst[0] = worst[0].max(db);
        worst[1] = worst[1].max(dc);
        worst[2] = worst[2].max(dr);
        worst[3] = worst[3].max(ll_err);
        if let Some(s) = &case.sccs {
            let (sb, sc) = deviations(&coef, &cov, &s.coef, &s.vcov);
            assert!(sb <= 4e-5, "{}: SCCS coefficients {sb} s", case.label);
            assert!(sc <= 1e-4, "{}: SCCS covariance {sc}", case.label);
            // loglik of summary.coxph: null then fitted.
            let fitted = (ll - s.loglik[1]).abs() / s.loglik[1].abs();
            assert!(fitted <= 1e-7, "{}: SCCS log likelihood", case.label);
            worst[4] = worst[4].max(sb);
            worst[5] = worst[5].max(sc);
        }
        if let Some(c) = &case.clogit {
            let (cb, cc) = deviations(&coef, &cov, &c.coef, &c.vcov);
            assert!(cb <= 4e-8 && cc <= 1e-6, "{}: clogit", case.label);
            let fitted = (ll - c.loglik[1]).abs() / c.loglik[1].abs();
            assert!(fitted <= 1e-10, "{}: clogit log likelihood", case.label);
        }
    }
    eprintln!(
        "worst: glm coef {:e} s, cov {:e}, robust {:e}, loglik {:e}; SCCS coef {:e} s, cov {:e}",
        worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]
    );
}

#[test]
fn separation_is_reported_not_repaired() {
    let reference: Reference = load_json("tests/data/sccs.json");
    let case = reference
        .cases
        .iter()
        .find(|c| c.separation == Some(true))
        .unwrap();
    let pieces = build(&case.spec);
    let length = pieces.length();
    let x = design(case, &pieces);
    let opts = SccsOptions {
        max_iterations: 100,
        ..options(false)
    };
    let f = fit(&pieces.data(&length), x.as_ref(), &opts).unwrap();
    assert!(!f.converged);
    assert_eq!(f.monotone_columns, vec![0]);
    assert!(f.coefficients[0].is_none() && f.log_likelihood.is_none());
    assert!(f.null_log_likelihood.is_some());
}
