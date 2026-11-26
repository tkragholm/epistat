//! `sccs::build_pieces` with several exposure series, seasons and calendar
//! periods, and `sccs::fit` on them, checked against R: `SCCS::standardsccs`
//! (SCCS 1.8) with two drugs, age groups and seasons, and an explicit R loop
//! for the pieces with R's `Date` and `glm` Poisson with person dummies for
//! calendar periods and three seasons. `tests/data/sccs_calendar.R` makes the
//! data and the references and records the versions.
//!
//! Tolerances, in standard errors `s`, follow `r_sccs`: the fit here stops at
//! `eps = 1e-12` with `|ln L|` below 3,000, `6e-9 s`; `standardsccs` stops at
//! the default of `clogit`, `1e-9`, which is `6e-6 s`, and the bound is ten
//! times that. `glm` runs at epsilon 1e-13 on the deviance, `6e-9 s` again,
//! bounded at `4e-8 s` as in `r_sccs`. The covariance of `clogit` is formed at
//! its last iterate, so its error relative to `sqrt(V_jj V_kk)` is first order
//! in the last step, at most `sqrt(2e-9 |ln L|)`, about `2.5e-3`; the bound of
//! `r_sccs`, `1e-4`, is kept, since the step is far smaller in practice. That
//! of `glm` is bounded at `1e-6` as in `r_sccs`.

use epistat_survival::sccs::{
    Calendar, Column, ExposureSeries, PieceSpec, Pieces, SccsOptions, SeasonStart, WindowSpec,
    build_pieces, fit,
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
}

#[derive(Deserialize)]
struct Case {
    label: String,
    start: Vec<f64>,
    end: Vec<f64>,
    event_offsets: Vec<usize>,
    events: Vec<f64>,
    series: Vec<Series>,
    age_cuts: Vec<f64>,
    origin: Vec<f64>,
    calendar_cuts: Vec<f64>,
    seasons: Vec<Season>,
    pieces: Option<Want>,
    coef: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    loglik: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Series {
    offsets: Vec<usize>,
    dates: Vec<f64>,
    windows: Vec<Win>,
}

#[derive(Deserialize)]
struct Win {
    from: f64,
    to: f64,
}

#[derive(Deserialize)]
struct Season {
    month: u32,
    day: u32,
}

#[derive(Deserialize)]
struct Want {
    person: Vec<u32>,
    start: Vec<f64>,
    end: Vec<f64>,
    events: Vec<f64>,
    season: Vec<u32>,
    period: Vec<u32>,
}

const EPS: f64 = 1e-12;

fn build(case: &Case) -> Pieces {
    let windows: Vec<Vec<WindowSpec>> = case
        .series
        .iter()
        .map(|s| {
            s.windows
                .iter()
                .map(|w| WindowSpec {
                    from: w.from,
                    to: w.to,
                })
                .collect()
        })
        .collect();
    let series: Vec<ExposureSeries<'_>> = case
        .series
        .iter()
        .zip(&windows)
        .map(|(s, w)| ExposureSeries {
            offsets: &s.offsets,
            dates: &s.dates,
            windows: w,
        })
        .collect();
    let seasons: Vec<SeasonStart> = case
        .seasons
        .iter()
        .map(|s| SeasonStart {
            month: s.month,
            day: s.day,
        })
        .collect();
    build_pieces(&PieceSpec {
        start: &case.start,
        end: &case.end,
        event_offsets: &case.event_offsets,
        events: &case.events,
        exposures: &series,
        age_cuts: &case.age_cuts,
        calendar: Some(Calendar {
            origin: &case.origin,
            cuts: &case.calendar_cuts,
            season_starts: &seasons,
        }),
    })
    .unwrap()
}

/// The largest deviation of the coefficients in standard errors and of the
/// covariance relative to `sqrt(V_jj V_kk)`.
fn deviations(beta: &[Option<f64>], cov: &[Option<f64>], case: &Case) -> (f64, f64) {
    let p = case.coef.len();
    let (mut db, mut dc) = (0.0f64, 0.0f64);
    for j in 0..p {
        let s = case.vcov[j][j].sqrt();
        db = db.max((beta[j].unwrap() - case.coef[j]).abs() / s);
        for k in 0..p {
            let scale = (case.vcov[j][j] * case.vcov[k][k]).sqrt();
            dc = dc.max((cov[j * p + k].unwrap() - case.vcov[j][k]).abs() / scale);
        }
    }
    (db, dc)
}

#[test]
fn several_series_seasons_and_calendar_periods_match_r() {
    let reference: Reference = load_json("tests/data/sccs_calendar.json");
    assert!(reference.versions.r.starts_with("R version 4.6"));
    assert_eq!(reference.versions.sccs, "1.8");
    let opts = SccsOptions {
        tolerance: EPS,
        ..SccsOptions::default()
    };
    for case in &reference.cases {
        let pieces = build(case);
        if let Some(want) = &case.pieces {
            // The cut points are integers, so the pieces compare exactly.
            assert_eq!(pieces.person, want.person, "{}", case.label);
            assert_eq!(pieces.start, want.start, "{}", case.label);
            assert_eq!(pieces.end, want.end, "{}", case.label);
            assert_eq!(pieces.events, want.events, "{}", case.label);
            assert_eq!(pieces.season, want.season, "{}", case.label);
            assert_eq!(pieces.calendar_period, want.period, "{}", case.label);
        }
        let length = pieces.length();
        let x = pieces.design();
        assert_eq!(x.ncols(), case.coef.len(), "{}", case.label);
        let f = fit(&pieces.data(&length), x.as_ref(), &opts).unwrap();
        assert!(f.converged, "{}", case.label);
        let (db, dc) = deviations(&f.coefficients, &f.covariance, case);
        let ll = f.log_likelihood.unwrap();
        if let Some(loglik) = &case.loglik {
            // standardsccs, through clogit.
            assert_eq!(
                pieces.columns(),
                vec![
                    Column::Window {
                        series: 0,
                        window: 0
                    },
                    Column::Window {
                        series: 0,
                        window: 1
                    },
                    Column::Window {
                        series: 1,
                        window: 0
                    },
                    Column::AgeBand(1),
                    Column::AgeBand(2),
                    Column::AgeBand(3),
                    Column::Season(1),
                ]
            );
            assert!(db <= 6e-5, "{}: coefficients {db} s", case.label);
            assert!(dc <= 1e-4, "{}: covariance {dc}", case.label);
            let fitted = (ll - loglik[1]).abs() / loglik[1].abs();
            assert!(fitted <= 1e-7, "{}: log likelihood {fitted}", case.label);
        } else {
            assert!(db <= 4e-8, "{}: coefficients {db} s", case.label);
            assert!(dc <= 1e-6, "{}: covariance {dc}", case.label);
        }
        eprintln!("{}: coef {db:e} s, cov {dc:e}", case.label);
    }
}
