//! `sccs::fit_event_dependent` checked against `SCCS::eventdepenexp` (SCCS
//! 1.8, gnm 1.1.5) on the rotavirus vaccine and intussusception data of the
//! package. `tests/data/sccs_eventdep.R` makes the references and records the
//! versions.
//!
//! Tolerances. Both sides repeat the weighted fit until the exposure
//! coefficients change by at most `1e-8` times their number, and the last
//! change is the error left by that iteration, a few `1e-9` in absolute terms
//! here. `eventdepenexp` fits each step with `gnm`, which stops when every
//! score is below `1e-6` times the square root of its information, so its
//! coefficients are within about `1e-6` standard errors of each step's
//! maximum, and the fits here (at `1e-12` on the log likelihood) within far
//! less. The bound is ten times that, `1e-5` standard errors. The sandwich is
//! a smooth function of the estimate and of the fitted values, so it moves by
//! the same relative order; the bound is `1e-4` relative to `sqrt(V_jj V_kk)`.

use epistat_survival::sccs::{
    EventDependentOptions, EventDependentSpec, WindowSpec, fit_event_dependent,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    start: Vec<f64>,
    end: Vec<f64>,
    event: Vec<f64>,
    dose_offsets: Vec<usize>,
    doses: Vec<f64>,
    age_cuts: Vec<f64>,
    fits: Vec<Want>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "SCCS")]
    sccs: String,
}

#[derive(Deserialize)]
struct Win {
    from: f64,
    to: f64,
}

#[derive(Deserialize)]
struct Want {
    same: bool,
    windows: Vec<Win>,
    coef: Vec<f64>,
    names: Vec<String>,
    vcov: Vec<Vec<f64>>,
}

#[test]
fn event_dependent_fits_match_eventdepenexp() {
    let r: Reference = load_json("tests/data/sccs_eventdep.json");
    assert_eq!(r.versions.sccs, "1.8");
    for want in &r.fits {
        let windows: Vec<WindowSpec> = want
            .windows
            .iter()
            .map(|w| WindowSpec {
                from: w.from,
                to: w.to,
            })
            .collect();
        let f = fit_event_dependent(
            &EventDependentSpec {
                start: &r.start,
                end: &r.end,
                event: &r.event,
                dose_offsets: &r.dose_offsets,
                doses: &r.doses,
                windows: &windows,
                age_cuts: &r.age_cuts,
                same_dose_parameters: want.same,
            },
            &EventDependentOptions::default(),
        )
        .unwrap();
        assert!(f.converged);
        let p = want.coef.len();
        assert_eq!(f.coefficients.len(), p, "{:?}", want.names);
        assert_eq!(
            f.n_exposure_columns,
            want.names.iter().filter(|n| n.starts_with("rv")).count()
        );
        let (mut db, mut dc) = (0.0f64, 0.0f64);
        for j in 0..p {
            let s = want.vcov[j][j].sqrt();
            db = db.max((f.coefficients[j].unwrap() - want.coef[j]).abs() / s);
            for k in 0..p {
                let scale = (want.vcov[j][j] * want.vcov[k][k]).sqrt();
                dc = dc.max((f.covariance.get(j, k).unwrap() - want.vcov[j][k]).abs() / scale);
            }
        }
        eprintln!(
            "windows {}, same {}: coef {db:e} s, cov {dc:e}, iterations {}",
            want.windows.len(),
            want.same,
            f.iterations
        );
        assert!(db <= 1e-5, "coefficients {db} s");
        assert!(dc <= 1e-4, "covariance {dc}");
        assert_eq!(f.n_cases, 566);
        assert_eq!(f.n_doses_after_event, 0);
    }
}
