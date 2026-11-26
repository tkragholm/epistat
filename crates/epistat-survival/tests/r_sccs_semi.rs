//! `sccs::sample_size` and `sccs::semiparametric_pieces` checked against R:
//! `SCCS::samplesize` and `SCCS::semisccs` (SCCS 1.8). `tests/data/sccs_semi.R`
//! makes the data and the references and records the versions.
//!
//! Tolerances. The sample size is a closed form of a few dozen operations and
//! two normal quantiles, each correct to a few units in the last place, so it
//! agrees with the same formula in R with unrounded quantiles to `1e-12`
//! relative, a hundred times the rounding of the operations. `samplesize`
//! rounds the quantiles to four decimals, which moves them by up to `5e-5`
//! and `n`, proportional to the square of their weighted sum of about 2.8, by
//! up to `4e-5 n`, so its ceiling must be equal where `n` is further than
//! `1e-4 n` from an integer and within one elsewhere. The semiparametric fit is
//! compared as the `standardsccs` cases of `r_sccs`: `semisccs` stops at the
//! default of `clogit`, `1e-9` on the log likelihood, which with `|ln L|`
//! about 800 is `4e-6 s` in the coefficients, bounded at ten times that, and
//! its covariance is formed at its last iterate, bounded at `1e-4` relative to
//! `sqrt(V_jj V_kk)`.

use epistat_survival::sccs::{
    ExposureSeries, PieceSpec, SampleSizeSpec, SccsOptions, WindowSpec, fit, sample_size,
    semiparametric_pieces,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    sizes: Vec<Size>,
    semi: Semi,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "SCCS")]
    sccs: String,
}

#[derive(Deserialize)]
struct Size {
    relative_incidence: f64,
    risk_length: f64,
    start: f64,
    end: f64,
    age_cuts: Vec<f64>,
    age_relative_incidence: Vec<f64>,
    exposed: Vec<f64>,
    alpha: f64,
    power: f64,
    samplesize: f64,
    unrounded: f64,
}

#[derive(Deserialize)]
struct Win {
    from: f64,
    to: f64,
}

#[derive(Deserialize)]
struct Semi {
    start: Vec<f64>,
    end: Vec<f64>,
    event_offsets: Vec<usize>,
    events: Vec<f64>,
    exposure_offsets: Vec<usize>,
    exposures: Vec<f64>,
    windows: Vec<Win>,
    n_event_times: usize,
    coef: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    loglik: Vec<f64>,
}

#[test]
fn sample_sizes_match_samplesize() {
    let reference: Reference = load_json("tests/data/sccs_semi.json");
    assert_eq!(reference.versions.sccs, "1.8");
    for s in &reference.sizes {
        let n = sample_size(&SampleSizeSpec {
            relative_incidence: s.relative_incidence,
            risk_length: s.risk_length,
            start: s.start,
            end: s.end,
            age_cuts: &s.age_cuts,
            age_relative_incidence: &s.age_relative_incidence,
            exposed: &s.exposed,
            alpha: s.alpha,
            power: s.power,
        })
        .unwrap();
        assert!(
            (n - s.unrounded).abs() <= 1e-12 * s.unrounded,
            "{n} against {}",
            s.unrounded
        );
        let fraction = n - n.floor();
        if fraction.min(1.0 - fraction) > 1e-4 * n {
            assert_eq!(n.ceil(), s.samplesize);
        } else {
            assert!((n.ceil() - s.samplesize).abs() <= 1.0);
        }
    }
}

#[test]
fn sample_size_refuses_designs_it_cannot_size() {
    let base = SampleSizeSpec {
        relative_incidence: 2.0,
        risk_length: 21.0,
        start: 0.0,
        end: 365.0,
        age_cuts: &[],
        age_relative_incidence: &[],
        exposed: &[1.0],
        alpha: 0.05,
        power: 0.8,
    };
    assert!(sample_size(&base).is_ok());
    let bad = |spec: SampleSizeSpec<'_>| assert!(sample_size(&spec).is_err());
    bad(SampleSizeSpec {
        relative_incidence: 1.0,
        ..base
    });
    bad(SampleSizeSpec {
        relative_incidence: -2.0,
        ..base
    });
    bad(SampleSizeSpec {
        risk_length: 400.0,
        ..base
    });
    bad(SampleSizeSpec {
        exposed: &[1.2],
        ..base
    });
    bad(SampleSizeSpec {
        exposed: &[0.0],
        ..base
    });
    bad(SampleSizeSpec {
        age_cuts: &[100.0],
        ..base
    });
    bad(SampleSizeSpec {
        age_cuts: &[400.0],
        age_relative_incidence: &[1.0],
        exposed: &[0.5, 0.5],
        ..base
    });
    bad(SampleSizeSpec { alpha: 0.0, ..base });
    bad(SampleSizeSpec { power: 1.0, ..base });
}

#[test]
fn the_semiparametric_fit_matches_semisccs() {
    let reference: Reference = load_json("tests/data/sccs_semi.json");
    let s = &reference.semi;
    let windows: Vec<WindowSpec> = s
        .windows
        .iter()
        .map(|w| WindowSpec {
            from: w.from,
            to: w.to,
        })
        .collect();
    let pieces = semiparametric_pieces(&PieceSpec {
        start: &s.start,
        end: &s.end,
        event_offsets: &s.event_offsets,
        events: &s.events,
        exposures: &[ExposureSeries {
            offsets: &s.exposure_offsets,
            dates: &s.exposures,
            windows: &windows,
        }],
        ..PieceSpec::default()
    })
    .unwrap();
    assert_eq!(pieces.event_times.len(), s.n_event_times);
    let x = pieces.design();
    assert_eq!(x.ncols(), s.coef.len());
    let opts = SccsOptions {
        tolerance: 1e-12,
        ..SccsOptions::default()
    };
    let f = fit(&pieces.data(), x.as_ref(), &opts).unwrap();
    assert!(f.converged);
    let p = s.coef.len();
    let (mut db, mut dc) = (0.0f64, 0.0f64);
    for j in 0..p {
        db = db.max((f.coefficients[j].unwrap() - s.coef[j]).abs() / s.vcov[j][j].sqrt());
        for k in 0..p {
            let scale = (s.vcov[j][j] * s.vcov[k][k]).sqrt();
            dc = dc.max((f.covariance[j * p + k].unwrap() - s.vcov[j][k]).abs() / scale);
        }
    }
    eprintln!("semisccs: coef {db:e} s, cov {dc:e}");
    assert!(db <= 5e-5, "coefficients {db} s");
    assert!(dc <= 1e-4, "covariance {dc}");
    let ll = f.log_likelihood.unwrap();
    assert!((ll - s.loglik[1]).abs() <= 1e-7 * s.loglik[1].abs());
    // Age cuts and a calendar are refused.
    assert!(
        semiparametric_pieces(&PieceSpec {
            start: &s.start,
            end: &s.end,
            event_offsets: &s.event_offsets,
            events: &s.events,
            age_cuts: &[100.0],
            ..PieceSpec::default()
        })
        .is_err()
    );
}
