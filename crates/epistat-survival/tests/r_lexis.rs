//! `split_lexis` checked against `Epi::splitLexis` applied scale by scale and
//! `popEpi::splitMulti`, and the Poisson rate models against `glm` with
//! `Epi::ci.lin` and `Epi::ci.exp`.
//!
//! The quarters case is on a grid that is exact in binary, so every output must
//! agree bit for bit. The others agree up to the rounding of R's way of
//! computing the pieces: R adds up the lengths of the earlier pieces to move
//! every other scale, so a value or a length of a row cut into `k` pieces is
//! off by at most `(k + 4)` units in the last place of the largest magnitude
//! of the case, which is the tolerance used.

use epistat_survival::SurvData;
use epistat_survival::lexis::rates::{RateOptions, fit_rates};
use epistat_survival::lexis::{Lexis, LexisOptions, TimeScale, split_lexis};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    r_version: String,
    epi_version: String,
    popepi_version: String,
    cases: Vec<Case>,
    rates: RateCase,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    dur: Vec<f64>,
    status: Vec<i32>,
    entry: std::collections::HashMap<String, Vec<f64>>,
    scales: Vec<String>,
    breaks: Vec<Vec<f64>>,
    epi: Pieces,
    multi: Pieces,
    crop: Option<Pieces>,
    crop_breaks: Option<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct Pieces {
    row: Vec<u32>,
    entry: Vec<Vec<f64>>,
    dur: Vec<f64>,
    status: Vec<i32>,
}

#[derive(Deserialize)]
struct RateCase {
    length: Vec<f64>,
    events: Vec<f64>,
    columns: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    vcov: Vec<f64>,
    deviance: f64,
    iterations: usize,
    ctr: Vec<Vec<f64>>,
    ci_lin: Vec<Vec<f64>>,
    ci_exp: Vec<Vec<f64>>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/lexis.json"
    ))
}

#[test]
fn reference_versions() {
    let r = reference();
    assert!(r.r_version.starts_with("R version 4.6"));
    assert!(r.epi_version.starts_with("2.66"));
    assert!(r.popepi_version.starts_with("0.5"));
}

fn run(case: &Case, breaks: &[Vec<f64>], crop: bool) -> Lexis {
    let n_codes = u32::try_from(*case.status.iter().max().unwrap())
        .unwrap()
        .max(1);
    let data = SurvData::builder(&case.dur, &case.status)
        .n_codes(n_codes)
        .build()
        .unwrap();
    let scales: Vec<TimeScale<'_>> = case
        .scales
        .iter()
        .zip(breaks)
        .map(|(name, b)| TimeScale {
            entry: &case.entry[name],
            breaks: b,
        })
        .collect();
    split_lexis(&data, &scales, &LexisOptions { crop }).unwrap()
}

/// Worst deviation over the entries and lengths, in units of the tolerance.
fn compare(case: &Case, got: &Lexis, want: &Pieces, breaks: &[Vec<f64>], exact: bool) -> f64 {
    let what = &case.name;
    // R drops the rows of duration 0, which give one piece of length 0 here.
    let kept: Vec<usize> = (0..got.len())
        .filter(|&i| case.dur[got.row[i] as usize] > 0.0)
        .collect();
    assert_eq!(kept.len(), want.row.len(), "{what}: pieces");
    assert_eq!(
        kept.iter().map(|&i| got.row[i] + 1).collect::<Vec<_>>(),
        want.row,
        "{what}: rows"
    );
    assert_eq!(
        kept.iter().map(|&i| got.status[i]).collect::<Vec<_>>(),
        want.status,
        "{what}: status"
    );
    let magnitude = want
        .entry
        .iter()
        .flatten()
        .chain(&want.dur)
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut per_row = std::collections::HashMap::new();
    for r in &want.row {
        *per_row.entry(*r).or_insert(0_usize) += 1;
    }
    let k = *per_row.values().max().unwrap() as f64;
    let tol = (k + 4.0) * f64::EPSILON * magnitude;
    let mut worst = 0.0_f64;
    for (w, &i) in kept.iter().enumerate() {
        if exact {
            assert_eq!(
                got.length[i].to_bits(),
                want.dur[w].to_bits(),
                "{what}: length {i}"
            );
        } else {
            worst = worst.max((got.length[i] - want.dur[w]).abs() / tol);
        }
        for (j, b) in breaks.iter().enumerate() {
            if exact {
                assert_eq!(
                    got.entry[j][i].to_bits(),
                    want.entry[j][w].to_bits(),
                    "{what}: entry {i}"
                );
            } else {
                worst = worst.max((got.entry[j][i] - want.entry[j][w]).abs() / tol);
            }
            let cell = b.iter().filter(|&&c| c <= want.entry[j][w] + 1e-9).count();
            assert_eq!(
                got.interval[j][i] as usize, cell,
                "{what}: interval {i} scale {j}"
            );
        }
    }
    assert!(worst <= 1.0, "{what}: deviation {worst} tolerances");
    worst
}

#[test]
fn split_matches_splitlexis_and_splitmulti() {
    let r = reference();
    for case in &r.cases {
        if case.name == "short_piece" {
            continue;
        }
        let exact = case.name == "quarters";
        let got = run(case, &case.breaks, false);
        let a = compare(case, &got, &case.epi, &case.breaks, exact);
        let b = compare(case, &got, &case.multi, &case.breaks, exact);
        println!(
            "{}: {} pieces, worst deviation {a:.3} and {b:.3} tolerances",
            case.name,
            got.len()
        );
        if let (Some(crop), Some(cb)) = (&case.crop, &case.crop_breaks) {
            let got = run(case, cb, true);
            let c = compare(case, &got, crop, cb, exact);
            println!(
                "{}: cropped {} pieces, worst deviation {c:.3}",
                case.name,
                got.len()
            );
        }
    }
}

/// Both R functions drop a piece shorter than 1.5e-8, and with it the
/// person-time and, here, the last piece of the row. The pieces are kept here.
#[test]
fn a_short_last_piece_is_kept() {
    let r = reference();
    let case = r.cases.iter().find(|c| c.name == "short_piece").unwrap();
    for pieces in [&case.epi, &case.multi] {
        // R: row 1 is one piece of length 5 carrying the event.
        assert_eq!(pieces.row, vec![1, 2, 2]);
        assert_eq!(f64::to_bits(pieces.dur[0]), f64::to_bits(5.0));
        assert_eq!(pieces.status, vec![1, 0, 1]);
    }
    let got = run(case, &case.breaks, false);
    assert_eq!(got.row, vec![0, 0, 1, 1]);
    assert_eq!(got.status, vec![0, 1, 0, 1]);
    assert_eq!(f64::to_bits(got.length[0]), f64::to_bits(5.0));
    assert_eq!(
        f64::to_bits(got.length[1]),
        f64::to_bits((5.0 + 1e-9) - 5.0)
    );
    assert!((got.length[1] - 1e-9).abs() < 1e-15);
}

#[test]
fn rates_match_glm_and_ci() {
    let r = reference().rates;
    let columns: Vec<&[f64]> = r.columns.iter().map(Vec::as_slice).collect();
    let fit = fit_rates(&r.length, &r.events, &columns, &RateOptions::default()).unwrap();
    assert_eq!(fit.n_zero_length, 0);
    assert_eq!(fit.fit.iterations, r.iterations);
    // The condition number of X'WX is below 1e4 for this design, and the
    // fits run the same iterations from the same start, so the difference is
    // that of the solves: n * kappa * eps.
    let p = r.columns.len();
    let n = r.length.len() as f64;
    let tol = n * 1e4 * f64::EPSILON;
    let mut worst = 0.0_f64;
    let mut check = |got: f64, want: f64| {
        worst = worst.max((got - want).abs() / (tol * want.abs().max(1.0)));
    };
    for j in 0..p {
        check(fit.fit.coefficients[j].unwrap(), r.coefficients[j]);
        for k in 0..p {
            check(fit.fit.covariance[j * p + k].unwrap(), r.vcov[j * p + k]);
        }
    }
    check(fit.fit.deviance, r.deviance);
    let contrasts: Vec<&[f64]> = r.ctr.iter().map(Vec::as_slice).collect();
    let lin = fit.ci_lin(&contrasts, 0.05).unwrap();
    let exp = fit.ci_exp(&contrasts, 0.05).unwrap();
    for (i, (l, e)) in lin.iter().zip(&exp).enumerate() {
        let want = &r.ci_lin[i];
        check(l.estimate.unwrap(), want[0]);
        check(l.se.unwrap(), want[1]);
        check(l.lower.unwrap(), want[2]);
        check(l.upper.unwrap(), want[3]);
        check(l.p_value.unwrap(), want[4]);
        let want = &r.ci_exp[i];
        check(e.estimate.unwrap(), want[0]);
        check(e.lower.unwrap(), want[1]);
        check(e.upper.unwrap(), want[2]);
    }
    println!("rates: worst deviation {worst:.3} tolerances (tolerance {tol:e})");
    assert!(worst <= 1.0, "deviation {worst} tolerances");
}
