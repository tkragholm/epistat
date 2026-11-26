//! The bias analysis of rates against `episensr` 2.2.0 and a direct
//! simulation in R (`tests/data/bias_rates.R`).
//!
//! The corrected cases and person-time are a few operations on numbers of
//! order 1 to 10^5, compared to 1e-12 relative. `episensr::probsens_irr`
//! reports `(A / (A + C)) / (B / (B + D))` as the corrected rate ratio; the
//! test checks that this differs from the rate ratio, which epistat reports.
//! The probabilistic results are compared as in `r_bias`: the means of the
//! medians and limits of ten runs on each side, in standard errors of their
//! difference.

use epistat_stats::bias::{
    Dist, Misclass, Pair, ProbOptions, RateTable, Summary, correct_exposure_rates,
    probabilistic_misclass_rates,
};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Deterministic {
    se: f64,
    sp: f64,
    cells: [f64; 4],
    episensr_irr: f64,
    irr: f64,
}

#[derive(Deserialize)]
struct Run {
    syst: [f64; 3],
    tot: [f64; 3],
}

#[derive(Deserialize)]
struct Reference {
    versions: HashMap<String, String>,
    counts: [f64; 2],
    pt: [f64; 2],
    deterministic: Vec<Deterministic>,
    #[serde(rename = "K")]
    k: usize,
    reps: usize,
    probabilistic: Vec<Run>,
}

fn reference() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/bias_rates.json"
    ));
    assert!(r.versions["r"].starts_with("R version 4.6"));
    assert_eq!(r.versions["episensr"], "2.2.0");
    r
}

fn table(r: &Reference) -> RateTable {
    RateTable {
        cases_exposed: r.counts[0],
        cases_unexposed: r.counts[1],
        time_exposed: r.pt[0],
        time_unexposed: r.pt[1],
    }
}

#[test]
fn the_correction_of_rates_agrees_with_probsens_irr() {
    let r = reference();
    let t = table(&r);
    for case in &r.deterministic {
        let c = correct_exposure_rates(
            &t,
            &Misclass {
                se: [case.se; 2],
                sp: [case.sp; 2],
            },
        )
        .unwrap();
        let cells = c.table.unwrap();
        let got = [
            cells.cases_exposed,
            cells.cases_unexposed,
            cells.time_exposed,
            cells.time_unexposed,
        ];
        for (g, w) in got.iter().zip(&case.cells) {
            assert!((g - w).abs() <= 1e-12 * w.abs(), "{g} against {w}");
        }
        let irr = c.rate_ratio.unwrap();
        assert!((irr - case.irr).abs() <= 1e-12 * case.irr);
        // episensr's ratio of cases over cases plus person-time is off by
        // more than rounding.
        assert!((irr - case.episensr_irr).abs() > 1e-4 * irr);
    }
}

fn mean_var(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

/// With ten runs a side, a t statistic of 5 has a two-sided probability near
/// 7e-4.
const Z_MAX: f64 = 5.0;

#[test]
fn probabilistic_rates_agree_with_a_direct_simulation() {
    let r = reference();
    let t = table(&r);
    let se = Pair::Same(Dist::Trapezoidal {
        a: 0.75,
        b: 0.8,
        c: 0.9,
        d: 0.95,
    });
    let sp = Pair::Same(Dist::Uniform { lo: 0.97, hi: 1.0 });
    let fits: Vec<(Summary, Summary)> = (0..r.k)
        .map(|i| {
            let p = probabilistic_misclass_rates(
                &t,
                &se,
                &sp,
                &ProbOptions {
                    draws: r.reps,
                    seed: 5_000 + i as u64,
                    random_error: true,
                    alpha: 0.05,
                },
            )
            .unwrap();
            (p.adjusted, p.adjusted_total.unwrap())
        })
        .collect();
    let mut worst = 0.0_f64;
    for (which, label) in [(0, "systematic"), (1, "total")] {
        for (k, name) in ["median", "lower", "upper"].iter().enumerate() {
            let from_r: Vec<f64> = r
                .probabilistic
                .iter()
                .map(|x| if which == 0 { x.syst[k] } else { x.tot[k] })
                .collect();
            let from_rust: Vec<f64> = fits
                .iter()
                .map(|(s, tot)| {
                    let x = if which == 0 { s } else { tot };
                    [x.median, x.lower, x.upper][k]
                })
                .collect();
            let ((m1, v1), (m2, v2)) = (mean_var(&from_r), mean_var(&from_rust));
            let z = (m1 - m2) / ((v1 + v2) / from_r.len() as f64).sqrt();
            println!("{label} {name}: R {m1:.4} Rust {m2:.4} z {z:.2}");
            worst = worst.max(z.abs());
        }
    }
    assert!(worst < Z_MAX, "{worst}");
}
