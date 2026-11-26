//! The bias analysis against `episensr` 2.2.0 (`tests/data/generate_episensr.R`).
//!
//! The deterministic corrections are compared to machine precision. The
//! probabilistic ones cannot reproduce R's draws, so the reference holds the
//! median and the 2.5th and 97.5th percentiles of ten runs of R, and the Rust
//! code is run ten times with other seeds: the means must agree to within
//! standard errors of their difference, estimated from the ten runs on each
//! side.

use epistat_epi::Counts2x2;
use epistat_epi::bias::{
    Confounder, ConfoundingMeasure, Dist, Kind, Misclass, Pair, ProbOptions, ProbResult, Summary,
    adjust_confounding, correct_exposure, correct_exposure_predictive, correct_outcome,
    correct_selection, probabilistic_confounding, probabilistic_misclass, probabilistic_selection,
};
use epistat_testkit::{Fingerprint, assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Deterministic {
    kind: String,
    table: [f64; 4],
    parms: Vec<f64>,
    corrected: Vec<f64>,
    rr: Option<f64>,
    or: Option<f64>,
    smr: Option<f64>,
    mh: Option<f64>,
    crude: Option<f64>,
}

#[derive(Deserialize)]
struct Replicate {
    syst_rr: Option<[f64; 3]>,
    tot_rr: Option<[f64; 3]>,
    syst_or: Option<[f64; 3]>,
    tot_or: Option<[f64; 3]>,
    syst: Option<[f64; 3]>,
}

#[derive(Deserialize)]
struct Probabilistic {
    exposure_nondiff: Vec<Replicate>,
    exposure_diff: Vec<Replicate>,
    outcome: Vec<Replicate>,
    selection: Vec<Replicate>,
    confounding: Vec<Replicate>,
}

#[derive(Deserialize)]
struct Reference {
    #[serde(rename = "K")]
    k: usize,
    reps: usize,
    table: [f64; 4],
    deterministic: Vec<Deterministic>,
    probabilistic: Probabilistic,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/episensr_reference.json"
    ))
}

fn table(c: [f64; 4]) -> Counts2x2 {
    Counts2x2 {
        exposed_cases: c[0],
        unexposed_cases: c[1],
        exposed_noncases: c[2],
        unexposed_noncases: c[3],
    }
}

/// A few operations of order 1 on numbers of order 1 to 10^4, with a
/// denominator that can be as small as 0.05: a relative 1e-12 is far above the
/// rounding and far below any difference of formula.
const REL: f64 = 1e-12;

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1.0)
}

#[test]
fn deterministic_corrections_agree_to_machine_precision() {
    let r = reference();
    let mut worst = 0.0_f64;
    let mut seen = std::collections::BTreeMap::new();
    let mut fp = Fingerprint::new();
    for case in &r.deterministic {
        let t = table(case.table);
        let p = &case.parms;
        let mut check = |got: f64, want: f64, what: &str| {
            let d = rel(got, want);
            worst = worst.max(d);
            assert!(d < REL, "{} {what}: {got} vs {want}", case.kind);
            fp = std::mem::take(&mut fp).f64(got);
        };
        let (cells, rr, or) = match case.kind.as_str() {
            "exposure" | "outcome" => {
                let m = Misclass {
                    se: [p[0], p[1]],
                    sp: [p[2], p[3]],
                };
                let c = if case.kind == "exposure" {
                    correct_exposure(&t, &m)
                } else {
                    correct_outcome(&t, &m)
                }
                .unwrap();
                (c.cells.unwrap(), c.risk_ratio, c.odds_ratio)
            }
            "exposure_pv" => {
                let c = correct_exposure_predictive(&t, [p[0], p[1]], [p[2], p[3]]).unwrap();
                (c.cells.unwrap(), c.risk_ratio, c.odds_ratio)
            }
            "selection" => {
                let c = correct_selection(&t, &[[p[0], p[1]], [p[2], p[3]]]).unwrap();
                (c.cells.unwrap(), c.risk_ratio, c.odds_ratio)
            }
            kind => {
                let measure = match kind {
                    "confounding_RR" => ConfoundingMeasure::RiskRatio,
                    "confounding_OR" => ConfoundingMeasure::OddsRatio,
                    _ => ConfoundingMeasure::RiskDifference,
                };
                let c = adjust_confounding(
                    &t,
                    measure,
                    &Confounder {
                        association: p[0],
                        prevalence_exposed: p[1],
                        prevalence_unexposed: p[2],
                    },
                )
                .unwrap();
                let got = [
                    c.present.exposed_cases,
                    c.present.unexposed_cases,
                    c.present.exposed_noncases,
                    c.present.unexposed_noncases,
                    c.absent.exposed_cases,
                    c.absent.unexposed_cases,
                    c.absent.exposed_noncases,
                    c.absent.unexposed_noncases,
                ];
                for (g, w) in got.iter().zip(&case.corrected) {
                    check(*g, *w, "stratum cell");
                }
                check(c.crude, case.crude.unwrap(), "crude");
                check(c.adjusted_mh, case.mh.unwrap(), "mh");
                if let Some(smr) = case.smr {
                    check(c.adjusted_smr.unwrap(), smr, "smr");
                }
                *seen.entry(case.kind.clone()).or_insert(0) += 1;
                continue;
            }
        };
        for (g, w) in [
            cells.exposed_cases,
            cells.unexposed_cases,
            cells.exposed_noncases,
            cells.unexposed_noncases,
        ]
        .iter()
        .zip(&case.corrected)
        {
            check(*g, *w, "cell");
        }
        check(rr.unwrap(), case.rr.unwrap(), "risk ratio");
        check(or.unwrap(), case.or.unwrap(), "odds ratio");
        *seen.entry(case.kind.clone()).or_insert(0) += 1;
    }
    for kind in [
        "exposure",
        "outcome",
        "exposure_pv",
        "selection",
        "confounding_RR",
        "confounding_OR",
        "confounding_RD",
    ] {
        assert!(
            seen.get(kind).copied().unwrap_or(0) >= 3,
            "{kind}: {seen:?}"
        );
    }
    println!("worst relative deviation {worst:e} against {REL:e}");
    assert_eq!(fp.finish(), 12_307_114_443_917_333_060, "golden hash");
}

fn mean_var(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

/// The largest standardised difference of the means over the medians and
/// limits of one summary.
fn compare(label: &str, r: &[Option<[f64; 3]>], rust: &[Summary]) -> f64 {
    let mut worst = 0.0_f64;
    for (k, name) in ["median", "lower", "upper"].iter().enumerate() {
        let from_r: Vec<f64> = r.iter().map(|x| x.unwrap()[k]).collect();
        let from_rust: Vec<f64> = rust
            .iter()
            .map(|s| [s.median, s.lower, s.upper][k])
            .collect();
        let ((m1, v1), (m2, v2)) = (mean_var(&from_r), mean_var(&from_rust));
        let z = (m1 - m2) / ((v1 + v2) / from_r.len() as f64).sqrt();
        println!("{label} {name}: R {m1:.4} Rust {m2:.4} z {z:.2}");
        worst = worst.max(z.abs());
    }
    worst
}

/// Beyond this many standard errors the difference is not noise: with ten runs
/// per side, a t statistic of 5 has a two-sided probability near 7e-4.
const Z_MAX: f64 = 5.0;

struct Runs {
    results: Vec<ProbResult>,
}

impl Runs {
    fn summaries(&self, f: impl Fn(&ProbResult) -> Option<Summary>) -> Vec<Summary> {
        self.results.iter().map(|r| f(r).unwrap()).collect()
    }
}

fn runs(k: usize, reps: usize, f: impl Fn(&ProbOptions) -> ProbResult) -> Runs {
    Runs {
        results: (0..k)
            .map(|i| {
                f(&ProbOptions {
                    draws: reps,
                    seed: 4_000 + i as u64,
                    random_error: true,
                    alpha: 0.05,
                })
            })
            .collect(),
    }
}

#[test]
fn probabilistic_misclassification_agrees_with_episensr() {
    let r = reference();
    let t = table(r.table);
    let (k, reps) = (r.k, r.reps);
    let mut worst = 0.0_f64;

    let ProbPair(se, sp) = nondifferential();
    let fits = runs(k, reps, |o| {
        probabilistic_misclass(&t, Kind::Exposure, &se, &sp, o).unwrap()
    });
    worst = worst.max(check_misclass(
        "exposure nondiff",
        &r.probabilistic.exposure_nondiff,
        &fits,
    ));

    let se = Pair::Correlated {
        dists: [
            Dist::Uniform { lo: 0.7, hi: 0.9 },
            Dist::Uniform { lo: 0.65, hi: 0.85 },
        ],
        rho: 0.8,
    };
    let sp = Pair::Correlated {
        dists: [
            Dist::Beta {
                alpha: 200.0,
                beta: 12.0,
            },
            Dist::Beta {
                alpha: 180.0,
                beta: 10.0,
            },
        ],
        rho: 0.8,
    };
    let fits = runs(k, reps, |o| {
        probabilistic_misclass(&t, Kind::Exposure, &se, &sp, o).unwrap()
    });
    worst = worst.max(check_misclass(
        "exposure diff",
        &r.probabilistic.exposure_diff,
        &fits,
    ));

    let se = Pair::Same(Dist::TruncatedNormal {
        lo: 0.7,
        hi: 0.95,
        mean: 0.85,
        sd: 0.05,
    });
    let sp = Pair::Same(Dist::Uniform { lo: 0.95, hi: 1.0 });
    let fits = runs(k, reps, |o| {
        probabilistic_misclass(&t, Kind::Outcome, &se, &sp, o).unwrap()
    });
    worst = worst.max(check_misclass("outcome", &r.probabilistic.outcome, &fits));
    println!("largest |z| {worst:.2} against {Z_MAX}");
}

struct ProbPair(Pair, Pair);

fn nondifferential() -> ProbPair {
    ProbPair(
        Pair::Same(Dist::Trapezoidal {
            a: 0.75,
            b: 0.85,
            c: 0.95,
            d: 1.0,
        }),
        Pair::Same(Dist::Triangular {
            lo: 0.9,
            mode: 0.97,
            hi: 1.0,
        }),
    )
}

fn check_misclass(label: &str, reference: &[Replicate], fits: &Runs) -> f64 {
    let pick = |f: fn(&Replicate) -> Option<[f64; 3]>| reference.iter().map(f).collect::<Vec<_>>();
    let mut worst = 0.0_f64;
    for (name, r, rust) in [
        (
            "rr",
            pick(|x| x.syst_rr),
            fits.summaries(|p| Some(p.risk_ratio)),
        ),
        (
            "rr total",
            pick(|x| x.tot_rr),
            fits.summaries(|p| p.risk_ratio_total),
        ),
        (
            "or",
            pick(|x| x.syst_or),
            fits.summaries(|p| Some(p.odds_ratio)),
        ),
        (
            "or total",
            pick(|x| x.tot_or),
            fits.summaries(|p| p.odds_ratio_total),
        ),
    ] {
        worst = worst.max(compare(&format!("{label} {name}"), &r, &rust));
    }
    for p in &fits.results {
        assert_eq!(p.n_draws_used + p.n_discarded, 50_000);
    }
    assert!(worst < Z_MAX, "{label}: {worst}");
    worst
}

#[test]
fn probabilistic_selection_agrees_with_episensr_on_the_systematic_error() {
    let r = reference();
    let t = table(r.table);
    let s = [
        [
            Dist::Uniform { lo: 0.8, hi: 0.95 },
            Dist::Uniform { lo: 0.7, hi: 0.9 },
        ],
        [
            Dist::Trapezoidal {
                a: 0.4,
                b: 0.5,
                c: 0.6,
                d: 0.7,
            },
            Dist::Beta {
                alpha: 30.0,
                beta: 30.0,
            },
        ],
    ];
    let fits = runs(r.k, r.reps, |o| probabilistic_selection(&t, &s, o).unwrap());
    let pick = |f: fn(&Replicate) -> Option<[f64; 3]>| {
        r.probabilistic.selection.iter().map(f).collect::<Vec<_>>()
    };
    let a = compare(
        "selection rr",
        &pick(|x| x.syst_rr),
        &fits.summaries(|p| Some(p.risk_ratio)),
    );
    let b = compare(
        "selection or",
        &pick(|x| x.syst_or),
        &fits.summaries(|p| Some(p.odds_ratio)),
    );
    assert!(a.max(b) < Z_MAX, "{a} {b}");
}

#[test]
fn probabilistic_confounding_agrees_with_a_direct_simulation() {
    let r = reference();
    let t = table(r.table);
    let prevalence = Pair::Independent([
        Dist::Beta {
            alpha: 20.0,
            beta: 30.0,
        },
        Dist::Beta {
            alpha: 10.0,
            beta: 40.0,
        },
    ]);
    let association = Dist::LogNormal {
        mu: 2.0_f64.ln(),
        sigma: 0.25,
    };
    let results: Vec<Summary> = (0..r.k)
        .map(|i| {
            probabilistic_confounding(
                &t,
                ConfoundingMeasure::RiskRatio,
                &prevalence,
                &association,
                &ProbOptions {
                    draws: r.reps,
                    seed: 9_000 + i as u64,
                    random_error: false,
                    alpha: 0.05,
                },
            )
            .unwrap()
            .adjusted
        })
        .collect();
    let reference: Vec<Option<[f64; 3]>> =
        r.probabilistic.confounding.iter().map(|x| x.syst).collect();
    let z = compare("confounding", &reference, &results);
    assert!(z < Z_MAX, "{z}");
}

#[test]
fn the_draws_do_not_depend_on_the_thread_pool() {
    let r = reference();
    let t = table(r.table);
    let ProbPair(se, sp) = nondifferential();
    assert_same_on_threads(|| {
        let p = probabilistic_misclass(
            &t,
            Kind::Exposure,
            &se,
            &sp,
            &ProbOptions {
                draws: 3_000,
                seed: 1,
                random_error: true,
                alpha: 0.05,
            },
        )
        .unwrap();
        let (rr, or) = (p.risk_ratio_total.unwrap(), p.odds_ratio);
        vec![
            rr.median,
            rr.lower,
            rr.upper,
            or.median,
            p.n_discarded as f64,
        ]
    });
}
