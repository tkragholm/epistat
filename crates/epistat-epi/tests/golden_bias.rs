//! Golden hashes of the probabilistic bias analyses: the risk ratio and odds
//! ratio summaries and the counts of draws, for misclassification of the
//! exposure and the outcome with and without random error, selection and
//! confounding. Recorded on aarch64 macOS.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_epi::bias::{
    ConfoundingMeasure, Dist, Kind, Pair, ProbOptions, Summary, Table2x2,
    probabilistic_confounding, probabilistic_misclass, probabilistic_selection,
};
use epistat_testkit::Fingerprint;

const T: Table2x2 = Table2x2 {
    a: 150.0,
    b: 900.0,
    c: 9_850.0,
    d: 89_100.0,
};

fn options(random_error: bool) -> ProbOptions {
    ProbOptions {
        draws: 20_000,
        seed: 2026,
        random_error,
        alpha: 0.05,
    }
}

fn summary(fp: Fingerprint, s: &Summary) -> Fingerprint {
    fp.f64(s.median).f64(s.lower).f64(s.upper)
}

#[test]
fn probabilistic_results_are_unchanged() {
    let se = Pair::Correlated {
        dists: [
            Dist::Trapezoidal {
                a: 0.70,
                b: 0.75,
                c: 0.85,
                d: 0.90,
            },
            Dist::Beta {
                alpha: 80.0,
                beta: 20.0,
            },
        ],
        rho: 0.8,
    };
    // Wide enough that some draws are discarded.
    let sp = Pair::Same(Dist::Uniform { lo: 0.985, hi: 1.0 });
    let mut fp = Fingerprint::new();
    for kind in [Kind::Exposure, Kind::Outcome] {
        for random_error in [false, true] {
            let r = probabilistic_misclass(&T, kind, &se, &sp, &options(random_error)).unwrap();
            fp = summary(fp, &r.risk_ratio);
            fp = summary(fp, &r.odds_ratio);
            if let (Some(rr), Some(or)) = (&r.risk_ratio_total, &r.odds_ratio_total) {
                fp = summary(fp, rr);
                fp = summary(fp, or);
            }
            fp = fp.u64(r.n_draws_used as u64).u64(r.n_discarded as u64);
        }
    }
    let s = [
        [
            Dist::Uniform { lo: 0.7, hi: 0.9 },
            Dist::Uniform { lo: 0.85, hi: 0.95 },
        ],
        [
            Dist::Uniform { lo: 0.93, hi: 0.97 },
            Dist::Uniform { lo: 0.93, hi: 0.97 },
        ],
    ];
    let r = probabilistic_selection(&T, &s, &options(true)).unwrap();
    fp = summary(fp, &r.risk_ratio);
    fp = summary(fp, &r.odds_ratio);
    fp = summary(fp, r.risk_ratio_total.as_ref().unwrap());
    let prevalence = Pair::Independent([
        Dist::Beta {
            alpha: 30.0,
            beta: 70.0,
        },
        Dist::Beta {
            alpha: 20.0,
            beta: 80.0,
        },
    ]);
    let association = Dist::LogNormal {
        mu: 2f64.ln(),
        sigma: 0.2,
    };
    for measure in [
        ConfoundingMeasure::RiskRatio,
        ConfoundingMeasure::OddsRatio,
        ConfoundingMeasure::RiskDifference,
    ] {
        // For the risk difference the association is a difference.
        let association = match measure {
            ConfoundingMeasure::RiskDifference => Dist::Uniform {
                lo: 0.005,
                hi: 0.015,
            },
            _ => association,
        };
        let a = probabilistic_confounding(&T, measure, &prevalence, &association, &options(true))
            .unwrap();
        fp = summary(fp, &a.adjusted);
        fp = summary(fp, a.adjusted_total.as_ref().unwrap());
        fp = fp.u64(a.n_discarded as u64);
    }
    assert_eq!(fp.finish(), 8_580_821_384_644_866_469, "golden hash");
}
