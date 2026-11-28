//! Golden hashes of the probabilistic bias analyses: the risk ratio and odds
//! ratio summaries and the counts of draws, for misclassification of the
//! exposure and the outcome with and without random error, selection and
//! confounding. Recorded on aarch64 macOS and asserted only there. Elsewhere the
//! test computes the hash and, when `EPISTAT_HASH_OUT` is set, appends it to
//! that file.
use epistat_core::{Bootstrap, Estimate, Level, Seed};
use epistat_epi::Counts2x2;
use epistat_epi::bias::{
    ConfoundingMeasure, Dist, Kind, Pair, ProbOptions, probabilistic_confounding,
    probabilistic_misclass, probabilistic_selection,
};
use epistat_testkit::Fingerprint;

const T: Counts2x2 = Counts2x2 {
    exposed_cases: 150.0,
    unexposed_cases: 900.0,
    exposed_noncases: 9_850.0,
    unexposed_noncases: 89_100.0,
};

fn options(random_error: bool) -> ProbOptions {
    ProbOptions {
        bootstrap: Bootstrap::new(20_000, Seed(2026)),
        random_error,
        level: Level::alpha(0.05).unwrap(),
    }
}

fn summary(fp: Fingerprint, s: Option<Estimate>) -> Fingerprint {
    let s = s.expect("a summary");
    fp.f64(s.estimate)
        .f64(s.lower().expect("a lower percentile"))
        .f64(s.upper().expect("an upper percentile"))
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
            fp = summary(fp, r.risk_ratio);
            fp = summary(fp, r.odds_ratio);
            if random_error {
                fp = summary(fp, r.risk_ratio_total);
                fp = summary(fp, r.odds_ratio_total);
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
    fp = summary(fp, r.risk_ratio);
    fp = summary(fp, r.odds_ratio);
    fp = summary(fp, r.risk_ratio_total);
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
        fp = summary(fp, a.adjusted);
        fp = summary(fp, a.adjusted_total);
        fp = fp.u64(a.n_discarded as u64);
    }
    let hash = fp.finish();
    epistat_testkit::hash_out("probabilistic_results_are_unchanged", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 4_485_323_065_721_477_026, "golden hash");
    }
}
