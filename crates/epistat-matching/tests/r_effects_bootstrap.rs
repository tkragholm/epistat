//! The bootstrap of matched sets checked against R's `boot` package 1.3.32.
//!
//! The bootstrap draws pair numbers from `epistat_rng`, which R cannot
//! reproduce, so R is given the draws. The ignored test
//! `write_bootstrap_draws` writes the pair numbers that
//! `effects::cluster_resamples` gives for 185 pairs, 200 replicates and a fixed
//! seed to `tests/data/effects_boot_draws.json`. `tests/data/effects_boot.R`
//! reads them, resamples the matched rows of `tests/data/effects.json` (the
//! 1:1 match without replacement and the 2:1 match with replacement, whose
//! pairs are the clusters) in the same order, refits the outcome model on each
//! replicate and writes the replicate statistics, with the intervals of
//! `boot.ci()`. The test first checks that the committed draws are what the
//! Rust code draws now, so the reference cannot go stale, and then compares:
//!
//! - every replicate of the difference, and of the ratio and odds ratio for the
//!   logistic model, to R's, relative 1e-6 for the logistic model for the
//!   reason given in `r_effects.rs` and 1e-9 for the linear one;
//! - the standard error to R's `sd()` of the replicates;
//! - the percentile interval to R's type 7 `quantile()`, and both it and
//!   `boot.ci(type = "perc")` to lie between the same two order statistics
//!   (`boot.ci` interpolates at rank `(R + 1) alpha / 2` on the normal scale
//!   and type 7 at `1 + (R - 1) alpha / 2`, and with 200 replicates and
//!   `alpha = 0.05` both ranks fall between the 5th and 6th smallest and
//!   between the 195th and 196th);
//! - the normal interval to `t0 -/+ z sd(t)`. `boot.ci(type = "norm")` centres
//!   on `2 t0 - mean(t)`, which removes the bootstrap bias, and the test checks
//!   that it is our interval moved by `-(mean(t) - t0)`. The bootstrap of
//!   Austin and Small reports the estimate plus and minus the normal quantile
//!   times the standard error, which is what the Rust code does.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_core::Bootstrap;
use epistat_matching::UnitRole;
use epistat_matching::effects::{
    BootstrapSpec, BootstrapStatistic, EffectSpec, MatchedSample, OutcomeModel, UnitData,
    bootstrap, cluster_resamples,
};
use epistat_matching::output::{LongMatches, MatchData};
use epistat_testkit::load_json;
use r_reference::{Versions, assert_r_version};
use serde::{Deserialize, Serialize};

const REPLICATES: usize = 200;
const SEED: u64 = 20_261_121;
const PAIRS: usize = 185;

#[derive(Serialize, Deserialize)]
struct Draws {
    replicates: usize,
    seed: u64,
    pairs: usize,
    resamples: Vec<Vec<usize>>,
}

fn data_path(name: &str) -> String {
    format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Regenerates the draws file that the R script reads.
#[test]
#[ignore = "writes tests/data/effects_boot_draws.json"]
fn write_bootstrap_draws() {
    let draws = Draws {
        replicates: REPLICATES,
        seed: SEED,
        pairs: PAIRS,
        resamples: cluster_resamples(PAIRS, Bootstrap::new(REPLICATES, SEED)),
    };
    let text = serde_json::to_string(&draws).unwrap();
    std::fs::write(data_path("effects_boot_draws.json"), text).unwrap();
}

#[derive(Deserialize)]
struct Matches {
    lalonde: Lalonde,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Lalonde {
    treat: Vec<f64>,
    re78: Vec<f64>,
    age: Vec<f64>,
    educ: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    rows: Vec<usize>,
    weight: Vec<f64>,
    subclass: Vec<Option<usize>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Stat {
    t0: f64,
    t: Vec<f64>,
    sd: f64,
    quantile7: (f64, f64),
    boot_norm: (f64, f64),
    boot_perc: (f64, f64),
}

#[derive(Deserialize)]
struct Run {
    case: String,
    outcome: String,
    difference: Stat,
    ratio: Option<Stat>,
    odds_ratio: Option<Stat>,
}

fn sample(matches: &Matches, case: &Case, binary: bool) -> MatchedSample {
    let l = &matches.lalonde;
    let n = l.treat.len();
    let units = UnitData::new(
        (1..=n).map(|i| format!("u{i}")).collect(),
        l.treat.iter().map(|t| *t > 0.5).collect(),
        l.re78
            .iter()
            .map(|y| {
                if binary {
                    f64::from(u8::from(*y > 0.0))
                } else {
                    *y
                }
            })
            .collect(),
    )
    .unwrap()
    .with_covariate("age", l.age.clone())
    .unwrap()
    .with_covariate("educ", l.educ.clone())
    .unwrap();
    let id: Vec<String> = case.rows.iter().map(|r| format!("u{r}")).collect();
    let role = vec![UnitRole::Case; id.len()];
    if case.name.starts_with("replace") {
        let long = LongMatches {
            id,
            role,
            pair_id: case.subclass.clone(),
            weight: case.weight.clone(),
        };
        MatchedSample::from_long_matches(&long, &units).unwrap()
    } else {
        let data = MatchData {
            id,
            role,
            weight: case.weight.clone(),
            subclass: case.subclass.clone(),
        };
        MatchedSample::from_match_data(&data, &units).unwrap()
    }
}

fn close(label: &str, got: f64, expected: f64, rel: f64, scale: f64) {
    let gap = (got - expected).abs();
    assert!(
        gap <= rel * expected.abs().max(1e-3 * scale),
        "{label}: got {got}, R {expected}"
    );
}

fn compare(label: &str, ours: &BootstrapStatistic, r: &Stat, rel: f64) {
    let scale = r.t0.abs().max(r.sd);
    assert_eq!(ours.replicates.len(), r.t.len(), "{label}");
    close(
        &format!("{label} estimate"),
        ours.summary.estimate,
        r.t0,
        rel,
        scale,
    );
    for (k, (a, b)) in ours.replicates.iter().zip(&r.t).enumerate() {
        close(&format!("{label} replicate {k}"), *a, *b, rel, scale);
    }
    close(&format!("{label} se"), ours.summary.se, r.sd, rel, scale);
    close(
        &format!("{label} perc low"),
        ours.summary.percentile.lower.unwrap(),
        r.quantile7.0,
        rel,
        scale,
    );
    close(
        &format!("{label} perc high"),
        ours.summary.percentile.upper.unwrap(),
        r.quantile7.1,
        rel,
        scale,
    );
    let mut sorted = r.t.clone();
    sorted.sort_by(f64::total_cmp);
    let (lo, hi) = (r.t.len() * 25 / 1000, r.t.len() * 975 / 1000);
    // Ranks 5 and 6, and 195 and 196, counting from one.
    for (value, bounds) in [
        (
            ours.summary.percentile.lower.unwrap(),
            (sorted[lo - 1], sorted[lo]),
        ),
        (r.boot_perc.0, (sorted[lo - 1], sorted[lo])),
        (
            ours.summary.percentile.upper.unwrap(),
            (sorted[hi - 1], sorted[hi]),
        ),
        (r.boot_perc.1, (sorted[hi - 1], sorted[hi])),
    ] {
        let slack = 1e-9 * scale;
        assert!(
            value >= bounds.0 - slack && value <= bounds.1 + slack,
            "{label}: {value} outside {bounds:?}"
        );
    }
    let mean = r.t.iter().sum::<f64>() / r.t.len() as f64;
    let bias = mean - r.t0;
    close(
        &format!("{label} norm low"),
        ours.summary.normal.lower.unwrap() - bias,
        r.boot_norm.0,
        rel,
        scale,
    );
    close(
        &format!("{label} norm high"),
        ours.summary.normal.upper.unwrap() - bias,
        r.boot_norm.1,
        rel,
        scale,
    );
}

#[test]
fn bootstrap_agrees_with_boot() {
    let draws: Draws = load_json(data_path("effects_boot_draws.json"));
    assert_eq!(
        draws.resamples,
        cluster_resamples(draws.pairs, Bootstrap::new(draws.replicates, draws.seed)),
        "the committed draws are not what the code draws; run write_bootstrap_draws and the R script"
    );
    assert_eq!(
        (draws.replicates, draws.seed, draws.pairs),
        (REPLICATES, SEED, PAIRS)
    );
    let matches: Matches = load_json(data_path("effects.json"));
    let reference: Reference = load_json(data_path("effects_boot.json"));
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["boot"], "1.3.32");
    assert_eq!(reference.runs.len(), 4);
    for run in &reference.runs {
        let case = matches.cases.iter().find(|c| c.name == run.case).unwrap();
        let binary = run.outcome == "binary";
        let sample = sample(&matches, case, binary);
        let mut spec = EffectSpec::new(if binary {
            OutcomeModel::Logistic
        } else {
            OutcomeModel::Linear
        });
        spec.interactions = true;
        let out = bootstrap(
            &sample,
            &spec,
            &BootstrapSpec::new(Bootstrap::new(REPLICATES, SEED)),
        )
        .unwrap();
        let rel = if binary { 1e-6 } else { 1e-9 };
        let label = format!("{} {}", run.case, run.outcome);
        compare(
            &format!("{label} difference"),
            &out.difference,
            &run.difference,
            rel,
        );
        match (&out.ratio, &run.ratio) {
            (Some(ours), Some(r)) => compare(&format!("{label} ratio"), ours, r, rel),
            (None, None) => {}
            _ => panic!("{label}: ratio present in only one"),
        }
        match (&out.odds_ratio, &run.odds_ratio) {
            (Some(ours), Some(r)) => compare(&format!("{label} odds ratio"), ours, r, rel),
            (None, None) => {}
            _ => panic!("{label}: odds ratio present in only one"),
        }
    }
}
