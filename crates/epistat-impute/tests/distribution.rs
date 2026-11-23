//! The distribution of the imputed values against mice 3.19.0, from
//! `tests/data/impute_distribution.R`.
//!
//! No random number agrees with R's, so the imputations are compared as
//! distributions. Each of `m` independent data sets gives a statistic of its
//! imputed values (mean, variance, quantiles, a share of a level) and of the
//! analysis of its completed data. Each statistic is a mean over `m` data
//! sets from each implementation, with a Monte Carlo standard error from the
//! spread over the sets, and the two means must lie within 4 combined standard
//! errors (decision 59). With about 40 comparisons per run, a correct
//! implementation fails with probability about 0.3 per cent; the seeds are
//! fixed in the test and were not chosen by trial. The between-imputation
//! variance of a pooled coefficient has relative standard error
//! `sqrt(2 / (m - 1))` for normally distributed estimates, and the
//! fraction of missing information has the delta-method error from the two.
//!
//! The data are the 25 rows of `mice::nhanes2` and 2,000 simulated register-like
//! rows with `inc` (skewed, missing at random), `edu` (nominal) and `smk`
//! (binary). The simulation has 2,000 rows and 200 data sets where the design
//! note has 10,000 and 500, so that a debug build of the test stays short.
//! mice's `polyreg` does not draw the coefficients; `Polyreg { proper: false }`
//! is compared with it, and the proper draw is checked to leave the mean
//! where it is and to raise the between-imputation spread.

mod common;

use common::{
    PooledStats, compare_pooled, compare_stats, lm, numeric_stats, pooled_stats, share_stats,
    column_of, variance,
};
use epistat_impute::{Column, Imputed, ImputeOptions, ImputeSpec, Incomplete, Method, impute};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    mice: String,
}

#[derive(Deserialize)]
struct Analysis {
    estimate: Vec<Vec<f64>>,
    variance: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Block {
    stats: std::collections::BTreeMap<String, Vec<Vec<f64>>>,
    analysis: Analysis,
}

#[derive(Deserialize)]
struct Nhanes {
    age: Vec<u32>,
    bmi: Vec<Option<f64>>,
    hyp: Vec<Option<u8>>,
    chl: Vec<Option<f64>>,
    stats: std::collections::BTreeMap<String, Vec<Vec<f64>>>,
    analysis: Analysis,
    norm: Block,
}

#[derive(Deserialize)]
struct Sim {
    x1: Vec<f64>,
    x2: Vec<u8>,
    inc: Vec<f64>,
    edu: Vec<u32>,
    smk: Vec<u8>,
    y: Vec<f64>,
    missing_inc: Vec<bool>,
    missing_edu: Vec<bool>,
    missing_smk: Vec<bool>,
    stats: std::collections::BTreeMap<String, Vec<Vec<f64>>>,
    analysis: Analysis,
    norm: Block,
    pmm: Block,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    nhanes: Nhanes,
    sim: Sim,
}

fn reference() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/impute_distribution.json"
    ));
    assert!(r.versions.r.starts_with("R version"));
    assert_eq!(r.versions.mice, "3.19.0");
    r
}

const K: f64 = 4.0;

fn nhanes_data(n: &Nhanes) -> Incomplete {
    let flags = |v: &[bool]| v.to_vec();
    let miss_bmi: Vec<bool> = n.bmi.iter().map(Option::is_none).collect();
    let miss_hyp: Vec<bool> = n.hyp.iter().map(Option::is_none).collect();
    let miss_chl: Vec<bool> = n.chl.iter().map(Option::is_none).collect();
    Incomplete::new(
        vec![
            Column::Nominal {
                codes: n.age.clone(),
                levels: 3,
            },
            Column::Continuous(n.bmi.iter().map(|v| v.unwrap_or(0.0)).collect()),
            Column::Binary(n.hyp.iter().map(|v| v.unwrap_or(0)).collect()),
            Column::Continuous(n.chl.iter().map(|v| v.unwrap_or(0.0)).collect()),
        ],
        &[vec![false; 25], flags(&miss_bmi), flags(&miss_hyp), flags(&miss_chl)],
    )
    .unwrap()
}

fn spec_of(data: &Incomplete, methods: Vec<Option<Method>>) -> ImputeSpec {
    let mut spec = ImputeSpec::default_for(data);
    spec.methods = methods;
    spec
}

fn nhanes_analysis(imputed: &Imputed, data: &Incomplete) -> Vec<PooledStats> {
    let (mut est, mut var) = (vec![vec![]; 2], vec![vec![]; 2]);
    for s in 0..imputed.m {
        let c = imputed.completed(data, s).unwrap();
        let (coef, v) = lm(&[&c[1], &c[2]], &c[3]);
        for t in 0..2 {
            est[t].push(coef[t + 1]);
            var[t].push(v[t + 1]);
        }
    }
    (0..2).map(|t| pooled_stats(&est[t], &var[t])).collect()
}

fn reference_pooled(a: &Analysis) -> Vec<PooledStats> {
    (0..a.estimate[0].len())
        .map(|t| pooled_stats(&column_of(&a.estimate, t), &column_of(&a.variance, t)))
        .collect()
}

#[test]
fn nhanes2_pmm_and_logreg_match_mice() {
    let r = reference().nhanes;
    let data = nhanes_data(&r);
    let spec = spec_of(
        &data,
        vec![None, Some(Method::pmm()), Some(Method::Logreg), Some(Method::pmm())],
    );
    let imputed = impute(
        &data,
        &spec,
        &ImputeOptions {
            m: 500,
            max_iterations: 5,
            seed: 20_261_123,
        },
    )
    .unwrap();
    compare_stats("nhanes bmi", &numeric_stats(&imputed, 1), &r.stats["bmi"], K);
    compare_stats("nhanes chl", &numeric_stats(&imputed, 3), &r.stats["chl"], K);
    compare_stats("nhanes hyp", &share_stats(&imputed, 2, 2), &r.stats["hyp"], K);
    let ours = nhanes_analysis(&imputed, &data);
    let theirs = reference_pooled(&r.analysis);
    for t in 0..2 {
        compare_pooled(&format!("nhanes coefficient {t}"), &ours[t], &theirs[t], K);
    }
}

#[test]
fn nhanes2_norm_matches_mice() {
    let r = reference().nhanes;
    let data = nhanes_data(&r);
    let spec = spec_of(
        &data,
        vec![None, Some(Method::Norm), Some(Method::Logreg), Some(Method::Norm)],
    );
    let imputed = impute(
        &data,
        &spec,
        &ImputeOptions {
            m: 500,
            max_iterations: 5,
            seed: 20_261_124,
        },
    )
    .unwrap();
    compare_stats("nhanes norm bmi", &numeric_stats(&imputed, 1), &r.norm.stats["bmi"], K);
    compare_stats("nhanes norm chl", &numeric_stats(&imputed, 3), &r.norm.stats["chl"], K);
    let ours = nhanes_analysis(&imputed, &data);
    let theirs = reference_pooled(&r.norm.analysis);
    for t in 0..2 {
        compare_pooled(&format!("nhanes norm coefficient {t}"), &ours[t], &theirs[t], K);
    }
}

fn sim_columns(s: &Sim, incomplete: [bool; 3]) -> Incomplete {
    let n = s.y.len();
    let mask = |flag: bool, m: &[bool]| if flag { m.to_vec() } else { vec![false; n] };
    Incomplete::new(
        vec![
            Column::Continuous(s.x1.clone()),
            Column::Binary(s.x2.clone()),
            Column::Continuous(s.inc.clone()),
            Column::Nominal {
                codes: s.edu.clone(),
                levels: 3,
            },
            Column::Binary(s.smk.clone()),
            Column::Continuous(s.y.clone()),
        ],
        &[
            vec![false; n],
            vec![false; n],
            mask(incomplete[0], &s.missing_inc),
            mask(incomplete[1], &s.missing_edu),
            mask(incomplete[2], &s.missing_smk),
            vec![false; n],
        ],
    )
    .unwrap()
}

fn sim_analysis(imputed: &Imputed, data: &Incomplete) -> Vec<PooledStats> {
    let (mut est, mut var) = (vec![vec![]; 2], vec![vec![]; 2]);
    for s in 0..imputed.m {
        let c = imputed.completed(data, s).unwrap();
        let (coef, v) = lm(&[&c[0], &c[2], &c[4]], &c[5]);
        for (t, k) in [(0, 2), (1, 3)] {
            est[t].push(coef[k]);
            var[t].push(v[k]);
        }
    }
    (0..2).map(|t| pooled_stats(&est[t], &var[t])).collect()
}

fn imputed_sets(imputed: &Imputed) -> usize {
    imputed.m
}

fn options(seed: u64) -> ImputeOptions {
    ImputeOptions {
        m: 200,
        max_iterations: 5,
        seed,
    }
}

fn sim_full(proper: bool, seed: u64) -> (Incomplete, Imputed) {
    let r = reference().sim;
    let data = sim_columns(&r, [true, true, true]);
    let spec = spec_of(
        &data,
        vec![None, None, Some(Method::pmm()), Some(Method::Polyreg { proper }), Some(Method::Logreg), None],
    );
    let imputed = impute(&data, &spec, &options(seed)).unwrap();
    (data, imputed)
}

#[test]
fn simulated_mixed_types_match_mice() {
    let r = reference().sim;
    // mice's polyreg does not draw the coefficients.
    let (data, imputed) = sim_full(false, 20_261_126);
    compare_stats("sim inc", &numeric_stats(&imputed, 2), &r.stats["inc"], K);
    compare_stats("sim edu", &share_stats(&imputed, 3, 3), &r.stats["edu"], K);
    compare_stats("sim smk", &share_stats(&imputed, 4, 2), &r.stats["smk"], K);
    let ours = sim_analysis(&imputed, &data);
    let theirs = reference_pooled(&r.analysis);
    for t in 0..2 {
        compare_pooled(&format!("sim coefficient {t}"), &ours[t], &theirs[t], K);
    }
}

#[test]
fn proper_polyreg_keeps_the_mean_and_widens_the_spread() {
    let r = reference().sim;
    let (_, improper) = sim_full(false, 20_261_126);
    let (_, proper) = sim_full(true, 20_261_129);
    compare_stats("proper edu", &share_stats(&proper, 3, 3), &r.stats["edu"], K);
    let spread = |imputed: &Imputed| {
        let stats = share_stats(imputed, 3, 3);
        (0..2)
            .map(|c| variance(&column_of(&stats, c)))
            .collect::<Vec<f64>>()
    };
    let (vi, vp) = (spread(&improper), spread(&proper));
    eprintln!("between-set variance of the level shares: improper {vi:?} proper {vp:?}");
    // A sample variance of m normal values has relative standard error
    // sqrt(2 / (m - 1)), so the ratio of two has sqrt(4 / (m - 1)); a ratio
    // above 1 plus four of those is a real excess.
    let bound = 1.0 + 4.0 * (4.0 / (imputed_sets(&proper) as f64 - 1.0)).sqrt();
    for c in 0..2 {
        assert!(vp[c] > vi[c] * bound, "level {c}: {} against {}", vp[c], vi[c]);
    }
}

#[test]
fn simulated_norm_and_pmm_of_a_single_column_match_mice() {
    let r = reference().sim;
    let data = sim_columns(&r, [true, false, false]);
    for (label, method, block, seed) in [
        ("norm", Method::Norm, &r.norm, 20_261_127),
        ("pmm", Method::pmm(), &r.pmm, 20_261_128),
    ] {
        let spec = spec_of(&data, vec![None, None, Some(method), None, None, None]);
        let imputed = impute(&data, &spec, &options(seed)).unwrap();
        compare_stats(&format!("sim {label} inc"), &numeric_stats(&imputed, 2), &block.stats["inc"], K);
        let ours = sim_analysis(&imputed, &data);
        let theirs = reference_pooled(&block.analysis);
        for t in 0..2 {
            compare_pooled(&format!("sim {label} coefficient {t}"), &ours[t], &theirs[t], K);
        }
    }
}
