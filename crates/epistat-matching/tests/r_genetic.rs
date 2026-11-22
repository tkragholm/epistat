//! Genetic matching checked against R 4.6's `MatchIt` 4.8.2
//! (`method = "genetic"`), `Matching` 4.10-15 (`Match`, `GenMatch`) and
//! `rgenoud`.
//!
//! The reference is `tests/data/matchit_genetic.json`, written by
//! `tests/data/matchit_genetic.R`. R's search cannot be reproduced, so the
//! deterministic parts are checked with the weights fixed, and the search by
//! its results.
//!
//! * **Pairs for given weights.** For 28 runs on `lalonde` (with jittered
//!   continuous covariates, so that no two controls tie) and a simulated
//!   cohort, the crate's pairs equal the pairs of `matchit(method =
//!   "genetic")` with the same weights, unit by unit. The order of the
//!   controls of one focal unit is not compared: the crate gives them nearest
//!   first, MatchIt in an order of its own. The runs
//!   cover ratios 1 to 3, with and without replacement, the orders `largest`,
//!   `smallest` and `data`, calipers on a covariate and on the distance,
//!   `exact` and `antiexact`. MatchIt's weights are fixed by giving rgenoud
//!   a range of relative width 2e-9 around them; the script requires that its
//!   pairs equal those of `Match(Weight.matrix = diag(w))`, and the crate is
//!   given the weights `GenMatch` reports.
//! * **The loss.** For the same weights the crate's loss equals `GenMatch`'s
//!   value for the default fitness (sorted p-values of paired t-tests and
//!   Kolmogorov-Smirnov tests), the minimum p-value, no Kolmogorov-Smirnov
//!   test, unpaired t-tests, and the quantile summaries `qqmean.mean`,
//!   `qqmean.max`, `qqmax.mean`, `qqmax.max`, `qqmedian.median` and
//!   `qqmedian.max`. R rejects the two `qqmax` names as invalid; their
//!   reference is `Matching::qqstats` summarised as GenMatch's code does.
//!   The bound is 1e-12: the exact Kolmogorov-Smirnov probability is a
//!   recursion of at most 400 convex combinations, each rounding at most
//!   1.1e-16, so it is within 4.4e-14. The t p-values are within a few
//!   rounding errors of 1 on each side, and the quantile summaries are means
//!   of at most 400 ratios of integers. Where a Kolmogorov-Smirnov p-value is
//!   asymptotic the bound is 4e-5, because R's is off by that much, see
//!   below.
//! * **The Kolmogorov-Smirnov test.** 60 pairs of samples with and without
//!   ties, sizes from 3 to 185, give the same statistic and p-value as
//!   `ks.test`, exact below a product of sizes of 10000 and asymptotic from
//!   there.
//! * **A defect in R's asymptotic p-value.** `ks.test` passes `tol = 1e-6` to
//!   the series for the limiting distribution, but for a scaled statistic
//!   below 1 it sums one term of the theta-function form and stops. The
//!   omitted term is `sqrt(2 pi) / x * exp(-9 pi^2 / (8 x^2))`, which is
//!   1.6e-5 at 0.96 and tends to 3.8e-5 at 1. `limit` in the reference holds
//!   R's value and the alternating series with 200 terms, which is exact to
//!   rounding; the test asserts R's error, which is the omitted term to
//!   within 1e-9, and the crate's agreement with the series to 1e-14. The
//!   crate sums the series to rounding.
//! * **The bootstrap p-value.** `GenMatch` with `nboots = 500` draws from R's
//!   stream. Thirty evaluations in R and thirty seeds here give sorted
//!   p-value vectors whose entries have means within 5 combined standard
//!   errors.
//! * **The search.** `GenMatch` (rgenoud) and the crate's search get the
//!   same budget on two data sets, and the crate's smallest balance p-value
//!   over twelve seeds is not worse than R's by more than two standard errors,
//!   and better than the unweighted start. The distributions are printed
//!   (`cargo test -- --nocapture`).

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_matching::{
    BalanceMeasure, GeneticFit, GeneticMatchOptions, GeneticProblem, GeneticSearch, LossRule,
    ks_limit_upper, ks_two_sample,
};
use epistat_testkit::{POOLS, assert_same_on_pools, assert_within_standard_errors};
use r_reference::reference as load_reference;
use serde::{Deserialize, Deserializer};
use std::collections::BTreeMap;

fn one_or_many<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Shape<T> {
        One(T),
        Many(Vec<T>),
    }
    Ok(match Shape::deserialize(deserializer)? {
        Shape::One(value) => vec![value],
        Shape::Many(values) => values,
    })
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    limit: Vec<LimitCase>,
    scenarios: Vec<Scenario>,
    runs: Vec<Run>,
    boot: Boot,
    ks: Vec<KsCase>,
    search: Vec<SearchRef>,
    budget: Budget,
}

#[derive(Deserialize)]
struct LimitCase {
    x: f64,
    r: f64,
    series: f64,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
    matching: String,
    rgenoud: String,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    cov: Vec<String>,
    exact: String,
    treat: Vec<u8>,
    ps: Vec<f64>,
    g: Vec<String>,
    columns: BTreeMap<String, Vec<f64>>,
}

#[derive(Deserialize)]
struct RunSpec {
    ratio: usize,
    replace: bool,
    exact: bool,
    anti: bool,
}

#[derive(Deserialize)]
struct CalCov {
    name: String,
    value: f64,
}

#[derive(Deserialize)]
struct Loss {
    fit: String,
    #[serde(deserialize_with = "one_or_many")]
    value: Vec<f64>,
}

#[derive(Deserialize)]
struct Run {
    scenario: String,
    #[serde(rename = "run")]
    spec: RunSpec,
    cal_cov: Option<CalCov>,
    cal_ps: Option<f64>,
    weights: Vec<f64>,
    order: Vec<usize>,
    #[serde(deserialize_with = "one_or_many")]
    treated: Vec<usize>,
    matchit: Vec<MatchRow>,
    losses: Vec<Loss>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MatchRow {
    One(usize),
    Many(Vec<usize>),
}

impl MatchRow {
    fn controls(&self) -> Vec<usize> {
        match self {
            Self::One(value) => vec![*value],
            Self::Many(values) => values.clone(),
        }
    }
}

#[derive(Deserialize)]
struct Boot {
    run: Vec<f64>,
    sorted: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct KsCase {
    #[serde(deserialize_with = "one_or_many")]
    x: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    y: Vec<f64>,
    p: f64,
    d: f64,
}

#[derive(Deserialize)]
struct SearchRef {
    scenario: String,
    #[serde(deserialize_with = "one_or_many")]
    ones: Vec<f64>,
    values: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Budget {
    pop: usize,
    generations: usize,
}

fn reference() -> Reference {
    load_reference("matchit_genetic.json")
}

/// The matching variables of a run, as the script builds them: the
/// covariates, the score, and for `exact` the factor codes of its variable.
fn variables(scenario: &Scenario, exact: bool) -> Vec<Vec<f64>> {
    let mut columns: Vec<Vec<f64>> = scenario
        .cov
        .iter()
        .map(|name| scenario.columns[name].clone())
        .collect();
    columns.push(scenario.ps.clone());
    if exact {
        let values = &scenario.columns[&scenario.exact];
        let mut levels: Vec<f64> = values.clone();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        columns.push(
            values
                .iter()
                .map(|v| levels.iter().position(|l| l.total_cmp(v).is_eq()).unwrap() as f64 + 1.0)
                .collect(),
        );
    }
    columns
}

fn balance_columns(scenario: &Scenario) -> Vec<Vec<f64>> {
    scenario
        .cov
        .iter()
        .map(|name| scenario.columns[name].clone())
        .collect()
}

fn problem(scenario: &Scenario, run: &Run, order: &[usize]) -> GeneticProblem {
    let vars = variables(scenario, run.spec.exact);
    let p = vars.len();
    let mut calipers = vec![None; p];
    if let Some(cal) = &run.cal_cov {
        let at = scenario.cov.iter().position(|c| *c == cal.name).unwrap();
        calipers[at] = Some(cal.value);
    }
    if let Some(width) = run.cal_ps {
        calipers[scenario.cov.len()] = Some(width);
    }
    let mut exact = vec![false; p];
    if run.spec.exact {
        exact[p - 1] = true;
    }
    let focal: Vec<bool> = scenario.treat.iter().map(|&t| t == 1).collect();
    let mut forbidden = Vec::new();
    if run.spec.anti {
        for (i, &fi) in focal.iter().enumerate() {
            if !fi {
                continue;
            }
            for (j, &fj) in focal.iter().enumerate() {
                if !fj && scenario.g[i] == scenario.g[j] {
                    forbidden.push((i, j));
                }
            }
        }
    }
    GeneticProblem::new(
        &focal,
        &vars,
        &balance_columns(scenario),
        order,
        GeneticMatchOptions {
            ratio: run.spec.ratio,
            replace: run.spec.replace,
            calipers,
            exact,
            forbidden,
        },
        f64::EPSILON.sqrt(),
    )
    .unwrap()
}

fn scenario<'a>(reference: &'a Reference, name: &str) -> &'a Scenario {
    reference
        .scenarios
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("scenario {name}"))
}

fn fit_of(name: &str) -> GeneticFit {
    let base = GeneticFit::default();
    match name {
        "pvals" => base,
        "pvals_minimum" => GeneticFit {
            loss: LossRule::Minimum,
            ..base
        },
        "pvals_no_ks" => GeneticFit { ks: false, ..base },
        "pvals_unpaired" => GeneticFit {
            paired: false,
            ..base
        },
        "qqmean.mean" => measure(BalanceMeasure::QqMeanMean),
        "qqmean.max" => measure(BalanceMeasure::QqMeanMax),
        "qqmax.mean" => measure(BalanceMeasure::QqMaxMean),
        "qqmax.max" => measure(BalanceMeasure::QqMaxMax),
        "qqmedian.median" => measure(BalanceMeasure::QqMedianMedian),
        "qqmedian.max" => measure(BalanceMeasure::QqMedianMax),
        other => panic!("fit {other}"),
    }
}

fn measure(measure: BalanceMeasure) -> GeneticFit {
    GeneticFit {
        measure,
        ..GeneticFit::default()
    }
}

/// See the module documentation for the bound.
const LOSS_BOUND: f64 = 1e-12;

/// The largest error of R's asymptotic Kolmogorov-Smirnov p-value, the
/// omitted term of the theta series at a scaled statistic of 1.
fn r_limit_error_bound() -> f64 {
    (2.0 * std::f64::consts::PI).sqrt() * (-9.0 * std::f64::consts::PI.powi(2) / 8.0).exp()
}

/// Whether a fit's p-values include an asymptotic Kolmogorov-Smirnov test for
/// `rows` matched rows.
fn has_asymptotic_ks(fit: &str, rows: usize) -> bool {
    let uses_ks = fit == "pvals" || fit == "pvals_minimum" || fit == "pvals_unpaired";
    uses_ks && rows * rows >= 10_000
}

#[test]
fn pairs_for_given_weights_equal_matchit() {
    let reference = reference();
    println!(
        "{}, MatchIt {}, Matching {}, rgenoud {}",
        reference.versions.r,
        reference.versions.matchit,
        reference.versions.matching,
        reference.versions.rgenoud
    );
    let mut compared = 0;
    for (k, run) in reference.runs.iter().enumerate() {
        let sc = scenario(&reference, &run.scenario);
        let order: Vec<usize> = run.order.iter().map(|i| i - 1).collect();
        let problem = problem(sc, run, &order);
        let matches = problem.match_pairs(&run.weights).unwrap();
        let mut ours: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (f, c) in matches.focal.iter().zip(&matches.control) {
            ours.entry(f + 1).or_default().push(c + 1);
        }
        for (treated, row) in run.treated.iter().zip(&run.matchit) {
            let mut expected = row.controls();
            expected.sort_unstable();
            let mut got = ours.remove(treated).unwrap_or_default();
            got.sort_unstable();
            assert_eq!(
                got, expected,
                "run {k} ({}), treated unit {treated}",
                run.scenario
            );
            compared += expected.len();
        }
        assert!(ours.is_empty(), "run {k}: pairs for units MatchIt left out");
        // Weights are the reciprocal of the controls of a focal unit.
        for w in &matches.weight {
            assert!(*w > 0.0 && *w <= 1.0);
        }
    }
    println!("{} runs, {compared} pairs", reference.runs.len());
}

#[test]
fn loss_for_given_weights_equals_genmatch() {
    let reference = reference();
    let mut worst: f64 = 0.0;
    let mut checked = 0;
    for (k, run) in reference.runs.iter().enumerate() {
        let sc = scenario(&reference, &run.scenario);
        let order: Vec<usize> = run.order.iter().map(|i| i - 1).collect();
        let problem = problem(sc, run, &order);
        let rows = problem.match_pairs(&run.weights).unwrap().len();
        for loss in &run.losses {
            let got = problem.loss(&run.weights, &fit_of(&loss.fit), 0).unwrap();
            let bound = if has_asymptotic_ks(&loss.fit, rows) {
                r_limit_error_bound() + LOSS_BOUND
            } else {
                LOSS_BOUND
            };
            assert_eq!(
                got.values().len(),
                loss.value.len(),
                "run {k} {}: length",
                loss.fit
            );
            for (g, e) in got.values().iter().zip(&loss.value) {
                let gap = (g - e).abs();
                worst = worst.max(gap);
                assert!(
                    gap <= bound,
                    "run {k} ({}) {}: {g} against {e}",
                    run.scenario,
                    loss.fit
                );
            }
            checked += 1;
        }
    }
    println!(
        "{checked} losses, worst deviation {worst:e}, bound {LOSS_BOUND:e} (4e-5 with an asymptotic test)"
    );
}

#[test]
fn kolmogorov_smirnov_equals_ks_test() {
    let reference = reference();
    let mut worst_exact: f64 = 0.0;
    let mut worst_asymptotic: f64 = 0.0;
    let bound = r_limit_error_bound() + LOSS_BOUND;
    for (k, case) in reference.ks.iter().enumerate() {
        let test = ks_two_sample(&case.x, &case.y).unwrap();
        let gap = (test.p_value - case.p).abs();
        assert!(
            (test.statistic - case.d).abs() <= 1e-15,
            "case {k}: statistic {} against {}",
            test.statistic,
            case.d
        );
        assert_eq!(
            test.exact,
            case.x.len() * case.y.len() < 10_000,
            "case {k}: exact or asymptotic"
        );
        if test.exact {
            worst_exact = worst_exact.max(gap);
            assert!(
                gap <= LOSS_BOUND,
                "case {k}: p {} against {}",
                test.p_value,
                case.p
            );
        } else {
            worst_asymptotic = worst_asymptotic.max(gap);
            assert!(
                gap <= bound,
                "case {k}: p {} against {}",
                test.p_value,
                case.p
            );
        }
    }
    println!(
        "{} cases, worst deviation {worst_exact:e} exact (bound {LOSS_BOUND:e}), \
         {worst_asymptotic:e} asymptotic (bound {bound:e})",
        reference.ks.len()
    );
}

#[test]
fn r_asymptotic_p_value_is_off_below_one() {
    let reference = reference();
    let mut worst_r: f64 = 0.0;
    for case in &reference.limit {
        // The crate: the series to rounding.
        let ours = ks_limit_upper(case.x);
        assert!(
            (ours - case.series).abs() <= 1e-14,
            "x {}: {ours} against the series {}",
            case.x,
            case.series
        );
        // R: the omitted term of the theta series below 1, nothing above.
        let r_error = case.r - case.series;
        worst_r = worst_r.max(r_error.abs());
        if case.x < 1.0 {
            let omitted = (2.0 * std::f64::consts::PI).sqrt() / case.x
                * (-9.0 * std::f64::consts::PI.powi(2) / (8.0 * case.x * case.x)).exp();
            assert!(
                (r_error - omitted).abs() <= 1e-9,
                "x {}: R's error {r_error:e}, omitted term {omitted:e}",
                case.x
            );
        } else {
            assert!(
                r_error.abs() <= 1e-13,
                "x {}: R's error {r_error:e}",
                case.x
            );
        }
    }
    assert!(
        worst_r > 3.0e-5,
        "the reference should reach the defect: {worst_r:e}"
    );
    println!("R's asymptotic p-value is off by up to {worst_r:e}");
}

#[test]
fn bootstrap_p_values_agree_in_distribution() {
    let reference = reference();
    let run = &reference.runs[0];
    let sc = scenario(&reference, &run.scenario);
    let order: Vec<usize> = run.order.iter().map(|i| i - 1).collect();
    let problem = problem(sc, run, &order);
    let fit = GeneticFit {
        bootstrap: 500,
        ..GeneticFit::default()
    };
    let ours: Vec<Vec<f64>> = (0..30)
        .map(|seed| {
            problem
                .loss(&reference.boot.run, &fit, 7000 + seed)
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    let theirs = &reference.boot.sorted;
    let entries = ours[0].len();
    assert_eq!(entries, theirs[0].len());
    let stats = |rows: &[Vec<f64>], i: usize| {
        let n = rows.len() as f64;
        let mean = rows.iter().map(|r| r[i]).sum::<f64>() / n;
        let var = rows.iter().map(|r| (r[i] - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (var / n).sqrt())
    };
    for i in 0..entries {
        let (a, se_a) = stats(&ours, i);
        let (b, se_b) = stats(theirs, i);
        println!("entry {i}: ours {a:.4} (se {se_a:.4}), R {b:.4} (se {se_b:.4})");
        if se_a < 1e-12 && se_b < 1e-12 {
            // A p-value of the t-test, the same on every draw. R computes
            // 1 - pt(), which is exactly 0 where the tail is below 1e-16.
            assert!((a - b).abs() <= 1e-12, "entry {i}: {a} against {b}");
        } else {
            assert_within_standard_errors(a, se_a, b, se_b, 5.0);
        }
    }
}

#[test]
fn search_is_as_good_as_genmatch() {
    let reference = reference();
    for sr in &reference.search {
        let sc = scenario(&reference, &sr.scenario);
        // The same inputs as the script's `genetic_inputs` for the default
        // order: units by decreasing score.
        let mut order: Vec<usize> = (0..sc.ps.len()).collect();
        order.sort_by(|&a, &b| sc.ps[b].total_cmp(&sc.ps[a]).then(a.cmp(&b)));
        let run = Run {
            scenario: sc.name.clone(),
            spec: RunSpec {
                ratio: 1,
                replace: false,
                exact: false,
                anti: false,
            },
            cal_cov: None,
            cal_ps: None,
            weights: Vec::new(),
            order: Vec::new(),
            treated: Vec::new(),
            matchit: Vec::new(),
            losses: Vec::new(),
        };
        let problem = problem(sc, &run, &order);
        let fit = GeneticFit::default();
        let mut ours = Vec::new();
        for seed in 1..=12_u64 {
            let search = GeneticSearch {
                population: reference.budget.pop,
                max_generations: reference.budget.generations,
                wait_generations: reference.budget.generations,
                hard_generation_limit: true,
                seed,
                ..GeneticSearch::default()
            };
            let outcome = problem.search(&fit, &search).unwrap();
            ours.push(outcome.loss.values().to_vec());
        }
        let start = problem
            .loss(&vec![1.0; problem.variables()], &fit, 0)
            .unwrap();
        assert!(
            (start.values()[0] - sr.ones[0]).abs() <= LOSS_BOUND,
            "the unweighted start differs from GenMatch's"
        );
        let stats = |rows: &[Vec<f64>], i: usize| {
            let n = rows.len() as f64;
            let mean = rows.iter().map(|r| r[i]).sum::<f64>() / n;
            let var = rows.iter().map(|r| (r[i] - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, (var / n).sqrt())
        };
        // Compared as the fitness compares: from the smallest p-value up, the
        // first entry whose means differ by more than two combined standard
        // errors decides, and ours must be the larger.
        let entries = ours[0].len();
        assert_eq!(entries, sr.values[0].len());
        let mut verdict = "no difference beyond noise".to_string();
        for i in 0..entries {
            let (a, se_a) = stats(&ours, i);
            let (b, se_b) = stats(&sr.values, i);
            let gap = a - b;
            if gap.abs() > 2.0 * se_a.hypot(se_b) && gap.abs() > LOSS_BOUND {
                verdict = format!(
                    "entry {i}: ours {a:.4} (se {se_a:.4}), GenMatch {b:.4} (se {se_b:.4})"
                );
                assert!(gap > 0.0, "{}: {verdict}", sr.scenario);
                break;
            }
        }
        let means = |rows: &[Vec<f64>]| -> Vec<f64> {
            (0..entries)
                .map(|i| (stats(rows, i).0 * 1e4).round() / 1e4)
                .collect()
        };
        println!(
            "{}: unweighted start {:?}\n  ours      {:?}\n  GenMatch  {:?}\n  {verdict}",
            sr.scenario,
            sr.ones
                .iter()
                .map(|x| (x * 1e4).round() / 1e4)
                .collect::<Vec<_>>(),
            means(&ours),
            means(&sr.values)
        );
        let (start_first, _) = (sr.ones[0], 0);
        let (mean_first, _) = stats(&ours, 0);
        assert!(
            mean_first >= start_first,
            "the search does not improve on the start"
        );
    }
}

#[test]
fn search_does_not_depend_on_threads() {
    let reference = reference();
    let sc = scenario(&reference, "sim");
    let mut order: Vec<usize> = (0..sc.ps.len()).collect();
    order.sort_by(|&a, &b| sc.ps[b].total_cmp(&sc.ps[a]).then(a.cmp(&b)));
    let run = Run {
        scenario: "sim".into(),
        spec: RunSpec {
            ratio: 1,
            replace: true,
            exact: false,
            anti: false,
        },
        cal_cov: None,
        cal_ps: None,
        weights: Vec::new(),
        order: Vec::new(),
        treated: Vec::new(),
        matchit: Vec::new(),
        losses: Vec::new(),
    };
    let problem = problem(sc, &run, &order);
    let fit = GeneticFit {
        bootstrap: 20,
        ..GeneticFit::default()
    };
    assert_same_on_pools(&POOLS, || {
        let outcome = problem
            .search(
                &fit,
                &GeneticSearch {
                    population: 20,
                    max_generations: 4,
                    wait_generations: 4,
                    hard_generation_limit: true,
                    seed: 99,
                    ..GeneticSearch::default()
                },
            )
            .unwrap();
        let mut out = outcome.weights.clone();
        out.extend_from_slice(outcome.loss.values());
        out.extend(outcome.matches.control.iter().map(|&c| c as f64));
        out.push(outcome.evaluations as f64);
        out
    });
}
