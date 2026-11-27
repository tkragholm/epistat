//! The interaction-weighted event study of Sun and Abraham, checked against
//! `fixest::feols(y ~ sunab(g, t) | id + t)` and `fixest::aggregate` 0.14.2
//! (`tests/data/generate_sunab_reference.R`), and against ATT(g,t)
//! aggregated by treated counts.

#[path = "common/feols_bound.rs"]
#[expect(dead_code, reason = "the iid covariance is not used here")]
mod feols_bound;

use epistat_core::InputError;
use epistat_did::att_gt;
use epistat_did::{
    StaggeredPanel, SunabBins, SunabControl, SunabOptions, SunabReference, SunabTerm, sunab,
};
use epistat_glm::{Absorb, Ssc};
use epistat_inference::{ClusterRule, Clusters};
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use feols_bound::{EPS, Meat, feols_bound};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Bins {
    kind: String,
    lo: Option<i64>,
    hi: Option<i64>,
    width: Option<i64>,
}

#[derive(Deserialize)]
struct Spec {
    control: String,
    cohorts: Option<Vec<i64>>,
    references: Vec<Value>,
    bins: Option<Bins>,
}

#[derive(Deserialize)]
struct Table {
    names: Vec<String>,
    estimate: Vec<f64>,
    se: Vec<f64>,
}

#[derive(Deserialize)]
struct Exact {
    error: f64,
    lambda_plus: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    spec: Spec,
    unit: Vec<u32>,
    time: Vec<i64>,
    first_treated: Vec<Option<i64>>,
    outcome: Vec<f64>,
    weights: Option<Vec<f64>>,
    covariate: Option<Vec<f64>>,
    clusters: Option<Vec<u32>>,
    names: Vec<String>,
    coef: Vec<f64>,
    vcov: Vec<f64>,
    vcov_unadjusted: Vec<f64>,
    df_k: usize,
    nobs: usize,
    shares: Vec<f64>,
    collinear: Vec<String>,
    period: Table,
    att: Table,
    cohort: Table,
    exact: Exact,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/sunab_ref.json"
    ))
}

const TIGHT: Absorb = Absorb {
    tolerance: 1e-13,
    max_iterations: 100_000,
    ..Absorb::new()
};

struct Setting {
    references: Vec<SunabReference>,
    bins: SunabBins,
    cohorts: Vec<i64>,
}

fn setting(case: &Case) -> Setting {
    let references = case
        .spec
        .references
        .iter()
        .map(|v| match v {
            Value::Number(n) => SunabReference::Period(n.as_i64().unwrap()),
            Value::String(s) if s == "first" => SunabReference::First,
            Value::String(s) if s == "last" => SunabReference::Last,
            Value::String(s) => SunabReference::Period(s.parse().unwrap()),
            other => panic!("reference {other}"),
        })
        .collect();
    let bins = match &case.spec.bins {
        None => SunabBins::None,
        Some(b) if b.kind == "clamp" => SunabBins::Clamp {
            lo: b.lo.unwrap(),
            hi: b.hi.unwrap(),
        },
        Some(b) => SunabBins::Width(b.width.unwrap()),
    };
    Setting {
        references,
        bins,
        cohorts: case.spec.cohorts.clone().unwrap_or_default(),
    }
}

fn config(case: &Case, s: &Setting) -> SunabOptions {
    SunabOptions {
        control: match case.spec.control.as_str() {
            "never" => SunabControl::NeverTreated,
            "last" => SunabControl::LastTreated,
            _ => SunabControl::Cohorts(s.cohorts.clone()),
        },
        references: s.references.clone(),
        bins: s.bins,
        absorb: TIGHT,
        ssc: Ssc::new(),
        ..SunabOptions::default()
    }
}

fn covariate(case: &Case) -> Option<Mat<f64>> {
    case.covariate
        .as_ref()
        .map(|x| Mat::from_fn(x.len(), 1, |i, _| x[i]))
}

fn panel<'a>(case: &'a Case, x: Option<&'a Mat<f64>>) -> StaggeredPanel<'a> {
    let mut panel =
        StaggeredPanel::builder(&case.unit, &case.time, &case.first_treated, &case.outcome);
    if let Some(w) = &case.weights {
        panel = panel.weights(w);
    }
    if let Some(x) = x {
        panel = panel.covariates(x.as_ref());
    }
    if let Some(c) = &case.clusters {
        panel = panel.clusters(c);
    }
    panel.build().expect("a valid panel")
}

/// The position in `table` of each name, which must all be there.
fn positions(table: &Table, names: &[String], case: &str) -> Vec<usize> {
    assert_eq!(table.names.len(), names.len(), "{case}: {:?}", table.names);
    names
        .iter()
        .map(|n| table.names.iter().position(|m| m == n).unwrap())
        .collect()
}

#[test]
fn sunab_matches_fixest() {
    for case in &reference().cases {
        let s = setting(case);
        let x = covariate(case);
        let result = sunab::fit(&panel(case, x.as_ref()), &config(case, &s)).unwrap();
        assert_eq!(result.n_rows, case.nobs, "{}", case.name);
        assert_eq!(result.df_k, case.df_k, "{}", case.name);
        assert!(case.collinear.is_empty(), "{}", case.name);
        // The cells in fixest's order and names.
        let mut labels: Vec<String> = result
            .cells
            .iter()
            .map(|c| format!("t::{}:cohort::{}", c.relative_period, c.cohort))
            .collect();
        if case.covariate.is_some() {
            labels.push("x1".to_string());
        }
        assert_eq!(labels, case.names, "{}", case.name);
        let p = labels.len();
        let k_cells = result.cells.len();

        // The design as the module documentation defines it, fitted plainly.
        // The last cohort of the panel without never-treated units.
        let cutoff = (case.spec.control == "last").then_some(7);
        let n = case.unit.len();
        let design = Mat::from_fn(n, p, |i, j| {
            if j >= k_cells {
                return x.as_ref().unwrap()[(i, 0)];
            }
            let Some(g) = case.first_treated[i] else {
                return 0.0;
            };
            let cell = &result.cells[j];
            let e = case.time[i] - g;
            let referenced = s.references.iter().any(|r| match *r {
                SunabReference::Period(q) => q == e,
                _ => e == result_first(case, cutoff),
            });
            if g != cell.cohort || referenced || cutoff.is_some_and(|c| case.time[i] >= c) {
                return 0.0;
            }
            let label = match s.bins {
                SunabBins::None => e,
                SunabBins::Clamp { lo, hi } => e.clamp(lo, hi),
                SunabBins::Width(w) => e.div_euclid(w) * w,
                _ => unreachable!("the bins of the reference cases"),
            };
            f64::from(u8::from(label == cell.relative_period))
        });
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if cutoff.is_some_and(|c| case.time[i] >= c) {
                    0.0
                } else {
                    case.weights.as_ref().map_or(1.0, |w| w[i])
                }
            })
            .collect();
        let t_min = *case.time.iter().min().unwrap();
        let time: Vec<u32> = case
            .time
            .iter()
            .map(|&t| u32::try_from(t - t_min).unwrap())
            .collect();
        let clusters = case.clusters.as_deref().unwrap_or(&case.unit);
        let g_count = result.n_clusters as f64;
        let nobs = case.nobs as f64;
        let factor = g_count / (g_count - 1.0) * (nobs - 1.0) / (nobs - case.df_k as f64);
        let bound = feols_bound(
            &[&case.unit, &time],
            Some(&weights),
            &design,
            &case.outcome,
            Meat::Cluster(clusters),
            TIGHT,
            case.exact.lambda_plus,
            case.exact.error,
            factor,
        );
        // The estimator is the plain fit.
        for (j, c) in bound.coefficients.iter().enumerate() {
            let mine = result
                .cells
                .get(j)
                .map_or_else(
                    || result.covariate_coefficients[j - k_cells],
                    |c| c.estimate,
                )
                .map(|e| e.estimate);
            assert_eq!(c.map(f64::to_bits), mine.map(f64::to_bits), "{}", case.name);
        }
        assert_eq!(bound.kept.len(), p, "{}", case.name);
        for j in 0..p {
            let got = bound.coefficients[j].unwrap();
            assert!(
                (got - case.coef[j]).abs() <= bound.beta,
                "{} {}: {got} {}",
                case.name,
                labels[j],
                case.coef[j]
            );
            for k in 0..p {
                let got = result.covariance.get(j, k).unwrap();
                let want = case.vcov[j * p + k];
                assert!(
                    (got - want).abs() <= bound.covariance[j * p + k],
                    "{} vcov {} {}: {got:e} {want:e}",
                    case.name,
                    labels[j],
                    labels[k]
                );
            }
        }
        // The shares: sums of the same weights over the same rows, in
        // different orders.
        let total_weight: f64 = weights.iter().sum();
        for (j, cell) in result.cells.iter().enumerate() {
            assert!(
                (cell.weight - case.shares[j]).abs() <= 2.0 * n as f64 * EPS * total_weight,
                "{} share {}",
                case.name,
                labels[j]
            );
        }

        // The influence functions give fixest's covariance without its
        // small-sample factors.
        let influence = result.influence.as_ref().unwrap();
        let unit_cluster: Vec<u32> = influence
            .units()
            .iter()
            .map(|&u| {
                let row = case.unit.iter().position(|&v| v == u).unwrap();
                clusters[row]
            })
            .collect();
        let of = Clusters::from_labels(&unit_cluster);
        assert_eq!(influence.n_columns(), p);
        let raw = influence
            .variance_with(Some(&of), ClusterRule::NoSmallSample)
            .unwrap();
        for j in 0..p {
            for k in 0..p {
                let want = case.vcov_unadjusted[j * p + k];
                assert!(
                    (raw.get(j, k).unwrap() - want).abs() <= bound.covariance[j * p + k] / factor,
                    "{} influence {} {}",
                    case.name,
                    labels[j],
                    labels[k]
                );
            }
        }

        // The summaries. Each is a mean of cells with shares that sum to one,
        // so its error is at most the cells' bound; its variance s' V s is
        // off by at most s' |dV| s.
        let check = |term: SunabTerm, table: &Table, row: usize| {
            let a = result
                .aggregates
                .iter()
                .position(|e| e.term == term)
                .unwrap();
            let est = &result.aggregates[a].estimate;
            assert!(
                (est.estimate - table.estimate[row]).abs()
                    <= bound.beta + 16.0 * EPS * est.estimate.abs(),
                "{} {term:?}: {} {}",
                case.name,
                est.estimate,
                table.estimate[row]
            );
            let members: Vec<(usize, f64)> = result
                .cells
                .iter()
                .enumerate()
                .filter(|(_, c)| match term {
                    SunabTerm::Period(e) => c.relative_period == e,
                    SunabTerm::Cohort(g) => c.cohort == g && c.relative_period >= 0,
                    SunabTerm::Att => c.relative_period >= 0,
                    _ => unreachable!("the terms of the module"),
                })
                .map(|(j, c)| (j, c.weight))
                .collect();
            let total: f64 = members.iter().map(|m| m.1).sum();
            let mut tol = 0.0;
            for &(i, wi) in &members {
                for &(j, wj) in &members {
                    tol += wi / total * wj / total * bound.covariance[i * p + j];
                }
            }
            let se = est.se.unwrap();
            let want = table.se[row];
            assert!(
                (se * se - want * want).abs() <= tol + 16.0 * EPS * want * want,
                "{} {term:?} se: {se} {want}",
                case.name
            );
            // The influence function of the summary gives its variance
            // without the small-sample factors, up to the rounding of sums
            // over the units in another order.
            let v = result
                .aggregate_influence
                .as_ref()
                .unwrap()
                .variance_with(Some(&of), ClusterRule::NoSmallSample)
                .unwrap()
                .get(a, a)
                .unwrap();
            let units = result.n_units as f64;
            assert!(
                (v * factor - se * se).abs() <= 64.0 * units * EPS * se * se,
                "{} {term:?} influence {} {}",
                case.name,
                v * factor,
                se * se
            );
        };
        let periods: Vec<i64> = result
            .aggregates
            .iter()
            .filter_map(|e| match e.term {
                SunabTerm::Period(r) => Some(r),
                _ => None,
            })
            .collect();
        let period_names: Vec<String> = periods.iter().map(|r| format!("t::{r}")).collect();
        for (row, &r) in positions(&case.period, &period_names, &case.name)
            .iter()
            .zip(&periods)
        {
            check(SunabTerm::Period(r), &case.period, *row);
        }
        let cohorts: Vec<i64> = result
            .aggregates
            .iter()
            .filter_map(|e| match e.term {
                SunabTerm::Cohort(g) => Some(g),
                _ => None,
            })
            .collect();
        let cohort_names: Vec<String> = cohorts.iter().map(|g| format!("cohort::{g}")).collect();
        for (row, &g) in positions(&case.cohort, &cohort_names, &case.name)
            .iter()
            .zip(&cohorts)
        {
            check(SunabTerm::Cohort(g), &case.cohort, *row);
        }
        check(SunabTerm::Att, &case.att, 0);
        // The covariance of the summaries has their variances on its
        // diagonal.
        let m = result.aggregates.len();
        for a in 0..m {
            let se = result.aggregates[a].estimate.se.unwrap();
            assert_eq!(
                result
                    .aggregate_covariance
                    .get(a, a)
                    .unwrap()
                    .sqrt()
                    .to_bits(),
                se.to_bits()
            );
        }
    }
}

/// `.F` of `sunab`: the smallest relative period over the rows fitted, with
/// -1 for the never treated.
fn result_first(case: &Case, cutoff: Option<i64>) -> i64 {
    (0..case.unit.len())
        .filter(|&i| cutoff.is_none_or(|c| case.time[i] < c))
        .map(|i| case.first_treated[i].map_or(-1, |g| case.time[i] - g))
        .min()
        .unwrap()
}

/// With the never treated as controls, a balanced panel and no covariates,
/// each cell is the ATT(g, g + e) of Callaway and Sant'Anna with the
/// universal base period, so the summaries by relative period and overall are
/// `aggte`'s dynamic and simple summaries, whose weights are the cohorts'
/// counts of units.
#[test]
fn sunab_is_att_gt_aggregated_by_treated_counts() {
    let reference = reference();
    let case = &reference.cases[0];
    // A balanced panel from the first case: the units seen in every period.
    let periods = 9;
    let mut count = vec![0usize; 200];
    for &u in &case.unit {
        count[u as usize] += 1;
    }
    let rows: Vec<usize> = (0..case.unit.len())
        .filter(|&i| count[case.unit[i] as usize] == periods)
        .collect();
    let unit: Vec<u32> = rows.iter().map(|&i| case.unit[i]).collect();
    let time: Vec<i64> = rows.iter().map(|&i| case.time[i]).collect();
    let first: Vec<Option<i64>> = rows.iter().map(|&i| case.first_treated[i]).collect();
    let outcome: Vec<f64> = rows.iter().map(|&i| case.outcome[i]).collect();
    let panel = StaggeredPanel::builder(&unit, &time, &first, &outcome)
        .build()
        .unwrap();
    let result = sunab::fit(
        &panel,
        &SunabOptions {
            absorb: TIGHT,
            ..SunabOptions::default()
        },
    )
    .unwrap();

    let options = att_gt::AttGtOptions {
        base_period: att_gt::BasePeriod::Universal,
        skip_failed_cells: false,
        refuse_small_never_treated: false,
        refuse_poor_overlap: false,
        ..att_gt::AttGtOptions::default()
    };
    let fitted = att_gt::estimate(&panel, &options).unwrap();
    // The tolerance: the regression's bound from its criterion (the
    // smallest positive eigenvalue of D'D for a balanced panel of N units and
    // T < N periods is T), and the rounding of the four means of ATT(g,t),
    // each at most n eps max |y|.
    let n = unit.len();
    let t_codes: Vec<u32> = time
        .iter()
        .map(|&t| u32::try_from(t - 1).unwrap())
        .collect();
    let design = Mat::from_fn(n, result.cells.len(), |i, j| {
        let c = &result.cells[j];
        let hit = first[i] == Some(c.cohort) && time[i] - c.cohort == c.relative_period;
        f64::from(u8::from(hit))
    });
    let bound = feols_bound(
        &[&unit, &t_codes],
        None,
        &design,
        &outcome,
        Meat::Cluster(&unit),
        TIGHT,
        f64::from(u8::try_from(periods).unwrap()),
        0.0,
        2.0,
    );
    let y_max = outcome.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let scale = bound.beta + 8.0 * n as f64 * EPS * y_max;
    for cell in &result.cells {
        let att = fitted
            .cells()
            .iter()
            .find(|e| e.group == cell.cohort && e.time == cell.cohort + cell.relative_period)
            .and_then(|e| e.estimate)
            .unwrap()
            .estimate;
        assert!(
            (cell.estimate.unwrap().estimate - att).abs() <= scale,
            "cell {} {}: {} {att}",
            cell.cohort,
            cell.relative_period,
            cell.estimate.unwrap().estimate
        );
    }
    for (aggregation, term) in [
        (att_gt::AggteType::Dynamic, None),
        (att_gt::AggteType::Simple, Some(SunabTerm::Att)),
    ] {
        let agg = fitted
            .aggregate(&att_gt::AggteOptions {
                aggregation,
                ..att_gt::AggteOptions::default()
            })
            .unwrap();
        match term {
            Some(t) => {
                let mine = result.aggregates.iter().find(|e| e.term == t).unwrap();
                let overall = agg.overall.unwrap().estimate.estimate;
                assert!((mine.estimate.estimate - overall).abs() <= scale);
            }
            None => {
                for point in &agg.points {
                    let mine = result
                        .aggregates
                        .iter()
                        .find(|e| e.term == SunabTerm::Period(point.key));
                    if point.key == -1 {
                        assert!(mine.is_none());
                        continue;
                    }
                    let mine = mine.unwrap();
                    let att = point.estimate.estimate;
                    assert!(
                        (mine.estimate.estimate - att).abs() <= scale,
                        "event {}: {} {att}",
                        point.key,
                        mine.estimate.estimate
                    );
                }
            }
        }
    }
}

#[test]
fn unusable_bins_are_an_error() {
    let reference = reference();
    let case = &reference.cases[0];
    let s = setting(case);
    for bins in [SunabBins::Clamp { lo: 2, hi: 1 }, SunabBins::Width(0)] {
        let config = SunabOptions {
            bins,
            ..config(case, &s)
        };
        assert!(matches!(
            sunab::fit(&panel(case, None), &config),
            Err(epistat_did::Error::Bins)
        ));
    }
}

/// An always-treated cohort is removed, and a cohort's classification does
/// not depend on the order of the rows.
#[test]
fn always_treated_rows_are_removed() {
    let reference = reference();
    let case = &reference.cases[0];
    let s = setting(case);
    let base = sunab::fit(&panel(case, None), &config(case, &s)).unwrap();
    // Units 0 to 4 become always treated: first treated in period 1.
    let first: Vec<Option<i64>> = case
        .unit
        .iter()
        .zip(&case.first_treated)
        .map(|(&u, &g)| if u < 5 { Some(1) } else { g })
        .collect();
    let mut changed = StaggeredPanel::builder(&case.unit, &case.time, &first, &case.outcome);
    if let Some(w) = &case.weights {
        changed = changed.weights(w);
    }
    let changed = changed.build().unwrap();
    let result = sunab::fit(&changed, &config(case, &s)).unwrap();
    let removed = case.unit.iter().filter(|&&u| u < 5).count();
    assert_eq!(result.excluded_rows, removed);
    assert!(result.cells.iter().all(|c| c.cohort != 1));
    assert!(result.n_rows < base.n_rows);
    // The rows come in the panel's order, so reversed rows are refused
    // rather than classified again.
    let order: Vec<usize> = (0..case.unit.len()).rev().collect();
    let unit: Vec<u32> = order.iter().map(|&i| case.unit[i]).collect();
    let time: Vec<i64> = order.iter().map(|&i| case.time[i]).collect();
    let first_rev: Vec<Option<i64>> = order.iter().map(|&i| first[i]).collect();
    let outcome: Vec<f64> = order.iter().map(|&i| case.outcome[i]).collect();
    let reversed = StaggeredPanel::builder(&unit, &time, &first_rev, &outcome).build();
    assert!(matches!(
        reversed,
        Err(epistat_did::Error::Input(InputError::Unsorted { .. }))
    ));
}

/// The values of an influence function, column after column.
fn influence_values(
    influence: Option<&epistat_inference::Influence>,
) -> impl Iterator<Item = f64> + '_ {
    influence
        .into_iter()
        .flat_map(|f| (0..f.n_columns()).flat_map(move |j| f.column(j).iter().copied()))
}

fn sunab_bits() -> Vec<f64> {
    let reference = reference();
    let mut out = Vec::new();
    for case in &reference.cases {
        let s = setting(case);
        let x = covariate(case);
        let mut c = config(case, &s);
        c.absorb = Absorb::new();
        let result = sunab::fit(&panel(case, x.as_ref()), &c).unwrap();
        out.extend(
            result
                .cells
                .iter()
                .filter_map(|t| t.estimate.map(|e| e.estimate)),
        );
        out.extend(result.aggregates.iter().map(|e| e.estimate.estimate));
        out.extend(result.aggregates.iter().filter_map(|e| e.estimate.se));
        out.extend(result.covariance.as_slice().iter().flatten());
        out.extend(influence_values(result.aggregate_influence.as_ref()).take(40));
        out.extend(influence_values(result.influence.as_ref()).take(40));
    }
    out
}

#[test]
fn sunab_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(sunab_bits);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sunab_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&sunab_bits())
        .finish();
    assert_eq!(hash, 12_002_447_259_631_101_414, "recorded hash");
}

#[test]
fn the_event_study_holds_the_aggregates_by_relative_period() {
    let reference = reference();
    for case in &reference.cases {
        let s = setting(case);
        let x = covariate(case);
        let fit = sunab::fit(&panel(case, x.as_ref()), &config(case, &s)).unwrap();
        let Some(&before) = fit.references.iter().rev().find(|&&r| r < 0) else {
            assert!(fit.event_study().is_err(), "{}", case.name);
            continue;
        };
        let study = fit.event_study().unwrap();
        assert_eq!(study.reference(), before, "{}", case.name);
        let periods: Vec<(usize, i64)> = fit
            .aggregates
            .iter()
            .enumerate()
            .filter_map(|(a, g)| match g.term {
                SunabTerm::Period(e) => Some((a, e)),
                _ => None,
            })
            .collect();
        assert_eq!(
            study.relative_periods(),
            periods.iter().map(|p| p.1).collect::<Vec<_>>().as_slice(),
            "{}",
            case.name
        );
        for (k, &(a, _)) in periods.iter().enumerate() {
            assert_eq!(
                study.coefficients()[k].to_bits(),
                fit.aggregates[a].estimate.estimate.to_bits()
            );
            for (l, &(b, _)) in periods.iter().enumerate() {
                assert_eq!(
                    study.covariance().get(k, l),
                    fit.aggregate_covariance.get(a, b)
                );
            }
        }
    }
}
