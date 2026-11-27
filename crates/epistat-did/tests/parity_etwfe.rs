//! Wooldridge's extended two-way fixed-effects regression and its average
//! effects, checked against `etwfe::etwfe` and `etwfe::emfx` 0.6.2 with
//! `marginaleffects` 1.0.0 and `fixest` 0.14.2
//! (`tests/data/generate_etwfe_reference.R`).

#[path = "common/feols_bound.rs"]
mod feols_bound;

use epistat_did::{
    EtwfeControl, EtwfeEffects, EtwfeOptions, EtwfeTerm, EtwfeType, EtwfeVcov, StaggeredPanel,
    StaggeredPanelBuilder, etwfe,
};
use epistat_glm::{Absorb, Ssc};
use epistat_inference::{ClusterRule, Influence};
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use feols_bound::{EPS, Meat, feols_bound};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Spec {
    control: String,
    effects: String,
    vcov: String,
}

#[derive(Deserialize)]
struct Exact {
    error: f64,
    lambda_plus: f64,
}

#[derive(Deserialize)]
struct Table {
    key: Option<Vec<i64>>,
    estimate: Vec<f64>,
    se: Vec<Option<f64>>,
    jacobian: Vec<f64>,
    se_analytic: Vec<f64>,
    /// The largest sum of the absolute terms of one prediction.
    terms: f64,
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
    covariates: Option<Vec<f64>>,
    n_covariates: usize,
    names: Vec<String>,
    coef: Vec<f64>,
    vcov: Vec<f64>,
    vcov_unadjusted: Vec<f64>,
    df_k: usize,
    nobs: usize,
    collinear: Vec<String>,
    simple: Table,
    group: Table,
    calendar: Table,
    event: Table,
    exact: Exact,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/etwfe_ref.json"
    ))
}

const TIGHT: Absorb = Absorb {
    tolerance: 1e-13,
    max_iterations: 100_000,
    ..Absorb::new()
};

fn covariates(case: &Case) -> Option<Mat<f64>> {
    let p = case.n_covariates;
    case.covariates
        .as_ref()
        .map(|x| Mat::from_fn(x.len() / p, p, |i, j| x[i * p + j]))
}

fn panel<'a>(case: &'a Case, x: Option<&'a Mat<f64>>) -> StaggeredPanel<'a> {
    columns(case, &case.outcome, case.weights.as_deref(), x)
        .build()
        .expect("a valid panel")
}

/// The panel of `case` with the given outcome and weights.
fn columns<'a>(
    case: &'a Case,
    outcome: &'a [f64],
    weights: Option<&'a [f64]>,
    x: Option<&'a Mat<f64>>,
) -> StaggeredPanelBuilder<'a> {
    let mut panel = StaggeredPanel::builder(&case.unit, &case.time, &case.first_treated, outcome);
    if let Some(w) = weights {
        panel = panel.weights(w);
    }
    if let Some(x) = x {
        panel = panel.covariates(x.as_ref());
    }
    panel
}

fn config(case: &Case) -> EtwfeOptions {
    EtwfeOptions {
        control: if case.spec.control == "never" {
            EtwfeControl::NeverTreated
        } else {
            EtwfeControl::NotYetTreated
        },
        effects: if case.spec.effects == "unit" {
            EtwfeEffects::Unit
        } else {
            EtwfeEffects::Cohort
        },
        vcov: match case.spec.vcov.as_str() {
            "iid" => EtwfeVcov::Iid,
            "hetero" => EtwfeVcov::Hetero,
            _ => EtwfeVcov::Clustered,
        },
        absorb: TIGHT,
        ssc: Ssc::new(),
        ..EtwfeOptions::default()
    }
}

/// fixest's name of a column.
fn name(term: EtwfeTerm) -> String {
    let x = |k: usize| format!("x{}", k + 1);
    match term {
        EtwfeTerm::Covariate(k) => x(k),
        EtwfeTerm::CohortSlope { cohort, covariate } => format!("g::{cohort}:{}", x(covariate)),
        EtwfeTerm::PeriodSlope { period, covariate } => format!("t::{period}:{}", x(covariate)),
        EtwfeTerm::Cell { cohort, period } => format!(".Dtreat:g::{cohort}:t::{period}"),
        EtwfeTerm::CellSlope {
            cohort,
            period,
            covariate,
        } => format!(".Dtreat:g::{cohort}:t::{period}:{}_dm", x(covariate)),
        other => panic!("{other:?} is not a column of the linear model"),
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one comparison per output")]
fn etwfe_matches_the_package() {
    for case in &reference().cases {
        let x = covariates(case);
        let result = etwfe::fit(&panel(case, x.as_ref()), &config(case)).unwrap();
        assert_eq!(result.n_rows, case.nobs, "{}", case.name);
        assert_eq!(result.df_k, case.df_k, "{}", case.name);
        let labels: Vec<String> = result.coefficients.iter().map(|c| name(c.term)).collect();
        assert!(result.coefficients.iter().all(|c| c.estimate.is_some()));
        // fixest's columns are these and the all-zero cells it drops.
        assert_eq!(labels, case.names, "{}", case.name);
        assert!(case.collinear.iter().all(|c| !labels.contains(c)));
        let q = labels.len();

        // The design as the module documentation defines it, fitted plainly.
        let n = case.unit.len();
        let p = case.n_covariates;
        let t_first = *case.time.iter().min().unwrap();
        // Without never-treated units the last cohort is the reference, from
        // its first treated period on.
        let cutoff = case
            .first_treated
            .iter()
            .all(Option::is_some)
            .then(|| case.first_treated.iter().flatten().copied().max().unwrap());
        let excluded: Vec<bool> = (0..n)
            .map(|i| {
                case.first_treated[i].is_some_and(|g| g <= t_first)
                    || cutoff.is_some_and(|c| case.time[i] >= c)
            })
            .collect();
        let cohort_key = |i: usize| case.first_treated[i].unwrap_or(0);
        let mean = |k: usize, g: i64| {
            let rows: Vec<usize> = (0..n).filter(|&i| cohort_key(i) == g).collect();
            rows.iter()
                .map(|&i| x.as_ref().unwrap()[(i, k)])
                .sum::<f64>()
                / rows.len() as f64
        };
        let mut means = std::collections::BTreeMap::new();
        for i in 0..n {
            for k in 0..p {
                means
                    .entry((k, cohort_key(i)))
                    .or_insert_with(|| mean(k, cohort_key(i)));
            }
        }
        let xv = |i: usize, k: usize| x.as_ref().unwrap()[(i, k)];
        let design = Mat::from_fn(n, q, |i, j| {
            let cell = |cohort: i64, period: i64| {
                case.first_treated[i] == Some(cohort) && case.time[i] == period
            };
            match result.coefficients[j].term {
                EtwfeTerm::Covariate(k) => xv(i, k),
                EtwfeTerm::CohortSlope { cohort, covariate } => {
                    if case.first_treated[i] == Some(cohort) {
                        xv(i, covariate)
                    } else {
                        0.0
                    }
                }
                EtwfeTerm::PeriodSlope { period, covariate } => {
                    if case.time[i] == period {
                        xv(i, covariate)
                    } else {
                        0.0
                    }
                }
                EtwfeTerm::Cell { cohort, period } => f64::from(u8::from(cell(cohort, period))),
                EtwfeTerm::CellSlope {
                    cohort,
                    period,
                    covariate,
                } => {
                    if cell(cohort, period) {
                        xv(i, covariate) - means[&(covariate, cohort)]
                    } else {
                        0.0
                    }
                }
                other => panic!("{other:?} is not a column of the linear model"),
            }
        });
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if excluded[i] {
                    0.0
                } else {
                    case.weights.as_ref().map_or(1.0, |w| w[i])
                }
            })
            .collect();
        let time: Vec<u32> = case
            .time
            .iter()
            .map(|&t| u32::try_from(t - t_first).unwrap())
            .collect();
        let cohort_codes: Vec<u32> = (0..n)
            .map(|i| u32::try_from(cohort_key(i)).unwrap())
            .collect();
        let first: &[u32] = if case.spec.effects == "unit" {
            &case.unit
        } else {
            &cohort_codes
        };
        let nobs = case.nobs as f64;
        let k_df = case.df_k as f64;
        let units = result.n_units as f64;
        let rows: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
        let (kind, factor) = match case.spec.vcov.as_str() {
            "iid" => (Meat::Iid, 1.0 / (nobs - k_df)),
            "hetero" => (Meat::Cluster(&rows), nobs / (nobs - k_df)),
            _ => (
                Meat::Cluster(&case.unit),
                units / (units - 1.0) * (nobs - 1.0) / (nobs - k_df),
            ),
        };
        let bound = |kind: Meat<'_>, factor: f64| {
            feols_bound(
                &[first, &time],
                Some(&weights),
                &design,
                &case.outcome,
                kind,
                TIGHT,
                case.exact.lambda_plus,
                case.exact.error,
                factor,
            )
        };
        let b = bound(kind, factor);
        assert_eq!(b.kept.len(), q, "{}", case.name);
        for (j, c) in b.coefficients.iter().enumerate() {
            assert_eq!(
                c.map(f64::to_bits),
                result.coefficients[j]
                    .estimate
                    .map(|e| e.estimate.to_bits()),
                "{} {}",
                case.name,
                labels[j]
            );
        }
        for j in 0..q {
            let got = result.coefficients[j].estimate.unwrap().estimate;
            assert!(
                (got - case.coef[j]).abs() <= b.beta,
                "{} {}: {got} {}",
                case.name,
                labels[j],
                case.coef[j]
            );
            for k in 0..q {
                let got = result.covariance.get(j, k).unwrap();
                let want = case.vcov[j * q + k];
                assert!(
                    (got - want).abs() <= b.covariance[j * q + k],
                    "{} vcov {} {}: {got:e} {want:e}",
                    case.name,
                    labels[j],
                    labels[k]
                );
            }
        }
        // The influence functions give the unit-clustered covariance without
        // small-sample factors.
        let unit_factor = units / (units - 1.0) * (nobs - 1.0) / (nobs - k_df);
        let bu = bound(Meat::Cluster(&case.unit), unit_factor);
        let influence = result.influence.as_ref().unwrap();
        assert_eq!(influence.n_columns(), q);
        let raw = influence
            .variance_with(None, ClusterRule::NoSmallSample)
            .unwrap();
        for j in 0..q * q {
            assert!(
                (raw.as_slice()[j].unwrap() - case.vcov_unadjusted[j]).abs()
                    <= bu.covariance[j] / unit_factor,
                "{} influence {} {}",
                case.name,
                labels[j / q],
                labels[j % q]
            );
        }

        // The average effects.
        let max_x = x.as_ref().map_or(1.0, |x| {
            (0..n)
                .flat_map(|i| (0..p).map(move |k| x[(i, k)].abs()))
                .fold(1.0, f64::max)
        });
        let tables: [(&Table, &dyn Fn(i64) -> EtwfeType); 4] = [
            (&case.simple, &|_| EtwfeType::Simple),
            (&case.group, &EtwfeType::Group),
            (&case.calendar, &EtwfeType::Calendar),
            (&case.event, &EtwfeType::Event),
        ];
        let mut checked = 0;
        for (table, kind) in tables {
            let keys = table.key.clone().unwrap_or_else(|| vec![0]);
            for (r, &key) in keys.iter().enumerate() {
                let term = kind(key);
                let a = result
                    .effects
                    .iter()
                    .position(|e| e.term == term)
                    .unwrap_or_else(|| panic!("{} {term:?}", case.name));
                let effect = &result.effects[a];
                checked += 1;
                // R's analytic Jacobian, rebuilt here from the coefficient
                // estimates: J beta is the estimate.
                let j_r: Vec<f64> = (0..q).map(|j| table.jacobian[r * q + j]).collect();
                let j_abs: f64 = j_r.iter().map(|v| v.abs()).sum();
                // marginaleffects averages differences of predictions of at
                // most K + 2 terms, each rounded to (K + 2) eps M.
                let round = 2.0 * (q as f64 + 2.0) * EPS * table.terms;
                assert!(
                    (effect.estimate.estimate - table.estimate[r]).abs()
                        <= j_abs * b.beta + round + 16.0 * EPS * effect.estimate.estimate.abs(),
                    "{} {term:?}: {} {}",
                    case.name,
                    effect.estimate.estimate,
                    table.estimate[r]
                );
                // J from means over at most n rows of values at most max_x.
                let dj = 4.0 * nobs * EPS * max_x;
                // The variance with the analytic Jacobian: the covariance's
                // bound and the Jacobian's rounding.
                let quad = |u: &[f64], v: &[f64], m: &dyn Fn(usize) -> f64| {
                    let mut total = 0.0;
                    for (i, ui) in u.iter().enumerate() {
                        for (j, vj) in v.iter().enumerate() {
                            total += ui * m(i * q + j) * vj;
                        }
                    }
                    total
                };
                let abs_j: Vec<f64> = j_r.iter().map(|v| v.abs()).collect();
                let dj_v = vec![dj; q];
                let v_abs = |ij: usize| case.vcov[ij].abs();
                let tol_v = quad(&abs_j, &abs_j, &|ij| b.covariance[ij])
                    + 2.0 * quad(&dj_v, &abs_j, &v_abs)
                    + quad(&dj_v, &dj_v, &v_abs);
                let Some(se) = effect.estimate.se else {
                    assert!(table.se[r].is_none(), "{} {term:?}", case.name);
                    assert!(j_r.iter().all(|&v| v == 0.0));
                    continue;
                };
                let analytic = table.se_analytic[r];
                assert!(
                    (se * se - analytic * analytic).abs() <= tol_v + 16.0 * EPS * se * se,
                    "{} {term:?} analytic se: {se} {analytic}",
                    case.name
                );
                // marginaleffects' forward differences with steps
                // h = max(|beta| sqrt(eps), 1e-10): the rounding of the step
                // and of the predictions.
                let fd: Vec<f64> = (0..q)
                    .map(|j| {
                        let beta = case.coef[j].abs();
                        let h = (beta * EPS.sqrt()).max(1e-10);
                        j_r[j].abs() * EPS * (beta + h) / h + 2.0 * round / h
                    })
                    .collect();
                let tol_fd = 2.0 * quad(&fd, &abs_j, &v_abs) + quad(&fd, &fd, &v_abs);
                let want = table.se[r].unwrap();
                assert!(
                    (se * se - want * want).abs() <= tol_v + tol_fd + 16.0 * EPS * se * se,
                    "{} {term:?} se: {se} {want}",
                    case.name
                );
                // The influence function of the effect gives J V_unit J'
                // without the small-sample factors.
                let v = result
                    .effect_influence
                    .as_ref()
                    .unwrap()
                    .variance_with(None, ClusterRule::NoSmallSample)
                    .unwrap()
                    .get(a, a)
                    .unwrap();
                let want = quad(&j_r, &j_r, &|ij| case.vcov_unadjusted[ij]);
                let tol = quad(&abs_j, &abs_j, &|ij| bu.covariance[ij] / unit_factor)
                    + 2.0 * quad(&dj_v, &abs_j, &|ij| case.vcov_unadjusted[ij].abs());
                assert!(
                    (v - want).abs() <= tol + 16.0 * EPS * want,
                    "{} {term:?} influence {v:e} {want:e} {tol:e}",
                    case.name
                );
            }
        }
        assert_eq!(checked, result.effects.len(), "{}", case.name);
    }
}

#[test]
fn never_treated_controls_need_never_treated_units() {
    let reference = reference();
    let case = reference
        .cases
        .iter()
        .find(|c| c.name == "notyet_last_hetero")
        .unwrap();
    let config = EtwfeOptions {
        control: EtwfeControl::NeverTreated,
        ..config(case)
    };
    assert!(matches!(
        etwfe::fit(&panel(case, None), &config),
        Err(epistat_did::Error::NoNeverTreated)
    ));
}

/// The values of an influence function, column after column.
fn influence_values(influence: Option<&Influence>) -> impl Iterator<Item = f64> + '_ {
    influence
        .into_iter()
        .flat_map(|f| (0..f.n_columns()).flat_map(move |j| f.column(j).iter().copied()))
}

fn etwfe_bits() -> Vec<f64> {
    let reference = reference();
    let mut out = Vec::new();
    for case in &reference.cases {
        let x = covariates(case);
        let mut c = config(case);
        c.absorb = Absorb::new();
        let result = etwfe::fit(&panel(case, x.as_ref()), &c).unwrap();
        out.extend(
            result
                .coefficients
                .iter()
                .filter_map(|t| t.estimate.map(|e| e.estimate)),
        );
        out.extend(result.effects.iter().map(|e| e.estimate.estimate));
        out.extend(result.effects.iter().filter_map(|e| e.estimate.se));
        out.extend(result.covariance.as_slice().iter().flatten().take(400));
        out.extend(influence_values(result.effect_influence.as_ref()).take(40));
        out.extend(influence_values(result.influence.as_ref()).take(40));
    }
    out
}

#[test]
fn etwfe_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(etwfe_bits);
}

#[test]
fn etwfe_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&etwfe_bits())
        .finish();
    epistat_testkit::hash_out("etwfe_has_its_recorded_bits", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 13_680_617_634_678_465_476, "recorded hash");
    }
}
