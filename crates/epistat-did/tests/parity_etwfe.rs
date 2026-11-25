//! Wooldridge's extended two-way fixed-effects regression and its average
//! effects, checked against `etwfe::etwfe` and `etwfe::emfx` 0.6.2 with
//! `marginaleffects` 1.0.0 and `fixest` 0.14.2
//! (`tests/data/generate_etwfe_reference.R`).

#[path = "common/feols_bound.rs"]
mod feols_bound;

use epistat_did::{
    EtwfeConfig, EtwfeControl, EtwfeEffects, EtwfeTerm, EtwfeType, StaggeredPanel, did_covariance,
    estimate_etwfe,
};
use epistat_glm::{Absorb, Ssc, Vcov};
use epistat_inference::Clusters;
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
    key: Option<Vec<i32>>,
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
    time: Vec<i32>,
    first_treated: Vec<Option<i32>>,
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
};

fn covariates(case: &Case) -> Option<Mat<f64>> {
    let p = case.n_covariates;
    case.covariates
        .as_ref()
        .map(|x| Mat::from_fn(x.len() / p, p, |i, j| x[i * p + j]))
}

fn panel<'a>(case: &'a Case, x: Option<&'a Mat<f64>>) -> StaggeredPanel<'a> {
    StaggeredPanel {
        unit: &case.unit,
        time: &case.time,
        first_treated: &case.first_treated,
        outcome: &case.outcome,
        weights: case.weights.as_deref(),
        covariates: x.map(Mat::as_ref),
    }
}

fn config<'a>(case: &Case, ways: &'a [&'a [u32]]) -> EtwfeConfig<'a> {
    EtwfeConfig {
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
            "iid" => Vcov::Iid,
            "hetero" => Vcov::Hetero,
            _ => Vcov::Cluster(ways),
        },
        absorb: TIGHT,
        ssc: Ssc::new(),
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
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one comparison per output")]
fn etwfe_matches_the_package() {
    for case in &reference().cases {
        let x = covariates(case);
        let ways = [&case.unit[..]];
        let result = estimate_etwfe(&panel(case, x.as_ref()), &config(case, &ways)).unwrap();
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
        let mean = |k: usize, g: i32| {
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
            let cell = |cohort: i32, period: i32| {
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
            .map(|&t| (t - t_first).cast_unsigned())
            .collect();
        let cohort_codes: Vec<u32> = (0..n).map(|i| cohort_key(i).cast_unsigned()).collect();
        let first: &[u32] = if case.spec.effects == "unit" {
            &case.unit
        } else {
            &cohort_codes
        };
        let nobs = case.nobs as f64;
        let k_df = case.df_k as f64;
        let units = result.units.len() as f64;
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
                result.coefficients[j].estimate.map(f64::to_bits),
                "{} {}",
                case.name,
                labels[j]
            );
        }
        for j in 0..q {
            let got = result.coefficients[j].estimate.unwrap();
            assert!(
                (got - case.coef[j]).abs() <= b.beta,
                "{} {}: {got} {}",
                case.name,
                labels[j],
                case.coef[j]
            );
            for k in 0..q {
                let got = result.covariance[j * q + k].unwrap();
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
        let of = Clusters::from_labels(&result.units);
        let psi: Vec<Vec<f64>> = (0..q)
            .map(|j| result.influence[j].clone().unwrap())
            .collect();
        let raw = did_covariance(&psi, &of).unwrap();
        for j in 0..q * q {
            assert!(
                (raw[j] - case.vcov_unadjusted[j]).abs() <= bu.covariance[j] / unit_factor,
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
        let tables: [(&Table, &dyn Fn(i32) -> EtwfeType); 4] = [
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
                    (effect.estimate - table.estimate[r]).abs()
                        <= j_abs * b.beta + round + 16.0 * EPS * effect.estimate.abs(),
                    "{} {term:?}: {} {}",
                    case.name,
                    effect.estimate,
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
                let Some(se) = effect.std_error else {
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
                let v = did_covariance(&[result.effect_influence[a].clone()], &of).unwrap()[0];
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
    let ways = [&case.unit[..]];
    let config = EtwfeConfig {
        control: EtwfeControl::NeverTreated,
        ..config(case, &ways)
    };
    assert!(estimate_etwfe(&panel(case, None), &config).is_err());
}

fn etwfe_bits() -> Vec<f64> {
    let reference = reference();
    let mut out = Vec::new();
    for case in &reference.cases {
        let x = covariates(case);
        let ways = [&case.unit[..]];
        let mut c = config(case, &ways);
        c.absorb = Absorb::new();
        let result = estimate_etwfe(&panel(case, x.as_ref()), &c).unwrap();
        out.extend(result.coefficients.iter().filter_map(|t| t.estimate));
        out.extend(result.effects.iter().map(|e| e.estimate));
        out.extend(result.effects.iter().filter_map(|e| e.std_error));
        out.extend(result.covariance.iter().flatten().take(400));
        out.extend(result.effect_influence.iter().flatten().take(40));
        out.extend(result.influence.iter().flatten().flatten().take(40));
    }
    out
}

#[test]
fn etwfe_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(etwfe_bits);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn etwfe_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&etwfe_bits())
        .finish();
    assert_eq!(hash, 2_658_182_323_512_888_873, "recorded hash");
}
