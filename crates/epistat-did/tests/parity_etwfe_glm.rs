//! The Poisson and logit extended two-way regressions and their average
//! effects, checked against `etwfe::etwfe(family = "poisson" / "logit")` and
//! `etwfe::emfx` 0.6.2 with `marginaleffects` 1.0.0 and `fixest` 0.14.2
//! (`tests/data/generate_etwfe_glm_reference.R`), and against a plain fit of
//! the design the module documentation defines.
//!
//! # Tolerances
//!
//! Both sides stop when the deviance changes by less than `tol (|D| + 0.1)`
//! between iterations. Near the optimum `b^` the deviance is `D^ + (b - b^)'
//! I (b - b^)` to second order, with `I = X'WX`, and the iteration (Newton's
//! for these canonical links) at least halves `D - D^` at each step, so at
//! the reported coefficients `D - D^ < tol (|D| + 0.1)` and at the iterate
//! before them, where the working weights of the covariance are taken,
//! `D - D^ < 2 tol (|D| + 0.1)`. With a factor 2 for the third-order term,
//! the reported coefficients lie within `e1 = sqrt(2 tol (|D| + 0.1))` of
//! `b^` in the norm of `I`, and the iterate before within
//! `e2 = sqrt(4 tol (|D| + 0.1))`. Between any two of the points the two
//! sides evaluate, `|x'(b1 - b2)| <= m = (e2_R + e2) sqrt(h)` for every row
//! `x` of the fit and of the effects, `h` the largest `x' I^-1 x`, which R
//! records.
//!
//! - A coefficient differs by at most `(e1_R + e1) sqrt([I^-1]_jj)`, and an
//!   average effect by `(e1_R + e1) sqrt(J I^-1 J')` and the second-order
//!   term `m^2 / 2 e^m mean(mu'_1 + mu'_0)`, as `|mu''| <= mu'` for both
//!   links.
//! - The working weights `W = w mu'` have `|d log W / d eta| <= 1`, so the
//!   two sides' `I` lie within a factor `e^m` of each other and so do their
//!   inverses: the iid variance of `c'b` differs by at most `(e^m - 1)` of
//!   itself.
//! - The clustered and robust variances `c'B (sum_g S_g S_g') B c` move with
//!   `dB = -B dI B`, `-m I <= dI <= m I`, and with the scores `w (y - mu)
//!   x`, whose change is at most `W m |x|` (and as much again for `fixest`'s
//!   working residuals, taken at the iterate before). The first-order change
//!   is at most `m` times the sensitivity `2 sum_g |a'S_g| (sqrt(c'Bc
//!   S_g'BS_g) + 2 sum_{i in g} W_i |a'x_i|)`, `a = Bc`, which R records;
//!   `e^{3m}` covers the growth of `W`, `mu'` and `B` along the way.
//! - The Jacobian of an average moves by at most `m e^m mean(mu'_1 |x_1| +
//!   mu'_0 |x_0|)` in each column.
//! - `marginaleffects` differentiates by forward differences with step `h_j
//!   = max(|b_j| sqrt(eps), 1e-10)`: the truncation error is at most `h_j / 2
//!   e^{h_j max|x_j|} mean(mu'_1 x_1j^2 + mu'_0 x_0j^2)`, the rounding of the
//!   two averages `2 eps r / h_j` with `r` their rounding bound, and the
//!   rounding of the step `|J_j| eps (|b_j| + h_j) / h_j`.
//! - The normal equations of either side round to `8 kappa eps`, with
//!   `kappa` the condition number of `I`.

#![expect(
    deprecated,
    reason = "the fits take their data as separate arguments, the epistat-glm 0.2 form"
)]

use epistat_core::InputError;
use epistat_did::{
    EtwfeControl, EtwfeEffects, EtwfeFamily, EtwfeOptions, EtwfeTerm, EtwfeType, EtwfeVcov,
    StaggeredPanel, StaggeredPanelBuilder, etwfe,
};
use epistat_glm::{Error as GlmError, Family, Glm, Link, Ssc};
use epistat_inference::{ClusterRule, Influence};
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Spec {
    family: String,
    control: String,
    vcov: String,
}

#[derive(Deserialize)]
struct Table {
    key: Option<Vec<i64>>,
    estimate: Vec<f64>,
    se: Vec<Option<f64>>,
    jacobian: Vec<f64>,
    se_analytic: Vec<f64>,
    jbj: Vec<f64>,
    jabs: Vec<f64>,
    curv: Vec<f64>,
    xmax: Vec<f64>,
    slope: Vec<f64>,
    round: Vec<f64>,
    n: Vec<usize>,
    sens: Vec<f64>,
    sens_unit: Vec<f64>,
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
    deviance: f64,
    tolerance: f64,
    collinear: Vec<String>,
    kappa: f64,
    b_diag: Vec<f64>,
    sens: Vec<f64>,
    sens_unit: Vec<f64>,
    h: f64,
    simple: Table,
    group: Table,
    calendar: Table,
    event: Table,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/etwfe_glm_ref.json"
    ))
}

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
        family: family(case),
        control: if case.spec.control == "never" {
            EtwfeControl::NeverTreated
        } else {
            EtwfeControl::NotYetTreated
        },
        vcov: match case.spec.vcov.as_str() {
            "iid" => EtwfeVcov::Iid,
            "hetero" => EtwfeVcov::Hetero,
            _ => EtwfeVcov::Clustered,
        },
        tolerance: case.tolerance,
        max_iterations: 100,
        ..EtwfeOptions::default()
    }
}

fn family(case: &Case) -> EtwfeFamily {
    if case.spec.family == "poisson" {
        EtwfeFamily::Poisson
    } else {
        EtwfeFamily::Logit
    }
}

/// fixest's name of a column.
fn name(term: EtwfeTerm) -> String {
    let x = |k: usize| format!("x{}", k + 1);
    match term {
        EtwfeTerm::Intercept => "(Intercept)".to_string(),
        EtwfeTerm::Cohort(g) => format!("g::{g}"),
        EtwfeTerm::Period(t) => format!("t::{t}"),
        EtwfeTerm::Covariate(k) => x(k),
        EtwfeTerm::CohortSlope { cohort, covariate } => {
            format!("g::{cohort}:{}", x(covariate))
        }
        EtwfeTerm::PeriodSlope { period, covariate } => {
            format!("t::{period}:{}", x(covariate))
        }
        EtwfeTerm::Cell { cohort, period } => {
            format!(".Dtreat:g::{cohort}:t::{period}")
        }
        EtwfeTerm::CellSlope {
            cohort,
            period,
            covariate,
        } => format!(".Dtreat:g::{cohort}:t::{period}:{}_dm", x(covariate)),
        other => panic!("{other:?} is not a column of etwfe"),
    }
}

/// `u' M v` for a row-major `q x q` matrix given by entry.
fn quad(u: &[f64], v: &[f64], m: &dyn Fn(usize) -> f64) -> f64 {
    let q = u.len();
    let mut total = 0.0;
    for (i, ui) in u.iter().enumerate() {
        for (j, vj) in v.iter().enumerate() {
            total += ui * m(i * q + j) * vj;
        }
    }
    total
}

/// The design of the module documentation for `terms`, with the cell
/// indicator of each row set to `cell` (`None`: the row's own indicator).
struct Plain<'a> {
    case: &'a Case,
    x: Option<&'a Mat<f64>>,
    terms: Vec<EtwfeTerm>,
    means: std::collections::BTreeMap<(usize, i64), f64>,
    indicator: Vec<bool>,
}

impl Plain<'_> {
    fn value(&self, i: usize, term: EtwfeTerm, treated: bool) -> f64 {
        let case = self.case;
        let xv = |k: usize| self.x.unwrap()[(i, k)];
        let in_cohort = |g: i64| case.first_treated[i] == Some(g);
        let one = |b: bool| f64::from(u8::from(b));
        match term {
            EtwfeTerm::Intercept => 1.0,
            EtwfeTerm::Cohort(g) => one(in_cohort(g)),
            EtwfeTerm::Period(t) => one(case.time[i] == t),
            EtwfeTerm::Covariate(k) => xv(k),
            EtwfeTerm::CohortSlope { cohort, covariate } => one(in_cohort(cohort)) * xv(covariate),
            EtwfeTerm::PeriodSlope { period, covariate } => {
                one(case.time[i] == period) * xv(covariate)
            }
            EtwfeTerm::Cell { cohort, period } => {
                one(treated && in_cohort(cohort) && case.time[i] == period)
            }
            EtwfeTerm::CellSlope {
                cohort,
                period,
                covariate,
            } => {
                if treated && in_cohort(cohort) && case.time[i] == period {
                    xv(covariate) - self.means[&(covariate, cohort)]
                } else {
                    0.0
                }
            }
            other => panic!("{other:?} is not a column of etwfe"),
        }
    }

    fn row(&self, i: usize, treated: bool) -> Vec<f64> {
        self.terms
            .iter()
            .map(|&t| self.value(i, t, treated))
            .collect()
    }
}

fn plain<'a>(case: &'a Case, x: Option<&'a Mat<f64>>, terms: Vec<EtwfeTerm>) -> Plain<'a> {
    let n = case.unit.len();
    let key = |i: usize| case.first_treated[i].unwrap_or(0);
    let mut means = std::collections::BTreeMap::new();
    for i in 0..n {
        for k in 0..case.n_covariates {
            means.entry((k, key(i))).or_insert_with(|| {
                let rows: Vec<usize> = (0..n).filter(|&r| key(r) == key(i)).collect();
                rows.iter().map(|&r| x.unwrap()[(r, k)]).sum::<f64>() / rows.len() as f64
            });
        }
    }
    let t_first = *case.time.iter().min().unwrap();
    let never = case.spec.control == "never";
    let indicator = (0..n)
        .map(|i| {
            case.first_treated[i].is_some_and(|g| {
                g > t_first
                    && if never {
                        case.time[i] != g - 1
                    } else {
                        case.time[i] >= g
                    }
            })
        })
        .collect();
    Plain {
        case,
        x,
        terms,
        means,
        indicator,
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one comparison per output")]
fn etwfe_glm_matches_the_package() {
    for case in &reference().cases {
        let x = covariates(case);
        let config = config(case);
        let result = etwfe::fit(&panel(case, x.as_ref()), &config).unwrap();
        assert!(result.converged, "{}", case.name);
        assert_eq!(result.n_rows, case.nobs, "{}", case.name);
        assert_eq!(result.df_k, case.df_k, "{}", case.name);
        let labels: Vec<String> = result.coefficients.iter().map(|c| name(c.term)).collect();
        assert!(result.coefficients.iter().all(|c| c.estimate.is_some()));
        // fixest's columns are these and the all-zero cells it drops.
        assert_eq!(labels, case.names, "{}", case.name);
        assert!(case.collinear.iter().all(|c| !labels.contains(c)));
        let q = labels.len();

        // The fit is that of Glm::fit on the design the documentation
        // defines, to the bit.
        let terms: Vec<EtwfeTerm> = result.coefficients.iter().map(|c| c.term).collect();
        let design = plain(case, x.as_ref(), terms);
        let n = case.unit.len();
        let t_first = *case.time.iter().min().unwrap();
        let cutoff = case
            .first_treated
            .iter()
            .all(Option::is_some)
            .then(|| case.first_treated.iter().flatten().copied().max().unwrap());
        let fit_rows: Vec<usize> = (0..n)
            .filter(|&i| {
                !(case.first_treated[i].is_some_and(|g| g <= t_first)
                    || cutoff.is_some_and(|c| case.time[i] >= c))
            })
            .collect();
        let dense = Mat::from_fn(fit_rows.len(), q, |r, j| {
            let i = fit_rows[r];
            design.value(i, design.terms[j], design.indicator[i])
        });
        let y: Vec<f64> = fit_rows.iter().map(|&i| case.outcome[i]).collect();
        let w: Option<Vec<f64>> = case
            .weights
            .as_ref()
            .map(|w| fit_rows.iter().map(|&i| w[i]).collect());
        let (fam, link) = if case.spec.family == "poisson" {
            (Family::Poisson, Link::Log)
        } else {
            (Family::Binomial, Link::Logit)
        };
        let glm = Glm {
            tolerance: case.tolerance,
            max_iterations: 100,
            ..Glm::new(fam, link)
        };
        let plain_fit = glm
            .fit_slices(dense.as_ref(), &y, w.as_deref(), None, None)
            .unwrap();
        for j in 0..q {
            assert_eq!(
                plain_fit.coefficients[j].map(f64::to_bits),
                result.coefficients[j]
                    .estimate
                    .map(|e| e.estimate.to_bits()),
                "{} {}",
                case.name,
                labels[j]
            );
        }
        let deviance = result.deviance.unwrap();
        assert_eq!(plain_fit.deviance.to_bits(), deviance.to_bits());

        // The distances of the module.
        let tol = case.tolerance;
        let e1 = |d: f64| (2.0 * tol * (d.abs() + 0.1)).sqrt();
        let e2 = |d: f64| (4.0 * tol * (d.abs() + 0.1)).sqrt();
        let dist = e1(case.deviance) + e1(deviance);
        let m = (e2(case.deviance) + e2(deviance)) * case.h.sqrt();
        let grow = m.exp();
        let nobs = case.nobs as f64;
        let k_df = case.df_k as f64;
        let units = result.n_units as f64;
        let ssc = match case.spec.vcov.as_str() {
            "iid" => (nobs - 1.0) / (nobs - k_df),
            "hetero" => nobs / (nobs - k_df),
            _ => units / (units - 1.0) * (nobs - 1.0) / (nobs - k_df),
        };
        let round = 8.0 * case.kappa * EPS;
        // The change of a variance c'Vc given its value and sensitivity.
        let dvar = |value: f64, sens: f64| -> f64 {
            if case.spec.vcov == "iid" {
                (grow - 1.0) * value
            } else {
                ssc * m * (3.0 * m).exp() * sens
            }
        };
        let max_beta = case.coef.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let mut worst = [0.0_f64; 3];
        for j in 0..q {
            let got = result.coefficients[j].estimate.unwrap().estimate;
            let bound = dist * case.b_diag[j].sqrt() + round * max_beta;
            assert!(
                (got - case.coef[j]).abs() <= bound,
                "{} {}: {got} {} {bound:e}",
                case.name,
                labels[j],
                case.coef[j]
            );
            worst[0] = worst[0].max((got - case.coef[j]).abs() / case.b_diag[j].sqrt());
            let v = result.covariance.get(j, j).unwrap();
            let want = case.vcov[j * q + j];
            let bound = dvar(want, case.sens[j]) + round * want;
            assert!(
                (v - want).abs() <= bound,
                "{} vcov {}: {v:e} {want:e} {bound:e}",
                case.name,
                labels[j]
            );
            worst[1] = worst[1].max((v - want).abs() / want);
        }
        // The influence functions give the unit-clustered covariance without
        // small-sample factors.
        let unit_variance = result
            .influence
            .as_ref()
            .unwrap()
            .variance_with(None, ClusterRule::NoSmallSample)
            .unwrap();
        let effect_variance = result
            .effect_influence
            .as_ref()
            .unwrap()
            .variance_with(None, ClusterRule::NoSmallSample)
            .unwrap();
        for j in 0..q {
            let v = unit_variance.get(j, j).unwrap();
            let want = case.vcov_unadjusted[j * q + j];
            let bound = m * (3.0 * m).exp() * case.sens_unit[j] + round * want;
            assert!(
                (v - want).abs() <= bound,
                "{} influence {}: {v:e} {want:e}",
                case.name,
                labels[j]
            );
        }

        // The average effects.
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
                assert_eq!(effect.n_rows, table.n[r], "{} {term:?}", case.name);
                let row = |v: &[f64]| v[r * q..(r + 1) * q].to_vec();
                let (j_r, jabs, curv, xmax) = (
                    row(&table.jacobian),
                    row(&table.jabs),
                    row(&table.curv),
                    row(&table.xmax),
                );
                let bound = dist * table.jbj[r].sqrt()
                    + 0.5 * m * m * grow * table.slope[r]
                    + 2.0 * EPS * table.round[r];
                assert!(
                    (effect.estimate.estimate - table.estimate[r]).abs() <= bound,
                    "{} {term:?}: {} {} {bound:e}",
                    case.name,
                    effect.estimate.estimate,
                    table.estimate[r]
                );
                worst[2] =
                    worst[2].max((effect.estimate.estimate - table.estimate[r]).abs() / bound);
                let Some(se) = effect.estimate.se else {
                    assert!(table.se[r].is_none(), "{} {term:?}", case.name);
                    continue;
                };
                // Rust's J against R's analytic one, and the variance.
                let dj: Vec<f64> = jabs
                    .iter()
                    .map(|v| m * grow * v + table.n[r] as f64 * EPS * v)
                    .collect();
                let abs_j: Vec<f64> = j_r.iter().map(|v| v.abs()).collect();
                let v_abs = |ij: usize| case.vcov[ij].abs();
                let analytic = table.se_analytic[r];
                let tol_v = dvar(analytic * analytic, table.sens[r])
                    + 2.0 * quad(&dj, &abs_j, &v_abs)
                    + quad(&dj, &dj, &v_abs)
                    + (round + 16.0 * EPS) * analytic * analytic;
                assert!(
                    (se * se - analytic * analytic).abs() <= tol_v,
                    "{} {term:?} analytic se: {se} {analytic} {tol_v:e}",
                    case.name
                );
                // marginaleffects' forward differences.
                let fd: Vec<f64> = (0..q)
                    .map(|j| {
                        let beta = case.coef[j].abs();
                        let h = (beta * EPS.sqrt()).max(1e-10);
                        0.5 * h * (h * xmax[j]).exp() * curv[j]
                            + 2.0 * EPS * table.round[r] / h
                            + j_r[j].abs() * EPS * (beta + h) / h
                    })
                    .collect();
                let tol_fd = 2.0 * quad(&fd, &abs_j, &v_abs) + quad(&fd, &fd, &v_abs);
                let want = table.se[r].unwrap();
                assert!(
                    (se * se - want * want).abs() <= tol_v + tol_fd,
                    "{} {term:?} se: {se} {want} {:e}",
                    case.name,
                    tol_v + tol_fd
                );
                // The influence function of the effect gives J V_unit J'
                // without the small-sample factors.
                let v = effect_variance.get(a, a).unwrap();
                let vu = |ij: usize| case.vcov_unadjusted[ij];
                let want = quad(&j_r, &j_r, &vu);
                let tol = m * (3.0 * m).exp() * table.sens_unit[r]
                    + 2.0 * quad(&dj, &abs_j, &|ij| vu(ij).abs())
                    + quad(&dj, &dj, &|ij| vu(ij).abs())
                    + (round + 16.0 * EPS) * want;
                assert!(
                    (v - want).abs() <= tol,
                    "{} {term:?} influence {v:e} {want:e} {tol:e}",
                    case.name
                );
            }
        }
        assert_eq!(checked, result.effects.len(), "{}", case.name);
        eprintln!(
            "{}: coefficients {:e} se, variances {:e} relative, effects {:e} of their bound",
            case.name, worst[0], worst[1], worst[2]
        );

        // The effects are the averages of the contrasts of the plain design.
        let link = if case.spec.family == "poisson" {
            Link::Log
        } else {
            Link::Logit
        };
        let beta: Vec<f64> = result
            .coefficients
            .iter()
            .map(|c| c.estimate.unwrap().estimate)
            .collect();
        let eta = |row: &[f64]| row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
        let simple = &result.effects[0];
        let rows: Vec<usize> = (0..n)
            .filter(|&i| {
                design.indicator[i]
                    && (case.spec.control != "never"
                        || case.time[i] >= case.first_treated[i].unwrap())
                    && fit_rows.contains(&i)
            })
            .collect();
        let mut total = 0.0;
        let mut size = 0.0;
        for &i in &rows {
            let hi = link.inverse(eta(&design.row(i, true)));
            let lo = link.inverse(eta(&design.row(i, false)));
            total += hi - lo;
            size += hi + lo;
        }
        // Two orders of summation of the contrasts and of the predictors.
        let bound = 4.0 * (q as f64 + rows.len() as f64) * EPS * size / rows.len() as f64;
        assert_eq!(simple.n_rows, rows.len());
        assert!(
            (simple.estimate.estimate - total / rows.len() as f64).abs() <= bound,
            "{} plain simple",
            case.name
        );
    }
}

#[test]
fn unusable_inputs_are_errors() {
    let reference = reference();
    let case = reference
        .cases
        .iter()
        .find(|c| c.name == "logit_last_cluster")
        .unwrap();
    let x = covariates(case);
    let base = config(case);
    let unit_effects = EtwfeOptions {
        effects: EtwfeEffects::Unit,
        ..base
    };
    assert!(matches!(
        etwfe::fit(&panel(case, x.as_ref()), &unit_effects),
        Err(epistat_did::Error::Unsupported { .. })
    ));
    let never = EtwfeOptions {
        control: EtwfeControl::NeverTreated,
        ..base
    };
    assert_eq!(
        etwfe::fit(&panel(case, x.as_ref()), &never).unwrap_err(),
        epistat_did::Error::NoNeverTreated
    );
    let mut weights = case.weights.clone().unwrap();
    weights[7] = -1.0;
    assert_eq!(
        columns(case, &case.outcome, Some(&weights), x.as_ref())
            .build()
            .unwrap_err(),
        epistat_did::Error::Input(InputError::Domain {
            what: "weights",
            value: -1.0,
            domain: "[0, inf)"
        })
    );
    // A response outside the family's support names its input row; the row
    // must be one of the fit.
    let mut outcome = case.outcome.clone();
    let row = (0..outcome.len())
        .rev()
        .find(|&i| case.time[i] == 1)
        .unwrap();
    outcome[row] = 2.0;
    let bad = columns(case, &outcome, case.weights.as_deref(), x.as_ref())
        .build()
        .unwrap();
    assert_eq!(
        etwfe::fit(&bad, &base).unwrap_err(),
        epistat_did::Error::Glm(GlmError::Response { row, value: 2.0 })
    );
}

/// The values of an influence function, column after column.
fn influence_values(influence: Option<&Influence>) -> impl Iterator<Item = f64> + '_ {
    influence
        .into_iter()
        .flat_map(|f| (0..f.n_columns()).flat_map(move |j| f.column(j).iter().copied()))
}

fn etwfe_glm_bits() -> Vec<f64> {
    let reference = reference();
    let mut out = Vec::new();
    for case in &reference.cases {
        let x = covariates(case);
        let c = EtwfeOptions {
            tolerance: 1e-8,
            max_iterations: 25,
            ssc: Ssc::new(),
            ..config(case)
        };
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
        out.extend(result.effect_covariance.as_slice().iter().flatten());
        out.extend(influence_values(result.effect_influence.as_ref()).take(40));
        out.extend(influence_values(result.influence.as_ref()).take(40));
        out.extend(result.deviance);
    }
    out
}

#[test]
fn etwfe_glm_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(etwfe_glm_bits);
}

#[test]
fn etwfe_glm_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&etwfe_glm_bits())
        .finish();
    epistat_testkit::hash_out("etwfe_glm_has_its_recorded_bits", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 17_652_008_239_643_160_384, "recorded hash");
    }
}
