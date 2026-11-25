//! The imputation estimator and the two-stage estimator against
//! `didimputation::did_imputation` 0.5.1 and `did2s::did2s` 1.2.1 (with
//! `fixest` 0.14.2), and against the same quantities computed exactly by
//! sparse solves in `tests/data/generate_imputation_reference.R`, on the
//! first 250 units of `df_hom`, on `castle` with its population weights and
//! on three simulated staggered panels (unbalanced, row and unit weights,
//! zero weights, covariates, state clusters, always-treated units, a panel
//! without never-treated units and periods two years apart).
//!
//! # Tolerances
//!
//! The first stage is absorbed with the criterion `tol = 1e-13` and the
//! comparisons with the exact values allow the bounds below; the comparisons
//! with the packages add the distance the script measured between the
//! package and the exact value (`package_error`), which comes from `fixest`
//! stopping its demeaning at `fixef.tol = 1e-6`. With prior weights that
//! vary, `did_imputation`'s standard errors come from its own formula with the
//! squared weights (module documentation of `epistat_did::imputation`), which
//! the script reproduces, so only the exact ones are compared there.
//!
//! - Imputed outcomes. The criterion bounds the error of a demeaned column by
//!   `||e||_W <= tol s sqrt(sum W_qg^2) / sqrt(lambda_D)` (module
//!   documentation of `epistat_glm::absorb`), `s` the column's root mean
//!   square, `W_qg` the weights of the levels and `lambda_D` the smallest
//!   positive eigenvalue of `D0' W D0`. The effects' error `delta` has
//!   `||D0 delta||_W = ||e||_W`, and the error of the sum of one unit's and
//!   one period's effect is at most `sqrt(2 / lambda_D) ||e||_W`. The fixed
//!   effects of the fit are those of `y` minus `beta_j` times those of each
//!   covariate, so with `s_y` and `s_j` the columns' root mean squares every
//!   imputed outcome is within `P = sqrt(2) tol sqrt(sum W_qg^2) / lambda_D
//!   (s_y + sum |beta_j| s_j)`. The error of `beta` is of second order. An
//!   estimate is a mean with weights summing to one, so it is within `P`.
//! - Weights of the variance. The exact `v*` satisfies `Z' v* = 0` over the
//!   rows of the estimation and `v*_0 = -W0 Z0 gamma*`; the returned `v_0`
//!   lies in the span of `W0 Z0` by construction, so with `r = Z' v`
//!   (unit and period sums and covariate products of the returned weights),
//!   `||v_0 - v*_0||_{W^-1} = sqrt(r' M0^+ r) <= ||r|| / sqrt(lambda_F)`,
//!   `lambda_F` the smallest positive eigenvalue of `[X0 D0]' W [X0 D0]`.
//! - Standard errors. The cluster sums `S_c = sum v e~` move by at most
//!   `||r|| / sqrt(lambda_F) sqrt(sum_0 w e~^2) + P (2 + sum_0 |v|)`: the
//!   first term from the weights by Cauchy-Schwarz, the second from the
//!   residuals and the cell means, each within `P` (twice on treated rows).
//!   A standard error, the norm of the sums, moves by no more; a unit's
//!   influence by at most `N` times that.
//! - Two-stage. Each coefficient is a weighted mean of imputed residuals,
//!   within `P`. The projections `h` satisfy `Z0' W h = Z' W x2` exactly, so
//!   with `r` the difference of the two sides `||h - h*||_W <= ||r|| /
//!   sqrt(lambda_F)`, and the cluster sums move by at most `(2 P sum_k w +
//!   ||h - h*||_W sqrt(sum_0 w u^2) + P sum_0 w |h|) / W_k`.
//! - Rounding. The exact values come from Cholesky solves of `M0`, whose
//!   relative error is of order `kappa eps` (Higham, Accuracy and Stability,
//!   thm 10.4), with `kappa = lambda_max / lambda_F`; the sums over `n` rows
//!   add `n eps` relative. `n kappa eps` times the scale covers both.
//! - Pre-trends. The coefficients are those of a `feols` fit, which agree
//!   with the exact projection to `(100 eps + e^2) kappa scale` (tests of
//!   `epistat_glm::feols`), `kappa` the condition number of the demeaned
//!   design from the script and `e` the relative demeaning error, at most
//!   `tol sqrt(sum W_qg^2 / (lambda_D W))`. The standard errors are compared
//!   with `fixest` at `fixef.tol = 1e-11`, whose residuals the script
//!   measured against the exact ones (`tight_error`); the meat is quadratic
//!   in the residuals, so the relative error is at most twice the sum of the
//!   two residual errors, plus the rounding.

use std::collections::HashMap;

use epistat_did::{
    ImputationConfig, ImputationTarget, Pretrends, StaggeredPanel, TwoStageConfig, TwoStageSpec,
    estimate_did_imputation, estimate_did2s,
};
use epistat_glm::Absorb;
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

const TIGHT: Absorb = Absorb {
    tolerance: 1e-13,
    max_iterations: 100_000,
    ..Absorb::new()
};

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize, Clone, Copy)]
struct Lambda {
    d_plus: f64,
    full_plus: f64,
    full_max: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    unit: Vec<u32>,
    time: Vec<i32>,
    g: Vec<Option<i32>>,
    y: Vec<f64>,
    w: Vec<f64>,
    cl: Vec<u32>,
    x: Vec<Vec<f64>>,
    units: Vec<u32>,
    units_unweighted: Vec<u32>,
    dropped: usize,
    dropped_unweighted: usize,
    lambda: Lambda,
    lambda_unweighted: Lambda,
    imputation: Vec<Imputation>,
    did2s: Vec<TwoStage>,
}

#[derive(Deserialize)]
struct PackageError {
    estimate: f64,
    std_error: f64,
}

#[derive(Deserialize)]
struct Exact {
    estimate: Vec<Option<f64>>,
    std_error: Vec<Option<f64>>,
    covariance: Vec<Vec<Option<f64>>>,
    influence: Vec<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct ImputationPackage {
    estimate: Vec<Option<f64>>,
    std_error: Vec<Option<f64>>,
    pre_terms: Vec<i32>,
}

#[derive(Deserialize)]
struct PretrendExact {
    periods: Vec<i32>,
    estimate: Vec<Option<f64>>,
    std_error: Vec<Option<f64>>,
    kappa: f64,
    tight_error: f64,
}

#[derive(Deserialize)]
struct Imputation {
    name: String,
    weighted: bool,
    target: String,
    horizons: Option<Vec<i32>>,
    all_horizons: bool,
    custom: Option<Vec<f64>>,
    pretrends: Option<serde_json::Value>,
    package: ImputationPackage,
    package_error: PackageError,
    exact: Exact,
    pretrend: Option<PretrendExact>,
}

#[derive(Deserialize)]
struct TwoStagePackage {
    estimate: Vec<f64>,
    std_error: Vec<f64>,
}

#[derive(Deserialize)]
struct TwoStage {
    name: String,
    spec: String,
    references: Vec<i32>,
    weighted: bool,
    package: TwoStagePackage,
    package_error: PackageError,
    exact: Exact,
    terms: Option<Vec<i32>>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/imputation_ref.json"
    ))
}

impl Case {
    fn covariates(&self) -> Option<Mat<f64>> {
        (!self.x.is_empty()).then(|| Mat::from_fn(self.y.len(), self.x.len(), |i, j| self.x[j][i]))
    }
}

/// The quantities of the tolerances for one weighting of a case.
struct Bounds {
    /// The rows of the estimation.
    used: Vec<bool>,
    /// The untreated rows of positive weight.
    fit: Vec<bool>,
    weight: Vec<f64>,
    /// `P`, the bound on every imputed outcome.
    p: f64,
    lambda: Lambda,
    /// `n kappa eps`.
    rounding: f64,
    /// `tol sqrt(sum W_qg^2 / (lambda_D W))`.
    relative_demeaning: f64,
    /// The unit and period levels and the covariates of the first stage.
    n_columns: usize,
}

fn rms_centred(values: &[f64], weight: &[f64], fit: &[bool]) -> f64 {
    let (mut sw, mut swv) = (0.0, 0.0);
    for i in 0..values.len() {
        if fit[i] {
            sw += weight[i];
            swv += weight[i] * values[i];
        }
    }
    let mean = swv / sw;
    let mut ss = 0.0;
    for i in 0..values.len() {
        if fit[i] {
            ss += weight[i] * (values[i] - mean).powi(2);
        }
    }
    (ss / sw).sqrt()
}

impl Bounds {
    fn new(case: &Case, weighted: bool, dropped: &[usize], beta: &[Option<f64>]) -> Self {
        let n = case.y.len();
        let weight: Vec<f64> = if weighted {
            case.w.clone()
        } else {
            vec![1.0; n]
        };
        let mut used = vec![true; n];
        for &i in dropped {
            used[i] = false;
        }
        let treated: Vec<bool> = (0..n)
            .map(|i| case.g[i].is_some_and(|g| case.time[i] >= g))
            .collect();
        let fit: Vec<bool> = (0..n).map(|i| !treated[i] && weight[i] > 0.0).collect();
        let mut by_unit: HashMap<u32, f64> = HashMap::new();
        let mut by_time: HashMap<i32, f64> = HashMap::new();
        let mut total = 0.0;
        for i in 0..n {
            if fit[i] {
                *by_unit.entry(case.unit[i]).or_default() += weight[i];
                *by_time.entry(case.time[i]).or_default() += weight[i];
                total += weight[i];
            }
        }
        let sw2: f64 = by_unit
            .values()
            .chain(by_time.values())
            .map(|w| w * w)
            .sum();
        let lambda = if weighted {
            case.lambda
        } else {
            case.lambda_unweighted
        };
        let mut s = rms_centred(&case.y, &weight, &fit);
        for (j, column) in case.x.iter().enumerate() {
            s += beta[j].map_or(0.0, f64::abs) * rms_centred(column, &weight, &fit);
        }
        let tol = TIGHT.tolerance;
        let p = 2f64.sqrt() * tol * sw2.sqrt() / lambda.d_plus * s;
        let kappa = lambda.full_max / lambda.full_plus;
        Self {
            used,
            fit,
            weight,
            p,
            lambda,
            rounding: n as f64 * kappa * EPS,
            relative_demeaning: tol * (sw2 / (lambda.d_plus * total)).sqrt(),
            n_columns: by_unit.len() + by_time.len() + case.x.len(),
        }
    }

    /// `||Z' v||` over the rows of the estimation, with every unit and period
    /// dummy and the covariates.
    fn normal_residual(&self, case: &Case, v: &[f64]) -> f64 {
        let mut by_unit: HashMap<u32, f64> = HashMap::new();
        let mut by_time: HashMap<i32, f64> = HashMap::new();
        let mut by_x = vec![0.0; case.x.len()];
        for i in 0..v.len() {
            if self.used[i] {
                *by_unit.entry(case.unit[i]).or_default() += v[i];
                *by_time.entry(case.time[i]).or_default() += v[i];
                for (j, column) in case.x.iter().enumerate() {
                    by_x[j] += v[i] * column[i];
                }
            }
        }
        by_unit
            .values()
            .chain(by_time.values())
            .chain(by_x.iter())
            .map(|r| r * r)
            .sum::<f64>()
            .sqrt()
    }
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: {got:e} against {want:e}, difference {:e} above {tol:e}",
        (got - want).abs()
    );
}

fn panel<'a>(case: &'a Case, weighted: bool, x: Option<&'a Mat<f64>>) -> StaggeredPanel<'a> {
    StaggeredPanel {
        unit: &case.unit,
        time: &case.time,
        first_treated: &case.g,
        outcome: &case.y,
        weights: weighted.then_some(case.w.as_slice()),
        covariates: x.map(Mat::as_ref),
    }
}

fn pretrend_values(value: &serde_json::Value) -> Option<Vec<i32>> {
    match value {
        serde_json::Value::String(s) if s == "all" => None,
        serde_json::Value::Array(a) => Some(
            a.iter()
                .map(|v| i32::try_from(v.as_i64().unwrap()).unwrap())
                .collect(),
        ),
        other => panic!("pretrends {other}"),
    }
}

#[test]
fn imputation_matches_the_exact_values_and_did_imputation() {
    let reference = reference();
    for case in &reference.cases {
        let x = case.covariates();
        for spec in &case.imputation {
            let weighted = spec.weighted;
            let label = format!("{} {}", case.name, spec.name);
            let horizons: Vec<i32>;
            let custom: Vec<&[f64]>;
            let target = match spec.target.as_str() {
                "static" => ImputationTarget::Static,
                "horizons" => {
                    if spec.all_horizons {
                        ImputationTarget::Horizons(None)
                    } else {
                        horizons = spec.horizons.clone().unwrap();
                        ImputationTarget::Horizons(Some(&horizons))
                    }
                }
                _ => {
                    custom = vec![spec.custom.as_deref().unwrap()];
                    ImputationTarget::Weights(&custom)
                }
            };
            let periods = spec.pretrends.as_ref().map(pretrend_values);
            let pretrends = periods.as_ref().map(|p| match p {
                None => Pretrends::All,
                Some(p) => Pretrends::Periods(p),
            });
            let config = ImputationConfig {
                target,
                pretrends,
                clusters: Some(&case.cl),
                absorb: TIGHT,
                keep_weights: true,
            };
            let out = estimate_did_imputation(&panel(case, weighted, x.as_ref()), &config).unwrap();
            let (dropped, units) = if weighted {
                (case.dropped, &case.units)
            } else {
                (case.dropped_unweighted, &case.units_unweighted)
            };
            assert_eq!(out.dropped_rows.len(), dropped, "{label}");
            assert_eq!(&out.units, units, "{label}");
            let b = Bounds::new(
                case,
                weighted,
                &out.dropped_rows,
                &out.covariate_coefficients,
            );
            let k = out.estimates.len();
            assert_eq!(k, spec.exact.estimate.len(), "{label}");
            let n_units = out.units.len() as f64;
            let mut s_bound = vec![0.0; k];
            for (j, e) in out.estimates.iter().enumerate() {
                let what = format!("{label} term {j}");
                let Some(want) = spec.exact.estimate[j] else {
                    assert!(e.estimate.is_none(), "{what}");
                    continue;
                };
                let got = e.estimate.unwrap();
                let tol = b.p + b.rounding * want.abs().max(1.0);
                close(got, want, tol, &format!("{what} estimate"));
                close(
                    got,
                    spec.package.estimate[j].unwrap(),
                    tol + spec.package_error.estimate,
                    &format!("{what} estimate against the package"),
                );
                // The standard error, from the a-posteriori bound on v.
                let v = &out.weights.as_ref().unwrap()[j];
                let r = b.normal_residual(case, v);
                let (mut adjusted_ss, mut v0) = (0.0, 0.0);
                for i in 0..v.len() {
                    if b.fit[i] && b.used[i] {
                        v0 += v[i].abs();
                        adjusted_ss += b.weight[i] * out.residualised[i].unwrap().powi(2);
                    }
                }
                let bound = r / b.lambda.full_plus.sqrt() * adjusted_ss.sqrt()
                    + b.p * (2.0 + v0)
                    + b.rounding * spec.exact.std_error[j].unwrap();
                s_bound[j] = bound;
                let se = e.std_error.unwrap();
                close(
                    se,
                    spec.exact.std_error[j].unwrap(),
                    bound,
                    &format!("{what} se"),
                );
                if !weighted {
                    close(
                        se,
                        spec.package.std_error[j].unwrap(),
                        bound + spec.package_error.std_error,
                        &format!("{what} se against the package"),
                    );
                }
                let influence = out.influence[j].as_ref().unwrap();
                for (u, value) in influence.iter().enumerate() {
                    close(
                        *value,
                        spec.exact.influence[u][j].unwrap(),
                        n_units * bound,
                        &format!("{what} influence of unit {u}"),
                    );
                }
            }
            for a in 0..k {
                for c in 0..k {
                    if let (Some(want), Some(got)) =
                        (spec.exact.covariance[a][c], out.covariance[a * k + c])
                    {
                        let (sa, sc) = (
                            spec.exact.std_error[a].unwrap(),
                            spec.exact.std_error[c].unwrap(),
                        );
                        let tol = s_bound[a] * sc + sa * s_bound[c] + s_bound[a] * s_bound[c];
                        close(got, want, tol, &format!("{label} covariance {a} {c}"));
                    }
                }
            }
            if let Some(pre) = &spec.pretrend {
                check_pretrends(&label, spec, pre, &out, &b);
            }
        }
    }
}

fn check_pretrends(
    label: &str,
    spec: &Imputation,
    pre: &PretrendExact,
    out: &epistat_did::ImputationResult,
    b: &Bounds,
) {
    let result = out.pretrends.as_ref().unwrap();
    let periods: Vec<i32> = result.terms.iter().map(|t| t.relative_period).collect();
    assert_eq!(periods, pre.periods, "{label} pre-trend periods");
    let kept: Vec<i32> = result
        .terms
        .iter()
        .filter(|t| t.estimate.is_some())
        .map(|t| t.relative_period)
        .collect();
    assert_eq!(kept, spec.package.pre_terms, "{label} pre-trend terms kept");
    let scale = pre
        .estimate
        .iter()
        .flatten()
        .fold(0.0_f64, |a, &v| a.max(v.abs()));
    let e = b.relative_demeaning;
    // lm's Householder QR of the n x m design with every dummy carries an
    // error of order n m eps (Higham, Accuracy and Stability, thm 20.3).
    let n_fit = b.fit.iter().zip(&b.used).filter(|&(&f, &u)| f && u).count();
    let qr = (n_fit * b.n_columns) as f64 * EPS * scale;
    let tol_beta = (100.0 * EPS + e * e) * pre.kappa * scale + qr;
    for (j, term) in result.terms.iter().enumerate() {
        match (term.estimate, pre.estimate[j]) {
            (Some(got), Some(want)) => {
                close(got, want, tol_beta, &format!("{label} pre-trend {j}"));
                let se_want = pre.std_error[j].unwrap();
                let rel = 2.0 * (e + pre.tight_error) + 100.0 * pre.kappa * EPS;
                close(
                    term.std_error.unwrap(),
                    se_want,
                    rel * se_want,
                    &format!("{label} pre-trend se {j}"),
                );
            }
            (None, None) => {}
            other => panic!("{label} pre-trend {j}: {other:?}"),
        }
    }
}

#[test]
fn did2s_matches_the_exact_values_and_the_package() {
    let reference = reference();
    for case in &reference.cases {
        let x = case.covariates();
        let n = case.y.len();
        for spec in &case.did2s {
            let label = format!("{} did2s {}", case.name, spec.name);
            let config = TwoStageConfig {
                spec: if spec.spec == "static" {
                    TwoStageSpec::Static
                } else {
                    TwoStageSpec::EventStudy {
                        references: &spec.references,
                    }
                },
                clusters: Some(&case.cl),
                absorb: TIGHT,
                keep_projections: true,
            };
            let out = estimate_did2s(&panel(case, spec.weighted, x.as_ref()), &config).unwrap();
            let b = Bounds::new(
                case,
                spec.weighted,
                &out.dropped_rows,
                &out.covariate_coefficients,
            );
            let periods: Vec<Option<i32>> = out.terms.iter().map(|t| t.relative_period).collect();
            match &spec.terms {
                None => assert_eq!(periods, vec![None], "{label}"),
                Some(t) => assert_eq!(
                    periods,
                    t.iter().map(|&r| Some(r)).collect::<Vec<_>>(),
                    "{label}"
                ),
            }
            let k_terms = out.terms.len();
            let n_units = out.units.len() as f64;
            let mut s_bound = vec![0.0; k_terms];
            for (k, term) in out.terms.iter().enumerate() {
                let what = format!("{label} term {k}");
                let in_term = |i: usize| -> bool {
                    b.used[i]
                        && match term.relative_period {
                            None => case.g[i].is_some_and(|g| case.time[i] >= g),
                            Some(r) => case.g[i].is_some_and(|g| case.time[i] - g == r),
                        }
                };
                let want = spec.exact.estimate[k].unwrap();
                let got = term.estimate.unwrap();
                let tol = b.p + b.rounding * want.abs().max(1.0);
                close(got, want, tol, &format!("{what} estimate"));
                close(
                    got,
                    spec.package.estimate[k],
                    tol + spec.package_error.estimate,
                    &format!("{what} estimate against the package"),
                );
                // The normal equations of the projection: Z0' W h = Z' W x2.
                let h = &out.projections.as_ref().unwrap()[k];
                let mut lhs = vec![0.0; n];
                for i in 0..n {
                    if in_term(i) {
                        lhs[i] += b.weight[i];
                    }
                    if b.fit[i] && b.used[i] {
                        lhs[i] -= b.weight[i] * h[i];
                    }
                }
                let r = b.normal_residual(case, &lhs);
                let (mut w_k, mut uu, mut wh) = (0.0, 0.0, 0.0);
                for i in 0..n {
                    if in_term(i) {
                        w_k += b.weight[i];
                    }
                    if b.fit[i] && b.used[i] {
                        uu += b.weight[i] * out.residualised[i].unwrap().powi(2);
                        wh += b.weight[i] * h[i].abs();
                    }
                }
                let se_want = spec.exact.std_error[k].unwrap();
                let bound =
                    (2.0 * b.p * w_k + r / b.lambda.full_plus.sqrt() * uu.sqrt() + b.p * wh) / w_k
                        + b.rounding * se_want;
                s_bound[k] = bound;
                let se = term.std_error.unwrap();
                close(se, se_want, bound, &format!("{what} se"));
                close(
                    se,
                    spec.package.std_error[k],
                    bound + spec.package_error.std_error,
                    &format!("{what} se against the package"),
                );
                let influence = out.influence[k].as_ref().unwrap();
                for (u, value) in influence.iter().enumerate() {
                    close(
                        *value,
                        spec.exact.influence[u][k].unwrap(),
                        n_units * bound,
                        &format!("{what} influence of unit {u}"),
                    );
                }
            }
            for a in 0..k_terms {
                for c in 0..k_terms {
                    let want = spec.exact.covariance[a][c].unwrap();
                    let got = out.covariance[a * k_terms + c].unwrap();
                    let (sa, sc) = (
                        spec.exact.std_error[a].unwrap(),
                        spec.exact.std_error[c].unwrap(),
                    );
                    let tol = s_bound[a] * sc + sa * s_bound[c] + s_bound[a] * s_bound[c];
                    close(got, want, tol, &format!("{label} covariance {a} {c}"));
                }
            }
        }
    }
}

#[test]
fn did2s_and_imputation_agree_on_the_effects_after_treatment() {
    // Both estimators take their first stage from the same fit and average
    // the same imputed effects with the prior weights, so they agree to the
    // rounding of a weighted mean, n eps times the largest |tau|.
    let reference = reference();
    for case in &reference.cases {
        let x = case.covariates();
        for weighted in [false, true] {
            let p = panel(case, weighted, x.as_ref());
            let two = estimate_did2s(
                &p,
                &TwoStageConfig {
                    absorb: TIGHT,
                    ..TwoStageConfig::event_study()
                },
            )
            .unwrap();
            let imp = estimate_did_imputation(
                &p,
                &ImputationConfig {
                    absorb: TIGHT,
                    ..ImputationConfig::event_study()
                },
            )
            .unwrap();
            let largest = imp
                .residualised
                .iter()
                .flatten()
                .fold(0.0_f64, |a, &v| a.max(v.abs()));
            let tol = case.y.len() as f64 * EPS * largest;
            let mut compared = 0;
            for term in &two.terms {
                let Some(r) = term.relative_period.filter(|&r| r >= 0) else {
                    continue;
                };
                let e = imp
                    .estimates
                    .iter()
                    .find(|e| e.term == epistat_did::ImputationTerm::Horizon(r))
                    .unwrap();
                match (term.estimate, e.estimate) {
                    (Some(a), Some(b)) => close(a, b, tol, &format!("{} horizon {r}", case.name)),
                    (None, None) => {}
                    other => panic!("{} horizon {r}: {other:?}", case.name),
                }
                compared += 1;
            }
            assert!(compared > 0, "{}", case.name);
            let two_static = estimate_did2s(
                &p,
                &TwoStageConfig {
                    absorb: TIGHT,
                    ..TwoStageConfig::static_effect()
                },
            )
            .unwrap();
            let imp_static = estimate_did_imputation(
                &p,
                &ImputationConfig {
                    absorb: TIGHT,
                    ..ImputationConfig::static_effect()
                },
            )
            .unwrap();
            close(
                two_static.terms[0].estimate.unwrap(),
                imp_static.estimates[0].estimate.unwrap(),
                tol,
                &format!("{} static", case.name),
            );
        }
    }
}

#[test]
fn did_imputation_understates_the_weighted_variance_on_castle() {
    // With population weights the package's variance uses the squared
    // weights in its auxiliary system (module documentation). On castle its
    // standard error of the static effect is 0.0224 where the derivative of
    // its own estimate gives 0.0268; the parity test checks that
    // estimate_did_imputation gives the latter.
    let reference = reference();
    let case = reference.cases.iter().find(|c| c.name == "castle").unwrap();
    let spec = case
        .imputation
        .iter()
        .find(|s| s.name == "static_weighted")
        .unwrap();
    let package = spec.package.std_error[0].unwrap();
    let exact = spec.exact.std_error[0].unwrap();
    assert!(exact / package > 1.15, "{exact} {package}");
}
