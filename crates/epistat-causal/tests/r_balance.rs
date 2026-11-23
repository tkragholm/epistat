//! Entropy balancing and CBPS checked against R: `WeightIt::weightit(method =
//! "ebal")`, `ebal::ebalance`, `CBPS::CBPS`, and solvers written out in R with
//! numerical derivatives (`data/balance.R`), the infinitesimal jackknife of the
//! whole procedure for the stacked variance, and a bootstrap that refits the
//! weights.
//!
//! # The tolerances
//!
//! Two solutions of one balance problem differ by what their residuals allow.
//! For entropy balancing, with `u` the standardised columns, the multipliers
//! solve `g(lambda) = 0` with the Jacobian `H`, the weighted covariance of `u`,
//! whose smallest eigenvalue `lmin` R records. A residual of 2-norm `r` moves
//! the multipliers by at most `r / lmin` to first order, and the log weight of
//! person `i` by `|u_i| r / lmin`, plus the same for the normalising constant,
//! which is bounded by the largest `|u|`. The residual of each side is its
//! largest standardised imbalance times `sqrt(p)`. The bound is doubled for
//! second-order terms and `64 n eps` added for the sums.
//!
//! For CBPS the fitted linear predictor of person `i` differs between two points
//! by `x_i' (A^-1 g)`, at most `|A^-T x_i| |g|` with `A` the Jacobian of the
//! condition (the balance conditions, the Gauss-Newton matrix `G'WG` of the
//! package's weighting matrix, or the Hessian of the continuously updated
//! objective). R records `|A^-T x_i|` and `|g|` at the package's result. A
//! probability moves by at most a quarter of that. The package scales the
//! logistic fit by a factor from `optimize` to place its weighting matrix, so
//! its minimum is that of another problem, and R records the distance to the
//! minimum with its matrix, which is added.
//!
//! The jackknife is found by central differences of the case weight of each
//! person with Richardson extrapolation, each refit solving to a gradient of
//! 1e-15. The rounding of a refit, `3e-13` for a value of order 1, enters each
//! influence as `3e-13 / h`, so a covariance entry differs by at most `sqrt(n)
//! e (sqrt(V_aa) + sqrt(V_bb)) + n e^2` with `e = 3e-13 / h`.
//!
//! The stacked variance treats a sample weight as a probability weight: the score
//! of person `i` carries the factor `s_i`. The jackknife is the derivative with
//! respect to the case weight, which is the influence without that factor, so the
//! test multiplies it by `s_i`.
//!
//! The bootstrap standard error has the Monte Carlo error `se / sqrt(2 (B - 1))`;
//! the comparison allows five of it.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the cluster labels written by R are small whole numbers"
)]

use epistat_causal::aipw::Target;
use epistat_causal::balance::{
    CbpsInput, CbpsOptions, EntropyInput, EntropyOptions, Identification, WeightVariance,
    balance_columns, cbps, entropy_balance,
};
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;
const H: f64 = 1e-4;
/// Two solutions of the same condition differ by the condition's residual over its
/// curvature. The own solvers stop at a gradient of 1e-14 and use numerical
/// derivatives accurate to about 1e-12, the solvers here at 1e-12, with `hx` of at
/// most 35: 1e-9 on a probability.
const OWN_TOLERANCE: f64 = 1e-9;
/// The continuously updated objective is minimised in R with a numerical gradient
/// whose noise is about 1e-10, over a curvature as flat as `hx` of 17.
const OWN_TOLERANCE_UPDATED: f64 = 1e-8;
/// The error of one refit of the own solvers of R, in the value of a coefficient of
/// order 1. The balance conditions are solved to a score of 1e-15, so the refit is
/// accurate to 3e-13. The over-identified minimum is found by Newton steps on a
/// numerical gradient that stops at 1e-12, over a curvature of at least 0.2, which
/// leaves 5e-12 in the coefficients and, times a sensitivity of at most 6, 3e-11.
const REFIT_ERROR_JUST: f64 = 3e-13;
const REFIT_ERROR_OVER: f64 = 3e-11;
/// The continuously updated refits of `data/balance_cu.R` solve the first-order
/// condition with a closed-form gradient to 1e-15, as the balance conditions
/// are solved, so they are accurate to the same 3e-13.
const REFIT_ERROR_UPDATED: f64 = 3e-13;

#[derive(Deserialize)]
struct Versions {
    r: String,
    #[serde(rename = "WeightIt")]
    weightit: String,
    #[serde(rename = "CBPS")]
    cbps: String,
    ebal: String,
}

#[derive(Deserialize)]
struct Data {
    a: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    x3: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
    q: Vec<f64>,
    cl: Vec<f64>,
}

impl Data {
    fn treatment(&self) -> Vec<bool> {
        self.a.iter().map(|&a| a > 0.5).collect()
    }
    fn raw(&self) -> Mat<f64> {
        Mat::from_fn(self.a.len(), 3, |i, j| match j {
            0 => self.x1[i],
            1 => self.x2[i],
            _ => self.x3[i],
        })
    }
    fn design(&self) -> Mat<f64> {
        Mat::from_fn(self.a.len(), 4, |i, j| match j {
            0 => 1.0,
            1 => self.x1[i],
            2 => self.x2[i],
            _ => self.x3[i],
        })
    }
}

#[derive(Deserialize)]
struct EbalCase {
    name: String,
    estimand: String,
    moments: usize,
    int: bool,
    use_s: bool,
    use_q: bool,
    weights: Vec<f64>,
    weights_own: Vec<f64>,
    imbalance: f64,
    imbalance_own: f64,
    lambda_min: f64,
    unorm: Vec<f64>,
    ncol: usize,
}

#[derive(Deserialize)]
struct EbalPkg {
    weights_controls: Vec<f64>,
    tol: f64,
    sd_treated: Vec<f64>,
}

#[derive(Deserialize)]
struct Infeasible {
    x3: Vec<f64>,
    target_mean: f64,
    max_controls: f64,
    #[serde(deserialize_with = "epistat_testkit::na_f64")]
    weightit_imbalance: f64,
}

#[derive(Deserialize)]
struct Ij {
    estimand: String,
    #[serde(default)]
    ident: Option<String>,
    #[serde(default)]
    std: Option<bool>,
    theta: Vec<f64>,
    influence: Vec<f64>,
    k: usize,
}

#[derive(Deserialize)]
struct Boot {
    estimand: String,
    #[serde(default)]
    ident: Option<String>,
    estimate: f64,
    #[serde(rename = "se_boot")]
    se: f64,
    #[serde(rename = "B")]
    b: usize,
}

#[derive(Deserialize)]
struct CbpsCase {
    name: String,
    att: String,
    ident: String,
    twostep: bool,
    std: bool,
    use_s: bool,
    coef_own: Vec<f64>,
    fitted_pkg: Vec<f64>,
    weights_pkg: Vec<f64>,
    fitted_own: Vec<f64>,
    weights_own: Vec<f64>,
    #[serde(deserialize_with = "epistat_testkit::na_f64")]
    #[serde(rename = "J_own")]
    j_own: f64,
    hx: Vec<f64>,
    resid: f64,
    w_effect: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    main: Data,
    ebal: Vec<EbalCase>,
    ebal_pkg: EbalPkg,
    infeasible: Infeasible,
    ij_data: Data,
    ij_ebal: Vec<Ij>,
    ij_ebal_means: Vec<Ij>,
    ij_cbps: Vec<Ij>,
    boot_data: Data,
    boot_ebal: Vec<Boot>,
    boot_cbps: Vec<Boot>,
    cbps: Vec<CbpsCase>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/balance.json"
    ))
}

fn target(name: &str) -> Target {
    match name {
        "ATT" => Target::Att,
        "ATC" => Target::Atc,
        _ => Target::Ate,
    }
}

fn opts(name: &str) -> EntropyOptions {
    EntropyOptions {
        target: target(name),
        tolerance: 1e-12,
        max_iterations: 100,
    }
}

#[test]
fn the_reference_records_its_versions() {
    let r = reference();
    assert!(r.versions.r.starts_with("R version 4."), "{}", r.versions.r);
    assert_eq!(r.versions.weightit, "2.1.0");
    assert_eq!(r.versions.cbps, "0.24");
    assert_eq!(r.versions.ebal, "0.2.1");
}

fn assert_weights(
    got: &[f64],
    reference: &[f64],
    case: &EbalCase,
    rust_imbalance: f64,
    imb_ref: f64,
    what: &str,
) {
    let n = got.len() as f64;
    let p = case.ncol as f64;
    let umax = case.unorm.iter().copied().fold(0.0, f64::max);
    let residual = p.sqrt() * (imb_ref + rust_imbalance);
    let mut worst = 0.0_f64;
    for i in 0..got.len() {
        let bound = 2.0 * residual / case.lambda_min * (case.unorm[i] + umax) + 64.0 * n * EPS;
        let gap = (got[i] - reference[i]).abs() / reference[i];
        worst = worst.max(gap / bound);
        assert!(
            gap <= bound,
            "{} {what}: person {i}: {} against {} (relative gap {gap:e}, bound {bound:e})",
            case.name,
            got[i],
            reference[i]
        );
    }
    eprintln!("{} {what}: worst gap over bound {worst:.3e}", case.name);
}

#[test]
fn entropy_balancing_matches_weightit_and_the_own_solver() {
    let r = reference();
    let d = &r.main;
    let t = d.treatment();
    for case in &r.ebal {
        let cols = balance_columns(d.raw().as_ref(), case.moments, case.int).unwrap();
        assert_eq!(cols.ncols(), case.ncol, "{}", case.name);
        let res = entropy_balance(
            &EntropyInput {
                treatment: &t,
                covariates: cols.as_ref(),
                sample_weight: case.use_s.then_some(&d.s[..]),
                base_weight: case.use_q.then_some(&d.q[..]),
            },
            &opts(&case.estimand),
        )
        .unwrap();
        assert!(res.max_imbalance <= 1e-12, "{}", case.name);
        assert_weights(
            &res.weights.weights,
            &case.weights,
            case,
            res.max_imbalance,
            case.imbalance,
            "WeightIt",
        );
        assert_weights(
            &res.weights.weights,
            &case.weights_own,
            case,
            res.max_imbalance,
            case.imbalance_own,
            "own",
        );
        assert!(
            case.imbalance <= 1e-6,
            "{}: WeightIt imbalance {}",
            case.name,
            case.imbalance
        );
    }
}

#[test]
fn entropy_balancing_matches_the_ebal_package() {
    let r = reference();
    let d = &r.main;
    let t = d.treatment();
    let case = r.ebal.iter().find(|c| c.name == "att_m1_q").unwrap();
    let res = entropy_balance(
        &EntropyInput {
            treatment: &t,
            covariates: d.raw().as_ref(),
            sample_weight: None,
            base_weight: Some(&d.q),
        },
        &opts("ATT"),
    )
    .unwrap();
    let controls: Vec<f64> = (0..t.len())
        .filter(|&i| !t[i])
        .map(|i| res.weights.weights[i])
        .collect();
    // ebal's constraint tolerance is on the raw means.
    let sd_min = r
        .ebal_pkg
        .sd_treated
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let imb_ref = r.ebal_pkg.tol / sd_min;
    let residual = 3.0_f64.sqrt() * (imb_ref + res.max_imbalance);
    let umax = case.unorm.iter().copied().fold(0.0, f64::max);
    let idx: Vec<usize> = (0..t.len()).filter(|&i| !t[i]).collect();
    for (k, &i) in idx.iter().enumerate() {
        let bound = 2.0 * residual / case.lambda_min * (case.unorm[i] + umax);
        let gap =
            (controls[k] - r.ebal_pkg.weights_controls[k]).abs() / r.ebal_pkg.weights_controls[k];
        assert!(gap <= bound, "control {k}: gap {gap:e}, bound {bound:e}");
    }
}

#[test]
fn an_unreachable_target_is_refused_where_weightit_returns_unbalanced_weights() {
    let r = reference();
    let d = &r.main;
    let t = d.treatment();
    assert!(r.infeasible.max_controls < r.infeasible.target_mean);
    assert!(
        r.infeasible.weightit_imbalance > 1e-3,
        "{}",
        r.infeasible.weightit_imbalance
    );
    let c = Mat::from_fn(t.len(), 3, |i, j| match j {
        0 => d.x1[i],
        1 => d.x2[i],
        _ => r.infeasible.x3[i],
    });
    let err = entropy_balance(
        &EntropyInput {
            treatment: &t,
            covariates: c.as_ref(),
            sample_weight: None,
            base_weight: None,
        },
        &opts("ATT"),
    )
    .unwrap_err();
    assert!(
        matches!(err, epistat_causal::Error::Infeasible(ref m) if m.contains("column 2")),
        "{err}"
    );
}

fn influence_covariance(ij: &Ij, weight: &[f64], cluster: Option<&[f64]>) -> Vec<f64> {
    let n = ij.influence.len() / ij.k;
    // The jackknife is the derivative with respect to the case weight, which
    // leaves out the factor s_i that the weight puts on the person's score.
    let col = |a: usize, i: usize| weight[i] * ij.influence[a * n + i];
    let mut out = vec![0.0; ij.k * ij.k];
    let (labels, factor): (Vec<usize>, f64) = match cluster {
        None => ((0..n).collect(), 1.0),
        Some(c) => {
            let mut distinct: Vec<i64> = c.iter().map(|&x| x as i64).collect();
            distinct.sort_unstable();
            distinct.dedup();
            let g = distinct.len() as f64;
            (
                c.iter()
                    .map(|&x| distinct.binary_search(&(x as i64)).unwrap())
                    .collect(),
                g / (g - 1.0),
            )
        }
    };
    let groups = labels.iter().copied().max().unwrap() + 1;
    let sums = |a: usize| {
        let mut s = vec![0.0; groups];
        for i in 0..n {
            s[labels[i]] += col(a, i);
        }
        s
    };
    for a in 0..ij.k {
        for b in 0..ij.k {
            let (sa, sb) = (sums(a), sums(b));
            out[a * ij.k + b] = factor * sa.iter().zip(&sb).map(|(x, y)| x * y).sum::<f64>();
        }
    }
    out
}

fn assert_covariance(
    got_coef: &[f64],
    got_cov: &[f64],
    ij: &Ij,
    weight: &[f64],
    cluster: Option<&[f64]>,
    refit_error: f64,
    label: &str,
) {
    let n = ij.influence.len() / ij.k;
    let expected = influence_covariance(ij, weight, cluster);
    let e = refit_error / H;
    for a in 0..ij.k {
        assert!(
            (got_coef[a] - ij.theta[a]).abs() <= 1e-9 * ij.theta[a].abs().max(1.0),
            "{label}: coefficient {a}: {} against {}",
            got_coef[a],
            ij.theta[a]
        );
        for b in 0..ij.k {
            let (va, vb) = (expected[a * ij.k + a].sqrt(), expected[b * ij.k + b].sqrt());
            let bound = (n as f64).sqrt() * e * (va + vb)
                + n as f64 * e * e
                + 64.0 * n as f64 * EPS * expected[a * ij.k + b].abs();
            let gap = (got_cov[a * ij.k + b] - expected[a * ij.k + b]).abs();
            assert!(
                gap <= bound,
                "{label}: covariance ({a},{b}): {} against {}, gap {gap:e}, bound {bound:e}",
                got_cov[a * ij.k + b],
                expected[a * ij.k + b]
            );
        }
    }
}

#[test]
fn the_stacked_variance_of_entropy_balancing_is_the_infinitesimal_jackknife() {
    let r = reference();
    let d = &r.ij_data;
    let t = d.treatment();
    let n = t.len();
    let reg = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => d.a[i],
        _ => d.x1[i],
    });
    let means = Mat::from_fn(n, 2, |i, j| f64::from(u8::from((d.a[i] > 0.5) == (j == 1))));
    for (ijs, design) in [(&r.ij_ebal, &reg), (&r.ij_ebal_means, &means)] {
        for ij in ijs {
            let res = entropy_balance(
                &EntropyInput {
                    treatment: &t,
                    covariates: d.raw().as_ref(),
                    sample_weight: Some(&d.s),
                    base_weight: Some(&d.q),
                },
                &opts(&ij.estimand),
            )
            .unwrap();
            let fit = res
                .weights
                .weighted_regression(&d.y, design.as_ref(), None, WeightVariance::Estimated)
                .unwrap();
            assert_covariance(
                &fit.coefficients,
                &fit.covariance,
                ij,
                &d.s,
                None,
                REFIT_ERROR_JUST,
                &format!("ebal {}", ij.estimand),
            );
            let cl: Vec<u32> = d.cl.iter().map(|&x| x as u32).collect();
            let fit = res
                .weights
                .weighted_regression(&d.y, design.as_ref(), Some(&cl), WeightVariance::Estimated)
                .unwrap();
            assert_covariance(
                &fit.coefficients,
                &fit.covariance,
                ij,
                &d.s,
                Some(&d.cl),
                REFIT_ERROR_JUST,
                &format!("ebal {} clustered", ij.estimand),
            );
        }
    }
}

fn cbps_target(name: &str) -> Target {
    target(name)
}

fn cbps_opts(case_att: &str, ident: &str, twostep: bool, std: bool) -> CbpsOptions {
    CbpsOptions {
        target: cbps_target(case_att),
        identification: if ident == "exact" {
            Identification::Exact
        } else {
            Identification::Over
        },
        two_step: twostep,
        standardise: std,
        tolerance: 1e-12,
        max_iterations: 500,
    }
}

#[test]
fn cbps_matches_the_package_and_the_own_solver() {
    let r = reference();
    let d = &r.main;
    let t = d.treatment();
    let x = d.design();
    let n = t.len();
    for case in &r.cbps {
        let res = cbps(
            &CbpsInput {
                treatment: &t,
                design: x.as_ref(),
                sample_weight: case.use_s.then_some(&d.s[..]),
            },
            &cbps_opts(&case.att, &case.ident, case.twostep, case.std),
        )
        .unwrap();
        let k = 4.0_f64;
        let flip = case.att == "ATC";
        let pkg: Vec<f64> = case
            .fitted_pkg
            .iter()
            .map(|&p| if flip { 1.0 - p } else { p })
            .collect();
        // Distance to the package: its own residual, ours, and the weighting matrix.
        let mut worst_pkg = 0.0_f64;
        let mut worst_own = 0.0_f64;
        let own_tolerance = if case.twostep {
            OWN_TOLERANCE
        } else {
            OWN_TOLERANCE_UPDATED
        };
        let resid = case.resid * 1.5 + k.sqrt() * 1e-12;
        // The linear predictor of person i can differ by this much from the package's.
        let deta: Vec<f64> = (0..n)
            .map(|i| {
                let e = case.fitted_own[i];
                case.hx[i] * resid + case.w_effect[i] / (e * (1.0 - e))
            })
            .collect();
        let deta_max = deta.iter().copied().fold(0.0, f64::max);
        for i in 0..n {
            let bound = 0.25 * deta[i] + 64.0 * n as f64 * EPS;
            let gap = (res.propensity[i] - pkg[i]).abs();
            worst_pkg = worst_pkg.max(gap / bound);
            assert!(
                gap <= bound,
                "{} person {i}: {} against package {} (gap {gap:e}, bound {bound:e})",
                case.name,
                res.propensity[i],
                pkg[i]
            );
            // A weight is `exp` of the linear predictor up to its normalisation.
            let wg = (res.weights.weights[i] - case.weights_pkg[i]).abs() / case.weights_pkg[i];
            let wb = deta[i] + deta_max + 64.0 * n as f64 * EPS;
            assert!(
                wg <= wb,
                "{} person {i}: weight gap {wg:e}, bound {wb:e}",
                case.name
            );
            // The own solver of R, to the tolerance of the solvers.
            let og = (res.propensity[i] - case.fitted_own[i]).abs();
            worst_own = worst_own.max(og);
            assert!(
                og <= own_tolerance,
                "{} person {i}: own solver gap {og:e}",
                case.name
            );
            let owg = (res.weights.weights[i] - case.weights_own[i]).abs() / case.weights_own[i];
            assert!(
                owg <= 10.0 * own_tolerance,
                "{} person {i}: own solver weight gap {owg:e}",
                case.name
            );
        }
        eprintln!(
            "{}: worst gap over bound (package) {worst_pkg:.3e}, worst gap to own solver {worst_own:.3e}",
            case.name
        );
        if case.ident == "over" {
            let j = res.objective.unwrap();
            assert!(
                (j - case.j_own).abs() <= 1e-9 * case.j_own.abs().max(1e-12),
                "{}: J {j} against {}",
                case.name,
                case.j_own
            );
        }
        for (a, b) in res.coefficients.iter().zip(&case.coef_own) {
            let b = if flip { -b } else { *b };
            assert!(
                (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                "{}: coefficient {a} against {b}",
                case.name
            );
        }
    }
}

#[test]
fn the_stacked_variance_of_cbps_is_the_infinitesimal_jackknife() {
    let r = reference();
    let d = &r.ij_data;
    let t = d.treatment();
    let n = t.len();
    let x = d.design();
    let reg = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => d.a[i],
        _ => d.x1[i],
    });
    for ij in &r.ij_cbps {
        let res = cbps(
            &CbpsInput {
                treatment: &t,
                design: x.as_ref(),
                sample_weight: Some(&d.s),
            },
            &cbps_opts(
                &ij.estimand,
                ij.ident.as_deref().unwrap(),
                true,
                ij.std.unwrap(),
            ),
        )
        .unwrap();
        let fit = res
            .weights
            .weighted_regression(&d.y, reg.as_ref(), None, WeightVariance::Estimated)
            .unwrap();
        assert_covariance(
            &fit.coefficients,
            &fit.covariance,
            ij,
            &d.s,
            None,
            if ij.ident.as_deref() == Some("over") {
                REFIT_ERROR_OVER
            } else {
                REFIT_ERROR_JUST
            },
            &format!("cbps {} {}", ij.estimand, ij.ident.as_deref().unwrap()),
        );
    }
}

#[derive(Deserialize)]
struct UpdatedVersions {
    r: String,
    #[serde(rename = "numDeriv")]
    numderiv: String,
}

#[derive(Deserialize)]
struct UpdatedReference {
    versions: UpdatedVersions,
    ij_cbps_cu: Vec<Ij>,
}

/// The continuously updated CBPS against the infinitesimal jackknife of its
/// whole procedure (`data/balance_cu.R`, on the jackknife data of `balance.R`),
/// the weighting matrix updated with the coefficients and the case weights.
#[test]
fn the_stacked_variance_of_the_continuously_updated_cbps_is_the_infinitesimal_jackknife() {
    let r = reference();
    let updated: UpdatedReference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/balance_cu.json"
    ));
    assert!(updated.versions.r.starts_with("R version 4."));
    assert_ne!(updated.versions.numderiv, "");
    let d = &r.ij_data;
    let t = d.treatment();
    let n = t.len();
    let x = d.design();
    let reg = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => d.a[i],
        _ => d.x1[i],
    });
    for ij in &updated.ij_cbps_cu {
        let res = cbps(
            &CbpsInput {
                treatment: &t,
                design: x.as_ref(),
                sample_weight: Some(&d.s),
            },
            &cbps_opts(&ij.estimand, "over", false, ij.std.unwrap()),
        )
        .unwrap();
        let fit = res
            .weights
            .weighted_regression(&d.y, reg.as_ref(), None, WeightVariance::Estimated)
            .unwrap();
        assert_covariance(
            &fit.coefficients,
            &fit.covariance,
            ij,
            &d.s,
            None,
            REFIT_ERROR_UPDATED,
            &format!("cbps {} continuously updated", ij.estimand),
        );
    }
}

#[test]
fn the_stacked_standard_error_agrees_with_a_bootstrap_that_refits_the_weights() {
    let r = reference();
    let d = &r.boot_data;
    let t = d.treatment();
    let x = d.design();
    for b in &r.boot_ebal {
        let res = entropy_balance(
            &EntropyInput {
                treatment: &t,
                covariates: d.raw().as_ref(),
                sample_weight: None,
                base_weight: None,
            },
            &opts(&b.estimand),
        )
        .unwrap();
        let md = res
            .weights
            .mean_difference(&d.y, None, WeightVariance::Estimated, 0.05)
            .unwrap();
        check_boot(&md.difference, b, "ebal");
    }
    for b in &r.boot_cbps {
        let res = cbps(
            &CbpsInput {
                treatment: &t,
                design: x.as_ref(),
                sample_weight: None,
            },
            &cbps_opts(&b.estimand, b.ident.as_deref().unwrap(), true, true),
        )
        .unwrap();
        let md = res
            .weights
            .mean_difference(&d.y, None, WeightVariance::Estimated, 0.05)
            .unwrap();
        check_boot(&md.difference, b, "cbps");
    }
}

fn check_boot(e: &epistat_causal::aipw::Estimate, b: &Boot, what: &str) {
    assert!(
        (e.estimate - b.estimate).abs() <= 1e-9,
        "{what}: {} against {}",
        e.estimate,
        b.estimate
    );
    let mce = b.se / (2.0 * (b.b as f64 - 1.0)).sqrt();
    let gap = (e.se - b.se).abs();
    eprintln!(
        "{what} {}: se {} bootstrap {} ({} Monte Carlo errors)",
        b.estimand,
        e.se,
        b.se,
        gap / mce
    );
    assert!(
        gap <= 5.0 * mce,
        "{what} {}: {} against {} (Monte Carlo error {mce})",
        b.estimand,
        e.se,
        b.se
    );
}
