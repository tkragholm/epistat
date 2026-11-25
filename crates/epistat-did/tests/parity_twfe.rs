//! Two-way fixed-effects DiD, static and event study, checked against
//! `fixest::feols` 0.14.2 (`tests/data/generate_twfe_reference.R`).

use epistat_did::{
    TwfeConfig, TwfePanel, TwfeSpec, did_covariance, did_standard_error, estimate_twfe,
};
use epistat_glm::{Absorb, Feols, FixedEffects, Singletons, Ssc, Vcov};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Spec {
    kind: String,
    reference: Option<i32>,
    lo: Option<i32>,
    hi: Option<i32>,
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
    covariate: Option<Vec<f64>>,
    clusters: Option<Vec<u32>>,
    names: Vec<String>,
    coef: Vec<f64>,
    se: Vec<f64>,
    vcov: Vec<f64>,
    vcov_unadjusted: Vec<f64>,
    df_k: usize,
    nobs: usize,
    collinear: Vec<String>,
    exact: Exact,
}

#[derive(Deserialize)]
struct Exact {
    /// fixest's largest deviation from the exact projection over each
    /// column's root mean square.
    error: f64,
    /// The smallest positive eigenvalue of D'WD.
    lambda_plus: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/twfe_ref.json"
    ))
}

const EPS: f64 = f64::EPSILON;
const TIGHT: Absorb = Absorb {
    tolerance: 1e-13,
    max_iterations: 100_000,
};

fn covariate(case: &Case) -> Option<Mat<f64>> {
    case.covariate
        .as_ref()
        .map(|x| Mat::from_fn(x.len(), 1, |i, _| x[i]))
}

fn config(case: &Case) -> TwfeConfig<'_> {
    let spec = if case.spec.kind == "static" {
        TwfeSpec::Static
    } else {
        TwfeSpec::EventStudy {
            reference: case.spec.reference.unwrap(),
            bins: case.spec.lo.zip(case.spec.hi),
        }
    };
    TwfeConfig {
        spec,
        clusters: case.clusters.as_deref(),
        absorb: TIGHT,
        ssc: Ssc::new(),
    }
}

fn panel<'a>(case: &'a Case, x: Option<&'a Mat<f64>>) -> TwfePanel<'a> {
    TwfePanel {
        unit: &case.unit,
        time: &case.time,
        first_treated: &case.first_treated,
        outcome: &case.outcome,
        weights: case.weights.as_deref(),
        covariates: x.map(Mat::as_ref),
    }
}

/// The name fixest gives a term.
fn name(case: &Case, period: Option<i32>) -> String {
    match period {
        None => "treat".to_string(),
        Some(r) => {
            let prefix = if case.spec.lo.is_some() {
                "relb"
            } else {
                "rel"
            };
            format!("{prefix}::{r}")
        }
    }
}

#[test]
fn twfe_matches_fixest() {
    for case in &reference().cases {
        let x = covariate(case);
        let result = estimate_twfe(&panel(case, x.as_ref()), &config(case)).unwrap();
        assert_eq!(result.n_rows, case.nobs, "{}", case.name);
        assert_eq!(result.df_k, case.df_k, "{}", case.name);
        // The terms and covariates in the order of the result, with fixest's
        // position of each kept one.
        let mut labels: Vec<String> = result
            .terms
            .iter()
            .map(|t| name(case, t.relative_period))
            .collect();
        if case.covariate.is_some() {
            labels.push("x1".to_string());
        }
        let p = labels.len();
        let estimates: Vec<Option<f64>> = result
            .terms
            .iter()
            .map(|t| t.estimate)
            .chain(result.covariate_coefficients.iter().copied())
            .collect();
        let position: Vec<Option<usize>> = labels
            .iter()
            .map(|l| case.names.iter().position(|n| n == l))
            .collect();
        for (j, l) in labels.iter().enumerate() {
            assert_eq!(
                estimates[j].is_none(),
                case.collinear.contains(l),
                "{} {l} dropped",
                case.name
            );
            assert_eq!(
                estimates[j].is_some(),
                position[j].is_some(),
                "{} {l}",
                case.name
            );
        }

        // The tolerances follow the derivation of epistat-glm's r_feols
        // test. Each side's demeaned columns are off by at most e times the
        // column's root mean square per row: fixest's measured error, and
        // this side's bound from its criterion (see `covariance_bound`). The
        // coefficients move at second order in e, the clustered covariance
        // at first order.
        let (bound, e, kappa) = covariance_bound(case, x.as_ref());
        let largest = case.coef.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        // The rounding of two least-squares solutions, 100 kappa eps, as in
        // r_feols, and the demeaning error at second order.
        let tol_beta = (100.0 * EPS + e * e) * kappa * largest;
        for j in 0..p {
            if let (Some(a), Some(k)) = (estimates[j], position[j]) {
                assert!(
                    (a - case.coef[k]).abs() <= tol_beta,
                    "{} {}: {a} {}",
                    case.name,
                    labels[j],
                    case.coef[k]
                );
            }
        }
        let kept: Vec<usize> = (0..p).filter(|&j| position[j].is_some()).collect();
        let pk = case.names.len();
        for (a, &i) in kept.iter().enumerate() {
            for (b, &j) in kept.iter().enumerate() {
                let got = result.covariance[i * p + j].unwrap();
                let want = case.vcov[position[i].unwrap() * pk + position[j].unwrap()];
                assert!(
                    (got - want).abs() <= bound[a * kept.len() + b],
                    "{} vcov {} {}: {got:e} {want:e}",
                    case.name,
                    labels[i],
                    labels[j]
                );
            }
            let se = result.terms.get(i).and_then(|t| t.std_error);
            if let Some(se) = se {
                let want = case.se[position[i].unwrap()];
                assert!(
                    (se * se - want * want).abs() <= bound[a * kept.len() + a],
                    "{} se",
                    case.name
                );
            }
        }

        // The influence functions: summed within the clusters they give
        // fixest's covariance without its small-sample factors. With unit
        // clusters directly; with coarser clusters through the clusters of
        // the units.
        let unit_cluster: Vec<u32> = result
            .units
            .iter()
            .map(|&u| {
                let row = case.unit.iter().position(|&v| v == u).unwrap();
                case.clusters.as_ref().map_or(u, |c| c[row])
            })
            .collect();
        let clusters = Clusters::from_labels(&unit_cluster);
        let psi: Vec<Vec<f64>> = kept
            .iter()
            .map(|&j| result.influence[j].clone().unwrap())
            .collect();
        let raw = did_covariance(&psi, &clusters).unwrap();
        let factor_free: f64 = {
            // fixest's factor of this covariance over the unadjusted one.
            let g = clusters.count() as f64;
            let n = case.nobs as f64;
            g / (g - 1.0) * (n - 1.0) / (n - case.df_k as f64)
        };
        for (a, &i) in kept.iter().enumerate() {
            for (b, &j) in kept.iter().enumerate() {
                let want = case.vcov_unadjusted[position[i].unwrap() * pk + position[j].unwrap()];
                let got = raw[a * kept.len() + b];
                // The influence functions sum to zero up to rounding, so
                // centring them changes nothing above the bound.
                assert!(
                    (got - want).abs() <= bound[a * kept.len() + b] / factor_free,
                    "{} influence {} {}: {got:e} {want:e}",
                    case.name,
                    labels[i],
                    labels[j]
                );
            }
        }
        // The diagonal of did_covariance is the square of did_standard_error
        // up to the rounding of the square root.
        let se = did_standard_error(&psi[0], &clusters).unwrap();
        assert!(
            (se * se - raw[0]).abs() <= 4.0 * EPS * raw[0],
            "{}",
            case.name
        );
    }
}

/// The bound on the error of each entry of the clustered covariance over the
/// kept terms, and the demeaning error `e` of the two sides together relative
/// to each column's root mean square.
///
/// This side's error: at the returned point the level means of the residual
/// are at most tol s, so the level sums gamma have ||gamma|| <= tol s
/// sqrt(sum W_qg^2), the error has ||e||_W <= ||gamma|| / sqrt(lambda_+)
/// (epistat-glm's absorb module) and each row's error is at most
/// ||e||_W / sqrt(min w). The residual itself is a sum of three terms of the
/// size of the centred column, which adds 16 eps of its largest value.
/// The covariance: |dV| <= |B| |dM| |B| times the small-sample factors
/// (below 2 here, so 4 covers them twice over), with |dM| bounded from the
/// score sums and the errors of the scores, and rounding of 100 eps of the
/// largest entry.
fn covariance_bound(case: &Case, x: Option<&Mat<f64>>) -> (Vec<f64>, f64, f64) {
    // The same regression through epistat-glm, for its design and scores.
    let result = estimate_twfe(&panel(case, x), &config(case)).unwrap();
    let t_min = *case.time.iter().min().unwrap();
    let time: Vec<u32> = case
        .time
        .iter()
        .map(|&t| (t - t_min).cast_unsigned())
        .collect();
    let fe = FixedEffects::new(
        &[&case.unit, &time],
        case.weights.as_deref(),
        Singletons::Remove,
    )
    .unwrap();
    let terms: Vec<Option<i32>> = result.terms.iter().map(|t| t.relative_period).collect();
    let p = terms.len() + usize::from(x.is_some());
    let n = case.unit.len();
    let design = Mat::from_fn(n, p, |i, j| {
        if j >= terms.len() {
            return x.unwrap()[(i, 0)];
        }
        let Some(g) = case.first_treated[i] else {
            return 0.0;
        };
        let r = case.time[i] - g;
        match terms[j] {
            None => f64::from(u8::from(case.time[i] >= g)),
            Some(s) => {
                let r = match (case.spec.lo, case.spec.hi) {
                    (Some(lo), Some(hi)) => r.clamp(lo, hi),
                    _ => r,
                };
                f64::from(u8::from(r == s))
            }
        }
    });
    let fit = Feols { absorb: TIGHT }
        .fit(&case.outcome, design.as_ref(), &fe)
        .unwrap();
    // The wrapper is the plain fit.
    for (j, c) in fit.coefficients.iter().enumerate() {
        let mine = result.terms.get(j).map_or_else(
            || result.covariate_coefficients[j - terms.len()],
            |t| t.estimate,
        );
        assert_eq!(c.map(f64::to_bits), mine.map(f64::to_bits), "{}", case.name);
    }
    let rows = fe.rows();
    let m = rows.len();
    let w: Vec<f64> = rows
        .iter()
        .map(|&i| case.weights.as_ref().map_or(1.0, |w| w[i]))
        .collect();
    let tw: f64 = w.iter().sum();
    let rms = |v: &dyn Fn(usize) -> f64| {
        let mean: f64 = (0..m).map(|k| w[k] * v(rows[k])).sum::<f64>() / tw;
        ((0..m)
            .map(|k| w[k] * (v(rows[k]) - mean).powi(2))
            .sum::<f64>()
            / tw)
            .sqrt()
    };
    let kept = fit.kept_columns().to_vec();
    let k = kept.len();
    let rms_x: Vec<f64> = kept.iter().map(|&j| rms(&|i| design[(i, j)])).collect();
    let rms_y = rms(&|i| case.outcome[i]);
    let spread = |v: &dyn Fn(usize) -> f64, s: f64| {
        let mean: f64 = (0..m).map(|k| w[k] * v(rows[k])).sum::<f64>() / tw;
        (0..m)
            .map(|k| (v(rows[k]) - mean).abs())
            .fold(0.0, f64::max)
            / s
    };
    let ratio = kept
        .iter()
        .zip(&rms_x)
        .map(|(&j, &r)| spread(&|i| design[(i, j)], r))
        .fold(spread(&|i| case.outcome[i], rms_y), f64::max);
    let mut sum_w2 = 0.0;
    for q in 0..2 {
        let mut level = vec![0.0; fe.levels()[q]];
        for (kk, &c) in fe.codes(q).iter().enumerate() {
            level[c as usize] += w[kk];
        }
        sum_w2 += level.iter().map(|v| v * v).sum::<f64>();
    }
    let w_min = w.iter().copied().fold(f64::MAX, f64::min);
    let mine = TIGHT.tolerance * sum_w2.sqrt() / (case.exact.lambda_plus * w_min).sqrt()
        + 16.0 * EPS * ratio;
    let e = case.exact.error + mine;
    let delta_e = e
        * (rms_y
            + kept
                .iter()
                .zip(&rms_x)
                .map(|(&j, r)| fit.coefficients[j].unwrap().abs() * r)
                .sum::<f64>());
    let d = fit.design();
    let clusters: Vec<u32> = rows
        .iter()
        .map(|&i| case.clusters.as_ref().map_or(case.unit[i], |c| c[i]))
        .collect();
    let of = Clusters::from_labels(&clusters);
    let g = of.count();
    let mut s = vec![0.0; g * k];
    let mut u = vec![0.0; g * k];
    for r in 0..m {
        let c = of.of()[r];
        for a in 0..k {
            s[c * k + a] += w[r] * fit.residuals[r] * d[(r, a)];
            u[c * k + a] +=
                w[r] * (fit.residuals[r].abs() * e * rms_x[a] + d[(r, a)].abs() * delta_e);
        }
    }
    let mut dm = vec![0.0; k * k];
    for c in 0..g {
        for a in 0..k {
            for b in 0..k {
                dm[a * k + b] +=
                    s[c * k + a].abs() * u[c * k + b] + u[c * k + a] * s[c * k + b].abs();
            }
        }
    }
    let ways = [clusters_input(case)];
    let cov = fit.vcov(Vcov::Cluster(&ways), Ssc::new()).unwrap();
    let largest = cov
        .matrix
        .iter()
        .flatten()
        .fold(0.0_f64, |a, &b| a.max(b.abs()));
    let bread = fit.bread();
    let mut bound = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            let mut total = 0.0;
            for c in 0..k {
                for dd in 0..k {
                    total += bread[(a, c)].abs() * dm[c * k + dd] * bread[(dd, b)].abs();
                }
            }
            // The small-sample factors are below 2 for these panels.
            bound[a * k + b] = 4.0 * total + 100.0 * EPS * largest;
        }
    }
    // The condition number of the Gram matrix of the demeaned design with
    // its columns scaled to unit norm.
    let gram: Vec<f64> = (0..k * k)
        .map(|ab| (0..m).map(|r| w[r] * d[(r, ab / k)] * d[(r, ab % k)]).sum())
        .collect();
    let diag: Vec<f64> = (0..k).map(|a| gram[a * k + a].sqrt()).collect();
    let scaled: Vec<f64> = (0..k * k)
        .map(|ab| gram[ab] / (diag[ab / k] * diag[ab % k]))
        .collect();
    let (values, _) = epistat_linalg::eigen::symmetric_eigen_jacobi(&scaled, k);
    let kappa = values.iter().copied().fold(f64::MIN, f64::max)
        / values.iter().copied().fold(f64::MAX, f64::min);
    (bound, e, kappa)
}

fn clusters_input(case: &Case) -> &[u32] {
    case.clusters.as_deref().unwrap_or(&case.unit)
}

#[test]
fn bins_must_contain_the_reference() {
    let reference = reference();
    let case = &reference.cases[0];
    let mut config = config(case);
    config.spec = TwfeSpec::EventStudy {
        reference: -1,
        bins: Some((0, 3)),
    };
    assert!(estimate_twfe(&panel(case, None), &config).is_err());
}

#[test]
fn a_cohort_that_varies_within_a_unit_is_an_error() {
    let reference = reference();
    let case = &reference.cases[0];
    let mut first = case.first_treated.clone();
    let row = first.iter().position(Option::is_some).unwrap();
    first[row] = first[row].map(|g| g + 1);
    let panel = TwfePanel {
        first_treated: &first,
        ..panel(case, None)
    };
    assert!(estimate_twfe(&panel, &config(case)).is_err());
}

fn twfe_bits() -> Vec<f64> {
    let reference = reference();
    let mut out = Vec::new();
    for case in &reference.cases {
        let x = covariate(case);
        let mut c = config(case);
        c.absorb = Absorb::new();
        let result = estimate_twfe(&panel(case, x.as_ref()), &c).unwrap();
        out.extend(result.terms.iter().filter_map(|t| t.estimate));
        out.extend(result.covariance.iter().flatten());
        out.extend(result.influence.iter().flatten().flatten().take(40));
    }
    out
}

#[test]
fn twfe_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(twfe_bits);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn twfe_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&twfe_bits())
        .finish();
    assert_eq!(hash, 10_080_804_908_240_982_917, "recorded hash");
}
