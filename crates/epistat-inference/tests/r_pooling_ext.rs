//! The extensions of Rubin's rules checked against `mice` 3.19.0 and `mitml`
//! 0.4.5: the pooled vector, the tests D1, D2 and D3 of several parameters,
//! pooling on a transformed scale and the p-values.
//!
//! The reference is `tests/data/pooling_ext.json`, written by
//! `tests/data/pooling_ext.R`. It holds, for models fitted to every imputed data
//! set of `nhanes2`, `boys` and `lung`, the coefficients and covariance of each
//! fit, the tests `mitml::testModels` and `mice::D3` return, and tail
//! probabilities of the F and t distributions.
//!
//! Tolerance. The pooled quantities and the statistics are sums and a solve of
//! at most `k = 6` equations on data of moderate condition, so the error is a
//! few `k kappa eps`, below `1e-11`, and they are held to `1e-10` relative. The
//! logistic fits to `nhanes2` are close to separation, and the pooled within
//! covariance of their tested coefficients has condition numbers up to `1.4e10`
//! (`1.3e7` for two coefficients), so the quantities that go through a solve
//! are held to `1e-6` there. The
//! degrees of freedom of D1 and D3 are a square of a ratio that contains `1/r`,
//! and `r` itself is a trace of a solve, so they are held to `1e-9`. The
//! p-values are tails of the incomplete beta function, held to `1e-9` relative
//! or `1e-14` absolute. The checks print the largest deviation in units of its
//! bound.

use epistat_core::{InputError, Level};
use epistat_inference::Error;
use epistat_inference::pooling::{
    MultiTest, NegativeRiv, PooledVector, Transform, d1, d2, d3, dfcom_events, dfcom_residual,
    rubin, rubin_transformed, rubin_vector,
};
use epistat_linalg::{Mat, MatRef};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
    transformed: Vec<Transformed>,
    pf: Vec<PfPoint>,
    pt: Vec<PtPoint>,
}

#[derive(Deserialize)]
struct Fit {
    coef: Vec<f64>,
    vcov: Vec<f64>,
}

#[derive(Deserialize)]
struct Pooled {
    estimate: Vec<f64>,
    ubar: Vec<f64>,
    b: Vec<f64>,
    t: Vec<f64>,
    df: Vec<f64>,
    p: Vec<Option<f64>>,
    dfcom: Vec<f64>,
}

#[derive(Deserialize)]
struct Test {
    f: f64,
    df2: f64,
    p: f64,
    riv: f64,
}

#[derive(Deserialize)]
struct D3Reference {
    f: f64,
    df2: f64,
    p: f64,
    riv: f64,
    lr_own: Vec<f64>,
    lr_pooled: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    m: usize,
    k: usize,
    tested: Vec<usize>,
    dfcom: f64,
    fits1: Vec<Fit>,
    pooled: Pooled,
    d1: Test,
    d1_dfcom: Test,
    d2_wald: Test,
    d2_lik: Option<Test>,
    lik_stats: Option<Vec<f64>>,
    wald_stats: Vec<f64>,
    d3: Option<D3Reference>,
}

#[derive(Deserialize)]
struct Transformed {
    name: String,
    kind: String,
    q: Vec<f64>,
    se: Vec<f64>,
    dfcom: Option<f64>,
    qbar: f64,
    ubar: f64,
    b: f64,
    t: f64,
    df: f64,
    estimate: f64,
    lower: f64,
    upper: f64,
    pvalue_null: f64,
}

#[derive(Deserialize)]
struct PfPoint {
    k: f64,
    nu: f64,
    f: f64,
    p: f64,
}

#[derive(Deserialize)]
struct PtPoint {
    nu: f64,
    z: f64,
    p: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pooling_ext.json"
    ))
}

struct Worst(f64);

impl Worst {
    fn rel(&mut self, what: &str, got: f64, want: f64, bound: f64) {
        let r = (got - want).abs() / (bound * want.abs().max(1e-300));
        self.0 = self.0.max(r);
        assert!(r <= 1.0, "{what}: {got:e} against {want:e}");
    }

    /// A probability: relative `bound` or an absolute `1e-14`.
    fn prob(&mut self, what: &str, got: f64, want: f64, bound: f64) {
        let r = (got - want).abs() / (bound * want.abs()).max(1e-14);
        self.0 = self.0.max(r);
        assert!(r <= 1.0, "{what}: {got:e} against {want:e}");
    }
}

/// The estimates and covariances of the tested coefficients of each data set.
fn solve_bound(case: &Case) -> f64 {
    if case.name.starts_with("logistic") {
        1e-6
    } else {
        1e-10
    }
}

fn tested_parts(case: &Case) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let p = case.fits1[0].coef.len();
    let idx: Vec<usize> = case.tested.iter().map(|i| i - 1).collect();
    let q = case
        .fits1
        .iter()
        .map(|f| idx.iter().map(|&i| f.coef[i]).collect())
        .collect();
    let u = case
        .fits1
        .iter()
        .map(|f| {
            idx.iter()
                .flat_map(|&i| idx.iter().map(move |&j| (i, j)))
                .map(|(i, j)| f.vcov[i * p + j])
                .collect()
        })
        .collect();
    (q, u)
}

fn slices(v: &[Vec<f64>]) -> Vec<&[f64]> {
    v.iter().map(Vec::as_slice).collect()
}

/// The estimates as an `m x k` matrix and the row-major covariances as
/// `k x k` matrices, the forms `rubin_vector` and `d1` take.
fn matrices(q: &[&[f64]], u: &[&[f64]]) -> (Mat<f64>, Vec<Mat<f64>>) {
    let k = q[0].len();
    let estimates = Mat::from_fn(q.len(), k, |i, j| q[i][j]);
    let covariances = u
        .iter()
        .map(|u| Mat::from_fn(k, k, |i, j| u[i * k + j]))
        .collect();
    (estimates, covariances)
}

fn vector(q: &[&[f64]], u: &[&[f64]]) -> Result<PooledVector, Error> {
    let (estimates, covariances) = matrices(q, u);
    let views: Vec<MatRef<'_, f64>> = covariances.iter().map(Mat::as_ref).collect();
    rubin_vector(estimates.as_ref(), &views)
}

fn test_d1(
    q: &[&[f64]],
    u: &[&[f64]],
    null: Option<&[f64]>,
    dfcom: Option<f64>,
) -> Result<MultiTest, Error> {
    let (estimates, covariances) = matrices(q, u);
    let views: Vec<MatRef<'_, f64>> = covariances.iter().map(Mat::as_ref).collect();
    d1(estimates.as_ref(), &views, null, dfcom)
}

#[test]
fn pooled_vector_matches_mice_pool() {
    let mut worst = Worst(0.0);
    for case in reference().cases {
        let p = case.fits1[0].coef.len();
        let q: Vec<&[f64]> = case.fits1.iter().map(|f| f.coef.as_slice()).collect();
        let u: Vec<&[f64]> = case.fits1.iter().map(|f| f.vcov.as_slice()).collect();
        let v = vector(&q, &u).unwrap();
        assert_eq!(v.m, case.m);
        for j in 0..p {
            let name = format!("{} coefficient {j}", case.name);
            worst.rel(&name, v.estimate[j], case.pooled.estimate[j], 1e-10);
            worst.rel(
                &name,
                v.within.get(j, j).unwrap(),
                case.pooled.ubar[j],
                1e-10,
            );
            worst.rel(&name, v.between.get(j, j).unwrap(), case.pooled.b[j], 1e-10);
            worst.rel(&name, v.total.get(j, j).unwrap(), case.pooled.t[j], 1e-10);
            // The scalar rule per coefficient, with the p-value.
            let est: Vec<f64> = q.iter().map(|f| f[j]).collect();
            let var: Vec<f64> = u.iter().map(|f| f[j * p + j]).collect();
            let s = rubin(&est, &var, Some(case.pooled.dfcom[j])).unwrap();
            worst.rel(&name, s.df.unwrap(), case.pooled.df[j], 1e-9);
            if let Some(want) = case.pooled.p[j] {
                let got = s.p_value(0.0).unwrap().unwrap();
                worst.prob(&name, got, want, 1e-9);
            }
        }
        // The relative increase of the tested block is D1's.
        let (tq, tu) = tested_parts(&case);
        let sub = vector(&slices(&tq), &slices(&tu)).unwrap();
        worst.rel(
            &case.name,
            sub.riv,
            case.d1.riv,
            solve_bound(&case).max(1e-9),
        );
    }
    eprintln!("vector: worst deviation {:.3} of its bound", worst.0);
}

#[test]
fn d1_matches_mitml() {
    let mut worst = Worst(0.0);
    for case in reference().cases {
        let (q, u) = tested_parts(&case);
        let (q, u) = (slices(&q), slices(&u));
        let name = &case.name;
        let t = test_d1(&q, &u, None, None).unwrap();
        let bound = solve_bound(&case);
        if case.d1.f > 1e-6 {
            worst.rel(name, t.statistic, case.d1.f, bound);
        }
        worst.rel(name, t.riv, case.d1.riv, bound.max(1e-9));
        worst.rel(name, t.df2, case.d1.df2, bound.max(1e-9));
        worst.prob(name, t.p_value.unwrap(), case.d1.p, bound.max(1e-9));
        let c = test_d1(&q, &u, None, Some(case.dfcom));
        if case.k * (case.m - 1) <= 4 {
            assert!(matches!(c, Err(Error::Imputations { .. })), "{name}");
            continue;
        }
        let c = c.unwrap();
        if case.d1_dfcom.f > 1e-6 {
            worst.rel(name, c.statistic, case.d1_dfcom.f, bound);
        }
        worst.rel(name, c.df2, case.d1_dfcom.df2, bound.max(1e-9));
        worst.prob(name, c.p_value.unwrap(), case.d1_dfcom.p, bound.max(1e-9));
    }
    eprintln!("D1: worst deviation {:.3} of its bound", worst.0);
}

#[test]
fn d2_matches_mitml() {
    let mut worst = Worst(0.0);
    for case in reference().cases {
        let name = &case.name;
        let mut runs = vec![(&case.wald_stats, &case.d2_wald)];
        if let (Some(l), Some(t)) = (&case.lik_stats, &case.d2_lik) {
            runs.push((l, t));
        }
        for (stats, want) in runs {
            // A likelihood ratio statistic can be a rounding error below 0.
            let stats: Vec<f64> = stats.iter().map(|d| d.max(0.0)).collect();
            let t = d2(&stats, case.k).unwrap();
            if want.f < 0.0 {
                assert!(t.truncated && t.statistic == 0.0, "{name}");
                assert_eq!(t.p_value, Some(1.0));
            } else {
                worst.rel(name, t.statistic, want.f, 1e-9);
                worst.prob(name, t.p_value.unwrap(), want.p, 1e-9);
            }
            worst.rel(name, t.riv, want.riv, 1e-9);
            worst.rel(name, t.df2, want.df2, 1e-9);
        }
    }
    eprintln!("D2: worst deviation {:.3} of its bound", worst.0);
}

#[test]
fn d3_matches_mice() {
    let mut worst = Worst(0.0);
    let mut checked = 0;
    for case in reference().cases {
        let Some(w) = &case.d3 else { continue };
        let name = &case.name;
        checked += 1;
        let t = d3(&w.lr_own, &w.lr_pooled, case.k, NegativeRiv::Keep).unwrap();
        worst.rel(name, t.riv, w.riv, 1e-9);
        if w.f != 0.0 {
            worst.rel(name, t.statistic, w.f, 1e-9);
        }
        worst.rel(name, t.df2, w.df2, 1e-9);
        worst.prob(name, t.p_value.unwrap(), w.p, 1e-9);
        let truncated = d3(&w.lr_own, &w.lr_pooled, case.k, NegativeRiv::Truncate).unwrap();
        if w.riv < 0.0 {
            assert!(truncated.truncated && truncated.riv == 0.0 && truncated.df2.is_infinite());
            let dtilde = w.lr_pooled.iter().sum::<f64>() / case.m as f64;
            worst.rel(name, truncated.statistic, dtilde / case.k as f64, 1e-12);
        } else {
            assert!(!truncated.truncated);
            assert_eq!(truncated, t);
        }
    }
    assert!(checked >= 8);
    eprintln!("D3: worst deviation {:.3} of its bound", worst.0);
}

#[test]
fn transformed_pooling_matches_pool_scalar() {
    let mut worst = Worst(0.0);
    for c in reference().transformed {
        let kind = match c.kind.as_str() {
            "log" => Transform::Log,
            "logit" => Transform::Logit,
            "fisherz" => Transform::FisherZ,
            "cloglog" => Transform::Cloglog,
            _ => Transform::Sqrt,
        };
        let name = &c.name;
        let p = rubin_transformed(&c.q, &c.se, kind, c.dfcom).unwrap();
        worst.rel(name, p.pooled.estimate, c.qbar, 1e-12);
        worst.rel(name, p.pooled.within, c.ubar, 1e-12);
        worst.rel(name, p.pooled.between, c.b, 1e-12);
        worst.rel(name, p.pooled.total, c.t, 1e-12);
        worst.rel(name, p.pooled.df.unwrap(), c.df, 1e-9);
        worst.rel(name, p.estimate, c.estimate, 1e-12);
        let interval = p.interval(Level::alpha(0.05).unwrap());
        let (lo, hi) = (interval.lower.unwrap(), interval.upper.unwrap());
        worst.rel(name, lo, c.lower, 1e-9);
        worst.rel(name, hi, c.upper, 1e-9);
        let null = match kind {
            Transform::FisherZ => 0.2,
            Transform::Log | Transform::Sqrt => 1.0,
            _ => 0.4,
        };
        worst.prob(name, p.p_value(null).unwrap().unwrap(), c.pvalue_null, 1e-9);
    }
    eprintln!("transformed: worst deviation {:.3} of its bound", worst.0);
}

#[test]
fn transformations_refuse_values_outside_their_domain() {
    let q = [0.2, 1.2, 0.3];
    let se = [0.05; 3];
    assert!(matches!(
        rubin_transformed(&q, &se, Transform::Logit, None),
        Err(Error::Input(InputError::Domain { .. }))
    ));
    assert!(matches!(
        rubin_transformed(&[0.5, -0.1], &[0.1, 0.1], Transform::Log, None),
        Err(Error::Input(InputError::Domain { .. }))
    ));
    assert!(
        rubin_transformed(&[0.5, 0.6], &[0.1, 0.1], Transform::Log, None)
            .unwrap()
            .p_value(-1.0)
            .is_err()
    );
}

/// The tolerance on a tail probability. Up to `df2 = 1e7` the incomplete beta
/// function is accurate to `1e-9` relative on this grid. Above, the chi-square
/// tail with its first correction in `1 / df2` has an error of order
/// `c^4 / df2^2` with `c = k f / 2`, below `1e-6` relative for the grid.
fn tail_bound(nu: f64) -> f64 {
    if nu < 1e7 { 1e-9 } else { 1e-6 }
}

#[test]
fn distribution_tails_match_r() {
    let r = reference();
    let mut worst_f = Worst(0.0);
    for g in &r.pf {
        let got = epistat_stats::student::f_sf(g.f, g.k, g.nu);
        worst_f.prob(
            &format!("pf k {} nu {} f {}", g.k, g.nu, g.f),
            got,
            g.p,
            tail_bound(g.nu),
        );
    }
    eprintln!("F tails: worst deviation {:.3} of its bound", worst_f.0);
    let mut worst_t = Worst(0.0);
    for g in &r.pt {
        // A pooled estimate z standard errors from the null, on nu degrees of freedom.
        // Two equal sets of variance 1 pool to a total variance of 1; the
        // estimate and the degrees of freedom are then set to the case's.
        let mut pooled = rubin(&[0.0, 0.0], &[1.0, 1.0], None).unwrap();
        pooled.estimate = g.z;
        pooled.df = Some(g.nu);
        let got = pooled.p_value(0.0).unwrap().unwrap();
        worst_t.prob(
            &format!("pt nu {} z {}", g.nu, g.z),
            got,
            g.p,
            tail_bound(g.nu),
        );
    }
    eprintln!("t tails: worst deviation {:.3} of its bound", worst_t.0);
}

#[test]
fn degrees_of_freedom_helpers() {
    assert_eq!(dfcom_residual(100, 4).to_bits(), 96.0_f64.to_bits());
    assert_eq!(dfcom_residual(3, 4).to_bits(), 1.0_f64.to_bits());
    assert_eq!(dfcom_events(60, 3).to_bits(), 57.0_f64.to_bits());
    assert_eq!(dfcom_events(2, 3).to_bits(), 1.0_f64.to_bits());
}

/// The p-value goes through `ln`, `exp` and `powf`, whose last bits depend on
/// the maths library and, on Windows, on the processor the library dispatches
/// for, so the bits are checked only on aarch64 macOS, where they were recorded.
#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fingerprint_of_the_tests() {
    let case = reference()
        .cases
        .into_iter()
        .find(|c| c.name == "boys linear, m = 5, drop reg")
        .unwrap();
    let (q, u) = tested_parts(&case);
    let (q, u) = (slices(&q), slices(&u));
    let a = test_d1(&q, &u, None, Some(case.dfcom)).unwrap();
    let b = d2(&case.wald_stats, case.k).unwrap();
    let w = case.d3.as_ref().unwrap();
    let c = d3(&w.lr_own, &w.lr_pooled, case.k, NegativeRiv::Keep).unwrap();
    let hash = epistat_testkit::Fingerprint::new()
        .f64(a.statistic)
        .f64(a.df2)
        .option_f64(a.p_value)
        .f64(b.statistic)
        .f64(b.df2)
        .f64(c.statistic)
        .f64(c.df2)
        .finish();
    eprintln!("pooling fingerprint {hash:#018x}");
    assert_eq!(hash, 0xa9bc_5db4_43e5_013a);
}
