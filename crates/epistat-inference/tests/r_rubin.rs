//! Rubin's rules checked against `mice` 3.19.0, `pool()` and `pool.scalar()`.
//!
//! The reference is `tests/data/rubin.json`, written by
//! `tests/data/rubin.R`. It holds linear models fitted to every data set
//! imputed by `mice()` from `nhanes` (5 data sets) and `nhanes2` (20 data
//! sets), with the per-data-set estimates and variances of every coefficient
//! and the pooled quantities `pool()` returns. Two further cases come from
//! `pool.scalar()`: one with infinite complete-data degrees of freedom and one
//! with identical estimates.
//!
//! Tolerance. Both sides are a few dozen additions, multiplications and
//! divisions on numbers of order one to a few hundred, so they agree to about
//! 1e-14 relative. The bound is 1e-12, which allows R's `var()` summing in a
//! different order. The case without between-imputation variance is the one
//! departure: `mice` floors lambda at 1e-4, which turns an infinite `df_old`
//! into 3e8 and moves df by `df_obs^2 / df_old`, under 1e-5 relative, so that
//! case is compared at 1e-5 and the test also asserts that our value is the
//! limit, `df_obs`.

use epistat_inference::pooling::rubin;
use epistat_testkit::{Tolerance, assert_close, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    scalars: Vec<Scalar>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    mice: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    terms: Vec<String>,
    estimates: Vec<Vec<f64>>,
    variances: Vec<Vec<f64>>,
    dfcom: f64,
    pooled: PooledColumns,
}

#[derive(Deserialize)]
struct PooledColumns {
    qbar: Vec<f64>,
    ubar: Vec<f64>,
    b: Vec<f64>,
    t: Vec<f64>,
    df: Vec<f64>,
    riv: Vec<f64>,
    lambda: Vec<f64>,
    fmi: Vec<f64>,
}

#[derive(Deserialize)]
struct Scalar {
    name: String,
    q: Vec<f64>,
    u: Vec<f64>,
    n: Option<f64>,
    qbar: f64,
    ubar: f64,
    b: f64,
    t: f64,
    df: f64,
    r: f64,
    fmi: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/rubin.json"
    ))
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let r = reference();
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.mice, "3.19.0");
}

#[test]
fn pooled_coefficients_match_mice_pool() {
    let tight = Tolerance::exact().rel(1e-12).abs(1e-14);
    for case in &reference().cases {
        let mut got = Vec::new();
        let mut want = Vec::new();
        for j in 0..case.terms.len() {
            let p = rubin(&case.estimates[j], &case.variances[j], Some(case.dfcom)).unwrap();
            got.extend([
                p.estimate,
                p.within,
                p.between,
                p.total,
                p.df.unwrap(),
                p.riv.unwrap(),
                p.lambda.unwrap(),
                p.fmi.unwrap(),
            ]);
            let c = &case.pooled;
            want.extend([
                c.qbar[j],
                c.ubar[j],
                c.b[j],
                c.t[j],
                c.df[j],
                c.riv[j],
                c.lambda[j],
                c.fmi[j],
            ]);
        }
        assert_close(&got, &want, tight);
        eprintln!("{}: {} coefficients", case.name, case.terms.len());
    }
}

#[test]
fn pooled_scalars_match_mice_pool_scalar() {
    let tight = Tolerance::exact().rel(1e-12).abs(1e-14);
    for s in &reference().scalars {
        // pool.scalar() takes the sample size n and uses n - 1 degrees of
        // freedom for one parameter.
        let dfcom = s.n.map(|n| n - 1.0);
        let p = rubin(&s.q, &s.u, dfcom).unwrap();
        let (df_tol, fmi_tol) = if s.b == 0.0 {
            (1e-5, 1e-5)
        } else {
            (1e-12, 1e-12)
        };
        assert_close(
            &[p.estimate, p.within, p.between, p.total, p.riv.unwrap()],
            &[s.qbar, s.ubar, s.b, s.t, s.r],
            tight,
        );
        assert_close(&[p.df.unwrap()], &[s.df], Tolerance::exact().rel(df_tol));
        assert_close(&[p.fmi.unwrap()], &[s.fmi], Tolerance::exact().rel(fmi_tol));
        if s.b == 0.0 {
            // The limit: df is the observed-data value (n-1+1)/(n-1+3)*(n-1).
            let d = dfcom.unwrap();
            assert_eq!(
                p.df.unwrap().to_bits(),
                ((d + 1.0) / (d + 3.0) * d).to_bits()
            );
        }
        eprintln!("{}", s.name);
    }
}

#[test]
fn undefined_statistics_are_none_and_bad_input_is_an_error() {
    let p = rubin(&[1.0, 1.0], &[0.0, 0.0], None).unwrap();
    assert_eq!(p.se(), None);
    assert_eq!(p.df, None);
    assert_eq!(p.lambda, None);
    assert_eq!(p.interval(0.05).unwrap(), None);
    assert!(rubin(&[1.0], &[0.1], None).is_err());
    assert!(rubin(&[1.0, 2.0], &[0.1], None).is_err());
    assert!(rubin(&[1.0, f64::NAN], &[0.1, 0.1], None).is_err());
    assert!(rubin(&[1.0, 2.0], &[0.1, -0.1], None).is_err());
    assert!(rubin(&[1.0, 2.0], &[0.1, 0.1], Some(0.0)).is_err());
}

#[test]
fn interval_uses_the_t_quantile_on_the_pooled_degrees_of_freedom() {
    let p = rubin(&[1.1, 0.9, 1.4, 1.0], &[0.2, 0.25, 0.22, 0.21], None).unwrap();
    let (lo, hi) = p.interval(0.05).unwrap().unwrap();
    // qt(0.975, 68.2995918367348) = 1.99531 (R).
    let half = (hi - lo) / 2.0;
    assert!((half / p.se().unwrap() - 1.99531).abs() < 1e-5, "{half}");
    assert!((f64::midpoint(lo, hi) - p.estimate).abs() < 1e-14);
}
