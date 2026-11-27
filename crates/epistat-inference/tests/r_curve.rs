//! Contrast curves of spline terms against `rms::Predict(ref.zero = TRUE)` and
//! `Epi::ci.lin(ctr.mat =)`.
//!
//! The reference is `tests/data/curve.json`, written by `tests/data/curve.R`: a
//! logistic model and a Cox model with restricted cubic splines of 4 and 5
//! knots, and a Poisson model with a natural spline of 4 degrees of freedom,
//! each with the fitted coefficients, their covariance, the grid, the
//! reference value and the curve R reports. The basis comes from
//! `epistat-linalg`, which `epistat-linalg`'s own tests check against R.
//!
//! Tolerance. A value of the curve is a sum of at most five products of a basis
//! value (accurate to about `1e-14` of its size) and a coefficient, so the
//! estimate agrees to `1e-12` of the larger of its size and the size of the
//! terms, taken as `1e-12` absolute plus `1e-12` relative. The standard error
//! is a quadratic form of the same terms with a covariance that R prints to
//! 17 digits and has cancellation between its terms, so `1e-10` relative. The
//! limits are the estimate plus or minus a multiple of the standard error. The
//! test prints the largest deviation in units of its bound.

use epistat_core::{Bootstrap, Level};
use epistat_inference::curve::{Band, contrast_curve};
use epistat_inference::simultaneous::sup_t_from_covariance;
use epistat_linalg::Mat;
use epistat_linalg::basis::{Basis, NaturalSpline, RestrictedCubic};
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    rcs: Vec<RcsCase>,
    ns: NsCase,
}

#[derive(Deserialize)]
struct RcsCase {
    name: String,
    knots: Vec<f64>,
    reference: f64,
    level: f64,
    grid: Vec<f64>,
    coef: Vec<f64>,
    cov: Vec<Vec<f64>>,
    estimate: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Deserialize)]
struct NsCase {
    reference: f64,
    level: f64,
    grid: Vec<f64>,
    interior: Vec<f64>,
    boundary: Vec<f64>,
    coef: Vec<f64>,
    cov: Vec<Vec<f64>>,
    estimate: Vec<f64>,
    se: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/curve.json"
    ))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows.len(), |i, j| rows[i][j])
}

fn units(worst: &mut f64, got: f64, want: f64, relative: f64, absolute: f64) {
    *worst = worst.max((got - want).abs() / (absolute + relative * want.abs()));
}

#[test]
fn rcs_curves_match_predict() {
    let mut worst = 0.0_f64;
    for case in reference().rcs {
        let basis = RestrictedCubic::new(&case.knots, true).unwrap();
        let cov = matrix(&case.cov);
        let level = Level::alpha(1.0 - case.level).unwrap();
        let curve = contrast_curve(
            &basis,
            &case.grid,
            case.reference,
            &case.coef,
            cov.as_ref(),
            level,
            Band::Pointwise,
        )
        .unwrap();
        for i in 0..case.grid.len() {
            units(
                &mut worst,
                curve.estimate[i],
                case.estimate[i],
                1e-12,
                1e-12,
            );
            // R's limits are the estimate plus or minus z se, so compare the
            // half widths, which carry the standard error.
            let (got, want) = (
                curve.upper[i] - curve.lower[i],
                case.upper[i] - case.lower[i],
            );
            units(&mut worst, got, want, 1e-10, 1e-12);
            units(&mut worst, curve.lower[i], case.lower[i], 1e-10, 1e-10);
            units(&mut worst, curve.upper[i], case.upper[i], 1e-10, 1e-10);
        }
        let at_reference = case
            .grid
            .iter()
            .position(|&x| x.to_bits() == case.reference.to_bits());
        if let Some(i) = at_reference {
            assert_eq!(curve.estimate[i].to_bits(), 0, "{}", case.name);
            assert_eq!(curve.se[i].to_bits(), 0, "{}", case.name);
        }
    }
    eprintln!("restricted cubic: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn natural_spline_curve_matches_ci_lin() {
    let case = reference().ns;
    let basis =
        NaturalSpline::new(&case.interior, (case.boundary[0], case.boundary[1]), false).unwrap();
    let cov = matrix(&case.cov);
    let curve = contrast_curve(
        &basis,
        &case.grid,
        case.reference,
        &case.coef,
        cov.as_ref(),
        Level::alpha(1.0 - case.level).unwrap(),
        Band::Pointwise,
    )
    .unwrap();
    let mut worst = 0.0_f64;
    for i in 0..case.grid.len() {
        units(
            &mut worst,
            curve.estimate[i],
            case.estimate[i],
            1e-12,
            1e-12,
        );
        units(&mut worst, curve.se[i], case.se[i], 1e-10, 0.0);
        units(&mut worst, curve.lower[i], case.lower[i], 1e-10, 1e-10);
        units(&mut worst, curve.upper[i], case.upper[i], 1e-10, 1e-10);
    }
    eprintln!("natural spline: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

/// The simultaneous band uses the sup-t value of the covariance of the curve
/// over the grid, without the reference point, and does not depend on the
/// number of threads.
#[test]
fn simultaneous_band_uses_the_sup_t_value() {
    let case = reference().ns;
    let basis =
        NaturalSpline::new(&case.interior, (case.boundary[0], case.boundary[1]), false).unwrap();
    let cov = matrix(&case.cov);
    let band = Band::SupT(Bootstrap::new(20_000, 20_261_123));
    let alpha = Level::alpha(0.05).unwrap();
    let run = || {
        contrast_curve(
            &basis,
            &case.grid,
            case.reference,
            &case.coef,
            cov.as_ref(),
            alpha,
            band,
        )
        .unwrap()
    };
    let curve = run();
    let pointwise = contrast_curve(
        &basis,
        &case.grid,
        case.reference,
        &case.coef,
        cov.as_ref(),
        alpha,
        Band::Pointwise,
    )
    .unwrap();
    assert!(curve.critical > pointwise.critical);
    // Three grid points, whose covariance has full rank, written out with the
    // rows of the contrast matrix and given to `sup_t_from_covariance`. The
    // two critical values come from different draws, so they agree to the
    // Monte Carlo error of a quantile at 200,000 replicates, about 0.1%.
    let small = [24.0, 50.0, 70.0];
    let p = basis.len();
    let b0 = basis.values(case.reference);
    let rows: Vec<Vec<f64>> = small
        .iter()
        .map(|&x| {
            basis
                .values(x)
                .iter()
                .zip(&b0)
                .map(|(a, c)| a - c)
                .collect()
        })
        .collect();
    let full = Mat::from_fn(3, 3, |a, b| {
        let mut s = 0.0;
        for i in 0..p {
            for j in 0..p {
                s += rows[a][i] * case.cov[i][j] * rows[b][j];
            }
        }
        s
    });
    let draws = 200_000;
    let want = sup_t_from_covariance(full.as_ref(), alpha, Bootstrap::new(draws, 7)).unwrap();
    let got = contrast_curve(
        &basis,
        &small,
        case.reference,
        &case.coef,
        cov.as_ref(),
        alpha,
        Band::SupT(Bootstrap::new(draws, 11)),
    )
    .unwrap()
    .critical;
    eprintln!("sup-t {got} against {want}");
    assert!((got - want).abs() < 5e-3 * want, "{got} {want}");
    let values = || {
        let r = run();
        let mut v = vec![r.critical];
        v.extend(&r.lower);
        v.extend(&r.upper);
        v
    };
    assert_same_on_threads(values);
    // The band goes through `ln`, `exp` and the normal draws' `cos`, whose last
    // bits depend on the maths library and, on Windows, on the processor the
    // library dispatches for, so the bits are checked only on aarch64 macOS,
    // where they were recorded. Elsewhere the hash is written to
    // `EPISTAT_HASH_OUT` when that is set.
    let r = run();
    let hash = epistat_testkit::Fingerprint::new()
        .f64(r.critical)
        .f64s(&r.lower)
        .f64s(&r.upper)
        .finish();
    eprintln!("curve fingerprint {hash:#018x}");
    epistat_testkit::hash_out("curve", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 0x5f9d_e54c_4f54_6a2d);
    }
}

#[test]
fn input_errors() {
    use epistat_core::InputError;
    use epistat_inference::Error as E;
    let basis = RestrictedCubic::new(&[1.0, 2.0, 3.0], true).unwrap();
    let cov = Mat::<f64>::identity(2, 2);
    let ok = |grid: &[f64], coef: &[f64], cov: &Mat<f64>| {
        contrast_curve(
            &basis,
            grid,
            2.0,
            coef,
            cov.as_ref(),
            Level::default(),
            Band::Pointwise,
        )
    };
    assert!(ok(&[1.5], &[0.1, 0.2], &cov).is_ok());
    assert!(matches!(
        ok(&[], &[0.1, 0.2], &cov),
        Err(E::Input(InputError::Empty { what: "grid" }))
    ));
    assert!(matches!(
        ok(&[1.5], &[0.1], &cov),
        Err(E::Input(InputError::Length { .. }))
    ));
    assert!(matches!(
        ok(&[1.5], &[0.1, 0.2], &Mat::identity(3, 3)),
        Err(E::Input(InputError::Length { .. }))
    ));
    assert!(matches!(
        ok(&[f64::NAN], &[0.1, 0.2], &cov),
        Err(E::Input(InputError::NotFinite {
            what: "grid",
            row: 0,
            column: None
        }))
    ));
    let skew = Mat::from_fn(2, 2, |i, j| {
        if i == 0 && j == 1 {
            0.5
        } else {
            f64::from(i == j)
        }
    });
    assert_eq!(ok(&[1.5], &[0.1, 0.2], &skew), Err(E::NotSymmetric));
    let negative = Mat::from_fn(2, 2, |i, j| if i == j { -1.0 } else { 0.0 });
    assert!(matches!(
        ok(&[1.5], &[0.1, 0.2], &negative),
        Err(E::Input(InputError::Domain { .. }))
    ));
}
