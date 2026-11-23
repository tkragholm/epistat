//! Restricted cubic splines and natural cubic splines checked against R's
//! `Hmisc::rcspline.eval` and `splines::ns`, and thread independence and a
//! golden hash of the design matrices.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the tie counts are floor(fraction * n) of small non-negative numbers"
)]

use epistat_linalg::basis::{
    Basis, NaturalSpline, RcsScale, RestrictedCubic, basis_matrix, harrell_knots,
    natural_spline_knots,
};
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    rcs: Vec<RcsCase>,
    knots: Vec<KnotCase>,
    ns: Vec<NsCase>,
}

#[derive(Deserialize)]
struct RcsCase {
    knots: Vec<f64>,
    norm: u8,
    inclx: bool,
    x: Vec<f64>,
    basis: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct KnotCase {
    n: usize,
    nk: usize,
    tie_low: f64,
    tie_high: f64,
    knots: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct NsCase {
    n: usize,
    tie_low: f64,
    df: usize,
    intercept: bool,
    given_boundary: bool,
    boundary: Vec<f64>,
    #[serde(default)]
    knots: Vec<f64>,
    xnew: Vec<f64>,
    basis: Vec<Vec<f64>>,
    deriv: Vec<Vec<f64>>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/splines_rcs.json"
    ))
}

/// The integer sample rule of `splines_rcs.R`.
fn sample_x(n: usize, tie_low: f64, tie_high: f64) -> Vec<f64> {
    let mut x: Vec<f64> = (1..=n)
        .map(|i| ((i * 7919) % 10007) as f64 / 10007.0 * 100.0)
        .collect();
    let low = (tie_low * n as f64).floor() as usize;
    let high = (tie_high * n as f64).floor() as usize;
    for v in x.iter_mut().take(low) {
        *v = 0.0;
    }
    for v in x.iter_mut().skip(n - high) {
        *v = 100.0;
    }
    x
}

fn worst(got: f64, expected: f64) -> f64 {
    (got - expected).abs()
}

#[test]
fn restricted_cubic_values_match_hmisc() {
    let eps = f64::EPSILON;
    let mut worst_ratio = 0.0_f64;
    for case in &reference().rcs {
        let scale = match case.norm {
            0 => RcsScale::Raw,
            1 => RcsScale::LastGap,
            _ => RcsScale::Range,
        };
        let basis = RestrictedCubic::with_scale(&case.knots, case.inclx, scale).unwrap();
        let k = case.knots.len();
        let (first, last) = (case.knots[0], case.knots[k - 1]);
        let divisor = match case.norm {
            0 => 1.0,
            1 => (last - case.knots[k - 2]).powi(3),
            _ => (last - first).powi(2),
        };
        let x_max = case.x.iter().copied().fold(f64::MIN, f64::max);
        // Three terms of size up to (x - t_1)^3 / D are added in R's direct
        // form, so R's rounding error is a few times eps times that. The
        // linear column is exact.
        let bound = 8.0 * eps * (x_max - first).powi(3) / divisor;
        for (&x, row) in case.x.iter().zip(&case.basis) {
            for (j, (&got, &want)) in basis.values(x).iter().zip(row).enumerate() {
                let allowed = if case.inclx && j == 0 { 0.0 } else { bound };
                let deviation = worst(got, want);
                assert!(
                    deviation <= allowed,
                    "norm {} k {k} x {x} column {j}: {got:e} against {want:e}",
                    case.norm
                );
                if allowed > 0.0 {
                    worst_ratio = worst_ratio.max(deviation / allowed);
                }
            }
        }
    }
    eprintln!("restricted cubic: worst deviation over its bound {worst_ratio:.3}");
}

#[test]
fn harrell_knots_match_hmisc() {
    let mut cases = 0;
    for case in &reference().knots {
        let x = sample_x(case.n, case.tie_low, case.tie_high);
        let got = harrell_knots(&x, case.nk, 0.05);
        match (&case.knots, got) {
            (Some(want), Ok(got)) => {
                assert_eq!(
                    got.len(),
                    want.len(),
                    "{} {} {}",
                    case.n,
                    case.nk,
                    case.tie_low
                );
                for (g, w) in got.iter().zip(want) {
                    assert!(
                        (g - w).abs() <= 1e-13 * w.abs().max(1.0),
                        "n {} nk {} low {} high {}: {got:?} against {want:?}",
                        case.n,
                        case.nk,
                        case.tie_low,
                        case.tie_high
                    );
                }
                cases += 1;
            }
            (None, Err(_)) => cases += 1,
            (want, got) => panic!(
                "n {} nk {} low {} high {}: R {want:?}, here {got:?}",
                case.n, case.nk, case.tie_low, case.tie_high
            ),
        }
    }
    assert!(cases > 100);
}

#[test]
fn natural_splines_match_splines_ns() {
    let (mut worst_value, mut worst_derivative) = (0.0_f64, 0.0_f64);
    for case in &reference().ns {
        let mut x = sample_x(case.n.min(500), case.tie_low, 0.0);
        if case.n == 520 {
            let head: Vec<f64> = x[..20].to_vec();
            x.extend(head);
        }
        let boundary = case
            .given_boundary
            .then(|| (case.boundary[0], case.boundary[1]));
        let knots = natural_spline_knots(&x, case.df, case.intercept, boundary).unwrap();
        assert_eq!(
            knots.len(),
            case.knots.len(),
            "df {} knots {knots:?}",
            case.df
        );
        for (g, w) in knots.iter().zip(&case.knots) {
            assert!(
                (g - w).abs() <= 1e-13 * w.abs().max(1.0),
                "{knots:?} against {:?}",
                case.knots
            );
        }
        let basis =
            NaturalSpline::new(&knots, (case.boundary[0], case.boundary[1]), case.intercept)
                .unwrap();
        assert_eq!(basis.len(), case.basis[0].len());
        for (i, &x) in case.xnew.iter().enumerate() {
            let (values, slopes) = (basis.values(x), basis.derivatives(x));
            for j in 0..basis.len() {
                // A QR of a well conditioned constraint matrix and a sum of four
                // products of the size of the value: 1e-13 of it, and of 1 where
                // the value is small. Linear extrapolation reaches values near 100.
                let d = worst(values[j], case.basis[i][j]) / case.basis[i][j].abs().max(1.0);
                assert!(
                    d <= 1e-13,
                    "df {} int {} x {x} col {j}: {} {}",
                    case.df,
                    case.intercept,
                    values[j],
                    case.basis[i][j]
                );
                worst_value = worst_value.max(d);
                // R's reference slope is a five-point difference with step
                // 1e-3, whose error is of order h^4 times the fifth derivative
                // (1e-12 for the smooth parts here) and, across a knot, h^2
                // times the jump in the third derivative; 1e-9 covers both.
                let d = worst(slopes[j], case.deriv[i][j]);
                assert!(
                    d <= 1e-9,
                    "df {} x {x} col {j}: {} {}",
                    case.df,
                    slopes[j],
                    case.deriv[i][j]
                );
                worst_derivative = worst_derivative.max(d);
            }
        }
    }
    eprintln!("ns: worst value deviation {worst_value:e}, worst derivative {worst_derivative:e}");
}

fn designs() -> Vec<f64> {
    let x = sample_x(20_000, 0.0, 0.0);
    let rcs = RestrictedCubic::new(&[5.0, 27.5, 50.0, 72.5, 95.0], true).unwrap();
    let ns = NaturalSpline::new(&[20.0, 40.0, 60.0, 80.0], (0.0, 100.0), false).unwrap();
    let mut out = basis_matrix(&rcs, &x)
        .col_iter()
        .flat_map(|c| c.iter().copied().collect::<Vec<_>>())
        .collect::<Vec<f64>>();
    out.extend(
        basis_matrix(&ns, &x)
            .col_iter()
            .flat_map(|c| c.iter().copied().collect::<Vec<_>>()),
    );
    out
}

#[test]
fn design_matrices_do_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(designs);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn design_matrices_have_their_recorded_bits() {
    // The bits depend on the platform's maths library for none of the
    // operations used (sums, products, quotients, square roots), but the
    // hash is recorded on the platform of the other golden tests.
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&designs())
        .finish();
    assert_eq!(hash, 14_909_619_049_344_926_689, "recorded hash");
}
