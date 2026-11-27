//! `root::bisect` against survival's `log_product_limit` bisection, copied
//! here as it is written there: the same root to the bit.

use epistat_optim::Error;
use epistat_optim::root::{Bisection, bisect};

/// epistat-survival `cox/predict.rs`, the bisection of `log_product_limit`,
/// as written there after its bracket is found.
fn survival_bisection(g: impl Fn(f64) -> f64, s0: f64, mut lo: f64, mut hi: f64) -> f64 {
    for _ in 0..2200 {
        let mid = lo + (hi - lo) / 2.0;
        if mid <= lo || mid >= hi {
            break;
        }
        if g(mid) < s0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 2.0 * f64::EPSILON * mid.abs() {
            break;
        }
    }
    lo + (hi - lo) / 2.0
}

#[test]
fn bisection_is_survivals_to_the_bit() {
    let cases: [(&[f64], &[f64], f64, f64); 4] = [
        (&[1.0, 1.0], &[0.5, 2.0], 4.0, -1.0),
        (&[0.3, 1.7, 2.0], &[1.1, 0.4, 0.9], 6.5, -4.0),
        (&[1.0; 5], &[1.0, 1.0, 1.0, 1.0, 1.0], 5.000_001, -64.0),
        (&[2.5, 0.5], &[3.0, 0.01], 9.0, -2.0),
    ];
    for (w, theta, s0, lo) in cases {
        let g = |l: f64| {
            let mut total = 0.0;
            for (w, t) in w.iter().zip(theta) {
                total += w * t / -(t * l).exp_m1();
            }
            total
        };
        let hi = -f64::MIN_POSITIVE;
        assert!(g(lo) < s0 && g(hi) >= s0);
        let want = survival_bisection(g, s0, lo, hi);
        let got = bisect(|l| g(l) - s0, lo, hi, &Bisection::default()).unwrap();
        assert_eq!(got.x.to_bits(), want.to_bits(), "{w:?} {theta:?} {s0}");
        assert!(got.converged);
    }
}

#[test]
fn a_decreasing_function_and_the_stopping_rules() {
    let root = bisect(|x| 1.0 - x, 0.0, 3.0, &Bisection::default()).unwrap();
    // The last bracket has relative width at most 2 eps around the root.
    assert!((root.x - 1.0).abs() <= 2.0 * f64::EPSILON);
    assert!(root.converged);
    assert_eq!(root.evaluations, root.iterations + 3);
    let coarse = Bisection {
        absolute_tolerance: 0.25,
        ..Bisection::default()
    };
    let root = bisect(|x| x - 0.3, 0.0, 1.0, &coarse).unwrap();
    assert!(root.converged && root.iterations == 2 && (root.x - 0.3).abs() <= 0.125);
    let limited = Bisection {
        max_iterations: 3,
        ..Bisection::default()
    };
    let root = bisect(|x| x - 0.3, 0.0, 1.0, &limited).unwrap();
    assert!(!root.converged && root.iterations == 3);
}

#[test]
fn invalid_input_is_refused() {
    let f = |x: f64| x - 0.5;
    assert!(matches!(
        bisect(f, 0.0, 0.25, &Bisection::default()),
        Err(Error::NoSignChange)
    ));
    for (lower, upper) in [(f64::NAN, 1.0), (0.0, f64::INFINITY), (1.0, 0.0)] {
        assert!(matches!(
            bisect(f, lower, upper, &Bisection::default()),
            Err(Error::Input(_))
        ));
    }
    let negative = Bisection {
        relative_tolerance: -1.0,
        ..Bisection::default()
    };
    assert!(matches!(
        bisect(f, 0.0, 1.0, &negative),
        Err(Error::Input(_))
    ));
}
