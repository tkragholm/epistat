//! `root::itp` against closed-form roots and against the guarantees of
//! Oliveira and Takahashi (2020): the result lies within the tolerance of a
//! sign change, and the evaluations never exceed bisection's count plus the
//! slack.
//!
//! Where no closed form is at hand the check needs no reference: the
//! function changes sign (or is zero) between `x - tolerance` and
//! `x + tolerance`, which is exactly the method's claim. The evaluation bound
//! is `n_half + n0 + 3`: the two ends, at most `n_half + n0` steps (Theorem
//! 2.1), and the evaluation at the returned midpoint.

use epistat_optim::Error;
use epistat_optim::root::{Itp, Root, itp};

fn n_half(lower: f64, upper: f64, tolerance: f64) -> usize {
    let steps = ((upper - lower) / (2.0 * tolerance)).log2().ceil().max(0.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a small non-negative count"
    )]
    let steps = steps as usize;
    steps
}

fn options(tolerance: f64) -> Itp {
    Itp {
        tolerance,
        ..Itp::default()
    }
}

/// Runs `itp`, and checks the bound on evaluations, that the count returned
/// is the number of calls, and that a sign change lies within the tolerance.
fn solve(f: impl Fn(f64) -> f64, lower: f64, upper: f64, tolerance: f64) -> Root {
    let mut calls = 0;
    let root = itp(
        |x| {
            calls += 1;
            f(x)
        },
        lower,
        upper,
        &options(tolerance),
    )
    .unwrap();
    assert_eq!(root.evaluations, calls);
    assert!(root.converged);
    assert!(
        calls <= n_half(lower, upper, tolerance) + 1 + 3,
        "{calls} evaluations on [{lower}, {upper}]"
    );
    // At least one double away, for tolerances below the spacing of doubles.
    let (left, right) = (
        f((root.x - tolerance).min(root.x.next_down()).max(lower)),
        f((root.x + tolerance).max(root.x.next_up()).min(upper)),
    );
    assert!(
        root.value == 0.0 || left.signum() != right.signum() || left == 0.0 || right == 0.0,
        "no sign change within {tolerance} of {}",
        root.x
    );
    root
}

/// A name, the function, the bracket and the root.
type Case = (&'static str, fn(f64) -> f64, f64, f64, f64);

#[test]
fn closed_form_roots() {
    let tolerance = 1e-12;
    // Each root is within the tolerance of the double nearest the true root,
    // whose own error is half an ulp, far below it.
    let cases: [Case; 4] = [
        (
            "x^2 - 2",
            |x| x * x - 2.0,
            0.0,
            2.0,
            std::f64::consts::SQRT_2,
        ),
        (
            "exp(x) - 2",
            |x| x.exp() - 2.0,
            -1.0,
            3.0,
            std::f64::consts::LN_2,
        ),
        // mpmath.findroot(cos(x) - x) at 50 digits:
        // 0.73908513321516064165531208767387340401341175890076
        (
            "cos x - x",
            |x| x.cos() - x,
            0.0,
            1.0,
            0.739_085_133_215_160_6,
        ),
        ("2x + 1", |x| 2.0 * x + 1.0, -3.0, 1.0, -0.5),
    ];
    for (name, f, lower, upper, want) in cases {
        let root = solve(f, lower, upper, tolerance);
        assert!(
            (root.x - want).abs() <= tolerance,
            "{name}: {} against {want}",
            root.x
        );
    }
}

#[test]
fn roots_at_an_end_or_at_zero_need_no_step() {
    let root = itp(|x| x - 1.0, 1.0, 3.0, &Itp::default()).unwrap();
    assert_eq!(
        (root.x, root.value, root.iterations, root.evaluations),
        (1.0, 0.0, 0, 2)
    );
    let root = itp(|x| x - 3.0, 1.0, 3.0, &Itp::default()).unwrap();
    assert_eq!((root.x, root.iterations), (3.0, 0));
    // On a symmetric bracket the regula falsi point of a line through the
    // origin is 0 exactly.
    let root = itp(|x| x, -1.0, 1.0, &Itp::default()).unwrap();
    assert_eq!(
        (root.x, root.value, root.iterations, root.evaluations),
        (0.0, 0.0, 1, 3)
    );
}

/// Families of test functions of the kind Alefeld, Potra and Shi (1995)
/// collect: smooth, steep, nearly flat, with poles near the bracket and with
/// multiple roots.
#[test]
fn test_families_meet_the_bound() {
    for tolerance in [1e-4, 1e-8, 1e-12] {
        solve(
            |x| x.sin() - x / 2.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            tolerance,
        );
        for n in 1..=10 {
            let n = f64::from(n);
            let f = |x: f64| {
                -2.0 * (1..=20)
                    .map(|i| {
                        let i = f64::from(i);
                        (2.0 * i - 5.0).powi(2) / (x - i * i).powi(3)
                    })
                    .sum::<f64>()
            };
            solve(f, n * n + 1e-9, (n + 1.0).powi(2) - 1e-9, tolerance);
        }
        for (a, b) in [(-40.0, -1.0), (-100.0, -2.0), (-200.0, -3.0)] {
            solve(move |x: f64| a * x * (b * x).exp(), -9.0, 31.0, tolerance);
        }
        for a in [0.2, 1.0] {
            for n in [4, 6, 8, 10, 12] {
                solve(move |x: f64| x.powi(n) - a, 0.0, 5.0, tolerance);
            }
        }
        for n in [1, 2, 3, 5, 10, 20, 40, 100] {
            let n = f64::from(n);
            solve(
                move |x: f64| 2.0 * x * (-n).exp() - 2.0 * (-n * x).exp() + 1.0,
                0.0,
                1.0,
                tolerance,
            );
            solve(
                move |x: f64| (1.0 + (1.0 - n).powi(2)) * x - (1.0 - n * x).powi(2),
                0.0,
                1.0,
                tolerance,
            );
            solve(move |x: f64| x * x - (1.0 - x).powf(n), 0.0, 1.0, tolerance);
            solve(
                move |x: f64| (-n * x).exp() * (x - 1.0) + x.powf(n),
                0.0,
                1.0,
                tolerance,
            );
        }
        for n in [2, 5, 15, 20] {
            let n = f64::from(n);
            solve(
                move |x: f64| (n * x - 1.0) / ((n - 1.0) * x),
                0.01,
                1.0,
                tolerance,
            );
            solve(
                move |x: f64| x.powf(1.0 / n) - n.powf(1.0 / n),
                1.0,
                100.0,
                tolerance,
            );
        }
        solve(
            |x: f64| {
                if x == 0.0 {
                    0.0
                } else {
                    x * (-x.powi(-2)).exp()
                }
            },
            -1.0,
            4.0,
            tolerance,
        );
    }
}

#[test]
fn multiple_roots_and_flat_functions() {
    let tolerance = 1e-10;
    for (f, root) in [
        (
            (|x: f64| (x - 1.0 / 3.0).powi(3)) as fn(f64) -> f64,
            1.0 / 3.0,
        ),
        (|x: f64| x.powi(9), 0.0),
        (|x: f64| x.powi(19), 0.0),
    ] {
        let found = solve(f, -1.0, 2.0, tolerance);
        // The double nearest 1/3 is half an ulp from it, far below the
        // tolerance.
        assert!(
            (found.x - root).abs() <= tolerance,
            "{} against {root}",
            found.x
        );
    }
    // exp(-1/x^2) with the sign of x is 0 in double precision on
    // (-0.037, 0.037), so any point there is a root of the computed function.
    let flat = |x: f64| {
        if x == 0.0 {
            0.0
        } else {
            x.signum() * (-x.powi(-2)).exp()
        }
    };
    let found = solve(flat, -1.0, 2.0, tolerance);
    assert!(found.x.abs() < 0.04 && flat(found.x) == 0.0);
}

#[test]
fn a_jump_is_found_within_the_tolerance() {
    for jump in [0.1, 1.0 / 3.0, 0.7, 0.999_999] {
        let f = move |x: f64| if x < jump { -1.0 } else { 1.0 };
        for tolerance in [1e-3, 1e-9, 1e-15] {
            let found = solve(f, 0.0, 1.0, tolerance);
            assert!(
                (found.x - jump).abs() <= tolerance,
                "{} against {jump}",
                found.x
            );
        }
    }
}

#[test]
fn a_tolerance_below_the_spacing_of_doubles_ends_at_adjacent_doubles() {
    let found = solve(|x| x * x - 2.0, 1.0, 2.0, 1e-300);
    let (below, above) = (found.x.next_down(), found.x.next_up());
    assert!(below * below - 2.0 < 0.0 || above * above - 2.0 > 0.0);
    assert!((found.x - std::f64::consts::SQRT_2).abs() <= 2.0 * f64::EPSILON);
}

#[test]
fn infinite_values_count_by_their_sign() {
    let f = |x: f64| {
        if x < 0.25 {
            f64::NEG_INFINITY
        } else if x > 0.75 {
            f64::INFINITY
        } else {
            x - 0.5
        }
    };
    let found = solve(f, 0.0, 1.0, 1e-12);
    assert!((found.x - 0.5).abs() <= 1e-12);
    let pole = solve(|x: f64| (x - 0.4).recip(), 0.0, 1.0, 1e-12);
    assert!((pole.x - 0.4).abs() <= 1e-12);
}

#[test]
fn nan_is_an_error() {
    let f = |x: f64| {
        if (0.4..0.6).contains(&x) {
            f64::NAN
        } else {
            x - 0.5
        }
    };
    assert!(matches!(
        itp(f, 0.0, 1.0, &Itp::default()),
        Err(Error::NonFinite { x }) if (0.4..0.6).contains(&x)
    ));
    assert!(matches!(
        itp(
            |x| if x == 0.0 { f64::NAN } else { x },
            0.0,
            1.0,
            &Itp::default()
        ),
        Err(Error::NonFinite { x: 0.0 })
    ));
}

#[test]
fn invalid_input_is_refused() {
    let f = |x: f64| x - 0.5;
    assert!(matches!(
        itp(f, 0.0, 0.3, &Itp::default()),
        Err(Error::NoSignChange)
    ));
    for (lower, upper) in [
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (1.0, 1.0),
        (1.0, 0.0),
    ] {
        assert!(matches!(
            itp(f, lower, upper, &Itp::default()),
            Err(Error::Input(_))
        ));
    }
    for bad in [
        Itp {
            tolerance: 0.0,
            ..Itp::default()
        },
        Itp {
            tolerance: f64::INFINITY,
            ..Itp::default()
        },
        Itp {
            kappa1: Some(0.0),
            ..Itp::default()
        },
        Itp {
            kappa1: Some(f64::NAN),
            ..Itp::default()
        },
        Itp {
            kappa2: 0.99,
            ..Itp::default()
        },
        Itp {
            kappa2: 2.7,
            ..Itp::default()
        },
        Itp {
            kappa2: f64::NAN,
            ..Itp::default()
        },
    ] {
        assert!(
            matches!(itp(f, 0.0, 1.0, &bad), Err(Error::Input(_))),
            "{bad:?}"
        );
    }
}

#[test]
fn an_iteration_limit_below_the_bound_ends_unconverged() {
    let limited = Itp {
        tolerance: 1e-12,
        max_iterations: 2,
        ..Itp::default()
    };
    let found = itp(|x| x.cos() - x, 0.0, 1.0, &limited).unwrap();
    assert_eq!(
        (found.iterations, found.evaluations, found.converged),
        (2, 5, false)
    );
}

#[test]
fn smooth_simple_roots_take_far_fewer_steps_than_bisection() {
    // Superlinear convergence (Theorem 2.3): on these the steps are a small
    // fraction of bisection's 40.
    let tolerance = 1e-12;
    for f in [
        (|x: f64| x * x - 2.0) as fn(f64) -> f64,
        |x: f64| x.cos() - x,
        |x: f64| x.exp() - 2.0,
    ] {
        let found = solve(f, 0.0, 2.0, tolerance);
        assert!(found.iterations <= 12, "{} steps", found.iterations);
    }
}
