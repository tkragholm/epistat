//! `brent::minimize` against closed-form minima, and its rule for values that
//! are not finite.
//!
//! The search stops when the interval around `x` is no wider than about
//! `4 tol1`, `tol1 = sqrt(ε) |x| + tolerance / 3`, with the minimum inside it,
//! so on a unimodal function the point found is within `2 tol1` of the
//! minimum. Golden-section search needs `ln(w / (4 tol1)) / ln(1 / 0.618) + 1`
//! evaluations to shrink an interval of width `w` that far, which bounds the
//! count on the smooth cases.

use epistat_optim::Error;
use epistat_optim::brent::{Brent, Minimum, minimize};

fn run(f: impl Fn(f64) -> f64, lower: f64, upper: f64, tolerance: f64) -> Minimum {
    let options = Brent {
        tolerance,
        ..Brent::default()
    };
    let m = minimize::<Error>(|x| Ok(f(x)), lower, upper, &options).unwrap();
    assert!(m.converged);
    assert_eq!(m.evaluations, m.iterations + 1);
    m
}

fn tol1(x: f64, tolerance: f64) -> f64 {
    f64::EPSILON.sqrt() * x.abs() + tolerance / 3.0
}

fn golden_section_count(width: f64, x: f64, tolerance: f64) -> usize {
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let steps = ((width / (4.0 * tol1(x, tolerance))).ln() / (1.0 / golden).ln()).ceil();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a small non-negative count"
    )]
    let steps = steps as usize;
    steps + 1
}

/// A name, the function, the interval and the minimum.
type Case = (&'static str, fn(f64) -> f64, f64, f64, f64);

#[test]
fn smooth_minima_in_fewer_evaluations_than_golden_section() {
    let cases: [Case; 3] = [
        ("parabola", |x| (x - 1.0 / 3.0).powi(2), 0.0, 1.0, 1.0 / 3.0),
        (
            "cos",
            f64::cos,
            0.0,
            2.0 * std::f64::consts::PI,
            std::f64::consts::PI,
        ),
        (
            "shifted quartic",
            |x| (x - 2.0).powi(2) * (1.0 + x * x),
            0.5,
            4.0,
            2.0,
        ),
    ];
    for tolerance in [1e-4, 1e-8, 1e-12] {
        for (name, f, lower, upper, want) in cases {
            let m = run(f, lower, upper, tolerance);
            assert!(
                (m.x - want).abs() <= 2.0 * tol1(want, tolerance),
                "{name} at {tolerance}: {} against {want}",
                m.x
            );
            assert!(
                m.evaluations <= golden_section_count(upper - lower, want, tolerance),
                "{name} at {tolerance}: {} evaluations",
                m.evaluations
            );
        }
    }
}

#[test]
fn flat_and_kinked_minima() {
    // x^4 is flat to rounding within |x| < eps^(1/4) of 0, so only the width
    // of the final interval bounds the point.
    let tolerance = 1e-8;
    let m = run(|x| x.powi(4), -1.0, 2.0, tolerance);
    assert!(m.x.abs() <= 2.0 * tol1(0.0, tolerance) + f64::EPSILON.powf(0.25));
    // |x - 0.3| has no parabola to follow; the search falls back to golden
    // section and still brackets the kink.
    let m = run(|x| (x - 0.3).abs(), -1.0, 3.0, tolerance);
    assert!((m.x - 0.3).abs() <= 2.0 * tol1(0.3, tolerance));
}

#[test]
fn a_minimum_at_an_end_is_found_within_two_tol1() {
    let tolerance = 1e-7;
    let m = run(f64::exp, -2.0, 1.0, tolerance);
    assert!(m.x - -2.0 <= 2.0 * tol1(-2.0, tolerance) * (1.0 + 1e-12));
    let m = run(|x| -x, -2.0, 1.0, tolerance);
    assert!(1.0 - m.x <= 2.0 * tol1(1.0, tolerance) * (1.0 + 1e-12));
}

#[test]
fn a_constant_function_returns_a_point_of_the_interval() {
    let m = run(|_| 2.0, -1.0, 1.0, 1e-8);
    assert!(m.value.to_bits() == 2.0_f64.to_bits() && (-1.0..=1.0).contains(&m.x));
}

#[test]
fn plus_infinity_is_worse_than_any_finite_value() {
    let tolerance = 1e-9;
    for cut in [0.1, 0.5, 0.7] {
        let f = move |x: f64| {
            if x < cut {
                f64::INFINITY
            } else {
                (x - 0.8).powi(2)
            }
        };
        let m = run(f, 0.0, 1.0, tolerance);
        assert!(m.value.is_finite());
        assert!(
            (m.x - 0.8).abs() <= 2.0 * tol1(0.8, tolerance),
            "{cut}: {}",
            m.x
        );
    }
    let m = run(
        |x: f64| {
            if x > 0.6 {
                f64::INFINITY
            } else {
                (x - 0.2).powi(2)
            }
        },
        0.0,
        1.0,
        tolerance,
    );
    assert!((m.x - 0.2).abs() <= 2.0 * tol1(0.2, tolerance));
}

#[test]
fn nan_and_minus_infinity_are_errors() {
    let options = Brent::default();
    let nan = |x: f64| Ok(if x > 0.5 { f64::NAN } else { x * x });
    assert!(matches!(
        minimize::<Error>(nan, 0.0, 1.0, &options),
        Err(Error::NonFinite { x }) if x > 0.5
    ));
    let below = |x: f64| Ok(if x < 0.3 { f64::NEG_INFINITY } else { x });
    assert!(matches!(
        minimize::<Error>(below, 0.0, 1.0, &options),
        Err(Error::NonFinite { x }) if x < 0.3
    ));
}

#[derive(Debug, PartialEq)]
enum Caller {
    Failed(f64),
    Optim(Error),
}

impl From<Error> for Caller {
    fn from(error: Error) -> Self {
        Self::Optim(error)
    }
}

#[test]
fn an_error_of_the_objective_stops_the_search_and_is_returned() {
    let mut calls = 0;
    let result = minimize(
        |x: f64| {
            calls += 1;
            if calls == 3 {
                Err(Caller::Failed(x))
            } else {
                Ok(x * x)
            }
        },
        -1.0,
        1.0,
        &Brent::default(),
    );
    assert!(matches!(result, Err(Caller::Failed(_))));
    assert_eq!(calls, 3);
    let refused = minimize(|x: f64| Ok::<_, Caller>(x), 1.0, 0.0, &Brent::default());
    assert!(matches!(refused, Err(Caller::Optim(Error::Input(_)))));
}

#[test]
fn invalid_input_is_refused() {
    let f = |x: f64| Ok::<_, Error>(x * x);
    for (lower, upper, tolerance) in [
        (f64::NAN, 1.0, 1e-6),
        (0.0, f64::INFINITY, 1e-6),
        (1.0, 1.0, 1e-6),
        (2.0, 1.0, 1e-6),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, f64::NAN),
    ] {
        let options = Brent {
            tolerance,
            ..Brent::default()
        };
        assert!(matches!(
            minimize(f, lower, upper, &options),
            Err(Error::Input(_))
        ));
    }
}

#[test]
fn the_iteration_limit_ends_the_search_unconverged() {
    let options = Brent {
        tolerance: 1e-10,
        max_iterations: 3,
    };
    let m = minimize::<Error>(|x| Ok((x - 0.3).powi(2)), 0.0, 1.0, &options).unwrap();
    assert_eq!((m.iterations, m.evaluations, m.converged), (3, 4, false));
}
