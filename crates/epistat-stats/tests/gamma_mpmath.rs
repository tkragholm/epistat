//! `dist::gamma::{cdf, sf}` against mpmath at 60 digits, and the identities
//! of DLMF chapter 8 that need no reference.
//!
//! The reference is `tests/data/gamma_mpmath.json`, written by
//! `tests/data/gamma_mpmath.py`: shapes from the smallest subnormal to `1e4`
//! across `x = a λ` for `λ` from `1e-3` to 10, and shapes from `1e5` to
//! `1e7` near the centre, with both `P(a, x)` and `Q(a, x)`.

use epistat_stats::dist::gamma::{cdf, ln_gamma, quantile, sf};
use epistat_stats::dist::{chi_square, normal};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
    closed: Vec<Case>,
    recurrence: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    a: f64,
    x: f64,
    prefactor: f64,
}

#[derive(Deserialize)]
struct Case {
    a: f64,
    x: f64,
    lower: f64,
    upper: f64,
}

const EPS: f64 = f64::EPSILON;

/// The shape from which the prefactor `x^a e^-x / Γ(a + 1)` is formed about
/// `x = a` with the Stirling series (`STIRLING_FROM` in `dist/stirling.rs`).
const STIRLING_FROM: f64 = 10.0;

/// `λ - 1 - ln λ` at `λ = x / a`, accurate enough for a bound.
fn gap(a: f64, x: f64) -> f64 {
    let lambda = x / a;
    lambda - 1.0 - lambda.ln()
}

/// The relative error allowed in the tail that the function computes
/// directly.
///
/// Every region multiplies a sum or fraction, formed to a few eps, by the
/// prefactor `D(a, x) = x^a e^-x / Γ(a + 1)`, or (the uniform expansion) adds
/// a term of the same exponential size to `erfc`. An exponential carries the
/// absolute error of its exponent as a relative error. From
/// [`STIRLING_FROM`] on the exponent is `-a (λ - 1 - ln λ) - ln Γ*(a)` with
/// `ln Γ*(a) < 1/120`: the gap is formed to a few eps relative, and from
/// `λ = x / a` or `(x - a) / a`, which carry one rounding, whose effect on
/// `a (λ - 1 - ln λ)` is `a |λ - 1| eps`. Below it the exponent is
/// the product `x^a e^-x / Γ(a + 1)`, with `Γ(a + 1)` a product of at most
/// ten factors over a Taylor series: some sixteen roundings, which the
/// coefficient below covers with the constant term doubled. Where that
/// product leaves the normal range (`x >= 700`, or `x^a` or the product not
/// a normal double) it is the exponent `a ln x - x - ln Γ(a + 1)`, each term
/// with a relative error of a few eps. The coefficient 8 covers the few eps
/// of each and the roundings of the sum, fraction or series.
fn bound(a: f64, x: f64) -> f64 {
    let exponent = if a < STIRLING_FROM {
        let product = x.powf(a) * (-x).exp() / ln_gamma(a + 1.0).exp();
        if x < 700.0 && product.is_finite() && product >= 1.01 * f64::MIN_POSITIVE {
            1.0
        } else {
            (a * x.ln()).abs() + x + ln_gamma(a + 1.0).abs()
        }
    } else {
        a * gap(a, x) + a * (x / a - 1.0).abs()
    };
    8.0 * EPS * (1.0 + exponent)
}

/// The deviation of the two tails at a case from the reference, in units of
/// the allowed error.
///
/// The tail computed directly is at most `0.7` (`P(1/2, 1/2) = 0.68` at the
/// corner of the regions), so it is the smaller tail or above it by at most
/// a factor `7/3`, and the other tail, one minus it, inherits at most `7/3`
/// times its absolute error plus the rounding of the subtraction. Both tails
/// are held to `3 bound` times the smaller reference tail plus `2 eps` of
/// their own value.
fn deviation(c: &Case) -> (f64, f64) {
    let allowed = |want: f64| 3.0 * bound(c.a, c.x) * c.lower.min(c.upper) + 2.0 * EPS * want;
    let p = cdf(c.x, c.a);
    let q = sf(c.x, c.a);
    (
        (p - c.lower).abs() / allowed(c.lower),
        (q - c.upper).abs() / allowed(c.upper),
    )
}

fn reference() -> Reference {
    load_json(format!(
        "{}/tests/data/gamma_mpmath.json",
        env!("CARGO_MANIFEST_DIR")
    ))
}

#[test]
fn grid() {
    let r = reference();
    assert!(r.cases.len() > 300);
    let mut worst = 0.0_f64;
    let mut worst_label = String::new();
    // The largest relative error of the smaller tail, by shape class.
    let mut by_class = [0.0_f64; 4];
    for c in &r.cases {
        let class = match c.a {
            a if a < 1.0 => 0,
            a if a < 12.0 => 1,
            a if a <= 1e4 => 2,
            _ => 3,
        };
        let relative = if c.lower <= c.upper {
            ((cdf(c.x, c.a) - c.lower) / c.lower).abs()
        } else {
            ((sf(c.x, c.a) - c.upper) / c.upper).abs()
        };
        by_class[class] = by_class[class].max(relative);
        let (p, q) = deviation(c);
        for (name, units) in [("P", p), ("Q", q)] {
            let label = format!("{name}({}, {:e})", c.a, c.x);
            assert!(units <= 1.0, "{label}: {units:.3} of the bound");
            if units > worst {
                worst = units;
                worst_label = label;
            }
        }
    }
    println!(
        "largest relative error of the smaller tail: a < 1 {:.1e}, 1 <= a < 12 {:.1e}, \
         12 <= a <= 1e4 {:.1e}, a >= 1e5 {:.1e}",
        by_class[0], by_class[1], by_class[2], by_class[3]
    );
    println!("worst deviation from mpmath, in units of its bound: {worst:.3} ({worst_label})");
}

/// Integer shapes against `e^-x e_n(x)` in exact rationals (DLMF 8.4.10),
/// half-integer shapes against DLMF 8.4.14, and shape 1 and 1/2 against
/// `exp`, `expm1` and the normal tail.
#[test]
fn closed_forms() {
    let r = reference();
    assert!(r.closed.len() > 100);
    for c in &r.closed {
        let (p, q) = deviation(c);
        assert!(
            p <= 1.0 && q <= 1.0,
            "a = {}, x = {}: {p:.3}, {q:.3}",
            c.a,
            c.x
        );
    }
    for x in [1e-10_f64, 0.003, 0.5, 1.0, 4.0, 30.0, 700.0] {
        // exp and expm1 are correctly rounded to within an ulp or so.
        let tolerance = bound(1.0, x) + 2.0 * EPS;
        let p = -(-x).exp_m1();
        let q = (-x).exp();
        assert!((cdf(x, 1.0) - p).abs() <= tolerance * p, "P(1, {x})");
        assert!((sf(x, 1.0) - q).abs() <= tolerance * q, "Q(1, {x})");
    }
    for x in [1e-6_f64, 0.1, 0.5, 2.0, 10.0, 200.0] {
        // erfc(sqrt x) = 2 Φ(-sqrt(2x)). The normal tail at y = sqrt(2x)
        // carries the rounding of y as a relative error of y² eps = 2x eps,
        // and a few eps of its own.
        let y = (2.0 * x).sqrt();
        let (q, p) = (2.0 * normal::cdf(-y), 2.0 * normal::cdf(y) - 1.0);
        let small = p.min(q);
        let tolerance = (3.0 * bound(0.5, x) + 8.0 * EPS * (1.0 + 2.0 * x)) * small;
        assert!(
            (sf(x, 0.5) - q).abs() <= tolerance + 2.0 * EPS * q,
            "Q(1/2, {x})"
        );
        // P is formed as 2 Φ(y) - 1, which carries the absolute rounding
        // error of Φ(y) near 1/2 twice: 2 eps.
        assert!(
            (cdf(x, 0.5) - p).abs() <= tolerance + 2.0 * EPS * p + 2.0 * EPS,
            "P(1/2, {x})"
        );
    }
}

/// `P(a + 1, x) = P(a, x) - D(a, x)` and `Q(a + 1, x) = Q(a, x) + D(a, x)`
/// (DLMF 8.8.5, 8.8.6), with `D` from mpmath, on random `(a, x)` in
/// `(0, 1]^2` and `(0, 500]^2`. Each side carries the error allowed at its
/// shape, so the identity holds to their sum plus the rounding of the
/// difference.
#[test]
fn recurrence() {
    let r = reference();
    assert!(r.recurrence.len() > 150);
    for s in &r.recurrence {
        let (a, x, d) = (s.a, s.x, s.prefactor);
        let allowed = |shape: f64| {
            let (p, q) = (cdf(x, shape), sf(x, shape));
            3.0 * bound(shape, x) * p.min(q) + 2.0 * EPS * p.max(q)
        };
        let tolerance = allowed(a) + allowed(a + 1.0);
        let lower = cdf(x, a + 1.0) - (cdf(x, a) - d);
        let upper = sf(x, a + 1.0) - (sf(x, a) + d);
        assert!(lower.abs() <= tolerance, "P at a = {a}, x = {x}: {lower:e}");
        assert!(upper.abs() <= tolerance, "Q at a = {a}, x = {x}: {upper:e}");
    }
}

/// The density `x^(a-1) e^-x / Γ(a)`, accurate enough for a bound.
fn density(a: f64, x: f64) -> f64 {
    ((a - 1.0) * x.ln() - x - ln_gamma(a)).exp()
}

/// Values at a border of the regions and one ulp either side agree within
/// the error allowed there and the change of the function over the step.
fn continuous(points: &[(f64, f64)], label: &str) {
    for pair in points.windows(2) {
        let ((a0, x0), (a1, x1)) = (pair[0], pair[1]);
        for (name, f) in [("P", cdf as fn(f64, f64) -> f64), ("Q", sf)] {
            let (v0, v1) = (f(x0, a0), f(x1, a1));
            let (p, q) = (cdf(x0, a0), sf(x0, a0));
            // Over one ulp of a the function moves by about
            // |d P / d a| ulp(a), below a |ln x - ψ(a)| P eps; the bound
            // carries that as the a |λ - 1| and a ln x terms.
            let change = density(a0, x0) * (x1 - x0).abs();
            let tolerance =
                2.0 * (3.0 * bound(a0, x0) * p.min(q) + 2.0 * EPS * v0.max(v1)) + change;
            assert!(
                (v0 - v1).abs() <= tolerance,
                "{label}: {name} at ({a0}, {x0}) = {v0:e}, at ({a1}, {x1}) = {v1:e}"
            );
        }
    }
}

fn around(v: f64) -> [f64; 3] {
    [v.next_down(), v, v.next_up()]
}

/// The borders of the method: `a = α(x)`, `x = 1.5`, `a = 12`, `η = ±1` and
/// the switch of the prefactor at `a = 10`.
#[test]
fn continuity_across_the_borders() {
    // α(x) = x from x = 1/2, and ln(1/2) / ln(x / 2) below.
    for x in [0.5_f64, 0.7, 1.2, 1.5, 3.0, 11.0] {
        continuous(&around(x).map(|a| (a, x)), &format!("a = α({x})"));
    }
    for x in [1e-300_f64, 1e-8, 0.01, 0.2, 0.4999] {
        let alpha = -std::f64::consts::LN_2 / (x.ln() - std::f64::consts::LN_2);
        continuous(&around(alpha).map(|a| (a, x)), &format!("a = α({x})"));
    }
    for a in [1e-3, 0.3, 1.0, 1.4999] {
        continuous(&around(1.5).map(|x| (a, x)), &format!("x = 1.5, a = {a}"));
    }
    for lambda in [0.1, 0.5, 1.0, 2.0, 3.0] {
        let x = 12.0 * lambda;
        continuous(&around(12.0).map(|a| (a, x)), &format!("a = 12, x = {x}"));
    }
    for a in [12.0_f64, 30.0, 500.0, 1e4] {
        for (lo, hi) in [(0.25, 0.35), (2.2, 2.5)] {
            // The x where λ - 1 - ln λ = 1/2, by bisection on the formed gap.
            let (mut lo, mut hi) = (a * lo, a * hi);
            let below = |x: f64| x / a - 1.0 - (x / a).ln() <= 0.5;
            let inside_lo = below(lo);
            for _ in 0..200 {
                let mid = f64::midpoint(lo, hi);
                if below(mid) == inside_lo {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let x = lo;
            let mut points = vec![(a, x)];
            for _ in 0..3 {
                let (_, first) = points[0];
                let (_, last) = points[points.len() - 1];
                points.insert(0, (a, first.next_down()));
                points.push((a, last.next_up()));
            }
            continuous(&points, &format!("η = ±1, a = {a}"));
        }
    }
    for lambda in [0.05, 0.25, 1.0, 2.5, 4.0] {
        let x = 10.0 * lambda;
        continuous(&around(10.0).map(|a| (a, x)), &format!("a = 10, x = {x}"));
    }
}

/// The table of edges in the method's documentation.
#[test]
fn edges() {
    for shape in [1e-300, 0.5, 3.0, 1e10] {
        for x in [0.0, -0.0] {
            assert_eq!(cdf(x, shape).to_bits(), 0.0_f64.to_bits());
            assert_eq!(sf(x, shape).to_bits(), 1.0_f64.to_bits());
        }
        assert_eq!(cdf(f64::INFINITY, shape).to_bits(), 1.0_f64.to_bits());
        assert_eq!(sf(f64::INFINITY, shape).to_bits(), 0.0_f64.to_bits());
    }
    assert_eq!(
        chi_square::sf(f64::INFINITY, 3.0).to_bits(),
        0.0_f64.to_bits()
    );
    // An infinite shape: the limit at finite x, NaN at x = inf.
    assert_eq!(cdf(1e300, f64::INFINITY).to_bits(), 0.0_f64.to_bits());
    assert_eq!(sf(1e300, f64::INFINITY).to_bits(), 1.0_f64.to_bits());
    assert!(cdf(f64::INFINITY, f64::INFINITY).is_nan());
    for (x, shape) in [
        (1.0, 0.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
    ] {
        assert!(
            cdf(x, shape).is_nan() && sf(x, shape).is_nan(),
            "{x} {shape}"
        );
    }

    // The smallest subnormal shape: P = x^a to first order, which is 1, and
    // Q = a E1(x), from mpmath, to within a subnormal ulp.
    let tiny = f64::from_bits(1);
    for (x, want) in [
        (1e-300, "3.4100327489208971387e-321"),
        (1.0, "1.0839006523431638578e-324"),
        (5.0, "5.673334029200999105e-327"),
    ] {
        // Subnormal: parsed from mpmath's digits to the nearest double.
        let want: f64 = want.parse().unwrap();
        assert_eq!(cdf(x, tiny), 1.0, "P({tiny:e}, {x})");
        let q = sf(x, tiny);
        assert!((q - want).abs() <= tiny, "Q({tiny:e}, {x}) = {q:e}");
    }

    // a = x = 1e-300: Q = 6.902e-298 (mpmath), positive and to the bound.
    let (a, x) = (1e-300, 1e-300);
    let want = 6.901_983_122_333_122e-298;
    let q = sf(x, a);
    assert!(q > 0.0);
    assert!(
        (q - want).abs() <= 3.0 * bound(a, x) * want + 2.0 * EPS * want,
        "{q:e}"
    );

    // Q(3, 745) = 7.853e-319 (mpmath) keeps its gradual underflow: within a
    // subnormal ulp.
    let q = sf(745.0, 3.0);
    let want: f64 = "7.8534308073502550296e-319".parse().unwrap();
    assert!((q - want).abs() <= tiny, "{q:e}");
}

/// `P(a, a) = 1/2 + (1 / sqrt(2πa)) (1/3 + 1 / (540 a)) + O(a^-5/2)`, from
/// DLMF 8.12.15 with `c_0(0) = -1/3` and `c_1(0) = -1/540` (8.12.17); the
/// omitted terms are below `1e-37` at `a = 1e15`. The reference is formed in
/// double precision with an error of a few eps.
#[test]
fn the_centre_of_a_very_large_shape() {
    for a in [1e15_f64, 1e300] {
        let front = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * a.sqrt());
        let p = 0.5 + front * (1.0 / 3.0 + 1.0 / (540.0 * a));
        let q = 0.5 - front * (1.0 / 3.0 + 1.0 / (540.0 * a));
        let tolerance = 3.0 * bound(a, a) * q + 4.0 * EPS;
        assert!(
            (cdf(a, a) - p).abs() <= tolerance,
            "P({a:e}, {a:e}) = {}",
            cdf(a, a)
        );
        assert!(
            (sf(a, a) - q).abs() <= tolerance,
            "Q({a:e}, {a:e}) = {}",
            sf(a, a)
        );
    }
}

/// `quantile` returns an `x` whose tail is the given probability: the tail
/// at `x` carries the error allowed there, and the iteration stops within a
/// few ulps of the root, which moves the tail by the density times that.
#[test]
fn the_quantile_inverts_both_tails() {
    let mut worst = 0.0_f64;
    for shape in [
        1e-3_f64, 0.02, 0.3, 1.0, 2.5, 11.0, 12.5, 80.0, 1000.0, 1e5, 1e7,
    ] {
        for t in [1e-300_f64, 1e-100, 1e-20, 1e-8, 1e-3, 0.05, 0.3, 0.5] {
            for upper in [false, true] {
                // The upper tail is given as q = 1 - p, so its smallest
                // values are 1 - p for p near 1: from 1e-16 up.
                let p = if upper { 1.0 - t.max(1e-16) } else { t };
                let t = if upper { 1.0 - p } else { p };
                let x = quantile(p, shape);
                let label = format!("quantile({p}, {shape}) = {x:e}");
                assert!(x.is_finite() && x >= 0.0, "{label}");
                if x == 0.0 {
                    // The root is below the smallest subnormal.
                    assert!(!upper && cdf(f64::from_bits(1), shape) >= t, "{label}");
                    continue;
                }
                let got = if upper { sf(x, shape) } else { cdf(x, shape) };
                // The tail that was solved for is t; its own error and the
                // stop within 4 ulps (8 eps x) at the density.
                let tolerance =
                    3.0 * bound(shape, x) * t + 2.0 * EPS * t + density(shape, x) * 8.0 * EPS * x;
                let units = (got - t).abs() / tolerance;
                assert!(
                    units <= 1.0,
                    "{label}: tail {got:e}, {units:.3} of the bound"
                );
                worst = worst.max(units);
            }
        }
    }
    println!("worst deviation of the quantile's tail, in units of its bound: {worst:.3}");
    // Shape 1 and the chi-square with 2 degrees of freedom are -ln(1 - p).
    for p in [1e-300_f64, 1e-12, 0.1, 0.5, 0.9, 1.0 - 1e-12] {
        let want = -(-p).ln_1p();
        assert!((quantile(p, 1.0) - want).abs() <= 8.0 * EPS * want, "{p}");
        let want = 2.0 * want;
        let got = chi_square::quantile(p, 2.0);
        assert!(
            (got - want).abs() <= 8.0 * EPS * want,
            "qchisq({p}, 2) = {got:e}"
        );
    }
}
