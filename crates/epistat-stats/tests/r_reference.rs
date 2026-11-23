//! Values checked against R 4.6.2: bit for bit, except `qnorm`, which is held
//! to two units in the last place because R's own last bit depends on whether
//! it was compiled with fused multiply-add.

use epistat_stats::{gamma, normal, quantile};
use epistat_testkit::Tolerance;

mod normal_values {
    include!("data/normal.rs");
}
mod gamma_values {
    include!("data/gamma.rs");
}
mod quantile_values {
    include!("data/quantile.rs");
}

fn check(name: &str, input: f64, got: f64, expected: u64) {
    assert_eq!(
        got.to_bits(),
        expected,
        "{name}({input:e}) = {got:e}, R gives {:e}",
        f64::from_bits(expected)
    );
}

#[test]
fn qnorm() {
    let within_two = Tolerance::exact().ulps(2);
    for &(p, z) in normal_values::QNORM {
        let p = f64::from_bits(p);
        let (got, expected) = (normal::quantile(p), f64::from_bits(z));
        assert!(
            within_two.accepts(got, expected),
            "quantile({p:e}) = {got:e}, R gives {expected:e}"
        );
    }
}

#[test]
fn pnorm() {
    for &(x, p) in normal_values::PNORM {
        let x = f64::from_bits(x);
        check("cdf", x, normal::cdf(x), p);
    }
}

#[test]
fn quantile_types() {
    use quantile_values::*;
    let floats = |bits: &[u64]| bits.iter().map(|&b| f64::from_bits(b)).collect::<Vec<_>>();
    let probs = floats(PROBS);
    for (sample, type1, type7) in [(SAMPLE_A, TYPE1_A, TYPE7_A), (SAMPLE_B, TYPE1_B, TYPE7_B)] {
        let sample = floats(sample);
        for (i, &p) in probs.iter().enumerate() {
            check("type1", p, quantile::type1(&sample, p).unwrap(), type1[i]);
            check("type7", p, quantile::type7(&sample, p).unwrap(), type7[i]);
        }
    }
}

#[test]
fn pchisq_and_lgamma() {
    let close = Tolerance::exact().rel(1e-13).abs(1e-300);
    for &(x, df, lower, upper) in gamma_values::CHISQ {
        let (x, df) = (f64::from_bits(x), f64::from_bits(df));
        let (got_lower, got_upper) = (gamma::chi_square_cdf(x, df), gamma::chi_square_sf(x, df));
        assert!(
            close.accepts(got_lower, f64::from_bits(lower)),
            "pchisq({x}, {df}) = {got_lower:e}, R {:e}",
            f64::from_bits(lower)
        );
        assert!(
            close.accepts(got_upper, f64::from_bits(upper)),
            "pchisq({x}, {df}, upper) = {got_upper:e}, R {:e}",
            f64::from_bits(upper)
        );
    }
    for &(x, value) in gamma_values::LGAMMA {
        let x = f64::from_bits(x);
        let expected = f64::from_bits(value);
        let got = gamma::ln_gamma(x);
        assert!(
            Tolerance::exact()
                .rel(1e-14)
                .abs(1e-15)
                .accepts(got, expected),
            "lgamma({x}) = {got:e}, R {expected:e}"
        );
    }
}

mod digamma_values {
    include!("data/digamma.rs");
}

#[test]
fn digamma_and_trigamma() {
    // The series and the recurrence each add about one rounding error per step,
    // at most ten steps, so 1e-14 relative. Near the zero of digamma at 1.4616
    // the value is small against its terms, so an absolute 1e-15 is allowed.
    let close = Tolerance::exact().rel(1e-14).abs(1e-15);
    for &(x, psi, psi1) in digamma_values::DIGAMMA {
        let x = f64::from_bits(x);
        let (got, got1) = (gamma::digamma(x), gamma::trigamma(x));
        assert!(
            close.accepts(got, f64::from_bits(psi)),
            "digamma({x}) = {got:e}, R {:e}",
            f64::from_bits(psi)
        );
        assert!(
            close.accepts(got1, f64::from_bits(psi1)),
            "trigamma({x}) = {got1:e}, R {:e}",
            f64::from_bits(psi1)
        );
    }
}

mod cipoisson_values {
    include!("data/cipoisson.rs");
}

/// The relative deviation allowed in a gamma quantile `x` with shape `a` and
/// tail probability `t = min(p, 1 - p)`.
///
/// The incomplete gamma function is computed as a series or continued
/// fraction times `exp((a - 1) ln x + ... - x - lnGamma(a))`, so the sum of
/// the magnitudes of the exponent's terms times `eps` is the relative error in
/// the probability. A relative error `d` in the tail moves the quantile by
/// `d t / (x f(x))` in relative terms, `f` the density. Both the Rust and the R
/// quantile carry this error, so the coefficient is 8, plus eight units in the
/// last place for the Newton stopping rule.
fn qgamma_tolerance(x: f64, shape: f64, t: f64) -> f64 {
    let ln_norm = epistat_stats::gamma::ln_gamma(shape);
    let exponent = ((shape - 1.0) * x.ln()).abs() + x + ln_norm.abs();
    let density = ((shape - 1.0) * x.ln() - x - ln_norm).exp();
    8.0 * f64::EPSILON * exponent * t / (x * density) + 8.0 * f64::EPSILON
}

#[test]
fn qgamma_quantiles() {
    use epistat_stats::poisson::gamma_quantile;
    let mut worst = 0.0_f64;
    for &(p, shape, q) in cipoisson_values::QGAMMA {
        let (p, shape, expected) = (f64::from_bits(p), f64::from_bits(shape), f64::from_bits(q));
        let got = gamma_quantile(p, shape).unwrap();
        let deviation = ((got - expected) / expected).abs();
        let bound = qgamma_tolerance(expected, shape, p.min(1.0 - p));
        worst = worst.max(deviation / bound);
        assert!(
            deviation <= bound,
            "qgamma({p}, {shape}) = {got:e}, R gives {expected:e}, bound {bound:e}"
        );
    }
    println!("worst deviation from qgamma, in units of its bound: {worst:e}");
}

/// Relative deviation allowed in a limit. The shapes are at most 1001 and the
/// tails 0.005 to 0.05, where `qgamma_tolerance` is below 1e-13. Anscombe's
/// limits are a square root and a square of a normal quantile, a few units in
/// the last place. The bound is 1e-13 for both.
const CIPOISSON_REL: f64 = 1e-13;

#[test]
fn cipoisson_limits() {
    use epistat_stats::poisson::{Method, cipoisson};
    let mut worst = 0.0_f64;
    for (method, rows) in [
        (Method::Exact, cipoisson_values::CIPOISSON_EXACT),
        (Method::Anscombe, cipoisson_values::CIPOISSON_ANSCOMBE),
    ] {
        for &(k, time, level, lower, upper) in rows {
            let (k, time, level) = (
                f64::from_bits(k),
                f64::from_bits(time),
                f64::from_bits(level),
            );
            let got = cipoisson(k, time, level, method).unwrap();
            for (name, got, bits) in [("lower", got.lower, lower), ("upper", got.upper, upper)] {
                let expected = f64::from_bits(bits);
                match got {
                    None => assert!(
                        expected.is_nan(),
                        "{method:?} k={k} time={time} level={level}: {name} is None, R gives {expected}"
                    ),
                    Some(value) => {
                        let deviation = if expected == 0.0 {
                            value.abs()
                        } else {
                            ((value - expected) / expected).abs()
                        };
                        worst = worst.max(deviation);
                        assert!(
                            deviation <= CIPOISSON_REL,
                            "{method:?} k={k} time={time} level={level}: {name} {value:e}, R gives {expected:e}"
                        );
                    }
                }
            }
        }
    }
    println!("worst relative deviation from cipoisson: {worst:e}");
}

mod student_values {
    include!("data/student.rs");
}

/// The relative deviation allowed in `I_x(a, b)`. It is a continued fraction,
/// whose error is a few eps, times a prefactor `exp(a ln x + b ln y - ln B)`
/// formed on the log scale. The sum of the magnitudes of the exponent's terms,
/// times eps, is the relative error in the probability. R's own routine carries
/// an error of the same kind, so the coefficient is 8, plus 64 eps for the
/// fraction.
fn beta_bound(a: f64, b: f64, x: f64, y: f64) -> f64 {
    use epistat_stats::gamma::ln_gamma;
    let terms = (a * x.ln()).abs()
        + (b * y.ln()).abs()
        + ln_gamma(a).abs()
        + ln_gamma(b).abs()
        + ln_gamma(a + b).abs();
    8.0 * f64::EPSILON * terms + 64.0 * f64::EPSILON
}

/// The bound on the tail of the t distribution at `z`, the beta bound with
/// `x = nu / (nu + z^2)`.
fn t_bound(z: f64, nu: f64) -> f64 {
    let q = nu + z * z;
    beta_bound(0.5 * nu, 0.5, nu / q, z * z / q)
}

#[test]
fn pt_qt_and_pbeta() {
    use epistat_stats::student::{regularised_beta, t_cdf_sf, t_ln_density, t_quantile};
    let mut worst = 0.0_f64;
    let mut check = |name: &str, got: f64, want: f64, bound: f64| {
        if want == 0.0 {
            assert_eq!(got.to_bits(), 0.0_f64.to_bits(), "{name}");
            return;
        }
        let deviation = ((got - want) / want).abs();
        worst = worst.max(deviation / bound);
        assert!(
            deviation <= bound,
            "{name} = {got:e}, R {want:e}, bound {bound:e}"
        );
    };
    for &(z, nu, lower, upper) in student_values::PT {
        let (z, nu) = (f64::from_bits(z), f64::from_bits(nu));
        let (cdf, sf) = t_cdf_sf(z, nu);
        // The larger tail is one minus the smaller, with an absolute error of
        // the smaller's bound.
        let bound = t_bound(z, nu);
        let (small, big) = if z > 0.0 { (sf, cdf) } else { (cdf, sf) };
        let (want_small, want_big) = if z > 0.0 {
            (f64::from_bits(upper), f64::from_bits(lower))
        } else {
            (f64::from_bits(lower), f64::from_bits(upper))
        };
        if z == 0.0 {
            assert_eq!(cdf.to_bits(), 0.5_f64.to_bits());
            continue;
        }
        check(
            &format!("pt({z}, {nu}) small tail"),
            small,
            want_small,
            bound,
        );
        check(
            &format!("pt({z}, {nu}) large tail"),
            big,
            want_big,
            bound * want_small / want_big + 4.0 * f64::EPSILON,
        );
    }
    for &(p, nu, q) in student_values::QT {
        let (p, nu, want) = (f64::from_bits(p), f64::from_bits(nu), f64::from_bits(q));
        let got = t_quantile(p, nu).unwrap();
        if p.to_bits() == 0.5_f64.to_bits() {
            assert_eq!(got.to_bits(), 0.0_f64.to_bits());
            continue;
        }
        // A relative error d in the tail moves the quantile by
        // d tail / (t f(t)) in relative terms.
        let tail = p.min(1.0 - p);
        let density = t_ln_density(want, nu).exp();
        let mut bound = t_bound(want, nu) * tail / (want.abs() * density) + 16.0 * f64::EPSILON;
        if nu < 1.0 {
            // R's qt finds the root for fewer than one degree of freedom by a
            // search that stops at a relative step of 1e-11. Ours inverts the
            // distribution function to the bound above, which
            // the unit tests check.
            bound = bound.max(1e-11);
        }
        check(&format!("qt({p}, {nu})"), got, want, bound);
    }
    for &(x, a, b, want) in student_values::PBETA {
        let (x, a, b, want) = (
            f64::from_bits(x),
            f64::from_bits(a),
            f64::from_bits(b),
            f64::from_bits(want),
        );
        let y = 1.0 - x;
        let got = regularised_beta(a, b, x, y);
        // y is rounded, so a relative error eps / y in it enters through b ln y.
        let bound = beta_bound(a, b, x, y) + b * f64::EPSILON / y;
        check(&format!("pbeta({x}, {a}, {b})"), got, want, bound);
    }
    println!("worst deviation from pt, qt and pbeta, in units of its bound: {worst:e}");
}
