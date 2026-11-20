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
