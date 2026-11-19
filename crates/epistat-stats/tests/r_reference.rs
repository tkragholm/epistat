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
