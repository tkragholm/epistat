//! Values checked against R 4.6.2: bit for bit, except `qnorm`, which is held
//! to two units in the last place because R's own last bit depends on whether
//! it was compiled with fused multiply-add.

use epistat_stats::{normal, quantile};

mod normal_values {
    include!("data/normal.rs");
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
    for &(p, z) in normal_values::QNORM {
        let p = f64::from_bits(p);
        let got = normal::quantile(p);
        assert!(
            got.to_bits().abs_diff(z) <= 2,
            "quantile({p:e}) = {got:e}, R gives {:e}",
            f64::from_bits(z)
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
