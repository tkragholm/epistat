//! B-splines checked against R's `splines::splineDesign`.

use epistat_linalg::basis::{BSpline, Basis};
use epistat_testkit::{Tolerance, assert_close};

mod reference {
    include!("data/bspline.rs");
}

fn floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from_bits(b)).collect()
}

#[test]
fn values_and_derivatives() {
    let basis = BSpline::new(3, &[0.2, 0.45, 0.7], 0.0, 1.0);
    let tolerance = Tolerance::exact().abs(1e-14).ulps(4);
    for (i, &x) in reference::X.iter().enumerate() {
        assert_close(&basis.values(x), &floats(&reference::DERIV0[i]), tolerance);
        assert_close(
            &basis.derivatives(x),
            &floats(&reference::DERIV1[i]),
            Tolerance::exact().abs(1e-12).ulps(4),
        );
    }
}
