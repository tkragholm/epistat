//! `attributable_logistic` gives identical bits in rayon pools of 1, 2, 3, 4
//! and 8 threads, with and without clusters and weights.

use epistat_glm::attributable::{AfOptions, Design, attributable_logistic};
use epistat_glm::{Family, Glm, Link};
use epistat_linalg::Mat;
use epistat_testkit::{POOLS, SplitMix64, assert_same_on_pools};

#[test]
fn attributable_fraction_does_not_depend_on_the_number_of_threads() {
    let mut rng = SplitMix64::new(65);
    let n = 20_000_usize;
    let z: Vec<f64> = (0..n)
        .map(|_| {
            let (a, b) = (1.0 - rng.unit(), rng.unit());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect();
    let e: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(rng.unit() < 1.0 / (1.0 + (-0.4 * z[i]).exp()))))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let eta = -1.0 + 0.8 * e[i] + 0.5 * z[i];
            f64::from(u8::from(rng.unit() < 1.0 / (1.0 + (-eta).exp())))
        })
        .collect();
    let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.unit()).collect();
    let cluster: Vec<u32> = (0..n).map(|i| u32::try_from(i / 4).unwrap()).collect();
    let x = Mat::from_fn(n, 3, |i, j| [1.0, e[i], z[i]][j]);
    let x0 = Mat::from_fn(n, 3, |i, j| if j == 1 { 0.0 } else { x[(i, j)] });
    assert_same_on_pools(&POOLS, || {
        let fit = Glm::new(Family::Binomial, Link::Logit)
            .fit(x.as_ref(), &y, Some(&w), None, None)
            .unwrap();
        let mut out = vec![];
        for design in [Design::CrossSectional, Design::CaseControl] {
            for cl in [None, Some(&cluster[..])] {
                let a = attributable_logistic(
                    &fit,
                    x.as_ref(),
                    x0.as_ref(),
                    &y,
                    &AfOptions {
                        design,
                        weights: Some(&w),
                        cluster: cl,
                        ..AfOptions::default()
                    },
                )
                .unwrap();
                out.extend([a.af, a.se, a.lower, a.upper].map(Option::unwrap));
            }
        }
        out
    });
}
