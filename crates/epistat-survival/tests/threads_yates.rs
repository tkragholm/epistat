//! The population means, their covariance and the tests give identical bits in
//! rayon pools of 1, 2, 3, 4 and 8 threads, on populations of more rows than
//! one block of the sums holds.

#![expect(
    clippy::cast_precision_loss,
    reason = "the inputs are a few thousand rows"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::standardise::Counterfactual;
use epistat_survival::standardise::yates::{
    InverseLink, YatesLevel, YatesOptions, YatesTest, yates,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn population_means_are_the_same_on_every_pool() {
    let n = 5003;
    let mut rng = Stream::seeded(11);
    let base = Mat::from_fn(n, 4, |_, j| if j == 0 { 1.0 } else { rng.normal() });
    let weights: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
    let beta = [0.2, -0.4, 0.3, 0.1];
    let var: Vec<f64> = (0..16)
        .map(|i| if i / 4 == i % 4 { 0.01 } else { 0.001 })
        .collect();
    let set_a = [(1, 0.0), (2, 0.0)];
    let set_b = [(1, 1.0), (2, 0.0)];
    let set_c = [(1, 0.0), (2, 1.0)];
    for link in [
        InverseLink::Identity,
        InverseLink::Exp,
        InverseLink::Logistic,
    ] {
        for test in [
            YatesTest::Global,
            YatesTest::Pairwise,
            YatesTest::VersusMean,
        ] {
            assert_same_on_pools(&POOLS, || {
                let levels: Vec<YatesLevel<'_>> = [&set_a[..], &set_b[..], &set_c[..]]
                    .into_iter()
                    .map(|set| YatesLevel {
                        population: Counterfactual::Columns(set),
                        base: Some(base.as_ref()),
                        weights: Some(&weights),
                    })
                    .collect();
                let out = yates(
                    &beta,
                    &var,
                    &levels,
                    &YatesOptions {
                        link,
                        centre: None,
                        test,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut all = out.estimate.clone();
                all.extend(&out.se);
                all.extend(out.covariance.as_slice().iter().map(|v| v.unwrap()));
                for t in &out.tests {
                    all.push(t.chi_square);
                    all.push(t.df as f64);
                    all.push(t.p_value.unwrap_or(-1.0));
                }
                all
            });
        }
    }
}
