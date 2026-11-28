use approx::assert_relative_eq;
use epistat_core::CaseWeights;
use epistat_linalg::Mat;
use epistat_semicontinuous::CostData;
use epistat_semicontinuous::tweedie::{self, TweedieOptions};

#[test]
fn tweedie_input_uses_sample_weights_for_intercept_only_gamma_fit() {
    let x = Mat::from_fn(4, 1, |_row, _col| 1.0);
    let y = [1.0, 1.0, 10.0, 10.0];
    let weights = [10.0, 10.0, 1.0, 1.0];

    // Weighted mean of the gamma fit: Σ wᵢ·yᵢ / Σ wᵢ over the four rows above.
    let outcomes = [1.0, 1.0, 10.0, 10.0];
    let row_weights = [10.0, 10.0, 1.0, 1.0];
    let weighted_mean = outcomes
        .iter()
        .zip(row_weights.iter())
        .map(|(y, w)| y * w)
        .sum::<f64>()
        / row_weights.iter().sum::<f64>();
    let input = CostData::builder(x.as_ref(), &y)
        .weights(CaseWeights::new(&weights).expect("weights"))
        .build()
        .expect("data");

    let options = TweedieOptions {
        max_iterations: 100,
        tolerance: 1e-10,
        ..TweedieOptions::default()
    };
    let model = tweedie::fit(&input, 2.0, &options, None, None).expect("weighted gamma fit");

    assert_relative_eq!(model.coefficients[0].exp(), weighted_mean, epsilon = 1e-6);
}
