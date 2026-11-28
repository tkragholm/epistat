use epistat_linalg::Mat;
use epistat_semicontinuous::LongitudinalData;
use epistat_semicontinuous::mtp::{
    self, CounterfactualScenario, MtpOptions, counterfactual_effects_summary,
    posterior_predictive_summary,
};

fn idx_to_f64(idx: usize) -> f64 {
    f64::from(u32::try_from(idx).unwrap_or(u32::MAX))
}

const SUBJECTS: [u32; 20] = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5];
const TIME: [f64; 20] = [
    0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0,
    3.0,
];

#[test]
fn mtp_public_workflow_produces_effects_and_ppc() {
    let y: Vec<f64> = (0..20)
        .map(|row| {
            if row % 2 == 0 {
                0.0
            } else {
                1.0 + idx_to_f64(row % 5)
            }
        })
        .collect();
    let x = Mat::from_fn(20, 1, |_row, _col| 1.0);
    let input = LongitudinalData::builder(&y, x.as_ref(), x.as_ref(), &SUBJECTS, &TIME)
        .build()
        .expect("data");
    let options = MtpOptions {
        iterations: 160,
        burn_in: 40,
        chains: 1,
        keep_draws: true,
        ..MtpOptions::new(123)
    };

    let model = mtp::fit(&input, &options).expect("fit should succeed");
    let report = &model.chains[0];
    let samples = &model.draws.as_ref().expect("draws")[0];
    assert!(model.convergence.is_none());

    assert_eq!(samples.len(), report.diagnostics.retained_draws);
    assert!(report.diagnostics.acceptance_rates.is_some());
    assert!(report.posterior_summary.is_some());

    let scenario = CounterfactualScenario {
        binary_design_exposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
        mean_design_exposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
        binary_design_unexposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
        mean_design_unexposed: Mat::from_fn(4, 1, |_row, _col| 1.0),
    };

    let effects = counterfactual_effects_summary(samples, &scenario).expect("effects should run");
    assert_eq!(effects.per_period.len(), 4);
    assert!(effects.cumulative_additive_effect.mean.unwrap().is_finite());

    let ppc = posterior_predictive_summary(samples, &input, 8).expect("ppc should run");
    assert_eq!(ppc.calibration_bins.len(), 8);
    assert!(ppc.predicted_mean.mean.unwrap().is_finite());
    assert!(ppc.predicted_zero_rate.mean.unwrap() >= 0.0);
}
