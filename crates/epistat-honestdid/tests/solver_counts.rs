//! The solver counts in the summaries and the diagnostics of the optimisation
//! surface: a deterministic call reports the same counts each time.

use epistat_core::{Bootstrap, Covariance, EventStudy};
use epistat_honestdid::honest::{
    self, Calibration, Directions, HonestOptions, JointPathOptions, Restriction, Surface, Target,
};

#[test]
fn counts_are_reported_and_repeat() {
    let covariance: Vec<f64> = (0..25)
        .map(|i| if i / 5 == i % 5 { 0.04 } else { 0.005 })
        .collect();
    let study = EventStudy::new(
        vec![0.02, -0.01, 0.03, 0.5, 0.7],
        Covariance::from_values(5, &covariance).expect("covariance"),
        vec![-4, -3, -2, 0, 1],
        -1,
    )
    .expect("event study");
    let restriction = Restriction::Smoothness { m: vec![0.05] };
    let summarize = || {
        honest::summarize(
            &study,
            &Target::Weights(vec![1.0, 0.0]),
            &restriction,
            &HonestOptions::new(Bootstrap::new(1_000, 0)),
        )
        .expect("smoothness summary")
    };
    let summary = summarize();
    assert_eq!(summary.rows.len(), 1);
    assert_eq!(
        summary.solver,
        summarize().solver,
        "the counts of a deterministic call repeat"
    );

    let options = JointPathOptions {
        directions: Directions::Surface(Surface::new(250, 1729)),
        calibration: Calibration::Gaussian(Bootstrap::new(20_000, 42)),
        ..JointPathOptions::new(Bootstrap::new(1_000, 0))
    };
    let run = || {
        honest::joint_path(&study, &restriction, &options)
            .expect("surface region")
            .diagnostics
            .expect("diagnostics")
    };
    let first = run();
    assert_eq!(first, run(), "the counts of a deterministic call repeat");
    assert!(first.directions >= first.random);
}
