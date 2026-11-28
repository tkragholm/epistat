//! The functions of the crate that can refuse their input refuse an empty,
//! short, NaN or infinite column with an `Input` error and the code of the
//! shared check, through `epistat_testkit::assert_rejects_bad_input_with_codes`.
//! Order does not matter to any of them, so `Unsorted` is not tried. The
//! other functions return `None` where their value is undefined, NaN input
//! included, and the distribution functions of `dist` return NaN for a
//! parameter outside the domain, as R's do; neither is checked here. The
//! distribution objects of `dist` refuse a parameter that is NaN or infinite
//! with a domain error.

use epistat_stats::cluster::{KMeansAlgorithm, kmeans};
use epistat_stats::dist::{Beta, ChiSquared, FisherF, Gamma, Normal, Poisson, StudentT};
use epistat_stats::weighted::{ProbabilityBounds, WeightLimits, effective_sample_size, try_mean};
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const WEIGHTS: [f64; 6] = [1.0, 0.5, 2.0, 1.5, 0.25, 1.0];

/// The code of a bad column that stands alone.
fn single(bad: BadInput) -> &'static str {
    bad.code()
}

/// The kinds that spoil the one column an entry point takes.
fn one(bad: BadInput) -> bool {
    matches!(bad, BadInput::Empty | BadInput::Nan | BadInput::Infinite)
}

#[test]
fn weights_are_refused_when_empty_or_not_finite() {
    // No weights have no effective sample size, which is `None`.
    assert_rejects_bad_input_with_codes("effective_sample_size", single, |bad| {
        matches!(bad, BadInput::Nan | BadInput::Infinite)
            .then(|| effective_sample_size(&bad.column(&WEIGHTS)))
    });
    assert_rejects_bad_input_with_codes("try_mean weights", single, |bad| {
        matches!(
            bad,
            BadInput::Nan | BadInput::Infinite | BadInput::Mismatched
        )
        .then(|| try_mean(&WEIGHTS, Some(&bad.column(&WEIGHTS))))
    });
    assert_rejects_bad_input_with_codes("try_mean values", single, |bad| {
        one(bad).then(|| try_mean(&bad.column(&WEIGHTS), None))
    });
    assert_rejects_bad_input_with_codes("WeightLimits::apply", single, |bad| {
        one(bad).then(|| WeightLimits::P99.apply(&bad.column(&WEIGHTS)))
    });
    // A bound outside [0, 1], NaN and an infinity included, is a domain error.
    assert_rejects_bad_input_with_codes(
        "ProbabilityBounds::new",
        |_| "input.domain",
        |bad| {
            matches!(bad, BadInput::Nan | BadInput::Infinite)
                .then(|| ProbabilityBounds::new(bad.column(&[0.05])[0], 0.95))
        },
    );
}

#[test]
fn kmeans_refuses_bad_points_and_starts() {
    // Six points in one dimension and two starting centres.
    let points = [0.1, 0.2, 0.15, 3.0, 3.2, 2.9];
    let start = [0.0, 3.0];
    assert_rejects_bad_input_with_codes("kmeans points", single, |bad| {
        one(bad).then(|| kmeans(&bad.column(&points), 1, &start, KMeansAlgorithm::Lloyd, 10))
    });
    assert_rejects_bad_input_with_codes(
        "kmeans start",
        |bad| match bad {
            BadInput::Nan | BadInput::Infinite => "input.not_finite",
            // No starting centres ask for no clusters, outside 1 to the
            // number of points.
            _ => "input.domain",
        },
        |bad| one(bad).then(|| kmeans(&points, 1, &bad.column(&start), KMeansAlgorithm::Lloyd, 10)),
    );
}

#[test]
fn distribution_objects_refuse_parameters_that_are_not_finite() {
    let spoiled = |bad: BadInput| {
        matches!(bad, BadInput::Nan | BadInput::Infinite).then(|| bad.column(&[2.0])[0])
    };
    let domain = |_| "input.domain";
    assert_rejects_bad_input_with_codes("Normal::new mean", domain, |bad| {
        spoiled(bad).map(|v| Normal::new(v, 1.0))
    });
    assert_rejects_bad_input_with_codes("Normal::new sd", domain, |bad| {
        spoiled(bad).map(|v| Normal::new(0.0, v))
    });
    assert_rejects_bad_input_with_codes("StudentT::new", domain, |bad| {
        spoiled(bad).map(StudentT::new)
    });
    assert_rejects_bad_input_with_codes("ChiSquared::new", domain, |bad| {
        spoiled(bad).map(ChiSquared::new)
    });
    assert_rejects_bad_input_with_codes("FisherF::new", domain, |bad| {
        spoiled(bad).map(|v| FisherF::new(3.0, v))
    });
    assert_rejects_bad_input_with_codes("Gamma::with_rate", domain, |bad| {
        spoiled(bad).map(|v| Gamma::with_rate(v, 1.0))
    });
    assert_rejects_bad_input_with_codes("Gamma::with_scale", domain, |bad| {
        spoiled(bad).map(|v| Gamma::with_scale(1.0, v))
    });
    assert_rejects_bad_input_with_codes("Beta::new", domain, |bad| {
        spoiled(bad).map(|v| Beta::new(v, 1.0))
    });
    assert_rejects_bad_input_with_codes("Poisson::new", domain, |bad| {
        spoiled(bad).map(Poisson::new)
    });
}
