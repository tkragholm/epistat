//! Every entry point that takes numbers refuses an empty, short, NaN or
//! infinite argument with an `Input` error and the code of the shared check,
//! through `epistat_testkit::assert_rejects_bad_input_with_codes`. Order does
//! not matter to any of them (the replicates of an interval are sorted
//! inside), so `Unsorted` is not tried, and a kind an entry point accepts is
//! not tried either: a replicate that is infinite is a valid draw.

use epistat_core::{Bootstrap, Level, Seed};
use epistat_inference::Clusters;
use epistat_inference::bootstrap;
use epistat_inference::curve::{Band, contrast_curve};
use epistat_inference::influence::{ClusterRule, Influence, InfluenceScale, Keys};
use epistat_inference::interval::{self, Percentile};
use epistat_inference::multiplicity::{Adjustment, adjust};
use epistat_inference::pooling::{
    NegativeRiv, Transform, d1, d2, d3, rubin, rubin_transformed, rubin_vector,
};
use epistat_inference::sandwich::{meat, meat_from};
use epistat_inference::simultaneous::{multiplier_bands, sup_t_from_covariance};
use epistat_inference::wald::wald;
use epistat_linalg::Mat;
use epistat_linalg::basis::Polynomial;
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const VALUES: [f64; 6] = [0.4, -1.2, 0.3, 2.1, -0.7, 0.9];
const VARIANCES: [f64; 6] = [0.1, 0.2, 0.15, 0.3, 0.12, 0.2];

/// The code of a bad column among several: an empty column differs in length
/// from the others, so it is a length error.
fn several(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

/// The code of a bad column that stands alone.
fn single(bad: BadInput) -> &'static str {
    bad.code()
}

/// The kinds that spoil one of several columns.
fn columns(bad: BadInput) -> bool {
    !matches!(bad, BadInput::Unsorted)
}

/// The kinds that spoil the one column an entry point takes.
fn one(bad: BadInput) -> bool {
    matches!(bad, BadInput::Empty | BadInput::Nan | BadInput::Infinite)
}

/// The kinds that spoil a sample of draws, in which an infinity is valid.
fn draws(bad: BadInput) -> bool {
    matches!(bad, BadInput::Empty | BadInput::Nan)
}

fn alpha() -> Level {
    Level::alpha(0.05).unwrap()
}

#[test]
fn intervals_refuse_bad_replicates() {
    assert_rejects_bad_input_with_codes("percentile", single, |bad| {
        draws(bad)
            .then(|| interval::percentile(&mut bad.column(&VALUES), alpha(), Percentile::Type7))
    });
    assert_rejects_bad_input_with_codes("trimmed_percentile", single, |bad| {
        draws(bad).then(|| {
            interval::trimmed_percentile(&mut bad.column(&VALUES), alpha(), 0.05, Percentile::Type7)
        })
    });
    assert_rejects_bad_input_with_codes("bias_corrected", single, |bad| {
        draws(bad).then(|| interval::bias_corrected(&mut bad.column(&VALUES), 0.2, alpha()))
    });
}

#[test]
fn influence_and_bootstraps_refuse_bad_columns() {
    let units: Vec<u32> = (0..6).collect();
    assert_rejects_bad_input_with_codes("Influence::new", several, |bad| {
        columns(bad).then(|| {
            Influence::new(
                bad.column(&VALUES),
                units.clone(),
                1,
                ClusterRule::SmallSample,
            )
        })
    });
    assert_rejects_bad_input_with_codes("Influence::from_values", several, |bad| {
        columns(bad).then(|| {
            Influence::from_values(
                bad.column(&VALUES),
                Keys::Rows(units.clone()),
                1,
                InfluenceScale::Dfbeta,
                ClusterRule::NoSmallSample,
            )
        })
    });
    assert_rejects_bad_input_with_codes("bootstrap::multiplier", single, |bad| {
        one(bad)
            .then(|| bootstrap::multiplier(0.0, &bad.column(&VALUES), Bootstrap::new(5, Seed(1))))
    });
    // The per-cluster columns are resampled as they are, NaN included: the
    // function is Study III's and keeps its 0.3 behaviour.
    assert_rejects_bad_input_with_codes("bootstrap::cluster_totals", several, |bad| {
        matches!(bad, BadInput::Empty | BadInput::Mismatched).then(|| {
            bootstrap::cluster_totals(&[&VALUES, &bad.column(&VALUES)], Bootstrap::new(5, Seed(1)))
        })
    });
    assert_rejects_bad_input_with_codes("multiplier_bands", single, |bad| {
        one(bad).then(|| {
            let column = bad.column(&VALUES);
            let influence = Mat::from_fn(column.len(), 1, |i, _| column[i]);
            multiplier_bands(
                influence.as_ref(),
                None,
                alpha(),
                Bootstrap::new(5, Seed(1)),
            )
        })
    });
}

#[test]
fn pooling_and_tests_refuse_bad_columns() {
    assert_rejects_bad_input_with_codes("rubin", single, |bad| {
        columns(bad).then(|| rubin(&bad.column(&VALUES), &VARIANCES, None))
    });
    assert_rejects_bad_input_with_codes("rubin variances", several, |bad| {
        columns(bad).then(|| rubin(&VALUES, &bad.column(&VARIANCES), None))
    });
    assert_rejects_bad_input_with_codes("d2", single, |bad| {
        one(bad).then(|| d2(&bad.column(&VARIANCES), 2))
    });
    // A p-value outside [0, 1], NaN and an infinity included, is a domain
    // error. No p-values adjust to none, as R's `p.adjust` gives.
    assert_rejects_bad_input_with_codes(
        "adjust",
        |_| "input.domain",
        |bad| {
            matches!(bad, BadInput::Nan | BadInput::Infinite)
                .then(|| adjust(&bad.column(&[0.01, 0.2, 0.04]), Adjustment::Holm, None))
        },
    );
    assert_rejects_bad_input_with_codes("wald", single, |bad| {
        (bad == BadInput::Empty).then(|| wald(&[], Mat::<f64>::zeros(0, 0).as_ref()))
    });
    let x = Mat::from_fn(6, 2, |i, j| if j == 0 { 1.0 } else { VALUES[i] });
    assert_rejects_bad_input_with_codes(
        "meat_from residuals",
        |_| "input.length",
        |bad| {
            matches!(bad, BadInput::Empty | BadInput::Mismatched)
                .then(|| meat_from(x.as_ref(), Some(&bad.column(&VARIANCES)), None))
        },
    );
}

#[test]
fn vector_pooling_and_scales_refuse_bad_columns() {
    let estimates = |column: &[f64]| Mat::from_fn(column.len(), 1, |i, _| column[i]);
    let one_by_one = Mat::from_fn(1, 1, |_, _| 0.2);
    let covariances = vec![one_by_one.as_ref(); VALUES.len()];
    assert_rejects_bad_input_with_codes("rubin_vector", single, |bad| {
        columns(bad).then(|| rubin_vector(estimates(&bad.column(&VALUES)).as_ref(), &covariances))
    });
    assert_rejects_bad_input_with_codes("d1", single, |bad| {
        columns(bad).then(|| {
            d1(
                estimates(&bad.column(&VALUES)).as_ref(),
                &covariances,
                None,
                None,
            )
        })
    });
    assert_rejects_bad_input_with_codes("d3", single, |bad| {
        columns(bad).then(|| d3(&bad.column(&VARIANCES), &VARIANCES, 1, NegativeRiv::Keep))
    });
    assert_rejects_bad_input_with_codes("rubin_transformed", several, |bad| {
        columns(bad)
            .then(|| rubin_transformed(&VARIANCES, &bad.column(&VARIANCES), Transform::Log, None))
    });
}

#[test]
fn curves_bands_and_clusters_refuse_bad_input() {
    let basis = Polynomial { degree: 1 };
    let covariance = Mat::from_fn(2, 2, |i, j| if i == j { 0.1 } else { 0.02 });
    assert_rejects_bad_input_with_codes("contrast_curve coefficients", several, |bad| {
        one(bad).then(|| {
            contrast_curve(
                &basis,
                &[0.0, 0.5, 1.0],
                0.0,
                &bad.column(&[0.3, -0.1]),
                covariance.as_ref(),
                alpha(),
                Band::Pointwise,
            )
        })
    });
    let mut sorted = VALUES;
    sorted.sort_by(f64::total_cmp);
    assert_rejects_bad_input_with_codes("percentile_sorted", single, |bad| {
        matches!(bad, BadInput::Empty | BadInput::Nan | BadInput::Unsorted)
            .then(|| interval::percentile_sorted(&bad.column(&sorted), alpha(), Percentile::Type7))
    });
    // A variance that is not positive and finite is a numerical error, so only
    // an empty covariance is tried.
    assert_rejects_bad_input_with_codes("sup_t_from_covariance", single, |bad| {
        (bad == BadInput::Empty).then(|| {
            let column = bad.column(&VARIANCES);
            let covariance = Mat::from_fn(column.len(), column.len(), |i, j| {
                if i == j { column[i] } else { 0.0 }
            });
            sup_t_from_covariance(covariance.as_ref(), alpha(), Bootstrap::new(5, Seed(1)))
        })
    });
    let unit = [0_u32, 0, 1, 1, 2, 2];
    let cluster = [0_u32, 0, 0, 0, 1, 1];
    assert_rejects_bad_input_with_codes("Clusters::per_unit", several, |bad| {
        matches!(bad, BadInput::Mismatched).then(|| Clusters::per_unit(&unit, &cluster[..5]))
    });
    assert_rejects_bad_input_with_codes("Clusters::per_row", single, |bad| {
        (bad == BadInput::Empty).then(|| Clusters::per_row(&[]))
    });
    let x = Mat::from_fn(6, 2, |i, j| if j == 0 { 1.0 } else { VALUES[i] });
    let short = Clusters::per_row(&cluster[..5]).unwrap();
    assert_rejects_bad_input_with_codes("meat clusters", several, |bad| {
        matches!(bad, BadInput::Mismatched)
            .then(|| meat(x.as_ref(), |i| VARIANCES[i], Some(&short)))
    });
}
