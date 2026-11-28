//! Every entry point that takes numbers refuses an empty, short, NaN or
//! infinite argument with an `Input` error and the code of the shared check,
//! through `epistat_testkit::assert_rejects_bad_input_with_codes`. Order does
//! not matter to any of them (the replicates of an interval are sorted
//! inside), so `Unsorted` is not tried, and a kind an entry point accepts is
//! not tried either: a replicate that is infinite is a valid draw.

use epistat_core::{Bootstrap, Level};
use epistat_inference::bootstrap;
use epistat_inference::influence::{ClusterRule, Influence};
use epistat_inference::interval::{self, Percentile};
use epistat_inference::multiplicity::{Adjustment, adjust};
use epistat_inference::pooling::{d2, rubin};
use epistat_inference::sandwich::meat_from;
use epistat_inference::simultaneous::multiplier_bands;
use epistat_inference::wald::wald;
use epistat_linalg::Mat;
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
    assert_rejects_bad_input_with_codes("bootstrap::multiplier", single, |bad| {
        one(bad).then(|| bootstrap::multiplier(0.0, &bad.column(&VALUES), Bootstrap::new(5, 1)))
    });
    // The per-cluster columns are resampled as they are, NaN included: the
    // function is Study III's and keeps its 0.3 behaviour.
    assert_rejects_bad_input_with_codes("bootstrap::cluster_totals", several, |bad| {
        matches!(bad, BadInput::Empty | BadInput::Mismatched).then(|| {
            bootstrap::cluster_totals(&[&VALUES, &bad.column(&VALUES)], Bootstrap::new(5, 1))
        })
    });
    assert_rejects_bad_input_with_codes("multiplier_bands", single, |bad| {
        one(bad).then(|| {
            let column = bad.column(&VALUES);
            let influence = Mat::from_fn(column.len(), 1, |i, _| column[i]);
            multiplier_bands(influence.as_ref(), None, alpha(), Bootstrap::new(5, 1))
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
