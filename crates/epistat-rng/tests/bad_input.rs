//! Every draw that takes parameters refuses NaN and an infinity with an
//! `Input` error and the code of the shared check, and the Wishart draws
//! refuse an empty or short scale factor, through
//! `epistat_testkit::assert_rejects_bad_input_with_codes`. A parameter that
//! is a number rather than a column has no length and no order.

use epistat_rng::Stream;
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

/// The value `x` spoilt by `bad`, for the kinds that spoil a number.
fn number(bad: BadInput, x: f64) -> Option<f64> {
    matches!(bad, BadInput::Nan | BadInput::Infinite).then(|| bad.column(&[x])[0])
}

/// A number outside the range of its parameter is a domain error, NaN and an
/// infinity included.
fn domain(_: BadInput) -> &'static str {
    "input.domain"
}

/// A lower triangular 2 x 2 factor, row-major, whose middle entry is the one
/// below the diagonal.
const SCALE: [f64; 4] = [1.0, 0.0, 0.5, 1.0];

/// The code of a spoilt scale factor: an empty or short one does not hold
/// `dim` squared values.
fn scale_code(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty | BadInput::Mismatched => "input.length",
        other => other.code(),
    }
}

#[test]
fn draws_refuse_bad_parameters() {
    assert_rejects_bad_input_with_codes("gamma shape", domain, |bad| {
        number(bad, 2.0).map(|x| Stream::seeded(1).gamma(x, 1.0))
    });
    assert_rejects_bad_input_with_codes("gamma scale", domain, |bad| {
        number(bad, 2.0).map(|x| Stream::seeded(1).gamma(2.0, x))
    });
    assert_rejects_bad_input_with_codes("chi_square", domain, |bad| {
        number(bad, 3.0).map(|x| Stream::seeded(1).chi_square(x))
    });
    assert_rejects_bad_input_with_codes("beta alpha", domain, |bad| {
        number(bad, 2.0).map(|x| Stream::seeded(1).beta(x, 2.0))
    });
    assert_rejects_bad_input_with_codes("beta beta", domain, |bad| {
        number(bad, 2.0).map(|x| Stream::seeded(1).beta(2.0, x))
    });
    assert_rejects_bad_input_with_codes("binomial", domain, |bad| {
        number(bad, 0.3).map(|x| Stream::seeded(1).binomial(20, x))
    });
    assert_rejects_bad_input_with_codes("wishart df", domain, |bad| {
        number(bad, 5.0).map(|x| Stream::seeded(1).wishart(x, &SCALE, 2))
    });
    assert_rejects_bad_input_with_codes("inverse_wishart df", domain, |bad| {
        number(bad, 5.0).map(|x| Stream::seeded(1).inverse_wishart(x, &SCALE, 2))
    });
}

#[test]
fn wishart_refuses_a_bad_scale_factor() {
    let spoil = |bad: BadInput| (bad != BadInput::Unsorted).then(|| bad.column(&SCALE));
    assert_rejects_bad_input_with_codes("wishart scale", scale_code, |bad| {
        spoil(bad).map(|scale| Stream::seeded(1).wishart(3.0, &scale, 2))
    });
    assert_rejects_bad_input_with_codes("inverse_wishart scale", scale_code, |bad| {
        spoil(bad).map(|scale| Stream::seeded(1).inverse_wishart(3.0, &scale, 2))
    });
}
