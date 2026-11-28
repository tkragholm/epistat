//! The entry points that take columns or a system refuse an empty, short,
//! NaN, infinite or unsorted argument with an `Input` error and the code of
//! the check that refuses it, through
//! `epistat_testkit::assert_rejects_bad_input_with_codes`. A kind an entry
//! point accepts is not tried: an empty set of internal knots is a spline
//! without internal knots, and a matrix given to a solve may hold NaN, which
//! is refused as `Singular` once no factorisation gives a finite solution.

use epistat_linalg::basis::{
    BSpline, NaturalSpline, RestrictedCubic, harrell_knots, natural_spline_knots,
};
use epistat_linalg::design::{from_columns, independent_columns};
use epistat_linalg::{
    Attempt, Mat, congruence, inverse, small, solve, weighted_gram, weighted_xtz,
};
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const X: [f64; 6] = [0.3, 1.2, -0.4, 2.2, 0.9, 1.7];
const WEIGHTS: [f64; 6] = [1.0, 0.5, 2.0, 1.0, 0.25, 1.0];

/// The code of a bad column among several: an empty column differs in length
/// from the others, so it is a length error.
fn several(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

/// The kinds that spoil one of several columns of numbers in any order.
fn columns(bad: BadInput) -> bool {
    bad != BadInput::Unsorted
}

/// A design with an intercept and the column `X`.
fn design() -> Mat<f64> {
    from_columns(&[&X], true).unwrap()
}

#[test]
fn designs_and_products_refuse_bad_columns() {
    assert_rejects_bad_input_with_codes("from_columns", BadInput::code, |bad| {
        columns(bad).then(|| from_columns(&[&bad.column(&X), &X], false))
    });
    let x = design();
    assert_rejects_bad_input_with_codes("weighted_gram", several, |bad| {
        columns(bad).then(|| weighted_gram(x.as_ref(), &bad.column(&WEIGHTS)))
    });
    assert_rejects_bad_input_with_codes("weighted_xtz weights", several, |bad| {
        columns(bad).then(|| weighted_xtz(x.as_ref(), &bad.column(&WEIGHTS), &X))
    });
    assert_rejects_bad_input_with_codes(
        "weighted_xtz z",
        |_| "input.length",
        |bad| {
            matches!(bad, BadInput::Empty | BadInput::Mismatched)
                .then(|| weighted_xtz(x.as_ref(), &WEIGHTS, &bad.column(&X)))
        },
    );
    assert_rejects_bad_input_with_codes("independent_columns", several, |bad| {
        columns(bad).then(|| independent_columns(x.as_ref(), &bad.column(&WEIGHTS), None, 1e-8))
    });
}

#[test]
fn solves_refuse_a_right_hand_side_of_another_length() {
    let a = Mat::from_fn(3, 3, |i, j| if i == j { 2.0 } else { 0.5 });
    let b = [1.0, 2.0, 3.0];
    let rhs = |bad: BadInput| {
        let column = bad.column(&b);
        Mat::from_fn(column.len(), 1, |i, _| column[i])
    };
    let shape = |bad: BadInput| matches!(bad, BadInput::Empty | BadInput::Mismatched);
    let length = |_: BadInput| "input.length";
    assert_rejects_bad_input_with_codes("solve", length, |bad| {
        shape(bad).then(|| solve(a.as_ref(), rhs(bad).as_ref(), &Attempt::SYMMETRIC))
    });
    assert_rejects_bad_input_with_codes("lu_partial", length, |bad| {
        shape(bad).then(|| epistat_linalg::solve::lu_partial(a.as_ref(), rhs(bad).as_ref()))
    });
    assert_rejects_bad_input_with_codes("llt", length, |bad| {
        shape(bad).then(|| epistat_linalg::solve::llt(a.as_ref(), rhs(bad).as_ref()))
    });
    assert_rejects_bad_input_with_codes("least_squares", length, |bad| {
        shape(bad).then(|| epistat_linalg::solve::least_squares(a.as_ref(), rhs(bad).as_ref()))
    });
    assert_rejects_bad_input_with_codes("congruence", length, |bad| {
        shape(bad).then(|| {
            let side = bad.column(&b).len();
            congruence(a.as_ref(), Mat::<f64>::identity(side, side).as_ref())
        })
    });
    let not_square = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
    assert_rejects_bad_input_with_codes("inverse", length, |bad| {
        (bad == BadInput::Mismatched).then(|| inverse(not_square.as_ref(), &Attempt::SYMMETRIC))
    });
    let lower = [2.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.2, 0.3, 1.0];
    assert_rejects_bad_input_with_codes("small::cholesky_solve", length, |bad| {
        shape(bad).then(|| small::cholesky_solve(&lower, 3, &mut bad.column(&b)))
    });
    assert_rejects_bad_input_with_codes("small::cholesky_inverse", length, |bad| {
        shape(bad).then(|| small::cholesky_inverse(&bad.column(&lower), 3))
    });
}

#[test]
fn bases_refuse_bad_knots() {
    let internal = [0.3, 0.5, 0.7];
    // A knot that is NaN or infinite is not strictly inside the boundary.
    let inside = |bad: BadInput| match bad {
        BadInput::Nan | BadInput::Infinite => "linalg.outside_boundary",
        other => other.code(),
    };
    let spoils =
        |bad: BadInput| matches!(bad, BadInput::Nan | BadInput::Infinite | BadInput::Unsorted);
    assert_rejects_bad_input_with_codes("BSpline::new", inside, |bad| {
        spoils(bad).then(|| BSpline::new(3, &bad.column(&internal), 0.0, 1.0))
    });
    assert_rejects_bad_input_with_codes("NaturalSpline::new", inside, |bad| {
        spoils(bad).then(|| NaturalSpline::new(&bad.column(&internal), (0.0, 1.0), false))
    });
    let knots = [0.0, 1.0, 2.0, 3.0, 4.0];
    // Fewer than three knots, none included, is the crate's own refusal.
    let rcs = |bad: BadInput| match bad {
        BadInput::Empty => "linalg.too_few_knots",
        other => other.code(),
    };
    assert_rejects_bad_input_with_codes("RestrictedCubic::new", rcs, |bad| {
        (bad != BadInput::Mismatched).then(|| RestrictedCubic::new(&bad.column(&knots), false))
    });
    let sample: Vec<f64> = (0..40).map(|i| f64::from(i) / 4.0).collect();
    // No observations are too few to place knots on.
    let observations = |bad: BadInput| match bad {
        BadInput::Empty => "linalg.too_few_observations",
        other => other.code(),
    };
    let values =
        |bad: BadInput| matches!(bad, BadInput::Empty | BadInput::Nan | BadInput::Infinite);
    assert_rejects_bad_input_with_codes("harrell_knots", observations, |bad| {
        values(bad).then(|| harrell_knots(&bad.column(&sample), 4, 0.05))
    });
    // No observations have no range to place the boundary knots at.
    let boundary = |bad: BadInput| match bad {
        BadInput::Empty => "linalg.boundary",
        other => other.code(),
    };
    assert_rejects_bad_input_with_codes("natural_spline_knots", boundary, |bad| {
        values(bad).then(|| natural_spline_knots(&bad.column(&sample), 4, false, None))
    });
}
