//! Every solver that takes numbers refuses an empty, short, NaN or infinite
//! argument with an `Input` error and the code of the shared check, through
//! `epistat_testkit::assert_rejects_bad_input_with_codes`. No solver here
//! needs its input in order, so `Unsorted` is not tried. The minimum cost
//! flow takes integers only, which have no NaN, and its arcs are added one at
//! a time, so it is not here.

use epistat_optim::Error;
use epistat_optim::bfgs::Bfgs;
use epistat_optim::bounded_simplex::BoundedSimplex;
use epistat_optim::branch_and_bound::{self, MixedBinaryProgram};
use epistat_optim::brent::{self, Brent};
use epistat_optim::frank_wolfe::SimplexLeastSquares;
use epistat_optim::nelder_mead::NelderMead;
use epistat_optim::root::{self, Bisection, Itp};
use epistat_optim::simplex::DenseSimplex;
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

/// The value `x` spoilt by `bad`, for the kinds that spoil a number.
fn number(bad: BadInput, x: f64) -> Option<f64> {
    matches!(bad, BadInput::Nan | BadInput::Infinite).then(|| bad.column(&[x])[0])
}

/// The code of a number outside its range, NaN and an infinity included.
fn domain(_: BadInput) -> &'static str {
    "input.domain"
}

/// The code of a bad column among several: an empty column differs in length
/// from the others, so it is a length error.
fn several(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

/// The kinds that spoil one of several columns.
fn columns(bad: BadInput) -> bool {
    bad != BadInput::Unsorted
}

#[test]
fn one_dimensional_solvers_refuse_a_bad_bracket() {
    let f = |x: f64| x - 0.25;
    for (end, name) in [(0, "lower"), (1, "upper")] {
        let bracket = |x: f64| if end == 0 { (x, 1.0) } else { (0.0, x) };
        assert_rejects_bad_input_with_codes(&format!("bisect {name}"), domain, |bad| {
            number(bad, [0.0, 1.0][end]).map(|x| {
                let (lower, upper) = bracket(x);
                root::bisect(f, lower, upper, &Bisection::default())
            })
        });
        assert_rejects_bad_input_with_codes(&format!("itp {name}"), domain, |bad| {
            number(bad, [0.0, 1.0][end]).map(|x| {
                let (lower, upper) = bracket(x);
                root::itp(f, lower, upper, &Itp::default())
            })
        });
        assert_rejects_bad_input_with_codes(&format!("brent {name}"), domain, |bad| {
            number(bad, [0.0, 1.0][end]).map(|x| {
                let (lower, upper) = bracket(x);
                brent::minimize(
                    |x| Ok::<_, Error>(f(x) * f(x)),
                    lower,
                    upper,
                    &Brent::default(),
                )
            })
        });
    }
}

#[test]
fn minimisers_refuse_a_bad_start() {
    let quadratic = |x: &[f64]| x.iter().map(|v| (v - 1.0) * (v - 1.0)).sum::<f64>();
    let start = [0.5, -0.5, 2.0];
    assert_rejects_bad_input_with_codes("bfgs x0", BadInput::code, |bad| {
        (bad != BadInput::Unsorted && bad != BadInput::Mismatched).then(|| {
            Bfgs::default().minimize(
                &bad.column(&start),
                |x| (quadratic(x), x.iter().map(|v| 2.0 * (v - 1.0)).collect()),
                None,
            )
        })
    });
    assert_rejects_bad_input_with_codes(
        "bfgs inverse_hessian",
        |_| "input.length",
        |bad| {
            (bad == BadInput::Mismatched).then(|| {
                Bfgs::default().minimize(
                    &start,
                    |x| (quadratic(x), x.iter().map(|v| 2.0 * (v - 1.0)).collect()),
                    Some(bad.column(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
                )
            })
        },
    );
    let vertex = [0.0, 1.0, 2.0];
    let simplex = |first: Vec<f64>| {
        vec![
            first,
            vec![1.0, 1.0, 2.0],
            vec![0.0, 2.0, 2.0],
            vec![0.0, 1.0, 3.0],
        ]
    };
    assert_rejects_bad_input_with_codes("nelder_mead simplex", BadInput::code, |bad| {
        (bad != BadInput::Unsorted)
            .then(|| NelderMead::default().minimize(simplex(bad.column(&vertex)), quadratic))
    });
}

#[test]
fn linear_programs_refuse_bad_columns() {
    // The columns of x0 + x1 + s = 1: one row, three columns.
    let g = [1.0];
    assert_rejects_bad_input_with_codes("dense simplex columns", several, |bad| {
        columns(bad)
            .then(|| DenseSimplex::new(&[bad.column(&[1.0]), vec![1.0], vec![1.0]], &g).map(|_| ()))
    });
    assert_rejects_bad_input_with_codes("dense simplex g", BadInput::code, |bad| {
        matches!(bad, BadInput::Empty | BadInput::Nan | BadInput::Infinite).then(|| {
            DenseSimplex::new(&[vec![1.0], vec![1.0], vec![1.0]], &bad.column(&g)).map(|_| ())
        })
    });

    // Two rows, three columns, stored by column.
    let columns_ab = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let cost = [1.0, 2.0, 0.5];
    let (lo, hi) = ([0.0; 3], [1.0; 3]);
    let (row_lo, row_hi) = ([0.0, 0.0], [1.0, 1.5]);
    assert_rejects_bad_input_with_codes("bounded simplex columns", several, |bad| {
        columns(bad).then(|| {
            BoundedSimplex::new(
                2,
                &bad.column(&columns_ab),
                &cost,
                &lo,
                &hi,
                &row_lo,
                &row_hi,
            )
            .map(|_| ())
        })
    });
    assert_rejects_bad_input_with_codes(
        "bounded simplex cost",
        |_| "input.length",
        |bad| {
            (bad == BadInput::Mismatched).then(|| {
                BoundedSimplex::new(
                    2,
                    &columns_ab,
                    &bad.column(&cost),
                    &lo,
                    &hi,
                    &row_lo,
                    &row_hi,
                )
                .map(|_| ())
            })
        },
    );
    let binary = [true, false, false];
    let program = |columns: &[f64], col_lo: &[f64]| -> Result<_, Error> {
        branch_and_bound::solve(
            &MixedBinaryProgram {
                rows: 2,
                columns,
                cost: &cost,
                col_lo,
                col_hi: &hi,
                row_lo: &row_lo,
                row_hi: &row_hi,
                binary: &binary,
                integer: &[],
            },
            &branch_and_bound::Options::default(),
        )
    };
    assert_rejects_bad_input_with_codes("branch and bound columns", several, |bad| {
        columns(bad).then(|| program(&bad.column(&columns_ab), &lo))
    });
    assert_rejects_bad_input_with_codes(
        "branch and bound col_lo",
        |_| "input.length",
        |bad| (bad == BadInput::Mismatched).then(|| program(&columns_ab, &bad.column(&lo))),
    );

    let target = [0.5, 0.5];
    assert_rejects_bad_input_with_codes("frank wolfe a", several, |bad| {
        columns(bad).then(|| {
            SimplexLeastSquares::new(&bad.column(&columns_ab), 2, 3, &target, 0.0).map(|_| ())
        })
    });
    assert_rejects_bad_input_with_codes("frank wolfe b", several, |bad| {
        columns(bad).then(|| {
            SimplexLeastSquares::new(&columns_ab, 2, 3, &bad.column(&target), 0.0).map(|_| ())
        })
    });
    let program = SimplexLeastSquares::new(&columns_ab, 2, 3, &target, 0.0).unwrap();
    assert_rejects_bad_input_with_codes("frank wolfe start", several, |bad| {
        columns(bad).then(|| program.start(bad.column(&[0.2, 0.3, 0.5])).map(|_| ()))
    });
}
