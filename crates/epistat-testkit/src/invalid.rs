//! Checks that an entry point meets bad input with an error, not a panic, and
//! that the error codes a crate declares are unique.
//!
//! These helpers are for tests. They have no stability promise: names,
//! signatures and the set of bad inputs may change in any release.
//!
//! # Bad input
//!
//! [`BadInput`] lists the kinds of input every entry point must refuse: an
//! empty column, a column with NaN, a column with an infinity, columns of
//! different lengths and an unsorted column. [`assert_rejects_bad_input`] calls
//! a closure once per kind. The closure builds the call for that kind, usually
//! by passing [`BadInput::column`] of a valid column where the entry point
//! takes one, and returns `None` for a kind the entry point has no argument
//! for. Each call must return an error whose [`ErrorKind`] is
//! [`Input`](ErrorKind::Input). A panic, an `Ok` or an error of another kind
//! fails the test with the entry point's label and the kind of input.
//!
//! ```
//! use epistat_core::{InputError, check};
//! use epistat_testkit::{BadInput, assert_rejects_bad_input};
//!
//! // An entry point taking a covariate column and an outcome column.
//! fn entry(x: &[f64], y: &[f64]) -> Result<f64, InputError> {
//!     check::same_length("y", x.len(), y.len())?;
//!     check::finite("x", x)?;
//!     check::sorted_by("x", x, f64::total_cmp)?;
//!     Ok(x.iter().sum::<f64>() + y.iter().sum::<f64>())
//! }
//!
//! let x = [1.0, 2.0, 3.0, 4.0];
//! let y = [0.5, 0.1, 0.9, 0.3];
//! assert_rejects_bad_input("entry", |bad| {
//!     Some(match bad {
//!         BadInput::Mismatched => entry(&x, &bad.column(&y)),
//!         _ => entry(&bad.column(&x), &y),
//!     })
//! });
//! ```
//!
//! A single call is checked with [`assert_input_error`], which also catches a
//! panic. The kinds of input a particular entry point cannot see, such as an
//! unsorted column where order does not matter, are answered with `None`.
//!
//! [`assert_rejects_bad_input_with_codes`] checks the code as well as the
//! kind, so that a crate reporting a NaN under its own code instead of
//! `input.not_finite` fails. It takes the expected code of each kind, usually
//! [`BadInput::code`], the code of the shared check that refuses it. A crate
//! opts in per entry point once its errors use the shared codes; an entry
//! point that meets an empty column through a length check passes its own
//! mapping.
//!
//! ```
//! # use epistat_core::{InputError, check};
//! # use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};
//! # fn entry(x: &[f64], y: &[f64]) -> Result<f64, InputError> {
//! #     check::same_length("y", x.len(), y.len())?;
//! #     check::finite("x", x)?;
//! #     check::sorted_by("x", x, f64::total_cmp)?;
//! #     Ok(x.iter().sum::<f64>() + y.iter().sum::<f64>())
//! # }
//! # let x = [1.0, 2.0, 3.0, 4.0];
//! # let y = [0.5, 0.1, 0.9, 0.3];
//! // An empty x differs in length from y, so it is a length error here.
//! let code = |bad| match bad {
//!     BadInput::Empty => "input.length",
//!     other => other.code(),
//! };
//! assert_rejects_bad_input_with_codes("entry", code, |bad| {
//!     Some(match bad {
//!         BadInput::Mismatched => entry(&x, &bad.column(&y)),
//!         _ => entry(&bad.column(&x), &y),
//!     })
//! });
//! ```
//!
//! # Error codes
//!
//! [`assert_codes_unique`] takes one value of every variant of a crate's error
//! type and checks that no two share a code, that each code is lower case with
//! a prefix and a name joined by a dot, and that all start with the crate's
//! prefix.
//!
//! ```
//! use epistat_core::{Classify, ErrorKind};
//! use epistat_testkit::assert_codes_unique;
//!
//! enum Error { Singular, NotConverged }
//!
//! impl Classify for Error {
//!     fn kind(&self) -> ErrorKind { ErrorKind::Numerical }
//!     fn code(&self) -> &'static str {
//!         match self {
//!             Self::Singular => "demo.singular",
//!             Self::NotConverged => "demo.not_converged",
//!         }
//!     }
//! }
//!
//! assert_codes_unique("demo", &[Error::Singular, Error::NotConverged]);
//! ```

use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};

use epistat_core::{Classify, ErrorKind};

/// A kind of input every entry point must refuse with an `Input` error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BadInput {
    /// A column with no rows.
    Empty,
    /// A column with a NaN in the middle.
    Nan,
    /// A column with an infinity in the middle.
    Infinite,
    /// A column one row shorter than the columns it goes with. The entry
    /// point is called with this column in place of one valid column, the rest
    /// valid.
    Mismatched,
    /// A column in descending order, from a valid ascending one.
    Unsorted,
}

impl BadInput {
    /// Every kind, in the order the checks run.
    pub const ALL: [Self; 5] = [
        Self::Empty,
        Self::Nan,
        Self::Infinite,
        Self::Mismatched,
        Self::Unsorted,
    ];

    /// `good` spoiled in this way.
    ///
    /// `Empty` returns no rows. `Nan` and `Infinite` replace the middle row.
    /// `Mismatched` drops the last row. `Unsorted` reverses the rows, so a
    /// column with at least two distinct values ends up out of order.
    #[must_use]
    pub fn column(self, good: &[f64]) -> Vec<f64> {
        let mut column = good.to_vec();
        match self {
            Self::Empty => column.clear(),
            Self::Nan => {
                if let Some(value) = column.get_mut(good.len() / 2) {
                    *value = f64::NAN;
                }
            }
            Self::Infinite => {
                if let Some(value) = column.get_mut(good.len() / 2) {
                    *value = f64::INFINITY;
                }
            }
            Self::Mismatched => {
                column.pop();
            }
            Self::Unsorted => column.reverse(),
        }
        column
    }

    /// The code of the shared check that refuses this kind:
    /// `input.empty`, `input.not_finite` for NaN and infinities,
    /// `input.length` and `input.unsorted`.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Empty => "input.empty",
            Self::Nan | Self::Infinite => "input.not_finite",
            Self::Mismatched => "input.length",
            Self::Unsorted => "input.unsorted",
        }
    }
}

/// Calls `call` and checks that it returns an `Input` error.
///
/// # Panics
///
/// If `call` panics, returns `Ok`, or returns an error whose kind is not
/// [`ErrorKind::Input`]. The message starts with `label`.
pub fn assert_input_error<T, E>(label: &str, call: impl FnOnce() -> Result<T, E>)
where
    T: Debug,
    E: Classify + Debug,
{
    match catch_unwind(AssertUnwindSafe(call)) {
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            panic!("{label}: panicked instead of returning an Input error: {message}");
        }
        Ok(Ok(value)) => panic!("{label}: returned Ok({value:?}), expected an Input error"),
        Ok(Err(error)) => assert_eq!(
            error.kind(),
            ErrorKind::Input,
            "{label}: error {error:?} (code {}) is not of kind Input",
            error.code()
        ),
    }
}

/// Calls `call` and checks that it returns an `Input` error with the code
/// `code`.
///
/// # Panics
///
/// As [`assert_input_error`], and if the code is not `code`.
pub fn assert_input_error_code<T, E>(label: &str, code: &str, call: impl FnOnce() -> Result<T, E>)
where
    T: Debug,
    E: Classify + Debug,
{
    assert_input_error(label, || {
        call().map_err(|error| {
            assert_eq!(
                error.code(),
                code,
                "{label}: error {error:?} has code {}, expected {code}",
                error.code()
            );
            error
        })
    });
}

/// Calls `build` once for each [`BadInput`] and checks that each call returns
/// an `Input` error.
///
/// `build` returns `None` for a kind the entry point has no argument for.
///
/// # Panics
///
/// As [`assert_input_error`], with the kind of input in the message, and if
/// `build` returns `None` for every kind.
pub fn assert_rejects_bad_input<T, E>(label: &str, build: impl Fn(BadInput) -> Option<Result<T, E>>)
where
    T: Debug,
    E: Classify + Debug,
{
    let mut checked = 0;
    for bad in BadInput::ALL {
        let outcome = catch_unwind(AssertUnwindSafe(|| build(bad)));
        match outcome {
            Err(_) => assert_input_error(&format!("{label} on {bad:?}"), || {
                build(bad).unwrap_or_else(|| unreachable!("the first call returned Some"))
            }),
            Ok(None) => {}
            Ok(Some(result)) => {
                checked += 1;
                assert_input_error(&format!("{label} on {bad:?}"), || result);
            }
        }
    }
    assert!(checked > 0, "{label}: no kind of bad input was tried");
}

/// [`assert_rejects_bad_input`] that also checks the code of each error
/// against `code` of its kind, usually [`BadInput::code`].
///
/// # Panics
///
/// As [`assert_rejects_bad_input`], and if an error's code is not the one
/// `code` gives for its kind.
pub fn assert_rejects_bad_input_with_codes<T, E>(
    label: &str,
    code: impl Fn(BadInput) -> &'static str,
    build: impl Fn(BadInput) -> Option<Result<T, E>>,
) where
    T: Debug,
    E: Classify + Debug,
{
    let mut checked = 0;
    for bad in BadInput::ALL {
        let label = format!("{label} on {bad:?}");
        let outcome = catch_unwind(AssertUnwindSafe(|| build(bad)));
        match outcome {
            Err(_) => assert_input_error_code(&label, code(bad), || {
                build(bad).unwrap_or_else(|| unreachable!("the first call returned Some"))
            }),
            Ok(None) => {}
            Ok(Some(result)) => {
                checked += 1;
                assert_input_error_code(&label, code(bad), || result);
            }
        }
    }
    assert!(checked > 0, "{label}: no kind of bad input was tried");
}

/// Checks the codes of `errors`, one value of every variant of a crate's
/// error type.
///
/// # Panics
///
/// If two values share a code, if a code is not lower case ASCII letters,
/// digits and underscores in two parts joined by one dot, or if a code does
/// not start with `prefix` and a dot.
pub fn assert_codes_unique<E: Classify>(prefix: &str, errors: &[E]) {
    let mut seen: Vec<&'static str> = Vec::with_capacity(errors.len());
    for error in errors {
        let code = error.code();
        let part = |s: &str| {
            !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        };
        let well_formed = code
            .split_once('.')
            .is_some_and(|(a, b)| part(a) && part(b));
        assert!(
            well_formed,
            "code {code:?} is not of the form prefix.name in lower case"
        );
        assert!(
            code.strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with('.')),
            "code {code:?} does not start with {prefix:?}."
        );
        assert!(!seen.contains(&code), "code {code:?} is declared twice");
        seen.push(code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Fail(ErrorKind, &'static str);

    impl Classify for Fail {
        fn kind(&self) -> ErrorKind {
            self.0
        }
        fn code(&self) -> &'static str {
            self.1
        }
    }

    #[test]
    fn columns_are_spoiled_as_documented() {
        let good = [1.0, 2.0, 3.0];
        assert_eq!(BadInput::Empty.column(&good).len(), 0);
        assert!(BadInput::Nan.column(&good)[1].is_nan());
        assert!(BadInput::Infinite.column(&good)[1].is_infinite());
        assert_eq!(BadInput::Mismatched.column(&good), [1.0, 2.0]);
        assert_eq!(BadInput::Unsorted.column(&good), [3.0, 2.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "panicked instead")]
    fn a_panic_is_reported() {
        assert_input_error::<(), Fail>("f", || panic!("index out of bounds"));
    }

    #[test]
    #[should_panic(expected = "expected an Input error")]
    fn an_ok_is_reported() {
        assert_input_error::<_, Fail>("f", || Ok(1));
    }

    #[test]
    #[should_panic(expected = "not of kind Input")]
    fn another_kind_is_reported() {
        assert_input_error::<(), _>("f", || Err(Fail(ErrorKind::Numerical, "x.y")));
    }

    #[test]
    #[should_panic(expected = "on Nan")]
    fn the_kind_of_input_is_named() {
        assert_rejects_bad_input("f", |bad| {
            Some(if bad == BadInput::Nan {
                Ok(0)
            } else {
                Err(Fail(ErrorKind::Input, "x.y"))
            })
        });
    }

    #[test]
    fn the_shared_checks_give_the_codes_of_the_kinds() {
        use epistat_core::check;
        let good = [1.0, 2.0, 3.0];
        let entry = |x: &[f64]| -> Result<(), epistat_core::InputError> {
            check::non_empty("x", x.len())?;
            check::same_length("x", 3, x.len())?;
            check::finite("x", x)?;
            check::sorted_by("x", x, f64::total_cmp)
        };
        assert_rejects_bad_input_with_codes("entry", BadInput::code, |bad| {
            Some(entry(&bad.column(&good)))
        });
    }

    #[test]
    #[should_panic(expected = "has code x.y, expected input.not_finite")]
    fn another_code_is_reported() {
        assert_rejects_bad_input_with_codes("f", BadInput::code, |bad| {
            Some(if bad == BadInput::Nan {
                Err::<(), _>(Fail(ErrorKind::Input, "x.y"))
            } else {
                Err(Fail(ErrorKind::Input, bad.code()))
            })
        });
    }

    #[test]
    #[should_panic(expected = "no kind of bad input")]
    fn trying_nothing_fails() {
        assert_rejects_bad_input::<(), Fail>("f", |_| None);
    }

    #[test]
    #[should_panic(expected = "declared twice")]
    fn duplicate_codes_are_reported() {
        let a = Fail(ErrorKind::Input, "x.a");
        let b = Fail(ErrorKind::Input, "x.a");
        assert_codes_unique("x", &[a, b]);
    }

    #[test]
    #[should_panic(expected = "prefix.name")]
    fn malformed_codes_are_reported() {
        assert_codes_unique("x", &[Fail(ErrorKind::Input, "x.Bad")]);
    }

    #[test]
    #[should_panic(expected = "does not start with")]
    fn a_foreign_prefix_is_reported() {
        assert_codes_unique("x", &[Fail(ErrorKind::Input, "y.a")]);
    }
}
