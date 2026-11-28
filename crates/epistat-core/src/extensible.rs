//! The hidden field that lets an options struct gain options in a minor
//! release. The crate documentation, under "Options", says how a crate uses
//! it.

/// A field that holds nothing and that a caller never names, so a struct
/// literal of options ends in `..Default::default()` or another options
/// value.
///
/// An options struct carries it as `#[doc(hidden)] pub _extensible:
/// Extensible`, and its `Default` sets it to [`Extensible::NEW`]. Because
/// the field is hidden, cargo-semver-checks treats a new option as an
/// addition, where a new field of an exhaustive struct is a breaking change.
///
/// ```
/// use epistat_core::Extensible;
///
/// #[derive(Clone, Debug, Default, PartialEq)]
/// pub struct FitOptions {
///     pub max_iterations: usize,
///     #[doc(hidden)]
///     pub _extensible: Extensible,
/// }
///
/// let options = FitOptions { max_iterations: 50, ..Default::default() };
/// assert_eq!(options.max_iterations, 50);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extensible {
    _private: (),
}

impl Extensible {
    /// The value, for a `Default` written out by hand or a `const` options
    /// value.
    pub const NEW: Self = Self { _private: () };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Options {
        level: f64,
        _extensible: Extensible,
    }

    #[test]
    fn a_literal_ends_in_the_defaults() {
        let options = Options {
            level: 0.9,
            ..Default::default()
        };
        assert!((options.level - 0.9).abs() < f64::EPSILON);
        assert_eq!(Extensible::default(), Extensible::NEW);
    }
}
