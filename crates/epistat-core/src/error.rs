//! What kind of failure an error is, and a stable code for it.
//!
//! Every epistat crate has one error enum, and every one of them implements
//! [`Classify`]. A Rust caller matches on the typed variants. A binding needs
//! only two things: the [`ErrorKind`], which decides the host language's
//! exception or condition class, and the [`code`](Classify::code), a stable
//! string such as `"input.length"` or `"glm.not_converged"` that a test or a
//! user can match on without parsing a message.
//!
//! Codes are part of the API. They are lower case, a crate or area prefix and
//! a name joined by a dot, and a code once released keeps its meaning. A
//! variant that wraps a lower crate's error reports that error's kind and code.

/// What kind of failure an error is.
///
/// A binding maps the kinds to its own classes: Python's `ValueError` for
/// [`Input`](Self::Input) and a `RuntimeError` subclass for the other two, R's
/// condition classes in the same way.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ErrorKind {
    /// The caller's input cannot be used: a wrong length, a value outside its
    /// domain, unsorted or repeated keys. The same call with other input can
    /// succeed.
    Input,
    /// The input was accepted, and the computation failed on it: no
    /// convergence, a singular or aliased design, an infeasible program.
    Numerical,
    /// The combination of options asks for something the method does not do.
    Unsupported,
}

impl ErrorKind {
    /// The kind as a lower-case word: `"input"`, `"numerical"` or
    /// `"unsupported"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Numerical => "numerical",
            Self::Unsupported => "unsupported",
        }
    }
}

impl core::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The kind and the stable code of an error, what a binding reads.
pub trait Classify {
    /// What kind of failure this is.
    fn kind(&self) -> ErrorKind;

    /// A stable code naming the failure, such as `"input.length"`.
    fn code(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_have_lower_case_names() {
        assert_eq!(ErrorKind::Input.to_string(), "input");
        assert_eq!(ErrorKind::Numerical.as_str(), "numerical");
        assert_eq!(ErrorKind::Unsupported.as_str(), "unsupported");
    }
}
