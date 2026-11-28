//! The result of a hypothesis test: a statistic, its degrees of freedom, its
//! p-value and the distribution the p-value is read from.
//!
//! A Wald test, a log-rank test, a Breslow-Day test and a pre-trend test all
//! report the same three numbers. [`Test`] holds them, each `None` where the
//! method does not define it, and [`Reference`] says which distribution the
//! statistic is referred to, so the result prints as R prints it:
//!
//! ```
//! use epistat_core::{Reference, Test};
//!
//! let wald = Test::chi_square(Some(3.21), 2.0, Some(0.2009));
//! assert_eq!(wald.to_string(), "chi-square = 3.21 on 2 df, p = 0.20");
//! let z = Test::normal(Some(-1.96), Some(0.05));
//! assert_eq!(z.distribution, Reference::Normal);
//! assert_eq!(z.to_string(), "z = -1.96, p = 0.050");
//! ```
//!
//! The type holds numbers; the crate that runs the test computes them.

use core::fmt;

use crate::format;

/// The distribution a test statistic is referred to.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Reference {
    /// The standard normal; the statistic is a z value and has no degrees of
    /// freedom.
    Normal,
    /// Student's t with [`Test::df`] degrees of freedom.
    StudentT,
    /// The chi-square with [`Test::df`] degrees of freedom.
    ChiSquare,
    /// The F distribution with [`Test::df`] numerator degrees of freedom and
    /// `df2` denominator degrees of freedom, which may be infinite.
    F {
        /// The denominator degrees of freedom.
        df2: f64,
    },
    /// The 50:50 mixture of a point mass at 0 and the chi-square with
    /// [`Test::df`] degrees of freedom, the null distribution of a likelihood
    /// ratio for a variance on the boundary of its space (Self and Liang).
    ChiBarSquare,
    /// The exact null distribution of the statistic, as for Fisher's test or
    /// a permutation test.
    Exact,
}

/// A test statistic, its degrees of freedom and its p-value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Test {
    /// The statistic, `None` where it is undefined.
    pub statistic: Option<f64>,
    /// The degrees of freedom, or the numerator degrees of freedom of an F
    /// test; `None` for a normal or exact reference.
    pub df: Option<f64>,
    /// The p-value, `None` where it is undefined.
    pub p_value: Option<f64>,
    /// The distribution the p-value is read from.
    pub distribution: Reference,
}

impl Test {
    /// A test with every part given.
    #[must_use]
    pub const fn new(
        statistic: Option<f64>,
        df: Option<f64>,
        p_value: Option<f64>,
        distribution: Reference,
    ) -> Self {
        Self {
            statistic,
            df,
            p_value,
            distribution,
        }
    }

    /// A chi-square test on `df` degrees of freedom.
    #[must_use]
    pub const fn chi_square(statistic: Option<f64>, df: f64, p_value: Option<f64>) -> Self {
        Self::new(statistic, Some(df), p_value, Reference::ChiSquare)
    }

    /// A test of a statistic referred to the standard normal.
    #[must_use]
    pub const fn normal(statistic: Option<f64>, p_value: Option<f64>) -> Self {
        Self::new(statistic, None, p_value, Reference::Normal)
    }
}

impl fmt::Display for Test {
    /// `chi-square = 3.21 on 2 df, p = 0.20`, `z = -1.96, p = 0.050`,
    /// `F = 3.20 on 2 and 30 df, p = 0.054`; `NA` for an undefined part.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self.distribution {
            Reference::Normal => "z",
            Reference::StudentT => "t",
            Reference::ChiSquare => "chi-square",
            Reference::F { .. } => "F",
            Reference::ChiBarSquare => "chi-bar-square",
            Reference::Exact => "statistic",
        };
        write!(
            f,
            "{name} = {}",
            format::optional(self.statistic, 3, f.precision())
        )?;
        let df = self.df.map_or_else(|| "NA".to_owned(), degrees);
        match self.distribution {
            Reference::F { df2 } => write!(f, " on {df} and {} df", degrees(df2))?,
            Reference::StudentT | Reference::ChiSquare | Reference::ChiBarSquare => {
                write!(f, " on {df} df")?;
            }
            Reference::Normal | Reference::Exact => {}
        }
        match self.p_value {
            Some(p) => write!(f, ", p {}", format::p_clause(p)),
            None => f.write_str(", p = NA"),
        }
    }
}

/// Degrees of freedom: a whole number as an integer, `Inf` for infinity,
/// three significant digits otherwise.
fn degrees(df: f64) -> String {
    if df.is_infinite() {
        "Inf".to_owned()
    } else if df.fract() == 0.0 && df.abs() < 1e15 {
        format!("{df:.0}")
    } else {
        format::optional(Some(df), 3, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tests_print_with_their_reference() {
        assert_eq!(
            Test::new(Some(2.1), Some(14.0), Some(0.054), Reference::StudentT).to_string(),
            "t = 2.10 on 14 df, p = 0.054"
        );
        assert_eq!(
            Test::new(
                Some(3.2),
                Some(2.0),
                Some(0.054),
                Reference::F { df2: 30.0 }
            )
            .to_string(),
            "F = 3.20 on 2 and 30 df, p = 0.054"
        );
        assert_eq!(
            Test::new(
                Some(3.2),
                Some(2.0),
                None,
                Reference::F { df2: f64::INFINITY }
            )
            .to_string(),
            "F = 3.20 on 2 and Inf df, p = NA"
        );
        assert_eq!(
            Test::new(Some(2.7), Some(1.0), Some(0.05), Reference::ChiBarSquare).to_string(),
            "chi-bar-square = 2.70 on 1 df, p = 0.050"
        );
        assert_eq!(
            Test::chi_square(None, 1.5, None).to_string(),
            "chi-square = NA on 1.50 df, p = NA"
        );
        assert_eq!(
            Test::new(None, None, Some(1e-20), Reference::Exact).to_string(),
            "statistic = NA, p < 2e-16"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn tests_round_trip_through_serde() {
        let f_test = Test::new(Some(1.0), Some(2.0), Some(0.5), Reference::F { df2: 9.0 });
        let json = serde_json::to_string(&f_test).unwrap();
        assert_eq!(
            json,
            r#"{"statistic":1.0,"df":2.0,"p_value":0.5,"distribution":{"f":{"df2":9.0}}}"#
        );
        assert_eq!(serde_json::from_str::<Test>(&json).unwrap(), f_test);
    }
}
