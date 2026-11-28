//! The two tables every measure of the crate starts from.
//!
//! [`Counts2x2`] holds the four counts of a 2 x 2 table, [`PersonTime`] the
//! events and person-time of one group and [`PersonTime2`] those of two
//! groups. The fields are named, so a table
//! means the same thing in every module: textbooks and R packages order the
//! cells differently (`episensr` puts the unexposed cases second, `epiR` the
//! exposed non-cases), and a table built for one order and read in the other
//! gives a silently wrong answer.
//!
//! # Formulas
//!
//! With `a` exposed cases, `b` exposed non-cases, `c` unexposed cases and `d`
//! unexposed non-cases,
//!
//! ```text
//! risk ratio      = (a / (a + b)) / (c / (c + d)),
//! odds ratio      = (a d) / (c b),
//! risk difference = a / (a + b) - c / (c + d),
//! ```
//!
//! and with `x1`, `t1` the exposed events and person-time and `x0`, `t0` the
//! unexposed ones, the rate ratio is `(x1 / t1) / (x0 / t0)` and the rate
//! difference `x1 / t1 - x0 / t0`. Each is `None` where it is not a finite
//! number.

/// A 2 x 2 table of counts by exposure and outcome. Cells may be fractional,
/// as in a table corrected for bias or a weighted one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Counts2x2 {
    /// Exposed cases.
    pub exposed_cases: f64,
    /// Exposed non-cases.
    pub exposed_noncases: f64,
    /// Unexposed cases.
    pub unexposed_cases: f64,
    /// Unexposed non-cases.
    pub unexposed_noncases: f64,
}

impl Counts2x2 {
    /// That every cell is a count, finite and not negative.
    pub(crate) fn check_cells(&self) -> Result<(), crate::Error> {
        for (what, value) in [
            ("exposed_cases", self.exposed_cases),
            ("exposed_noncases", self.exposed_noncases),
            ("unexposed_cases", self.unexposed_cases),
            ("unexposed_noncases", self.unexposed_noncases),
        ] {
            crate::error::count(what, value, None)?;
        }
        Ok(())
    }

    /// The risk ratio, or `None` when a risk is undefined or the unexposed
    /// risk is zero.
    #[must_use]
    pub fn risk_ratio(&self) -> Option<f64> {
        let (r1, r0) = (
            self.exposed_cases / (self.exposed_cases + self.exposed_noncases),
            self.unexposed_cases / (self.unexposed_cases + self.unexposed_noncases),
        );
        defined(r1 / r0)
    }

    /// The odds ratio, or `None` when it is undefined.
    #[must_use]
    pub fn odds_ratio(&self) -> Option<f64> {
        defined(
            (self.exposed_cases * self.unexposed_noncases)
                / (self.unexposed_cases * self.exposed_noncases),
        )
    }

    /// The risk difference, or `None` when a risk is undefined.
    #[must_use]
    pub fn risk_difference(&self) -> Option<f64> {
        defined(
            self.exposed_cases / (self.exposed_cases + self.exposed_noncases)
                - self.unexposed_cases / (self.unexposed_cases + self.unexposed_noncases),
        )
    }
}

/// The events and person-time of one group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonTime {
    /// The events.
    pub events: f64,
    /// The person-time.
    pub time: f64,
}

/// The events and person-time of an exposed and an unexposed group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonTime2 {
    /// Events among the exposed.
    pub exposed_events: f64,
    /// Person-time of the exposed.
    pub exposed_time: f64,
    /// Events among the unexposed.
    pub unexposed_events: f64,
    /// Person-time of the unexposed.
    pub unexposed_time: f64,
}

impl PersonTime2 {
    /// The rate ratio, or `None` when it is undefined.
    #[must_use]
    pub fn rate_ratio(&self) -> Option<f64> {
        let ratio = (self.exposed_events / self.exposed_time)
            / (self.unexposed_events / self.unexposed_time);
        ratio.is_finite().then_some(ratio)
    }

    /// The rate difference, or `None` when it is undefined.
    #[must_use]
    pub fn rate_difference(&self) -> Option<f64> {
        let difference =
            self.exposed_events / self.exposed_time - self.unexposed_events / self.unexposed_time;
        difference.is_finite().then_some(difference)
    }
}

fn defined(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}
