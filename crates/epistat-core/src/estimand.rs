//! The population an average treatment effect is averaged over.
//!
//! With potential outcomes `Y(1)` and `Y(0)`, treatment `A` and propensity
//! score `e(x) = P(A = 1 | X = x)`, the estimands are weighted averages of
//! `Y(1) - Y(0)`:
//!
//! | estimand | population | weight of a unit with score `e` |
//! |---|---|---|
//! | ATE | everyone | `1` |
//! | ATT | the treated | `e` |
//! | ATC | the untreated | `1 - e` |
//! | ATO | the overlap population | `e (1 - e)` |
//! | ATM | the matching population | `min(e, 1 - e)` |
//!
//! The overlap weights are those of Li, Morgan and Zaslavsky \[1\] and the
//! matching weights those of Li and Greene \[2\]. A method that does not
//! estimate an estimand returns an error saying so; it does not substitute
//! another.
//!
//! # References
//!
//! 1. Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score
//!    weighting. J Am Stat Assoc. 2018;113(521):390-400.
//!    doi:10.1080/01621459.2016.1260466
//! 2. Li L, Greene T. A weighting analogue to pair matching in propensity
//!    score analysis. Int J Biostat. 2013;9(2):215-34.
//!    doi:10.1515/ijb-2012-0030

/// The target population of an average treatment effect.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Estimand {
    /// The average treatment effect over everyone.
    Ate,
    /// The average treatment effect on the treated.
    Att,
    /// The average treatment effect on the controls, the untreated.
    Atc,
    /// The average treatment effect in the overlap population.
    Ato,
    /// The average treatment effect in the matching population.
    Atm,
}

impl Estimand {
    /// The estimand's abbreviation in capitals, as R's packages print it:
    /// `"ATE"`, `"ATT"`, `"ATC"`, `"ATO"` or `"ATM"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ate => "ATE",
            Self::Att => "ATT",
            Self::Atc => "ATC",
            Self::Ato => "ATO",
            Self::Atm => "ATM",
        }
    }
}

impl core::fmt::Display for Estimand {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimands_print_as_r_does() {
        let names: Vec<String> = [
            Estimand::Ate,
            Estimand::Att,
            Estimand::Atc,
            Estimand::Ato,
            Estimand::Atm,
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        assert_eq!(names, ["ATE", "ATT", "ATC", "ATO", "ATM"]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_uses_lower_case_names() {
        assert_eq!(serde_json::to_string(&Estimand::Att).unwrap(), r#""att""#);
        assert_eq!(
            serde_json::from_str::<Estimand>(r#""ato""#).unwrap(),
            Estimand::Ato
        );
    }
}
