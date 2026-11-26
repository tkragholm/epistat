//! Bias analysis of rates: cases and person-time by exposure.
//!
//! # Algorithm
//!
//! Exposure misclassification moves cases and person-time between the
//! exposure groups. With sensitivity `Se_0` and specificity `Sp_0` of the
//! exposure classification among the cases, and `Se_1` and `Sp_1` for the
//! person-time, the true exposed cases and person-time are
//!
//! ```text
//! A = ( a - (1 - Sp_0) (a + b) ) / ( Se_0 - (1 - Sp_0) ),      B = (a + b) - A
//! C = ( c - (1 - Sp_1) (c + d) ) / ( Se_1 - (1 - Sp_1) ),      D = (c + d) - C
//! ```
//!
//! with `a`, `b` the cases and `c`, `d` the person-time of the exposed and
//! unexposed, as `episensr::probsens_irr` corrects them \[1\]. The
//! rate ratio is `(A / C) / (B / D)`. `probsens_irr` reports
//! `(A / (A + C)) / (B / (B + D))` instead, which treats the person-time as
//! non-cases and is close to the rate ratio only when the cases are few
//! beside the person-time.
//!
//! For the outcome, a validation study's positive predictive value and an
//! assumed sensitivity give the true events of a group as the recorded times
//! `PPV / Se`. The person-time is left as it is, which assumes that the
//! person-time a missed or a false event would move is small beside the
//! total, as it is for a rare outcome.
//!
//! The probabilistic version draws the parameters as
//! [`super::probabilistic_misclass`] does and, with random error, shifts the
//! corrected log rate ratio by a standard normal draw times `sqrt(1 / a + 1 /
//! b)`, the standard error of the observed log rate ratio. `probsens_irr`
//! takes the standard error from the exact binomial interval instead.
//!
//! # References
//!
//! 1. Lash TL, Fox MP, Fink AK. Applying Quantitative Bias Analysis to
//!    Epidemiologic Data. New York: Springer; 2009.

use epistat_rng::{Stream, replicate_seed};

use super::{
    AdjustedResult, BiasError, Draws, Misclass, Pair, ProbOptions, check_options, finish_single,
    names,
};

/// Cases and person-time by exposure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateTable {
    /// Cases among the exposed.
    pub cases_exposed: f64,
    /// Cases among the unexposed.
    pub cases_unexposed: f64,
    /// Person-time of the exposed.
    pub time_exposed: f64,
    /// Person-time of the unexposed.
    pub time_unexposed: f64,
}

impl RateTable {
    /// The rate ratio, or `None` when it is undefined.
    #[must_use]
    pub fn rate_ratio(&self) -> Option<f64> {
        let ratio =
            (self.cases_exposed / self.time_exposed) / (self.cases_unexposed / self.time_unexposed);
        ratio.is_finite().then_some(ratio)
    }

    /// The rate difference, or `None` when it is undefined.
    #[must_use]
    pub fn rate_difference(&self) -> Option<f64> {
        let difference =
            self.cases_exposed / self.time_exposed - self.cases_unexposed / self.time_unexposed;
        difference.is_finite().then_some(difference)
    }

    fn check(&self) -> Result<(), BiasError> {
        let all = [
            self.cases_exposed,
            self.cases_unexposed,
            self.time_exposed,
            self.time_unexposed,
        ];
        if all.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(BiasError::Table(
                "cases and person-time must be finite and not negative",
            ));
        }
        if self.cases_exposed + self.cases_unexposed <= 0.0
            || self.time_exposed <= 0.0
            || self.time_unexposed <= 0.0
        {
            return Err(BiasError::Table(
                "there must be cases and person-time in both groups",
            ));
        }
        Ok(())
    }

    fn positive(&self) -> bool {
        [
            self.cases_exposed,
            self.cases_unexposed,
            self.time_exposed,
            self.time_unexposed,
        ]
        .iter()
        .all(|x| *x > 0.0)
    }
}

/// A corrected rate table and its measures; `None` throughout when a corrected
/// count or person-time is negative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectedRates {
    /// The corrected cases and person-time.
    pub table: Option<RateTable>,
    /// The corrected rate ratio.
    pub rate_ratio: Option<f64>,
    /// The corrected rate difference.
    pub rate_difference: Option<f64>,
}

impl CorrectedRates {
    fn from_table(t: RateTable) -> Self {
        let cells = [
            t.cases_exposed,
            t.cases_unexposed,
            t.time_exposed,
            t.time_unexposed,
        ];
        if cells.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Self {
                table: None,
                rate_ratio: None,
                rate_difference: None,
            };
        }
        Self {
            table: Some(t),
            rate_ratio: t.rate_ratio(),
            rate_difference: t.rate_difference(),
        }
    }
}

fn exposure_rate_cells(t: &RateTable, se: [f64; 2], sp: [f64; 2]) -> RateTable {
    let cases = t.cases_exposed + t.cases_unexposed;
    let time = t.time_exposed + t.time_unexposed;
    let big_a = (t.cases_exposed - (1.0 - sp[0]) * cases) / (se[0] - (1.0 - sp[0]));
    let big_c = (t.time_exposed - (1.0 - sp[1]) * time) / (se[1] - (1.0 - sp[1]));
    RateTable {
        cases_exposed: big_a,
        cases_unexposed: cases - big_a,
        time_exposed: big_c,
        time_unexposed: time - big_c,
    }
}

/// Corrects cases and person-time for misclassification of the exposure,
/// with the sensitivity and specificity of the classification of the cases
/// (stratum 0) and of the person-time (stratum 1).
///
/// # Errors
///
/// [`BiasError`] when the table or a parameter is invalid.
pub fn correct_exposure_rates(t: &RateTable, m: &Misclass) -> Result<CorrectedRates, BiasError> {
    t.check()?;
    m.check()?;
    Ok(CorrectedRates::from_table(exposure_rate_cells(
        t, m.se, m.sp,
    )))
}

/// Corrects the cases of a rate table for misclassification of the outcome
/// from the positive predictive value and the sensitivity of the outcome
/// classification among the exposed (index 0) and the unexposed (index 1):
/// the true events are the recorded times `PPV / Se`, and the person-time is
/// kept.
///
/// # Errors
///
/// [`BiasError`] when the table is invalid, a predictive value is outside
/// `[0, 1]` or a sensitivity outside `(0, 1]`.
pub fn correct_outcome_rates_ppv(
    t: &RateTable,
    ppv: [f64; 2],
    se: [f64; 2],
) -> Result<CorrectedRates, BiasError> {
    t.check()?;
    super::check_ppv_sensitivity(ppv, se)?;
    Ok(CorrectedRates::from_table(RateTable {
        cases_exposed: t.cases_exposed * ppv[0] / se[0],
        cases_unexposed: t.cases_unexposed * ppv[1] / se[1],
        ..*t
    }))
}

/// A probabilistic correction of the rate ratio for misclassification of the
/// exposure, as `episensr::probsens_irr`. `se` and `sp` are the parameter
/// pairs of [`correct_exposure_rates`], and the result is the corrected rate
/// ratio. The parameters of each draw are named as in
/// [`super::probabilistic_misclass`].
///
/// # Errors
///
/// [`BiasError`] when the table, a distribution or an option is invalid, and
/// [`BiasError::NoValidDraws`] when every draw is discarded.
pub fn probabilistic_misclass_rates(
    t: &RateTable,
    se: &Pair,
    sp: &Pair,
    opts: &ProbOptions,
) -> Result<AdjustedResult, BiasError> {
    t.check()?;
    se.validate()?;
    sp.validate()?;
    check_options(opts)?;
    if !(t.cases_exposed > 0.0 && t.cases_unexposed > 0.0) {
        return Err(BiasError::Table("both groups need cases"));
    }
    let standard_error = (1.0 / t.cases_exposed + 1.0 / t.cases_unexposed).sqrt();
    let mut draws = Draws::new(
        names(&[
            "sensitivity 0",
            "sensitivity 1",
            "specificity 0",
            "specificity 1",
        ]),
        opts.draws,
        opts.random_error,
    );
    for r in 0..opts.draws {
        let mut stream = Stream::seeded(replicate_seed(opts.seed, r as u64));
        let mut parameters = [f64::NAN; 4];
        let corrected = (|| {
            let s_e = se.draw(&mut stream)?;
            parameters[..2].copy_from_slice(&s_e);
            let s_p = sp.draw(&mut stream)?;
            parameters[2..].copy_from_slice(&s_p);
            if s_e.iter().chain(&s_p).any(|x| !(0.0..=1.0).contains(x))
                || s_e.iter().zip(&s_p).any(|(e, p)| e - (1.0 - p) <= 0.0)
            {
                return None;
            }
            let cells = exposure_rate_cells(t, s_e, s_p);
            if !cells.positive() {
                return None;
            }
            cells.rate_ratio()
        })();
        draws.parameters.extend_from_slice(&parameters);
        draws.systematic.push(corrected);
        if opts.random_error {
            draws.total.push(corrected.map(|irr| {
                let z = stream.normal();
                (irr.ln() - z * standard_error).exp()
            }));
        }
    }
    finish_single(draws, opts.alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bias::Dist;

    const TRUE: RateTable = RateTable {
        cases_exposed: 120.0,
        cases_unexposed: 300.0,
        time_exposed: 20_000.0,
        time_unexposed: 90_000.0,
    };

    fn close(x: f64, y: f64) -> bool {
        (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0)
    }

    #[test]
    fn the_correction_inverts_the_misclassification() {
        let m = Misclass {
            se: [0.9, 0.8],
            sp: [0.97, 0.95],
        };
        let seen = |exposed: f64, unexposed: f64, k: usize| {
            (
                m.se[k] * exposed + (1.0 - m.sp[k]) * unexposed,
                (1.0 - m.se[k]) * exposed + m.sp[k] * unexposed,
            )
        };
        let (a, b) = seen(TRUE.cases_exposed, TRUE.cases_unexposed, 0);
        let (c, d) = seen(TRUE.time_exposed, TRUE.time_unexposed, 1);
        let observed = RateTable {
            cases_exposed: a,
            cases_unexposed: b,
            time_exposed: c,
            time_unexposed: d,
        };
        let corrected = correct_exposure_rates(&observed, &m).unwrap();
        let t = corrected.table.unwrap();
        assert!(close(t.cases_exposed, TRUE.cases_exposed));
        assert!(close(t.time_unexposed, TRUE.time_unexposed));
        assert!(close(
            corrected.rate_ratio.unwrap(),
            TRUE.rate_ratio().unwrap()
        ));
    }

    #[test]
    fn equal_ppv_and_sensitivity_leave_the_rate_ratio() {
        let c = correct_outcome_rates_ppv(&TRUE, [0.9, 0.9], [0.8, 0.8]).unwrap();
        assert!(close(c.rate_ratio.unwrap(), TRUE.rate_ratio().unwrap()));
        let d = correct_outcome_rates_ppv(&TRUE, [0.9, 0.9], [0.9, 0.75]).unwrap();
        assert!(close(
            d.rate_ratio.unwrap(),
            TRUE.rate_ratio().unwrap() * 0.75 / 0.9
        ));
    }

    #[test]
    fn constant_parameters_give_the_simple_correction() {
        let m = Misclass {
            se: [0.9, 0.85],
            sp: [0.99, 0.98],
        };
        let simple = correct_exposure_rates(&TRUE, &m).unwrap();
        let r = probabilistic_misclass_rates(
            &TRUE,
            &Pair::Independent([Dist::Constant(0.9), Dist::Constant(0.85)]),
            &Pair::Independent([Dist::Constant(0.99), Dist::Constant(0.98)]),
            &ProbOptions {
                draws: 20,
                seed: 1,
                random_error: false,
                alpha: 0.05,
            },
        )
        .unwrap();
        assert_eq!(
            r.adjusted.median.to_bits(),
            simple.rate_ratio.unwrap().to_bits()
        );
        assert_eq!(r.draws.parameters_of(3), &[0.9, 0.85, 0.99, 0.98]);
    }
}
