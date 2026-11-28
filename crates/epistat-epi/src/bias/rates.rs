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

use crate::Error;
use epistat_rng::{Stream, replicate_seed};

use crate::tables::PersonTime2;

use super::{
    AdjustedResult, BiasError, Draws, Misclass, Pair, ProbOptions, check_options, finish_single,
    names,
};

impl PersonTime2 {
    fn check(&self) -> Result<(), Error> {
        let all = [
            self.exposed_events,
            self.unexposed_events,
            self.exposed_time,
            self.unexposed_time,
        ];
        if all.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(
                BiasError::Table("cases and person-time must be finite and not negative").into(),
            );
        }
        if self.exposed_events + self.unexposed_events <= 0.0
            || self.exposed_time <= 0.0
            || self.unexposed_time <= 0.0
        {
            return Err(
                BiasError::Table("there must be cases and person-time in both groups").into(),
            );
        }
        Ok(())
    }

    fn positive(&self) -> bool {
        [
            self.exposed_events,
            self.unexposed_events,
            self.exposed_time,
            self.unexposed_time,
        ]
        .iter()
        .all(|x| *x > 0.0)
    }
}

/// A corrected rate table and its measures; `None` throughout when a corrected
/// count or person-time is negative.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrectedRates {
    /// The corrected cases and person-time.
    pub table: Option<PersonTime2>,
    /// The corrected rate ratio.
    pub rate_ratio: Option<f64>,
    /// The corrected rate difference.
    pub rate_difference: Option<f64>,
}

impl CorrectedRates {
    fn from_table(t: PersonTime2) -> Self {
        let cells = [
            t.exposed_events,
            t.unexposed_events,
            t.exposed_time,
            t.unexposed_time,
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

fn exposure_rate_cells(t: &PersonTime2, se: [f64; 2], sp: [f64; 2]) -> PersonTime2 {
    let cases = t.exposed_events + t.unexposed_events;
    let time = t.exposed_time + t.unexposed_time;
    let big_a = (t.exposed_events - (1.0 - sp[0]) * cases) / (se[0] - (1.0 - sp[0]));
    let big_c = (t.exposed_time - (1.0 - sp[1]) * time) / (se[1] - (1.0 - sp[1]));
    PersonTime2 {
        exposed_events: big_a,
        unexposed_events: cases - big_a,
        exposed_time: big_c,
        unexposed_time: time - big_c,
    }
}

/// Corrects cases and person-time for misclassification of the exposure,
/// with the sensitivity and specificity of the classification of the cases
/// (stratum 0) and of the person-time (stratum 1).
///
/// # Errors
///
/// [`crate::Error`] when the table or a parameter is invalid.
pub fn correct_exposure_rates(t: &PersonTime2, m: &Misclass) -> Result<CorrectedRates, Error> {
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
/// [`crate::Error`] when the table is invalid, a predictive value is outside
/// `[0, 1]` or a sensitivity outside `(0, 1]`.
pub fn correct_outcome_rates_ppv(
    t: &PersonTime2,
    ppv: [f64; 2],
    se: [f64; 2],
) -> Result<CorrectedRates, Error> {
    t.check()?;
    super::check_ppv_sensitivity(ppv, se)?;
    Ok(CorrectedRates::from_table(PersonTime2 {
        exposed_events: t.exposed_events * ppv[0] / se[0],
        unexposed_events: t.unexposed_events * ppv[1] / se[1],
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
/// [`crate::Error`] when the table, a distribution or an option is invalid, and
/// [`crate::Error`] when every draw is discarded.
pub fn probabilistic_misclass_rates(
    t: &PersonTime2,
    se: &Pair,
    sp: &Pair,
    opts: &ProbOptions,
) -> Result<AdjustedResult, Error> {
    t.check()?;
    se.validate()?;
    sp.validate()?;
    check_options(opts)?;
    if !(t.exposed_events > 0.0 && t.unexposed_events > 0.0) {
        return Err(BiasError::Table("both groups need cases").into());
    }
    let standard_error = (1.0 / t.exposed_events + 1.0 / t.unexposed_events).sqrt();
    let mut draws = Draws::new(
        names(&[
            "sensitivity 0",
            "sensitivity 1",
            "specificity 0",
            "specificity 1",
        ]),
        opts.bootstrap.replicates,
        opts.random_error,
    );
    for r in 0..opts.bootstrap.replicates {
        let mut stream = Stream::seeded(replicate_seed(opts.bootstrap.seed.value(), r as u64));
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
        draws.push_parameters(&parameters);
        draws.systematic.push(corrected);
        if opts.random_error {
            draws.total.push(corrected.map(|irr| {
                let z = stream.normal();
                (irr.ln() - z * standard_error).exp()
            }));
        }
    }
    finish_single(draws, opts.level)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bias::Dist;
    use epistat_core::{Bootstrap, Level, Seed};
    use epistat_testkit::Tolerance;

    const TRUE: PersonTime2 = PersonTime2 {
        exposed_events: 120.0,
        unexposed_events: 300.0,
        exposed_time: 20_000.0,
        unexposed_time: 90_000.0,
    };

    /// Agreement to `1e-12`, absolute below 1 and relative above.
    const CLOSE: Tolerance = Tolerance::exact().abs(1e-12).rel(1e-12);

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
        let (a, b) = seen(TRUE.exposed_events, TRUE.unexposed_events, 0);
        let (c, d) = seen(TRUE.exposed_time, TRUE.unexposed_time, 1);
        let observed = PersonTime2 {
            exposed_events: a,
            unexposed_events: b,
            exposed_time: c,
            unexposed_time: d,
        };
        let corrected = correct_exposure_rates(&observed, &m).unwrap();
        let t = corrected.table.unwrap();
        assert!(CLOSE.accepts(t.exposed_events, TRUE.exposed_events));
        assert!(CLOSE.accepts(t.unexposed_time, TRUE.unexposed_time));
        assert!(CLOSE.accepts(corrected.rate_ratio.unwrap(), TRUE.rate_ratio().unwrap()));
    }

    #[test]
    fn equal_ppv_and_sensitivity_leave_the_rate_ratio() {
        let c = correct_outcome_rates_ppv(&TRUE, [0.9, 0.9], [0.8, 0.8]).unwrap();
        assert!(CLOSE.accepts(c.rate_ratio.unwrap(), TRUE.rate_ratio().unwrap()));
        let d = correct_outcome_rates_ppv(&TRUE, [0.9, 0.9], [0.9, 0.75]).unwrap();
        assert!(CLOSE.accepts(
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
                bootstrap: Bootstrap::new(20, Seed(1)),
                random_error: false,
                level: Level::alpha(0.05).unwrap(),
            },
        )
        .unwrap();
        assert_eq!(
            r.adjusted.unwrap().estimate.to_bits(),
            simple.rate_ratio.unwrap().to_bits()
        );
        assert_eq!(
            r.draws.parameters_of(3),
            &[Some(0.9), Some(0.85), Some(0.99), Some(0.98)]
        );
    }
}
