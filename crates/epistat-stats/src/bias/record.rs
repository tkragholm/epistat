//! Record-level correction for misclassification, with any model.
//!
//! The corrections of a table work on crude counts. A register study usually
//! reports an estimate adjusted for covariates by a regression. The
//! record-level correction of Fox, Lash and Greenland \[1\] reclassifies each
//! person's exposure or outcome in each draw and refits the model, so that
//! the covariates are handled as in the main analysis.
//!
//! # Algorithm
//!
//! A draw draws the sensitivities and specificities as
//! [`super::probabilistic_misclass`] does and corrects the crude table of the
//! records. From the corrected table it takes the predictive values of the
//! classification in each stratum (cases and non-cases for the exposure,
//! exposed and unexposed for the outcome): `PPV_k = Se_k T_k / O_k` and `NPV_k
//! = Sp_k (N_k - T_k) / (N_k - O_k)`, with `T_k` the true positives of the
//! corrected table, `O_k` the recorded positives and `N_k` the stratum size.
//! Each record recorded positive is then positive with probability `PPV_k`,
//! and each recorded negative with probability `1 - NPV_k`, by one uniform
//! draw a record in the order of the records. The model is fitted to the
//! reclassified variable and returns its estimate on the log scale with its
//! standard error. The systematic value is the exponential of the estimate,
//! and with random error the estimate is shifted by a standard normal draw
//! times its standard error first.
//!
//! A draw is discarded when a parameter is out of range, a corrected cell is
//! not positive, or the model returns `None` or a value that is not finite.
//!
//! # References
//!
//! 1. Fox MP, Lash TL, Greenland S. A method to automate probabilistic
//!    sensitivity analyses of misclassified binary variables. Int J
//!    Epidemiol. 2005;34(6):1370-6.

use epistat_rng::{Stream, replicate_seed};

use super::{
    AdjustedResult, BiasError, Draws, Kind, Pair, ProbOptions, Table2x2, check_options,
    exposure_cells, finish_single, names, outcome_cells, positive_cells,
};

/// A probabilistic record-level correction for misclassification of the
/// exposure or the outcome.
///
/// `exposed` and `case` are the recorded exposure and outcome of each
/// record. `fit` receives the reclassified variable, the exposure for
/// [`Kind::Exposure`] and the outcome for [`Kind::Outcome`], in the order of
/// the records, and returns the estimate on the log scale and its standard
/// error, or `None` when the model cannot be fitted. The result is on the
/// ratio scale. The parameters of each draw are named as in
/// [`super::probabilistic_misclass`].
///
/// # Errors
///
/// [`BiasError::Table`] when the records have different lengths or their
/// table is invalid, [`BiasError`] for an invalid distribution or option, and
/// [`BiasError::NoValidDraws`] when every draw is discarded.
pub fn record_level_misclass<F>(
    exposed: &[bool],
    case: &[bool],
    kind: Kind,
    se: &Pair,
    sp: &Pair,
    opts: &ProbOptions,
    mut fit: F,
) -> Result<AdjustedResult, BiasError>
where
    F: FnMut(&[bool]) -> Option<(f64, f64)>,
{
    if exposed.len() != case.len() {
        return Err(BiasError::Table(
            "exposure and outcome need one value a record",
        ));
    }
    se.validate()?;
    sp.validate()?;
    check_options(opts)?;
    let mut t = Table2x2 {
        a: 0.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
    };
    for (&e, &y) in exposed.iter().zip(case) {
        match (e, y) {
            (true, true) => t.a += 1.0,
            (false, true) => t.b += 1.0,
            (true, false) => t.c += 1.0,
            (false, false) => t.d += 1.0,
        }
    }
    t.check()?;
    // The recorded variable and the stratum of each record.
    let (recorded, stratum): (&[bool], &[bool]) = match kind {
        Kind::Exposure => (exposed, case),
        Kind::Outcome => (case, exposed),
    };
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
    let mut reclassified = vec![false; recorded.len()];
    for r in 0..opts.draws {
        let mut stream = Stream::seeded(replicate_seed(opts.seed, r as u64));
        let mut parameters = [f64::NAN; 4];
        let fitted = (|| {
            let s_e = se.draw(&mut stream)?;
            parameters[..2].copy_from_slice(&s_e);
            let s_p = sp.draw(&mut stream)?;
            parameters[2..].copy_from_slice(&s_p);
            if s_e.iter().chain(&s_p).any(|x| !(0.0..=1.0).contains(x))
                || s_e.iter().zip(&s_p).any(|(e, p)| e - (1.0 - p) <= 0.0)
            {
                return None;
            }
            let cells = match kind {
                Kind::Exposure => exposure_cells(&t, s_e, s_p),
                Kind::Outcome => outcome_cells(&t, s_e, s_p),
            };
            if !positive_cells(&cells) {
                return None;
            }
            // Per stratum (0 is cases for the exposure, exposed for the
            // outcome): true positives, recorded positives, size.
            let (strata_true, strata_recorded, strata_size) = match kind {
                Kind::Exposure => ([cells.a, cells.c], [t.a, t.c], [t.a + t.b, t.c + t.d]),
                Kind::Outcome => ([cells.a, cells.b], [t.a, t.b], [t.a + t.c, t.b + t.d]),
            };
            let mut ppv = [0.0; 2];
            let mut npv = [0.0; 2];
            for k in 0..2 {
                let (tp, rec, n) = (strata_true[k], strata_recorded[k], strata_size[k]);
                ppv[k] = if rec > 0.0 { s_e[k] * tp / rec } else { 1.0 };
                npv[k] = if n - rec > 0.0 {
                    s_p[k] * (n - tp) / (n - rec)
                } else {
                    1.0
                };
                if !((0.0..=1.0).contains(&ppv[k]) && (0.0..=1.0).contains(&npv[k])) {
                    return None;
                }
            }
            for ((out, &value), &group) in reclassified.iter_mut().zip(recorded).zip(stratum) {
                // Stratum 0 is the records with the stratum variable true.
                let k = usize::from(!group);
                let u = stream.uniform();
                *out = if value { u < ppv[k] } else { u >= npv[k] };
            }
            let (estimate, standard_error) = fit(&reclassified)?;
            (estimate.is_finite() && standard_error.is_finite() && standard_error >= 0.0)
                .then_some((estimate, standard_error))
        })();
        draws.parameters.extend_from_slice(&parameters);
        draws.systematic.push(fitted.map(|(e, _)| e.exp()));
        if opts.random_error {
            draws.total.push(fitted.map(|(e, s)| {
                let z = stream.normal();
                (e - z * s).exp()
            }));
        }
    }
    finish_single(draws, opts.alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bias::{Dist, Misclass, correct_exposure};

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the test tables hold whole counts"
    )]
    fn records(t: &Table2x2) -> (Vec<bool>, Vec<bool>) {
        let mut exposed = Vec::new();
        let mut case = Vec::new();
        for (e, y, n) in [
            (true, true, t.a),
            (false, true, t.b),
            (true, false, t.c),
            (false, false, t.d),
        ] {
            for _ in 0..n as usize {
                exposed.push(e);
                case.push(y);
            }
        }
        (exposed, case)
    }

    /// The log risk ratio of a table and its standard error.
    fn log_rr(exposed: &[bool], case: &[bool]) -> Option<(f64, f64)> {
        let mut t = [0.0_f64; 4];
        for (&e, &y) in exposed.iter().zip(case) {
            t[usize::from(!e) + 2 * usize::from(!y)] += 1.0;
        }
        let [a, b, c, d] = t;
        let rr = (a / (a + c)) / (b / (b + d));
        let se = (1.0 / a - 1.0 / (a + c) + 1.0 / b - 1.0 / (b + d)).sqrt();
        rr.is_finite().then(|| (rr.ln(), se))
    }

    #[test]
    fn perfect_classification_refits_the_records_as_they_are() {
        let t = Table2x2 {
            a: 40.0,
            b: 60.0,
            c: 360.0,
            d: 1540.0,
        };
        let (exposed, case) = records(&t);
        let r = record_level_misclass(
            &exposed,
            &case,
            Kind::Exposure,
            &Pair::Same(Dist::Constant(1.0)),
            &Pair::Same(Dist::Constant(1.0)),
            &ProbOptions {
                draws: 10,
                seed: 3,
                random_error: false,
                alpha: 0.05,
            },
            |x| log_rr(x, &case),
        )
        .unwrap();
        let rr = t.risk_ratio().unwrap();
        assert!((r.adjusted.median - rr).abs() < 1e-14 * rr);
        assert_eq!(r.adjusted.lower.to_bits(), r.adjusted.upper.to_bits());
    }

    #[test]
    fn the_reclassified_records_centre_on_the_corrected_table() {
        // The expected reclassified table is the corrected one, so the
        // median of many draws of the crude risk ratio lies near the
        // corrected risk ratio. With 2000 records the risk ratio of one
        // draw has a standard deviation of about 0.1 on the log scale; the
        // median of 400 draws is within 0.02 of its centre with room to
        // spare.
        let t = Table2x2 {
            a: 40.0,
            b: 60.0,
            c: 360.0,
            d: 1540.0,
        };
        let m = Misclass {
            se: [0.85, 0.85],
            sp: [0.97, 0.97],
        };
        let corrected = correct_exposure(&t, &m).unwrap().risk_ratio.unwrap();
        let (exposed, case) = records(&t);
        let r = record_level_misclass(
            &exposed,
            &case,
            Kind::Exposure,
            &Pair::Same(Dist::Constant(0.85)),
            &Pair::Same(Dist::Constant(0.97)),
            &ProbOptions {
                draws: 400,
                seed: 9,
                random_error: true,
                alpha: 0.05,
            },
            |x| log_rr(x, &case),
        )
        .unwrap();
        assert!(
            (r.adjusted.median.ln() - corrected.ln()).abs() < 0.02,
            "{} {corrected}",
            r.adjusted.median
        );
        let total = r.adjusted_total.unwrap();
        assert!(total.lower < r.adjusted.lower && total.upper > r.adjusted.upper);
    }
}
