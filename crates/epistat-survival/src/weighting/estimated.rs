//! The weights of an outcome model when they come from estimated models: of
//! treatment, of censoring, or of both.
//!
//! An [`Estimated`] names the models. [`cox::fit_estimated`](super::cox::fit_estimated)
//! and [`km::estimate_estimated`](super::km::estimate_estimated) weight the rows
//! of the data with their product and give the variance that accounts for every
//! model being estimated. The theory is in the [module documentation of
//! `weighting`](super), and the term each model adds in `adjust`.
//!
//! The row weight is the weight of the subject from the propensity model, which
//! is indexed by `subject`, times the weight of the row from the censoring model.
//! The censoring models are fitted to the data without the treatment weights, and
//! the propensity model without the censoring weights, as is usual. The
//! variance treats the two as separate estimating equations whose parameters do
//! not affect each other.
//!
//! The units that estimating functions are summed over are, in order of
//! preference, the clusters of the data, the subjects of the propensity weights
//! (`subject`), the `id` of the data, and the rows.

use super::adjust::{IptwNuisance, Nuisance, PooledNuisance, WeightAt};
use super::censoring::ProcessNuisance;
use super::ipcw::{CensoringModel, IpcwOptions};
use super::iptw::Iptw;
use super::pooled::PooledIpcw;
use super::stack::{check_subjects, units};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// The model of censoring whose weights an outcome uses.
#[derive(Debug, Clone, Copy)]
pub enum Censoring<'a> {
    /// The Kaplan-Meier or Cox model of the censoring hazard of
    /// [`ipcw::fit`](super::ipcw::fit), fitted again to the data of the outcome
    /// model with the options given.
    Hazard {
        /// The model of the censoring hazard.
        model: CensoringModel<'a>,
        /// The options of the weights.
        options: &'a IpcwOptions,
        /// The weight of each row that the outcome uses.
        at: WeightAt,
    },
    /// The pooled logistic model of [`pooled::fit`](super::pooled::fit), whose
    /// rows are those of the data. A fit of
    /// [`pooled::fit_pooled`](super::pooled::fit_pooled) to a treatment that
    /// changes over time goes here too: its weights multiply the rows and its
    /// models add their terms in the same way.
    Pooled {
        /// The fit.
        fit: &'a PooledIpcw<'a>,
        /// The weight of each row that the outcome uses.
        at: WeightAt,
    },
}

/// The estimated weights of an outcome model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Estimated<'a> {
    /// The propensity weights of the subjects, `None` for none.
    pub iptw: Option<&'a Iptw<'a>>,
    /// The subject of each row, a position in the weights of `iptw`. `None` when
    /// the rows are the subjects. A subject may have several rows, and a subject
    /// may have none. It needs `iptw`.
    pub subject: Option<&'a [u32]>,
    /// The censoring weights, `None` for none.
    pub censoring: Option<Censoring<'a>>,
}

/// What the outcome fits need: the weight of every row, the units, and one term
/// of the variance for each model of weights.
pub(super) struct Prepared<'a> {
    pub weight: Vec<f64>,
    pub unit_of_row: Vec<usize>,
    pub n_units: usize,
    pub nuisances: Vec<Box<dyn Nuisance + 'a>>,
}

/// Fit the models of the weights that are fitted here, check the input, and form
/// the weights and the terms.
pub(super) fn prepare<'a>(data: &SurvData<'_>, est: &Estimated<'a>) -> Result<Prepared<'a>> {
    let n = data.n_rows();
    if est.iptw.is_none() && est.censoring.is_none() {
        return Err(Error::input(
            "weights: neither treatment nor censoring weights are given",
        ));
    }
    if est.iptw.is_none() && est.subject.is_some() {
        return Err(Error::input(
            "subject: it indexes the treatment weights, which are not given",
        ));
    }
    let id = data.id();
    let (n_subjects, subject_of_row): (usize, Vec<u32>) = match (est.iptw, id) {
        (Some(iptw), _) => {
            check_subjects(n, iptw.len(), est.subject)?;
            (
                iptw.len(),
                est.subject
                    .map_or_else(|| (0..n as u32).collect(), <[u32]>::to_vec),
            )
        }
        (None, Some(id)) => (data.n_ids(), id.to_vec()),
        (None, None) => (n, (0..n as u32).collect()),
    };
    let subj = |row: usize| subject_of_row[row] as usize;
    let (unit_of_subject, n_units) = units(n_subjects, &subj, data.cluster(), data.n_clusters())?;
    let unit_of_row: Vec<usize> = (0..n).map(|row| unit_of_subject[subj(row)]).collect();

    let mut weight = vec![1.0; n];
    let mut nuisances: Vec<Box<dyn Nuisance + 'a>> = Vec::new();
    if let Some(iptw) = est.iptw {
        for (row, w) in weight.iter_mut().enumerate() {
            *w *= iptw.weights[subj(row)];
        }
        nuisances.push(Box::new(IptwNuisance::new(
            iptw,
            subject_of_row.clone(),
            unit_of_subject.clone(),
        )?));
    }
    match est.censoring {
        None => {}
        Some(Censoring::Pooled { fit, at }) => {
            if fit.len() != n {
                return Err(Error::input(format!(
                    "censoring: the pooled fit has {} rows for {n} rows of data",
                    fit.len()
                )));
            }
            let w = match at {
                WeightAt::Stop => &fit.weight,
                WeightAt::Start => &fit.weight_before,
            };
            for (a, b) in weight.iter_mut().zip(w) {
                *a *= b;
            }
            nuisances.push(Box::new(PooledNuisance::new(fit, at, unit_of_row.clone())?));
        }
        Some(Censoring::Hazard { model, options, at }) => {
            let (w, nuisance) =
                ProcessNuisance::new(data, model, options, at, unit_of_row.clone(), n_units)?;
            for (a, b) in weight.iter_mut().zip(&w) {
                *a *= b;
            }
            nuisances.push(Box::new(nuisance));
        }
    }
    Ok(Prepared {
        weight,
        unit_of_row,
        n_units,
        nuisances,
    })
}
