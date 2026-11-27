//! The canonical 2x2 difference in means, behind
//! [`two_period::two_by_two`](crate::two_period::two_by_two).

use crate::error::Error;
use crate::inference::normal_interval;
use crate::types::{CellSummary, DidCell, DidInputSummary, TimePeriod, TreatmentGroup};

#[derive(Debug, Clone, Copy, Default)]
struct CellStats {
    observations: usize,
    weight: f64,
    weight_sq: f64,
    outcome_w: f64,
    outcome_sq_w: f64,
}

impl CellStats {
    fn add(&mut self, weight: f64, outcome: f64) {
        self.observations += 1;
        self.weight += weight;
        self.weight_sq = weight.mul_add(weight, self.weight_sq);
        self.outcome_w = weight.mul_add(outcome, self.outcome_w);
        self.outcome_sq_w = (weight * outcome).mul_add(outcome, self.outcome_sq_w);
    }

    fn validate_non_empty(self, cell: DidCell) -> Result<Self, Error> {
        if self.weight > 0.0 {
            Ok(self)
        } else {
            let (treated, post) = cell.flags();
            Err(Error::EmptyCell { treated, post })
        }
    }

    fn into_summary(self, cell: DidCell) -> CellSummary {
        CellSummary {
            cell,
            observations: self.observations,
            weight_sum: self.weight,
            effective_n: self.effective_n(),
            mean_outcome: self.mean(),
            variance: self.variance(),
        }
    }

    fn mean(self) -> f64 {
        self.outcome_w / self.weight
    }

    fn effective_n(self) -> f64 {
        self.weight * self.weight / self.weight_sq
    }

    fn variance(self) -> f64 {
        let mean = self.mean();
        mean.mul_add(-mean, self.outcome_sq_w / self.weight)
            .max(0.0)
    }
}

/// The four cells of a repeated cross-section: rows `(treated, post,
/// outcome, weight)`, validated by the caller.
pub(crate) fn summarize_two_by_two(
    rows: impl Iterator<Item = (bool, bool, f64, f64)>,
) -> Result<DidInputSummary, Error> {
    let mut treated_pre = CellStats::default();
    let mut treated_post = CellStats::default();
    let mut control_pre = CellStats::default();
    let mut control_post = CellStats::default();

    for (treated, post, outcome, weight) in rows {
        match DidCell::from_parts(
            TreatmentGroup::from_bool(treated),
            TimePeriod::from_bool(post),
        ) {
            DidCell::TreatedPre => treated_pre.add(weight, outcome),
            DidCell::TreatedPost => treated_post.add(weight, outcome),
            DidCell::ControlPre => control_pre.add(weight, outcome),
            DidCell::ControlPost => control_post.add(weight, outcome),
        }
    }

    let treated_pre = treated_pre.validate_non_empty(DidCell::TreatedPre)?;
    let treated_post = treated_post.validate_non_empty(DidCell::TreatedPost)?;
    let control_pre = control_pre.validate_non_empty(DidCell::ControlPre)?;
    let control_post = control_post.validate_non_empty(DidCell::ControlPost)?;

    Ok(DidInputSummary {
        treated_pre: treated_pre.into_summary(DidCell::TreatedPre),
        treated_post: treated_post.into_summary(DidCell::TreatedPost),
        control_pre: control_pre.into_summary(DidCell::ControlPre),
        control_post: control_post.into_summary(DidCell::ControlPost),
    })
}

/// The 2x2 ATT of the four cells, with its standard error and the normal
/// interval at `confidence_level`.
pub(crate) fn estimate_att_from_summary(
    summary: DidInputSummary,
    confidence_level: f64,
) -> (f64, f64, (f64, f64)) {
    let att = (summary.treated_post.mean_outcome - summary.treated_pre.mean_outcome)
        - (summary.control_post.mean_outcome - summary.control_pre.mean_outcome);
    let se = ((summary.treated_pre.variance / summary.treated_pre.effective_n)
        + (summary.treated_post.variance / summary.treated_post.effective_n)
        + (summary.control_pre.variance / summary.control_pre.effective_n)
        + (summary.control_post.variance / summary.control_post.effective_n))
        .sqrt();
    (att, se, normal_interval(att, se, confidence_level))
}
