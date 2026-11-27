//! The owned copy of the person-period data that the sampler reads, and its
//! rows grouped by subject.

use std::collections::BTreeMap;

use epistat_linalg::Mat;

/// The columns of a checked [`LongitudinalData`](crate::LongitudinalData),
/// copied once per fit.
#[derive(Debug, Clone)]
pub(crate) struct LongitudinalInput {
    pub outcome: Mat<f64>,
    pub x_binary: Mat<f64>,
    pub x_mean: Mat<f64>,
    pub subject_ids: Vec<u64>,
    pub time: Vec<f64>,
    pub family_ids: Option<Vec<u64>>,
}

#[cfg(test)]
impl LongitudinalInput {
    pub(crate) const fn new(
        outcome: Mat<f64>,
        x_binary: Mat<f64>,
        x_mean: Mat<f64>,
        subject_ids: Vec<u64>,
        time: Vec<f64>,
    ) -> Self {
        Self {
            outcome,
            x_binary,
            x_mean,
            subject_ids,
            time,
            family_ids: None,
        }
    }

    pub(crate) fn with_family_ids(mut self, family_ids: Vec<u64>) -> Self {
        self.family_ids = Some(family_ids);
        self
    }
}

/// Subject-wise row index bundle.
#[derive(Debug, Clone)]
pub(crate) struct SubjectRows {
    pub rows: Vec<usize>,
}

/// Prepared input view reused by sampler components.
#[derive(Debug, Clone)]
pub(crate) struct MtpPreparedInput<'a> {
    pub input: &'a LongitudinalInput,
    pub subjects: Vec<SubjectRows>,
}

/// The rows of each subject, subjects in the order of their codes.
pub(crate) fn prepare_input(input: &LongitudinalInput) -> MtpPreparedInput<'_> {
    let mut grouped: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (row, subject_id) in input.subject_ids.iter().copied().enumerate() {
        grouped.entry(subject_id).or_default().push(row);
    }
    let subjects = grouped
        .into_values()
        .map(|rows| SubjectRows { rows })
        .collect();
    MtpPreparedInput { input, subjects }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_input_groups_rows_by_subject() {
        let outcome = Mat::from_fn(4, 1, |_i, _| 1.0);
        let x_binary = Mat::from_fn(4, 1, |_i, _| 1.0);
        let x_mean = Mat::from_fn(4, 1, |_i, _| 1.0);
        let input = LongitudinalInput::new(
            outcome,
            x_binary,
            x_mean,
            vec![10, 11, 10, 11],
            vec![0.0, 0.0, 1.0, 1.0],
        );
        let prepared = prepare_input(&input);
        assert_eq!(prepared.subjects.len(), 2);
        assert_eq!(prepared.subjects[0].rows, vec![0, 2]);
    }
}
