//! Which matched sets and strata carry information about a covariate.
//!
//! A conditional logistic model and a stratified Cox model compare rows within
//! a set or a stratum, so a set that cannot be compared adds nothing to the
//! estimate: one without a case or without a control, and one in which the
//! covariates do not differ between its rows. The sets that remain are the
//! ones the estimate is made from, and a study with matched or sibling sets
//! reports how many they are, since the number of sets in the data can be far
//! above it.
//!
//! # Definition
//!
//! A set is informative when it has rows of both outcome levels (at least one
//! case and at least one control, or in a Cox model at least one row with an
//! event and one without) and at least one covariate takes more than one value
//! among its rows. For [`informative_sets`] this is exact: the contribution of
//! a set to the conditional likelihood is the probability that the cases are
//! the cases, and it depends on the coefficients only when the covariates
//! differ. A set with both outcome levels and constant covariates adds a
//! constant to the log likelihood and nothing to its score and information. For a stratified Cox model ([`informative_strata`]) the conditions
//! are necessary and not sufficient, since the risk sets at the event times
//! within a stratum are subsets of its rows and an event may have no one
//! with another covariate value at risk; a stratum that fails them
//! contributes nothing, and one that meets them may still contribute nothing
//! when the times keep the rows apart.
//!
//! The count by column gives, for each covariate, the number of sets with both
//! outcome levels in which that covariate varies. A binary exposure's count is
//! the number of sets in which both exposure levels occur, which is the number
//! of sets that identify its effect on its own.
//!
//! Two covariate values are the same when they compare equal as numbers.
//! Rows of weight 0 are left out, as the fits do. A value that is not finite
//! is an error.

use std::collections::HashMap;

use epistat_core::check;
use epistat_linalg::MatRef;

use crate::check::{check_lengths, check_row_count, refuse_non_finite_matrix};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// What one set or stratum holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SetInformation {
    /// The label of the set, as given.
    pub set: u32,
    /// The rows used.
    pub rows: usize,
    /// The rows with the outcome (cases, or events).
    pub cases: usize,
    /// Whether some covariate takes more than one value among the rows.
    pub varies: bool,
}

impl SetInformation {
    /// Whether the set has rows of both outcome levels.
    #[must_use]
    pub const fn has_both_outcomes(&self) -> bool {
        self.cases > 0 && self.cases < self.rows
    }

    /// Whether the set carries information: both outcome levels and a
    /// covariate that varies.
    #[must_use]
    pub const fn is_informative(&self) -> bool {
        self.has_both_outcomes() && self.varies
    }
}

/// The sets of a data set and how many carry information.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InformativeSets {
    /// Every set, in the order of its first row.
    pub sets: Vec<SetInformation>,
    /// The number of sets.
    pub n_sets: usize,
    /// The number of informative sets.
    pub n_informative_sets: usize,
    /// The number of rows used.
    pub n_rows: usize,
    /// The number of rows in informative sets.
    pub n_informative_rows: usize,
    /// The number of cases in informative sets.
    pub n_informative_cases: usize,
    /// For each column of the design, the number of sets with both outcome
    /// levels in which it varies.
    pub informative_by_column: Vec<usize>,
}

/// Counts the informative matched sets of a conditional logistic model with the
/// design `x`, one row per entry of `case` and `set`. Rows with `weight` 0 are
/// left out.
///
/// # Errors
///
/// An input error when the columns differ in length, `x` has no column or a
/// value that is not finite, or a weight is negative or not finite.
pub fn informative_sets(
    case: &[bool],
    set: &[u32],
    x: MatRef<'_, f64>,
    weight: Option<&[f64]>,
) -> Result<InformativeSets> {
    check_lengths(&[("case", case.len()), ("set", set.len()), ("x", x.nrows())])?;
    if let Some(w) = weight {
        check_lengths(&[("case", case.len()), ("weight", w.len())])?;
        check::weights("weight", w)?;
    }
    count(case, set, x, weight)
}

/// Counts the strata of a stratified Cox model that can carry information
/// about the columns of `x`: those with at least one event and one row without
/// an event, in which a column varies. A row has the event when its status is
/// not 0. Without strata the data are one stratum. See the module documentation
/// for what the count means for a Cox model.
///
/// # Errors
///
/// An input error when `x` does not have one row per row of the data, has no
/// column or has a value that is not finite, and for interval-censored data.
pub fn informative_strata(data: &SurvData<'_>, x: MatRef<'_, f64>) -> Result<InformativeSets> {
    let status = data.followup().status();
    check_lengths(&[("status", status.len()), ("x", x.nrows())])?;
    let case: Vec<bool> = status.iter().map(|&s| s != 0).collect();
    match data.stratum() {
        Some(stratum) => count(&case, stratum, x, data.weight()),
        None => count(&case, &vec![0; case.len()], x, data.weight()),
    }
}

#[expect(
    clippy::float_cmp,
    reason = "two covariate values are the same when they are equal as numbers"
)]
fn count(
    case: &[bool],
    set: &[u32],
    x: MatRef<'_, f64>,
    weight: Option<&[f64]>,
) -> Result<InformativeSets> {
    let p = x.ncols();
    if p == 0 {
        return Err(Error::missing("x has no column".to_owned()));
    }
    check_row_count(case.len())?;
    refuse_non_finite_matrix(x, "x")?;
    let used = |row: usize| weight.is_none_or(|w| w[row] > 0.0);
    // Group the used rows by set, sets in the order of their first row.
    let mut index: HashMap<u32, usize> = HashMap::new();
    let mut labels = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();
    let mut slot = vec![usize::MAX; case.len()];
    for (row, &label) in set.iter().enumerate() {
        if !used(row) {
            continue;
        }
        let s = *index.entry(label).or_insert_with(|| {
            labels.push(label);
            sizes.push(0);
            labels.len() - 1
        });
        sizes[s] += 1;
        slot[row] = s;
    }
    let mut start = vec![0usize; labels.len() + 1];
    for (s, &size) in sizes.iter().enumerate() {
        start[s + 1] = start[s] + size;
    }
    let mut cursor = start.clone();
    let mut members = vec![0usize; start[labels.len()]];
    for (row, &s) in slot.iter().enumerate() {
        if s != usize::MAX {
            members[cursor[s]] = row;
            cursor[s] += 1;
        }
    }

    let mut out = InformativeSets {
        sets: Vec::with_capacity(labels.len()),
        n_sets: labels.len(),
        n_informative_sets: 0,
        n_rows: members.len(),
        n_informative_rows: 0,
        n_informative_cases: 0,
        informative_by_column: vec![0; p],
    };
    for (s, &label) in labels.iter().enumerate() {
        let rows = &members[start[s]..start[s + 1]];
        let cases = rows.iter().filter(|&&r| case[r]).count();
        let first = rows[0];
        let mut varies = false;
        let both = cases > 0 && cases < rows.len();
        for j in 0..p {
            let column_varies = rows.iter().any(|&r| x[(r, j)] != x[(first, j)]);
            varies |= column_varies;
            if both && column_varies {
                out.informative_by_column[j] += 1;
            }
        }
        let info = SetInformation {
            set: label,
            rows: rows.len(),
            cases,
            varies,
        };
        if info.is_informative() {
            out.n_informative_sets += 1;
            out.n_informative_rows += rows.len();
            out.n_informative_cases += cases;
        }
        out.sets.push(info);
    }
    Ok(out)
}
