//! Encoding covariate records into a numeric design matrix.

use super::EstimationError;
use super::config::{CovariateEncodingConfig, MissingValuePolicy};
use crate::matching::{ColumnReader, CovariateRecord, NameScan, to_f64};
use crate::types::CovariateRef;
use itertools::Itertools;
use std::collections::BTreeSet;

const MISSING_LEVEL: &str = "__missing__";

/// Row-major dense covariate matrix.
#[derive(Debug, Clone)]
pub struct CovariateMatrix {
    /// Row ids aligned with matrix rows.
    pub row_ids: Vec<String>,
    /// Column names aligned with matrix columns.
    pub column_names: Vec<String>,
    /// Row-major values with length `n_rows * n_cols`.
    pub values: Vec<f64>,
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
    /// The columns left out because every row has the same value in them.
    pub dropped_columns: Vec<String>,
}

impl CovariateMatrix {
    pub(super) fn row(&self, row_idx: usize) -> &[f64] {
        let start = row_idx * self.n_cols;
        let end = start + self.n_cols;
        &self.values[start..end]
    }
}

#[derive(Debug, Clone)]
enum CovariateSchema {
    Numeric {
        key: String,
        mean: f64,
    },
    Categorical {
        key: String,
        emitted_levels: Vec<String>,
    },
}

pub(super) fn build_covariate_matrix<R: CovariateRecord>(
    records: &[&R],
    config: &CovariateEncodingConfig,
) -> Result<CovariateMatrix, EstimationError> {
    build_covariate_matrix_marked(records, config).map(|(matrix, _)| matrix)
}

/// The matrix and, for each of its columns, whether it is a level indicator of
/// a categorical covariate.
pub(super) fn build_covariate_matrix_marked<R: CovariateRecord>(
    records: &[&R],
    config: &CovariateEncodingConfig,
) -> Result<(CovariateMatrix, Vec<bool>), EstimationError> {
    if records.is_empty() {
        return Err(EstimationError::EmptyInput);
    }

    let keys = resolve_covariate_keys(records, &config.covariate_keys);
    let mut schemas = Vec::new();
    for key in keys {
        let Some(schema) = build_schema_for_key(records, &key, config)? else {
            continue;
        };
        schemas.push(schema);
    }
    if schemas.is_empty() && !config.include_intercept {
        return Err(EstimationError::NoCovariatesAvailable);
    }

    let mut column_names = Vec::new();
    let mut indicator = Vec::new();
    if config.include_intercept {
        column_names.push("intercept".to_string());
        indicator.push(false);
    }
    for schema in &schemas {
        match schema {
            CovariateSchema::Numeric { key, .. } => {
                column_names.push(key.clone());
                indicator.push(false);
            }
            CovariateSchema::Categorical {
                key,
                emitted_levels,
                ..
            } => {
                column_names.extend(emitted_levels.iter().map(|level| format!("{key}={level}")));
                indicator.extend(emitted_levels.iter().map(|_| true));
            }
        }
    }

    let n_rows = records.len();
    let n_cols = column_names.len();
    let mut values = vec![0.0_f64; n_rows.saturating_mul(n_cols)];
    let row_ids = records
        .iter()
        .map(|record| record.id().to_string())
        .collect_vec();

    let mut readers = schemas
        .iter()
        .map(|schema| match schema {
            CovariateSchema::Numeric { key, .. } | CovariateSchema::Categorical { key, .. } => {
                ColumnReader::new(key)
            }
        })
        .collect_vec();
    for (row_idx, record_ref) in records.iter().enumerate() {
        let record = *record_ref;
        let mut col_idx = 0usize;
        if config.include_intercept {
            values[row_idx * n_cols + col_idx] = 1.0;
            col_idx += 1;
        }

        for (schema, reader) in schemas.iter().zip(&mut readers) {
            match schema {
                CovariateSchema::Numeric { key, mean } => {
                    let value = encode_numeric_value(
                        record,
                        reader.read(record),
                        key,
                        *mean,
                        config.missing_value_policy,
                    )?;
                    values[row_idx * n_cols + col_idx] = value;
                    col_idx += 1;
                }
                CovariateSchema::Categorical {
                    key,
                    emitted_levels,
                } => {
                    let level = encode_categorical_level(
                        reader.read(record),
                        key,
                        config.missing_value_policy,
                    )?;
                    for emitted in emitted_levels {
                        values[row_idx * n_cols + col_idx] =
                            if level == emitted.as_str() { 1.0 } else { 0.0 };
                        col_idx += 1;
                    }
                }
            }
        }
    }

    let mut matrix = CovariateMatrix {
        row_ids,
        column_names,
        values,
        n_rows,
        n_cols,
        dropped_columns: Vec::new(),
    };
    drop_constant_columns(&mut matrix, &mut indicator)?;
    Ok((matrix, indicator))
}

fn resolve_covariate_keys<R: CovariateRecord>(records: &[&R], requested: &[String]) -> Vec<String> {
    if !requested.is_empty() {
        return requested.to_vec();
    }

    let mut keys = BTreeSet::new();
    let mut scan = NameScan::default();
    for record in records {
        if !scan.needs_names(*record) {
            continue;
        }
        for name in record.covariate_names() {
            if !keys.contains(name) {
                keys.insert(name.to_owned());
            }
        }
    }
    keys.into_iter().collect_vec()
}

fn build_schema_for_key<R: CovariateRecord>(
    records: &[&R],
    key: &str,
    config: &CovariateEncodingConfig,
) -> Result<Option<CovariateSchema>, EstimationError> {
    let mut numeric_sum = 0.0_f64;
    let mut numeric_count = 0usize;
    let mut categorical_levels = BTreeSet::new();
    let mut observed_numeric = false;
    let mut observed_categorical = false;
    let mut saw_missing = false;

    let mut reader = ColumnReader::new(key);
    for record in records {
        match reader.read(record) {
            Some(CovariateRef::Numeric(value)) => {
                if !value.is_finite() {
                    return Err(EstimationError::NonFiniteNumericCovariate {
                        record_id: record.id().to_string(),
                        covariate: key.to_string(),
                    });
                }
                observed_numeric = true;
                numeric_sum += value;
                numeric_count += 1;
            }
            Some(CovariateRef::Categorical(value)) => {
                observed_categorical = true;
                if !categorical_levels.contains(value) {
                    categorical_levels.insert(value.to_owned());
                }
            }
            Some(CovariateRef::Missing) | None => {
                saw_missing = true;
            }
        }
    }

    if observed_numeric && observed_categorical {
        return Err(EstimationError::MixedCovariateType {
            covariate: key.to_string(),
        });
    }

    if !observed_numeric && !observed_categorical {
        if config.covariate_keys.is_empty() {
            return Ok(None);
        }
        return Err(EstimationError::NoObservedValuesForCovariate {
            covariate: key.to_string(),
        });
    }

    if observed_numeric {
        if saw_missing && matches!(config.missing_value_policy, MissingValuePolicy::Error) {
            return Err(EstimationError::MissingValuesNotAllowed {
                covariate: key.to_string(),
            });
        }
        let mean = if numeric_count == 0 {
            0.0
        } else {
            numeric_sum / to_f64(numeric_count)
        };
        return Ok(Some(CovariateSchema::Numeric {
            key: key.to_string(),
            mean,
        }));
    }

    if saw_missing && matches!(config.missing_value_policy, MissingValuePolicy::Impute) {
        categorical_levels.insert(MISSING_LEVEL.to_string());
    }
    if saw_missing && matches!(config.missing_value_policy, MissingValuePolicy::Error) {
        return Err(EstimationError::MissingValuesNotAllowed {
            covariate: key.to_string(),
        });
    }
    let levels = categorical_levels.into_iter().collect_vec();
    if levels.is_empty() {
        return Err(EstimationError::NoObservedValuesForCovariate {
            covariate: key.to_string(),
        });
    }

    let emitted_levels = if config.drop_first_categorical_level && levels.len() > 1 {
        levels.into_iter().skip(1).collect_vec()
    } else {
        levels
    };
    Ok(Some(CovariateSchema::Categorical {
        key: key.to_string(),
        emitted_levels,
    }))
}

fn encode_numeric_value<R: CovariateRecord>(
    record: &R,
    value: Option<CovariateRef<'_>>,
    key: &str,
    mean: f64,
    policy: MissingValuePolicy,
) -> Result<f64, EstimationError> {
    match value {
        Some(CovariateRef::Numeric(value)) => {
            if !value.is_finite() {
                return Err(EstimationError::NonFiniteNumericCovariate {
                    record_id: record.id().to_string(),
                    covariate: key.to_string(),
                });
            }
            Ok(value)
        }
        Some(CovariateRef::Categorical(_)) => Err(EstimationError::MixedCovariateType {
            covariate: key.to_string(),
        }),
        Some(CovariateRef::Missing) | None => match policy {
            MissingValuePolicy::Error => Err(EstimationError::MissingValuesNotAllowed {
                covariate: key.to_string(),
            }),
            MissingValuePolicy::Impute => Ok(mean),
        },
    }
}

fn encode_categorical_level<'r>(
    value: Option<CovariateRef<'r>>,
    key: &str,
    policy: MissingValuePolicy,
) -> Result<&'r str, EstimationError> {
    match value {
        Some(CovariateRef::Categorical(value)) => Ok(value),
        Some(CovariateRef::Numeric(_)) => Err(EstimationError::MixedCovariateType {
            covariate: key.to_string(),
        }),
        Some(CovariateRef::Missing) | None => match policy {
            MissingValuePolicy::Error => Err(EstimationError::MissingValuesNotAllowed {
                covariate: key.to_string(),
            }),
            MissingValuePolicy::Impute => Ok(MISSING_LEVEL),
        },
    }
}

/// Removes the columns whose value is exactly the same in every row, the
/// intercept excepted, and lists them in `dropped_columns`. A column with a
/// small but nonzero spread is kept, since the spread depends on the unit.
fn drop_constant_columns(
    matrix: &mut CovariateMatrix,
    indicator: &mut Vec<bool>,
) -> Result<(), EstimationError> {
    if matrix.n_cols == 0 {
        return Err(EstimationError::NoCovariatesAvailable);
    }

    let keep = (0..matrix.n_cols)
        .filter(|col_idx| {
            if matrix.column_names[*col_idx] == "intercept" {
                return true;
            }
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for row_idx in 0..matrix.n_rows {
                let value = matrix.values[row_idx * matrix.n_cols + *col_idx];
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }
            max_value > min_value
        })
        .collect_vec();

    if keep.is_empty() {
        return Err(EstimationError::NoCovariatesAvailable);
    }
    if keep.len() == matrix.n_cols {
        return Ok(());
    }

    let mut reduced = vec![0.0_f64; matrix.n_rows * keep.len()];
    for row_idx in 0..matrix.n_rows {
        for (new_col_idx, old_col_idx) in keep.iter().enumerate() {
            reduced[row_idx * keep.len() + new_col_idx] =
                matrix.values[row_idx * matrix.n_cols + *old_col_idx];
        }
    }

    matrix.dropped_columns = (0..matrix.n_cols)
        .filter(|idx| !keep.contains(idx))
        .map(|idx| matrix.column_names[idx].clone())
        .collect_vec();
    *indicator = keep.iter().map(|idx| indicator[*idx]).collect_vec();
    matrix.column_names = keep
        .iter()
        .map(|idx| matrix.column_names[*idx].clone())
        .collect_vec();
    matrix.values = reduced;
    matrix.n_cols = matrix.column_names.len();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::types::{BalanceRecord, CovariateValue};
    use crate::{Cohort, CohortRow};

    /// A deterministic mix of numeric and categorical covariates with
    /// missing values, as owned records and as two cohorts whose columns are
    /// in different orders.
    fn units(n: usize) -> (Vec<BalanceRecord>, Cohort, Cohort) {
        let mut state = 12345_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize
        };
        let mut owned = Vec::new();
        let mut ids = Vec::new();
        let mut age = Vec::new();
        let mut region = Vec::new();
        let mut income = Vec::new();
        for i in 0..n {
            let id = format!("u{i}");
            let a = (next() % 60) as f64;
            let r = match next() % 4 {
                0 => None,
                k => Some(["n", "s", "e"][k % 3]),
            };
            let inc = (next() % 5 != 0).then(|| (next() % 1000) as f64 / 7.0);
            let mut row = BalanceRecord::new(&id, date(2010, 1, 1));
            row.covariates
                .insert("age".to_string(), CovariateValue::Numeric(a));
            row.covariates.insert(
                "region".to_string(),
                r.map_or(CovariateValue::Missing, |r| {
                    CovariateValue::Categorical(r.to_string())
                }),
            );
            row.covariates.insert(
                "income".to_string(),
                inc.map_or(CovariateValue::Missing, CovariateValue::Numeric),
            );
            owned.push(row);
            ids.push(id);
            age.push(Some(a));
            region.push(r);
            income.push(inc);
        }
        let first_half = n / 2;
        let birth = vec![date(2010, 1, 1); n];
        let build = |range: std::ops::Range<usize>, reversed: bool| {
            let builder =
                Cohort::builder(ids[range.clone()].to_vec(), birth[range.clone()].to_vec());
            let age = age[range.clone()].to_vec();
            let region = region[range.clone()].to_vec();
            let income = income[range].to_vec();
            if reversed {
                builder
                    .numeric_covariate("income", income)
                    .categorical_covariate("region", region)
                    .numeric_covariate("age", age)
            } else {
                builder
                    .numeric_covariate("age", age)
                    .categorical_covariate("region", region)
                    .numeric_covariate("income", income)
            }
            .build()
            .expect("valid cohort")
        };
        (
            owned,
            build(0..first_half, false),
            build(first_half..n, true),
        )
    }

    #[test]
    fn rows_read_by_slot_encode_as_owned_records_read_by_name() {
        let (owned, first, second) = units(300);
        let mixed: Vec<CohortRow<'_>> = first.rows().chain(second.rows()).collect();
        let owned_refs: Vec<&BalanceRecord> = owned.iter().collect();
        let mixed_refs: Vec<&CohortRow<'_>> = mixed.iter().collect();
        let config = CovariateEncodingConfig {
            missing_value_policy: MissingValuePolicy::Impute,
            ..CovariateEncodingConfig::default()
        };
        let by_name = build_covariate_matrix(&owned_refs, &config).expect("owned");
        let by_slot = build_covariate_matrix(&mixed_refs, &config).expect("rows");
        assert_eq!(by_name.column_names, by_slot.column_names);
        assert_eq!(by_name.row_ids, by_slot.row_ids);
        let same = by_name
            .values
            .iter()
            .zip(&by_slot.values)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "design matrices differ");
    }
}
