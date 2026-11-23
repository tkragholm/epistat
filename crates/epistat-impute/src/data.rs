use crate::error::{ImputeError, Result};

/// A set of flags, one bit per row, that marks the missing cells of a column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    bits: Vec<u64>,
    len: usize,
}

impl Mask {
    /// A mask of `len` rows with the rows where `flags` is true set.
    #[must_use]
    pub fn from_flags(flags: &[bool]) -> Self {
        let mut bits = vec![0_u64; flags.len().div_ceil(64)];
        for (row, _) in flags.iter().enumerate().filter(|&(_, &flag)| flag) {
            bits[row / 64] |= 1 << (row % 64);
        }
        Self {
            bits,
            len: flags.len(),
        }
    }

    /// The number of rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the mask has no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether row `row` is set. Rows past the end are not.
    #[must_use]
    pub fn get(&self, row: usize) -> bool {
        row < self.len && (self.bits[row / 64] >> (row % 64)) & 1 == 1
    }

    /// The number of rows that are set.
    #[must_use]
    pub fn count(&self) -> usize {
        self.bits
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// The rows that are set, in increasing order.
    #[must_use]
    pub fn rows(&self) -> Vec<usize> {
        (0..self.len).filter(|&row| self.get(row)).collect()
    }
}

/// One variable of the data.
///
/// Values at missing cells are ignored and may be anything. A binary value is
/// 0 or 1, a categorical code lies in `0..levels`.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// A numeric variable.
    Continuous(Vec<f64>),
    /// A variable with the values 0 and 1.
    Binary(Vec<u8>),
    /// An unordered categorical variable with `levels` levels, coded
    /// `0..levels`. As a predictor it enters as `levels - 1` indicators, the
    /// level 0 being the reference.
    Nominal {
        /// The level of each row.
        codes: Vec<u32>,
        /// The number of levels.
        levels: usize,
    },
    /// An ordered categorical variable. It can be a predictor, where its code
    /// enters as a score. It cannot be imputed, because ordered logistic
    /// imputation is not implemented.
    Ordered {
        /// The level of each row.
        codes: Vec<u32>,
        /// The number of levels.
        levels: usize,
    },
}

impl Column {
    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Continuous(v) => v.len(),
            Self::Binary(v) => v.len(),
            Self::Nominal { codes, .. } | Self::Ordered { codes, .. } => codes.len(),
        }
    }

    /// Whether the column has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value of row `row` as a number, a code for a categorical column.
    pub(crate) fn value(&self, row: usize) -> f64 {
        match self {
            Self::Continuous(v) => v[row],
            Self::Binary(v) => f64::from(v[row]),
            Self::Nominal { codes, .. } | Self::Ordered { codes, .. } => f64::from(codes[row]),
        }
    }

    /// The number of levels of a categorical column, 0 for the others.
    pub(crate) const fn levels(&self) -> usize {
        match self {
            Self::Nominal { levels, .. } | Self::Ordered { levels, .. } => *levels,
            _ => 0,
        }
    }
}

/// Columns of equal length and, for each, the cells that are missing.
#[derive(Clone, Debug, PartialEq)]
pub struct Incomplete {
    columns: Vec<Column>,
    missing: Vec<Mask>,
    rows: usize,
}

impl Incomplete {
    /// Data from its columns and, for each column, a flag per row that is true
    /// where the cell is missing.
    ///
    /// # Errors
    ///
    /// [`ImputeError::Length`] when the columns or masks differ in length,
    /// [`ImputeError::Levels`] for a categorical column with fewer than two
    /// levels, and [`ImputeError::Value`] for an observed value that is not
    /// finite, a binary value other than 0 and 1, or a code outside the
    /// levels.
    pub fn new(columns: Vec<Column>, missing: &[Vec<bool>]) -> Result<Self> {
        if missing.len() != columns.len() {
            return Err(ImputeError::Length {
                what: "missing",
                expected: columns.len(),
                found: missing.len(),
            });
        }
        let rows = columns.first().map_or(0, Column::len);
        for (j, (column, flags)) in columns.iter().zip(missing).enumerate() {
            if column.len() != rows {
                return Err(ImputeError::Length {
                    what: "column",
                    expected: rows,
                    found: column.len(),
                });
            }
            if flags.len() != rows {
                return Err(ImputeError::Length {
                    what: "missing mask",
                    expected: rows,
                    found: flags.len(),
                });
            }
            let levels = column.levels();
            if matches!(column, Column::Nominal { .. } | Column::Ordered { .. }) && levels < 2 {
                return Err(ImputeError::Levels { column: j, levels });
            }
            for row in (0..rows).filter(|&r| !flags[r]) {
                let ok = match column {
                    Column::Continuous(v) => v[row].is_finite(),
                    Column::Binary(v) => v[row] <= 1,
                    Column::Nominal { codes, levels } | Column::Ordered { codes, levels } => {
                        (codes[row] as usize) < *levels
                    }
                };
                if !ok {
                    return Err(ImputeError::Value { column: j, row });
                }
            }
        }
        Ok(Self {
            missing: missing
                .iter()
                .map(|flags| Mask::from_flags(flags))
                .collect(),
            columns,
            rows,
        })
    }

    /// The number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[must_use]
    pub const fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// The columns.
    #[must_use]
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// The mask of the missing cells of column `column`.
    ///
    /// # Panics
    ///
    /// If `column` is out of range.
    #[must_use]
    pub fn missing(&self, column: usize) -> &Mask {
        &self.missing[column]
    }
}
