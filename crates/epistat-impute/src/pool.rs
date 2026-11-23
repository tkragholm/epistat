use epistat_inference::pooling::{Pooled, PooledVector, rubin, rubin_vector};

use crate::data::Incomplete;
use crate::error::{ImputeError, Result};
use crate::impute::Imputed;

impl Imputed {
    /// The completed data of set `set`: every column as numbers (a binary or
    /// categorical value as its code), observed values from `data` and
    /// missing cells from the imputation.
    ///
    /// # Errors
    ///
    /// [`ImputeError::Length`] when `set` is out of range or `data` is not the
    /// data that was imputed.
    pub fn completed(&self, data: &Incomplete, set: usize) -> Result<Vec<Vec<f64>>> {
        if set >= self.m {
            return Err(ImputeError::Length {
                what: "set",
                expected: self.m,
                found: set,
            });
        }
        if data.n_columns() != self.values.len() {
            return Err(ImputeError::Length {
                what: "columns",
                expected: self.values.len(),
                found: data.n_columns(),
            });
        }
        Ok((0..data.n_columns())
            .map(|j| {
                let mut column: Vec<f64> =
                    (0..data.rows()).map(|r| data.columns()[j].value(r)).collect();
                for (&row, &value) in self.missing_rows[j].iter().zip(&self.values[j][set]) {
                    column[row] = value;
                }
                column
            })
            .collect())
    }

    /// Analyses each completed data set with `analysis`, which returns an
    /// estimate and its variance, and pools them by Rubin's rules with the
    /// Barnard-Rubin degrees of freedom for `dfcom` complete-data degrees of
    /// freedom (`None` for infinite).
    ///
    /// # Errors
    ///
    /// [`ImputeError::Analysis`] for an analysis that fails, and the errors of
    /// [`completed`](Self::completed) and of
    /// [`rubin`](epistat_inference::pooling::rubin).
    pub fn pool<E>(
        &self,
        data: &Incomplete,
        dfcom: Option<f64>,
        analysis: impl Fn(&[Vec<f64>]) -> std::result::Result<(f64, f64), E>,
    ) -> Result<Pooled>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let (mut estimates, mut variances) = (Vec::new(), Vec::new());
        for set in 0..self.m {
            let completed = self.completed(data, set)?;
            let (q, u) = analysis(&completed).map_err(|e| ImputeError::Analysis {
                set,
                source: Box::new(e),
            })?;
            estimates.push(q);
            variances.push(u);
        }
        Ok(rubin(&estimates, &variances, dfcom)?)
    }

    /// As [`pool`](Self::pool) for `k` estimates and their `k` by `k`
    /// covariance matrix, row-major, by
    /// [`rubin_vector`](epistat_inference::pooling::rubin_vector).
    ///
    /// # Errors
    ///
    /// As [`pool`](Self::pool).
    pub fn pool_vector<E>(
        &self,
        data: &Incomplete,
        k: usize,
        analysis: impl Fn(&[Vec<f64>]) -> std::result::Result<(Vec<f64>, Vec<f64>), E>,
    ) -> Result<PooledVector>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let (mut estimates, mut covariances) = (Vec::new(), Vec::new());
        for set in 0..self.m {
            let completed = self.completed(data, set)?;
            let (q, u) = analysis(&completed).map_err(|e| ImputeError::Analysis {
                set,
                source: Box::new(e),
            })?;
            estimates.push(q);
            covariances.push(u);
        }
        let q: Vec<&[f64]> = estimates.iter().map(Vec::as_slice).collect();
        let u: Vec<&[f64]> = covariances.iter().map(Vec::as_slice).collect();
        Ok(rubin_vector(&q, &u, k)?)
    }
}
