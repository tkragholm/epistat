//! The data view of the models the estimators fit with `epistat-glm`, from
//! the columns an estimator holds.

use epistat_glm::GlmData;
use epistat_glm::rows::RowsFit;
use epistat_linalg::MatRef;

use crate::error::{Error, Result};

/// The data of a model fitted to `x` and `y`, with optional prior weights,
/// offset and starting coefficients, checked as `GlmData` checks them.
pub(crate) fn data<'a>(
    x: MatRef<'a, f64>,
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    start: Option<&'a [f64]>,
) -> Result<GlmData<'a>> {
    let mut builder = GlmData::builder(x, y);
    if let Some(w) = weights {
        builder = builder.weights(w);
    }
    if let Some(o) = offset {
        builder = builder.offset(o);
    }
    if let Some(s) = start {
        builder = builder.start(s);
    }
    Ok(builder.build()?)
}

/// Every coefficient of a fit over rows, as `Fit::full_rank` gives them for a
/// fit over a design: [`epistat_glm::Error::NotConverged`] for a fit that did
/// not converge and [`epistat_glm::Error::Aliased`] at the first aliased
/// column, both as the failure of `model`.
pub(crate) fn full_rank(fit: &RowsFit, model: impl Into<String>) -> Result<Vec<f64>> {
    let model = model.into();
    let failure = |source| Error::Model {
        model: model.clone(),
        source,
    };
    if !fit.converged {
        return Err(failure(epistat_glm::Error::NotConverged {
            iterations: fit.iterations,
            column: None,
        }));
    }
    fit.coefficients
        .iter()
        .enumerate()
        .map(|(column, c)| c.ok_or_else(|| failure(epistat_glm::Error::Aliased { column })))
        .collect()
}
