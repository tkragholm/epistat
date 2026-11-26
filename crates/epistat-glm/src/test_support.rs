//! The models fitted from separate arguments, for the unit tests.

use epistat_linalg::MatRef;

use crate::{ElasticNet, ElasticNetFit, Error, Fit, Glm, GlmData};

impl Glm {
    /// [`Glm::fit`] on the view of the parts.
    pub(crate) fn fit5(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<Fit, Error> {
        self.fit(&GlmData::from_parts(x, y, weights, offset, start)?)
    }
}

impl ElasticNet {
    /// [`ElasticNet::fit`] on the view of the parts, with the penalty factors.
    pub(crate) fn fit6(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        penalty_factor: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<ElasticNetFit, Error> {
        let model = Self {
            penalty_factor: penalty_factor.map(<[f64]>::to_vec),
            ..self.clone()
        };
        model.fit(&GlmData::from_parts(x, y, weights, offset, start)?)
    }
}
