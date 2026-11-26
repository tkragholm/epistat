//! The models fitted from separate arguments: the data views built from
//! optional columns, so that a test can pass `None` where a model has no
//! weights, offset or start.
#![allow(dead_code)]

use epistat_glm::{
    ElasticNet, ElasticNetFit, Error, Fit, Gee, GeeData, GeeFit, Glm, GlmData, Multinom,
    MultinomData, MultinomFit, NegBin, NegBinFit, Ordinal, OrdinalData, OrdinalFit,
};
use epistat_linalg::MatRef;

/// The view of the parts.
pub fn glm_data<'a>(
    x: MatRef<'a, f64>,
    y: &'a [f64],
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    start: Option<&'a [f64]>,
) -> Result<GlmData<'a>, Error> {
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
    builder.build()
}

pub trait GlmParts {
    fn fit5(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<Fit, Error>;
}

impl GlmParts for Glm {
    fn fit5(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<Fit, Error> {
        self.fit(&glm_data(x, y, weights, offset, start)?)
    }
}

pub trait NegBinParts {
    fn fit4(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> Result<NegBinFit, Error>;
    fn fit_from(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<NegBinFit, Error>;
}

impl NegBinParts for NegBin {
    fn fit4(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> Result<NegBinFit, Error> {
        self.fit(&glm_data(x, y, weights, offset, None)?)
    }
    fn fit_from(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<NegBinFit, Error> {
        self.fit(&glm_data(x, y, weights, offset, start)?)
    }
}

pub trait ElasticNetParts {
    fn fit6(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        penalty_factor: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<ElasticNetFit, Error>;
}

impl ElasticNetParts for ElasticNet {
    fn fit6(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        penalty_factor: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<ElasticNetFit, Error> {
        let model = ElasticNet {
            penalty_factor: penalty_factor.map(<[f64]>::to_vec),
            ..self.clone()
        };
        model.fit(&glm_data(x, y, weights, offset, start)?)
    }
}

pub trait MultinomParts {
    fn fit4(
        &self,
        x: MatRef<'_, f64>,
        y: &[usize],
        classes: usize,
        weights: Option<&[f64]>,
    ) -> Result<MultinomFit, Error>;
}

impl MultinomParts for Multinom {
    fn fit4(
        &self,
        x: MatRef<'_, f64>,
        y: &[usize],
        classes: usize,
        weights: Option<&[f64]>,
    ) -> Result<MultinomFit, Error> {
        let mut builder = MultinomData::builder(x, y, classes);
        if let Some(w) = weights {
            builder = builder.weights(w);
        }
        self.fit(&builder.build()?)
    }
}

pub trait GeeParts {
    fn fit6(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        cluster: &[u32],
        wave: Option<&[u32]>,
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> Result<GeeFit, Error>;
}

impl GeeParts for Gee {
    fn fit6(
        &self,
        x: MatRef<'_, f64>,
        y: &[f64],
        cluster: &[u32],
        wave: Option<&[u32]>,
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> Result<GeeFit, Error> {
        let mut builder = GeeData::builder(x, y, cluster);
        if let Some(w) = wave {
            builder = builder.waves(w);
        }
        if let Some(w) = weights {
            builder = builder.weights(w);
        }
        if let Some(o) = offset {
            builder = builder.offset(o);
        }
        self.fit(&builder.build()?)
    }
}

/// The ordinal view of the parts.
pub fn ordinal_data<'a>(
    x: MatRef<'a, f64>,
    nominal: Option<MatRef<'a, f64>>,
    y: &'a [u32],
    weights: Option<&'a [f64]>,
    offset: Option<&'a [f64]>,
    cluster: Option<&'a [u32]>,
) -> Result<OrdinalData<'a>, Error> {
    let mut builder = OrdinalData::builder(x, y);
    if let Some(m) = nominal {
        builder = builder.nominal(m);
    }
    if let Some(w) = weights {
        builder = builder.weights(w);
    }
    if let Some(o) = offset {
        builder = builder.offset(o);
    }
    if let Some(c) = cluster {
        builder = builder.clusters(c);
    }
    builder.build()
}

pub trait OrdinalParts {
    fn fit5(
        &self,
        x: MatRef<'_, f64>,
        y: &[u32],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        cluster: Option<&[u32]>,
    ) -> Result<OrdinalFit, Error>;
    fn fit_partial(
        &self,
        x: MatRef<'_, f64>,
        nominal: MatRef<'_, f64>,
        y: &[u32],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        cluster: Option<&[u32]>,
    ) -> Result<OrdinalFit, Error>;
}

impl OrdinalParts for Ordinal {
    fn fit5(
        &self,
        x: MatRef<'_, f64>,
        y: &[u32],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        cluster: Option<&[u32]>,
    ) -> Result<OrdinalFit, Error> {
        self.fit(&ordinal_data(x, None, y, weights, offset, cluster)?)
    }
    fn fit_partial(
        &self,
        x: MatRef<'_, f64>,
        nominal: MatRef<'_, f64>,
        y: &[u32],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        cluster: Option<&[u32]>,
    ) -> Result<OrdinalFit, Error> {
        self.fit(&ordinal_data(
            x,
            Some(nominal),
            y,
            weights,
            offset,
            cluster,
        )?)
    }
}
