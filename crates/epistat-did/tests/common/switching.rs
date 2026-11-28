//! Long rows with labels, as the R references and the tests hold them, to
//! the sorted columns of a [`SwitchingPanel`].

#![allow(dead_code, reason = "each test uses part of the helper")]
#![allow(clippy::float_cmp, reason = "a weight of exactly 1 is the default")]

use epistat_core::{Codes, Panel};
use epistat_did::SwitchingPanel;
use epistat_linalg::Mat;

/// One input row with the caller's labels.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub group: i64,
    pub time: i64,
    pub outcome: Option<f64>,
    pub treatment: Option<f64>,
    pub weight: f64,
    pub controls: Vec<f64>,
    pub stratum: i64,
    pub cluster: Option<i64>,
}

impl Row {
    /// A row of weight 1 with no controls, stratum or cluster.
    pub fn new(group: i64, time: i64, treatment: f64, outcome: f64) -> Self {
        Self {
            group,
            time,
            outcome: Some(outcome),
            treatment: Some(treatment),
            weight: 1.0,
            controls: Vec::new(),
            stratum: 0,
            cluster: None,
        }
    }
}

/// The columns of a panel, sorted by unit and period, with the codes of the
/// labels.
pub struct Columns {
    pub units: Codes<i64>,
    pub clusters_of: Option<Codes<i64>>,
    pub subgroups_of: Option<Codes<i64>>,
    /// For each sorted row, its position in the input.
    pub order: Vec<usize>,
    unit: Vec<u32>,
    period: Vec<i64>,
    outcome: Vec<Option<f64>>,
    treatment: Vec<Option<f64>>,
    weights: Option<Vec<f64>>,
    controls: Option<Mat<f64>>,
    strata: Option<Vec<u32>>,
    clusters: Option<Vec<u32>>,
    subgroups: Option<Vec<u32>>,
    het: Option<Mat<f64>>,
}

impl Columns {
    /// The columns of `rows`; weights, controls, strata and clusters are
    /// kept when some row carries them.
    pub fn new(rows: &[Row]) -> Self {
        let group: Vec<i64> = rows.iter().map(|r| r.group).collect();
        let time: Vec<i64> = rows.iter().map(|r| r.time).collect();
        let units = Codes::from_labels(&group).expect("labels");
        let order: Vec<usize> = Panel::sort_order(units.codes(), &time)
            .expect("order")
            .into_iter()
            .map(|i| usize::try_from(i).expect("row"))
            .collect();
        let pick = |i: usize| &rows[order[i]];
        let n = rows.len();
        let k = rows.first().map_or(0, |r| r.controls.len());
        let codes = |labels: Vec<i64>| {
            let codes = Codes::from_labels(&labels).expect("labels");
            let sorted = order.iter().map(|&i| codes.codes()[i]).collect();
            (codes, sorted)
        };
        let clusters = rows
            .iter()
            .any(|r| r.cluster.is_some())
            .then(|| codes(rows.iter().map(|r| r.cluster.expect("cluster")).collect()));
        let strata = rows
            .iter()
            .any(|r| r.stratum != 0)
            .then(|| codes(rows.iter().map(|r| r.stratum).collect()).1);
        let (clusters_of, clusters) = clusters.map_or((None, None), |(c, s)| (Some(c), Some(s)));
        Self {
            unit: order.iter().map(|&i| units.codes()[i]).collect(),
            period: (0..n).map(|i| pick(i).time).collect(),
            outcome: (0..n).map(|i| pick(i).outcome).collect(),
            treatment: (0..n).map(|i| pick(i).treatment).collect(),
            weights: rows
                .iter()
                .any(|r| r.weight != 1.0)
                .then(|| (0..n).map(|i| pick(i).weight).collect()),
            controls: (k > 0).then(|| Mat::from_fn(n, k, |i, j| pick(i).controls[j])),
            strata,
            clusters,
            clusters_of,
            subgroups_of: None,
            subgroups: None,
            het: None,
            units,
            order,
        }
    }

    /// The subgroups of `by`, one label per input row.
    pub fn with_subgroups(mut self, labels: &[i64]) -> Self {
        let codes = Codes::from_labels(labels).expect("labels");
        self.subgroups = Some(self.order.iter().map(|&i| codes.codes()[i]).collect());
        self.subgroups_of = Some(codes);
        self
    }

    /// The heterogeneity covariates, one row per input row.
    pub fn with_het(mut self, x: &[Vec<f64>]) -> Self {
        let k = x.first().map_or(0, Vec::len);
        self.het = Some(Mat::from_fn(self.order.len(), k, |i, j| {
            x[self.order[i]][j]
        }));
        self
    }

    /// The panel.
    pub fn panel(&self) -> SwitchingPanel<'_> {
        let mut b =
            SwitchingPanel::builder(&self.unit, &self.period, &self.outcome, &self.treatment);
        if let Some(w) = &self.weights {
            b = b.weights(epistat_core::CaseWeights::new(w).expect("weights"));
        }
        if let Some(x) = &self.controls {
            b = b.controls(x.as_ref());
        }
        if let Some(s) = &self.strata {
            b = b.strata(s);
        }
        if let Some(c) = &self.clusters {
            b = b.clusters(c);
        }
        if let Some(s) = &self.subgroups {
            b = b.subgroups(s);
        }
        if let Some(h) = &self.het {
            b = b.het_covariates(h.as_ref());
        }
        b.build().expect("panel")
    }

    /// The labels of unit codes.
    pub fn labels(&self, codes: &[u32]) -> Vec<i64> {
        codes
            .iter()
            .map(|&c| *self.units.label(c).expect("code"))
            .collect()
    }

    /// The labels of cluster codes.
    pub fn cluster_labels(&self, codes: &[u32]) -> Vec<i64> {
        let of = self.clusters_of.as_ref().expect("clusters");
        codes.iter().map(|&c| *of.label(c).expect("code")).collect()
    }
}
