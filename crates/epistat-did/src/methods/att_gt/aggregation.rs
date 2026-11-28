//! The fixed-weight aggregations of ATT(g,t): [`AggteWeights::Equal`],
//! [`AggteWeights::TreatedCount`] and [`AggteWeights::TotalWeight`].
//!
//! The weights within a key are normalised to sum to 1 and taken as known,
//! so the influence function of an aggregate is the weighted sum of its
//! cells' and nothing more. It counts the covariance between cells that
//! share units, which `sum(w^2 se^2)` would not.
//!
//! [`AggteWeights::Equal`]: super::AggteWeights::Equal
//! [`AggteWeights::TreatedCount`]: super::AggteWeights::TreatedCount
//! [`AggteWeights::TotalWeight`]: super::AggteWeights::TotalWeight

use std::collections::BTreeMap;

use epistat_inference::Clusters;

use crate::inference::did_clusters::standard_error;
use crate::inference::standard_error_from_influence;
use crate::util::usize_to_f64;

use super::AggteWeights;
use super::aggte::{Cell, Point, combine_fixed};

/// The weight of a cell before normalisation within its key.
pub(super) fn raw_weight(cell: &Cell, weights: AggteWeights) -> f64 {
    match weights {
        AggteWeights::TreatedCount => usize_to_f64(cell.treated_n),
        AggteWeights::TotalWeight => cell.total_weight,
        _ => 1.0,
    }
}

/// The weighted mean of the cells by event time, every cell of each event
/// time with the weights taken as known.
///
/// The estimate and the influence function are built cell by cell in index
/// order with one rounding per multiply-add. The standard error is that of
/// the combined influence function: clustered as `did` clusters it when
/// `clusters` is given, and otherwise `sqrt(mean((psi - mean psi)^2) / n)`.
pub(super) fn by_event_time(
    cells: &[Cell],
    influence: &[&[f64]],
    weights: AggteWeights,
    clusters: Option<&Clusters>,
    confidence_level: f64,
) -> Vec<Point> {
    let n = influence.first().map_or(0, |column| column.len());
    let mut grouped = BTreeMap::<i64, Vec<usize>>::new();
    for (index, cell) in cells.iter().enumerate() {
        grouped.entry(cell.event_time).or_default().push(index);
    }

    let mut points = Vec::with_capacity(grouped.len());
    for (key, bucket) in grouped {
        let raw_total = bucket
            .iter()
            .map(|&c| raw_weight(&cells[c], weights))
            .sum::<f64>();
        let mut estimate = 0.0;
        let mut total_influence = vec![0.0; n];
        let mut total_weight = 0.0;
        for &c in &bucket {
            let normalized_weight = raw_weight(&cells[c], weights) / raw_total;
            estimate = normalized_weight.mul_add(cells[c].att, estimate);
            total_weight += cells[c].total_weight;
            for (dst, src) in total_influence.iter_mut().zip(influence[c]) {
                *dst = normalized_weight.mul_add(*src, *dst);
            }
        }
        let se = match clusters {
            Some(clusters) => standard_error(&total_influence, clusters),
            None => standard_error_from_influence(&total_influence),
        };
        points.push(Point::normal(
            key,
            (estimate, total_influence),
            se,
            confidence_level,
            bucket.len(),
            total_weight,
        ));
    }
    points
}

/// The weighted mean of the cells in each bucket, with the weights taken as
/// known, and its standard error from the combined influence function
/// clustered as `did` clusters it.
pub(super) fn by_key(
    buckets: BTreeMap<i64, Vec<usize>>,
    cells: &[Cell],
    influence: &[&[f64]],
    weights: AggteWeights,
    clusters: &Clusters,
    confidence_level: f64,
) -> Vec<Point> {
    let n_units = clusters.len();
    buckets
        .into_iter()
        .map(|(key, bucket)| {
            let raw = bucket
                .iter()
                .map(|&c| raw_weight(&cells[c], weights))
                .collect::<Vec<_>>();
            let raw_total = raw.iter().sum::<f64>();
            let normalised = raw.iter().map(|w| w / raw_total).collect::<Vec<_>>();
            let (att, combined) = combine_fixed(&bucket, &normalised, cells, influence, n_units);
            let se = standard_error(&combined, clusters);
            Point::normal(
                key,
                (att, combined),
                se,
                confidence_level,
                bucket.len(),
                bucket.iter().map(|&c| cells[c].total_weight).sum(),
            )
        })
        .collect()
}
