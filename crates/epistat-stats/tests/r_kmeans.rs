//! k-means partitions checked against R 4.6's `stats::kmeans()`.
//!
//! The reference is `tests/data/kmeans.json`, written by
//! `tests/data/kmeans.R`: twenty simulated data sets of 240 points in two to
//! four dimensions, clustered from the same starting rows by Lloyd's
//! algorithm and by Hartigan and Wong's.
//!
//! Both algorithms are deterministic given their start, so the partition
//! must be R's and the centres and within sums of squares agree to rounding.
//! The centres are means of up to 240 values, so the bound is 1e-12 relative.
//! The Hartigan-Wong runs are compared the same way, on data whose clusters
//! overlap, where a different order of transfers would end in a different
//! local minimum.

use epistat_stats::cluster::{KMeansAlgorithm, kmeans};
use epistat_testkit::{Tolerance, assert_close, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
}

#[derive(Deserialize)]
struct Run {
    algorithm: String,
    d: usize,
    points: Vec<f64>,
    start: Vec<f64>,
    cluster: Vec<usize>,
    centres: Vec<f64>,
    withinss: Vec<f64>,
    size: Vec<usize>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/kmeans.json"
    ))
}

#[test]
fn partitions_and_centres_match_r() {
    let r = reference();
    assert!(r.versions.r.starts_with("R version 4.6"));
    let close = Tolerance::exact().rel(1e-12).abs(1e-12);
    let (mut lloyd, mut hw) = (0, 0);
    for run in &r.runs {
        let algorithm = if run.algorithm == "Lloyd" {
            lloyd += 1;
            KMeansAlgorithm::Lloyd
        } else {
            hw += 1;
            KMeansAlgorithm::HartiganWong
        };
        let fit = kmeans(&run.points, run.d, &run.start, algorithm, 1000).unwrap();
        assert!(fit.converged);
        assert_eq!(fit.cluster, run.cluster, "{} partition", run.algorithm);
        assert_eq!(fit.size, run.size);
        assert_close(&fit.centres, &run.centres, close);
        assert_close(&fit.within_ss, &run.withinss, close);
    }
    assert!(
        lloyd >= 15 && hw >= 15,
        "{lloyd} Lloyd and {hw} Hartigan-Wong runs"
    );
}
