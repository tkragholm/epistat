//! Times `match_cardinality` on a CSV written by
//! `tests/data/cardinality_timing.R`: a `treat` column followed by covariates.
//!
//! ```text
//! cargo run --release -p epistat-matching --example cardinality_timing -- data.csv [seconds]
//! ```
//!
//! Prints the units, the selected units, the status of the search, the bound and
//! the wall time. The optional second argument is the time limit in seconds.

use epistat_matching::{CardinalityLimits, CardinalityRequest, match_cardinality};
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a CSV path");
    let seconds: Option<f64> = args.next().map(|s| s.parse().expect("seconds"));
    let text = std::fs::read_to_string(&path).expect("readable file");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("a header").split(',').collect();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); header.len()];
    for line in lines {
        for (c, field) in line.split(',').enumerate() {
            columns[c].push(field.parse().expect("a number"));
        }
    }
    let treated: Vec<bool> = columns[0].iter().map(|&t| t > 0.5).collect();
    let covariates: Vec<Vec<f64>> = columns[1..].to_vec();
    let mut request = CardinalityRequest::new(&treated, &covariates);
    request.limits = CardinalityLimits {
        time: seconds.map(Duration::from_secs_f64),
        nodes: None,
    };
    let start = Instant::now();
    let outcome = match_cardinality(&request).expect("a solution");
    let elapsed = start.elapsed();
    let report = &outcome.strata[0];
    println!(
        "units {} selected {} proven {} bound {} nodes {} pivots {} seconds {:.2}",
        treated.len(),
        report.selected_treated + report.selected_control,
        report.proven_optimal,
        report.bound,
        report.nodes,
        report.pivots,
        elapsed.as_secs_f64()
    );
}
