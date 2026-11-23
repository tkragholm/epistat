//! Wall time of the group-time average treatment effects on a panel, set
//! against `did::att_gt` (`bench/tier1/bench.R`). The driver and the
//! environment are described in `bench/tier1/common.rs`.
//!
//! The data (`did`) are a panel of six periods, three treated cohorts (first
//! treated in period 3, 4 and 5) and a never-treated group, with one
//! covariate. `N` is the number of rows. Method: `att_gt` (the doubly robust
//! estimator, never-treated comparison group, varying base period, analytic
//! standard errors).

#![allow(
    clippy::cast_possible_truncation,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_did::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, DrDidConfig, estimate_att_gt_dr_panel,
};

const SEED: u64 = 20_261_208;
const PERIODS: usize = 6;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("did", n), "id,time,g,x1,y");
    for id in 0..n / PERIODS {
        let u = rng.normal();
        let x1 = rng.normal();
        let g = match rng.below(5) {
            0 => 3.0,
            1 => 4.0,
            2 => 5.0,
            _ => 0.0,
        };
        // The cohort depends on the covariate through a shift of the outcome
        // trend, so that the covariate matters for the comparison.
        let trend = 0.3 + 0.1 * x1;
        for time in 1..=PERIODS {
            let t = time as f64;
            let effect = if g > 0.0 && t >= g {
                1.0 + 0.2 * (t - g)
            } else {
                0.0
            };
            let y = u + 0.5 * x1 + trend * t + effect + 0.5 * rng.normal();
            out.row(&[id as f64, t, g, x1, y]);
        }
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_did" {
        generate(n);
        return;
    }
    assert_eq!(method, "att_gt", "unknown method {method}");
    let t = Table::read(&common::data_path("did", n));
    let rows: Vec<AttGtDrObservation> = (0..t.len())
        .map(|i| {
            let g = t.col("g")[i];
            AttGtDrObservation {
                unit_id: Some(t.col("id")[i] as i64),
                first_treated_time: (g > 0.0).then_some(g as i32),
                time: t.col("time")[i] as i32,
                outcome: t.col("y")[i],
                weight: 1.0,
                covariates: vec![t.col("x1")[i]],
                comparison_cohort: None,
            }
        })
        .collect();
    let config = AttGtDrConfig {
        att_gt: AttGtConfig::default(),
        drdid: DrDidConfig::default(),
    };
    let (times, estimates) = common::timed(common::reps(), || {
        estimate_att_gt_dr_panel(&rows, config).expect("att_gt")
    });
    let mut lines = Lines::default();
    for e in &estimates {
        lines.push(&format!("att_g{}_t{}", e.group, e.time), e.att);
        lines.push(&format!("se_g{}_t{}", e.group, e.time), e.se);
    }
    common::finish(&method, n, &times, &lines);
}
