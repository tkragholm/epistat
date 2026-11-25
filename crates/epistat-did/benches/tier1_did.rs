//! Wall time of the group-time average treatment effects on a panel, set
//! against `did::att_gt` (`bench/tier1/bench.R`). The driver and the
//! environment are described in `bench/tier1/common.rs`.
//!
//! The data (`did`) are a panel of six periods, three treated cohorts (first
//! treated in period 3, 4 and 5) and a never-treated group, with one
//! covariate. `N` is the number of rows. Method: `att_gt` (the doubly robust
//! estimator, never-treated comparison group, varying base period, analytic
//! standard errors).
//!
//! The data (`twfe`) are `N / 10` units observed for ten consecutive periods
//! out of fifteen, starting at a period drawn for each unit, first treated in
//! period 5, 8 or 11 or never, with effects that grow with the time since
//! treatment. Method: `twfe_es`, the event study
//! `feols(y ~ i(rel, ref = c(-1, -1000)) | id + t, cluster = ~id)` with the
//! relative periods binned at -5 and 5, including the influence functions of
//! the coefficients by unit.
//!
//! The data `dcdh` are a panel of eight periods and a binary treatment that
//! goes on and off: the baseline is 1 for 40% of the groups, the first change
//! comes in a random period or never, and after it the treatment flips with
//! probability 0.3 each period. Method: `dcdh`, `estimate_dcdh` with three
//! effects and two placebos, set against `DIDmultiplegtDYN::did_multiplegt_dyn`
//! 2.2.0 (the last release that runs without polars). On the same data,
//! `dcdh_trends_lin`, `dcdh_same_switchers` and `dcdh_less_conservative` add
//! those options, and `dcdh_bootstrap` takes 50 bootstrap replications.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a benchmark program: sizes, codes and days are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_did::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, DcdhConfig, DcdhObservation, DrDidConfig,
    StaggeredPanel, TwfeConfig, TwfeSpec, estimate_att_gt_dr_panel, estimate_dcdh,
    estimate_dcdh_bootstrap, estimate_twfe,
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

fn generate_twfe(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("twfe", n), "id,t,g,y");
    let period_effect: Vec<f64> = (0..15).map(|_| rng.normal()).collect();
    for id in 0..n / PERIODS_TWFE {
        let u = rng.normal();
        let start = rng.below(6) as usize;
        let g = match rng.below(4) {
            0 => 5.0,
            1 => 8.0,
            2 => 11.0,
            _ => -1.0,
        };
        for s in 0..PERIODS_TWFE {
            let t = (start + s) as f64;
            let effect = if g > 0.0 && t >= g {
                1.0 + 0.2 * (t - g)
            } else {
                0.0
            };
            let y = u + period_effect[start + s] + effect + rng.normal();
            out.row(&[id as f64, t, g, y]);
        }
    }
}

const DCDH_PERIODS: usize = 8;

fn generate_dcdh(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut out = Writer::create(&common::data_path("dcdh", n), "id,time,d,y");
    for id in 0..n / DCDH_PERIODS {
        let u = rng.normal();
        let mut d = if rng.bernoulli(0.4) { 1.0 } else { 0.0 };
        // First change in period 2..=8, or never (two chances in nine).
        let first = 2 + rng.below(9) as usize;
        let mut previous = d;
        for time in 1..=DCDH_PERIODS {
            if time == first || (time > first && rng.bernoulli(0.3)) {
                d = 1.0 - d;
            }
            let t = time as f64;
            let y = u + 0.3 * t + d + 0.5 * previous + rng.normal();
            previous = d;
            out.row(&[id as f64, t, d, y]);
        }
    }
}

const PERIODS_TWFE: usize = 10;

fn twfe(n: usize) {
    let t = Table::read_sized(&common::data_path("twfe", n));
    let unit: Vec<u32> = t.col("id").iter().map(|&v| v as u32).collect();
    let time: Vec<i32> = t.col("t").iter().map(|&v| v as i32).collect();
    let first: Vec<Option<i32>> = t
        .col("g")
        .iter()
        .map(|&g| (g > 0.0).then_some(g as i32))
        .collect();
    let panel = StaggeredPanel {
        unit: &unit,
        time: &time,
        first_treated: &first,
        outcome: t.col("y"),
        weights: None,
        covariates: None,
    };
    let config = TwfeConfig {
        spec: TwfeSpec::EventStudy {
            reference: -1,
            bins: Some((-5, 5)),
        },
        ..TwfeConfig::event_study()
    };
    let (times, result) = common::timed(common::reps(), || {
        estimate_twfe(&panel, &config).expect("twfe")
    });
    let mut lines = Lines::default();
    for term in &result.terms {
        let r = term.relative_period.unwrap_or(0);
        lines.push_opt(&format!("coef_rel{r}"), term.estimate);
        lines.push_opt(&format!("se_rel{r}"), term.std_error);
    }
    common::finish("twfe_es", n, &times, &lines);
}

fn run_dcdh(n: usize, method: &str) {
    let t = Table::read(&common::data_path("dcdh", n));
    let rows: Vec<DcdhObservation> = (0..t.len())
        .map(|i| DcdhObservation {
            group: t.col("id")[i] as i64,
            time: t.col("time")[i] as i64,
            outcome: Some(t.col("y")[i]),
            treatment: Some(t.col("d")[i]),
            weight: 1.0,
            controls: Vec::new(),
            stratum: 0,
            cluster: None,
        })
        .collect();
    let config = DcdhConfig {
        effects: 3,
        placebo: 2,
        trends_lin: method == "dcdh_trends_lin",
        same_switchers: method == "dcdh_same_switchers",
        less_conservative_se: method == "dcdh_less_conservative",
        ..DcdhConfig::default()
    };
    let mut lines = Lines::default();
    if method == "dcdh_bootstrap" {
        let (times, r) = common::timed(common::reps(), || {
            estimate_dcdh_bootstrap(&rows, &config, 50, 1).expect("dcdh")
        });
        for e in &r.effects {
            lines.push(
                &format!("est_effect{}", e.relative_period),
                e.estimate.unwrap_or(f64::NAN),
            );
            lines.push(
                &format!("bse_effect{}", e.relative_period),
                e.se.unwrap_or(f64::NAN),
            );
        }
        common::finish(method, n, &times, &lines);
        return;
    }
    let (times, r) = common::timed(common::reps(), || {
        estimate_dcdh(&rows, &config).expect("dcdh")
    });
    let rows = r
        .effects
        .iter()
        .map(|e| (format!("effect{}", e.relative_period), e))
        .chain(
            r.placebos
                .iter()
                .map(|e| (format!("placebo{}", -e.relative_period), e)),
        )
        .chain(std::iter::once((
            "average".to_string(),
            &r.average_total_effect,
        )));
    for (name, e) in rows {
        if e.estimate.is_none() && name == "average" {
            continue;
        }
        lines.push(&format!("est_{name}"), e.estimate.unwrap_or(f64::NAN));
        lines.push(&format!("se_{name}"), e.se.unwrap_or(f64::NAN));
    }
    common::finish(method, n, &times, &lines);
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_did" {
        generate(n);
        return;
    }
    if method == "gen_twfe" {
        generate_twfe(n);
        return;
    }
    if method == "twfe_es" {
        twfe(n);
        return;
    }
    if method == "gen_dcdh" {
        generate_dcdh(n);
        return;
    }
    if method.starts_with("dcdh") {
        run_dcdh(n, &method);
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
