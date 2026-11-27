//! Wall time of the case-cohort analysis of `epistat_survival::casecohort`
//! against `survival::cch`, and of the nested case-control weights and weighted
//! fit of `epistat_survival::ncc` against `multipleNCC` (`bench/tier3/designs.R`).
//! The driver environment is described in `bench/tier1/common.rs`: one process
//! times one method on one size.
//!
//! Run by hand, with `SCRATCH` outside the repository, and not above 10^5 rows
//! on the Mac:
//!
//! ```text
//! N=100000 METHOD=gen_cc SCRATCH=/x cargo bench -p epistat-survival --bench tier3_designs
//! N=100000 METHOD=cch_prentice SCRATCH=/x REPS=3 cargo bench -p epistat-survival --bench tier3_designs
//! ```
//!
//! `N` is the size of the cohort. The case-cohort data `cc` hold the cases, about
//! 5% of the cohort, and a random subcohort of 5%, with time, status, subcohort
//! indicator, stratum and two covariates, and `cc_sizes` the cohort size of each
//! of three strata. The nested case-control data `ncc` hold the whole cohort
//! with exit time, case indicator, sampled indicator (three controls per case,
//! drawn from the others still at risk) and two covariates.
//!
//! * `cch_prentice`, `cch_selfprentice`, `cch_linying`: the unstratified
//!   methods. The numbers compared are the coefficients and the standard errors.
//! * `cch_borgan1`, `cch_borgan2`: Borgan's methods on the three strata.
//! * `ncc_probabilities`: the inclusion probabilities against
//!   `multipleNCC::KMprob`. The numbers compared are the sum of the
//!   probabilities and the probabilities of the first ten sampled controls.
//! * `ncc_fit`: the probabilities, the weights and the weighted Cox fit against
//!   `wpl`. The numbers compared are the coefficients and the robust standard
//!   errors. The R side has no direct counterpart of the absolute risk, so it
//!   is not part of this method.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::casecohort::{CaseCohortMethod, CaseCohortOptions, fit as cch_fit};
use epistat_survival::cox::CoxOptions;
use epistat_survival::ncc::{NccDesign, fit_weighted, inclusion_probabilities, weights};

const SEED: u64 = 20_261_209;
const CONTROLS: usize = 3;

/// The case-cohort sample of a cohort of `n`: cases and a 5% subcohort.
fn generate_cc(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("cc", n), "time,status,sub,stratum,x1,x2");
    let mut sizes = [0.0_f64; 3];
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.bernoulli(0.4)));
        let stratum = usize::try_from(rng.below(3)).expect("a small code");
        sizes[stratum] += 1.0;
        let t = (-rng.uniform().ln()) / (0.02 * (0.5 * x1 - 0.4 * x2).exp());
        let c = (-rng.uniform().ln()) / 0.03;
        let time = t.min(c).min(10.0);
        let case = t <= c.min(10.0);
        let sub = rng.bernoulli(0.05);
        if case || sub {
            out.row(&[
                time,
                f64::from(u8::from(case)),
                f64::from(u8::from(sub)),
                stratum as f64,
                x1,
                x2,
            ]);
        }
    }
    let mut size_file = Writer::create(&common::data_path("cc_sizes", n), "stratum,size");
    for (j, s) in sizes.iter().enumerate() {
        size_file.row(&[j as f64, *s]);
    }
}

/// The cohort with three controls per case drawn from those still at risk after
/// the case time, by rank in the order of exit times.
fn generate_ncc(n: usize) {
    let mut rng = Rng(SEED + 1);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut exit = Vec::with_capacity(n);
    let mut case = Vec::with_capacity(n);
    for _ in 0..n {
        let a = rng.normal();
        let b = f64::from(u8::from(rng.bernoulli(0.4)));
        let t = (-rng.uniform().ln()) / (0.02 * (0.5 * a - 0.4 * b).exp());
        let c = (-rng.uniform().ln()) / 0.03;
        exit.push(t.min(c).min(10.0));
        case.push(t <= c.min(10.0));
        x1.push(a);
        x2.push(b);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| exit[a].total_cmp(&exit[b]));
    let mut sampled = vec![false; n];
    for r in 0..n {
        let j = order[r];
        if !case[j] {
            continue;
        }
        sampled[j] = true;
        let after = n - r - 1;
        assert!(after > CONTROLS, "too few people at risk after a case");
        let mut chosen = Vec::new();
        while chosen.len() < CONTROLS {
            let k = r + 1 + usize::try_from(rng.below(after as u64)).expect("an index");
            if !chosen.contains(&k) {
                chosen.push(k);
                sampled[order[k]] = true;
            }
        }
    }
    let mut out = Writer::create(&common::data_path("ncc", n), "exit,case,sampled,x1,x2");
    for i in 0..n {
        out.row(&[
            exit[i],
            f64::from(u8::from(case[i])),
            f64::from(u8::from(sampled[i])),
            x1[i],
            x2[i],
        ]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    match method.as_str() {
        "gen_cc" => return generate_cc(n),
        "gen_ncc" => return generate_ncc(n),
        _ => {}
    }
    let mut lines = Lines::default();
    let times = if method.starts_with("cch_") {
        let t = Table::read(&common::data_path("cc", n));
        let sizes = Table::read(&common::data_path("cc_sizes", n));
        let status: Vec<i32> = t.ints("status");
        let sub = t.flags("sub");
        let stratum: Vec<u32> = t.ints("stratum");
        let (x1, x2) = (t.col("x1"), t.col("x2"));
        let x = Mat::from_fn(t.len(), 2, |i, j| [x1[i], x2[i]][j]);
        let data = SurvData::builder(t.col("time"), &status)
            .build()
            .expect("data");
        let (m, stratified) = match method.as_str() {
            "cch_prentice" => (CaseCohortMethod::Prentice, false),
            "cch_selfprentice" => (CaseCohortMethod::SelfPrentice, false),
            "cch_linying" => (CaseCohortMethod::LinYing, false),
            "cch_borgan1" => (CaseCohortMethod::BorganI, true),
            "cch_borgan2" => (CaseCohortMethod::BorganII, true),
            other => panic!("unknown method {other}"),
        };
        let cohort = vec![n as f64];
        let by_stratum = sizes.col("size").to_vec();
        let cohort_size: &[f64] = if stratified { &by_stratum } else { &cohort };
        let strata = stratified.then_some(stratum.as_slice());
        let (times, f) = common::timed(common::reps(), || {
            cch_fit(
                &data,
                x.as_ref(),
                &sub,
                strata,
                &CaseCohortOptions::new(m, cohort_size),
            )
            .expect("case-cohort fit")
        });
        for j in 0..2 {
            lines.push_opt(&format!("coef{j}"), f.coefficients[j]);
            lines.push_opt(&format!("se{j}"), f.covariance.get(j, j).map(f64::sqrt));
        }
        times
    } else {
        let t = Table::read(&common::data_path("ncc", n));
        let exit = t.col("exit");
        let case = t.flags("case");
        let sampled = t.flags("sampled");
        let design = NccDesign {
            entry: None,
            exit,
            case: &case,
            controls_per_case: CONTROLS,
            stratum: None,
        };
        match method.as_str() {
            "ncc_probabilities" => {
                let (times, inc) = common::timed(common::reps(), || {
                    inclusion_probabilities(&design).expect("probabilities")
                });
                lines.push("sum_p", inc.probability.iter().sum());
                let first: Vec<f64> = (0..n)
                    .filter(|&i| sampled[i] && !case[i])
                    .take(10)
                    .map(|i| inc.probability[i])
                    .collect();
                lines.push_all("p", &first);
                times
            }
            "ncc_fit" => {
                let rows: Vec<usize> = (0..n).filter(|&i| sampled[i]).collect();
                let stop: Vec<f64> = rows.iter().map(|&i| exit[i]).collect();
                let status: Vec<i32> = rows.iter().map(|&i| i32::from(case[i])).collect();
                let (x1, x2) = (t.col("x1"), t.col("x2"));
                let x = Mat::from_fn(rows.len(), 2, |i, j| [x1[rows[i]], x2[rows[i]]][j]);
                let data = SurvData::builder(&stop, &status).build().expect("data");
                let (times, model) = common::timed(common::reps(), || {
                    let inc = inclusion_probabilities(&design).expect("probabilities");
                    let w_all = weights(&inc, &sampled, &case).expect("weights");
                    let w: Vec<f64> = rows.iter().map(|&i| w_all[i]).collect();
                    fit_weighted(&data, x.as_ref(), &w, &CoxOptions::default()).expect("fit")
                });
                let robust = model.robust_covariance.expect("robust covariance");
                for (j, &coefficient) in model.coefficients.iter().enumerate().take(2) {
                    lines.push_opt(&format!("coef{j}"), coefficient);
                    lines.push_opt(&format!("se{j}"), robust.get(j, j).map(f64::sqrt));
                }
                times
            }
            other => panic!("unknown method {other}"),
        }
    };
    common::finish(&method, n, &times, &lines);
}
