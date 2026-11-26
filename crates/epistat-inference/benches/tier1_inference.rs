//! Wall time of the p-value adjustments and the pooling rules, set against
//! `stats::p.adjust`, `mice::pool` and `mitml::testModels`
//! (`bench/tier1/bench.R`). The driver and the environment are described in
//! `bench/tier1/common.rs`.
//!
//! Methods: `padj_holm`, `padj_hochberg`, `padj_hommel`, `padj_bh` and
//! `padj_by` adjust one vector of p-values, and `pool` pools 50 imputed fits
//! of three tested coefficients: the scalar rule for one coefficient, the
//! vector rule, and the D1 and D2 tests. The fits come from R (`gen_pool`, run
//! by R), so the pooling is timed from the coefficients and covariances.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_inference::multiplicity::{Adjustment, adjust};
use epistat_inference::pooling::{d1, d2, rubin, rubin_vector};

const SEED: u64 = 20_261_204;
const K: usize = 3;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("pv", n), "p");
    for _ in 0..n {
        let p = if rng.bernoulli(0.1) {
            rng.uniform().powi(6)
        } else {
            rng.uniform()
        };
        out.row(&[p]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_pv" {
        generate(n);
        return;
    }
    let mut lines = Lines::default();
    let times = if let Some(name) = method.strip_prefix("padj_") {
        let rule = match name {
            "holm" => Adjustment::Holm,
            "hochberg" => Adjustment::Hochberg,
            "hommel" => Adjustment::Hommel,
            "bh" => Adjustment::BenjaminiHochberg,
            "by" => Adjustment::BenjaminiYekutieli,
            other => panic!("unknown adjustment {other}"),
        };
        let p = Table::read(&common::data_path("pv", n)).col("p").to_vec();
        let (times, adjusted) =
            common::timed(common::reps(), || adjust(&p, rule, None).expect("adjust"));
        lines.push("sum", adjusted.iter().sum());
        lines.push(
            "min",
            adjusted.iter().copied().fold(f64::INFINITY, f64::min),
        );
        for i in [0, 1, 2, n / 2, n - 1] {
            lines.push(&format!("at{i}"), adjusted[i]);
        }
        times
    } else if method == "pool" {
        let t = Table::read(&common::data_path("pool", n));
        let m = t.len();
        let q: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..K).map(|j| t.col(&format!("q{j}"))[i]).collect())
            .collect();
        let u: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..K * K).map(|j| t.col(&format!("u{j}"))[i]).collect())
            .collect();
        let qs = epistat_linalg::Mat::from_fn(m, K, |i, j| q[i][j]);
        let um: Vec<epistat_linalg::Mat<f64>> = u
            .iter()
            .map(|u| epistat_linalg::Mat::from_fn(K, K, |i, j| u[i * K + j]))
            .collect();
        let us: Vec<epistat_linalg::MatRef<'_, f64>> =
            um.iter().map(epistat_linalg::Mat::as_ref).collect();
        let scalar_q = t.col("bq").to_vec();
        let scalar_u = t.col("bu").to_vec();
        let wald = t.col("wald").to_vec();
        let dfcom = t.col("dfcom")[0];
        let (times, r) = common::timed(common::reps().max(50), || {
            (
                rubin(&scalar_q, &scalar_u, Some(dfcom)).expect("rubin"),
                rubin_vector(qs.as_ref(), &us).expect("vector"),
                d1(qs.as_ref(), &us, None, Some(dfcom)).expect("d1"),
                d2(&wald, K).expect("d2"),
            )
        });
        lines.push("scalar_estimate", r.0.estimate);
        lines.push("scalar_total", r.0.total);
        lines.push_opt("scalar_df", r.0.df);
        lines.push_all("vector_estimate", &r.1.estimate);
        lines.push("d1_statistic", r.2.statistic);
        lines.push("d1_df2", r.2.df2);
        lines.push_opt("d1_p", r.2.p_value);
        lines.push("d2_statistic", r.3.statistic);
        lines.push("d2_df2", r.3.df2);
        times
    } else {
        panic!("unknown method {method}")
    };
    common::finish(&method, n, &times, &lines);
}
