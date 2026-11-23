//! Wall time of the restricted cubic spline and natural spline bases, set
//! against `Hmisc::rcspline.eval` and `splines::ns` on the same values
//! (`bench/tier1/bench.R`). The driver and the environment are described in
//! `bench/tier1/common.rs`.
//!
//! Methods: `rcs` (Harrell's five knots and the basis with the linear column)
//! and `ns` (a natural spline of 5 degrees of freedom). The time of each
//! includes finding the knots.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_linalg::Mat;
use epistat_linalg::basis::{
    NaturalSpline, RestrictedCubic, basis_matrix, harrell_knots, natural_spline_knots,
};

const SEED: u64 = 20_261_201;
const SHOWN_ROWS: [usize; 3] = [0, 1, 2];

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("spl", n), "x");
    for _ in 0..n {
        let x = (50.0 + 15.0 * rng.normal()) * 100.0;
        out.row(&[x.round() / 100.0]);
    }
}

fn describe(knots: &[f64], basis: &Mat<f64>) -> Lines {
    let mut lines = Lines::default();
    lines.push_all("knot", knots);
    for j in 0..basis.ncols() {
        let sum: f64 = (0..basis.nrows()).map(|i| basis[(i, j)]).sum();
        lines.push(&format!("colsum{j}"), sum);
        for &i in &SHOWN_ROWS {
            lines.push(&format!("row{i}_col{j}"), basis[(i, j)]);
        }
    }
    lines
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_spl" {
        generate(n);
        return;
    }
    let table = Table::read(&common::data_path("spl", n));
    let x = table.col("x");
    let (times, lines) = match method.as_str() {
        "rcs" => {
            let (times, (knots, basis)) = common::timed(common::reps(), || {
                let knots = harrell_knots(x, 5, 0.05).expect("knots");
                let spline = RestrictedCubic::new(&knots, true).expect("spline");
                let basis = basis_matrix(&spline, x);
                (knots, basis)
            });
            (times, describe(&knots, &basis))
        }
        "ns" => {
            let (lo, hi) = x
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            let (times, (knots, basis)) = common::timed(common::reps(), || {
                let knots = natural_spline_knots(x, 5, false, None).expect("knots");
                let spline = NaturalSpline::new(&knots, (lo, hi), false).expect("spline");
                let basis = basis_matrix(&spline, x);
                (knots, basis)
            });
            (times, describe(&knots, &basis))
        }
        other => panic!("unknown method {other}"),
    };
    common::finish(&method, n, &times, &lines);
}
