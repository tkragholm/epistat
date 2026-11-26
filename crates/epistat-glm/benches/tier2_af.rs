//! Wall time of the attributable fraction of a logistic model, cross-sectional
//! and case-control, with a confounder, an interaction and clusters of four
//! rows. The driver and the environment are those of `bench/tier1/common.rs`,
//! run by `bench/run.sh`. The data kind `afd` is written by `METHOD=gen_afd`
//! from a fixed seed and read by both sides (`bench/tier2/af_bench.R` is the R
//! side, which runs `AFglm` of the package AF from its source, see there); the
//! fit is not timed.
//!
//! Methods: `af_cross` and `af_case_control`, `N` rows.

#![allow(
    clippy::cast_precision_loss,
    reason = "a benchmark program: row numbers are small whole numbers"
)]

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_glm::attributable::{AfOptions, Design, attributable_logistic};
use epistat_glm::{Family, Glm, GlmData, Link};
use epistat_inference::Clusters;
use epistat_linalg::Mat;

const SEED: u64 = 20_261_208;

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("afd", n), "cluster,e,z,y");
    for i in 0..n {
        let z = rng.normal();
        let e = f64::from(u8::from(rng.bernoulli(1.0 / (1.0 + (-0.4 * z).exp()))));
        let eta = -1.0 + 0.8 * e + 0.5 * z + 0.3 * e * z;
        let y = f64::from(u8::from(rng.bernoulli(1.0 / (1.0 + (-eta).exp()))));
        out.row(&[(i / 4) as f64, e, z, y]);
    }
}

fn main() {
    let (method, n) = (common::method(), common::size());
    if method == "gen_afd" {
        return generate(n);
    }
    let design = match method.as_str() {
        "af_cross" => Design::CrossSectional,
        "af_case_control" => Design::CaseControl,
        other => panic!("unknown method {other}"),
    };
    let t = Table::read(&common::data_path("afd", n));
    let (e, z, y) = (t.col("e"), t.col("z"), t.col("y"));
    let cluster: Vec<u32> = t.ints("cluster");
    let x = Mat::from_fn(n, 4, |i, j| [1.0, e[i], z[i], e[i] * z[i]][j]);
    let x0 = Mat::from_fn(n, 4, |i, j| if j == 1 || j == 3 { 0.0 } else { x[(i, j)] });
    let data = GlmData::builder(x.as_ref(), y).build().expect("data");
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(&data)
        .expect("fit");
    let opts = AfOptions {
        design,
        ..AfOptions::default()
    };
    let (times, result) = common::timed(common::reps(), || {
        let clusters = Clusters::from_labels(&cluster);
        attributable_logistic(&fit, &data, x0.as_ref(), Some(&clusters), &opts).expect("af")
    });
    let af = result.af.expect("a fraction");
    let mut lines = Lines::default();
    lines.push_opt("af", Some(af.estimate));
    lines.push_opt("se", af.se);
    lines.push_opt("lower", af.lower());
    lines.push_opt("upper", af.upper());
    common::finish(&method, n, &times, &lines);
}
