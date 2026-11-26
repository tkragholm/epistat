//! Wall time of the parametric g-formula, set against `gfoRmula::gformula` on
//! the same data (`bench/tier3/gformula.R`). The driver and the environment are
//! described in `bench/tier1/common.rs`; `bench/run.sh` runs
//! both sides. `N` is the number of persons, each followed for up to six
//! periods, so the data have about `4.5 N` rows.
//!
//! The data (`gf`) have baseline covariates `V` and `W`, a binary covariate
//! `L1`, a normal covariate `L2` and a binary treatment `A` in each period, an
//! event `Y` and a competing event `D`, as in `tests/data/gformula.R`. Methods,
//! each simulating every person once:
//!
//! * `gf_static`: the natural course, never and always treat;
//! * `gf_dynamic`: the natural course, treat when `L2 > 0.5`, and treat when
//!   `L1 = 1`;
//! * `gf_boot`: `gf_static` with 20 bootstrap replicates.
//!
//! The coefficients of the five models are the same quantity on both sides.
//! The risks at the last period and the bootstrap standard errors differ by the
//! Monte Carlo error and are written as `info_`.

#[path = "../../../bench/tier1/common.rs"]
mod common;

use common::{Lines, Rng, Table, Writer};
use epistat_causal::gformula::{
    Action, Competing, Covariate, CovariateKind, GFormulaOptions, History, Intervention, Outcome,
    PersonPeriods, Rule, Term, gformula,
};

const SEED: u64 = 20_261_210;
const PERIODS: usize = 6;

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn generate(n: usize) {
    let mut rng = Rng(SEED);
    let mut out = Writer::create(&common::data_path("gf", n), "id,t0,V,W,L1,L2,A,D,Y");
    let flag = |b: bool| f64::from(u8::from(b));
    for i in 0..n {
        let v = flag(rng.bernoulli(0.4));
        let w = rng.normal();
        let (mut l1p, mut l2p, mut ap, mut sum_a, mut sum_l2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for k in 0..PERIODS {
            let kf = k as f64;
            let (l1, l2, a) = if k == 0 {
                let l1 = flag(rng.bernoulli(expit(-0.5 + 0.5 * v)));
                let l2 = 0.3 * w + rng.normal();
                (
                    l1,
                    l2,
                    flag(rng.bernoulli(expit(-0.5 + 0.8 * l1 + 0.4 * l2))),
                )
            } else {
                let l1 = flag(rng.bernoulli(expit(
                    -1.0 + 1.2 * l1p - 0.8 * sum_a / kf + 0.3 * v + 0.2 * w,
                )));
                let l2 = 0.6 * l2p + 0.5 * l1 - 0.4 * ap + 0.2 * w + 0.8 * rng.normal();
                let a = flag(rng.bernoulli(expit(-1.0 + 2.0 * ap + 0.8 * l1 + 0.4 * l2 - 0.2 * v)));
                (l1, l2, a)
            };
            sum_a += a;
            sum_l2 += l2;
            let d = rng.bernoulli(expit(-4.0 + 0.5 * l1 + 0.3 * w + 0.1 * kf));
            let y = !d
                && rng.bernoulli(expit(
                    -3.0 + 0.8 * l1 + 0.4 * sum_l2 / (kf + 1.0) - 0.7 * a
                        + 0.4 * v
                        + 0.3 * a * v
                        + 0.1 * kf,
                ));
            let yv = if d { f64::NAN } else { flag(y) };
            out.row(&[i as f64, kf, v, w, l1, l2, a, flag(d), yv]);
            if y || d || rng.bernoulli(0.02) {
                break;
            }
            (l1p, l2p, ap) = (l1, l2, a);
        }
    }
}

fn covariates() -> Vec<Covariate> {
    use Term::{Baseline, Current, Intercept, Lag, LaggedCumulativeAverage};
    let model = |kind, terms| Covariate {
        kind,
        terms,
        restriction: None,
    };
    vec![
        model(
            CovariateKind::Binary,
            vec![
                Intercept,
                Lag(0, 1),
                LaggedCumulativeAverage(2, 1),
                Baseline(0),
                Baseline(1),
            ],
        ),
        model(
            CovariateKind::Normal,
            vec![Intercept, Lag(1, 1), Current(0), Lag(2, 1), Baseline(1)],
        ),
        model(
            CovariateKind::Binary,
            vec![Intercept, Lag(2, 1), Current(0), Current(1), Baseline(0)],
        ),
    ]
}

fn interventions(method: &str) -> Vec<Intervention> {
    if method == "gf_dynamic" {
        vec![
            Intervention {
                rules: vec![Rule {
                    variable: 2,
                    action: Action::Dynamic(Box::new(|h: &History<'_>, _| {
                        f64::from(u8::from(h.current(1) > 0.5))
                    })),
                    periods: None,
                }],
            },
            Intervention {
                rules: vec![Rule {
                    variable: 2,
                    action: Action::Dynamic(Box::new(|h: &History<'_>, _| h.current(0))),
                    periods: None,
                }],
            },
        ]
    } else {
        vec![Intervention::set(2, 0.0), Intervention::set(2, 1.0)]
    }
}

fn main() {
    use Term::{Baseline, CumulativeAverage, Current, Intercept, Period};
    let (method, n) = (common::method(), common::size());
    if method == "gen_gf" {
        generate(n);
        return;
    }
    assert!(
        ["gf_static", "gf_dynamic", "gf_boot"].contains(&method.as_str()),
        "unknown method {method}"
    );
    let t = Table::read(&common::data_path("gf", n));
    let id: Vec<u64> = t.ints("id");
    let period: Vec<usize> = t.ints("t0");
    let baseline = [t.col("V"), t.col("W")];
    let time_varying = [t.col("L1"), t.col("L2"), t.col("A")];
    let event: Vec<bool> = t.col("Y").iter().map(|&y| y > 0.5).collect();
    let competing = t.flags("D");
    let data = PersonPeriods {
        id: &id,
        period: &period,
        baseline: &baseline,
        time_varying: &time_varying,
    };
    let outcome = Outcome::Survival {
        event: &event,
        terms: vec![
            Intercept,
            Current(0),
            CumulativeAverage(1),
            Current(2),
            Baseline(0),
            Period,
            Term::product(Current(2), Baseline(0)),
        ],
        competing: Some(Competing {
            event: &competing,
            terms: vec![Intercept, Current(0), Baseline(1), Period],
        }),
        censoring: None,
    };
    let options = GFormulaOptions {
        bootstrap: if method == "gf_boot" { 20 } else { 0 },
        ..GFormulaOptions::default()
    };
    let covariates = covariates();
    let interventions = interventions(&method);
    let (times, g) = common::timed(common::reps(), || {
        gformula(&data, &covariates, &outcome, &interventions, &options).expect("g-formula")
    });
    let mut lines = Lines::default();
    for (name, model) in ["L1", "L2", "A"].iter().zip(&g.models.covariates) {
        lines.push_all(
            &format!("coef_{name}_"),
            &model.as_ref().expect("fitted").coefficients,
        );
    }
    lines.push_all("coef_Y_", &g.models.outcome.coefficients);
    lines.push_all(
        "coef_D_",
        &g.models.competing.as_ref().expect("competing").coefficients,
    );
    let last = PERIODS - 1;
    for (s, strategy) in g.strategies.iter().enumerate() {
        lines.push_opt(&format!("info_risk_{s}"), strategy.value[last].estimate);
        if method == "gf_boot" {
            lines.push_opt(&format!("info_se_{s}"), strategy.value[last].se);
        }
    }
    common::finish(&method, n, &times, &lines);
}
