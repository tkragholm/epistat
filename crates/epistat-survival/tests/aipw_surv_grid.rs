//! The grid form of the augmentation of `epistat_survival::aipw`: with a cell for
//! every censoring time it is the exact sum, and with fewer cells the risk
//! approaches the exact one as the cells get narrower.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::aipw::{
    AipwCensoring, AipwSurv, AipwSurvOptions, AipwSurvSpec, Estimator, aipw_cox,
};
use epistat_survival::cox::Ties;
use epistat_survival::standardise::Counterfactual;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Scenario {
    name: String,
    times: Vec<f64>,
    time: Vec<f64>,
    status: Vec<i32>,
    a: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    scenarios: Vec<Scenario>,
}

fn run(s: &Scenario, km: bool, grid: Option<usize>) -> AipwSurv {
    let n = s.time.len();
    let treated: Vec<bool> = s.a.iter().map(|&a| a > 0.5).collect();
    let x = Mat::from_fn(n, 3, |i, j| [s.a[i], s.x1[i], s.x2[i]][j]);
    let z = Mat::from_fn(n, 3, |i, j| [1.0, s.x1[i], s.x2[i]][j]);
    let group: Vec<u32> = s.a.iter().map(|&a| u32::from(a > 0.5)).collect();
    let data = SurvData::builder(&s.time, &s.status).build().unwrap();
    let (c0, c1) = ([(0usize, 0.0)], [(0usize, 1.0)]);
    let spec = AipwSurvSpec {
        treatment: &treated,
        outcome_x: x.as_ref(),
        levels: [Counterfactual::Columns(&c0), Counterfactual::Columns(&c1)],
        propensity_x: z.as_ref(),
        censoring: if km {
            AipwCensoring::KaplanMeier {
                groups: Some(&group),
            }
        } else {
            AipwCensoring::Cox { x: x.as_ref() }
        },
    };
    aipw_cox(
        &data,
        &spec,
        &s.times,
        &AipwSurvOptions {
            estimator: Estimator::Aiptw,
            ties: Ties::Breslow,
            grid,
            ..AipwSurvOptions::default()
        },
    )
    .unwrap()
}

fn worst(a: &AipwSurv, b: &AipwSurv) -> f64 {
    let se = |r: &AipwSurv| r.risks.se.clone().unwrap();
    let mut w: f64 = 0.0;
    for (x, y) in a.risks.risk.iter().zip(&b.risks.risk) {
        w = w.max((x.unwrap() - y.unwrap()).abs());
    }
    for (x, y) in se(a).iter().zip(se(b).iter()) {
        w = w.max((x.unwrap() - y.unwrap()).abs());
    }
    w
}

#[test]
fn the_grid_approaches_the_exact_sum() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/aipw_surv.json"
    ));
    for s in r
        .scenarios
        .iter()
        .filter(|s| s.name == "km" || s.name == "cox")
    {
        let km = s.name == "km";
        let exact = run(s, km, None);
        // The number of distinct censoring times is below the number of rows, so a
        // grid of that many cells has a cell for every time.
        let all = run(s, km, Some(s.time.len()));
        // The difference of two sums of the same terms, over at most 500
        // jumps of terms of order 1 with relative error 1e-16.
        assert!(
            worst(&exact, &all) < 1e-12,
            "{}: {}",
            s.name,
            worst(&exact, &all)
        );
        let errors: Vec<f64> = [2, 8, 32]
            .iter()
            .map(|&k| worst(&exact, &run(s, km, Some(k))))
            .collect();
        println!("{} {errors:?}", s.name);
        assert!(errors[2] < errors[0], "{}: {errors:?}", s.name);
        assert!(errors[2] < 5e-3, "{}: {errors:?}", s.name);
    }
}

#[test]
fn the_error_of_the_grid_falls_with_many_censoring_times() {
    // 6000 persons with continuous times, so that there are about 1500 censoring
    // times and a cell of a grid of 64 holds some twenty of them.
    let n = 6000;
    let mut rng = Stream::seeded(5);
    let (mut time, mut status, mut a, mut x1, mut x2) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let (u, v) = (rng.normal(), f64::from(u8::from(rng.uniform() < 0.4)));
        let t = f64::from(u8::from(
            rng.uniform() < 1.0 / (1.0 + (-(0.5 * u - 0.4 * v)).exp()),
        ));
        let te = -rng.uniform().ln() / (0.12 * (0.5 * u - 0.4 * v - 0.6 * t).exp());
        let tc = (-rng.uniform().ln() / (0.06 * (0.3 * u + 0.3 * t).exp())).min(12.0);
        time.push(te.min(tc));
        status.push(i32::from(te <= tc));
        a.push(t);
        x1.push(u);
        x2.push(v);
    }
    let s = Scenario {
        name: "large".into(),
        times: vec![2.0, 5.0, 8.0],
        time,
        status,
        a,
        x1,
        x2,
    };
    for km in [true, false] {
        let exact = run(&s, km, None);
        let errors: Vec<f64> = [8, 64, 512]
            .iter()
            .map(|&k| worst(&exact, &run(&s, km, Some(k))))
            .collect();
        println!("large km={km} {errors:?}");
        assert!(errors[1] < errors[0] && errors[2] < errors[1], "{errors:?}");
        assert!(errors[2] < 1e-4, "{errors:?}");
    }
}
