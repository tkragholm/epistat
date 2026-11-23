//! Input checks, the counted repairs, passive and sampled columns, the
//! stationarity of the chains and the hand-off to pooling.

use epistat_impute::{
    Column, ImputeError, ImputeOptions, ImputeSpec, Incomplete, Method, PassiveFn, Ties, impute,
    nelson_aalen,
};
use epistat_inference::pooling::rubin;
use epistat_testkit::{SplitMix64, assert_within_standard_errors};

fn normal(rng: &mut SplitMix64) -> f64 {
    let (u, v) = (1.0 - rng.unit(), rng.unit());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

const OPTIONS: ImputeOptions = ImputeOptions {
    m: 4,
    max_iterations: 3,
    seed: 5,
};

fn small(n: usize) -> (Incomplete, ImputeSpec) {
    let mut rng = SplitMix64::new(77);
    let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let y: Vec<f64> = x.iter().map(|v| 0.5 * v + normal(&mut rng)).collect();
    let miss: Vec<bool> = (0..n).map(|_| rng.unit() < 0.3).collect();
    let data = Incomplete::new(
        vec![Column::Continuous(x), Column::Continuous(y)],
        &[miss, vec![false; n]],
    )
    .unwrap();
    let spec = ImputeSpec::default_for(&data);
    (data, spec)
}

fn error_of(data: &Incomplete, spec: &ImputeSpec, options: &ImputeOptions) -> ImputeError {
    impute(data, spec, options).expect_err("an error")
}

#[test]
fn invalid_requests_are_errors() {
    let (data, spec) = small(60);
    let zero_m = ImputeOptions { m: 0, ..OPTIONS };
    assert!(matches!(error_of(&data, &spec, &zero_m), ImputeError::Count { what: "m" }));
    let zero_it = ImputeOptions { max_iterations: 0, ..OPTIONS };
    assert!(matches!(error_of(&data, &spec, &zero_it), ImputeError::Count { .. }));

    let mut bad = spec.clone();
    bad.methods[0] = None;
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Method { column: 0, .. }));
    let mut bad = spec.clone();
    bad.methods[1] = Some(Method::Norm);
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Method { column: 1, .. }));
    let mut bad = spec.clone();
    bad.predictors[0] = vec![0];
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Predictor { .. }));
    let mut bad = spec.clone();
    bad.predictors[0] = vec![1, 1];
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Predictor { .. }));
    let mut bad = spec.clone();
    bad.predictors[0] = vec![7];
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Predictor { .. }));
    let mut bad = spec.clone();
    bad.visit_order = vec![];
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::VisitOrder));
    let mut bad = spec.clone();
    bad.methods.pop();
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Length { what: "methods", .. }));
    // A continuous column cannot be imputed by logistic regression.
    let mut bad = spec.clone();
    bad.methods[0] = Some(Method::Logreg);
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Unsupported { column: 0 }));
    let mut bad = spec;
    bad.methods[0] = Some(Method::Pmm {
        donors: 0,
        match_type: epistat_impute::MatchType::Mixed,
    });
    assert!(matches!(error_of(&data, &bad, &OPTIONS), ImputeError::Method { .. }));
}

#[test]
fn invalid_data_are_errors() {
    let col = |v: Vec<f64>| Column::Continuous(v);
    assert!(matches!(
        Incomplete::new(vec![col(vec![1.0, 2.0]), col(vec![1.0])], &[vec![false; 2], vec![false]]),
        Err(ImputeError::Length { .. })
    ));
    assert!(matches!(
        Incomplete::new(vec![col(vec![1.0, f64::NAN])], &[vec![false; 2]]),
        Err(ImputeError::Value { column: 0, row: 1 })
    ));
    // A missing cell's value is ignored.
    assert!(Incomplete::new(vec![col(vec![1.0, f64::NAN])], &[vec![false, true]]).is_ok());
    assert!(matches!(
        Incomplete::new(vec![Column::Binary(vec![0, 2])], &[vec![false; 2]]),
        Err(ImputeError::Value { .. })
    ));
    assert!(matches!(
        Incomplete::new(
            vec![Column::Nominal { codes: vec![0, 3], levels: 3 }],
            &[vec![false; 2]]
        ),
        Err(ImputeError::Value { .. })
    ));
    assert!(matches!(
        Incomplete::new(
            vec![Column::Nominal { codes: vec![0, 0], levels: 1 }],
            &[vec![false; 2]]
        ),
        Err(ImputeError::Levels { .. })
    ));
    // A column with no observed value.
    let data = Incomplete::new(
        vec![col(vec![0.0; 5]), col(vec![1.0, 2.0, 3.0, 4.0, 5.0])],
        &[vec![true; 5], vec![false; 5]],
    )
    .unwrap();
    let spec = ImputeSpec::default_for(&data);
    assert!(matches!(error_of(&data, &spec, &OPTIONS), ImputeError::NothingObserved { column: 0 }));
    // Too few observed rows for the model: two observed rows, two parameters.
    let data = Incomplete::new(
        vec![col(vec![1.0, 2.0, 0.0, 0.0]), col(vec![1.0, 2.0, 3.0, 4.0])],
        &[vec![false, false, true, true], vec![false; 4]],
    )
    .unwrap();
    let spec = ImputeSpec::default_for(&data);
    assert!(matches!(error_of(&data, &spec, &OPTIONS), ImputeError::TooFewObserved { .. }));
    // A nominal level that never occurs among the observed values.
    let data = Incomplete::new(
        vec![
            Column::Nominal { codes: vec![0, 1, 0, 1, 0, 0], levels: 3 },
            col(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
        &[vec![false, false, false, false, false, true], vec![false; 6]],
    )
    .unwrap();
    let spec = ImputeSpec::default_for(&data);
    assert!(matches!(error_of(&data, &spec, &OPTIONS), ImputeError::EmptyLevel { column: 0, level: 2 }));
    // An incomplete ordered column cannot be imputed.
    let data = Incomplete::new(
        vec![
            Column::Ordered { codes: vec![0, 1, 0, 1, 0, 1], levels: 2 },
            col(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
        &[vec![false, false, false, false, false, true], vec![false; 6]],
    )
    .unwrap();
    let spec = ImputeSpec::default_for(&data);
    assert!(matches!(error_of(&data, &spec, &OPTIONS), ImputeError::Method { column: 0, .. }));
}

#[test]
fn a_passive_function_of_the_wrong_length_is_an_error() {
    let (data, mut spec) = small(60);
    spec.methods[0] = Some(Method::Passive(PassiveFn::new(|_| vec![1.0])));
    assert!(matches!(error_of(&data, &spec, &OPTIONS), ImputeError::Passive { column: 0 }));
}

#[test]
fn survival_inputs_are_checked() {
    assert!(nelson_aalen(&[1.0, 2.0], &[1], Ties::Efron).is_err());
    assert!(nelson_aalen(&[1.0, -2.0], &[1, 0], Ties::Efron).is_err());
    assert!(nelson_aalen(&[1.0, f64::NAN], &[1, 0], Ties::Breslow).is_err());
    assert!(nelson_aalen(&[1.0, 2.0], &[1, 2], Ties::Breslow).is_err());
    // Three people, events at 1 and 3, censoring at 2.
    let h = nelson_aalen(&[1.0, 2.0, 3.0], &[1, 0, 1], Ties::Breslow).unwrap();
    assert_eq!(h, vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 + 1.0]);
    assert!(nelson_aalen(&[], &[], Ties::Efron).unwrap().is_empty());
}

#[test]
fn imputed_values_are_valid_for_their_column() {
    let n = 400;
    let mut rng = SplitMix64::new(3);
    let a: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let income: Vec<f64> = a.iter().map(|v| (0.4 * v + 0.5 * normal(&mut rng)).exp()).collect();
    let flag: Vec<u8> = a.iter().map(|v| u8::from(*v + 0.5 * normal(&mut rng) > 0.0)).collect();
    let level: Vec<u32> = a
        .iter()
        .map(|v| if *v + 0.7 * normal(&mut rng) > 0.5 { 2 } else { u32::from(rng.unit() < 0.5) })
        .collect();
    let sampled: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let miss = |rng: &mut SplitMix64| (0..n).map(|_| rng.unit() < 0.3).collect::<Vec<bool>>();
    let masks = vec![
        vec![false; n],
        miss(&mut rng),
        miss(&mut rng),
        miss(&mut rng),
        miss(&mut rng),
    ];
    let data = Incomplete::new(
        vec![
            Column::Continuous(a),
            Column::Continuous(income.clone()),
            Column::Binary(flag),
            Column::Nominal { codes: level, levels: 3 },
            Column::Continuous(sampled.clone()),
        ],
        &masks,
    )
    .unwrap();
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[4] = Some(Method::Sample);
    spec.predictors[4] = vec![];
    let imputed = impute(&data, &spec, &ImputeOptions { m: 5, max_iterations: 4, seed: 9 }).unwrap();

    let observed_income: Vec<f64> = (0..n).filter(|&i| !masks[1][i]).map(|i| income[i]).collect();
    for set in &imputed.values[1] {
        assert_eq!(set.len(), masks[1].iter().filter(|&&m| m).count());
        // Predictive mean matching only returns observed values.
        assert!(set.iter().all(|v| observed_income.contains(v)));
    }
    for set in &imputed.values[2] {
        assert!(set.iter().all(|&v| v == 0.0 || v == 1.0));
    }
    for set in &imputed.values[3] {
        assert!(set.iter().all(|&v| v == 0.0 || v == 1.0 || v == 2.0));
        assert!(set.contains(&2.0), "the third level is imputed");
    }
    for set in &imputed.values[4] {
        assert!(set.iter().all(|v| sampled.contains(v)));
    }
    assert!(imputed.values[0].iter().all(Vec::is_empty));
    assert!(imputed.n_augmented > 0);
    // chain statistics: one entry per set and iteration for each incomplete column.
    for j in 1..5 {
        assert_eq!(imputed.chain_mean[j].len(), 5 * 4);
        assert_eq!(imputed.chain_variance[j].len(), 5 * 4);
    }
    assert!(imputed.chain_mean[0].is_empty());
    // Completed data hold the observed values where they exist.
    let done = imputed.completed(&data, 2).unwrap();
    for i in (0..n).filter(|&i| !masks[1][i]) {
        assert_eq!(done[1][i], income[i]);
    }
    assert!(imputed.completed(&data, 5).is_err());
}

#[test]
fn passive_columns_follow_the_columns_they_are_computed_from() {
    let n = 300;
    let mut rng = SplitMix64::new(4);
    let a: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let b: Vec<f64> = a.iter().map(|v| v + normal(&mut rng)).collect();
    let product: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    let miss_a: Vec<bool> = (0..n).map(|_| rng.unit() < 0.3).collect();
    let miss_b: Vec<bool> = (0..n).map(|_| rng.unit() < 0.2).collect();
    // The product is missing wherever a or b is.
    let miss_p: Vec<bool> = (0..n).map(|i| miss_a[i] || miss_b[i]).collect();
    let data = Incomplete::new(
        vec![
            Column::Continuous(a),
            Column::Continuous(b),
            Column::Continuous(product),
            Column::Continuous((0..n).map(|_| normal(&mut rng)).collect()),
        ],
        &[miss_a, miss_b, miss_p, vec![false; n]],
    )
    .unwrap();
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[2] = Some(Method::Passive(PassiveFn::new(|c| {
        c[0].iter().zip(&c[1]).map(|(x, y)| x * y).collect()
    })));
    spec.visit_order = vec![0, 1, 2];
    // The incomplete columns use each other but not the passive column.
    spec.predictors[0] = vec![1, 3];
    spec.predictors[1] = vec![0, 3];
    let imputed = impute(&data, &spec, &ImputeOptions { m: 3, max_iterations: 3, seed: 8 }).unwrap();
    for s in 0..3 {
        let c = imputed.completed(&data, s).unwrap();
        for i in 0..n {
            assert_eq!(c[2][i], c[0][i] * c[1][i], "set {s} row {i}");
        }
    }
}

#[test]
fn a_collinear_predictor_is_dropped_and_counted() {
    let n = 200;
    let mut rng = SplitMix64::new(6);
    let a: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let copy: Vec<f64> = a.iter().map(|v| 2.0 * v + 1.0).collect();
    let y: Vec<f64> = a.iter().map(|v| v + normal(&mut rng)).collect();
    let miss: Vec<bool> = (0..n).map(|_| rng.unit() < 0.3).collect();
    let data = Incomplete::new(
        vec![Column::Continuous(y), Column::Continuous(a), Column::Continuous(copy)],
        &[miss, vec![false; n], vec![false; n]],
    )
    .unwrap();
    let spec = ImputeSpec::default_for(&data);
    let imputed = impute(&data, &spec, &OPTIONS).unwrap();
    assert_eq!(imputed.n_aliased_dropped, OPTIONS.m * OPTIONS.max_iterations);
    assert!(imputed.values[0].iter().flatten().all(|v| v.is_finite()));
}

#[test]
fn perfect_prediction_is_handled_by_augmentation() {
    // The binary outcome equals the sign of x among the observed rows.
    let n = 120;
    let mut rng = SplitMix64::new(10);
    let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let y: Vec<u8> = x.iter().map(|v| u8::from(*v > 0.0)).collect();
    let miss: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let data = Incomplete::new(
        vec![Column::Binary(y), Column::Continuous(x.clone())],
        &[miss.clone(), vec![false; n]],
    )
    .unwrap();
    let spec = ImputeSpec::default_for(&data);
    let imputed = impute(&data, &spec, &ImputeOptions { m: 20, max_iterations: 3, seed: 1 }).unwrap();
    assert_eq!(imputed.n_augmented, 20 * 3);
    // The imputed value follows the sign of x for almost every cell.
    let rows = &imputed.missing_rows[0];
    let agree: usize = imputed.values[0]
        .iter()
        .map(|set| {
            set.iter()
                .zip(rows)
                .filter(|&(&v, &r)| (v == 1.0) == (x[r] > 0.0))
                .count()
        })
        .sum();
    let total = 20 * rows.len();
    assert!(agree as f64 / total as f64 > 0.85, "{agree} of {total}");
}

#[test]
fn chains_started_at_the_observed_distribution_stay_there() {
    // Three standard normals with correlations 0.5 and two incomplete by
    // design, completely at random. The imputed cells have the distribution of
    // the data, so their mean and variance do not drift over the iterations.
    let n = 1500;
    let mut rng = SplitMix64::new(11);
    let (mut x1, mut x2, mut x3) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let common = normal(&mut rng);
        let w = 0.5_f64.sqrt();
        x1[i] = w * common + w * normal(&mut rng);
        x2[i] = w * common + w * normal(&mut rng);
        x3[i] = w * common + w * normal(&mut rng);
    }
    let miss = |rng: &mut SplitMix64| (0..n).map(|_| rng.unit() < 0.4).collect::<Vec<bool>>();
    let masks = vec![miss(&mut rng), miss(&mut rng), vec![false; n]];
    let data = Incomplete::new(
        vec![Column::Continuous(x1), Column::Continuous(x2), Column::Continuous(x3)],
        &masks,
    )
    .unwrap();
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[0] = Some(Method::Norm);
    spec.methods[1] = Some(Method::Norm);
    let (m, maxit) = (200, 8);
    let imputed = impute(&data, &spec, &ImputeOptions { m, max_iterations: maxit, seed: 21 }).unwrap();
    for j in 0..2 {
        // The mean over data sets of the imputed variance, by iteration.
        let at = |t: usize| -> (f64, f64) {
            let v: Vec<f64> = (0..m)
                .map(|s| imputed.chain_variance[j][s * maxit + t].unwrap())
                .collect();
            let mean = v.iter().sum::<f64>() / m as f64;
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
            (mean, (var / m as f64).sqrt())
        };
        let first = at(0);
        for t in 1..maxit {
            let later = at(t);
            assert_within_standard_errors(later.0, later.1, first.0, first.1, 4.0);
        }
        // And the variance is the population's, 1, up to the sampling error of the
        // variance of the observed values, sqrt(2 / n_observed).
        let observed = (n - imputed.missing_rows[j].len()) as f64;
        assert_within_standard_errors(first.0, first.1, 1.0, (2.0 / observed).sqrt(), 4.0);
    }
}

#[test]
fn pooling_the_completed_sets_is_rubins_rule() {
    let (data, spec) = small(150);
    let imputed = impute(&data, &spec, &OPTIONS).unwrap();
    let analysis = |c: &[Vec<f64>]| -> Result<(f64, f64), std::convert::Infallible> {
        let n = c[0].len() as f64;
        let mean = c[0].iter().sum::<f64>() / n;
        let var = c[0].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Ok((mean, var / n))
    };
    let pooled = imputed.pool(&data, Some(149.0), analysis).unwrap();
    let (mut q, mut u) = (vec![], vec![]);
    for s in 0..OPTIONS.m {
        let (a, b) = analysis(&imputed.completed(&data, s).unwrap()).unwrap();
        q.push(a);
        u.push(b);
    }
    assert_eq!(pooled, rubin(&q, &u, Some(149.0)).unwrap());
    let failing = |_: &[Vec<f64>]| -> Result<(f64, f64), std::fmt::Error> { Err(std::fmt::Error) };
    assert!(matches!(
        imputed.pool(&data, None, failing),
        Err(ImputeError::Analysis { set: 0, .. })
    ));
    let vector = imputed
        .pool_vector(&data, 1, |c| {
            let (a, b) = analysis(c)?;
            Ok::<_, std::convert::Infallible>((vec![a], vec![b]))
        })
        .unwrap();
    assert!((vector.estimate[0] - pooled.estimate).abs() < 1e-15);
}
