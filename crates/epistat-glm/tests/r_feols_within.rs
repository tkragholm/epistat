//! The within-family estimator, `fixest::feols(y ~ x | family, cluster =
//! ~family)`, as [`Feols`] with one factor, checked against fixest (see the R
//! script for the version used) and against `lm` with one dummy variable per
//! family; and [`between_within`].

use epistat_glm::{Feols, FixedEffects, Singletons, Ssc, Vcov, between_within};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    fam: Vec<u32>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    coef: Vec<Option<f64>>,
    lm_coef: Vec<Option<f64>>,
    cluster_vcov: Vec<Option<f64>>,
    iid_vcov: Vec<Option<f64>>,
    r2_within: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/within.json"
    ))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn worst(got: &[Option<f64>], want: &[Option<f64>], label: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}");
    let scale = want.iter().flatten().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let mut worst = 0.0_f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        match (g, w) {
            (None, None) => {}
            (Some(g), Some(w)) => worst = worst.max((g - w).abs() / scale),
            _ => panic!("{label} entry {i}: {g:?} against {w:?}"),
        }
    }
    worst
}

#[test]
fn one_factor_matches_fixest_and_dummies() {
    for case in &reference().cases {
        let x = matrix(&case.x);
        let fe =
            FixedEffects::new(&[&case.fam], case.weights.as_deref(), Singletons::Remove).unwrap();
        let fit = Feols::default().fit(x.as_ref(), &case.y, &fe).unwrap();
        // The coefficients equal the dummy-variable regression to the
        // condition of the demeaned Gram matrix times epsilon, about 1e-12
        // here, plus the demeaning's tolerance of 1e-10 of a column's scale;
        // 1e-9 allows for the conditioning of lm's own QR on top.
        let d = worst(&fit.coefficients, &case.coef, &case.name);
        assert!(d <= 1e-9, "{} coefficients {d:e}", case.name);
        // With a column constant within families lm keeps that column and
        // drops a dummy instead, so there is nothing to compare.
        if case.name != "constant_column" {
            let d = worst(&fit.coefficients, &case.lm_coef, &case.name);
            assert!(d <= 1e-9, "{} against lm {d:e}", case.name);
        }
        let iid = fit.vcov(&fe, Vcov::Iid, Ssc::new()).unwrap();
        let clustered = fit
            .vcov(&fe, Vcov::Cluster(&[&case.fam]), Ssc::new())
            .unwrap();
        // The covariances are products of the coefficients' rounding, the
        // same 1e-9 bound relative to the largest entry.
        let d1 = worst(iid.matrix.as_slice(), &case.iid_vcov, &case.name);
        let d2 = worst(clustered.matrix.as_slice(), &case.cluster_vcov, &case.name);
        assert!(d1 <= 1e-9, "{} classical {d1:e}", case.name);
        assert!(d2 <= 1e-9, "{} clustered {d2:e}", case.name);
        assert!(
            (fit.r2_within.unwrap() - case.r2_within).abs() <= 1e-9,
            "{} r2",
            case.name
        );
        if case.name == "constant_column" {
            assert_eq!(fit.dropped_columns, vec![2]);
        } else {
            assert!(fit.dropped_columns.is_empty(), "{}", case.name);
        }
    }
}

#[test]
fn between_within_splits_each_column_into_family_mean_and_deviation() {
    let x = Mat::from_fn(5, 2, |i, j| {
        [[1.0, 4.0], [3.0, 4.0], [5.0, 4.0], [2.0, 9.0], [4.0, 9.0]][i][j]
    });
    let clusters = Clusters::from_labels(&[7, 7, 7, 9, 9]);
    let (means, deviations) = between_within(x.as_ref(), &clusters).unwrap();
    assert_eq!((means.nrows(), means.ncols()), (2, 2));
    assert!((means[(0, 0)] - 3.0).abs() < 1e-15 && (means[(1, 0)] - 3.0).abs() < 1e-15);
    assert!((means[(0, 1)] - 4.0).abs() < 1e-15);
    // A column that is constant within a family has deviation exactly 0.
    assert_eq!(deviations[(0, 1)].to_bits(), 0.0_f64.to_bits());
    assert!((deviations[(0, 0)] + 2.0).abs() < 1e-15 && (deviations[(4, 0)] - 1.0).abs() < 1e-15);
}
