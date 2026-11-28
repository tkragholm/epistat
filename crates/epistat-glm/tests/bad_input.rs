//! Every entry point meets an empty column, a NaN, an infinity and a column of
//! the wrong length with an `Input` error carrying the code of the shared
//! check, not a panic. No entry point here takes sorted input, so the unsorted
//! kind is not tried.

use epistat_glm::park::{classic_park_test, modified_park_test, modified_park_test_with};
use epistat_glm::{
    ElasticNet, Family, Feols, FixedEffects, Gee, GeeData, Glm, GlmData, Link, Multinom,
    MultinomData, NegBin, Ordinal, OrdinalData, OrdinalLink, Singletons, between_within, brant,
    robust_covariance,
};
use epistat_glm::{RiskScale, SandwichType, risk_regression};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const N: usize = 24;

/// The code of the shared check for each kind of bad column. The design
/// fixes the number of rows, so an empty column beside it is a length error.
fn code(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

fn design() -> Mat<f64> {
    Mat::from_fn(N, 2, |i, j| if j == 0 { 1.0 } else { (i % 7) as f64 / 3.0 })
}

/// The design with its second column spoiled as `bad` spoils a column.
fn spoiled(bad: BadInput) -> Mat<f64> {
    let x = design();
    let column: Vec<f64> = bad.column(&(0..N).map(|i| x[(i, 1)]).collect::<Vec<_>>());
    Mat::from_fn(column.len(), 2, |i, j| if j == 0 { 1.0 } else { column[i] })
}

fn counts() -> Vec<f64> {
    (0..N)
        .map(|i| f64::from(u8::try_from(i % 5).unwrap()))
        .collect()
}

fn binary() -> Vec<f64> {
    (0..N).map(|i| f64::from(u8::from(i % 3 == 0))).collect()
}

fn weights() -> Vec<f64> {
    (0..N).map(|i| 0.5 + (i % 4) as f64 / 4.0).collect()
}

#[test]
fn the_glm_views_refuse_bad_columns() {
    let x = design();
    let (y, w) = (counts(), weights());
    for which in ["x", "y", "weights", "offset"] {
        // The design fixes the number of rows: an empty design is empty.
        let code = |bad: BadInput| if which == "x" { bad.code() } else { code(bad) };
        assert_rejects_bad_input_with_codes(&format!("GlmData {which}"), code, |bad| {
            if bad == BadInput::Unsorted {
                return None;
            }
            let column = |good: &[f64]| bad.column(good);
            Some(match which {
                "x" => {
                    let x = spoiled(bad);
                    GlmData::builder(x.as_ref(), &y).build().map(|_| ())
                }
                "y" => GlmData::builder(x.as_ref(), &column(&y))
                    .build()
                    .map(|_| ()),
                "weights" => GlmData::builder(x.as_ref(), &y)
                    .weights(&column(&w))
                    .build()
                    .map(|_| ()),
                _ => GlmData::builder(x.as_ref(), &y)
                    .offset(&column(&w))
                    .build()
                    .map(|_| ()),
            })
        });
    }
}

#[test]
fn the_models_refuse_bad_columns() {
    let x = design();
    let (y, b) = (counts(), binary());
    let cluster: Vec<u32> = (0..N).map(|i| u32::try_from(i / 3).unwrap()).collect();
    let category: Vec<u32> = (0..N).map(|i| u32::try_from(i % 3).unwrap()).collect();
    let class: Vec<usize> = (0..N).map(|i| i % 3).collect();
    let skip = |bad: BadInput| bad == BadInput::Unsorted;
    assert_rejects_bad_input_with_codes("Glm::fit", code, |bad| {
        (!skip(bad)).then(|| {
            let y = bad.column(&y);
            GlmData::builder(x.as_ref(), &y)
                .build()
                .and_then(|d| Glm::new(Family::Poisson, Link::Log).fit(&d))
        })
    });
    assert_rejects_bad_input_with_codes("NegBin::fit", code, |bad| {
        (!skip(bad)).then(|| {
            let y = bad.column(&y);
            GlmData::builder(x.as_ref(), &y)
                .build()
                .and_then(|d| NegBin::new().fit(&d))
        })
    });
    assert_rejects_bad_input_with_codes("ElasticNet::fit", code, |bad| {
        (!skip(bad)).then(|| {
            let y = bad.column(&y);
            GlmData::builder(x.as_ref(), &y)
                .build()
                .and_then(|d| ElasticNet::new(Family::Gaussian, 0.1, 0.5).fit(&d))
        })
    });
    assert_rejects_bad_input_with_codes("Gee::fit", code, |bad| {
        (!skip(bad)).then(|| {
            let y = bad.column(&y);
            GeeData::builder(x.as_ref(), &y, &cluster)
                .build()
                .and_then(|d| {
                    Gee::new(
                        Family::Poisson,
                        Link::Log,
                        epistat_glm::Correlation::Exchangeable,
                    )
                    .fit(&d)
                })
        })
    });
    assert_rejects_bad_input_with_codes("Ordinal::fit", BadInput::code, |bad| {
        (!skip(bad)).then(|| {
            let x = spoiled(bad);
            let z = Mat::from_fn(x.nrows(), 1, |i, _| x[(i, 1)]);
            OrdinalData::builder(z.as_ref(), &category)
                .build()
                .and_then(|d| Ordinal::new(OrdinalLink::Logit).fit(&d))
        })
    });
    assert_rejects_bad_input_with_codes("brant", BadInput::code, |bad| {
        (!skip(bad)).then(|| {
            let x = spoiled(bad);
            let z = Mat::from_fn(x.nrows(), 1, |i, _| x[(i, 1)]);
            OrdinalData::builder(z.as_ref(), &category)
                .build()
                .and_then(|d| brant(&d))
        })
    });
    assert_rejects_bad_input_with_codes("Multinom::fit", BadInput::code, |bad| {
        (!skip(bad)).then(|| {
            let x = spoiled(bad);
            MultinomData::builder(x.as_ref(), &class, 3)
                .build()
                .and_then(|d| Multinom::default().fit(&d))
        })
    });
    assert_rejects_bad_input_with_codes("risk_regression", code, |bad| {
        (!skip(bad)).then(|| {
            let b = bad.column(&b);
            GlmData::builder(x.as_ref(), &b)
                .build()
                .and_then(|d| risk_regression(RiskScale::Ratio, &d, None, SandwichType::Hc1))
        })
    });
    let model = Glm::new(Family::Poisson, Link::Log);
    let fit = model
        .fit(&GlmData::builder(x.as_ref(), &y).build().unwrap())
        .unwrap();
    assert_rejects_bad_input_with_codes("robust_covariance", code, |bad| {
        (!skip(bad)).then(|| {
            let y = bad.column(&y);
            robust_covariance(&model, &fit, x.as_ref(), &y, None, SandwichType::Hc0)
        })
    });
}

#[test]
fn feols_and_the_park_tests_refuse_bad_columns() {
    let x = design();
    let y = counts();
    let unit: Vec<u32> = (0..N).map(|i| u32::try_from(i / 4).unwrap()).collect();
    let fe = FixedEffects::new(&[&unit], None, Singletons::Keep).unwrap();
    let z = Mat::from_fn(N, 1, |i, _| x[(i, 1)]);
    assert_rejects_bad_input_with_codes("Feols::fit", code, |bad| {
        (bad != BadInput::Unsorted).then(|| Feols::default().fit(z.as_ref(), &bad.column(&y), &fe))
    });
    assert_rejects_bad_input_with_codes("FixedEffects::new", code, |bad| {
        (bad != BadInput::Unsorted)
            .then(|| FixedEffects::new(&[&unit], Some(&bad.column(&weights())), Singletons::Keep))
    });
    let fitted: Vec<f64> = (0..N).map(|i| 1.0 + i as f64).collect();
    assert_rejects_bad_input_with_codes("modified_park_test", code, |bad| {
        matches!(bad, BadInput::Mismatched)
            .then(|| modified_park_test(&fitted, &bad.column(&y), None))
    });
    assert_rejects_bad_input_with_codes("classic_park_test", code, |bad| {
        matches!(bad, BadInput::Mismatched).then(|| classic_park_test(&fitted, &bad.column(&y)))
    });
}

#[test]
fn the_park_controls_and_the_between_within_split_refuse_a_wrong_length() {
    let y = counts();
    let fitted: Vec<f64> = (0..N).map(|i| 1.0 + i as f64).collect();
    let length = |_: BadInput| "input.length";
    assert_rejects_bad_input_with_codes("modified_park_test_with", length, |bad| {
        matches!(bad, BadInput::Mismatched).then(|| {
            modified_park_test_with(
                Glm::new(Family::Gamma, Link::Log),
                &fitted,
                &bad.column(&y),
                None,
            )
        })
    });
    let x = design();
    let short: Vec<u32> = (0..N - 1).map(|i| u32::try_from(i / 4).unwrap()).collect();
    let clusters = Clusters::from_codes(&short);
    assert_rejects_bad_input_with_codes("between_within", length, |bad| {
        matches!(bad, BadInput::Mismatched).then(|| between_within(x.as_ref(), &clusters))
    });
}
