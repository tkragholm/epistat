//! `SurvData` built from Arrow arrays borrows the arrays' buffers and gives
//! the bits of the same data built from slices, for every builder path. Nulls
//! and wrong types are errors that name the column.

#![cfg(feature = "arrow")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use std::sync::Arc;

use arrow_array::types::UInt32Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, UInt8Array, UInt32Array,
};
use epistat_rng::Stream;
use epistat_survival::aalen_johansen::{self, AalenJohansenOptions};
use epistat_survival::arrow::{column, follow_up, interval, interval_status};
use epistat_survival::cox::{self, CoxOptions};
use epistat_survival::survfit::{self, SurvfitOptions};
use epistat_survival::{Error, SurvData};
use faer::Mat;

const N: usize = 300;

struct Sample {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    cluster: Vec<u32>,
    id: Vec<u32>,
    istate: Vec<u32>,
    x: Mat<f64>,
}

fn sample() -> Sample {
    let mut rng = Stream::seeded(77);
    let mut s = Sample {
        start: Vec::new(),
        stop: Vec::new(),
        status: Vec::new(),
        weight: Vec::new(),
        stratum: Vec::new(),
        cluster: Vec::new(),
        id: Vec::new(),
        istate: Vec::new(),
        x: Mat::zeros(N, 2),
    };
    for i in 0..N {
        let a = rng.normal();
        s.x[(i, 0)] = a;
        s.x[(i, 1)] = f64::from(rng.below(3) as u32);
        let t = (rng.exp1() * (-0.3 * a).exp() * 8.0).ceil() / 10.0 + 0.1;
        s.stop.push(t);
        s.start.push((rng.uniform() * t * 8.0).floor() / 10.0 * 0.9);
        s.status.push(match rng.below(6) {
            0..=1 => 1,
            2 => 2,
            3 => 3,
            _ => 0,
        });
        s.weight.push(match rng.below(5) {
            0 => 0.0,
            1 => 2.0,
            _ => 1.0,
        });
        s.stratum.push(rng.below(3) as u32 * 7 + 5);
        s.cluster.push(rng.below(20) as u32);
        s.id.push(rng.below(150) as u32);
        s.istate.push(1 + rng.below(3) as u32);
    }
    s
}

/// A dictionary column with `UInt32` keys `keys` over some values.
fn dictionary(keys: &[u32]) -> DictionaryArray<UInt32Type> {
    let width = keys.iter().max().map_or(0, |&m| m as usize + 1);
    let values = Int64Array::from_iter_values((0..width as i64).map(|v| v * 100));
    DictionaryArray::new(UInt32Array::from(keys.to_vec()), Arc::new(values))
}

#[test]
fn follow_up_paths_give_the_bits_of_slices() {
    let s = sample();
    let stop = Float64Array::from(s.stop.clone());
    let status = Int32Array::from(s.status.clone());
    let start = Float64Array::from(s.start.clone());
    let weight = Float64Array::from(s.weight.clone());
    let stratum = UInt32Array::from(s.stratum.clone());
    let stratum_dict = dictionary(&s.stratum);
    let cluster = UInt32Array::from(s.cluster.clone());
    let id = dictionary(&s.id);
    let istate = UInt32Array::from(s.istate.clone());
    let mut checked = 0;
    for entry in [false, true] {
        for weighted in [false, true] {
            for grouping in 0..3 {
                for multi in [false, true] {
                    let n_codes = if multi { 3 } else { 1 };
                    let flat: Vec<i32> = s
                        .status
                        .iter()
                        .map(|&c| if multi { c } else { i32::from(c == 1) })
                        .collect();
                    let arrow_status = Int32Array::from(flat.clone());
                    let (status_ref, status_slice): (&dyn Array, &[i32]) = if multi {
                        (&status, &s.status)
                    } else {
                        (&arrow_status, &flat)
                    };
                    let mut a = follow_up(&stop, status_ref).n_codes(n_codes);
                    let mut b = SurvData::builder(&s.stop, status_slice).n_codes(n_codes);
                    if entry {
                        a = a.start(&start);
                        b = b.start(&s.start);
                    }
                    if weighted {
                        a = a.weight(&weight);
                        b = b.weight(&s.weight);
                    }
                    match grouping {
                        1 => {
                            a = a.stratum(&stratum).cluster(&cluster);
                            b = b.stratum(&s.stratum).cluster(&s.cluster);
                        }
                        2 => {
                            // Dictionary keys are the same codes as the slice.
                            a = a.stratum(&stratum_dict).id(&id).cluster(&cluster);
                            b = b.stratum(&s.stratum).id(&s.id).cluster(&s.cluster);
                        }
                        _ => {}
                    }
                    if multi && entry {
                        a = a.istate(&istate);
                        b = b.istate(&s.istate);
                    }
                    let from_arrow = a.build().unwrap();
                    let from_slices = b.build().unwrap();
                    assert_eq!(format!("{from_arrow:?}"), format!("{from_slices:?}"));

                    let opts = CoxOptions {
                        cause: multi.then_some(1),
                        ..CoxOptions::default()
                    };
                    let fit_a = cox::fit(&from_arrow, s.x.as_ref(), &opts).unwrap();
                    let fit_b = cox::fit(&from_slices, s.x.as_ref(), &opts).unwrap();
                    assert_eq!(format!("{fit_a:?}"), format!("{fit_b:?}"));

                    if !multi {
                        let sf_a = survfit::estimate(&from_arrow, &SurvfitOptions::default());
                        let sf_b = survfit::estimate(&from_slices, &SurvfitOptions::default());
                        assert_eq!(
                            format!("{:?}", sf_a.unwrap()),
                            format!("{:?}", sf_b.unwrap())
                        );
                    }
                    let aj_a =
                        aalen_johansen::estimate(&from_arrow, &AalenJohansenOptions::default());
                    let aj_b =
                        aalen_johansen::estimate(&from_slices, &AalenJohansenOptions::default());
                    assert_eq!(
                        format!("{:?}", aj_a.unwrap()),
                        format!("{:?}", aj_b.unwrap())
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 24);
}

#[test]
fn interval_paths_give_the_bits_of_slices() {
    let s = sample();
    let lower: Vec<f64> = s.stop.iter().map(|t| t - 0.05).collect();
    let mut upper: Vec<f64> = s.stop.iter().map(|t| t + 0.05).collect();
    let mut lower2 = lower.clone();
    upper[3] = f64::INFINITY;
    lower2[4] = f64::NEG_INFINITY;
    upper[5] = lower[5];
    let lo = Float64Array::from(lower2.clone());
    let up = Float64Array::from(upper.clone());
    let weight = Float64Array::from(s.weight.clone());
    let stratum = dictionary(&s.stratum);
    let cluster = UInt32Array::from(s.cluster.clone());
    let a = interval(&lo, &up)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&cluster)
        .build()
        .unwrap();
    let b = SurvData::interval_builder(&lower2, &upper)
        .weight(&s.weight)
        .stratum(&s.stratum)
        .cluster(&s.cluster)
        .build()
        .unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));

    let interval_status_codes: Vec<i32> = (0..N as i32).map(|i| i % 4).collect();
    let time2: Vec<f64> = s.stop.iter().map(|t| t + 0.5).collect();
    let t1 = Float64Array::from(s.stop.clone());
    let t2 = Float64Array::from(time2.clone());
    let st = Int32Array::from(interval_status_codes.clone());
    let a = interval_status(&t1, &t2, &st).id(&cluster).build().unwrap();
    let b = SurvData::interval_status_builder(&s.stop, &time2, &interval_status_codes)
        .id(&s.cluster)
        .build()
        .unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn columns_are_borrowed_not_copied() {
    let s = sample();
    let stop = Float64Array::from(s.stop.clone());
    let status = Int32Array::from(s.status.clone());
    let start = Float64Array::from(s.start.clone());
    let weight = Float64Array::from(s.weight.clone());
    let dense: Vec<u32> = (0..N as u32).map(|i| i % 4).collect();
    let stratum = UInt32Array::from(dense.clone());
    let istate = UInt32Array::from(s.istate.clone());
    let dict = dictionary(&dense);
    let data = follow_up(&stop, &status)
        .n_codes(3)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .cluster(&dict)
        .istate(&istate)
        .build()
        .unwrap();
    let surv = data.followup().unwrap();
    assert!(std::ptr::eq(surv.stop().as_ptr(), stop.values().as_ptr()));
    assert!(std::ptr::eq(
        surv.status().as_ptr(),
        status.values().as_ptr()
    ));
    assert!(std::ptr::eq(
        surv.start().unwrap().as_ptr(),
        start.values().as_ptr()
    ));
    assert!(std::ptr::eq(
        data.weight().unwrap().as_ptr(),
        weight.values().as_ptr()
    ));
    assert!(std::ptr::eq(
        data.istate().unwrap().as_ptr(),
        istate.values().as_ptr()
    ));
    // Dense codes are used as given.
    assert!(std::ptr::eq(
        data.stratum().unwrap().as_ptr(),
        stratum.values().as_ptr()
    ));
    assert!(std::ptr::eq(
        data.cluster().unwrap().as_ptr(),
        dict.keys().values().as_ptr()
    ));

    let lo = Float64Array::from(s.stop.clone());
    let up = Float64Array::from(s.stop.iter().map(|t| t + 1.0).collect::<Vec<_>>());
    let data = interval(&lo, &up).build().unwrap();
    let surv = data.interval().unwrap();
    assert!(std::ptr::eq(surv.lower().as_ptr(), lo.values().as_ptr()));
    assert!(std::ptr::eq(surv.upper().as_ptr(), up.values().as_ptr()));
}

#[test]
fn a_sliced_array_is_borrowed_at_its_offset() {
    let s = sample();
    let stop = Float64Array::from(s.stop.clone());
    let status = Int32Array::from(s.status.clone());
    let (from, len) = (17, 120);
    let stop_cut = stop.slice(from, len);
    let status_cut = status.slice(from, len);
    let data = follow_up(&stop_cut, &status_cut)
        .n_codes(3)
        .build()
        .unwrap();
    let surv = data.followup().unwrap();
    assert_eq!(data.n_rows(), len);
    assert!(std::ptr::eq(
        surv.stop().as_ptr(),
        stop.values()[from..].as_ptr()
    ));
    let want = SurvData::builder(&s.stop[from..from + len], &s.status[from..from + len])
        .n_codes(3)
        .build()
        .unwrap();
    assert_eq!(format!("{data:?}"), format!("{want:?}"));
    // A null outside the slice is not in the data.
    let mut with_null: Vec<Option<f64>> = s.stop.iter().copied().map(Some).collect();
    with_null[2] = None;
    let nullable = Float64Array::from(with_null);
    let cut = nullable.slice(from, len);
    assert!(follow_up(&cut, &status_cut).n_codes(3).build().is_ok());
}

fn message(result: Result<SurvData<'_>, Error>) -> String {
    result.expect_err("an error").to_string()
}

#[test]
fn nulls_are_errors_naming_column_and_row() {
    let status = Int32Array::from(vec![1, 0, 1, 1]);
    let stop_ok = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
    let stop = Float64Array::from(vec![Some(1.0), Some(2.0), None, None]);
    assert_eq!(
        message(follow_up(&stop, &status).build()),
        "column stop has a null at row 2; drop or resolve missing values before building the data"
    );
    let bad_status = Int32Array::from(vec![Some(1), None, Some(1), Some(1)]);
    assert!(
        message(follow_up(&stop_ok, &bad_status).build())
            .starts_with("column status has a null at row 1")
    );
    let weight = Float64Array::from(vec![None, Some(1.0), Some(1.0), Some(1.0)]);
    assert!(
        message(follow_up(&stop_ok, &status).weight(&weight).build())
            .starts_with("column weight has a null at row 0")
    );
    let start = Float64Array::from(vec![Some(0.0), Some(0.0), Some(0.0), None]);
    assert!(
        message(follow_up(&stop_ok, &status).start(&start).build())
            .starts_with("column start has a null at row 3")
    );
    let stratum = UInt32Array::from(vec![Some(0), Some(1), None, Some(0)]);
    assert!(
        message(follow_up(&stop_ok, &status).stratum(&stratum).build())
            .starts_with("column stratum has a null at row 2")
    );
    // A null key and a key that points at a null dictionary value.
    let keys = UInt32Array::from(vec![Some(0), None, Some(1), Some(0)]);
    let values = Int64Array::from(vec![5, 6]);
    let dict = DictionaryArray::new(keys, Arc::new(values));
    assert!(
        message(follow_up(&stop_ok, &status).cluster(&dict).build())
            .starts_with("column cluster has a null at row 1")
    );
    let keys = UInt32Array::from(vec![0, 1, 1, 0]);
    let values = Int64Array::from(vec![Some(5), None]);
    let dict = DictionaryArray::new(keys, Arc::new(values));
    assert!(
        message(follow_up(&stop_ok, &status).id(&dict).build())
            .starts_with("column id has a null at row 1")
    );
    let istate = UInt32Array::from(vec![Some(1), Some(1), None, Some(1)]);
    assert!(
        message(follow_up(&stop_ok, &status).istate(&istate).build())
            .starts_with("column istate has a null at row 2")
    );
    // The interval paths.
    let lower = Float64Array::from(vec![Some(1.0), None]);
    let upper = Float64Array::from(vec![2.0, 3.0]);
    assert!(
        message(interval(&lower, &upper).build()).starts_with("column lower has a null at row 1")
    );
    let st = Int32Array::from(vec![Some(1), None]);
    assert!(
        message(interval_status(&upper, &upper, &st).build())
            .starts_with("column status has a null at row 1")
    );
    // A validity buffer without a null is accepted.
    let all_valid = Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
    assert!(follow_up(&all_valid, &status).build().is_ok());
}

#[test]
fn wrong_types_are_errors_naming_the_column() {
    let status = Int32Array::from(vec![1, 0, 1]);
    let stop = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let f32_stop = Float32Array::from(vec![1.0, 2.0, 3.0]);
    assert_eq!(
        message(follow_up(&f32_stop, &status).build()),
        "column stop has type Float32, expected Float64; cast it before building the data"
    );
    let i64_status = Int64Array::from(vec![1, 0, 1]);
    assert!(
        message(follow_up(&stop, &i64_status).build())
            .starts_with("column status has type Int64, expected Int32")
    );
    let signed = Int32Array::from(vec![0, 1, 2]);
    assert!(
        message(follow_up(&stop, &status).stratum(&signed).build())
            .starts_with("column stratum has type Int32, expected UInt32 or a dictionary")
    );
    let small_keys = DictionaryArray::<arrow_array::types::UInt8Type>::new(
        UInt8Array::from(vec![0, 1, 0]),
        Arc::new(Int64Array::from(vec![1, 2])),
    );
    assert!(
        message(follow_up(&stop, &status).cluster(&small_keys).build())
            .starts_with("column cluster is a dictionary with UInt8 keys, expected UInt32")
    );
    assert!(
        message(follow_up(&stop, &status).istate(&signed).build())
            .starts_with("column istate has type Int32, expected UInt32")
    );
    assert!(
        message(follow_up(&stop, &status).weight(&status).build())
            .starts_with("column weight has type Int32, expected Float64")
    );
    assert!(
        message(interval(&stop, &f32_stop).build()).starts_with("column upper has type Float32")
    );
}

#[test]
fn the_checks_of_surv_data_still_apply() {
    let stop = Float64Array::from(vec![1.0, f64::NAN, 3.0]);
    let status = Int32Array::from(vec![1, 0, 1]);
    assert!(message(follow_up(&stop, &status).build()).contains("stop"));
    let short = Int32Array::from(vec![1, 0]);
    let stop = Float64Array::from(vec![1.0, 2.0, 3.0]);
    assert!(message(follow_up(&stop, &short).build()).contains("status"));
    let status = Int32Array::from(vec![1, 0, 4]);
    assert!(message(follow_up(&stop, &status).build()).contains("status"));
}

#[test]
fn a_record_batch_is_read_by_column_name() {
    let s = sample();
    let batch = RecordBatch::try_from_iter([
        (
            "time",
            Arc::new(Float64Array::from(s.stop.clone())) as ArrayRef,
        ),
        (
            "event",
            Arc::new(Int32Array::from(s.status.clone())) as ArrayRef,
        ),
        ("arm", Arc::new(dictionary(&s.cluster)) as ArrayRef),
    ])
    .unwrap();
    let data = follow_up(
        column(&batch, "time").unwrap(),
        column(&batch, "event").unwrap(),
    )
    .n_codes(3)
    .stratum(column(&batch, "arm").unwrap())
    .build()
    .unwrap();
    let want = SurvData::builder(&s.stop, &s.status)
        .n_codes(3)
        .stratum(&s.cluster)
        .build()
        .unwrap();
    assert_eq!(format!("{data:?}"), format!("{want:?}"));
    let surv = data.followup().unwrap();
    let time = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!(std::ptr::eq(surv.stop().as_ptr(), time.values().as_ptr()));
    let missing = column(&batch, "weight").unwrap_err().to_string();
    assert_eq!(
        missing,
        "column weight is not in the record batch; it has time, event, arm"
    );
}
