//! `split`, `tmerge` and `condense` checked against `survival::survSplit`,
//! `tmerge` and `survcondense`. They produce times, indices and sums of a few
//! numbers in table order, so every output must agree exactly, bit for bit.

use epistat_survival::SurvData;
use epistat_survival::condense::{CondenseOptions, condense};
use epistat_survival::split::{SplitOptions, split};
use epistat_survival::timefix::TimeFix;
use epistat_survival::tmerge::{Counts, Observations, Spans, Tmerge, TmergeOptions};
use epistat_testkit::{bits, load_json, na_f64_vec};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    r_version: String,
    survival_version: String,
    split: Vec<SplitCase>,
    tmerge: Vec<TmCase>,
    condense: Vec<CondenseCase>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/reshape.json"
    ))
}

fn opt_bits(values: &[Option<f64>]) -> Vec<Option<u64>> {
    values.iter().map(|v| v.map(f64::to_bits)).collect()
}

#[test]
fn reference_versions() {
    let r = reference();
    assert!(r.r_version.starts_with("R version 4.6"));
    assert!(r.survival_version.starts_with("3.8"));
}

#[derive(Deserialize)]
struct SplitCase {
    name: String,
    counting: bool,
    zero: f64,
    timefix: bool,
    #[serde(deserialize_with = "na_f64_vec")]
    cut: Vec<f64>,
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    status: Vec<i32>,
    error: Option<String>,
    out_start: Option<Vec<f64>>,
    out_stop: Option<Vec<f64>>,
    out_status: Option<Vec<i32>>,
    out_row: Option<Vec<u32>>,
    out_episode: Option<Vec<u32>>,
    out_added: Option<Vec<bool>>,
}

#[test]
fn split_matches_survsplit_exactly() {
    let mut compared = 0;
    for case in &reference().split {
        let mut builder = SurvData::builder(&case.stop, &case.status).n_codes(1);
        if case.counting {
            builder = builder.start(case.start.as_ref().unwrap());
        }
        let options = SplitOptions {
            zero: case.zero,
            timefix: if case.timefix {
                TimeFix::default()
            } else {
                TimeFix::Exact
            },
        };
        let got = builder
            .build()
            .map_err(|e| e.to_string())
            .and_then(|data| split(&data, &case.cut, &options).map_err(|e| e.to_string()));
        if case.error.is_some() {
            assert!(got.is_err(), "{}: R refuses, this accepts", case.name);
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let name = &case.name;
        assert_eq!(
            bits(&got.start),
            bits(case.out_start.as_ref().unwrap()),
            "{name} start"
        );
        assert_eq!(
            bits(&got.stop),
            bits(case.out_stop.as_ref().unwrap()),
            "{name} stop"
        );
        assert_eq!(
            &got.status,
            case.out_status.as_ref().unwrap(),
            "{name} status"
        );
        assert_eq!(&got.row, case.out_row.as_ref().unwrap(), "{name} row");
        assert_eq!(
            &got.episode,
            case.out_episode.as_ref().unwrap(),
            "{name} episode"
        );
        assert_eq!(
            &got.ends_at_cut,
            case.out_added.as_ref().unwrap(),
            "{name} added"
        );
        compared += 1;
    }
    assert!(compared >= 25);
}

#[derive(Deserialize)]
struct Op {
    kind: String,
    name: String,
    id: Vec<u32>,
    time: Vec<Option<f64>>,
    value: Option<Vec<Option<f64>>>,
    init: Option<f64>,
}

#[derive(Deserialize)]
struct SpanIn {
    id: Vec<u32>,
    tstart: Option<Vec<f64>>,
    tstop: Vec<f64>,
}

#[derive(Deserialize)]
struct Var {
    name: String,
    values: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct CountsIn {
    early: usize,
    late: usize,
    gap: usize,
    within: usize,
    boundary: usize,
    leading: usize,
    trailing: usize,
    tied: usize,
    missid: usize,
}

#[derive(Deserialize)]
struct TmCase {
    name: String,
    base_id: Vec<u32>,
    delay: f64,
    na_rm: bool,
    tdcstart: Option<f64>,
    ops: Vec<Op>,
    span: Option<SpanIn>,
    error: Option<String>,
    out_id: Option<Vec<u32>>,
    out_row: Option<Vec<u32>>,
    out_tstart: Option<Vec<f64>>,
    out_tstop: Option<Vec<f64>>,
    out_vars: Option<Vec<Var>>,
    out_counts: Option<Vec<CountsIn>>,
}

fn run_tmerge(case: &TmCase) -> Result<Tmerge, String> {
    let options = TmergeOptions {
        delay: case.delay,
        na_rm: case.na_rm,
        tdcstart: case.tdcstart,
    };
    let mut tm = if let Some(span) = &case.span {
        Tmerge::new(
            &case.base_id,
            &Spans {
                id: &span.id,
                tstart: span.tstart.as_deref(),
                tstop: &span.tstop,
            },
            options,
        )
    } else {
        let first = &case.ops[0];
        Tmerge::from_first_events(&case.base_id, &first.id, &first.time, options)
    }
    .map_err(|e| e.to_string())?;
    for op in &case.ops {
        let obs = Observations {
            id: &op.id,
            time: &op.time,
            value: op.value.as_deref(),
        };
        match op.kind.as_str() {
            "tdc" => tm.tdc(&op.name, &obs, op.init),
            "cumtdc" => tm.cumtdc(&op.name, &obs, op.init),
            "event" => tm.event(&op.name, &obs),
            "cumevent" => tm.cumevent(&op.name, &obs),
            other => panic!("{other}"),
        }
        .map_err(|e| e.to_string())?;
    }
    Ok(tm)
}

fn same_counts(got: &Counts, want: &CountsIn) -> bool {
    [
        got.early,
        got.late,
        got.gap,
        got.within,
        got.boundary,
        got.leading,
        got.trailing,
        got.tied,
        got.missid,
    ] == [
        want.early,
        want.late,
        want.gap,
        want.within,
        want.boundary,
        want.leading,
        want.trailing,
        want.tied,
        want.missid,
    ]
}

#[test]
fn tmerge_matches_tmerge_exactly() {
    let (mut compared, mut refused, mut skipped) = (0, 0, 0);
    for case in &reference().tmerge {
        let name = &case.name;
        let got = run_tmerge(case);
        if let Some(message) = &case.error {
            // R stops when no observation of an operation is valid. Here the
            // operation changes nothing and the call succeeds.
            if message.contains("No valid entries") {
                skipped += 1;
                continue;
            }
            assert!(
                got.is_err(),
                "{name}: R refuses with {message}, this accepts"
            );
            refused += 1;
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(&got.id(), case.out_id.as_ref().unwrap(), "{name} id");
        assert_eq!(&got.row(), case.out_row.as_ref().unwrap(), "{name} row");
        assert_eq!(
            bits(got.tstart()),
            bits(case.out_tstart.as_ref().unwrap()),
            "{name} tstart"
        );
        assert_eq!(
            bits(got.tstop()),
            bits(case.out_tstop.as_ref().unwrap()),
            "{name} tstop"
        );
        let want = case.out_vars.as_ref().unwrap();
        assert_eq!(got.variables().len(), want.len(), "{name} variables");
        for w in want {
            let v = got
                .variable(&w.name)
                .unwrap_or_else(|| panic!("{name}: no variable {}", w.name));
            assert_eq!(
                opt_bits(&v.values),
                opt_bits(&w.values),
                "{name} {}",
                w.name
            );
        }
        let want_counts = case.out_counts.as_ref().unwrap();
        assert_eq!(got.counts().len(), want_counts.len(), "{name} counts");
        for (i, (g, w)) in got.counts().iter().zip(want_counts).enumerate() {
            assert!(same_counts(&g.1, w), "{name} counts of call {i}");
        }
        compared += 1;
    }
    assert!(compared >= 90, "{compared} compared");
    assert!(refused >= 8, "{refused} refused");
    assert!(skipped <= 8);
}

#[derive(Deserialize)]
struct CondenseCase {
    name: String,
    merge_over_events: bool,
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    x1: Vec<u64>,
    x2: Vec<u64>,
    error: Option<String>,
    out_start: Option<Vec<f64>>,
    out_stop: Option<Vec<f64>>,
    out_status: Option<Vec<i32>>,
    out_row: Option<Vec<u32>>,
}

#[test]
fn condense_matches_survcondense_exactly() {
    let (mut merged, mut unmerged) = (0, 0);
    for case in &reference().condense {
        let name = &case.name;
        assert!(case.error.is_none(), "{name}");
        let key: Vec<u64> = case
            .x1
            .iter()
            .zip(&case.x2)
            .map(|(a, b)| a * 1000 + b)
            .collect();
        let data = SurvData::builder(&case.stop, &case.status)
            .start(&case.start)
            .id(&case.id)
            .n_codes(1)
            .build()
            .unwrap();
        let got = condense(
            &data,
            &key,
            &CondenseOptions {
                merge_over_events: case.merge_over_events,
            },
        )
        .unwrap();
        let want_row = case.out_row.as_ref().unwrap();
        if want_row.is_empty() {
            // R returns no rows at all when nothing is merged, because it
            // drops rows with `x[-integer(0), ]`. Here every row stays.
            let n = u32::try_from(case.stop.len()).unwrap();
            assert_eq!(got.row, (0..n).collect::<Vec<_>>(), "{name}");
            assert_eq!(bits(&got.start), bits(&case.start), "{name}");
            assert_eq!(bits(&got.stop), bits(&case.stop), "{name}");
            assert_eq!(got.status, case.status, "{name}");
            unmerged += 1;
            continue;
        }
        assert_eq!(&got.row, want_row, "{name} row");
        assert_eq!(
            bits(&got.start),
            bits(case.out_start.as_ref().unwrap()),
            "{name} start"
        );
        assert_eq!(
            bits(&got.stop),
            bits(case.out_stop.as_ref().unwrap()),
            "{name} stop"
        );
        assert_eq!(
            &got.status,
            case.out_status.as_ref().unwrap(),
            "{name} status"
        );
        merged += 1;
    }
    assert!(
        merged >= 15 && unmerged >= 2,
        "{merged} merged, {unmerged} not"
    );
}
