//! `surv2::counting` checked against R's `fromtimeline`, which is
//! `survival:::surv2counting`, the conversion `coxph`, `survfit` and `survcheck`
//! run on a `Surv2` response, in survival 3.8-12.
//!
//! The cases are the examples of the `Surv2` help page, hand-made edge cases,
//! the `pbcseq` timelines of the `timeline` vignette, and 60 randomized
//! timelines with unsorted rows, missing statuses and covariates, binary and
//! multi-state statuses, and every choice of `repeated` and `lvcf`. See
//! `data/surv2.R`. Where R stops with an error, so must the Rust function.
//!
//! # Tolerances
//!
//! None. Every output is a time or covariate copied from the input, a
//! row number, or a status code, so they are compared exactly.

use epistat_survival::surv2::{
    Counting, Repeated, StatusKind, Timeline, TimelineOptions, counting,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    m: u32,
    repeated: String,
    lvcf: bool,
    input: Input,
    error: Option<String>,
    result: Option<Result>,
}

#[derive(Deserialize)]
struct Input {
    id: Vec<u32>,
    time: Vec<f64>,
    status: Vec<Option<u32>>,
    covs: Vec<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct Result {
    row: Vec<u32>,
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    status: Vec<Option<u32>>,
    istate: Option<Vec<u32>>,
    covs: Vec<Vec<Option<f64>>>,
}

fn options(case: &Case) -> TimelineOptions {
    TimelineOptions {
        status: if case.m == 0 {
            StatusKind::Binary
        } else {
            StatusKind::States(case.m)
        },
        repeated: match case.repeated.as_str() {
            "events" => Repeated::Events,
            "collapse" => Repeated::Collapse,
            "first" => Repeated::First,
            other => panic!("repeated {other}"),
        },
        lvcf: case.lvcf,
    }
}

fn run(case: &Case) -> epistat_survival::Result<Counting> {
    let columns: Vec<Vec<f64>> = case
        .input
        .covs
        .iter()
        .map(|c| c.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        .collect();
    let cols: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    counting(
        &Timeline {
            id: &case.input.id,
            time: &case.input.time,
            status: &case.input.status,
            covariates: &cols,
        },
        &options(case),
    )
}

#[test]
fn counting_matches_fromtimeline() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/surv2.json"
    ));
    let (mut ok, mut errors) = (0, 0);
    for case in &r.cases {
        let got = run(case);
        if let Some(message) = &case.error {
            assert!(got.is_err(), "{}: R stops with `{message}`", case.name);
            errors += 1;
            continue;
        }
        let want = case.result.as_ref().unwrap();
        let got = got.unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let name = &case.name;
        assert_eq!(got.row, want.row, "{name}: row");
        assert_eq!(got.start, want.start, "{name}: start");
        assert_eq!(got.stop, want.stop, "{name}: stop");
        assert_eq!(got.status, want.status, "{name}: status");
        assert_eq!(got.istate, want.istate, "{name}: istate");
        assert_eq!(got.covariates, want.covs, "{name}: covariates");
        // The interval ends at the next observation of the same subject.
        for i in 0..got.len() {
            let (a, b) = (got.row[i] as usize, got.stop_row[i] as usize);
            assert_eq!(case.input.id[a], case.input.id[b]);
            assert_eq!(case.input.id[a], got.id[i]);
            assert_eq!(case.input.time[b].to_bits(), got.stop[i].to_bits());
            if let Some(start) = &got.start {
                assert_eq!(case.input.time[a].to_bits(), start[i].to_bits());
            }
            assert!(case.input.time[a] < case.input.time[b]);
        }
        ok += 1;
    }
    eprintln!("{ok} timelines converted and {errors} refused, as R");
    assert!(ok >= 80 && errors >= 3);
}

/// The example of the `Surv2` help page: the states (CU, CU, MCI, CU, MCI, MCI,
/// dementia) with `repeated = FALSE` become (CU, censor, MCI, CU, MCI, censor,
/// dementia) and with `repeated = "first"` (CU, censor, MCI, censor, censor,
/// censor, dementia). The first status is the initial state, so the intervals
/// carry the statuses of the second to the last observation.
#[test]
fn the_help_page_example() {
    let status: Vec<Option<u32>> = [1, 1, 2, 1, 2, 2, 3].map(Some).to_vec();
    let id = vec![1_u32; 7];
    let time: Vec<f64> = (1..=7).map(f64::from).collect();
    let run = |repeated| {
        counting(
            &Timeline {
                id: &id,
                time: &time,
                status: &status,
                covariates: &[],
            },
            &TimelineOptions {
                status: StatusKind::States(3),
                repeated,
                lvcf: true,
            },
        )
        .unwrap()
    };
    let of = |c: Counting| c.status.iter().map(|s| s.unwrap()).collect::<Vec<_>>();
    assert_eq!(of(run(Repeated::Collapse)), [0, 2, 1, 2, 0, 3]);
    assert_eq!(of(run(Repeated::First)), [0, 2, 0, 0, 0, 3]);
    assert_eq!(of(run(Repeated::Events)), [1, 2, 1, 2, 2, 3]);
}
