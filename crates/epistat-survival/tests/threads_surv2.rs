//! The conversion of timelines to intervals gives identical bits in rayon pools
//! of 1, 2, 3, 4 and 8 threads, with unsorted rows, missing statuses and
//! covariates, and enough rows for rayon's parallel sorts to split.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the grid values and counts are far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::surv2::{Repeated, StatusKind, Timeline, TimelineOptions, counting};
use epistat_testkit::{POOLS, absent, assert_same_on_pools};

#[test]
fn counting_does_not_depend_on_the_number_of_threads() {
    let mut rng = Stream::seeded(77);
    let (mut id, mut time, mut status, mut x) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for subject in 0..9001_u32 {
        let k = 1 + rng.below(5);
        for j in 0..k {
            id.push(subject * 7 + 3);
            time.push(j as f64 * 2.0 + f64::from(rng.below(2) as u32) * 0.5);
            status.push(match rng.below(10) {
                0 => None,
                1..=3 => Some(rng.below(3) as u32 + 1),
                _ => Some(0),
            });
            x.push(if rng.below(4) == 0 {
                f64::NAN
            } else {
                rng.uniform()
            });
        }
    }
    // Put the rows in a scrambled order, in a fixed way.
    let n = id.len();
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rng.below(i + 1));
    }
    let pick = |v: &[f64]| perm.iter().map(|&i| v[i]).collect::<Vec<_>>();
    let id: Vec<u32> = perm.iter().map(|&i| id[i]).collect();
    let status: Vec<Option<u32>> = perm.iter().map(|&i| status[i]).collect();
    let (time, x) = (pick(&time), pick(&x));
    for (kind, repeated) in [
        (StatusKind::Binary, Repeated::Events),
        (StatusKind::States(3), Repeated::Collapse),
        (StatusKind::States(3), Repeated::First),
    ] {
        let status: Vec<Option<u32>> = if kind == StatusKind::Binary {
            status.iter().map(|s| s.map(|v| u32::from(v > 0))).collect()
        } else {
            status.clone()
        };
        assert_same_on_pools(&POOLS, || {
            let covs = [x.as_slice()];
            // Equal times within a subject make this refuse, so jitter them out.
            let t: Vec<f64> = time
                .iter()
                .zip(0..)
                .map(|(t, i)| t + f64::from(i) * 1e-9)
                .collect();
            let got = counting(
                &Timeline {
                    id: &id,
                    time: &t,
                    status: &status,
                    covariates: &covs,
                },
                &TimelineOptions {
                    status: kind,
                    repeated,
                    lvcf: true,
                },
            );
            let mut out = Vec::new();
            match got {
                Err(e) => out.push(e.to_string().len() as f64),
                Ok(c) => {
                    out.extend(c.id.iter().map(|&v| f64::from(v)));
                    out.extend(c.start.iter().flatten());
                    out.extend(&c.stop);
                    out.extend(c.status.iter().map(|s| s.map_or(absent(None), f64::from)));
                    out.extend(c.istate.iter().flatten().map(|&v| f64::from(v)));
                    for col in &c.covariates {
                        out.extend(col.iter().copied().map(absent));
                    }
                    out.extend(c.row.iter().map(|&v| f64::from(v)));
                    out.extend(c.stop_row.iter().map(|&v| f64::from(v)));
                }
            }
            out
        });
    }
}
