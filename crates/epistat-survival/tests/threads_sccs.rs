//! `sccs::build_pieces` and `sccs::fit` give identical bits in rayon pools of
//! 1, 2, 3, 4 and 8 threads.
//!
//! The data are seeded, with 3,000 people so that they fall into six blocks of
//! 512 people, each handled by its own task, a risk window with a pre-exposure
//! part, age bands and one to four events per person.

use epistat_rng::Stream;
use epistat_survival::sccs::{PieceSpec, SccsOptions, WindowSpec, build_pieces, fit};
use epistat_testkit::{POOLS, assert_same_on_pools};

#[test]
fn sccs_does_not_depend_on_the_number_of_threads() {
    let mut rng = Stream::seeded(61);
    let n = 3_000_usize;
    let (mut start, mut end) = (vec![], vec![]);
    let (mut event_offsets, mut events) = (vec![0_usize], vec![]);
    let (mut exposure_offsets, mut exposures) = (vec![0_usize], vec![]);
    for _ in 0..n {
        let a = rng.below(100) as f64;
        let length = 300 + rng.below(400);
        let b = a + length as f64;
        start.push(a);
        end.push(b);
        let doses = rng.below(3);
        for _ in 0..doses {
            exposures.push(a + rng.below(300) as f64);
        }
        exposure_offsets.push(exposures.len());
        let k = 1 + rng.below(4);
        for _ in 0..k {
            events.push(a + rng.below(length) as f64);
        }
        event_offsets.push(events.len());
    }
    let windows = [
        WindowSpec {
            from: -10.0,
            to: 0.0,
        },
        WindowSpec {
            from: 0.0,
            to: 14.0,
        },
        WindowSpec {
            from: 14.0,
            to: f64::INFINITY,
        },
    ];
    let spec = PieceSpec {
        start: &start,
        end: &end,
        event_offsets: &event_offsets,
        events: &events,
        exposure_offsets: &exposure_offsets,
        exposures: &exposures,
        age_cuts: &[150.0, 300.0, 450.0],
        windows: &windows,
    };
    assert_same_on_pools(&POOLS, || {
        let pieces = build_pieces(&spec).unwrap();
        let length = pieces.length();
        let x = pieces.design();
        let opts = SccsOptions {
            robust: true,
            ..SccsOptions::default()
        };
        let f = fit(&pieces.data(&length), x.as_ref(), &opts).unwrap();
        let mut out = vec![];
        out.extend(&pieces.start);
        out.extend(&pieces.end);
        out.extend(&pieces.events);
        out.extend(f.coefficients.iter().map(|c| c.unwrap()));
        out.extend(f.covariance.iter().map(|c| c.unwrap()));
        out.extend(
            f.robust_covariance
                .as_ref()
                .unwrap()
                .iter()
                .map(|c| c.unwrap()),
        );
        out.extend(f.log_likelihood);
        out.extend(f.null_log_likelihood);
        out.push(f.iterations as f64);
        out
    });
}
