//! `root::zeroin` against R's `uniroot` (`tests/data/root.R`): the same points
//! are evaluated in the same order, and the same root and iteration count are
//! returned.
//!
//! Bisection steps agree bit for bit. Interpolation steps can differ from R's
//! in the last bits, which a fused multiply-add in R's build of `R_zeroin2`
//! would explain and which epistat does not use (decision 1). A difference of
//! a few units in the last place in one step changes every later point by
//! about as much relative to the step, so each point is compared within
//! `1e-12` of its magnitude, far below the tolerance of any case and far above
//! the `2^-52` rounding of one step; the iteration counts must be equal.
//!
//! `root::bisect` against survival's `log_product_limit` bisection, copied
//! here as it is written there: the same root to the bit.

use epistat_optim::root::{Bisection, Zeroin, bisect, zeroin};

type Case = (&'static str, f64, f64, f64, f64, usize, &'static [f64]);

// R version 4.6.2 (2026-10-29), stats::uniroot
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "R's values as R printed them, with 17 significant digits"
)]
const CASES: &[Case] = &[
    (
        "cubic",
        1.0,
        2.0,
        0.0001220703125,
        1.5213860590522479,
        6,
        &[
            1.3333333333333333,
            1.6666666666666665,
            1.5,
            1.5191489361702128,
            1.5213860590522479,
            1.5213250238959972,
        ],
    ),
    (
        "cosine",
        0.0,
        1.0,
        9.9999999999999998e-13,
        0.7390851332151559,
        6,
        &[
            0.68507335732604513,
            0.73629899761365403,
            0.73909105828536981,
            0.73908512956477512,
            0.7390851332151559,
            0.73908513321565628,
        ],
    ),
    (
        "exponential",
        0.0,
        3.0,
        1.0000000000000001e-09,
        1.6094379124145011,
        8,
        &[
            0.62874835789507144,
            1.8143741789475356,
            1.4980102695802378,
            1.5982192496742944,
            1.6095214300422174,
            1.6094374430867704,
            1.6094379124145011,
            1.6094379129145018,
        ],
    ),
    (
        "flat cube",
        0.0,
        1.0,
        1e-10,
        0.33333333334016274,
        69,
        &[
            0.11111111111111108,
            0.15622389306599829,
            0.57811194653299913,
            0.27212887848142275,
            0.27704126718865996,
            0.42757660686082954,
            0.30348550831768178,
            0.30797557618290922,
            0.36777609152186941,
            0.325032754007125,
            0.32564421606555238,
            0.3467101537937109,
            0.32900649532537479,
            0.32971165999639779,
            0.33821090689505434,
            0.33218037905653103,
            0.33226161054724895,
            0.33523625872115165,
            0.33271246465185972,
            0.33281762544075211,
            0.33402694208095185,
            0.33316989339794517,
            0.33318132587328797,
            0.33360413397711991,
            0.33324486799677616,
            0.33325994278642901,
            0.33343203838177449,
            0.33331007621492598,
            0.33331170279671796,
            0.33337187058924622,
            0.33332074364810055,
            0.3333228892286122,
            0.33334737990892921,
            0.33333002364574377,
            0.33333025512229869,
            0.33333881751561395,
            0.33333154171307205,
            0.33333184704759,
            0.33333533228160195,
            0.33333286233638243,
            0.33333289527748294,
            0.33333411377954247,
            0.33333307837039267,
            0.33333312182211527,
            0.33333361780082887,
            0.33333326630642462,
            0.33333327099422594,
            0.33333344439752743,
            0.33333329704991854,
            0.3333333032334716,
            0.33333337381549955,
            0.33333332379482888,
            0.33333332446194325,
            0.33333334913872137,
            0.33333332816989197,
            0.33333332904986462,
            0.33333333909429297,
            0.33333333197592235,
            0.33333333207085841,
            0.33333333558257572,
            0.33333333259853126,
            0.33333333272375892,
            0.33333333415316735,
            0.33333333314016206,
            0.33333333319016223,
            0.33333333367166479,
            0.3333333332401624,
            0.33333333329016257,
            0.33333333334016274,
        ],
    ),
    (
        "decreasing",
        0.0,
        10.0,
        1e-08,
        2.3333333333827984,
        9,
        &[
            7.7000000000000002,
            3.8500000000000001,
            1.925,
            2.5191250000000003,
            2.3560928125,
            2.333177935165752,
            2.3333343943677409,
            2.3333333333827984,
            2.3333333283827975,
        ],
    ),
    (
        "step",
        0.0,
        1.0,
        9.9999999999999995e-07,
        0.59999942779541016,
        20,
        &[
            0.5,
            0.75,
            0.625,
            0.5625,
            0.59375,
            0.609375,
            0.6015625,
            0.59765625,
            0.599609375,
            0.6005859375,
            0.60009765625,
            0.599853515625,
            0.5999755859375,
            0.60003662109375,
            0.600006103515625,
            0.5999908447265625,
            0.59999847412109375,
            0.60000228881835938,
            0.60000038146972656,
            0.59999942779541016,
        ],
    ),
];

fn function(name: &str, x: f64) -> f64 {
    match name {
        "cubic" => x.powi(3) - x - 2.0,
        "cosine" => x.cos() - x,
        "exponential" => x.exp() - 5.0,
        "flat cube" => (x - 1.0 / 3.0).powi(3),
        "decreasing" => 1.0 / (1.0 + x) - 0.3,
        "step" => {
            if x < 0.6 {
                -1.0
            } else {
                1.0
            }
        }
        other => panic!("unknown case {other}"),
    }
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-12 * want.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn points_root_and_iterations_equal_uniroot() {
    for &(name, lower, upper, tol, root, iterations, points) in CASES {
        let mut seen = Vec::new();
        let found = zeroin(
            |x| {
                seen.push(x);
                function(name, x)
            },
            lower,
            upper,
            &Zeroin {
                tolerance: tol,
                max_iterations: 1_000,
            },
        )
        .unwrap();
        assert_eq!(&seen[..2], &[lower, upper], "{name}: the ends first");
        let search = &seen[2..];
        assert_eq!(search.len(), points.len(), "{name}: evaluations");
        for (k, (&got, &want)) in search.iter().zip(points).enumerate() {
            assert!(close(got, want), "{name}: point {k}, {got} against {want}");
        }
        assert!(
            close(found.x, root),
            "{name}: root {} against {root}",
            found.x
        );
        assert_eq!(found.iterations, iterations, "{name}: iterations");
        assert!(found.converged, "{name}");
        assert_eq!(found.evaluations, seen.len(), "{name}: count");
    }
}

/// epistat-survival `cox/predict.rs`, the bisection of `log_product_limit`,
/// as written there after its bracket is found.
fn survival_bisection(g: impl Fn(f64) -> f64, s0: f64, mut lo: f64, mut hi: f64) -> f64 {
    for _ in 0..2200 {
        let mid = lo + (hi - lo) / 2.0;
        if mid <= lo || mid >= hi {
            break;
        }
        if g(mid) < s0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 2.0 * f64::EPSILON * mid.abs() {
            break;
        }
    }
    lo + (hi - lo) / 2.0
}

#[test]
fn bisection_is_survivals_to_the_bit() {
    let cases: [(&[f64], &[f64], f64, f64); 4] = [
        (&[1.0, 1.0], &[0.5, 2.0], 4.0, -1.0),
        (&[0.3, 1.7, 2.0], &[1.1, 0.4, 0.9], 6.5, -4.0),
        (&[1.0; 5], &[1.0, 1.0, 1.0, 1.0, 1.0], 5.000_001, -64.0),
        (&[2.5, 0.5], &[3.0, 0.01], 9.0, -2.0),
    ];
    for (w, theta, s0, lo) in cases {
        let g = |l: f64| {
            let mut total = 0.0;
            for (w, t) in w.iter().zip(theta) {
                total += w * t / -(t * l).exp_m1();
            }
            total
        };
        let hi = -f64::MIN_POSITIVE;
        assert!(g(lo) < s0 && g(hi) >= s0);
        let want = survival_bisection(g, s0, lo, hi);
        let got = bisect(|l| g(l) - s0, lo, hi, &Bisection::default()).unwrap();
        assert_eq!(got.x.to_bits(), want.to_bits(), "{w:?} {theta:?} {s0}");
        assert!(got.converged);
    }
}
