//! `brent::minimize` against R's `optimize` (`tests/data/brent.R`): the same
//! number of points is evaluated, in the same order, and the same minimum is
//! returned. `optimize` evaluates the function once more at the minimum to
//! report its value, so its list has one point more than the search's.
//!
//! Golden-section steps agree bit for bit. Parabolic steps can differ in the
//! last bits from R's, which a fused multiply-add in R's build of
//! `Brent_fmin` would explain and which epistat does not use (decision 1).
//! Near the end of a search such a difference can swap the order of the two
//! probes at `x +- tol1`, so each point is compared within `2 tol1`, taken at
//! the larger of the two points and widened by `1e-12` of itself for rounding,
//! the spacing below which the search does not tell points apart, and the
//! minimum within `tol1`.

#![expect(
    clippy::excessive_precision,
    clippy::approx_constant,
    reason = "R's values as R printed them, with 17 significant digits"
)]

use epistat_optim::brent::minimize;

type Case = (&'static str, f64, f64, f64, f64, &'static [f64]);

// R version 4.6.2 (2026-10-29), stats::optimize
const CASES: &[Case] = &[
    (
        "parabola",
        0.0,
        1.0,
        0.000_122_070_312_5,
        0.333_333_333_333_333_31,
        &[
            0.381_966_011_250_105_1,
            0.618_033_988_749_894_79,
            0.236_067_977_499_789_69,
            0.333_333_333_333_333_31,
            0.333_374_028_404_553_71,
            0.333_292_638_262_112_92,
            0.333_333_333_333_333_31,
        ],
    ),
    (
        "sine",
        0.0,
        6.283_185_307_179_586_2,
        1e-10,
        4.712_388_981_532_385_8,
        &[
            2.399_963_229_728_653_1,
            3.883_222_077_450_932_7,
            4.799_926_459_457_306_1,
            5.089_839_033_597_529_6,
            4.705_817_100_294_146,
            4.711_801_320_683_421_5,
            4.712_391_081_200_088_6,
            4.712_388_981_532_385_8,
            4.712_388_911_278_984_5,
            4.712_389_051_785_787_2,
            4.712_388_981_532_385_8,
        ],
    ),
    (
        "kink",
        -1.0,
        3.0,
        1e-08,
        0.699_999_996_576_580_19,
        &[
            0.527_864_045_000_420_39,
            1.472_135_954_999_579_2,
            -0.055_728_090_000_841_224,
            0.703_187_495_328_866_04,
            0.996_899_671_326_200_61,
            0.730_613_356_890_238_21,
            0.665_274_505_028_265_24,
            0.699_066_330_015_949_2,
            0.682_088_738_066_438_44,
            0.697_397_266_903_869_16,
            0.700_103_368_931_155_21,
            0.701_281_400_389_479_95,
            0.700_077_308_076_376_91,
            0.699_809_612_013_412_85,
            0.699_975_057_278_979_13,
            0.699_958_837_526_592_38,
            0.700_006_120_891_350_39,
            0.700_033_311_976_467_12,
            0.700_001_578_864_434_48,
            0.699_995_592_396_186_92,
            0.700_000_274_891_081_85,
            0.699_999_336_920_482_22,
            0.700_000_134_032_610_3,
            0.699_999_922_569_659_37,
            0.699_999_963_913_609_08,
            0.700_000_010_340_726_56,
            0.700_000_057_586_822_04,
            0.699_999_996_576_580_19,
            0.699_999_982_812_434_03,
            0.699_999_996_576_580_19,
        ],
    ),
    (
        "quartic",
        -1.0,
        4.0,
        9.999_999_999_999_999_5e-07,
        2.250_000_082_561_606_4,
        &[
            0.909_830_056_250_525_49,
            2.090_169_943_749_473_6,
            2.819_660_112_501_051,
            1.971_590_536_512_468,
            2.192_384_502_354_649_5,
            2.431_982_465_116_746_5,
            2.240_809_658_254_146_4,
            2.253_678_859_406_362_4,
            2.250_086_285_741_847_1,
            2.249_989_817_981_356_2,
            2.250_000_082_561_606_4,
            2.249_999_715_700_659,
            2.250_000_449_422_553_8,
            2.250_000_082_561_606_4,
        ],
    ),
    (
        "flat edge",
        -2.0,
        1.0,
        9.999_999_999_999_999_5e-08,
        -1.999_999_888_396_977_2,
        &[
            -0.854_101_966_249_684_71,
            -0.145_898_033_750_315_63,
            -1.291_796_067_500_631,
            -1.792_362_798_874_178_2,
            -1.601_163_321_126_921_6,
            -1.871_673_152_375_344_3,
            -1.920_689_646_498_834_1,
            -1.950_983_505_876_510_1,
            -1.969_706_140_622_323_7,
            -1.981_277_365_254_186_4,
            -1.988_428_775_368_137_6,
            -1.992_848_589_886_049_1,
            -1.995_580_185_482_088_5,
            -1.997_268_404_403_960_4,
            -1.998_311_781_078_127_9,
            -1.998_956_623_325_832_2,
            -1.999_355_157_752_295_4,
            -1.999_601_465_573_536_8,
            -1.999_753_692_178_758_9,
            -1.999_847_773_394_778_1,
            -1.999_905_918_783_980_9,
            -1.999_941_854_610_797_2,
            -1.999_964_064_173_183_5,
            -1.999_977_790_437_613_5,
            -1.999_986_273_735_569_8,
            -1.999_991_516_702_043_7,
            -1.999_994_757_033_526_4,
            -1.999_996_759_668_517_3,
            -1.999_997_997_365_008_8,
            -1.999_998_762_303_508_5,
            -1.999_999_235_061_500_6,
            -1.999_999_527_242_008_1,
            -1.999_999_707_819_492_7,
            -1.999_999_819_422_515_7,
            -1.999_999_888_396_977_2,
            -1.999_999_888_396_977_2,
        ],
    ),
];

fn function(name: &str, x: f64) -> f64 {
    match name {
        "parabola" => (x - 1.0 / 3.0).powi(2),
        "sine" => x.sin(),
        "kink" => (x - 0.7).abs(),
        "quartic" => x.powf(4.0) - 3.0 * x.powf(3.0) + 2.0,
        "flat edge" => x.exp(),
        other => panic!("unknown case {other}"),
    }
}

#[test]
fn points_and_minimum_equal_optimize() {
    let eps = f64::EPSILON.sqrt();
    for &(name, lower, upper, tol, minimum, points) in CASES {
        let mut seen = Vec::new();
        let m = minimize(
            |x| {
                seen.push(x);
                function(name, x)
            },
            lower,
            upper,
            tol,
        )
        .unwrap();
        let search = &points[..points.len() - 1];
        assert_eq!(seen.len(), search.len(), "{name}: evaluations");
        assert_eq!(m.evaluations, search.len(), "{name}: count");
        assert_eq!(m.iterations + 1, m.evaluations, "{name}: iterations");
        for (k, (&got, &want)) in seen.iter().zip(search).enumerate() {
            let tol1 = eps * got.abs().max(want.abs()) + tol / 3.0;
            assert!(
                (got - want).abs() <= 2.0 * tol1 * (1.0 + 1e-12),
                "{name}: point {k}, {got} against {want}"
            );
        }
        let tol1 = eps * minimum.abs() + tol / 3.0;
        assert!(
            (m.x - minimum).abs() <= tol1,
            "{name}: minimum {} against {minimum}",
            m.x
        );
    }
}
