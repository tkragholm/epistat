//! Draws checked against an independent Python implementation of the same
//! stream, seeded with 20261119.

use epistat_rng::{Stream, dirichlet_weights, replicate_seed, resample_counts};

const SEED: u64 = 20_261_119;

#[test]
fn raw_outputs() {
    let mut stream = Stream::seeded(SEED);
    let outputs: Vec<u64> = (0..4).map(|_| stream.next_u64()).collect();
    assert_eq!(
        outputs,
        [
            0x66DD_8061_A4A6_372B,
            0xBDC3_1D10_C7E7_DB52,
            0x77F9_83A2_BF69_AD49,
            0xF666_38C1_5FE9_CBB7,
        ]
    );
}

#[test]
fn uniforms() {
    let mut stream = Stream::seeded(SEED);
    let bits: Vec<u64> = (0..3).map(|_| stream.uniform().to_bits()).collect();
    assert_eq!(
        bits,
        [
            0x3FD9_B760_1869_298C,
            0x3FE7_B863_A218_FCFB,
            0x3FDD_FE60_E8AF_DA6A
        ]
    );
}

#[test]
fn bounded_integers() {
    let mut stream = Stream::seeded(SEED);
    let values: Vec<usize> = (0..6).map(|_| stream.below(1_000)).collect();
    assert_eq!(values, [401, 741, 468, 962, 754, 166]);
}

#[test]
fn shuffle() {
    let mut items: Vec<u8> = (0..10).collect();
    Stream::seeded(SEED).shuffle(&mut items);
    assert_eq!(items, [7, 5, 1, 2, 0, 9, 8, 3, 6, 4]);
}

#[test]
fn replicate_seeds() {
    assert_eq!(replicate_seed(SEED, 3), 0xB620_93D6_5767_7A54);
}

#[test]
fn resample() {
    assert_eq!(resample_counts(8, SEED), [0, 1, 0, 2, 0, 3, 1, 1]);
}

#[test]
fn dirichlet() {
    // These go through `ln`, so they are held to two units in the last place
    // rather than to the bit: libm may round differently between platforms.
    let expected: [u64; 4] = [
        0x3FD6_C0DE_D604_9899,
        0x3FED_EE67_2531_0590,
        0x3FDB_FFDF_7C3A_4858,
        0x4002_2C4E_6C6B_E27D,
    ];
    let mut weights = Vec::new();
    dirichlet_weights(4, SEED, &mut weights);
    for (weight, bits) in weights.iter().zip(expected) {
        assert!(weight.to_bits().abs_diff(bits) <= 2, "{weight}");
    }
}
