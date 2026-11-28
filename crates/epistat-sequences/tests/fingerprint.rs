//! The main results of the crate, hashed to the bit on a fixed panel and fixed
//! seeds: the dynamic Hamming and optimal-matching distances, the k-medoids
//! fits and their silhouettes under each distance, the typologies, the null
//! draws and null silhouettes, and the refit agreements.
//!
//! The digest is the same on any thread count, which is checked everywhere.
//! The costs go through `ln`, whose last bit depends on the maths library, so
//! the recorded value is compared only on aarch64 macOS. To record it again,
//! run `GOLDEN_PRINT=1 cargo test -p epistat-sequences --test fingerprint -- --nocapture`
//! on that target and paste the printed constant.

use epistat_core::{Bootstrap, Seed};
use epistat_sequences::positionwise::{self, Panel};
use epistat_sequences::{
    Attempt, Distance, FitSettings, NullModel, Restarts, condensed, fit_typology, indel_for,
    null_sequences, null_silhouettes, partition_agreement, refit_agreements, substitution_costs,
    transition_costs,
};
use epistat_testkit::{Fingerprint, SplitMix64, on_threads};

/// The digest recorded on aarch64 macOS.
const RECORDED: Option<u64> = Some(0xc361_bd8a_73f8_b1fb);

const COUNT: usize = 36;
const LENGTH: usize = 10;
const ALPHABET: usize = 4;
const K: usize = 3;

/// Sequences that stay in a state for a few positions and then move on, so
/// that the transition and substitution costs differ between states.
fn encoded() -> Vec<u8> {
    let mut generator = SplitMix64::new(20_261_128);
    let mut symbols = Vec::with_capacity(COUNT * LENGTH);
    for _ in 0..COUNT {
        let mut state = generator.below(ALPHABET as u64);
        for _ in 0..LENGTH {
            if generator.unit() < 0.3 {
                state = (state + 1 + generator.below(2)) % ALPHABET as u64;
            }
            symbols.push(u8::try_from(state).unwrap());
        }
    }
    symbols
}

fn weights() -> Vec<f64> {
    let mut generator = SplitMix64::new(7);
    (0..COUNT).map(|_| generator.uniform(0.5, 3.0)).collect()
}

fn attempt(h: Fingerprint, attempt: &Attempt) -> Fingerprint {
    let h = attempt
        .medoids
        .iter()
        .chain(&attempt.labels)
        .fold(h, |h, &m| h.u64(u64::from(m)));
    h.u64(attempt.iterations as u64)
        .u64(u64::from(attempt.converged))
        .f64(attempt.objective)
}

fn settings(distance: Distance) -> FitSettings {
    FitSettings {
        distance,
        k: K,
        restarts: 4,
        max_iterations: 50,
    }
}

fn fingerprint() -> u64 {
    let encoded = encoded();
    let weights = weights();
    let panel = Panel::new(&encoded, LENGTH, ALPHABET, None).unwrap();
    let restarts = Restarts::from_seed(11, 4).unwrap();
    let mut h = Fingerprint::new();

    // Dynamic Hamming: the costs, the fit, its sums and silhouette.
    let costs = transition_costs(&panel, &weights).unwrap();
    h = h.f64s(&costs);
    let costed = Panel::new(&encoded, LENGTH, ALPHABET, Some(&costs)).unwrap();
    let fit = positionwise::kmedoids(&costed, &weights, K, &restarts, 50).unwrap();
    h = attempt(h, &fit);
    h = h.f64s(&positionwise::cluster_distance_sums(&costed, &weights, &fit.labels, K).unwrap());
    h = h.option_f64(positionwise::silhouette(&costed, &weights, &fit.labels).unwrap());

    // Optimal matching: the costs, the triangle, the fit and its silhouette.
    let substitution =
        substitution_costs((0..COUNT).map(|i| panel.at(i)), &weights, ALPHABET).unwrap();
    h = h.f64s(&substitution);
    let indel = indel_for(&substitution, ALPHABET).unwrap();
    h = h.f64(indel);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the triangle holds single-precision costs, as the fits narrow them"
    )]
    let narrowed: Vec<f32> = substitution.iter().map(|&c| c as f32).collect();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the triangle holds single-precision costs, as the fits narrow them"
    )]
    let indel = indel as f32;
    let lengths = vec![u32::try_from(LENGTH).unwrap(); COUNT];
    let matrix = condensed(&encoded, &lengths, &narrowed, ALPHABET, indel).unwrap();
    h = matrix
        .values()
        .iter()
        .fold(h, |h, &d| h.u64(u64::from(d.to_bits())));
    let offsets: Vec<u32> = (0..u32::try_from(COUNT).unwrap()).collect();
    let fit = matrix
        .kmedoids(&offsets, &weights, K, &restarts, 50)
        .unwrap();
    h = attempt(h, &fit);
    h = h.f64s(
        &matrix
            .cluster_distance_sums(&weights, &fit.labels, K)
            .unwrap(),
    );
    h = h.option_f64(matrix.silhouette(&offsets, &weights, &fit.labels).unwrap());

    // The one-call typologies under each distance.
    let mut reference = Vec::new();
    for distance in [
        Distance::Hamming,
        Distance::DynamicHamming,
        Distance::OptimalMatching,
    ] {
        let typology = fit_typology(&panel, &weights, settings(distance), 3).unwrap();
        h = attempt(h, &typology.attempt);
        h = h.option_f64(typology.silhouette);
        if distance == Distance::Hamming {
            reference = typology.attempt.labels;
        }
    }

    // The validation draws: null sequences and silhouettes under each model,
    // and refits under replicate weights.
    for model in NullModel::ALL {
        let drawn = null_sequences(&panel, &weights, model, 19).unwrap();
        h = drawn.iter().fold(h, |h, &s| h.u64(u64::from(s)));
        for distance in [Distance::Hamming, Distance::OptimalMatching] {
            let silhouettes = null_silhouettes(
                &panel,
                &weights,
                model,
                settings(distance),
                Bootstrap::new(5, Seed(23)),
            )
            .unwrap();
            h = silhouettes.iter().fold(h, |h, &s| h.option_f64(s));
        }
    }
    let mut generator = SplitMix64::new(31);
    let replicate_weights: Vec<f64> = (0..6 * COUNT).map(|_| generator.below(3) as f64).collect();
    for distance in [Distance::DynamicHamming, Distance::OptimalMatching] {
        let agreements = refit_agreements(
            &panel,
            &weights,
            &replicate_weights,
            &reference,
            settings(distance),
            3,
        )
        .unwrap();
        for entry in agreements {
            h = match entry {
                None => h.u64(0),
                Some((rand, share)) => h.u64(1).option_f64(rand).option_f64(share),
            };
        }
    }
    let shifted: Vec<u32> = reference.iter().map(|&l| (l + 1) % 3).rev().collect();
    let agreement = partition_agreement(&reference, &shifted, &weights).unwrap();
    h = h
        .option_f64(agreement.adjusted_rand)
        .option_f64(agreement.share_agreeing);
    h.finish()
}

#[test]
fn the_main_results_have_fixed_bits() {
    let expected = fingerprint();
    let counts = [1, 4, 12];
    for (run, threads) in on_threads(&counts, fingerprint).into_iter().zip(counts) {
        assert_eq!(run, expected, "{threads} threads");
    }
    if std::env::var_os("GOLDEN_PRINT").is_some() {
        println!("const RECORDED: Option<u64> = Some({expected:#018x});");
    }
    epistat_testkit::hash_out("main_results", expected);
    if epistat_testkit::RECORDED_PLATFORM
        && let Some(recorded) = RECORDED
    {
        assert_eq!(expected, recorded, "the main results changed");
    }
}
