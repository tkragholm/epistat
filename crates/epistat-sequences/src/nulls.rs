//! Null models for the parametric bootstrap of a typology.
//!
//! A typology is only a finding if it beats data that have the same state
//! frequencies and spell structure but no clustering. Each null model keeps
//! some aspects of the observed sequences and destroys one, so which nulls a
//! typology beats says what its types are made of, and not merely that they
//! exist. [`null_sequences`] draws one set of sequences from a [`NullModel`].
//!
//! # Algorithm
//!
//! The observed sequences carry weights `w_i`. All draws use one stream
//! seeded with `seed`, in the order stated, and a weighted choice among
//! symbols with weights `v_a` is
//!
//! ```text
//! target = u * sum_a v_a,    u uniform on [0, 1)
//! choice = the first symbol a, in rank order, with target <= sum_{b <= a} v_b
//! ```
//!
//! over the symbols with `v_a > 0`, and the last such symbol if no cumulative
//! weight reaches the target. A target on a boundary takes the earlier symbol.
//! Each draw consumes one uniform.
//!
//! - [`NullModel::StateIndependence`]. For each sequence in turn and each
//!   position in turn, a symbol is drawn from the weighted distribution of
//!   symbols at that position. The cross-sectional distribution is kept, and
//!   both sequencing and duration are destroyed.
//! - [`NullModel::FirstOrderMarkov`]. The first state is drawn from the
//!   weighted distribution of first states and each later state from the
//!   weighted transitions out of the previous state, pooled over positions.
//!   The transition structure is kept, so run lengths stay realistic. A state
//!   with no observed transition out of it is absorbing: the sequence stays
//!   there and no draw is taken.
//! - [`NullModel::RandomisedSequencing`]. Each sequence is cut into spells,
//!   maximal runs of one state, and the spells are shuffled with
//!   [`Stream::shuffle`]. The durations of the person are kept and the order is
//!   destroyed.
//! - [`NullModel::RandomisedDuration`]. The spells keep their order and the
//!   spell lengths are shuffled among them. The order is kept and the timing
//!   is destroyed.
//!
//! The weights are not changed and stay with the caller, so the null and the
//! observed data have the same number of sequences with the same total weight
//! and differ only in which sequences they are.
//!
//! # Edge cases
//!
//! - A symbol with zero total weight is never drawn.
//! - The shuffles carry the bias of [`Stream::below`], documented in
//!   `epistat-rng`.
//! - A panel of length 1 gives, under the Markov model, only first states.
//!
//! # References
//!
//! 1. Studer M. WeightedCluster library manual: a practical guide to creating
//!    typologies of trajectories in the social sciences with R. LIVES Working
//!    Papers 24. Lausanne: NCCR LIVES; 2013.
//! 2. Studer M, Ritschard G. What matters in differences between life
//!    trajectories: a comparative review of sequence dissimilarity measures.
//!    J R Stat Soc Ser A. 2016;179(2):481-511. doi:10.1111/rssa.12125

#![expect(
    clippy::cast_possible_truncation,
    reason = "sequence indices are bounded by the u32 limit checked when a panel or triangle is built, symbol ranks by the alphabet, and cluster ranks by the number of labels"
)]

use epistat_rng::Stream;

use crate::SequenceError;
use crate::positionwise::Panel;

/// Which aspects of the observed sequences a null draw preserves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullModel {
    /// Each position drawn independently from its cross-sectional state
    /// distribution.
    StateIndependence,
    /// States drawn from the pooled first-order transition probabilities.
    FirstOrderMarkov,
    /// Each sequence's spells in random order.
    RandomisedSequencing,
    /// Each sequence's spell lengths in random order among its spells.
    RandomisedDuration,
}

impl NullModel {
    /// The four models, in the order of their indices 0 to 3.
    pub const ALL: [Self; 4] = [
        Self::StateIndependence,
        Self::FirstOrderMarkov,
        Self::RandomisedSequencing,
        Self::RandomisedDuration,
    ];

    /// The model at `index` in [`NullModel::ALL`], or `None` above 3.
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }
}

/// Cumulative weights over the symbols present, in rank order.
///
/// A dense table over the whole alphabet rather than a map: skipping the
/// symbols with no weight is what the cumulative sum does anyway.
// `usize as u8` on a symbol rank, which `Panel::new` has already checked is
// below the alphabet size, and ranks are `u8`.
fn cumulative(weight_by_symbol: &[f64]) -> (Vec<u8>, Vec<f64>) {
    let mut symbols = Vec::new();
    let mut running = 0.0;
    let mut totals = Vec::new();
    for (rank, &weight) in weight_by_symbol.iter().enumerate() {
        if weight > 0.0 {
            running += weight;
            symbols.push(rank as u8);
            totals.push(running);
        }
    }
    (symbols, totals)
}

/// One symbol, drawn against its cumulative weights.
///
/// `target <= bound`, scanning forward, and the last symbol if the scan falls
/// through. A target landing on a boundary takes the earlier symbol.
fn weighted_choice(symbols: &[u8], totals: &[f64], stream: &mut Stream) -> u8 {
    let target = stream.uniform() * totals[totals.len() - 1];
    for (symbol, &bound) in symbols.iter().zip(totals) {
        if target <= bound {
            return *symbol;
        }
    }
    symbols[symbols.len() - 1]
}

/// A sequence as (symbol, run length) pairs, in order.
fn spells(sequence: &[u8]) -> Vec<(u8, usize)> {
    let mut out: Vec<(u8, usize)> = Vec::new();
    for &symbol in sequence {
        match out.last_mut() {
            Some(last) if last.0 == symbol => last.1 += 1,
            _ => out.push((symbol, 1)),
        }
    }
    out
}

/// Similar but unclustered sequences, for the parametric bootstrap.
///
/// Returns `count * length` symbol ranks, sequence after sequence, drawn from
/// `model` as described in the module documentation. `weights` are those of
/// the observed sequences of `panel`.
///
/// # Errors
///
/// [`SequenceError::Input`] when the weights do not match the panel one to
/// one, or the panel has sequences and none has a positive weight.
pub fn null_sequences(
    panel: &Panel<'_>,
    weights: &[f64],
    model: NullModel,
    seed: u64,
) -> Result<Vec<u8>, SequenceError> {
    if weights.len() != panel.count {
        return Err(SequenceError::input(format!(
            "{} sequences but {} weights; they must correspond one to one",
            panel.count,
            weights.len()
        )));
    }
    if panel.count > 0 && !weights.iter().any(|&weight| weight > 0.0) {
        return Err(SequenceError::input(
            "no sequence has a positive weight; there is nothing to draw from",
        ));
    }
    let (length, alphabet) = (panel.length, panel.alphabet);

    let drawn = {
        let mut stream = Stream::seeded(seed);
        let mut drawn: Vec<u8> = Vec::with_capacity(panel.count * length);

        match model {
            NullModel::StateIndependence => {
                let tables: Vec<(Vec<u8>, Vec<f64>)> = (0..length)
                    .map(|position| {
                        let mut weight_by_symbol = vec![0.0; alphabet];
                        for index in 0..panel.count {
                            weight_by_symbol[panel.at(index)[position] as usize] += weights[index];
                        }
                        cumulative(&weight_by_symbol)
                    })
                    .collect();
                for _ in 0..panel.count {
                    for (symbols, totals) in &tables {
                        drawn.push(weighted_choice(symbols, totals, &mut stream));
                    }
                }
            }
            NullModel::FirstOrderMarkov => {
                let mut first = vec![0.0; alphabet];
                let mut rows = vec![vec![0.0; alphabet]; alphabet];
                for (index, &weight) in weights.iter().enumerate() {
                    let sequence = panel.at(index);
                    first[sequence[0] as usize] += weight;
                    for pair in sequence.windows(2) {
                        rows[pair[0] as usize][pair[1] as usize] += weight;
                    }
                }
                let start = cumulative(&first);
                let tables: Vec<(Vec<u8>, Vec<f64>)> =
                    rows.iter().map(|row| cumulative(row)).collect();
                for _ in 0..panel.count {
                    let mut state = weighted_choice(&start.0, &start.1, &mut stream);
                    drawn.push(state);
                    for _ in 1..length {
                        let (symbols, totals) = &tables[state as usize];
                        // An absorbing state has no outgoing transitions;
                        // staying put is the only behaviour consistent with the
                        // observed data, and it takes no draw.
                        if !symbols.is_empty() {
                            state = weighted_choice(symbols, totals, &mut stream);
                        }
                        drawn.push(state);
                    }
                }
            }
            NullModel::RandomisedSequencing => {
                for index in 0..panel.count {
                    let mut runs = spells(panel.at(index));
                    stream.shuffle(&mut runs);
                    for (symbol, run) in runs {
                        drawn.extend(std::iter::repeat_n(symbol, run));
                    }
                }
            }
            NullModel::RandomisedDuration => {
                for index in 0..panel.count {
                    let runs = spells(panel.at(index));
                    let mut durations: Vec<usize> = runs.iter().map(|s| s.1).collect();
                    stream.shuffle(&mut durations);
                    for ((symbol, _), run) in runs.iter().zip(&durations) {
                        drawn.extend(std::iter::repeat_n(*symbol, *run));
                    }
                }
            }
        }
        drawn
    };
    Ok(drawn)
}

#[cfg(test)]
mod tests {
    use super::{NullModel, null_sequences};
    use crate::Panel;

    // Three sequences over three states, the first with two spells.
    const ENCODED: [u8; 12] = [0, 0, 1, 1, 2, 2, 2, 0, 1, 0, 1, 2];
    const WEIGHTS: [f64; 3] = [3.0, 1.0, 2.0];

    fn panel() -> Panel<'static> {
        Panel::new(&ENCODED, 4, 3, None).expect("a valid panel")
    }

    #[test]
    fn a_draw_is_reproducible_and_has_the_panel_shape() {
        for model in NullModel::ALL {
            let first = null_sequences(&panel(), &WEIGHTS, model, 11).unwrap();
            let again = null_sequences(&panel(), &WEIGHTS, model, 11).unwrap();
            assert_eq!(first.len(), 12);
            assert_eq!(first, again);
            assert!(first.iter().all(|&symbol| symbol < 3));
        }
    }

    #[test]
    fn the_spell_models_keep_each_sequences_symbols_or_lengths() {
        let sorted = |sequence: &[u8]| {
            let mut copy = sequence.to_vec();
            copy.sort_unstable();
            copy
        };
        let sequencing =
            null_sequences(&panel(), &WEIGHTS, NullModel::RandomisedSequencing, 5).unwrap();
        for (drawn, observed) in sequencing.chunks(4).zip(ENCODED.chunks(4)) {
            assert_eq!(sorted(drawn), sorted(observed));
        }
        let duration =
            null_sequences(&panel(), &WEIGHTS, NullModel::RandomisedDuration, 5).unwrap();
        for (drawn, observed) in duration.chunks(4).zip(ENCODED.chunks(4)) {
            assert_eq!(drawn[0], observed[0]);
        }
    }

    #[test]
    fn the_markov_model_stays_in_an_absorbing_state() {
        // State 1 is never left, so once drawn it repeats.
        let encoded = [0u8, 1, 1, 1, 1, 1, 1, 1];
        let panel = Panel::new(&encoded, 4, 2, None).unwrap();
        let drawn = null_sequences(&panel, &[1.0, 1.0], NullModel::FirstOrderMarkov, 3).unwrap();
        for sequence in drawn.chunks(4) {
            if let Some(at) = sequence.iter().position(|&symbol| symbol == 1) {
                assert!(sequence[at..].iter().all(|&symbol| symbol == 1));
            }
        }
    }

    #[test]
    fn bad_weights_are_refused_and_models_are_indexed() {
        assert!(null_sequences(&panel(), &[1.0], NullModel::StateIndependence, 0).is_err());
        assert!(null_sequences(&panel(), &[0.0; 3], NullModel::StateIndependence, 0).is_err());
        assert_eq!(
            NullModel::from_index(3),
            Some(NullModel::RandomisedDuration)
        );
        assert_eq!(NullModel::from_index(4), None);
    }
}
