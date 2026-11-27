//! The semiparametric self-controlled case series of Farrington and Whitaker
//! \[1\], whose age effect is left unspecified, as the data of [`super::fit`].
//!
//! # Algorithm
//!
//! The age effect is a step function with one level at each distinct time at
//! which any case has an event, `u_1 < u_2 < .. < u_m`. A case observed on
//! `[start_i, end_i)` contributes the event times that fall in it, and the
//! probability that its event lies at `u_k` is
//!
//! ```text
//! exp(alpha_k + x_i(u_k)' beta) / sum_(l: u_l in [start_i, end_i)) exp(alpha_l + x_i(u_l)' beta)
//! ```
//!
//! with `x_i(u)` the risk-window indicators of the case at `u` and `alpha_1 =
//! 0`. This is the conditional likelihood of [`super::fit`] on pieces of unit
//! length, one per case and event time in its observation, with a column for
//! each event time after the first. [`semiparametric_pieces`] builds those
//! pieces and their design, so the fit, its covariance, robust variance and
//! intervals are those of the parametric case series. The relative incidence
//! is free of any assumption on the shape of the age effect, at the cost of
//! `m - 1` nuisance columns.
//!
//! # Scale
//!
//! The design has one row per case and event time in its observation and
//! `m - 1` age columns, and a Newton step of the fit costs `O(P p^2)` for `P`
//! rows and `p` columns. With a few hundred distinct event times and a few
//! thousand cases that is seconds; with event times in days over many years it
//! is not, and age bands or a smooth age effect are the practical choice.
//!
//! # Agreement with R
//!
//! The rows and the likelihood are those of `SCCS::semisccs` (SCCS 1.8), which
//! fits them with `survival::clogit` and `factor(age)`. `semisccs` keeps one
//! event when a person has two at the same time and these pieces keep both, as
//! for `standardsccs`.
//!
//! # References
//!
//! 1. Farrington CP, Whitaker HJ. Semiparametric analysis of case series data.
//!    J R Stat Soc Ser C Appl Stat 2006;55(5):553-94.

use epistat_linalg::Mat;

use super::{PieceSpec, SccsData, check_spec};
use crate::error::{Error, Result};

/// The pieces of the semiparametric case series: one of unit length per case
/// and distinct event time in its observation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SemiparametricPieces {
    /// The position of each piece's person in the input.
    pub person: Vec<u32>,
    /// The event time of each piece.
    pub time: Vec<f64>,
    /// The length of each piece, 1.
    pub length: Vec<f64>,
    /// The person's events at the time of each piece.
    pub events: Vec<f64>,
    /// The index of each piece's time in `event_times`.
    pub age_level: Vec<u32>,
    /// The risk window of each piece in each exposure series, `window[s][k]`.
    pub window: Vec<Vec<Option<u32>>>,
    /// The number of risk windows of each exposure series.
    pub n_windows: Vec<usize>,
    /// The distinct event times in increasing order. The first is the
    /// reference of the age effect.
    pub event_times: Vec<f64>,
}

impl SemiparametricPieces {
    /// The pieces as the input of [`super::fit`].
    #[must_use]
    pub fn data(&self) -> SccsData<'_> {
        SccsData {
            person: &self.person,
            length: &self.length,
            events: &self.events,
        }
    }

    /// The design: one indicator column per risk window of each exposure
    /// series in turn, then one per event time after the first.
    #[must_use]
    pub fn design(&self) -> Mat<f64> {
        let w: usize = self.n_windows.iter().sum();
        let columns: Vec<(usize, usize)> = self
            .n_windows
            .iter()
            .enumerate()
            .flat_map(|(s, &n)| (0..n).map(move |k| (s, k)))
            .collect();
        let m = self.event_times.len().saturating_sub(1);
        Mat::from_fn(self.person.len(), w + m, |i, j| {
            let on = if j < w {
                let (s, k) = columns[j];
                self.window[s][i] == u32::try_from(k).ok()
            } else {
                self.age_level[i] as usize == j - w + 1
            };
            f64::from(u8::from(on))
        })
    }
}

/// Builds the pieces of the semiparametric case series from the observation,
/// events and exposure series of `spec`, which may not have age cuts or a
/// calendar, since the age effect is free. See the module documentation.
///
/// # Errors
///
/// An input error for the input [`super::build_pieces`] refuses, and when
/// `spec` has age cuts or a calendar.
#[expect(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "people are checked against u32, windows and event times are fewer; an event is at a distinct event time or not"
)]
pub fn semiparametric_pieces(spec: &PieceSpec<'_>) -> Result<SemiparametricPieces> {
    let n = check_spec(spec)?;
    if !spec.age_cuts.is_empty() || spec.calendar.is_some() {
        return Err(Error::not_applicable(
            "the semiparametric case series has a free age effect and takes no age cuts or calendar",
        ));
    }
    for i in 0..n {
        let (a, b) = (spec.start[i], spec.end[i]);
        if let Some(&t) = spec.events[spec.event_offsets[i]..spec.event_offsets[i + 1]]
            .iter()
            .find(|&&t| !(t >= a && t < b))
        {
            return Err(Error::value(format!(
                "person {i} has an event at {t} outside the observation period [{a}, {b})"
            )));
        }
    }
    let mut times = spec.events.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut out = SemiparametricPieces {
        person: Vec::new(),
        time: Vec::new(),
        length: Vec::new(),
        events: Vec::new(),
        age_level: Vec::new(),
        window: vec![Vec::new(); spec.exposures.len()],
        n_windows: spec.exposures.iter().map(|s| s.windows.len()).collect(),
        event_times: Vec::new(),
    };
    for i in 0..n {
        let (a, b) = (spec.start[i], spec.end[i]);
        let events = &spec.events[spec.event_offsets[i]..spec.event_offsets[i + 1]];
        let first = times.partition_point(|&t| t < a);
        let last = times.partition_point(|&t| t < b);
        for (k, &u) in times.iter().enumerate().take(last).skip(first) {
            out.person.push(i as u32);
            out.time.push(u);
            out.length.push(1.0);
            out.events
                .push(events.iter().filter(|&&t| t == u).count() as f64);
            out.age_level.push(k as u32);
            for (series, label) in spec.exposures.iter().zip(&mut out.window) {
                let dates = &series.dates[series.offsets[i]..series.offsets[i + 1]];
                let mut best: Option<(f64, u32)> = None;
                for &d in dates {
                    for (w, win) in series.windows.iter().enumerate() {
                        if d + win.from <= u
                            && u < d + win.to
                            && best.is_none_or(|(top, _)| d > top)
                        {
                            best = Some((d, w as u32));
                        }
                    }
                }
                label.push(best.map(|(_, w)| w));
            }
        }
    }
    out.event_times = times;
    Ok(out)
}
