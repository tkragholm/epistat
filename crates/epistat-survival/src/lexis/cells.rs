//! Summing the pieces of a Lexis split into cells.
//!
//! When the design of a rate model depends on categorical cells only, such as
//! age band, period band and sex, the Poisson likelihood of the pieces is
//! the likelihood of the cells: the log-likelihood of a piece with exposure
//! `y` and `e` events is `e ln(mu) - mu` up to a constant, with
//! `mu = y exp(x beta)`, and pieces with the same `x` add up to a cell with
//! exposure `sum y` and `sum e`. The estimates and the covariance are those
//! of the pieces, and the work is one pass over the pieces and a fit over the
//! cells, which at register scale are thousands where the pieces are tens of
//! millions. The equality is shown in the tests, and the deviance differs only
//! by a constant that the saturated model of a piece and of a cell do not
//! share, see [`crate::lexis::rates`].
//!
//! # What is summed
//!
//! [`aggregate`] takes one or more categorical keys, each a code per piece
//! below its number of levels, and returns the cells that have at least one
//! piece, in the order of their combined key: the first key varies fastest.
//! The person-time and the events of a cell are the sums over its pieces in
//! the order of the pieces, so the sums do not depend on the strategy below
//! or on the number of threads. Neither [`crate::lexis::Lexis::interval`]
//! nor a column made from [`crate::lexis::Lexis::row`] needs a copy: both are
//! keys.
//!
//! # Algorithm
//!
//! With `K` the number of possible cells, the product of the levels, a table
//! of `K` sums is filled in one pass over the pieces when `K` is at most the
//! number of pieces. Otherwise the pieces are sorted by key and piece number,
//! which is a total order, and each run is summed in order. Both visit the
//! pieces of a cell in piece order and give the same bits. The first takes
//! `24 K` bytes and the second `16` bytes per piece.

use rayon::prelude::*;

use crate::error::{Error, Result};

/// A categorical key: the level of every piece.
#[derive(Debug, Clone, Copy)]
pub struct CellKey<'a> {
    /// The level of each piece, in `0..levels`.
    pub codes: &'a [u32],
    /// The number of levels, at least 1.
    pub levels: usize,
}

/// The cells of an aggregation.
#[derive(Debug, Clone, PartialEq)]
pub struct Cells {
    /// For each key, the level of each cell.
    pub codes: Vec<Vec<u32>>,
    /// The person-time of each cell.
    pub length: Vec<f64>,
    /// The events of each cell.
    pub events: Vec<f64>,
    /// The number of pieces in each cell, at least 1.
    pub pieces: Vec<u64>,
    /// The number of pieces that went in.
    pub n_input: usize,
}

impl Cells {
    /// The number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length.len()
    }

    /// Whether there are no cells, which is the case without pieces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length.is_empty()
    }
}

/// Sum `length` and `events` over the cells of `keys`.
///
/// # Errors
///
/// An input error when a key or a column has the wrong length, a key has no
/// levels or a code outside its levels, a length or an event count is not
/// finite or is negative, or the number of possible cells does not fit a
/// `u64`.
pub fn aggregate(keys: &[CellKey<'_>], length: &[f64], events: &[f64]) -> Result<Cells> {
    let n = length.len();
    if events.len() != n {
        return Err(Error::invalid(format!(
            "events has {} values against {n} lengths",
            events.len()
        )));
    }
    if let Some(i) = length.iter().position(|v| !(v.is_finite() && *v >= 0.0)) {
        return Err(Error::invalid(format!(
            "the length of piece {i} is not finite and non-negative"
        )));
    }
    if let Some(i) = events.iter().position(|v| !(v.is_finite() && *v >= 0.0)) {
        return Err(Error::invalid(format!(
            "the events of piece {i} are not finite and non-negative"
        )));
    }
    let mut strides = Vec::with_capacity(keys.len());
    let mut possible = 1_usize;
    for (k, key) in keys.iter().enumerate() {
        if key.codes.len() != n {
            return Err(Error::invalid(format!(
                "key {k} has {} codes against {n} pieces",
                key.codes.len()
            )));
        }
        if key.levels == 0 {
            return Err(Error::invalid(format!("key {k} has no levels")));
        }
        if let Some(i) = key
            .codes
            .iter()
            .position(|&c| usize::try_from(c).map_or(true, |c| c >= key.levels))
        {
            return Err(Error::invalid(format!(
                "key {k}: the code {} of piece {i} is not below its {} levels",
                key.codes[i], key.levels
            )));
        }
        strides.push(possible);
        possible = possible
            .checked_mul(key.levels)
            .ok_or_else(|| Error::invalid("the number of possible cells does not fit a usize"))?;
    }
    let combined = |i: usize| -> usize {
        keys.iter()
            .zip(&strides)
            .map(|(key, &stride)| key.codes[i] as usize * stride)
            .sum()
    };

    let mut ids: Vec<usize> = Vec::new();
    let mut sum_length: Vec<f64> = Vec::new();
    let mut sum_events: Vec<f64> = Vec::new();
    let mut count: Vec<u64> = Vec::new();
    if possible <= n {
        let size = possible;
        let mut l = vec![0.0; size];
        let mut e = vec![0.0; size];
        let mut c = vec![0_u64; size];
        for i in 0..n {
            let cell = combined(i);
            l[cell] += length[i];
            e[cell] += events[i];
            c[cell] += 1;
        }
        for cell in 0..size {
            if c[cell] > 0 {
                ids.push(cell);
                sum_length.push(l[cell]);
                sum_events.push(e[cell]);
                count.push(c[cell]);
            }
        }
    } else {
        let mut order: Vec<(usize, usize)> = (0..n).map(|i| (combined(i), i)).collect();
        order.par_sort_unstable();
        let mut at = 0;
        while at < order.len() {
            let id = order[at].0;
            let (mut l, mut e, mut c) = (0.0, 0.0, 0_u64);
            while at < order.len() && order[at].0 == id {
                let i = order[at].1;
                l += length[i];
                e += events[i];
                c += 1;
                at += 1;
            }
            ids.push(id);
            sum_length.push(l);
            sum_events.push(e);
            count.push(c);
        }
    }
    let levels: Vec<usize> = keys.iter().map(|k| k.levels).collect();
    Ok(finish(
        &levels, &strides, &ids, sum_length, sum_events, count, n,
    ))
}

/// The cells with the given combined keys, their codes decoded.
pub(crate) fn finish(
    levels: &[usize],
    strides: &[usize],
    ids: &[usize],
    length: Vec<f64>,
    events: Vec<f64>,
    pieces: Vec<u64>,
    n_input: usize,
) -> Cells {
    let codes = levels
        .iter()
        .zip(strides)
        .map(|(&l, &stride)| {
            ids.iter()
                .map(|&id| u32::try_from((id / stride) % l).unwrap_or(u32::MAX))
                .collect()
        })
        .collect();
    Cells {
        codes,
        length,
        events,
        pieces,
        n_input,
    }
}
