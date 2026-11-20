//! Reading the influence of every cluster at chosen times, or its integral,
//! from a recursion that returns the influence at the times it is asked for.
//!
//! The influence of a cluster on a curve is a step function of time. A pseudo
//! value of the curve at a time reads it there, and a pseudo value of the area
//! under the curve reads its integral, a sum of rectangles whose heights are
//! the influence at each knot. Holding the influence of every cluster at every
//! knot takes clusters times knots times width values, which is the memory the
//! area targets need. The integral is a running sum over the knots, so it
//! can be formed a few knots at a time: the recursion is asked for the influence
//! at one chunk of knots, the running sums of every cluster take it in, and the
//! chunk is dropped. A chunk costs a pass of the recursion, and `per_pass`
//! knots in a pass bound the values held at once to
//! `clusters * per_pass * width`.
//!
//! The running sum of a cluster adds the rectangles in the order of the knots
//! whatever the chunks, so the values do not depend on `per_pass`.

use crate::error::Result;

/// What the influence is read at: `knots` ascending, `at` ascending.
pub(super) struct Steps<'a> {
    /// The times the recursion is asked for.
    pub knots: &'a [f64],
    /// The times the pseudo-values are for. With `integrate` unset these are
    /// the knots.
    pub at: &'a [f64],
    /// Whether to read the integral of the influence from the first knot to each
    /// time of `at`, with the influence 0 before the first knot, and the last
    /// knot's value holding afterwards.
    pub integrate: bool,
}

/// The influence, or its integral, of every cluster of each block at `at`.
///
/// `blocks` holds the number of clusters of each group. `fetch(chunk)` returns,
/// for each block, the influence of its clusters at the knots of `chunk`,
/// row-major by cluster, then knot, then `width`. The result is, for each block,
/// row-major by cluster, then time of `at`, then `width`.
///
/// # Errors
///
/// Whatever `fetch` returns.
pub(super) fn read<F>(
    steps: &Steps<'_>,
    blocks: &[usize],
    width: usize,
    per_pass: usize,
    mut fetch: F,
) -> Result<Vec<Vec<f64>>>
where
    F: FnMut(&[f64]) -> Result<Vec<Vec<f64>>>,
{
    let Steps {
        knots,
        at,
        integrate,
    } = *steps;
    let (t, k) = (knots.len(), at.len());
    let mut out: Vec<Vec<f64>> = blocks.iter().map(|&n| vec![0.0; n * k * width]).collect();
    let mut running: Vec<Vec<f64>> = if integrate {
        blocks.iter().map(|&n| vec![0.0; n * width]).collect()
    } else {
        Vec::new()
    };
    // The times before the first knot have the value 0.
    let mut slot = 0;
    if integrate {
        while slot < k && (t == 0 || at[slot] < knots[0]) {
            slot += 1;
        }
    }
    let per_pass = per_pass.max(1);
    let mut first = 0;
    while first < t {
        let end = (first + per_pass).min(t);
        let chunk = end - first;
        let got = fetch(&knots[first..end])?;
        for i in first..end {
            let local = i - first;
            if integrate {
                // The influence is a step function, the value at each knot
                // holding until the next, so the area is a sum of rectangles.
                let next = if i + 1 < t {
                    knots[i + 1]
                } else {
                    f64::INFINITY
                };
                while slot < k && at[slot] < next {
                    let step = at[slot] - knots[i];
                    for (b, &n) in blocks.iter().enumerate() {
                        for c in 0..n {
                            for w in 0..width {
                                let u = got[b][(c * chunk + local) * width + w];
                                out[b][(c * k + slot) * width + w] =
                                    running[b][c * width + w] + u * step;
                            }
                        }
                    }
                    slot += 1;
                }
                if i + 1 < t {
                    let step = next - knots[i];
                    for (b, &n) in blocks.iter().enumerate() {
                        for c in 0..n {
                            for w in 0..width {
                                running[b][c * width + w] +=
                                    got[b][(c * chunk + local) * width + w] * step;
                            }
                        }
                    }
                }
            } else {
                for (b, &n) in blocks.iter().enumerate() {
                    for c in 0..n {
                        for w in 0..width {
                            out[b][(c * k + i) * width + w] =
                                got[b][(c * chunk + local) * width + w];
                        }
                    }
                }
            }
        }
        first = end;
    }
    Ok(out)
}
