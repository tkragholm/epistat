//! The reference hazard of each row of a cohort over its follow-up.

use crate::error::{Error, Result};
use crate::pyears::{Layout, PyearsOptions};
use crate::ratetable::RateTable;
use crate::surv::SurvData;

/// The reference hazard of every row of a cohort as a function of the time
/// since the origin of the follow-up scale.
///
/// The hazard of a row is piecewise constant between the cuts of the rate
/// table, so its cumulative hazard `H(t)` is piecewise linear and its expected
/// survival is `exp(-H(t))`. The pieces run from time 0, not from the entry
/// time of a row with delayed entry, because the weights of the Pohar Perme
/// estimator are the inverse of the expected survival since the origin. Each
/// piece takes 24 bytes, and a row has about one piece per cut of each time
/// dimension it crosses, plus one.
#[derive(Debug, Clone)]
pub struct ExpectedHazard {
    first: Vec<u32>,
    last: Vec<u32>,
    ends: Vec<f64>,
    hazards: Vec<f64>,
    cumulative: Vec<f64>,
}

impl ExpectedHazard {
    /// Builds the hazards of the rows of `data` from the table, the values of
    /// each time dimension at time 0 of the follow-up scale (`origins`, one
    /// slice per time dimension, such as age in days at the origin and the
    /// date of the origin) and the levels of each factor (`factors`), as
    /// [`RateTable::expected`] takes them. Rows of weight 0 get no pieces.
    ///
    /// The hazards are per unit of the follow-up time: the rate of the table
    /// divided by its scale. The cumulative hazard of a row is summed over its
    /// pieces in order, as [`RateTable::expected`] sums the count of a row.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] as for [`RateTable::expected`], and when a used row
    /// has follow-up off the table, which has no expected hazard and is not
    /// assumed to be zero, or the pieces number more than `u32::MAX`.
    pub fn from_table(
        table: &RateTable,
        data: &SurvData<'_>,
        origins: &[&[f64]],
        factors: &[&[u32]],
    ) -> Result<Self> {
        let surv = data.followup()?;
        let n = data.n_rows();
        let from_origin = {
            let mut b = SurvData::builder(surv.stop(), surv.status());
            if let Some(w) = data.weight() {
                b = b.weight(w);
            }
            b.build()?
        };
        let dims = table.dimensions(origins, factors)?;
        let Layout {
            data: walked,
            mut cursor,
            ..
        } = Layout::new(&from_origin, &dims, &PyearsOptions::default(), false)?;
        let scale = table.scale();
        let rates = table.rates();
        let mut me = Self {
            first: vec![0; n],
            last: vec![0; n],
            ends: Vec::new(),
            hazards: Vec::new(),
            cumulative: Vec::new(),
        };
        let mut current = usize::MAX;
        let mut elapsed = 0.0_f64;
        let mut total = 0.0_f64;
        let mut off: Option<usize> = None;
        let mut too_many = false;
        cursor.walk(
            &walked,
            &mut |i, cell, length| {
                if i != current {
                    current = i;
                    elapsed = 0.0;
                    total = 0.0;
                    if let Ok(at) = u32::try_from(me.ends.len()) {
                        me.first[i] = at;
                    } else {
                        too_many = true;
                    }
                }
                let Some(cell) = cell else {
                    off.get_or_insert(i);
                    return;
                };
                let hazard = rates[cell] / scale;
                elapsed += length;
                total += length * hazard;
                me.ends.push(elapsed);
                me.hazards.push(hazard);
                me.cumulative.push(total);
                if let Ok(at) = u32::try_from(me.ends.len()) {
                    me.last[i] = at;
                } else {
                    too_many = true;
                }
            },
            &mut |_, _| {},
        )?;
        if let Some(row) = off {
            return Err(Error::input(format!(
                "row {row} has follow-up off the rate table, which has no expected hazard"
            )));
        }
        if too_many {
            return Err(Error::input(
                "the follow-up has more than 2^32 pieces in the rate table",
            ));
        }
        // The last piece of a row ends where its follow-up ends, whatever the
        // rounding of the sum of the lengths.
        let stop = walked.followup()?.stop();
        for ((&first, &last), &end) in me.first.iter().zip(&me.last).zip(stop) {
            if last > first {
                me.ends[last as usize - 1] = end;
            }
        }
        Ok(me)
    }

    /// The number of rows of the data it was built from.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.first.len()
    }

    /// The end of the follow-up of row `i`, 0 for a row without pieces.
    #[must_use]
    pub fn horizon(&self, i: usize) -> f64 {
        let (a, b) = (self.first[i] as usize, self.last[i] as usize);
        if b > a { self.ends[b - 1] } else { 0.0 }
    }

    /// The cumulative reference hazard of row `i` from time 0 to `t`, `None`
    /// for `t` below 0, beyond the follow-up of the row, or not a number.
    #[must_use]
    pub fn cumulative(&self, i: usize, t: f64) -> Option<f64> {
        (t >= 0.0 && t <= self.horizon(i)).then(|| self.cumulative_within(i, t))
    }

    /// [`Self::cumulative`] for a `t` known to be in `[0, horizon(i)]`.
    pub(crate) fn cumulative_within(&self, i: usize, t: f64) -> f64 {
        let (a, b) = (self.first[i] as usize, self.last[i] as usize);
        if b == a {
            return 0.0;
        }
        let ends = &self.ends[a..b];
        let k = ends.partition_point(|&e| e < t).min(ends.len() - 1);
        let (before, begin) = if k == 0 {
            (0.0, 0.0)
        } else {
            (self.cumulative[a + k - 1], ends[k - 1])
        };
        before + (t - begin) * self.hazards[a + k]
    }

    /// Calls `f(a, b, hazard)` for each stretch `(a, b]` of `(lo, hi]` over
    /// which the reference hazard of row `i` is constant, in order, for
    /// `0 <= lo < hi <= horizon(i)`. The stretches end exactly at `hi`.
    pub(crate) fn pieces_within(
        &self,
        i: usize,
        lo: f64,
        hi: f64,
        mut f: impl FnMut(f64, f64, f64),
    ) {
        let (a, b) = (self.first[i] as usize, self.last[i] as usize);
        if b == a {
            return;
        }
        let ends = &self.ends[a..b];
        let last = ends.len() - 1;
        let mut k = ends.partition_point(|&e| e <= lo).min(last);
        let mut from = lo;
        loop {
            let to = if k == last { hi } else { ends[k].min(hi) };
            if to > from {
                f(from, to, self.hazards[a + k]);
            }
            if to >= hi {
                break;
            }
            from = to;
            k += 1;
        }
    }

    /// The reference hazard of row `i` at time `t`, that of the piece that
    /// ends at or after `t`, so that at a cut of the table it is the hazard
    /// before the cut. `None` for `t` not above 0, beyond the follow-up of the
    /// row, or not a number.
    #[must_use]
    pub fn hazard(&self, i: usize, t: f64) -> Option<f64> {
        let (a, b) = (self.first[i] as usize, self.last[i] as usize);
        if t.is_nan() || t <= 0.0 || b == a {
            return None;
        }
        let k = self.ends[a..b].partition_point(|&e| e < t);
        (a + k < b).then(|| self.hazards[a + k])
    }
}
