//! The panel of a treatment that may switch on and off, the data view of the
//! dCDH estimator.

use epistat_core::{InputError, Panel, check};
use epistat_linalg::{Mat, MatRef};

use crate::error::Error;

/// A validated panel of groups over periods whose treatment may switch on
/// and off and take several values, with outcomes and treatments that may
/// be missing, as `DIDmultiplegtDYN::did_multiplegt_dyn` takes it.
///
/// It holds one row per unit (a group, in the package's word) and period:
/// the unit as a `u32` code, the period as an `i64`, the outcome and the
/// treatment, each `None` where missing. Optional columns add sampling
/// weights, the controls, the strata of `trends_nonparam`, the clusters of
/// the variance, the subgroups of [`by`](super::by) and the covariates of
/// [`heterogeneity`](super::heterogeneity). Its key is
/// [`epistat_core::Panel`]: the rows are sorted by unit and then period with
/// no pair repeated, and [`Panel::sort_order`] gives the order for unsorted
/// input. Labels become codes with [`epistat_core::Codes::from_labels`],
/// whose sorted order keeps the order of the units, strata and clusters that
/// of their labels; the estimator's internal orders, and so its sums, follow
/// the codes.
///
/// [`SwitchingPanel::builder`] takes the four required columns, the builder's
/// methods add the optional ones, and [`SwitchingPanelBuilder::build`] checks
/// every column once:
///
/// | column | check |
/// |---|---|
/// | `unit`, `period` | the same length, at least one row, sorted by unit then period, no repeated pair |
/// | every other column | one entry per row |
/// | `outcome`, `treatment` | finite where given |
/// | `weights` | finite and not negative |
/// | `controls`, `het_covariates` | finite |
/// | `strata`, `clusters`, `subgroups`, `het_covariates` | constant within a unit |
///
/// Periods are ranked, so only their order matters. A unit whose outcome or
/// treatment is never observed takes no part.
///
/// ```
/// use epistat_did::SwitchingPanel;
///
/// // Two units over three periods; the first switches on in period 2 and
/// // off again in period 3, the second is never treated.
/// let unit = [0, 0, 0, 1, 1, 1];
/// let period = [1, 2, 3, 1, 2, 3];
/// let y = [Some(1.0), Some(2.0), Some(1.2), Some(0.9), None, Some(1.1)];
/// let d = [Some(0.0), Some(1.0), Some(0.0), Some(0.0), Some(0.0), Some(0.0)];
/// let panel = SwitchingPanel::builder(&unit, &period, &y, &d).build()?;
/// assert_eq!(panel.n_units(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SwitchingPanel<'a> {
    key: Panel<'a>,
    outcome: &'a [Option<f64>],
    treatment: &'a [Option<f64>],
    weights: Option<&'a [f64]>,
    controls: Option<MatRef<'a, f64>>,
    strata: Option<&'a [u32]>,
    clusters: Option<&'a [u32]>,
    subgroups: Option<&'a [u32]>,
    het_covariates: Option<MatRef<'a, f64>>,
}

/// The builder of a [`SwitchingPanel`].
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct SwitchingPanelBuilder<'a> {
    unit: &'a [u32],
    period: &'a [i64],
    outcome: &'a [Option<f64>],
    treatment: &'a [Option<f64>],
    weights: Option<&'a [f64]>,
    controls: Option<MatRef<'a, f64>>,
    strata: Option<&'a [u32]>,
    clusters: Option<&'a [u32]>,
    subgroups: Option<&'a [u32]>,
    het_covariates: Option<MatRef<'a, f64>>,
}

impl<'a> SwitchingPanel<'a> {
    /// The panel of the unit codes `unit`, the periods `period`, and the
    /// outcome and treatment of each row, `None` where missing.
    pub const fn builder(
        unit: &'a [u32],
        period: &'a [i64],
        outcome: &'a [Option<f64>],
        treatment: &'a [Option<f64>],
    ) -> SwitchingPanelBuilder<'a> {
        SwitchingPanelBuilder {
            unit,
            period,
            outcome,
            treatment,
            weights: None,
            controls: None,
            strata: None,
            clusters: None,
            subgroups: None,
            het_covariates: None,
        }
    }

    /// The key: the unit code and period of each row.
    #[must_use]
    pub const fn key(&self) -> Panel<'a> {
        self.key
    }

    /// The unit code of each row.
    #[must_use]
    pub const fn unit(&self) -> &'a [u32] {
        self.key.unit()
    }

    /// The period of each row.
    #[must_use]
    pub const fn period(&self) -> &'a [i64] {
        self.key.period()
    }

    /// The outcome of each row, `None` where missing.
    #[must_use]
    pub const fn outcome(&self) -> &'a [Option<f64>] {
        self.outcome
    }

    /// The treatment of each row, `None` where missing.
    #[must_use]
    pub const fn treatment(&self) -> &'a [Option<f64>] {
        self.treatment
    }

    /// The sampling weight of each row, `None` when every row weighs 1.
    #[must_use]
    pub const fn weights(&self) -> Option<&'a [f64]> {
        self.weights
    }

    /// The controls, one row per row, `None` without controls.
    #[must_use]
    pub const fn controls(&self) -> Option<MatRef<'a, f64>> {
        self.controls
    }

    /// The stratum code of each row (`trends_nonparam`), `None` for one
    /// stratum.
    #[must_use]
    pub const fn strata(&self) -> Option<&'a [u32]> {
        self.strata
    }

    /// The cluster code of each row, `None` when each unit is its own
    /// cluster.
    #[must_use]
    pub const fn clusters(&self) -> Option<&'a [u32]> {
        self.clusters
    }

    /// The subgroup code of each row (`by`), `None` without subgroups.
    #[must_use]
    pub const fn subgroups(&self) -> Option<&'a [u32]> {
        self.subgroups
    }

    /// The covariates of the heterogeneity regression (`predict_het`), one
    /// row per row, `None` without them.
    #[must_use]
    pub const fn het_covariates(&self) -> Option<MatRef<'a, f64>> {
        self.het_covariates
    }

    /// The number of rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.key.len()
    }

    /// Whether the panel has no rows, which the builder refuses, so always
    /// false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.key.is_empty()
    }

    /// The number of distinct units.
    #[must_use]
    pub fn n_units(&self) -> usize {
        self.key.n_units()
    }

    /// The weight of row `i`.
    pub(super) fn weight(&self, i: usize) -> f64 {
        self.weights.map_or(1.0, |w| w[i])
    }

    /// The rows `rows`, in their order, with the weight of each multiplied
    /// by `scale` when given, as owned columns. The subgroups and the
    /// heterogeneity covariates are left out.
    pub(super) fn select(&self, rows: &[usize], scale: Option<&[f64]>) -> Columns {
        let pick = |v: &[u32]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let weights = match scale {
            Some(s) => Some(
                rows.iter()
                    .zip(s)
                    .map(|(&i, m)| self.weight(i) * m)
                    .collect(),
            ),
            None => self.weights.map(|w| rows.iter().map(|&i| w[i]).collect()),
        };
        Columns {
            unit: pick(self.unit()),
            period: rows.iter().map(|&i| self.period()[i]).collect(),
            outcome: rows.iter().map(|&i| self.outcome[i]).collect(),
            treatment: rows.iter().map(|&i| self.treatment[i]).collect(),
            weights,
            controls: self
                .controls
                .map(|x| Mat::from_fn(rows.len(), x.ncols(), |r, k| x[(rows[r], k)])),
            strata: self.strata.map(pick),
            clusters: self.clusters.map(pick),
        }
    }
}

/// Owned columns of some rows of a [`SwitchingPanel`].
pub(super) struct Columns {
    unit: Vec<u32>,
    period: Vec<i64>,
    outcome: Vec<Option<f64>>,
    treatment: Vec<Option<f64>>,
    weights: Option<Vec<f64>>,
    controls: Option<Mat<f64>>,
    strata: Option<Vec<u32>>,
    clusters: Option<Vec<u32>>,
}

impl Columns {
    /// The panel of these columns.
    pub(super) fn panel(&self) -> Result<SwitchingPanel<'_>, Error> {
        let mut builder =
            SwitchingPanel::builder(&self.unit, &self.period, &self.outcome, &self.treatment);
        if let Some(w) = &self.weights {
            builder = builder.weights(w);
        }
        if let Some(x) = &self.controls {
            builder = builder.controls(x.as_ref());
        }
        if let Some(s) = &self.strata {
            builder = builder.strata(s);
        }
        if let Some(c) = &self.clusters {
            builder = builder.clusters(c);
        }
        builder.build()
    }
}

impl<'a> SwitchingPanelBuilder<'a> {
    /// Sampling weights, one per row (`weight`).
    pub const fn weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Controls, one row per row (`controls`).
    pub const fn controls(mut self, controls: MatRef<'a, f64>) -> Self {
        self.controls = Some(controls);
        self
    }

    /// Stratum codes, one per row and constant within a unit
    /// (`trends_nonparam`): comparisons are made within strata.
    pub const fn strata(mut self, strata: &'a [u32]) -> Self {
        self.strata = Some(strata);
        self
    }

    /// Cluster codes, one per row and constant within a unit (`cluster`).
    pub const fn clusters(mut self, clusters: &'a [u32]) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// Subgroup codes, one per row and constant within a unit, for
    /// [`by`](super::by).
    pub const fn subgroups(mut self, subgroups: &'a [u32]) -> Self {
        self.subgroups = Some(subgroups);
        self
    }

    /// Covariates of the heterogeneity regression, one row per row and
    /// constant within a unit, for [`heterogeneity`](super::heterogeneity).
    pub const fn het_covariates(mut self, covariates: MatRef<'a, f64>) -> Self {
        self.het_covariates = Some(covariates);
        self
    }

    /// The validated panel.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for the first check of [`SwitchingPanel`] that
    /// fails: [`InputError::Length`], [`InputError::Empty`],
    /// [`InputError::Unsorted`] or [`InputError::Duplicate`] for the key,
    /// [`InputError::Length`] for another column of the wrong length,
    /// [`InputError::NotFinite`] for a value that is not finite,
    /// [`InputError::Domain`] for a negative weight and
    /// [`InputError::NotConstant`] for a column that changes within a unit.
    pub fn build(self) -> Result<SwitchingPanel<'a>, Error> {
        let key = Panel::new(self.unit, self.period)?;
        let n = key.len();
        for (what, column) in [("outcome", self.outcome), ("treatment", self.treatment)] {
            check::same_length(what, n, column.len())?;
            if let Some(row) = column
                .iter()
                .position(|v| v.is_some_and(|v| !v.is_finite()))
            {
                return Err(InputError::NotFinite {
                    what,
                    row,
                    column: None,
                }
                .into());
            }
        }
        if let Some(weights) = self.weights {
            check::same_length("weights", n, weights.len())?;
            check::finite("weights", weights)?;
            if let Some(&value) = weights.iter().find(|&&w| w < 0.0) {
                return Err(InputError::Domain {
                    what: "weights",
                    value,
                    domain: "[0, inf)",
                    row: None,
                }
                .into());
            }
        }
        for (what, x) in [
            ("controls", self.controls),
            ("het_covariates", self.het_covariates),
        ] {
            if let Some(x) = x {
                check::same_length(what, n, x.nrows())?;
                for j in 0..x.ncols() {
                    if let Some(row) = (0..n).find(|&i| !x[(i, j)].is_finite()) {
                        return Err(InputError::NotFinite {
                            what,
                            row,
                            column: None,
                        }
                        .into());
                    }
                }
            }
        }
        for (what, column) in [
            ("strata", self.strata),
            ("clusters", self.clusters),
            ("subgroups", self.subgroups),
        ] {
            if let Some(column) = column {
                check::same_length(what, n, column.len())?;
                constant_by_unit(key, what, |i, j| column[i] == column[j])?;
            }
        }
        if let Some(x) = self.het_covariates {
            constant_by_unit(key, "het_covariates", |i, j| {
                (0..x.ncols()).all(|k| x[(i, k)].to_bits() == x[(j, k)].to_bits())
            })?;
        }
        Ok(SwitchingPanel {
            key,
            outcome: self.outcome,
            treatment: self.treatment,
            weights: self.weights,
            controls: self.controls,
            strata: self.strata,
            clusters: self.clusters,
            subgroups: self.subgroups,
            het_covariates: self.het_covariates,
        })
    }
}

/// Every row of a unit equals the unit's first row under `same`. The rows of
/// a unit are adjacent in a validated key.
fn constant_by_unit(
    key: Panel<'_>,
    what: &'static str,
    same: impl Fn(usize, usize) -> bool,
) -> Result<(), InputError> {
    for (unit, rows) in key.units() {
        let first = rows.start;
        if rows.into_iter().any(|i| !same(first, i)) {
            return Err(InputError::NotConstant { what, unit });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::Classify;

    const UNIT: [u32; 6] = [0, 0, 0, 2, 2, 2];
    const PERIOD: [i64; 6] = [1, 2, 3, 1, 2, 3];
    const Y: [Option<f64>; 6] = [Some(1.0), Some(2.0), None, Some(1.5), Some(1.5), Some(2.5)];
    const D: [Option<f64>; 6] = [Some(0.0), Some(1.0), Some(0.0), None, Some(0.0), Some(0.0)];

    fn input(result: Result<SwitchingPanel<'_>, Error>) -> InputError {
        match result {
            Err(Error::Input(error)) => error,
            other => panic!("expected an input error, got {other:?}"),
        }
    }

    #[test]
    fn a_valid_panel_keeps_its_columns() {
        let w = [1.0, 1.0, 1.0, 0.0, 2.0, 2.0];
        let x = Mat::from_fn(6, 1, |i, _| f64::from(u32::try_from(i).unwrap()));
        let h = Mat::from_fn(6, 1, |i, _| if i < 3 { 1.0 } else { 2.0 });
        let codes = [7, 7, 7, 3, 3, 3];
        let panel = SwitchingPanel::builder(&UNIT, &PERIOD, &Y, &D)
            .weights(&w)
            .controls(x.as_ref())
            .strata(&codes)
            .clusters(&codes)
            .subgroups(&codes)
            .het_covariates(h.as_ref())
            .build()
            .unwrap();
        assert_eq!((panel.len(), panel.n_units()), (6, 2));
        assert!(!panel.is_empty());
        assert_eq!(panel.unit(), &UNIT);
        assert_eq!(panel.period(), &PERIOD);
        assert_eq!(panel.outcome(), &Y);
        assert_eq!(panel.treatment(), &D);
        assert_eq!(panel.weights(), Some(&w[..]));
        assert_eq!(panel.controls().map(|x| x.ncols()), Some(1));
        assert_eq!(panel.strata(), Some(&codes[..]));
        assert_eq!(panel.clusters(), Some(&codes[..]));
        assert_eq!(panel.subgroups(), Some(&codes[..]));
        assert_eq!(panel.het_covariates().map(|x| x.ncols()), Some(1));
    }

    #[test]
    fn bad_columns_are_refused() {
        let period = [1, 1, 3, 1, 2, 3];
        let error = input(SwitchingPanel::builder(&UNIT, &period, &Y, &D).build());
        assert_eq!(error.code(), "input.duplicate");
        let error = input(SwitchingPanel::builder(&UNIT, &PERIOD, &Y[..5], &D).build());
        assert_eq!(
            error,
            InputError::Length {
                what: "outcome",
                expected: 6,
                found: 5
            }
        );
        let mut y = Y;
        y[4] = Some(f64::NAN);
        let error = input(SwitchingPanel::builder(&UNIT, &PERIOD, &y, &D).build());
        assert_eq!(
            error,
            InputError::NotFinite {
                what: "outcome",
                row: 4,
                column: None
            }
        );
        let w = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
        let error = input(
            SwitchingPanel::builder(&UNIT, &PERIOD, &Y, &D)
                .weights(&w)
                .build(),
        );
        assert_eq!(error.code(), "input.domain");
        let x = Mat::from_fn(6, 1, |i, _| if i == 2 { f64::INFINITY } else { 0.0 });
        let error = input(
            SwitchingPanel::builder(&UNIT, &PERIOD, &Y, &D)
                .controls(x.as_ref())
                .build(),
        );
        assert_eq!(
            error,
            InputError::NotFinite {
                what: "controls",
                row: 2,
                column: None
            }
        );
    }

    #[test]
    fn unit_level_columns_must_be_constant_within_a_unit() {
        let strata = [1, 1, 1, 4, 5, 4];
        let error = input(
            SwitchingPanel::builder(&UNIT, &PERIOD, &Y, &D)
                .strata(&strata)
                .build(),
        );
        assert_eq!(
            error,
            InputError::NotConstant {
                what: "strata",
                unit: 2
            }
        );
        let h = Mat::from_fn(6, 1, |i, _| if i == 1 { 1.0 } else { 0.0 });
        let error = input(
            SwitchingPanel::builder(&UNIT, &PERIOD, &Y, &D)
                .het_covariates(h.as_ref())
                .build(),
        );
        assert_eq!(
            error,
            InputError::NotConstant {
                what: "het_covariates",
                unit: 0
            }
        );
    }

    #[test]
    fn bad_input_is_an_input_error() {
        use epistat_testkit::{BadInput, assert_rejects_bad_input};
        let reversed: Vec<i64> = PERIOD.iter().rev().copied().collect();
        let w = [1.0; 6];
        assert_rejects_bad_input("SwitchingPanel::build", |bad| {
            let weights = bad.column(&w);
            let built = match bad {
                BadInput::Unsorted => SwitchingPanel::builder(&UNIT, &reversed, &Y, &D).build(),
                BadInput::Empty => SwitchingPanel::builder(&[], &[], &[], &[]).build(),
                _ => SwitchingPanel::builder(&UNIT, &PERIOD, &Y, &D)
                    .weights(&weights)
                    .build(),
            };
            Some(built.map(|panel| panel.len()))
        });
    }
}
