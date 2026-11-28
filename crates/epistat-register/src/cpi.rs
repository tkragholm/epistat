//! Constant-price deflation of nominal values to a reference year.
//!
//! The deflation is table-agnostic: it deflates any nominal value to the prices
//! of a reference year given an annual price index, which the caller passes as
//! a [`CpiTable`]. With the cargo feature `denmark`, the Statistics Denmark
//! annual consumer price index is available as `DENMARK_ANNUAL_CPI`.
//!
//! # Algorithm
//!
//! With `cpi[y]` the index of year `y`, a value `v` recorded in year `y` is
//! expressed in the prices of year `r` by
//!
//! ```text
//! factor(y, r) = cpi[r] / cpi[y]
//! deflated     = v * factor(y, r)
//! ```
//!
//! The factor is computed first and then multiplied, so a caller that joins the
//! factor to its rows and multiplies gets the same bits as
//! [`CpiTable::deflate`] and [`CpiTable::deflate_series`]. Under inflation a
//! reference year after the recorded year has the higher index, so the value is
//! scaled up.
//!
//! # Edge cases
//!
//! - A year with no index is an error, never a silent factor of one. A series
//!   with several uncovered years reports all of them at once, sorted and
//!   without duplicates, and a missing reference year is reported separately.
//! - [`CpiTable::try_new`] refuses a year given twice and an index that is
//!   not finite and positive, which would give an infinite or NaN factor.
//!   [`CpiTable::new`] checks neither: the later of two rows for a year wins.
//!
//! ```
//! use epistat_register::CpiTable;
//!
//! let cpi = CpiTable::try_new([(2015, 100.0), (2016, 100.3), (2017, 101.4)])?;
//! assert_eq!(cpi.years(), Some(2015..=2017));
//! let deflated = cpi.deflate(1_000.0, 2017, 2015)?;
//! assert!((deflated - 1_000.0 * 100.0 / 101.4).abs() < 1e-9);
//! assert!(CpiTable::try_new([(2015, 100.0), (2015, 101.0)]).is_err());
//! assert!(CpiTable::try_new([(2015, 0.0)]).is_err());
//! # Ok::<(), epistat_register::Error>(())
//! ```

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use epistat_core::InputError;

use crate::Error;

/// An annual price index keyed by year.
#[derive(Debug, Clone, Default)]
pub struct CpiTable {
    index_by_year: BTreeMap<i32, f64>,
}

impl CpiTable {
    /// A table from `(year, index)` rows, each year once and each index
    /// finite and positive.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] with [`InputError::Domain`] at the row of an index
    /// that is NaN, infinite, zero or negative, or of a year given in an
    /// earlier row.
    pub fn try_new(rows: impl IntoIterator<Item = (i32, f64)>) -> Result<Self, Error> {
        let mut index_by_year = BTreeMap::new();
        for (row, (year, index)) in rows.into_iter().enumerate() {
            if !(index.is_finite() && index > 0.0) {
                return Err(InputError::Domain {
                    what: "price index",
                    value: index,
                    domain: "(0, inf)",
                    row: Some(row),
                }
                .into());
            }
            if index_by_year.insert(year, index).is_some() {
                return Err(InputError::Domain {
                    what: "year",
                    value: f64::from(year),
                    domain: "the years not given in an earlier row",
                    row: Some(row),
                }
                .into());
            }
        }
        Ok(Self { index_by_year })
    }

    /// Build a table from `(year, index)` rows without checking them. Later
    /// rows win on duplicate years, and a zero or NaN index is kept;
    /// [`CpiTable::try_new`] refuses both.
    pub fn new(rows: impl IntoIterator<Item = (i32, f64)>) -> Self {
        let mut index_by_year = BTreeMap::new();
        for (year, index) in rows {
            index_by_year.insert(year, index);
        }
        Self { index_by_year }
    }

    /// The first and the last year with an index, `None` for an empty table.
    /// Years between them without an index are listed by
    /// [`CpiTable::missing_years`].
    #[must_use]
    pub fn years(&self) -> Option<RangeInclusive<i32>> {
        let first = *self.index_by_year.keys().next()?;
        let last = *self.index_by_year.keys().next_back()?;
        Some(first..=last)
    }

    /// The Statistics Denmark annual consumer price index,
    /// [`DENMARK_ANNUAL_CPI`], as a table.
    #[cfg(feature = "denmark")]
    #[must_use]
    pub fn denmark() -> Self {
        Self::new(DENMARK_ANNUAL_CPI.iter().copied())
    }

    /// The `(year, index)` rows in ascending year order.
    pub fn rows(&self) -> impl Iterator<Item = (i32, f64)> + '_ {
        self.index_by_year
            .iter()
            .map(|(&year, &index)| (year, index))
    }

    /// The index for `year`, if present.
    #[must_use]
    pub fn index(&self, year: i32) -> Option<f64> {
        self.index_by_year.get(&year).copied()
    }

    /// The years, ascending and without duplicates, among `years` that have no
    /// index.
    pub fn missing_years(&self, years: impl IntoIterator<Item = i32>) -> Vec<i32> {
        let mut missing: Vec<i32> = years
            .into_iter()
            .filter(|year| !self.index_by_year.contains_key(year))
            .collect();
        missing.sort_unstable();
        missing.dedup();
        missing
    }

    /// The multiplicative factor that converts `year` values to `reference_year`
    /// prices: `cpi[reference_year] / cpi[year]`.
    ///
    /// # Errors
    ///
    /// [`Error::MissingReferenceYear`] or [`Error::MissingYears`]
    /// when either year is not in the table.
    pub fn adjustment_factor(&self, year: i32, reference_year: i32) -> Result<f64, Error> {
        let reference = self
            .index(reference_year)
            .ok_or(Error::MissingReferenceYear(reference_year))?;
        let current = self
            .index(year)
            .ok_or_else(|| Error::MissingYears(vec![year]))?;
        Ok(reference / current)
    }

    /// Deflate a single nominal value from `year` to `reference_year` prices.
    ///
    /// # Errors
    ///
    /// As [`Self::adjustment_factor`].
    pub fn deflate(&self, nominal: f64, year: i32, reference_year: i32) -> Result<f64, Error> {
        Ok(nominal * self.adjustment_factor(year, reference_year)?)
    }

    /// Deflate a column of `(nominal, year)` pairs to `reference_year` prices.
    ///
    /// # Errors
    ///
    /// [`Error::MissingYears`] listing every uncovered year, so the
    /// caller sees the full gap at once, or
    /// [`Error::MissingReferenceYear`] if the reference year itself is
    /// absent.
    pub fn deflate_series(
        &self,
        costs: &[(f64, i32)],
        reference_year: i32,
    ) -> Result<Vec<f64>, Error> {
        let reference = self
            .index(reference_year)
            .ok_or(Error::MissingReferenceYear(reference_year))?;
        let missing = self.missing_years(costs.iter().map(|&(_, year)| year));
        if !missing.is_empty() {
            return Err(Error::MissingYears(missing));
        }
        Ok(costs
            .iter()
            .map(|&(nominal, year)| {
                let current = self.index_by_year[&year];
                nominal * (reference / current)
            })
            .collect())
    }
}

/// Statistics Denmark annual consumer price index, 1900 to 2025, as
/// `(year, index)` on a 2015=100 base.
///
/// [`CpiTable::denmark`] builds a [`CpiTable`] from it.
///
/// The series is spliced from two tables of Statistics Denmark at 1994. From
/// 1994 the values are the four-decimal 2015=100 index. Before it they are
/// derived from PRIS8, "Consumer price index, annual average", which is
/// published from 1900 on a 1900=100 base and rounded to whole index points,
/// and rescaled so that the 1993 to 1994 ratio is PRIS8's own and the join
/// carries no artefact. The rows before 1994 therefore have the granularity of
/// PRIS8, about 0.02% at 1994 and 1% at 1900. They are written to four decimals
/// for arithmetic continuity and are not accurate to four decimals.
#[cfg(feature = "denmark")]
pub const DENMARK_ANNUAL_CPI: &[(i32, f64)] = &[
    (1900, 1.4510),
    (1901, 1.4510),
    (1902, 1.4655),
    (1903, 1.4655),
    (1904, 1.4800),
    (1905, 1.4800),
    (1906, 1.4945),
    (1907, 1.5380),
    (1908, 1.5525),
    (1909, 1.5671),
    (1910, 1.5816),
    (1911, 1.5816),
    (1912, 1.6396),
    (1913, 1.6831),
    (1914, 1.7267),
    (1915, 2.0314),
    (1916, 2.3941),
    (1917, 2.7714),
    (1918, 3.2357),
    (1919, 3.8306),
    (1920, 4.5706),
    (1921, 3.8886),
    (1922, 3.3082),
    (1923, 3.4388),
    (1924, 3.6420),
    (1925, 3.5404),
    (1926, 3.0035),
    (1927, 2.9020),
    (1928, 2.8875),
    (1929, 2.8729),
    (1930, 2.7278),
    (1931, 2.5827),
    (1932, 2.5682),
    (1933, 2.6263),
    (1934, 2.7278),
    (1935, 2.8439),
    (1936, 2.8729),
    (1937, 2.9745),
    (1938, 3.0035),
    (1939, 3.0906),
    (1940, 3.8596),
    (1941, 4.4255),
    (1942, 4.5706),
    (1943, 4.6141),
    (1944, 4.7157),
    (1945, 4.7592),
    (1946, 4.7302),
    (1947, 4.8608),
    (1948, 4.9914),
    (1949, 5.1075),
    (1950, 5.5718),
    (1951, 6.2247),
    (1952, 6.3698),
    (1953, 6.3263),
    (1954, 6.4424),
    (1955, 6.8776),
    (1956, 7.2259),
    (1957, 7.3129),
    (1958, 7.3855),
    (1959, 7.5306),
    (1960, 7.7047),
    (1961, 8.0529),
    (1962, 8.5753),
    (1963, 9.0251),
    (1964, 9.3588),
    (1965, 9.9537),
    (1966, 10.6357),
    (1967, 11.4192),
    (1968, 12.3333),
    (1969, 12.7686),
    (1970, 13.5957),
    (1971, 14.3937),
    (1972, 15.3514),
    (1973, 16.7733),
    (1974, 19.3416),
    (1975, 21.1988),
    (1976, 23.0996),
    (1977, 25.6678),
    (1978, 28.2361),
    (1979, 30.9494),
    (1980, 34.7655),
    (1981, 38.8427),
    (1982, 42.7749),
    (1983, 45.7349),
    (1984, 48.6078),
    (1985, 50.8859),
    (1986, 52.7576),
    (1987, 54.8761),
    (1988, 57.3573),
    (1989, 60.0996),
    (1990, 61.6812),
    (1991, 63.1612),
    (1992, 64.4961),
    (1993, 65.2941),
    (1994, 66.6000),
    (1995, 68.0000),
    (1996, 69.5000),
    (1997, 71.0000),
    (1998, 72.3000),
    (1999, 74.1000),
    (2000, 76.2167),
    (2001, 78.0250),
    (2002, 79.9167),
    (2003, 81.5750),
    (2004, 82.5167),
    (2005, 84.0167),
    (2006, 85.6333),
    (2007, 87.0833),
    (2008, 90.0583),
    (2009, 91.2333),
    (2010, 93.3417),
    (2011, 95.9167),
    (2012, 98.2167),
    (2013, 98.9917),
    (2014, 99.5500),
    (2015, 100.0000),
    (2016, 100.2500),
    (2017, 101.4000),
    (2018, 102.2250),
    (2019, 103.0000),
    (2020, 103.4333),
    (2021, 105.3500),
    (2022, 113.4583),
    (2023, 117.2083),
    (2024, 118.8167),
    (2025, 120.9900),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// A table with the gap years 2011 to 2014 and 2016 to 2019 absent.
    fn table() -> CpiTable {
        CpiTable::new([
            (2010, 93.3417),
            (2015, 100.0),
            (2020, 103.4333),
            (2023, 117.2083),
        ])
    }

    #[test]
    fn a_checked_table_refuses_bad_indices_and_repeated_years() {
        let checked = CpiTable::try_new(table().rows()).unwrap();
        assert_eq!(
            checked.rows().collect::<Vec<_>>(),
            table().rows().collect::<Vec<_>>()
        );
        assert_eq!(checked.years(), Some(2010..=2023));
        assert_eq!(CpiTable::default().years(), None);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                CpiTable::try_new([(2010, 90.0), (2011, bad)]),
                Err(Error::Input(InputError::Domain {
                    what: "price index",
                    row: Some(1),
                    ..
                }))
            ));
        }
        assert!(matches!(
            CpiTable::try_new([(2010, 90.0), (2011, 91.0), (2010, 92.0)]),
            Err(Error::Input(InputError::Domain {
                what: "year",
                row: Some(2),
                ..
            }))
        ));
    }

    #[cfg(feature = "denmark")]
    #[test]
    fn the_danish_table_passes_the_checks() {
        let checked = CpiTable::try_new(DENMARK_ANNUAL_CPI.iter().copied()).unwrap();
        assert_eq!(
            CpiTable::denmark().rows().collect::<Vec<_>>(),
            checked.rows().collect::<Vec<_>>()
        );
        assert_eq!(CpiTable::denmark().years(), Some(1900..=2025));
    }

    #[test]
    fn reference_year_factor_is_one() {
        let factor = table().adjustment_factor(2023, 2023).unwrap();
        assert_eq!(factor.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn the_factor_is_the_reference_index_over_the_year_index() {
        let factor = table().adjustment_factor(2015, 2023).unwrap();
        assert_eq!(factor.to_bits(), (117.2083_f64 / 100.0).to_bits());
    }

    #[test]
    fn deflate_multiplies_by_the_factor_computed_first() {
        let deflated = table().deflate(1000.0, 2010, 2023).unwrap();
        assert_eq!(
            deflated.to_bits(),
            (1000.0 * (117.2083_f64 / 93.3417)).to_bits()
        );
    }

    #[test]
    fn deflate_series_reports_all_missing_years_sorted() {
        let costs = [(10.0, 2030), (20.0, 2015), (30.0, 1899), (40.0, 2030)];
        let err = table().deflate_series(&costs, 2023).unwrap_err();
        assert_eq!(err, Error::MissingYears(vec![1899, 2030]));
    }

    #[test]
    fn missing_reference_year_is_distinct_error() {
        let err = table().deflate(10.0, 2015, 1899).unwrap_err();
        assert_eq!(err, Error::MissingReferenceYear(1899));
    }

    #[test]
    fn errors_name_the_years_in_the_message() {
        let err = table().deflate_series(&[(1.0, 2030), (1.0, 1899)], 2023);
        assert_eq!(
            err.unwrap_err().to_string(),
            "Missing CPI adjustment factors for years: 1899, 2030"
        );
        assert_eq!(
            Error::MissingReferenceYear(1899).to_string(),
            "Reference price year 1899 is missing from the CPI table."
        );
    }

    #[test]
    fn deflate_series_matches_per_element_deflate() {
        let costs = [(100.0, 2010), (250.0, 2020), (5.0, 2023)];
        let series = table().deflate_series(&costs, 2023).unwrap();
        for (value, &(nominal, year)) in series.iter().zip(costs.iter()) {
            let single = table().deflate(nominal, year, 2023).unwrap();
            assert_eq!(value.to_bits(), single.to_bits());
        }
    }

    #[test]
    fn a_later_row_wins_a_duplicate_year() {
        let table = CpiTable::new([(2000, 1.0), (2001, 2.0), (2000, 4.0)]);
        assert_eq!(table.index(2000).unwrap().to_bits(), 4.0_f64.to_bits());
        let rows: Vec<_> = table.rows().collect();
        assert_eq!(rows, [(2000, 4.0), (2001, 2.0)]);
    }

    #[cfg(feature = "denmark")]
    #[test]
    fn the_danish_table_covers_1900_to_2025_in_ascending_years() {
        let years: Vec<i32> = DENMARK_ANNUAL_CPI.iter().map(|&(year, _)| year).collect();
        assert_eq!(years, (1900..=2025).collect::<Vec<_>>());
        let table = CpiTable::new(DENMARK_ANNUAL_CPI.iter().copied());
        assert_eq!(table.index(2015).unwrap().to_bits(), 100.0_f64.to_bits());
        let factor = table.adjustment_factor(2015, 2023).unwrap();
        assert_eq!(factor.to_bits(), (117.2083_f64 / 100.0).to_bits());
    }
}
