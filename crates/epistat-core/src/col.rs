//! A column of values in which some rows may have none, held as Arrow holds
//! it: the values, and a validity bitmap with one bit per row.
//!
//! An Arrow or polars column lends its value buffer and its null buffer as
//! they are, so a [`Col`] over them copies nothing. A `&[Option<T>]` cannot
//! be lent by either, and takes 16 bytes a row for an `i64` where the bitmap
//! takes one bit.
//!
//! ```
//! use epistat_core::{Col, InputError};
//!
//! // A column without nulls borrows its values.
//! let ages = [34_i64, 51, 29];
//! let col = Col::new(&ages);
//! assert_eq!(col.null_count(), 0);
//! assert_eq!(col.get(1), Some(51));
//!
//! // Arrow's parts: the values, and bit i of the validity set where row i
//! // has a value, least significant bit first.
//! let first_treated = [2004_i64, 0, 2006, 0];
//! let validity = [0b0000_0101_u8];
//! let col = Col::from_arrow(&first_treated, Some(&validity[..]), 0, 4)?;
//! assert_eq!(col.null_count(), 2);
//! assert_eq!(col.iter().collect::<Vec<_>>(), [Some(2004), None, Some(2006), None]);
//!
//! // Options are copied once, for a caller that holds them.
//! let col = Col::from_options(&[Some(1.5), None]);
//! assert_eq!(col.get(1), None);
//! # Ok::<(), InputError>(())
//! ```

use core::fmt;
use core::iter::FusedIterator;

use crate::check::InputError;

/// A column of `len` rows, each a value or null, borrowed from the caller's
/// buffers where it can be.
///
/// The values are a slice with one entry per row; what a null row holds
/// there is unspecified. The validity is a bitmap in Arrow's layout: row `i`
/// has a value where bit `offset + i` is set, counting from the least
/// significant bit of the first byte. A column in which no row is null keeps
/// no bitmap.
///
/// [`get`](Self::get) and [`iter`](Self::iter) read a row as an
/// `Option<T>`. A loop that must go faster reads [`values`](Self::values)
/// and [`validity`](Self::validity) separately.
///
/// A view or an estimator takes a nullable column as `Col<'a, T>`, or as
/// `impl Into<Col<'a, T>>`, where it would take `&'a [Option<T>]`. A call
/// that passes `&[Option<T>]` or `&[T]` still compiles through `From`: the
/// options are copied once, the plain values are borrowed.
///
/// ```
/// use epistat_core::Col;
///
/// fn treated(first_treated: impl Into<Col<'static, i64>>) -> usize {
///     let col = first_treated.into();
///     col.len() - col.null_count()
/// }
///
/// assert_eq!(treated(&[Some(2004), None, Some(2006)][..]), 2);
/// assert_eq!(treated(&[2004, 2006][..]), 2);
/// ```
pub struct Col<'a, T> {
    values: Buf<'a, T>,
    validity: Option<Bitmap<'a>>,
    null_count: usize,
}

/// A slice borrowed from the caller or copied into the column.
enum Buf<'a, T> {
    Borrowed(&'a [T]),
    Owned(Box<[T]>),
}

impl<T> Buf<'_, T> {
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Borrowed(values) => values,
            Self::Owned(values) => values,
        }
    }
}

impl<T: Clone> Clone for Buf<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(values) => Self::Borrowed(values),
            Self::Owned(values) => Self::Owned(values.clone()),
        }
    }
}

/// Validity bits from `offset` on, at least as many as the column has rows.
#[derive(Clone)]
struct Bitmap<'a> {
    bytes: Buf<'a, u8>,
    offset: usize,
}

impl Bitmap<'_> {
    fn is_set(&self, row: usize) -> bool {
        let bit = self.offset + row;
        self.bytes.as_slice()[bit / 8] >> (bit % 8) & 1 == 1
    }
}

/// The number of bits set in `bytes` from bit `offset` for `len` bits; the
/// bytes hold them all.
fn count_set(bytes: &[u8], offset: usize, len: usize) -> usize {
    let end = offset + len;
    let mut bit = offset;
    let mut set = 0;
    while bit < end && !bit.is_multiple_of(8) {
        set += usize::from(bytes[bit / 8] >> (bit % 8) & 1);
        bit += 1;
    }
    while bit + 8 <= end {
        set += bytes[bit / 8].count_ones() as usize;
        bit += 8;
    }
    while bit < end {
        set += usize::from(bytes[bit / 8] >> (bit % 8) & 1);
        bit += 1;
    }
    set
}

impl<'a, T> Col<'a, T> {
    /// A column of `values` in which no row is null. Nothing is copied.
    ///
    /// ```
    /// use epistat_core::Col;
    ///
    /// let col = Col::new(&[3.0, 4.5]);
    /// assert_eq!(col.len(), 2);
    /// assert_eq!(col.null_count(), 0);
    /// ```
    #[must_use]
    pub const fn new(values: &'a [T]) -> Self {
        Self {
            values: Buf::Borrowed(values),
            validity: None,
            null_count: 0,
        }
    }

    /// A column over an Arrow array's parts, checked once. Nothing is
    /// copied.
    ///
    /// `values` holds the column's `len` rows, as arrow-rs and polars give
    /// them with the array's offset already applied. `validity` is the null
    /// buffer, `None` when the array has none, and `offset` the position of
    /// row 0's bit in it, as arrow-rs's `NullBuffer::offset` and polars'
    /// `Bitmap::as_slice` give it. A bitmap longer than needed is accepted,
    /// as Arrow pads its buffers.
    ///
    /// ```
    /// use epistat_core::{Col, InputError};
    ///
    /// // The rows of a sliced array start at bit 3 of its null buffer.
    /// let values = [7_i32, 8, 9];
    /// let validity = [0b0010_1000_u8];
    /// let col = Col::from_arrow(&values, Some(&validity[..]), 3, 3)?;
    /// assert_eq!(col.iter().collect::<Vec<_>>(), [Some(7), None, Some(9)]);
    ///
    /// assert!(Col::from_arrow(&values, Some(&validity[..]), 6, 3).is_err());
    /// # Ok::<(), InputError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] when `values` does not have `len` rows or the
    /// bitmap has fewer than `offset + len` bits, and
    /// [`InputError::TooLarge`] when `offset + len` overflows.
    pub fn from_arrow(
        values: &'a [T],
        validity: Option<&'a [u8]>,
        offset: usize,
        len: usize,
    ) -> Result<Self, InputError> {
        if values.len() != len {
            return Err(InputError::Length {
                what: "values",
                expected: len,
                found: values.len(),
            });
        }
        let Some(bytes) = validity else {
            return Ok(Self::new(values));
        };
        let end = offset.checked_add(len).ok_or(InputError::TooLarge {
            what: "validity offset",
            limit: (usize::MAX - len) as u64,
            found: offset as u64,
        })?;
        let needed = end.div_ceil(8);
        if bytes.len() < needed {
            return Err(InputError::Length {
                what: "validity",
                expected: needed,
                found: bytes.len(),
            });
        }
        let null_count = len - count_set(bytes, offset, len);
        let validity = (null_count > 0).then_some(Bitmap {
            bytes: Buf::Borrowed(bytes),
            offset,
        });
        Ok(Self {
            values: Buf::Borrowed(values),
            validity,
            null_count,
        })
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.as_slice().len()
    }

    /// Whether the column has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of null rows, counted when the column was made.
    #[must_use]
    pub const fn null_count(&self) -> usize {
        self.null_count
    }

    /// Whether row `row` has a value.
    ///
    /// # Panics
    ///
    /// When `row` is not below [`len`](Self::len).
    #[must_use]
    pub fn is_valid(&self, row: usize) -> bool {
        assert!(row < self.len(), "row {row} of a column of {}", self.len());
        self.validity.as_ref().is_none_or(|bits| bits.is_set(row))
    }

    /// The values, one per row. What a null row holds is unspecified: read
    /// [`validity`](Self::validity) or [`is_valid`](Self::is_valid) with it.
    #[must_use]
    pub fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    /// The validity bitmap and the position of row 0's bit in it, `None`
    /// when no row is null. Row `i` has a value where bit `offset + i` is
    /// set, least significant bit first.
    ///
    /// ```
    /// use epistat_core::Col;
    ///
    /// let col = Col::from_options(&[Some(1_u32), None, Some(3)]);
    /// let (bits, offset) = col.validity().expect("a null row");
    /// assert_eq!((bits, offset), (&[0b101_u8][..], 0));
    /// assert_eq!(Col::new(&[1_u32]).validity(), None);
    /// ```
    #[must_use]
    pub fn validity(&self) -> Option<(&[u8], usize)> {
        self.validity
            .as_ref()
            .map(|bits| (bits.bytes.as_slice(), bits.offset))
    }
}

impl<T: Copy> Col<'_, T> {
    /// Row `row`, `None` where it is null.
    ///
    /// ```
    /// use epistat_core::Col;
    ///
    /// let col = Col::from_options(&[Some(2.0), None]);
    /// assert_eq!(col.get(0), Some(2.0));
    /// assert_eq!(col.get(1), None);
    /// ```
    ///
    /// # Panics
    ///
    /// When `row` is not below [`len`](Self::len).
    #[must_use]
    pub fn get(&self, row: usize) -> Option<T> {
        self.is_valid(row).then(|| self.values.as_slice()[row])
    }

    /// The rows in order, each `None` where it is null.
    ///
    /// ```
    /// use epistat_core::Col;
    ///
    /// let col = Col::from_options(&[Some(1_i64), None, Some(3)]);
    /// let present: i64 = col.iter().flatten().sum();
    /// assert_eq!(present, 4);
    /// ```
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            col: self,
            front: 0,
            back: self.len(),
        }
    }
}

impl<'a, T> From<&'a [T]> for Col<'a, T> {
    fn from(values: &'a [T]) -> Self {
        Self::new(values)
    }
}

impl<'a, T> From<&'a Vec<T>> for Col<'a, T> {
    fn from(values: &'a Vec<T>) -> Self {
        Self::new(values)
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for Col<'a, T> {
    fn from(values: &'a [T; N]) -> Self {
        Self::new(values)
    }
}

impl<T: Copy + Default> Col<'_, T> {
    /// A column of `rows`, copied: the values into a vector, in which a null
    /// row holds `T::default()`, and the nulls into a bitmap. A caller with
    /// an Arrow column uses [`from_arrow`](Self::from_arrow), which copies
    /// nothing.
    ///
    /// ```
    /// use epistat_core::Col;
    ///
    /// let col = Col::from_options(&[Some(5_u32), None]);
    /// assert_eq!(col.null_count(), 1);
    /// assert_eq!(col.values(), [5, 0]);
    /// ```
    #[must_use]
    pub fn from_options(rows: &[Option<T>]) -> Self {
        let values: Box<[T]> = rows.iter().map(|row| row.unwrap_or_default()).collect();
        let mut bytes = vec![0_u8; rows.len().div_ceil(8)];
        let mut null_count = 0;
        for (i, row) in rows.iter().enumerate() {
            if row.is_some() {
                bytes[i / 8] |= 1 << (i % 8);
            } else {
                null_count += 1;
            }
        }
        let validity = (null_count > 0).then(|| Bitmap {
            bytes: Buf::Owned(bytes.into_boxed_slice()),
            offset: 0,
        });
        Self {
            values: Buf::Owned(values),
            validity,
            null_count,
        }
    }
}

/// Copies the rows, as [`Col::from_options`].
impl<T: Copy + Default> From<&[Option<T>]> for Col<'_, T> {
    fn from(rows: &[Option<T>]) -> Self {
        Self::from_options(rows)
    }
}

/// Copies the rows, as [`Col::from_options`].
impl<T: Copy + Default> From<&Vec<Option<T>>> for Col<'_, T> {
    fn from(rows: &Vec<Option<T>>) -> Self {
        Self::from_options(rows)
    }
}

/// Copies the rows, as [`Col::from_options`], so an array literal such as
/// `&[Some(1.0), None]` converts as a slice does.
///
/// ```
/// use epistat_core::Col;
///
/// fn nulls<'a>(column: impl Into<Col<'a, f64>>) -> usize {
///     column.into().null_count()
/// }
/// assert_eq!(nulls(&[Some(1.0), None, Some(3.0)]), 1);
/// assert_eq!(nulls(&[1.0, 2.0]), 0);
/// ```
impl<T: Copy + Default, const N: usize> From<&[Option<T>; N]> for Col<'_, T> {
    fn from(rows: &[Option<T>; N]) -> Self {
        Self::from_options(rows)
    }
}

impl<T: Clone> Clone for Col<'_, T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            validity: self.validity.clone(),
            null_count: self.null_count,
        }
    }
}

impl<T> fmt::Debug for Col<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Col")
            .field("len", &self.len())
            .field("null_count", &self.null_count)
            .finish_non_exhaustive()
    }
}

/// Two columns are equal when they have the same rows, whatever their null
/// rows hold among the values and however their bitmaps are offset.
impl<T: Copy + PartialEq> PartialEq for Col<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<'c, T: Copy> IntoIterator for &'c Col<'_, T> {
    type Item = Option<T>;
    type IntoIter = Iter<'c, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The rows of a [`Col`] in order, from [`Col::iter`].
#[derive(Clone, Debug)]
pub struct Iter<'c, T> {
    col: &'c Col<'c, T>,
    front: usize,
    back: usize,
}

impl<T: Copy> Iterator for Iter<'_, T> {
    type Item = Option<T>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.front < self.back).then(|| {
            self.front += 1;
            self.col.get(self.front - 1)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.back - self.front;
        (left, Some(left))
    }
}

impl<T: Copy> DoubleEndedIterator for Iter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        (self.front < self.back).then(|| {
            self.back -= 1;
            self.col.get(self.back)
        })
    }
}

impl<T: Copy> ExactSizeIterator for Iter<'_, T> {}

impl<T: Copy> FusedIterator for Iter<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_counted_across_byte_edges() {
        let bytes = [0b1111_0000_u8, 0b1111_1111, 0b0000_0001];
        assert_eq!(count_set(&bytes, 0, 24), 13);
        assert_eq!(count_set(&bytes, 3, 7), 6);
        assert_eq!(count_set(&bytes, 4, 13), 13);
        assert_eq!(count_set(&bytes, 9, 0), 0);
        for offset in 0..24 {
            for len in 0..=24 - offset {
                let slow = (offset..offset + len)
                    .filter(|bit| bytes[bit / 8] >> (bit % 8) & 1 == 1)
                    .count();
                assert_eq!(count_set(&bytes, offset, len), slow, "{offset} {len}");
            }
        }
    }

    #[test]
    fn arrow_parts_read_as_their_options() {
        let rows = [
            Some(1_i64),
            None,
            Some(3),
            Some(4),
            None,
            None,
            Some(7),
            None,
            Some(9),
        ];
        let copied = Col::from_options(&rows[..]);
        let (bits, offset) = copied.validity().unwrap();
        assert_eq!(offset, 0);
        let values: Vec<i64> = rows.iter().map(|r| r.unwrap_or(-1)).collect();
        let lent = Col::from_arrow(&values, Some(bits), 0, rows.len()).unwrap();
        assert_eq!(lent, copied);
        assert_eq!(lent.null_count(), 4);
        assert_eq!(lent.iter().collect::<Vec<_>>(), rows);
        assert_eq!(
            lent.iter().rev().collect::<Vec<_>>(),
            rows.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(lent.iter().len(), rows.len());
    }

    #[test]
    fn a_shifted_bitmap_reads_from_its_offset() {
        // Rows 0..3 are bits 5, 6 and 7 of the first byte, row 3 bit 0 of
        // the second.
        let bytes = [0b1010_0000_u8, 0b0000_0001];
        let col = Col::from_arrow(&[1, 2, 3, 4], Some(&bytes[..]), 5, 4).unwrap();
        assert_eq!(
            col.iter().collect::<Vec<_>>(),
            [Some(1), None, Some(3), Some(4)]
        );
    }

    #[test]
    fn a_full_bitmap_is_dropped() {
        let col = Col::from_arrow(&[1, 2], Some(&[0b11][..]), 0, 2).unwrap();
        assert_eq!(col.validity(), None);
        assert_eq!(Col::from_options(&[Some(1), Some(2)]).validity(), None);
        assert_eq!(col, Col::new(&[1, 2]));
    }

    #[test]
    fn short_or_mismatched_parts_are_refused() {
        assert_eq!(
            Col::from_arrow(&[1, 2, 3], None, 0, 2),
            Err(InputError::Length {
                what: "values",
                expected: 2,
                found: 3
            })
        );
        assert_eq!(
            Col::from_arrow(&[0_u8; 9], Some(&[0xff][..]), 0, 9),
            Err(InputError::Length {
                what: "validity",
                expected: 2,
                found: 1
            })
        );
        assert!(matches!(
            Col::from_arrow(&[0_u8; 2], Some(&[0xff][..]), usize::MAX, 2),
            Err(InputError::TooLarge { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "row 2 of a column of 2")]
    fn a_row_past_the_end_panics() {
        let _ = Col::from_options(&[Some(1), None]).get(2);
    }

    #[test]
    fn a_parameter_of_type_col_takes_either_slice() {
        fn total(col: impl Into<Col<'static, i64>>) -> i64 {
            col.into().iter().flatten().sum()
        }
        const PLAIN: [i64; 2] = [1, 2];
        assert_eq!(total(&PLAIN[..]), 3);
        assert_eq!(total(&[Some(1), None, Some(4)][..]), 5);
        assert_eq!(total(&vec![Some(2), None]), 2);
    }

    #[test]
    fn an_empty_column_has_no_rows() {
        let col = Col::<f64>::new(&[]);
        assert!(col.is_empty());
        assert_eq!(col.iter().next(), None);
        let empty: [Option<f64>; 0] = [];
        assert!(Col::from_options(&empty[..]).is_empty());
    }
}
