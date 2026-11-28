//! Number formatting shared by the `Display` impls of the crate.
//!
//! Printing is for reading, so the rules follow R's print methods rather than
//! Rust's shortest round-trip form: a number shows a fixed count of
//! significant digits, in scientific notation when it is very large or very
//! small, and a p-value below the machine epsilon prints as `<2e-16`.

use core::fmt;

/// `value` with `digits` significant digits, trailing zeros kept: `0.510`,
/// `12.3`, `4.90e-14`. Scientific notation is used below `1e-4` and from
/// `1e6` on. A non-finite value prints as Rust prints it.
pub(crate) struct Significant(pub f64, pub usize);

impl fmt::Display for Significant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&significant(self.0, self.1, false))
    }
}

/// As [`Significant`], with trailing zeros and a trailing point removed:
/// `95`, `97.5`.
pub(crate) struct Trimmed(pub f64, pub usize);

impl fmt::Display for Trimmed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&significant(self.0, self.1, true))
    }
}

/// A number as a `Display` impl prints it: `decimals` fixed decimals when the
/// caller gave a precision (`{:.2}`), otherwise `digits` significant digits.
pub(crate) fn number(value: f64, digits: usize, decimals: Option<usize>) -> String {
    match decimals {
        Some(decimals) if value.is_finite() => format!("{value:.decimals$}"),
        _ => significant(value, digits, false),
    }
}

/// An optional number, `NA` when undefined, as R prints it.
pub(crate) fn optional(value: Option<f64>, digits: usize, decimals: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_owned(), |v| number(v, digits, decimals))
}

/// A p-value with two significant digits, `<2e-16` below the machine epsilon
/// as R's `format.pval` prints it, `NA` when undefined.
pub(crate) fn p_value(p: Option<f64>) -> String {
    match p {
        None => "NA".to_owned(),
        Some(p) if p < 2.2e-16 => "<2e-16".to_owned(),
        Some(p) => significant(p, 2, false),
    }
}

/// `= 0.20` or `< 2e-16`, the p-value after its name in a sentence.
pub(crate) fn p_clause(p: f64) -> String {
    if p < 2.2e-16 {
        "< 2e-16".to_owned()
    } else {
        format!("= {}", significant(p, 2, false))
    }
}

fn significant(value: f64, digits: usize, trim: bool) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return if trim || digits <= 1 {
            "0".to_owned()
        } else {
            format!("{:.*}", digits - 1, 0.0)
        };
    }
    let digits = digits.max(1);
    // The exponent after rounding to `digits` significant digits, read from
    // the scientific form so that 0.9995 at three digits counts as 1.00.
    let scientific = format!("{:.*e}", digits - 1, value);
    let exponent: i32 = scientific
        .rsplit_once('e')
        .and_then(|(_, e)| e.parse().ok())
        .unwrap_or(0);
    let mut text = if (-4..6).contains(&exponent) {
        let digits = i64::try_from(digits).unwrap_or(i64::MAX);
        let decimals = usize::try_from(digits - 1 - i64::from(exponent)).unwrap_or(0);
        format!("{value:.decimals$}")
    } else {
        scientific
    };
    if trim {
        let (mantissa, rest) = match text.find('e') {
            Some(at) => (text[..at].to_owned(), text[at..].to_owned()),
            None => (text.clone(), String::new()),
        };
        let mantissa = if mantissa.contains('.') {
            mantissa.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            mantissa
        };
        text = mantissa + &rest;
    }
    text
}

/// Right-aligned columns: each row a label and cells, the header above.
pub(crate) fn table(
    f: &mut fmt::Formatter<'_>,
    header: &[&str],
    rows: &[(String, Vec<String>)],
) -> fmt::Result {
    let label_width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    let mut widths: Vec<usize> = header.iter().map(|h| h.len()).collect();
    for (_, cells) in rows {
        for (width, cell) in widths.iter_mut().zip(cells) {
            *width = (*width).max(cell.len());
        }
    }
    write!(f, "{:label_width$}", "")?;
    for (h, width) in header.iter().zip(&widths) {
        write!(f, "  {h:>width$}")?;
    }
    for (label, cells) in rows {
        write!(f, "\n{label:<label_width$}")?;
        for (cell, width) in cells.iter().zip(&widths) {
            write!(f, "  {cell:>width$}")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn significant_digits_follow_the_rounded_magnitude() {
        assert_eq!(significant(0.510, 3, false), "0.510");
        assert_eq!(significant(0.428_31, 3, false), "0.428");
        assert_eq!(significant(12.345, 3, false), "12.3");
        assert_eq!(significant(0.9995, 3, false), "1.00");
        assert_eq!(significant(-2.0, 3, false), "-2.00");
        assert_eq!(significant(123_456.0, 3, false), "123456");
        assert_eq!(significant(1_234_567.0, 3, false), "1.23e6");
        assert_eq!(significant(4.9e-14, 2, false), "4.9e-14");
        assert_eq!(significant(0.000_12, 2, false), "0.00012");
        assert_eq!(significant(0.0, 3, false), "0.00");
        assert_eq!(significant(f64::NAN, 3, false), "NaN");
    }

    #[test]
    fn trimmed_numbers_drop_trailing_zeros() {
        assert_eq!(significant(95.000_000_000_000_01, 12, true), "95");
        assert_eq!(significant(97.5, 12, true), "97.5");
        assert_eq!(significant(0.0, 12, true), "0");
    }

    #[test]
    fn p_values_print_as_r_prints_them() {
        assert_eq!(p_value(Some(0.2)), "0.20");
        assert_eq!(p_value(Some(0.054_3)), "0.054");
        assert_eq!(p_value(Some(4.9e-14)), "4.9e-14");
        assert_eq!(p_value(Some(1e-300)), "<2e-16");
        assert_eq!(p_value(None), "NA");
        assert_eq!(p_clause(0.01), "= 0.010");
        assert_eq!(p_clause(0.0), "< 2e-16");
    }

    #[test]
    fn a_precision_gives_fixed_decimals() {
        assert_eq!(number(0.512_34, 3, Some(2)), "0.51");
        assert_eq!(number(0.512_34, 3, None), "0.512");
        assert_eq!(optional(None, 3, None), "NA");
    }
}
