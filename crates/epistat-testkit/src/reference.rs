use serde::{Deserialize, Deserializer, de::DeserializeOwned};
use std::path::Path;

/// Reads and parses a JSON reference file.
///
/// # Panics
///
/// With the path in the message, if the file cannot be read or parsed.
#[must_use]
#[track_caller]
pub fn load_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

/// Deserialises a number written by R's `jsonlite`, where `NA` and `NaN` are
/// `null` and the strings `"NA"`, `"NaN"`, `"Inf"` and `"-Inf"` may appear.
/// `NA` and `NaN` become NaN.
///
/// Use with `#[serde(deserialize_with = "epistat_testkit::na_f64")]`.
///
/// # Errors
///
/// When the value is neither a number, `null` nor one of those strings.
pub fn na_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    Number::deserialize(deserializer)?.value()
}

/// [`na_f64`] for an array.
///
/// # Errors
///
/// When an element is neither a number, `null` nor one of the strings.
pub fn na_f64_vec<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
    Vec::<Number>::deserialize(deserializer)?
        .into_iter()
        .map(Number::value)
        .collect()
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Number {
    Value(f64),
    Missing(()),
    Text(String),
}

impl Number {
    fn value<E: serde::de::Error>(self) -> Result<f64, E> {
        match self {
            Self::Value(value) => Ok(value),
            Self::Missing(()) => Ok(f64::NAN),
            Self::Text(text) => match text.as_str() {
                "NA" | "NaN" => Ok(f64::NAN),
                "Inf" => Ok(f64::INFINITY),
                "-Inf" => Ok(f64::NEG_INFINITY),
                other => Err(E::custom(format!("not a number: {other:?}"))),
            },
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Reference {
        #[serde(deserialize_with = "na_f64")]
        estimate: f64,
        #[serde(deserialize_with = "na_f64_vec")]
        bounds: Vec<f64>,
    }

    #[test]
    fn r_missing_values_become_nan() {
        let parsed: Reference =
            serde_json::from_str(r#"{"estimate": null, "bounds": [1.5, "NA", "Inf", "-Inf"]}"#)
                .unwrap();
        assert!(parsed.estimate.is_nan());
        assert_eq!(parsed.bounds[0].to_bits(), 1.5f64.to_bits());
        assert!(parsed.bounds[1].is_nan());
        assert_eq!(parsed.bounds[2], f64::INFINITY);
        assert_eq!(parsed.bounds[3], f64::NEG_INFINITY);
        assert!(serde_json::from_str::<Reference>(r#"{"estimate": "x", "bounds": []}"#).is_err());
    }

    #[test]
    #[should_panic(expected = "no-such-file.json")]
    fn a_missing_file_names_its_path() {
        let _: Reference = load_json("no-such-file.json");
    }
}
