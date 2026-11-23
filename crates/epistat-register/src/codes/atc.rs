//! The five levels of an ATC code.
//!
//! An ATC code names a drug by a path of five levels, written with 1, 3, 4, 5
//! and 7 characters \[1\]. `N02BE01` is paracetamol: `N` is the anatomical
//! main group, `N02` the therapeutic subgroup, `N02B` the pharmacological
//! subgroup, `N02BE` the chemical subgroup and `N02BE01` the chemical
//! substance. The pattern of each level is fixed: a letter, two digits, a
//! letter, a letter and two digits. A code has the level of its length, and
//! its ancestors are its prefixes of the shorter lengths.
//!
//! The classification, the defined daily doses and the national lists are WHO
//! data under licence and are not part of the crate.
//!
//! # References
//!
//! 1. WHO Collaborating Centre for Drug Statistics Methodology. ATC/DDD Index.
//!    atcddd.fhi.no/atc_ddd_index/

/// A level of the ATC classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// Level 1, the anatomical main group, one letter.
    Anatomical,
    /// Level 2, the therapeutic subgroup, a letter and two digits.
    Therapeutic,
    /// Level 3, the pharmacological subgroup, four characters.
    Pharmacological,
    /// Level 4, the chemical subgroup, five characters.
    Chemical,
    /// Level 5, the chemical substance, seven characters.
    Substance,
}

impl Level {
    /// The number of characters of a code at this level.
    #[must_use]
    pub const fn code_len(self) -> usize {
        match self {
            Self::Anatomical => 1,
            Self::Therapeutic => 3,
            Self::Pharmacological => 4,
            Self::Chemical => 5,
            Self::Substance => 7,
        }
    }
}

/// The level of an upper-case ATC code, or [`None`] when the code does not
/// follow the pattern of any level.
#[must_use]
pub fn level_of(code: &str) -> Option<Level> {
    let b = code.as_bytes();
    let letter = |i: usize| b.get(i).is_some_and(u8::is_ascii_uppercase);
    let digit = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
    let level = match b.len() {
        1 => Level::Anatomical,
        3 => Level::Therapeutic,
        4 => Level::Pharmacological,
        5 => Level::Chemical,
        7 => Level::Substance,
        _ => return None,
    };
    let ok = letter(0)
        && (b.len() < 3 || (digit(1) && digit(2)))
        && (b.len() < 4 || letter(3))
        && (b.len() < 5 || letter(4))
        && (b.len() < 7 || (digit(5) && digit(6)));
    ok.then_some(level)
}

/// Whether the code is a well-formed ATC code of some level.
#[must_use]
pub fn is_valid(code: &str) -> bool {
    level_of(code).is_some()
}

/// The ancestor of a code at `level`, the code itself at its own level.
/// [`None`] when the code is not well formed or is of a higher level than
/// `level`.
#[must_use]
pub fn ancestor(code: &str, level: Level) -> Option<&str> {
    let own = level_of(code)?;
    (own >= level).then(|| &code[..level.code_len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_follow_the_length_and_pattern() {
        assert_eq!(level_of("N"), Some(Level::Anatomical));
        assert_eq!(level_of("N02"), Some(Level::Therapeutic));
        assert_eq!(level_of("N02B"), Some(Level::Pharmacological));
        assert_eq!(level_of("N02BE"), Some(Level::Chemical));
        assert_eq!(level_of("N02BE01"), Some(Level::Substance));
        for bad in [
            "", "N0", "N02BE0", "N02BE011", "2N2", "N0AB", "n02", "N02B1", "N02BEAB",
        ] {
            assert_eq!(level_of(bad), None, "{bad}");
            assert!(!is_valid(bad));
        }
    }

    #[test]
    fn ancestors_are_prefixes_up_to_the_own_level() {
        assert_eq!(ancestor("N02BE01", Level::Anatomical), Some("N"));
        assert_eq!(ancestor("N02BE01", Level::Therapeutic), Some("N02"));
        assert_eq!(ancestor("N02BE01", Level::Pharmacological), Some("N02B"));
        assert_eq!(ancestor("N02BE01", Level::Chemical), Some("N02BE"));
        assert_eq!(ancestor("N02BE01", Level::Substance), Some("N02BE01"));
        assert_eq!(ancestor("N02B", Level::Chemical), None);
        assert_eq!(ancestor("N02B", Level::Pharmacological), Some("N02B"));
        assert_eq!(ancestor("junk", Level::Anatomical), None);
    }
}
