#![expect(
    clippy::cast_possible_truncation,
    reason = "test sizes are small draws from a generator"
)]

use super::*;
use epistat_testkit::SplitMix64;

/// A naive matcher over the original items, the definition in the module docs.
fn naive(items: &[(&str, &str)], code: &str) -> bool {
    items.iter().any(|(lo, hi)| {
        code.len() >= lo.len()
            && lo.as_bytes() <= &code.as_bytes()[..lo.len()]
            && &code.as_bytes()[..lo.len()] <= hi.as_bytes()
    })
}

const ALPHABET: &[u8] = b"0189ABCEIZ";

fn all_codes(max_len: usize) -> Vec<String> {
    let mut out = vec![String::new()];
    let mut frontier = vec![String::new()];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for prefix in &frontier {
            for &b in ALPHABET {
                let mut s = prefix.clone();
                s.push(b as char);
                next.push(s);
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out.retain(|s| !s.is_empty());
    out
}

#[test]
fn parse_reads_prefixes_ranges_and_dots() {
    let set = CodeSet::parse("I21, I252, C00-C26, E10.2-E10.5 g83.0-G83.4").unwrap();
    for yes in [
        "I21", "I219", "I252", "I2521", "C00", "C26", "C269", "E102", "E105", "E1049", "G830",
        "G834",
    ] {
        assert!(set.contains(yes), "{yes}");
    }
    for no in [
        "I2", "I25", "I20", "C27", "E10", "E101", "E106", "G835", "", "C2",
    ] {
        assert!(!set.contains(no), "{no}");
    }
    assert_eq!(set.longest_item(), 4);
    assert!(CodeSet::parse("").unwrap().is_empty());
    assert!(CodeSet::parse(" , ").unwrap().is_empty());
}

#[test]
fn parse_refuses_malformed_items() {
    for bad in [
        "C00-C2",
        "C26-C00",
        "C00-C10-C20",
        "C0$",
        "123456789",
        "-C10",
        "C10-",
    ] {
        assert!(CodeSet::parse(bad).is_err(), "{bad}");
    }
    assert!(CodeSet::from_prefixes(["I21", ""]).is_err());
    assert!(CodeSet::from_prefixes(["I2!"]).is_err());
}

#[test]
fn parse_reads_the_wildcard_and_the_en_dash_as_tables_print_them() {
    let printed =
        CodeSet::parse("I43.x, I42.5\u{2013}I42.9, C37.x\u{2013}C41.x, D65\u{2013}D68.x, R64")
            .unwrap();
    let plain = CodeSet::parse("I43, I425-I429, C37-C41, D65-D68, R64").unwrap();
    assert_eq!(printed, plain);
    for yes in [
        "I43", "I430", "I4381", "I426", "C37", "C419", "D65", "D689", "R64", "R640",
    ] {
        assert!(printed.contains(yes), "{yes}");
    }
    for no in ["I42", "I424", "C36", "C42", "D64", "D69", "R63"] {
        assert!(!printed.contains(no), "{no}");
    }
    // The wildcard is the dot and `x` at the end of an end; elsewhere `x` is
    // a letter of the code.
    assert!(CodeSet::parse("I43X").unwrap().contains("I43X"));
    assert!(!CodeSet::parse("I43X").unwrap().contains("I430"));
    assert_eq!(
        CodeSet::parse("i43.X").unwrap(),
        CodeSet::parse("I43").unwrap()
    );
    for bad in [
        ".x",
        "I42.5\u{2013}I43.x",
        "I42.5\u{2013}I42.9\u{2013}I43.0",
        "I42.5\u{2014}I42.9",
    ] {
        assert!(CodeSet::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn range_edges_and_the_codes_just_outside() {
    let set = CodeSet::parse("C09-C10, E10.2-E10.5").unwrap();
    for (code, expected) in [
        ("C08", false),
        ("C09", true),
        ("C099", true),
        ("C10", true),
        ("C109", true),
        ("C11", false),
        ("E1019", false),
        ("E102", true),
        ("E105Z", true),
        ("E106", false),
    ] {
        assert_eq!(set.contains(code), expected, "{code}");
    }
}

#[test]
fn membership_equals_the_naive_rule_on_every_short_code() {
    let items = [
        ("A0", "A9"),
        ("B1", "B1"),
        ("C18", "C19"),
        ("E1", "E1"),
        ("E101", "E109"),
        ("Z", "Z"),
        ("018", "019"),
    ];
    let spec = "A0-A9 B1 C18-C19 E1 E101-E109 Z 018-019";
    let set = CodeSet::parse(spec).unwrap();
    for code in all_codes(5) {
        assert_eq!(set.contains(&code), naive(&items, &code), "{code}");
    }
}

fn random_items(rng: &mut SplitMix64) -> Vec<(String, String)> {
    (0..=rng.below(4))
        .map(|_| {
            let len = 1 + rng.below(4) as usize;
            let pick = |rng: &mut SplitMix64| -> String {
                (0..len)
                    .map(|_| ALPHABET[rng.below(ALPHABET.len() as u64) as usize] as char)
                    .collect()
            };
            let (a, b) = (pick(rng), pick(rng));
            if a <= b { (a, b) } else { (b, a) }
        })
        .collect()
}

fn spec_of(items: &[(String, String)]) -> String {
    items
        .iter()
        .map(|(a, b)| format!("{a}-{b}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn set_algebra_equals_the_naive_rule_on_random_sets() {
    let codes = all_codes(5);
    let mut rng = SplitMix64::new(20_261_122);
    for round in 0..60 {
        let (ia, ib) = (random_items(&mut rng), random_items(&mut rng));
        let (a, b) = (
            CodeSet::parse(&spec_of(&ia)).unwrap(),
            CodeSet::parse(&spec_of(&ib)).unwrap(),
        );
        let refs = |items: &[(String, String)]| -> Vec<(String, String)> { items.to_vec() };
        let pairs = |items: &[(String, String)]| -> Vec<(String, String)> { refs(items) };
        let m = |items: &[(String, String)], code: &str| {
            let v: Vec<(&str, &str)> = items
                .iter()
                .map(|(x, y)| (x.as_str(), y.as_str()))
                .collect();
            naive(&v, code)
        };
        let (u, i, d) = (a.union(&b), a.intersection(&b), a.difference(&b));
        for code in &codes {
            let (in_a, in_b) = (m(&pairs(&ia), code), m(&pairs(&ib), code));
            assert_eq!(a.contains(code), in_a, "round {round} {code}");
            assert_eq!(u.contains(code), in_a || in_b, "union round {round} {code}");
            assert_eq!(
                i.contains(code),
                in_a && in_b,
                "intersection round {round} {code}"
            );
            assert_eq!(
                d.contains(code),
                in_a && !in_b,
                "difference round {round} {code}"
            );
        }
        assert_eq!(
            i.is_empty(),
            !codes.iter().any(|c| i.contains(c)) && i.is_empty()
        );
        // a = (a minus b) union (a and b), as sets, whatever the construction.
        assert_eq!(a, d.union(&i), "round {round}");
        assert_eq!(a.difference(&a), CodeSet::default());
    }
}

#[test]
fn difference_of_a_prefix_and_a_longer_item_keeps_the_short_codes() {
    let a = CodeSet::parse("E10").unwrap();
    let b = CodeSet::parse("E101").unwrap();
    let d = a.difference(&b);
    assert!(d.contains("E10"));
    assert!(d.contains("E100"));
    assert!(!d.contains("E101"));
    assert!(!d.contains("E1019"));
    assert!(d.contains("E102"));
}

#[test]
fn equal_sets_are_equal_however_they_are_built() {
    let a = CodeSet::parse("C00-C09 C0A-C0Z C10-C19").unwrap();
    let b = CodeSet::parse("C00-C19").unwrap();
    let c = CodeSet::from_prefixes(
        (0..20)
            .map(|n| format!("C{n:02}"))
            .collect::<Vec<_>>()
            .iter()
            .map(String::as_str),
    )
    .unwrap();
    assert_eq!(a, b);
    // In byte order C0A lies between C09 and C10, so a range over the two
    // decades holds codes that the twenty prefixes do not.
    assert_ne!(b, c);
    assert_eq!(c, CodeSet::parse("C00-C09 C10-C19").unwrap());
}

#[test]
fn classify_gives_one_bit_per_set_and_counts_short_codes() {
    let map = CodeMap::new(vec![
        ("diabetes".into(), CodeSet::parse("E10-E14").unwrap()),
        ("with".into(), CodeSet::parse("E10.2-E10.5").unwrap()),
    ])
    .unwrap();
    let got = map.classify(&["E109", "E102", "E10", "I21"]);
    assert_eq!(got.masks, [0b01, 0b11, 0b01, 0]);
    assert_eq!(got.short_codes, 2);
    assert!(
        CodeMap::new(vec![
            ("a".into(), CodeSet::default()),
            ("a".into(), CodeSet::default())
        ])
        .is_err()
    );
    let too_many = (0..65)
        .map(|k| (k.to_string(), CodeSet::default()))
        .collect();
    assert!(CodeMap::new(too_many).is_err());
}

#[test]
fn longest_match_takes_the_most_specific_set() {
    let map = CodeMap::new(vec![
        ("n02".into(), CodeSet::parse("N02").unwrap()),
        ("n02aa".into(), CodeSet::parse("N02AA").unwrap()),
        ("n02aa01".into(), CodeSet::parse("N02AA01").unwrap()),
        ("c".into(), CodeSet::parse("C").unwrap()),
    ])
    .unwrap();
    let got = map
        .longest_match(&["N02BE01", "N02AA05", "N02AA01", "C09AA01", "A01"])
        .unwrap();
    assert_eq!(got, [Some(0), Some(1), Some(2), Some(3), None]);
    let tie = CodeMap::new(vec![
        ("a".into(), CodeSet::parse("N02").unwrap()),
        ("b".into(), CodeSet::parse("N0-N2").unwrap()),
    ])
    .unwrap();
    assert!(tie.longest_match(&["N02A"]).is_ok());
    let tie = CodeMap::new(vec![
        ("a".into(), CodeSet::parse("N02").unwrap()),
        ("b".into(), CodeSet::parse("N01-N03").unwrap()),
    ])
    .unwrap();
    assert!(tie.longest_match(&["N02A"]).is_err());
    let derived = CodeMap::new(vec![(
        "a".into(),
        CodeSet::parse("N02")
            .unwrap()
            .difference(&CodeSet::parse("N02A").unwrap()),
    )])
    .unwrap();
    assert!(derived.longest_match(&["N02"]).is_err());
}

#[test]
fn a_dictionary_classifies_each_distinct_code_once() {
    let rows = ["I21", "E10", "I21", "A01", "E10", "I21"];
    let dict = Dictionary::new(&rows).unwrap();
    assert_eq!(dict.distinct(), ["A01", "E10", "I21"]);
    assert_eq!(dict.row_code(), [2, 1, 2, 0, 1, 2]);
    let map = CodeMap::new(vec![("mi".into(), CodeSet::parse("I21").unwrap())]).unwrap();
    let masks = map.classify(dict.distinct()).masks;
    assert_eq!(dict.gather(&masks), [1, 0, 1, 0, 0, 1]);
}

#[test]
fn codes_longer_than_the_stored_length_match_on_their_first_characters() {
    let set = CodeSet::parse("ABCDEFGH").unwrap();
    assert!(set.contains("ABCDEFGH"));
    assert!(set.contains("ABCDEFGHIJ"));
    assert!(!set.contains("ABCDEFGI"));
    assert!(!set.contains("ABCDEFG"));
}
