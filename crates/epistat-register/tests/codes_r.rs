//! Code sets, ATC levels, ICD-8 routing and crosswalks against the rules
//! written out in base R (`tests/data/codes_reference.R`). Every result is a
//! set, a level, a string or a count, so agreement is exact.

use epistat_register::codes::atc::{self, Level};
use epistat_register::codes::{
    CodeMap, CodeSet, Crosswalk, EraCodeSet, MapKind, Revision, normalize_any,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Sets {
    a: String,
    b: String,
    a_in: String,
    b_in: String,
    union: String,
    intersection: String,
    difference: String,
    longest: String,
}

#[derive(Deserialize)]
struct Atc {
    code: Vec<String>,
    level: Vec<i32>,
    ancestors: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct Routing {
    raw: Vec<String>,
    revision: Vec<Option<String>>,
    code: Vec<Option<String>>,
    member: Vec<bool>,
}

#[derive(Deserialize)]
struct Coverage {
    codes: usize,
    mapped: usize,
    unmapped: Vec<String>,
    one_to_one: usize,
    one_to_many: usize,
    many_to_one: usize,
    approximate: usize,
}

#[derive(Deserialize)]
struct Table {
    from: Vec<String>,
    to: Vec<String>,
    kind: Vec<String>,
    ok: bool,
    query: Vec<String>,
    coverage: Option<Coverage>,
}

#[derive(Deserialize)]
struct Reference {
    universe: Vec<String>,
    sets: Vec<Sets>,
    atc: Atc,
    routing: Routing,
    crosswalks: Vec<Table>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/codes_reference.json"
    ))
}

fn bits(set: &CodeSet, universe: &[String]) -> String {
    universe
        .iter()
        .map(|c| if set.contains(c) { '1' } else { '0' })
        .collect()
}

#[test]
fn sets_and_their_combinations_agree_with_r() {
    let r = reference();
    assert_eq!(r.universe.len(), 1554);
    for (k, s) in r.sets.iter().enumerate() {
        let a = CodeSet::parse(&s.a).unwrap();
        let b = CodeSet::parse(&s.b).unwrap();
        let label = format!("pair {k}: {:?} and {:?}", s.a, s.b);
        assert_eq!(bits(&a, &r.universe), s.a_in, "{label}");
        assert_eq!(bits(&b, &r.universe), s.b_in, "{label}");
        assert_eq!(bits(&a.union(&b), &r.universe), s.union, "{label}");
        assert_eq!(bits(&a.intersection(&b), &r.universe), s.intersection, "{label}");
        assert_eq!(bits(&a.difference(&b), &r.universe), s.difference, "{label}");
        let map = CodeMap::new(vec![("a".to_owned(), a), ("b".to_owned(), b)]).unwrap();
        for (code, want) in r.universe.iter().zip(s.longest.chars()) {
            let got = map.longest_match(&[code.as_str()]);
            let got = match got {
                Err(_) => 't',
                Ok(v) => match v[0] {
                    None => 'n',
                    Some(0) => 'a',
                    Some(_) => 'b',
                },
            };
            assert_eq!(got, want, "{label}, code {code}");
        }
    }
}

#[test]
fn atc_levels_and_ancestors_agree_with_r() {
    let r = reference();
    let levels = [
        Level::Anatomical,
        Level::Therapeutic,
        Level::Pharmacological,
        Level::Chemical,
        Level::Substance,
    ];
    for (i, code) in r.atc.code.iter().enumerate() {
        let want = usize::try_from(r.atc.level[i]).ok().map(|l| levels[l]);
        assert_eq!(atc::level_of(code), want, "{code:?}");
        assert_eq!(atc::is_valid(code), want.is_some(), "{code:?}");
        for (l, &level) in levels.iter().enumerate() {
            let expected = Some(r.atc.ancestors[i][l].as_str()).filter(|s| !s.is_empty());
            assert_eq!(atc::ancestor(code, level), expected, "{code:?} at {level:?}");
        }
    }
    assert!(r.atc.level.iter().filter(|&&l| l == 4).count() > 50);
    assert!(r.atc.level.iter().filter(|&&l| l < 0).count() > 50);
}

#[test]
fn routing_by_revision_agrees_with_r() {
    let r = reference();
    let condition = EraCodeSet {
        icd8: CodeSet::parse("410").unwrap(),
        icd10: CodeSet::parse("I21").unwrap(),
    };
    for (i, raw) in r.routing.raw.iter().enumerate() {
        let got = normalize_any(raw);
        let want = r.routing.revision[i].as_deref().map(|rev| {
            let revision = if rev == "Icd8" {
                Revision::Icd8
            } else {
                Revision::Icd10
            };
            (revision, r.routing.code[i].clone().unwrap())
        });
        assert_eq!(got, want, "{raw:?}");
        let member = got.is_some_and(|(_, code)| condition.contains(&code));
        assert_eq!(member, r.routing.member[i], "{raw:?}");
    }
    assert!(r.routing.member.iter().filter(|&&m| m).count() > 20);
}

#[test]
fn crosswalk_validation_and_coverage_agree_with_r() {
    let r = reference();
    let kind = |k: &str| match k {
        "OneToOne" => MapKind::OneToOne,
        "OneToMany" => MapKind::OneToMany,
        "ManyToOne" => MapKind::ManyToOne,
        _ => MapKind::Approximate,
    };
    let mut accepted = 0;
    for (k, t) in r.crosswalks.iter().enumerate() {
        let pairs: Vec<(&str, &str, MapKind)> = (0..t.from.len())
            .map(|i| (t.from[i].as_str(), t.to[i].as_str(), kind(&t.kind[i])))
            .collect();
        let built = Crosswalk::from_pairs(&pairs);
        assert_eq!(built.is_ok(), t.ok, "table {k}");
        let (Ok(walk), Some(want)) = (built, t.coverage.as_ref()) else {
            continue;
        };
        accepted += 1;
        let got = walk.coverage(t.query.iter().map(String::as_str));
        assert_eq!(
            (got.codes, got.mapped, &got.unmapped),
            (want.codes, want.mapped, &want.unmapped),
            "table {k}"
        );
        assert_eq!(
            (
                got.one_to_one,
                got.one_to_many,
                got.many_to_one,
                got.approximate
            ),
            (
                want.one_to_one,
                want.one_to_many,
                want.many_to_one,
                want.approximate
            ),
            "table {k}"
        );
        for code in &t.query {
            let mut targets: Vec<&str> = (0..t.from.len())
                .filter(|&i| &t.from[i] == code)
                .map(|i| t.to[i].as_str())
                .collect();
            targets.sort_unstable();
            let entries: Vec<&str> = walk.map_icd8(code).iter().map(|e| e.to.as_str()).collect();
            assert_eq!(entries, targets, "table {k}, {code}");
        }
    }
    assert!(accepted > 10 && accepted < r.crosswalks.len());
}
