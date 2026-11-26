//! Relatedness against `kinship2::kinship`, and pair kinds, family clusters,
//! multiple births and birth order against direct loops in base R
//! (`tests/data/pedigree_reference.R`). Kinship values are sums of powers of
//! one half, exact in a double, so the comparison is exact.

use epistat_register::pedigree::{
    CousinKinds, FamilyLinks, Pedigree, SiblingKinds, delivery_order, multiple_birth_groups,
};
use epistat_testkit::{Fingerprint, assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Kinship {
    person: Vec<f64>,
    mother: Vec<Option<f64>>,
    father: Vec<Option<f64>>,
    founder: Vec<bool>,
    relatedness: Vec<Vec<f64>>,
    inbred_count: usize,
}

#[derive(Deserialize)]
struct Pairs {
    a: Vec<u32>,
    b: Vec<u32>,
    kind: Vec<String>,
}

#[derive(Deserialize)]
struct Kinds {
    person: Vec<f64>,
    mother: Vec<Option<f64>>,
    father: Vec<Option<f64>>,
    birth_mother: Vec<Option<f64>>,
    birth_day: Vec<Option<f64>>,
    siblings: Pairs,
    cousins1: Pairs,
    cousins2: Pairs,
    clusters: Vec<u32>,
    group: Vec<Option<u32>>,
    size: Vec<Option<u32>>,
    order: Vec<Option<u32>>,
    birth_day2: Vec<Option<f64>>,
    tolerance: Vec<ByTolerance>,
    pedigree_clusters: Vec<u32>,
}

#[derive(Deserialize)]
struct ByTolerance {
    tolerance: u32,
    group: Vec<Option<u32>>,
    size: Vec<Option<u32>>,
    order: Vec<Option<u32>>,
}

#[derive(Deserialize)]
struct Reference {
    kinship: Kinship,
    kinds: Kinds,
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "whole ids from R"
)]
fn ids(values: &[f64]) -> Vec<u64> {
    values.iter().map(|&v| v as u64).collect()
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "whole ids from R"
)]
fn opt(values: &[Option<f64>]) -> Vec<Option<u64>> {
    values.iter().map(|v| v.map(|x| x as u64)).collect()
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pedigree_reference.json"
    ))
}

#[test]
#[expect(clippy::cast_possible_truncation, reason = "fewer than 200 persons")]
fn relatedness_equals_twice_kinship2() {
    let k = reference().kinship;
    let n = k.person.len();
    let ped = Pedigree::new(&ids(&k.person), &opt(&k.mother), &opt(&k.father)).unwrap();
    let pairs: Vec<(u32, u32)> = (0..n)
        .flat_map(|i| (i..n).map(move |j| (i as u32, j as u32)))
        .collect();
    let r = ped.relatedness(&pairs, 0).unwrap();
    let mut fp = Fingerprint::new();
    for (&(i, j), got) in pairs.iter().zip(&r) {
        let want = k.relatedness[i as usize][j as usize];
        assert_eq!(got.unwrap().to_bits(), want.to_bits(), "pair ({i}, {j})");
        fp = fp.f64(want);
    }
    assert!(k.inbred_count > 10, "the pedigree has inbred persons");
    // Serial code: the result cannot depend on a thread pool around it.
    assert_same_on_threads(|| {
        ped.relatedness(&pairs, 0)
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect()
    });
    assert_eq!(fp.finish(), 2_817_513_417_404_181_425, "golden hash");

    // Dropping the founders makes their children's parents nodes without a
    // row, and leaves every relatedness among the others unchanged.
    let keep: Vec<usize> = (0..n).filter(|&i| !k.founder[i]).collect();
    let person: Vec<f64> = keep.iter().map(|&i| k.person[i]).collect();
    let mother: Vec<Option<f64>> = keep.iter().map(|&i| k.mother[i]).collect();
    let father: Vec<Option<f64>> = keep.iter().map(|&i| k.father[i]).collect();
    let ped = Pedigree::new(&ids(&person), &opt(&mother), &opt(&father)).unwrap();
    assert!(ped.nodes() > ped.persons());
    let m = keep.len();
    let pairs: Vec<(u32, u32)> = (0..m)
        .flat_map(|i| (i..m).map(move |j| (i as u32, j as u32)))
        .collect();
    let r = ped.relatedness(&pairs, 0).unwrap();
    for (&(i, j), got) in pairs.iter().zip(&r) {
        let want = k.relatedness[keep[i as usize]][keep[j as usize]];
        assert_eq!(got.unwrap().to_bits(), want.to_bits(), "pair ({i}, {j})");
    }
}

fn kind_names<K: std::fmt::Debug>(kinds: &[K]) -> Vec<String> {
    kinds.iter().map(|k| format!("{k:?}")).collect()
}

#[test]
fn pair_kinds_clusters_and_births_agree_with_r() {
    let k = reference().kinds;
    let ped = Pedigree::new(&ids(&k.person), &opt(&k.mother), &opt(&k.father)).unwrap();
    let mut fp = Fingerprint::new();

    let s = ped.sibling_pairs(SiblingKinds::ALL, 100).unwrap();
    assert_eq!((&s.a, &s.b), (&k.siblings.a, &k.siblings.b));
    assert_eq!(kind_names(&s.kind), k.siblings.kind);
    for kind in [
        "Full",
        "MaternalHalf",
        "PaternalHalf",
        "MotherOnly",
        "FatherOnly",
    ] {
        assert!(k.siblings.kind.iter().any(|x| x == kind), "{kind}");
    }
    for (i, name) in kind_names(&s.kind).iter().enumerate() {
        fp = fp
            .u64(u64::from(s.a[i]))
            .u64(u64::from(s.b[i]))
            .u64(name.len() as u64);
    }

    for (degree, want) in [(1, &k.cousins1), (2, &k.cousins2)] {
        let c = ped.cousin_pairs(degree, CousinKinds::ALL, 100).unwrap();
        assert_eq!((&c.a, &c.b), (&want.a, &want.b), "degree {degree}");
        assert_eq!(kind_names(&c.kind), want.kind, "degree {degree}");
        assert!(want.kind.iter().any(|x| x == "Full") && want.kind.iter().any(|x| x == "Half"));
        for i in 0..c.a.len() {
            fp = fp.u64(u64::from(c.a[i])).u64(u64::from(c.b[i]));
        }
    }

    assert_eq!(
        ped.family_clusters(FamilyLinks::SharedParents).unwrap(),
        k.clusters
    );
    fp = fp.u64(k.clusters.iter().map(|&c| u64::from(c)).sum());

    let births = multiple_birth_groups(&opt(&k.birth_mother), &opt_i(&k.birth_day), 0).unwrap();
    assert_eq!(births.group, k.group);
    assert_eq!(births.size, k.size);
    assert!(k.size.contains(&Some(2)));
    let order = delivery_order(&opt(&k.birth_mother), &opt_i(&k.birth_day), 0).unwrap();
    assert_eq!(order, k.order);
    assert!(k.order.iter().any(|o| *o >= Some(2)));
    assert_eq!(fp.finish(), 12_360_024_851_349_320_220, "golden hash");

    assert_eq!(
        ped.family_clusters(FamilyLinks::ParentsAndChildren)
            .unwrap(),
        k.pedigree_clusters
    );
    let day2 = opt_i(&k.birth_day2);
    for t in &k.tolerance {
        let births = multiple_birth_groups(&opt(&k.birth_mother), &day2, t.tolerance).unwrap();
        assert_eq!(births.group, t.group, "tolerance {}", t.tolerance);
        assert_eq!(births.size, t.size, "tolerance {}", t.tolerance);
        let order = delivery_order(&opt(&k.birth_mother), &day2, t.tolerance).unwrap();
        assert_eq!(order, t.order, "tolerance {}", t.tolerance);
    }
    // The tolerance changes the deliveries of these data.
    assert_ne!(k.tolerance[0].size, k.tolerance[1].size);
    assert_ne!(k.tolerance[1].size, k.tolerance[2].size);
}

#[expect(clippy::cast_possible_truncation, reason = "whole days from R")]
fn opt_i(values: &[Option<f64>]) -> Vec<Option<i64>> {
    values.iter().map(|v| v.map(|x| x as i64)).collect()
}
