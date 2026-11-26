//! Family structure from the mother and father of each person: full and half
//! siblings, cousins, multiple births, birth order, pedigree depth and the
//! coefficient of relationship.
//!
//! Persons are rows. A parent is an arbitrary `u64` identifier and need not be
//! a row, so a father who is not in the population file still links the
//! children he has in it. Missing parents are `None`.
//!
//! # Relationships
//!
//! Two persons are
//!
//! - [`SiblingKind::Full`] siblings when both have the same known mother and the
//!   same known father,
//! - [`SiblingKind::MaternalHalf`] when they have the same mother and known,
//!   different fathers, and [`SiblingKind::PaternalHalf`] the other way round,
//! - [`SiblingKind::MotherOnly`] when they have the same mother and the father
//!   of one or both is unknown, and [`SiblingKind::FatherOnly`] likewise. These
//!   two are not promoted to full or to half siblings: the data do not say
//!   which they are, and the caller decides.
//!
//! First cousins have different parents who are siblings, which is to say they
//! share a grandparent and no parent. The degree generalises: degree `k`
//! cousins share an ancestor `k + 1` generations up and no ancestor closer.
//! The pair is [`CousinKind::Full`] when it shares at least two ancestors of
//! that generation (the grandparents of a couple) and [`CousinKind::Half`] when
//! it shares one.
//!
//! # Coefficient of relationship
//!
//! [`Pedigree::relatedness`] returns `r = 2 phi`, with `phi` the kinship
//! coefficient, the probability that an allele drawn from each person at the
//! same locus is identical by descent. For persons `a` and `b` who are not the
//! same person, and `a` not an ancestor of `b`,
//!
//! ```text
//! phi(a, b) = ( phi(mother(a), b) + phi(father(a), b) ) / 2
//! phi(a, a) = ( 1 + F_a ) / 2,        F_a = phi(mother(a), father(a))
//! ```
//!
//! where an unknown parent contributes 0 and `F_a` is 0 when either parent is
//! unknown. Taking the person in the later generation to expand makes the
//! recursion valid, and each value is computed once per requested pair.
//! Full siblings have `r = 0.5`, half siblings 0.25, first cousins 0.125 and a
//! parent and child 0.5. With a parent link missing, the sum over known paths
//! is a lower bound. Persons with no known common ancestor have `r = 0` only
//! under the assumption that the unknown ancestors are unrelated, which is
//! the convention of `kinship2`. [`Pedigree::relatedness`] therefore reports
//! `None` for a zero where either ancestry is incomplete to the number of
//! generations the caller requires, rather than a zero that is not known.
//!
//! # Algorithm
//!
//! Persons and parents are nodes, rows first. A level is computed for every node,
//! 0 when no parent is known and one more than the highest level of its parents
//! otherwise, by processing nodes whose parents are done. A node that is never
//! reached lies on or below a cycle, and the cycle is reported by id. An
//! ancestor has a strictly lower level than its descendant.
//!
//! Sibling and cousin pairs are found by grouping persons by the parent, or by
//! the ancestor of the given generation, and pairing the members of each
//! group. Pairs are produced person by person in increasing row order, and
//! within a person in increasing order of the other row, so the output does not
//! depend on hash order or threads. A group larger than a given limit is
//! refused with the parent's id, because a group of `k` people has `k(k-1)/2`
//! pairs and a corrupt record would give millions.
//!
//! Family clusters follow the links the caller names in [`FamilyLinks`]: a
//! shared parent, by [`connected_components`] over the mother and father
//! columns, or every person-to-parent link, by a disjoint-set forest over all
//! nodes whose roots are the smallest node, so that the root of a person's
//! set is a person and the numbering is by first row.
//!
//! Multiple births and birth order group a mother's births into deliveries.
//! Her births are sorted by day, and a birth opens a new delivery when it lies
//! more than the caller's tolerance of days after the first birth of the
//! current one. With a tolerance of 0 a delivery is one day.
//!
//! # Not included
//!
//! A pair kind is never inferred from birth dates, and full siblings are not
//! deduced from a shared mother alone. The inbreeding coefficient of an
//! unrelated founder is taken as 0. Coverage of parent links in older
//! generations is the caller's concern.
//!
//! # References
//!
//! 1. Malecot G. Les mathematiques de l'heredite. Paris: Masson; 1948.
//! 2. Lange K. Mathematical and Statistical Methods for Genetic Analysis.
//!    2nd ed. New York: Springer; 2002. Chapter 5.
//! 3. Sinnwell JP, Therneau TM, Schaid DJ. The kinship2 R package for
//!    pedigree data. Hum Hered. 2014;78(2):91-3.
//! 4. Frisell T, Oberg S, Kuja-Halkola R, Sjolander A. Sibling comparison
//!    designs: bias from non-shared confounders and measurement error.
//!    Epidemiology. 2012;23(5):713-20.

#![expect(
    clippy::cast_possible_truncation,
    reason = "node and row numbers are checked against the u32 limit before they are narrowed"
)]

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::relations::{DisjointSets, connected_components};
use crate::RegisterError;

/// How [`consensus_parents`] picks one parent from the rows of a person.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentRule {
    /// The smallest non-missing identifier. Independent of row order.
    SmallestId,
    /// The identifier that occurs most often, the smallest among those tied.
    /// Independent of row order.
    MostFrequent,
    /// The first non-missing identifier in row order. Depends on row order.
    FirstObserved,
}

/// Which links [`Pedigree::family_clusters`] follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FamilyLinks {
    /// A shared mother or father, so a cluster is a sibship and the half
    /// siblings it reaches.
    SharedParents,
    /// Every link from a person to a parent, so a cluster is a connected
    /// component of the whole pedigree.
    ParentsAndChildren,
}

/// One parent per person, chosen from rows that may disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsensusParents {
    /// The distinct persons, sorted.
    pub person: Vec<u64>,
    /// The chosen mother of each, or `None` when no row names one.
    pub mother: Vec<Option<u64>>,
    /// The chosen father of each, or `None` when no row names one.
    pub father: Vec<Option<u64>>,
}

/// Chooses one mother and one father for each person from repeated rows, such
/// as snapshots that disagree. The mother and the father are chosen
/// independently by `rule`.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length.
pub fn consensus_parents(
    rows_person: &[u64],
    mother: &[Option<u64>],
    father: &[Option<u64>],
    rule: ParentRule,
) -> Result<ConsensusParents, RegisterError> {
    let n = rows_person.len();
    if mother.len() != n || father.len() != n {
        return Err(RegisterError::input(
            "person, mother and father must have one entry per row",
        ));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&row| (rows_person[row], row));
    let mut out = ConsensusParents {
        person: Vec::new(),
        mother: Vec::new(),
        father: Vec::new(),
    };
    let mut i = 0;
    while i < n {
        let person = rows_person[order[i]];
        let mut j = i;
        while j < n && rows_person[order[j]] == person {
            j += 1;
        }
        let run = &order[i..j];
        out.person.push(person);
        out.mother
            .push(choose(run.iter().map(|&r| mother[r]), rule));
        out.father
            .push(choose(run.iter().map(|&r| father[r]), rule));
        i = j;
    }
    Ok(out)
}

fn choose(ids: impl Iterator<Item = Option<u64>>, rule: ParentRule) -> Option<u64> {
    let mut ids = ids.flatten();
    match rule {
        ParentRule::SmallestId => ids.min(),
        ParentRule::FirstObserved => ids.next(),
        ParentRule::MostFrequent => {
            let mut all: Vec<u64> = ids.collect();
            all.sort_unstable();
            let mut best: Option<(usize, u64)> = None;
            let mut i = 0;
            while i < all.len() {
                let mut j = i;
                while j < all.len() && all[j] == all[i] {
                    j += 1;
                }
                if best.is_none_or(|(count, _)| j - i > count) {
                    best = Some((j - i, all[i]));
                }
                i = j;
            }
            best.map(|(_, id)| id)
        }
    }
}

/// The relationship of two persons who share a parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SiblingKind {
    /// The same known mother and the same known father.
    Full,
    /// The same mother, known and different fathers.
    MaternalHalf,
    /// The same father, known and different mothers.
    PaternalHalf,
    /// The same mother, and the father of one or both unknown.
    MotherOnly,
    /// The same father, and the mother of one or both unknown.
    FatherOnly,
}

/// Which sibling kinds to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one switch for each kind, as the caller chooses them"
)]
pub struct SiblingKinds {
    /// Report [`SiblingKind::Full`].
    pub full: bool,
    /// Report [`SiblingKind::MaternalHalf`].
    pub maternal_half: bool,
    /// Report [`SiblingKind::PaternalHalf`].
    pub paternal_half: bool,
    /// Report [`SiblingKind::MotherOnly`].
    pub mother_only: bool,
    /// Report [`SiblingKind::FatherOnly`].
    pub father_only: bool,
}

impl SiblingKinds {
    /// Every kind.
    pub const ALL: Self = Self {
        full: true,
        maternal_half: true,
        paternal_half: true,
        mother_only: true,
        father_only: true,
    };

    fn wants(self, kind: SiblingKind) -> bool {
        match kind {
            SiblingKind::Full => self.full,
            SiblingKind::MaternalHalf => self.maternal_half,
            SiblingKind::PaternalHalf => self.paternal_half,
            SiblingKind::MotherOnly => self.mother_only,
            SiblingKind::FatherOnly => self.father_only,
        }
    }
}

/// Sibling pairs `(a[i], b[i])` with `a < b`, in order of `a` then `b`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SiblingPairs {
    /// The lower row of each pair.
    pub a: Vec<u32>,
    /// The higher row of each pair.
    pub b: Vec<u32>,
    /// The kind of each pair.
    pub kind: Vec<SiblingKind>,
}

/// The relationship of two cousins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CousinKind {
    /// At least two shared ancestors of the cousin generation, as when the
    /// parents are full siblings.
    Full,
    /// One shared ancestor of the cousin generation, as when the parents are
    /// half siblings.
    Half,
}

/// Which cousin kinds to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CousinKinds {
    /// Report [`CousinKind::Full`].
    pub full: bool,
    /// Report [`CousinKind::Half`].
    pub half: bool,
}

impl CousinKinds {
    /// Both kinds.
    pub const ALL: Self = Self {
        full: true,
        half: true,
    };
}

/// Cousin pairs `(a[i], b[i])` with `a < b`, in order of `a` then `b`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CousinPairs {
    /// The lower row of each pair.
    pub a: Vec<u32>,
    /// The higher row of each pair.
    pub b: Vec<u32>,
    /// The kind of each pair.
    pub kind: Vec<CousinKind>,
}

/// Whether a parent is a mother or a father.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParentRole {
    /// The mother column.
    Mother,
    /// The father column.
    Father,
}

/// A parent with many children in the rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LargeGroup {
    /// The parent's identifier.
    pub parent: u64,
    /// Which column the parent is in.
    pub role: ParentRole,
    /// The number of rows that name the parent.
    pub size: usize,
}

/// The ancestors of each person within some generations, distinct, at the
/// lowest generation at which they occur, in order of generation then node.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ancestors {
    /// Row range of each person, of length persons plus one.
    pub offsets: Vec<usize>,
    /// The node of each ancestor; [`Pedigree::node_id`] gives its identifier.
    pub node: Vec<u32>,
    /// The generation: 1 for a parent, 2 for a grandparent.
    pub generation: Vec<u8>,
}

/// Multiple births among the rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MultipleBirths {
    /// A number for each delivery, the rows of one mother born on one day or
    /// within the tolerance of its first birth, from 0 in order of first row,
    /// or `None` when the mother or day is missing.
    pub group: Vec<Option<u32>>,
    /// The size of the group: 1 for a single birth, 2 for twins.
    pub size: Vec<Option<u32>>,
}

/// The rows with a mother and a birth day, sorted by mother, day and row, and
/// for each the number of its delivery among all mothers' deliveries in that
/// order. A delivery is a run of a mother's births whose days lie within
/// `tolerance_days` of the first birth of the run.
fn deliveries(
    mother: &[Option<u64>],
    birth_day: &[Option<i64>],
    tolerance_days: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = mother.len();
    let mut rows: Vec<u32> = (0..n as u32)
        .filter(|&r| mother[r as usize].is_some() && birth_day[r as usize].is_some())
        .collect();
    rows.sort_by_key(|&r| (mother[r as usize], birth_day[r as usize], r));
    let mut delivery = Vec::with_capacity(rows.len());
    let mut current = 0_u32;
    let mut first_day = i64::MIN;
    for (k, &r) in rows.iter().enumerate() {
        let r = r as usize;
        let day = birth_day[r].unwrap_or(i64::MIN);
        let same_mother = k > 0 && mother[rows[k - 1] as usize] == mother[r];
        let joins = same_mother && day.saturating_sub(first_day) <= i64::from(tolerance_days);
        if k > 0 && !joins {
            current += 1;
        }
        if !joins {
            first_day = day;
        }
        delivery.push(current);
    }
    (rows, delivery)
}

/// Groups rows by mother and delivery. A group of two or more is a multiple
/// birth.
///
/// With `tolerance_days` 0 a delivery is the births of one mother on one day.
/// With a tolerance of `k`, a birth within `k` days of the first birth of the
/// mother's current delivery joins it, so twins born either side of midnight
/// are one delivery with a tolerance of 1. The window is counted from the
/// first birth of the delivery, not from the birth before, so a chain of
/// births a day apart does not grow a delivery beyond `k` days. Two
/// deliveries of one mother cannot be days apart, so a tolerance of a day or
/// two joins only births the register dates on different days; a tolerance of
/// weeks would also join a delayed delivery of a second twin, which is rare
/// and is the caller's choice to make.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length or there are more
/// than `u32::MAX` rows.
pub fn multiple_birth_groups(
    mother: &[Option<u64>],
    birth_day: &[Option<i64>],
    tolerance_days: u32,
) -> Result<MultipleBirths, RegisterError> {
    let n = mother.len();
    check_rows(n, birth_day.len())?;
    let (rows, delivery) = deliveries(mother, birth_day, tolerance_days);
    let groups = delivery.last().map_or(0, |&d| d as usize + 1);
    let mut head = vec![u32::MAX; groups];
    let mut count = vec![0_u32; groups];
    for (&r, &d) in rows.iter().zip(&delivery) {
        head[d as usize] = head[d as usize].min(r);
        count[d as usize] += 1;
    }
    // Number the groups in order of their first row.
    let mut by_head: Vec<u32> = (0..groups as u32).collect();
    by_head.sort_by_key(|&g| head[g as usize]);
    let mut id = vec![0_u32; groups];
    for (k, &g) in by_head.iter().enumerate() {
        id[g as usize] = k as u32;
    }
    let mut out = MultipleBirths {
        group: vec![None; n],
        size: vec![None; n],
    };
    for (&r, &d) in rows.iter().zip(&delivery) {
        out.group[r as usize] = Some(id[d as usize]);
        out.size[r as usize] = Some(count[d as usize]);
    }
    Ok(out)
}

/// The birth order of each row within its mother: 0 for the first delivery.
/// Rows of one delivery share an order, so twins are one delivery. A delivery
/// is the births of one mother on one day, or within `tolerance_days` of its
/// first birth, as in [`multiple_birth_groups`]. `None` when the mother or
/// the day is missing.
///
/// The rows must hold every birth of every mother present, or the order is
/// too low.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length or there are more
/// than `u32::MAX` rows.
pub fn delivery_order(
    mother: &[Option<u64>],
    birth_day: &[Option<i64>],
    tolerance_days: u32,
) -> Result<Vec<Option<u32>>, RegisterError> {
    let n = mother.len();
    check_rows(n, birth_day.len())?;
    let (rows, delivery) = deliveries(mother, birth_day, tolerance_days);
    let mut out = vec![None; n];
    let mut first_of_mother = 0_u32;
    for (k, &r) in rows.iter().enumerate() {
        if k == 0 || mother[rows[k - 1] as usize] != mother[r as usize] {
            first_of_mother = delivery[k];
        }
        out[r as usize] = Some(delivery[k] - first_of_mother);
    }
    Ok(out)
}

fn check_rows(n: usize, other: usize) -> Result<(), RegisterError> {
    if other != n {
        return Err(RegisterError::input("columns must have one entry per row"));
    }
    if u32::try_from(n).is_err() {
        return Err(RegisterError::input(format!(
            "{n} rows is more than a u32 row number holds"
        )));
    }
    Ok(())
}

/// A pedigree of persons and their parents.
#[derive(Clone, Debug)]
pub struct Pedigree {
    persons: usize,
    ids: Vec<u64>,
    mother: Vec<Option<u32>>,
    father: Vec<Option<u32>>,
    /// Parents before children.
    topo: Vec<u32>,
    level: Vec<u32>,
}

impl Pedigree {
    /// Builds a pedigree. Persons are the rows, numbered from 0; a parent who
    /// is not a row becomes a node without a row.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the columns differ in length, a person
    /// occurs twice, a person has the same mother and father, there are more
    /// nodes than a `u32` holds, or a person is their own ancestor (the error
    /// names a person on the cycle).
    pub fn new(
        person_id: &[u64],
        mother_id: &[Option<u64>],
        father_id: &[Option<u64>],
    ) -> Result<Self, RegisterError> {
        let n = person_id.len();
        if mother_id.len() != n || father_id.len() != n {
            return Err(RegisterError::input(
                "person, mother and father must have one entry per row",
            ));
        }
        if u32::try_from(n.saturating_mul(3)).is_err() {
            return Err(RegisterError::input(format!(
                "{n} persons is more than a pedigree takes"
            )));
        }
        let (ids, mut mother, mut father) = code_nodes(person_id, mother_id, father_id)?;
        mother.resize(ids.len(), None);
        father.resize(ids.len(), None);
        let mut pedigree = Self {
            persons: n,
            ids,
            mother,
            father,
            topo: Vec::new(),
            level: Vec::new(),
        };
        pedigree.order()?;
        Ok(pedigree)
    }

    /// Levels and a parents-before-children order, or the cycle.
    fn order(&mut self) -> Result<(), RegisterError> {
        let m = self.ids.len();
        let mut pending = vec![0_u8; m];
        let mut child_count = vec![0_usize; m + 1];
        for (x, waiting) in pending.iter_mut().enumerate() {
            for p in [self.mother[x], self.father[x]].into_iter().flatten() {
                *waiting += 1;
                child_count[p as usize + 1] += 1;
            }
        }
        for i in 0..m {
            child_count[i + 1] += child_count[i];
        }
        let mut fill = child_count.clone();
        let mut children = vec![0_u32; child_count[m]];
        for x in 0..m {
            for p in [self.mother[x], self.father[x]].into_iter().flatten() {
                children[fill[p as usize]] = x as u32;
                fill[p as usize] += 1;
            }
        }
        let mut level = vec![0_u32; m];
        let mut topo: Vec<u32> = (0..m as u32)
            .filter(|&x| pending[x as usize] == 0)
            .collect();
        let mut head = 0;
        while head < topo.len() {
            let x = topo[head] as usize;
            head += 1;
            for &c in &children[child_count[x]..child_count[x + 1]] {
                let c = c as usize;
                level[c] = level[c].max(level[x] + 1);
                pending[c] -= 1;
                if pending[c] == 0 {
                    topo.push(c as u32);
                }
            }
        }
        if topo.len() < m {
            // Every node left has a parent that is also left. Walking to such
            // a parent must repeat a node, and that node is on a cycle.
            let mut seen = vec![false; m];
            let mut x = (0..m).find(|&x| pending[x] > 0).unwrap_or(0);
            while !seen[x] {
                seen[x] = true;
                x = [self.mother[x], self.father[x]]
                    .into_iter()
                    .flatten()
                    .map(|p| p as usize)
                    .find(|&p| pending[p] > 0)
                    .unwrap_or(x);
            }
            return Err(RegisterError::input(format!(
                "person {} is their own ancestor",
                self.ids[x]
            )));
        }
        self.topo = topo;
        self.level = level;
        Ok(())
    }

    /// The number of persons, which are nodes `0..persons`.
    #[must_use]
    pub fn persons(&self) -> usize {
        self.persons
    }

    /// The number of nodes: the persons and the parents who are not persons.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.ids.len()
    }

    /// The identifier of a node.
    ///
    /// # Panics
    ///
    /// When `node` is not below [`Pedigree::nodes`].
    #[must_use]
    pub fn node_id(&self, node: u32) -> u64 {
        self.ids[node as usize]
    }

    /// The generations of recorded ancestry of each person: 0 when no parent is
    /// known, and one more than the deepest of the parents otherwise. Saturates
    /// at 255.
    #[must_use]
    pub fn depth(&self) -> Vec<u8> {
        self.level[..self.persons]
            .iter()
            .map(|&l| u8::try_from(l).unwrap_or(u8::MAX))
            .collect()
    }

    /// The generations through which every ancestor slot of each person is
    /// known: 0 when either parent is unknown, 1 when both are known and
    /// neither has both of theirs, and so on. Saturates at 255.
    #[must_use]
    pub fn complete_generations(&self) -> Vec<u8> {
        self.complete_all()[..self.persons].to_vec()
    }

    fn complete_all(&self) -> Vec<u8> {
        let mut complete = vec![0_u8; self.ids.len()];
        for &x in &self.topo {
            let x = x as usize;
            if let (Some(m), Some(f)) = (self.mother[x], self.father[x]) {
                complete[x] = complete[m as usize]
                    .min(complete[f as usize])
                    .saturating_add(1);
            }
        }
        complete
    }

    /// The family cluster of each person under the links the caller names.
    ///
    /// - [`FamilyLinks::SharedParents`]: persons who share a mother or a
    ///   father, directly or through a chain of persons who do, have the same
    ///   number. Half siblings joined through a third are one cluster, a
    ///   mother is not in the cluster of her children, and persons with no
    ///   known parent are their own cluster.
    /// - [`FamilyLinks::ParentsAndChildren`]: every link from a person to a
    ///   parent joins them, whether or not the parent is a row, so a cluster
    ///   is a connected component of the pedigree. Parents and their children,
    ///   grandparents, aunts, uncles and cousins who are rows are together.
    ///
    /// Clusters are numbered from 0 in order of first row. The second reading
    /// suits rows that hold parents and children together, or a cluster that
    /// should keep cousins together; the first suits rows of children only,
    /// clustered on their sibship.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] if the pedigree is too large for
    /// [`connected_components`], which cannot happen for one built by
    /// [`Pedigree::new`].
    pub fn family_clusters(&self, links: FamilyLinks) -> Result<Vec<u32>, RegisterError> {
        match links {
            FamilyLinks::SharedParents => self.sibship_clusters(),
            FamilyLinks::ParentsAndChildren => Ok(self.pedigree_components()),
        }
    }

    fn sibship_clusters(&self) -> Result<Vec<u32>, RegisterError> {
        // The same code in the mother and the father column is two parents, and
        // a column holds at most one distinct parent per person, so each
        // column is renumbered from 0 and stays below the number of rows.
        let dense = |parent: &[Option<u32>]| -> Vec<Option<u32>> {
            let mut map = vec![u32::MAX; self.ids.len()];
            let mut next = 0_u32;
            parent[..self.persons]
                .iter()
                .map(|p| {
                    p.map(|node| {
                        let slot = &mut map[node as usize];
                        if *slot == u32::MAX {
                            *slot = next;
                            next += 1;
                        }
                        *slot
                    })
                })
                .collect()
        };
        let (m, f) = (dense(&self.mother), dense(&self.father));
        connected_components(&[&m, &f])
    }

    /// Components of the graph of nodes joined by person-to-parent links. The
    /// root of a set is its smallest node, and persons are the first nodes, so
    /// the root of a person's component is a person.
    fn pedigree_components(&self) -> Vec<u32> {
        let mut sets = DisjointSets::new(self.ids.len());
        for x in 0..self.persons {
            for p in [self.mother[x], self.father[x]].into_iter().flatten() {
                sets.union(x as u32, p);
            }
        }
        let mut number = vec![u32::MAX; self.persons];
        let mut next = 0_u32;
        (0..self.persons)
            .map(|x| {
                let root = sets.find(x as u32) as usize;
                if number[root] == u32::MAX {
                    number[root] = next;
                    next += 1;
                }
                number[root]
            })
            .collect()
    }

    /// The parents with more than `threshold` children among the rows, mothers
    /// first and then fathers, each in order of identifier.
    #[must_use]
    pub fn large_groups(&self, threshold: usize) -> Vec<LargeGroup> {
        let nodes = self.ids.len();
        let by_mother = Groups::new(&self.mother[..self.persons], nodes);
        let by_father = Groups::new(&self.father[..self.persons], nodes);
        self.large_groups_of([&by_mother, &by_father], threshold)
    }

    /// [`Pedigree::large_groups`] from the groups of the mother and the father
    /// column.
    fn large_groups_of(&self, groups: [&Groups; 2], threshold: usize) -> Vec<LargeGroup> {
        let mut out = Vec::new();
        for (role, groups) in [ParentRole::Mother, ParentRole::Father]
            .into_iter()
            .zip(groups)
        {
            let mut part: Vec<LargeGroup> = (0..self.ids.len())
                .filter(|&node| groups.members(node).len() > threshold)
                .map(|node| LargeGroup {
                    parent: self.ids[node],
                    role,
                    size: groups.members(node).len(),
                })
                .collect();
            part.sort_by_key(|g| g.parent);
            out.extend(part);
        }
        out
    }

    /// The sibling pairs of the requested kinds.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when a parent has more than `max_group_size`
    /// children among the rows; the error names the parent.
    pub fn sibling_pairs(
        &self,
        kinds: SiblingKinds,
        max_group_size: usize,
    ) -> Result<SiblingPairs, RegisterError> {
        let mut out = SiblingPairs::default();
        self.for_each_sibling_pair(kinds, max_group_size, |a, b, kind| {
            out.a.push(a);
            out.b.push(b);
            out.kind.push(kind);
        })?;
        Ok(out)
    }

    /// As [`Pedigree::sibling_pairs`], handing each pair to `emit(a, b, kind)`
    /// in order of `a` then `b` without collecting them.
    ///
    /// # Errors
    ///
    /// As [`Pedigree::sibling_pairs`]; nothing is emitted when it fails.
    pub fn for_each_sibling_pair(
        &self,
        kinds: SiblingKinds,
        max_group_size: usize,
        mut emit: impl FnMut(u32, u32, SiblingKind),
    ) -> Result<(), RegisterError> {
        let nodes = self.ids.len();
        let by_mother = Groups::new(&self.mother[..self.persons], nodes);
        let by_father = Groups::new(&self.father[..self.persons], nodes);
        if let Some(g) = self
            .large_groups_of([&by_mother, &by_father], max_group_size)
            .first()
        {
            return Err(RegisterError::input(format!(
                "{:?} {} has {} children among the rows, more than the limit {max_group_size}",
                g.role, g.parent, g.size
            )));
        }
        let mut found: Vec<(u32, SiblingKind)> = Vec::new();
        for a in 0..self.persons as u32 {
            let (ma, fa) = (self.mother[a as usize], self.father[a as usize]);
            found.clear();
            if let Some(m) = ma {
                for &b in later(by_mother.members(m as usize), a) {
                    let fb = self.father[b as usize];
                    let kind = match (fa, fb) {
                        (Some(x), Some(y)) if x == y => SiblingKind::Full,
                        (Some(_), Some(_)) => SiblingKind::MaternalHalf,
                        _ => SiblingKind::MotherOnly,
                    };
                    found.push((b, kind));
                }
            }
            if let Some(f) = fa {
                for &b in later(by_father.members(f as usize), a) {
                    let mb = self.mother[b as usize];
                    let kind = match (ma, mb) {
                        // Already found as full siblings through the mother.
                        (Some(x), Some(y)) if x == y => continue,
                        (Some(_), Some(_)) => SiblingKind::PaternalHalf,
                        _ => SiblingKind::FatherOnly,
                    };
                    found.push((b, kind));
                }
            }
            found.sort_unstable_by_key(|&(b, _)| b);
            for &(b, kind) in &found {
                if kinds.wants(kind) {
                    emit(a, b, kind);
                }
            }
        }
        Ok(())
    }

    /// The ancestors of each person within `generations` generations.
    #[must_use]
    pub fn ancestors(&self, generations: u32) -> Ancestors {
        let mut out = Ancestors {
            offsets: vec![0],
            ..Ancestors::default()
        };
        let mut seen: HashMap<u32, u8> = HashMap::new();
        for x in 0..self.persons as u32 {
            seen.clear();
            let mut frontier = vec![x];
            let mut step = 0_u8;
            let mut list: Vec<(u8, u32)> = Vec::new();
            while u32::from(step) < generations && !frontier.is_empty() {
                step = step.saturating_add(1);
                let mut next = Vec::new();
                for &y in &frontier {
                    for p in [self.mother[y as usize], self.father[y as usize]]
                        .into_iter()
                        .flatten()
                    {
                        if seen.insert(p, step).is_none() {
                            list.push((step, p));
                            next.push(p);
                        }
                    }
                }
                frontier = next;
            }
            list.sort_unstable();
            for (g, p) in list {
                out.node.push(p);
                out.generation.push(g);
            }
            out.offsets.push(out.node.len());
        }
        out
    }

    /// The cousin pairs of `degree` (1 for first cousins) and the requested
    /// kinds.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when `degree` is 0 or more than 10, or a
    /// shared ancestor has more than `max_group_size` descendants of the
    /// cousin generation among the rows; the error names the ancestor.
    pub fn cousin_pairs(
        &self,
        degree: u32,
        kinds: CousinKinds,
        max_group_size: usize,
    ) -> Result<CousinPairs, RegisterError> {
        let mut out = CousinPairs::default();
        self.for_each_cousin_pair(degree, kinds, max_group_size, |a, b, kind| {
            out.a.push(a);
            out.b.push(b);
            out.kind.push(kind);
        })?;
        Ok(out)
    }

    /// As [`Pedigree::cousin_pairs`], handing each pair to `emit(a, b, kind)` in
    /// order of `a` then `b`.
    ///
    /// # Errors
    ///
    /// As [`Pedigree::cousin_pairs`]; nothing is emitted when it fails.
    pub fn for_each_cousin_pair(
        &self,
        degree: u32,
        kinds: CousinKinds,
        max_group_size: usize,
        mut emit: impl FnMut(u32, u32, CousinKind),
    ) -> Result<(), RegisterError> {
        if degree == 0 || degree > 10 {
            return Err(RegisterError::input(
                "the cousin degree must be from 1 to 10",
            ));
        }
        let top = degree as usize + 1;
        // levels[k - 1] holds the distinct ancestors of each person exactly k
        // generations up, sorted.
        let levels = self.ancestors_by_generation(top);
        let last = &levels[top - 1];
        let nodes = self.ids.len();
        // Members of each ancestor's group, ascending by row: a counting sort
        // of the persons by ancestor, filled in row order.
        let mut group_start = vec![0_usize; nodes + 1];
        for &anc in &last.node {
            group_start[anc as usize + 1] += 1;
        }
        for i in 0..nodes {
            group_start[i + 1] += group_start[i];
        }
        let mut fill = group_start.clone();
        // Each member carries its parents, which the test for a closer shared
        // ancestor reads first, so that it does not look them up at random.
        let mut members = vec![(0_u32, [NO_NODE; 2]); last.node.len()];
        for x in 0..self.persons {
            for &anc in last.of(x) {
                members[fill[anc as usize]] = (x as u32, self.parent_pair(x));
                fill[anc as usize] += 1;
            }
        }
        drop(fill);
        if let Some(anc) =
            (0..nodes).find(|&a| group_start[a + 1] - group_start[a] > max_group_size)
        {
            return Err(RegisterError::input(format!(
                "ancestor {} has {} descendants {top} generations down among the rows, more than the limit {max_group_size}",
                self.ids[anc],
                group_start[anc + 1] - group_start[anc]
            )));
        }
        let mut candidates: Vec<(u32, [u32; 2])> = Vec::new();
        for a in 0..self.persons as u32 {
            candidates.clear();
            for &anc in last.of(a as usize) {
                let group = &members[group_start[anc as usize]..group_start[anc as usize + 1]];
                candidates.extend_from_slice(&group[group.partition_point(|m| m.0 <= a)..]);
            }
            // Equal rows carry equal parents, so the order is unique.
            candidates.sort_unstable_by_key(|c| c.0);
            let parents_a = self.parent_pair(a as usize);
            let mut i = 0;
            while i < candidates.len() {
                let (b, parents_b) = candidates[i];
                let mut j = i;
                while j < candidates.len() && candidates[j].0 == b {
                    j += 1;
                }
                let shared = j - i;
                i = j;
                let closer = parents_a
                    .iter()
                    .any(|&x| x != NO_NODE && parents_b.contains(&x))
                    || (1..top - 1)
                        .any(|k| intersects(levels[k].of(a as usize), levels[k].of(b as usize)));
                if closer {
                    continue;
                }
                let kind = if shared >= 2 {
                    CousinKind::Full
                } else {
                    CousinKind::Half
                };
                if (kind == CousinKind::Full && kinds.full)
                    || (kind == CousinKind::Half && kinds.half)
                {
                    emit(a, b, kind);
                }
            }
        }
        Ok(())
    }

    /// The known parents of `x`, sorted, with [`NO_NODE`] for an unknown one.
    fn parent_pair(&self, x: usize) -> [u32; 2] {
        let (m, f) = (
            self.mother[x].unwrap_or(NO_NODE),
            self.father[x].unwrap_or(NO_NODE),
        );
        [m.min(f), m.max(f)]
    }

    /// For generations 1 to `top`, the distinct ancestors of each person exactly
    /// that many generations up, sorted by node.
    fn ancestors_by_generation(&self, top: usize) -> Vec<PerPerson> {
        let parents = |y: usize| [self.mother[y], self.father[y]].into_iter().flatten();
        let mut levels: Vec<PerPerson> = Vec::with_capacity(top);
        let mut scratch: Vec<u32> = Vec::new();
        for k in 0..top {
            let mut level = PerPerson {
                start: Vec::with_capacity(self.persons + 1),
                node: Vec::new(),
            };
            level.start.push(0);
            for x in 0..self.persons {
                scratch.clear();
                match levels.last() {
                    Some(below) if k > 0 => {
                        for &y in below.of(x) {
                            scratch.extend(parents(y as usize));
                        }
                    }
                    _ => scratch.extend(parents(x)),
                }
                scratch.sort_unstable();
                scratch.dedup();
                level.node.extend_from_slice(&scratch);
                level.start.push(level.node.len());
            }
            levels.push(level);
        }
        levels
    }

    /// The coefficient of relationship `r = 2 phi` of each pair of rows, `1 + F`
    /// for a pair of the same person.
    ///
    /// A pair with no known common ancestor has `r = 0` only if nothing unknown
    /// could change it. With `complete_generations` above 0, a zero is reported
    /// as `None` unless both persons have every ancestor slot known through that
    /// many generations ([`Pedigree::complete_generations`]). With 0, a zero is
    /// reported as `Some(0.0)`, the convention of `kinship2`, and every value
    /// is a lower bound when parents are unknown. A positive value is reported
    /// as it is, whatever is unknown.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when a row is not below the number of persons.
    pub fn relatedness(
        &self,
        pairs: &[(u32, u32)],
        complete_generations: u32,
    ) -> Result<Vec<Option<f64>>, RegisterError> {
        if let Some(&(a, b)) = pairs
            .iter()
            .find(|&&(a, b)| a as usize >= self.persons || b as usize >= self.persons)
        {
            return Err(RegisterError::input(format!(
                "pair ({a}, {b}) has a row not below the number of persons, {}",
                self.persons
            )));
        }
        let complete = if complete_generations > 0 {
            self.complete_all()
        } else {
            Vec::new()
        };
        let mut memo: HashMap<(u32, u32), f64> = HashMap::new();
        Ok(pairs
            .iter()
            .map(|&(a, b)| {
                let r = 2.0 * self.phi(a, b, &mut memo);
                let known = |x: u32| u32::from(complete[x as usize]) >= complete_generations;
                if r == 0.0 && complete_generations > 0 && !(known(a) && known(b)) {
                    None
                } else {
                    Some(r)
                }
            })
            .collect())
    }

    fn phi(&self, a: u32, b: u32, memo: &mut HashMap<(u32, u32), f64>) -> f64 {
        // The later generation is expanded, so it is the first of the pair; at
        // the same level neither is an ancestor of the other.
        let (a, b) = if (self.level[a as usize], a) >= (self.level[b as usize], b) {
            (a, b)
        } else {
            (b, a)
        };
        if let Some(&v) = memo.get(&(a, b)) {
            return v;
        }
        let (m, f) = (self.mother[a as usize], self.father[a as usize]);
        let value = if a == b {
            let inbreeding = match (m, f) {
                (Some(m), Some(f)) => self.phi(m, f, memo),
                _ => 0.0,
            };
            f64::midpoint(1.0, inbreeding)
        } else {
            let from = |p: Option<u32>, memo: &mut HashMap<(u32, u32), f64>| {
                p.map_or(0.0, |p| self.phi(p, b, memo))
            };
            let left = from(m, memo);
            let right = from(f, memo);
            f64::midpoint(left, right)
        };
        memo.insert((a, b), value);
        value
    }
}

/// The node of each identifier: the persons are nodes `0..n` in row order and a
/// parent who is not a person takes the next node at its first occurrence, in
/// row order with the mother before the father. Returns the identifier of each
/// node and the mother and father node of each row.
///
/// Identifiers below four times the number of rows, as identifiers numbered
/// from 0 or 1 are, are looked up in a table indexed by the identifier, and
/// larger ones in a hash map.
#[expect(
    clippy::type_complexity,
    reason = "the three columns of the coded pedigree"
)]
fn code_nodes(
    person_id: &[u64],
    mother_id: &[Option<u64>],
    father_id: &[Option<u64>],
) -> Result<(Vec<u64>, Vec<Option<u32>>, Vec<Option<u32>>), RegisterError> {
    let n = person_id.len();
    let dense = (n as u64).saturating_mul(4).saturating_add(64);
    let mut table: Vec<u32> = Vec::new();
    let mut sparse: HashMap<u64, u32, BuildHasherDefault<IdHasher>> = HashMap::default();
    let mut ids = person_id.to_vec();
    for (row, &id) in person_id.iter().enumerate() {
        let fresh = if id < dense {
            let slot = id as usize;
            if table.len() <= slot {
                table.resize(slot + 1, u32::MAX);
            }
            std::mem::replace(&mut table[slot], row as u32) == u32::MAX
        } else {
            sparse.insert(id, row as u32).is_none()
        };
        if !fresh {
            return Err(RegisterError::input(format!("person {id} occurs twice")));
        }
    }
    let mut code = |id: u64, ids: &mut Vec<u64>| -> u32 {
        let next = ids.len() as u32;
        let node = if id < dense {
            let slot = id as usize;
            if table.len() <= slot {
                table.resize(slot + 1, u32::MAX);
            }
            if table[slot] == u32::MAX {
                table[slot] = next;
            }
            table[slot]
        } else {
            *sparse.entry(id).or_insert(next)
        };
        if node == next {
            ids.push(id);
        }
        node
    };
    let mut mother = vec![None; n];
    let mut father = vec![None; n];
    for row in 0..n {
        if let (Some(m), Some(f)) = (mother_id[row], father_id[row])
            && m == f
        {
            return Err(RegisterError::input(format!(
                "person {} has the same mother and father, {m}",
                person_id[row]
            )));
        }
        mother[row] = mother_id[row].map(|id| code(id, &mut ids));
        father[row] = father_id[row].map(|id| code(id, &mut ids));
    }
    Ok((ids, mother, father))
}

/// A hash of a `u64` identifier by the finaliser of SplitMix64, which spreads
/// consecutive identifiers over all bits; the keys are not chosen by an
/// adversary, so a keyed hash is not needed.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

/// The members of `group` after `a`; a group is ascending.
fn later(group: &[u32], a: u32) -> &[u32] {
    &group[group.partition_point(|&b| b <= a)..]
}

fn intersects(x: &[u32], y: &[u32]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < x.len() && j < y.len() {
        match x[i].cmp(&y[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// No node, for an unknown parent.
const NO_NODE: u32 = u32::MAX;

/// Nodes listed for each person: those of person `x` are
/// `node[start[x]..start[x + 1]]`.
struct PerPerson {
    start: Vec<usize>,
    node: Vec<u32>,
}

impl PerPerson {
    fn of(&self, x: usize) -> &[u32] {
        &self.node[self.start[x]..self.start[x + 1]]
    }
}

/// Rows grouped by the node in one parent column, ascending within a group.
struct Groups {
    start: Vec<usize>,
    rows: Vec<u32>,
}

impl Groups {
    fn new(column: &[Option<u32>], nodes: usize) -> Self {
        let mut start = vec![0_usize; nodes + 1];
        for node in column.iter().flatten() {
            start[*node as usize + 1] += 1;
        }
        for i in 0..nodes {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut rows = vec![0_u32; start[nodes]];
        for (row, node) in column.iter().enumerate() {
            if let Some(node) = node {
                rows[fill[*node as usize]] = row as u32;
                fill[*node as usize] += 1;
            }
        }
        Self { start, rows }
    }

    fn members(&self, node: usize) -> &[u32] {
        &self.rows[self.start[node]..self.start[node + 1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coding of the nodes by one hash map, the form `code_nodes` replaces.
    #[expect(
        clippy::type_complexity,
        reason = "the three columns of the coded pedigree"
    )]
    fn reference_nodes(
        person_id: &[u64],
        mother_id: &[Option<u64>],
        father_id: &[Option<u64>],
    ) -> Option<(Vec<u64>, Vec<Option<u32>>, Vec<Option<u32>>)> {
        let n = person_id.len();
        let mut node_of: HashMap<u64, u32> = HashMap::with_capacity(n * 2);
        let mut ids = person_id.to_vec();
        for (row, &id) in person_id.iter().enumerate() {
            if node_of.insert(id, row as u32).is_some() {
                return None;
            }
        }
        let mut code = |id: u64, ids: &mut Vec<u64>| -> u32 {
            *node_of.entry(id).or_insert_with(|| {
                ids.push(id);
                (ids.len() - 1) as u32
            })
        };
        let mut mother = vec![None; n];
        let mut father = vec![None; n];
        for row in 0..n {
            mother[row] = mother_id[row].map(|id| code(id, &mut ids));
            father[row] = father_id[row].map(|id| code(id, &mut ids));
        }
        Some((ids, mother, father))
    }

    /// The family clusters with the parents renumbered by a hash map.
    fn reference_family_clusters(p: &Pedigree) -> Vec<u32> {
        let dense = |parent: &[Option<u32>]| -> Vec<Option<u32>> {
            let mut map: HashMap<u32, u32> = HashMap::new();
            parent[..p.persons]
                .iter()
                .map(|q| {
                    q.map(|node| {
                        let next = map.len() as u32;
                        *map.entry(node).or_insert(next)
                    })
                })
                .collect()
        };
        let (m, f) = (dense(&p.mother), dense(&p.father));
        connected_components(&[&m, &f]).unwrap()
    }

    /// The cousin pairs from ancestors in a vector per person and groups by a
    /// sort of (ancestor, person), the forms the flat lists replace.
    fn reference_cousin_pairs(p: &Pedigree, degree: u32, kinds: CousinKinds) -> CousinPairs {
        let top = degree as usize + 1;
        let mut levels: Vec<Vec<Vec<u32>>> = Vec::with_capacity(top);
        let first: Vec<Vec<u32>> = (0..p.persons)
            .map(|x| {
                let mut v: Vec<u32> = [p.mother[x], p.father[x]].into_iter().flatten().collect();
                v.sort_unstable();
                v
            })
            .collect();
        levels.push(first);
        for k in 1..top {
            let next: Vec<Vec<u32>> = (0..p.persons)
                .map(|x| {
                    let mut v: Vec<u32> = levels[k - 1][x]
                        .iter()
                        .flat_map(|&y| {
                            [p.mother[y as usize], p.father[y as usize]]
                                .into_iter()
                                .flatten()
                        })
                        .collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                })
                .collect();
            levels.push(next);
        }
        let nodes = p.ids.len();
        let mut flat: Vec<(u32, u32)> = Vec::new();
        for (x, ancestors) in levels[top - 1].iter().enumerate() {
            for &anc in ancestors {
                flat.push((anc, x as u32));
            }
        }
        flat.sort_unstable();
        let mut group_start = vec![0_usize; nodes + 1];
        for &(anc, _) in &flat {
            group_start[anc as usize + 1] += 1;
        }
        for i in 0..nodes {
            group_start[i + 1] += group_start[i];
        }
        let members: Vec<u32> = flat.iter().map(|&(_, x)| x).collect();
        let mut out = CousinPairs::default();
        let mut candidates: Vec<u32> = Vec::new();
        for a in 0..p.persons as u32 {
            candidates.clear();
            for &anc in &levels[top - 1][a as usize] {
                let group = &members[group_start[anc as usize]..group_start[anc as usize + 1]];
                candidates.extend_from_slice(later(group, a));
            }
            candidates.sort_unstable();
            let mut i = 0;
            while i < candidates.len() {
                let b = candidates[i];
                let mut j = i;
                while j < candidates.len() && candidates[j] == b {
                    j += 1;
                }
                let shared = j - i;
                i = j;
                let closer = (0..top - 1)
                    .any(|k| intersects(&levels[k][a as usize], &levels[k][b as usize]));
                if closer {
                    continue;
                }
                let kind = if shared >= 2 {
                    CousinKind::Full
                } else {
                    CousinKind::Half
                };
                if (kind == CousinKind::Full && kinds.full)
                    || (kind == CousinKind::Half && kinds.half)
                {
                    out.a.push(a);
                    out.b.push(b);
                    out.kind.push(kind);
                }
            }
        }
        out
    }

    /// Generations of persons whose parents are drawn from a few couples of the
    /// generation before, some unknown and some outside the rows, with the
    /// identifiers `id(row)`.
    fn random_pedigree(
        n: usize,
        seed: u64,
        id: impl Fn(u64) -> u64,
    ) -> (Vec<u64>, Vec<Option<u64>>, Vec<Option<u64>>) {
        let mut state = seed;
        let mut below = |k: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % k
        };
        let per = n / 4;
        let (mut ids, mut mother, mut father) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..n as u64 {
            ids.push(id(row));
            let g = row / per as u64;
            if g == 0 {
                mother.push(None);
                father.push(None);
                continue;
            }
            let base = (g - 1) * per as u64;
            let couple = below(per as u64 / 3);
            let m = base + (couple * 7) % per as u64;
            let f = base + (couple * 13 + 1) % per as u64;
            let pick = |p: u64, r: u64| match r {
                0 => None,
                1 => Some(id(n as u64 + p)),
                _ => Some(id(p)),
            };
            let (rm, rf) = (below(12), below(12));
            mother.push(pick(m, rm));
            father.push(if m == f { None } else { pick(f, rf) });
        }
        (ids, mother, father)
    }

    #[test]
    fn the_flat_forms_equal_the_references() {
        type Identifier = fn(u64) -> u64;
        let forms: [(&str, Identifier); 3] = [
            ("small", |r| r + 1),
            ("large", |r| r.wrapping_mul(1_000_003) + 1_000_000_000_000),
            (
                "mixed",
                |r| if r % 3 == 0 { r } else { r * 977 + (1 << 40) },
            ),
        ];
        for (name, id) in forms {
            for seed in 1..4 {
                let (ids, mother, father) = random_pedigree(4000, seed, id);
                let p = Pedigree::new(&ids, &mother, &father).unwrap();
                let (rid, rm, rf) = reference_nodes(&ids, &mother, &father).unwrap();
                assert_eq!(p.ids, rid, "{name}");
                assert_eq!(&p.mother[..p.persons], &rm[..], "{name}");
                assert_eq!(&p.father[..p.persons], &rf[..], "{name}");
                assert_eq!(p.family_clusters(FamilyLinks::SharedParents).unwrap(), reference_family_clusters(&p));
                for degree in 1..4 {
                    let got = p.cousin_pairs(degree, CousinKinds::ALL, 100_000).unwrap();
                    assert!(degree > 1 || !got.a.is_empty(), "{name}: no cousins");
                    assert_eq!(got, reference_cousin_pairs(&p, degree, CousinKinds::ALL));
                }
            }
        }
        // A person twice is refused on both routes.
        assert!(Pedigree::new(&[5, 1 << 50, 5], &[None; 3], &[None; 3]).is_err());
        assert!(Pedigree::new(&[1 << 50, 5, 1 << 50], &[None; 3], &[None; 3]).is_err());
    }

    // Persons: 1 and 2 are the founders. 3, 4 and 5 are their children, 6 is
    // the child of 3 and an outside father 90, 7 is the child of 4 and 91.
    // 8 is a child of 6 and 7, so a first cousin pair has a child.
    // 9 is the child of mother 1 and father 92 (a half sibling of 3, 4, 5).
    fn family() -> Pedigree {
        let id = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mother = [
            None,
            None,
            Some(1),
            Some(1),
            Some(1),
            Some(3),
            Some(4),
            Some(6),
            Some(1),
        ];
        let father = [
            None,
            None,
            Some(2),
            Some(2),
            Some(2),
            Some(90),
            Some(91),
            Some(7),
            Some(92),
        ];
        Pedigree::new(&id, &mother, &father).unwrap()
    }

    fn r(p: &Pedigree, a: u32, b: u32) -> f64 {
        p.relatedness(&[(a, b)], 0).unwrap()[0].unwrap()
    }

    #[test]
    fn expected_coefficients() {
        let p = family();
        // Rows: 0 -> id 1, 2 -> id 3, 3 -> id 4, 5 -> id 6, 6 -> id 7.
        assert_eq!(r(&p, 2, 3).to_bits(), 0.5_f64.to_bits()); // full siblings
        assert_eq!(r(&p, 2, 8).to_bits(), 0.25_f64.to_bits()); // half siblings
        assert_eq!(r(&p, 0, 2).to_bits(), 0.5_f64.to_bits()); // parent and child
        assert_eq!(r(&p, 5, 6).to_bits(), 0.125_f64.to_bits()); // first cousins
        assert_eq!(r(&p, 3, 5).to_bits(), 0.25_f64.to_bits()); // uncle and nephew
        assert_eq!(r(&p, 5, 5).to_bits(), 1.0_f64.to_bits());
        assert_eq!(r(&p, 0, 1).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn inbreeding_enters_the_diagonal() {
        let p = family();
        // Person 8 has parents 6 and 7, first cousins, so F = 1/16.
        assert_eq!(r(&p, 7, 7).to_bits(), 1.0625_f64.to_bits());
    }

    #[test]
    fn unknown_ancestry_is_not_reported_as_zero_when_asked() {
        let p = family();
        // Rows 0 and 1 are founders with no known parents: zero is a lower bound.
        assert_eq!(p.relatedness(&[(0, 1)], 0).unwrap(), [Some(0.0)]);
        assert_eq!(p.relatedness(&[(0, 1)], 1).unwrap(), [None]);
        // Persons 3 and 4 have both parents, so one generation is complete.
        assert_eq!(p.relatedness(&[(2, 3)], 1).unwrap(), [Some(0.5)]);
        assert_eq!(p.complete_generations(), [0, 0, 1, 1, 1, 1, 1, 2, 1]);
        assert_eq!(p.depth(), [0, 0, 1, 1, 1, 2, 2, 3, 1]);
    }

    #[test]
    fn mother_only_is_never_promoted() {
        let p = Pedigree::new(
            &[1, 2, 3],
            &[Some(10), Some(10), Some(10)],
            &[None, Some(20), Some(20)],
        )
        .unwrap();
        let s = p.sibling_pairs(SiblingKinds::ALL, 10).unwrap();
        assert_eq!((s.a, s.b), (vec![0, 0, 1], vec![1, 2, 2]));
        assert_eq!(
            s.kind,
            [
                SiblingKind::MotherOnly,
                SiblingKind::MotherOnly,
                SiblingKind::Full
            ]
        );
        let only_full = SiblingKinds {
            full: true,
            ..SiblingKinds {
                full: false,
                maternal_half: false,
                paternal_half: false,
                mother_only: false,
                father_only: false,
            }
        };
        assert_eq!(p.sibling_pairs(only_full, 10).unwrap().a, [1]);
    }

    #[test]
    fn a_large_group_is_refused_with_the_parent() {
        let ids: Vec<u64> = (0..30).collect();
        let mothers = vec![Some(500); 30];
        let p = Pedigree::new(&ids, &mothers, &vec![None; 30]).unwrap();
        let e = p
            .sibling_pairs(SiblingKinds::ALL, 25)
            .unwrap_err()
            .to_string();
        assert!(e.contains("500") && e.contains("30"), "{e}");
        assert_eq!(p.large_groups(25).len(), 1);
        assert!(p.sibling_pairs(SiblingKinds::ALL, 30).is_ok());
    }

    #[test]
    fn bad_pedigrees_are_refused() {
        assert!(Pedigree::new(&[1, 1], &[None, None], &[None, None]).is_err());
        assert!(Pedigree::new(&[1], &[Some(5)], &[Some(5)]).is_err());
        let cycle = Pedigree::new(&[1, 2], &[Some(2), Some(1)], &[None, None]).unwrap_err();
        assert!(cycle.to_string().contains("own ancestor"), "{cycle}");
        let own = Pedigree::new(&[7], &[Some(7)], &[None]).unwrap_err();
        assert!(own.to_string().contains('7'));
        assert!(Pedigree::new(&[1, 2], &[None], &[None, None]).is_err());
        let p = family();
        assert!(p.relatedness(&[(0, 99)], 0).is_err());
        assert!(p.cousin_pairs(0, CousinKinds::ALL, 10).is_err());
    }

    #[test]
    fn family_clusters_chain_half_siblings() {
        let p = family();
        // 3, 4, 5 and 9 share mother 1; 6, 7, 8 are linked through 7 and 6:
        // 8's mother is 6, father 7 (rows), different codes from 6's and 7's
        // own parents, so only 8 is alone with them unrelated.
        let c = p.family_clusters(FamilyLinks::SharedParents).unwrap();
        assert_eq!(c[2], c[3]);
        assert_eq!(c[2], c[8]);
        assert_ne!(c[2], c[5]);
        assert_eq!(c[0], 0);
    }

    #[test]
    fn whole_pedigree_clusters_join_parents_and_children() {
        // The family of the guide: mother 10 and her brother 11 are children
        // of 1 and 2; 20 to 24 are her children, 25 is his; 30 is a father
        // who is a row.
        let person = [10, 11, 20, 21, 22, 23, 24, 25, 30];
        let mother = [
            Some(1),
            Some(1),
            Some(10),
            Some(10),
            Some(10),
            Some(10),
            Some(10),
            Some(40),
            None,
        ];
        let father = [
            Some(2),
            Some(2),
            Some(30),
            Some(30),
            Some(31),
            Some(31),
            None,
            Some(11),
            None,
        ];
        let p = Pedigree::new(&person, &mother, &father).unwrap();
        assert_eq!(
            p.family_clusters(FamilyLinks::SharedParents).unwrap(),
            [0, 0, 1, 1, 1, 1, 1, 2, 3]
        );
        assert_eq!(
            p.family_clusters(FamilyLinks::ParentsAndChildren).unwrap(),
            [0; 9]
        );
        // Without the link from 25 to 11 and from the children to 30, the
        // brother's child and the father are clusters of their own.
        let p = Pedigree::new(
            &person,
            &mother,
            &[Some(2), Some(2), None, None, Some(31), Some(31), None, None, None],
        )
        .unwrap();
        assert_eq!(
            p.family_clusters(FamilyLinks::ParentsAndChildren).unwrap(),
            [0, 0, 0, 0, 0, 0, 0, 1, 2]
        );
    }

    #[test]
    fn deliveries_within_a_tolerance() {
        // Mother 50: twins either side of midnight, then a birth a year later.
        // Mother 60: births on days 0, 1 and 2, where the window counts from
        // the first birth, and day 2 is beyond a tolerance of 1.
        let mother = [Some(50), Some(50), Some(50), Some(60), Some(60), Some(60), None];
        let day = [
            Some(100),
            Some(101),
            Some(466),
            Some(0),
            Some(1),
            Some(2),
            Some(5),
        ];
        assert_eq!(
            delivery_order(&mother, &day, 0).unwrap(),
            [Some(0), Some(1), Some(2), Some(0), Some(1), Some(2), None]
        );
        assert_eq!(
            delivery_order(&mother, &day, 1).unwrap(),
            [Some(0), Some(0), Some(1), Some(0), Some(0), Some(1), None]
        );
        assert_eq!(
            delivery_order(&mother, &day, 2).unwrap(),
            [Some(0), Some(0), Some(1), Some(0), Some(0), Some(0), None]
        );
        let b = multiple_birth_groups(&mother, &day, 1).unwrap();
        assert_eq!(b.group, [Some(0), Some(0), Some(1), Some(2), Some(2), Some(3), None]);
        assert_eq!(b.size, [Some(2), Some(2), Some(1), Some(2), Some(2), Some(1), None]);
        // Groups are numbered by first row, also when rows are not in day order.
        let b = multiple_birth_groups(&[Some(1), Some(1), Some(1)], &[Some(9), Some(0), Some(8)], 1)
            .unwrap();
        assert_eq!(b.group, [Some(0), Some(1), Some(0)]);
    }

    #[test]
    fn ancestors_by_generation() {
        let p = family();
        let a = p.ancestors(2);
        let of = |row: usize| -> Vec<(u64, u8)> {
            (a.offsets[row]..a.offsets[row + 1])
                .map(|i| (p.node_id(a.node[i]), a.generation[i]))
                .collect()
        };
        assert_eq!(of(0), []);
        assert_eq!(of(5), [(3, 1), (90, 1), (1, 2), (2, 2)]);
    }

    #[test]
    fn consensus_rules() {
        let person = [1, 1, 1, 2];
        let mother = [Some(9), Some(5), Some(5), None];
        let father = [None, Some(7), Some(8), None];
        let by = |rule| consensus_parents(&person, &mother, &father, rule).unwrap();
        let s = by(ParentRule::SmallestId);
        assert_eq!(s.person, [1, 2]);
        assert_eq!(
            (s.mother, s.father),
            (vec![Some(5), None], vec![Some(7), None])
        );
        let f = by(ParentRule::FirstObserved);
        assert_eq!(
            (f.mother, f.father),
            (vec![Some(9), None], vec![Some(7), None])
        );
        let m = by(ParentRule::MostFrequent);
        assert_eq!(
            (m.mother, m.father),
            (vec![Some(5), None], vec![Some(7), None])
        );
        assert!(consensus_parents(&[1], &[None, None], &[None], ParentRule::SmallestId).is_err());
    }

    #[test]
    fn births_and_order() {
        let mother = [Some(1), Some(1), Some(1), Some(2), None, Some(1)];
        let day = [Some(10), Some(10), Some(30), Some(10), Some(10), None];
        let b = multiple_birth_groups(&mother, &day, 0).unwrap();
        assert_eq!(b.group, [Some(0), Some(0), Some(1), Some(2), None, None]);
        assert_eq!(b.size, [Some(2), Some(2), Some(1), Some(1), None, None]);
        assert_eq!(
            delivery_order(&mother, &day, 0).unwrap(),
            [Some(0), Some(0), Some(1), Some(0), None, None]
        );
    }
}
