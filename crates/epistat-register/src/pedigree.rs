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

use crate::{RegisterError, relations::connected_components};

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
    /// A number for each group of rows with the same mother and birth day,
    /// from 0 in order of first row, or `None` when the mother or day is
    /// missing.
    pub group: Vec<Option<u32>>,
    /// The size of the group: 1 for a single birth, 2 for twins.
    pub size: Vec<Option<u32>>,
}

/// Groups rows by mother and birth day. A group of two or more is a multiple
/// birth.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length or there are more
/// than `u32::MAX` rows.
pub fn multiple_birth_groups(
    mother: &[Option<u64>],
    birth_day: &[Option<i64>],
) -> Result<MultipleBirths, RegisterError> {
    let n = mother.len();
    check_rows(n, birth_day.len())?;
    let mut rows: Vec<u32> = (0..n as u32)
        .filter(|&r| mother[r as usize].is_some() && birth_day[r as usize].is_some())
        .collect();
    rows.sort_by_key(|&r| (mother[r as usize], birth_day[r as usize], r));
    let mut first = vec![u32::MAX; n];
    let mut count = vec![0_u32; n];
    let mut i = 0;
    while i < rows.len() {
        let key = (mother[rows[i] as usize], birth_day[rows[i] as usize]);
        let mut j = i;
        while j < rows.len() && (mother[rows[j] as usize], birth_day[rows[j] as usize]) == key {
            j += 1;
        }
        let head = rows[i];
        for &r in &rows[i..j] {
            first[r as usize] = head;
        }
        count[head as usize] = (j - i) as u32;
        i = j;
    }
    let mut out = MultipleBirths {
        group: vec![None; n],
        size: vec![None; n],
    };
    let mut ids = vec![0_u32; n];
    let mut next = 0_u32;
    for r in 0..n {
        if first[r] == r as u32 {
            ids[r] = next;
            next += 1;
        }
        if first[r] != u32::MAX {
            let head = first[r] as usize;
            out.group[r] = Some(ids[head]);
            out.size[r] = Some(count[head]);
        }
    }
    Ok(out)
}

/// The birth order of each row within its mother: 0 for the first delivery.
/// Rows with the same mother and birth day share an order, so twins are one
/// delivery. `None` when the mother or the day is missing.
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
) -> Result<Vec<Option<u32>>, RegisterError> {
    let n = mother.len();
    check_rows(n, birth_day.len())?;
    let mut rows: Vec<u32> = (0..n as u32)
        .filter(|&r| mother[r as usize].is_some() && birth_day[r as usize].is_some())
        .collect();
    rows.sort_by_key(|&r| (mother[r as usize], birth_day[r as usize], r));
    let mut out = vec![None; n];
    let mut rank = 0_u32;
    for (k, &r) in rows.iter().enumerate() {
        if k > 0 {
            let p = rows[k - 1] as usize;
            if mother[p] != mother[r as usize] {
                rank = 0;
            } else if birth_day[p] != birth_day[r as usize] {
                rank += 1;
            }
        }
        out[r as usize] = Some(rank);
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
        let mut node_of: HashMap<u64, u32> = HashMap::with_capacity(n * 2);
        let mut ids = person_id.to_vec();
        for (row, &id) in person_id.iter().enumerate() {
            if node_of.insert(id, row as u32).is_some() {
                return Err(RegisterError::input(format!("person {id} occurs twice")));
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

    /// The family cluster of each person: persons who share a mother or a
    /// father, directly or through a chain of persons who do, have the same
    /// number. Numbered from 0 in order of first row. Half siblings joined
    /// through a third are one cluster. Persons with no known parent are their
    /// own cluster.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] if the pedigree is too large for
    /// [`connected_components`], which cannot happen for one built by
    /// [`Pedigree::new`].
    pub fn family_clusters(&self) -> Result<Vec<u32>, RegisterError> {
        // The same code in the mother and the father column is two parents, and
        // a column holds at most one distinct parent per person, so each
        // column is renumbered from 0 and stays below the number of rows.
        let dense = |parent: &[Option<u32>]| -> Vec<Option<u32>> {
            let mut map: HashMap<u32, u32> = HashMap::new();
            parent[..self.persons]
                .iter()
                .map(|p| {
                    p.map(|node| {
                        let next = map.len() as u32;
                        *map.entry(node).or_insert(next)
                    })
                })
                .collect()
        };
        let (m, f) = (dense(&self.mother), dense(&self.father));
        connected_components(&[&m, &f])
    }

    /// The parents with more than `threshold` children among the rows, mothers
    /// first and then fathers, each in order of identifier.
    #[must_use]
    pub fn large_groups(&self, threshold: usize) -> Vec<LargeGroup> {
        let mut out = Vec::new();
        for (role, column) in [
            (ParentRole::Mother, &self.mother),
            (ParentRole::Father, &self.father),
        ] {
            let groups = Groups::new(&column[..self.persons], self.ids.len());
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
        if let Some(g) = self.large_groups(max_group_size).first() {
            return Err(RegisterError::input(format!(
                "{:?} {} has {} children among the rows, more than the limit {max_group_size}",
                g.role, g.parent, g.size
            )));
        }
        let nodes = self.ids.len();
        let by_mother = Groups::new(&self.mother[..self.persons], nodes);
        let by_father = Groups::new(&self.father[..self.persons], nodes);
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
        // levels[k - 1][x] holds the distinct ancestors of x exactly k
        // generations up, sorted.
        let levels = self.ancestors_by_generation(top);
        let nodes = self.ids.len();
        let mut flat: Vec<(u32, u32)> = Vec::new();
        for (x, ancestors) in levels[top - 1].iter().enumerate() {
            for &anc in ancestors {
                flat.push((anc, x as u32));
            }
        }
        flat.sort_unstable();
        // Members of each ancestor's group, ascending by row.
        let mut group_start = vec![0_usize; nodes + 1];
        for &(anc, _) in &flat {
            group_start[anc as usize + 1] += 1;
        }
        for i in 0..nodes {
            group_start[i + 1] += group_start[i];
        }
        let members: Vec<u32> = flat.iter().map(|&(_, x)| x).collect();
        if let Some(anc) =
            (0..nodes).find(|&a| group_start[a + 1] - group_start[a] > max_group_size)
        {
            return Err(RegisterError::input(format!(
                "ancestor {} has {} descendants {top} generations down among the rows, more than the limit {max_group_size}",
                self.ids[anc],
                group_start[anc + 1] - group_start[anc]
            )));
        }
        let mut candidates: Vec<u32> = Vec::new();
        for a in 0..self.persons as u32 {
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
                    emit(a, b, kind);
                }
            }
        }
        Ok(())
    }

    /// For generations 1 to `top`, the distinct ancestors of each person exactly
    /// that many generations up, sorted by node.
    fn ancestors_by_generation(&self, top: usize) -> Vec<Vec<Vec<u32>>> {
        let mut levels: Vec<Vec<Vec<u32>>> = Vec::with_capacity(top);
        let first: Vec<Vec<u32>> = (0..self.persons)
            .map(|x| {
                let mut v: Vec<u32> = [self.mother[x], self.father[x]]
                    .into_iter()
                    .flatten()
                    .collect();
                v.sort_unstable();
                v
            })
            .collect();
        levels.push(first);
        for k in 1..top {
            let next: Vec<Vec<u32>> = (0..self.persons)
                .map(|x| {
                    let mut v: Vec<u32> = levels[k - 1][x]
                        .iter()
                        .flat_map(|&y| {
                            [self.mother[y as usize], self.father[y as usize]]
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
        let c = p.family_clusters().unwrap();
        assert_eq!(c[2], c[3]);
        assert_eq!(c[2], c[8]);
        assert_ne!(c[2], c[5]);
        assert_eq!(c[0], 0);
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
        let b = multiple_birth_groups(&mother, &day).unwrap();
        assert_eq!(b.group, [Some(0), Some(0), Some(1), Some(2), None, None]);
        assert_eq!(b.size, [Some(2), Some(2), Some(1), Some(1), None, None]);
        assert_eq!(
            delivery_order(&mother, &day).unwrap(),
            [Some(0), Some(0), Some(1), Some(0), None, None]
        );
    }
}
