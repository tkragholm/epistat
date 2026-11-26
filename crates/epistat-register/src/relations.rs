//! Connected components of rows that share a key.
//!
//! Each row carries one code in each of several key columns, or none. Two rows
//! with the same code in the same column are joined, and the joins are
//! transitive, so a component is a connected component of that graph. A code in
//! one column is unrelated to the same code in another, so a caller keeps two
//! kinds of link apart by putting them in two columns. For siblings, the rows
//! are people and the columns their mother and their father, and two people who
//! share either parent end up in one group.
//!
//! # Algorithm
//!
//! A disjoint-set forest over the row indices \[1\] with path halving \[2\].
//! Each column is scanned once. The first row seen with a code is remembered
//! and every later row with the same code is joined to it. A union points the
//! larger root at the smaller, and path halving only moves a row's parent to an
//! ancestor, so `parent[row] <= row` throughout, and the root of a set is its
//! first row. The work is near-linear in the number of rows and columns.
//!
//! Components are then numbered from 0 in the order of their first row: a row
//! that is a root takes the next number, and any other row takes the number of
//! its root, which is already numbered. The numbering depends only on the row
//! order the caller fixes, not on the order of the columns or on how the unions
//! were made.
//!
//! # Edge cases
//!
//! - A row with no code in a column joins nothing through that column.
//! - A code must be below the number of rows, which is what makes the first-seen
//!   table a dense array.
//! - No rows give no components. No key columns, columns of different lengths
//!   and more than `u32::MAX` rows are refused.
//!
//! # References
//!
//! 1. Tarjan RE. Efficiency of a good but not linear set union algorithm.
//!    J ACM. 1975;22(2):215-25. doi:10.1145/321879.321884
//! 2. Tarjan RE, van Leeuwen J. Worst-case analysis of set union algorithms.
//!    J ACM. 1984;31(2):245-81. doi:10.1145/62.2160

#![expect(
    clippy::cast_possible_truncation,
    reason = "row numbers are checked against the u32 limit before they are narrowed"
)]

use crate::RegisterError;

/// Disjoint-set forest over row indices whose root is the smallest index in
/// the set.
///
/// A union points the larger root at the smaller, and path halving only moves
/// a row's parent to an ancestor, so `parent[row] <= row` throughout and each
/// root is its set's first row.
pub(crate) struct DisjointSets {
    parent: Vec<u32>,
}

impl DisjointSets {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }

    pub(crate) fn find(&mut self, mut row: u32) -> u32 {
        while self.parent[row as usize] != row {
            let grandparent = self.parent[self.parent[row as usize] as usize];
            self.parent[row as usize] = grandparent;
            row = grandparent;
        }
        row
    }

    pub(crate) fn union(&mut self, left: u32, right: u32) {
        let left = self.find(left);
        let right = self.find(right);
        if left != right {
            let (low, high) = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            self.parent[high as usize] = low;
        }
    }
}

/// The component of each row, joining rows that share a code in a key column.
///
/// `keys` holds one or more columns, each with one entry per row. An entry is a
/// code in `0..n`, where `n` is the number of rows, or `None` for no key.
/// Returns one component number per row. Components are numbered from 0 in
/// order of their first row, so a row whose component has not appeared at an
/// earlier row starts the next number.
///
/// # Errors
///
/// [`RegisterError::Input`] when there are no key columns, when the columns
/// differ in length, when there are more than `u32::MAX` rows, or when a code
/// is not below the number of rows.
pub fn connected_components(keys: &[&[Option<u32>]]) -> Result<Vec<u32>, RegisterError> {
    let Some(first_column) = keys.first() else {
        return Err(RegisterError::input(
            "connected_components needs a key column",
        ));
    };
    let n = first_column.len();
    if keys.iter().any(|column| column.len() != n) {
        return Err(RegisterError::input(
            "every key column must have one entry per row",
        ));
    }
    if u32::try_from(n).is_err() {
        return Err(RegisterError::input(format!(
            "{n} rows is more than connected_components takes"
        )));
    }

    let mut sets = DisjointSets::new(n);
    // The first row seen with each code, reused across columns.
    let mut first_row = vec![u32::MAX; n];
    for column in keys {
        first_row.fill(u32::MAX);
        for (row, code) in column.iter().enumerate() {
            let Some(code) = *code else { continue };
            let Some(slot) = first_row.get_mut(code as usize) else {
                return Err(RegisterError::input(format!(
                    "key code {code} is not below the number of rows, {n}"
                )));
            };
            if *slot == u32::MAX {
                *slot = row as u32;
            } else {
                sets.union(*slot, row as u32);
            }
        }
    }

    let mut component = vec![0_u32; n];
    let mut next = 0_u32;
    for row in 0..n as u32 {
        let root = sets.find(row);
        component[row as usize] = if root == row {
            next += 1;
            next - 1
        } else {
            // The root is the set's first row, so it is numbered already.
            component[root as usize]
        };
    }
    Ok(component)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn components(keys: &[&[Option<u32>]]) -> Vec<u32> {
        connected_components(keys).expect("valid keys")
    }

    #[test]
    fn rows_sharing_a_code_are_joined_and_numbered_by_first_row() {
        let mother = [Some(1), Some(0), Some(1), None];
        assert_eq!(components(&[&mother]), [0, 1, 0, 2]);
    }

    #[test]
    fn links_in_two_columns_chain_transitively() {
        // Rows 0 and 1 share a mother, rows 1 and 2 a father, row 3 neither.
        let mother = [Some(0), Some(0), Some(1), Some(2)];
        let father = [Some(0), Some(1), Some(1), Some(3)];
        assert_eq!(components(&[&mother, &father]), [0, 0, 0, 1]);
    }

    #[test]
    fn a_code_in_one_column_does_not_join_the_same_code_in_another() {
        let mother = [Some(0), None];
        let father = [None, Some(0)];
        assert_eq!(components(&[&mother, &father]), [0, 1]);
    }

    #[test]
    fn a_late_union_renumbers_nothing_already_numbered() {
        // Row 3 joins rows 0 and 2 through two different codes; row 1 stays alone.
        let left = [Some(0), Some(1), Some(2), Some(0)];
        let right = [None, None, Some(3), Some(3)];
        assert_eq!(components(&[&left, &right]), [0, 1, 0, 0]);
    }

    #[test]
    fn no_rows_gives_no_components() {
        assert_eq!(components(&[&[]]), Vec::<u32>::new());
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(connected_components(&[]).is_err());
        assert!(connected_components(&[&[Some(0)], &[Some(0), None]]).is_err());
        assert!(connected_components(&[&[Some(1)]]).is_err());
    }

    /// The forest against a quadratic label propagation, on random chains.
    #[test]
    fn matches_label_propagation() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut draw = |bound: u32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(bound)) as u32
        };
        for _ in 0..200 {
            let n = 1 + draw(60) as usize;
            let spread = 1 + draw(n as u32);
            let columns: Vec<Vec<Option<u32>>> = (0..2)
                .map(|_| {
                    (0..n)
                        .map(|_| (draw(4) != 0).then(|| draw(spread)))
                        .collect()
                })
                .collect();
            let keys: Vec<&[Option<u32>]> = columns.iter().map(Vec::as_slice).collect();

            let mut label: Vec<usize> = (0..n).collect();
            loop {
                let mut changed = false;
                for column in &columns {
                    for a in 0..n {
                        for b in 0..n {
                            if column[a].is_some() && column[a] == column[b] && label[b] < label[a]
                            {
                                label[a] = label[b];
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            let mut numbered = std::collections::HashMap::new();
            let expected: Vec<u32> = label
                .iter()
                .map(|root| {
                    let next = numbered.len() as u32;
                    *numbered.entry(*root).or_insert(next)
                })
                .collect();
            assert_eq!(components(&keys), expected);
        }
    }
}
