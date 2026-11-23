//! The rows of each cluster, with the clusters in increasing order of their
//! labels, which is the order of R's factor levels for integer labels.

/// The rows of each cluster.
#[derive(Clone, Debug)]
pub(crate) struct Groups {
    /// The distinct labels, increasing.
    pub labels: Vec<u32>,
    /// `rows[starts[g]..starts[g + 1]]` are the rows of cluster `g`.
    pub starts: Vec<usize>,
    /// Row numbers, cluster by cluster, increasing within a cluster.
    pub rows: Vec<usize>,
    /// Whether `rows` is `0..n`, so that data in row order are already in
    /// cluster order.
    pub sorted: bool,
}

impl Groups {
    /// Groups the rows by `cluster`.
    pub fn new(cluster: &[u32]) -> Self {
        let mut labels = cluster.to_vec();
        labels.sort_unstable();
        labels.dedup();
        let index: Vec<usize> = cluster
            .iter()
            .map(|c| labels.binary_search(c).unwrap_or_else(|at| at))
            .collect();
        let g = labels.len();
        let mut starts = vec![0_usize; g + 1];
        for &i in &index {
            starts[i + 1] += 1;
        }
        for i in 0..g {
            starts[i + 1] += starts[i];
        }
        let mut next = starts.clone();
        let mut rows = vec![0_usize; cluster.len()];
        for (row, &i) in index.iter().enumerate() {
            rows[next[i]] = row;
            next[i] += 1;
        }
        let sorted = rows.iter().enumerate().all(|(k, &r)| k == r);
        Self {
            labels,
            starts,
            rows,
            sorted,
        }
    }

    /// The number of clusters.
    pub fn count(&self) -> usize {
        self.labels.len()
    }

    /// `values` in cluster order.
    pub fn gather(&self, values: &[f64]) -> Vec<f64> {
        if self.sorted {
            return values.to_vec();
        }
        self.rows.iter().map(|&r| values[r]).collect()
    }

    /// `out[rows[k]] = values[k]`, the inverse of [`Groups::gather`].
    pub fn scatter(&self, values: &[f64], out: &mut [f64]) {
        if self.sorted {
            out.copy_from_slice(values);
            return;
        }
        for (&r, &v) in self.rows.iter().zip(values) {
            out[r] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_grouped_by_increasing_label() {
        let groups = Groups::new(&[7, 3, 7, 5, 3]);
        assert_eq!(groups.labels, [3, 5, 7]);
        assert_eq!(groups.starts, [0, 2, 3, 5]);
        assert_eq!(groups.rows, [1, 4, 3, 0, 2]);
        assert!(!groups.sorted);
        let values = [0.0, 1.0, 2.0, 3.0, 4.0];
        let gathered = groups.gather(&values);
        assert_eq!(gathered, [1.0, 4.0, 3.0, 0.0, 2.0]);
        let mut back = [0.0; 5];
        groups.scatter(&gathered, &mut back);
        assert_eq!(back, values);
        assert!(Groups::new(&[1, 1, 2, 4]).sorted);
    }
}
