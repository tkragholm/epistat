# epistat-sequences

Distances between state sequences, weighted k-medoids over them, the
silhouette of the result and validation of a typology against null models.

- Optimal-matching distance, held as a condensed triangle, with k-medoids and
  cluster distance sums over it.
- Hamming and dynamic Hamming distance, with k-medoids and silhouette that need
  no distance matrix.
- Weighted average silhouette width, `None` where it is undefined.
- `fit_typology`: costs, restarts, k-medoids and silhouette in one call.
- Four null models for a parametric bootstrap, partition agreement (adjusted
  Rand index and greedy share), transition and substitution costs, and replicate
  loops for null silhouettes and refit agreement.

Sequences are rank-encoded `u8` slices with a weight each. Every draw comes from
the `epistat-rng` stream, restarts and replicates run in parallel, and the
results are the same on any number of threads.
