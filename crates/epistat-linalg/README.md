# epistat-linalg

Dense linear algebra on [faer](https://crates.io/crates/faer) whose results
do not depend on the number of threads: solves that fall back through a
chosen sequence of factorisations (Cholesky, Bunch-Kaufman, LU with partial
or full pivoting, each optionally retried with a growing ridge), and the
products of regression (`X'WX`, `X'Wz`, `X beta`) computed over fixed blocks
of rows.
