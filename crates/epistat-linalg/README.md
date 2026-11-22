# epistat-linalg

Dense linear algebra on [faer](https://crates.io/crates/faer) whose results
do not depend on the number of threads: solves that fall back through a
chosen sequence of factorisations (Cholesky, Bunch-Kaufman, LU with partial
or full pivoting, each optionally retried with a growing ridge), and the
products of regression (`X'WX`, `X'Wz`, `X beta`) computed over fixed blocks
of rows. Also the eigendecomposition of a small symmetric matrix by Jacobi rotations and
the quadratic form of its generalised inverse, B-spline and polynomial bases, restricted cubic splines (Harrell's form and knot rules, as `Hmisc::rcspline.eval`) and natural cubic splines (as `splines::ns`) with prediction at new values, a test for the columns of a
design that each group of rows can identify, and row and column selection.

The `nightly` feature enables faer's AVX-512 kernels and needs Rust 1.89 on x86.
