use clarabel::algebra::CscMatrix;
use epistat_linalg::{
    Accum, Mat, MatMut, MatRef, Par, Regularisation, Side, Solve, cholesky_factor, matmul,
};
use epistat_stats::dist::normal;

use crate::util::usize_to_f64;

pub(super) fn linear_grid(lb: f64, ub: f64, points: usize) -> Vec<f64> {
    if points <= 1 {
        return vec![lb];
    }
    let denom = usize_to_f64(points.saturating_sub(1));
    (0..points)
        .map(|idx| (ub - lb).mul_add(usize_to_f64(idx) / denom, lb))
        .collect()
}

pub(super) fn select_rows(matrix: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter().map(|idx| matrix[*idx].clone()).collect()
}

pub(super) fn subset_square_matrix(matrix: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row_idx| {
            rows.iter()
                .map(|col_idx| matrix[*row_idx][*col_idx])
                .collect()
        })
        .collect()
}

pub(super) fn diag_sqrt(matrix: &[Vec<f64>]) -> Vec<f64> {
    (0..matrix.len())
        .map(|idx| matrix[idx][idx].max(0.0).sqrt())
        .collect()
}

pub(super) fn drop_zero_columns(matrix: &[Vec<f64>], tol: f64) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let keep: Vec<bool> = (0..matrix[0].len())
        .map(|col_idx| matrix.iter().any(|row| row[col_idx].abs() > tol))
        .collect();
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(col_idx, value)| keep[col_idx].then_some(*value))
                .collect()
        })
        .collect()
}

pub(super) fn cholesky_lower(matrix: &[Vec<f64>]) -> Result<Mat<f64>, String> {
    if matrix.iter().any(|row| row.len() != matrix.len()) {
        return Err("covariance matrix must be square".to_string());
    }
    if matrix.is_empty() {
        return Ok(Mat::zeros(0, 0));
    }
    let sym = Mat::from_fn(matrix.len(), matrix.len(), |row, col| {
        f64::midpoint(matrix[row][col], matrix[col][row])
    });
    if let Some(lower) = cholesky_factor(sym.as_ref(), Regularisation::default()) {
        return Ok(lower);
    }
    let max_diag = (0..sym.nrows())
        .map(|idx| sym[(idx, idx)].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let jitter = Regularisation {
        delta: max_diag * 1e-12,
        epsilon: 1e-12,
    };
    if let Some(lower) = cholesky_factor(sym.as_ref(), jitter) {
        return Ok(lower);
    }
    cholesky_lower_tolerant(&sym)
}

fn cholesky_lower_tolerant(matrix: &Mat<f64>) -> Result<Mat<f64>, String> {
    let n = matrix.nrows();
    let max_diag = (0..n)
        .map(|idx| matrix[(idx, idx)].abs())
        .fold(0.0_f64, f64::max);
    let neg_diag_tol = 1e-10_f64 * max_diag.max(1.0);
    let mut lower = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[(i, j)];
            for k in 0..j {
                sum = lower[(i, k)].mul_add(-lower[(j, k)], sum);
            }
            if i == j {
                if sum < -neg_diag_tol {
                    return Err("covariance matrix is not positive semidefinite".to_string());
                }
                lower[(i, j)] = sum.max(0.0).sqrt();
            } else if lower[(j, j)].abs() > 1e-12 {
                lower[(i, j)] = sum / lower[(j, j)];
            }
        }
    }
    Ok(lower)
}

pub(super) fn lower_mat_vec_mul_into(lower: &Mat<f64>, vec: &[f64], out: &mut [f64]) {
    debug_assert_eq!(lower.nrows(), out.len());
    debug_assert_eq!(lower.ncols(), vec.len());
    if lower.nrows() >= 16 {
        let vector = MatRef::from_column_major_slice(vec, vec.len(), 1);
        let out_view = MatMut::from_column_major_slice_mut(out, out.len(), 1);
        matmul(
            out_view,
            Accum::Replace,
            lower.as_ref(),
            vector,
            1.0,
            Par::Seq,
        );
        return;
    }
    for row in 0..lower.nrows() {
        let mut sum = 0.0;
        for col in 0..=row {
            sum = lower[(row, col)].mul_add(vec[col], sum);
        }
        out[row] = sum;
    }
}

pub(super) fn pointwise_confidence_level_from_critical(critical: f64) -> f64 {
    (2.0 * normal::cdf(critical) - 1.0).clamp(1e-12, 1.0 - 1e-12)
}

pub(super) fn critical_value_from_pointwise_confidence(pointwise: f64) -> f64 {
    normal::quantile(f64::midpoint(1.0, pointwise))
}

/// The sup-t critical value of the estimator whose covariance is `covariance`,
/// for a joint confidence level of `confidence_level`.
pub(super) fn sup_t_critical_value(
    covariance: &[Vec<f64>],
    confidence_level: f64,
    simulation_draws: usize,
    simulation_seed: u64,
) -> Result<f64, String> {
    let dim = covariance.len();
    if covariance.iter().any(|row| row.len() != dim) {
        return Err("covariance matrix must be square".to_string());
    }
    let matrix = Mat::from_fn(dim, dim, |row, col| covariance[row][col]);
    epistat_inference::simultaneous::sup_t_from_covariance(
        matrix.as_ref(),
        1.0 - confidence_level,
        simulation_draws,
        simulation_seed,
    )
    .map_err(|error| format!("sup-t critical value: {error}"))
}

/// Covariance of the linear functionals `weights[i]' beta` of an estimator
/// whose covariance is `covariance`. Fails with `message` when a functional
/// has a variance that is not above `1e-14`.
pub(super) fn functional_covariance_matrix(
    covariance: &[Vec<f64>],
    weight_sets: &[Vec<f64>],
    message: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let projected = weight_sets
        .iter()
        .map(|weights| {
            covariance
                .iter()
                .map(|row| dot(weights, row))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let variances = projected
        .iter()
        .zip(weight_sets.iter())
        .map(|(projection, weights)| dot(weights, projection).max(0.0))
        .collect::<Vec<_>>();
    if variances.iter().any(|variance| *variance <= 1e-14) {
        return Err(message.to_string());
    }
    let count = weight_sets.len();
    let mut result = vec![vec![0.0; count]; count];
    for i in 0..count {
        result[i][i] = variances[i];
        for j in (i + 1)..count {
            let value = dot(&weight_sets[i], &projected[j]);
            result[i][j] = value;
            result[j][i] = value;
        }
    }
    Ok(result)
}

pub(super) fn post_covariance_block(
    covariance: &[Vec<f64>],
    num_pre_periods: usize,
    num_post_periods: usize,
) -> Vec<Vec<f64>> {
    (0..num_post_periods)
        .map(|row| {
            (0..num_post_periods)
                .map(|col| covariance[num_pre_periods + row][num_pre_periods + col])
                .collect()
        })
        .collect()
}

pub(super) fn solve_square_linear_system(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Vec<f64>, String> {
    let (rows, cols) = matrix_shape(matrix, "coefficient")?;
    if rows != cols {
        return Err(format!(
            "square linear solve requires a square coefficient matrix; got {rows}x{cols}"
        ));
    }
    if rhs.len() != rows {
        return Err(format!(
            "square linear solve rhs length {} does not match matrix size {rows}",
            rhs.len()
        ));
    }
    let dense = Mat::from_fn(rows, cols, |row, col| matrix[row][col]);
    let rhs_mat = Mat::from_fn(rows, 1, |row, _| rhs[row]);
    let solution = dense.as_ref().llt(Side::Lower).map_or_else(
        |_| dense.as_ref().partial_piv_lu().solve(&rhs_mat),
        |cholesky| cholesky.solve(&rhs_mat),
    );
    let out: Vec<f64> = (0..rows).map(|row| solution[(row, 0)]).collect();
    let residual = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, rhs_value)| {
            row.iter()
                .zip(out.iter())
                .map(|(coeff, value)| coeff * value)
                .sum::<f64>()
                - rhs_value
        })
        .map(f64::abs)
        .fold(0.0, f64::max);
    let scale = rhs
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    if residual > 1e-7 * scale {
        return Err(format!(
            "square linear solve residual {residual} exceeds tolerance {}",
            1e-7 * scale
        ));
    }
    Ok(out)
}

pub(super) fn solve_square_linear_system_transposed(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Vec<f64>, String> {
    let (rows, cols) = matrix_shape(matrix, "coefficient")?;
    if rows != cols {
        return Err(format!(
            "transposed square linear solve requires a square coefficient matrix; got {rows}x{cols}"
        ));
    }
    if rhs.len() != rows {
        return Err(format!(
            "transposed square linear solve rhs length {} does not match matrix size {rows}",
            rhs.len()
        ));
    }
    let dense_transposed = Mat::from_fn(rows, cols, |row, col| matrix[col][row]);
    let rhs_mat = Mat::from_fn(rows, 1, |row, _| rhs[row]);
    let solution = dense_transposed.as_ref().partial_piv_lu().solve(&rhs_mat);
    let out: Vec<f64> = (0..rows).map(|row| solution[(row, 0)]).collect();
    let residual = (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| matrix[col][row] * out[col])
                .sum::<f64>()
                - rhs[row]
        })
        .map(f64::abs)
        .fold(0.0, f64::max);
    let scale = rhs
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    if residual > 1e-7 * scale {
        return Err(format!(
            "transposed square linear solve residual {residual} exceeds tolerance {}",
            1e-7 * scale
        ));
    }
    Ok(out)
}

pub(super) fn invert_square_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let (rows, cols) = matrix_shape(matrix, "coefficient")?;
    if rows != cols {
        return Err(format!(
            "matrix inverse requires a square coefficient matrix; got {rows}x{cols}"
        ));
    }
    let dense = Mat::from_fn(rows, cols, |row, col| matrix[row][col]);
    let identity = Mat::from_fn(rows, cols, |row, col| if row == col { 1.0 } else { 0.0 });
    let inverse = dense.as_ref().partial_piv_lu().solve(&identity);
    let out: Vec<Vec<f64>> = (0..rows)
        .map(|row| (0..cols).map(|col| inverse[(row, col)]).collect())
        .collect();
    let residual = (0..rows)
        .flat_map(|row| {
            let out_ref = &out;
            (0..cols).map(move |col| {
                let expected = if row == col { 1.0 } else { 0.0 };
                matrix[row]
                    .iter()
                    .zip(out_ref.iter())
                    .map(|(left, inverse_row)| left * inverse_row[col])
                    .sum::<f64>()
                    - expected
            })
        })
        .map(f64::abs)
        .fold(0.0, f64::max);
    if residual > 1e-7 {
        return Err(format!(
            "matrix inverse residual {residual} exceeds tolerance 1e-7"
        ));
    }
    Ok(out)
}

pub(super) fn try_invert_square_matrix_row_major_into(
    matrix: &[f64],
    n: usize,
    out: &mut Vec<f64>,
) -> bool {
    let dense = MatRef::from_row_major_slice(matrix, n, n).to_owned();
    let identity = Mat::from_fn(n, n, |row, col| if row == col { 1.0 } else { 0.0 });
    let inverse = dense.as_ref().partial_piv_lu().solve(&identity);
    out.clear();
    out.resize(n * n, 0.0);
    for row in 0..n {
        for col in 0..n {
            out[row * n + col] = inverse[(row, col)];
        }
    }
    let mut residual = 0.0_f64;
    for row in 0..n {
        for col in 0..n {
            let dot = (0..n)
                .map(|k| matrix[row * n + k] * out[k * n + col])
                .sum::<f64>();
            let target = if row == col { 1.0 } else { 0.0 };
            residual = residual.max((dot - target).abs());
        }
    }
    residual <= 1e-7
}

pub(super) fn matrix_rank(matrix: &[Vec<f64>], tol: f64) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let (rows, cols) = matrix_shape_unchecked(matrix);
    let mut flat = flatten_row_major(matrix, rows * cols);
    let mut rank = 0usize;
    let mut row = 0usize;
    for col in 0..cols {
        let pivot = (row..rows).max_by(|&left, &right| {
            flat[left * cols + col]
                .abs()
                .total_cmp(&flat[right * cols + col].abs())
        });
        let Some(pivot_row) = pivot else { break };
        if flat[pivot_row * cols + col].abs() <= tol {
            continue;
        }
        if pivot_row != row {
            for offset in 0..cols {
                flat.swap(row * cols + offset, pivot_row * cols + offset);
            }
        }
        let row_start = row * cols;
        let pivot_value = flat[row_start + col];
        for value in &mut flat[(row_start + col)..(row_start + cols)] {
            *value /= pivot_value;
        }
        for r in 0..rows {
            if r == row {
                continue;
            }
            let target_start = r * cols;
            let factor = flat[target_start + col];
            if factor.abs() <= tol {
                continue;
            }
            for offset in col..cols {
                let target_idx = target_start + offset;
                flat[target_idx] = factor.mul_add(-flat[row_start + offset], flat[target_idx]);
            }
        }
        rank += 1;
        row += 1;
        if row == rows {
            break;
        }
    }
    rank
}

pub(super) fn mat_vec_mul_into(matrix: &[Vec<f64>], vector: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.reserve(matrix.len());
    for row in matrix {
        let value = row
            .iter()
            .zip(vector)
            .fold(0.0, |acc, (left, right)| left.mul_add(*right, acc));
        out.push(value);
    }
}

pub(super) fn sandwich_covariance(a: &[Vec<f64>], sigma: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (a_rows, a_cols) = matrix_shape_unchecked(a);
    if a_rows == 0 {
        return Vec::new();
    }
    let (sigma_rows, sigma_cols) = matrix_shape_unchecked(sigma);
    assert_eq!(
        a_cols, sigma_rows,
        "sandwich covariance dimension mismatch: a has {a_cols} columns, sigma has {sigma_rows} rows"
    );
    assert_eq!(
        sigma_rows, sigma_cols,
        "sandwich covariance requires square sigma; got {sigma_rows}x{sigma_cols}"
    );

    let a_flat = flatten_row_major(a, a_rows * a_cols);
    let sigma_flat = flatten_row_major(sigma, sigma_rows * sigma_cols);
    let a_mat = MatRef::from_row_major_slice(&a_flat, a_rows, a_cols).to_owned();
    let sigma_mat = MatRef::from_row_major_slice(&sigma_flat, sigma_rows, sigma_cols).to_owned();

    let mut tmp = Mat::<f64>::zeros(a_rows, sigma_cols);
    matmul(
        tmp.as_mut(),
        Accum::Replace,
        a_mat.as_ref(),
        sigma_mat.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut out = Mat::<f64>::zeros(a_rows, a_rows);
    matmul(
        out.as_mut(),
        Accum::Replace,
        tmp.as_ref(),
        a_mat.as_ref().transpose(),
        1.0,
        Par::Seq,
    );
    mat_to_row_vecs(&out)
}

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

pub(super) fn bilinear_form_into(
    left: &[f64],
    matrix: &[Vec<f64>],
    right: &[f64],
    tmp: &mut Vec<f64>,
) -> f64 {
    let (rows, cols) = matrix_shape_unchecked(matrix);
    assert_eq!(
        left.len(),
        rows,
        "bilinear form dimension mismatch: left len {} != matrix rows {rows}",
        left.len()
    );
    assert_eq!(
        right.len(),
        cols,
        "bilinear form dimension mismatch: right len {} != matrix cols {cols}",
        right.len()
    );
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    tmp.clear();
    tmp.resize(cols, 0.0);
    for col in 0..cols {
        tmp[col] = left
            .iter()
            .enumerate()
            .map(|(row, l)| l * matrix[row][col])
            .sum::<f64>();
    }
    dot(tmp, right)
}

fn matrix_shape(matrix: &[Vec<f64>], name: &str) -> Result<(usize, usize), String> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    if matrix.iter().any(|row| row.len() != cols) {
        return Err(format!("{name} matrix has non-rectangular rows"));
    }
    Ok((rows, cols))
}

fn matrix_shape_unchecked(matrix: &[Vec<f64>]) -> (usize, usize) {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    debug_assert!(matrix.iter().all(|row| row.len() == cols));
    (rows, cols)
}

fn flatten_row_major(matrix: &[Vec<f64>], capacity: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(capacity);
    for row in matrix {
        flat.extend_from_slice(row);
    }
    flat
}

fn mat_to_row_vecs(matrix: &Mat<f64>) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; matrix.ncols()]; matrix.nrows()];
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            out[row][col] = matrix[(row, col)];
        }
    }
    out
}

pub(super) fn build_clarabel_matrix(
    inequalities: &[Vec<f64>],
    equalities: &[Vec<f64>],
) -> CscMatrix<f64> {
    let total_rows = inequalities.len() + equalities.len();
    let total_cols = inequalities
        .first()
        .map_or_else(|| equalities.first().map_or(0, Vec::len), Vec::len);
    let cap = total_rows * total_cols;
    let mut row_idx = Vec::with_capacity(cap);
    let mut col_idx = Vec::with_capacity(cap);
    let mut values = Vec::with_capacity(cap);
    for (r, row) in inequalities.iter().chain(equalities.iter()).enumerate() {
        for (c, value) in row.iter().enumerate() {
            if *value != 0.0 {
                row_idx.push(r);
                col_idx.push(c);
                values.push(*value);
            }
        }
    }
    CscMatrix::new_from_triplets(total_rows, total_cols, row_idx, col_idx, values)
}

#[cfg(test)]
pub(super) fn dense_rows_to_csc(rows: &[Vec<f64>]) -> CscMatrix<f64> {
    build_clarabel_matrix(rows, &[])
}

/// Quantile of a standard normal truncated to `[l, u]`, as
/// `TruncatedNormal::norminvp` in R. A degenerate interval `l == u` gives that
/// point. An interval so far in a tail that its probability underflows, which
/// the conditional tests meet when the truncation region lies dozens of
/// standard deviations from zero, goes to [`far_tail_truncated_quantile`]. An
/// interval with `l > u`, a `NaN` bound or a probability outside `[0, 1]` is an
/// error.
#[expect(
    clippy::float_cmp,
    reason = "an interval of one point has that point as every quantile"
)]
pub(super) fn truncated_normal_quantile(p: f64, l: f64, u: f64) -> Result<f64, String> {
    let undefined =
        || format!("truncated normal quantile is undefined for p = {p}, interval [{l}, {u}]");
    if p.is_nan() || l.is_nan() || u.is_nan() || !(0.0..=1.0).contains(&p) {
        return Err(undefined());
    }
    if l == u {
        return Ok(l);
    }
    epistat_stats::dist::normal::truncated_quantile(p, l, u)
        .or_else(|| far_tail_truncated_quantile(p, l, u))
        .ok_or_else(undefined)
}

/// The `p` quantile of a standard normal truncated to `[l, u]`, for an
/// interval on one side of zero, computed without the normal distribution
/// function so that it holds where that function underflows.
///
/// # Algorithm
///
/// An interval below zero is mirrored to `[a, b] = [-u, -l]` with the
/// quantile `1 - p`, so that `0 <= a < b`. With `Q(x) = 1 - Phi(x)` the upper
/// tail and `r(x) = Q(x) / Q(a)`, the quantile `x` solves
///
/// ```text
/// r(x) = 1 - p (1 - r(b)),
/// ln r(x) = (a - x)(a + x) / 2 + ln M(x) - ln M(a),
/// ```
///
/// where `M(x) = Q(x) / phi(x)` is Mills' ratio, evaluated by Laplace's
/// continued fraction `M(x) = 1 / (x + 1 / (x + 2 / (x + 3 / (x + ...))))`
/// with [`MILLS_TERMS`] terms from the inside out. Every quantity stays a
/// ratio or a logarithm, so nothing underflows. Since
/// `d ln r(x) / dx = -1 / M(x)`, Newton's step is `x + (ln r(x) - target)
/// M(x)`, started from `sqrt(a^2 - 2 target)`, the root when `M` is taken as
/// constant, and kept inside `[a, b]`. `M` decreases, so the start lies above
/// the root, and `ln r` is concave and decreasing, so from there every step
/// stays above the root and moves down to it. The iteration stops when a step
/// no longer moves `x`.
///
/// `None` for an interval that contains zero, whose probability cannot
/// underflow, or for `a` infinite.
///
/// # References
///
/// - Laplace, P. S. (1805). *Traité de mécanique céleste*, vol. 4, the
///   continued fraction of the normal tail.
/// - Botev, Z. I. (2017). The normal law under linear restrictions:
///   simulation and estimation via minimax tilting. *Journal of the Royal
///   Statistical Society B* 79, 125-148, whose `TruncatedNormal::norminvp`
///   inverts the truncated normal in the tails by Newton's method on the same
///   logarithmic scale.
#[expect(
    clippy::float_cmp,
    reason = "Newton's iteration stops when a step leaves x unchanged"
)]
fn far_tail_truncated_quantile(p: f64, l: f64, u: f64) -> Option<f64> {
    let (a, b, q, sign) = if l >= 0.0 {
        (l, u, p, 1.0)
    } else if u <= 0.0 {
        (-u, -l, 1.0 - p, -1.0)
    } else {
        return None;
    };
    if !a.is_finite() || a >= b {
        return None;
    }
    if q == 0.0 {
        return Some(sign * a);
    }
    if q == 1.0 {
        return Some(sign * b);
    }
    let ln_mills_a = mills_ratio(a).ln();
    let ln_ratio = |x: f64| {
        if x == f64::INFINITY {
            f64::NEG_INFINITY
        } else {
            (a - x) * (a + x) / 2.0 + mills_ratio(x).ln() - ln_mills_a
        }
    };
    // ln(1 - q (1 - r(b))), with 1 - r(b) = -expm1(ln r(b)).
    let target = (-q * -ln_ratio(b).exp_m1()).ln_1p();
    let mut x = (a * a - 2.0 * target).sqrt().clamp(a, b);
    for _ in 0..MAX_NEWTON_STEPS {
        let next = (x + (ln_ratio(x) - target) * mills_ratio(x)).clamp(a, b);
        if next == x {
            break;
        }
        x = next;
    }
    Some(sign * x)
}

/// Terms of the continued fraction in [`mills_ratio`]. The far-tail quantile
/// meets it above about 37 standard deviations, where a few terms reach full
/// precision; with 200 the relative error stays below `1e-15` down to `x = 2`.
const MILLS_TERMS: u32 = 200;

/// Newton steps allowed in [`far_tail_truncated_quantile`]; it converges in
/// under ten.
const MAX_NEWTON_STEPS: usize = 100;

/// Mills' ratio `Q(x) / phi(x)` of the standard normal for `x > 0`, by
/// Laplace's continued fraction evaluated from the inside out.
fn mills_ratio(x: f64) -> f64 {
    let mut denominator = x;
    for k in (1..=MILLS_TERMS).rev() {
        denominator = x + f64::from(k) / denominator;
    }
    1.0 / denominator
}

#[cfg(test)]
mod tests {
    use super::truncated_normal_quantile;

    #[test]
    fn truncated_quantile_follows_norminvp_on_degenerate_and_empty_intervals() {
        assert!((truncated_normal_quantile(0.5, -1.0, 1.0).expect("interval")).abs() < 1e-12);
        assert_eq!(truncated_normal_quantile(0.9, 0.3, 0.3), Ok(0.3));
        assert!(truncated_normal_quantile(0.9, 1.0, 0.0).is_err());
        assert!(truncated_normal_quantile(0.9, f64::NAN, 1.0).is_err());
        assert!(truncated_normal_quantile(1.5, 0.0, 1.0).is_err());
    }

    #[derive(serde::Deserialize)]
    struct NorminvpCase {
        p: f64,
        l: f64,
        u: f64,
        quantile: f64,
    }

    /// Against `TruncatedNormal::norminvp`, on intervals where the normal
    /// distribution function underflows and on moderate tails where it does
    /// not. Beyond 35 standard deviations R inverts by Newton's method and
    /// stops once a step falls below `1e-10`, so its quantile is within that
    /// of the root; the quantiles here are between 20 and 80 in size, so
    /// `1e-10` relative is at least that bound.
    #[test]
    fn truncated_quantile_matches_norminvp_in_the_tails() {
        let cases: Vec<NorminvpCase> =
            epistat_testkit::load_json("tests/data/truncated_normal_quantile_ref.json");
        assert!(!cases.is_empty());
        let mut worst = 0.0_f64;
        for case in &cases {
            let ours = truncated_normal_quantile(case.p, case.l, case.u).expect("defined");
            let relative = ((ours - case.quantile) / case.quantile).abs();
            worst = worst.max(relative);
            assert!(
                relative < 1e-10,
                "p = {}, [{}, {}]: {ours} against R's {}",
                case.p,
                case.l,
                case.u,
                case.quantile
            );
        }
        println!("worst relative deviation from norminvp: {worst:e}");
    }

    /// Where the distribution function does not underflow, the far-tail
    /// route agrees with the direct one, which inverts `Phi` and so loses
    /// relative precision in proportion to the tail's depth: `1e-12`
    /// relative holds to 30 standard deviations.
    #[test]
    fn far_tail_route_agrees_with_the_direct_inversion() {
        for &(p, l, u) in &[
            (0.5, 5.0, 6.0),
            (0.9, 10.0, 12.0),
            (0.1, -30.0, -20.0),
            (0.7, 20.0, f64::INFINITY),
            (0.3, f64::NEG_INFINITY, -8.0),
        ] {
            let direct = epistat_stats::dist::normal::truncated_quantile(p, l, u).expect("direct");
            let tail = super::far_tail_truncated_quantile(p, l, u).expect("tail");
            assert!(
                ((direct - tail) / direct).abs() < 1e-12,
                "p = {p}, [{l}, {u}]: direct {direct}, far tail {tail}"
            );
        }
    }

    #[test]
    fn mills_ratio_matches_the_normal_tail() {
        // Q(x) / phi(x): at 2, 5 and 8 R's pnorm(x, lower.tail = FALSE) /
        // dnorm(x); at 40, where pnorm underflows, the asymptotic series
        // (1 / x) sum_k (-1)^k (2k - 1)!! / x^(2k) to eight terms, whose
        // truncation error is below 1e-20.
        for &(x, expected) in &[
            (2.0, 0.421_369_229_288_054_5),
            (5.0, 0.192_808_104_715_315_76),
            (8.0, 0.123_131_963_257_932_3),
            (40.0, 0.024_984_404_205_720_574),
        ] {
            let ours = super::mills_ratio(x);
            assert!(
                ((ours - expected) / expected).abs() < 1e-15,
                "x = {x}: {ours}"
            );
        }
    }
}
