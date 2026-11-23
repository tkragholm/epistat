//! Distribution draws on a [`Stream`]: the standard normal, the gamma family,
//! the Beta, and the Wishart and inverse Wishart matrices.
//!
//! # Algorithm
//!
//! [`Stream::normal`] is the Box-Muller transform \[1\]. For two successive
//! uniforms `u1` and `u2` on `[0, 1)`,
//!
//! ```text
//! z = sqrt(-2 ln(max(u1, MIN_POSITIVE))) cos(2 pi u2),
//! ```
//!
//! where `MIN_POSITIVE` is the smallest positive normal `f64`, so that a
//! uniform of exactly 0 does not give an infinite radius. The sine of the same
//! angle is an independent standard normal, which [`Stream::fill_normals`] uses
//! for every second value of a slice.
//!
//! [`Stream::gamma`] is Marsaglia and Tsang \[2\]. For shape `a >= 1` let
//! `d = a - 1/3` and `c = 1 / sqrt(9 d)`. Draw a standard normal `z` and a
//! uniform `u`, and with `v = (1 + c z)^3` accept `d v` if `v > 0` and
//!
//! ```text
//! u < 1 - 0.0331 z^4,    or    ln u < z^2 / 2 + d (1 - v + ln v).
//! ```
//!
//! Otherwise draw again. For `a < 1` the draw is `gamma(a + 1) * w^(1/a)`
//! for a uniform `w = 1 - u` (floored at `MIN_POSITIVE`) drawn before the
//! gamma variate \[2\]. The result is multiplied by the scale. A chi-square
//! variate with `k` degrees of freedom is `gamma(k / 2, 2)`, and a Beta
//! variate is `x / (x + y)` for `x ~ gamma(a, 1)` and `y ~ gamma(b, 1)`.
//!
//! [`Stream::wishart`] is the Bartlett decomposition \[3\]. For a scale matrix
//! `S = L L'` with `L` lower triangular and `p` rows, let `A` be lower
//! triangular with `A[i][i] = sqrt(chi-square(df - i))` and `A[i][j]` a
//! standard normal for `j < i`, drawn row by row, the diagonal first. Then
//! `W = (L A)(L A)'` has the Wishart distribution with `df` degrees of
//! freedom and scale `S`, and mean `df S`. [`Stream::inverse_wishart`] draws
//! the same `A` and returns `(L A^-T)(L A^-T)'`, which is the inverse of a
//! Wishart matrix with scale `S^-1` and so has mean `S / (df - p - 1)` for
//! `df > p + 1`. It needs only the triangular inverse of `A`, not a general
//! matrix inverse.
//!
//! # Edge cases
//!
//! A non-positive or non-finite shape, scale or degrees of freedom, a Wishart
//! with `df <= p - 1`, or a scale factor with a non-positive or non-finite
//! diagonal is a [`DrawError`]. No draw is taken before the check.
//!
//! # Reproducibility
//!
//! The uniforms are exact (see the crate documentation). The draws also use
//! `ln`, `sqrt`, `cos`, `sin` and `powf` from the platform's maths library, so
//! they agree between platforms only as far as that library does. No
//! fused multiply-add is used, so a build does not change them with the
//! target's instruction set, and `sin` and `cos` are always called
//! separately, so debug and release builds give the same draws.
//!
//! # References
//!
//! 1. Box GEP, Muller ME. A note on the generation of random normal deviates.
//!    Ann Math Stat. 1958;29(2):610-1.
//! 2. Marsaglia G, Tsang WW. A simple method for generating gamma variables.
//!    ACM Trans Math Softw. 2000;26(3):363-72. doi:10.1145/358407.358414
//! 3. Smith WB, Hocking RR. Algorithm AS 53: Wishart variate generator.
//!    J R Stat Soc Ser C. 1972;21(3):341-5. doi:10.2307/2346290

use std::f64::consts::PI;
use std::fmt;

use crate::Stream;

/// A parameter outside the range of a distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawError(&'static str);

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for DrawError {}

fn positive(value: f64, what: &'static str) -> Result<(), DrawError> {
    if value > 0.0 && value.is_finite() {
        Ok(())
    } else {
        Err(DrawError(what))
    }
}

impl Stream {
    /// A standard normal variate by Box-Muller from two uniforms: the first
    /// is floored at `f64::MIN_POSITIVE`, and the result is
    /// `sqrt(-2 ln u1) cos(2 pi u2)`.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * PI * u2).cos()
    }

    /// Fills `out` with standard normal variates, two per pair of uniforms:
    /// the cosine and the sine of the same Box-Muller angle. A last odd
    /// position takes one [`Stream::normal`].
    pub fn fill_normals(&mut self, out: &mut [f64]) {
        let (pairs, rest) = out.as_chunks_mut::<2>();
        for pair in pairs {
            let u1 = self.uniform().max(f64::MIN_POSITIVE);
            let u2 = self.uniform();
            let radius = (-2.0_f64 * u1.ln()).sqrt();
            let theta = 2.0_f64 * PI * u2;
            pair[0] = radius * theta.cos();
            // An optimised build may otherwise merge the two calls into one
            // `sincos`, whose last bit can differ from the separate `sin`.
            pair[1] = radius * std::hint::black_box(theta).sin();
        }
        if let [last] = rest {
            *last = self.normal();
        }
    }

    /// Fills `out` with independent signs, `+1.0` for a set bit and `-1.0`
    /// for a clear one, taking 64 signs from each output of the stream from
    /// the lowest bit up, as [`crate::rademacher_sum`] does.
    pub fn fill_signs(&mut self, out: &mut [f64]) {
        for chunk in out.chunks_mut(64) {
            let mut bits = self.next_u64();
            for sign in chunk {
                *sign = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
            }
        }
    }

    /// A gamma variate with the given shape and scale, mean `shape * scale`,
    /// by the method of Marsaglia and Tsang.
    ///
    /// # Errors
    ///
    /// [`DrawError`] when the shape or the scale is not positive and finite.
    pub fn gamma(&mut self, shape: f64, scale: f64) -> Result<f64, DrawError> {
        positive(shape, "the gamma shape must be positive and finite")?;
        positive(scale, "the gamma scale must be positive and finite")?;
        Ok(self.gamma_unchecked(shape, scale))
    }

    fn gamma_unchecked(&mut self, shape: f64, scale: f64) -> f64 {
        if shape < 1.0 {
            let u = (1.0_f64 - self.uniform()).max(f64::MIN_POSITIVE);
            return self.gamma_unchecked(shape + 1.0, scale) * u.powf(1.0 / shape);
        }
        let shape_minus_third = shape - (1.0 / 3.0);
        let coefficient = (1.0 / (9.0 * shape_minus_third)).sqrt();
        loop {
            let z = self.normal();
            let base = 1.0 + coefficient * z;
            if base <= 0.0 {
                continue;
            }
            let cube = base * base * base;
            let u = self.uniform();
            if u < 1.0 + (0.0331 * z * z * z) * (-z) {
                return scale * shape_minus_third * cube;
            }
            if u.ln() < shape_minus_third * (1.0 - cube + cube.ln()) + (0.5 * z) * z {
                return scale * shape_minus_third * cube;
            }
        }
    }

    /// A chi-square variate with `df` degrees of freedom, `gamma(df / 2, 2)`.
    ///
    /// # Errors
    ///
    /// [`DrawError`] when `df` is not positive and finite.
    pub fn chi_square(&mut self, df: f64) -> Result<f64, DrawError> {
        positive(
            df,
            "the chi-square degrees of freedom must be positive and finite",
        )?;
        Ok(self.gamma_unchecked(0.5 * df, 2.0))
    }

    /// A Beta(`alpha`, `beta`) variate, `x / (x + y)` for independent
    /// `x ~ gamma(alpha, 1)` and `y ~ gamma(beta, 1)`.
    ///
    /// The values are not those of `rand_distr::Beta` driven by the same
    /// stream, which uses another algorithm.
    ///
    /// # Errors
    ///
    /// [`DrawError`] when a shape is not positive and finite.
    pub fn beta(&mut self, alpha: f64, beta: f64) -> Result<f64, DrawError> {
        positive(alpha, "the Beta shapes must be positive and finite")?;
        positive(beta, "the Beta shapes must be positive and finite")?;
        let x = self.gamma_unchecked(alpha, 1.0);
        let y = self.gamma_unchecked(beta, 1.0);
        let total = x + y;
        // Both gamma variates can underflow to 0 for very small shapes.
        Ok(if total > 0.0 { x / total } else { 0.5 })
    }

    /// The Bartlett factor `A` of a Wishart draw: row-major, lower triangular,
    /// `A[i][i] = sqrt(chi-square(df - i))` and standard normals below it,
    /// drawn row by row with the diagonal first.
    fn bartlett(&mut self, df: f64, dim: usize) -> Vec<f64> {
        let mut factor = vec![0.0; dim * dim];
        for row in 0..dim {
            factor[row * dim + row] = self.gamma_unchecked(0.5 * (df - row as f64), 2.0).sqrt();
            for col in 0..row {
                factor[row * dim + col] = self.normal();
            }
        }
        factor
    }

    /// A Wishart matrix with `df` degrees of freedom and scale `L L'`, as a
    /// row-major `dim` by `dim` matrix, by the Bartlett decomposition.
    ///
    /// `scale_cholesky` is `L`, row-major and lower triangular. Its upper
    /// triangle is not read. The mean of the result is `df L L'`.
    ///
    /// # Errors
    ///
    /// [`DrawError`] when `dim` is 0, `scale_cholesky` does not hold `dim`
    /// squared values, its diagonal is not positive and finite, or `df` is not
    /// above `dim - 1`.
    pub fn wishart(
        &mut self,
        df: f64,
        scale_cholesky: &[f64],
        dim: usize,
    ) -> Result<Vec<f64>, DrawError> {
        check_wishart(df, scale_cholesky, dim)?;
        let bartlett = self.bartlett(df, dim);
        let mut product = vec![0.0; dim * dim];
        for row in 0..dim {
            for col in 0..=row {
                let mut sum = 0.0;
                for k in col..=row {
                    sum += scale_cholesky[row * dim + k] * bartlett[k * dim + col];
                }
                product[row * dim + col] = sum;
            }
        }
        Ok(gram_of_lower(&product, dim))
    }

    /// An inverse Wishart matrix with `df` degrees of freedom and scale
    /// matrix `L L'`, row-major, so that its inverse is Wishart with scale
    /// `(L L')^-1`. The mean is `L L' / (df - dim - 1)` for `df > dim + 1`.
    ///
    /// `scale_cholesky` is `L`, row-major and lower triangular. The draws are
    /// those of [`Stream::wishart`] for the same stream and arguments.
    ///
    /// # Errors
    ///
    /// As [`Stream::wishart`].
    pub fn inverse_wishart(
        &mut self,
        df: f64,
        scale_cholesky: &[f64],
        dim: usize,
    ) -> Result<Vec<f64>, DrawError> {
        check_wishart(df, scale_cholesky, dim)?;
        let bartlett = self.bartlett(df, dim);
        // `inverse` is A^-1, lower triangular, by forward substitution.
        let mut inverse = vec![0.0; dim * dim];
        for col in 0..dim {
            inverse[col * dim + col] = 1.0 / bartlett[col * dim + col];
            for row in col + 1..dim {
                let mut sum = 0.0;
                for k in col..row {
                    sum += bartlett[row * dim + k] * inverse[k * dim + col];
                }
                inverse[row * dim + col] = -sum / bartlett[row * dim + row];
            }
        }
        // P = L A^-T, where A^-T[k][col] = inverse[col][k] is nonzero for
        // k <= col, so k runs to the smaller of the row and the column.
        let mut product = vec![0.0; dim * dim];
        for row in 0..dim {
            for col in 0..dim {
                let mut sum = 0.0;
                for k in 0..=row.min(col) {
                    sum += scale_cholesky[row * dim + k] * inverse[col * dim + k];
                }
                product[row * dim + col] = sum;
            }
        }
        let mut out = vec![0.0; dim * dim];
        for row in 0..dim {
            for col in 0..=row {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += product[row * dim + k] * product[col * dim + k];
                }
                out[row * dim + col] = sum;
                out[col * dim + row] = sum;
            }
        }
        Ok(out)
    }
}

fn check_wishart(df: f64, scale_cholesky: &[f64], dim: usize) -> Result<(), DrawError> {
    if dim == 0 || scale_cholesky.len() != dim * dim {
        return Err(DrawError(
            "the Wishart scale factor must be a non-empty square matrix",
        ));
    }
    if !(df.is_finite() && df > dim as f64 - 1.0) {
        return Err(DrawError(
            "the Wishart degrees of freedom must be finite and above the dimension minus one",
        ));
    }
    for i in 0..dim {
        let diagonal = scale_cholesky[i * dim + i];
        if !(diagonal > 0.0 && diagonal.is_finite()) {
            return Err(DrawError(
                "the Wishart scale factor needs a positive, finite diagonal",
            ));
        }
    }
    Ok(())
}

/// `P P'` for lower triangular `P`, symmetric by construction.
fn gram_of_lower(lower: &[f64], dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim * dim];
    for row in 0..dim {
        for col in 0..=row {
            let mut sum = 0.0;
            for k in 0..=col {
                sum += lower[row * dim + k] * lower[col * dim + k];
            }
            out[row * dim + col] = sum;
            out[col * dim + row] = sum;
        }
    }
    out
}
