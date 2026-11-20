//! The Student t distribution: log density and derivatives for the location-scale
//! family. The distribution and quantile functions are in [`epistat_stats::student`].

use epistat_stats::student::{t_cdf_sf, t_ln_density, t_quantile};

/// `ln f(z)`, `(ln f)'` and `(ln f)''` of the density with `nu` degrees of
/// freedom: `(ln f)' = -(nu + 1) z / (nu + z^2)` and
/// `(ln f)'' = -(nu + 1) (nu - z^2) / (nu + z^2)^2`.
pub(super) fn log_density(z: f64, nu: f64) -> (f64, f64, f64) {
    let q = nu + z * z;
    (
        t_ln_density(z, nu),
        -(nu + 1.0) * z / q,
        -(nu + 1.0) * (nu - z * z) / (q * q),
    )
}

/// The distribution function and the survival function at `z`.
pub(super) fn cdf_sf(z: f64, nu: f64) -> (f64, f64) {
    t_cdf_sf(z, nu)
}

/// The quantile function for `p` in `(0, 1)`, NaN when it is out of range.
pub(super) fn quantile(p: f64, nu: f64) -> f64 {
    t_quantile(p, nu).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::{cdf_sf, log_density, quantile};
    use std::f64::consts::PI;

    /// One and two degrees of freedom have closed forms, the Cauchy
    /// `1/2 + atan(z) / pi` and `1/2 + z / (2 sqrt(2 + z^2))`. The error of
    /// the evaluation is a few eps in the tail it computes, and the check
    /// allows 64 eps relative to the smaller of the two tails.
    #[test]
    fn closed_forms_at_one_and_two_degrees_of_freedom() {
        for z in [-30.0, -3.0, -0.4, 0.0, 0.7, 2.0, 25.0] {
            let (cdf, sf) = cdf_sf(z, 1.0);
            let (want_cdf, want_sf) = (0.5 + z.atan() / PI, 0.5 - z.atan() / PI);
            let want_small = if z > 0.0 { want_sf } else { want_cdf };
            let small = if z > 0.0 { sf } else { cdf };
            assert!(
                (small - want_small).abs() <= 64.0 * f64::EPSILON * want_small,
                "{z}"
            );
            let (cdf2, _) = cdf_sf(z, 2.0);
            let want2 = 0.5 + z / (2.0 * (2.0 + z * z).sqrt());
            assert!((cdf2 - want2).abs() <= 64.0 * f64::EPSILON, "{z}");
            assert!((cdf + sf - 1.0).abs() <= 4.0 * f64::EPSILON);
        }
    }

    /// The density is the derivative of the distribution function, by
    /// central differences with step `h`: truncation about `h^2 f''/6` and
    /// rounding about `eps / h`, both well below the `1e-8` allowed at
    /// `h = 1e-5`.
    #[test]
    fn the_density_is_the_derivative_of_the_distribution_function() {
        let h = 1e-5;
        for nu in [2.5, 4.0, 30.0] {
            for z in [-4.0, -1.0, 0.0, 0.3, 2.0] {
                let slope = (cdf_sf(z + h, nu).0 - cdf_sf(z - h, nu).0) / (2.0 * h);
                let density = log_density(z, nu).0.exp();
                assert!((slope - density).abs() < 1e-8, "{nu} {z}");
            }
        }
    }

    #[test]
    fn the_quantile_inverts_the_distribution_function() {
        for nu in [1.0, 2.5, 4.0, 30.0, 400.0] {
            for p in [1e-10, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-10] {
                let z = quantile(p, nu);
                let (cdf, sf) = cdf_sf(z, nu);
                let (got, want) = if p < 0.5 { (cdf, p) } else { (sf, 1.0 - p) };
                assert!((got - want).abs() <= 1e-13 * want, "{nu} {p}: {z}");
            }
        }
    }
}
