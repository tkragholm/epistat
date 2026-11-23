//! Sieve-based continuous-treatment `DiD` estimation.
//!
//! The Average Causal Response on the Treated (ACRT) of Callaway,
//! Goodman-Bacon and Sant'Anna (2024) with a sieve approximation to the
//! dose-response function: the control trend is estimated, the treated units'
//! outcome changes are projected onto a basis in the dose, and the derivative
//! of the fitted function is averaged over the treated doses.
//!
//! # Algorithm
//!
//! With `ΔY_i` the outcome change, `D_i` the dose, `w_i` the weight and
//! `ψ(d)` a `K`-dimensional basis, the units with `D = 0` are the controls `C`
//! and those with `D > 0` the treated `T`. The estimator solves
//!
//! ```text
//! m0 = sum_C w_i ΔY_i / W_C,                    W_C = sum_C w_i,
//! θ  = H^-1 sum_T w_i ψ(D_i) (ΔY_i - m0),       H   = sum_T w_i ψ(D_i) ψ(D_i)',
//! ACRT = (1 / n_T) sum_T ψ'(D_i)' θ,
//! ```
//!
//! the last an unweighted mean over the `n_T` treated units.
//!
//! # Influence function
//!
//! The three equations are estimating equations of `(m0, θ, ACRT)` averaged
//! over all `n` units, so the influence function of a unit is the linearisation
//! of the whole procedure, scaled so that `ACRT - ACRT_0 ≈ (1/n) sum_i IF_i`,
//! the convention of [`standard_error_from_influence`]. With `α = H^-1 dbar`,
//! `dbar = (1 / n_T) sum_T ψ'(D_i)`, `r_i = ΔY_i - m0 - ψ(D_i)' θ` and
//! `s = sum_T w_i ψ(D_i)`,
//!
//! ```text
//! treated:   IF_i = (n / n_T)(ψ'(D_i)' θ - ACRT) + n w_i α' ψ(D_i) r_i
//! control:   IF_i = - α' s · n w_i (ΔY_i - m0) / W_C
//! ```
//!
//! and 0 for a unit with a negative dose, which is in neither group. The control
//! term is the effect of the estimated trend through `θ`. It is zero, up to
//! rounding, when the basis spans the constants, as the polynomial basis and
//! the full B-spline basis do: the trend then moves only the constant of the
//! fit, whose derivative is zero.
//!
//! # References
//!
//! 1. Callaway B, Goodman-Bacon A, Sant'Anna PHC. Difference-in-differences
//!    with a continuous treatment. NBER Working Paper 32117, 2024; R package
//!    `contdid`.

use crate::estimators::common::basis::Basis;
use crate::estimators::common::linalg::{solve_spd_system, solve_spd_system_multi_rhs};
use crate::inference::standard_error_from_influence;
use crate::types::{ACRTResult, ContinuousDidError, ContinuousObservation};
use crate::util::usize_to_f64;
use epistat_linalg::{Col, Mat};

/// Estimates the global Average Causal Response on the Treated (ACRT) with a
/// sieve regression of the outcome change on the dose.
///
/// Units with dose exactly 0 are the control group and units with a positive
/// dose are treated. Units with a negative dose enter neither group, and are
/// only given a zero influence value. See the module documentation for the
/// estimator and its influence function; the standard error is
/// [`standard_error_from_influence`] of it.
///
/// # Errors
/// Returns `ContinuousDidError::EmptyInput` if observations are empty,
/// `ContinuousDidError::InvalidObservation` for a dose, outcome or weight that
/// is not finite or a negative weight, `ContinuousDidError::NoTreatedUnits` if
/// no unit has a positive dose, `ContinuousDidError::NoControlUnits` if the
/// units with dose 0 have no positive weight, and
/// `ContinuousDidError::SingularBasis` if the design matrix is singular.
pub fn estimate_acrt_sieve<B: Basis>(
    observations: &[ContinuousObservation],
    basis: &B,
) -> Result<ACRTResult, ContinuousDidError> {
    if observations.is_empty() {
        return Err(ContinuousDidError::EmptyInput);
    }
    if observations.iter().any(|obs| {
        !(obs.dose.is_finite() && obs.delta_outcome.is_finite() && obs.weight.is_finite())
            || obs.weight < 0.0
    }) {
        return Err(ContinuousDidError::InvalidObservation);
    }

    let n = observations.len();

    // 1. The control trend m0.
    let mut control_sum = 0.0;
    let mut control_weight = 0.0;
    for obs in observations {
        if obs.dose == 0.0 {
            control_sum = obs.weight.mul_add(obs.delta_outcome, control_sum);
            control_weight += obs.weight;
        }
    }
    if control_weight <= 0.0 {
        return Err(ContinuousDidError::NoControlUnits);
    }
    let m0 = control_sum / control_weight;

    // 2. The treated units.
    let treated_indices: Vec<usize> = observations
        .iter()
        .enumerate()
        .filter(|(_, o)| o.dose > 0.0)
        .map(|(i, _)| i)
        .collect();
    let nt = treated_indices.len();
    if nt == 0 {
        return Err(ContinuousDidError::NoTreatedUnits);
    }

    let k = basis.num_basis();

    // 3. The normal equations of the treated units.
    let mut xtwx = vec![0.0; k * k];
    let mut xtw_outcome = vec![0.0; k];

    let mut psi_vecs = Vec::with_capacity(nt);
    let mut deriv_vecs = Vec::with_capacity(nt);

    for &idx in &treated_indices {
        let obs = &observations[idx];
        let b_vals = basis.eval(obs.dose);
        let d_vals = basis.eval_deriv(obs.dose);
        let w = obs.weight;

        let outcome_star = obs.delta_outcome - m0;

        for r in 0..k {
            xtw_outcome[r] = (w * b_vals[r]).mul_add(outcome_star, xtw_outcome[r]);
            for c in 0..k {
                xtwx[r * k + c] = (w * b_vals[r]).mul_add(b_vals[c], xtwx[r * k + c]);
            }
        }
        psi_vecs.push(b_vals);
        deriv_vecs.push(d_vals);
    }

    let theta_vec =
        solve_spd_system(&xtwx, &xtw_outcome).map_err(|_| ContinuousDidError::SingularBasis)?;
    let theta = Col::<f64>::from_fn(k, |i| theta_vec[i]);

    // 4. ACRT_glob.
    let mut acrt_sum = 0.0;
    let mut acrt_i_vals = Vec::with_capacity(nt);
    let mut mean_deriv = vec![0.0; k];

    for deriv in &deriv_vecs {
        let acrt_i = (0..k).map(|j| deriv[j] * theta[j]).sum::<f64>();
        acrt_sum += acrt_i;
        acrt_i_vals.push(acrt_i);
        for (acc, &value) in mean_deriv.iter_mut().zip(deriv.iter()) {
            *acc += value;
        }
    }
    let acrt_glob = acrt_sum / usize_to_f64(nt);
    for value in &mut mean_deriv {
        *value /= usize_to_f64(nt);
    }

    // 5. The influence function.
    let mean_deriv_col = Col::<f64>::from_fn(k, |j| mean_deriv[j]);
    let mean_deriv_mat = Mat::<f64>::from_fn(k, 1, |row, _| mean_deriv_col[row]);
    let alpha_mat = solve_spd_system_multi_rhs(&xtwx, &mean_deriv_mat)
        .map_err(|_| ContinuousDidError::SingularBasis)?;
    let alpha = Col::<f64>::from_fn(k, |j| alpha_mat[(j, 0)]);

    let (units, treated) = (usize_to_f64(n), usize_to_f64(nt));
    let mut influence_function = vec![0.0; n];
    let mut alpha_s = 0.0;
    for (slot, &idx) in treated_indices.iter().enumerate() {
        let obs = &observations[idx];
        let psi_i = &psi_vecs[slot];
        let alpha_psi = (0..k).map(|j| alpha[j] * psi_i[j]).sum::<f64>();
        alpha_s += obs.weight * alpha_psi;
        let fitted = (0..k).map(|j| psi_i[j] * theta[j]).sum::<f64>();
        let resid = obs.delta_outcome - m0 - fitted;
        influence_function[idx] = units / treated * (acrt_i_vals[slot] - acrt_glob)
            + units * obs.weight * alpha_psi * resid;
    }
    for (value, obs) in influence_function.iter_mut().zip(observations) {
        if obs.dose == 0.0 {
            *value = -alpha_s * units * obs.weight * (obs.delta_outcome - m0) / control_weight;
        }
    }
    let se = standard_error_from_influence(&influence_function);

    Ok(ACRTResult {
        acrt_glob,
        se,
        coefficients: theta_vec,
        influence_function,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimators::common::basis::PolynomialBasis;

    #[test]
    fn acrt_sieve_linear_case_with_control() {
        let mut obs = Vec::new();
        // Control group: trend = 2.0
        for _ in 0..50 {
            obs.push(ContinuousObservation::new(0.0, 2.0));
        }
        // Treated group: trend = 2.0 + 5.0 * dose
        for i in 1..=50 {
            let dose = f64::from(i);
            obs.push(ContinuousObservation::new(dose, 5.0f64.mul_add(dose, 2.0)));
        }

        let basis = PolynomialBasis::new(1);
        let result = estimate_acrt_sieve(&obs, &basis).unwrap();

        assert!((result.acrt_glob - 5.0).abs() < 1e-10);
    }
}
