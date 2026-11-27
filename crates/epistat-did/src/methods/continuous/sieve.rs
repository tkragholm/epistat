//! The sieve regression behind [`super::estimate`].

use epistat_linalg::basis::Basis;
use epistat_linalg::{Col, Mat};

use crate::error::Error;
use crate::estimators::common::linalg::{solve_spd_system, solve_spd_system_multi_rhs};
use crate::inference::standard_error_from_influence;
use crate::util::usize_to_f64;

/// The numbers of one sieve fit, before they become a result.
pub(super) struct SieveFit {
    pub(super) acrt: f64,
    pub(super) coefficients: Vec<f64>,
    pub(super) se: f64,
    pub(super) influence: Vec<f64>,
    pub(super) n_treated: usize,
    pub(super) n_control: usize,
}

/// The sieve ACRT of the columns `dose`, `change` and `weight`, validated by
/// the caller, on `basis`; see the module documentation.
pub(super) fn estimate(
    dose: &[f64],
    change: &[f64],
    weight: &dyn Fn(usize) -> f64,
    basis: &dyn Basis,
) -> Result<SieveFit, Error> {
    let n = dose.len();

    // 1. The control trend m0.
    let mut control_sum = 0.0;
    let mut control_weight = 0.0;
    let mut n_control = 0;
    for i in 0..n {
        if dose[i] == 0.0 {
            control_sum = weight(i).mul_add(change[i], control_sum);
            control_weight += weight(i);
            n_control += 1;
        }
    }
    if control_weight <= 0.0 {
        return Err(Error::NoComparison);
    }
    let m0 = control_sum / control_weight;

    // 2. The treated units.
    let treated_indices: Vec<usize> = (0..n).filter(|&i| dose[i] > 0.0).collect();
    let nt = treated_indices.len();
    if nt == 0 {
        return Err(Error::NoTreated);
    }

    let k = basis.len();

    // 3. The normal equations of the treated units.
    let mut xtwx = vec![0.0; k * k];
    let mut xtw_outcome = vec![0.0; k];

    let mut psi_vecs = Vec::with_capacity(nt);
    let mut deriv_vecs = Vec::with_capacity(nt);

    for &idx in &treated_indices {
        let b_vals = basis.values(dose[idx]);
        let d_vals = basis.derivatives(dose[idx]);
        let w = weight(idx);

        let outcome_star = change[idx] - m0;

        for r in 0..k {
            xtw_outcome[r] = (w * b_vals[r]).mul_add(outcome_star, xtw_outcome[r]);
            for c in 0..k {
                xtwx[r * k + c] = (w * b_vals[r]).mul_add(b_vals[c], xtwx[r * k + c]);
            }
        }
        psi_vecs.push(b_vals);
        deriv_vecs.push(d_vals);
    }

    let theta_vec = solve_spd_system(&xtwx, &xtw_outcome).map_err(|_| Error::Singular)?;
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
    let alpha_mat =
        solve_spd_system_multi_rhs(&xtwx, &mean_deriv_mat).map_err(|_| Error::Singular)?;
    let alpha = Col::<f64>::from_fn(k, |j| alpha_mat[(j, 0)]);

    let (units, treated) = (usize_to_f64(n), usize_to_f64(nt));
    let mut influence_function = vec![0.0; n];
    let mut alpha_s = 0.0;
    for (slot, &idx) in treated_indices.iter().enumerate() {
        let psi_i = &psi_vecs[slot];
        let alpha_psi = (0..k).map(|j| alpha[j] * psi_i[j]).sum::<f64>();
        alpha_s += weight(idx) * alpha_psi;
        let fitted = (0..k).map(|j| psi_i[j] * theta[j]).sum::<f64>();
        let resid = change[idx] - m0 - fitted;
        influence_function[idx] = units / treated * (acrt_i_vals[slot] - acrt_glob)
            + units * weight(idx) * alpha_psi * resid;
    }
    for (i, value) in influence_function.iter_mut().enumerate() {
        if dose[i] == 0.0 {
            *value = -alpha_s * units * weight(i) * (change[i] - m0) / control_weight;
        }
    }
    let se = standard_error_from_influence(&influence_function);

    Ok(SieveFit {
        acrt: acrt_glob,
        coefficients: theta_vec,
        se,
        influence: influence_function,
        n_treated: nt,
        n_control,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::basis::Polynomial;

    #[test]
    fn acrt_sieve_linear_case_with_control() {
        let mut dose = vec![0.0; 50];
        let mut change = vec![2.0; 50];
        for i in 1..=50 {
            let d = f64::from(i);
            dose.push(d);
            change.push(5.0f64.mul_add(d, 2.0));
        }
        let result = estimate(&dose, &change, &|_| 1.0, &Polynomial { degree: 1 }).unwrap();
        assert!((result.acrt - 5.0).abs() < 1e-10);
    }
}
