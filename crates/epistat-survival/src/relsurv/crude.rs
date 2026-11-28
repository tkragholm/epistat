//! The crude probability of death from the disease and from other causes, in
//! the presence of the other.

#![expect(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "times are compared with the values they were copied from, and counts of rows are far below 2^53"
)]

use super::ExpectedHazard;
use super::cohort::{SWEEP_LIMIT, rows_by_stratum};
use super::expected::Pieces;
use crate::error::{Error, Result};
use crate::surv::SurvData;
use epistat_core::Level;
use epistat_stats::dist::normal;

/// The crude probabilities of death of one stratum.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CrudeMortality {
    /// The level of the stratum, 0 when the data has no strata.
    pub stratum: usize,
    /// The distinct exit times.
    pub times: Vec<f64>,
    /// The probability of having died of the disease by each time, the
    /// cumulative excess hazard weighted by the observed survival.
    pub disease: Vec<f64>,
    /// Its variance.
    pub disease_variance: Vec<f64>,
    /// The probability of having died of other causes, at the population's
    /// rates, by each time.
    pub other: Vec<f64>,
    /// Its variance.
    pub other_variance: Vec<f64>,
    /// The normal limits of `disease`, `estimate -/+ z se`.
    pub disease_limits: Vec<(f64, f64)>,
    /// The normal limits of `other`.
    pub other_limits: Vec<(f64, f64)>,
    /// The observed survival of Kaplan and Meier, `1 - disease - other`.
    pub survival: Vec<f64>,
    /// The number at risk at each time.
    pub n_at_risk: Vec<f64>,
    /// The number of deaths at each time.
    pub events: Vec<f64>,
}

/// The crude probability of death from the disease and from other causes of
/// Cronin and Feuer, as `relsurv::cmp.rel`, for each stratum of the data.
///
/// Net survival describes a world in which the disease is the only cause of
/// death. The crude probabilities describe the cohort as it is: with `S` the
/// observed survival of Kaplan and Meier, `lambda_P` the mean reference hazard
/// of those at risk and `dLambda_E = dN / Y - lambda_P dt` the excess hazard
/// of Ederer II,
///
/// ```text
/// F_E(t) = integral_0^t S(u-) dLambda_E(u)          (the disease)
/// F_P(t) = integral_0^t S(u-) lambda_P(u) du        (other causes)
/// ```
///
/// and `F_E + F_P = 1 - S`. Between two consecutive entry or exit times the
/// risk set is fixed and `S` constant, so each integral over the interval is
/// `S` times the mean of the increments `H*_i(b) - H*_i(a)` over those at
/// risk, exactly; `cmp.rel` forms the same sums on a grid of days, which gives
/// these values whatever its step when entries fall on its grid. The
/// variances are those of `cmp.rel` (relsurv 2.3.3, `cmpfast.c`), with
/// `sigma_k = d_k / n_k^2` at the death times `k`:
///
/// ```text
/// Var F_E(t_j) = sum_{k <= j} sigma_k (S(t_k) - (F_E(t_j) - F_E(t_k)))^2
/// Var F_P(t_j) = sum_{k <= j} sigma_k (F_P(t_j) - F_P(t_k))^2
/// ```
///
/// formed by recurrences over `j` in terms whose signs do not cancel, so the
/// cost is linear in the number of times. The limits are normal, as in
/// `cmp.rel`. A row entering at `t` is at risk after `t`.
///
/// # Errors
///
/// An input error for expected hazards of another number of rows than the
/// data, a status with more than one event code, weights other than 0 and 1,
/// an entry before time 0, follow-up beyond what
/// `expected` covers, or more than 3e8 pairs of a person at risk and a time
/// point.
pub fn crude_mortality(
    data: &SurvData<'_>,
    expected: &ExpectedHazard,
    level: Level,
) -> Result<Vec<CrudeMortality>> {
    if data.followup().n_codes() != 1 {
        return Err(Error::not_applicable(
            "the crude probabilities need a binary status of death from any cause",
        ));
    }
    let groups = rows_by_stratum(data, expected)?;
    let surv = data.followup();
    let (start, stop, status) = (surv.start(), surv.stop(), surv.status());
    let z = normal::quantile(level.upper_probability());
    groups
        .into_iter()
        .enumerate()
        .map(|(stratum, rows)| one(stratum, &rows, start, stop, status, expected, z))
        .collect()
}

fn one(
    stratum: usize,
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    expected: &ExpectedHazard,
    z: f64,
) -> Result<CrudeMortality> {
    let begin = |i: usize| start.map_or(0.0, |s| s[i]);
    let mut points: Vec<f64> = rows.iter().map(|&i| stop[i]).collect();
    points.extend(rows.iter().map(|&i| begin(i)).filter(|&s| s > 0.0));
    points.sort_by(f64::total_cmp);
    points.dedup();
    let work: f64 = rows
        .iter()
        .map(|&i| {
            let lo = points.partition_point(|&p| p <= begin(i));
            let hi = points.partition_point(|&p| p <= stop[i]);
            (hi - lo) as f64
        })
        .sum();
    if work > SWEEP_LIMIT {
        return Err(Error::options(format!(
            "the crude probabilities would visit {work:.3e} pairs of a person and a time point, \
             above the limit of {SWEEP_LIMIT:.0e}"
        )));
    }
    let mut stops: Vec<f64> = rows.iter().map(|&i| stop[i]).collect();
    stops.sort_by(f64::total_cmp);
    stops.dedup();
    let mut by_start: Vec<usize> = rows.to_vec();
    by_start.sort_by(|&a, &b| begin(a).total_cmp(&begin(b)).then(a.cmp(&b)));
    let mut next_entrant = 0;
    let mut active: Vec<(usize, f64, (usize, usize))> = Vec::new();
    let mut reader = expected.reader();
    let mut pieces = Pieces::default();
    let mut out = CrudeMortality {
        stratum,
        times: Vec::new(),
        disease: Vec::new(),
        disease_variance: Vec::new(),
        other: Vec::new(),
        other_variance: Vec::new(),
        disease_limits: Vec::new(),
        other_limits: Vec::new(),
        survival: Vec::new(),
        n_at_risk: Vec::new(),
        events: Vec::new(),
    };
    let (mut s, mut f_e, mut f_p) = (1.0_f64, 0.0_f64, 0.0_f64);
    // The recurrences of the variances: for the disease A = sum sigma D^2,
    // B = sum sigma D with D_k = S_k - (F_E - F_E,k), and for other causes
    // P = sum sigma (F_P - F_P,k)^2, Q = sum sigma (F_P - F_P,k), and
    // C = sum sigma for both.
    let (mut a, mut b_sum, mut c) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut p, mut q) = (0.0_f64, 0.0_f64);
    let mut previous = 0.0_f64;
    for &b in &points {
        while next_entrant < by_start.len() && begin(by_start[next_entrant]) < b {
            let i = by_start[next_entrant];
            let at = reader.append(i, &mut pieces);
            let h = pieces.slice(at.clone()).cumulative_within(previous);
            active.push((i, h, (at.start, at.end)));
            next_entrant += 1;
        }
        if active.is_empty() {
            previous = b;
            continue;
        }
        let n = active.len() as f64;
        let (mut dh, mut deaths) = (0.0_f64, 0.0_f64);
        for (i, h_a, at) in &mut active {
            let h_b = pieces.slice(at.0..at.1).cumulative_within(b);
            dh += h_b - *h_a;
            if stop[*i] == b && status[*i] != 0 {
                deaths += 1.0;
            }
            *h_a = h_b;
        }
        let population = dh / n;
        let observed = deaths / n;
        let delta_e = s * (observed - population);
        let delta_p = s * population;
        f_e += delta_e;
        f_p += delta_p;
        s *= 1.0 - observed;
        // Move the sums to the new F, then add the term of this time.
        a = a - 2.0 * delta_e * b_sum + delta_e * delta_e * c;
        b_sum -= delta_e * c;
        p = p + 2.0 * delta_p * q + delta_p * delta_p * c;
        q += delta_p * c;
        if deaths > 0.0 {
            let sigma = deaths / (n * n);
            a += sigma * s * s;
            b_sum += sigma * s;
            c += sigma;
        }
        if stops.binary_search_by(|x| x.total_cmp(&b)).is_ok() {
            let (ve, vp) = (a.max(0.0), p.max(0.0));
            out.times.push(b);
            out.disease.push(f_e);
            out.disease_variance.push(ve);
            out.other.push(f_p);
            out.other_variance.push(vp);
            out.disease_limits
                .push((f_e - z * ve.sqrt(), f_e + z * ve.sqrt()));
            out.other_limits
                .push((f_p - z * vp.sqrt(), f_p + z * vp.sqrt()));
            out.survival.push(s);
            out.n_at_risk.push(n);
            out.events.push(deaths);
        }
        active.retain(|&(i, _, _)| stop[i] > b);
        previous = b;
    }
    Ok(out)
}
