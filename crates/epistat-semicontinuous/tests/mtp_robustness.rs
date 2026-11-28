use epistat_linalg::Mat;
use epistat_core::Seed;
use epistat_rng::Stream;
use epistat_semicontinuous::LongitudinalData;
use epistat_semicontinuous::mtp::{
    self, MtpModel, MtpOptions, MtpPosteriorSamples, MtpProposalTuning,
    posterior_predictive_summary,
};

/// The columns of a simulated panel.
struct Panel {
    outcome: Vec<f64>,
    x_binary: Mat<f64>,
    x_mean: Mat<f64>,
    subject_ids: Vec<u32>,
    time: Vec<f64>,
}

impl Panel {
    fn data(&self) -> LongitudinalData<'_> {
        LongitudinalData::builder(
            &self.outcome,
            self.x_binary.as_ref(),
            self.x_mean.as_ref(),
            &self.subject_ids,
            &self.time,
        )
        .build()
        .expect("data")
    }
}

/// The options with one chain, keeping its draws.
fn one_chain(options: MtpOptions) -> MtpOptions {
    MtpOptions {
        chains: 1,
        keep_draws: true,
        ..options
    }
}

fn draws(model: &MtpModel) -> &MtpPosteriorSamples {
    &model.draws.as_ref().expect("draws")[0]
}

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn sample_standard_normal(rng: &mut Stream) -> f64 {
    let u1 = (1.0_f64 - rng.uniform()).max(f64::MIN_POSITIVE);
    let u2 = rng.uniform();
    (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn centered_time(period: usize, periods: usize) -> f64 {
    if periods <= 1 {
        0.0
    } else {
        2.0f64.mul_add(usize_to_f64(period) / usize_to_f64(periods - 1), -1.0)
    }
}

fn simulate_mtp_panel(
    n_subjects: usize,
    periods: usize,
    alpha: [f64; 2],
    beta: [f64; 2],
    omega_sq: f64,
    seed: u64,
) -> Panel {
    let nrows = n_subjects * periods;
    let omega = omega_sq.sqrt();
    let mut rng = Stream::seeded(seed);

    let mut outcome = Vec::with_capacity(nrows);
    let mut x_binary = Vec::with_capacity(nrows * 2);
    let mut x_mean = Vec::with_capacity(nrows * 2);
    let mut subject_ids = Vec::with_capacity(nrows);
    let mut time = Vec::with_capacity(nrows);

    for subject in 0..n_subjects {
        for period in 0..periods {
            let t = centered_time(period, periods);
            let eta_binary = alpha[1].mul_add(t, alpha[0]);
            let p_positive = logistic(eta_binary);
            let marginal_log_mean = beta[1].mul_add(t, beta[0]);
            let xi = (-0.5 * omega_sq).mul_add(1.0, marginal_log_mean - p_positive.ln());

            let y = if rng.uniform() < p_positive {
                (omega.mul_add(sample_standard_normal(&mut rng), xi)).exp()
            } else {
                0.0
            };

            outcome.push(y);
            x_binary.extend_from_slice(&[1.0, t]);
            x_mean.extend_from_slice(&[1.0, t]);
            subject_ids.push(u32::try_from(subject + 1).unwrap_or(u32::MAX));
            time.push(usize_to_f64(period));
        }
    }

    Panel {
        outcome,
        x_binary: Mat::from_fn(nrows, 2, |row, col| x_binary[2 * row + col]),
        x_mean: Mat::from_fn(nrows, 2, |row, col| x_mean[2 * row + col]),
        subject_ids,
        time,
    }
}

fn zero_rate(input: &Panel) -> f64 {
    let zeros = input.outcome.iter().filter(|&&y| y <= 0.0).count();
    usize_to_f64(zeros) / usize_to_f64(input.outcome.len())
}

fn high_tail_ratio(input: &Panel, threshold: f64) -> f64 {
    let positives = input.outcome.iter().filter(|&&y| y > threshold).count();
    usize_to_f64(positives) / usize_to_f64(input.outcome.len())
}

#[test]
fn mtp_recovers_key_parameters_on_synthetic_data() {
    let true_alpha = [-0.35, 0.8];
    let true_beta = [1.2, 0.55];
    let true_omega_sq = 0.45;

    let input = simulate_mtp_panel(120, 4, true_alpha, true_beta, true_omega_sq, 42);
    let options = one_chain(MtpOptions {
        iterations: 3000,
        burn_in: 1000,
        thin: 6,
        proposal: MtpProposalTuning {
            adaptation_interval: 40,
            ..MtpProposalTuning::default()
        },
        ..MtpOptions::new(Seed(42))
    });

    let model = mtp::fit(&input.data(), &options).expect("fit should run");
    let summary = model
        .posterior_summary
        .expect("posterior summary should be present");

    assert_eq!(summary.alpha.len(), 2);
    assert_eq!(summary.beta.len(), 2);
    for (name, truth, parameter) in [
        ("alpha[1]", true_alpha[1], summary.alpha[1]),
        ("beta[1]", true_beta[1], summary.beta[1]),
    ] {
        let (mean, sd) = (parameter.mean.unwrap(), parameter.std_dev.unwrap());
        let deviation = (mean - truth).abs();
        assert!(
            deviation < 4.0 * sd,
            "{name}: truth {truth} is {deviation} from the posterior mean, posterior sd {sd}"
        );
    }
    let omega_mean = summary.omega_sq.expect("omega summary").mean.unwrap();
    assert!(omega_mean.is_finite());
    assert!(omega_mean > 0.05);
    assert!(omega_mean < 3.0);
}

/// The recovery panel at the crate's default sampler settings (4000 iterations, 1000 of
/// burn-in, every fourth draw kept), run as four chains from the same starting point,
/// holds a split R-hat below 1.05 for alpha[1], beta[1] and omega_sq.
#[test]
fn mtp_split_rhat_below_threshold_on_synthetic_data() {
    let input = simulate_mtp_panel(120, 4, [-0.35, 0.8], [1.2, 0.55], 0.45, 42);
    let model = mtp::fit(&input.data(), &MtpOptions::new(Seed(42))).expect("fit should run");
    let convergence = model.convergence.expect("convergence");
    for (name, rhat) in [
        ("alpha[1]", convergence.alpha_split_rhat[1]),
        ("beta[1]", convergence.beta_split_rhat[1]),
        ("omega_sq", convergence.omega_sq_split_rhat),
    ] {
        let rhat = rhat.unwrap_or_else(|| panic!("{name}: split R-hat is undefined"));
        assert!(rhat < 1.05, "{name}: split R-hat {rhat}");
    }
}

#[test]
fn mtp_stays_stable_under_extreme_zero_inflation() {
    let input = simulate_mtp_panel(140, 3, [-3.5, -0.4], [1.0, 0.2], 0.8, 7);
    let observed_zero_rate = zero_rate(&input);
    assert!(observed_zero_rate > 0.9);

    let options = one_chain(MtpOptions {
        iterations: 320,
        burn_in: 80,
        ..MtpOptions::new(Seed(42))
    });

    let model = mtp::fit(&input.data(), &options).expect("fit should run");
    let posterior = draws(&model);
    assert_ne!(posterior.draws.len(), 0);
    assert!(model.posterior_summary.is_some());

    let ppc = posterior_predictive_summary(posterior, &input.data(), 10).expect("ppc should run");
    assert!(ppc.predicted_zero_rate.mean.unwrap().is_finite());
    assert!(ppc.predicted_zero_rate.mean.unwrap() > 0.75);
    assert!(ppc.predicted_mean.mean.unwrap().is_finite());
}

#[test]
fn mtp_stays_stable_under_highly_skewed_positive_tail() {
    let input = simulate_mtp_panel(120, 4, [-0.2, 0.25], [0.9, 0.35], 2.4, 99);
    assert!(high_tail_ratio(&input, 30.0) > 0.01);

    let options = one_chain(MtpOptions {
        iterations: 320,
        burn_in: 80,
        proposal: MtpProposalTuning {
            log_omega_draw_scale: 0.02,
            ..MtpProposalTuning::default()
        },
        ..MtpOptions::new(Seed(42))
    });

    let model = mtp::fit(&input.data(), &options).expect("fit should run");
    let summary = model
        .posterior_summary
        .as_ref()
        .expect("posterior summary should be present");
    assert!(
        summary
            .omega_sq
            .as_ref()
            .expect("omega summary")
            .mean
            .unwrap()
            > 0.5
    );

    let ppc =
        posterior_predictive_summary(draws(&model), &input.data(), 10).expect("ppc should run");
    assert!(ppc.predicted_mean.mean.unwrap().is_finite());
    assert!(ppc.brier_score.mean.unwrap().is_finite());
}
