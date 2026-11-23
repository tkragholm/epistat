# The parametric g-formula, checked against gfoRmula::gformula and against
# quantities written out here that share no code with the Rust implementation.
#
# Four data sets, each simulated here:
#
#   main   survival with a competing event and censoring, six periods; a binary
#          covariate L1, a normal covariate L2 and a binary treatment A, with
#          baseline covariates V and W. The models are the forms the data were
#          generated from (lags, lagged cumulative average, cumulative average,
#          an interaction), so the g-formula is consistent.
#   eof    an outcome at the end of four periods, binary and continuous; an
#          absorbing binary covariate L and a normal covariate Z, the variable
#          intervened on.
#   two    two periods with a binary covariate and treatment, for the exact
#          g-formula sum.
#
# What is recorded:
#
#   - the coefficients and residual root mean squares of every model, as
#     gformula fits them (`coeffs`, `rmses`), and its nonparametric risk;
#   - deterministic runs, in which every simulated variable is set by a static
#     intervention and each person is simulated once, so gformula's result
#     does not depend on its random draws: the risks by period (main, with lags,
#     cumulative averages and an interaction; again with baselags and four
#     periods) and the end-of-follow-up means (eof);
#   - runs with simulated covariates at nsimul = 50000, with the Monte Carlo
#     standard error of each risk and of each difference from the natural
#     course, from the simulated persons of sim_data;
#   - a bootstrap of 200 replicates of main (standard errors and percentile
#     limits);
#   - the counterfactual risks of main under four strategies from the
#     data-generating process itself, with 10^6 persons per strategy;
#   - for `two`, the g-formula as the exact sum over the covariate and treatment
#     values of period 1, from glm fits of the same models.
#
# Numbers are written with 17 digits.
# Run: Rscript gformula.R > gformula.json
suppressMessages({
  library(jsonlite)
  library(data.table)
  library(gfoRmula)
})
set.seed(20261124)
expit <- plogis

# ----------------------------------------------------------------------------
# main: the data-generating process. `rule(k, l1, l2, a)` sets the treatment of
# period k from the covariates and the natural value; NULL keeps it natural.
gen_main <- function(n, K, rule = NULL, censor = 0.02, long = TRUE) {
  V <- rbinom(n, 1, 0.4)
  W <- round(rnorm(n), 3)
  rows <- vector("list", K)
  alive <- rep(TRUE, n)
  event_by <- matrix(0, n, K)
  l1p <- l2p <- ap <- rep(0, n)
  sum_a <- sum_l2 <- rep(0, n)
  ever_y <- rep(0, n)
  for (k in 0:(K - 1)) {
    if (k == 0) {
      l1 <- rbinom(n, 1, expit(-0.5 + 0.5 * V))
      l2 <- round(rnorm(n, 0.3 * W, 1), 4)
      a <- rbinom(n, 1, expit(-0.5 + 0.8 * l1 + 0.4 * l2))
    } else {
      l1 <- rbinom(n, 1, expit(-1 + 1.2 * l1p - 0.8 * sum_a / k + 0.3 * V + 0.2 * W))
      l2 <- round(rnorm(n, 0.6 * l2p + 0.5 * l1 - 0.4 * ap + 0.2 * W, 0.8), 4)
      a <- rbinom(n, 1, expit(-1 + 2 * ap + 0.8 * l1 + 0.4 * l2 - 0.2 * V))
    }
    if (!is.null(rule)) a <- rule(k, l1, l2, a)
    sum_a <- sum_a + a
    sum_l2 <- sum_l2 + l2
    d <- rbinom(n, 1, expit(-4 + 0.5 * l1 + 0.3 * W + 0.1 * k))
    y <- rbinom(n, 1, expit(-3 + 0.8 * l1 + 0.4 * sum_l2 / (k + 1) - 0.7 * a +
                              0.4 * V + 0.3 * a * V + 0.1 * k))
    y[d == 1] <- 0
    if (long) {
      rows[[k + 1]] <- data.table(id = which(alive), t0 = k, V = V[alive], W = W[alive],
                                  L1 = l1[alive], L2 = l2[alive], A = as.numeric(a[alive]),
                                  D = d[alive], Y = ifelse(d[alive] == 1, NA, y[alive]))
    }
    ever_y <- ever_y + alive * y
    event_by[, k + 1] <- ever_y
    gone <- (y == 1) | (d == 1) | (rbinom(n, 1, censor) == 1)
    alive <- alive & !gone
    l1p <- l1; l2p <- l2; ap <- a
  }
  if (long) rbindlist(rows)[order(id, t0)] else colMeans(event_by)
}

K <- 6
main <- gen_main(1500, K)
# Treatment intervention functions in gformula's interface.
dyn_l2 <- function(newdf, pool, intvar, intvals, time_name, t) {
  newdf[, (intvar) := as.numeric(newdf$L2 > intvals[[1]])]
}

covparams_main <- list(covmodels = c(
  L1 ~ lag1_L1 + lag_cumavg1_A + V + W,
  L2 ~ lag1_L2 + L1 + lag1_A + W,
  A ~ lag1_A + L1 + L2 + V))
ymodel_main <- Y ~ L1 + cumavg_L2 + A + V + A:V + t0
dmodel_main <- D ~ L1 + W + t0

run_main <- function(nsimul = NA, nsamples = 0, sim = FALSE) {
  gformula(obs_data = copy(main), id = "id", time_points = K, time_name = "t0",
           covnames = c("L1", "L2", "A"), covtypes = c("binary", "normal", "binary"),
           covparams = covparams_main,
           histvars = list(c("L1", "L2", "A"), c("L2"), c("A")),
           histories = c(lagged, cumavg, lagavg), basecovs = c("V", "W"),
           outcome_name = "Y", outcome_type = "survival", ymodel = ymodel_main,
           compevent_name = "D", compevent_model = dmodel_main,
           intervention1.A = list(static, rep(0, K)),
           intervention2.A = list(static, rep(1, K)),
           intervention3.A = list(dyn_l2, 0.5),
           intervention4.A = list(static, rep(1, K), int_times = 2:(K - 1)),
           nsimul = nsimul, nsamples = nsamples, sim_data_b = sim, seed = 1234,
           show_progress = FALSE)
}

# The Monte Carlo standard error of the mean of `col` by period over the
# simulated persons, and of its difference from the natural course.
mc_summary <- function(fit, col, periods) {
  pools <- fit$sim_data
  nat <- pools[[1]]
  lapply(seq_along(pools), function(s) {
    p <- pools[[s]]
    setorder(p, id, t0)
    setorder(nat, id, t0)
    est <- se <- dse <- numeric(length(periods))
    for (j in seq_along(periods)) {
      x <- p[t0 == periods[j]][[col]]
      x0 <- nat[t0 == periods[j]][[col]]
      est[j] <- mean(x)
      se[j] <- sd(x) / sqrt(length(x))
      dse[j] <- sd(x - x0) / sqrt(length(x))
    }
    list(estimate = est, mc_se = se, difference_mc_se = dse)
  })
}

coef_list <- function(fit) {
  lapply(fit$coeffs, function(b) unname(as.numeric(b)))
}
rmse_list <- function(fit) lapply(fit$rmses, function(r) unname(as.numeric(r)))
result_col <- function(fit, col) {
  r <- fit$result
  lapply(sort(unique(r[["Interv."]])), function(s) as.numeric(r[r[["Interv."]] == s][[col]]))
}

fit_main <- run_main(nsimul = 50000, sim = TRUE)
main_mc <- mc_summary(fit_main, "poprisk", 0:(K - 1))
boot_main <- run_main(nsamples = 200)

# Deterministic: treatment is the only time-varying covariate and the
# interventions set it at every period.
exact_main <- function(baselags, time_points, ymodel, dmodel) {
  gformula(obs_data = copy(main), id = "id", time_points = time_points, time_name = "t0",
           covnames = c("A"), covtypes = c("binary"),
           covparams = list(covmodels = c(A ~ lag1_A + V)),
           histvars = list(c("A"), c("A"), c("A")), histories = c(lagged, cumavg, lagavg),
           basecovs = c("V", "W"), outcome_name = "Y", outcome_type = "survival",
           ymodel = ymodel, compevent_name = "D", compevent_model = dmodel,
           intervention1.A = list(static, rep(0, time_points)),
           intervention2.A = list(static, rep(1, time_points)),
           intervention3.A = list(static, rep(c(1, 0), length.out = time_points)),
           baselags = baselags, seed = 99, show_progress = FALSE)
}
exact1 <- exact_main(FALSE, K, Y ~ A + lag1_A + cumavg_A + V + W + A:W + t0,
                     D ~ lag1_A + W + t0)
exact2 <- exact_main(TRUE, 4, Y ~ A + lag1_A + lag2_A + lag_cumavg1_A + V + t0,
                     D ~ lag2_A + W)

# Truth: the risks of the data-generating process under each strategy.
truth_n <- 1000000
truth <- list(
  never = gen_main(truth_n, K, function(k, l1, l2, a) 0 * a, censor = 0, long = FALSE),
  always = gen_main(truth_n, K, function(k, l1, l2, a) 0 * a + 1, censor = 0, long = FALSE),
  dynamic = gen_main(truth_n, K, function(k, l1, l2, a) as.numeric(l2 > 0.5), censor = 0, long = FALSE),
  from_two = gen_main(truth_n, K, function(k, l1, l2, a) if (k >= 2) 0 * a + 1 else a,
                      censor = 0, long = FALSE))

# ----------------------------------------------------------------------------
# eof: an outcome at the end of four periods.
gen_eof <- function(n, K) {
  V <- rbinom(n, 1, 0.5)
  rows <- vector("list", K)
  alive <- rep(TRUE, n)
  lp <- zp <- sum_z <- rep(0, n)
  for (k in 0:(K - 1)) {
    if (k == 0) {
      l <- rbinom(n, 1, 0.3)
      z <- round(rnorm(n, 0.5 * V, 1), 4)
    } else {
      l <- ifelse(lp == 1, 1, rbinom(n, 1, expit(-1.5 + 0.6 * zp)))
      z <- round(rnorm(n, 0.5 * zp + 0.7 * l + 0.3 * V, 1), 4)
    }
    sum_z <- sum_z + z
    y <- yc <- rep(NA_real_, n)
    if (k == K - 1) {
      y <- rbinom(n, 1, expit(-1 + 0.8 * l + 0.5 * z + 0.4 * sum_z / K + 0.3 * V))
      yc <- round(2 + l + 0.5 * z + 0.5 * sum_z / K + V + rnorm(n), 4)
    }
    rows[[k + 1]] <- data.table(id = which(alive), t0 = k, V = V[alive], L = l[alive],
                                Z = z[alive], Y = y[alive], Yc = yc[alive])
    alive <- alive & rbinom(n, 1, 0.05) == 0
    lp <- l; zp <- z
  }
  rbindlist(rows)[order(id, t0)]
}
Ke <- 4
eof <- gen_eof(1000, Ke)
run_eof <- function(type, yname, nsimul = NA, sim = FALSE) {
  gformula(obs_data = copy(eof), id = "id", time_points = Ke, time_name = "t0",
           covnames = c("L", "Z"), covtypes = c("absorbing", "normal"),
           covparams = list(covmodels = c(L ~ lag1_Z, Z ~ lag1_Z + L + V)),
           histvars = list(c("L", "Z"), c("Z")), histories = c(lagged, cumavg),
           basecovs = c("V"), outcome_name = yname, outcome_type = type,
           ymodel = stats::as.formula(paste(yname, "~ L + Z + cumavg_Z + V")),
           intervention1.Z = list(threshold, -0.5, 0.5),
           intervention2.Z = list(static, rep(0, Ke)),
           nsimul = nsimul, sim_data_b = sim, seed = 4321, show_progress = FALSE)
}
eof_mc <- function(fit, col) {
  pools <- fit$sim_data
  nat <- pools[[1]][t0 == Ke - 1]
  setorder(nat, id)
  unname(lapply(pools, function(p) {
    p <- p[t0 == Ke - 1]
    setorder(p, id)
    list(estimate = mean(p[[col]]), mc_se = sd(p[[col]]) / sqrt(nrow(p)),
         difference_mc_se = sd(p[[col]] - nat[[col]]) / sqrt(nrow(p)))
  }))
}
fit_eof_bin <- run_eof("binary_eof", "Y", nsimul = 50000, sim = TRUE)
fit_eof_cont <- run_eof("continuous_eof", "Yc", nsimul = 50000, sim = TRUE)
exact_eof <- function(type, yname) {
  gformula(obs_data = copy(eof), id = "id", time_points = Ke, time_name = "t0",
           covnames = c("Z"), covtypes = c("normal"),
           covparams = list(covmodels = c(Z ~ lag1_Z + V)),
           histvars = list(c("Z"), c("Z")), histories = c(lagged, cumavg),
           basecovs = c("V"), outcome_name = yname, outcome_type = type,
           ymodel = stats::as.formula(paste(yname, "~ Z + lag1_Z + cumavg_Z + V")),
           intervention1.Z = list(static, rep(0, Ke)),
           intervention2.Z = list(static, rep(1, Ke)),
           seed = 7, show_progress = FALSE)
}
exact_eof_bin <- exact_eof("binary_eof", "Y")
exact_eof_cont <- exact_eof("continuous_eof", "Yc")

# ----------------------------------------------------------------------------
# two: two periods, and the g-formula as an exact sum.
gen_two <- function(n) {
  V <- rbinom(n, 1, 0.5)
  l0 <- rbinom(n, 1, 0.4)
  a0 <- rbinom(n, 1, expit(-0.3 + 0.8 * l0))
  d0 <- rbinom(n, 1, expit(-3 + 0.5 * l0))
  y0 <- rbinom(n, 1, expit(-2.5 + 0.9 * l0 - 0.6 * a0 + 0.3 * V))
  y0[d0 == 1] <- 0
  r0 <- data.table(id = 1:n, t0 = 0, V = V, L = l0, A = as.numeric(a0), D = d0,
                   Y = ifelse(d0 == 1, NA, y0))
  stay <- (d0 == 0) & (y0 == 0)
  l1 <- rbinom(n, 1, expit(-0.8 + 1.0 * l0 - 0.9 * a0 + 0.3 * V))
  a1 <- rbinom(n, 1, expit(-0.5 + 1.5 * a0 + 0.8 * l1))
  d1 <- rbinom(n, 1, expit(-3 + 0.5 * l1 + 0.2))
  y1 <- rbinom(n, 1, expit(-2.5 + 0.9 * l1 - 0.6 * a1 + 0.3 * V + 0.2))
  y1[d1 == 1] <- 0
  r1 <- data.table(id = 1:n, t0 = 1, V = V, L = l1, A = as.numeric(a1), D = d1,
                   Y = ifelse(d1 == 1, NA, y1))[stay]
  rbindlist(list(r0, r1))[order(id, t0)]
}
two <- gen_two(2000)
two[, lag1_L := shift(L, fill = 0), by = id]
two[, lag1_A := shift(A, fill = 0), by = id]
fL <- glm(L ~ lag1_L + lag1_A + V, binomial(), data = two[t0 > 0])
fA <- glm(A ~ lag1_A + L, binomial(), data = two[t0 > 0])
fD <- glm(D ~ L + t0, binomial(), data = two)
fY <- glm(Y ~ L + A + V + t0, binomial(), data = two)
pr <- function(f, nd) as.numeric(predict(f, newdata = as.data.frame(nd), type = "response"))
# Exact risks by period for a strategy: `rule(l, a)` gives the treatment from the
# covariate and the natural value, NULL is the natural course.
exact_two <- function(rule) {
  b <- two[t0 == 0]
  a0 <- if (is.null(rule)) b$A else rule(b$L, b$A)
  pd0 <- pr(fD, list(L = b$L, t0 = 0))
  py0 <- pr(fY, list(L = b$L, A = a0, V = b$V, t0 = 0))
  risk0 <- py0 * (1 - pd0)
  s0 <- (1 - py0) * (1 - pd0)
  inc <- 0
  for (l in 0:1) {
    pl1 <- pr(fL, list(lag1_L = b$L, lag1_A = a0, V = b$V))
    pl <- if (l == 1) pl1 else 1 - pl1
    for (a in 0:1) {
      if (is.null(rule)) {
        pa1 <- pr(fA, list(lag1_A = a0, L = rep(l, nrow(b))))
        pa <- if (a == 1) pa1 else 1 - pa1
      } else {
        # Every natural value maps to the rule's value, so each gets its natural probability.
        pa1 <- pr(fA, list(lag1_A = a0, L = rep(l, nrow(b))))
        pa <- if (a == 1) pa1 else 1 - pa1
        a_set <- rule(rep(l, nrow(b)), rep(a, nrow(b)))
      }
      a_used <- if (is.null(rule)) rep(a, nrow(b)) else a_set
      pd1 <- pr(fD, list(L = rep(l, nrow(b)), t0 = 1))
      py1 <- pr(fY, list(L = rep(l, nrow(b)), A = a_used, V = b$V, t0 = 1))
      inc <- inc + pl * pa * py1 * (1 - pd1)
    }
  }
  c(mean(risk0), mean(risk0 + s0 * inc))
}
two_exact <- list(natural = exact_two(NULL),
                  never = exact_two(function(l, a) 0 * a),
                  always = exact_two(function(l, a) 0 * a + 1),
                  treat_if_l = exact_two(function(l, a) as.numeric(l)))
two_coef <- list(L = unname(coef(fL)), A = unname(coef(fA)), D = unname(coef(fD)),
                 Y = unname(coef(fY)))
two[, c("lag1_L", "lag1_A") := NULL]

# ----------------------------------------------------------------------------
cols <- function(d) as.list(d)
out <- list(
  versions = list(r = R.version.string, gfoRmula = as.character(packageVersion("gfoRmula")),
                  data.table = as.character(packageVersion("data.table"))),
  main = list(
    data = cols(main), periods = K,
    coefficients = coef_list(fit_main), rmse = rmse_list(fit_main),
    observed = as.numeric(fit_main$result[fit_main$result[["Interv."]] == 0][["NP Risk"]]),
    simulated = 50000, mc = main_mc,
    bootstrap = list(replicates = 200,
                     risk = result_col(boot_main, "g-form risk"),
                     risk_se = result_col(boot_main, "Risk SE"),
                     risk_lower = result_col(boot_main, "Risk lower 95% CI"),
                     risk_upper = result_col(boot_main, "Risk upper 95% CI"),
                     difference_se = result_col(boot_main, "RD SE")),
    truth = truth, truth_n = truth_n,
    exact1 = list(coefficients = coef_list(exact1),
                  risk = result_col(exact1, "g-form risk"),
                  ratio = result_col(exact1, "Risk ratio"),
                  difference = result_col(exact1, "Risk difference"),
                  observed = as.numeric(exact1$result[exact1$result[["Interv."]] == 0][["NP Risk"]])),
    exact2 = list(coefficients = coef_list(exact2),
                  risk = result_col(exact2, "g-form risk"),
                  observed = as.numeric(exact2$result[exact2$result[["Interv."]] == 0][["NP Risk"]]))),
  eof = list(
    data = cols(eof), periods = Ke, simulated = 50000,
    binary = list(coefficients = coef_list(fit_eof_bin), rmse = rmse_list(fit_eof_bin),
                  observed = fit_eof_bin$result[["NP mean"]][1],
                  mc = eof_mc(fit_eof_bin, "Py")),
    continuous = list(coefficients = coef_list(fit_eof_cont), rmse = rmse_list(fit_eof_cont),
                      observed = fit_eof_cont$result[["NP mean"]][1],
                      mc = eof_mc(fit_eof_cont, "Ey")),
    exact_binary = list(coefficients = coef_list(exact_eof_bin),
                        value = as.numeric(exact_eof_bin$result[["g-form mean"]])),
    exact_continuous = list(coefficients = coef_list(exact_eof_cont),
                            value = as.numeric(exact_eof_cont$result[["g-form mean"]]))),
  two = list(data = cols(two), coefficients = two_coef, exact = two_exact))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
