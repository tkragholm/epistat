# The inverse probability of censoring weighted observed risk of the
# g-formula, against gfoRmula::gformula with a censoring model (its "IP
# weighted risk"), and the models it fits when rows are censored.
#
# Survival over five periods with a competing event D and censoring C: a
# binary covariate L1, a normal covariate L2 and a binary treatment A, with a
# baseline covariate V. Censoring depends on L1 and L2, so the unweighted
# proportions are biased and the weights matter. A censored row has C = 1 and
# no outcome or competing event observed (NA), and is the person's last row.
# The record holds the data, gfoRmula's coefficients of the outcome,
# competing-event and censoring models, and its IP weighted risk of each
# period, also with the weights cut at their 0.95 quantile
# (ipw_cutoff_quantile) and at 3 (ipw_cutoff_value). gfoRmula's simulated
# risks are not used.
# Numbers are written with 17 digits.
#
# Run from the crate directory:
#   Rscript tests/data/gformula_censored.R > tests/data/gformula_censored.json
suppressMessages({
  library(jsonlite)
  library(data.table)
  library(gfoRmula)
})
set.seed(20261126)
n <- 1500
K <- 5
rows <- list()
for (i in seq_len(n)) {
  V <- rbinom(1, 1, 0.4)
  l1p <- 0; l2p <- 0; ap <- 0
  for (k in 0:(K - 1)) {
    l1 <- rbinom(1, 1, plogis(-0.8 + 1.0 * l1p - 0.5 * ap + 0.3 * V))
    l2 <- round(rnorm(1, 0.5 * l2p + 0.4 * l1 - 0.3 * ap, 0.8), 4)
    a <- rbinom(1, 1, plogis(-0.8 + 2 * ap + 0.7 * l1 + 0.3 * l2))
    cens <- rbinom(1, 1, plogis(-2.6 + 0.9 * l1 + 0.5 * l2))
    if (cens == 1) {
      rows[[length(rows) + 1]] <- data.table(id = i, t0 = k, V = V, L1 = l1, L2 = l2,
        A = a, C = 1, D = NA_real_, Y = NA_real_)
      break
    }
    d <- rbinom(1, 1, plogis(-3.5 + 0.4 * l1 + 0.1 * k))
    y <- if (d == 1) NA_real_ else rbinom(1, 1, plogis(-2.6 + 0.8 * l1 + 0.3 * l2 - 0.6 * a + 0.3 * V))
    rows[[length(rows) + 1]] <- data.table(id = i, t0 = k, V = V, L1 = l1, L2 = l2,
      A = a, C = 0, D = d, Y = y)
    l1p <- l1; l2p <- l2; ap <- a
    if (d == 1 || (!is.na(y) && y == 1)) break
  }
}
d <- rbindlist(rows)
fit <- gformula(obs_data = d, id = "id", time_name = "t0",
  covnames = c("L1", "L2", "A"), covtypes = c("binary", "normal", "binary"),
  histories = c(lagged), histvars = list(c("L1", "L2", "A")),
  covparams = list(covmodels = c(L1 ~ lag1_L1 + lag1_A + V,
                                 L2 ~ lag1_L2 + L1 + lag1_A,
                                 A ~ lag1_A + L1 + L2)),
  outcome_name = "Y", outcome_type = "survival",
  ymodel = Y ~ L1 + L2 + A + V,
  compevent_name = "D", compevent_model = D ~ L1 + t0,
  censor_name = "C", censor_model = C ~ L1 + L2,
  basecovs = c("V"), nsimul = 2000, seed = 1, sim_data_b = FALSE)
res <- as.data.frame(fit$result)
nat <- res[res$`Interv.` == 0, ]
# The same with the weights cut at their 0.95 quantile and at 3.
observed_with <- function(...) {
  f <- gformula(obs_data = d, id = "id", time_name = "t0",
    covnames = c("L1", "L2", "A"), covtypes = c("binary", "normal", "binary"),
    histories = c(lagged), histvars = list(c("L1", "L2", "A")),
    covparams = list(covmodels = c(L1 ~ lag1_L1 + lag1_A + V,
                                   L2 ~ lag1_L2 + L1 + lag1_A,
                                   A ~ lag1_A + L1 + L2)),
    outcome_name = "Y", outcome_type = "survival",
    ymodel = Y ~ L1 + L2 + A + V,
    compevent_name = "D", compevent_model = D ~ L1 + t0,
    censor_name = "C", censor_model = C ~ L1 + L2,
    basecovs = c("V"), nsimul = 200, seed = 1, ...)
  r <- as.data.frame(f$result)
  as.numeric(r[r$`Interv.` == 0, "IP weighted risk"])
}
out <- list(
  versions = list(R = as.character(getRversion()),
    gfoRmula = as.character(packageVersion("gfoRmula")),
    data.table = as.character(packageVersion("data.table")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  data = list(id = d$id, t0 = d$t0, V = d$V, L1 = d$L1, L2 = d$L2, A = d$A, C = d$C,
    D = d$D, Y = d$Y),
  outcome_coef = as.numeric(fit$coeffs$Y),
  competing_coef = as.numeric(fit$coeffs$D),
  censoring_coef = as.numeric(fit$coeffs[[length(fit$coeffs)]]),
  ip_weights = as.numeric(fit$IP_weights),
  observed = as.numeric(nat$`IP weighted risk`),
  observed_quantile_95 = observed_with(ipw_cutoff_quantile = 0.95),
  observed_value_3 = observed_with(ipw_cutoff_value = 3))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, na = "null"))
