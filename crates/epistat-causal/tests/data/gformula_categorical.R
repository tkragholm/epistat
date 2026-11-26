# A categorical covariate in the g-formula, against gfoRmula::gformula with
# covtype "categorical" (fitted by nnet::multinom).
#
# Survival over four periods with a binary covariate L1, a three-class
# covariate L3 (classes 0, 1, 2) that depends on its own lag and on L1, and a
# binary treatment A that depends on L3. The record holds the data; the
# coefficients of the model of L3, refitted by nnet::multinom with reltol =
# 1e-14 on the rows gfoRmula fits it on (periods 1 to 3), since its default
# stopping rule leaves the coefficients about 1e-4 from the maximum; and the
# risks by period of gfoRmula under the natural course and under "always
# treat" with nsimul = 40000, with the Monte Carlo standard error of each from
# the simulated persons (sim_data).
# Numbers are written with 17 digits.
#
# Run from the crate directory:
#   Rscript tests/data/gformula_categorical.R > tests/data/gformula_categorical.json
suppressMessages({
  library(jsonlite)
  library(data.table)
  library(gfoRmula)
  library(nnet)
})
set.seed(20261126)
n <- 2500
K <- 4
rows <- list()
softmax <- function(e) exp(c(0, e)) / sum(exp(c(0, e)))
for (i in seq_len(n)) {
  V <- rbinom(1, 1, 0.5)
  l1p <- 0; l3p <- 0; ap <- 0
  for (k in 0:(K - 1)) {
    l1 <- rbinom(1, 1, plogis(-0.6 + 0.9 * l1p - 0.5 * ap + 0.3 * V))
    pr <- softmax(c(-0.8 + 1.1 * (l3p == 1) + 0.4 * (l3p == 2) + 0.6 * l1,
                    -1.5 + 0.5 * (l3p == 1) + 1.6 * (l3p == 2) + 0.3 * l1))
    l3 <- sample(0:2, 1, prob = pr)
    a <- rbinom(1, 1, plogis(-0.7 + 1.8 * ap + 0.5 * l1 + 0.4 * (l3 == 1) + 0.9 * (l3 == 2)))
    y <- rbinom(1, 1, plogis(-2.6 + 0.6 * l1 + 0.4 * (l3 == 1) + 0.9 * (l3 == 2) - 0.6 * a + 0.2 * V))
    rows[[length(rows) + 1]] <- data.table(id = i, t0 = k, V = V, L1 = l1, L3 = l3, A = a, Y = y)
    l1p <- l1; l3p <- l3; ap <- a
    if (y == 1) break
  }
}
d <- rbindlist(rows)
dg <- copy(d)
dg[, L3 := factor(L3, levels = 0:2)]
fit <- gformula(obs_data = dg, id = "id", time_name = "t0",
  covnames = c("L1", "L3", "A"), covtypes = c("binary", "categorical", "binary"),
  histories = c(lagged), histvars = list(c("L1", "L3", "A")),
  covparams = list(covmodels = c(L1 ~ lag1_L1 + lag1_A + V,
                                 L3 ~ lag1_L3 + L1,
                                 A ~ lag1_A + L1 + L3)),
  outcome_name = "Y", outcome_type = "survival",
  ymodel = Y ~ L1 + L3 + A + V,
  intvars = list("A"), interventions = list(list(c(static, rep(1, K)))),
  basecovs = c("V"), nsimul = 40000, seed = 7, sim_data_b = TRUE)
res <- as.data.frame(fit$result)
mc_se <- function(s) {
  s <- as.data.table(s)
  sapply(0:(K - 1), function(k) {
    v <- s[t0 == k, poprisk]
    sd(v) / sqrt(length(v))
  })
}
# The model of L3 on the rows of periods 1 to K - 1, with lag1_L3 as a factor
# whose value before period 0 is the reference class.
dl <- copy(d)
dl[, lag1_L3 := shift(L3, fill = 0), by = id]
dl <- dl[t0 >= 1]
dl[, lag1_L3 := factor(lag1_L3, levels = 0:2)]
tight <- multinom(factor(L3, levels = 0:2) ~ lag1_L3 + L1, data = dl, reltol = 1e-14,
  maxit = 10000, trace = FALSE)
out <- list(
  versions = list(R = as.character(getRversion()),
    gfoRmula = as.character(packageVersion("gfoRmula")),
    nnet = as.character(packageVersion("nnet")),
    data.table = as.character(packageVersion("data.table")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  data = list(id = d$id, t0 = d$t0, V = d$V, L1 = d$L1, L3 = d$L3, A = d$A, Y = d$Y),
  l3_coef = as.numeric(t(coef(tight))),
  natural = as.numeric(res[res$`Interv.` == 0, "g-form risk"]),
  natural_se = mc_se(fit$sim_data[[1]]),
  treat = as.numeric(res[res$`Interv.` == 1, "g-form risk"]),
  treat_se = mc_se(fit$sim_data[[2]]))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE))
