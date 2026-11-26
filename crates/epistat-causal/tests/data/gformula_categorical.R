# A categorical covariate in the g-formula, against the g-formula written out
# here as an exact sum, with the models fitted by glm and nnet::multinom.
#
# Survival over four periods with a binary covariate L1, a three-class
# covariate L3 (classes 0, 1, 2) that depends on its own lag and on L1, and a
# binary treatment A that depends on L3. The models are fitted on the data:
# L1, L3 and A on the rows of periods 1 to 3 (L3 by nnet::multinom with
# reltol = 1e-14, since its default stopping rule leaves the coefficients about
# 1e-4 from the maximum), the outcome on every row. The risk by the end of
# period 1 under the natural course and under "always treat" is then the exact
# sum over the values of L1, L3 and A in period 1, averaged over the persons'
# period-0 rows:
#   risk_1 = mean_i [ p_0 + (1 - p_0) sum_{l1, l3, a} P(l1) P(l3 | l1) P(a | l1, l3) p_1 ],
# with p_0 the outcome model on the period-0 row (A set to 1 under "always
# treat") and p_1 on the period-1 values.
#
# gfoRmula::gformula 1.1.1 with covtype "categorical" is not the reference:
# on these data its simulated L3 keeps the period-0 distribution in every
# period (the class proportions of its sim_data do not change with t0), while
# the observed proportions rise from 0.31 and 0.14 to 0.39 and 0.19.
# Numbers are written with 17 digits.
#
# Run from the crate directory:
#   Rscript tests/data/gformula_categorical.R > tests/data/gformula_categorical.json
suppressMessages({
  library(jsonlite)
  library(data.table)
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
dl <- copy(d)
dl[, `:=`(lag1_L1 = shift(L1, fill = 0), lag1_L3 = shift(L3, fill = 0),
  lag1_A = shift(A, fill = 0)), by = id]
later <- dl[t0 >= 1]
later[, lag1_L3f := factor(lag1_L3, levels = 0:2)]
m_l1 <- glm(L1 ~ lag1_L1 + lag1_A + V, family = binomial, data = later)
tight <- multinom(factor(L3, levels = 0:2) ~ lag1_L3f + L1, data = later, reltol = 1e-14,
  maxit = 10000, trace = FALSE)
m_a <- glm(A ~ lag1_A + L1 + factor(L3, levels = 0:2), family = binomial, data = later)
m_y <- glm(Y ~ L1 + factor(L3, levels = 0:2) + A + V, family = binomial, data = dl)
b1 <- coef(m_l1); bl3 <- coef(tight); ba <- coef(m_a); by <- coef(m_y)
ind <- function(x) c(x == 1, x == 2)
py <- function(l1, l3, a, v) plogis(sum(by * c(1, l1, ind(l3), a, v)))
exact <- function(treat) {
  base <- d[t0 == 0]
  mean(sapply(seq_len(nrow(base)), function(i) {
    r <- base[i]
    a0 <- if (treat) 1 else r$A
    p0 <- py(r$L1, r$L3, a0, r$V)
    p_l1 <- plogis(sum(b1 * c(1, r$L1, a0, r$V)))
    total <- 0
    for (l1 in 0:1) {
      pl1 <- if (l1 == 1) p_l1 else 1 - p_l1
      e <- exp(drop(bl3 %*% c(1, ind(r$L3), l1)))
      pl3 <- c(1, e) / (1 + sum(e))
      for (l3 in 0:2) {
        p_a <- plogis(sum(ba * c(1, a0, l1, ind(l3))))
        for (a in 0:1) {
          pa <- if (treat) as.numeric(a == 1) else if (a == 1) p_a else 1 - p_a
          total <- total + pl1 * pl3[l3 + 1] * pa * py(l1, l3, a, r$V)
        }
      }
    }
    p0 + (1 - p0) * total
  }))
}
out <- list(
  versions = list(R = as.character(getRversion()),
    nnet = as.character(packageVersion("nnet")),
    data.table = as.character(packageVersion("data.table")),
    jsonlite = as.character(packageVersion("jsonlite"))),
  data = list(id = d$id, t0 = d$t0, V = d$V, L1 = d$L1, L3 = d$L3, A = d$A, Y = d$Y),
  l3_coef = as.numeric(t(coef(tight))),
  natural = exact(FALSE),
  treat = exact(TRUE))
cat(toJSON(out, digits = I(17), auto_unbox = TRUE))
