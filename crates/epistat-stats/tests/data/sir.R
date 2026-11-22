# Reference values for epistat_stats::sir.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/sir.R
#
# Exact limits and the exact p-value come from stats::poisson.test over a grid
# of observed and expected counts. The profile limits are the roots of the
# deviance equation found with uniroot at a tolerance of 1e-14, since
# popEpi::sir's profile interval interpolates a profile on a grid and rounds to
# four decimals; popEpi's own values are written as well, with the rounding. The
# Wald limits are exp(log(O/E) -/+ qnorm(1 - a) / sqrt(O)). Strata for the tests
# of homogeneity and trend come from a simulated cohort in 24 cells (6 age bands,
# 2 periods, 2 sexes) summed to three groups: the likelihood ratio test of
# homogeneity is popEpi::sir's lrt.test, and the test of trend is the deviance
# test of glm(O ~ score + offset(log(E)), poisson) against the null model, since
# popEpi's test.type = "trend" does not use its covariates and returns the test
# of homogeneity. Values are written with 17 significant digits.
suppressMessages({
  library(jsonlite)
  library(popEpi)
})
set.seed(20261122)

grid <- list()
for (o in c(0, 1, 2, 3, 5, 10, 25, 60, 120, 200)) {
  for (e in c(0.01, 0.5, 2, 7.5, 40, 130, 500)) {
    for (level in c(0.9, 0.95, 0.99)) {
      pt <- poisson.test(o, e, conf.level = level)   # the interval is for o / e
      z <- qnorm(1 - (1 - level) / 2)
      dev <- function(mu) if (o == 0) 2 * mu else 2 * (o * log(o / mu) - o + mu)
      prof <- if (o == 0) c(0, z^2 / 2) else c(
        uniroot(function(mu) dev(mu) - z^2, c(1e-12 * o, o), tol = 1e-15)$root,
        uniroot(function(mu) dev(mu) - z^2, c(o, o + 10 * z * sqrt(o) + 10 * z^2), tol = 1e-15)$root)
      grid[[length(grid) + 1]] <- list(o = o, e = e, level = level,
        exact_p = pt$p.value, exact_lower = pt$conf.int[1], exact_upper = pt$conf.int[2],
        profile_lower = prof[1] / e, profile_upper = prof[2] / e,
        wald_lower = if (o > 0) exp(log(o / e) - z / sqrt(o)) else NULL,
        wald_upper = if (o > 0) exp(log(o / e) + z / sqrt(o)) else NULL)
    }
  }
}

# A cohort in cells of age band, period and sex, and a reference rate for each.
cells <- expand.grid(age = 1:6, period = 1:2, sex = 1:2)
cells$pyrs <- round(runif(nrow(cells), 300, 4000))
cells$rate <- 0.002 * 1.5^(cells$age - 1) * ifelse(cells$sex == 1, 1.3, 1) * ifelse(cells$period == 2, 0.9, 1)
cells$group <- (cells$age + 1) %/% 2          # 1, 2, 3
cells$rr <- c(1, 1.4, 2.2)[cells$group]
cells$obs <- rpois(nrow(cells), cells$pyrs * cells$rate * cells$rr)
cells$obs[cells$group == 1 & cells$period == 1 & cells$sex == 1] <- 0
ref <- cells[, c("age", "period", "sex", "rate")]
cells$expected <- cells$pyrs * cells$rate
groups <- do.call(rbind, lapply(1:3, function(g) {
  d <- cells[cells$group == g, ]
  data.frame(group = g, observed = sum(d$obs), expected = sum(d$expected), pyrs = sum(d$pyrs))
}))
pop <- list()
for (ct in c("univariate", "wald", "profile")) {
  r <- suppressMessages(sir(coh.data = cells, coh.obs = obs, coh.pyrs = pyrs, ref.data = ref, ref.rate = rate,
                            adjust = list(age, period, sex), print = group, conf.type = ct))
  pop[[ct]] <- list(sir = r$sir, lower = r$sir.lo, upper = r$sir.hi)
  lrt <- attr(r, "sir.meta")$lrt.test
}
trend_fit <- function(o, e, s) {
  full <- glm(o ~ s, offset = log(e), family = poisson)
  null <- glm(o ~ 1, offset = log(e), family = poisson)
  a <- anova(null, full, test = "Chisq")
  list(statistic = a$Deviance[2], p = a[["Pr(>Chi)"]][2])
}
tr <- trend_fit(groups$observed, groups$expected, c(1, 2, 3))
tr2 <- trend_fit(groups$observed[c(1, 3)], groups$expected[c(1, 3)], c(1, 4))
hom <- list(statistic = {
  pooled <- sum(groups$observed) / sum(groups$expected)
  fitted <- groups$expected * pooled
  2 * sum(ifelse(groups$observed > 0, groups$observed * log(groups$observed / fitted), 0) - (groups$observed - fitted))
}, p = lrt, df = nrow(groups) - 1)
# A larger set of strata with skewed counts and scores 0..7.
big_e <- c(4.2, 8.1, 15, 22, 31, 40, 52, 66)
big_o <- rpois(8, big_e * exp(0.15 * (0:7)))
tb <- trend_fit(big_o, big_e, 0:7)
pooled <- sum(big_o) / sum(big_e)
hom_big <- 2 * sum(ifelse(big_o > 0, big_o * log(big_o / (big_e * pooled)), 0) - (big_o - big_e * pooled))
pop_hom_big <- anova(glm(big_o ~ 1, offset = log(big_e), family = poisson),
                     glm(big_o ~ factor(0:7), offset = log(big_e), family = poisson), test = "Chisq")

write(toJSON(list(versions = list(r = R.version.string, popEpi = as.character(packageVersion("popEpi"))),
                  grid = grid,
                  strata = list(observed = I(groups$observed), expected = I(groups$expected), pyrs = I(groups$pyrs),
                                popepi = pop, homogeneity = hom, trend = tr, trend_two = tr2,
                                score = I(c(1, 2, 3))),
                  big = list(observed = I(big_o), expected = I(big_e), score = I(0:7), trend = tb,
                             homogeneity = list(statistic = hom_big, p = pop_hom_big[["Pr(>Chi)"]][2], df = 7))),
             auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
      "tests/data/sir.json")
