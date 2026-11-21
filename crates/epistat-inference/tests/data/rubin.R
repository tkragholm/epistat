# Reference pooled estimates from mice::pool, which applies Rubin's rules
# with the Barnard-Rubin degrees of freedom.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/rubin.R
#
# The script imputes mice::nhanes and mice::nhanes2 with a fixed seed, fits a
# linear model to every imputed data set and records, for every coefficient,
# the per-data-set estimates and variances and the pooled estimate, between
# and within variances, total variance, degrees of freedom, relative increase
# in variance, lambda and fraction of missing information that mice::pool()
# returns. Two further cases call mice::pool.scalar() with a complete-data
# degrees of freedom of infinity and with identical estimates, where the
# between-imputation variance is zero.

suppressMessages({
  library(mice)
  library(jsonlite)
})

cases <- list()

add_pool <- function(name, imp, formula) {
  analyses <- lapply(seq_len(imp$m), function(i) lm(formula, data = complete(imp, i)))
  fits <- as.mira(analyses)
  pooled <- pool(fits)
  pt <- pooled$pooled
  per_set <- lapply(analyses, function(f) {
    s <- summary(f)$coefficients
    list(estimate = unname(s[, "Estimate"]), variance = unname(s[, "Std. Error"]^2))
  })
  k <- nrow(pt)
  cases[[length(cases) + 1]] <<- list(
    name = name,
    terms = pt$term,
    m = imp$m,
    estimates = lapply(seq_len(k), function(j) sapply(per_set, function(s) s$estimate[j])),
    variances = lapply(seq_len(k), function(j) sapply(per_set, function(s) s$variance[j])),
    dfcom = pt$dfcom[1],
    pooled = list(qbar = pt$estimate, ubar = pt$ubar, b = pt$b, t = pt$t,
                  df = pt$df, riv = pt$riv, lambda = pt$lambda, fmi = pt$fmi)
  )
}

imp1 <- mice(nhanes, m = 5, seed = 20261121, printFlag = FALSE)
add_pool("nhanes_m5", imp1, bmi ~ age + hyp + chl)
imp2 <- mice(nhanes2, m = 20, seed = 20261122, printFlag = FALSE)
add_pool("nhanes2_m20", imp2, chl ~ age + bmi + hyp)

scalar <- function(name, q, u, n) {
  p <- pool.scalar(q, u, n = n)
  list(name = name, q = q, u = u, n = if (is.infinite(n)) NULL else n,
       qbar = p$qbar, ubar = p$ubar, b = p$b, t = p$t, df = p$df,
       r = p$r, fmi = p$fmi)
}
scalars <- list(
  scalar("infinite_dfcom", c(1.1, 0.9, 1.4, 1.0), c(0.20, 0.25, 0.22, 0.21), Inf),
  scalar("no_between_variance", rep(2.5, 4), c(0.2, 0.2, 0.2, 0.2), 60)
)

out <- list(versions = list(r = R.version.string, mice = as.character(packageVersion("mice"))),
            cases = cases, scalars = scalars)
write(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), "tests/data/rubin.json")
