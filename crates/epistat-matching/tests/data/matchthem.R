# Reference matches and pooled effects after multiple imputation, from
# MatchThem and mice.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchthem.R
#
# MatchIt's `lalonde` with values removed completely at random from age, educ,
# re74, re75 and re78 (fixed seed), imputed five times by mice() (Bayesian linear
# regression, which draws continuous values so that no two imputed rows tie;
# fixed seed). The data sets and the treatment, outcome and
# covariates of each are written to the file so that the Rust test matches
# the same sets. For each approach, "within" and "across", matchthem() runs
# nearest neighbour matching 1:1 on the logistic propensity score of age,
# educ, married, nodegree, re74 and re75. The file records, per data set:
#   pairs      the control row matched to each treated row (match.matrix)
#   distance   the score the matching used (averaged for "across")
#   estimate, se   the coefficient of `treat` in lm(re78 ~ treat) on the
#              matched rows with the matching weights, and its standard error
#              from sandwich::vcovCL() clustered by subclass
# and, per approach, mice::pool.scalar() of those estimates and variances,
# with the complete-data sample size n = matched rows and k = 2 parameters,
# and the pooled coefficient of MatchThem's own with() and pool() on the same
# fits, whose model-based standard errors differ from the cluster-robust ones
# and so are recorded but not compared.

suppressMessages({
  library(MatchIt)
  library(MatchThem)
  library(mice)
  library(sandwich)
  library(jsonlite)
})

data(lalonde)
d <- lalonde[, c("treat", "age", "educ", "married", "nodegree", "re74", "re75", "re78")]
# lalonde has rows with identical covariates, hence identical scores, between
# which MatchIt does not break ties by row order (see r_nearest.rs). A
# perturbation of age by 1e-5 times the row number separates them.
d$age <- d$age + 1e-5 * seq_len(nrow(d))
set.seed(20261121)
for (v in c("age", "educ", "re74", "re75", "re78")) {
  d[[v]][runif(nrow(d)) < 0.12] <- NA
}
imp <- mice(d, m = 5, method = "norm", seed = 20261122, printFlag = FALSE)
f <- treat ~ age + educ + married + nodegree + re74 + re75

sets <- lapply(seq_len(imp$m), function(i) as.list(complete(imp, i)))

run <- function(approach) {
  mt <- matchthem(f, imp, approach = approach, method = "nearest", ratio = 1)
  per_set <- lapply(seq_len(imp$m), function(i) {
    m <- mt$models[[i]]
    mm <- m$match.matrix
    md <- match_data(m, data = complete(imp, i))
    fit <- lm(re78 ~ treat, data = md, weights = weights)
    v <- vcovCL(fit, cluster = md$subclass)
    list(treated = match(rownames(mm), rownames(d)),
         control = match(as.vector(mm), rownames(d)),
         distance = unname(m$distance),
         rows = nrow(md),
         estimate = unname(coef(fit)["treat"]),
         variance = unname(v["treat", "treat"]))
  })
  q <- sapply(per_set, function(s) s$estimate)
  u <- sapply(per_set, function(s) s$variance)
  n <- per_set[[1]]$rows
  stopifnot(all(sapply(per_set, function(s) s$rows) == n))
  p <- pool.scalar(q, u, n = n, k = 2)
  fits <- with(mt, lm(re78 ~ treat, weights = weights))
  pooled_mt <- summary(pool(fits))
  list(approach = approach, sets = per_set,
       pooled = list(qbar = p$qbar, ubar = p$ubar, b = p$b, t = p$t, df = p$df,
                     fmi = p$fmi, r = p$r),
       matchthem_estimate = pooled_mt$estimate[pooled_mt$term == "treat"],
       matchthem_model_se = pooled_mt$std.error[pooled_mt$term == "treat"])
}

out <- list(
  versions = list(r = R.version.string, MatchThem = as.character(packageVersion("MatchThem")),
                  mice = as.character(packageVersion("mice")),
                  MatchIt = as.character(packageVersion("MatchIt"))),
  data = sets,
  approaches = list(run("within"), run("across"))
)
write(toJSON(out, auto_unbox = TRUE, digits = NA), "tests/data/matchthem.json")
