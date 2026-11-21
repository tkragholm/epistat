# Reference replicates of the bootstrap of matched sets, from R.
#
# Run from the crate directory after tests/data/effects.R and the ignored Rust
# test `write_bootstrap_draws` in tests/r_effects_bootstrap.rs:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/effects_boot.R
#
# `effects_boot_draws.json` holds the pair numbers of each replicate, 0-based
# into the sorted distinct pairs, as drawn by the Rust code. For the 1:1 match
# without replacement and the 2:1 match with replacement of effects.json, each
# replicate takes every row of each drawn pair in row order, refits the
# outcome model (lm for re78, glm for re78 > 0, treatment with age and educ
# and their interactions with the treatment, matching weights as weights) and
# averages the predictions with the treatment set to 1 and to 0 over the
# treated rows of the replicate with their weights. The statistics are the
# difference, and for the logistic model the ratio and the odds ratio of the
# averaged predictions. Intervals at the 95 percent level come from boot.ci()
# on a boot object built from the replicates, and from quantile() type 7.

suppressMessages({
  library(boot)
  library(jsonlite)
})

draws <- fromJSON("tests/data/effects_boot_draws.json", simplifyVector = FALSE)
matches <- fromJSON("tests/data/effects.json", simplifyVector = TRUE)
lal <- matches$lalonde

stat_fn <- function(rows_df, outcome) {
  d <- rows_df
  rhs <- "treat * (age + educ)"
  fit <- suppressWarnings(if (outcome == "continuous")
    lm(as.formula(paste("y ~", rhs)), data = d, weights = w)
  else glm(as.formula(paste("b ~", rhs)), data = d, weights = w,
           family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100)))
  nd <- d[d$treat == 1, ]
  wt <- nd$w
  nd1 <- transform(nd, treat = 1)
  nd0 <- transform(nd, treat = 0)
  m1 <- sum(wt * predict(fit, nd1, type = "response")) / sum(wt)
  m0 <- sum(wt * predict(fit, nd0, type = "response")) / sum(wt)
  if (outcome == "continuous") return(c(difference = m1 - m0))
  c(difference = m1 - m0, ratio = m1 / m0,
    odds_ratio = (m1 / (1 - m1)) / (m0 / (1 - m0)))
}

summarise <- function(t0, t, B) {
  ci <- structure(list(t0 = t0, t = matrix(t, ncol = 1), R = B,
                       data = NULL, seed = NULL, statistic = NULL,
                       sim = "ordinary", call = quote(boot()), stype = "i",
                       strata = rep(1, 2), weights = c(0.5, 0.5)),
                  class = "boot")
  b <- boot.ci(ci, conf = 0.95, type = c("norm", "perc"))
  list(t0 = t0, t = t, sd = sd(t),
       quantile7 = unname(quantile(t, c(0.025, 0.975), type = 7)),
       boot_norm = b$normal[2:3], boot_perc = b$percent[4:5])
}

runs <- list()
for (cs in matches$cases[matches$cases$name %in% c("nearest", "replace_ratio2"), "name"]) {
  case <- matches$cases[matches$cases$name == cs, ]
  rows <- case$rows[[1]]
  d <- data.frame(treat = lal$treat[rows], y = lal$re78[rows],
                  b = as.numeric(lal$re78[rows] > 0), age = lal$age[rows],
                  educ = lal$educ[rows], w = case$weight[[1]],
                  pair = case$subclass[[1]])
  pairs <- sort(unique(d$pair))
  members <- lapply(pairs, function(p) which(d$pair == p))
  for (outcome in c("continuous", "binary")) {
    t0 <- stat_fn(d, outcome)
    ts <- do.call(rbind, lapply(draws$resamples, function(draw) {
      idx <- unlist(members[unlist(draw) + 1])
      stat_fn(d[idx, ], outcome)
    }))
    run <- list(case = cs, outcome = outcome)
    for (nm in names(t0)) {
      run[[nm]] <- summarise(unname(t0[nm]), ts[, nm], length(draws$resamples))
    }
    runs[[length(runs) + 1]] <- run
  }
}

out <- list(
  versions = list(r = R.version.string, boot = as.character(packageVersion("boot")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  runs = runs)
write_json(out, "tests/data/effects_boot.json", digits = I(17), auto_unbox = TRUE,
           na = "null")
