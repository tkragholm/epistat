# Reference for a caliper in standard deviations on a covariate, with
# common-support trimming, from MatchIt's `caliper` and `std.caliper`.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_std_caliper_field.R
#
# The data are simulated with a fixed seed. The distance is a numeric vector,
# the fitted probabilities of a logistic model on `x`, so the matching does not
# depend on a model fit. `z` is related to `x`, so the units that `discard =
# "both"` removes are in the tails of `z` and the standard deviation of `z`
# differs before and after trimming.
#
# For each caliper width and each of discard = "none" and "both" the script
# records the pairs of nearest-neighbour matching (ratio 1, m.order = "data",
# no replacement) with `caliper = c(z = width)` and `std.caliper = TRUE`. It
# also records the pairs that an unstandardised caliper of width times the
# standard deviation over all units, and over the units left after trimming,
# gives, and checks which of the two the standardised caliper equals.
#
# The same is done with `reestimate = TRUE`: the distance is then the logistic
# propensity score of `treat ~ x`, fitted again on the units that discard
# leaves. The units are found from the first fit, so the standard deviation of
# `z` is taken over the units the first fit keeps, and the script checks that
# the standardised result equals an unstandardised caliper of width times it.

suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

set.seed(20261121)
n <- 400
x <- rnorm(n)
z <- rnorm(n) + 0.5 * x
treat <- rbinom(n, 1, plogis(-0.4 + 1.6 * x))
df <- data.frame(treat = treat, x = x, z = z)
distance <- unname(fitted(glm(treat ~ x, data = df, family = binomial)))

pairs_of <- function(m) {
  mt <- m$match.matrix
  ids <- as.integer(rownames(mt))
  ctl <- suppressWarnings(as.integer(mt[, 1]))
  keep <- !is.na(ctl)
  list(treated = ids[keep], control = ctl[keep])
}

run <- function(policy, caliper, std) {
  suppressWarnings(matchit(treat ~ z, data = df, method = "nearest",
                           distance = distance, discard = policy,
                           m.order = "data", caliper = c(z = caliper),
                           std.caliper = std))
}

widths <- c(0.02, 0.05, 0.1, 0.2, 0.5)
cases <- list()
for (policy in c("none", "both")) {
  discarded <- unname(suppressWarnings(
    matchit(treat ~ z, data = df, method = NULL, distance = distance,
            discard = policy))$discarded)
  sd_all <- sd(df$z)
  sd_kept <- sd(df$z[!discarded])
  for (w in widths) {
    std <- pairs_of(run(policy, w, TRUE))
    all <- pairs_of(run(policy, w * sd_all, FALSE))
    kept <- pairs_of(run(policy, w * sd_kept, FALSE))
    cases[[length(cases) + 1]] <- list(
      discard = policy, width = w, sd_all = sd_all, sd_kept = sd_kept,
      n_discarded = sum(discarded),
      std_equals_sd_over_kept = identical(std, kept),
      std_equals_sd_over_all = identical(std, all),
      kept_and_all_differ = !identical(all, kept),
      treated = std$treated, control = std$control)
  }
}

run_re <- function(policy, caliper, std) {
  suppressWarnings(matchit(treat ~ x, data = df, method = "nearest",
                           distance = "glm", discard = policy,
                           reestimate = TRUE, m.order = "data",
                           caliper = c(z = caliper), std.caliper = std))
}
re_cases <- list()
for (policy in c("both", "treated")) {
  discarded <- unname(suppressWarnings(
    matchit(treat ~ x, data = df, method = NULL, distance = "glm",
            discard = policy))$discarded)
  sd_all <- sd(df$z)
  sd_kept <- sd(df$z[!discarded])
  for (w in widths) {
    std <- pairs_of(run_re(policy, w, TRUE))
    all <- pairs_of(run_re(policy, w * sd_all, FALSE))
    kept <- pairs_of(run_re(policy, w * sd_kept, FALSE))
    re_cases[[length(re_cases) + 1]] <- list(
      discard = policy, width = w, sd_all = sd_all, sd_kept = sd_kept,
      n_discarded = sum(discarded),
      std_equals_sd_over_kept = identical(std, kept),
      std_equals_sd_over_all = identical(std, all),
      kept_and_all_differ = !identical(all, kept),
      treated = std$treated, control = std$control)
  }
}
stopifnot(all(vapply(re_cases, function(c) c$std_equals_sd_over_kept, TRUE)))
stopifnot(any(vapply(re_cases, function(c) c$kept_and_all_differ, TRUE)))

stopifnot(all(vapply(cases, function(c) c$discard == "none" || c$std_equals_sd_over_kept, TRUE)))
stopifnot(any(vapply(cases, function(c) c$discard == "both" && c$kept_and_all_differ, TRUE)))

out <- list(
  versions = list(r = R.version.string, matchit = as.character(packageVersion("MatchIt"))),
  treat = treat, x = x, z = z, distance = distance, cases = cases,
  reestimate_cases = re_cases)
write_json(out, "tests/data/matchit_std_caliper_field.json", digits = NA,
           auto_unbox = TRUE)
