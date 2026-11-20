# The log-rank and G-rho tests of survival::survdiff, for the Rust module
# survdiff. Cases: two groups with ties, three groups, strata with groups
# missing from some strata, six groups over strata, several seeds, and the
# one-sample test with an offset. For each case the script records the data
# (time, status, group codes, stratum codes), rho, and from survdiff the
# observed and expected counts (per stratum and group, flattened stratum
# major), the variance matrix, the chi-square statistic, the p-value, the
# eigenvalues of the variance matrix (for the condition number in the error
# bound), and the number of distinct event times counted within strata (the
# m of the error bound).
# Run: Rscript survdiff.R > survdiff.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261121)

make <- function(n, k, strata = 1, shift = 0.4, grid = 0.5, cens = 3) {
  g <- sample(seq_len(k), n, replace = TRUE)
  st <- sample(seq_len(strata), n, replace = TRUE)
  rate <- exp(shift * (g - 1) / max(k - 1, 1) + 0.3 * (st - 1))
  t <- round(rexp(n, rate) / grid) * grid
  cn <- round(runif(n, 0.2, cens) / grid) * grid
  time <- pmax(grid, pmin(t, cn))
  list(time = time, status = as.integer(t <= cn), group = g * 5L + 2L,
       stratum = st * 3L + 1L)
}

cases <- list()
add <- function(d, rho, use_strata, label) {
  g <- factor(d$group, levels = sort(unique(d$group)))
  if (use_strata) {
    f <- Surv(d$time, d$status) ~ g + strata(d$stratum)
  } else {
    f <- Surv(d$time, d$status) ~ g
  }
  fit <- survdiff(f, rho = rho)
  ev <- eigen(fit$var, symmetric = TRUE, only.values = TRUE)$values
  m <- if (use_strata) {
    sum(sapply(split(seq_along(d$time), d$stratum), function(ix)
      length(unique(d$time[ix][d$status[ix] == 1]))))
  } else length(unique(d$time[d$status == 1]))
  cases[[length(cases) + 1]] <<- list(
    label = label, rho = rho, use_strata = use_strata,
    time = d$time, status = d$status, group = d$group,
    stratum = if (use_strata) d$stratum else NULL,
    observed = as.numeric(fit$obs), expected = as.numeric(fit$exp),
    variance = as.numeric(fit$var), chisq = fit$chisq, pvalue = fit$pvalue,
    eigenvalues = ev, n_event_times = m, n_groups = length(unique(d$group)))
}

d2 <- make(60, 2)
for (rho in c(0, 1, 0.37, 2, -0.5)) add(d2, rho, FALSE, paste("two groups, rho", rho))
d3 <- make(150, 3, shift = 0.9)
for (rho in c(0, 1, 0.37)) add(d3, rho, FALSE, paste("three groups, rho", rho))
ds <- make(240, 3, strata = 4, shift = 0.8)
# groups absent from the first stratum and one stratum with few rows
keep <- !(ds$stratum == 4L & ds$group == 7L) & !(ds$stratum == 10L & ds$group == 12L)
ds <- lapply(ds, function(x) x[keep])
for (rho in c(0, 1, 0.37)) add(ds, rho, TRUE, paste("strata, rho", rho))
d6 <- make(400, 6, strata = 3, shift = 1.2, grid = 0.25, cens = 5)
for (rho in c(0, 1)) add(d6, rho, TRUE, paste("six groups strata, rho", rho))
for (s in 1:4) {
  dr <- make(sample(80:300, 1), sample(2:5, 1), strata = sample(1:3, 1),
             shift = runif(1, 0, 1), grid = sample(c(0.1, 0.5, 1), 1))
  add(dr, c(0, 1, 0.5, 0)[s], length(unique(dr$stratum)) > 1, paste("random", s))
}

# One-sample: expected survival from an exponential reference, offset in (0,1]
one <- list()
for (rho in c(0, 1, 0.37, -0.5)) {
  d <- make(70, 2)
  off <- exp(-0.25 * d$time * (1 + (d$group > 8) * 0.5))
  fit <- survdiff(Surv(d$time, d$status) ~ offset(off), rho = rho)
  one[[length(one) + 1]] <- list(
    rho = rho, time = d$time, status = d$status, offset = off,
    observed = fit$obs, expected = fit$exp, variance = fit$var,
    chisq = fit$chisq, pvalue = fit$pvalue)
}

out <- list(versions = list(R = R.version.string,
                            survival = as.character(packageVersion("survival"))),
            cases = cases, one_sample = one)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null"))
