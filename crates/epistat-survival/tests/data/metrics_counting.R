# survival::rttright and survival::brier on counting-process data with an id
# and one common start time. Subjects are split into one to three rows at
# random cut points, with the intermediate rows censored; times are rounded so
# that deaths and censorings tie; case weights and strata are constant within a
# subject. The same subjects in one row each, without a start column, give the
# reference that does not depend on how the follow-up is cut.
#
# survival 3.8-12 gives wrong weights with reporting times for counting-process
# data: its branch selects the last row of each subject with `Y[, 2] > 0`, the
# stop time, where the status is meant. The output below keeps R's numbers so
# that the test can assert the defect. brier counts every row of an id as a
# subject, which the output shows by the score of the split data.
# Run: Rscript metrics_counting.R > metrics_counting.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261124)
n <- 140
s0 <- 0.5
x <- rnorm(n)
t1 <- rexp(n, 0.25 * exp(0.4 * x))
t2 <- rexp(n, 0.1)
cc <- rexp(n, 0.15)
stop_end <- s0 + round(pmin(t1, t2, cc), 1)
stop_end[stop_end <= s0] <- s0 + 0.1
cause <- ifelse(pmin(t1, t2) > cc, 0L, ifelse(t1 <= t2, 1L, 2L))
grp <- sample(1:2, n, replace = TRUE)
wt <- sample(c(1, 2, 3), n, replace = TRUE)
subj <- data.frame(id = 1:n, stop = stop_end, cause = cause, grp = grp, w = wt, x = x)

# Cut each subject at 0 to 2 interior points, rounded as the times are.
rows <- list()
for (i in 1:n) {
  k <- sample(0:2, 1)
  e <- stop_end[i]
  cuts <- if (k > 0) sort(unique(round(runif(k, s0, e), 1))) else numeric(0)
  cuts <- cuts[cuts > s0 & cuts < e]
  edges <- c(s0, cuts, e)
  m <- length(edges) - 1
  for (r in 1:m) {
    rows[[length(rows) + 1]] <- data.frame(
      id = i, start = edges[r], stop = edges[r + 1],
      cause = if (r == m) cause[i] else 0L, grp = grp[i], w = wt[i], x = x[i])
  }
}
sp <- do.call(rbind, rows)
sp <- sp[sample(nrow(sp)), ]   # the row order of the data is not time order
rownames(sp) <- NULL
stopifnot(all(tapply(sp$w, sp$id, function(v) diff(range(v))) == 0))

times <- c(1.5, 3, 5)
status_f <- function(d) factor(d$cause, 0:2)
mat <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))

cases <- list()
for (competing in c(FALSE, TRUE)) {
  for (renorm in c(TRUE, FALSE)) {
    for (strat in c(FALSE, TRUE)) {
      Ysp <- if (competing) Surv(sp$start, sp$stop, status_f(sp)) else Surv(sp$start, sp$stop, sp$cause > 0)
      Yun <- if (competing) Surv(subj$stop, status_f(subj)) else Surv(subj$stop, subj$cause > 0)
      f_sp <- as.formula(paste0("Ysp ~ ", if (strat) "strata(grp)" else "1"))
      f_un <- as.formula(paste0("Yun ~ ", if (strat) "strata(grp)" else "1"))
      none_sp <- rttright(f_sp, data = sp, weights = w, id = id, renorm = renorm)
      none_un <- rttright(f_un, data = subj, weights = w, renorm = renorm)
      tm_sp <- rttright(f_sp, data = sp, weights = w, id = id, times = times, renorm = renorm)
      tm_un <- rttright(f_un, data = subj, weights = w, times = times, renorm = renorm)
      cases[[length(cases) + 1]] <- list(
        competing = competing, renorm = renorm, strata = strat,
        none_split = as.numeric(none_sp), none_unsplit = as.numeric(none_un),
        times_split = mat(tm_sp), times_unsplit = mat(tm_un))
    }
  }
}

# brier: one row per subject with a start column and an id, against the
# right-censored data, and the same subjects split into rows.
subj$time <- subj$stop
subj$status <- as.integer(subj$cause > 0)
subj$start <- s0
fit_un <- coxph(Surv(time, status) ~ x, data = subj)
b_un <- brier(fit_un, times = c(2, 4), detail = TRUE)
fit_st <- coxph(Surv(start, time, status) ~ x + cluster(id), data = subj, id = id)
b_st <- brier(fit_st, times = c(2, 4), detail = TRUE)
sp$status <- as.integer(sp$cause > 0)
sp$time <- sp$stop
fit_sp <- coxph(Surv(start, stop, status) ~ x + cluster(id), data = sp, id = id)
b_sp <- brier(fit_sp, times = c(2, 4), detail = TRUE)

cat(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  split = list(id = sp$id, start = sp$start, stop = sp$stop, cause = sp$cause, grp = sp$grp,
               w = sp$w),
  subjects = list(id = subj$id, stop = subj$stop, cause = subj$cause, grp = subj$grp, w = subj$w),
  times = times, origin = s0, cases = cases,
  brier = list(times = c(2, 4),
               unsplit = list(brier = b_un$brier, rsquared = b_un$rsquared, p0 = b_un$p0,
                              eff_n = b_un$eff.n, phat = mat(b_un$phat)),
               start = list(brier = b_st$brier, rsquared = b_st$rsquared, p0 = b_st$p0,
                            eff_n = b_st$eff.n, phat = mat(b_st$phat)),
               split = list(brier = b_sp$brier, rsquared = b_sp$rsquared)),
  x = subj$x
), digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
