# State occupation probabilities and restricted time in state by
# survival::survfit(Surv(start, stop, state) ~ 1, id = id, istate = istate):
# an illness-death process with recovery (states 1, 2, 3; 1 <-> 2, both to the
# absorbing 3), integer times so that transitions and censorings tie, an initial
# state distribution that is a mix of states 1 and 2, case weights (multiples of
# 1/8, whose sums are exact in binary, and two-decimal weights, whose sums are
# not), delayed entry (subjects who come under observation at a time after 0, in
# the state they are in then) and a subset of the subjects.
# survfit takes its starting time t0 to be 0 when everyone starts in one state
# or start.time says so, and otherwise the first transition time. Each case
# says which it used. The initial distribution p0 is what survfit used.
# For each case: the intervals, the event times, p0, t0, the state
# probabilities and the weight at risk at each time, and the restricted mean
# time in each state that summary(fit, rmean = tau) reports for several tau,
# among them an event time, a time between two and a time past the last.
# Run: Rscript multistate.R > multistate.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)
n <- 90
rows <- list()
for (i in 1:n) {
  s <- sample(1:2, 1, prob = c(0.7, 0.3))
  t <- 0
  repeat {
    t <- t + 1
    u <- runif(1)
    if (t >= 12 || u < 0.05) {
      rows[[length(rows) + 1]] <- c(i, t - 1, t, 0, s)
      break
    }
    p <- if (s == 1) c(0.1, 0.03) else c(0.15, 0.08)
    if (u < 0.05 + p[1]) {
      ns <- if (s == 1) 2 else 1
      rows[[length(rows) + 1]] <- c(i, t - 1, t, ns, s)
      s <- ns
    } else if (u < 0.05 + p[1] + p[2]) {
      rows[[length(rows) + 1]] <- c(i, t - 1, t, 3, s)
      break
    }
  }
}
d <- as.data.frame(do.call(rbind, rows))
names(d) <- c("id", "start", "stop", "state", "from")
# A row starts where the subject's previous row stopped.
d$start <- ave(d$stop, d$id, FUN = function(x) c(0, head(x, -1)))
weight_exact <- sample(c(0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3), n, replace = TRUE)
weight_decimal <- round(runif(n, 0.5, 3), 2)
entry_time <- ifelse(runif(n) < 0.35, sample(1:4, n, replace = TRUE), 0)

# Subjects who enter late: the rows before the entry time are dropped.
delay <- function(d) {
  e <- entry_time[d$id]
  d <- d[d$stop > e, ]
  d$start <- pmax(d$start, entry_time[d$id])
  d
}
# Everyone who starts in state 1.
common <- function(d) d[d$id %in% d$id[d$start == 0 & d$from == 1 & !duplicated(d$id)], ]
odd <- function(d) d[d$id %% 2 == 1, ]

specs <- list(
  list("common_start", function(d) common(d), "none", FALSE),
  list("mixed_start_default", function(d) d, "none", FALSE),
  list("mixed_start_zero", function(d) d, "none", TRUE),
  list("subset_start_zero", function(d) odd(d), "none", TRUE),
  list("weights_exact", function(d) d, "exact", TRUE),
  list("weights_decimal", function(d) d, "decimal", TRUE),
  list("entry", function(d) delay(d), "none", TRUE),
  list("entry_weights_exact", function(d) delay(d), "exact", TRUE),
  list("entry_weights_decimal", function(d) delay(d), "decimal", TRUE)
)

columns <- function(m) unname(lapply(seq_len(ncol(m)), function(j) m[, j]))

cases <- list()
for (s in specs) {
  name <- s[[1]]; wkind <- s[[3]]; start_zero <- s[[4]]
  x <- s[[2]](d)
  x$w <- switch(wkind, none = 1, exact = weight_exact[x$id], decimal = weight_decimal[x$id])
  x$st <- factor(x$state, 0:3, c("cens", "1", "2", "3"))
  x$istate <- factor(x$from, 1:3, c("1", "2", "3"))
  args <- list(formula = Surv(start, stop, st) ~ 1, data = x, id = x$id, istate = x$istate)
  if (wkind != "none") args$weights <- x$w
  if (start_zero) args$start.time <- 0
  warns <- character()
  fit <- withCallingHandlers(do.call(survfit, args), warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  stopifnot(identical(fit$states, c("1", "2", "3")))
  t0 <- if (is.null(fit$t0)) min(0, fit$time) else fit$t0
  taus <- sort(unique(c(
    fit$time[1], fit$time[3], fit$time[5] + 0.5, fit$time[length(fit$time) - 1],
    max(fit$time), max(fit$time) + 3
  )))
  rmean <- lapply(taus, function(tau) {
    tab <- summary(fit, rmean = tau)$table
    stopifnot(rownames(tab) == c("1", "2", "3"))
    unname(tab[, "rmean"])
  })
  common_max <- summary(fit)$table
  cases[[length(cases) + 1]] <- list(
    name = name, weights = wkind, warnings = warns,
    start = x$start, stop = x$stop,
    from = match(as.character(x$from), c("1", "2", "3")) - 1L,
    to = ifelse(x$state == 0, 3L, x$state - 1L),
    weight = x$w,
    time = fit$time, p0 = unname(fit$p0), t0 = t0,
    n_risk = rowSums(fit$n.risk),
    pstate = columns(fit$pstate),
    tau = taus, rmean = rmean,
    rmean_common = unname(common_max[, "rmean"])
  )
}

versions <- list(
  R = R.version.string,
  survival = as.character(packageVersion("survival")),
  jsonlite = as.character(packageVersion("jsonlite"))
)
cat(toJSON(list(versions = versions, cases = cases), digits = I(17), auto_unbox = TRUE, na = "null"))
