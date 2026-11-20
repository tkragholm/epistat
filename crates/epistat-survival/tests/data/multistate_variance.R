# The infinitesimal jackknife standard errors of a multi-state fit by
# survival::survfit(Surv(start, stop, state) ~ 1, id = id, istate = istate):
# std.err of the state probabilities, std.chaz of the cumulative hazards,
# std.auc of the restricted time in each state, se0 of the initial
# distribution, the influence of each cluster (influence = TRUE) and the
# se(rmean) of summary(fit, rmean = tau), which uses that influence and is
# exact for a tau between event times.
# The process is that of multistate.R: an illness-death process with recovery
# (states 1, 2, 3), integer times, case weights, delayed entry. Cases with an
# estimated initial distribution have everyone under observation at the start
# time; cases with delayed entry give survfit the initial distribution, since
# survfit and the estimate here form it from different rows when people enter
# late. Some cases cluster three subjects together.
# Run: Rscript multistate_variance.R > multistate_variance.json
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


cl_of <- function(id) (id - 1) %/% 3 + 1
# name, subset, weights, start.time = 0?, p0 given?, cluster?, influence kept?
specs <- list(
  list("common_start", function(d) common(d), "none", FALSE, FALSE, FALSE, FALSE),
  list("mixed_start_zero", function(d) d, "none", TRUE, FALSE, FALSE, TRUE),
  list("subset_start_zero", function(d) odd(d), "none", TRUE, FALSE, FALSE, FALSE),
  list("weights_exact", function(d) d, "exact", TRUE, FALSE, FALSE, FALSE),
  list("weights_decimal_cluster", function(d) d, "decimal", TRUE, FALSE, TRUE, TRUE),
  list("mixed_start_default_p0", function(d) d, "none", FALSE, TRUE, FALSE, FALSE),
  list("entry_weights_exact_p0", function(d) delay(d), "exact", TRUE, TRUE, FALSE, TRUE),
  list("entry_weights_decimal_cluster_p0", function(d) delay(d), "decimal", TRUE, TRUE, TRUE, FALSE)
)

columns <- function(m) unname(lapply(seq_len(ncol(m)), function(j) m[, j]))

cases <- list()
for (s in specs) {
  name <- s[[1]]; wkind <- s[[3]]; start_zero <- s[[4]]
  give_p0 <- s[[5]]; use_cluster <- s[[6]]; keep_influence <- s[[7]]
  x <- s[[2]](d)
  x$w <- switch(wkind, none = 1, exact = weight_exact[x$id], decimal = weight_decimal[x$id])
  x$st <- factor(x$state, 0:3, c("cens", "1", "2", "3"))
  x$istate <- factor(x$from, 1:3, c("1", "2", "3"))
  x$cl <- cl_of(x$id)
  args <- list(formula = Surv(start, stop, st) ~ 1, data = x, id = x$id,
               istate = x$istate, influence = TRUE)
  if (wkind != "none") args$weights <- x$w
  if (start_zero) args$start.time <- 0
  if (use_cluster) args$cluster <- x$cl
  if (give_p0) {
    # the distribution of states of everyone at risk at the first time
    at <- x$start == 0
    args$p0 <- as.numeric(tapply(x$w[at & !duplicated(x$id)], x$from[at & !duplicated(x$id)], sum)[as.character(1:3)])
    args$p0[is.na(args$p0)] <- 0
    args$p0 <- args$p0 / sum(args$p0)
  }
  warns <- character()
  fit <- withCallingHandlers(do.call(survfit, args), warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  stopifnot(identical(fit$states, c("1", "2", "3")))
  t0 <- if (is.null(fit$t0)) min(0, fit$time) else fit$t0
  last_t <- max(fit$time)
  taus <- sort(unique(c(fit$time[1], fit$time[3], fit$time[5] + 0.5,
                        last_t - 1, last_t, last_t + 3)))
  rmean <- lapply(taus, function(tau) {
    tab <- summary(fit, rmean = tau)$table
    stopifnot(rownames(tab) == c("1", "2", "3"))
    list(rmean = unname(tab[, "rmean"]), se = unname(tab[, "se(rmean)"]))
  })
  entry <- list(
    name = name, weights = wkind, warnings = warns, estimated_p0 = !give_p0,
    start = x$start, stop = x$stop, status = x$state, from = x$from,
    id = x$id, cluster = if (use_cluster) x$cl else integer(0), weight = x$w,
    time = fit$time, p0 = unname(fit$p0), t0 = t0,
    se0 = if (is.null(fit$se0)) numeric(0) else unname(fit$se0),
    cname = colnames(fit$cumhaz),
    pstate = columns(fit$pstate), std_err = columns(fit$std.err),
    cumhaz = columns(fit$cumhaz), std_chaz = columns(fit$std.chaz),
    std_auc = columns(fit$std.auc),
    tau = taus, rmean = rmean
  )
  if (keep_influence) {
    inf <- fit$influence.pstate
    pos <- as.integer(rownames(inf))
    cluster_all <- if (use_cluster) x$cl else x$id
    entry$uclust <- unique(cluster_all)[pos]
    entry$influence <- lapply(seq_len(dim(inf)[3]), function(j)
      lapply(seq_len(dim(inf)[1]), function(g) inf[g, , j]))
  }
  cases[[length(cases) + 1]] <- entry
}

versions <- list(
  R = R.version.string,
  survival = as.character(packageVersion("survival")),
  jsonlite = as.character(packageVersion("jsonlite"))
)
cat(toJSON(list(versions = versions, cases = cases), digits = I(17), auto_unbox = TRUE, na = "null"))
