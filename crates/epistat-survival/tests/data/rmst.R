# Restricted mean survival time and time lost in two groups by survRM2::rmst2:
# the area under each arm's Kaplan-Meier curve with its variance, the
# difference and the ratios, the default horizon in each of the ways the
# arms can end, and the horizons rmst2 refuses.
#
# Data: survival::lung with sex as the arm, and simulated sets with tied integer
# times in which each arm's last time is an event or a censoring, the first arm's
# last time is before, at or after the second's, and one arm has an event and
# a censoring at its last time.
# Run: Rscript rmst.R > rmst.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(survRM2)
})
set.seed(20261120)
v <- function(x) I(as.numeric(x))

sim <- function(n, ends, order, mixed = FALSE) {
  arm <- rbinom(n, 1, 0.5)
  time <- pmax(1, ceiling(rexp(n, 0.15 * c(1, 0.7)[arm + 1])))
  time <- pmin(time, 20)
  status <- rbinom(n, 1, c(0.6, 0.45)[arm + 1])
  # Put each arm's largest time beyond all others, as set by `order`.
  big <- c(30, 30)
  if (order == "less") big <- c(26, 30)
  if (order == "greater") big <- c(30, 26)
  for (a in 0:1) {
    i <- which(arm == a)[1:2]
    time[i[1]] <- big[a + 1]
    status[i[1]] <- ends[a + 1]
    if (mixed && a == 0) {
      time[i[2]] <- big[a + 1]
      status[i[2]] <- 1 - ends[a + 1]
    }
  }
  data.frame(time = time, status = status, arm = arm)
}

datasets <- list()
datasets$lung <- with(lung, data.frame(time = time, status = status - 1, arm = sex - 1))
for (e0 in 0:1) for (e1 in 0:1) for (o in c("less", "equal", "greater")) {
  datasets[[sprintf("sim_e%d%d_%s", e0, e1, o)]] <- sim(120, c(e0, e1), o)
}
datasets$sim_mixed_last <- sim(120, c(1, 1), "equal", mixed = TRUE)

run <- function(d, tau, alpha) {
  out <- tryCatch(
    suppressMessages(rmst2(d$time, d$status, d$arm, tau = tau, alpha = alpha)),
    error = function(e) NULL)
  if (is.null(out)) return(list(error = TRUE, tau_given = if (is.null(tau)) NA else tau, alpha = alpha))
  arm <- function(w) list(rmst = w$rmst[["Est."]], se = w$rmst[["se"]],
                          rmst_lower = w$rmst[[3]], rmst_upper = w$rmst[[4]],
                          rmtl = w$rmtl[["Est."]], rmtl_lower = w$rmtl[[3]],
                          rmtl_upper = w$rmtl[[4]])
  u <- out$unadjusted.result
  row <- function(i) list(est = u[i, 1], lower = u[i, 2], upper = u[i, 3], p = u[i, 4])
  list(error = FALSE, tau_given = if (is.null(tau)) NA else tau, alpha = alpha, tau = out$tau,
       arm0 = arm(out$RMST.arm0), arm1 = arm(out$RMST.arm1),
       difference = row(1), ratio = row(2), loss_ratio = row(3))
}

cases <- list()
for (nm in names(datasets)) {
  d <- datasets[[nm]]
  mx <- max(d$time)
  taus <- list(NULL, 0.5 * mx, 0.9 * mx, mx, 1.5 * mx)
  if (nm == "lung") taus <- list(NULL, 200, 500, 800, 1022)
  fits <- lapply(taus, function(t) run(d, t, 0.05))
  if (nm %in% c("lung", "sim_e11_equal")) fits[[length(fits) + 1]] <- run(d, taus[[3]], 0.1)
  cases[[length(cases) + 1]] <- list(name = nm, fits = fits)
}
ds <- lapply(datasets, function(d) list(time = v(d$time), status = v(d$status), arm = v(d$arm)))
versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 survRM2 = as.character(packageVersion("survRM2")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, datasets = ds, cases = cases),
           digits = I(17), auto_unbox = TRUE, na = "null"))
