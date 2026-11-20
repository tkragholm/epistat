# Aalen-Johansen cumulative incidence by survival::survfit with a factor
# status, checked further against etm::etm and cmprsk::cuminc where they cover
# the case, and prodlim::prodlim for the unweighted cases. Three causes, three groups, integer times so that events and
# censorings tie, delayed entry as counting-process Surv(entry, time, status)
# with entries that coincide with other rows' exit times, and two sets of case
# weights: multiples of 1/8, whose sums are exact in binary, and
# two-decimal weights, whose sums are not.
# For each case and group: the distinct exit times, the weighted and the
# unweighted number at risk, the number of events, the event-free probability
# and the cumulative incidence of each cause. Cases without weights and
# without delayed entry also carry cmprsk's estimate, cases without weights
# carry etm's and prodlim's.
# The numbers are written with 17 significant digits, which digits = NA does not
# (it writes 15), so that a double survives the round trip.
# Run: Rscript aalen_johansen.R > aalen_johansen.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(etm)
  library(cmprsk)
  library(prodlim)
})
set.seed(20261120)
n <- 240
group <- sample(1:3, n, replace = TRUE, prob = c(0.4, 0.35, 0.25))
time <- pmax(1, ceiling(rexp(n, 0.12 * c(1, 0.7, 1.4)[group])))
time <- pmin(time, 24)
status <- sample(0:3, n, replace = TRUE, prob = c(0.3, 0.35, 0.2, 0.15))
entry <- pmax(0, time - sample(1:9, n, replace = TRUE))
entry[sample(n, 60)] <- 0
# The rows at the largest time of a group are censored, so that no
# group's event-free probability reaches zero.
for (g in 1:3) {
  last <- group == g & time == max(time[group == g])
  status[last] <- 0
}
weight_exact <- sample(c(0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3), n, replace = TRUE)
weight_decimal <- round(runif(n, 0.5, 3), 2)
fstatus <- factor(status, levels = 0:3, labels = c("cens", "1", "2", "3"))

specs <- list(
  list("plain", FALSE, "none", FALSE),
  list("groups", FALSE, "none", TRUE),
  list("weights_exact", FALSE, "exact", FALSE),
  list("groups_weights_exact", FALSE, "exact", TRUE),
  list("groups_weights_decimal", FALSE, "decimal", TRUE),
  list("entry", TRUE, "none", FALSE),
  list("groups_entry", TRUE, "none", TRUE),
  list("entry_weights_exact", TRUE, "exact", FALSE),
  list("groups_entry_weights_exact", TRUE, "exact", TRUE),
  list("groups_entry_weights_decimal", TRUE, "decimal", TRUE)
)

columns <- function(m) unname(lapply(seq_len(ncol(m)), function(j) m[, j]))

cases <- list()
for (s in specs) {
  name <- s[[1]]; use_entry <- s[[2]]; wkind <- s[[3]]; use_group <- s[[4]]
  w <- switch(wkind, none = rep(1, n), exact = weight_exact, decimal = weight_decimal)
  grp <- if (use_group) group else rep(1L, n)
  d <- data.frame(entry, time, fstatus, grp, w, id = seq_len(n))
  lhs <- if (use_entry) "Surv(entry, time, fstatus)" else "Surv(time, fstatus)"
  rhs <- if (use_group) "factor(grp)" else "1"
  args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d)
  if (wkind != "none") args$weights <- d$w
  if (use_entry) args$id <- d$id
  fit <- do.call(survfit, args)
  args$weights <- NULL
  args$formula <- as.formula(paste(lhs, "~", rhs))
  unweighted <- do.call(survfit, args)
  stopifnot(identical(fit$states, c("(s0)", "1", "2", "3")))
  ends <- cumsum(if (is.null(fit$strata)) length(fit$time) else fit$strata)
  begins <- c(1, head(ends, -1) + 1)
  groups <- list()
  for (k in seq_along(ends)) {
    rows <- begins[k]:ends[k]
    ps <- fit$pstate[rows, , drop = FALSE]
    stopifnot(all(ps[, 1] > 0))
    entry_g <- list(
      group = k - 1L,
      time = fit$time[rows],
      n_risk = rowSums(fit$n.risk[rows, , drop = FALSE]),
      n_at_risk = rowSums(unweighted$n.risk[rows, , drop = FALSE]),
      n_event = rowSums(fit$n.event[rows, , drop = FALSE]),
      event_free = ps[, 1],
      cif = columns(ps[, 2:4, drop = FALSE])
    )
    sel <- which(grp == k)
    # etm and cmprsk take no case weights.
    if (wkind == "none") {
      ed <- data.frame(
        id = sel, from = "0",
        to = ifelse(status[sel] == 0, "cens", as.character(status[sel])),
        entry = entry[sel], exit = time[sel]
      )
      tra <- matrix(FALSE, 4, 4, dimnames = list(c("0", "1", "2", "3"), c("0", "1", "2", "3")))
      tra["0", c("1", "2", "3")] <- TRUE
      if (use_entry) {
        fit_etm <- etm(ed, c("0", "1", "2", "3"), tra, "cens", s = 0)
      } else {
        fit_etm <- etm(ed[, c("id", "from", "to", "exit")] |> setNames(c("id", "from", "to", "time")),
                       c("0", "1", "2", "3"), tra, "cens", s = 0)
      }
      entry_g$etm_time <- fit_etm$time
      entry_g$etm_event_free <- fit_etm$est["0", "0", ]
      entry_g$etm_cif <- unname(lapply(c("1", "2", "3"), function(j) fit_etm$est["0", j, ]))
      if (!use_entry) {
        ci <- cuminc(time[sel], status[sel])
        # cuminc repeats each time, first with the value before it and then
        # with the value at it, for plotting. The value at it is kept.
        keep <- lapply(1:3, function(j) !duplicated(ci[[paste("1", j)]]$time, fromLast = TRUE))
        entry_g$cmprsk_time <- unname(lapply(1:3, function(j) ci[[paste("1", j)]]$time[keep[[j]]]))
        entry_g$cmprsk_cif <- unname(lapply(1:3, function(j) ci[[paste("1", j)]]$est[keep[[j]]]))
      }
    }
    if (wkind == "none") {
      pd <- data.frame(time = time[sel], status = status[sel], entry = entry[sel])
      pf <- if (use_entry) prodlim(Hist(time, status, entry = entry) ~ 1, data = pd)
            else prodlim(Hist(time, status) ~ 1, data = pd)
      entry_g$prodlim_cif <- unname(lapply(1:3, function(j) {
        as.numeric(predict(pf, cause = j, times = entry_g$time, type = "cuminc"))
      }))
    }
    groups[[k]] <- entry_g
  }
  cases[[length(cases) + 1]] <- list(
    name = name, use_entry = use_entry, weights = wkind, use_group = use_group,
    groups = groups
  )
}

versions <- list(
  R = R.version.string,
  survival = as.character(packageVersion("survival")),
  etm = as.character(packageVersion("etm")),
  cmprsk = as.character(packageVersion("cmprsk")),
  prodlim = as.character(packageVersion("prodlim")),
  jsonlite = as.character(packageVersion("jsonlite"))
)
cat(toJSON(list(
  versions = versions,
  time = time, status = status, entry = entry, group = group - 1L,
  weight_exact = weight_exact, weight_decimal = weight_decimal,
  cases = cases
), digits = I(17), auto_unbox = TRUE, na = "null"))
