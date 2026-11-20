# Pseudo-observations of a multi-state process by survival::pseudo (the
# infinitesimal jackknife on a multi-state survfit): the state occupation
# probabilities (pstate), the time in each state (auc) and the cumulative
# transition hazards (cumhaz), at several times.
#
# Data: simulated illness-death processes on three states (a, b, c), with c
# absorbing, integer times so that events tie, subjects with one to three rows,
# and variants in which subjects start in either of the first two states and with
# two strata. Subjects all start at time 0: survfit forms the initial distribution
# from the subjects at risk at the start time, which differs from the opening rows
# when entry is delayed.
# Run: Rscript pseudo_states.R > pseudo_states.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261125)
v <- function(x) I(as.numeric(x))

simulate <- function(n, entry = FALSE) {  # entry: some subjects start in state b
  rows <- list()
  for (i in 1:n) {
    s <- 1
    t0 <- 0
    if (entry) s <- sample(1:2, 1, prob = c(0.7, 0.3))
    repeat {
      t1 <- t0 + ceiling(rexp(1, if (s == 1) 0.15 else 0.2))
      cens <- runif(1) < 0.15 || t1 >= 20
      if (t1 > 20) t1 <- 20
      if (cens) {
        rows[[length(rows) + 1]] <- data.frame(id = i * 3L + 1L, start = t0, stop = t1, state = 0, ist = s)
        break
      }
      nxt <- if (s == 1) sample(c(2, 3), 1, prob = c(0.6, 0.4)) else 3
      rows[[length(rows) + 1]] <- data.frame(id = i * 3L + 1L, start = t0, stop = t1, state = nxt, ist = s)
      if (nxt == 3) break
      t0 <- t1
      s <- nxt
    }
  }
  do.call(rbind, rows)
}

datasets <- list()
d <- simulate(70)
d$group <- 0L
datasets$plain <- d
d <- simulate(70, entry = TRUE)
d$group <- 0L
datasets$mixed_start <- d
d <- simulate(90)
d$group <- c(4L, 9L)[1 + (match(d$id, unique(d$id)) %% 2)]
datasets$strata <- d

times <- c(2, 5, 9, 14, 20, 26)
cases <- list()
for (nm in names(datasets)) {
  d <- datasets[[nm]]
  # The data frame is read again by pseudo, from this environment.
  d$st <- factor(d$state, 0:3, c("cens", "a", "b", "c"))
  d$ist <- factor(d$ist, 1:3, c("a", "b", "c"))
  strata <- length(unique(d$group)) > 1
  fit <- if (strata) survfit(Surv(start, stop, st) ~ group, id = id, data = d, istate = ist)
         else survfit(Surv(start, stop, st) ~ 1, id = id, data = d, istate = ist)
  ncurve <- if (strata) length(fit$strata) else 1
  nn <- if (is.null(fit$n.id)) fit$n else fit$n.id
  for (type in c("pstate", "auc", "cumhaz")) {
    rf <- pseudo(fit, times, type = type, data.frame = TRUE)
    idcol <- intersect(c("id", "(id)"), names(rf))[1]
    rf[["(id)"]] <- rf[[idcol]]
    col <- if (type == "cumhaz") "transition" else "state"
    outs <- unique(rf[[col]])
    rf$curve <- if (strata) rf$curve else 1L
    est <- rf$pseudo - nn[rf$curve] * rf$resid
    groups <- sort(unique(d$group))
    vals <- c(); labels <- c(); grp <- c(); nsub <- c(); ests <- c()
    for (ci in seq_len(ncurve)) {
      sel <- rf$curve == ci
      ids <- sort(unique(rf[["(id)"]][sel]))
      nsub <- c(nsub, length(ids))
      for (id in ids) for (tm in times) for (o in outs) {
        j <- which(sel & rf[["(id)"]] == id & rf$time == tm & rf[[col]] == o)
        stopifnot(length(j) == 1)
        vals <- c(vals, rf$pseudo[j])
      }
      for (tm in times) for (o in outs) {
        j <- which(sel & rf$time == tm & rf[[col]] == o)[1]
        ests <- c(ests, est[j])
      }
      labels <- c(labels, ids)
      grp <- c(grp, rep(groups[ci], length(ids)))
    }
    cases[[length(cases) + 1]] <- list(name = nm, type = type, times = v(times),
      origin = as.numeric(fit$t0)[1], outputs = I(as.character(outs)), group = v(grp),
      subject = v(labels), n_subjects = v(nsub), values = v(vals), estimate = v(ests))
  }
}
ds <- lapply(datasets, function(d) lapply(d, v))
versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, datasets = ds, cases = cases),
           digits = I(17), auto_unbox = TRUE, na = "null"))
