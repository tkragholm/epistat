# Curves read at chosen times by summary.survfit(times =), and the starting
# point added by survfit0, for survival::survfit 3.8-12.
#
# Data: survival::lung by sex, and a simulated counting-process set with entry
# times, three groups, case weights that are multiples of 1/8 (so that every
# sum of weights is exact), and subjects with several rows given through id.
# The curves of R's own fit are written out beside what summary() and survfit0()
# return, so that the Rust reader can be run on R's curve and compared with no
# estimator error in between.
# The standard error is that of S: std.err times surv when the fit's std.err
# is on the log scale (fit$logse), as summary() does.
# Run: Rscript survfit_times.R > survfit_times.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261121)

lung <- survival::lung
lung_d <- data.frame(stop = lung$time, status = lung$status - 1, g = lung$sex * 5L)

n <- 130
grp <- sample(c(2L, 7L, 11L), n, replace = TRUE, prob = c(0.4, 0.35, 0.25))
time <- pmin(pmax(1, ceiling(rexp(n, 0.15))), 22)
status <- rbinom(n, 1, 0.6)
entry <- pmax(0, time - sample(1:8, n, replace = TRUE))
entry[sample(n, 40)] <- 0
w <- sample(c(0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3), n, replace = TRUE)
rows <- list()
for (i in seq_len(n)) {
  span <- if (time[i] - 1 >= entry[i] + 1) (entry[i] + 1):(time[i] - 1) else integer(0)
  k <- if (length(span) >= 2) sample(1:3, 1) else if (length(span) == 1) sample(1:2, 1) else 1
  cuts <- if (k > 1) sort(span[sample.int(length(span), min(k - 1, length(span)))]) else integer(0)
  k <- length(cuts) + 1
  st <- rep(0, k)
  st[k] <- status[i]
  rows[[i]] <- data.frame(id = i, start = c(entry[i], cuts), stop = c(cuts, time[i]),
                          status = st, g = grp[i], w = w[i])
}
cp_d <- do.call(rbind, rows)
# The last time of every group is a censoring, as in the other survfit data.
for (k in unique(cp_d$g)) {
  last <- cp_d$g == k & cp_d$stop == max(cp_d$stop[cp_d$g == k])
  cp_d$status[last] <- 0
}

col <- function(v) if (is.null(v)) numeric(0) else I(unname(v))

# The curve of every stratum as the Rust GroupCurve holds it.
curves <- function(fit, levs) {
  nstrat <- if (is.null(fit$strata)) 1 else length(fit$strata)
  ends <- cumsum(if (is.null(fit$strata)) length(fit$time) else fit$strata)
  begins <- c(1, head(ends, -1) + 1)
  se <- if (fit$logse) fit$std.err * fit$surv else fit$std.err
  counts <- if (is.null(fit$counts)) cbind(fit$n.risk, fit$n.event, fit$n.censor)
            else fit$counts[, 1:3, drop = FALSE]
  lapply(seq_len(nstrat), function(k) {
    r <- begins[k]:ends[k]
    list(group = as.integer(levs[k]), time = fit$time[r],
         n_risk = fit$n.risk[r], n_event = fit$n.event[r], n_censor = fit$n.censor[r],
         count_risk = counts[r, 1], count_event = counts[r, 2], count_censor = counts[r, 3],
         surv = fit$surv[r], se = se[r], cumhaz = fit$cumhaz[r], se_cumhaz = fit$std.chaz[r],
         lower = col(fit$lower[r]), upper = col(fit$upper[r]))
  })
}

# What survfit0 returns, per stratum, with the standard error on the scale of S.
starts <- function(fit, levs) {
  f0 <- survfit0(fit)
  se <- if (fit$logse) f0$std.err * f0$surv else f0$std.err
  nstrat <- if (is.null(f0$strata)) 1 else length(f0$strata)
  ends <- cumsum(if (is.null(f0$strata)) length(f0$time) else f0$strata)
  begins <- c(1, head(ends, -1) + 1)
  lapply(seq_len(nstrat), function(k) {
    r <- begins[k]:ends[k]
    list(group = as.integer(levs[k]), time = I(f0$time[r]), n_risk = I(f0$n.risk[r]),
         n_event = I(f0$n.event[r]), n_censor = I(f0$n.censor[r]), surv = I(f0$surv[r]),
         se = I(se[r]), cumhaz = I(f0$cumhaz[r]), se_cumhaz = I(f0$std.chaz[r]),
         lower = col(f0$lower[r]), upper = col(f0$upper[r]))
  })
}

read <- function(fit, levs, times, extend, dosum) {
  args <- list(object = fit, times = times, extend = extend)
  if (!is.na(dosum)) args$dosum <- dosum
  s <- do.call(summary, args)
  nstrat <- if (is.null(fit$strata)) 1 else length(fit$strata)
  sid <- if (is.null(s$strata)) rep(1L, length(s$time)) else as.integer(s$strata)
  lapply(seq_len(nstrat), function(k) {
    i <- sid == k
    list(group = as.integer(levs[k]), time = I(s$time[i]), n_risk = I(s$n.risk[i]),
         n_event = I(s$n.event[i]), n_censor = I(s$n.censor[i]), surv = I(s$surv[i]),
         se = I(s$std.err[i]), cumhaz = I(s$cumhaz[i]), se_cumhaz = I(s$std.chaz[i]),
         lower = col(s$lower[i]), upper = col(s$upper[i]))
  })
}

specs <- list(
  list(name = "lung_sex", d = lung_d, counting = FALSE, w = FALSE, id = FALSE,
       fits = list(c(1, 1, "log"), c(2, 1, "log-log"), c(1, 2, "plain"), c(1, 1, "none"))),
  list(name = "cp", d = cp_d, counting = TRUE, w = TRUE, id = TRUE,
       fits = list(c(1, 1, "log"), c(2, 2, "log"), c(1, 1, "none"))),
  list(name = "cp_plain", d = transform(cp_d, w = 1), counting = TRUE, w = FALSE, id = TRUE,
       fits = list(c(1, 1, "log"), c(1, 1, "logit")))
)

cases <- list()
for (s in specs) {
  d <- s$d
  lhs <- if (s$counting) "Surv(start, stop, status)" else "Surv(stop, status)"
  for (f in s$fits) {
    args <- list(formula = as.formula(paste(lhs, "~ factor(g)")), data = d,
                 stype = as.integer(f[1]), ctype = as.integer(f[2]), conf.type = f[3])
    if (s$w) args$weights <- d$w
    if (s$id) args$id <- d$id
    fit <- do.call(survfit, args)
    levs <- levels(factor(d$g))
    ft <- sort(unique(fit$time))
    m <- length(ft)
    sets <- list(
      increasing = c(-1, 0, ft[1] / 2, ft[1], (ft[3] + ft[4]) / 2, ft[10], ft[m %/% 2],
                     ft[m - 1], ft[m] + 0.25, ft[m] + 10),
      repeated = c(ft[10], ft[2], ft[10], 0, ft[m] + 3, ft[m %/% 3]),
      single = ft[m %/% 2] + 0.5,
      after = c(ft[m] + 1, ft[m] + 2)
    )
    reads <- list()
    for (nm in names(sets)) {
      for (extend in c(FALSE, TRUE)) {
        for (dosum in c(NA, TRUE, FALSE)) {
          t <- sets[[nm]]
          if (isTRUE(dosum) && !all(diff(t) > 0)) next
          groups <- tryCatch(read(fit, levs, t, extend, dosum), error = function(e) NULL)
          if (is.null(groups)) next
          reads[[length(reads) + 1]] <- list(set = nm, times = I(t), extend = extend,
                                             dosum = if (is.na(dosum)) NA else dosum,
                                             groups = groups)
        }
      }
    }
    cases[[length(cases) + 1]] <- list(
      name = s$name, stype = as.integer(f[1]), ctype = as.integer(f[2]), conf_type = f[3],
      curves = curves(fit, levs), starts = starts(fit, levs), reads = reads)
  }
}
# Competing risks: the Aalen-Johansen curves of survfit read by summary.survfitms.
cause <- sample(0:2, nrow(lung_d), replace = TRUE, prob = c(0.2, 0.5, 0.3))
cr_d <- data.frame(stop = lung_d$stop, cause = cause, g = lung_d$g)
cr_fit <- survfit(Surv(stop, factor(cause, 0:2, c("cen", "a", "b"))) ~ factor(g), data = cr_d)
cr_times <- c(0, 15, 100, 100.5, 300, 700, 1100)
cr_reads <- list()
for (extend in c(FALSE, TRUE)) {
  s <- summary(cr_fit, times = cr_times, extend = extend)
  sid <- as.integer(s$strata)
  cr_reads[[length(cr_reads) + 1]] <- list(extend = extend, groups = lapply(1:2, function(k) {
    i <- sid == k
    list(group = c(5L, 10L)[k], time = I(s$time[i]), pstate = lapply(1:3, function(j) I(s$pstate[i, j])))
  }))
}
cr <- list(stop = cr_d$stop, cause = cr_d$cause, g = cr_d$g, times = I(cr_times), reads = cr_reads)

versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 jsonlite = as.character(packageVersion("jsonlite")))
dataset <- function(d, counting) list(stop = d$stop, status = d$status, g = d$g,
  start = if (counting) d$start else numeric(0), id = if (counting) d$id else integer(0),
  w = if (counting) d$w else numeric(0))
datasets <- list(lung_sex = dataset(lung_d, FALSE), cp = dataset(cp_d, TRUE),
                 cp_plain = dataset(transform(cp_d, w = 1), TRUE))
cat(toJSON(list(versions = versions, datasets = datasets, cases = cases, cr = cr), digits = I(17), auto_unbox = TRUE,
           na = "null"))
