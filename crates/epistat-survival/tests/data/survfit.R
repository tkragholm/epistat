# Survival curves of single-event data by survival::survfit: the Kaplan-Meier
# and Nelson-Aalen estimators and their stype and ctype variants, Greenwood and
# infinitesimal jackknife standard errors, every conf.type and conf.lower,
# quantiles with limits (quantile.survfit) and restricted means
# (survival:::survmean, which print.survfit and summary.survfit call).
#
# Data: survival::lung, and a simulated set with tied integer times, delayed
# entry, three groups with different event rates, three kinds of case weights
# (whole numbers, multiples of 1/8 whose sums are exact, two decimals), a
# cluster column, subjects with several rows (and some that switch group or
# have a gap), and times that tie only up to rounding error.
# The last rows of every group are censored, so that no curve reaches zero,
# except in the cases named "to_zero".
# Rows of weight 0 are left out of R's fit; the Rust function drops them.
#
# Under the jackknife variance survfit records the wrong scale for its
# standard error when stype and ctype differ (logse is TRUE where std.err is
# that of S for stype 1 with ctype 2, and FALSE where it is that of log S for
# stype 2 with ctype 1), so its summary standard error and limits are wrong
# there. For those fits "se", "lower" and "upper" hold the values formed with
# std.err on its own scale, and "bug" is true.
# Run: Rscript survfit.R > survfit.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)
n <- 240
group <- sample(1:3, n, replace = TRUE, prob = c(0.4, 0.35, 0.25))
time <- pmax(1, ceiling(rexp(n, 0.12 * c(1, 0.7, 1.4)[group])))
time <- pmin(time, 24)
status <- rbinom(n, 1, c(0.7, 0.5, 0.15)[group])
entry <- pmax(0, time - sample(1:9, n, replace = TRUE))
entry[sample(n, 60)] <- 0
code_of <- c(7L, 3L, 12L)
grp <- code_of[group]
w_int <- sample(1:3, n, replace = TRUE) + 0
w_exact <- sample(c(0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3), n, replace = TRUE)
w_decimal <- round(runif(n, 0.5, 3), 2)
w_zero <- w_exact
w_zero[sample(n, 25)] <- 0
cluster <- sample(1:60, n, replace = TRUE)
noisy <- time + runif(n, -1e-12, 1e-12) * time

# The last rows of every group are censored.
censor_last <- function(time, status, g) {
  for (k in unique(g)) {
    last <- g == k & time == max(time[g == k])
    status[last] <- 0
  }
  status
}
status_c <- censor_last(time, status, group)

# Subjects with several rows. Each of the n subjects is cut at integer times
# into one to three intervals. A row ending in an event may be followed by
# another row (repeated events), a gap may precede the next row, and the later
# rows may belong to another group.
make_split <- function(repeat_events) {
  rows <- list()
  for (i in seq_len(n)) {
    span <- if (time[i] - 1 >= entry[i] + 1) (entry[i] + 1):(time[i] - 1) else integer(0)
    k <- if (length(span) >= 2) sample(1:3, 1) else if (length(span) == 1) sample(1:2, 1) else 1
    cuts <- if (k > 1) sort(span[sample.int(length(span), min(k - 1, length(span)))]) else integer(0)
    k <- length(cuts) + 1
    starts <- c(entry[i], cuts)
    stops <- c(cuts, time[i])
    if (k > 1 && runif(1) < 0.1 && stops[1] + 1 < stops[2]) starts[2] <- starts[2] + 1
    st <- rep(0, k)
    st[k] <- status[i]
    if (repeat_events && k > 1 && runif(1) < 0.2) st[1] <- 1
    gg <- rep(group[i], k)
    if (k > 1 && runif(1) < 0.15) gg[2:k] <- (group[i] %% 3) + 1
    rows[[i]] <- data.frame(id = i, start = starts, stop = stops, status = st, g = gg)
  }
  d <- do.call(rbind, rows)
  d$status <- censor_last(d$stop, d$status, d$g)
  d
}
split_plain <- make_split(FALSE)
split_events <- make_split(TRUE)

# Specifications. surv is the Surv call, rhs the right side.
base <- data.frame(time = time, status = status_c, start = entry, g = grp,
                   w = 1, cl = cluster)
spec <- function(name, d, counting = FALSE, w = NULL, grouped = TRUE, id = FALSE,
                 cl = FALSE, robust = NULL, drop_zero = FALSE, taus = c(0.5, 3, 8, 15, 30)) {
  list(name = name, d = d, counting = counting, w = w, grouped = grouped, id = id,
       cl = cl, robust = robust, drop_zero = drop_zero, taus = taus)
}
lung <- survival::lung
lung_d <- data.frame(time = lung$time, status = lung$status - 1, g = lung$sex * 5L, w = 1)
lung_taus <- c(10, 100, 300, 800, 1200)
sd <- function(d, g) { d$g <- code_of[g]; d }
# The last rows of every group are events, so each curve ends at zero.
status_z <- status
for (k in unique(group)) status_z[group == k & time == max(time[group == k])] <- 1
flat <- data.frame(time = 1:10, status = c(1, 1, 1, 1, 1, 0, 0, 0, 1, 0), start = 0, g = 1L,
                   w = 1, cl = 1L)

specs <- list(
  spec("lung", lung_d, grouped = FALSE, taus = lung_taus),
  spec("lung_sex", lung_d, taus = lung_taus),
  spec("groups", base),
  spec("entry", base, counting = TRUE),
  spec("weights_whole", transform(base, w = w_int), w = "w"),
  spec("weights_exact", transform(base, w = w_exact), w = "w"),
  spec("weights_exact_model", transform(base, w = w_exact), w = "w", robust = FALSE),
  spec("weights_decimal_entry", transform(base, w = w_decimal), counting = TRUE, w = "w", robust = FALSE),
  spec("cluster", transform(base, w = w_exact), w = "w", cl = TRUE),
  spec("zero_weights", transform(base, w = w_zero), w = "w", drop_zero = TRUE),
  spec("noisy_times", transform(base, time = noisy)),
  spec("to_zero_whole", transform(base, status = status_z, w = w_int), w = "w"),
  spec("to_zero_exact", transform(base, status = status_z, w = w_exact), w = "w"),
  spec("to_zero_entry", transform(base, status = status_z), counting = TRUE),
  spec("flat", flat, grouped = FALSE, taus = c(0.5, 5, 7.5, 10, 12)),
  spec("id", split_plain, counting = TRUE, id = TRUE),
  spec("id_events", split_events, counting = TRUE, id = TRUE),
  spec("id_robust", split_plain, counting = TRUE, id = TRUE, robust = TRUE)
)

# survfit's default rule for the variance.
robust_used <- function(s, d) {
  if (!is.null(s$robust)) return(s$robust)
  has_w <- !is.null(s$w) && any(d$w != floor(d$w))
  dup <- s$id && anyDuplicated(d$id[d$status == 1]) > 0
  s$cl || has_w || dup
}

col <- function(v) if (is.null(v)) numeric(0) else unname(v)

run <- function(s, stype, ctype, conf.type, conf.lower, conf.int) {
  d <- s$d
  if (s$drop_zero) d <- d[d$w > 0, ]
  lhs <- if (s$counting) "Surv(start, stop, status)" else "Surv(time, status)"
  if (s$counting && !s$id) d$stop <- d$time
  rhs <- if (s$grouped) "factor(g)" else "1"
  args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d, stype = stype,
               ctype = ctype, conf.type = conf.type, conf.lower = conf.lower,
               conf.int = conf.int)
  if (!is.null(s$w)) args$weights <- d$w
  if (s$id) args$id <- d$id
  if (s$cl) args$cluster <- d$cl
  if (!is.null(s$robust)) args$robust <- s$robust
  fit <- do.call(survfit, args)
  rb <- robust_used(s, d)
  nstrat <- if (is.null(fit$strata)) 1 else length(fit$strata)
  ends <- cumsum(if (is.null(fit$strata)) length(fit$time) else fit$strata)
  begins <- c(1, head(ends, -1) + 1)
  levs <- if (s$grouped) levels(factor(d$g)) else "0"
  # The scale of std.err: log S for the model-based variance, and with the
  # jackknife S for stype 1 and log S (the cumulative hazard's) for stype 2.
  true_logse <- if (rb) stype == 2 else TRUE
  bug <- conf.type != "none" && !identical(true_logse, isTRUE(fit$logse))
  se_s <- if (true_logse) fit$std.err * fit$surv else fit$std.err
  lower <- fit$lower
  upper <- fit$upper
  if (bug) {
    ci <- survival:::survfit_confint(fit$surv, fit$std.err, logse = true_logse, conf.type,
                                     conf.int, fit$std.err)
    lower <- ci$lower
    upper <- ci$upper
    fit$lower <- lower
    fit$upper <- upper
  }
  probs <- c(0.25, 0.5, 0.75)
  qs <- if (conf.type == "none") quantile(fit, probs, conf.int = FALSE)
        else quantile(fit, probs, conf.int = TRUE)
  tomat <- function(x) if (is.matrix(x)) x else matrix(x, nrow = 1)
  if (conf.type == "none") {
    qm <- tomat(qs); ql <- NULL; qu <- NULL
  } else {
    qm <- tomat(qs$quantile); ql <- tomat(qs$lower); qu <- tomat(qs$upper)
  }
  counts <- if (is.null(fit$counts)) cbind(fit$n.risk, fit$n.event, fit$n.censor) else fit$counts[, 1:3, drop = FALSE]
  groups <- list()
  for (k in seq_len(nstrat)) {
    r <- begins[k]:ends[k]
    rm <- lapply(s$taus, function(tau) {
      m <- survival:::survmean(fit, 1, tau)$matrix
      if (!is.matrix(m)) m <- matrix(m, nrow = 1, dimnames = list(NULL, names(m)))
      c(m[k, "rmean"], m[k, "se(rmean)"])
    })
    groups[[k]] <- list(
      group = as.integer(levs[k]),
      n = fit$n[k],
      n_id = if (is.null(fit$n.id)) NA else fit$n.id[k],
      time = fit$time[r],
      n_risk = fit$n.risk[r], n_event = fit$n.event[r], n_censor = fit$n.censor[r],
      count_risk = counts[r, 1], count_event = counts[r, 2], count_censor = counts[r, 3],
      surv = fit$surv[r], se = se_s[r], cumhaz = fit$cumhaz[r], se_cumhaz = fit$std.chaz[r],
      lower = col(lower[r]), upper = col(upper[r]),
      q = qm[k, ], q_lower = col(ql[k, ]), q_upper = col(qu[k, ]),
      rmean = sapply(rm, `[`, 1), rmean_se = sapply(rm, `[`, 2)
    )
  }
  list(stype = stype, ctype = ctype, conf_type = conf.type, conf_lower = conf.lower,
       conf_int = conf.int, robust = rb, bug = bug, logse = isTRUE(fit$logse),
       groups = groups)
}

cases <- list()
for (s in specs) {
  fits <- list()
  for (st in 1:2) for (ct in 1:2) {
    fits[[length(fits) + 1]] <- run(s, st, ct, "log", "usual", 0.95)
  }
  if (s$name %in% c("lung_sex", "weights_exact", "entry", "id_events", "to_zero_whole", "flat")) {
    for (cty in c("plain", "log-log", "logit", "arcsin")) {
      fits[[length(fits) + 1]] <- run(s, 1, 1, cty, "usual", 0.95)
    }
    fits[[length(fits) + 1]] <- run(s, 1, 1, "log", "peto", 0.95)
    fits[[length(fits) + 1]] <- run(s, 1, 1, "log", "modified", 0.95)
    fits[[length(fits) + 1]] <- run(s, 2, 1, "log-log", "usual", 0.9)
    fits[[length(fits) + 1]] <- run(s, 1, 1, "none", "usual", 0.95)
  }
  cases[[length(cases) + 1]] <- list(name = s$name, taus = s$taus,
                                     robust_arg = if (is.null(s$robust)) NA else s$robust, fits = fits)
}

dataset <- function(s) {
  d <- s$d
  stop_col <- if (s$counting && s$id) d$stop else d$time
  list(stop = stop_col, status = d$status,
       start = if (s$counting) d$start else numeric(0),
       group = if (s$grouped) d$g else integer(0),
       weight = if (!is.null(s$w)) d$w else numeric(0),
       id = if (s$id) d$id else integer(0),
       cluster = if (s$cl) d$cl else integer(0))
}
datasets <- setNames(lapply(specs, dataset), sapply(specs, function(s) s$name))
versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, datasets = datasets, cases = cases),
           digits = I(17), auto_unbox = TRUE, na = "null"))
