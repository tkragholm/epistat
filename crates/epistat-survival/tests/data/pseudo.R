# Pseudo-observations by survival::pseudo (the infinitesimal jackknife) and by
# leave-one-subject-out refits of survival::survfit (the exact jackknife), and
# by the pseudo package's pseudosurv, pseudomean and pseudoci.
#
# Data: simulated with ties, competing causes 1 and 2 (status 0 is censored),
# three strata with the codes 7, 3 and 12, delayed entry, subjects with several
# rows (counting-process data with an id), whole-number and binary-exact case
# weights and a cluster column. A dataset is described by which of these it has.
# "km" cases treat any event as the event and use survfit's single-event curve,
# "cr" cases use the multi-state curve with the cause as the status.
#
# survival::pseudo is used as it is, except where it differs from the weighted
# influence its own variance uses (see `weighted_default`): with case weights and
# no subject with more than one row it leaves the weights out of the residuals,
# and with a cluster column and no id it returns one value per row, with
# the number of rows as the multiplier. For those the reference is built from
# residuals.survfit(weighted = TRUE) and collapsed by rowsum, which is what the
# Rust code means, and `r_default` holds what pseudo gives.
#
# The exact jackknife is n theta - (n - 1) theta(-i), with theta(-i) from survfit
# on the data without the subject's rows, within the subject's stratum, and n the
# number of subjects in the stratum.
# Run: Rscript pseudo.R > pseudo.json
suppressMessages({
  library(jsonlite)
  library(survival)
  library(pseudo)
})
set.seed(20261121)
v <- function(x) I(as.numeric(x))
n <- 150
group <- sample(1:3, n, replace = TRUE, prob = c(0.4, 0.35, 0.25))
code_of <- c(7L, 3L, 12L)
time <- pmin(pmax(1, ceiling(rexp(n, 0.1 * c(1, 0.7, 1.4)[group]))), 22)
status <- sample(0:2, n, replace = TRUE, prob = c(0.35, 0.4, 0.25))
entry <- pmax(0, time - sample(1:9, n, replace = TRUE))
entry[sample(n, 50)] <- 0
w_exact <- sample(c(0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3), n, replace = TRUE)
# Clusters lie within a stratum, since a subject cannot be in two curves.
cluster <- group * 100L + sample(1:20, n, replace = TRUE)

# Subjects cut at integer times into one to three rows; the last row carries the
# status. Rows of one subject share the stratum and weight.
split_rows <- function() {
  rows <- list()
  for (i in 1:n) {
    cuts <- sort(sample(seq_len(max(0, time[i] - entry[i] - 1)), min(2, max(0, time[i] - entry[i] - 1)))) + entry[i]
    if (time[i] - entry[i] <= 1) cuts <- numeric(0)
    edges <- c(entry[i], cuts, time[i])
    k <- length(edges) - 1
    rows[[i]] <- data.frame(id = i * 2L + 1L, start = edges[-(k + 1)], stop = edges[-1],
                            status = c(rep(0, k - 1), status[i]), group = code_of[group[i]],
                            weight = w_exact[i])
  }
  do.call(rbind, rows)
}
rows <- split_rows()

mkset <- function(name, strata = FALSE, entry_ = FALSE, id = FALSE, weight = FALSE, cl = FALSE) {
  if (id) {
    d <- rows
    list(name = name, stop = d$stop, start = d$start, status = d$status,
         group = if (strata) d$group else integer(0), weight = if (weight) d$weight else numeric(0),
         id = d$id, cluster = integer(0))
  } else {
    list(name = name, stop = time, start = if (entry_) entry else numeric(0), status = status,
         group = if (strata) code_of[group] else integer(0), weight = if (weight) w_exact else numeric(0),
         id = integer(0), cluster = if (cl) cluster else integer(0))
  }
}
sets <- list(
  mkset("plain"),
  mkset("strata", strata = TRUE),
  mkset("entry", entry_ = TRUE),
  mkset("id", strata = TRUE, entry_ = TRUE, id = TRUE),
  mkset("weights", weight = TRUE),
  mkset("weights_id", id = TRUE, weight = TRUE),
  mkset("cluster", weight = TRUE, cl = TRUE, strata = TRUE)
)

# The data frame of a dataset in the form survfit takes it.
frame <- function(s) {
  d <- data.frame(stop = s$stop, status = s$status)
  if (length(s$start)) d$start <- s$start
  if (length(s$group)) d$g <- s$group
  if (length(s$weight)) d$w <- s$weight
  if (length(s$id)) d$id <- s$id
  if (length(s$cluster)) d$cl <- s$cluster
  d
}

fit_of <- function(d, kind) {
  has <- function(x) x %in% names(d)
  ev <- if (kind == "km") quote(status > 0) else quote(factor(status, 0:2))
  lhs <- if (has("start")) bquote(Surv(start, stop, .(ev))) else bquote(Surv(stop, .(ev)))
  f <- eval(bquote(.(lhs) ~ .(if (has("g")) quote(g) else 1)))
  args <- list(formula = f, data = d)
  if (has("w")) args$weights <- d$w
  if (has("id")) args$id <- d$id
  else if (kind == "cr" && has("start")) args$id <- seq_len(nrow(d))
  if (has("cl")) args$cluster <- d$cl
  do.call(survfit, args)
}

state_of <- c(survival = "(s0)", cif1 = "1", cif2 = "2", rmst = "(s0)", rmtl1 = "1", rmtl2 = "2")
state_col <- c(survival = 1, cif1 = 2, cif2 = 3, rmst = 1, rmtl1 = 2, rmtl2 = 3)
is_auc <- function(target) target %in% c("rmst", "rmtl1", "rmtl2")

# The label (id, cluster or row number), and the group, of each row.
labels_of <- function(d) {
  lab <- if ("id" %in% names(d)) d$id else if ("cl" %in% names(d)) d$cl else seq_len(nrow(d))
  grp <- if ("g" %in% names(d)) d$g else rep(0L, nrow(d))
  list(label = lab, group = grp)
}

# survfit's own estimate of a target of one stratum's rows.
estimate <- function(d, kind, target, times) {
  fit <- fit_of(d, kind)
  if (!is_auc(target)) {
    sm <- summary(fit, times = times, extend = TRUE)
    if (kind == "km") return(as.numeric(sm$surv))
    return(as.numeric(sm$pstate[, state_col[[target]]]))
  }
  vapply(times, function(tt) {
    tab <- summary(fit, rmean = tt)$table
    if (kind == "km") as.numeric(tab["rmean"]) else as.numeric(tab[state_col[[target]], "rmean"])
  }, numeric(1))
}

# The residuals of one stratum's fit for a target: rows by times.
residual_matrix <- function(dg, kind, target, times) {
  fit <- fit_of(dg, kind)
  rr <- residuals(fit, times = times, type = if (is_auc(target)) "auc" else "pstate",
                  weighted = FALSE, collapse = FALSE, data.frame = TRUE)
  if (kind == "cr") rr <- rr[rr$state == state_of[[target]], ]
  matrix(rr$resid, nrow = nrow(dg))
}

build_case <- function(s, kind, target, times, method) {
  d <- frame(s)
  L <- labels_of(d)
  groups <- sort(unique(L$group))
  est <- list(); vals <- list(); labels <- c(); grp_out <- c(); nsubs <- c()
  for (g in groups) {
    rowsg <- which(L$group == g)
    dg <- d[rowsg, , drop = FALSE]
    lab_g <- L$label[rowsg]
    labs <- sort(unique(lab_g))
    nsub <- length(labs)
    theta <- estimate(dg, kind, target, times)
    m <- matrix(0, nsub, length(times))
    if (method == "jack") {
      for (j in seq_len(nsub)) {
        keep <- lab_g != labs[j]
        m[j, ] <- nsub * theta - (nsub - 1) * estimate(dg[keep, , drop = FALSE], kind, target, times)
      }
    } else {
      wt <- if ("w" %in% names(dg)) dg$w else rep(1, nrow(dg))
      infl <- residual_matrix(dg, kind, target, times) * wt
      for (j in seq_len(nsub)) {
        m[j, ] <- theta + nsub * colSums(infl[lab_g == labs[j], , drop = FALSE])
      }
    }
    est[[length(est) + 1]] <- theta; vals[[length(vals) + 1]] <- m
    labels <- c(labels, labs); grp_out <- c(grp_out, rep(g, nsub)); nsubs <- c(nsubs, nsub)
  }
  out <- list(name = s$name, kind = kind, target = target, method = method, times = v(times),
              group = v(grp_out), subject = v(labels),
              values = v(t(do.call(rbind, vals))), estimate = v(unlist(est)),
              n_subjects = v(nsubs))
  # What survival::pseudo gives itself, to show where it is the same.
  if (method == "ij") {
    rf <- tryCatch(pseudo(fit_of(d, kind), times, type = if (is_auc(target)) "auc" else "pstate",
                          data.frame = TRUE), error = function(e) NULL)
    if (!is.null(rf)) {
      if (kind == "cr") rf <- rf[rf$state == state_of[[target]], ]
      out$r_rows <- nrow(rf) / length(times)
      # Same shape: compare in the order of the subjects.
      if (out$r_rows == length(labels)) {
        id <- rf[["(id)"]]
        grp_r <- if ("curve" %in% names(rf)) sort(unique(L$group))[rf$curve] else rep(0L, nrow(rf))
        ord <- order(grp_r, id, rf$time)
        out$r_pseudo <- v(matrix(rf$pseudo[ord], ncol = length(times), byrow = TRUE) |> t())
      }
    }
  }
  out
}

times_of <- function(kind, target) c(2, 5, 9, 14, 20, 25)

cases <- list()
for (s in sets) {
  for (kind in c("km", "cr")) {
    targets <- if (kind == "km") c("survival", "rmst") else c("survival", "cif1", "cif2", "rmst", "rmtl1", "rmtl2")
    for (target in targets) for (method in c("ij", "jack")) {
      times <- times_of(kind, target)
      r <- tryCatch(build_case(s, kind, target, times, method),
                    error = function(e) { message("FAILED ", s$name, " ", kind, " ", target, " ", method, ": ", conditionMessage(e)); NULL })
      if (!is.null(r)) cases[[length(cases) + 1]] <- r
    }
  }
}

# The pseudo package on the plain data: pseudosurv and pseudomean on a binary
# event, pseudoci on the causes. The data has times from 1.
ev <- as.integer(status > 0)
tm <- c(2, 5, 9, 14, 20)
pk_surv <- pseudosurv(time, ev, tm)$pseudo
pk_mean <- sapply(tm, function(t) pseudomean(time, ev, t))
pk_ci <- pseudoci(time, status, tm)$pseudo
pkg <- list(times = v(tm), surv = v(t(pk_surv)), mean = v(t(pk_mean)),
            ci1 = v(t(pk_ci$cause1)), ci2 = v(t(pk_ci$cause2)))

datasets <- setNames(lapply(sets, function(s) list(stop = v(s$stop), start = v(s$start), status = v(s$status),
                    group = v(s$group), weight = v(s$weight), id = v(s$id), cluster = v(s$cluster))),
                    sapply(sets, function(s) s$name))
versions <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
                 pseudo = as.character(packageVersion("pseudo")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, datasets = datasets, cases = cases, pkg = pkg,
                plain = list(time = v(time), status = v(status))),
           digits = I(17), auto_unbox = TRUE, na = "null"))
