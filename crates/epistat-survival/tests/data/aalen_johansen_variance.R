# The infinitesimal jackknife standard errors of the Aalen-Johansen estimate
# from survival::survfit with a factor status: std.err for the event-free
# probability and the cumulative incidence, std.chaz for the cumulative hazards,
# std.auc for the restricted time in each state, the influence of each cluster
# (influence = TRUE) and the se(rmean) of summary(fit, rmean = tau), which
# uses that influence and so is exact for a tau between event times.
# The data are those of aalen_johansen.R: three causes, three groups, integer
# times so that events and censorings tie, delayed entry as counting-process
# data, case weights that are multiples of 1/8 (exact sums) or have two
# decimals, plus clusters of about three rows and subjects whose follow-up is
# split into two rows at an interior time.
# Run: Rscript aalen_johansen_variance.R > aalen_johansen_variance.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)
n <- 240
group <- sample(1:3, n, replace = TRUE, prob = c(0.4, 0.35, 0.25))
time <- pmax(1, ceiling(rexp(n, 0.12 * c(1, 0.7, 1.4)[group])))
time <- pmin(time, 24)
status <- sample(0:3, n, replace = TRUE, prob = c(0.3, 0.35, 0.2, 0.15))
entry <- pmax(0, time - sample(1:9, n, replace = TRUE))
entry[sample(n, 60)] <- 0
for (g in 1:3) {
  last <- group == g & time == max(time[group == g])
  status[last] <- 0
}
weight_exact <- sample(c(0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3), n, replace = TRUE)
weight_decimal <- round(runif(n, 0.5, 3), 2)
cl <- sample(1:70, n, replace = TRUE)
cl_group <- tapply(group, cl, function(x) x[1])
# a cluster lies in one group, as an id would
group <- unname(cl_group[as.character(cl)])
for (g in 1:3) {
  last <- group == g & time == max(time[group == g])
  status[last] <- 0
}
fstatus <- factor(status, levels = 0:3, labels = c("cens", "1", "2", "3"))
d0 <- data.frame(id = seq_len(n), entry, time, status, group, cl,
                 w_exact = weight_exact, w_decimal = weight_decimal)

# Split the follow-up of every third subject with room for an interior time.
split_rows <- function(d) {
  out <- list()
  for (i in seq_len(nrow(d))) {
    r <- d[i, ]
    if (i %% 3 == 0 && r$time - r$entry >= 2) {
      mid <- floor((r$entry + r$time) / 2)
      a <- r; a$time <- mid; a$status <- 0
      b <- r; b$entry <- mid
      out[[length(out) + 1]] <- a
      out[[length(out) + 1]] <- b
    } else out[[length(out) + 1]] <- r
  }
  do.call(rbind, out)
}

specs <- list(
  list("plain", d0, "none", FALSE, FALSE, "none", FALSE),
  list("groups_weights_decimal", d0, "w_decimal", TRUE, FALSE, "none", FALSE),
  list("cluster_weights_exact", d0, "w_exact", TRUE, FALSE, "cl", TRUE),
  list("entry_groups", d0, "none", TRUE, TRUE, "id", FALSE),
  list("entry_cluster_weights_decimal", d0, "w_decimal", TRUE, TRUE, "cl", TRUE),
  list("split_id_weights_exact", split_rows(d0), "w_exact", TRUE, TRUE, "id", TRUE)
)

columns <- function(m) unname(lapply(seq_len(ncol(m)), function(j) m[, j]))

cases <- list()
for (s in specs) {
  name <- s[[1]]; d <- s[[2]]; wcol <- s[[3]]; use_group <- s[[4]]
  use_entry <- s[[5]]; idcol <- s[[6]]; keep_influence <- s[[7]]
  d$fstatus <- factor(d$status, levels = 0:3, labels = c("cens", "1", "2", "3"))
  lhs <- if (use_entry) "Surv(entry, time, fstatus)" else "Surv(time, fstatus)"
  rhs <- if (use_group) "factor(group)" else "1"
  args <- list(formula = as.formula(paste(lhs, "~", rhs)), data = d,
               influence = TRUE)
  w <- if (wcol == "none") rep(1, nrow(d)) else d[[wcol]]
  if (wcol != "none") args$weights <- w
  # cluster given alone, an id alone, or neither (competing risks, no id)
  if (name %in% c("cluster_weights_exact", "entry_cluster_weights_decimal")) {
    args$cluster <- d$cl
    if (use_entry) args$id <- d$id
  } else if (idcol == "id") {
    args$id <- d$id
  } else if (use_entry) args$id <- seq_len(nrow(d))
  fit <- do.call(survfit, args)
  stopifnot(identical(fit$states, c("(s0)", "1", "2", "3")))
  ngrp <- if (is.null(fit$strata)) 1 else length(fit$strata)
  ends <- cumsum(if (is.null(fit$strata)) length(fit$time) else fit$strata)
  begins <- c(1, head(ends, -1) + 1)
  groups <- list()
  tau_all <- NULL
  for (k in seq_len(ngrp)) {
    rows <- begins[k]:ends[k]
    tt <- fit$time[rows]
    ent <- list(
      group = k - 1L, time = tt,
      pstate = columns(fit$pstate[rows, , drop = FALSE]),
      std_err = columns(fit$std.err[rows, , drop = FALSE]),
      cumhaz = columns(fit$cumhaz[rows, , drop = FALSE]),
      std_chaz = columns(fit$std.chaz[rows, , drop = FALSE]),
      std_auc = columns(fit$std.auc[rows, , drop = FALSE])
    )
    if (keep_influence) {
      inf <- if (ngrp == 1) fit$influence.pstate else fit$influence.pstate[[k]]
      # cluster x time x state; rownames are positions in the order the
      # clusters first appear, which index the labels
      pos <- as.integer(rownames(inf))
      cluster_all <- if (!is.null(args$cluster)) args$cluster
                     else if (!is.null(args$id)) args$id else seq_len(nrow(d))
      labels <- unique(cluster_all)
      ent$uclust <- labels[pos]
      ent$influence <- lapply(seq_len(dim(inf)[3]), function(j)
        lapply(seq_len(dim(inf)[1]), function(g) inf[g, , j]))
    }
    groups[[k]] <- ent
  }
  last_t <- max(fit$time)
  taus <- sort(unique(c(fit$time[1], fit$time[3], fit$time[3] + 0.5,
                        last_t - 1, last_t, last_t + 4)))
  rmean <- lapply(taus, function(tau) {
    tab <- summary(fit, rmean = tau)$table
    list(rmean = unname(matrix(tab[, "rmean"], nrow = ngrp)),
         se = unname(matrix(tab[, "se(rmean)"], nrow = ngrp)))
  })
  cases[[length(cases) + 1]] <- list(
    name = name,
    time = d$time, status = d$status, entry = d$entry, group = d$group - 1L,
    id = d$id, cluster = d$cl, weight = w, use_entry = use_entry,
    use_group = use_group, uses = if (name %in% c("cluster_weights_exact",
      "entry_cluster_weights_decimal")) "cluster" else
      if (idcol == "id") "id" else "none",
    cname = colnames(fit$cumhaz), t0 = fit$t0,
    groups = groups, tau = taus,
    rmean = lapply(rmean, function(x) lapply(seq_len(nrow(x$rmean)), function(g)
      list(rmean = x$rmean[g, ], se = x$se[g, ])))
  )
}
versions <- list(R = R.version.string,
                 survival = as.character(packageVersion("survival")),
                 jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = versions, cases = cases), digits = I(17),
           auto_unbox = TRUE, na = "null"))
