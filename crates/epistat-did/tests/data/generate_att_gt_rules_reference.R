# did::att_gt references for the IPW and outcome-regression routes and for
# att_gt's pre-processing, did 2.5.1 and DRDID 1.3.0.
#
# Run from the crate root:  Rscript tests/data/generate_att_gt_rules_reference.R
#
# Writes att_gt_rules_ref.json. Every run uses bstrap = FALSE and
# cband = FALSE, so the standard errors are the analytic ones, and the default
# faster_mode = TRUE route. Each run stores the cells, the estimates, the
# standard errors, n, the influence matrix (row-major, units by cells) and the
# ids its rows stand for: the unit ids on a panel, and on repeated
# cross-sections the row numbers of the data as given (no row is dropped
# there). The data are drawn here and stored with the runs.

suppressMessages({
  library(did)
  library(jsonlite)
})
stopifnot(packageVersion("did") == "2.5.1",
          packageVersion("DRDID") == "1.3.0",
          packageVersion("jsonlite") == "2.0.0")
cat(R.version.string, "\n")

# A panel of n units over `periods`, cohort drawn from `cohorts` (0 for never
# treated) with probabilities that depend on x1, a time-invariant x2, a weight
# per unit and an effect that grows with time since treatment. With
# `varying_x` the continuous covariate changes over time; with
# `varying_weight` so does the weight.
sim_panel <- function(n, periods, cohorts, seed, varying_x = FALSE,
                      varying_weight = FALSE) {
  set.seed(seed)
  x1 <- rnorm(n)
  x2 <- rbinom(n, 1, 0.4)
  k <- length(cohorts)
  scores <- sapply(seq_len(k), function(j) 0.4 * (j - (k + 1) / 2) * x1)
  probs <- exp(scores) / rowSums(exp(scores))
  g <- apply(probs, 1, function(p) cohorts[sample.int(k, 1, prob = p)])
  w <- runif(n, 0.5, 2)
  alpha <- rnorm(n)
  rows <- expand.grid(id = seq_len(n), period = periods)
  rows <- rows[order(rows$id, rows$period), ]
  i <- rows$id
  t <- rows$period
  x <- x1[i] + if (varying_x) 0.3 * (t - min(periods)) * rnorm(length(i)) else 0
  ww <- w[i] * if (varying_weight) runif(length(i), 0.8, 1.25) else 1
  treated <- g[i] > 0 & t >= g[i]
  y <- alpha[i] + 0.5 * x + 0.3 * x2[i] + 0.2 * t + 0.15 * t * x1[i] +
    treated * (1 + 0.5 * (t - g[i])) + rnorm(length(i))
  # Four decimals keep the stored data short; the runs read the rounded values.
  data.frame(id = i, period = t, g = g[i], y = round(y, 4), x1 = round(x, 4),
             x2 = x2[i], w = round(ww, 4))
}

run <- function(data, panel = TRUE, est_method = "dr", control_group = "nevertreated",
                base_period = "varying", anticipation = 0,
                allow_unbalanced_panel = FALSE, xformla = ~ x1 + x2, aggregate = FALSE) {
  notes <- character(0)
  out <- withCallingHandlers(
    att_gt(yname = "y", tname = "period", idname = if (panel) "id" else NULL,
           gname = "g", xformla = xformla, data = data, panel = panel,
           allow_unbalanced_panel = allow_unbalanced_panel,
           control_group = control_group, anticipation = anticipation,
           weightsname = "w", bstrap = FALSE, cband = FALSE,
           est_method = est_method, base_period = base_period),
    warning = function(w) {
      notes <<- c(notes, conditionMessage(w))
      invokeRestart("muffleWarning")
    },
    message = function(m) {
      notes <<- c(notes, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
  dp <- out$DIDparams
  ids <- unique(dp$time_invariant_data[[dp$idname]])
  # aggte with max_e = 1, which did counts in steps of its period list for the
  # simple and group summaries and in calendar periods for the dynamic one.
  aggregates <- NULL
  if (aggregate) {
    grab <- function(type) {
      a <- suppressWarnings(aggte(out, type = type, max_e = 1, bstrap = FALSE, cband = FALSE))
      vec <- function(x) I(if (is.null(x)) numeric(0) else as.numeric(x))
      list(overall_att = a$overall.att, overall_se = a$overall.se, egt = vec(a$egt),
           att_egt = vec(a$att.egt), se_egt = vec(a$se.egt))
    }
    aggregates <- list(simple = grab("simple"), group = grab("group"),
                       dynamic = grab("dynamic"))
  }
  list(
    aggregates = aggregates,
    panel = panel, est_method = est_method, control_group = control_group,
    base_period = base_period, anticipation = anticipation,
    allow_unbalanced_panel = allow_unbalanced_panel,
    group = out$group, t = out$t, att = out$att, se = out$se, n = out$n,
    inffunc = as.matrix(out$inffunc), ids = as.numeric(ids), notes = I(notes)
  )
}

ref <- list()

# 1. IPW and outcome regression on a balanced panel and on repeated
#    cross-sections, with two covariates and weights.
balanced <- sim_panel(120, 1:4, c(0, 2, 3, 4), 101)
ref$balanced <- balanced
ref$methods_panel <- list(
  run(balanced, est_method = "ipw", base_period = "universal"),
  run(balanced, est_method = "ipw", control_group = "notyettreated"),
  run(balanced, est_method = "reg", base_period = "universal",
      control_group = "notyettreated"),
  run(balanced, est_method = "reg")
)
set.seed(102)
repeated <- sim_panel(320, 1:4, c(0, 2, 3, 4), 103)
repeated <- repeated[sample.int(nrow(repeated), 320), ]
repeated$id <- NULL
rownames(repeated) <- NULL
ref$repeated <- repeated
ref$methods_rc <- list(
  run(repeated, panel = FALSE, est_method = "ipw", base_period = "universal"),
  run(repeated, panel = FALSE, est_method = "ipw", control_group = "notyettreated"),
  run(repeated, panel = FALSE, est_method = "reg", base_period = "universal",
      control_group = "notyettreated"),
  run(repeated, panel = FALSE, est_method = "reg"),
  run(repeated, panel = FALSE, est_method = "dr")
)

# 2. Pre-processing: periods with gaps, a cohort treated in the first period,
#    one treated at the second (dropped only with anticipation) and one after
#    the last period.
gaps <- sim_panel(160, c(1, 2, 4, 5, 7), c(0, 1, 2, 4, 5, 9), 201)
ref$gaps <- gaps
ref$gaps_counts <- list(
  treated_first = length(unique(gaps$id[gaps$g == 1])),
  treated_second = length(unique(gaps$id[gaps$g == 2])),
  after_last = length(unique(gaps$id[gaps$g == 9]))
)
ref$gaps_runs <- list(
  run(gaps, aggregate = TRUE),
  run(gaps, base_period = "universal", control_group = "notyettreated", aggregate = TRUE),
  run(gaps, anticipation = 1),
  run(gaps, anticipation = 1, base_period = "universal",
      control_group = "notyettreated", est_method = "reg")
)

# 3. No never-treated units: the last cohort is the comparison group.
late <- sim_panel(150, 1:6, c(3, 4, 5), 301)
ref$late <- late
ref$late_runs <- list(
  run(late, control_group = "notyettreated"),
  run(late, control_group = "nevertreated"),
  run(late, control_group = "notyettreated", anticipation = 1,
      base_period = "universal")
)

# 4. An unbalanced panel: a fifth of the units miss one period.
unbalanced <- sim_panel(150, 1:5, c(0, 3, 4), 401)
set.seed(402)
missing_units <- sample(unique(unbalanced$id), 30)
missing_period <- sample(1:5, 30, replace = TRUE)
drop <- paste(unbalanced$id, unbalanced$period) %in% paste(missing_units, missing_period)
unbalanced <- unbalanced[!drop, ]
rownames(unbalanced) <- NULL
ref$unbalanced <- unbalanced
ref$unbalanced_counts <- list(missing_units = length(missing_units))
ref$unbalanced_runs <- list(
  run(unbalanced),
  run(unbalanced, allow_unbalanced_panel = TRUE),
  run(unbalanced, allow_unbalanced_panel = TRUE, est_method = "ipw",
      base_period = "universal"),
  run(unbalanced, allow_unbalanced_panel = TRUE, est_method = "reg",
      control_group = "notyettreated")
)

# 5. A time-varying covariate and time-varying weights under a universal
#    base: which period's covariates and weights each cell uses.
varying <- sim_panel(150, 1:5, c(0, 3, 4), 501, varying_x = TRUE,
                     varying_weight = TRUE)
ref$varying <- varying
ref$varying_runs <- list(
  run(varying, base_period = "universal"),
  run(varying, base_period = "varying", est_method = "ipw")
)

# 6. Anticipation on mpdta, with lpop.
data(mpdta)
run_mpdta <- function(...) {
  out <- att_gt(yname = "lemp", tname = "year", idname = "countyreal",
                gname = "first.treat", xformla = ~ lpop, data = mpdta,
                bstrap = FALSE, cband = FALSE, ...)
  list(group = out$group, t = out$t, att = out$att, se = out$se, n = out$n)
}
ref$mpdta_anticipation <- list(
  run_mpdta(anticipation = 1),
  run_mpdta(anticipation = 1, base_period = "universal",
            control_group = "notyettreated")
)

# The influence matrices of the runs below differ from a stored one only in
# what the estimates and standard errors already check, so they are left out
# to keep the file small.
for (path in list(c("methods_rc", 2), c("methods_rc", 4), c("methods_rc", 5),
                  c("gaps_runs", 2), c("gaps_runs", 4), c("late_runs", 2),
                  c("late_runs", 3), c("varying_runs", 2))) {
  ref[[path[1]]][[as.integer(path[2])]]["inffunc"] <- list(NULL)
}

write_json(ref, "tests/data/att_gt_rules_ref.json", digits = 17,
           auto_unbox = TRUE, matrix = "rowmajor", na = "null", null = "null")
cat("att_gt_rules_ref.json written\n")
