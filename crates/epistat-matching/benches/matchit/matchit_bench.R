# Wall time of MatchIt (and optmatch and Epi) on the cohort that
# benches/matchit_compare.rs writes, for the same methods and the same options.
#
# Usage: Rscript matchit_bench.R N METHOD SCRATCH
#   N        number of units, as in SCRATCH/cohort_N.csv
#   METHOD   load | nn_ps | nn_ps_cal | nn_strata | nn_cal_strata | nn_1to3 |
#            nn_ps_r4 | nn_strata_r4 | nn_replace | nn_mahal | nn_mahal_strata |
#            nn_euclid | nn_euclid_strata | subclass | exact | cem | quick |
#            optimal_cal | full_cal | cardinality | profile | genetic_eval |
#            genetic_short | riskset | riskset_replace | balance | fit_logit |
#            fit_probit | fit_enet_cv | fit_cbps | effects | abadie |
#            sens_wilcoxon | sens_hl | sens_mscore
#   SCRATCH  directory with the cohort file; results go to OUT, by default
#            SCRATCH/out
#
# The environment variable REPS sets the number of timed repetitions (default
# 3), CALIPER_RAW the caliper of optimal_cal and full_cal (default 0.2, as in
# the Rust program). Only the matching is timed, as in the Rust program: the
# model fit, the distances and the selection. Reading the file is not. The
# time limit is set from outside by the driver (run.sh).
suppressMessages({
  library(data.table)
  library(MatchIt)
})
args <- commandArgs(TRUE)
n <- args[1]
method <- args[2]
scratch <- args[3]
reps <- as.integer(Sys.getenv("REPS", "3"))
caliper_raw <- as.numeric(Sys.getenv("CALIPER_RAW", "0.2"))
out_dir <- Sys.getenv("OUT", file.path(scratch, "out"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

d <- as.data.frame(fread(file.path(scratch, sprintf("cohort_%s.csv", n))))
d$f3 <- factor(d$x3)
d$f5 <- factor(d$x5)
f <- treat ~ x1 + x2 + x3 + x4 + x5
if (method == "load") {
  cat(sprintf("loaded %d rows, %d treated\n", nrow(d), sum(d$treat)))
  quit(save = "no")
}

# Pairs of treated and control row numbers from a match matrix, sorted.
pairs_of <- function(m) {
  mm <- m$match.matrix
  out <- data.table(anchor = integer(), control = integer())
  for (k in seq_len(ncol(mm))) {
    ok <- !is.na(mm[, k])
    out <- rbind(out, data.table(anchor = as.integer(rownames(mm)[ok]),
                                 control = as.integer(mm[ok, k])))
  }
  setorder(out, anchor, control)
  out
}
n_matched_anchors <- function(m) sum(rowSums(!is.na(m$match.matrix)) > 0)

run_nearest <- function(...) matchit(f, data = d, method = "nearest", ...)
# Each returns the object whose time is taken.
call <- switch(method,
  nn_ps = function() run_nearest(distance = "glm"),
  nn_ps_cal = function() run_nearest(distance = "glm", caliper = 0.2),
  nn_strata = function() run_nearest(distance = "glm", exact = ~s),
  nn_cal_strata = function() run_nearest(distance = "glm", caliper = 0.2, exact = ~s),
  nn_1to3 = function() run_nearest(distance = "glm", ratio = 3),
  nn_ps_r4 = function() run_nearest(distance = "glm", ratio = 4),
  nn_strata_r4 = function() run_nearest(distance = "glm", exact = ~s, ratio = 4),
  nn_replace = function() run_nearest(distance = "glm", replace = TRUE),
  nn_mahal = function() run_nearest(distance = "mahalanobis"),
  nn_mahal_strata = function() run_nearest(distance = "mahalanobis", exact = ~s),
  nn_euclid = function() run_nearest(distance = "scaled_euclidean"),
  nn_euclid_strata = function() run_nearest(distance = "scaled_euclidean", exact = ~s),
  cardinality = function() matchit(ctreat ~ x1 + x2 + x3 + x4 + x5, data = d,
                                   method = "cardinality", ratio = 1, tols = 0.05,
                                   std.tols = TRUE, solver = "highs", time = 120),
  profile = function() matchit(ctreat ~ x1 + x2 + x3 + x4 + x5, data = d,
                               method = "cardinality", ratio = NA, tols = 0.05,
                               std.tols = TRUE, solver = "highs", time = 120),
  genetic_eval = function() suppressWarnings(
    matchit(f, data = d, method = "genetic", pop.size = 1, max.generations = 0,
            wait.generations = 0, distance = "glm")),
  genetic_short = function() suppressWarnings(
    matchit(f, data = d, method = "genetic", pop.size = 10, max.generations = 2,
            wait.generations = 1, hard.generation.limit = TRUE, distance = "glm")),
  subclass = function() matchit(f, data = d, method = "subclass", subclass = 6,
                                distance = "glm"),
  exact = function() matchit(treat ~ s + x3 + x5, data = d, method = "exact"),
  cem = function() matchit(treat ~ x1 + x2 + f3 + f5, data = d, method = "cem",
                           cutpoints = list(x1 = "q5", x2 = "q5")),
  quick = function() matchit(f, data = d, method = "quick", distance = "glm"),
  NULL)

# optmatch on the propensity score within the fine strata g, with the caliper
# as forbidden pairs, which is how matchit(method = "optimal") arranges the
# call, with matchit()'s default tol of 1e-7. matchit() itself ignores a caliper
# for these methods.
if (method %in% c("optimal_cal", "full_cal")) {
  suppressMessages(library(optmatch))
  options(fullmatch_try_recovery = FALSE)
  call <- function() {
    fit <- glm(f, data = d, family = binomial())
    ps <- as.numeric(fitted(fit))
    names(ps) <- rownames(d)
    z <- setNames(d$treat, rownames(d))
    em <- exactMatch(z ~ g, data = d)
    dm <- match_on(ps, z = z, within = em, caliper = caliper_raw)
    if (method == "optimal_cal") pairmatch(dm, data = d, tol = 1e-7) else fullmatch(dm, data = d, tol = 1e-7)
  }
}

elapsed <- function(expr_fn) {
  gc(FALSE)
  t0 <- proc.time()[["elapsed"]]
  res <- expr_fn()
  list(res = res, secs = proc.time()[["elapsed"]] - t0)
}

times <- numeric()
res <- NULL
matched <- NA_integer_

# Model fits: the propensity score alone.
if (method == "fit_logit") call <- function() glm(f, data = d, family = binomial())
if (method == "fit_probit") call <- function() glm(f, data = d, family = binomial("probit"))
if (method == "fit_enet_cv") {
  suppressMessages(library(glmnet))
  xm <- model.matrix(f, d)[, -1, drop = FALSE]
  call <- function() cv.glmnet(xm, d$treat, family = "binomial", alpha = 1, nfolds = 10)
}
if (method == "fit_cbps") {
  suppressMessages(library(CBPS))
  call <- function() suppressMessages(CBPS(f, data = d, ATT = 1, method = "over",
                                          print.level = 0))
}

# Effects, Abadie-Imbens and sensitivity bounds start from a 1:1 nearest match
# on the propensity score, as the Rust program does. Only what follows is timed.
if (method %in% c("effects", "abadie", "sens_wilcoxon", "sens_hl", "sens_mscore")) {
  m0 <- run_nearest(distance = "glm")
  mm <- m0$match.matrix
  ok <- !is.na(mm[, 1])
  tr_rows <- as.integer(rownames(mm)[ok])
  co_rows <- as.integer(mm[ok, 1])
  gam <- 3
}
if (method == "effects") {
  suppressMessages({library(sandwich)})
  call <- function() {
    md <- match.data(m0)
    fit <- lm(y ~ treat + x1 + x2, data = md, weights = weights)
    vcovCL(fit, cluster = ~subclass)
  }
}
if (method == "abadie") {
  suppressMessages(library(Matching))
  call <- function() Match(Y = d$y, Tr = d$treat, X = as.matrix(d[, c("x1", "x2")]),
                           estimand = "ATT", M = 1, BiasAdjust = TRUE, Var.calc = 2)
}
if (method == "sens_wilcoxon") {
  suppressMessages(library(rbounds))
  call <- function() psens(d$y[tr_rows], d$y[co_rows], Gamma = gam, GammaInc = 0.5)
}
if (method == "sens_hl") {
  suppressMessages(library(rbounds))
  call <- function() hlsens(d$y[tr_rows], d$y[co_rows], Gamma = gam, GammaInc = 0.5)
}
if (method == "sens_mscore") {
  suppressMessages(library(sensitivitymv))
  ymv <- cbind(d$y[tr_rows], d$y[co_rows])
  call <- function() lapply(c(1, 1.5, 2, 2.5, 3), function(g) senmv(ymv, gamma = g))
}

if (method %in% c("riskset", "riskset_replace")) {
  suppressMessages(library(Epi))
  # ccwc keeps a person with exit equal to the failure time in the risk set, so
  # a person who dies on day t leaves on day t - 1 to be out on day t.
  d$exit <- ifelse(!is.na(d$t_case), d$t_case,
                   ifelse(!is.na(d$t_death), d$t_death - 1, 99999))
  d$fail <- as.integer(!is.na(d$t_case))
  call <- function() {
    suppressWarnings(ccwc(entry = 0, exit = exit, fail = fail, origin = 0,
                          controls = 1, match = list(s), include = list(id),
                          data = d, silent = TRUE))
  }
}

if (method == "balance") {
  m0 <- run_nearest(distance = "glm")
  for (r in seq_len(reps)) {
    e <- elapsed(function() summary(m0))
    times <- c(times, e$secs)
    res <- e$res
    if (sum(times) > 20) break
  }
  s <- res$sum.matched
  fwrite(data.table(name = rownames(s), smd_post = s[, "Std. Mean Diff."],
                    var_ratio_post = s[, "Var. Ratio"],
                    ecdf_mean_post = s[, "eCDF Mean"]),
         file.path(out_dir, sprintf("r_balance_%s.csv", n)))
  su <- res$sum.all
  fwrite(data.table(name = rownames(su), smd_pre = su[, "Std. Mean Diff."]),
         file.path(out_dir, sprintf("r_balance_pre_%s.csv", n)))
  matched <- n_matched_anchors(m0)
} else {
  if (is.null(call)) stop("unknown method ", method)
  for (r in seq_len(reps)) {
    e <- elapsed(call)
    times <- c(times, e$secs)
    res <- e$res
    if (sum(times) > 20) break
    if (r < reps) rm(res)
  }
  if (method %in% c("nn_ps", "nn_ps_cal", "nn_strata", "nn_cal_strata", "nn_1to3",
                    "nn_ps_r4", "nn_strata_r4", "nn_replace", "nn_mahal",
                    "nn_mahal_strata", "nn_euclid", "nn_euclid_strata")) {
    p <- pairs_of(res)
    matched <- n_matched_anchors(res)
    fwrite(p, file.path(out_dir, sprintf("r_%s_%s.csv", method, n)))
  } else if (method %in% c("cardinality", "profile")) {
    matched <- sum(res$weights > 0)
  } else if (method %in% c("genetic_eval", "genetic_short")) {
    matched <- n_matched_anchors(res)
  } else if (method %in% c("subclass", "exact", "cem", "quick")) {
    ok <- !is.na(res$subclass)
    matched <- sum(ok)
    fwrite(data.table(id = as.integer(rownames(d))[ok], subclass = as.integer(res$subclass[ok])),
           file.path(out_dir, sprintf("r_%s_%s.csv", method, n)))
  } else if (method == "optimal_cal") {
    ids <- as.integer(rownames(d))
    g <- as.character(res)[match(rownames(d), names(res))]
    ok <- !is.na(g)
    tr <- data.table(g = g[ok & d$treat == 1], anchor = ids[ok & d$treat == 1])
    co <- data.table(g = g[ok & d$treat == 0], control = ids[ok & d$treat == 0])
    p <- merge(tr, co, by = "g")[, .(anchor, control)]
    setorder(p, anchor, control)
    matched <- length(unique(p$anchor))
    fwrite(p, file.path(out_dir, sprintf("r_%s_%s.csv", method, n)))
  } else if (method == "full_cal") {
    g <- as.character(res)[match(rownames(d), names(res))]
    ok <- !is.na(g)
    ids <- as.integer(rownames(d))
    matched <- sum(ok)
    fwrite(data.table(id = ids[ok], subclass = as.integer(factor(g[ok]))),
           file.path(out_dir, sprintf("r_%s_%s.csv", method, n)))
  } else if (method %in% c("riskset", "riskset_replace")) {
    # Cases in a set that got at least one control. Cases that fail on the same
    # day in the same stratum share a set.
    sets <- as.data.table(res)
    with_controls <- sets[Fail == 0, unique(Set)]
    matched <- sets[Fail == 1 & Set %in% with_controls, .N]
  }
}
med <- median(times)
cat(sprintf("n %s %s: median %.3f s, min %.3f s over %d runs, matched %s\n",
            n, method, med, min(times), length(times), matched))
cat(sprintf("%s,%s,%d,%.4f,%.4f,%s\n", n, method, length(times), med, min(times), matched),
    file = file.path(out_dir, "r_times.csv"), append = TRUE)
