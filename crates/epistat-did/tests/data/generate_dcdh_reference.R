# Reference values for the de Chaisemartin and D'Haultfoeuille event-study
# estimator (`epistat_did::estimate_dcdh`), read by tests/parity_dcdh.rs.
#
# The reference is DIDmultiplegtDYN::did_multiplegt_dyn. The current CRAN
# release, 2.4.0, needs the R package polars, which is not on CRAN, so it
# cannot run here. This script runs 2.2.0, the last release without polars,
# from the CRAN archive, installed into its own library:
#
#   R -e 'install.packages(c("matlib", "plm"))'
#   R -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/DIDmultiplegtDYN/DIDmultiplegtDYN_2.2.0.tar.gz",
#         repos = NULL, type = "source", lib = "~/.cache/r-didmultiplegtdyn-2.2.0")'
#   DCDH_LIB=~/.cache/r-didmultiplegtdyn-2.2.0 Rscript generate_dcdh_reference.R
#
# Where 2.4.0 computes something else than 2.2.0 for the options tested here,
# the test file says so.
#
# Each case is a simulated panel. The script records the package's results
# (`$results`) and, from a second call of the internal
# `did_multiplegt_main`, the variance terms of every group: the columns
# `U_Gg_var_glob_<l>_XX`, `U_Gg_var_glob_pl_<l>_XX` and `U_Gg_var_global_XX`
# on the group's first row. With a cluster the package overwrites those
# columns with their sums by cluster, so the script records the cluster sums
# instead.

lib <- Sys.getenv("DCDH_LIB", path.expand("~/.cache/r-didmultiplegtdyn-2.2.0"))
suppressMessages(library(DIDmultiplegtDYN, lib.loc = lib))
suppressMessages(library(jsonlite))
suppressMessages(library(data.table))
pkg_version <- as.character(packageVersion("DIDmultiplegtDYN", lib.loc = lib))
stopifnot(pkg_version == "2.2.0")

# The outcome: a unit effect, a common trend, the effect of the current and
# the previous period's treatment, an optional control and noise.
outcome_of <- function(d, beta_x = 0) {
  d <- d[order(d$g, d$t), ]
  lag_d <- ave(d$D, d$g, FUN = function(x) c(0, head(x, -1)))
  unit <- rnorm(max(d$g))[d$g]
  x <- if ("x1" %in% names(d)) d$x1 else 0
  d$Y <- unit + 0.3 * d$t + 1.0 * d$D + 0.5 * lag_d + beta_x * x + rnorm(nrow(d))
  d
}

# Absorbing binary treatment: first treated in period 3, 5 or 7, or never.
sim_absorbing <- function(G, T) {
  d <- expand.grid(t = 1:T, g = 1:G)
  first <- sample(c(3, 5, 7, 99), G, replace = TRUE)
  d$D <- as.numeric(d$t >= first[d$g])
  d
}

# Binary treatment that goes on and off: the baseline is 0 or 1, the first
# change is at a random period or never, and after it the treatment flips
# with probability 0.3 each period.
sim_on_off <- function(G, T, p1 = 0.4) {
  rows <- vector("list", G)
  for (g in 1:G) {
    d1 <- rbinom(1, 1, p1)
    first <- sample(c(2:T, 99, 99), 1)
    D <- rep(d1, T)
    if (first <= T) {
      D[first] <- 1 - d1
      if (first < T) {
        for (t in (first + 1):T) D[t] <- if (runif(1) < 0.3) 1 - D[t - 1] else D[t - 1]
      }
    }
    rows[[g]] <- data.frame(t = 1:T, g = g, D = D)
  }
  do.call(rbind, rows)
}

# A treatment with levels 0, 1 and 2: the baseline is any level, and after
# the first change the level moves by one step with probability 0.3.
sim_three_levels <- function(G, T) {
  rows <- vector("list", G)
  for (g in 1:G) {
    d1 <- sample(0:2, 1)
    first <- sample(c(2:T, 99, 99), 1)
    D <- rep(d1, T)
    if (first <= T) {
      D[first] <- if (d1 == 0) 1 else if (d1 == 2) 1 else sample(c(0, 2), 1)
      if (first < T) {
        for (t in (first + 1):T) {
          step <- if (runif(1) < 0.3) sample(c(-1, 1), 1) else 0
          D[t] <- min(2, max(0, D[t - 1] + step))
        }
      }
    }
    rows[[g]] <- data.frame(t = 1:T, g = g, D = D)
  }
  do.call(rbind, rows)
}

# Removes rows, and sets outcomes and treatments to missing, at random.
unbalance <- function(d, drop = 0.08, y_na = 0.05, d_na = 0.03) {
  d <- d[runif(nrow(d)) > drop, ]
  d$Y[runif(nrow(d)) < y_na] <- NA
  d$D[runif(nrow(d)) < d_na] <- NA
  d
}

run_case <- function(name, d, args) {
  d <- as.data.frame(d)
  base <- list(df = d, outcome = "Y", group = "g", time = "t", treatment = "D",
               graph_off = TRUE)
  full <- c(base, args)
  res <- suppressMessages(suppressWarnings(do.call(did_multiplegt_dyn, full)))$results

  # The variance terms by group, from the internal main routine.
  margs <- list(df = d, outcome = "Y", group = "g", time = "t", treatment = "D",
                effects = full$effects, placebo = if (is.null(full$placebo)) 0 else full$placebo,
                ci_level = 95, switchers = if (is.null(full$switchers)) "" else full$switchers,
                only_never_switchers = isTRUE(full$only_never_switchers),
                trends_nonparam = full$trends_nonparam, weight = full$weight,
                controls = full$controls,
                dont_drop_larger_lower = isTRUE(full$dont_drop_larger_lower),
                drop_if_d_miss_before_first_switch = isTRUE(full$drop_if_d_miss_before_first_switch),
                cluster = full$cluster, same_switchers = FALSE, same_switchers_pl = FALSE,
                effects_equal = isTRUE(full$effects_equal), save_results = NULL,
                normalized = isTRUE(full$normalized), predict_het = NULL, trends_lin = FALSE,
                less_conservative_se = FALSE, continuous = NULL)
  main <- suppressMessages(suppressWarnings(do.call(DIDmultiplegtDYN:::did_multiplegt_main, margs)))
  df <- as.data.table(main$df)
  # The balancing leaves the input's group column empty in the cells it
  # adds; recover it from the package's group index.
  map <- unique(df[!is.na(group), .(group_XX, gid = group)])
  df <- merge(df, map, by = "group_XX", all.x = TRUE)
  df[, group := gid]
  setorder(df, group_XX, time_XX)
  l <- main$l_XX
  lp <- if (is.null(main$l_placebo_XX)) 0 else main$l_placebo_XX
  clustered <- !is.null(full$cluster)
  key <- if (clustered) "cluster_XX" else "group"
  first <- if (clustered) df[first_obs_by_clust_XX == 1] else df[first_obs_by_gp_XX == 1]
  first <- first[order(get(key))]
  col_or_na <- function(col) if (col %in% names(first)) first[[col]] else rep(NA_real_, nrow(first))
  var_terms <- list(
    ids = first[[key]],
    effects = lapply(seq_len(l), function(i) col_or_na(sprintf("U_Gg_var_glob_%d_XX", i))),
    placebos = lapply(seq_len(lp), function(i) col_or_na(sprintf("U_Gg_var_glob_pl_%d_XX", i))),
    average = col_or_na("U_Gg_var_global_XX")
  )
  if (clustered) {
    # The average's column is not summed by cluster in the main routine.
    # With a cluster the main routine overwrites first_obs_by_gp_XX by the
    # first row of each cluster, so the groups' first rows are found anew.
    df[, first_of_group := seq_len(.N) == 1, by = group_XX]
    s <- df[first_of_group == TRUE, .(v = sum(U_Gg_var_global_XX)), by = cluster_XX][order(cluster_XX)]
    var_terms$average <- s$v
  }

  # The condition number of the Gram matrix of the controls' centred first
  # differences, which bounds how far two solvers of the theta_d regressions
  # can differ.
  kappa_x <- NULL
  if (!is.null(full$controls)) {
    o <- d[order(d$g, d$t), ]
    fd <- sapply(full$controls, function(v) ave(o[[v]], o$g, FUN = function(x) c(NA, diff(x))))
    fd <- scale(fd[complete.cases(fd), , drop = FALSE], scale = FALSE)
    kappa_x <- kappa(crossprod(fd), exact = TRUE)
  }

  mat <- function(m) if (is.null(m)) NULL else unname(lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ])))
  list(
    name = name,
    args = args[setdiff(names(args), "df")],
    data = list(g = d$g, t = d$t, D = d$D, Y = d$Y,
                w = if (is.null(full$weight)) NULL else d[[full$weight]],
                x1 = if ("x1" %in% names(d)) d$x1 else NULL,
                x2 = if ("x2" %in% names(d)) d$x2 else NULL,
                stratum = if ("s" %in% names(d)) d$s else NULL,
                cluster = if ("cl" %in% names(d)) d$cl else NULL),
    effects = mat(res$Effects),
    ate = as.numeric(res$ATE[1, ]),
    placebos = mat(res$Placebos),
    p_jointeffects = if (is.null(res$p_jointeffects)) NULL else res$p_jointeffects,
    p_jointplacebo = if (is.null(res$p_jointplacebo)) NULL else res$p_jointplacebo,
    p_equality_effects = if (is.null(res$p_equality_effects)) NULL else res$p_equality_effects,
    delta_D_avg_total = res$delta_D_avg_total,
    kappa_x = kappa_x,
    var_terms = var_terms
  )
}

cases <- list()
G <- 80; T <- 8

set.seed(101)
d <- outcome_of(sim_absorbing(G, T))
cases[[length(cases) + 1]] <- run_case("absorbing", d, list(effects = 4, placebo = 3))

set.seed(102)
d <- outcome_of(sim_on_off(G, T))
cases[[length(cases) + 1]] <- run_case("on_off", d, list(effects = 3, placebo = 2))
cases[[length(cases) + 1]] <- run_case("on_off_in", d, list(effects = 3, placebo = 2, switchers = "in"))
cases[[length(cases) + 1]] <- run_case("on_off_out", d, list(effects = 3, placebo = 2, switchers = "out"))
cases[[length(cases) + 1]] <- run_case("on_off_never", d, list(effects = 3, placebo = 2, only_never_switchers = TRUE))
cases[[length(cases) + 1]] <- run_case("on_off_normalized", d,
  list(effects = 3, placebo = 2, normalized = TRUE, effects_equal = TRUE))

set.seed(103)
d <- outcome_of(sim_three_levels(G, T))
cases[[length(cases) + 1]] <- run_case("three_levels", d, list(effects = 3, placebo = 2))
cases[[length(cases) + 1]] <- run_case("three_levels_keep", d,
  list(effects = 3, placebo = 2, dont_drop_larger_lower = TRUE, normalized = TRUE))

set.seed(104)
d <- unbalance(outcome_of(sim_on_off(G, T)))
cases[[length(cases) + 1]] <- run_case("unbalanced", d, list(effects = 3, placebo = 2))
cases[[length(cases) + 1]] <- run_case("unbalanced_drop_missing", d,
  list(effects = 3, placebo = 2, drop_if_d_miss_before_first_switch = TRUE))

set.seed(105)
d <- outcome_of(sim_on_off(G, T))
d$w <- runif(nrow(d), 0.5, 2)
d$cl <- (d$g - 1) %/% 4 + 1
cases[[length(cases) + 1]] <- run_case("weighted", d, list(effects = 3, placebo = 2, weight = "w"))
cases[[length(cases) + 1]] <- run_case("clustered", d, list(effects = 3, placebo = 2, cluster = "cl"))
cases[[length(cases) + 1]] <- run_case("weighted_clustered", d,
  list(effects = 3, placebo = 2, weight = "w", cluster = "cl"))

# With controls, 2.2.0 and 2.4.0 scale the variance term of theta_d by
# different sums (`rsum`) when there is more than one baseline treatment, so
# the case with standard errors has one baseline, and the case with two is
# compared on its estimates only.
with_controls <- function(d) {
  d$x1 <- rnorm(nrow(d)) + 0.2 * d$D
  d$x2 <- rnorm(nrow(d))
  outcome_of(d, beta_x = 0.5)
}
set.seed(106)
d <- with_controls(sim_on_off(G, T, p1 = 0))
cases[[length(cases) + 1]] <- run_case("controls", d, list(effects = 3, placebo = 2, controls = c("x1", "x2")))
set.seed(108)
d <- with_controls(sim_on_off(G, T))
cases[[length(cases) + 1]] <- run_case("controls_two_baselines", d,
  list(effects = 3, placebo = 2, controls = c("x1", "x2")))

# The same with 2.2.0's main routine patched to 2.4.0's rsum, the sum of
# N_gt over the not-yet switchers of the baseline level that enter its
# regression (did_multiplegt_main.R lines 1055-1073 of 2.4.0).
src <- deparse(DIDmultiplegtDYN:::did_multiplegt_main)
sample_240 <- "ever_change_d_XX == 0 & !is.na(diff_y_XX) & fd_X_all_non_missing_XX == 1L & d_sq_int_XX == l"
i <- grep("rmax <- df[, max(F_g_XX, na.rm = TRUE)]", src, fixed = TRUE)
stopifnot(length(i) == 1)
src[i] <- sub("df[, max(F_g_XX, na.rm = TRUE)]",
              sprintf("df[%s, max(F_g_XX, na.rm = TRUE)]", sample_240), src[i], fixed = TRUE)
j <- grep("rsum <- df[get(col_temp) == TRUE, ", src, fixed = TRUE)
stopifnot(length(j) == 1)
src[j] <- sub("get(col_temp) == TRUE, ", sprintf("get(col_temp) == TRUE & %s, ", sample_240), src[j], fixed = TRUE)
original_main <- DIDmultiplegtDYN:::did_multiplegt_main
patched_main <- eval(parse(text = src))
environment(patched_main) <- asNamespace("DIDmultiplegtDYN")
assignInNamespace("did_multiplegt_main", patched_main, "DIDmultiplegtDYN")
cases[[length(cases) + 1]] <- run_case("controls_two_baselines_rsum_240", d,
  list(effects = 3, placebo = 2, controls = c("x1", "x2")))
assignInNamespace("did_multiplegt_main", original_main, "DIDmultiplegtDYN")

set.seed(107)
d <- outcome_of(sim_on_off(G, T))
d$s <- d$g %% 2
cases[[length(cases) + 1]] <- run_case("trends_nonparam", d, list(effects = 3, placebo = 2, trends_nonparam = "s"))

out <- list(
  versions = list(r = paste(R.version$major, R.version$minor, sep = "."),
                  DIDmultiplegtDYN = pkg_version,
                  data.table = as.character(packageVersion("data.table")),
                  fixest = as.character(packageVersion("fixest")),
                  plm = as.character(packageVersion("plm"))),
  cases = cases
)
path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "dcdh_ref.json")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), path)
cat("wrote", path, "\n")
