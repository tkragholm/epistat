# Reference values for the options of `epistat_did::estimate_dcdh` and the
# functions around it, read by tests/parity_dcdh_options.rs.
#
# The reference is DIDmultiplegtDYN::did_multiplegt_dyn 2.2.0 from the CRAN
# archive, the last release that runs without the R package polars, which
# is not on CRAN (see generate_dcdh_reference.R for the installation):
#
#   DCDH_LIB=~/.cache/r-didmultiplegtdyn-2.2.0 Rscript generate_dcdh_options_reference.R
#
# Where the crate follows a rule of 2.4.0 that 2.2.0 does not have, the
# script patches 2.2.0 to 2.4.0's rule, or runs 2.4.0's own code:
#
# - `predict_het`: 2.4.0 uses HC2 standard errors where 2.2.0 uses HC1; the
#   main routine is patched to HC2, with the rule for switchers alone in
#   their cell below.
# - `avg_time_periods` exists only in 2.4.0, where it is C++
#   (`compute_Tg_cpp` and `avg_cumul_cpp` in src/did_loops.cpp). The script
#   downloads the 2.4.0 source from CRAN, compiles that file with Rcpp and
#   applies it to the panel of 2.2.0's main routine, as 2.4.0's
#   `did_multiplegt_avg_cumul` does.
#
# The descriptive outputs (`design`, `date_first_switch`,
# `normalized_weights`) are formatted as text by the package; the script
# runs copies of those functions with the formatting line removed.

lib <- Sys.getenv("DCDH_LIB", path.expand("~/.cache/r-didmultiplegtdyn-2.2.0"))
suppressMessages(library(DIDmultiplegtDYN, lib.loc = lib))
suppressMessages(library(jsonlite))
suppressMessages(library(data.table))
pkg_version <- as.character(packageVersion("DIDmultiplegtDYN", lib.loc = lib))
stopifnot(pkg_version == "2.2.0")
ns <- asNamespace("DIDmultiplegtDYN")

# --- Simulated panels, as in generate_dcdh_reference.R, with a group-level
# linear trend and group-level covariates.

outcome_of <- function(d, beta_x = 0, trend = 0) {
  d <- d[order(d$g, d$t), ]
  lag_d <- ave(d$D, d$g, FUN = function(x) c(0, head(x, -1)))
  unit <- rnorm(max(d$g))[d$g]
  slope <- rnorm(max(d$g), sd = trend)[d$g]
  x <- if ("x1" %in% names(d)) d$x1 else 0
  d$Y <- unit + 0.3 * d$t + slope * d$t + 1.0 * d$D + 0.5 * lag_d + beta_x * x + rnorm(nrow(d))
  d
}

sim_absorbing <- function(G, T) {
  d <- expand.grid(t = 1:T, g = 1:G)
  first <- sample(c(3, 5, 7, 99), G, replace = TRUE)
  d$D <- as.numeric(d$t >= first[d$g])
  d
}

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

# A continuous baseline: every group starts at its own dose and, from a
# random period on (or never), moves to a higher or lower one for good.
sim_continuous <- function(G, T) {
  rows <- vector("list", G)
  for (g in 1:G) {
    d1 <- runif(1, 0, 2)
    first <- sample(c(2:T, 99), 1)
    D <- rep(d1, T)
    if (first <= T) D[first:T] <- d1 + sample(c(-0.5, 0.5, 1), 1)
    rows[[g]] <- data.frame(t = 1:T, g = g, D = D)
  }
  do.call(rbind, rows)
}

unbalance <- function(d, drop = 0.08, y_na = 0.05, d_na = 0.03) {
  d <- d[runif(nrow(d)) > drop, ]
  d$Y[runif(nrow(d)) < y_na] <- NA
  d$D[runif(nrow(d)) < d_na] <- NA
  d
}

# Group-level covariates and subgroups.
with_group_vars <- function(d) {
  G <- max(d$g)
  d$h1 <- rnorm(G)[d$g]
  d$h2 <- rbinom(G, 1, 0.5)[d$g]
  d$sub <- (d$g %% 3 == 0) * 1
  d
}

# --- Copies of the package's descriptive functions without the lines that
# format their numbers as text.
unformatted <- function(f, pattern) {
  src <- deparse(f, width.cutoff = 500L)
  i <- grep(pattern, src, fixed = TRUE)
  stopifnot(length(i) >= 1)
  src <- src[-i]
  g <- eval(parse(text = src))
  environment(g) <- ns
  g
}
design_raw <- unformatted(ns$did_multiplegt_dyn_design, "desmat[, 2] <- noquote")
dfs_raw <- unformatted(ns$did_multiplegt_dyn_dfs, "dfsmat[, 2] <- sprintf")
normweights_raw <- unformatted(ns$did_multiplegt_dyn_normweights, "weight_mat[, ] <- sprintf")

# The main routine with 2.4.0's HC2 standard errors in predict_het.
main_src <- deparse(ns$did_multiplegt_main, width.cutoff = 500L)
hc <- grep('type = "HC1"', main_src, fixed = TRUE)
stopifnot(length(hc) >= 2)
# A switcher alone in its cell of fixed effects has leverage 1, so HC2 is
# 0/0 for it and NaN for every coefficient; such a switcher has no part in
# the covariates' coefficients, and the crate gives it no part in their
# covariance, which is the HC2 covariance of the regression without it.
hc2_omega_XX <- function(residuals, diaghat, df) {
  ifelse(diaghat > 1 - 1e-10, 0, residuals^2 / (1 - diaghat))
}
main_src <- gsub('type = "HC1"', 'type = "HC2", omega = hc2_omega_XX', main_src, fixed = TRUE)
# Each regression also records the condition numbers of its weighted design
# and of the HC2 covariance of the covariates, which set the test's
# tolerances.
fs <- grep("f_stat <- linearHypothesis(", main_src, fixed = TRUE)
stopifnot(length(fs) == 2)
main_src[fs] <- paste0("het_kappa_XX <<- c(het_kappa_XX, list(c(kappa(model, exact = TRUE), ",
                       "kappa(vcovHC(model, type = \"HC2\", omega = hc2_omega_XX)[predict_het_good, predict_het_good, drop = FALSE], exact = TRUE)))); ", main_src[fs])
main_hc2 <- eval(parse(text = main_src))
environment(main_hc2) <- ns
original_main <- ns$did_multiplegt_main

# avg_time_periods: 2.4.0's C++ from the CRAN source.
cpp_dir <- file.path(tempdir(), "dcdh240")
dir.create(cpp_dir, showWarnings = FALSE)
tarball <- file.path(cpp_dir, "DIDmultiplegtDYN_2.4.0.tar.gz")
if (!file.exists(tarball)) {
  download.file("https://cran.r-project.org/src/contrib/DIDmultiplegtDYN_2.4.0.tar.gz", tarball, quiet = TRUE)
}
untar(tarball, files = "DIDmultiplegtDYN/src/did_loops.cpp", exdir = cpp_dir)
Rcpp::sourceCpp(file.path(cpp_dir, "DIDmultiplegtDYN/src/did_loops.cpp"))

avg_cumul_240 <- function(df, l_XX, T_max_XX, same_switchers, switchers) {
  s <- as.data.table(df)[, .(group_XX, time_XX, treatment_XX, outcome_XX, D1 = d_sq_XX,
                             F_g_XX, T_g_XX, d_sq_int_XX, S_g_XX, N_gt_XX)]
  setorderv(s, c("group_XX", "time_XX"))
  EV <- as.integer(!is.na(s$F_g_XX) & s$F_g_XX < (T_max_XX + 1L))
  NGT <- as.integer(!is.na(s$outcome_XX))
  Tg <- compute_Tg_cpp(as.integer(s$group_XX), as.integer(s$time_XX), as.numeric(s$outcome_XX),
                       as.numeric(s$F_g_XX), as.numeric(s$T_g_XX), EV,
                       as.integer(s$d_sq_int_XX), NGT)
  Mg <- pmin(as.numeric(l_XX), Tg - s$F_g_XX + 1)
  avg_cumul_cpp(as.integer(s$group_XX), as.integer(s$time_XX), as.numeric(s$treatment_XX),
                as.numeric(s$outcome_XX), as.numeric(s$D1), as.numeric(s$F_g_XX), Tg, EV,
                as.integer(s$d_sq_int_XX), Mg, NGT, as.numeric(s$S_g_XX), as.numeric(s$N_gt_XX),
                as.integer(l_XX), as.integer(isTRUE(same_switchers)), switchers)
}

# --- One case: the wrapper's results, and from the main routine the
# variance terms of every group and the extra outputs.

main_args <- function(d, full) {
  list(df = d, outcome = "Y", group = "g", time = "t", treatment = "D",
       effects = full$effects, placebo = if (is.null(full$placebo)) 0 else full$placebo,
       ci_level = 95, switchers = if (is.null(full$switchers)) "" else full$switchers,
       only_never_switchers = isTRUE(full$only_never_switchers),
       trends_nonparam = full$trends_nonparam, weight = full$weight,
       controls = full$controls,
       dont_drop_larger_lower = isTRUE(full$dont_drop_larger_lower),
       drop_if_d_miss_before_first_switch = isTRUE(full$drop_if_d_miss_before_first_switch),
       cluster = full$cluster, same_switchers = isTRUE(full$same_switchers),
       same_switchers_pl = isTRUE(full$same_switchers_pl),
       effects_equal = isTRUE(full$effects_equal), save_results = NULL,
       normalized = isTRUE(full$normalized), predict_het = full$predict_het,
       trends_lin = isTRUE(full$trends_lin),
       less_conservative_se = isTRUE(full$less_conservative_se),
       continuous = full$continuous)
}

var_terms_of <- function(main, clustered) {
  df <- as.data.table(main$df)
  map <- unique(df[!is.na(group), .(group_XX, gid = group)])
  df <- merge(df, map, by = "group_XX", all.x = TRUE)
  df[, group := gid]
  setorder(df, group_XX, time_XX)
  l <- main$l_XX
  lp <- if (is.null(main$l_placebo_XX)) 0 else main$l_placebo_XX
  key <- if (clustered) "cluster_XX" else "group"
  first <- if (clustered) df[first_obs_by_clust_XX == 1] else df[first_obs_by_gp_XX == 1]
  first <- first[order(get(key))]
  col_or_na <- function(col) if (col %in% names(first)) first[[col]] else rep(NA_real_, nrow(first))
  out <- list(
    ids = first[[key]],
    effects = lapply(seq_len(l), function(i) col_or_na(sprintf("U_Gg_var_glob_%d_XX", i))),
    placebos = lapply(seq_len(lp), function(i) col_or_na(sprintf("U_Gg_var_glob_pl_%d_XX", i))),
    average = col_or_na("U_Gg_var_global_XX")
  )
  if (clustered && "U_Gg_var_global_XX" %in% names(df)) {
    df[, first_of_group := seq_len(.N) == 1, by = group_XX]
    s <- df[first_of_group == TRUE, .(v = sum(U_Gg_var_global_XX)), by = cluster_XX][order(cluster_XX)]
    out$average <- s$v
  }
  out
}

# The condition number of the Gram matrix of the controls' first
# differences among the not-yet switchers, demeaned within period and
# baseline, as the controls' regressions use them; columns that are zero
# there are left out, as the fit leaves them out.
kappa_of <- function(df, cols) {
  df <- as.data.table(df)
  setorder(df, group_XX, time_XX)
  fd <- sapply(cols, function(v) ave(df[[v]], df$group_XX, FUN = function(x) c(NA, diff(x))))
  dy <- ave(df$outcome_XX, df$group_XX, FUN = function(x) c(NA, diff(x)))
  ok <- complete.cases(fd) & !is.na(dy) & df$ever_change_d_XX == 0 & df$N_gt_XX > 0
  fd <- fd[ok, , drop = FALSE]
  cell <- interaction(df$time_XX[ok], df$d_sq_XX[ok], drop = TRUE)
  w <- df$N_gt_XX[ok]
  fd <- apply(fd, 2, function(x) sqrt(w) * (x - ave(w * x, cell, FUN = sum) / ave(w, cell, FUN = sum)))
  fd <- fd[, apply(abs(fd), 2, max) > 1e-12, drop = FALSE]
  kappa(crossprod(fd), exact = TRUE)
}

mat <- function(m) {
  if (is.null(m)) return(NULL)
  if (is.null(dim(m))) m <- matrix(m, nrow = 1)
  unname(lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ])))
}

results_of <- function(res) {
  list(effects = mat(res$Effects),
       ate = if (is.null(res$ATE)) NULL else as.numeric(res$ATE[1, ]),
       placebos = mat(res$Placebos),
       p_jointeffects = res$p_jointeffects,
       p_jointplacebo = res$p_jointplacebo,
       p_equality_effects = res$p_equality_effects,
       delta_D_avg_total = res$delta_D_avg_total)
}

data_of <- function(d) {
  list(g = d$g, t = d$t, D = d$D, Y = d$Y,
       w = if ("w" %in% names(d)) d$w else NULL,
       x1 = if ("x1" %in% names(d)) d$x1 else NULL,
       x2 = if ("x2" %in% names(d)) d$x2 else NULL,
       h1 = if ("h1" %in% names(d)) d$h1 else NULL,
       h2 = if ("h2" %in% names(d)) d$h2 else NULL,
       sub = if ("sub" %in% names(d)) d$sub else NULL,
       stratum = if ("s" %in% names(d)) d$s else NULL,
       cluster = if ("cl" %in% names(d)) d$cl else NULL)
}

run_case <- function(name, d, args, extra = list()) {
  d <- as.data.frame(d)
  base <- list(df = d, outcome = "Y", group = "g", time = "t", treatment = "D", graph_off = TRUE)
  full <- c(base, args)
  out <- suppressMessages(suppressWarnings(do.call(did_multiplegt_dyn, full)))
  case <- c(list(name = name, args = args, data = data_of(d)), results_of(out$results))

  main <- suppressMessages(suppressWarnings(do.call(ns$did_multiplegt_main, main_args(d, full))))
  case$var_terms <- var_terms_of(main, !is.null(full$cluster))
  ctl <- grep("^time_fe_XX_", names(main$df), value = TRUE)
  if (!is.null(full$controls)) ctl <- c(full$controls, ctl)
  if (length(ctl) > 0) case$kappa_x <- kappa_of(main$df, ctl)
  if (isTRUE(full$normalized)) {
    w <- normweights_raw(main, TRUE, TRUE, isTRUE(full$same_switchers), full$continuous)$norm_weight_mat
    case$normalized_weights <- mat(w[seq_len(main$l_XX), , drop = FALSE])
  }
  if (isTRUE(extra$design)) {
    for (p in c(1, 0.5)) {
      ds <- design_raw(main, list(p, "console"), full$weight, NULL, "_no_by", list())
      case[[paste0("design_", p * 100)]] <- list(
        paths = mat(ds$design_mat), const = unname(as.numeric(ds$design_const)))
    }
    for (opt in c("", "by_baseline_treat")) {
      fs <- dfs_raw(main, c(opt, "console"), NULL, "_no_by", list())
      tabs <- if (opt == "") list(fs$dfs_mat) else lapply(seq_len(fs$levels_baseline_treat), function(j) fs[[paste0("dfs_mat", j)]])
      levels <- if (opt == "") list(NULL) else lapply(seq_len(fs$levels_baseline_treat), function(j) as.numeric(fs[[paste0("level", j)]]))
      case[[if (opt == "") "first_switch" else "first_switch_by_baseline"]] <- lapply(seq_along(tabs), function(j) list(
        baseline = levels[[j]],
        times = as.numeric(rownames(tabs[[j]])),
        n = as.numeric(tabs[[j]][, 1]),
        share = as.numeric(tabs[[j]][, 2])))
    }
  }
  if (isTRUE(extra$avg_time_periods)) {
    a <- avg_cumul_240(main$df, main$l_XX, main$T_max_XX, full$same_switchers,
                       if (is.null(full$switchers)) "" else full$switchers)
    case$avg_time_periods <- list(average = a$avg_cumul, num = a$num, den = a$den, nswitch = a$nswitch)
  }
  if (!is.null(full$predict_het)) {
    het_kappa_XX <<- list()
    main2 <- suppressMessages(suppressWarnings(do.call(main_hc2, main_args(d, full))))
    case$het_kappa <- het_kappa_XX
    het <- main2$did_multiplegt_dyn$predict_het
    case$predict_het <- lapply(seq_len(nrow(het)), function(i) list(
      effect = het$effect[i], covariate = het$covariate[i], estimate = het$Estimate[i],
      se = het$SE[i], t = het$t[i], lb = het$LB[i], ub = het$UB[i], n = het$N[i], p_f = het$pF[i]))
  }
  if (!is.null(full$by) || !is.null(full$by_path)) {
    lv <- out$by_levels
    case$by_levels <- as.character(lv)
    case$by_results <- lapply(seq_along(lv), function(j) results_of(out[[paste0("by_level_", j)]]$results))
  }
  case
}

cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- run_case(...)
G <- 80; T <- 8

# trends_lin: an absorbing and an on-off treatment with group-specific trends,
# weighted and clustered, with controls at one baseline, with
# trends_nonparam, normalised and with predict_het.
set.seed(201)
d <- outcome_of(sim_absorbing(G, T), trend = 0.3)
add("trends_lin_absorbing", d, list(effects = 4, placebo = 2, trends_lin = TRUE))
set.seed(202)
d <- with_group_vars(outcome_of(sim_on_off(G, T), trend = 0.3))
d$w <- runif(nrow(d), 0.5, 2)
d$cl <- (d$g - 1) %/% 4 + 1
d$s <- d$g %% 2
add("trends_lin", d, list(effects = 3, placebo = 2, trends_lin = TRUE), list(avg_time_periods = TRUE))
add("trends_lin_weighted_clustered", d, list(effects = 3, placebo = 2, trends_lin = TRUE, weight = "w", cluster = "cl"))
add("trends_lin_nonparam", d, list(effects = 3, placebo = 2, trends_lin = TRUE, trends_nonparam = "s"))
add("trends_lin_normalized", d, list(effects = 3, placebo = 2, trends_lin = TRUE, normalized = TRUE))
add("trends_lin_het", d, list(effects = 3, placebo = 2, trends_lin = TRUE, predict_het = list(c("h1", "h2"), -1)))
set.seed(203)
d <- outcome_of(sim_on_off(G, T, p1 = 0), trend = 0.3)
d$x1 <- rnorm(nrow(d)) + 0.2 * d$D
d$x2 <- rnorm(nrow(d))
d$Y <- d$Y + 0.5 * d$x1
add("trends_lin_controls", d, list(effects = 3, placebo = 2, trends_lin = TRUE, controls = c("x1", "x2")))

# same_switchers, same_switchers_pl and less_conservative_se on balanced and
# unbalanced panels, with three treatment levels and with clusters.
set.seed(204)
d <- with_group_vars(outcome_of(sim_on_off(G, T)))
d$cl <- (d$g - 1) %/% 4 + 1
add("same_switchers", d, list(effects = 3, placebo = 2, same_switchers = TRUE), list(avg_time_periods = TRUE))
add("same_switchers_pl", d, list(effects = 3, placebo = 2, same_switchers = TRUE, same_switchers_pl = TRUE))
add("less_conservative", d, list(effects = 3, placebo = 2, less_conservative_se = TRUE))
add("less_conservative_clustered", d, list(effects = 3, placebo = 2, less_conservative_se = TRUE, cluster = "cl"))
add("normalized_weights", d, list(effects = 4, placebo = 1, normalized = TRUE), list(design = TRUE, avg_time_periods = TRUE))
add("normalized_weights_same", d, list(effects = 3, normalized = TRUE, same_switchers = TRUE))
add("predict_het", d, list(effects = 3, placebo = 2, predict_het = list(c("h1", "h2"), -1)))
add("predict_het_some", d, list(effects = 3, placebo = 2, predict_het = list(c("h1"), c(1, 2))))
add("by", d, list(effects = 3, placebo = 1, by = "sub"))
add("by_path", d, list(effects = 2, placebo = 1, by_path = 3))
add("by_path_all", d, list(effects = 2, by_path = -1))
set.seed(205)
d <- unbalance(outcome_of(sim_on_off(G, T)))
add("same_switchers_unbalanced", d, list(effects = 3, placebo = 2, same_switchers = TRUE, same_switchers_pl = TRUE), list(avg_time_periods = TRUE))
add("less_conservative_unbalanced", d, list(effects = 3, placebo = 2, less_conservative_se = TRUE), list(design = TRUE))
set.seed(206)
d <- with_group_vars(outcome_of(sim_three_levels(G, T)))
d$w <- runif(nrow(d), 0.5, 2)
add("three_levels_less_conservative", d, list(effects = 3, placebo = 2, less_conservative_se = TRUE, same_switchers = TRUE))
add("three_levels_design", d, list(effects = 3, placebo = 1, weight = "w"), list(design = TRUE, avg_time_periods = TRUE))
add("three_levels_het", d, list(effects = 2, placebo = 1, weight = "w", predict_het = list(c("h1"), -1)))
add("three_levels_out", d, list(effects = 3, switchers = "out", normalized = TRUE), list(avg_time_periods = TRUE))

# continuous: a continuous baseline with polynomials of order 1 and 2.
set.seed(207)
d <- sim_continuous(120, T)
d <- d[order(d$g, d$t), ]
d1 <- ave(d$D, d$g, FUN = function(x) x[1])
d$Y <- rnorm(120)[d$g] + 0.3 * d$t + 0.2 * d1 * d$t + d$D + rnorm(nrow(d))
add("continuous_1", d, list(effects = 3, placebo = 2, continuous = 1))
add("continuous_2", d, list(effects = 3, placebo = 1, continuous = 2, normalized = TRUE))

# The bootstrap. 2.2.0 draws sample(1:n, n, replace = TRUE) over the sorted
# resampling units after set.seed(seed), and a group drawn twice enters as
# one group of twice the weight; 2.4.0 gives each unit its number of draws
# as a weight. The script records the draws and the estimates of each
# replicate with the draws as weights, and checks that their standard
# deviation is 2.2.0's bootstrap standard error. With a cluster, 2.2.0's
# bootstrap fails (its collapse of duplicated rows drops the cluster
# variable), so that case has the replicates only.
estimates_of <- function(res) {
  list(effects = as.numeric(matrix(res$Effects, ncol = 8)[, 1]),
       placebos = if (is.null(res$Placebos)) NULL else as.numeric(matrix(res$Placebos, ncol = 8)[, 1]),
       ate = as.numeric(res$ATE[1]))
}
boot_case <- function(name, d, args, reps, seed) {
  d <- as.data.frame(d)
  full <- c(list(df = d, outcome = "Y", group = "g", time = "t", treatment = "D", graph_off = TRUE), args)
  unit_of <- if (is.null(args$cluster)) d$g else d[[args$cluster]]
  units <- sort(unique(unit_of))
  set.seed(seed)
  draws <- lapply(seq_len(reps), function(j) {
    s <- sample(seq_along(units), length(units), replace = TRUE)
    tabulate(s, nbins = length(units))
  })
  replicates <- lapply(draws, function(counts) {
    db <- d[counts[match(unit_of, units)] > 0, ]
    db$bw <- counts[match(if (is.null(args$cluster)) db$g else db[[args$cluster]], units)]
    r <- suppressMessages(suppressWarnings(do.call(did_multiplegt_dyn,
      c(list(df = db, outcome = "Y", group = "g", time = "t", treatment = "D", graph_off = TRUE,
             weight = "bw"), args))))
    estimates_of(r$results)
  })
  case <- c(list(name = name, args = args, data = data_of(d), reps = reps, units = units,
                 draws = draws, replicates = replicates),
            results_of(suppressMessages(suppressWarnings(do.call(did_multiplegt_dyn, full)))$results))
  if (is.null(args$cluster)) {
    set.seed(seed)
    b <- suppressMessages(suppressWarnings(do.call(did_multiplegt_dyn, c(full, list(bootstrap = reps)))))
    case$bootstrap_effects <- mat(b$results$Effects)
    case$bootstrap_placebos <- mat(b$results$Placebos)
    case$bootstrap_ate <- as.numeric(b$results$ATE[1, ])
    se <- apply(do.call(rbind, lapply(replicates, `[[`, "effects")), 2, sd, na.rm = TRUE)
    stopifnot(max(abs(se - b$results$Effects[, 2])) < 1e-10)
  }
  case
}
set.seed(208)
d <- outcome_of(sim_on_off(G, T))
d$cl <- (d$g - 1) %/% 4 + 1
boots <- list(
  boot_case("bootstrap", d, list(effects = 3, placebo = 2), reps = 200, seed = 11),
  boot_case("bootstrap_clustered", d, list(effects = 2, placebo = 1, cluster = "cl"), reps = 100, seed = 12)
)

out <- list(
  versions = list(r = paste(R.version$major, R.version$minor, sep = "."),
                  DIDmultiplegtDYN = pkg_version,
                  DIDmultiplegtDYN_cpp = "2.4.0",
                  data.table = as.character(packageVersion("data.table")),
                  fixest = as.character(packageVersion("fixest")),
                  plm = as.character(packageVersion("plm")),
                  sandwich = as.character(packageVersion("sandwich")),
                  car = as.character(packageVersion("car")),
                  Rcpp = as.character(packageVersion("Rcpp"))),
  cases = cases,
  bootstrap = boots
)
path <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "dcdh_options_ref.json")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null", na = "null"), path)
cat("wrote", path, "\n")
