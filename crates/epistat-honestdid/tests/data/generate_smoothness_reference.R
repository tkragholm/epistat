# The smoothness (Delta SD) reference: fixed-length and conditional-FLCI
# intervals from HonestDiD.
#
# Run from the crate root:  Rscript tests/data/generate_smoothness_reference.R
#
# Writes honest_did_smoothness_ref.json. For HonestDiD's two example event
# studies, BCdata_EventStudy (4 pre and 4 post periods) and LWdata_EventStudy
# (9 pre and 23 post periods), and for several post-period functionals, it
# records over a grid of M:
#
# - the FLCI of findOptimalFLCI, which createSensitivityResults(method =
#   "FLCI") calls, with its optimal weights and half-length;
# - the C-F interval of computeConditionalCS_DeltaSD, and of its sign
#   (DeltaSDB) and monotone (DeltaSDM) variants, with hybrid_flag = "FLCI" as
#   createSensitivityResults(method = "C-F") calls them, with the spacing of
#   the theta grid the interval is read from and the number of separate runs
#   of accepted grid points, whose outermost points are the interval;
# - the identified set of the internal .compute_IDset_DeltaSD at the estimated
#   coefficients, and whether its linear programs were feasible.
#
# The M grids run from 0 to beyond the largest second difference of the
# pre-period coefficients (with the reference period's 0), below which the
# identified set's linear programs are infeasible, and past the largest
# adjacent pre-period change.
#
# The FLCI's half-length is cv * h, with cv HonestDiD's .qfoldednormal: the
# 95% quantile of |Z + mu| over 10^6 normal draws at seed 0. The interval is
# therefore a Monte Carlo quantity; the tests derive their tolerance from it.
#
# A handful of cases are also run through createSensitivityResults itself and
# checked equal to the direct calls, so the file records what that function
# returns.

library(HonestDiD)
library(jsonlite)

# Versions: written with R 4.6.2, HonestDiD 0.2.8, CVXR 1.9.1, Rglpk 0.6.5.1
# and jsonlite 2.0.0.
stopifnot(packageVersion("HonestDiD") == "0.2.8",
          packageVersion("CVXR") == "1.9.1",
          packageVersion("Rglpk") == "0.6.5.1",
          packageVersion("jsonlite") == "2.0.0")
cat(R.version.string, "\n")

out_path <- "tests/data/honest_did_smoothness_ref.json"
alpha <- 0.05

basis <- function(i, n) {
  v <- rep(0, n)
  v[i] <- 1
  v
}

identified_set <- function(betahat, num_pre, num_post, l_vec, m) {
  feasible <- TRUE
  id <- withCallingHandlers(
    HonestDiD:::.compute_IDset_DeltaSD(M = m, trueBeta = betahat, l_vec = l_vec,
                                       numPrePeriods = num_pre,
                                       numPostPeriods = num_post),
    warning = function(w) {
      if (grepl("Solver did not find an optimum", conditionMessage(w))) {
        feasible <<- FALSE
        invokeRestart("muffleWarning")
      }
    }
  )
  list(feasible = feasible, lb = as.numeric(id$id.lb), ub = as.numeric(id$id.ub))
}

flci_case <- function(betahat, sigma, num_pre, num_post, l_vec, m) {
  r <- findOptimalFLCI(betahat = betahat, sigma = sigma, M = m,
                       numPrePeriods = num_pre, numPostPeriods = num_post,
                       l_vec = l_vec, alpha = alpha)
  list(lb = as.numeric(r$FLCI[1]), ub = as.numeric(r$FLCI[2]),
       half_length = as.numeric(r$optimalHalfLength),
       optimal_vec = as.numeric(r$optimalVec), status = r$status)
}

conditional_case <- function(betahat, sigma, num_pre, num_post, l_vec, m, delta) {
  args <- list(betahat = betahat, sigma = sigma, numPrePeriods = num_pre,
               numPostPeriods = num_post, l_vec = l_vec, alpha = alpha, M = m,
               hybrid_flag = "FLCI")
  grid <- suppressWarnings(switch(delta,
    DeltaSD = do.call(computeConditionalCS_DeltaSD, args),
    DeltaSDPB = do.call(computeConditionalCS_DeltaSDB,
                        c(args, biasDirection = "positive")),
    DeltaSDNB = do.call(computeConditionalCS_DeltaSDB,
                        c(args, biasDirection = "negative")),
    DeltaSDI = do.call(computeConditionalCS_DeltaSDM,
                       c(args, monotonicityDirection = "increasing")),
    DeltaSDD = do.call(computeConditionalCS_DeltaSDM,
                       c(args, monotonicityDirection = "decreasing"))
  ))
  accepted <- grid$grid[grid$accept == 1]
  runs <- rle(grid$accept)
  list(lb = if (length(accepted)) min(accepted) else NA,
       ub = if (length(accepted)) max(accepted) else NA,
       accepted_runs = sum(runs$values == 1),
       grid_lb = min(grid$grid), grid_ub = max(grid$grid),
       grid_step = diff(grid$grid)[1], grid_points = nrow(grid))
}

run_dataset <- function(name, data, m_values, flci_targets, conditional_targets) {
  betahat <- data$betahat
  sigma <- data$sigma
  num_pre <- length(data$prePeriodIndices)
  num_post <- length(data$postPeriodIndices)
  pre <- c(betahat[seq_len(num_pre)], 0)
  cat(sprintf("%s: largest adjacent pre change %.6f, largest second difference %.6f\n",
              name, max(abs(diff(betahat[seq_len(num_pre)]))),
              max(abs(diff(pre, differences = 2)))))
  targets <- c(flci_targets,
               conditional_targets[setdiff(names(conditional_targets), names(flci_targets))])
  cases <- list()
  for (target in names(targets)) {
    l_vec <- targets[[target]]
    for (m in m_values) {
      t0 <- Sys.time()
      case <- list(target = target, l_vec = l_vec, m = m,
                   identified = identified_set(betahat, num_pre, num_post, l_vec, m))
      if (target %in% names(flci_targets)) {
        case$flci <- flci_case(betahat, sigma, num_pre, num_post, l_vec, m)
      }
      if (target %in% names(conditional_targets)) {
        case$conditional <- lapply(
          c(DeltaSD = "DeltaSD", DeltaSDPB = "DeltaSDPB", DeltaSDNB = "DeltaSDNB",
            DeltaSDI = "DeltaSDI", DeltaSDD = "DeltaSDD"),
          function(delta) conditional_case(betahat, sigma, num_pre, num_post,
                                           l_vec, m, delta)
        )
      }
      cat(sprintf("  %s M = %g: %.1f s\n", target, m,
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      cases[[length(cases) + 1]] <- case
    }
  }
  list(name = name, betahat = betahat, sigma = sigma, num_pre = num_pre,
       num_post = num_post, cases = cases)
}

data(BCdata_EventStudy)
data(LWdata_EventStudy)

bc_targets <- list(
  first = basis(1, 4), second = basis(2, 4), third = basis(3, 4), fourth = basis(4, 4),
  average = rep(1 / 4, 4), declining = c(0.4, 0.3, 0.2, 0.1),
  contrast = c(1, -1, 0, 0)
)
bc <- run_dataset("BCdata_EventStudy", BCdata_EventStudy,
                  m_values = c(0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3),
                  flci_targets = bc_targets, conditional_targets = bc_targets)

lw_post <- 23
lw_periods <- setNames(lapply(seq_len(lw_post), function(i) basis(i, lw_post)),
                       paste0("period", seq_len(lw_post)))
lw_extra <- list(average = rep(1 / lw_post, lw_post),
                 declining = rev(seq_len(lw_post)) / sum(seq_len(lw_post)))
lw <- run_dataset("LWdata_EventStudy", LWdata_EventStudy,
                  m_values = c(0, 0.5, 1, 1.5, 2.5, 4),
                  flci_targets = c(lw_periods, lw_extra),
                  # Not the declining weights: the sign-restricted C-F there
                  # did not finish in ten minutes at M = 1.5.
                  conditional_targets = c(lw_periods[c(1, 2, 12, 23)],
                                          lw_extra["average"]))

# createSensitivityResults returns the same numbers as the direct calls above.
check <- function(dataset, data, target, m, method, delta, extra) {
  case <- Filter(function(x) x$target == target && x$m == m, dataset$cases)[[1]]
  args <- c(list(betahat = data$betahat, sigma = data$sigma,
                 numPrePeriods = dataset$num_pre, numPostPeriods = dataset$num_post,
                 method = method, Mvec = m, l_vec = case$l_vec, alpha = alpha), extra)
  r <- suppressWarnings(do.call(createSensitivityResults, args))
  expected <- if (method == "FLCI") case$flci else case$conditional[[delta]]
  stopifnot(identical(r$lb, expected$lb), identical(r$ub, expected$ub))
}
check(bc, BCdata_EventStudy, "average", 0.2, "FLCI", NULL, list())
check(bc, BCdata_EventStudy, "first", 0, "FLCI", NULL, list())
check(bc, BCdata_EventStudy, "average", 0.05, "C-F", "DeltaSD", list())
check(bc, BCdata_EventStudy, "second", 0.1, "C-F", "DeltaSDI",
      list(monotonicityDirection = "increasing"))
check(bc, BCdata_EventStudy, "first", 0.025, "C-F", "DeltaSDNB",
      list(biasDirection = "negative"))
check(lw, LWdata_EventStudy, "period2", 1, "C-F", "DeltaSD", list())

write_json(
  list(
    source = "HonestDiD 0.2.8 findOptimalFLCI and computeConditionalCS_DeltaSD(B,M), as createSensitivityResults calls them",
    r_version = R.version.string,
    alpha = alpha,
    folded_normal_draws = 1e6,
    datasets = list(bc, lw)
  ),
  out_path, digits = NA, auto_unbox = TRUE, pretty = FALSE
)
cat("wrote", out_path, "\n")
