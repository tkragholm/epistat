# The did, triplediff and HonestDiD fixtures, the continuous-treatment data,
# the Wald test and, through generate_did_cc_reference_data.R, the
# compositional-changes fixture.
#
# Run from the crate root:  Rscript tests/data/generate_reference_data.R
#
# The joint-path, directional and Gaussian-calibration references of HonestDiD
# are written by generate_joint_directional_reference.R, which gives each
# direction an explicit confidence grid.

library(did)
library(triplediff)
library(HonestDiD)
library(jsonlite)

# Versions: rerun with R 4.6.2, did 2.5.1, triplediff 0.2.4, HonestDiD 0.2.8
# and jsonlite 2.0.0, every file is written as stored except the ATT(g,t)
# estimates and standard errors of sections 1 and 2, which differ from the
# stored ones by at most 2e-13 relative. The stored files date from
# before versions were recorded; R's linear algebra (OpenBLAS 0.3.34 here)
# accounts for differences of that size, and the parity tests compare at 1e-8.
stopifnot(packageVersion("did") == "2.5.1",
          packageVersion("triplediff") == "0.2.4",
          packageVersion("HonestDiD") == "0.2.8",
          packageVersion("jsonlite") == "2.0.0")
cat(R.version.string, "\n")

base_path <- "tests/data/"

# --- 1. did Reference (Staggered ATT(g,t)) ---
tryCatch({
  data(mpdta)
  out_did <- att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat",
                    data = mpdta, control_group = "nevertreated", bstrap = FALSE, cband = FALSE)
  did_ref <- list(
    group = out_did$group, t = out_did$t, att = out_did$att, se = out_did$se,
    data_subset = mpdta[, c("year", "countyreal", "lemp", "first.treat")]
  )
  write_json(did_ref, paste0(base_path, "did_attgt_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("1. did Reference generated.\n")
}, error = function(e) cat("1. did Reference FAILED:", e$message, "\n"))

# --- 2. did DR Reference ---
tryCatch({
  data(mpdta)
  out_did_dr <- att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat",
                       xformla = ~ lpop, data = mpdta, control_group = "nevertreated",
                       est_method = "dr", bstrap = FALSE, cband = FALSE)
  did_dr_ref <- list(
    group = out_did_dr$group, t = out_did_dr$t, att = out_did_dr$att, se = out_did_dr$se,
    data_subset = mpdta[, c("year", "countyreal", "lemp", "first.treat", "lpop")]
  )
  write_json(did_dr_ref, paste0(base_path, "did_attgt_dr_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("2. did DR Reference generated.\n")
}, error = function(e) cat("2. did DR Reference FAILED:", e$message, "\n"))

# --- 3. Continuous DiD data (one two-period cell, simulated in base R) ---
tryCatch({
  set.seed(1234)
  n <- 2000
  df_2p <- data.frame(
    id = rep(1:n, each = 2),
    G = rep(sample(c(2, 0), n, replace = TRUE), each = 2),
    period = rep(1:2, n),
    D = rep(runif(n, 1, 10), each = 2),
    Y = rnorm(2*n)
  )
  df_2p$D[df_2p$G == 0] <- 0
  
  df_pre <- subset(df_2p, period == 1)
  df_post <- subset(df_2p, period == 2)
  
  cont_ref <- list(
    delta_y = df_post$Y - df_pre$Y,
    dose = df_post$D,
    treated = df_post$G == 2
  )
  write_json(cont_ref, paste0(base_path, "contdid_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("3. Continuous DiD data generated.\n")
}, error = function(e) cat("3. Continuous DiD FAILED:", e$message, "\n"))

# --- 4. Triple-Difference ---
tryCatch({
  df_ddd <- triplediff:::generate_test_panel(seed = 123, num_ids = 100)
  df_ddd$G <- ifelse(df_ddd$treat == 1, 2020, 0)
  out_ddd <- ddd(yname = "outcome", tname = "year", idname = "id",
                 gname = "G", pname = "partition",
                 xformla = ~ x1 + x2, data = df_ddd, panel = TRUE)
  ddd_ref <- list(
    att = out_ddd$ATT,
    se = out_ddd$se,
    data = df_ddd
  )
  write_json(ddd_ref, paste0(base_path, "triple_did_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("4. Triple-Difference generated.\n")
}, error = function(e) cat("4. Triple-Difference FAILED:", e$message, "\n"))

# --- 5. Honest DiD Data (Components only) ---
tryCatch({
  library(HonestDiD)
  data(BCdata_EventStudy)
  honest_ref <- list(
    betahat = BCdata_EventStudy$betahat,
    sigma = BCdata_EventStudy$sigma,
    pre_indices = BCdata_EventStudy$prePeriodIndices,
    post_indices = BCdata_EventStudy$postPeriodIndices
  )
  write_json(honest_ref, paste0(base_path, "honest_did_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("5. Honest DiD components generated.\n")
}, error = function(e) cat("5. Honest DiD FAILED:", e$message, "\n"))

# --- 6. Wald Test ---
tryCatch({
  est_pre <- c(-0.01, 0.02, 0.05)
  vcov_pre <- matrix(c(0.01, 0.002, 0.001,
                       0.002, 0.015, 0.003,
                       0.001, 0.003, 0.02), nrow=3)
  stat_wald <- as.numeric(t(est_pre) %*% solve(vcov_pre) %*% est_pre)
  p_wald <- 1 - pchisq(stat_wald, df=3)
  wald_ref <- list(
    estimates = est_pre,
    vcov = vcov_pre,
    statistic = stat_wald,
    p_value = p_wald,
    df = 3
  )
  write_json(wald_ref, paste0(base_path, "wald_test_ref.json"), digits = 15, auto_unbox = TRUE)
  cat("6. Wald Test generated.\n")
}, error = function(e) cat("6. Wald Test FAILED:", e$message, "\n"))


# --- 7. Honest DiD Multi-FLCI Scaffold (input vectors for Rust parity harness) ---
tryCatch({
  library(HonestDiD)
  data(BCdata_EventStudy)

  num_post <- length(BCdata_EventStudy$postPeriodIndices)
  period_0 <- rep(0, num_post)
  period_0[1] <- 1
  avg_0_1 <- rep(0, num_post)
  avg_0_1[1] <- 0.5
  avg_0_1[2] <- 0.5
  diff_0_1 <- rep(0, num_post)
  diff_0_1[1] <- 1
  diff_0_1[2] <- -1

  multi_scaffold <- list(
    meta = list(
      source = "Scaffold for Rust multi-FLCI parity",
      note = "HonestDiD does not expose a direct joint multi-functional DeltaRM FLCI API.",
      confidence_level = 0.95,
      mbar = 1.0
    ),
    l_vecs = list(period_0, avg_0_1, diff_0_1)
  )

  write_json(multi_scaffold, paste0(base_path, "honest_did_multi_flci_scaffold.json"), digits = 15, auto_unbox = TRUE)
  cat("7. Honest DiD multi-FLCI scaffold generated.\n")
}, error = function(e) cat("7. Honest DiD multi-FLCI scaffold FAILED:", e$message, "\n"))

# --- 8. DiD_CC external-reference scaffold ---
tryCatch({
  source("tests/data/generate_did_cc_reference_data.R")
  cat("8. DiD_CC scaffold generated.\n")
}, error = function(e) cat("8. DiD_CC scaffold FAILED:", e$message, "\n"))

cat("All reference data generation attempts complete.\n")
