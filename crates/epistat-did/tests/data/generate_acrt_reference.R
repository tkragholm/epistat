# Reference values for epistat_did::estimate_acrt_sieve.
#
# The estimator: the control trend m0 is the weighted mean outcome change of the
# units with dose 0, the sieve coefficients are the weighted least squares fit of
# dy - m0 on a basis of the dose among the units with a positive dose, and the
# ACRT is the unweighted mean over those units of the derivative of the fit.
#
# The influence function is the infinitesimal jackknife of that whole procedure:
# each unit's case weight v_i multiplies its weight in m0 and in the fit and
# counts it v_i times in the mean, and IF_i = n d ACRT / d v_i at v = 1, by
# central differences with Richardson extrapolation (steps h and h / 2). The
# plain central difference is written too, so the test can bound the truncation.
#
# Cases:
#   contdid  the two-period panel of generate_reference_data.R (n = 2000, half
#            treated, doses uniform on (1, 10)) with an effect of the dose added,
#            and a cubic basis: contdid's cont_did(target_parameter = "slope",
#            dose_est_method = "parametric", degree = 3, num_knots = 0) fits the
#            same cubic polynomials with splines2::bSpline, so its overall ACRT
#            is the same number. Its standard error is written for comparison:
#            it is formed from an influence function scaled by the treated units
#            but averaged over all units, so it is about n_T / n of the jackknife.
#   weighted unequal weights, units with a negative dose, and the basis (d, d^2)
#            without a constant, so the estimated trend moves the fit.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript generate_acrt_reference.R
suppressMessages({
  library(jsonlite)
  library(contdid)
  library(splines2)
})

acrt_fn <- function(dose, dy, w, basis, dbasis, v) {
  ww <- w * v
  ctl <- dose == 0
  trt <- dose > 0
  m0 <- sum(ww[ctl] * dy[ctl]) / sum(ww[ctl])
  fit <- lm.wfit(basis(dose[trt]), dy[trt] - m0, ww[trt])
  slope <- drop(dbasis(dose[trt]) %*% fit$coefficients)
  sum(v[trt] * slope) / sum(v[trt])
}

jackknife_if <- function(dose, dy, w, basis, dbasis, h) {
  n <- length(dose)
  one <- rep(1, n)
  rich <- plain <- numeric(n)
  for (i in seq_len(n)) {
    d <- function(step) {
      vp <- one; vm <- one
      vp[i] <- 1 + step; vm[i] <- 1 - step
      (acrt_fn(dose, dy, w, basis, dbasis, vp) - acrt_fn(dose, dy, w, basis, dbasis, vm)) /
        (2 * step)
    }
    d1 <- d(h); d2 <- d(h / 2)
    rich[i] <- n * (4 * d2 - d1) / 3
    plain[i] <- n * d1
  }
  list(rich = rich, plain = plain)
}

delete_one_se <- function(dose, dy, w, basis, dbasis) {
  n <- length(dose)
  est <- sapply(seq_len(n), function(i) {
    v <- rep(1, n); v[i] <- 0
    acrt_fn(dose, dy, w, basis, dbasis, v)
  })
  sqrt((n - 1) / n * sum((est - mean(est))^2))
}

# The 2-norm condition number of the weighted design of the fit.
design_kappa <- function(dose, w, basis) {
  trt <- dose > 0
  kappa(sqrt(w[trt]) * basis(dose[trt]), exact = TRUE)
}

make_case <- function(name, dose, dy, w, basis, dbasis, h = 1e-3) {
  n <- length(dose)
  ij <- jackknife_if(dose, dy, w, basis, dbasis, h)
  list(
    name = name, dose = dose, delta_y = dy, weight = w,
    acrt = acrt_fn(dose, dy, w, basis, dbasis, rep(1, n)),
    influence = ij$rich, influence_plain = ij$plain, h = h,
    se_ij = sqrt(sum((ij$rich - mean(ij$rich))^2)) / n,
    se_delete_one = delete_one_se(dose, dy, w, basis, dbasis),
    kappa = design_kappa(dose, w, basis)
  )
}

cubic <- function(d) cbind(1, d, d^2, d^3)
dcubic <- function(d) cbind(0, 1, 2 * d, 3 * d^2)

# contdid: the panel of generate_reference_data.R, with a dose effect in period 2.
set.seed(1234)
n <- 2000
df_2p <- data.frame(
  id = rep(1:n, each = 2),
  G = rep(sample(c(2, 0), n, replace = TRUE), each = 2),
  period = rep(1:2, n),
  D = rep(runif(n, 1, 10), each = 2),
  Y = rnorm(2 * n)
)
df_2p$D[df_2p$G == 0] <- 0
df_2p$Y <- df_2p$Y + (df_2p$period == 2) * (0.5 * df_2p$D - 0.03 * df_2p$D^2)
cd <- suppressWarnings(suppressMessages(cont_did(
  yname = "Y", tname = "period", idname = "id", dname = "D", gname = "G", data = df_2p,
  target_parameter = "slope", aggregation = "dose", treatment_type = "continuous",
  dose_est_method = "parametric", degree = 3, num_knots = 0, control_group = "nevertreated",
  bstrap = FALSE
)))
pre <- subset(df_2p, period == 1)
post <- subset(df_2p, period == 2)
c1 <- make_case("contdid", post$D, post$Y - pre$Y, rep(1, n), cubic, dcubic)
c1$contdid_acrt <- cd$overall_acrt
c1$contdid_se <- cd$overall_acrt_se
c1$boundary <- range(post$D[post$D > 0])

# weighted: doses on (0.2, 2) with a tenth of the units at dose 0 and a few
# negative, weights on (0.5, 2), and the basis (d, d^2).
set.seed(20261123)
m <- 300
dose <- round(runif(m, 0.2, 2), 3)
group <- sample(c("control", "treated", "negative"), m, replace = TRUE, prob = c(0.3, 0.65, 0.05))
dose[group == "control"] <- 0
dose[group == "negative"] <- -dose[group == "negative"]
dy <- round(0.4 + 1.5 * pmax(dose, 0) - 0.4 * pmax(dose, 0)^2 + rnorm(m, 0, 0.7), 4)
w <- round(runif(m, 0.5, 2), 3)
quad <- function(d) cbind(d, d^2)
dquad <- function(d) cbind(1, 2 * d)
c2 <- make_case("weighted", dose, dy, w, quad, dquad)

out <- list(
  versions = list(
    r = R.version.string, contdid = as.character(packageVersion("contdid")),
    splines2 = as.character(packageVersion("splines2")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  cases = list(c1, c2)
)
write_json(out, "acrt_ref.json", digits = I(17), auto_unbox = TRUE)
