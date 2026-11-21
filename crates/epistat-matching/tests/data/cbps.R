# Reference fits of the covariate balancing propensity score (Imai and
# Ratkovic 2014) from CBPS 0.24 and from MatchIt's matchit(distance = "cbps").
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/cbps.R
#
# Two data sets: MatchIt's lalonde (raw scale, covariates up to 6e4) and a
# simulated data set with continuous and binary covariates. For each, CBPS is
# run for the ATT and the ATE with method = "over" and method = "exact", at its
# defaults (twostep = TRUE, iterations = 1000, MatchIt's call). MatchIt's
# distance is written for the same four fits.
#
# CBPS stops at the tolerance of R's optim(), a relative change of the loss of
# 1.5e-8, which is far from the minimum when the loss is flat. So that the
# Rust fit can be compared with a bound that follows from the arithmetic, the
# script also finds the minimum itself. It builds the design the way CBPS
# documents (centre and scale the covariates, then the left singular vectors of
# the standardised design with its intercept, so U'U = I), writes the
# objective from the paper, with the weight matrix fixed at the logistic fit
# scaled by the minimiser over [0.8, 1.1] (optimize()), and
#   - for the over-identified fit, minimises it by BFGS with a tight tolerance
#     followed by Newton steps with numerical derivatives of Richardson's
#     extrapolation (numDeriv);
#   - for the exact fit, finds the root of the balance moments by Newton steps.
# The reference value of CBPS's J is checked against this objective at CBPS's own
# coefficients, which confirms the objective.
#
# For each fit the script writes
#   J, mle_J                CBPS's J (weight matrix fixed at the start) and J at
#                           the logistic fit
#   v_cond                  the condition number of the covariance of the
#                           moments on the eigenvalues kept in its inverse
#   j_scale                 |gbar|^2 times the spectral norm of the weight matrix at
#                           star, the size of the terms of J
#   loss_at_cbps            the script's objective at CBPS's coefficients
#   star_*                  the minimum (over) or root (exact) found here
#   grad_cbps, grad_star    norms of the numerical gradient of the loss at CBPS's
#                           fit and at star, in the orthogonal coordinates
#   lambda_min              the smallest eigenvalue of the Hessian of J at star
#   sigma_min, moment_cbps  the smallest singular value of the Jacobian of the
#                           balance moments at star, and the norm of the
#                           moments at CBPS's fit (exact)
#   jhess_norm              the spectral norm of the Hessian of J at star
#   t_norm                  the spectral norm of the map from the orthogonal
#                           coordinates to the coefficients
#   balance_*               (1/n) sum w_i x_i over the covariates with an
#                           intercept, the balance moment, at CBPS and at star

suppressMessages({
  library(MatchIt)
  library(CBPS)
  library(MASS)
  library(numDeriv)
  library(jsonlite)
})

data(lalonde)
lal <- data.frame(
  treat = lalonde$treat, age = lalonde$age, educ = lalonde$educ,
  black = as.numeric(lalonde$race == "black"),
  hispan = as.numeric(lalonde$race == "hispan"),
  married = lalonde$married, nodegree = lalonde$nodegree,
  re74 = lalonde$re74, re75 = lalonde$re75)

set.seed(20261121)
n <- 300
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.4)
x3 <- round(runif(n, 18, 60))
sim <- data.frame(
  treat = rbinom(n, 1, plogis(-0.4 + 0.8 * x1 + 0.6 * x2 - 0.03 * (x3 - 40))),
  x1 = x1, x2 = x2, x3 = x3)

clip <- function(p) pmax(1e-6, pmin(1 - 1e-6, p))

fit_one <- function(name, df) {
  vars <- setdiff(names(df), "treat")
  f <- as.formula(paste("treat ~", paste(vars, collapse = " + ")))
  X0 <- model.matrix(f, df)
  y <- df$treat
  n <- nrow(X0)
  nt <- sum(y)
  xs <- apply(X0[, -1], 2, sd)
  xm <- colMeans(X0[, -1])
  Z <- cbind(1, sweep(sweep(X0[, -1], 2, xm), 2, xs, "/"))
  sv <- svd(Z)
  U <- sv$u
  bmap <- function(g) {
    b <- as.vector(sv$v %*% (g / sv$d))
    b[-1] <- b[-1] / xs
    b[1] <- b[1] - sum(xm * b[-1])
    b
  }
  k <- ncol(U)
  tmat <- sapply(seq_len(k), function(j) bmap(diag(k)[, j]))
  prob <- function(g) as.vector(clip(plogis(U %*% g)))
  wts <- function(p, att) if (att) n / nt * (y - p) / (1 - p) else 1 / (p - 1 + y)
  mom_bal <- function(g, att) crossprod(U, wts(prob(g), att)) / n
  mom <- function(g, att) {
    p <- prob(g)
    c(crossprod(U, y - p) / n, crossprod(U, wts(p, att)) / n)
  }
  covm <- function(g, att) {
    p <- prob(g)
    a <- crossprod(U * sqrt(p * (1 - p)))
    if (att) {
      b <- crossprod(U * sqrt(p)) * n / nt
      cc <- crossprod(U * sqrt(p / (1 - p))) * n^2 / nt^2
    } else {
      b <- crossprod(U)
      cc <- crossprod(U / sqrt(p * (1 - p)))
    }
    rbind(cbind(a, b), cbind(t(b), cc)) / n
  }
  jloss_at <- function(g, att) {
    m <- mom(g, att)
    as.numeric(t(m) %*% ginv(covm(g, att)) %*% m)
  }
  g0 <- coef(glm(y ~ U - 1, family = binomial))
  balance_orig <- function(p, att) as.vector(crossprod(X0, wts(p, att)) / n)

  by_estimand <- lapply(c("att", "ate"), function(est) {
    att <- est == "att"
    alpha <- optimize(function(a) jloss_at(g0 * a, att), c(0.8, 1.1))$minimum
    ginit <- g0 * alpha
    vev <- eigen(covm(ginit, att), symmetric = TRUE, only.values = TRUE)$values
    vkept <- vev[vev > sqrt(.Machine$double.eps) * max(vev)]
    iv <- ginv(covm(ginit, att))
    jloss <- function(g) {
      m <- mom(g, att)
      as.numeric(t(m) %*% iv %*% m)
    }
    by_method <- lapply(c("over", "exact"), function(method) {
      cat(name, est, method, "\n")
      out <- capture.output(r <- CBPS(
        as.formula(paste("treat ~", paste(vars, collapse = " + "))), df,
        ATT = if (att) 1 else 0, method = method))
      beta <- as.vector(coef(r))
      # Orthogonal coordinates of CBPS's coefficients.
      bs <- beta
      bs[1] <- beta[1] + sum(xm * beta[-1])
      bs[-1] <- beta[-1] * xs
      g_r <- sv$d * as.vector(t(sv$v) %*% bs)
      stopifnot(max(abs(U %*% g_r - r$linear.predictor)) < 1e-9)
      stopifnot(abs(jloss(g_r) - r$J) < 1e-9 * max(1, abs(r$J)))
      stopifnot(abs(jloss(g0) - r$mle.J) < 1e-9 * max(1, abs(r$mle.J)))

      mi <- matchit(as.formula(paste("treat ~", paste(vars, collapse = " + "))),
                    data = df, method = NULL, distance = "cbps",
                    estimand = toupper(est),
                    distance.options = list(over = method == "over"))
      mi_lin <- matchit(as.formula(paste("treat ~", paste(vars, collapse = " + "))),
                        data = df, method = NULL, distance = "cbps",
                        link = "linear.logit", estimand = toupper(est),
                        distance.options = list(over = method == "over"))
      stopifnot(max(abs(unname(mi$distance) - r$fitted.values)) == 0)

      if (method == "over") {
        g <- optim(g_r, jloss, method = "BFGS",
                   control = list(reltol = 1e-16, maxit = 10000))$par
        for (i in 1:30) {
          step <- solve(hessian(jloss, g), grad(jloss, g))
          g <- g - step
          if (sqrt(sum(step^2)) < 1e-13) break
        }
        hs <- hessian(jloss, g)
        ev <- eigen(hs, symmetric = TRUE, only.values = TRUE)$values
        lambda_min <- min(ev)
        sigma_min <- NA_real_
        moment_cbps <- NA_real_
      } else {
        g <- g_r
        for (i in 1:30) {
          step <- solve(jacobian(function(x) mom_bal(x, att), g),
                        as.vector(mom_bal(g, att)))
          g <- g - step
          if (sqrt(sum(step^2)) < 1e-13) break
        }
        sigma_min <- min(svd(jacobian(function(x) mom_bal(x, att), g))$d)
        moment_cbps <- sqrt(sum(mom_bal(g_r, att)^2))
        ev <- eigen(hessian(jloss, g), symmetric = TRUE, only.values = TRUE)$values
        lambda_min <- NA_real_
      }
      list(
        estimand = est, method = method,
        coefficient_names = colnames(X0),
        coefficients = beta,
        fitted = as.vector(r$fitted.values),
        linear = as.vector(r$linear.predictor),
        J = r$J, mle_J = r$mle.J, converged = r$converged,
        matchit_distance = unname(mi$distance),
        matchit_linear = unname(mi_lin$distance),
        start_scale = alpha,
        v_cond = max(vkept) / min(vkept),
        j_scale = sum(mom(g, att)^2) / min(vkept),
        loss_at_cbps = jloss(g_r),
        star_coefficients = bmap(g),
        star_linear = as.vector(U %*% g),
        star_J = jloss(g),
        grad_cbps = sqrt(sum(grad(jloss, g_r)^2)),
        grad_star = sqrt(sum(grad(jloss, g)^2)),
        lambda_min = lambda_min, sigma_min = sigma_min,
        moment_cbps = moment_cbps,
        jhess_norm = max(abs(ev)),
        t_norm = max(svd(tmat)$d),
        balance_cbps = balance_orig(prob(g_r), att),
        balance_star = balance_orig(prob(g), att))
    })
    by_method
  })
  list(name = name, treat = y, covariate_names = vars,
       covariates = lapply(vars, function(v) df[[v]]),
       fits = unlist(by_estimand, recursive = FALSE))
}

out <- list(
  versions = list(r = R.version.string,
                  cbps = as.character(packageVersion("CBPS")),
                  matchit = as.character(packageVersion("MatchIt")),
                  numDeriv = as.character(packageVersion("numDeriv"))),
  data = list(fit_one("lalonde", lal), fit_one("sim", sim)))
write_json(out, "tests/data/cbps.json", digits = I(17), auto_unbox = TRUE,
           na = "null")
