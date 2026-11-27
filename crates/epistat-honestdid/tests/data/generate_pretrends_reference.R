# Reference for the pre-trends power module (src/pretrends.rs).
#
# Reference package: pretrends 0.1.0 (Roth), installed from its authors'
# GitHub repository, jonathandroth/pretrends commit
# 122731d082a5990e274f57fd9af0968e44977e7a. It is not on CRAN. Its
# probabilities come from mvtnorm::pmvnorm with the package defaults
# (Genz-Bretz, abseps 1e-3, random shifts), so the package's own numbers carry a
# Monte Carlo error. Two things are recorded to bound it:
#   * the package is run under `n_seeds` seeds, and the mean and standard
#     deviation over the seeds of every number are stored (`pkg_sd`);
#   * a tight reference written from the paper: pass probability by
#     mvtnorm::pmvnorm with abseps 1e-8 and its reported error, and the
#     conditional mean of the pre block by Tallis's formula over tight
#     pmvnorm calls, checked in this script against tmvtnorm::mtmvnorm.
# Run: Rscript generate_pretrends_reference.R <out.json> <HeAndWangResults.rda>
suppressMessages({library(pretrends); library(mvtnorm); library(tmvtnorm); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
e <- new.env(); load(args[2], e); hw <- e$HeAndWangResults
n_seeds <- 20
alpha <- 0.05
z <- qnorm(1 - alpha / 2)
tight <- GenzBretz(maxpts = 3e6, abseps = 1e-8, releps = 0)

rect <- function(lower, upper, mean, sigma) {
  if (length(mean) == 1) {
    s <- sqrt(sigma[1, 1])
    return(list(value = pnorm(upper, mean, s) - pnorm(lower, mean, s), error = 0))
  }
  set.seed(1)
  p <- pmvnorm(lower = lower, upper = upper, mean = mean, sigma = sigma, algorithm = tight)
  list(value = as.numeric(p), error = attr(p, "error"))
}
tight_pass <- function(mu, S) { s <- sqrt(diag(S)); rect(-z * s, z * s, mu, S) }
# Tallis (1961): E[X | box] = mu + S g / P, g_i = f_i(a_i) P_i(a_i) - f_i(b_i) P_i(b_i).
tight_mean <- function(mu, S) {
  p <- length(mu); s <- sqrt(diag(S)); P <- tight_pass(mu, S)
  g <- numeric(p); gerr <- numeric(p)
  for (i in 1:p) for (side in c(1, -1)) {
    bound <- side * (-z * s[i])  # side 1: lower bound -z s, side -1: upper bound z s
    dens <- dnorm(bound, mu[i], s[i])
    if (p == 1) { pr <- list(value = 1, error = 0) } else {
      o <- setdiff(1:p, i)
      cm <- mu[o] + S[o, i] / S[i, i] * (bound - mu[i])
      cs <- S[o, o, drop = FALSE] - tcrossprod(S[o, i]) / S[i, i]
      so <- sqrt(diag(S))[o]
      pr <- rect(-z * so, z * so, cm, cs)
    }
    g[i] <- g[i] + side * dens * pr$value; gerr[i] <- gerr[i] + dens * pr$error
  }
  shift <- as.numeric(S %*% g) / P$value
  err <- as.numeric(abs(S) %*% gerr) / P$value + abs(shift) * P$error / P$value
  list(value = mu + shift, error = err)
}

one_case <- function(name, betahat, sigma, tvec, ref, delta, target_powers) {
  pre <- which(tvec < ref); post <- which(tvec > ref)
  runs <- lapply(1:n_seeds, function(s) {
    set.seed(s)
    r <- pretrends(betahat = betahat, sigma = sigma, deltatrue = delta, tVec = tvec,
                   referencePeriod = ref)
    ev <- r$df_eventplot
    ev <- ev[ev$t != ref, ]
    list(power = r$df_power$Power, bf = r$df_power$Bayes.Factor,
         lr = r$df_power$Likelihood.Ratio, mean_after = ev$meanAfterPretesting,
         t = ev$t)
  })
  stack <- function(f) do.call(rbind, lapply(runs, f))
  m <- function(x) if (is.matrix(x)) colMeans(x) else mean(x)
  sdv <- function(x) if (is.matrix(x)) apply(x, 2, sd) else sd(x)
  pw <- sapply(runs, `[[`, "power"); bf <- sapply(runs, `[[`, "bf")
  lr <- sapply(runs, `[[`, "lr")
  ma <- t(sapply(runs, `[[`, "mean_after"))
  spre <- sigma[pre, pre, drop = FALSE]
  tp <- tight_pass(delta[pre], spre); tp0 <- tight_pass(0 * delta[pre], spre)
  tm <- tight_mean(delta[pre], spre)
  if (length(pre) > 1) {
    chk <- mtmvnorm(mean = delta[pre], sigma = spre, lower = -z * sqrt(diag(spre)),
                    upper = z * sqrt(diag(spre)), doComputeVariance = FALSE)$tmean
    stopifnot(max(abs(chk - tm$value)) < 1e-4)
  }
  cross <- sigma[post, pre, drop = FALSE]
  post_mean <- delta[post] + as.numeric(cross %*% solve(spre, tm$value - delta[pre]))
  post_err <- as.numeric(abs(cross %*% solve(spre)) %*% tm$error)
  slopes <- lapply(target_powers, function(tp_) {
    sv <- sapply(1:n_seeds, function(s) { set.seed(s)
      slope_for_power(sigma = sigma, targetPower = tp_, tVec = tvec, referencePeriod = ref) })
    list(target = tp_, pkg_mean = mean(sv), pkg_sd = sd(sv))
  })
  list(name = name, betahat = betahat, sigma = sigma, tvec = tvec, reference = ref,
       delta = delta, pre = pre - 1L, post = post - 1L,
       pkg = list(power = mean(pw), power_sd = sd(pw), bayes_factor = mean(bf),
                  bayes_factor_sd = sd(bf), likelihood_ratio = mean(lr),
                  likelihood_ratio_sd = sd(lr),
                  mean_after = colMeans(ma), mean_after_sd = apply(ma, 2, sd),
                  mean_after_t = runs[[1]]$t),
       tight = list(pass = tp$value, pass_error = tp$error, pass_null = tp0$value,
                    pass_null_error = tp0$error, mean_pre = tm$value,
                    mean_pre_error = tm$error, mean_post = post_mean, mean_post_error = post_err),
       slopes = slopes)
}

cases <- list()
# He and Wang (2017), the package's own example: the linear trend with 50 percent power
# and the quadratic trend of its README.
tv <- hw$tVec; ref <- -1
cases[[1]] <- one_case("hewang_linear50", hw$beta, hw$sigma, tv, ref,
  slope_for_power(sigma = hw$sigma, targetPower = 0.5, tVec = tv, referencePeriod = ref) * (tv - ref), c(0.5, 0.8))
cases[[2]] <- one_case("hewang_quadratic", hw$beta, hw$sigma, tv, ref, 0.024 * (tv - ref)^2, c(0.8))
cases[[3]] <- one_case("hewang_linear_neg", hw$beta, hw$sigma, tv, ref, -0.04 * (tv - ref), c(0.5))

# Synthetic event studies: one, two, five and eight pre-periods, with positive and
# negative correlations.
mk <- function(k, rho_fun, seed) {
  set.seed(seed)
  sd_ <- runif(k, 0.05, 0.15)
  R <- outer(1:k, 1:k, rho_fun)
  S <- diag(sd_) %*% R %*% diag(sd_)
  (S + t(S)) / 2
}
syn <- function(name, npre, npost, rho_fun, seed, slope) {
  k <- npre + npost
  tvec <- c(-(npre + 1):-2, 0:(npost - 1)); tvec <- tvec[seq_len(k)]
  S <- mk(k, rho_fun, seed)
  set.seed(seed + 1)
  b <- rnorm(k, 0.03, 0.08)
  one_case(name, b, S, tvec, -1, slope * (tvec + 1), c(0.5, 0.8))
}
cases[[4]] <- syn("syn_pre1", 1, 3, function(i, j) 0.5^abs(i - j), 11, 0.07)
cases[[5]] <- syn("syn_pre2", 2, 2, function(i, j) 0.6^abs(i - j), 12, 0.06)
cases[[6]] <- syn("syn_pre5_pos", 5, 3, function(i, j) 0.7^abs(i - j), 13, 0.05)
cases[[7]] <- syn("syn_pre5_neg", 5, 3, function(i, j) ifelse(i == j, 1, -0.15), 14, -0.05)
cases[[8]] <- syn("syn_pre8", 8, 4, function(i, j) 0.4^abs(i - j) , 15, 0.04)

write_json(list(
  package = "pretrends", package_version = as.character(packageVersion("pretrends")),
  package_commit = "122731d082a5990e274f57fd9af0968e44977e7a",
  mvtnorm_version = as.character(packageVersion("mvtnorm")),
  tmvtnorm_version = as.character(packageVersion("tmvtnorm")),
  r_version = R.version.string, alpha = alpha, n_seeds = n_seeds,
  readme = list(slope50 = 0.05205871, power = 0.4999776, bayes_factor = 0.56901,
                likelihood_ratio = 0.1057627,
                mean_after = c(-0.09230764, -0.05555285, -0.02790950, 0.06490184, 0.12084694, 0.16946498, 0.22453827),
                quad_power = 0.6624282, quad_bayes_factor = 0.3841516,
                quad_likelihood_ratio = 0.4332635),
  cases = cases), out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
