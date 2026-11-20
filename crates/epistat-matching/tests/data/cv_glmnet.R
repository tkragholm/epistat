# Reference for the cross-validated elastic net, from R's cv.glmnet().
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/cv_glmnet.R
#
# cv.glmnet draws its folds with R's generator, which the Rust fit does not
# reproduce. The folds come instead from the Rust fit: tests/data/cv_folds.json
# holds, for each case, the fold of every row that epistat-rng gives for the
# case's seed, for rows ordered treated first. It is written by
#   cargo test -p epistat-matching --test r_cv_glmnet -- --ignored write_folds
# and the test `folds_are_the_ones_the_reference_used` checks that the Rust
# fit still draws them. The script puts the treated rows first, as the design
# matrix of the Rust fit has them. They are passed to cv.glmnet as `foldid`, so both sides
# fit the same models on the same rows.
#
# MatchIt calls cv.glmnet with its defaults, so standardize = TRUE, no lambda
# given (each fold builds its own path and the folds are lined up with the
# master path), the deviance as the loss and lambda.1se as the penalty, except
# in the ridge case, which sets standardize = FALSE. thresh is 1e-20 so that
# glmnet's own convergence error is far below the comparison tolerance, set
# for all fits with glmnet.control().

suppressMessages({
  library(glmnet)
  library(MatchIt)
  library(jsonlite)
})

glmnet.control(thresh = 1e-20, maxit = 1e7)
folds <- fromJSON("tests/data/cv_folds.json", simplifyVector = FALSE)

design <- function(name, n) {
  set.seed(switch(name, lasso_unweighted = 11, elasticnet_weighted = 12,
                  ridge_raw = 13, lalonde_elasticnet = 14))
  if (name == "lalonde_elasticnet") {
    data(lalonde)
    x <- cbind(age = lalonde$age, educ = lalonde$educ,
               black = as.numeric(lalonde$race == "black"),
               hispan = as.numeric(lalonde$race == "hispan"),
               married = lalonde$married, nodegree = lalonde$nodegree,
               re74 = lalonde$re74, re75 = lalonde$re75)
    return(list(x = x, y = lalonde$treat))
  }
  p <- 6
  z <- matrix(rnorm(n * p), n, p)
  shared <- rnorm(n)
  x <- sqrt(0.7) * z + sqrt(0.3) * shared
  x[, 5] <- as.numeric(x[, 5] > 0.2)
  if (name == "ridge_raw") x[, 1] <- 3 * x[, 1] + 40
  colnames(x) <- paste0("x", seq_len(p))
  eta <- -0.2 + 0.7 * x[, 2] - 0.5 * x[, 3] + 0.6 * x[, 5] +
    0.04 * (x[, 1] - mean(x[, 1]))
  list(x = x, y = rbinom(n, 1, plogis(eta)))
}

cases <- lapply(folds$cases, function(case) {
  d <- design(case$name, case$n)
  stopifnot(nrow(d$x) == case$n)
  w <- if (isTRUE(case$weighted)) round(runif(case$n, 0.4, 2.5), 3) else NULL
  # Treated rows first, as the design matrix of the Rust fit has them, so
  # that a fold is the fold of the same row on both sides.
  ord <- order(-d$y)
  d$x <- d$x[ord, , drop = FALSE]
  d$y <- d$y[ord]
  if (!is.null(w)) w <- w[ord]
  foldid <- unlist(case$foldid) + 1L
  fit <- cv.glmnet(d$x, d$y, family = "binomial", alpha = case$alpha,
                   nfolds = case$nfolds, foldid = foldid, weights = w,
                   standardize = isTRUE(case$standardize))
  b <- as.vector(coef(fit, s = "lambda.1se"))
  list(name = case$name, alpha = case$alpha, nfolds = case$nfolds,
       standardize = isTRUE(case$standardize), seed = case$seed,
       x = unname(d$x), covariate_names = colnames(d$x), y = d$y,
       weights = w, foldid = unlist(case$foldid),
       lambda = fit$lambda, cvm = fit$cvm, cvsd = fit$cvsd,
       lambda_min = fit$lambda.min, lambda_1se = fit$lambda.1se,
       index_min = which(fit$lambda == fit$lambda.min) - 1L,
       index_1se = which(fit$lambda == fit$lambda.1se) - 1L,
       coefficients_1se = b)
})

out <- list(
  versions = list(r = R.version.string,
                  glmnet = as.character(packageVersion("glmnet")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  cases = cases)
write_json(out, "tests/data/cv_glmnet.json", digits = I(17),
           auto_unbox = TRUE, null = "null")
