# Reference values for epistat_glm::absorb on poorly connected designs: the
# exact projections on the dummies (the QR of lm.fit), the smallest positive
# eigenvalue of D'WD, and fixest::feols (fixest 0.14.2) at fixef.tol = 1e-10
# with its iterations and its distance from the exact projection.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript absorb_hard.R > absorb_hard.json
suppressMessages({
  library(fixest)
  library(jsonlite)
})
stopifnot(packageVersion("fixest") == "0.14.2")
set.seed(20261126)
setFixest_nthreads(1)

TOL <- 1e-10

# Workers observed for five years who move to a neighbouring firm on a line
# of firms with probability 0.1 a year, so that the firms form a chain; the
# occupation, on a line as well, changes with probability 0.05.
chain <- function(workers, firms, occupations) {
  d <- expand.grid(s = 0:4, worker = 0:(workers - 1))
  start <- sample(0:5, workers, TRUE)
  firm <- integer(nrow(d)); occ <- integer(nrow(d))
  for (i in 0:(workers - 1)) {
    rows <- which(d$worker == i)
    f <- sample(0:(firms - 1), 1); o <- sample(0:(occupations - 1), 1)
    for (k in rows) {
      if (runif(1) < 0.1) f <- min(max(f + sample(c(-1, 1), 1), 0), firms - 1)
      if (runif(1) < 0.05) o <- min(max(o + sample(c(-1, 1), 1), 0), occupations - 1)
      firm[k] <- f; occ[k] <- o
    }
  }
  d$firm <- firm; d$occ <- occ; d$year <- start[d$worker + 1] + d$s
  d$y <- rnorm(workers)[d$worker + 1] + rnorm(firms)[d$firm + 1] + rnorm(nrow(d))
  d
}

# Worker k twice at firm k and twice at firm k + 1: a path of firms.
path <- function(workers) {
  d <- data.frame(worker = rep(0:(workers - 1), each = 4))
  d$firm <- d$worker + rep(c(0, 0, 1, 1), workers)
  d$y <- d$worker / workers + rnorm(nrow(d))
  d
}

# Workers who rarely move, to a random firm: a sparse random graph.
sparse <- function(workers, firms) {
  d <- expand.grid(s = 0:4, worker = 0:(workers - 1))
  f <- sample(0:(firms - 1), workers, TRUE)
  firm <- integer(nrow(d))
  for (k in seq_len(nrow(d))) {
    i <- d$worker[k] + 1
    if (runif(1) < 0.05) f[i] <- sample(0:(firms - 1), 1)
    firm[k] <- f[i]
  }
  d$firm <- firm
  d$y <- rnorm(nrow(d)) + d$firm %% 7
  d
}

make <- function(name, d, fe_names, weights = NULL) {
  fml <- as.formula(paste("y ~ 1 |", paste(fe_names, collapse = " + ")))
  m <- feols(fml, d, weights = weights, fixef.tol = TOL, fixef.iter = 100000,
             notes = FALSE, warn = FALSE)
  kept <- obs(m)
  dk <- d[kept, ]
  ww <- if (is.null(weights)) rep(1, nrow(dk)) else weights[kept]
  D <- do.call(cbind, lapply(fe_names, function(f) model.matrix(~ factor(dk[[f]]) - 1)))
  exact <- lm.fit(D * sqrt(ww), dk$y * sqrt(ww))$residuals / sqrt(ww)
  ev <- eigen(crossprod(D * sqrt(ww)), symmetric = TRUE, only.values = TRUE)$values
  positive <- ev[ev > 1e-9 * max(ev)]
  yc <- dk$y - weighted.mean(dk$y, ww)
  scale <- sqrt(sum(ww * yc^2) / sum(ww))
  list(name = name, factors = lapply(fe_names, function(f) d[[f]]), weights = weights,
       y = d$y, kept = kept - 1, exact_residuals = unname(exact),
       lambda_plus = min(positive), fixest_iterations = m$iterations,
       fixest_error = max(abs(resid(m) - exact)) / scale)
}

ch <- chain(600, 30, 10)
sp <- sparse(1000, 250)
cases <- list(
  make("chain_two_way", ch, c("worker", "firm")),
  make("chain_with_years", ch, c("worker", "firm", "year"),
       weights = runif(nrow(ch), 0.5, 2)),
  make("chain_with_occupations", ch, c("worker", "firm", "occ")),
  make("path", path(150), c("worker", "firm")),
  make("sparse_two_way", sp, c("worker", "firm"))
)
cat(toJSON(list(fixest = as.character(packageVersion("fixest")), tolerance = TOL,
                cases = cases), digits = NA, auto_unbox = TRUE, null = "null"))
