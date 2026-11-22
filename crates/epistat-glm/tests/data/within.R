# Reference values for epistat_glm::within: fixest::feols(y ~ x | family,
# cluster = ~family), and lm with one dummy per family for the coefficients.
# Run: PATH=/opt/homebrew/opt/r/bin:$PATH Rscript within.R > within.json
suppressMessages({
  library(fixest)
  library(jsonlite)
})
set.seed(20261122)

vec <- function(m) as.vector(t(m))

make <- function(name, sizes, weights = FALSE, constant_col = FALSE, all_equal_family = FALSE) {
  G <- length(sizes)
  fam <- rep(seq_len(G), sizes)
  n <- length(fam)
  fe <- rep(rnorm(G), sizes)
  x1 <- rnorm(n) + 0.8 * fe
  x2 <- rbinom(n, 1, plogis(fe))
  if (all_equal_family) x2[fam == 3] <- 1
  x3 <- if (constant_col) fe else rnorm(n)
  y <- 1 + 0.5 * x1 + 0.3 * x2 - 0.2 * x3 + fe + rnorm(n)
  w <- if (weights) runif(n, 0.5, 2) else NULL
  d <- data.frame(y, x1, x2, x3, fam)
  m <- if (is.null(w)) feols(y ~ x1 + x2 + x3 | fam, d, cluster = ~fam, notes = FALSE)
       else feols(y ~ x1 + x2 + x3 | fam, d, weights = w, cluster = ~fam, notes = FALSE)
  mi <- if (is.null(w)) feols(y ~ x1 + x2 + x3 | fam, d, vcov = "iid", notes = FALSE)
        else feols(y ~ x1 + x2 + x3 | fam, d, weights = w, vcov = "iid", notes = FALSE)
  names_all <- c("x1", "x2", "x3")
  keep <- names_all %in% names(coef(m))
  expand_vec <- function(v) { out <- rep(NA_real_, 3); out[keep] <- v; out }
  expand_mat <- function(M) {
    out <- matrix(NA_real_, 3, 3); out[keep, keep] <- M; vec(out)
  }
  lmf <- if (is.null(w)) lm(y ~ x1 + x2 + x3 + factor(fam), d)
         else lm(y ~ x1 + x2 + x3 + factor(fam), d, weights = w)
  list(name = name, fam = fam, x = unname(cbind(x1, x2, x3)), y = y, weights = w,
       coef = expand_vec(coef(m)), lm_coef = unname(coef(lmf)[2:4]),
       cluster_vcov = expand_mat(vcov(m)), iid_vcov = expand_mat(vcov(mi)),
       r2_within = r2(m, "wr2"), nobs = m$nobs)
}

cases <- list(
  make("balanced_pairs", rep(2, 200)),
  make("unbalanced", sample(2:5, 150, TRUE)),
  make("singletons", c(sample(2:5, 120, TRUE), rep(1, 15))),
  make("weights", c(sample(2:4, 100, TRUE), rep(1, 10)), weights = TRUE),
  make("constant_column", sample(2:4, 100, TRUE), constant_col = TRUE),
  make("all_equal_family", sample(2:4, 100, TRUE), all_equal_family = TRUE)
)
cat(toJSON(list(r = R.version.string, fixest = as.character(packageVersion("fixest")),
                cases = cases),
           digits = NA, auto_unbox = TRUE, null = "null", na = "null"))
