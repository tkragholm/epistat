# Balance statistics of MatchIt's summary.matchit for methods whose weights are
# not pair counts, for no matching, for sampling weights, for discarded units,
# and for the percent improvement, the pair distance and interactions with the
# dummies of a factor.
#
# Per fit the script writes the estimand, the units' weights and discarded flags
# as MatchIt returns them, the pairs for nearest neighbour matching, and the
# tables of summary(): standardised and unstandardised statistics (with
# pair.dist for pair methods), both with improvement = TRUE, and the sample
# sizes. A few fits also carry the tables with interactions = TRUE. The
# interaction fits are made with the covariates in alphabetical order, which is
# the order this crate gives its columns. Numbers are written with 17
# significant digits, NA as "NA", Inf as "Inf".
#
# Data are lalonde and the simulated data set of balance.R (same seed), each
# with a positive sampling weight `sw`.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/balance_weights.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
set.seed(20261120)

data(lalonde)
lal <- lalonde
lal$id <- paste0("L", seq_len(nrow(lal)))
n <- 300
sim <- data.frame(id = paste0("S", seq_len(n)))
sim$x1 <- round(rnorm(n), 3)
sim$x2 <- round(exp(rnorm(n, 0.5, 0.8)), 3)
sim$z <- ifelse(runif(n) < 0.6, 0, round(rexp(n, 0.01)))
sim$b <- rbinom(n, 1, 0.4)
sim$cnt <- rpois(n, 3)
sim$f <- factor(sample(c("a", "b", "c", "d"), n, replace = TRUE,
                       prob = c(0.4, 0.3, 0.2, 0.1)))
sim$w <- round(rnorm(n, 10, 2), 3)
sim$treat <- rbinom(n, 1, plogis(-0.6 + 0.5 * sim$x1 + 0.3 * log(sim$x2) +
  0.6 * sim$b + 0.004 * sim$z - 0.2 * sim$cnt + 0.4 * (sim$f == "a")))
lal$sw <- round(runif(nrow(lal), 0.4, 3), 3)
sim$sw <- round(runif(n, 0.4, 3), 3)
rownames(lal) <- lal$id
rownames(sim) <- sim$id

tab <- function(m) {
  if (is.null(m)) return(NULL)
  out <- list()
  for (r in rownames(m)) out[[r]] <- setNames(as.list(as.numeric(m[r, ])), colnames(m))
  out
}
summ <- function(m, ...) {
  a <- summary(m, ...)
  matched <- if (!is.null(a$sum.matched)) a$sum.matched else a$sum.across
  list(all = tab(a$sum.all), matched = tab(matched), nn = tab(a$nn),
       reduction = tab(a$reduction))
}
pairs_of <- function(m, data) {
  mm <- m$match.matrix
  if (is.null(mm)) return(list())
  rn <- rownames(data)
  out <- list()
  for (i in seq_len(nrow(mm))) for (j in seq_len(ncol(mm))) {
    if (!is.na(mm[i, j])) {
      a <- match(rownames(mm)[i], rn)
      b <- match(mm[i, j], rn)
      out[[length(out) + 1]] <- if (data$treat[a] == 1) c(data$id[a], data$id[b])
                                else c(data$id[b], data$id[a])
    }
  }
  out
}

fit_one <- function(name, data, formula, interactions = FALSE, ...) {
  m <- suppressWarnings(matchit(formula, data = data, ...))
  paired <- !is.null(m$match.matrix)
  res <- list(
    name = name,
    estimand = m$estimand,
    weights = as.numeric(m$weights),
    discarded = as.logical(m$discarded),
    pairs = pairs_of(m, data),
    std = summ(m, standardize = TRUE, pair.dist = paired, improvement = TRUE),
    raw = summ(m, standardize = FALSE, pair.dist = paired, improvement = TRUE)
  )
  if (interactions) {
    res$int_std <- summ(m, standardize = TRUE, interactions = TRUE, pair.dist = FALSE)
    res$int_raw <- summ(m, standardize = FALSE, interactions = TRUE, pair.dist = FALSE)
    res$int_order <- rownames(summary(m, interactions = TRUE)$sum.all)
  }
  res
}

fml_l <- treat ~ age + educ + race + married + nodegree + re74
fml_s <- treat ~ x1 + x2 + z + b + cnt + f
fml_l_sorted <- treat ~ age + educ + married + nodegree + race + re74
fml_s_sorted <- treat ~ b + cnt + f + x1 + x2 + z
exact_l <- treat ~ married + nodegree + race
exact_s <- treat ~ b + cnt + f

fits_for <- function(data, fml, fml_sorted, fml_exact, k) {
  list(
    fit_one("subclass_att", data, fml, method = "subclass", subclass = k),
    fit_one("subclass_atc", data, fml, method = "subclass", subclass = k, estimand = "ATC"),
    fit_one("subclass_ate", data, fml, method = "subclass", subclass = k, estimand = "ATE"),
    fit_one("exact_att", data, fml_exact, method = "exact"),
    fit_one("exact_ate", data, fml_exact, method = "exact", estimand = "ATE"),
    fit_one("none", data, fml, method = NULL),
    fit_one("none_sampling", data, fml, method = NULL, s.weights = "sw"),
    fit_one("att_replace_sampling", data, fml, method = "nearest", replace = TRUE,
            s.weights = "sw"),
    fit_one("att_ratio2_sampling", data, fml, method = "nearest", ratio = 2,
            s.weights = "sw"),
    fit_one("subclass_ate_sampling", data, fml, method = "subclass", subclass = k,
            estimand = "ATE", s.weights = "sw"),
    fit_one("exact_att_sampling", data, fml_exact, method = "exact", s.weights = "sw"),
    fit_one("att_replace", data, fml, method = "nearest", replace = TRUE),
    fit_one("atc_replace", data, fml, method = "nearest", replace = TRUE, estimand = "ATC"),
    fit_one("att_ratio2", data, fml, method = "nearest", ratio = 2),
    fit_one("att_noreplace", data, fml, method = "nearest"),
    fit_one("att_discard", data, fml, method = "nearest", replace = TRUE, discard = "both"),
    fit_one("subclass_ate_discard", data, fml, method = "subclass", subclass = k,
            estimand = "ATE", discard = "both"),
    fit_one("none_discard", data, fml, method = NULL, discard = "both"),
    fit_one("att_replace_sorted", data, fml_sorted, interactions = TRUE,
            method = "nearest", replace = TRUE),
    fit_one("subclass_ate_sorted", data, fml_sorted, interactions = TRUE,
            method = "subclass", subclass = k, estimand = "ATE")
  )
}

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  lalonde = list(
    data = lal[, c("id", "treat", "age", "educ", "race", "married", "nodegree",
                   "re74", "sw")],
    fits = fits_for(lal, fml_l, fml_l_sorted, exact_l, 6)),
  sim = list(
    data = sim[, c("id", "treat", "x1", "x2", "z", "b", "cnt", "f", "sw")],
    fits = fits_for(sim, fml_s, fml_s_sorted, exact_s, 5))
)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
path <- file.path(dirname(script), "balance_weights.json")
write_json(out, path, digits = 17, na = "string", auto_unbox = TRUE,
           dataframe = "columns")
