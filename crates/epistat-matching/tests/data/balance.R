# Balance statistics from MatchIt's summary.matchit, on lalonde and on a
# simulated data set, for matched samples that R chose. The pairs and the unit
# weights of each fit are written out, so the Rust side computes balance on the
# same units and weights without repeating the pair selection.
#
# Per fit: the estimand, the pairs (match.matrix as treated id, control id),
# MatchIt's unit weights, and the tables of summary(): standardised
# statistics, unstandardised statistics (which carry the eQQ distances),
# both with interactions = TRUE, with addlvariables, and the sample sizes.
# pair.dist is off. Numbers are written with 17 significant digits, NA as "NA".
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/balance.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
set.seed(20261120)

# ---- data -------------------------------------------------------------------
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

# ---- helpers ----------------------------------------------------------------
tab <- function(m) {
  if (is.null(m)) return(NULL)
  out <- list()
  for (r in rownames(m)) {
    out[[r]] <- setNames(as.list(as.numeric(m[r, ])), colnames(m))
  }
  out
}
nn_tab <- function(m) {
  out <- list()
  for (r in rownames(m)) out[[r]] <- setNames(as.list(as.numeric(m[r, ])), colnames(m))
  out
}
summ <- function(m, ...) {
  a <- summary(m, pair.dist = FALSE, ...)
  list(all = tab(a$sum.all), matched = tab(a$sum.matched), nn = nn_tab(a$nn))
}
pairs_of <- function(m, ids) {
  mm <- m$match.matrix
  if (is.null(mm)) return(list())
  rows <- rownames(mm)
  out <- list()
  for (i in seq_len(nrow(mm))) {
    for (j in seq_len(ncol(mm))) {
      if (!is.na(mm[i, j])) out[[length(out) + 1]] <- c(rows[i], mm[i, j])
    }
  }
  # match.matrix is indexed by row name of the data; for ATC the rows are the
  # control units and the entries treated units.
  out
}

fit_one <- function(name, data, formula, addl, ...) {
  m <- matchit(formula, data = data, ...)
  rn <- rownames(data)
  prs <- pairs_of(m, data$id)
  # (treated id, control id) in every case
  prs <- lapply(prs, function(p) {
    a <- data$id[match(p[1], rn)]
    b <- data$id[match(p[2], rn)]
    if (data$treat[match(p[1], rn)] == 1) c(a, b) else c(b, a)
  })
  list(
    name = name,
    estimand = m$estimand,
    pairs = prs,
    weights = as.numeric(m$weights),
    standardised = summ(m, standardize = TRUE),
    unstandardised = summ(m, standardize = FALSE),
    interactions_std = summ(m, standardize = TRUE, interactions = TRUE),
    interactions_raw = summ(m, standardize = FALSE, interactions = TRUE),
    addl_std = summ(m, standardize = TRUE, addlvariables = addl, data = data),
    addl_raw = summ(m, standardize = FALSE, addlvariables = addl, data = data)
  )
}

rownames(lal) <- lal$id
rownames(sim) <- sim$id
fml_l <- treat ~ age + educ + race + married + nodegree + re74
fml_s <- treat ~ x1 + x2 + z + b + cnt + f

lal_fits <- list(
  fit_one("att_replace", lal, fml_l, ~ re75, method = "nearest", replace = TRUE),
  fit_one("atc_replace", lal, fml_l, ~ re75, method = "nearest", replace = TRUE,
          estimand = "ATC"),
  fit_one("att_ratio2", lal, fml_l, ~ re75, method = "nearest", ratio = 2),
  fit_one("att_noreplace", lal, fml_l, ~ re75, method = "nearest"),
  fit_one("ate_subclass", lal, fml_l, ~ re75, method = "subclass",
          estimand = "ATE", subclass = 6)
)
sim_fits <- list(
  fit_one("att_replace", sim, fml_s, ~ w, method = "nearest", replace = TRUE),
  fit_one("atc_replace", sim, fml_s, ~ w, method = "nearest", replace = TRUE,
          estimand = "ATC"),
  fit_one("att_ratio2", sim, fml_s, ~ w, method = "nearest", ratio = 2),
  fit_one("att_noreplace", sim, fml_s, ~ w, method = "nearest"),
  fit_one("ate_subclass", sim, fml_s, ~ w, method = "subclass",
          estimand = "ATE", subclass = 5)
)

# ---- zero variance: what MatchIt prints -------------------------------------
deg <- lal
deg$a <- ifelse(deg$treat == 1, 3, 0)
deg$k <- 1
deg$tz <- ifelse(deg$treat == 1, 3, rnorm(nrow(deg)))
md <- matchit(fml_l, data = deg, method = "nearest", replace = TRUE)
degen <- list(
  data = data.frame(id = deg$id, treat = deg$treat, a = deg$a, k = deg$k, tz = deg$tz),
  standardised = summ(md, standardize = TRUE, addlvariables = ~ a + k + tz, data = deg),
  unstandardised = summ(md, standardize = FALSE, addlvariables = ~ a + k + tz, data = deg),
  pairs = lapply(pairs_of(md, deg$id), function(p) c(deg$id[match(p[1], rownames(deg))],
                                                     deg$id[match(p[2], rownames(deg))])),
  weights = as.numeric(md$weights)
)

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  cobalt = as.character(packageVersion("cobalt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  lalonde = list(data = lal[, c("id", "treat", "age", "educ", "race", "married",
                                "nodegree", "re74", "re75")],
                 fits = lal_fits),
  sim = list(data = sim[, c("id", "treat", "x1", "x2", "z", "b", "cnt", "f", "w")],
             fits = sim_fits),
  degenerate = degen
)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
path <- file.path(dirname(script), "balance.json")
write_json(out, path, digits = 17, na = "string", auto_unbox = TRUE,
           dataframe = "columns")
