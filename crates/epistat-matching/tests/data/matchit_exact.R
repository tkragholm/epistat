# Reference exact matching from MatchIt's matchit(method = "exact").
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_exact.R
#
# For each data set and each estimand the script writes the subclass MatchIt
# gives every unit (null for a unit in a stratum that lacks one group), the
# weights MatchIt returns, which are rescaled so that the weights of each group
# sum to the number of its units that keep a weight, and the unscaled subclass
# weights computed here from the subclass counts: for ATT a control weighs
# n1 / n0 of its subclass and a treated unit 1, for ATC the reverse, and for
# ATE a treated unit weighs n / n1 and a control n / n0. The script checks that
# the two agree after rescaling, so the unscaled weights are tied to MatchIt's.
#
# Data (the exact variables are text in every set):
#   lalonde_2, lalonde_3  race and married; race, married and nodegree. The
#                         treated units come first, so MatchIt numbers the
#                         subclasses by the order of the treated units.
#   lalonde_3_shuffled    the same rows in random order, which starts with a
#                         control, so MatchIt numbers the subclasses by the
#                         order of the controls.
#   sim_*                 small data sets over 24 cells with empty cells and
#                         cells that hold one group only, in treated-first and
#                         shuffled order.
#   sim_disjoint          no stratum holds both groups, which MatchIt refuses.

suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

estimands <- c("ATT", "ATC", "ATE")

unscaled <- function(subclass, treat, estimand) {
  w <- rep(0, length(treat))
  for (s in sort(unique(subclass[!is.na(subclass)]))) {
    i <- which(subclass == s)
    n1 <- sum(treat[i] == 1)
    n0 <- sum(treat[i] == 0)
    n <- n1 + n0
    w[i] <- switch(estimand,
      ATT = ifelse(treat[i] == 1, 1, n1 / n0),
      ATC = ifelse(treat[i] == 1, n0 / n1, 1),
      ATE = ifelse(treat[i] == 1, n / n1, n / n0))
  }
  w
}

run_dataset <- function(name, treat, vars) {
  df <- data.frame(treat = treat, vars, stringsAsFactors = FALSE)
  f <- as.formula(paste("treat ~", paste(names(vars), collapse = " + ")))
  runs <- list()
  for (estimand in estimands) {
    m <- tryCatch(matchit(f, data = df, method = "exact", estimand = estimand),
                  error = function(e) NULL)
    if (is.null(m)) {
      runs[[estimand]] <- list(estimand = estimand, error = TRUE)
      next
    }
    sub <- ifelse(is.na(m$subclass), NA, as.integer(m$subclass))
    raw <- unscaled(sub, treat, estimand)
    scaled <- raw
    for (g in 0:1) {
      i <- which(treat == g & !is.na(sub))
      scaled[i] <- raw[i] * length(i) / sum(raw[i])
    }
    stopifnot(all(abs(scaled - unname(m$weights)) <=
                  1e-12 * pmax(1, abs(unname(m$weights)))))
    runs[[estimand]] <- list(estimand = estimand, error = FALSE,
                             subclass = sub, weights = unname(m$weights),
                             unscaled_weights = raw)
  }
  list(name = name, treat = treat, vars = as.list(vars), runs = unname(runs))
}

data(lalonde)
lal <- data.frame(
  treat = lalonde$treat, race = as.character(lalonde$race),
  married = as.character(lalonde$married),
  nodegree = as.character(lalonde$nodegree), stringsAsFactors = FALSE)

set.seed(20261120)
shuf <- sample(nrow(lal))

sim_data <- function(n) {
  data.frame(
    a = sample(c("p", "q", "r", "s"), n, TRUE),
    b = sample(c("u", "v", "w"), n, TRUE),
    c = sample(c("m", "f"), n, TRUE),
    stringsAsFactors = FALSE)
}
sim_treat <- function(v, n) {
  rbinom(n, 1, plogis(-0.3 + 0.8 * (v$a %in% c("p", "q")) - 0.6 * (v$c == "m")))
}
n1 <- 70
v1 <- sim_data(n1)
t1 <- sim_treat(v1, n1)
o1 <- order(-t1)
n2 <- 55
v2 <- sim_data(n2)
t2 <- sim_treat(v2, n2)
o2 <- sample(n2)
# Treated units in one half of the cells and controls in the other half.
n3 <- 30
v3 <- sim_data(n3)
t3 <- as.integer(v3$a %in% c("p", "q"))

datasets <- list(
  run_dataset("lalonde_2", lal$treat, lal[c("race", "married")]),
  run_dataset("lalonde_3", lal$treat, lal[c("race", "married", "nodegree")]),
  run_dataset("lalonde_3_shuffled", lal$treat[shuf],
              lal[shuf, c("race", "married", "nodegree")]),
  run_dataset("sim_treated_first", t1[o1], v1[o1, ]),
  run_dataset("sim_shuffled", t2[o2], v2[o2, ]),
  run_dataset("sim_disjoint", t3, v3))
datasets <- lapply(datasets, function(d) {
  d$vars <- lapply(d$vars, as.character)
  d
})

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  datasets = datasets)
write_json(out, "tests/data/matchit_exact.json", digits = I(17),
           auto_unbox = TRUE, na = "null")
