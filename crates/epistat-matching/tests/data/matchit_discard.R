# Reference common-support trimming from MatchIt's `discard` argument.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_discard.R
#
# For each case the script writes the distance MatchIt used and, for each of
# discard = "none", "treated", "control" and "both", which units it discarded.
# `discard` trims on the distance as it is returned, so a linear-scale distance
# is trimmed on the linear scale.
#
# Cases:
#   lalonde, sim_discrete    glm distance, on the probability and the linear
#                            predictor scale. sim_discrete has three discrete
#                            covariates, so many units share a score.
#   grid_*                   distance supplied as a numeric vector on a grid of
#                            0.05, so that units sit exactly at the ends of the
#                            other group's range, in both groups.
#   disjoint                 groups whose ranges do not overlap.

suppressMessages({
  library(MatchIt)
  library(jsonlite)
})

policies <- c("none", "treated", "control", "both")

discards <- function(df, distance, link = "logit") {
  f <- treat ~ z
  res <- list()
  for (policy in policies) {
    m <- if (is.character(distance)) {
      matchit(as.formula(paste("treat ~", paste(setdiff(names(df), "treat"),
                                                collapse = " + "))),
              data = df, method = NULL, distance = distance, link = link,
              discard = policy)
    } else {
      matchit(f, data = df, method = NULL, distance = distance,
              discard = policy)
    }
    res[[policy]] <- unname(m$discarded)
    dist <- unname(m$distance)
  }
  list(distance = dist, discarded = res)
}

cases <- list()
add <- function(name, scale, treat, d) {
  cases[[length(cases) + 1]] <<- list(
    name = name, scale = scale, treat = treat, distance = d$distance,
    discarded = d$discarded)
}

data(lalonde)
lal <- data.frame(
  treat = lalonde$treat, age = lalonde$age, educ = lalonde$educ,
  black = as.numeric(lalonde$race == "black"),
  hispan = as.numeric(lalonde$race == "hispan"),
  married = lalonde$married, nodegree = lalonde$nodegree,
  re74 = lalonde$re74, re75 = lalonde$re75)

set.seed(20261120)
n <- 400
d1 <- rbinom(n, 1, 0.5)
d2 <- sample(0:3, n, replace = TRUE)
d3 <- sample(0:2, n, replace = TRUE)
sim <- data.frame(
  treat = rbinom(n, 1, plogis(-0.5 + 0.9 * d1 + 0.5 * d2 - 0.4 * d3)),
  d1 = d1, d2 = d2, d3 = d3)
# Shape the ends of the score range. The lowest-score pattern has only
# controls and the highest only treated units. The second lowest has a single
# treated unit and the second highest a single control, so those units and
# their pattern sit exactly at the end of the other group's range.
cell <- function(a, b, c) which(d1 == a & d2 == b & d3 == c)
sim$treat[cell(0, 0, 2)] <- 0
sim$treat[cell(1, 3, 0)] <- 1
sim$treat[cell(0, 0, 1)] <- c(1, rep(0, length(cell(0, 0, 1)) - 1))
sim$treat[cell(1, 3, 1)] <- c(0, rep(1, length(cell(1, 3, 1)) - 1))

for (nm in c("lalonde", "sim_discrete")) {
  df <- if (nm == "lalonde") lal else sim
  add(nm, "probability", df$treat, discards(df, "glm", "logit"))
  add(nm, "linear", df$treat, discards(df, "glm", "linear.logit"))
}

grid <- function(from, to) seq(from, to, by = 0.05)
make_grid <- function(treated_from, treated_to, control_from, control_to,
                      n_t, n_c) {
  # The ends of both ranges are always present, so units sit exactly at the
  # boundary, and the rest are drawn from the grid with repeats.
  zt <- c(treated_from, treated_to, sample(grid(treated_from, treated_to), n_t - 2, replace = TRUE))
  zc <- c(control_from, control_to, sample(grid(control_from, control_to), n_c - 2, replace = TRUE))
  list(treat = c(rep(1, n_t), rep(0, n_c)), p = round(c(zt, zc), 2))
}
grid_cases <- list(
  grid_treated_narrower = make_grid(0.25, 0.75, 0.05, 0.9, 30, 60),
  grid_controls_narrower = make_grid(0.1, 0.95, 0.3, 0.7, 40, 30),
  grid_shifted = make_grid(0.3, 0.95, 0.05, 0.7, 30, 50),
  grid_equal_ends = make_grid(0.2, 0.8, 0.2, 0.8, 25, 25))
for (nm in names(grid_cases)) {
  g <- grid_cases[[nm]]
  df <- data.frame(treat = g$treat, z = seq_along(g$treat))
  add(nm, "probability", g$treat, discards(df, g$p))
  add(nm, "linear", g$treat, discards(df, qlogis(g$p)))
}
dj <- data.frame(treat = c(1, 1, 1, 0, 0, 0), z = 1:6)
add("disjoint", "probability", dj$treat,
    discards(dj, c(0.1, 0.2, 0.3, 0.6, 0.7, 0.8)))

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  cases = cases)
write_json(out, "tests/data/matchit_discard.json", digits = I(17),
           auto_unbox = TRUE)
