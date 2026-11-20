# Pairs of MatchIt 4.8.2's matchit(method = "nearest", distance = "glm",
# discard = policy, reestimate = TRUE): the units discarded from the first fit,
# a second fit on the others, and nearest neighbour matching of the kept units
# on the second fit's distance. For each data set, policy, ratio and caliper
# (in standard deviations of the distance) the file holds, per treated unit in
# row order, the units matched to it in order of assignment, and the discarded
# units.
#
# The covariates of lalonde are made unique by a perturbation of re74 far below
# any difference that matters, so that no two units share a score. The other
# data sets are simulated with continuous covariates.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_reestimate_match.R
suppressMessages({
  library(MatchIt)
  library(jsonlite)
})
args <- commandArgs(FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

data(lalonde)
set.seed(20261124)
lal <- lalonde[sample(nrow(lalonde), 400), ]
lal <- data.frame(
  treat = lal$treat, age = lal$age, educ = lal$educ,
  black = as.numeric(lal$race == "black"),
  hispan = as.numeric(lal$race == "hispan"),
  married = lal$married, nodegree = lal$nodegree,
  re74 = lal$re74, re75 = lal$re75)
lal$re74 <- lal$re74 + 1e-6 * seq_len(nrow(lal))

n <- 300
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.4)
x3 <- runif(n, 18, 60)
sim_cont <- data.frame(
  treat = rbinom(n, 1, plogis(-0.4 + 0.8 * x1 + 0.6 * x2 - 0.03 * (x3 - 40))),
  x1 = x1, x2 = x2, x3 = x3)

n <- 400
z1 <- rnorm(n)
z2 <- rnorm(n)
sim_overlap <- data.frame(
  treat = rbinom(n, 1, plogis(-1.2 + 1.6 * z1 + 0.5 * z2)),
  z1 = z1, z2 = z2)

runs <- list()
datasets <- list(lalonde = lal, sim_continuous = sim_cont, sim_overlap = sim_overlap)
for (name in names(datasets)) {
  df <- datasets[[name]]
  vars <- setdiff(names(df), "treat")
  f <- as.formula(paste("treat ~", paste(vars, collapse = " + ")))
  for (policy in c("treated", "control", "both")) {
    for (spec in list(list(ratio = 1, cal = NULL), list(ratio = 2, cal = NULL),
                      list(ratio = 1, cal = 0.25))) {
      m <- matchit(f, data = df, method = "nearest", distance = "glm",
                   discard = policy, reestimate = TRUE, ratio = spec$ratio,
                   caliper = spec$cal)
      mm <- m$match.matrix
      rows <- lapply(seq_len(nrow(mm)), function(i) {
        v <- mm[i, ]
        as.integer(v[!is.na(v)])
      })
      treated_rows <- as.integer(rownames(mm))
      runs[[length(runs) + 1]] <- list(
        data = name, policy = policy, ratio = spec$ratio, caliper = spec$cal,
        treated_rows = treated_rows, matchit = rows,
        discarded = which(m$discarded)
      )
    }
  }
}

out <- list(
  versions = list(r = R.version.string, matchit = as.character(packageVersion("MatchIt")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  data = lapply(datasets, function(df) list(
    treat = df$treat, names = setdiff(names(df), "treat"),
    covariates = lapply(setdiff(names(df), "treat"), function(v) df[[v]]))),
  runs = runs
)
write(toJSON(out, auto_unbox = TRUE, digits = I(17), null = "null", na = "null"),
      file.path(here, "matchit_reestimate_match.json"))
cat(length(runs), "runs\n")
