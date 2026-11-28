# Pair matching under the ATC with MatchIt 4.8.2, and its balance with
# summary.matchit() and cobalt 5.1.0's bal.tab(), for tests/r_atc.rs.
#
# Under estimand = "ATC" the control group is focal: each control unit is
# given treated units, match.matrix has the controls as rows, the controls
# weigh 1 and a treated unit the sum of 1/m over the controls it serves
# (matchit(), "How Matching Weights Are Computed"). The standardised mean
# differences are treated minus control, divided by the standard deviation of
# the controls before matching (summary.matchit(); bal.tab() picks
# s.d.denom = "control" for the ATC).
#
# Data: lalonde with race as two indicators, and the propensity score of a
# logistic model on the eight covariates, perturbed by 1e-10 times the row
# number so that no two units share a distance. The score is given to
# matchit() as a numeric distance, so the crate matches on the same numbers.
# A second set keeps every treated unit and 80 controls, so that every
# control can take two treated units without replacement. Absolute
# differences on a line give optimal matching many optima, so the optimal run
# is given a treated x control matrix of those differences plus uniform noise
# below 1e-3, whose optimum is unique: a second, smaller perturbation of the
# noise leaves match.matrix as it is. The matrix is written out. optmatch is
# given tol = 1e-9, as in matchit_optimal.R, so that its solution is the
# optimum and not one within its default relative tolerance of 1e-3.
#
# Runs: nearest 1:2 with replacement on the whole set, and nearest 1:1 and
# 1:2 without replacement and optimal 1:2 on the smaller one, the nearest runs
# with the default m.order (smallest under the ATC). Every focal control is matched in
# every run. Per run the file
# holds match.matrix as focal id and matched ids in assignment order, the
# weights with normalize = FALSE and TRUE, and the standardised mean
# differences of every covariate before and after matching from summary()
# and from bal.tab(binary = "std").
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/matchit_atc.R
suppressMessages({
  library(MatchIt)
  library(cobalt)
  library(jsonlite)
})
set.seed(20261128)

data(lalonde, package = "MatchIt")
lal <- lalonde
lal$id <- rownames(lal)
lal$black <- as.numeric(lal$race == "black")
lal$hispan <- as.numeric(lal$race == "hispan")
covariates <- c("age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75")
fml <- treat ~ age + educ + black + hispan + married + nodegree + re74 + re75
fit <- glm(fml, data = lal, family = binomial)
lal$ps <- unname(predict(fit, type = "response")) + 1e-10 * seq_len(nrow(lal))

small <- lal[sort(c(which(lal$treat == 1), sample(which(lal$treat == 0), 80))), ]

tr <- which(small$treat == 1)
co <- which(small$treat == 0)
noise <- matrix(runif(length(tr) * length(co), 0, 1e-3), length(tr))
noisy <- abs(outer(small$ps[tr], small$ps[co], "-")) + noise
dimnames(noisy) <- list(small$id[tr], small$id[co])

run <- function(name, d, method, ratio, replace, distance = d$ps) {
  args <- list(fml, data = d, method = method, distance = distance, estimand = "ATC",
               ratio = ratio)
  if (method == "nearest") args$replace <- replace
  if (method == "optimal") args$tol <- 1e-9
  m <- suppressWarnings(do.call(matchit, args))
  if (is.matrix(distance)) {
    again <- args
    again$distance <- distance + matrix(runif(length(distance), 0, 1e-9), nrow(distance))
    stopifnot(identical(suppressWarnings(do.call(matchit, again))$match.matrix,
                        m$match.matrix))
  }
  raw <- suppressWarnings(do.call(matchit, c(args, list(normalize = FALSE))))
  stopifnot(identical(m$match.matrix, raw$match.matrix))
  mm <- m$match.matrix
  sets <- lapply(seq_len(nrow(mm)), function(i) {
    got <- mm[i, ]
    list(focal = rownames(mm)[i], matched = unname(got[!is.na(got)]))
  })
  stopifnot(all(vapply(sets, function(s) length(s$matched) == ratio, logical(1))))
  s <- summary(m, standardize = TRUE)
  b <- bal.tab(m, un = TRUE, binary = "std", disp = c("means"))$Balance
  smd <- function(table, col) unname(table[covariates, col])
  list(
    name = name, method = method, ratio = ratio, replace = replace,
    ids = d$id, treat = d$treat,
    sets = sets,
    weights_raw = unname(raw$weights),
    weights = unname(m$weights),
    summary_smd_all = smd(s$sum.all, "Std. Mean Diff."),
    summary_smd_matched = smd(s$sum.matched, "Std. Mean Diff."),
    bal_tab_smd_un = smd(b, "Diff.Un"),
    bal_tab_smd_adj = smd(b, "Diff.Adj")
  )
}

runs <- list(
  run("small_nearest_1_without", small, "nearest", 1, FALSE),
  run("nearest_2_with", lal, "nearest", 2, TRUE),
  run("small_nearest_2_without", small, "nearest", 2, FALSE),
  run("small_optimal_2", small, "optimal", 2, FALSE, noisy)
)

out <- list(
  versions = list(r = R.version.string,
                  matchit = as.character(packageVersion("MatchIt")),
                  cobalt = as.character(packageVersion("cobalt")),
                  optmatch = as.character(packageVersion("optmatch")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  covariates = covariates,
  data = lal[, c("id", "treat", covariates, "ps")],
  matrix = list(treated = rownames(noisy), controls = colnames(noisy),
                values = unname(split(noisy, row(noisy)))),
  runs = runs
)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
write_json(out, file.path(dirname(script), "matchit_atc.json"), digits = 17,
           na = "string", auto_unbox = TRUE, dataframe = "columns")
