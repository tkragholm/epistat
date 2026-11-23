# Distributions of mice's imputations, for comparison with epistat-impute.
#
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/impute_distribution.R
#
# Two data sets: mice::nhanes2 (25 rows, m = 500) and a simulated register-like
# set (5,000 rows, m = 200): a complete normal x1 and binary x2, a skewed
# income `inc`, a three-level nominal `edu`, a binary `smk` and a continuous
# outcome `y`, with `inc`, `edu` and `smk` missing at random given the
# complete variables (about a third of the rows each). The simulated set
# is imputed twice: all three variables by pmm, polyreg and logreg, and `inc`
# alone by norm.
#
# For each imputed variable the file holds one row per data set with the mean,
# variance and the 0.1, 0.5 and 0.9 quantiles of the imputed values (the
# proportion of each level for a nominal variable), and for each analysis the
# estimate and variance of its coefficients in each completed data set, so
# that Rubin's rules can be applied by the test. The analyses are
# lm(chl ~ bmi + hyp) for nhanes2 and lm(y ~ x1 + inc + smk) for the simulated
# set. The data are written with the file so the test imputes the same values.

suppressMessages({
  library(mice)
  library(jsonlite)
})

set_stats <- function(imp, var, kind) {
  cells <- imp$imp[[var]]
  out <- t(apply(cells, 2, function(v) {
    if (kind == "num") {
      v <- as.numeric(v)
      c(mean(v), var(v), quantile(v, c(0.1, 0.5, 0.9), names = FALSE))
    } else if (kind == "bin") {
      v <- as.numeric(as.character(v))
      c(mean(v))
    } else {
      v <- as.character(v)
      c(mean(v == "1"), mean(v == "2"))
    }
  }))
  # One statistic per set: a matrix with a single column, not a vector.
  if (kind == "bin") out <- matrix(as.vector(out), ncol = 1)
  out
}

analysis <- function(imp, formula, terms) {
  est <- var <- matrix(NA_real_, imp$m, length(terms))
  for (i in seq_len(imp$m)) {
    fit <- lm(formula, data = complete(imp, i))
    s <- summary(fit)$coefficients
    est[i, ] <- s[terms, "Estimate"]
    var[i, ] <- s[terms, "Std. Error"]^2
  }
  list(terms = terms, estimate = est, variance = var)
}

versions <- list(r = R.version.string, mice = as.character(packageVersion("mice")))

# ---- nhanes2 ---------------------------------------------------------------
nh <- nhanes2
nh$hyp <- factor(as.integer(nh$hyp) - 1L)
meth <- c(age = "", bmi = "pmm", hyp = "logreg", chl = "pmm")
imp1 <- mice(nh, m = 500, maxit = 5, method = meth, printFlag = FALSE, seed = 20261123)
nhanes <- list(
  age = as.integer(nh$age) - 1L, bmi = nh$bmi, hyp = as.integer(as.character(nh$hyp)), chl = nh$chl,
  stats = list(bmi = set_stats(imp1, "bmi", "num"), hyp = set_stats(imp1, "hyp", "bin"),
               chl = set_stats(imp1, "chl", "num")),
  analysis = analysis(imp1, chl ~ bmi + hyp, c("bmi", "hyp1")))
meth2 <- c(age = "", bmi = "norm", hyp = "logreg", chl = "norm")
imp1n <- mice(nh, m = 500, maxit = 5, method = meth2, printFlag = FALSE, seed = 20261124)
nhanes$norm <- list(stats = list(bmi = set_stats(imp1n, "bmi", "num"), chl = set_stats(imp1n, "chl", "num")),
                    analysis = analysis(imp1n, chl ~ bmi + hyp, c("bmi", "hyp1")))

# ---- simulated register-like data -----------------------------------------
set.seed(20261125)
n <- 2000
x1 <- round(rnorm(n), 4)
x2 <- rbinom(n, 1, 0.5)
inc <- round(exp(3 + 0.3 * x1 + 0.2 * x2 + 0.5 * rnorm(n)), 3)
li <- log(inc) - 3
eta1 <- -0.2 + 0.5 * x1 + 0.6 * li
eta2 <- -0.5 - 0.3 * x1 + 0.9 * li
pe <- cbind(1, exp(eta1), exp(eta2)); pe <- pe / rowSums(pe)
edu <- apply(pe, 1, function(p) sample(0:2, 1, prob = p))
smk <- rbinom(n, 1, plogis(-0.5 - 0.4 * x1 + 0.3 * x2 - 0.4 * (edu == 2)))
y <- round(2 + 0.5 * x1 + 0.05 * inc + 0.8 * smk + rnorm(n), 4)
ys <- (y - mean(y)) / sd(y)
m_inc <- rbinom(n, 1, plogis(-1 + 0.8 * x1 + 0.5 * ys)) == 1
m_edu <- rbinom(n, 1, plogis(-1.2 + 0.6 * x2 + 0.4 * x1)) == 1
m_smk <- rbinom(n, 1, plogis(-1 + 0.5 * x1 - 0.3 * ys)) == 1
d <- data.frame(x1 = x1, x2 = x2, inc = inc, edu = factor(edu), smk = factor(smk), y = y)
dm <- d
dm$inc[m_inc] <- NA; dm$edu[m_edu] <- NA; dm$smk[m_smk] <- NA
cat("missing shares:", mean(m_inc), mean(m_edu), mean(m_smk), "\n")

meth <- c(x1 = "", x2 = "", inc = "pmm", edu = "polyreg", smk = "logreg", y = "")
t0 <- Sys.time()
imp2 <- mice(dm, m = 200, maxit = 5, method = meth, printFlag = FALSE, seed = 20261126)
cat("mice A", format(Sys.time() - t0), "\n")
f <- y ~ x1 + inc + smk
sim <- list(
  x1 = x1, x2 = x2, inc = inc, edu = edu, smk = smk, y = y,
  missing_inc = m_inc, missing_edu = m_edu, missing_smk = m_smk,
  stats = list(inc = set_stats(imp2, "inc", "num"), edu = set_stats(imp2, "edu", "nom"),
               smk = set_stats(imp2, "smk", "bin")),
  analysis = analysis(imp2, f, c("inc", "smk1")))
# inc alone, by norm
dn <- d; dn$inc[m_inc] <- NA
imp3 <- mice(dn, m = 200, maxit = 5, method = c(x1 = "", x2 = "", inc = "norm", edu = "", smk = "", y = ""),
             printFlag = FALSE, seed = 20261127)
sim$norm <- list(stats = list(inc = set_stats(imp3, "inc", "num")),
                 analysis = analysis(imp3, f, c("inc", "smk1")))
# the same by pmm, inc alone
imp4 <- mice(dn, m = 200, maxit = 5, method = c(x1 = "", x2 = "", inc = "pmm", edu = "", smk = "", y = ""),
             printFlag = FALSE, seed = 20261128)
sim$pmm <- list(stats = list(inc = set_stats(imp4, "inc", "num")),
                analysis = analysis(imp4, f, c("inc", "smk1")))

write(toJSON(list(versions = versions, nhanes = nhanes, sim = sim), auto_unbox = TRUE,
             digits = NA, matrix = "rowmajor", na = "null"),
      "tests/data/impute_distribution.json")
