# Likelihood-ratio tests between Cox models by survival::anova.coxph and
# anova.coxphlist, on the lung data and on a simulated counting-process data
# set with strata and case weights, with Efron and Breslow ties.
# For each case: the data and the design matrix of the full model with the
# columns of each term, the table of anova(fit) (the null model, then each term
# added in order) and the table of anova(fit_1, ..., fit_m) for the nested
# models with the first 1, ..., m terms. A table has the log likelihood, the
# chi-square statistic, the degrees of freedom and the p-value of each row, NA
# (written null) where R has none.
# The versions of R, survival and jsonlite are in the file. Numbers are written
# with 17 significant digits.
# Run: Rscript cox_anova.R > cox_anova.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)

n <- 150
x1 <- round(rnorm(n), 3)
x2 <- round(runif(n, 0, 2), 3)
g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
x3 <- rbinom(n, 1, 0.45)
stop <- pmax(ceiling(rexp(n, 0.08 * exp(0.4 * x1 - 0.5 * x3 + 0.3 * x2))), 2)
start <- pmin(sample(0:5, n, replace = TRUE), stop - 1)
event <- rbinom(n, 1, 0.75)
stratum <- sample(0:2, n, replace = TRUE)
weight <- round(runif(n, 0.5, 3), 2)
sim <- data.frame(start, stop, event, stratum, weight, x1, x2, g, x3)

l <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog", "wt.loss")])
l$ecog <- factor(l$ph.ecog)
lungd <- data.frame(start = 0, stop = l$time, event = as.integer(l$status == 2),
                    stratum = 0, weight = 1, x1 = l$age, x2 = l$wt.loss,
                    g = l$ecog, x3 = as.integer(l$sex == 2))

num <- function(x) ifelse(is.na(x), NA_real_, as.numeric(x))
table_of <- function(a) {
  list(loglik = num(a$loglik), chisq = num(a$Chisq), df = num(a$Df),
       p = num(a[["Pr(>|Chi|)"]]))
}

one <- function(name, d, ties, entry, strata, weights) {
  surv <- if (entry) "Surv(start, stop, event)" else "Surv(stop, event)"
  rhs <- c("x1", "g", "x2", "x3")
  if (strata) rhs <- c(rhs, "strata(stratum)")
  fml <- function(terms) as.formula(paste(surv, "~", paste(terms, collapse = " + ")))
  w <- if (weights) d$weight else rep(1, nrow(d))
  d$w <- w
  assign("d", d, envir = globalenv())
  fitk <- function(terms) {
    do.call(coxph, list(fml(terms), data = quote(d), weights = quote(w), ties = ties, robust = FALSE))
  }
  full <- fitk(rhs)
  mm <- model.matrix(full)
  asg <- attr(mm, "assign")
  # terms x1, g, x2, x3 are the first four terms; strata is not a column
  groups <- lapply(1:4, function(k) I(which(asg == k) - 1L))
  fits <- lapply(1:4, function(k) fitk(rhs[c(seq_len(k), if (strata) 5)]))
  tab_multi <- do.call(anova, fits)
  tab_single <- anova(full)
  # drop the strata row of the single table, which is not a term of the design
  keep <- !grepl("strata", rownames(tab_single))
  # anova.coxph refits the intermediate models without the case weights
  # (survival 3.8-12), so with weights its table is not the sequential test of
  # the weighted models. single_definition is that test from the weighted
  # fits, with the null model's log likelihood from the full fit.
  ll <- c(full$loglik[1], sapply(fits, function(f) f$loglik[2]))
  dfs <- diff(c(0, sapply(fits, function(f) sum(!is.na(coef(f))))))
  chisq <- 2 * diff(ll)
  definition <- list(loglik = ll, chisq = c(NA, chisq), df = c(NA, dfs),
                     p = c(NA, pchisq(chisq, dfs, lower.tail = FALSE)))
  list(
    name = name, ties = ties, entry = entry, strata = strata, weights = weights,
    x = unname(mm), groups = groups,
    start = d$start, stop = d$stop, event = d$event, stratum = d$stratum,
    weight = w,
    single = table_of(tab_single[keep, ]),
    single_definition = definition,
    multi = table_of(tab_multi)
  )
}

cases <- list(
  one("sim_efron_plain", sim, "efron", FALSE, FALSE, FALSE),
  one("sim_efron_entry_strata_weights", sim, "efron", TRUE, TRUE, TRUE),
  one("sim_breslow_strata", sim, "breslow", FALSE, TRUE, FALSE),
  one("lung_efron", lungd, "efron", FALSE, FALSE, FALSE),
  one("lung_efron_weights", transform(lungd, weight = round(runif(nrow(lungd), 0.5, 2), 2)),
      "efron", FALSE, FALSE, TRUE)
)
cat(toJSON(list(
  versions = list(
    r = R.version.string,
    survival = as.character(packageVersion("survival")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  cases = cases
), digits = I(17), auto_unbox = TRUE, na = "null"))
