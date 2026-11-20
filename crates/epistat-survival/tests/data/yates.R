# Population marginal means and their tests, as survival::yates, for a linear
# model, Poisson and logistic regression and a Cox model.
# For each case: the coefficients and covariance of the fit, the design matrix
# of the population of every level (the rows yates averages over, built by the
# internal functions of survival that yates itself calls, so that what is
# compared is the averaging, the variance and the tests), the centring vector
# of a Cox fit, and the yates result: estimate, std, the covariance of the
# estimates and the test table.
# The linear predictor has an exact variance. For the response and the risk,
# yates simulates `nsim` coefficient vectors (here 200000, so that its
# Monte Carlo error is small), and the file also holds the delta method
# computed here from a central difference of the estimates in the
# coefficients, which is what the Rust code evaluates analytically.
# Versions of R, survival and jsonlite are in the file. Numbers are written
# with 17 significant digits.
# Run: Rscript yates.R > yates.json
suppressMessages({
  library(jsonlite)
  library(survival)
})
set.seed(20261120)
nsim <- 200000

num <- function(m) {
  m <- as.matrix(m)
  lapply(seq_len(nrow(m)), function(i) I(as.numeric(m[i, ])))
}

# The populations, as yates builds them.
populations <- function(fit, term, population, test) {
  Terms <- delete.response(terms(fit))
  Tatt <- attributes(Terms)
  Tatt$dataClasses <- Tatt$dataClasses[row.names(Tatt$factors)]
  iscat <- sapply(Tatt$dataClasses, function(x) x %in% c("character", "factor"))
  mframe <- fit$model
  if (is.null(mframe)) mframe <- model.frame(fit)
  xassign <- if (is.null(fit$assign)) attr(model.matrix(fit), "assign") else fit$assign
  contr <- survival:::cmatrix(fit, term, test, assign = xassign)
  survival:::yates_xmat(Terms, Tatt, contr, population, mframe, fit, iscat)
}

delta_sd <- function(pmmfun, beta, vmat) {
  p <- length(beta)
  g <- sapply(seq_len(p), function(j) {
    h <- 1e-6 * max(abs(beta[j]), 1)
    b1 <- beta; b1[j] <- b1[j] + h
    b2 <- beta; b2[j] <- b2[j] - h
    (pmmfun(b1) - pmmfun(b2)) / (2 * h)
  })
  g <- matrix(g, ncol = p)
  list(var = g %*% vmat %*% t(g), gradient = g)
}

run <- function(name, fit, term, population, predict, test, link, cox = FALSE) {
  xm <- populations(fit, term, population, test)
  beta <- coef(fit)
  vmat <- vcov(fit)
  if (cox) {
    xm <- lapply(xm, function(x) x[, -1, drop = FALSE])
    centre <- as.numeric(fit$means)
  } else centre <- NULL
  set.seed(1953)
  y <- yates(fit, term, population = population, test = test, predict = predict, nsim = nsim)
  linkinv <- switch(link, identity = function(e) e, exp = exp, logit = plogis)
  cen <- if (is.null(centre)) rep(0, length(beta)) else centre
  pmmfun <- function(b) sapply(xm, function(x) mean(linkinv(sweep(x, 2, cen) %*% b)))
  dl <- delta_sd(pmmfun, as.numeric(beta), vmat)
  dvar <- dl$var
  xassign <- if (is.null(fit$assign)) attr(model.matrix(fit), "assign") else fit$assign
  cmat <- survival:::cmatrix(fit, term, test, assign = xassign)$cmat
  pmm0 <- pmmfun(as.numeric(beta))
  dtests <- if (is.list(cmat)) t(sapply(cmat, function(x) survival:::testfun(x, pmm0, dvar, NULL)))
            else matrix(survival:::testfun(cmat, pmm0, dvar, NULL), nrow = 1,
                        dimnames = list("global", c("chisq", "df")))
  list(name = name, link = link, population = population, predict = predict, test = test,
       coefficients = I(as.numeric(beta)), var = num(vmat), centre = centre,
       designs = lapply(xm, num),
       estimate = I(as.numeric(y$estimate$pmm)), std = I(as.numeric(y$estimate$std)),
       mvar = num(y$mvar), tests = lapply(seq_len(nrow(y$test)), function(i)
         list(label = rownames(y$test)[i], chisq = unname(y$test[i, "chisq"]), df = unname(y$test[i, "df"]))),
       nsim = if (link == "identity") NULL else nsim,
       delta_mvar = num(dvar), gradient = num(dl$gradient), fd_step = 1e-6,
       delta_tests = lapply(seq_len(nrow(dtests)), function(i)
         list(label = rownames(dtests)[i], chisq = unname(dtests[i, "chisq"]), df = unname(dtests[i, "df"]))),
       pmm_direct = I(as.numeric(pmm0)))
}

cases <- list()
sold <- solder
fit_lm <- lm(skips ~ Opening + Solder + Mask + PadType + Panel, data = sold, subset = (Mask != "A6"))
fit_lm2 <- lm(skips ~ Opening + Mask * PadType + Panel, data = sold, subset = (Mask != "A6"))
cases[[1]] <- run("lm_factorial_global", fit_lm, "Opening", "factorial", "linear", "global", "identity")
cases[[2]] <- run("lm_data_pairwise", fit_lm, "Opening", "data", "linear", "pairwise", "identity")
cases[[3]] <- run("lm_interaction_factorial", fit_lm2, "Opening", "factorial", "linear", "global", "identity")
fit_pois <- glm(skips ~ Opening + Mask + PadType + Solder, data = sold, family = poisson)
cases[[4]] <- run("pois_link_data", fit_pois, "Opening", "data", "link", "pairwise", "identity")
cases[[5]] <- run("pois_response_data", fit_pois, "Opening", "data", "response", "pairwise", "exp")
cases[[6]] <- run("pois_response_factorial", fit_pois, "Opening", "factorial", "response", "global", "exp")
lg <- transform(lung, ecog = factor(ph.ecog), male = factor(sex, 1:2, c("m", "f")), dead = as.numeric(status == 2))
lg <- lg[complete.cases(lg[, c("ecog", "male", "age", "dead")]) & lg$ph.ecog != 3, ]
lg$ecog <- droplevels(lg$ecog)
fit_logit <- glm(dead ~ ecog * male + age, data = lg, family = binomial)
cases[[7]] <- run("logit_response_data", fit_logit, "ecog", "data", "response", "global", "logit")
cases[[8]] <- run("logit_response_sas", fit_logit, "ecog", "sas", "response", "pairwise", "logit")
fit_cox <- coxph(Surv(time, status) ~ ecog * male + age, data = lg)
cases[[9]] <- run("cox_linear_data", fit_cox, "ecog", "data", "linear", "global", "identity", cox = TRUE)
cases[[10]] <- run("cox_risk_data", fit_cox, "ecog", "data", "risk", "pairwise", "exp", cox = TRUE)
cases[[11]] <- run("cox_risk_sas", fit_cox, "ecog", "sas", "risk", "global", "exp", cox = TRUE)

vers <- list(r = R.version.string, survival = as.character(packageVersion("survival")),
             jsonlite = as.character(packageVersion("jsonlite")))
cat(toJSON(list(versions = vers, cases = cases), digits = I(17), auto_unbox = TRUE,
           null = "null", na = "null"))
