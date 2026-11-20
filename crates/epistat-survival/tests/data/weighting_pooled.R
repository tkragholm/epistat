# Pooled logistic censoring weights, checked against ipw::ipwtm(type = "cens",
# family = "binomial") and against weights built from glm and cumprod directly.
# Person-period data in the layout of a target trial emulation: one row per
# subject and interval, a baseline treatment and covariates, a covariate that
# changes between intervals, a censoring indicator and an event indicator.
# Case 1: unstabilised weights. Case 2: stabilised weights. Case 3: stabilised
# weights on data that continue after the censoring row (ipwtm keeps the weight of
# the censoring row there).
# The versions are in the file. Numbers are written with 17 digits.
# Run: Rscript weighting_pooled.R > weighting_pooled.json
suppressMessages({
  library(jsonlite)
  library(ipw)
})
set.seed(20261121)
n <- 300
sim_pp <- function(n) {
  rows <- list()
  for (i in seq_len(n)) {
    x1 <- round(rnorm(1), 3)
    x2 <- rbinom(1, 1, 0.4)
    a <- rbinom(1, 1, plogis(0.3 * x1 - 0.2 * x2))
    l <- round(rnorm(1), 3)
    k <- 0
    repeat {
      k <- k + 1
      l <- round(0.6 * l + rnorm(1, 0, 0.7) + 0.2 * a, 3)
      ev <- rbinom(1, 1, plogis(-3.2 + 0.4 * l - 0.5 * a + 0.3 * x1))
      ce <- if (ev == 1) 0 else rbinom(1, 1, plogis(-2.4 + 0.6 * l + 0.4 * a - 0.3 * x2 + 0.05 * k))
      rows[[length(rows) + 1]] <- data.frame(id = i, k = k, a = a, x1 = x1, x2 = x2, l = l,
                                             event = ev, cens = ce)
      if (ev == 1 || ce == 1) break
    }
  }
  d <- do.call(rbind, rows)
  d$start <- d$k - 1
  d$stop <- d$k
  d
}
d <- sim_pp(n)
fden <- ~ a + x1 + x2 + l + k
fnum <- ~ a + x1 + k

hand <- function(dat, stab) {
  # the rows up to and including the first censoring row of each subject
  sel <- ave(dat$cens, dat$id, FUN = function(x) c(0, cumsum(x)[-length(x)])) == 0
  md <- glm(update(fden, cens ~ .), data = dat[sel, ], family = binomial)
  pd <- predict(md, newdata = dat, type = "response")
  pnum <- rep(NA_real_, nrow(dat)); cn <- NULL; Xn <- NULL; kn <- NA
  fd <- ifelse(sel, 1 - pd, 1)
  fn <- rep(1, nrow(dat))
  if (stab) {
    mn <- glm(update(fnum, cens ~ .), data = dat[sel, ], family = binomial)
    pnum <- predict(mn, newdata = dat, type = "response")
    fn <- ifelse(sel, 1 - pnum, 1)
    cn <- unname(coef(mn))
  }
  w <- ave(fn / fd, dat$id, FUN = cumprod)
  # weight before the row: the product over earlier rows of the subject
  wbefore <- w / (fn / fd)
  kap <- function(f, p) {
    X <- unname(model.matrix(f, dat[sel, ]))
    g <- crossprod(X * sqrt((p * (1 - p))[sel]))
    s <- sqrt(diag(g))
    kappa(diag(1 / s) %*% g %*% diag(1 / s), exact = TRUE)
  }
  list(weights = unname(w), weights_before = unname(wbefore),
       probability = unname(pd), coefficients = unname(coef(md)),
       numerator_probability = if (stab) unname(pnum) else NULL,
       numerator_coefficients = cn, active = as.integer(sel),
       kappa_den = kap(fden, pd), kappa_num = if (stab) kap(fnum, pnum) else NULL)
}

case <- function(name, dat, stab) {
  ip <- ipwtm(exposure = cens, family = "binomial", link = "logit",
              numerator = if (stab) fnum else NULL, denominator = fden,
              id = id, timevar = k, type = "cens", data = dat)
  h <- hand(dat, stab)
  c(list(name = name, stabilise = stab, id = dat$id, start = dat$start, stop = dat$stop,
         status = dat$event, censored = dat$cens,
         design = unname(model.matrix(fden, dat)),
         numerator_design = if (stab) unname(model.matrix(fnum, dat)) else NULL,
         ipw_weights = unname(ip$ipw.weights)), h)
}

# Rows after the first censoring row: ipwtm gives them the weight of the censoring row.
ext <- d
cs <- which(d$cens == 1)
cs <- cs[seq(1, length(cs), by = 6)]
extra <- do.call(rbind, lapply(cs, function(r) {
  m <- d[r, ]
  do.call(rbind, lapply(1:2, function(j) {
    z <- m; z$k <- m$k + j; z$start <- z$k - 1; z$stop <- z$k
    z$l <- round(m$l + rnorm(1, 0, 0.5), 3); z$event <- 0; z$cens <- 0; z
  }))
}))
ext <- rbind(d, extra)
ext <- ext[order(ext$id, ext$k), ]
rownames(ext) <- NULL

out <- list(
  versions = list(r = R.version.string, ipw = as.character(packageVersion("ipw")),
                  jsonlite = as.character(packageVersion("jsonlite"))),
  cases = list(case("pooled", d, FALSE), case("pooled_stab", d, TRUE),
               case("pooled_stab_extra_rows", ext, TRUE))
)
cat(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", na = "null"))
