# Expected survival of a cohort, Ederer I and Hakulinen relative survival,
# age-standardised net survival, the crude probabilities of death and the test
# of equal net survival, against R references.
#
# The rate table is the synthetic daily table of relsurv.R: Gompertz in age in
# days, a factor for sex and a trend in the calendar year, whose first and last
# cells reach without limit as in every R ratetable. Deaths fall on whole days,
# censorings on half days, and entries at censoring times. The sources:
#  * survival::survexp for the expected survival of Ederer I ("ederer"),
#    Hakulinen ("hakulinen", the potential follow-up as the response) and the
#    conditional curve ("conditional"), with times that include every exit and
#    potential follow-up time, where survexp's interval weighting gives the
#    exact values.
#  * relsurv::rs.surv with method "ederer1" and "hakulinen" and type
#    "fleming-harrington". Its Ederer I is exact. Its Hakulinen hazard is the
#    sum of S*_i(b) (H*_i(b) - H*_i(a)) over the sum of S*_i(b) on each step,
#    a rule that differs from the exact log of the ratio of the sums by less
#    than the largest squared increment H*_i(b) - H*_i(a) of the step; the sum
#    of those over the steps is written out as the bound. The exact Hakulinen
#    relative survival is the observed survival over survexp's curve.
#  * relsurv::rs.surv(method = "ederer2") by age group with precision 0.5,
#    which is exact without delayed entry, and the age-standardised curve with
#    the ICSS 1 weights written out from it.
#  * relsurv::cmp.rel for the crude probabilities and their variances, with
#    delayed entry at exit times, where its grid of days gives the exact
#    integrals.
#  * relsurv::rs.diff for the test, with and without strata and with delayed
#    entry at exit times, on its grid of exit times and whole days.
# Run from the crate directory:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/data/relsurv_extras.R
suppressMessages({
  library(jsonlite); library(survival); library(relsurv)
})
set.seed(20261126)

day0 <- as.numeric(as.Date("1980-01-01"))
age_cuts <- (0:110) * 365
year_cuts <- day0 + (0:45) * 365
na <- length(age_cuts); ny <- length(year_cuts)
haz <- array(0, dim = c(na, 2, ny))
for (a in 1:na) for (s in 1:2) for (y in 1:ny) {
  haz[a, s, y] <- exp(-10.5 + 0.09 * (a - 1) + 0.35 * (s == 1) - 0.012 * (y - 1)) / 365
}
rt <- haz
attributes(rt) <- list(
  dim = c(na, 2L, ny),
  dimnames = list(as.character(0:(na - 1)), c("male", "female"), as.character(1980 + 0:(ny - 1))),
  dimid = c("age", "sex", "year"), type = c(2L, 1L, 3L),
  cutpoints = list(age_cuts, NULL, as.Date(year_cuts, origin = "1970-01-01")),
  class = "ratetable")
stopifnot(is.ratetable(rt))

# The cumulative reference hazard of one person at times t since the origin.
H_at <- function(age0, date0, sex, t) {
  tmax <- max(t)
  ta <- age_cuts[age_cuts > age0] - age0
  ty <- year_cuts[year_cuts > date0] - date0
  b <- sort(unique(c(ta, ty))); b <- b[b < tmax]
  ends <- c(b, tmax); starts <- c(0, b); mid <- (starts + ends) / 2
  ai <- pmin(pmax(findInterval(age0 + mid, age_cuts), 1), na)
  yi <- pmin(pmax(findInterval(date0 + mid, year_cuts), 1), ny)
  h <- haz[cbind(ai, sex, yi)]
  approx(c(0, ends), c(0, cumsum((ends - starts) * h)), xout = t)$y
}

make_cohort <- function(n, entry) {
  d <- data.frame(
    age0 = floor(runif(n, 18, 92) * 365),
    date0 = floor(runif(n, day0 + 5 * 365, day0 + 20 * 365)),
    sex = sample(1:2, n, TRUE))
  d$grp <- sample(1:3, n, TRUE)
  rate <- c(1 / 2000, 1 / 2600, 1 / 3200)[d$grp] * exp(0.02 * (d$age0 / 365 - 60))
  tt <- 1 + floor(-log(runif(n)) / rate)
  cc <- 150.5 + floor(runif(n, 0, 3000))
  d$time <- pmin(tt, cc); d$status <- as.numeric(tt <= cc)
  d$start <- 0
  if (entry) {
    censor_times <- sort(unique(d$time[d$status == 0]))
    late <- which(runif(n) < 0.3)
    for (i in late) {
      cand <- censor_times[censor_times < 0.6 * d$time[i]]
      if (length(cand) > 0) d$start[i] <- cand[sample.int(length(cand), 1)]
    }
  }
  a <- d$age0 / 365
  d$agegr <- as.integer(cut(a, c(0, 45, 55, 65, 75, Inf), right = FALSE))
  d$dt <- as.Date(d$date0, origin = "1970-01-01")
  rownames(d) <- NULL
  d
}
pack <- function(d) as.list(d[, c("age0", "date0", "sex", "grp", "agegr", "time", "status", "start")])

dA <- make_cohort(600, FALSE)
dB <- make_cohort(500, TRUE)
fin <- max(dA$time + dA$date0)
dA$potential <- ifelse(dA$status == 1, fin - dA$date0, dA$time)
rmap <- quote(list(age = age0, sex = sex, year = dt))
asked <- c(0, 100.5, 365, 730.25, 1500, 2500, 3100)

# ---- 1. survexp ----
by_group <- function(fit, times) {
  s <- as.matrix(fit$surv)
  lapply(seq_len(ncol(s)), function(k) {
    idx <- match(asked, times)
    list(surv = s[idx, k], n = as.matrix(fit$n.risk)[idx, k])
  })
}
ederer <- survexp(~ grp, data = dA, ratetable = rt, rmap = list(age = age0, sex = sex, year = dt),
                  times = asked, method = "ederer")
grid_h <- sort(unique(c(asked, dA$potential)))
hakulinen <- survexp(potential ~ grp, data = dA, ratetable = rt,
                     rmap = list(age = age0, sex = sex, year = dt), times = grid_h,
                     method = "hakulinen")
grid_c <- sort(unique(c(asked, dA$time)))
conditional <- survexp(time ~ grp, data = dA, ratetable = rt,
                       rmap = list(age = age0, sex = sex, year = dt), times = grid_c,
                       method = "conditional")
survexp_ref <- list(asked = asked,
                    ederer = lapply(seq_len(3), function(k) as.matrix(ederer$surv)[, k]),
                    hakulinen = by_group(hakulinen, grid_h),
                    conditional = by_group(conditional, grid_c))

# ---- 2. Ederer I and Hakulinen relative survival ----
rs_run <- function(method) {
  f <- rs.surv(Surv(time, status) ~ grp, data = dA, ratetable = rt,
               rmap = list(age = age0, sex = sex, year = dt),
               method = method, type = "fleming-harrington")
  st <- rep(seq_along(f$strata), f$strata)
  lapply(seq_along(f$strata), function(k) {
    w <- st == k
    list(time = f$time[w], surv = f$surv[w], std_err = f$std.err[w])
  })
}
rs_e1 <- rs_run("ederer1")
rs_hak <- rs_run("hakulinen")
# The exact Hakulinen relative survival: Nelson-Aalen observed survival over
# survexp's Hakulinen curve at every exit time, and the bound on relsurv's rule.
hak_exact <- lapply(1:3, function(k) {
  dk <- dA[dA$grp == k, ]
  times <- sort(unique(dk$time))
  n <- sapply(times, function(t) sum(dk$time >= t))
  dd <- sapply(times, function(t) sum(dk$time == t & dk$status == 1))
  grid <- sort(unique(c(times, dk$potential)))
  he <- survexp(potential ~ 1, data = dk, ratetable = rt,
                rmap = list(age = age0, sex = sex, year = dt), times = grid,
                method = "hakulinen")
  se <- he$surv[match(times, grid)]
  # relsurv's steps: its tis, clipped at the last exit time.
  tis <- sort(unique(pmin(max(times), c(times, dk$potential))))
  H <- t(sapply(seq_len(nrow(dk)), function(i) H_at(dk$age0[i], dk$date0[i], dk$sex[i], c(0, tis))))
  bound <- 0
  for (j in seq_along(tis)) {
    inset <- dk$potential >= tis[j]
    if (any(inset)) bound <- bound + max((H[inset, j + 1] - H[inset, j])^2)
  }
  list(time = times, surv = exp(-cumsum(dd / n)) / se, bound = bound)
})

# ---- 3. Age-standardised net survival, Ederer II by age group, ICSS 1 ----
icss1 <- c(0.07, 0.12, 0.23, 0.29, 0.29)
e2 <- rs.surv(Surv(time, status) ~ agegr, data = dA, ratetable = rt,
              rmap = list(age = age0, sex = sex, year = dt),
              method = "ederer2", type = "fleming-harrington", precision = 0.5)
st <- rep(seq_along(e2$strata), e2$strata)
std_at <- function(t) {
  r <- 0; v <- 0
  for (k in 1:5) {
    w <- st == k; tk <- e2$time[w]
    j <- max(c(0, which(tk <= t)))
    rk <- if (j == 0) 1 else e2$surv[w][j]
    vk <- if (j == 0) 0 else e2$std.err[w][j]^2
    r <- r + icss1[k] * rk; v <- v + icss1[k]^2 * rk^2 * vk
  }
  se <- sqrt(v); lam <- -log(r); z <- qnorm(0.975); shift <- z * se / (r * lam)
  c(r, se, exp(-exp(log(lam) + shift)), exp(-exp(log(lam) - shift)))
}
std_times <- c(30.5, 365, 1000, 2000)
standardised <- list(times = std_times, weights = icss1,
                     values = lapply(std_times, std_at))

# ---- 4. Crude probabilities of death, with delayed entry ----
cm <- cmp.rel(Surv(start, time, status) ~ grp, data = dB, ratetable = rt,
              rmap = list(age = age0, sex = sex, year = dt))
crude <- lapply(1:3, function(k) {
  e <- cm[[2 * k - 1]]; p <- cm[[2 * k]]
  dk <- dB[dB$grp == k, ]
  idx <- match(sort(unique(dk$time)), e$time)
  list(time = e$time[idx], disease = e$est[idx], disease_var = e$var[idx],
       other = p$est[idx], other_var = p$var[idx])
})

# cmp.rel and rs.diff call exp_prep without the entry times (relsurv 2.3-3),
# so they treat a row with delayed entry as at risk from time 0. The values
# with delayed entry are therefore written out here from the definitions, and
# the relsurv values on cohort B are kept to show that difference: they equal
# those of cohort B with every entry at 0.
H_matrix <- function(d, points) {
  t(sapply(seq_len(nrow(d)), function(i) H_at(d$age0[i], d$date0[i], d$sex[i], c(0, points))))
}
crude_ref <- function(d) {
  points <- sort(unique(c(d$time, d$start[d$start > 0])))
  H <- H_matrix(d, points)
  s <- 1; fe <- 0; fp <- 0
  rows <- list(); keep <- list()
  for (j in seq_along(points)) {
    b <- points[j]
    at <- d$start < b & d$time >= b
    if (!any(at)) next
    n <- sum(at); dd <- sum(at & d$time == b & d$status == 1)
    pop <- mean(H[at, j + 1] - H[at, j])
    fe <- fe + s * (dd / n - pop); fp <- fp + s * pop
    s <- s * (1 - dd / n)
    rows[[length(rows) + 1]] <- c(b, fe, fp, s, dd / n^2)
  }
  m <- do.call(rbind, rows)
  ve <- vp <- numeric(nrow(m))
  for (j in seq_len(nrow(m))) {
    k <- seq_len(j)
    ve[j] <- sum(m[k, 5] * (m[k, 4] - (m[j, 2] - m[k, 2]))^2)
    vp[j] <- sum(m[k, 5] * (m[j, 3] - m[k, 3])^2)
  }
  idx <- match(sort(unique(d$time)), m[, 1])
  list(time = m[idx, 1], disease = m[idx, 2], disease_var = ve[idx],
       other = m[idx, 3], other_var = vp[idx])
}
crude_entry <- lapply(1:3, function(k) crude_ref(dB[dB$grp == k, ]))

test_ref <- function(d, strat) {
  points <- sort(unique(c(d$time, d$start[d$start > 0], seq(1, max(d$time), by = 1))))
  H <- H_matrix(d, points)
  G <- 3; z <- rep(0, G); V <- matrix(0, G, G)
  for (s in unique(strat)) {
    for (j in seq_along(points)) {
      b <- points[j]
      at <- strat == s & d$start < b & d$time >= b
      if (!any(at)) next
      w <- exp(H[, j + 1]); dh <- H[, j + 1] - H[, j]
      dead <- at & d$time == b & d$status == 1
      risk <- sapply(1:G, function(g) sum(w[at & d$grp == g]))
      ev <- sapply(1:G, function(g) sum(w[dead & d$grp == g]) - sum((w * dh)[at & d$grp == g]))
      sq <- sapply(1:G, function(g) sum(w[dead & d$grp == g]^2))
      p <- risk / sum(risk)
      z <- z + ev - p * sum(ev)
      for (k in 1:G) {
        e <- (seq_len(G) == k) - p
        V <- V + outer(e, e) * sq[k]
      }
    }
  }
  r <- G - 1
  list(statistic = as.numeric(z[1:r] %*% solve(V[1:r, 1:r]) %*% z[1:r]), df = r,
       z = z[1:r], covariance = as.numeric(V[1:r, 1:r]))
}

# ---- 5. The test of equal net survival ----
diff_run <- function(d, formula) {
  f <- rs.diff(formula, data = d, ratetable = rt, rmap = list(age = age0, sex = sex, year = dt))
  list(statistic = as.numeric(f$test.stat), p = as.numeric(f$p.value), df = f$df,
       z = as.numeric(f$zh), covariance = as.numeric(f$covmat))
}
tests <- list(
  plain = diff_run(dA, Surv(time, status) ~ grp),
  strata = diff_run(dA, Surv(time, status) ~ grp + strata(sex)),
  entry_relsurv = diff_run(dB, Surv(start, time, status) ~ grp),
  entry = test_ref(dB, rep(1, nrow(dB))),
  plain_written = test_ref(dA, rep(1, nrow(dA))))

write(toJSON(list(
  versions = list(r = R.version.string, survival = as.character(packageVersion("survival")),
                  relsurv = as.character(packageVersion("relsurv"))),
  age_cuts = age_cuts, year_cuts = year_cuts, rates = as.vector(haz),
  cohort_a = c(pack(dA), list(potential = dA$potential)), cohort_b = pack(dB),
  survexp = survexp_ref, ederer1 = rs_e1, hakulinen = rs_hak, hakulinen_exact = hak_exact,
  standardised = standardised, crude_relsurv = crude, crude = crude_entry, tests = tests),
  auto_unbox = TRUE, digits = 17, null = "null", na = "null"),
  "tests/data/relsurv_extras.json")
