# The R side of the tier 1 benchmarks: one process runs one method on one size
# and writes the numbers the Rust side writes, so that compare.R can check that
# both solved the same problem. See bench/run.sh for the driver and common.rs for the
# Rust side and the data files.
#
# Usage: Rscript bench.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (timed repetitions,
# default 3). The method gen_pool simulates the imputed fits that `pool`
# reads; the other data files come from the Rust programs.
suppressMessages(library(data.table))
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
# Integer columns are read as doubles, as the Rust side reads them: survival's
# pyears treats integer dates and ages differently from numeric ones.
rd <- function(kind) {
  d <- fread(file.path(scratch, sprintf("%s_%d.csv", kind, n)))
  for (j in names(d)) if (is.integer(d[[j]])) set(d, j = j, value = as.numeric(d[[j]]))
  d
}
setDTthreads(0)

timed <- function(f, reps = get("reps", envir = globalenv())) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    value <- f()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  list(times = times, value = value)
}
finish <- function(times, values) {
  d <- data.table(name = names(values), value = sprintf("%.17g", as.numeric(values)))
  fwrite(d, file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
  path <- file.path(out, "r_times.csv")
  row <- data.table(n = n, method = method, reps = length(times),
                    median_s = median(times), min_s = min(times))
  fwrite(row, path, append = file.exists(path))
}
v <- function(...) c(...)
indexed <- function(prefix, x) setNames(as.numeric(x), paste0(prefix, seq_along(x) - 1))
at <- function(i, x) setNames(x[i + 1], paste0("at", format(i, scientific = FALSE)))

# 5 knots of Harrell, linear column included, as the Rust side
basis_summary <- function(b, knots) {
  b <- unclass(b)
  vals <- indexed("knot", knots)
  for (j in seq_len(ncol(b))) {
    vals[sprintf("colsum%d", j - 1)] <- sum(b[, j])
    for (i in 0:2) vals[sprintf("row%d_col%d", i, j - 1)] <- b[i + 1, j]
  }
  vals
}

pooled <- function(r, exp_scale, name) {
  f <- if (exp_scale) exp else identity
  setNames(c(f(r$beta), r$se, f(r$ci.lb), f(r$ci.ub)),
           paste0(name, c("_estimate", "_se", "_lower", "_upper")))
}

coefs <- function(names, coef, se) {
  c(setNames(as.numeric(coef), paste0("coef_", names)), setNames(as.numeric(se), paste0("se_", names)))
}

if (method == "gen_pool") {
  suppressMessages({library(mice); library(mitml)})
  imp <- mice(nhanes2, m = 50, printFlag = FALSE, seed = 12)
  fits1 <- lapply(1:50, function(i) lm(chl ~ bmi + age + hyp, data = complete(imp, i)))
  fits0 <- lapply(1:50, function(i) lm(chl ~ bmi, data = complete(imp, i)))
  saveRDS(list(fits1 = fits1, fits0 = fits0), file.path(scratch, "pool_fits.rds"))
  tested <- setdiff(names(coef(fits1[[1]])), names(coef(fits0[[1]])))
  rows <- lapply(fits1, function(f) {
    q <- coef(f)[tested]; u <- vcov(f)[tested, tested]
    c(setNames(q, paste0("q", seq_along(q) - 1)),
      setNames(as.vector(t(u)), paste0("u", seq_len(length(u)) - 1)),
      bq = unname(coef(f)["bmi"]), bu = unname(vcov(f)["bmi", "bmi"]),
      wald = drop(t(q) %*% solve(u) %*% q), dfcom = f$df.residual)
  })
  fwrite(as.data.table(do.call(rbind, rows)), file.path(scratch, "pool_50.csv"))
  quit(save = "no")
}

if (method %in% c("rcs", "ns")) {
  x <- rd("spl")$x
  if (method == "rcs") {
    # Hmisc::rcspline.eval(x, nk = 5, inclx = TRUE) with its default knots and
    # norm = 2, written out here because Hmisc did not install on the Linux
    # machine (no Fortran compiler). The knots are the quantiles of Harrell.
    rcs_basis <- function(x) {
      knots <- unname(quantile(x, c(0.05, 0.275, 0.5, 0.725, 0.95), type = 7))
      k <- length(knots)
      kd <- (knots[k] - knots[1])^2
      cube <- function(z) pmax(z, 0)^3
      b <- matrix(0, length(x), k - 1)
      b[, 1] <- x
      for (j in 1:(k - 2)) {
        b[, j + 1] <- (cube(x - knots[j]) -
                         cube(x - knots[k - 1]) * (knots[k] - knots[j]) / (knots[k] - knots[k - 1]) +
                         cube(x - knots[k]) * (knots[k - 1] - knots[j]) / (knots[k] - knots[k - 1])) / kd
      }
      attr(b, "knots") <- knots
      b
    }
    r <- timed(function() rcs_basis(x))
    finish(r$times, basis_summary(r$value, attr(r$value, "knots")))
  } else {
    suppressMessages(library(splines))
    r <- timed(function() ns(x, df = 5))
    finish(r$times, basis_summary(r$value, attr(r$value, "knots")))
  }
} else if (method == "negbin") {
  suppressMessages(library(MASS))
  d <- rd("reg")
  r <- timed(function() glm.nb(y ~ x1 + x2 + x3 + offset(off), data = d))
  f <- r$value
  finish(r$times, c(coefs(c("int", "x1", "x2", "x3"), coef(f), sqrt(diag(vcov(f)))),
                    theta = f$theta, se_theta = f$SE.theta))
} else if (method %in% c("risk_robust", "risk_cluster")) {
  suppressMessages(library(sandwich))
  d <- rd("reg")
  r <- timed(function() {
    f <- glm(b ~ x1 + x2 + x3, data = d, family = poisson(link = "log"))
    vc <- if (method == "risk_cluster") vcovCL(f, cluster = d$fam, type = "HC1", cadjust = TRUE)
          else vcovHC(f, type = "HC1")
    list(coef = coef(f), vc = vc)
  })
  finish(r$times, coefs(c("int", "x1", "x2", "x3"), r$value$coef, sqrt(diag(r$value$vc))))
} else if (method == "within") {
  suppressMessages(library(fixest))
  d <- rd("reg")
  r <- timed(function() feols(c ~ x1 + x2 + x3 | fam, data = d, cluster = ~fam, notes = FALSE))
  finish(r$times, coefs(c("x1", "x2", "x3"), coef(r$value), se(r$value)))
} else if (method == "mh_or") {
  suppressMessages(library(metafor))
  d <- rd("mh")
  r <- timed(function() {
    s <- d[, .(a = sum(exposed == 1 & case == 1), b = sum(exposed == 1 & case == 0),
               c = sum(exposed == 0 & case == 1), d = sum(exposed == 0 & case == 0)), by = stratum]
    mh <- function(m) suppressWarnings(rma.mh(ai = s$a, bi = s$b, ci = s$c, di = s$d, measure = m,
                                             add = 0, to = "none", drop00 = FALSE, correct = FALSE))
    list(or = mh("OR"), rr = mh("RR"), rd = mh("RD"))
  })
  finish(r$times, c(pooled(r$value$or, TRUE, "or"), pooled(r$value$rr, TRUE, "rr"),
                    pooled(r$value$rd, FALSE, "rd")))
} else if (method == "mh_rr") {
  suppressMessages(library(metafor))
  d <- rd("mhr")
  r <- timed(function() {
    mh <- function(m) suppressWarnings(rma.mh(x1i = d$x1, x2i = d$x0, t1i = d$t1, t2i = d$t0, measure = m,
                                             add = 0, to = "none", drop00 = FALSE, correct = FALSE))
    list(irr = mh("IRR"), ird = mh("IRD"))
  })
  finish(r$times, c(pooled(r$value$irr, TRUE, "irr"), pooled(r$value$ird, FALSE, "ird")))
} else if (startsWith(method, "padj_")) {
  p <- rd("pv")$p
  rule <- c(holm = "holm", hochberg = "hochberg", hommel = "hommel", bh = "BH", by = "BY")[[sub("padj_", "", method)]]
  r <- timed(function() p.adjust(p, method = rule))
  q <- r$value
  finish(r$times, c(sum = sum(q), min = min(q), at(0, q), at(1, q), at(2, q),
                    at(n %/% 2, q), at(n - 1, q)))
} else if (method == "pool") {
  suppressMessages({library(mice); library(mitml)})
  fits <- readRDS(file.path(scratch, "pool_fits.rds"))
  fits1 <- fits$fits1; fits0 <- fits$fits0
  dfcom <- fits1[[1]]$df.residual
  tested <- setdiff(names(coef(fits1[[1]])), names(coef(fits0[[1]])))
  r <- timed(function() {
    p <- summary(pool(as.mira(fits1)), "all", conf.int = FALSE)
    d1 <- testModels(fits1, fits0, method = "D1", df.com = dfcom)$test
    d2 <- testModels(fits1, fits0, method = "D2", use = "wald")$test
    list(p = p, d1 = d1, d2 = d2)
  }, reps = 20)
  p <- r$value$p
  b <- p[p$term == "bmi", ]
  vec <- p$estimate[match(tested, p$term)]
  finish(r$times, c(scalar_estimate = b$estimate, scalar_total = b$t, scalar_df = b$df,
                    indexed("vector_estimate", vec),
                    d1_statistic = unname(r$value$d1[1, "F.value"]), d1_df2 = unname(r$value$d1[1, "df2"]),
                    d1_p = unname(r$value$d1[1, "P(>F)"]),
                    d2_statistic = unname(r$value$d2[1, "F.value"]), d2_df2 = unname(r$value$d2[1, "df2"])))
} else if (method == "events") {
  rows <- rd("evt"); persons <- rd("evtp"); persons[, id := .I - 1L]
  rows[persons, `:=`(entry = i.entry, exit = i.exit), on = "id"]
  r <- timed(function() {
    first_event <- function(bit) {
      a <- rows[bitwAnd(mask, bit) > 0L]
      w <- a[day >= entry & day <= exit]
      list(a = a, w = w, first = w[, .(first = min(day)), by = id])
    }
    ra <- first_event(1L)
    prior <- ra$a[day >= entry - 365L & day < entry, uniqueN(id)]
    rb <- first_event(2L)
    u <- unique(rb$w[, .(id, day)])
    u[, nx := shift(day, -1L), by = id]
    case_b <- u[nx - day <= 30L, .(case = min(day)), by = id]
    list(ra = ra, rb = rb, prior = prior, case_b = case_b)
  })
  x <- r$value
  finish(r$times, c(
    rule0_nfirst = nrow(x$ra$first), rule0_sumfirst = sum(as.numeric(x$ra$first$first)),
    rule0_ncase = nrow(x$ra$first), rule0_sumcase = sum(as.numeric(x$ra$first$first)),
    rule0_nevents = nrow(x$ra$w), rule0_nprevalent = x$prior,
    rule1_nfirst = nrow(x$rb$first), rule1_sumfirst = sum(as.numeric(x$rb$first$first)),
    rule1_ncase = nrow(x$case_b), rule1_sumcase = sum(as.numeric(x$case_b$case)),
    rule1_nevents = nrow(x$rb$w)))
} else if (method == "comorbidity") {
  suppressMessages(library(comorbidity))
  x <- rd("cmb")
  r <- timed(function() {
    cm <- comorbidity(x, id = "id", code = "code", map = "charlson_icd10_quan", assign0 = TRUE,
                      labelled = FALSE, tidy.codes = FALSE)
    cm[, "score"] <- score(cm, weights = "quan", assign0 = TRUE)
    cm
  })
  cm <- as.data.table(r$value); cm <- cm[order(id)]
  s <- cm$score
  finish(r$times, c(persons = nrow(cm), score_sum = sum(s), score_square_sum = sum(s * s),
                    n_nonzero = sum(s != 0), setNames(s[1:8], paste0("score", 0:7))))
} else if (method == "exposure") {
  suppressMessages(library(AdhereR))
  d <- rd("exp")
  d[, date := as.character(as.Date(day, origin = "1970-01-01"))]
  d[, dur := quantity / 2][, med := "A"]
  d <- as.data.frame(d)
  r <- timed(function() {
    compute.treatment.episodes(
      d, ID.colname = "id", event.date.colname = "date", event.duration.colname = "dur",
      medication.class.colname = "med", carryover.within.obs.window = TRUE,
      carry.only.for.same.medication = FALSE, consider.dosage.change = FALSE,
      medication.change.means.new.treatment.episode = FALSE,
      dosage.change.means.new.treatment.episode = FALSE,
      maximum.permissible.gap = 30, maximum.permissible.gap.unit = "days",
      maximum.permissible.gap.append.to.episode = FALSE,
      followup.window.start = 0, followup.window.start.unit = "days",
      followup.window.duration = 40000, followup.window.duration.unit = "days",
      date.format = "%Y-%m-%d", suppress.warnings = TRUE)
  })
  e <- r$value; epoch <- as.Date("1970-01-01")
  finish(r$times, c(episodes = nrow(e), sum_start = sum(as.numeric(e$episode.start - epoch)),
                    sum_end = sum(as.numeric(e$episode.end - epoch))))
} else if (method %in% c("pyears_rates", "sir_ratetable")) {
  suppressMessages(library(survival))
  d <- as.data.frame(rd("pyr"))
  if (method == "pyears_rates") {
    # Cut points rounded half up, as f64::round does on the Rust side.
    age_cuts <- floor((0:18) * 5 * 365.25 + 0.5)
    year_cuts <- as.numeric(as.Date(sprintf("%d-01-01", 2000:2023)))
    w <- 1:18
    r <- timed(function() {
      p <- pyears(Surv(dur, ev) ~ tcut(age0, age_cuts, labels = 1:18) +
                    tcut(date0, year_cuts, labels = 1:23) + factor(sex), data = d, scale = 1)
      by_sex <- lapply(1:2, function(s) {
        x <- apply(p$event[, , s], 1, sum); y <- apply(p$pyears[, , s], 1, sum)
        wn <- w / sum(w)
        list(events = sum(x), time = sum(y), rate = sum(wn * x / y), se = sqrt(sum(wn^2 * x / y^2)))
      })
      list(by_sex = by_sex)
    })
    s <- r$value$by_sex
    vals <- c()
    for (k in 1:2) vals <- c(vals, setNames(c(s[[k]]$events, s[[k]]$time, s[[k]]$rate, s[[k]]$se),
      sprintf(c("events_sex%d", "person_time_sex%d", "std_rate_sex%d", "std_se_sex%d"), k - 1)))
    vals["rate_ratio"] <- (s[[2]]$events / s[[2]]$time) / (s[[1]]$events / s[[1]]$time)
    finish(r$times, vals)
  } else {
    age_cuts <- floor((0:120) * 365.25 + 0.5)
    year_cuts <- as.numeric(as.Date(sprintf("%d-01-01", 1990:2030)))
    rates <- array(0, c(120, 2, 40))
    for (a in 1:120) for (s in 1:2) for (y in 1:40)
      rates[a, s, y] <- 1e-6 * exp(0.09 * (a - 1)) * (1 + 0.3 * (s - 1)) * (1 - 0.01 * (y - 1))
    rt <- structure(rates, dimnames = list(age = as.character(age_cuts[1:120]), sex = c("male", "female"),
                                           year = as.character(year_cuts[1:40])),
                    dimid = c("age", "sex", "year"),
                    cutpoints = list(age_cuts[1:120], NULL, as.Date(year_cuts[1:40], origin = "1970-01-01")),
                    type = c(2L, 1L, 3L), class = "ratetable")
    d$sexc <- d$sex + 1L
    observed <- sum(d$ev)
    r <- timed(function() {
      p <- pyears(Surv(dur, ev) ~ 1, data = d, rmap = list(age = age0, sex = sexc, year = date0),
                  ratetable = rt, scale = 1, expect = "event")
      list(p = p, ci = poisson.test(observed, p$expected)$conf.int)
    })
    e <- r$value$p$expected
    finish(r$times, c(observed = observed, expected = e, offtable = r$value$p$offtable,
                      sir = observed / e, lower = r$value$ci[1], upper = r$value$ci[2]))
  }
} else if (method == "overlap") {
  suppressMessages(library(WeightIt))
  d <- as.data.frame(rd("ps"))
  fit_w <- function() tryCatch(weightit(a ~ x1 + x2 + x3 + x4, data = d, estimand = "ATO", method = "glm"),
                               error = function(e) weightit(a ~ x1 + x2 + x3 + x4, data = d, estimand = "ATO", method = "ps"))
  r <- timed(fit_w)
  w <- r$value$weights
  cf <- coef(glm(a ~ x1 + x2 + x3 + x4, data = d, family = binomial))
  finish(r$times, c(indexed("coef", cf),
                    sum_w_treated = sum(w[d$a == 1]), sum_w_control = sum(w[d$a == 0]),
                    sum_wx1_treated = sum((w * d$x1)[d$a == 1]), sum_wx1_control = sum((w * d$x1)[d$a == 0]),
                    mean_propensity = mean(r$value$ps)))
} else if (method == "ipwtm") {
  suppressMessages(library(ipw))
  d <- as.data.frame(rd("ppw"))
  r <- timed(function() ipwtm(exposure = a, family = "binomial", link = "logit", numerator = ~x1,
                              denominator = ~l + x1 + aprev, id = id, timevar = k, type = "all", data = d))
  w <- r$value$ipw.weights; rows <- length(w)
  finish(r$times, c(indexed("coef_den", coef(r$value$den.mod)), indexed("coef_num", coef(r$value$num.mod)),
                    sum_w = sum(w), sum_w_square = sum(w^2), max_w = max(w),
                    setNames(w[c(1, 8, 9, rows %/% 2 + 1, rows)],
                             paste0("w", format(c(0, 7, 8, rows %/% 2, rows - 1), scientific = FALSE, trim = TRUE)))))
} else if (method == "aipw") {
  d <- rd("aipw")
  r <- timed(function() {
    f1 <- glm(y ~ x1 + x2, data = d[a == 1]); f0 <- glm(y ~ x1 + x2, data = d[a == 0])
    e <- fitted(glm(a ~ x1 + x2, data = d, family = binomial))
    m1 <- predict(f1, d); m0 <- predict(f0, d)
    u1 <- m1 + d$a * (d$y - m1) / e; u0 <- m0 + (1 - d$a) * (d$y - m0) / (1 - e)
    list(u1 = u1, u0 = u0)
  })
  u1 <- r$value$u1; u0 <- r$value$u0; psi <- u1 - u0
  finish(r$times, c(mean1 = mean(u1), mean0 = mean(u0), difference = mean(psi),
                    info_se_mean1 = sqrt(var(u1) / n), info_se_mean0 = sqrt(var(u0) / n),
                    info_se_difference = sqrt(var(psi) / n)))
} else if (method == "att_gt") {
  suppressMessages(library(did))
  d <- as.data.frame(rd("did"))
  r <- timed(function() att_gt(yname = "y", tname = "time", idname = "id", gname = "g", xformla = ~x1,
                               data = d, panel = TRUE, control_group = "nevertreated",
                               base_period = "varying", est_method = "dr", bstrap = FALSE,
                               cband = FALSE, print_details = FALSE))
  o <- r$value
  finish(r$times, c(setNames(o$att, sprintf("att_g%d_t%d", o$group, o$t)),
                    setNames(o$se, sprintf("se_g%d_t%d", o$group, o$t))))
} else if (method == "two_part") {
  d <- rd("sc")
  r <- timed(function() {
    f1 <- glm(I(y > 0) ~ x1 + x2, data = d, family = binomial)
    f2 <- glm(y ~ x1 + x2, data = d[y > 0], family = Gamma(link = "log"))
    list(f1 = f1, f2 = f2, disp = summary(f2)$dispersion)
  })
  x <- r$value
  nm <- c("int", "x1", "x2")
  finish(r$times, c(setNames(coef(x$f1), paste0("coef_logit_", nm)), setNames(sqrt(diag(vcov(x$f1))), paste0("se_logit_", nm)),
                    setNames(coef(x$f2), paste0("coef_gamma_", nm)),
                    setNames(sqrt(diag(vcov(x$f2)) / x$disp), paste0("se_gamma_", nm))))
} else stop("unknown method ", method)
