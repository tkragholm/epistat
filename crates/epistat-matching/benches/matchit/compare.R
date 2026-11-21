# Checks that the Rust program and the R script solved the same problem.
#
# Usage: Rscript compare.R N SCRATCH
# Reads SCRATCH/out/rust_METHOD_N.csv and r_METHOD_N.csv and prints one line
# per method: the number of matched units on each side, and how far the
# matches agree.
#   pairs      share of anchors whose set of controls is the same, and the
#              number of pairs in common over the larger number of pairs
#   partition  whether both sides kept the same units and grouped them the
#              same way (subclass numbers are renamed by the lowest unit id)
#   optimal    also the total distance of the pairs on the propensity score
#              fitted here with glm, since two optimal matchings can differ in
#              pairs and agree in total
#   balance    largest difference of each statistic over the covariates
suppressMessages(library(data.table))
args <- commandArgs(TRUE)
n <- args[1]
scratch <- args[2]
out <- file.path(scratch, "out")
rd <- function(impl, method) {
  p <- file.path(out, sprintf("%s_%s_%s.csv", impl, method, n))
  if (file.exists(p)) fread(p) else NULL
}
cat("n,method,rust_units,r_units,agreement,detail\n")
line <- function(method, ru, ru2, agree, detail = "") {
  cat(sprintf("%s,%s,%s,%s,%s,%s\n", n, method, ru, ru2, agree, detail))
}

pair_methods <- c("nn_ps", "nn_ps_cal", "nn_strata", "nn_cal_strata", "nn_1to3",
                  "nn_mahal", "optimal_cal")
part_methods <- c("subclass", "exact", "cem", "quick", "full_cal")

d <- NULL
ps_of <- function() {
  if (is.null(d)) {
    d <<- as.data.frame(fread(file.path(scratch, sprintf("cohort_%s.csv", n))))
    fit <- glm(treat ~ x1 + x2 + x3 + x4 + x5, data = d, family = binomial())
    d$ps <<- as.numeric(fitted(fit))
  }
  d$ps
}

for (m in pair_methods) {
  a <- rd("rust", m); b <- rd("r", m)
  if (is.null(a) || is.null(b)) next
  setnames(a, c("anchor", "control")); setnames(b, c("anchor", "control"))
  common <- nrow(merge(a, b, by = c("anchor", "control")))
  ga <- a[, .(k = paste(sort(control), collapse = " ")), by = anchor]
  gb <- b[, .(k = paste(sort(control), collapse = " ")), by = anchor]
  gm <- merge(ga, gb, by = "anchor", all = TRUE)
  same_anchor <- mean(!is.na(gm$k.x) & !is.na(gm$k.y) & gm$k.x == gm$k.y)
  detail <- sprintf("pairs_in_common=%.6f", common / max(nrow(a), nrow(b)))
  if (m == "optimal_cal") {
    ps <- ps_of()
    tot <- function(p) sum(abs(ps[p$anchor] - ps[p$control]))
    detail <- sprintf("%s total_rust=%.6f total_r=%.6f", detail, tot(a), tot(b))
  }
  line(m, nrow(a), nrow(b), sprintf("anchors_same=%.6f", same_anchor), detail)
}

canon <- function(x) {
  x <- copy(x); setnames(x, c("id", "sub"))
  x[, lab := min(id), by = sub]
  setorder(x, id)
  x[, .(id, lab)]
}
for (m in part_methods) {
  a <- rd("rust", m); b <- rd("r", m)
  if (is.null(a) || is.null(b)) next
  ca <- canon(a); cb <- canon(b)
  same_units <- identical(ca$id, cb$id)
  same_part <- same_units && identical(ca$lab, cb$lab)
  share <- if (same_units) mean(ca$lab == cb$lab) else NA
  detail <- sprintf("units_in_same_subclass=%s", format(share))
  if (m == "full_cal") {
    # The cost of a full matching is the distance of every anchor-candidate
    # pair inside a subclass (each subclass is a star).
    ps <- ps_of()
    cost <- function(x) {
      setnames(x, c("id", "sub"))
      x[, treat := d$treat[id]]
      x[, ps := ps[id]]
      x[, .(c = sum(abs(outer(ps[treat == 1], ps[treat == 0], "-")))), by = sub][, sum(c)]
    }
    detail <- sprintf("%s total_rust=%.6f total_r=%.6f", detail, cost(copy(a)), cost(copy(b)))
  }
  line(m, nrow(a), nrow(b), sprintf("same_units=%s same_partition=%s", same_units, same_part),
       detail)
}

a <- rd("rust", "balance"); b <- rd("r", "balance")
bp <- file.path(out, sprintf("r_balance_pre_%s.csv", n))
if (!is.null(a) && !is.null(b)) {
  setnames(a, c("name", "smd_pre", "smd_post", "var_ratio_post", "ecdf_mean_post"))
  mg <- merge(a, b, by = "name", suffixes = c(".rust", ".r"))
  pre <- merge(a[, .(name, smd_pre)], fread(bp), by = "name", suffixes = c(".rust", ".r"))
  line("balance", nrow(a), nrow(b), "see detail",
       sprintf("max_abs_diff smd_pre=%.2e smd_post=%.2e var_ratio=%.2e ecdf_mean=%.2e",
               max(abs(pre$smd_pre.rust - pre$smd_pre.r), na.rm = TRUE),
               max(abs(mg$smd_post.rust - mg$smd_post.r), na.rm = TRUE),
               max(abs(mg$var_ratio_post.rust - mg$var_ratio_post.r), na.rm = TRUE),
               max(abs(mg$ecdf_mean_post.rust - mg$ecdf_mean_post.r), na.rm = TRUE)))
}
