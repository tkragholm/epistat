# Checks that the Rust program and the R script solved the same problem.
#
# Usage: Rscript compare.R N SCRATCH
# Reads OUT/rust_METHOD_N.csv and r_METHOD_N.csv and prints one line per
# method: the largest absolute difference over the numbers both sides wrote,
# the largest relative difference (against the R value, at least 1e-12 in
# size) and the name where it occurs. Numbers whose name starts with `info_`
# are not the same quantity on the two sides, and are reported apart.
suppressMessages(library(data.table))
args <- commandArgs(TRUE)
n <- args[1]; scratch <- args[2]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
cat("n,method,values,max_abs_diff,max_rel_diff,worst_name,info_max_rel_diff,missing_in_r,missing_in_rust\n")
files <- list.files(out, pattern = sprintf("^rust_.*_%s\\.csv$", n))
for (f in files) {
  method <- sub(sprintf("^rust_(.*)_%s\\.csv$", n), "\\1", f)
  if (method == "times") next
  rp <- file.path(out, sprintf("r_%s_%s.csv", method, n))
  if (!file.exists(rp)) next
  a <- fread(file.path(out, f)); b <- fread(rp)
  m <- merge(a, b, by = "name", suffixes = c("_rust", "_r"))
  info <- grepl("^info_", m$name)
  main <- m[!info]
  ad <- abs(main$value_rust - main$value_r)
  rd <- ad / pmax(abs(main$value_r), 1e-12)
  rd[is.na(rd)] <- 0
  k <- if (length(rd)) which.max(rd) else NA_integer_
  inf <- m[info]
  info_rel <- if (nrow(inf)) max(abs(inf$value_rust - inf$value_r) / abs(inf$value_r)) else NA
  cat(sprintf("%s,%s,%d,%.3g,%.3g,%s,%.3g,%d,%d\n", n, method, nrow(main),
              if (length(ad)) max(ad, na.rm = TRUE) else NA, if (length(rd)) max(rd) else NA,
              if (is.na(k)) "" else main$name[k], info_rel,
              sum(!(a$name %in% b$name)), sum(!(b$name %in% a$name))))
}
