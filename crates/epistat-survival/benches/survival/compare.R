# Checks that the Rust program and the R script solved the same problem.
#
# Usage: Rscript compare.R N SCRATCH
# Reads OUT/rust_METHOD_N.csv and r_METHOD_N.csv and prints one line per
# method: the largest absolute and relative difference over the numbers both
# sides wrote. For aj_boot the two sides draw their weights from different
# streams, so the means are compared in units of the Monte Carlo standard
# error and the standard deviations as a ratio.
suppressMessages(library(data.table))
args <- commandArgs(TRUE)
n <- args[1]; scratch <- args[2]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
cat("n,method,values,max_abs_diff,max_rel_diff,detail\n")
compare <- function(f, method, r_method) {
  rp <- file.path(out, sprintf("r_%s_%s.csv", r_method, n))
  if (!file.exists(rp)) return(invisible())
  a <- fread(file.path(out, f)); b <- fread(rp)
  m <- merge(a, b, by = "name", suffixes = c("_rust", "_r"))
  if (grepl("^aj_boot", method)) {
    boot <- as.integer(Sys.getenv("BOOT", "50"))
    sds <- m[grepl("^sd_", name)]
    means <- m[grepl("^mean_", name)]
    se <- sds$value_r[match(sub("^mean_", "sd_", means$name), sds$name)] / sqrt(boot)
    z <- abs(means$value_rust - means$value_r) / sqrt(2 * se^2)
    ratio <- sds$value_rust / sds$value_r
    cat(sprintf("%s,%s,%d,%.3g,NA,max mean difference %.2f MC SE; sd ratio %.2f to %.2f\n",
                n, r_method, nrow(m), max(abs(means$value_rust - means$value_r)),
                max(z, na.rm = TRUE), min(ratio, na.rm = TRUE), max(ratio, na.rm = TRUE)))
    return(invisible())
  }
  ad <- abs(m$value_rust - m$value_r)
  rd <- ad / pmax(abs(m$value_r), 1e-12)
  cat(sprintf("%s,%s,%d,%.3g,%.3g,\n", n, r_method, nrow(m), max(ad, na.rm = TRUE),
              max(rd, na.rm = TRUE)))
}
files <- list.files(out, pattern = sprintf("^rust_.*_%s\\.csv$", n))
for (f in files) {
  method <- sub(sprintf("^rust_(.*)_%s\\.csv$", n), "\\1", f)
  if (method == "times") next
  compare(f, method, method)
  if (grepl("^multistate", method)) compare(f, method, sub("^multistate", "msurvfit", method))
}
