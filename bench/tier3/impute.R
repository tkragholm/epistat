# The R side of the imputation benchmark: mice::mice on the data file written
# by crates/epistat-impute/benches/tier3_impute.rs, one process per method and
# size, writing the numbers the Rust side writes. See bench/run.sh.
#
# Usage: Rscript impute.R N METHOD SCRATCH
# Environment: OUT (results, default SCRATCH/out) and REPS (default 3).
suppressMessages({library(data.table); library(mice)})
args <- commandArgs(TRUE)
n <- as.integer(args[1]); method <- args[2]; scratch <- args[3]
out <- Sys.getenv("OUT", file.path(scratch, "out"))
reps <- as.integer(Sys.getenv("REPS", "3"))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
d <- fread(file.path(scratch, sprintf("imp_%d.csv", n)))
target <- switch(method, pmm = "inc", norm = "inc", logreg = "flag", polyreg = "cat")
dd <- data.frame(x1 = d$x1, x2 = d$x2, x3 = d$x3, y = d[[target]])
if (method == "logreg") dd$y <- factor(dd$y)
if (method == "polyreg") dd$y <- factor(dd$y)
meth <- c(x1 = "", x2 = "", x3 = "", y = method)
times <- numeric(reps)
for (i in seq_len(reps)) {
  t0 <- proc.time()[["elapsed"]]
  imp <- mice(dd, m = 1, maxit = 5, method = meth, printFlag = FALSE, seed = 20261209 + i)
  times[i] <- proc.time()[["elapsed"]] - t0
}
cells <- imp$imp$y[, 1]
if (method == "polyreg") {
  values <- c(imputed = length(cells), share1 = mean(cells == "1"), share2 = mean(cells == "2"))
} else {
  v <- as.numeric(as.character(cells))
  values <- c(imputed = length(v), mean = mean(v), variance = var(v))
}
v <- data.table(name = names(values), value = sprintf("%.17g", as.numeric(values)))
fwrite(v, file.path(out, sprintf("r_%s_%d.csv", method, n)), quote = FALSE)
path <- file.path(out, "r_times.csv")
fwrite(data.table(n = n, method = method, reps = reps, median_s = median(times), min_s = min(times)),
       path, append = file.exists(path))
