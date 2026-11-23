# Reference flags and scores from the R package comorbidity, for
# tests/comorbidity_r.rs. The package has no dates, so this checks the code
# lists, the hierarchy and the weights; windows are checked in events_r.rs.
# Run from the crate directory:
#   Rscript tests/data/comorbidity_reference.R > tests/data/comorbidity_reference.json
suppressMessages({library(comorbidity); library(jsonlite)})
set.seed(20261122)
maps <- comorbidity:::.maps
names_of <- c(charlson = "charlson_icd10_quan", elixhauser = "elixhauser_icd10_quan")

# Codes: every letter + 2 digits and letter + 3 digits, and around each prefix
# of the maps the prefix, shorter and longer forms and the next code.
L <- LETTERS
d2 <- sprintf("%02d", 0:99)
d3 <- sprintf("%03d", 0:999)
alphabet <- c(as.vector(outer(L, d2, paste0)), as.vector(outer(L, d3, paste0)))
prefixes <- unique(unlist(lapply(maps[names_of], unlist)))
around <- unlist(lapply(prefixes, function(p) {
  n <- nchar(p)
  c(p, paste0(p, c("0", "9", "A", "Z")), paste0(p, "A9"), substr(p, 1, n - 1),
    paste0(substr(p, 1, n - 1), c("0", "9", "A")))
}))
codes <- sort(unique(c(alphabet, around[nchar(around) >= 1])))

flag_masks <- function(map, assign0) {
  x <- data.frame(id = seq_along(codes), code = codes)
  r <- comorbidity(x, id = "id", code = "code", map = map, assign0 = assign0,
                   labelled = FALSE, tidy.codes = FALSE)
  m <- as.matrix(r[, names(maps[[map]])])
  as.numeric(m %*% (2^(seq_len(ncol(m)) - 1)))
}
nonzero <- function(mask) {
  k <- which(mask != 0)
  list(code = codes[k], mask = mask[k])
}
single <- list()
for (nm in names(names_of)) {
  single[[nm]] <- nonzero(flag_masks(names_of[[nm]], FALSE))
}

# Persons with 1 to 30 codes drawn from the codes that match something, with a
# few that match nothing.
interesting <- sort(unique(c(single$charlson$code, single$elixhauser$code)))
n_persons <- 300
sizes <- sample(1:30, n_persons, replace = TRUE)
person_codes <- lapply(sizes, function(s) {
  c(sample(interesting, s, replace = TRUE), if (runif(1) < 0.3) sample(alphabet, 1))
})
long <- data.frame(id = rep(seq_len(n_persons), lengths(person_codes)),
                   code = unlist(person_codes))
person_ref <- function(map, assign0, weights) {
  r <- comorbidity(long, id = "id", code = "code", map = map, assign0 = assign0,
                   labelled = FALSE, tidy.codes = FALSE)
  m <- as.matrix(r[, names(maps[[map]])])
  out <- list(mask = as.numeric(m %*% (2^(seq_len(ncol(m)) - 1))))
  for (w in weights) out[[w]] <- unname(score(r, weights = w, assign0 = assign0))
  out
}
persons <- list(
  codes = lapply(person_codes, I),
  charlson_raw = person_ref("charlson_icd10_quan", FALSE, c("charlson", "quan")),
  charlson = person_ref("charlson_icd10_quan", TRUE, c("charlson", "quan")),
  elixhauser_raw = person_ref("elixhauser_icd10_quan", FALSE, c("vw")),
  elixhauser = person_ref("elixhauser_icd10_quan", TRUE, c("vw"))
)

raw <- c("e10.2", "E 10 2", "i21-0", "J45.0", "k25.", " c50 9 ", "dj45", "Z.95.8")
tidy <- list(raw = raw, tidy = comorbidity:::.tidy(data.frame(code = raw), "code")$code)

# Codes around the prefixes that match nothing in either map, a sample.
pool <- setdiff(unique(around), interesting)
zero_around <- sort(sample(pool, min(1500, length(pool))))
for (nm in names(names_of)) {
  x <- data.frame(id = seq_along(zero_around), code = zero_around)
  r <- comorbidity(x, id = "id", code = "code", map = names_of[[nm]], assign0 = FALSE,
                   labelled = FALSE, tidy.codes = FALSE)
  m <- as.matrix(r[, names(maps[[names_of[[nm]]]])])
  stopifnot(all(m == 0))
}

cat(toJSON(list(
  r_version = R.version.string,
  package_version = as.character(packageVersion("comorbidity")),
  zero_around = zero_around,
  single = single, persons = persons, tidy = tidy
), digits = NA, auto_unbox = TRUE))
