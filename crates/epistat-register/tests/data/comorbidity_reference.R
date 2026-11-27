# Flags and scores of the R package comorbidity for tests/comorbidity_r.rs.
# The package is run as a black box on generated codes: its maps are never
# read. The codes are every WHO ICD-10 category (a letter and two digits),
# each with one subdivision digit, and around every code that the
# specifications in src/comorbidity/quan.rs name, the code with a fifth
# character and the code one character shorter.
# Run from the crate directory:
#   Rscript tests/data/comorbidity_reference.R > tests/data/comorbidity_reference.json
suppressMessages({library(comorbidity); library(jsonlite)})
set.seed(20261127)
maps <- c(charlson = "charlson_icd10_quan", elixhauser = "elixhauser_icd10_quan")

categories <- as.vector(outer(LETTERS, sprintf("%02d", 0:99), paste0))
subdivisions <- as.vector(outer(categories, 0:9, paste0))

# The ends of every item of the specifications, as the crate reads them: dots
# and a trailing `.x` dropped, ranges split at the hyphen or en dash.
source_lines <- readLines("src/comorbidity/quan.rs", encoding = "UTF-8")
specs <- sub('.*icd10: "([^"]*)".*', "\\1", grep("icd10: \"", source_lines, value = TRUE))
items <- unlist(strsplit(specs, ",\\s*"))
ends <- unlist(strsplit(items, "[-–]"))
ends <- gsub(".", "", sub("\\.x$", "", ends), fixed = TRUE)
named <- unique(ends)
around <- unique(c(named, paste0(named, "0"), paste0(named, "9"), paste0(named, "A"),
                   substr(named, 1, nchar(named) - 1)))
codes <- sort(unique(c(categories, subdivisions, around[nchar(around) >= 1])))

flags <- function(codes, map, assign0) {
  x <- data.frame(id = seq_along(codes), code = codes)
  r <- comorbidity(x, id = "id", code = "code", map = map, assign0 = assign0,
                   labelled = FALSE, tidy.codes = FALSE)
  as.data.frame(r)[order(r$id), setdiff(names(r), "id")]
}
masks <- function(m) as.numeric(as.matrix(m) %*% (2^(seq_len(ncol(m)) - 1)))

single <- list()
for (nm in names(maps)) {
  m <- flags(codes, maps[[nm]], FALSE)
  mask <- masks(m)
  k <- which(mask != 0)
  single[[nm]] <- list(conditions = names(m), code = codes[k], mask = mask[k])
}

# Persons with 1 to 30 codes drawn from the codes that match something, with a
# few that match nothing.
interesting <- sort(unique(c(single$charlson$code, single$elixhauser$code)))
n_persons <- 300
sizes <- sample(1:30, n_persons, replace = TRUE)
person_codes <- lapply(sizes, function(s) {
  c(sample(interesting, s, replace = TRUE), if (runif(1) < 0.3) sample(categories, 1))
})
long <- data.frame(id = rep(seq_len(n_persons), lengths(person_codes)),
                   code = unlist(person_codes))
person_ref <- function(map, assign0, weights) {
  r <- comorbidity(long, id = "id", code = "code", map = map, assign0 = assign0,
                   labelled = FALSE, tidy.codes = FALSE)
  o <- order(r$id)
  out <- list(mask = masks(as.data.frame(r)[o, setdiff(names(r), "id")]))
  for (w in weights) out[[w]] <- unname(score(r, weights = w, assign0 = assign0))[o]
  out
}
persons <- list(
  codes = lapply(person_codes, I),
  charlson_raw = person_ref(maps[["charlson"]], FALSE, c("charlson", "quan")),
  charlson = person_ref(maps[["charlson"]], TRUE, c("charlson", "quan")),
  elixhauser_raw = person_ref(maps[["elixhauser"]], FALSE, c("vw")),
  elixhauser = person_ref(maps[["elixhauser"]], TRUE, c("vw"))
)

cat(toJSON(list(
  r_version = R.version.string,
  package_version = as.character(packageVersion("comorbidity")),
  codes = codes, single = single, persons = persons
), digits = NA, auto_unbox = TRUE))
