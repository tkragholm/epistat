#!/usr/bin/env bash
# Runs the Rust survival program and the R script on the same simulated
# cohorts, one process per method and size, and records wall time, peak memory
# and whether the process finished within the time limit.
#
#   SCRATCH=/some/dir/outside/the/repo benches/survival/run.sh
#
# Settings, all optional:
#   SIZES     default "10000 100000 1000000"
#   METHODS   default every method, each also with the suffix _split
#   LIMIT     seconds allowed to one process, default 60
#   IMPLS     "rust r" (default), or just one of them
#   FORCE_REPS  timed repetitions for every run, instead of the defaults
#   BOOT      bootstrap replicates of aj_boot, default 50
#   OUT       results directory, default SCRATCH/out
#   THREADS   sets RAYON_NUM_THREADS for the Rust program
#
# Results, in OUT:
#   runs.csv        impl,n,method,status,process_wall_s,peak_rss_bytes
#   rust_times.csv  n,method,reps,median_s,min_s   (estimation only)
#   r_times.csv     the same for R
#   compare_N.csv   what compare.R prints for each size
#
# A method is not run at a size when it timed out, failed or was skipped at
# the previous size. Peak memory is `maximum resident set size` of
# /usr/bin/time -l, which includes reading the cohort. The Rust name
# msurvfit is the same computation as multistate and is run on the R side
# only.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CRATE="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000}"
OUT="${OUT:-$SCRATCH/out}"
BASE="split cox_breslow cox_efron cox_exact aj aj_boot finegray survfit multistate msurvfit"
if [ -z "${METHODS:-}" ]; then
  METHODS="load"
  for m in $BASE; do
    METHODS="$METHODS $m"
    case $m in split|aj_boot|finegray) ;; *) METHODS="$METHODS ${m}_split" ;; esac
  done
fi
LIMIT="${LIMIT:-60}"
IMPLS="${IMPLS:-rust r}"
export PATH="/opt/homebrew/opt/r/bin:$PATH"
export BOOT="${BOOT:-50}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_bytes" > "$OUT/runs.csv"

BIN="$(cd "$CRATE" && cargo bench --bench survival_compare --no-run --message-format=json 2>/dev/null |
  python3 -c "
import json, sys
last = None
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'survival_compare':
        last = j['executable']
print(last)")"

reps_for() { # n
  if [ -n "${FORCE_REPS:-}" ]; then echo "$FORCE_REPS"
  elif [ "$1" -ge 1000000 ]; then echo 1
  else echo 3; fi
}

record() { # impl n method status wallfile
  local rss wall
  rss=$(awk '/maximum resident set size/ {print $1}' "$5")
  wall=$(awk '/ real / {print $1}' "$5")
  echo "$1,$2,$3,$4,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
}

run_one() { # impl n method
  local impl=$1 n=$2 m=$3 st status tf
  [ "$impl" = rust ] && [ "$m" = msurvfit ] && return
  [ "$impl" = r ] && [ "$m" = load ] && return
  if [ -n "$prev" ] && grep -q "^$impl,$prev,$m,\(timeout\|skipped\|error[0-9]*\)" "$OUT/runs.csv"; then
    echo "$impl,$n,$m,skipped,NA,NA" >> "$OUT/runs.csv"
    return
  fi
  tf="$OUT/time_${impl}_${m}_${n}.txt"
  if [ "$impl" = rust ]; then
    N=$n METHOD=$m REPS="$(reps_for "$n")" SCRATCH="$SCRATCH" OUT="$OUT" \
      /usr/bin/time -l -o "$tf" gtimeout -k 10 "$LIMIT" "$BIN" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
  else
    REPS="$(reps_for "$n")" OUT="$OUT" /usr/bin/time -l -o "$tf" \
      gtimeout -k 10 "$LIMIT" Rscript "$HERE/survival_bench.R" "$n" "$m" "$SCRATCH" \
      > "$OUT/log_r_${m}_${n}.txt" 2>&1
  fi
  st=$?
  status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
  record "$impl" "$n" "$m" "$status" "$tf"
}

[ -n "${THREADS:-}" ] && export RAYON_NUM_THREADS="$THREADS"
prev=""
for n in $SIZES; do
  [ -f "$SCRATCH/cohort_$n.csv" ] || N=$n METHOD=gen SCRATCH="$SCRATCH" "$BIN"
  for m in $METHODS; do
    for impl in $IMPLS; do run_one "$impl" "$n" "$m"; done
  done
  Rscript "$HERE/compare.R" "$n" "$SCRATCH" > "$OUT/compare_$n.csv" 2>&1
  prev=$n
done
