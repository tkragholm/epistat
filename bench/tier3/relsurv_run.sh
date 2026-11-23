#!/usr/bin/env bash
# Runs the Rust program and the R script of the tier 3 relative survival
# benchmark on the same simulated data, one process per method and size, and
# records wall time, peak memory and whether the process finished within the
# time limit.
#
#   SCRATCH=/some/dir/outside/the/repo bench/tier3/relsurv_run.sh
#
# Settings, all optional: SIZES (default "10000 100000 1000000"), METHODS
# (default every method of crates/epistat-survival/benches/tier3_relsurv.rs),
# LIMIT (seconds for one process, default 600), MEMORY (default 11G), IMPLS
# ("rust r", the default, or one of them), OUT (default SCRATCH/out), and
# BIN_tier3_relsurv (an already built program). Results go to OUT as
# bench/tier2/run.sh writes them. Not for the Mac above 10^5 rows. The exact
# estimators of net_pp and net_e2 are refused above 3e8 pairs of a person at
# risk and a time point, which 10^6 rows exceed, and the process then fails.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000}"
METHODS="${METHODS:-net_pp net_e2 net_pp_monthly excess_poisson excess_flexpar}"
OUT="${OUT:-$SCRATCH/out}"
LIMIT="${LIMIT:-600}"
MEMORY="${MEMORY:-11G}"
IMPLS="${IMPLS:-rust r}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_kb" > "$OUT/runs.csv"

BIN="${BIN_tier3_relsurv:-}"
if [ -z "$BIN" ]; then
  BIN="$(cd "$ROOT" && cargo bench -p epistat-survival --bench tier3_relsurv --no-run --message-format=json 2>/dev/null |
    python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'tier3_relsurv':
        print(j['executable'])")"
fi

guarded() { # timefile command...
  local tf=$1; shift
  if systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$MEMORY" -p MemorySwapMax=0 -- \
      python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" "$@"
  else
    ( ulimit -v 12000000 2>/dev/null; python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" "$@" )
  fi
}
record() { # impl n method status timefile
  local rss wall
  rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$5")
  wall=$(awk -F': ' '/Elapsed/ {print $2}' "$5")
  echo "$1,$2,$3,$4,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
}
reps_for() { if [ "$1" -ge 1000000 ]; then echo 1; elif [ "$1" -ge 100000 ]; then echo 3; else echo 5; fi; }

for n in $SIZES; do
  [ -f "$SCRATCH/rel_$n.csv" ] || N=$n METHOD=gen_rel SCRATCH="$SCRATCH" OUT="$OUT" "$BIN" > "$OUT/log_gen_rel_$n.txt" 2>&1
  for m in $METHODS; do
    for impl in $IMPLS; do
      tf="$OUT/time_${impl}_${m}_${n}.txt"
      if [ "$impl" = rust ]; then
        N=$n METHOD=$m REPS="$(reps_for "$n")" SCRATCH="$SCRATCH" OUT="$OUT" \
          guarded "$tf" "$BIN" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
      else
        REPS="$(reps_for "$n")" OUT="$OUT" \
          guarded "$tf" Rscript "$HERE/relsurv.R" "$n" "$m" "$SCRATCH" > "$OUT/log_r_${m}_${n}.txt" 2>&1
      fi
      st=$?
      status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
      record "$impl" "$n" "$m" "$status" "$tf"
    done
  done
  Rscript "$HERE/../tier1/compare.R" "$n" "$SCRATCH" > "$OUT/compare_$n.csv" 2>&1
  [ -n "${KEEP_DATA:-}" ] || find "$SCRATCH" -maxdepth 1 -name "*_$n.csv" -delete
done
