#!/usr/bin/env bash
# Times the parametric g-formula in Rust and in gfoRmula on the same simulated
# data, one process per method and size, with the memory cap and time limit of
# bench/tier2/run.sh.
#
#   SCRATCH=/some/dir/outside/the/repo bench/tier3/gformula_run.sh
#
# Settings, all optional: SIZES (persons, default "10000 100000 1000000"; the
# data have about 4.5 rows per person), METHODS (default "gf_static gf_dynamic
# gf_boot"), LIMIT (seconds per process, default 600), REPS (default 1 at 10^6
# persons, otherwise 3), OUT, BIN (a built tier3_gformula program instead of
# cargo bench). Not for the Mac above 2 * 10^4 persons.
#
# Results in OUT: runs.csv (impl,n,method,status,process_wall_s,peak_rss_kb),
# rust_times.csv and r_times.csv (n,method,reps,median_s,min_s), and
# compare_N.csv from bench/tier1/compare.R: the model coefficients agree to
# rounding, the risks and bootstrap standard errors (info_) within the Monte
# Carlo error.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000}"
METHODS="${METHODS:-gf_static gf_dynamic gf_boot}"
LIMIT="${LIMIT:-600}"
OUT="${OUT:-$SCRATCH/out}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_kb" > "$OUT/runs.csv"
BIN="${BIN:-$(cd "$ROOT" && cargo bench -p epistat-causal --bench tier3_gformula --no-run --message-format=json 2>/dev/null |
  python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'] == 'tier3_gformula':
        print(j['executable'])")}"
for n in $SIZES; do
  reps=${REPS:-3}; [ "$n" -ge 1000000 ] && reps=${REPS:-1}
  [ -f "$SCRATCH/gf_$n.csv" ] || N=$n METHOD=gen_gf SCRATCH="$SCRATCH" OUT="$OUT" "$BIN"
  for m in $METHODS; do
    for impl in rust r; do
      tf="$OUT/time_${impl}_${m}_${n}.txt"
      if [ "$impl" = rust ]; then
        N=$n METHOD=$m REPS=$reps SCRATCH="$SCRATCH" OUT="$OUT" \
          python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" "$BIN" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
      else
        REPS=$reps OUT="$OUT" \
          python3 "$HERE/../tier1/measure.py" "$tf" "$LIMIT" Rscript "$HERE/gformula.R" "$n" "$m" "$SCRATCH" \
          > "$OUT/log_r_${m}_${n}.txt" 2>&1
      fi
      st=$?; status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
      rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$tf")
      wall=$(awk -F': ' '/Elapsed/ {print $2}' "$tf")
      echo "$impl,$n,$m,$status,${wall:-NA},${rss:-NA}" >> "$OUT/runs.csv"
    done
  done
  Rscript "$HERE/../tier1/compare.R" "$n" "$SCRATCH" > "$OUT/compare_$n.csv" 2>&1
  [ -n "${KEEP_DATA:-}" ] || rm -f "$SCRATCH/gf_$n.csv"
done
