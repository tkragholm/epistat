#!/usr/bin/env bash
# Runs the Rust programs and the R script of the tier 3 benchmarks on the same
# simulated data, one process per method and size, and records wall time, peak
# memory and whether the process finished within the time limit.
#
#   SCRATCH=/some/dir/outside/the/repo bench/tier3/run.sh
#
# Settings, all optional:
#   SIZES     default "10000 100000 1000000 10000000"; a method runs up to its
#             own largest size
#   METHODS   default every method in the table below
#   LIMIT     seconds allowed to one process, default 120
#   MEMORY    memory cap of one process, default 11G (systemd-run when it
#             works, ulimit otherwise)
#   BIN_<bench>  path of an already built program, instead of cargo bench
#   IMPLS     "rust r" (default), or just one of them
#   OUT       results directory, default SCRATCH/out
#
# Results, in OUT:
#   runs.csv        impl,n,method,status,process_wall_s,peak_rss_kb
#   rust_times.csv  n,method,reps,median_s,min_s   (the method only)
#   r_times.csv     the same for R
#   compare_N.csv   what compare.R prints for each size
#
# A method is not run at a size when it timed out, failed or was skipped at
# the previous size. Peak memory is the maximum resident set size of the
# process tree, from measure.py, and includes reading the data file.
set -u
: "${SCRATCH:?set SCRATCH to a directory outside the repository}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SIZES="${SIZES:-10000 100000 1000000 10000000}"
OUT="${OUT:-$SCRATCH/out}"
LIMIT="${LIMIT:-120}"
MEMORY="${MEMORY:-11G}"
IMPLS="${IMPLS:-rust r}"
mkdir -p "$OUT"
[ -f "$OUT/runs.csv" ] || echo "impl,n,method,status,process_wall_s,peak_rss_kb" > "$OUT/runs.csv"

# method:crate:bench:kind:largest size
TABLE="
ebal_att:epistat-causal:tier3_balance:bal:10000000
ebal_ate_m2:epistat-causal:tier3_balance:bal:10000000
cbps_exact:epistat-causal:tier3_balance:bal:1000000
cbps_over:epistat-causal:tier3_balance:bal:100000
"
METHODS="${METHODS:-$(echo "$TABLE" | cut -d: -f1 | tr '\n' ' ')}"
field() { echo "$TABLE" | awk -F: -v m="$1" -v f="$2" '$1 == m {print $f}'; }

declare -A BIN
build_all() { # builds every program once, in one cargo run so features agree
  local crates="" m benches="" c b missing=""
  for m in $METHODS; do
    b=$(field "$m" 3); given="BIN_$b"
    [ -z "${!given:-}" ] && missing=1
  done
  if [ -z "$missing" ]; then
    for m in $METHODS; do b=$(field "$m" 3); given="BIN_$b"; BIN[$b]="${!given}"; done
    return
  fi
  for m in $(echo "$TABLE" | cut -d: -f1); do
    c=$(field "$m" 2); b=$(field "$m" 3)
    case " $crates " in *" $c "*) ;; *) crates="$crates $c" ;; esac
    case " $benches " in *" $b "*) ;; *) benches="$benches $b" ;; esac
  done
  local pkgs=""
  for c in $crates; do pkgs="$pkgs -p $c"; done
  local found
  found="$(cd "$ROOT" && cargo bench $pkgs --benches --no-run --message-format=json 2>/dev/null |
    python3 -c "
import json, sys
for line in sys.stdin:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    if j.get('reason') == 'compiler-artifact' and j.get('executable') and j['target']['name'].startswith('tier3_'):
        print(j['target']['name'], j['executable'])")"
  while read -r name path; do
    [ -n "${name:-}" ] && BIN[$name]=$path
  done <<< "$found"
  for b in $benches; do
    local given="BIN_$b"
    [ -n "${!given:-}" ] && BIN[$b]="${!given}"
    echo "built $b: ${BIN[$b]:-MISSING}" >&2
  done
}
bin_of() { echo "${BIN[$1]}"; }
build_all

reps_for() { # n
  if [ -n "${FORCE_REPS:-}" ]; then echo "$FORCE_REPS"
  elif [ "$1" -ge 1000000 ]; then echo 1
  elif [ "$1" -ge 100000 ]; then echo 3
  else echo 5; fi
}

# Runs a command under the memory cap and the time limit.
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

run_one() { # impl n method
  local impl=$1 n=$2 m=$3 st status tf
  if [ -n "$prev" ] && grep -q "^$impl,$prev,$m,\(timeout\|skipped\|error[0-9]*\)" "$OUT/runs.csv"; then
    echo "$impl,$n,$m,skipped,NA,NA" >> "$OUT/runs.csv"
    return
  fi
  tf="$OUT/time_${impl}_${m}_${n}.txt"
  if [ "$impl" = rust ]; then
    N=$n METHOD=$m REPS="$(reps_for "$n")" SCRATCH="$SCRATCH" OUT="$OUT" \
      guarded "$tf" "$(bin_of "$(field "$m" 3)")" > "$OUT/log_rust_${m}_${n}.txt" 2>&1
  else
    REPS="$(reps_for "$n")" OUT="$OUT" \
      guarded "$tf" Rscript "$HERE/bench.R" "$n" "$m" "$SCRATCH" > "$OUT/log_r_${m}_${n}.txt" 2>&1
  fi
  st=$?
  status=ok; [ $st -eq 124 ] && status=timeout; [ $st -ne 0 ] && [ $st -ne 124 ] && status="error$st"
  record "$impl" "$n" "$m" "$status" "$tf"
}

generate() { # kind n
  local kind=$1 n=$2 m crate bench
  [ "$kind" = none ] && return
  [ -f "$SCRATCH/${kind}_$n.csv" ] && return
  if [ "$kind" = pool ]; then
    OUT="$OUT" Rscript "$HERE/bench.R" "$n" gen_pool "$SCRATCH" > "$OUT/log_gen_pool.txt" 2>&1
    return
  fi
  m=$(echo "$TABLE" | awk -F: -v k="$kind" '$4 == k {print $1; exit}')
  N=$n METHOD="gen_$kind" SCRATCH="$SCRATCH" OUT="$OUT" \
    "$(bin_of "$(field "$m" 3)")" > "$OUT/log_gen_${kind}_$n.txt" 2>&1
}

prev_size=""
for n in $SIZES; do
  for m in $METHODS; do
    maxn=$(field "$m" 5); kind=$(field "$m" 4)
    size=$n
    [ "$kind" = pool ] && { [ "$n" = "${SIZES%% *}" ] || continue; size=50; }
    [ "$size" -gt "$maxn" ] && continue
    generate "$kind" "$size"
  done
  for m in $METHODS; do
    maxn=$(field "$m" 5); kind=$(field "$m" 4)
    size=$n
    [ "$kind" = pool ] && { [ "$n" = "${SIZES%% *}" ] || continue; size=50; }
    [ "$size" -gt "$maxn" ] && continue
    prev=""
    [ -n "$prev_size" ] && [ "$kind" != pool ] && prev=$prev_size
    for impl in $IMPLS; do run_one "$impl" "$size" "$m"; done
  done
  Rscript "$HERE/../tier1/compare.R" "$n" "$SCRATCH" > "$OUT/compare_$n.csv" 2>&1
  # Data files of this size are not needed again.
  [ -n "${KEEP_DATA:-}" ] || find "$SCRATCH" -maxdepth 1 -name "*_$n.csv" -delete
  prev_size=$n
done
[ -f "$OUT/compare_50.csv" ] || Rscript "$HERE/../tier1/compare.R" 50 "$SCRATCH" > "$OUT/compare_50.csv" 2>&1
