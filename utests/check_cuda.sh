#!/bin/bash
# check_cuda.sh – compare SADL CPU (int16) vs SADL CUDA (int16) inference
# on every model in utests/model_int16/.
#
# Usage:
#   ./check_cuda.sh [--tol T] [--amplitude N] [--model MODEL_NAME]
#
#   --tol T          max absolute int16 difference allowed (default: 0)
#   --amplitude N    random input range [-N, N] (default: 512)
#   --model NAME     run only the model whose filename stem matches NAME

SPATH=$(dirname $(realpath ${BASH_SOURCE[0]}))
BDIR=$(realpath $SPATH)

TOL=0
AMPLITUDE=1024
FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tol)       TOL="$2";       shift 2 ;;
    --amplitude) AMPLITUDE="$2"; shift 2 ;;
    --model)     FILTER="$2";    shift 2 ;;
    *) echo "[ERROR] unknown option $1"; exit 1 ;;
  esac
done

mkdir -p "$BDIR/build"
cd "$BDIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null
make -j test_cpu_int16 test_cuda_int16 2>&1 | grep -E "error:|warning:|Built target|CUDA"
cd "$BDIR"

CPU_BIN="$BDIR/build/test_cpu_int16"
GPU_BIN="$BDIR/build/test_cuda_int16"
MODEL_DIR="$BDIR/model_int16"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ ! -x "$CPU_BIN" ]]; then
  echo "[ERROR] $CPU_BIN not found – CPU build failed"
  exit 1
fi
if [[ ! -x "$GPU_BIN" ]]; then
  echo "[ERROR] $GPU_BIN not found – CUDA build failed or no GPU compiler"
  exit 1
fi

N_PASS=0
N_FAIL=0
N_SKIP=0

for MODEL in "$MODEL_DIR"/*.sadl; do
  BNAME=$(basename "$MODEL" .sadl)

  if [[ -n "$FILTER" && "$BNAME" != *"$FILTER"* ]]; then
    continue
  fi

  REF_BIN="$TMP_DIR/${BNAME}.bin"

  "$CPU_BIN" "$MODEL" "$REF_BIN" "$AMPLITUDE" > /dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    echo -e "   \e[33m[SKIP]\e[0m $BNAME  (CPU inference failed)"
    (( N_SKIP++ ))
    continue
  fi

  OUT=$("$GPU_BIN" "$MODEL" "$REF_BIN" "$TOL" 2>&1)
  E=$?
  if (( E == 0 )); then
    echo -e "   \e[32m[PASS]\e[0m $BNAME"
    (( N_PASS++ ))
  else
    echo -e "   \e[31m[FAIL]\e[0m $BNAME"
    echo "$OUT" | sed 's/^/         /'
    (( N_FAIL++ ))
  fi
done

TOTAL=$(( N_PASS + N_FAIL + N_SKIP ))
echo ""
echo "Results: $N_PASS/$TOTAL passed, $N_FAIL failed, $N_SKIP skipped"
echo "  (amplitude=$AMPLITUDE, tol=$TOL)"

[[ $N_FAIL -eq 0 ]]
