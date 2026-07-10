#!/bin/bash
set -e

SPATH=$(dirname "$(realpath "${BASH_SOURCE[0]}")");
BDIR=$(realpath "$SPATH");

ANCHOR_SUFFIX=".int16_anchor"
MODELS_DIR=""

usage() {
  echo "usage: $0 [--suffix suffix] models_dir"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --suffix)
      ANCHOR_SUFFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "[ERROR] unknown option $1"
      usage
      exit 1
      ;;
    *)
      if [ -n "$MODELS_DIR" ]; then
        echo "[ERROR] too many positional arguments"
        usage
        exit 1
      fi
      MODELS_DIR="$1"
      shift
      ;;
  esac
done

if [ -z "$MODELS_DIR" ]; then
  usage
  exit 1
fi
if [ ! -d "$MODELS_DIR" ]; then
  echo "[ERROR] no directory $MODELS_DIR"
  exit 1
fi

MODELS_DIR=$(realpath "$MODELS_DIR")

cd "$BDIR"
mkdir -p build
cd build
if ! cmake .. > /dev/null; then
  echo -e "   \e[31m[FAIL]\e[0m cmake"
  exit 1
fi
if ! make -j > /dev/null; then
  echo -e "   \e[31m[FAIL]\e[0m build"
  exit 1
fi

cpu_supports() {
  local feature="$1"
  if [ ! -r /proc/cpuinfo ]; then
    return 0
  fi
  case "$feature" in
    avx512)
      grep -m1 -qw avx512f /proc/cpuinfo && grep -m1 -qw avx512bw /proc/cpuinfo && grep -m1 -qw avx512dq /proc/cpuinfo
      ;;
    *)
      grep -m1 -qw "$feature" /proc/cpuinfo
      ;;
  esac
}

RUNNERS=()
add_runner() {
  local runner="$1"
  local feature="$2"
  if [ ! -x "$runner" ]; then
    return
  fi
  if [ -n "$feature" ] && ! cpu_supports "$feature"; then
    echo "[SKIP] $(basename "$runner") requires $feature"
    return
  fi
  RUNNERS+=("$runner")
}

add_runner "$BDIR/build/test_sadl_int16_scalar" ""
add_runner "$BDIR/build/test_sadl_int16_avx2" "avx2"
add_runner "$BDIR/build/test_sadl_int16_avx512" "avx512"
add_runner "$BDIR/build/test_sadl_int16_avx2_mempool" "avx2"
add_runner "$BDIR/build/test_sadl_int16_avx2_sparse" "avx2"

if [ "${#RUNNERS[@]}" -eq 0 ]; then
  echo "[ERROR] no int16 check runners found"
  exit 1
fi

COUNT=0
FAIL=0
while IFS= read -r -d '' MODEL; do
  COUNT=$((COUNT + 1))
  ANCHOR="${MODEL%.sadl}${ANCHOR_SUFFIX}"
  if [ ! -f "$ANCHOR" ]; then
    echo -e "   \e[31m[FAIL]\e[0m $(basename "$MODEL") missing anchor $(basename "$ANCHOR")"
    FAIL=1
    continue
  fi

  for RUNNER in "${RUNNERS[@]}"; do
    NAME=$(basename "$RUNNER")
    if "$RUNNER" check "$MODEL" "$ANCHOR" > /dev/null; then
      echo -e "   \e[32m[PASS]\e[0m $(basename "$MODEL") $NAME"
    else
      echo -e "   \e[31m[FAIL]\e[0m $(basename "$MODEL") $NAME"
      FAIL=1
    fi
  done
done < <(find "$MODELS_DIR" -maxdepth 1 -type f -name '*.sadl' -print0 | sort -z)

if [ "$COUNT" -eq 0 ]; then
  echo "[ERROR] no .sadl models found in $MODELS_DIR"
  exit 1
fi

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
