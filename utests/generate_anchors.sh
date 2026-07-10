#!/bin/bash
set -e

SPATH=$(dirname "$(realpath "${BASH_SOURCE[0]}")");
BDIR=$(realpath "$SPATH");

RANGE_MIN=-512
RANGE_MAX=512
SEED=42
ANCHOR_SUFFIX=".int16_anchor"
MODELS_DIR=""

usage() {
  echo "usage: $0 [--min value] [--max value] [--seed value] [--suffix suffix] models_dir"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --min)
      RANGE_MIN="$2"
      shift 2
      ;;
    --max)
      RANGE_MAX="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
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

GENERATOR="$BDIR/build/test_sadl_int16_scalar"
if [ ! -x "$GENERATOR" ]; then
  echo "[ERROR] missing generator $GENERATOR"
  exit 1
fi

COUNT=0
FAIL=0
while IFS= read -r -d '' MODEL; do
  ANCHOR="${MODEL%.sadl}${ANCHOR_SUFFIX}"
  if "$GENERATOR" generate "$MODEL" "$ANCHOR" "$RANGE_MIN" "$RANGE_MAX" "$SEED" > /dev/null; then
    echo -e "   \e[32m[PASS]\e[0m $(basename "$MODEL") generate"
  else
    echo -e "   \e[31m[FAIL]\e[0m $(basename "$MODEL") generate"
    FAIL=1
  fi
  COUNT=$((COUNT + 1))
done < <(find "$MODELS_DIR" -maxdepth 1 -type f -name '*.sadl' -print0 | sort -z)

if [ "$COUNT" -eq 0 ]; then
  echo "[ERROR] no .sadl models found in $MODELS_DIR"
  exit 1
fi

if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
