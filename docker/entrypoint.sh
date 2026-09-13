#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# FlashHead API entrypoint — configure via environment variables:
#
#   HOST          bind address              (default 0.0.0.0)
#   PORT          port                      (default 8000)
#   MODEL_TYPE    lite | pro                (default lite)
#   CKPT_DIR      FlashHead checkpoint dir  (default models/SoulX-FlashHead-1_3B)
#   WAV2VEC_DIR   wav2vec2 dir              (default models/wav2vec2-base-960h)
#   IDLE_TIMEOUT  idle session TTL (sec)    (default 300)
#   GC_INTERVAL   session GC interval (sec) (default 60)
#   AUTOLOAD      1 = load model on boot    (default 1; set 0 to load via API)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_TYPE="${MODEL_TYPE:-lite}"
CKPT_DIR="${CKPT_DIR:-models/SoulX-FlashHead-1_3B}"
WAV2VEC_DIR="${WAV2VEC_DIR:-models/wav2vec2-base-960h}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-300}"
GC_INTERVAL="${GC_INTERVAL:-60}"
AUTOLOAD="${AUTOLOAD:-1}"

ARGS=( --host "$HOST" --port "$PORT"
       --idle-timeout "$IDLE_TIMEOUT" --gc-interval "$GC_INTERVAL" )

if [[ "$AUTOLOAD" == "1" ]]; then
  if [[ ! -d "$CKPT_DIR" || ! -d "$WAV2VEC_DIR" ]]; then
    echo "[entrypoint] ERROR: AUTOLOAD=1 but model directories were not found:" >&2
    echo "             CKPT_DIR   = $CKPT_DIR" >&2
    echo "             WAV2VEC_DIR= $WAV2VEC_DIR" >&2
    echo "             Checkpoints are not included in the image." >&2
    echo "             Mount them, e.g.:  -v \"\$(pwd)/models:/checkpoints:ro\"" >&2
    echo "             (or set AUTOLOAD=0 to boot without a model and load it via POST /model/load)" >&2
    exit 1
  fi

  # Validate that the externally mounted checkpoint contains the selected model.
  if [[ "$MODEL_TYPE" == "lite" ]]; then required=( Model_Lite VAE_LTX )
  else                                    required=( Model_Pro  VAE_Wan  ); fi
  for sub in "${required[@]}"; do
    if [[ ! -d "$CKPT_DIR/$sub" ]]; then
      echo "[entrypoint] ERROR: MODEL_TYPE=$MODEL_TYPE needs $CKPT_DIR/$sub, which is missing." >&2
      echo "             Mount a checkpoint containing the required $sub directory." >&2
      exit 1
    fi
  done

  ARGS+=( --model-type "$MODEL_TYPE" --ckpt-dir "$CKPT_DIR" --wav2vec-dir "$WAV2VEC_DIR" )
fi

echo "[entrypoint] starting FlashHead API:"
echo "             python server2.py ${ARGS[*]}"
exec python server2.py "${ARGS[@]}"
