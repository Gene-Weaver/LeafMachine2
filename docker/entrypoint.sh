#!/usr/bin/env bash
set -euo pipefail

# ---- Default cache locations (writable under non-root) ----
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics}"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf/transformers}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor}"

# ---- Ensure dirs exist ----
mkdir -p \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$YOLO_CONFIG_DIR"

echo "LeafMachine2 container starting"
echo "  Working dir: $(pwd)"
echo "  Running as: $(id -u):$(id -g)"
echo "  Python: $(python --version 2>/dev/null || true)"
echo "  YOLO_CONFIG_DIR: ${YOLO_CONFIG_DIR}"
echo "  MPLCONFIGDIR: ${MPLCONFIGDIR}"
echo "  HF_HOME: ${HF_HOME}"
echo "  TORCH_HOME: ${TORCH_HOME}"
echo "  XDG_CACHE_HOME: ${XDG_CACHE_HOME}"

exec "$@"
