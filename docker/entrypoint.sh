#!/usr/bin/env bash
set -euo pipefail

# ---- Cache dirs (always writable) ----
mkdir -p \
  /tmp/matplotlib \
  /tmp/.cache \
  /tmp/torchinductor \
  /tmp/torch \
  /tmp/hf/transformers \
  /tmp/ultralytics

echo "LeafMachine2 container starting"
echo "  Working dir: $(pwd)"
echo "  Running as: $(id -u):$(id -g)"
echo "  Python: $(python --version 2>/dev/null || true)"
echo "  YOLO_CONFIG_DIR: ${YOLO_CONFIG_DIR:-}"
echo "  MPLCONFIGDIR: ${MPLCONFIGDIR:-}"

# IMPORTANT:
# Your Dockerfiles run as a non-root user (USER lm2).
# Therefore, do NOT attempt useradd/groupadd/chown here, and do NOT use gosu.
exec "$@"
