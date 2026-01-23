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

# ---- Create a user that matches host UID/GID (prevents lock icons + getpwuid errors) ----
LM2_UID="${LM2_UID:-1000}"
LM2_GID="${LM2_GID:-1000}"

# Create group if needed
if ! getent group "${LM2_GID}" >/dev/null 2>&1; then
  groupadd -g "${LM2_GID}" lm2host >/dev/null 2>&1 || true
fi

# Create user if needed
if ! getent passwd "${LM2_UID}" >/dev/null 2>&1; then
  useradd -m -u "${LM2_UID}" -g "${LM2_GID}" -s /bin/bash lm2host >/dev/null 2>&1 || true
fi

# Make sure repo is writable (best-effort; if host forbids, we'll see it)
chown -R "${LM2_UID}:${LM2_GID}" /app >/dev/null 2>&1 || true

echo "LeafMachine2 container starting"
echo "  Working dir: $(pwd)"
echo "  Running as UID:GID ${LM2_UID}:${LM2_GID}"
echo "  Python: $(python --version 2>/dev/null || true)"
echo "  YOLO_CONFIG_DIR: ${YOLO_CONFIG_DIR:-}"
echo "  MPLCONFIGDIR: ${MPLCONFIGDIR:-}"

# Run as the host-matching user
exec gosu "${LM2_UID}:${LM2_GID}" "$@"
