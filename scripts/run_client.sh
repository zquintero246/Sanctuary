#!/usr/bin/env bash
set -euo pipefail

python Sanctuary/voice_client.py \
  --uri "${SANCTUARY_URI:-ws://localhost:8000/voice}" \
  --sample-rate "${SANCTUARY_CLIENT_SR:-16000}" \
  --frame-ms "${SANCTUARY_CLIENT_FRAME_MS:-20}" \
  --print-events
