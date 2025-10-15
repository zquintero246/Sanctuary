#!/usr/bin/env bash
set -euo pipefail

export SANCTUARY_SR=${SANCTUARY_SR:-16000}
export SANCTUARY_FRAME_MS=${SANCTUARY_FRAME_MS:-20}

python -m Sanctuary.main
