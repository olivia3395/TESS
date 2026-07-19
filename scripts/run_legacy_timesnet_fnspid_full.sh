#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lyc/workspace/TESS-RC2}"

OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/legacy_timesnet_fnspid_full}" \
DEVICE="${DEVICE:-cuda}" \
EPOCHS="${EPOCHS:-25}" \
BATCH_SIZE="${BATCH_SIZE:-32}" \
LR="${LR:-0.0005}" \
SEED="${SEED:-42}" \
D_MODEL="${D_MODEL:-512}" \
E_LAYERS="${E_LAYERS:-2}" \
TOP_K="${TOP_K:-5}" \
NUM_KERNELS="${NUM_KERNELS:-6}" \
"$ROOT/scripts/run_legacy_timesnet_fnspid_numeric.sh"
