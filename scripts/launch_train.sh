#!/usr/bin/env bash
set -euo pipefail
torchrun --standalone --nproc-per-node=1 -m dgvlm_equitable_glioma.commands.train "$@"

