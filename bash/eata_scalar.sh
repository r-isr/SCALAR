#!/bin/bash
# EATA with SCALAR on ImageNet-C (ViT-B/16), dynamic label skew, aggressive adaptation.
# Run from the repository root.
#   bash bash/eata_scalar.sh                     # EATA with SCALAR (uniform target prior)
#   PRIOR_MODE=base bash bash/eata_scalar.sh     # EATA alone on the same stream
# Extra config overrides can be appended, e.g.:
#   bash bash/eata_scalar.sh CORRUPTION.TYPE "['gaussian_noise']"

DATA_DIR=${DATA_DIR:-./data}
PRIOR_MODE=${PRIOR_MODE:-uniform}

PYTHONHASHSEED=22 python main.py --cfg cfgs/vit/eata.yaml --data_set imagenet --data_dir "$DATA_DIR" \
  --skew dynamic --prior_mode "$PRIOR_MODE" --tau 0.7 --window 4096 \
  OPTIM.LR 6e-4 "$@"
