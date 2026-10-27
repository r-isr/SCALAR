#!/bin/bash
# SCALAR on ImageNet-C (ViT-B/16), fixed label skew, high adaptation speed. Run from the repository root.
#   bash bash/scalar_skew.sh                     # SCALAR (uniform target prior)
#   PRIOR_MODE=base bash bash/scalar_skew.sh     # teacher-student baseline without SCALAR
# Extra config overrides can be appended, e.g.:
#   bash bash/scalar_skew.sh CORRUPTION.TYPE "['gaussian_noise']"

DATA_DIR=${DATA_DIR:-./data}
CKPT=${CKPT:-./ckpt/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz}
PRIOR_MODE=${PRIOR_MODE:-uniform}

python main.py --cfg cfgs/vit/scalar.yaml --data_set imagenet --data_dir "$DATA_DIR" \
  --checkpoint "$CKPT" --skew fixed --prior_mode "$PRIOR_MODE" --tau 0.1 --window 1024 \
  OPTIM.LR 1e-5 OPTIM.EMA_ATT 0.9 OPTIM.EMA_WEIGH 0.9 "$@"
