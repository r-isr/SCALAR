#!/bin/bash
# Tent on ImageNet-C (ResNet-50). Run from the repository root.
# Extra config overrides can be appended, e.g.:
#   bash bash/tent_cnn.sh CORRUPTION.TYPE "['gaussian_noise']"

DATA_DIR=${DATA_DIR:-./data}

python main.py --cfg cfgs/cnn/tent.yaml --data_set imagenet --data_dir "$DATA_DIR" "$@"
