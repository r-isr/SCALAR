# Data and models

All datasets are read from the directory given by `--data_dir`.

## Downloads

- ImageNet-C: [Zenodo](https://zenodo.org/record/2235448)
- CIFAR-10-C: [Zenodo](https://zenodo.org/record/2535967)
- CIFAR-100-C: [Zenodo](https://zenodo.org/record/3555552)
- ImageNet validation set: [image-net.org](https://image-net.org/) (used for the clean evaluation on ImageNet)

## Expected layout

```
<DATA_DIR>/
├── ImageNet-C/
│   ├── gaussian_noise/
│   │   └── 5/<class folders>
│   ├── ...                          # the other corruption types
│   └── clean/
│       └── 1/<class folders>        # ImageNet validation images
├── CIFAR-10-C/                       # *.npy files
└── CIFAR-100-C/                      # *.npy files
```

## Clean ImageNet split

The clean ImageNet evaluation is loaded as an additional "corruption" named `clean` with severity `1`.
This folder is not part of the ImageNet-C download: place the ImageNet validation set there,
with one sub-folder per class, for example with a symbolic link:

```bash
mkdir -p <DATA_DIR>/ImageNet-C/clean
ln -s /path/to/imagenet/val <DATA_DIR>/ImageNet-C/clean/1
```

## Source checkpoints

| Dataset | Model | Checkpoint |
|---|---|---|
| ImageNet | ViT-B/16 (AugReg, IN-21k → IN-1k) | [B_16-…-res_224.npz](https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz) |
| CIFAR-10 | ViT-B/16 (384) | [vit_cifar10_16_384.t7](https://drive.google.com/file/d/1pAoz4Wwos74DjWPQ5d-6ntyjQkmp9FPE/view) (from ViDA) |
| CIFAR-100 | ViT-B/16 (384) | [vit_cifar100_16_384.t7](https://drive.google.com/file/d/1yRekkpkIdwX_LFsOh4Ba9ndaECnY-UC-/view) (from ViDA) |
| ImageNet (Tent) | ResNet-50 | `Standard_R50`, loaded through RobustBench |

The ViT checkpoints are passed with `--checkpoint`:

```bash
mkdir -p ckpt
wget -P ckpt https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
```

Datasets and checkpoints are obtained from their original sources and are subject to their respective licenses.
