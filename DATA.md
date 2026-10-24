# Data preparation

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
