# Controlling Temporal Pseudo-Label Marginals for Stable Online Test-Time Adaptation

Code for **SCALAR** (Sinkhorn Class-Marginal Regularization).

![SCALAR overview](figures/scalar-overview.png)

Continual test-time adaptation (CTTA) updates models online with streams of
unlabeled data as distributions shift. However, CTTA can become unstable by
reinforcing early prediction biases, causing pseudo-labels to concentrate on a few
classes and resulting in drift or collapse. We found this instability is linked to the
narrowing of pseudo-label class usage over time. Most CTTA methods aim to
improve reliability or diversity by selecting samples or adjusting loss weights, but
they do not directly control pseudo-label distribution across classes over time. We
introduce SCALAR (Sinkhorn Class-Marginal Regularization), a simple plug-in
module that enables temporal control over pseudo-label distribution within a short
window of recent predictions.

## Installation

Tested with Python 3.8, PyTorch 1.13.1 and CUDA 11.7. The main dependencies are listed in `requirements.txt`.

```bash
conda create -n scalar python=3.8
conda activate scalar
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Data and models

Datasets (ImageNet-C, CIFAR-10-C, CIFAR-100-C), source checkpoints and the expected folder layout are described in [DATA.md](DATA.md).
Datasets and checkpoints are obtained from their original sources and are subject to their respective licenses.

## Usage

Run the scripts in `bash/` from the repository root. Paths are set with `DATA_DIR` (default `./data`) and, for `scalar.sh` and `scalar_skew.sh`, `CKPT` (default: the ViT checkpoint in `./ckpt`).

```bash
bash bash/scalar.sh            # SCALAR on ImageNet-C (ViT-B/16)
bash bash/scalar_skew.sh       # SCALAR (ViT-B/16), fixed label skew, high adaptation speed
bash bash/eata_scalar.sh       # EATA with SCALAR (ViT-B/16), dynamic label skew, aggressive adaptation
bash bash/tent_cnn.sh          # Tent (ResNet-50)
```

Main options of `main.py` (set inside the scripts):

| Option | Values | Description |
|---|---|---|
| `--prior_mode` | `base`, `uniform`, `correct` | Set in the scripts with `PRIOR_MODE` (default `uniform`). `base`: no SCALAR; `uniform`: uniform target prior; `correct`: true class prior of the stream (requires `--skew`; with `--skew none` it acts as `uniform`) |
| `--tau` | float | temperature of the logit window before Sinkhorn |
| `--window` | int | size of the FIFO logit window |
| `--skew` | `none`, `fixed`, `dynamic` | label skew of the test stream (default: `none`) |
| `--data_set` | `imagenet`, `cifar10`, `cifar100` | dataset |

## Acknowledgement

This code builds on [Tent](https://github.com/DequanWang/tent), [EATA](https://github.com/mr-eggplant/EATA),
[CoTTA](https://github.com/qinenergy/cotta), [ViDA](https://github.com/Yangsenqiao/vida)
(code base, adapted RobustBench copy and CIFAR source checkpoints) and [RobustBench](https://github.com/RobustBench/robustbench).

## License

MIT, see [LICENSE](LICENSE).
