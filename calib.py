# calib.py
# Evaluation-only calibration metrics (top-1 error, top-label ECE, NLL).
#
# Consumes the per-batch model outputs that are ALREADY returned to the main
# eval loop (the pre-adaptation logits for that batch) plus ground-truth labels.
# Ground-truth is used strictly for evaluation here; this module never touches
# the adaptation loss, filtering, queue, or any SCALAR computation.
import math
import torch
import torch.nn.functional as F


class CalibrationMeter:
    """Accumulate top-1 error, top-label ECE (equal-width bins), and NLL.

    Aggregates globally across all batches passed to `update`. Call `compute`
    once at the end to read the metrics.
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        device = torch.device("cpu")
        self.bin_count = torch.zeros(n_bins, dtype=torch.float64, device=device)
        self.bin_conf_sum = torch.zeros(n_bins, dtype=torch.float64, device=device)
        self.bin_correct_sum = torch.zeros(n_bins, dtype=torch.float64, device=device)
        self.nll_sum = 0.0
        self.total_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """Update accumulators from one batch.

        logits: [B, C] pre-adaptation model outputs for the batch.
        labels: [B]    ground-truth class ids (evaluation only).
        """
        if logits.numel() == 0:
            return

        logits = logits.detach()
        labels = labels.detach()

        # NLL over ground-truth labels (sum; averaged at compute time).
        self.nll_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())

        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)                      # top-label confidence + prediction
        correct = (pred == labels).to(torch.float64)

        conf = conf.to(torch.float64).cpu()
        correct = correct.cpu()

        # 15 equal-width bins over [0, 1]; conf == 1.0 falls into the last bin.
        idx = torch.clamp((conf * self.n_bins).floor().long(), 0, self.n_bins - 1)

        ones = torch.ones_like(conf)
        self.bin_count.scatter_add_(0, idx, ones)
        self.bin_conf_sum.scatter_add_(0, idx, conf)
        self.bin_correct_sum.scatter_add_(0, idx, correct)

        self.total_count += int(labels.shape[0])

    def compute(self):
        """Return {'top1_err', 'ece', 'nll', 'total_count'}; NaNs if empty."""
        n = self.total_count
        if n <= 0:
            return {"top1_err": float("nan"), "ece": float("nan"),
                    "nll": float("nan"), "total_count": 0}

        total_correct = float(self.bin_correct_sum.sum().item())
        top1_err = 1.0 - total_correct / n
        nll = self.nll_sum / n

        ece = 0.0
        for b in range(self.n_bins):
            cnt = float(self.bin_count[b].item())
            if cnt <= 0:
                continue
            bin_avg_conf = float(self.bin_conf_sum[b].item()) / cnt
            bin_acc = float(self.bin_correct_sum[b].item()) / cnt
            ece += (cnt / n) * abs(bin_acc - bin_avg_conf)

        return {"top1_err": top1_err, "ece": ece, "nll": nll, "total_count": n}
