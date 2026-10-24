# stats.py
# stats.py
from dataclasses import dataclass, field
from typing import Optional, Any, Deque, Tuple
from collections import deque
import torch
import csv
import math



@dataclass
class PseudoLabelStatsLogger:
    csv_path: str
    num_classes: int
    head_class_ids: Optional[torch.Tensor] = None  # shape [H]
    write_header: bool = True

    # --- new: window + sparsification ---
    window_images: int = 1024          # rolling window size in *images*
    log_every_k: int = 1               # write every k batches (still updates window every batch)

    step: int = 0
    _fh: Any = field(init=False, default=None)
    _writer: Any = field(init=False, default=None)

    # Rolling window stores per-batch summaries to avoid storing per-image probs
    _q: Deque[Tuple[int, torch.Tensor, torch.Tensor]] = field(init=False, default_factory=deque)
    _win_n: int = field(init=False, default=0)
    _win_sum_probs: torch.Tensor = field(init=False)
    _win_top1_counts: torch.Tensor = field(init=False)

    # Per-corruption accumulators
    _corr_name: str = field(init=False, default="")
    _corr_n: int = field(init=False, default=0)
    _corr_sum_probs: torch.Tensor = field(init=False)
    _corr_top1_counts: torch.Tensor = field(init=False)
    _corr_acc_sum: float = field(init=False, default=0.0)  # sum of (acc1_batch * batch_size)

    def __post_init__(self):
        device = torch.device("cpu")
        self._win_sum_probs = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self._win_top1_counts = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self._corr_sum_probs = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self._corr_top1_counts = torch.zeros(self.num_classes, dtype=torch.float64, device=device)

    def open(self):
        self._fh = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._fh)
        if self.write_header:
            self._writer.writerow([
                "step","phase","corruption","batch_idx",
                "batch_size","acc1_batch",

                # batch stats (mean probs over this batch)
                "entropy_batch","effective_classes_batch",

                # rolling window stats (last window_images, approx within a batch)
                "entropy_win","effective_classes_win",

                # concentration diagnostics (window)
                "top1_prob_mass_win","top5_prob_mass_win","top10_prob_mass_win","top20_prob_mass_win",
                "top1_top1_mass_win","top5_top1_mass_win","top10_top1_mass_win","top20_top1_mass_win",

                # keep the original single-number peakedness too (same as top1_top1_mass_win)
                "top1_max_frac_win",


                # optional head/tail (only if head_class_ids is provided)
                "top1_head_frac_win","top1_tail_frac_win",
                "prob_head_mass_win","prob_tail_mass_win",

                # corruption summary fields (filled only on summary rows)
                "acc1_corruption"
            ])
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None

    # ---- helpers ----
    @staticmethod
    def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> float:
        p = torch.clamp(p, min=eps)
        return float(-(p * torch.log(p)).sum().item())

    def start_corruption(self, corruption: str):
        """Call once at the beginning of each corruption stream."""
        self._corr_name = corruption
        self._corr_n = 0
        self._corr_sum_probs.zero_()
        self._corr_top1_counts.zero_()
        self._corr_acc_sum = 0.0
        self._q.clear()
        self._win_n = 0
        self._win_sum_probs.zero_()
        self._win_top1_counts.zero_()


    def end_corruption(self):
        """Write one summary row for the current corruption."""
        if self._corr_n <= 0:
            return

        p_corr = (self._corr_sum_probs / max(1, self._corr_n)).to(torch.float64)
        ent_corr = self._entropy_from_probs(p_corr)
        eff_corr = float(math.exp(ent_corr))
        acc_corr = float(self._corr_acc_sum / max(1, self._corr_n))

        # For summary row, we also report window-derived fields as NaN (or reuse corr stats).
        # Here: reuse corruption stats for entropy/effective classes, window fields left as NaN.
        self._writer.writerow([
            self.step, "corruption_summary", self._corr_name, -1,
            self._corr_n, float("nan"),

            ent_corr, eff_corr,

            # window-level fields (not defined for summary; keep NaN)
            float("nan"), float("nan"),  # entropy_win, effective_classes_win

            # concentration diagnostics (NaN in summary row)
            float("nan"), float("nan"), float("nan"), float("nan"),  # topK_prob_mass_win
            float("nan"), float("nan"), float("nan"), float("nan"),  # topK_top1_mass_win

            # original peakedness (NaN in summary row)
            float("nan"),

            # head/tail metrics (NaN or could compute at corruption-level if you want later)
            float("nan"), float("nan"),
            float("nan"), float("nan"),

            acc_corr

        ])
        self._fh.flush()

    @torch.no_grad()
    def log_batch(
        self,
        *,
        phase: str,
        corruption: str,
        batch_idx: int,
        acc1: float,
        teacher_probs: torch.Tensor,   # [B,C]
    ):
        """
        Updates:
          (1) rolling window over last window_images (approx, to the nearest batch)
          (2) per-corruption aggregates
        Writes:
          a CSV row every log_every_k batches.
        """

        probs = teacher_probs.detach()
        B, C = probs.shape
        assert C == self.num_classes, f"teacher_probs C={C} != num_classes={self.num_classes}"

        # --- per-batch summary ---
        sum_probs_b = probs.sum(dim=0).to(dtype=torch.float64, device="cpu")  # [C]
        p_mean_b = (sum_probs_b / max(1, B))
        ent_b = self._entropy_from_probs(p_mean_b)
        eff_b = float(math.exp(ent_b))

        top1 = torch.argmax(probs, dim=1)  # [B]
        top1_counts_b = torch.bincount(top1.detach().cpu(), minlength=self.num_classes).to(torch.float64)  # [C]

        # --- update rolling window queue ---
        self._q.append((B, sum_probs_b, top1_counts_b))
        self._win_n += B
        self._win_sum_probs += sum_probs_b
        self._win_top1_counts += top1_counts_b

        # maintain approx window: drop whole oldest batches until <= window_images
        while self._win_n > self.window_images and len(self._q) > 0:
            n_old, sum_probs_old, top1_counts_old = self._q.popleft()
            self._win_n -= n_old
            self._win_sum_probs -= sum_probs_old
            self._win_top1_counts -= top1_counts_old

        # --- update per-corruption aggregates ---
        # (you should have called start_corruption(corruption) at beginning)
        self._corr_n += B
        self._corr_sum_probs += sum_probs_b
        self._corr_top1_counts += top1_counts_b
        self._corr_acc_sum += float(acc1) * B

        # --- compute window stats ---
        p_win = (self._win_sum_probs / max(1, self._win_n))
        ent_win = self._entropy_from_probs(p_win)
        eff_win = float(math.exp(ent_win))

        top1_frac_win = (self._win_top1_counts / max(1, self._win_n))
        top1_max_frac_win = float(top1_frac_win.max().item())
        # --- Top-K mass diagnostics (window) ---
        # A) probability-marginal concentration from p_win
        p_sorted, _ = torch.sort(p_win, descending=True)
        top1_prob_mass_win  = float(p_sorted[:1].sum().item())
        top5_prob_mass_win  = float(p_sorted[:5].sum().item())
        top10_prob_mass_win = float(p_sorted[:10].sum().item())
        top20_prob_mass_win = float(p_sorted[:20].sum().item())

        # B) argmax-histogram concentration from top1_frac_win
        f_sorted, _ = torch.sort(top1_frac_win, descending=True)
        top1_top1_mass_win  = float(f_sorted[:1].sum().item())   # == top1_max_frac_win
        top5_top1_mass_win  = float(f_sorted[:5].sum().item())
        top10_top1_mass_win = float(f_sorted[:10].sum().item())
        top20_top1_mass_win = float(f_sorted[:20].sum().item())


        # head/tail metrics (optional)
        if self.head_class_ids is not None:
            head_set = self.head_class_ids.detach().cpu().long()
            head_mask = torch.zeros(self.num_classes, dtype=torch.bool)
            head_mask[head_set] = True

            # top1 fractions in window
            top1_head_frac_win = float(top1_frac_win[head_mask].sum().item())
            top1_tail_frac_win = 1.0 - top1_head_frac_win

            # prob mass in window mean probs
            prob_head_mass_win = float(p_win[head_mask].sum().item())
            prob_tail_mass_win = 1.0 - prob_head_mass_win
        else:
            top1_head_frac_win = float("nan")
            top1_tail_frac_win = float("nan")
            prob_head_mass_win = float("nan")
            prob_tail_mass_win = float("nan")

        # --- write row sparsely ---
        write_now = (self.log_every_k <= 1) or ((batch_idx % self.log_every_k) == 0)
        if write_now:
            self._writer.writerow([
                self.step, phase, corruption, batch_idx,
                B, float(acc1),

                ent_b, eff_b,
                ent_win, eff_win,
                top1_prob_mass_win, top5_prob_mass_win, top10_prob_mass_win, top20_prob_mass_win,
                top1_top1_mass_win, top5_top1_mass_win, top10_top1_mass_win, top20_top1_mass_win,                
                top1_max_frac_win,

                top1_head_frac_win, top1_tail_frac_win,
                prob_head_mass_win, prob_tail_mass_win,

                float("nan")  # acc1_corruption filled only on summary row
            ])
            self._fh.flush()

        self.step += 1
