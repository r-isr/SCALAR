"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight.

Extended with optional SCALAR class-marginal balancing (prior_mode "uniform" / "correct").
With prior_mode "base" the update is identical to the original EATA.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import math
import torch.nn.functional as F


class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0,
                 steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05,
                 prior_mode="base", tau=0.3, window=1024, warmup=1024):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin  # hyper-parameter epsilon for cosine similarity thresholding (Eqn. 5)

        self.current_model_probs = None  # moving average of probability vector (Eqn. 4)

        self.fishers = fishers  # fisher regularizer items for anti-forgetting
        self.fisher_alpha = fisher_alpha  # trade-off beta for two losses (Eqn. 8)

        self.freeze_updates = False

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # ----- SCALAR class-marginal balancing (inactive when prior_mode == "base") -----
        if prior_mode not in {"base", "uniform", "correct"}:
            raise ValueError(f"EATA: unsupported prior_mode '{prior_mode}' "
                             f"(expected base / uniform / correct)")
        self.prior_mode = prior_mode
        self.cls_prior = None                 # target prior pi (None -> uniform); set by main for "correct"
        self.cls_queue_size = window          # FIFO window of logits of selected samples
        self.cls_queue_warmup_min = min(warmup, window)  # balancing starts once this many rows are available
        self.sink_tau = tau                   # softmax temperature before Sinkhorn
        self.sink_iters = 5                   # Sinkhorn iterations
        self.register_buffer("cls_queue", torch.empty(0))   # allocated on first use (needs C)
        self.register_buffer("cls_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("cls_queue_filled", torch.zeros(1, dtype=torch.long))

    def disable_adapt(self):
        self.freeze_updates = True

    def enable_adapt(self):
        self.freeze_updates = False

    def forward(self, x):
        if self.episodic:
            self.reset()

        if self.freeze_updates:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
            return outputs

        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(
                    self, x, self.model, self.optimizer, self.fishers, self.e_margin,
                    self.current_model_probs, fisher_alpha=self.fisher_alpha,
                    num_samples_update=self.num_samples_update_2, d_margin=self.d_margin
                )
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        if self.cls_queue.numel() > 0:
            self.cls_queue.zero_()
        self.cls_queue_ptr.zero_()
        self.cls_queue_filled.zero_()

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    @torch.no_grad()
    def _enqueue_logits(self, logits_sel):
        """Write the logits of the selected samples into the FIFO ring buffer."""
        B = logits_sel.shape[0]
        ptr = int(self.cls_queue_ptr.item())
        end = min(ptr + B, self.cls_queue_size)
        n1 = end - ptr
        if n1 > 0:
            self.cls_queue[ptr:end].copy_(logits_sel[:n1])
        n2 = B - n1
        if n2 > 0:
            self.cls_queue[0:n2].copy_(logits_sel[n1:])
        self.cls_queue_ptr[0] = (ptr + B) % self.cls_queue_size
        self.cls_queue_filled[0] = min(self.cls_queue_filled.item() + B, self.cls_queue_size)

    @torch.no_grad()
    def _sinkhorn_balance_class(self, logits_eff, pi=None):
        """Balance softmax(logits_eff / tau) so its column marginals approach N * pi (uniform if pi is None)."""
        Q = F.softmax(logits_eff / self.sink_tau, dim=1)
        N, C = Q.shape
        if pi is None:
            target_col = torch.full((1, C), float(N) / C, device=Q.device)
        else:
            pi = pi.view(1, C).to(Q)
            target_col = (pi * N).clamp_min(1e-12)

        eps = 1e-12
        for _ in range(self.sink_iters):
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
            col_sums = Q.sum(dim=0, keepdim=True) + eps
            Q = Q * (target_col / col_sums)

        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        return Q


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(eata_obj, x, model, optimizer, fishers, e_margin, current_model_probs,
                           fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    With SCALAR (prior_mode "uniform" / "correct"), the logits of the selected samples are
    balanced over a FIFO window and the loss mixes EATA's weighted entropy with the
    cross-entropy to the balanced targets (0.5 / 0.5).
    Return:
    1. model outputs;
    2. the number of reliable and non-redundant samples;
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model(x)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    entropys = entropys[filter_ids_1]
    logits_sel = outputs[filter_ids_1]
    # filter redundant samples
    if current_model_probs is not None:
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        logits_sel = logits_sel[filter_ids_2]
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    n_sel = logits_sel.size(0)

    # ---------- SCALAR: balanced targets for the selected samples ----------
    use_scalar = (eata_obj.prior_mode != "base") and n_sel > 0
    if use_scalar:
        with torch.no_grad():
            logits_batch = logits_sel.detach()
            if eata_obj.cls_queue.numel() == 0:
                eata_obj.cls_queue = logits_batch.new_zeros(eata_obj.cls_queue_size, logits_batch.size(1))
            filled = int(eata_obj.cls_queue_filled.item())
            logits_eff = torch.cat([eata_obj.cls_queue[:filled], logits_batch], dim=0) if filled > 0 else logits_batch
            q_bal = None
            if logits_eff.size(0) >= eata_obj.cls_queue_warmup_min:
                Q_eff = eata_obj._sinkhorn_balance_class(logits_eff, pi=eata_obj.cls_prior)
                q_bal = Q_eff[-n_sel:]  # last rows = current selected samples
        if q_bal is not None:
            bal_loss = -(q_bal * torch.log_softmax(logits_sel, dim=1)).sum(dim=1).mean()
            loss = 0.5 * loss + 0.5 * bal_loss

    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if n_sel != 0:
        loss.backward()
        optimizer.step()
        if use_scalar:
            eata_obj._enqueue_logits(logits_sel.detach())
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def collect_params(model):
    """Collect the affine scale + shift parameters from normalization layers.

    Walk the model's modules and collect normalization parameters.
    Supports both CNN-style BN naming and ViT-style norm naming.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if ("norm" in nm or "bn" in nm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with EATA on CNNs / ViTs."""
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if ("norm" in nm or "bn" in nm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    p.requires_grad_(True)

        if isinstance(m, nn.BatchNorm2d):
            # force use of batch stats in both train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model


def check_model(model):
    """Check model for compatibility with EATA."""
    is_training = model.training
    assert is_training, "EATA needs train mode: call model.train()"

    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    assert has_any_params, "EATA needs params to update: check which require grad"

    has_norm = any(("norm" in nm or "bn" in nm) for nm, _ in model.named_modules())
    assert has_norm, "EATA needs normalization layers for its optimization"
