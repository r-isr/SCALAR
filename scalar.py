from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import torchvision.transforms as transforms
import cotta_transforms as my_transforms

import time

from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

def _bytes_to_mb(x: int) -> float:
    return x / (1024.0 ** 2)

def get_tta_transforms(width, height, soft=False):
    """Augmentation for the student view (img_v). The teacher view is the raw input."""
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=[0, 0],
            translate=(1 / (width/32), 1 / (height/32)),
            scale=(0.98, 1.02),
            shear=None,
            interpolation=InterpolationMode.BILINEAR,
            fill=0
        ),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.RandomGrayscale(p=0.2),
    ])


def update_ema_variables(ema_model, model, alpha_teacher, alpha_atten):
    for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
        if ('proj' in name) or ('qkv' in name):
            ema_param.data[:] = alpha_atten * ema_param[:].data[:] + (1 - alpha_atten) * param[:].data[:]
        else:
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class SCALAR(nn.Module):


    def __init__(self, model, optimizer, steps=1, ema=0.99, ema_atten = 0.99,
     width=224, height=224, prior_mode="base", tau=0.3, window=1024):
        super().__init__()

        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SCALAR requires >= 1 step(s) to forward and update"
        # 2️ reconstruction head

        model.proj = nn.Linear(768, 256, bias=True)
        nn.init.trunc_normal_( model.proj.weight, std=0.02)
        nn.init.zeros_(model.proj.bias)
        self.model = model

        self.model_ema = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform_v = get_tta_transforms(width, height, soft=False)
        self.alpha_teacher = ema
        self.alpha_atten = ema_atten



        # ----- CLASS uniform prior -----
        # Sinkhorn params
        #Sets the target effective batch size for balancing. 
        #With tiny runtime batches (B=1–8), 
        #we accumulate past teacher logits to reach ~1024 rows for reliable batch marginals.
        self.cls_queue_size = window          
        self.cls_queue_warmup_min =  1024#1024#256    # start balancing at >= 256
        self.register_buffer("cls_queue", torch.empty(0))  # empty until we know C
        #A write pointer into the ring buffer
        self.register_buffer("cls_queue_ptr", torch.zeros(1, dtype=torch.long))
        #Tracks how many rows of the queue are actually filled
        self.register_buffer("cls_queue_filled", torch.zeros(1, dtype=torch.long))
        #Temperature to soften teacher logits before balancing
        self.sink_tau = tau#0.3
        #num of Sinkhorn alternations
        self.sink_iters = 5



        # Optional non-uniform class prior π (length C); when None → uniform
        self.cls_prior = None  # set from main when using "Correct" prior

        if prior_mode not in {"base", "uniform", "correct"}:
            raise ValueError(f"SCALAR: unsupported prior_mode '{prior_mode}' "
                             f"(expected base / uniform / correct)")
        self.prior_mode = prior_mode

        self.freeze_updates = False

        self.prior_source_weight = 0.5

        # stats (for analysis / logging)
        self.stats_target_probs = None

        self.enable_scalar = True   # master switch for Sinkhorn class-balancing
        self._acc_time = 0.0
        self._acc_steps = 0
        self._print_every = 500


    def disable_adapt(self):
        self.freeze_updates = True

    def enable_adapt(self):
        self.freeze_updates = False
    
    
    def forward(self, x):

        for _ in range(self.steps):

            if self._acc_steps == 0 and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()

            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

            dt = time.time() - t0

            self._acc_time += dt
            self._acc_steps += 1

            if self._acc_steps == self._print_every:
                avg = self._acc_time / self._acc_steps

                if torch.cuda.is_available():
                    alloc_peak = torch.cuda.max_memory_allocated()
                    reserv_peak = torch.cuda.max_memory_reserved()
                    alloc_peak_mb = _bytes_to_mb(alloc_peak)
                    reserv_peak_mb = _bytes_to_mb(reserv_peak)

                    print(  f"[SCALAR={self.enable_scalar}] "
                            f"avg step time over last {self._print_every}: {avg:.4f}s | "
                            f"peak_alloc={alloc_peak_mb:.1f}MB peak_reserved={reserv_peak_mb:.1f}MB"
                         )
                else:
                    print(  f"[SCALAR={self.enable_scalar}] "
                            f"avg step time over last {self._print_every}: {avg:.4f}s"
                    )

                self._acc_time = 0
                self._acc_steps = 0

        return outputs

    def _forward_logits(self, model, x):

        try:
            out = model(x, None, None, True)   # ViT path
            logits = out[0]
            extra = None
        except TypeError:
            out = model(x)                     # ConvNet path

            if isinstance(out, (tuple, list)):
                logits = out[0]
                extra  = out[1] if len(out) > 1 else None
            else:
                logits = out
                extra  = None
        return logits, extra

    def configure_for_tta_convnet(self,model):
        # Convnet-only helper: freeze all params, keep BN affine trainable.
        # A ViT (LayerNorm, no BatchNorm2d) would end up with NOTHING trainable,
        # so loss.backward() raises "does not require grad". Skip freezing when the
        # model has no BatchNorm2d and leave requires_grad as set by
        # configure_model/collect_params (i.e. all collected params trainable).
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
        if not has_bn:
            return model

        # freeze all
        for p in model.parameters():
            p.requires_grad_(False)

        # unfreeze BN affine only + freeze running stats
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # freezes running_mean/var updates
                if m.affine:
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)

        return model
    
    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optim):
        
        self.configure_for_tta_convnet(self.model)

        self.model_ema.eval()

        self.configure_for_tta_convnet(self.model_ema)

        # ── build two global views ──────────────────────────
        img_u = x 
        img_v = self.transform_v(x)

        
        # ── forward through student / teacher ──────────────
        logS_v, tokS_v = self._forward_logits(self.model, img_v) #self.model(img_v, None, None, True) #different view than teacher but unmasked

        with torch.no_grad():
            logT_u, tokT_u = self._forward_logits(self.model_ema, img_u) #self.model_ema(img_u, None, None, True)
            # teacher probabilities before Sinkhorn (used only by the stats logger / C_eff)
            self.stats_target_probs = torch.softmax(logT_u.detach(), dim=1)

        if self.freeze_updates:
             return logT_u   

        # ── projection to 256-d (already on correct device) ─────
        proj_S, proj_T = self.model.proj.to(device=x.device), self.model_ema.proj.to(device=x.device)


        # ── cross-view logits consistency (teacher ↔ student) ──     
        L_cons = (softmax_entropy(logS_v, logT_u.detach() ).mean())


        # ---- build effective batch for balancing ----
        with torch.no_grad():
            #Grab shapes and current queue fill
            B, C = logT_u.shape
            if self.cls_queue.numel() == 0:
                self.cls_queue = torch.zeros(self.cls_queue_size, C, device=logT_u.device)
            M = int(self.cls_queue_filled.item())
            #Build the effective matrix for balancing
            if M > 0:
                logits_eff = torch.cat([logT_u.detach(), self.cls_queue[:M]], dim=0)  # [(B+M), C]
            else:
                logits_eff = logT_u.detach()

            #How many total rows we’ll balance over
            eff_rows = logits_eff.shape[0]

                   
        #Only enable balancing when we’ve reached the warm-up threshold (e.g., ≥256 rows).
        #Before that, skip (too noisy).
        use_balancing = (self.enable_scalar and eff_rows >= self.cls_queue_warmup_min
                         and self.prior_mode in {"uniform", "correct"})

        if use_balancing:
            # existing modes: uniform (pi=None) or external π in self.cls_prior
            Q_eff = self._sinkhorn_balance_class(
                logits_eff, tau=self.sink_tau,
                iters=self.sink_iters,
                pi=self.cls_prior
            )

            q_bal = Q_eff[:B]  # [B, C] for current batch

            log_p_s = F.log_softmax(logS_v, dim=-1)                  # [B, C]
            L_bal = -(q_bal * log_p_s).sum(dim=-1).mean()
        else:
            q_bal = None
            L_bal = torch.zeros((), device=logS_v.device)


        if self.prior_mode in {"uniform", "correct"}:
            loss =  (1-self.prior_source_weight) * L_cons + self.prior_source_weight * L_bal #0.7,0.3
        else:  # "base"
            loss =  L_cons

        # ---- enqueue for next steps ----
        with torch.no_grad():
            if self.enable_scalar and self.prior_mode in {"uniform", "correct"}:
                self._enqueue_teacher_logits(logT_u.detach())

        # update ────────────────────────────
        loss.backward()
        optim.step()
        optim.zero_grad()

        # ── EMA update ─────────────────────────────────────
        update_ema_variables(self.model_ema, self.model,
                             self.alpha_teacher, self.alpha_atten)

        return logT_u


    #Convenience function to push current step’s teacher(clean) logits into the queue
    def _enqueue_teacher_logits(self, logits_T_clean): 
        #Batch size and number of classes
        B, C = logits_T_clean.shape
        #Current write index in the ring buffer
        ptr = int(self.cls_queue_ptr.item())
        #Compute the non-wrapping end index
        end = min(ptr + B, self.cls_queue_size)
        # How many rows we can write contiguously from ptr to end.
        n1 = end - ptr
        if n1 > 0:
            self.cls_queue[ptr:end].copy_(logits_T_clean[:n1])
        # If we ran past the end, wrap around and write the remaining rows at the beginning of the buffer.
        n2 = B - n1
        if n2 > 0:
            self.cls_queue[0:n2].copy_(logits_T_clean[n1:])
        # Advance the ring pointer, wrapping modulo the queue size
        new_ptr = (ptr + B) % self.cls_queue_size
        self.cls_queue_ptr[0] = new_ptr
        #Increase the filled count but cap at the queue capacity
        self.cls_queue_filled[0] = min(self.cls_queue_filled.item() + B, self.cls_queue_size)

    @torch.no_grad()
    def _sinkhorn_balance_class(self, logits_eff, tau=None, iters=None, pi=None):
        """
        Runs Sinkhorn on the effective matrix of teacher(clean) logits to produce balanced soft targets
        logits_eff: [(B+M), C] teacher(clean) logits for current batch + queue
        returns balanced probs Q_bal: same shape, rows sum to 1, columns ~ uniform
        """
        tau = self.sink_tau if tau is None else tau
        iters = self.sink_iters if iters is None else iters

        #Turn logits into probabilities (soft codes) with temperature τ
        Q = F.softmax(logits_eff / tau, dim=1)  # [(B+M), C], rows sum ~1

        # We want each column sum ≈ N/C (uniform class prior across the effective batch).
        N, C = Q.shape

        if pi is None:
            # We want each column sum ≈ N/C (uniform class prior across the effective batch).
            target_col = torch.full((1, C), float(N)/C, device=Q.device)
        else:
            # non-uniform prior: columns sum to N·π_c
            pi = pi.view(1, C).to(Q)                       # sums to 1
            target_col = (pi * N).clamp_min(1e-12)         # [1, C]        

        #Alternating row/column scaling to satisfy both constraints
        eps = 1e-12
        for _ in range(iters):
            # Row normalization: ensure each row is a proper probability vector
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
            # Column normalization: rescale columns so each column total moves toward target_col (= N/C)
            col_sums = Q.sum(dim=0, keepdim=True) + eps  # [1, C]
            scale = (target_col / col_sums)              # [1, C]
            Q = Q * scale

        # one final row-normalize (keeps rows exactly probs)
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)


        return Q
            


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)



def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    model_params_list = []
    atten_param_list = []

    for name, param in model.named_parameters():
        if ('proj' in name) or ('qkv' in name) :
            atten_param_list.append(param)
        else:
            model_params_list.append(param)

    return model_params_list, atten_param_list


def copy_model_and_optimizer(model, optimizer):
    """Return a detached deep copy of the model (the EMA teacher)."""
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return ema_model


def configure_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cpu()


    # 5️  single DataParallel wrap – now mask_token is replicated, too
    model = torch.nn.DataParallel(model).to(device).train()
    return model
