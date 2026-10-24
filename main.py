import logging

import torch
import torch.optim as optim

from robustbench.data import load_imagenetc, load_cifar10c, load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import eata_vit
import scalar

from conf import cfg, load_cfg_fom_args

import vit_patch                    # <-- import FIRST

import operators
import numpy as np


from timm.models import load_checkpoint
from pathlib import Path
import os

import argparse

from typing import Optional

from stats import PseudoLabelStatsLogger
from calib import CalibrationMeter

import math

logger = logging.getLogger(__name__)

# ---- head–tail skew controls (hard-coded for reproducibility) ----
# The mode is chosen on the command line: --skew {none, fixed, dynamic}
SKEW_TAIL_COUNT  = 1             # number of samples to keep per tail class
SKEW_BASE_SEED   = 54321         # master seed for deterministic selection

# --skew fixed: same sizes for every corruption (head classes re-drawn per corruption)
SKEW_FIXED_HEAD       = 50       # number of head classes
SKEW_FIXED_HEAD_COUNT = 100      # samples per head class

# --skew dynamic: sizes change per corruption (hard jumps), cycling through the lists
SKEW_DYNAMIC_HEAD_SCHEDULE       = [30, 150, 20, 200, 40, 100]
SKEW_DYNAMIC_HEAD_COUNT_SCHEDULE = [170, 33, 250, 25, 125, 50]

NUM_ROUNDS = 1

def make_skew_a1(
    x, y,
    head: int,
    tail: int,
    base_seed: int,
    tag: str = "default",
    head_upsample: int = 100,
    head_ids: Optional[np.ndarray] = None, 
    num_classes: Optional[int] = None,
):
    """
    Build a strong head–tail skew on the RobustBench 5k split.

    If head_ids is given, it defines the exact head classes.
    Otherwise, a deterministic permutation (base_seed, tag) picks them.

    Returns:
        sel_idx:  np.ndarray[int64] indices into (x, y) after selection (duplicates allowed)
        pi_emp:   np.ndarray[float64] class prior over the SELECTED stream (length C)
        head_ids: np.ndarray[int64] the exact class IDs used as 'head'
    """
    y_np = y.detach().cpu().numpy().astype(np.int64)
    # Use the model's class count, not the max label present: on a partial split
    # (e.g. small NUM_EX) the top class ids may be missing, which would make pi
    # shorter than the logits. Identical to before whenever all classes appear.
    C = int(num_classes) if num_classes is not None else int(y_np.max()) + 1

    g = np.random.RandomState(abs(hash((base_seed, str(tag)))) % (2**31 - 1))

    if head_ids is None:
        all_ids = np.arange(C, dtype=np.int64)
        g.shuffle(all_ids)
        head_ids = all_ids[: min(head, C)]
    else:
        head_ids = np.asarray(head_ids, dtype=np.int64)
        # (optional) validate bounds
        head_ids = head_ids[(head_ids >= 0) & (head_ids < C)]
        # ensure at most 'head' classes
        if head_ids.size > head:
            head_ids = head_ids[:head]

    # tail = everything not in head_ids
    mask = np.ones(C, dtype=bool)
    mask[head_ids] = False
    tail_ids = np.nonzero(mask)[0]

    sel = []

    # HEAD: upsample to fixed count per head class (with replacement)
    for c in head_ids:
        idx = np.where(y_np == c)[0]
        if idx.size == 0:
            continue
        pick = g.choice(idx, size=head_upsample, replace=True)
        sel.extend(pick.tolist())

    # TAIL: keep exactly `tail` (no replacement)
    for c in tail_ids:
        idx = np.where(y_np == c)[0]
        if idx.size == 0:
            continue
        g.shuffle(idx)
        k = min(idx.size, tail)
        if k > 0:
            sel.extend(idx[:k].tolist())

    sel_idx = np.array(sel, dtype=np.int64)

    cnt = np.bincount(y_np[sel_idx], minlength=C).astype(np.float64)
    pi_emp = cnt / (cnt.sum() + 1e-12)

    return sel_idx, pi_emp, head_ids


def build_class_permutation(C: int, seed: int) -> np.ndarray:
    g = np.random.RandomState(seed)
    perm = np.arange(C, dtype=np.int64)
    g.shuffle(perm)
    return perm

# set model's class prior handle (DataParallel-safe)
def set_cls_prior(model, prior_tensor_or_none):
    try:
        model.module.cls_prior = prior_tensor_or_none
    except AttributeError:
        model.cls_prior = prior_tensor_or_none

def evaluate(description):
    args = load_cfg_fom_args(description)

    # ---- skew mode (from --skew) ----
    skew_enable = (args.skew != "none")
    if args.skew == "dynamic":
        SKEW_HEAD_SCHEDULE = SKEW_DYNAMIC_HEAD_SCHEDULE
        SKEW_HEAD_COUNT_SCHEDULE = SKEW_DYNAMIC_HEAD_COUNT_SCHEDULE
    else:  # "fixed" (unused when "none")
        SKEW_HEAD_SCHEDULE = [SKEW_FIXED_HEAD]
        SKEW_HEAD_COUNT_SCHEDULE = [SKEW_FIXED_HEAD_COUNT]
    print(f"[Skew] mode={args.skew}  head_schedule={SKEW_HEAD_SCHEDULE}  "
          f"count_schedule={SKEW_HEAD_COUNT_SCHEDULE}  tail={SKEW_TAIL_COUNT}")
    # ------------------------------------------------------------------
    # 1)  Delete cached RobustBench checkpoints in  <CKPT_DIR>/<dataset>/corruptions/*.pt
    #     (keeps the cache from growing across runs). Every deleted file is printed.
    # ------------------------------------------------------------------
    corrupt_dir = Path(cfg.CKPT_DIR) / cfg.CORRUPTION.DATASET / "corruptions"
    if corrupt_dir.exists():
        for f in corrupt_dir.glob("*.pt"):
            print(f"[ckpt-cleanup] deleting cached checkpoint: {f}")
            f.unlink()

    # ------------------------------------------------------------------
    # 2)  Prevent RobustBench from re-downloading anything
    # ------------------------------------------------------------------
    os.environ["ROBUSTBENCH_SKIP_DOWNLOAD"] = "1"

    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(args, base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "eata":
        logger.info("test-time adaptation: EATA")
        model = setup_eata(args, base_model)
    if cfg.MODEL.ADAPTATION == "scalar":
        logger.info("test-time adaptation: SCALAR")
        if args.data_set in ["cifar10", "cifar100"]:
            width = height = 384
        elif args.data_set == "imagenet":
            width = height = 224
        else:
            print("data set name error")
            exit()
        model = setup_scalar(args, base_model,width, height)
        
    print("[DEBUG] cfg.prior_mode:", args.prior_mode)

    print("[DEBUG] cfg.tau:", args.tau)

    print("[DEBUG] cfg.window:", args.window)
    
    # evaluate on each severity and type of corruption in turn
    # ------------------------------------------------------------------
    # ❷  pick the proper corruption loader
    # ------------------------------------------------------------------
    def get_corrupted_split(dataset_name, severity, corruption):
        if dataset_name == "imagenet":
            return load_imagenetc(cfg.CORRUPTION.NUM_EX, severity,
                                  cfg.DATA_DIR, False, [corruption])
        elif dataset_name == "cifar10":
            return load_cifar10c(cfg.CORRUPTION.NUM_EX, severity,
                                 cfg.DATA_DIR, False, [corruption])
        elif dataset_name == "cifar100":
            return load_cifar100c(cfg.CORRUPTION.NUM_EX, severity,
                                  cfg.DATA_DIR, False, [corruption])
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

    def get_clean_split(dataset_name):
        if dataset_name == "imagenet":
            return load_imagenetc(n_examples=cfg.CORRUPTION.NUM_EX, severity=1,
            data_dir=cfg.DATA_DIR, shuffle=False,
            corruptions=["clean"])
        elif dataset_name == "cifar10":
            return load_cifar10c(n_examples=cfg.CORRUPTION.NUM_EX, severity=1,
            data_dir=cfg.DATA_DIR, shuffle=False, corruptions=["jpeg_compression"])
        elif dataset_name == "cifar100":
            return load_cifar100c(n_examples=cfg.CORRUPTION.NUM_EX, severity=1,
            data_dir=cfg.DATA_DIR, shuffle=False, corruptions=["jpeg_compression"])
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

    # ------------------------------------------------------------------
    # ❸  evaluate
    # ------------------------------------------------------------------  

    width = height = 384 if args.data_set in {"cifar10", "cifar100"} else 224
    All_error = [[] for _ in range(NUM_ROUNDS)]
    clean_err_per_round = [None] * NUM_ROUNDS
    calib_meter = CalibrationMeter(n_bins=15)   # global calibration over the corruption stream

    # Clean-before (no adapt)
    x_clean, y_clean = get_clean_split(args.data_set)

    if args.data_set in {"cifar10", "cifar100"}:
                x_clean = torch.nn.functional.interpolate(
                    x_clean, size=(height, width),
                    mode='bilinear', align_corners=False
                )

    err_clean_pre = eval_one_split(model, x_clean, y_clean, adapt=False, tag="clean_pre")

    logger.info(f"clean error [pre]: {err_clean_pre:.2%}")
    print(f"pre - error % [clean]: {err_clean_pre:.2%}")

    # set up stats logger
    # Clean-before (no adapt)
    if args.data_set == "imagenet":
        num_classes = 1000
    elif args.data_set == "cifar10":
        num_classes = 10
    elif args.data_set == "cifar100":
        num_classes = 100

    # ---- pseudo-label stats logger: DISABLED (saves run time). ----
    # To re-enable, uncomment every block marked "[stats logger]" in this function.
    # [stats logger]
    # head_class_ids = None
    # run_tag = f"{cfg.MODEL.ADAPTATION}_skew-{args.skew}_{args.prior_mode}"
    # csv_path = f"pseudo_label_stats_{run_tag}.csv"
    # stats_logger = PseudoLabelStatsLogger(
    #     csv_path=csv_path,
    #     num_classes=num_classes,
    #     head_class_ids=head_class_ids,   # optional
    #     write_header=True,
    #     window_images=1024,   # last 1024 images (approx by batches)
    #     log_every_k=10,       # log every 10 batches (still updates window every batch)
    # )
    # stats_logger.open()
    # print(f"[Stats] writing to: {csv_path}")



    for round_idx in range(NUM_ROUNDS):
        for severity in cfg.CORRUPTION.SEVERITY:
            for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                global_slot = round_idx * len(cfg.CORRUPTION.TYPE) + i_c

                # episodic reset logic (unchanged)
                try:
                    if round_idx == 0 and i_c == 0:
                        model.reset()
                    else:
                        logger.warning("not resetting model")
                except:
                    logger.warning("not resetting model")

                # Corruption sequence (adapt on/off per your experiment)
                x_test, y_test = get_corrupted_split(args.data_set, severity, corruption_type)

            
                # Upsample CIFAR tensors once (ImageNet already 224)
                if args.data_set in {"cifar10", "cifar100"}:
                    x_test = torch.nn.functional.interpolate(
                        x_test, size=(height, width),
                        mode='bilinear', align_corners=False
                    )

                current_head_ids_t = None

                if skew_enable:
                    print(f"[SplitSize] pre-skew x_test={x_test.shape[0]} y_test={y_test.shape[0]}")
                    head_ids_ci = None  # random (per tag/seed) head set

                    # dynamic head per corruption (hard jumps)
                    head_ci = SKEW_HEAD_SCHEDULE[global_slot % len(SKEW_HEAD_SCHEDULE)]
                    head_ci_count = SKEW_HEAD_COUNT_SCHEDULE[global_slot % len(SKEW_HEAD_COUNT_SCHEDULE)]
                

                    tag = f"{args.data_set}|{corruption_type}|s{severity}|i_c{global_slot}|h{head_ci}"

                    print(f"[Skew] corruption {i_c}: head={head_ci}")

                    sel_idx, pi_np, head_ids = make_skew_a1(
                        x_test, y_test,
                        head=head_ci,
                        tail=SKEW_TAIL_COUNT,
                        base_seed=SKEW_BASE_SEED,
                        tag=tag,
                        head_upsample=head_ci_count,
                        head_ids=head_ids_ci,
                        num_classes=num_classes,
                    )


                    # make head ids available for window stats logging (CPU long tensor)
                    current_head_ids_t = torch.as_tensor(head_ids, dtype=torch.long, device="cpu")

                    # reindex tensors
                    sel_idx_t = torch.from_numpy(sel_idx).to(x_test.device)
                    x_test = x_test.index_select(0, sel_idx_t)
                    y_test = y_test.index_select(0, sel_idx_t)

                    # prior for "correct" mode
                    pi = torch.from_numpy(pi_np).to(x_test.device, dtype=torch.float32)
                    # AFTER: print post-skew size
                    print(f"[SplitSize] post-skew x_test={x_test.shape[0]} y_test={y_test.shape[0]}")                    
                else:
                    pi = None



                if args.prior_mode == "correct" and skew_enable:
                    set_cls_prior(model, pi)         # use global π each step (Correct)
                else:                                # "uniform" / "base"
                    set_cls_prior(model, None)       # uniform columns in Sinkhorn (unused in base)

                model.enable_adapt()
                model.train()

                B = cfg.TEST.BATCH_SIZE
                num_batches = math.ceil(len(x_test) / B)

                correct_total = 0.0
                n_total = 0

                # [stats logger]
                # stats_logger.head_class_ids = current_head_ids_t if skew_enable else None
                # stats_logger.start_corruption(f"{corruption_type}|s{severity}|r{round_idx}")

                for b in range(num_batches):
                    x_b = x_test[b*B:(b+1)*B].cuda()
                    y_b = y_test[b*B:(b+1)*B].cuda()

                    # triggers adaptation (because SCALAR.forward calls forward_and_adapt)
                    logits = model(x_b)

                    with torch.no_grad():
                        pred = logits.argmax(dim=1)
                        correct = (pred == y_b).float().sum().item()
                        bs = y_b.shape[0]

                        correct_total += correct
                        n_total += bs

                        # calibration (eval-only): uses the pre-adaptation logits + GT labels
                        calib_meter.update(logits, y_b)

                        # [stats logger]
                        # teacher_probs = getattr(model, "stats_target_probs", None)
                        # if teacher_probs is not None:
                        #     stats_logger.log_batch(
                        #         phase="corruption",
                        #         corruption=f"{corruption_type}|s{severity}|r{round_idx}",
                        #         batch_idx=b,
                        #         acc1=correct / bs,
                        #         teacher_probs=teacher_probs,
                        #     )

                # [stats logger]
                # stats_logger.end_corruption()

                acc = correct_total / n_total
                err = 1. - acc
                All_error[round_idx].append(err)
                logger.info(f"error % [r{round_idx}][{corruption_type}{severity}]: {err:.2%}")

                if skew_enable:
                    print(f"[A1] N={len(sel_idx)}  head={head_ci}  tail={SKEW_TAIL_COUNT}  "
                        f"pi_min={pi_np[pi_np>0].min():.6f}  pi_max={pi_np.max():.6f}")


        # ---- per-round clean checkpoint + summary (always) ----
        x_clean_r, y_clean_r = get_clean_split(args.data_set)
        if args.data_set in {"cifar10", "cifar100"}:
            x_clean_r = torch.nn.functional.interpolate(
                x_clean_r, size=(height, width),
                mode='bilinear', align_corners=False)
        err_r = eval_one_split(model, x_clean_r, y_clean_r, adapt=False, tag=f"clean_round{round_idx}")
        clean_err_per_round[round_idx] = err_r
        _round_errs = All_error[round_idx]
        _round_mean = (sum(_round_errs) / len(_round_errs)) if _round_errs else float("nan")
        logger.info(f"clean error [r{round_idx}]: {err_r:.2%}")
        logger.info(f"round summary [r{round_idx}]: err={_round_mean:.2%}  clean={err_r:.2%}")
        print(f"[Round {round_idx}] clean error: {err_r:.2%}")
        model.enable_adapt(); model.train()   # re-arm before next round
    # [stats logger]
    # stats_logger.close()

    # Clean-after (no adapt)
    x_clean2, y_clean2 = get_clean_split(args.data_set)
    if args.data_set in {"cifar10", "cifar100"}:
                x_clean2 = torch.nn.functional.interpolate(
                    x_clean2, size=(height, width),
                    mode='bilinear', align_corners=False
                )    
    err_clean_post = eval_one_split(model, x_clean2, y_clean2, adapt=False, tag="clean_post")
    logger.info(f"clean error [post]: {err_clean_post:.2%}")
    print(f"post - error % [clean]: {err_clean_post:.2%}")

    # ==== Calibration summary (eval-only; always emitted) ====
    _calib = calib_meter.compute()
    logger.info(
        f"calibration: top1_err={_calib['top1_err']:.2%}  "
        f"ECE={_calib['ece']:.4f}  NLL={_calib['nll']:.4f}  "
        f"(N={_calib['total_count']})"
    )

    # ==== OVERALL SUMMARY + collapse detection (always emitted, any NUM_ROUNDS) ====
    COLLAPSE_THR = 0.98   # per-corruption error threshold
    COLLAPSE_RUN = 2      # consecutive corruptions above threshold to "blow up"
    round_mean_err = [
        (sum(errs) / len(errs)) if errs else float("nan")
        for errs in All_error
    ]

    def _has_consecutive_blowup(errs, thr=COLLAPSE_THR, run=COLLAPSE_RUN):
        c = 0
        for e in errs:
            c = c + 1 if e > thr else 0
            if c >= run:
                return True
        return False

    blow_up = [_has_consecutive_blowup(errs) for errs in All_error]

    collapse_round = None
    for R in range(NUM_ROUNDS):
        if not blow_up[R]:
            continue
        if all(round_mean_err[r] > COLLAPSE_THR for r in range(R + 1, NUM_ROUNDS)):
            collapse_round = R
            break

    valid_means = [m for m in round_mean_err if m == m]  # drop nan
    overall_err = (sum(valid_means) / len(valid_means)) if valid_means else float("nan")
    valid_cleans = [c for c in clean_err_per_round if c is not None]
    overall_clean = (sum(valid_cleans) / len(valid_cleans)) if valid_cleans else float("nan")

    logger.info("==== OVERALL SUMMARY ====")
    for r in range(NUM_ROUNDS):
        cl = clean_err_per_round[r]
        cl_str = f"{cl:.2%}" if cl is not None else "n/a"
        logger.info(f"round {r}: err={round_mean_err[r]:.2%}  clean={cl_str}")
    logger.info(f"overall error (avg over rounds): {overall_err:.2%}")
    logger.info(f"overall clean (avg over rounds): {overall_clean:.2%}")
    logger.info(f"clean [pre]={err_clean_pre:.2%}   clean [post]={err_clean_post:.2%}")
    logger.info("collapse round: " + (f"r{collapse_round}" if collapse_round is not None else "none"))


def eval_one_split(model, x_test, y_test, adapt: bool, tag: str):
    try:
        model.enable_adapt() if adapt else model.disable_adapt()
    except AttributeError:
        pass
    (model.train() if adapt else model.eval())

    acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device='cuda')
    err = 1.0 - acc
    print(f"[Eval:{tag}] adapt={adapt}  acc={acc:.4f}  err={err:.4f}")
    return err

def setup_source(args, model):

    # --- optional checkpoint override -----------------
    if args.checkpoint:
        if args.checkpoint.endswith(".npz"):
            # timm converts the NPZ → state-dict internally
            load_checkpoint(model, args.checkpoint, strict=False)
            print(f"[SCALAR] loaded NPZ checkpoint {args.checkpoint}")
        else:
            state = torch.load(args.checkpoint, map_location='cpu')
            if 'model' in state:          # Lightning/EMA wrappers
                state = state['model']
            if next(iter(state)).startswith('module.'):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            print(f"[SCALAR] loaded Torch checkpoint {args.checkpoint}")
    # --------------------------------------------------
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_eata(args, model):
    """Set up EATA for ViT/CNN normalization parameters (prior_mode "base" = original EATA)."""
    model = eata_vit.configure_model(model)
    eata_vit.check_model(model)

    params, param_names = eata_vit.collect_params(model)
    optimizer = setup_optimizer(params)

    eata_model = eata_vit.EATA(
        model,
        optimizer,
        fishers=None,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        e_margin=math.log(1000) / 2 - 1,
        d_margin=0.05,
        prior_mode=args.prior_mode,
        tau=args.tau,
        window=args.window,
    )

    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    logger.info(f"prior_mode: %s  tau: %s  window: %s", args.prior_mode, args.tau, args.window)
    return eata_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_scalar(args, model, width, height):


    # --- optional checkpoint override -----------------
    if args.checkpoint:
        if args.checkpoint.endswith(".npz"):
            # timm converts the NPZ → state-dict internally
            load_checkpoint(model, args.checkpoint, strict=False)
            print(f"[SCALAR] loaded NPZ checkpoint {args.checkpoint}")
        else:
            state = torch.load(args.checkpoint, map_location='cpu')
            if 'model' in state:          # Lightning/EMA wrappers
                state = state['model']
            if next(iter(state)).startswith('module.'):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            print(f"[SCALAR] loaded Torch checkpoint {args.checkpoint}")
    # --------------------------------------------------

    model = scalar.configure_model(model)

    model_param, atten_param  = scalar.collect_params(model)

    optimizer = setup_optimizer_scalar(model_param, atten_param, cfg.OPTIM.LR)


    scalar_model = scalar.SCALAR(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           ema = cfg.OPTIM.EMA_WEIGH,
                           ema_atten = cfg.OPTIM.EMA_ATT,
                           width=width,
                           height=height,
                           prior_mode = args.prior_mode,
                           tau = args.tau,
                           window = args.window
                              )
    #logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return scalar_model

def setup_optimizer_scalar(params, params_atten, model_lr):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": model_lr},
            {"params": params_atten, "lr": model_lr}],
            lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": model_lr},
            {"params": params_atten, "lr": model_lr}],
            momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING, nesterov=cfg.OPTIM.NESTEROV,lr=1e-5,weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    evaluate('Imagenet-C evaluation.')
