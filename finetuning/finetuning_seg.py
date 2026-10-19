"""
finetuning/finetuning_seg.py
============================
Fine-tunes ResNet-34, ViT-B/16, or DINOv2 for binary segmentation, with
optional spectral-norm regularization on the 1x1 convolution head.

Segmentation counterpart of ``finetuning_cls.py``.  Everything that made the
classification protocol defensible is kept identical, so the two sets of
results are comparable:

The two differences are the head and the metric.  The head is a 1x1
convolution, whose Lipschitz constant is still exactly ``sigma_max(W)`` (see
:mod:`shared.SegmentationModels`).  

Usage (see ``run_finetuning_seg.sh`` for the full sweep)::

    python -m finetuning.finetuning_seg \
        --dataset      kvasir_seg \
        --save_root    /path/to/output/run_name \
        --model_type   dinov2 \
        --freeze_mode  head+last \
        --lambda_sn    100 \
        --split_seed   0
"""

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.nn as nn
import argparse
import json
import os
import sys
import time
import torch
import wandb
import copy

# Ensure the project root is on sys.path when running the script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.DatasetRegistry import build_dataset, get_spec
from shared.metrics import (SMOOTH_FREE_CONVENTION, foreground_dice,
                            foreground_iou, nanmean_count)
from shared.ModelFactory import compute_spectral_norm
from shared.runtime import pin_cpu_cores
import shared.featcache as featcache
import shared.wandb_logging as wb
from shared.seed import reset_seed, seed_worker
from shared.SegmentationModels import (build_segmentation_model,
                                       set_trainable_layers_seg)
from shared.splits import load_or_make_split


# =============================================================================
# Argument parser
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Segmentation fine-tuning with spectral-norm "
                    "regularization on the head"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--dataset", type=str, default="kvasir_seg",
                   help="Registry key, see shared/DatasetRegistry.py.")
    p.add_argument("--save_root", type=str, required=True,
                   help="Directory where checkpoints and logs are written.")
    p.add_argument("--split_dir", type=str, default="splits",
                   help="Where the train/val/test index file is stored.")
    p.add_argument("--split_seed", type=int, default=0,
                   help="Vary this for independent repetitions.")

    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--device_id",   type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_cores",   type=int, default=0,
                   help="Pin the process to the last N CPU cores (0 = off).")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"))
    p.add_argument("--num_classes", type=int, default=2,
                   help="2 for binary segmentation (background and lesion).")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=100,
                   help="Upper bound on training length. Segmentation sets "
                        "here are small (95-720 training images), so they need "
                        "more passes than the classification ones.")
    p.add_argument("--batch_size",  type=int,   default=32,
                   help="Lower than classification on purpose: dense "
                        "prediction keeps a full feature grid plus upsampled "
                        "logits in memory for every sample.")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--optimizer",    type=str,   default="adam",
                   choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--momentum",     type=float, default=0.9,
                   help="Only used by SGD.")
    p.add_argument("--lambda_sn",   type=float, default=0.0,
                   help="Coefficient for the spectral-norm regularization.")
    p.add_argument("--dice_weight", type=float, default=1.0,
                   help="Weight of the soft-Dice term added to cross-entropy.")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_type", type=str, required=True,
                   choices=["resnet", "vit", "dinov2"])
    p.add_argument("--freeze_mode", type=str, required=True,
                   choices=["head", "head+last"])

    # ── Backbone feature cache ────────────────────────────────────────────────
    # Only ever used when freeze_mode == "head"; see shared/featcache.py for
    # why replaying cached features is the same arithmetic as recomputing them.
    p.add_argument("--featcache", type=str, default="auto",
                   choices=["auto", "off"],
                   help="'auto' caches the frozen backbone's features for "
                        "freeze_mode=head runs; 'off' always recomputes them "
                        "(use it to verify the two paths agree).")
    p.add_argument("--featcache_dir", type=str, default=None,
                   help="Where the cache lives. Defaults to $FEATCACHE_DIR, "
                        "then to featcache/ in the working directory.")

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="'offline' buffers to disk; sync later with "
                        "`wandb sync <dir>`.")

    return p


# Registry keys use torchvision-style names; the CLI keeps the short ones used
# by the classification pipeline, so the two sweeps look the same.
ARCH_MAP = {"resnet": "resnet34", "vit": "vit_b_16", "dinov2": "dinov2"}


# =============================================================================
# Optimizer
# =============================================================================
def build_optimizer(model, args):
    """Build the optimizer named by ``--optimizer`` over the trainable params."""
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer '{args.optimizer}'.")


# =============================================================================
# Loss and metrics
# =============================================================================
def dice_ce_loss(logits, target, dice_weight=1.0, num_classes=2):
    """
    Cross-entropy plus soft Dice.

    Cross-entropy alone is a poor objective here: the background covers most of
    the frame, so a model that predicts "background everywhere" already reaches
    a low loss.  The Dice term is computed on softmax probabilities (so it stays
    differentiable) and balances that.

    Args:
        logits      : ``(B, K, H, W)`` at full resolution.
        target      : ``(B, H, W)`` int64 class indices.
        dice_weight : Weight of the Dice term.
        num_classes : Number of classes.
    """
    ce = F.cross_entropy(logits, target)
    if dice_weight <= 0:
        return ce

    probs = logits.softmax(1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + 1.0) / (denom + 1.0)          # smoothed, per class
    return ce + dice_weight * (1.0 - dice.mean())


# =============================================================================
# One batch, with or without the feature cache
# =============================================================================
def iter_batches(loader, device, cache=None):
    """
    Yield ``(inputs, masks)`` on ``device``.

    Without a cache ``inputs`` are images and the loader hands them over
    directly.  With one, the loader hands over sample INDICES and the batch is
    gathered from the resident feature tensor, so ``inputs`` are the backbone
    feature grids that the images would have produced (see
    :mod:`shared.featcache`).
    """
    if cache is None:
        for images, masks in loader:
            yield (images.to(device, non_blocking=True),
                   masks.to(device, non_blocking=True))
        return

    feats, masks_all = cache
    for (idx,) in loader:
        idx = idx.to(feats.device, non_blocking=True)
        yield (feats[idx].to(device, non_blocking=True),
               masks_all[idx].to(device, non_blocking=True).long())


def forward_batch(model, inputs, size, using_cache: bool):
    """
    Full-resolution logits for one batch.

    With the cache the backbone is skipped and only the 1x1 convolution head
    plus the upsampling run -- which is exactly ``SegmentationModel.forward``
    minus a step whose output is already known, so the logits, and therefore
    the gradients reaching the head, are the same either way.
    """
    if using_cache:
        return model.head_from_features(inputs, size)
    return model(inputs)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=2, dice_weight=1.0, cache=None):
    """
    Run one evaluation pass and return a dict of metrics.

    Dice and IoU are computed **per image and then averaged**.  The per-image
    mean and a dataset-level pooled score differ a lot when lesion sizes vary
    by orders of magnitude: pooling is dominated by the few largest lesions,
    while the per-image mean is what the segmentation literature reports and
    what a clinician cares about.

    Both go through :mod:`shared.metrics`, which is now the only definition of
    Dice in the project.  This function used to add a ``1e-6`` smoothing term,
    which scored a lesion-free image on which the model predicted nothing as a
    perfect 1.0 and therefore inflated the reported Dice by the fraction of
    such frames.  The smoothing is gone; those images are undefined and
    excluded, and ``n_dice`` reports how many of ``n_images`` were actually
    scored so the mean can be read for what it is.
    """
    model.eval()
    # Accumulated on the GPU: a per-batch .item() (or .cpu()) is a
    # synchronisation point that stalls the CPU until the batch is finished,
    # which stops it from queueing the next one.  The numbers are identical,
    # they just cross to the host once, at the end.
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    n_batches = 0
    dices, ious, pixaccs = [], [], []

    for inputs, masks in iter_batches(loader, device, cache):
        logits = forward_batch(model, inputs, masks.shape[-2:], cache is not None)
        total_loss += dice_ce_loss(logits, masks, dice_weight,
                                   num_classes).detach().double()
        n_batches += 1

        pred = logits.argmax(1)
        pixaccs.append((pred == masks).float().mean(dim=(1, 2)))
        dices.append(foreground_dice(pred, masks))
        ious.append(foreground_iou(pred, masks))

    dice, n_dice, n_images = nanmean_count(dices)
    iou, n_iou, _ = nanmean_count(ious)

    return {
        "loss":      total_loss.item() / max(n_batches, 1),
        "dice":      dice,
        "iou":       iou,
        "pixel_acc": torch.cat(pixaccs).mean().item(),
        "n_dice":    n_dice,
        "n_iou":     n_iou,
        "n_images":  n_images,
    }


# =============================================================================
# Experiment log
# =============================================================================
def save_experiment_log(save_path, model_type, freeze_mode, args, curves,
                        selection, total_time_sec):
    """
    Save the full experiment log as JSON.

    Mirrors the classification log so that the same downstream readers work:
    a ``selection`` block with the epoch that was kept and the single test
    measurement, plus per-epoch ``training_curves``.  
    """
    os.makedirs(save_path, exist_ok=True)
    log = {
        "task":        "segmentation",
        "dataset":     args.dataset,
        "model_type":  model_type,
        "freeze_mode": freeze_mode,
        "hyperparameters": {
            "batch_size":  args.batch_size,
            "lr":           args.lr,
            "optimizer":     args.optimizer,
            "weight_decay":  args.weight_decay,
            "epochs":      args.epochs,
            "lambda_sn":   args.lambda_sn,
            "dice_weight": args.dice_weight,
            "input_size":  args.input_size,
            "num_classes": args.num_classes,
            "model_type":  args.model_type,
            "freeze_mode": args.freeze_mode,
            "split_seed":  args.split_seed,
            "split_dir":   args.split_dir,
        },
        "selection":       selection,
        "training_curves": curves,
        "timing": {"total_time_sec": float(total_time_sec)},
    }
    path = os.path.join(save_path, f"{model_type}_{freeze_mode}_log.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=4)
    print(f"[LOG SAVED] at {path}")


# =============================================================================
# Data loading
# =============================================================================
def build_dataset_and_split(args):
    """
    Build the dataset and its 70/10/20 split.

    The split is stratified by lesion size and persisted to disk, so that
    ``corruption_experiments_seg.py`` and ``drift_analysis_seg.py`` evaluate on
    exactly the same held-out images.

    Separate from the loaders so that ``train_model`` can build the MODEL in
    between: the feature cache needs the backbone to extract anything, so it
    has to run after the model exists but before the loaders do.  Neither this
    function nor ``shared.splits`` touches the global torch RNG -- the split
    draws from its own ``np.random.default_rng`` -- so moving it ahead of the
    model leaves the head initialisation and every shuffling order exactly
    where they were.
    """
    dataset = build_dataset(args.dataset, image_size=tuple(args.input_size))
    split_dir = os.path.join(args.split_dir, args.dataset)
    split = load_or_make_split(dataset, split_dir, args.split_seed,
                               fractions=(0.7, 0.1, 0.2))
    return dataset, split


def build_loaders_from_split(args, dataset, split):
    """Loaders that read images from disk: the original path, unchanged."""
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        # Torch seeds each worker's torch RNG but leaves numpy and random
        # untouched, so anything drawn from those inside __getitem__ would
        # depend on the worker count.  Nothing here does today; this keeps it
        # that way if augmentation is ever added.
        worker_init_fn=seed_worker,
    )

    # No `generator=` on the training loader on purpose: the shuffling order of
    # every reported run came from the global torch RNG seeded by
    # reset_seed(split_seed), and pinning a separate generator here would give
    # a different (equally valid, but different) order.

    train_ds = Subset(dataset, split["train"])
    val_ds   = Subset(dataset, split["val"])
    test_ds  = Subset(dataset, split["test"])

    # If a final batch contains only 1 image BatchNorm in train() will crashes.
    drop_last = (len(train_ds) % args.batch_size) == 1

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=drop_last, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_cached_loaders(args, dataset, split, model, device):
    """
    Path taken when ``freeze_mode == "head"``.

    The frozen backbone is run once over the whole dataset and its feature
    grids -- together with the masks, which are all that is still read from the
    images afterwards -- are kept resident (see :mod:`shared.featcache`).  The
    loaders returned here hand out sample indices; ``iter_batches`` turns each
    one into a batch of ``(features, mask)`` with a single gather, and the
    training loop then runs the 1x1 head on it.  No JPEG is decoded and no
    backbone forward or backward happens again for the rest of the run.

    Returns:
        ``(train_loader, val_loader, test_loader, cache)`` where ``cache`` is
        the ``(features, masks)`` pair the loaders index into.
    """
    feats, masks = featcache.features_seg(
        model, dataset, args.dataset, args.model_type, args.input_size, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        cache_dir=args.featcache_dir,
    )

    # Same drop_last rule as the image path, so that the two paths run the same
    # number of steps per epoch on the same data.
    drop_last = (len(split["train"]) % args.batch_size) == 1

    train_loader = featcache.index_loader(split["train"], args.batch_size,
                                          shuffle=True, drop_last=drop_last)
    val_loader = featcache.index_loader(split["val"], args.batch_size,
                                        shuffle=False)
    test_loader = featcache.index_loader(split["test"], args.batch_size,
                                         shuffle=False)
    return train_loader, val_loader, test_loader, (feats, masks)


# =============================================================================
# Training loop
# =============================================================================
def train_model(args, device):
    """
    Full pipeline: data -> model -> epochs -> checkpoint -> log.

    Model selection is based on validation Dice only.  The test split is
    evaluated once, after training and selection are complete.
    """
    # Dataset and split first, then the model, then the loaders.  The model has
    # to exist before the loaders because the cached path extracts features
    # with it; the global torch RNG is untouched by everything that moved, so
    # it is consumed in the same order as before (see build_dataset_and_split).
    dataset, split = build_dataset_and_split(args)

    model = build_segmentation_model(ARCH_MAP[args.model_type],
                                     args.num_classes, device=device)
    model.train()
    # Only sets requires_grad; the optimizer is built below from --lr /
    # --optimizer / --weight_decay so the CLI actually controls it.
    set_trainable_layers_seg(model, args.freeze_mode)
    optimizer = build_optimizer(model, args)

    cache = None
    if args.freeze_mode == "head" and args.featcache != "off":
        train_loader, val_loader, test_loader, cache = build_cached_loaders(
            args, dataset, split, model, device)
    else:
        train_loader, val_loader, test_loader = build_loaders_from_split(
            args, dataset, split)
    using_cache = cache is not None

    curves = {"train_loss": [], "train_loss_clean": [], "val_loss": [], "val_dice": [],
              "val_iou": [], "val_pixel_acc": [], "val_n_dice": [],
              "spectral_norm": []}
    best_val_dice, best_state, best_opt_state, best_epoch = -1.0, None, None, -1
    start_time = time.time()

    for epoch in range(args.epochs):

        # ── Train phase ───────────────────────────────────────────────────────
        model.train()

        # frezee also the BatchNorm also ResNet
        for m in model.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

        # Accumulators on the GPU, so the loop contains no .item(): each one is
        # a synchronisation point that blocks the CPU until the batch is done
        # and stops it from queueing the next one (twice per batch, all epoch
        # long).  float64 because that is what `x += loss.item()` was already
        # summing in -- a Python float is a double -- so the logged numbers are
        # not merely close, they are the same numbers; what changed is that
        # they cross to the host once per epoch.
        train_loss = torch.zeros((), dtype=torch.float64, device=device)
        train_loss_clean = torch.zeros((), dtype=torch.float64, device=device)
        n_batches = 0

        for inputs, masks in iter_batches(train_loader, device, cache):
            optimizer.zero_grad()
            logits = forward_batch(model, inputs, masks.shape[-2:], using_cache)
            loss = dice_ce_loss(logits, masks, args.dice_weight,
                                args.num_classes)
            clean_loss = loss.detach().double()  # dice+CE only, stays on the GPU

            # Spectral-norm penalty.  The SVD is only computed when the term is
            # active.  compute_spectral_norm flattens the (K, D, 1, 1) kernel
            # to (K, D), which is the matrix of the per-pixel linear map.
            # lambda_sn is applied as given: same coefficient as in
            # finetuning_cls.py, independent of the batch size.
            if args.lambda_sn > 0:
                sn_value = compute_spectral_norm(model)
                loss = loss + args.lambda_sn * sn_value

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().double()
            train_loss_clean += clean_loss
            n_batches += 1

        # One synchronisation per epoch, here.  The divisions happen on the
        # host, in double, exactly where they used to.
        curves["train_loss"].append(train_loss.item() / max(n_batches, 1))
        curves["train_loss_clean"].append(
            train_loss_clean.item() / max(n_batches, 1))
        curves["spectral_norm"].append(compute_spectral_norm(model).item())

        # ── Eval phase: validation only ───────────────────────────────────────
        val = evaluate(model, val_loader, device, args.num_classes,
                       args.dice_weight, cache)
        curves["val_loss"].append(val["loss"])
        curves["val_dice"].append(val["dice"])
        curves["val_iou"].append(val["iou"])
        curves["val_pixel_acc"].append(val["pixel_acc"])
        curves["val_n_dice"].append(val["n_dice"])

        # Keep the weights of the best epoch seen so far, by validation Dice
        if val["dice"] > best_val_dice:
            best_val_dice = val["dice"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_opt_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch

        if args.use_wandb:
            wandb.log({
                "epoch":         epoch,
                "train_loss":    curves["train_loss"][-1],
                "train_loss_clean": curves["train_loss_clean"][-1],
                "val_loss":      val["loss"],
                "val_dice":      val["dice"],
                "val_iou":       val["iou"],
                "spectral_norm": curves["spectral_norm"][-1],
            })

        print(f"[{epoch + 1:02d}/{args.epochs}]  "
              f"train_loss={curves['train_loss'][-1]:.4f}  "
              f"val_dice={val['dice']:.4f}  val_iou={val['iou']:.4f}  "
              f"sn={curves['spectral_norm'][-1]:.4f}")

    total_time = time.time() - start_time

    # ── Load the best-on-val weights, then evaluate on test once ─────────────
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, args.num_classes,
                    args.dice_weight, cache)
    selection = {
        "best_epoch":    best_epoch,
        "best_val_dice": best_val_dice,
        "test_dice":     test["dice"],
        "test_iou":      test["iou"],
        "test_pixel_acc": test["pixel_acc"],
        "test_loss":     test["loss"],
        # How many test images the Dice mean is actually over.  When these two
        # differ, the gap is the number of lesion-free frames on which the
        # model also predicted nothing: undefined, not perfect.  The paper
        # should quote n_test_dice next to the Dice.
        "n_test_dice":   test["n_dice"],
        "n_test_images": test["n_images"],
        "dice_convention": SMOOTH_FREE_CONVENTION,
    }
    print(f"[FINAL] best_epoch={best_epoch}  val_dice={best_val_dice:.4f}  "
          f"test_dice={test['dice']:.4f}  test_iou={test['iou']:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(args.save_root, exist_ok=True)
    ckpt_path = os.path.join(args.save_root,
                             f"{args.model_type}_{args.freeze_mode}.pth")
    torch.save({
        "model_state_dict":     best_state,
        "optimizer_state_dict": best_opt_state,
        "task":                 "segmentation",
        "arch":                 ARCH_MAP[args.model_type],
        "dataset":              args.dataset,
        "num_classes":          args.num_classes,
        "freeze_mode":          args.freeze_mode,
        "lambda_sn":            args.lambda_sn,
        "split_seed":           args.split_seed,
        **selection,
    }, ckpt_path)
    print(f"[CHECKPOINT SAVED] {ckpt_path}")

    save_experiment_log(args.save_root, args.model_type, args.freeze_mode,
                        args, curves, selection, total_time)

    if args.use_wandb:
            for k, v in selection.items():
                wandb.summary[k] = v
            wandb.summary["total_time_sec"] = total_time
            wandb.finish()

# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    args = _build_parser().parse_args()

    spec = get_spec(args.dataset)
    if spec.task != "segmentation":
        raise ValueError(f"Dataset '{args.dataset}' is a {spec.task} dataset. "
                         "Use finetuning_cls.py for classification.")

    device = torch.device(f"cuda:{args.device_id}"
                          if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_root, exist_ok=True)

    if args.use_wandb:
        wb.init(
            stage=wb.FINETUNING, task="segmentation",
            name=wb.run_label(args.dataset, args.model_type, args.freeze_mode,
                              args.lambda_sn, args.split_seed),
            config=vars(args), mode=args.wandb_mode,
        )

    pin_cpu_cores(args.pin_cores, torch_threads=2)
    # split_seed drives both the data split and the weight initialisation, so
    # --split_seed 0 1 2 gives three genuinely independent repetitions.  
    reset_seed(args.split_seed)
    print(f"Running: {args.dataset} | {args.model_type} | {args.freeze_mode} | "
          f"split_seed={args.split_seed} | device={device}")
    train_model(args, device)
