"""
finetuning/finetuning_cls.py
========================
Fine-tunes ResNet-34, ViT-B/16, or DINOv2 on medical datasets with optional 
spectral-norm regularization on the classification head.

Two fine-tuning strategies are supported:

* ``head``      : only the linear classification head is updated.
* ``head+last`` : the head **plus** the last backbone stage are updated
                  (``layer4`` for ResNet-34; last 4 encoder blocks for ViT
                  and DINOv2).

The regularization term adds ``λ * σ_max(W_head)`` to the cross-entropy loss,
where ``σ_max`` is the largest singular value (spectral norm) of the head
weight matrix.  The coefficient is applied as given: the cross-entropy is
already a per-sample mean and the penalty depends only on the weights, so the
objective is identical at any batch size and ``λ`` means the same thing in
``finetuning_seg.py``.


Data splitting and model selection
----------------------------------
The dataset is partitioned 70/10/20 into train/val/test by `shared.splits`, and the 
indices are written to disk so that every other script in the project evaluates on 
the same images. Only the validation split is used during training: it selects the 
epoch whose weights are saved. The test split is evaluated exactly once, at the end.

Datasets are referred to by registry key (see ``shared/DatasetRegistry.py``), not by path.
``--num_classes`` is read from the dataset itself, which matters for Hyper-Kvasir and Kvasir-Capsule 
where the label set is discovered from the folder tree rather than known in advance.

Outputs (written to ``--save_root``):

* ``<model>_<freeze>.pth``      — model + optimizer checkpoint.
* ``<model>_<freeze>_log.json`` — full training/validation curves and hyperparameters.
* ``log.txt``                   — stdout captured by ``run_finetuning_cls.sh``.

Usage (see ``run_finetuning_cls.sh`` for the full sweep)::

    python -m finetuning.finetuning_cls \
        --dataset      kvasir_v2 \
        --save_root    /path/to/output/run_name   \
        --model_type   resnet \
        --freeze_mode  head   \
        --lambda_sn    10.0   \
        --split_seed   0
"""

import argparse
import json
import os
import time
import sys
import copy
import torch
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

from shared.DatasetRegistry import build_dataset, get_spec
from shared.DINOv2Wrapper import DINOv2Wrapper
from shared.ModelFactory import get_head, compute_spectral_norm
from shared.runtime import pin_cpu_cores
import shared.featcache as featcache
import shared.wandb_logging as wb
from shared.seed import reset_seed, seed_worker
from shared.splits import load_or_make_split

# =============================================================================
# Argument parser
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="Finetuning on a particular dataset with spectral-norm regularization on the head"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--dataset", type=str, default="kvasir_v2",
                   help="Registry key, see shared/DatasetRegistry.py "
                        "(kvasir_v2, hyperkvasir, kvasir_capsule, ...).")
    p.add_argument("--save_root", type=str, required=True,
                   help="Directory where checkpoints and logs are written.")

    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--device_id",   type=int, default=0,
                   help="CUDA device index (ignored if no GPU is available).")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of DataLoader worker processes.")
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="'offline' buffers to disk; sync later with "
                        "`wandb sync <dir>`.")
    p.add_argument("--pin_cores", type=int, default=0,
                   help="Pin the process to the last N CPU cores (0 = off).")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"),
                   help="Spatial size to resize images to.")
    p.add_argument("--num_classes", type=int, default=None,
                   help="Number of output classes. Left unset it is read from "
                        "the dataset, which is required for Hyper-Kvasir and "
                        "Kvasir-Capsule whose label set is discovered at load "
                        "time. Pass a value only to override that.")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap the number of images per class. Hyper-Kvasir and "
                        "Kvasir-Capsule are heavily imbalanced (some classes "
                        "have tens of thousands of frames, others a few "
                        "dozen); capping keeps a couple of categories from "
                        "dominating the loss.")
    p.add_argument("--min_per_class", type=int, default=20,
                   help="Drop classes with fewer images than this: a class "
                        "with a handful of examples cannot be split 70/10/20.")

    p.add_argument("--split_dir", type=str, default="splits",
                   help="Where the train/val/test index file is stored.")
    p.add_argument("--split_seed", type=int, default=0,
                   help="Vary this for independent repetitions.")


    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",     type=int,   default=50,
                   help="Upper bound on training length. The checkpoint kept "
                        "is the best epoch on validation.")
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",  type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9,
                   help="Only used by SGD.")
    p.add_argument("--lambda_sn",  type=float, default=0.0,
                   help="Coefficient λ for the spectral-norm regularization term.")

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

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model_type", type=str, required=True,
                   choices=["resnet", "vit", "dinov2"],
                   help="Backbone architecture.")
    p.add_argument("--freeze_mode", type=str, required=True,
                   choices=["head", "head+last"],
                   help="Which parameters to keep trainable.")

    return p



# ===============================
# Experiment log saving function
# ===============================
def save_experiment_log(
    save_path: str,
    model_type: str,
    freeze_mode: str,
    args: argparse.Namespace,
    train_losses: list[float],
    val_losses: list[float],
    train_accuracies: list[float],
    train_ce: list[float],
    val_accuracies: list[float],
    spectral_norms: list[float],
    total_time_sec: float,
    best_epoch: int,
    best_val_acc: float,
    test_acc: float,
    test_loss: float,
):
    """
    Save a complete experiment log as JSON for post-hoc analysis.

    The file includes hyperparameters, per-epoch loss/accuracy curves, the
    spectral-norm trajectory, and wall-clock training time.  Saved to::

        <save_path>/<model_type>_<freeze_mode>_log.json

    Args:
        save_path        : Directory where the JSON file is written.
        model_type       : Architecture key (``"resnet"``, ``"vit"``, or ``"dinov2"``).
        freeze_mode      : Fine-tuning strategy key.
        args             : Parsed command-line arguments (for hyperparameter recording).
        train_losses     : Per-epoch training loss values.
        train_ce         : Per-epoch training pure CE loss values.
        val_losses       : Per-epoch validation loss values.
        train_accuracies : Per-epoch training accuracy values.
        val_accuracies   : Per-epoch validation accuracy values.
        spectral_norms   : Per-epoch spectral norm of the head weight matrix.
        total_time_sec   : Total wall-clock training time in seconds.
        best_epoch       : Index of the epoch selected on the validation split.
        best_val_acc     : Validation accuracy at ``best_epoch``.
        test_acc         : Test accuracy of the selected model, measured once.
        test_loss        : Test loss of the selected model, measured once.
    """

    os.makedirs(save_path, exist_ok=True)

    # Structure of the log file
    log = {
        "task":        "classification",
        "dataset":     args.dataset,
        "model_type":  model_type,
        "freeze_mode": freeze_mode,
        "hyperparameters": {
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "optimizer":     args.optimizer,
            "weight_decay":  args.weight_decay,
            "epochs":       args.epochs,
            "lambda_sn":    args.lambda_sn,
            "input_size":   args.input_size,
            "num_classes":  args.num_classes,
            "model_type":   args.model_type,
            "freeze_mode":  args.freeze_mode,
            "num_workers":  args.num_workers,
            "split_seed":   args.split_seed,
            "split_dir":    args.split_dir,
        },

        # Model selection: which epoch was kept, and the single test measurement
        "selection": {
            "best_epoch":   int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "test_acc":     float(test_acc),
            "test_loss":    float(test_loss),
        },

        # Training and evaluation metrics over epochs
        "training_curves": {
            "train_loss":    train_losses,
            "train_ce":      train_ce,
            "val_loss":      val_losses,
            "train_acc":     train_accuracies,
            "val_acc":       val_accuracies,
            "spectral_norm": spectral_norms,
        },

        # Training time information
        "timing": {
            "total_time_sec": float(total_time_sec),
        },
    }

    # File name for saving the experiment log
    file_path = os.path.join(
        save_path,
        f"{model_type}_{freeze_mode}_log.json"
    )

    # Write JSON log to disk
    with open(file_path, "w") as f:
        json.dump(log, f, indent=4)

    print(f"[LOG SAVED] at {file_path}")






# =============================================================================
# Model initialisation and parameter freezing
# =============================================================================
def build_optimizer(model: nn.Module,
                    args: argparse.Namespace) -> torch.optim.Optimizer:
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


def set_trainable_layers(
    model_type: str,
    num_classes: int,
    freeze_mode: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """
    Build a pretrained backbone, replace its head, and selectively unfreeze
    parameters according to ``freeze_mode``.

    Freeze modes:

    * ``"head"``      : only the classification head is trainable.
    * ``"head+last"`` : head **plus** last backbone stage are trainable
                        (``layer4`` for ResNet-34; last 4 encoder blocks
                        for ViT-B/16 and DINOv2).

    Args:
        model_type  : Architecture key — ``"resnet"``, ``"vit"``, or ``"dinov2"``.
        num_classes : Number of target output classes.
        freeze_mode : Parameter-freezing strategy.
        device      : Target device.

    Returns:
        Tuple ``(model, optimizer)`` where ``model`` is on ``device`` and
        ``optimizer`` is the optimizer named by ``--optimizer`` (lr ``--lr``)
        over all trainable parameters.

    Raises:
        ValueError : If ``model_type`` or ``freeze_mode`` is not recognised.
    """
    # ── Build backbone with fresh head ────────────────────────────────────────
    if model_type == "resnet":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif model_type == "vit":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)

    elif model_type == "dinov2":
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
        )
        m = DINOv2Wrapper(backbone, num_classes)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose 'resnet', 'vit', or 'dinov2'."
        )

    m = m.to(device)

    # Freeze all parameters initially
    for p in m.parameters():
        p.requires_grad = False

    # Always unfreeze the classification head
    for p in get_head(m).parameters():
        p.requires_grad = True

    # Optionally unfreeze the last backbone stage
    if freeze_mode == "head+last":
        if model_type == "resnet":
            for p in m.layer4.parameters():
                p.requires_grad = True

        elif model_type == "vit":
            n = len(m.encoder.layers)
            for i in range(n - 4, n):
                for p in m.encoder.layers[i].parameters():
                    p.requires_grad = True
            for p in m.encoder.ln.parameters():
                p.requires_grad = True

        elif model_type == "dinov2":
            for block in m.backbone.blocks[-4:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in m.backbone.norm.parameters():
                p.requires_grad = True

    elif freeze_mode != "head":
        raise ValueError(
            f"Unknown freeze_mode '{freeze_mode}'. "
            "Choose 'head' or 'head+last'."
        )

    # Optimizer over all trainable parameters
    optimizer = build_optimizer(m, args)

    return m, optimizer



# =============================================================================
# Data loading
# =============================================================================
def build_dataset_and_split(args: argparse.Namespace):
    """
    Build the dataset and its 70/10/20 split, filling in ``args.num_classes``
    when it was left unset.

    The dataset comes from `shared.DatasetRegistry`, the same registry
    used by ``finetuning_seg.py`` and by the corruption / drift scripts, so a
    given key always resolves to the same folders and the same class-discovery
    rules everywhere. The dataset is split 70/10/20 (train/val/test), stratified
    by class, and the indices are persisted by `shared.splits` under a per-dataset
    subfolder so that different datasets never overwrite each other's split.

    Separate from the loaders so that ``train_model`` can build the MODEL in
    between: the feature cache needs the backbone to extract anything, so it
    has to run after the model exists but before the loaders do.  Neither this
    function nor ``shared.splits`` touches the global torch RNG -- the split
    draws from its own ``np.random.default_rng`` -- so moving it ahead of the
    model leaves the head initialisation and every shuffling order exactly
    where they were.

    Args:
        args : Parsed command-line arguments.  ``args.num_classes`` is filled
               in from the dataset when it was left unset.

    Returns:
        ``(dataset, split)`` where ``split`` has keys ``train``/``val``/``test``.
    """
    dataset = build_dataset(
        args.dataset,
        image_size=tuple(args.input_size),
        max_per_class=args.max_per_class,
        min_per_class=args.min_per_class,
    )

    # The number of classes is a property of the data, not a CLI choice;
    discovered = getattr(dataset, "num_classes", None)
    if args.num_classes is None:
        args.num_classes = discovered
        print(f"  [{args.dataset}] num_classes = {discovered} (from the dataset)")
    elif discovered is not None and args.num_classes != discovered:
        raise ValueError(
            f"--num_classes={args.num_classes} but '{args.dataset}' has "
            f"{discovered} classes. Drop the flag to use the discovered value."
        )

    # Three-way split, stratified, written to disk so that every script
    # (finetuning, drift_analysis, corruption_experiments) reads exactly the
    # same indices instead of re-deriving them from a seed.  One subfolder per
    # dataset, mirroring finetuning_seg.py.
    split = load_or_make_split(dataset,
                               os.path.join(args.split_dir, args.dataset),
                               args.split_seed, fractions=(0.7, 0.1, 0.2))

    return dataset, split


def build_loaders_from_split(
    args: argparse.Namespace, dataset, split,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders that read images from disk: the original path, unchanged."""
    train_ds = Subset(dataset, split["train"])
    val_ds   = Subset(dataset, split["val"])
    test_ds  = Subset(dataset, split["test"])

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

    # If a final batch contains only 1 image BatchNorm in train() will crashes.
    drop_last = (len(train_ds) % args.batch_size) == 1

    # Shuffling matters only while optimising; evaluation order is irrelevant
    # to the averages and staying deterministic keeps runs reproducible.
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=drop_last, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def build_cached_loaders(args: argparse.Namespace, dataset, split,
                         model: nn.Module, device: torch.device):
    """
    Path taken when ``freeze_mode == "head"``.

    The frozen backbone is run once over the whole dataset and its pre-head
    activations are kept resident (see :mod:`shared.featcache`).  The loaders
    returned here hand out sample indices; ``iter_batches`` turns each one into
    a batch of ``(feature, label)`` with a single gather, and the training loop
    then runs the linear head on it.  No JPEG is decoded and no backbone
    forward or backward happens again for the rest of the run.

    Returns:
        ``(train_loader, val_loader, test_loader, cache)`` where ``cache`` is
        the ``(features, labels)`` pair the loaders index into.
    """
    feats = featcache.features_cls(
        model, dataset, args.dataset, args.model_type, args.input_size, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        cache_dir=args.featcache_dir,
    )
    labels = torch.as_tensor(dataset.labels, dtype=torch.long,
                             device=feats.device)

    # Same drop_last rule as the image path, so that the two paths run the same
    # number of steps per epoch on the same data.
    drop_last = (len(split["train"]) % args.batch_size) == 1

    train_loader = featcache.index_loader(split["train"], args.batch_size,
                                          shuffle=True, drop_last=drop_last)
    val_loader   = featcache.index_loader(split["val"], args.batch_size,
                                          shuffle=False)
    test_loader  = featcache.index_loader(split["test"], args.batch_size,
                                          shuffle=False)
    return train_loader, val_loader, test_loader, (feats, labels)


# =============================================================================
# One batch, with or without the feature cache
# =============================================================================
def iter_batches(loader, device, cache=None):
    """
    Yield ``(inputs, labels)`` on ``device``.

    Without a cache ``inputs`` are images and the loader hands them over
    directly.  With one, the loader hands over sample INDICES and the batch is
    gathered from the resident feature tensor, so ``inputs`` are the pre-head
    activations the images would have produced (see :mod:`shared.featcache`).
    """
    if cache is None:
        for images, labels in loader:
            yield (images.to(device, non_blocking=True),
                   labels.to(device, non_blocking=True))
        return

    feats, labels_all = cache
    for (idx,) in loader:
        idx = idx.to(feats.device, non_blocking=True)
        yield (feats[idx].to(device, non_blocking=True),
               labels_all[idx].to(device, non_blocking=True))


def forward_batch(model: nn.Module, inputs: torch.Tensor, using_cache: bool):
    """
    Logits for one batch.

    With the cache only the head runs.  ``get_head(model)`` is the very same
    ``nn.Linear`` that ``compute_spectral_norm`` regularises and that
    ``model(images)`` would have ended on, so the logits -- and therefore the
    gradients reaching the head -- are the same either way.
    """
    if using_cache:
        return get_head(model)(inputs)
    return model(inputs)


# =============================================================================
# Training loop
# =============================================================================
def train_model(model_type: str, freeze_mode: str, args: argparse.Namespace, device: torch.device) -> None:
    """
    Full training pipeline: data loading -> model init -> training epochs ->
    checkpoint saving -> log saving.

    Model selection is based on the validation split only.  The test split is
    evaluated exactly once, after training and model selection are complete.
    """
    # Dataset and split first, then the model, then the loaders.  The model has
    # to exist before the loaders because the cached path extracts features
    # with it; the global torch RNG is untouched by everything that moved, so
    # it is consumed in the same order as before (see build_dataset_and_split).
    dataset, split = build_dataset_and_split(args)
    model, optimizer = set_trainable_layers(model_type, args.num_classes, freeze_mode, device, args)

    cache = None
    if freeze_mode == "head" and args.featcache != "off":
        train_loader, val_loader, test_loader, cache = build_cached_loaders(
            args, dataset, split, model, device)
    else:
        train_loader, val_loader, test_loader = build_loaders_from_split(
            args, dataset, split)
    using_cache = cache is not None

    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses         = [], []
    train_accuracies, val_accuracies = [], []
    train_ce                         = []
    spectral_norms                   = []

    best_val_acc = -1.0
    best_state   = None
    best_opt_state = None
    best_epoch   = -1

    start_time = time.time()

    for epoch in range(args.epochs):

        # ── Train phase ─────────────────────────────────────────────────────
        model.train()

        # frezee also the BatchNorm also ResNet
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

        # Accumulators on the GPU, so the loop contains no .item(): each one is
        # a synchronisation point that blocks the CPU until the batch is done
        # and stops it from queueing the next one (three stalls per batch, all
        # epoch long).  float64 for the losses and int64 for the count, which
        # is exactly what `x += loss.item()` and `n += (...).sum().item()` were
        # accumulating in before -- a Python float is a double.  The logged
        # numbers are therefore not merely close, they are the same numbers;
        # what changed is that they cross to the host once per epoch instead of
        # three times per batch.
        train_loss   = torch.zeros((), dtype=torch.float64, device=device)
        train_ce_sum = torch.zeros((), dtype=torch.float64, device=device)
        correct      = torch.zeros((), dtype=torch.long, device=device)
        total        = 0

        for inputs, labels in iter_batches(train_loader, device, cache):
            optimizer.zero_grad()
            outputs = forward_batch(model, inputs, using_cache)
            loss    = criterion(outputs, labels)
            ce_value = loss.detach().double()   # pure CE, stays on the GPU

            # Spectral-norm penalty.  The SVD is only needed when the term is
            # actually active; with lambda_sn = 0 it was pure overhead.
            # lambda_sn is applied as given, so every batch (including a
            # smaller final one) sees exactly the same coefficient.
            if args.lambda_sn > 0:
                sn_value = compute_spectral_norm(model)
                loss = loss + args.lambda_sn * sn_value

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().double()
            train_ce_sum += ce_value
            preds    = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum()
            total   += labels.size(0)

        # One synchronisation per epoch, here.  The divisions happen on the
        # host, in double, exactly where they used to.
        train_losses.append(train_loss.item() / len(train_loader))
        train_accuracies.append(correct.item() / total)
        train_ce.append(train_ce_sum.item() / len(train_loader))
        # Measured once per epoch on the current weights, so the logged value
        # is the norm of the model being evaluated below.
        spectral_norms.append(compute_spectral_norm(model).item())

        # ── Eval phase ────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, cache)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Keep the weights of the best epoch seen so far, based on val accuracy
        if val_accuracies[-1] > best_val_acc:
            best_val_acc = val_accuracies[-1]
            best_state = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
            best_opt_state = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch

        # ── WandB logging ─────────────────────────────────────────────────────
        if args.use_wandb:
            wandb.log({
                "epoch":         epoch,
                "lr":            optimizer.param_groups[0]["lr"],
                "train_loss":    train_losses[-1],
                "val_loss":      val_losses[-1],
                "train_acc":     train_accuracies[-1],
                "train_ce":      train_ce[-1],
                "val_acc":       val_accuracies[-1],
                "spectral_norm": spectral_norms[-1],
            })

        print(
            f"[{epoch+1:02d}/{args.epochs}]  "
            f"train_loss={train_losses[-1]:.4f}  "
            f"val_acc={val_accuracies[-1]:.4f}  "
            f"sn={spectral_norms[-1]:.4f}"
        )

    total_time = time.time() - start_time

    # Load the best-on-val weights, then evaluate on the test split once
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, cache)
    print(f"[FINAL] best_epoch={best_epoch}  "
          f"best_val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.save_root, f"{model_type}_{freeze_mode}.pth")
    # Metadata travels with the weights so that drift_analysis_cls.py and
    # corruption_experiments_cls.py do not have to infer the architecture from the
    # file name, and can verify they are using the same split.
    torch.save({
        "model_state_dict":     best_state,
        "optimizer_state_dict": best_opt_state,
        "arch":                 model_type,
        "task":                 "classification",
        "dataset":              args.dataset,
        "num_classes":          args.num_classes,
        "freeze_mode":          freeze_mode,
        "lambda_sn":            args.lambda_sn,
        "split_seed":           args.split_seed,
        "best_epoch":           best_epoch,
        "best_val_acc":         best_val_acc,
        "test_acc":             test_acc,
    }, ckpt_path)
    print(f"[CHECKPOINT SAVED] {ckpt_path}")

    # ── Save experiment log ───────────────────────────────────────────────────
    save_experiment_log(
        save_path        = args.save_root,
        model_type       = model_type,
        freeze_mode      = freeze_mode,
        args             = args,
        train_losses     = train_losses,
        val_losses       = val_losses,
        train_accuracies = train_accuracies,
        train_ce         = train_ce,
        val_accuracies   = val_accuracies,
        spectral_norms   = spectral_norms,
        total_time_sec   = total_time,
        best_epoch       = best_epoch,
        best_val_acc     = best_val_acc,
        test_acc         = test_acc,
        test_loss        = test_loss,
    )

    if args.use_wandb:
        wandb.summary["best_epoch"]     = best_epoch
        wandb.summary["best_val_acc"]   = best_val_acc
        wandb.summary["test_acc"]       = test_acc
        wandb.summary["test_loss"]      = test_loss
        wandb.summary["total_time_sec"] = total_time
        wandb.finish()

def evaluate(model, loader, criterion, device, cache=None):
    """Run one evaluation pass and return (avg_loss, accuracy)."""
    model.eval()
    # Same reasoning as the training loop: accumulate on the GPU, cross to the
    # host once at the end instead of three times per batch.
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    correct    = torch.zeros((), dtype=torch.long, device=device)
    total      = 0
    with torch.no_grad():
        for inputs, labels in iter_batches(loader, device, cache):
            outputs = forward_batch(model, inputs, cache is not None)
            loss = criterion(outputs, labels)
            total_loss += loss.detach().double()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum()
            total += labels.size(0)
    return total_loss.item() / len(loader), correct.item() / total



# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    spec = get_spec(args.dataset)
    if spec.task != "classification":
        raise ValueError(
            f"Dataset '{args.dataset}' is a {spec.task} dataset. "
            "Use finetuning_seg.py for segmentation."
        )

    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.save_root, exist_ok=True)

    # Initialise WandB tracking
    if args.use_wandb:
        wb.init(
            stage=wb.FINETUNING, task="classification",
            name=wb.run_label(args.dataset, args.model_type, args.freeze_mode,
                            args.lambda_sn, args.split_seed),
            config=vars(args), mode=args.wandb_mode,
        )

    pin_cpu_cores(args.pin_cores, torch_threads=2)
    # split_seed drives BOTH the data split and the weight initialisation, so
    # --split_seed 0 1 2 gives three genuinely independent repetitions.  Kept
    # identical to finetuning_seg.py so "seed 0" means the same thing in both.
    reset_seed(args.split_seed)
    print(f"Running: {args.dataset} | {args.model_type} | {args.freeze_mode} | "
          f"split_seed={args.split_seed} | device={device}")
    train_model(args.model_type, args.freeze_mode, args, device)


