# Medical_Robustness_Lipschitz

Research codebase for studying how **localized image corruptions** propagate
through the internal representations of medical imaging models, and how
**Lipschitz control of the task head** bounds their effect on the metrics that
are actually reported: top-1 accuracy for classification and Dice for
segmentation.

The pipeline covers **two tasks** (classification and segmentation), **three
backbones** (ResNet-34, ViT-B/16, DINOv2) and **multiple datasets**, all behind
one dataset registry and one output layout.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository structure](#2-repository-structure)
3. [Dependencies](#3-dependencies)
4. [Datasets](#4-datasets)
5. [Splits, seeds and repetitions](#5-splits-seeds-and-repetitions)
6. [Output layout](#6-output-layout)
7. [Step-by-step reproduction](#7-step-by-step-reproduction)
8. [Metrics and bounds](#8-metrics-and-bounds)
9. [Building the paper tables and figures](#9-building-the-paper-tables-and-figures)
10. [Experiment tracking (Weights & Biases)](#10-experiment-tracking-weights--biases)
11. [Sanity-check tools](#11-sanity-check-tools)

---

## 1. Overview

A corruption applied to part of an image shifts the features a backbone
extracts. The pipeline measures that shift, separates it inside and outside
the corrupted region, and relates it to the change in the final prediction.

Three stages, each available for both tasks:

| stage | classification | segmentation |
|---|---|---|
| fine-tuning | `finetuning/finetuning_cls.py` | `finetuning/finetuning_seg.py` |
| corruption sweep | `corruption_experiments/corruption_experiments_cls.py` | `..._seg.py` |
| drift / Lipschitz | `drift_analysis/drift_analysis_cls.py` | `..._seg.py` |

Every script takes `--dataset <key>` resolved through
`shared/DatasetRegistry.py`, so the same key always means the same folders and
the same class-discovery rules everywhere.

**The head is affine in both tasks.** For classification it is an `nn.Linear`;
for segmentation a `1x1` convolution, which applies the *same* affine map
independently at every spatial location. That is what lets the Lipschitz
analysis carry over to segmentation unchanged, one pixel at a time, and makes
the spectral norm exactly computable in both cases.

---

## 2. Repository structure

```
shared/
  gpu_pool.sh            memory-aware GPU admission, used by all 8 launchers
  metrics.py             the one definition of Dice / IoU
  DatasetRegistry.py     every dataset: path, task, class discovery, mask pairing
  paths.py               single definition of the output layout and run names
  wandb_logging.py       entity, project naming and init for every script
  splits.py              stratified train/val/test indices, persisted to JSON
  featcache.py           frozen-backbone feature cache for freeze_mode=head
  SegmentationModels.py  spatial feature extractors + 1x1 conv head
  ModelFactory.py        classification model builder, get_head, spectral norm
  DINOv2Wrapper.py       DINOv2 backbone + linear head
  Datasets.py            legacy Kvasir-v2 loaders (npy / folder backends)
  constants.py           ImageNet normalisation statistics
  runtime.py             optional CPU pinning, never fatal
  seed.py                reset_seed()

finetuning/              training, one script per task + launchers
  check_runs.sh          missing / duplicated fine-tuning runs
  check_performance.sh   runs with suspiciously low test metric
ablation/                lambda ablation launchers
  select_lambda.py       lambda* selection on validation (Appendix lambda)
corruption_experiments/  corruption sweep, one script per task + launchers
  corruption_metrics.py  masked MSE / cosine on the feature grid
drift_analysis/          margin certificate (both Dice bounds) + launchers
  perturbations.py       pixel-space gaussian and ImageNet-C perturbations
corruptions/             mask shapes (circular, squared, superpixel, global)
  imagenet_c.py          ImageNet-C corruption functions (do not edit)
  preview.py             renders corrupted images for visual inspection

analysis/                every paper table and figure
  make_paper_figures.sh  the single driver; writes paper_figures/ and paper_tables/
  io.py                  the only code that knows what a result file looks like
  common.py              seed aggregation, paired deltas, LaTeX, plot style
  pack_results.py        compress the whole tree into a few small CSVs
  finetuning/            clean metrics table (Tab. 2) and head spectral norm figure
  1_depth_leakage.py           propagation with depth (R_leak)
  2_drift_predicts_damage.py   inside / outside drift as predictors
  3_in_out_dose_response.py    severity and corrupted area
  4_geometry_beats_area.py     mask geometry at matched area
  5_adaptation_not_robustness.py  clean and corrupted performance
  6_certificate_needs_sigma.py    margin certificate against observed drops
  qualitative_maps.py          qualitative feature-distortion maps
  stack_tasks.py               stacks classification / segmentation rows
  appendix/
    datasets.py                Appendix A: datasets, classes, splits
    effective_ratio.py         Appendix B: corrupted area vs nominal ratio
    perturbation_magnitude.py  Appendix B: input-space strength
    per_dataset.py             Appendix F: results per dataset
    full_grid.py               Appendix G: full corruption grid
    claims_per_dataset.py      Appendix F: main claims checked on each dataset
    corruption_audit.py        Appendix C: effective area and confinement of the masks

tests/
  test_segmentation_models.py  smoke-tests the three segmentation backbones
  smoke_test_corruption.sh     one-checkpoint run of the corruption sweep

run_all.sh               the whole chain, 8 stages: ablations -> lambda* ->
                         fine-tuning sweeps -> corruption -> drift
splits/<dataset>/split_seed<N>.json    train/val/test indices
```

Every path in the pipeline is an environment variable with a default, so
nothing has to be edited to move machine:

| variable | default | meaning |
|---|---|---|
| `RESULTS_ROOT` | `/home/a.dorizza/OUTPUTS_MEDICAL` | where results are written |
| `SPLIT_ROOT` | `<repo>/splits` | the split files, versioned with the code |
| `MEDICAL_ROOT` | `/home/datasets/medical` | dataset root, read by the registry |
| `PYTHONHASHSEED` | `0` | exported by the launchers; the interpreter reads it at startup |
| `FEATCACHE_DIR` | `<RESULTS_ROOT>/featcache` | frozen-backbone features, derived data, deletable |
| `GPU_POOL_MAX_PER_GPU` | `8` | ceiling on concurrent jobs per card; `0` = memory rule only |

`run_all.sh` exports them all, so the eight launchers it calls inherit one
consistent set rather than each falling back to its own default.

---

Every script that exists in a classification and a segmentation flavour is named
for it: `finetuning_cls.py` / `finetuning_seg.py`, `run_ablation_cls.sh` /
`run_ablation_seg.sh`, and so on. The classification ones used to carry no
suffix, which made a directory listing look like the segmentation pipeline was
a bolt-on rather than half of the experiment.

## 3. Dependencies

```bash
pip install -r requirements.txt
```

**ImageMagick is required at system level**, not only via pip:
`corruptions/imagenet_c.py` imports `wand` at module level, and `wand` binds to
ImageMagick.

```bash
sudo apt-get install libmagickwand-dev     # Debian/Ubuntu
brew install imagemagick                   # macOS
```

DINOv2 is fetched with `torch.hub`, so the first run needs network access.
Set `TORCH_HOME` to a shared writable directory to download it once:

```bash
export TORCH_HOME=/path/to/shared/torch_cache
```

---

## 4. Datasets

All paths live in `shared/DatasetRegistry.py`; edit `MEDICAL_ROOT` once.

```bash
python3 -c "
from shared.DatasetRegistry import list_datasets
print('classification:', list_datasets('classification'))
print('segmentation  :', list_datasets('segmentation'))"
```

### Classification

| key | notes |
|---|---|
| `kvasir_v2` | 8 classes |
| `hyperkvasir` | `labeled-images` only; classes are the leaf folders of the tract/category tree |
| `kvasir_capsule` | same mechanism; extract the per-class `.tar.gz` first |

Class counts are **discovered at load time** for the last two.
`finetuning_cls.py` reads `--num_classes` from the dataset automatically, and
saves it in the checkpoint; `corruption_experiments_cls.py` and
`drift_analysis_cls.py` then read it back from the checkpoint (see
`shared.ModelFactory.peek_num_classes`) instead of assuming Kvasir-v2's fixed
8, which would otherwise crash on these two with a shape mismatch. Classes with fewer than
`--min_per_class` images (default 20) are dropped and reported: a class with a
handful of examples cannot be split 70/10/20. `--max_per_class` caps very
large classes, which matters because both datasets are heavily imbalanced.

### Segmentation

| key | role |
|---|---|
| `kvasir_seg` | polyp training set |
| `cvc_clinicdb`, `cvc_colondb`, `etis_larib` | trainable with their own 70/10/20 split, and the usual zero-shot test sets for Kvasir-SEG models |
| `aircare_hypopharynx`, `aircare_larynx`, `aircare_oral_cavity`, `aircare_oropharynx` | Brescia head-and-neck sites |
| `aircare_all` | the four AIRCARE sites pooled (1031 images) |

Two protocols are supported for every segmentation dataset:

* **A — in-distribution.** Train with the dataset's own 70/10/20 split. Any key
  can be used, including the CVC/ETIS ones.
* **B — external.** Train on one dataset and evaluate zero-shot on the others.
  Training on `kvasir_seg` and testing on CVC/ETIS is the standard protocol in
  polyp segmentation and is what provides external validation.

The launchers pick the external sets automatically and **always exclude the
training set from that list**, since it is already evaluated on its own
held-out split; an `aircare_*` training set is evaluated on the remaining
AIRCARE sites instead of on CVC/ETIS.

`aircare_all` pools the four sites because individually they are small
(Hypopharynx has 39 images, so its validation fold would be about 4). It
deliberately excludes `4. Brescia_Larynx_best_selected`, verified to be an
exact duplicate subset of `Larynx_(611_images)`.

Before training on a new dataset, check the image/mask pairing:

```bash
python3 -c "
from shared.DatasetRegistry import build_dataset, list_datasets
for name in list_datasets('segmentation'):
    ds = build_dataset(name)
    img, msk = ds.raw(0)
    print(f'{name:22s} n={len(ds):5d} img={img.shape} mask_fg={msk.mean():.3f}')"
```

A `WARNING: N images without a mask were skipped` means `images_dir`,
`masks_dir` or `mask_prefix` need fixing in the registry. A `mask_fg` of ~0 or
~1 everywhere means the masks are inverted or not binary.

---

## 5. Splits, seeds and repetitions

### Three-way split

Every dataset is split **70/10/20** into train/val/test, stratified (by class
for classification, by lesion size for segmentation), and written to
`splits/<dataset>/split_seed<N>.json`.

| split | used for |
|---|---|
| train | weight updates |
| val | every decision: which epoch to keep, which lambda to use |
| test | evaluated **once**, after all decisions are frozen |

The saved checkpoint is the **best epoch on validation**, never the last one.
Lambda must therefore be selected on validation accuracy, never on test.

The split file stores a fingerprint of the dataset and is loaded, not
regenerated, by the corruption and drift scripts, so a mismatch fails loudly
instead of silently evaluating on images seen during training.

### Two independent seeds, on purpose

| flag | default | controls |
|---|---|---|
| `--split_seed` | 0 | the data split **and** the weight initialisation |
| `--seed` | 29 | the corruption masks and perturbation noise |

`--split_seed` is what makes a repetition independent: it changes both which
images are in each fold and how the head is initialised.
`reset_seed(args.split_seed)` is used identically in both training scripts, so
"seed 0" means the same thing for classification and segmentation.

`reset_seed()` covers torch (CPU and CUDA), numpy and `random`, locks cuDNN to
deterministic mode, and installs `seed_worker` as the DataLoader
`worker_init_fn` so a worker's random state depends on (seed, worker id) rather
than on `--num_workers`. `reset_seed(seed, strict=True)` additionally requests
deterministic CUDA kernels; it is off by default because a few backward kernels
have no deterministic implementation and would raise mid-training. The training
loaders deliberately do **not** take a separate `generator=`: the shuffling
order of every reported run comes from the global torch RNG seeded by
`reset_seed(split_seed)`, and pinning a second generator would give a
different — equally valid, but different — order.

`--seed` is deliberately **not** tied to `--split_seed`: the corruption mask
for a given image must be identical across architectures, settings and
repetitions, otherwise the comparison between them is no longer paired.

### Multi-seed runs

```bash
SPLIT_SEEDS=(0 1 2)      # in run_finetuning_cls.sh / run_finetuning_seg.sh
```

Seed 0 keeps its plain run name; seeds 1+ append `_seed<N>`. Nothing is
overwritten, and `analysis/io.py` recovers the seed from that suffix to
report mean +/- std together with the number of repetitions.

---

## 6. Output layout

Defined once in `shared/paths.py`:

```
<RESULTS_ROOT>/<stage>/<task>/<dataset>/<run>
```

```
OUTPUTS_MEDICAL/
  finetuning/
    classification/kvasir_v2/dinov2_head+last_lambda100/
        dinov2_head+last.pth          best-on-val weights + metadata
        dinov2_head+last_log.json     curves + "selection" block
    segmentation/kvasir_seg/dinov2_head+last_lambda100_seed1/
  corruption_experiments/
    classification/kvasir_v2/dinov2_head+last_lambda100_rawfeat/metrics.json
    segmentation/kvasir_seg/dinov2_head+last_lambda100_on_etis_larib_rawfeat/
  drift_analysis/
    classification/kvasir_v2/drift_gaussian/dinov2_head+last_lambda100/
    segmentation/kvasir_seg/drift_Contrast/dinov2_head+last_lambda100/
```

Run name: `<arch>_<freeze>_lambda<L>[_seed<N>]`. Corruption runs append the
feature variant (`rawfeat` / `normfeat`) and `_softmasking` when applicable.

`RESULTS_ROOT` and `SPLIT_ROOT` are declared at the top of **every** launcher
and must be identical across all six.

---

## 7. Step-by-step reproduction

```bash
cd /path/to/Medical_Robustness_Lipschitz_v2
export TORCH_HOME=/path/to/shared/torch_cache
```

### 7.0 Verify the corruptions first

Two minutes, and it avoids discovering a problem after days of GPU time.

```bash
python -m analysis.appendix.effective_ratio --size 224 --images /path/to/Kvasir-SEG/images
python -m corruptions.preview --mode all --images /path/to/Kvasir-SEG/images \
    --out OUTPUTS/corruption_preview
python -m analysis.appendix.perturbation_magnitude --dataset kvasir_seg \
    --out paper_assets/perturbation_audit.csv
```

The third one is CPU-only and answers a question the other two cannot: how hard
each corruption actually hits, in input space, restricted to its own mask. It
writes one row per (shape, perturbation, severity, ratio) and prints the spread
across shapes at fixed perturbation and severity.

That matters because "severity 3" names a parameter of an ImageNet-C function,
not a fixed amount of damage. If the spread is flat, a cross-shape comparison in
the paper is measuring spatial propagation, which is the claim. If it is not,
the shape effect has to be reported at matched `delta_rms_in`.

It is also what caught a real bug. `corruptions/utility.py` used to perturb the
tight **bounding box** of the mask rather than the whole image. For functions
that normalise against a statistic of their input -- ImageNet-C's Contrast
rescales around the ROI mean, Brightness works in HSV -- that made the effective
severity depend on the mask **shape**: `circular` with `invert=True` has a
bounding box equal to the whole image, while `squared` was applied patch by
patch, each normalised against its own local mean. Measured on Kvasir-SEG at
severity 5, ratio 0.3, Contrast moved the masked pixels by 73 grey levels under
`circular` and 30 under `squared` -- the same declared severity, less than half
the perturbation.

The module now perturbs the whole image and composites the result through the
mask, so the perturbation no longer depends on the mask at all and severity
means the same thing everywhere. Comparisons across architectures, settings or
lambda were never affected either way: those are paired, and every configuration
sees the identical corrupted image.

The audit also prints `delta_max_out` per configuration, which must be 0
everywhere. A non-zero value means the corruption escaped its spatial support
and the whole inside/outside decomposition is invalid.

### 7.1 Fine-tuning

```bash
cd finetuning
./run_finetuning_cls.sh                  # kvasir_v2 (classification)
./run_finetuning_cls.sh hyperkvasir
./run_finetuning_seg.sh              # kvasir_seg (segmentation)
./run_finetuning_seg.sh aircare_all
```

Hyperparameters are declared at the top of each launcher and passed on the
command line, never left to an argparse default. The values every reported run
used:

| | classification | segmentation |
|---|---|---|
| epochs | 50 | 100 |
| batch size | 128 | 32 |
| optimizer | Adam | Adam |
| learning rate | 1e-4 | 1e-4 |
| weight decay | 0 | 0 |
| loss | cross-entropy | cross-entropy + soft Dice (weight 1.0) |
| λ (sweep) | 0 and 0.3 | 0 and 0.1 |
| λ (ablation) | 0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10 | same |
| seeds | 0–4 | 0–4 |
| `min_per_class` | 20 | n/a |
| `max_per_class` | **none** | n/a |

Segmentation uses more epochs because its datasets are far smaller, and a
smaller batch because dense prediction holds a full feature grid plus upsampled
logits per sample. Both are upper bounds: the best-on-validation checkpoint
makes a generous value safe.

`max_per_class` being unset is a deliberate choice with consequences worth
stating in a paper: Hyper-Kvasir and Kvasir-Capsule are trained on their full,
heavily imbalanced label distributions. Set `MAX_PER_CLASS` in
`run_finetuning_cls.sh` to cap them.

Check convergence afterwards:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('<RESULTS_ROOT>/finetuning/*/*/*/*_log.json')):
    d = json.load(open(f)); s = d['selection']
    print(d['task'][:3], d['dataset'], f.split('/')[-2], 'ep', s['best_epoch'])"
```

If `best_epoch` lands on the last epoch, raise `EPOCHS` for that dataset.

### GPU scheduling and resuming

No launcher declares a jobs-per-GPU limit. `shared/gpu_pool.sh` admits a job
onto a card only when there is room for it now *and* there would still be room
if every job the launcher has running there grew to its peak, so the concurrency
follows the real size of the card. The per-job peak calibrates itself: the first
job runs alone, its memory is read back and scaled by `GPU_POOL_GROWTH`, and the
estimate is revised upward whenever a job is seen using more than it — which is
what covers jobs whose footprint grows as they run. Tune with
`GPU_POOL_HEADROOM_MB`, `GPU_POOL_GROWTH`, `GPU_POOL_DEVICES`, or pin the
estimate outright with `GPU_POOL_PEAK_MB`.

On top of the memory rule the launchers set `GPU_POOL_MAX_PER_GPU=8`, a plain
ceiling on how many of our own jobs go on one card. The memory rule is about
not running out of VRAM; once the `head` runs replay cached features (below)
they are small enough that it would admit far more of them than the card has
compute for, and past roughly eight they stop getting faster and only queue on
each other and on the node's CPUs. `GPU_POOL_MAX_PER_GPU=0` goes back to the
memory rule alone.

### Backbone feature cache (`freeze_mode=head`)

With `freeze_mode=head` the backbone is frozen, every BatchNorm is pinned to
`.eval()`, the preprocessing is deterministic and there is no augmentation and
no dropout — so the backbone's output for a given image is the same at every
epoch and the same in every run that shares `(dataset, model_type, input_size)`.
`shared/featcache.py` computes it once, writes it to `$FEATCACHE_DIR`
(default `RESULTS_ROOT/featcache`) and replays it: those runs then train the
head alone, with no image decoding and no backbone forward or backward. It is
the same arithmetic, not an approximation — what disappears is a computation
whose result was already known — and it covers half of each grid, the half that
`head+last` does not.

The first job of each `(dataset, model_type)` group pays for one extraction
pass; concurrent jobs of the same group wait on a lock and then read the file,
so it is extracted once, not eight times. `head+last` runs are untouched and
take the original path. `--featcache off` forces recomputation, which is how to
check that the two paths agree:

```bash
python3 -m finetuning.finetuning_seg --dataset kvasir_seg --model_type resnet \
    --freeze_mode head --lambda_sn 0.1 --split_seed 0 --epochs 100 \
    --save_root /tmp/check_off --featcache off
# then the same command with --featcache auto, and diff the two _log.json
```

The cache is derived data and can be deleted at any time; it is keyed by the
dataset key, the input size and the full list of sample paths, so a dataset
that changes gets a new entry instead of a stale one.

Every stage resumes. A configuration that already produced a complete result is
skipped, so the chain can be restarted after a crash, a preemption or an added
seed and will launch only what is missing. "Complete" means the result file
exists **and is non-empty**, and for training it means the checkpoint **and** its
log: a job killed after the best epoch leaves a `.pth` with no `_log.json`, and
treating that as done would put a silent hole in the analysis. `FORCE=1` redoes
everything.

Or run the whole chain unattended:

```bash
nohup setsid ./run_all.sh > run_all.log 2>&1 < /dev/null & disown
```

Eight stages: the two lambda ablations, the selection on validation, the two
fine-tuning sweeps, the two corruption sweeps and the two drift analyses.
Stages 5-8 run strictly one at a time, because each launcher throttles only the
jobs it started itself and two of them sharing the GPU pool oversubscribe it.
`SKIP_EVAL=1` stops after the sweeps.

By default the corruption sweep is **in-distribution only**: each checkpoint is
evaluated on the held-out test split of the dataset it was trained on, giving
one run per checkpoint, named `<run>_on_<same dataset>_rawfeat_softmasking`.
`CORRUPTION_INCLUDE_EXTERNAL=1` adds the zero-shot runs, which multiplies the
job count by the size of the polyp family. The drift stage does include the
zero-shot evaluations by default, since it is one pass per dataset rather than
108 corruption combinations; `DRIFT_INCLUDE_EXTERNAL=0` turns them off.

### 7.2 Corruption sweep

```bash
cd corruption_experiments
./run_corruption_experiments_cls.sh hyperkvasir
./run_corruption_experiments_seg.sh kvasir_seg     # + CVC/ETIS zero-shot
```

`squared_size` must divide the image size exactly: at 224 px use 32 (7x7 grid,
full coverage). With 50 the grid covers only 80% of the image and ratio 0.9 is
unreachable.

### 7.3 Drift and Lipschitz bound

```bash
cd drift_analysis
./run_drift_analysis_cls.sh hyperkvasir
./run_drift_analysis_seg.sh kvasir_seg
```

Both gaussian and ImageNet-C perturbations are applied in **pixel space**, so
severities match those of the corruption sweep exactly. `--eps` is in `[0,1]`
pixel units.

In the segmentation results, check `"bound_violations": 0` in every JSON. A
non-zero value means the bound is broken somewhere in the chain
feature -> head -> logit; stop and investigate before reporting anything.

---

## 8. Metrics and bounds

### Feature-level

Masked MSE and cosine similarity between clean and corrupted features, in
three regions: inside the corruption mask, outside it, and over the whole
image. The image-space mask is projected onto the feature grid by **area
averaging**, so a feature cell receives the fraction of its footprint that is
corrupted. An empty region yields NaN, never 0, and batches are combined with
a weighted mean so a partial last batch does not count as a full one.

### Output-level

Classification reports clean and corrupted accuracy. Segmentation reports
Dice and IoU.

**One definition of Dice, in `shared/metrics.py`, used by all three stages.**

```
Dice = 2|P n G| / (|P| + |G|)      no smoothing
|P| + |G| = 0  ->  undefined (NaN), excluded from the mean, counted separately
```

The three stages used to disagree. `finetuning_seg.py` added a `1e-6` smoothing
term, which scores a lesion-free image on which the model predicts nothing as a
perfect 1.0 — a score awarded for a measurement that was never taken, and one
that rewards a conservative model exactly where it should be silent. The other
two already excluded that case. Every reported Dice now carries the number of
images it was averaged over, because a mean over 150 of 200 images is a
different claim from a mean over 200.

One difference remains and is deliberate: `drift_analysis_seg.py` evaluates on
the **feature grid**, before bilinear upsampling, because that is the grid the
per-pixel bound is a statement about. It is systematically higher than the
full-resolution Dice and must never be put in the same column; the tables label
it as such.

Segmentation also reports two decompositions the classification task cannot
express:

* **Dice inside vs. outside** the corrupted region, mirroring `MSE_in` / `MSE_out`;
* **Dice drop by overlap** between the corruption mask and the ground-truth
  lesion (`none` / `partial` / `full`). A positive drop in the `none` stratum
  means a corruption that never touches the lesion still degrades its
  segmentation, i.e. propagation measured at the output.

### The bound

With `Gamma = max_{a!=b} ||w_a - w_b||` (the Lipschitz constant of the
*margin*, always `<= sqrt(2)*sigma_max`), a prediction is unchanged whenever
`rho > Gamma*||dz||_2`. For segmentation this applies per pixel, and the set R
of pixels that may flip gives two bounds, both computed and checked:

* loose: `|dDice| <= 3n/(S-n)`, with `n = |R|` and `S = |P| + |G|`;
* exact (the one in the paper): `Dice - Dice' <= 2[n_tp(N-T) + n_bg T] /
  (N(N - n_tp + n_bg))`, with `T = |P & G|`, `N = |P| + |G|`, `n_tp` the true
  positives in R and `n_bg` the background pixels in R outside `P | G`.

The bound is certified: violations must be exactly zero, and the reported
violation rate is a correctness check, not a result.

---

## 9. Building the paper tables and figures

```bash
export RESULTS_ROOT=/path/to/OUTPUTS_MEDICAL
QUALITATIVE=1 DATASETS=1 bash analysis/make_paper_figures.sh paper_results .cache_analysis
```

`QUALITATIVE=1` also draws the qualitative maps (needs a GPU and the
checkpoints), `DATASETS=1` also writes Table 1 and the dataset appendix (needs
the images), `REFRESH=1` re-reads the results tree after a rerun. The final
assets land in two folders that mirror the Overleaf project:

```
paper_results/paper_figures/{main,appendix}/   ->  figs/{main,appendix}/
paper_results/paper_tables/{main,appendix}/    ->  tables/{main,appendix}/
```

Walks `RESULTS_ROOT` itself, groups by task and dataset, averages over seeds,
and writes LaTeX **and** CSV side by side so every number in the paper is
traceable to the frame that produced it without re-running anything. Each
script also runs on its own with the same flags, which is what you want while
iterating on a single table:

```bash
python -m analysis.6_certificate_needs_sigma --task segmentation --out paper_results --cache .cache_analysis
```

The cache matters: the drift files carry hundreds of thousands of per-pixel
values each, so parsing the tree takes minutes and the driver does it once.
Parquet is used when `pyarrow` is available and pickle otherwise.

`--results_root` also accepts a **results pack** -- the directory of gzipped
CSVs that `analysis/pack_results.py` writes -- so every figure is reproducible off
the cluster from the 44 MB pack, with no tree and no cache:

```bash
tar xzf results_pack.tgz
python -m analysis.4_geometry_beats_area --results_root results_pack --out paper_results
```

Every insight script takes `--task classification` or `--task segmentation`;
scripts 1-4 also take `--task both`,
which pools the two. Pooling is offered exactly where the quantity is unit-free
-- the retained fraction `m^c/m` and the relative damage `1 - m^c/m` are shares
of each model's own clean metric, so an accuracy and a Dice are the same
quantity there. It is not offered for the certificate figures, whose bounds are
in accuracy points against Dice points and whose units are images against
pixels.

Three rules the analysis layer follows, and which the captions state:

* **every table in the body varies at most two dimensions and fixes the rest**,
  with the fixed values printed in the caption; full disaggregation goes to the
  appendix scripts in `analysis/appendix/`;
* **deltas between settings are paired within seed.** Seed *k* of `H+L` and
  seed *k* of `H+L+SN` share the split and the initialisation, so differencing
  inside the seed removes the split variance that dominates. The difference of
  two column means is not the same quantity;
* **no p-values on five repetitions.** The non-parametric statement — "did not
  hurt on 5 of 5 seeds" — is reported instead.

| paper asset | produced by |
|---|---|
| Tab. 1 `datasets` | `analysis/appendix/datasets.py` (with `DATASETS=1`) |
| Tab. 2 `clean_metrics`, Fig. 2 | `analysis/finetuning/generate_tables.py`, `generate_figures.py` |
| Fig. 3 leakage with depth | `analysis/1_depth_leakage.py` |
| Fig. 4 severity and area | `analysis/3_in_out_dose_response.py` |
| Fig. 5 mask geometry | `analysis/4_geometry_beats_area.py` |
| Fig. 6 inside / outside predictors | `analysis/2_drift_predicts_damage.py` |
| Fig. 7 fine-tuning settings | `analysis/5_adaptation_not_robustness.py` |
| Fig. 8 margin certificate (+ Contrast, appendix) | `analysis/6_certificate_needs_sigma.py` |
| qualitative maps | `analysis/qualitative_maps.py` (with `QUALITATIVE=1`) |
| lambda tables (appendix) | `ablation/select_lambda.py` |
| MSE appendix | scripts 1, 2, 3 with `--metric mse` |
| per-dataset appendix | `analysis/appendix/per_dataset.py`, `claims_per_dataset.py` |
| full grid appendix | `analysis/appendix/full_grid.py` |
| dataset appendix (B) | `analysis/appendix/datasets.py` (with `DATASETS=1`) |
| corruption audit (C) | `analysis/appendix/corruption_audit.py` |

### Reading both on-disk formats

The drift stage has been through a rename and a results tree may contain runs
from either era (`pixel_flip_rate` / `true_dice_drop` / `mean_dice_bound`
versus `true_flip_rate` / `true_drop` / `predicted_drop`, and
`margin_bound` versus `summary` + `per_sample`). `analysis/io.py` and
`analysis/pack_results.py` resolve every field through both spellings, and the
drift frame carries a `format` column so a mixed tree is visible rather than
silently averaged. Re-running the drift stage with the current code puts
everything in the current format.

## 10. Experiment tracking (Weights & Biases)

Configured once in `shared/wandb_logging.py`, used by all six scripts.

```
entity   medical_robustness          (override with the WANDB_ENTITY env var)
project  <stage>-<task>
run      <dataset>_<arch>_<freeze>_lambda<L>_seed<N>
```

Six projects, one per stage and task:

```
finetuning-classification    finetuning-segmentation
corruption-classification    corruption-segmentation
drift-classification         drift-segmentation
```

Every launcher exposes `USE_WANDB` and `WANDB_MODE`; the mode is `online`,
`offline` (buffer to disk, then `wandb sync <dir>`) or `disabled`. The two
fine-tuning launchers pass `--wandb_mode offline`: with dozens of concurrent
jobs, syncing to the network while they run is contention that buys nothing.
Push everything afterwards — `run_all.sh` does it as its last step, and by hand
it is

```bash
wandb sync --sync-all wandb/
```

which can also be run while the sweep is still going; it picks up only the runs
that have already finished. The entity
and project names are deliberately *not* per-launcher settings, so runs cannot
drift into four differently-named projects again.

The run name always spells out the seed, including `_seed0`, so repetitions of
one configuration sort next to each other in the UI. On disk the seed-0 folder
omits the suffix instead, to keep the names of already-completed runs stable.

---

## 11. Sanity-check tools

### `analysis/appendix/effective_ratio.py`

Measures the fraction of pixels each configuration actually corrupts. Pass
`--images` to use real data: the synthetic fallback uses structured images
because SLIC collapses uniform noise into a single segment, which would make
`superpixel` look like a full-image corruption at every ratio.

### `corruptions/preview.py`

Renders PNG grids (`--mode shapes|severity|ratio|leakage|all`). The `leakage`
page is the important one: the difference panel must be black outside the
mask, and `max outside` must print `0`, otherwise the corruption escaped its
spatial support and the IN/OUT decomposition is invalid.

### `tests/test_segmentation_models.py`

Smoke-tests the three segmentation backbones: output shapes, feature grid
sizes (7x7 / 14x14 / 16x16), the linear-head identity `W*z + b == logits`, the
spectral norm, and drift under small noise. With `--checkpoint_dir` it also
renders vanilla vs fine-tuned predictions side by side.

```bash
python -m tests.test_segmentation_models --root /path/to/Kvasir-SEG \
    --out OUTPUTS/seg_model_preview
```

---

## Authors

Alberto Dorizza — [RETIS Lab, Scuola Superiore Sant'Anna, Pisa](https://retis.santannapisa.it)
