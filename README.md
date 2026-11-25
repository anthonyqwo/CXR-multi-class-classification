# Lab3: CXR Multi-class Classification

Classify chest X-rays into 4 classes: **normal / bacteria / virus / COVID-19**. The competition metric is **macro-F1**.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Reads the CSV or folder-per-class layout; preprocessing (border crop, resize, CLAHE); train / eval / TTA augmentation |
| `models.py` | timm backbone construction (any timm model name), TorchXRayVision CXR-pretrained backbones (`xrv_*`), Hugging Face DINOv2-style encoders such as RAD-DINO (`hf_*`) and a hand-written EMA |
| `losses.py` | Class-weighted focal / cross-entropy loss (supports soft labels and label smoothing); sampler weights |
| `train.py` | Baseline training: official split, K-fold CV, or full train+val; bf16 AMP, warmup + cosine LR, EMA, MixUp / CutMix |
| `train_v2.py` | Upgraded trainer: LLRD, DropPath, 5-branch Multi-Sample Dropout, lung-crop (`--lung_crop`) & 3-ch fusion (`--rad3ch`), gradient accumulation, composite checkpointing, Top-K Model Soup (`soup.pt`), 3-view TTA OOF, soft pseudo-labels, and binary B-vs-V specialist mode (`--bv_only`) |
| `predict.py` | Test inference: multi-run ensemble, `--ckpt auto` (`soup.pt > best.pt > last.pt`), multi-view TTA, optional decision-bias tuning; writes Kaggle CSV & `.probs.npy` |
| `stack.py` | Legacy v1 logistic-regression stacking with optional raw k-NN features |
| `stack_v2.py` | Champion v2–v7 decoupled meta-ensemble: strictly OOF temperature calibration, 7-foundation PCA-whitened Re-ID, 8/10-foundation frozen probe, 4/5-family NBV meta-learners, transductive peer smoothing, and clean binary COVID-19 head |
| `stack_v3_spatial.py` | Final `#1` (`0.93640`, 0% pseudo-labels) GPU-accelerated Spatial-Biometric Meta-Stacker: multi-layer 3×3 & 2×2 spatial quadrant embeddings (RAD-DINO/MAIRA-2 L6+L9, XraySigLIP-L L16+L20, DINOv2-Large L16) + Dual-Gram Shrinkage-ZCA + Exact Scale-Matched CSLS + 12 clean models + 2 binary B-vs-V specialists |
| `tools/extract_spatial_reid.py` | Extracts 2×2 spatial quadrant + Layer-9 embeddings for RAD-DINO & MAIRA-2 |
| `tools/extract_spatial_v5.py` | Extracts 3×3 spatial (L6+L9) for RAD-DINO/MAIRA-2 and 2×2 spatial mid-layer embeddings for XraySigLIP-L & DINOv2-Large |
| `blend_stacks.py` | Evaluates strictly 5-fold OOF and generates final probability-blended champion CSVs |
| `blend.py` | Averages each legacy model's 5-fold test probabilities with its full-data model |
| `embed.py` | Extracts frozen embeddings of all train / val / test images from a Hugging Face encoder |
| `tools/run_5fold_v2.py` | Auto-resuming 5-fold runner for `train_v2.py` with pooled OOF summary and TTA test prediction |
| `tools/full_train_v2.py` | Full-data (`--merge_val`) SWA retraining and 50/50 blending with 5-fold test probs into `submissions/per_tag_v2_blend/` |
| `tools/eval_bv_oof_all.py` | Strictly OOF 3-view TTA inference for `--bv_only` specialists across all 4 classes (prevents OOF/test distribution shift) |
| `summarize.py` | Compares all runs per tag: mean fold F1, per-class F1, and pooled OOF macro-F1; `--folds 0 4` limits to shared folds, `--csv` saves the table |
| `utils.py` | macro-F1 / per-class F1, confusion matrix, bias tuning |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

Tested with Python 3.14, torch 2.15.0+cu132, torchvision 0.29.2, and timm 1.0.31 on an RTX 4060 Ti (8 GB). For an older driver, pick the matching CUDA wheel index (cu128 / cu130).

Measured speed with bf16, batch 16 and 384 px:

| Model | Training speed | Peak VRAM |
|---|---|---|
| ConvNeXt-Tiny | ~127 img/s (~32 s/epoch on 4,017 images) | 3.3 GB |
| EfficientNet-B3 | ~124 img/s | 4.1 GB |

## Data layout

Put the Kaggle data under `data/`. Two layouts are supported:

```
data/
  train_images/  train_data.csv     # CSV columns: new_filename,normal,bacteria,virus,COVID-19 (one-hot)
  val_images/    val_data.csv
  test_images/
```

or one folder per class (`data/train/normal/*.jpeg`, ...). In that case, omit `--train_csv`.

## Training

```bash
# Baseline: official train / val split
python train.py --train_dir data/train_images --train_csv data/train_data.csv \
                --val_dir data/val_images --val_csv data/val_data.csv

# 5-fold CV (recommended: val has only about 7 COVID-19 images, so its F1 is very noisy)
for f in 0 1 2 3 4; do
  python train.py --train_dir data/train_images --train_csv data/train_data.csv \
                  --val_dir data/val_images --val_csv data/val_data.csv \
                  --folds 5 --fold $f --tag cv
done
```

Each run writes to `runs/<time>_<model>_.../`:
- `config.json`: all hyperparameters
- `log.csv`: per-epoch loss, acc, macro-F1, per-class F1
- `best.pt`: the checkpoint with the best val macro-F1
- `val_probs.npy`
- `val_report.txt`: confusion matrix

These files can go straight into the report.

### Image cache

Decoding the original 1–2k px JPEGs every epoch is the main CPU bottleneck. Before the first epoch, `train.py` decodes, border-crops and resizes every image once (in parallel threads) and stores the result as a lossless PNG under `cache/s<size>_c<crop>/`. Later epochs, later runs and `predict.py` read these small PNGs instead.

- The cache key includes the source path, file size, mtime, `img_size` and `crop_border`, so changing any of them builds a new cache instead of reusing a stale one.
- CLAHE is applied after the cache, so switching `--clahe` does not need a rebuild.
- `--cache_dir ''` disables the cache; deleting `cache/` is always safe.

### RAD-DINO

`hf_*` models load a Hugging Face encoder and classify on the concatenated CLS and mean patch token. Inputs are re-normalised to the encoder's own statistics. Gradient checkpointing is on by default so that 518 px with a batch of 8 fits in 8 GB (about 3.8 GB, ~270 s/epoch on an RTX 4060 Ti); set the environment variable `LAB3_GRAD_CKPT=0` to turn it off when memory allows.

### Main options (for ablations)

| Option | Default | Notes |
|---|---|---|
| `--model` | `convnext_tiny.fb_in22k_ft_in1k` | Any timm model, e.g. `efficientnet_b3.ra2_in1k`, `tf_efficientnetv2_s.in21k_ft_in1k`; `xrv_densenet121-res224-all` (needs `torchxrayvision`); or `hf_<org>--<name>`, e.g. `hf_microsoft--rad-dino` (needs `transformers`; use `--img_size 518 --batch_size 8 --lr 2e-5`) |
| `--img_size` | 384 | Input resolution |
| `--clahe` | 0 | CLAHE clip limit (try 2.0) |
| `--crop_border` | 0 | Crop this fraction from each side (removes markers and borders) |
| `--aug` | light | `none` / `light` / `strong` |
| `--hflip` | off | Horizontal flip (moves the heart to the other side, so test it as an ablation) |
| `--mixup` / `--cutmix` | 0 | Alpha values |
| `--focal_gamma` | 0 | 0 = CE; try 2.0 |
| `--class_weight_power` | 0.5 | Loss weight = (1/freq)^p |
| `--sampler_power` | 0 | Oversampling; set `--class_weight_power 0` when using it |
| `--label_smoothing` | 0.05 | |
| `--ema` | 0 | Try 0.999 |
| `--pseudo_dir` / `--pseudo_csv` | none | Unlabeled images with a one-hot pseudo-label CSV (e.g. confident test predictions); added to the training part only, and images byte-identical to a validation image are dropped |
| `--split_seed` | = `--seed` | Seed of the K-fold split; keep it fixed (42) to vary only the training seed and keep OOF predictions comparable |
| `--amp_dtype` | auto | `bf16` or `fp16`; force `fp16` on GPUs that report bf16 support but run it slowly (e.g. T4) |
| `--val_list` | none | A `val_split.csv` from another run; reproduces exactly that validation set (K-fold splits can differ between machines) |
| `--lr` / `--epochs` / `--batch_size` | 1e-4 / 25 / 16 | A batch of 16 at 384 px fits in 8 GB of VRAM |

## Inference and submission

```bash
python predict.py --runs runs/<run1> runs/<run2> ... \
                  --test_dir data/test_images --test_csv test_data_sample.csv \
                  --tta orig,zoom --tune_bias --out submissions/sub.csv
```

- The output keeps the row order of `test_data_sample.csv`.
- With several runs, softmax probabilities are averaged.
- `--tune_bias` searches per-class log-probability offsets that maximize macro-F1:
  - If the runs share the same val set, their predictions are averaged before tuning.
  - If they are K-fold runs, the pooled out-of-fold predictions are used, which is more stable.

### Stacking (final submissions)

```bash
# 1. test probabilities of each 5-fold model (mean of its 5 fold checkpoints)
python predict.py --runs runs/*of5_<tag> --test_dir data/test_images --test_csv test_data_sample.csv                   --tta orig --out submissions/per_tag/<tag>.csv
# 2. class-balanced logistic regression on the concatenated OOF log-probabilities
python stack.py --tags e2_clahe e4_effb3 e6_effv2s_clahe e8_effv2s_clahe448 e11_effv2s448_pl                 --C 0.01 --out submissions/stack5.csv
```

`stack.py` needs every tag trained on the same folds (same `--split_seed`). A tag written as `tagA~tagB` averages the OOF probabilities of several training seeds of one model into a single input; its test probabilities are read from `<test_probs_dir>/tagA~tagB.probs.npy`. It prints the simple-average OOF macro-F1 and the nested-CV estimate of the stacked macro-F1.

### Blending with full-data models

For the final submissions, each model's test probabilities are the mean of two sources: the 5 fold checkpoints, and one model retrained on all labelled data (train + val, no validation, last epoch).

```bash
# 1. retrain a 5-fold tag on all data with the same hyperparameters (run directory *_full_full_<tag>)
python tools/full_train.py <tag>
# 2. its test probabilities
python predict.py --runs runs/*_full_full_<tag> --test_dir data/test_images --test_csv test_data_sample.csv                   --tta orig --out submissions/per_tag_full/<tag>.csv
# 3. half fold mean, half full-data model, for every tag of the stack
python blend.py --tags e2_clahe e4_effb3 e6_effv2s_clahe e8_effv2s_clahe448 e11_effv2s448_pl e23_effv2m448_pl                 "e27_raddino518_pl~e28_maira518_pl" --out_dir submissions/per_tag_blend_e27
```

Then pass `--test_probs_dir submissions/per_tag_blend_e27` to `stack.py`. The meta-model is still fit on the OOF probabilities of the fold models only.

### Nearest-neighbour features

Images of the same patient look alike and almost always share a label. `embed.py` embeds every image with a frozen (not fine-tuned) encoder, and `stack.py --knn_feats` adds, for each image, the similarity-weighted labels of its most similar labelled images to the meta-model inputs.

```bash
python embed.py --model microsoft/rad-dino --out features/rad-dino.npz
python stack.py --tags e2_clahe e4_effb3 e6_effv2s_clahe e8_effv2s_clahe448 e11_effv2s448_pl e23_effv2m448_pl                 "e27_raddino518_pl~e28_maira518_pl" --C 0.01 --test_probs_dir submissions/per_tag_blend_e27                 --knn_feats features/rad-dino.npz --out submissions/stack7_e27e28_knn.csv
```

- For a labelled image, neighbours are taken from the other folds only, so these features are out-of-fold like the model probabilities. For a test image, neighbours are taken from all labelled images.
- Only train / val labels are used; no test label and no external data.
- `--knn_scale` (default 0.3) is the weight of these features; several `.npz` files can be given.
- `a.npz+b.npz` averages the similarities of several embeddings before the neighbour search.
- `--knn_gate 0.6` also adds the votes multiplied by max(0, similarity - 0.6), so that very similar neighbours count more. The best public submission (`stack7_e27e28_knn1g.csv`, 0.89447) is the command above plus this option.

### Champion Pipeline (`train_v2.py` + `stack_v3_spatial.py`, Public LB `0.93640` #1, 0% Pseudo-Labels)

All clean v2 models (`e33`, `e34`, `e35`, `e36`, `e36b`, `e37`, `e38`, `e42`, `e44`) use the exact same `StratifiedGroupKFold(md5, seed=42)` splits as `train.py`:

```bash
# 1. Extract multi-layer 3x3 & 2x2 spatial biometric embeddings (all 5,908 train+val+test images)
python tools/extract_spatial_reid.py
python tools/extract_spatial_v5.py

# 2. Reproduce #1 submission stack_v10_v9_superblend.csv (strict 5-fold CV 0.9189, bvAUC 0.9508, LL 0.2325 -> Public LB 0.93640)
#    and single-stack stack_v10_clean11_bv2_reidv5.csv (Public LB 0.93636)
python stack_v3_spatial.py
```

### Legacy Pseudo-labeling

Confident stack predictions on the test set (max probability >= 0.7) are written as a one-hot CSV and passed to `train.py --pseudo_dir data/test_images --pseudo_csv <csv>`. Validation folds still contain only labeled train/val images, so OOF scores remain comparable. They can be slightly optimistic, because the pseudo labels come from models that saw the other folds. (The final `#1` `stack_v5` submissions use 0% pseudo-labels.)

## What is not in this repository

- **Data** (`data/`, `test_data_sample.csv`): download it from the Kaggle competition page.
- **Training outputs** (`runs/`, `submissions/`): checkpoints, out-of-fold predictions and test probabilities are too large to version. `stack_v2.py` and `stack_v3_spatial.py` need them, so the stacks can only be reproduced after the models have been retrained with `train.py` / `train_v2.py`.
- **Frozen embeddings** (`features/*.npz`): regenerate them with `embed.py` and `tools/extract_spatial_*.py`. Only `features/lung_boxes.csv` (lung bounding boxes used by `--lung_crop`) is included.
- The scripts that produced `features/lung_boxes.csv`, `features/cxr_anatomy.npz`, the lung-cropped embeddings (`*-lc.npz`) and the zero-shot similarities in `features/xray-siglip-l384.npz` were run ad hoc and are not included. They use the public `ianpan/chest-x-ray-basic` and `StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli` models.
