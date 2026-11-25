"""Upgraded training script (v2) for CXR 4-class classification (and Bacteria-vs-Virus specialist).

Key improvements over train.py:
- 100% compatible with existing 5-fold StratifiedGroupKFold (seed 42) and runs/ directory layout.
- Supports Layer-wise LR Decay (LLRD) + Multi-Sample Dropout + DropPath (hfsiglip_*, hfdinov2_*, timm).
- Supports 3-channel radiological fusion (--rad3ch), lung cropping (--lung_crop), letterbox (--letterbox).
- Supports gradient accumulation (--grad_accum) for large ViTs in 8GB VRAM.
- Supports soft probability pseudo-labels (--soft_pseudo_npy / --soft_pseudo_csv) with tunable weight.
- Supports dedicated Bacteria-vs-Virus binary specialist mode (--bv_only).
- Composite checkpoint selection + Top-K Epoch Weight Averaging (Model Soup -> soup.pt) + multi-view TTA OOF.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T
from PIL import Image

from dataset import (CLASSES, NUM_CLASSES, CXRDataset, build_eval_transform, build_train_transform,
                     make_preprocessor, read_split, warm_cache)
from losses import SoftFocalLoss, class_counts, class_weights, sample_weights
from models import ModelEMA, build_llrd_param_groups, build_model
from utils import compute_metrics, format_confusion, seed_everything


class SoftCXRDataset(Dataset):
    """Dataset supporting hard int labels OR soft probability targets + per-sample loss weight."""

    def __init__(self, paths: list[str], targets: np.ndarray, weights: np.ndarray, pre, transform):
        self.paths = paths
        self.targets = targets.astype(np.float32)  # (N, C)
        self.weights = weights.astype(np.float32)  # (N,)
        self.pre = pre
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.pre(self.paths[idx])
        img = Image.fromarray(img).convert("RGB")
        return self.transform(img), torch.from_numpy(self.targets[idx]), float(self.weights[idx])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--train_dir", default="data/train_images")
    g.add_argument("--train_csv", default="data/train_data.csv")
    g.add_argument("--val_dir", default="data/val_images")
    g.add_argument("--val_csv", default="data/val_data.csv")
    g.add_argument("--folds", type=int, default=5)
    g.add_argument("--fold", type=int, default=0)
    g.add_argument("--merge_val", action="store_true")
    g.add_argument("--bv_only", action="store_true", help="Bacteria (0) vs Virus (1) specialist mode")
    g.add_argument("--soft_pseudo_dir", default=None)
    g.add_argument("--soft_pseudo_csv", default=None, help="CSV with new_filename matching soft_pseudo_npy rows")
    g.add_argument("--soft_pseudo_npy", default=None, help="(N, 4) float probabilities for soft pseudo-labeling")
    g.add_argument("--pseudo_conf", type=float, default=0.0, help="min max-prob to keep a soft pseudo sample")
    g.add_argument("--pseudo_weight", type=float, default=0.5, help="loss weight multiplier for pseudo samples")
    g.add_argument("--pseudo_exclude_covid", action="store_true",
                   help="exclude pseudo samples whose argmax is COVID-19 (prevents COVID-19 FP poisoning)")

    g = p.add_argument_group("preprocessing / augmentation")
    g.add_argument("--img_size", type=int, default=384)
    g.add_argument("--clahe", type=float, default=0.0)
    g.add_argument("--rad3ch", action="store_true", help="3-ch fusion: [raw, CLAHE(2), CLAHE(4)]")
    g.add_argument("--letterbox", action="store_true", help="aspect-preserving square pad")
    g.add_argument("--lung_crop", default=None, help="path to lung_boxes.csv")
    g.add_argument("--lung_margin", type=float, default=0.12)
    g.add_argument("--crop_border", type=float, default=0.0)
    g.add_argument("--aug", choices=["none", "light", "strong"], default="light")
    g.add_argument("--hflip", action="store_true")
    g.add_argument("--mixup", type=float, default=0.0)
    g.add_argument("--cutmix", type=float, default=0.0)
    g.add_argument("--mix_prob", type=float, default=0.5)

    g = p.add_argument_group("model")
    g.add_argument("--model", required=True)
    g.add_argument("--no_pretrained", action="store_true")
    g.add_argument("--drop_rate", type=float, default=0.25)
    g.add_argument("--drop_path", type=float, default=0.15)
    g.add_argument("--freeze_layers", type=int, default=0,
                   help="freeze embeddings/stem and bottom N transformer layers (saves VRAM/time on ViT-L)")

    g = p.add_argument_group("loss / imbalance")
    g.add_argument("--focal_gamma", type=float, default=0.0)
    g.add_argument("--label_smoothing", type=float, default=0.05)
    g.add_argument("--class_weight_power", type=float, default=0.5)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=14)
    g.add_argument("--batch_size", type=int, default=8)
    g.add_argument("--grad_accum", type=int, default=2)
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--layer_decay", type=float, default=0.75)
    g.add_argument("--head_lr_mult", type=float, default=5.0)
    g.add_argument("--weight_decay", type=float, default=0.05)
    g.add_argument("--warmup_epochs", type=float, default=1.0)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--ema", type=float, default=0.0)
    g.add_argument("--soup_k", type=int, default=3, help="top-K epochs to weight-average into soup.pt")
    g.add_argument("--tta", default="orig,zoom5,zoom", help="comma-separated TTA views for final OOF eval")
    g.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto")

    g = p.add_argument_group("misc")
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--cache_dir", default="cache")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--split_seed", type=int, default=42)
    g.add_argument("--out", default="runs")
    g.add_argument("--tag", required=True)
    return p.parse_args(argv)


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def make_splits(args):
    train_df = read_split(args.train_dir, args.train_csv)
    val_df = read_split(args.val_dir, args.val_csv) if args.val_dir else None
    full = pd.concat([train_df, val_df], ignore_index=True) if val_df is not None else train_df
    if args.merge_val:
        return full.reset_index(drop=True), None
    groups = [_md5(p) for p in full.path]
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    tr_idx, va_idx = list(sgkf.split(full, full.label, groups))[args.fold]
    return full.iloc[tr_idx].reset_index(drop=True), full.iloc[va_idx].reset_index(drop=True)


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, bv_only=False):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss = F.cross_entropy(logits, labels).item()
    probs = logits.softmax(1).numpy()
    return probs, labels.numpy(), loss


@torch.no_grad()
def evaluate_tta(model, df, pre, img_size, tta_modes, device, amp_dtype, batch_size, workers):
    model.eval()
    preds_all = []
    for mode in tta_modes:
        ds = CXRDataset(df, pre, build_eval_transform(img_size, mode))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        logits = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits.append(model(x).float().cpu())
        preds_all.append(torch.cat(logits).softmax(1).numpy())
    return np.mean(preds_all, axis=0)


def average_state_dicts(sds: list[dict]) -> dict:
    avg = {}
    for k in sds[0].keys():
        v0 = sds[0][k]
        if v0.is_floating_point():
            avg[k] = sum(sd[k].float() for sd in sds).div(len(sds)).to(v0.dtype)
        else:
            avg[k] = v0.clone()
    return avg


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (args.amp_dtype in ("auto", "bf16") and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    num_classes = 2 if args.bv_only else NUM_CLASSES
    short_name = args.model.split(".")[0].replace("/", "--")
    name = datetime.now().strftime("%m%d-%H%M%S") + "_" + short_name
    if not args.merge_val and args.folds > 0:
        name += f"_f{args.fold}of{args.folds}"
    if args.merge_val:
        name += "_full"
    name += "_" + args.tag
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["num_classes"] = num_classes
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    train_df, val_df = make_splits(args)
    if args.bv_only:
        train_df = train_df[train_df.label.isin([1, 2])].copy().reset_index(drop=True)
        train_df["label"] = (train_df["label"] == 2).astype(int)  # 0=bacteria, 1=virus
        if val_df is not None:
            val_df = val_df[val_df.label.isin([1, 2])].copy().reset_index(drop=True)
            val_df["label"] = (val_df["label"] == 2).astype(int)

    if val_df is not None:
        val_df.to_csv(run_dir / "val_split.csv", index=False)

    # Build targets & sample weights (supports optional soft pseudo-labels!)
    tr_paths = train_df.path.tolist()
    tr_y = train_df.label.to_numpy()
    tr_targets = np.eye(num_classes, dtype=np.float32)[tr_y]
    if args.label_smoothing > 0:
        tr_targets = tr_targets * (1.0 - args.label_smoothing) + args.label_smoothing / num_classes
    tr_sample_w = np.ones(len(tr_paths), dtype=np.float32)

    if args.soft_pseudo_npy and args.soft_pseudo_csv:
        ps_df = pd.read_csv(args.soft_pseudo_csv)
        ps_probs = np.load(args.soft_pseudo_npy).astype(np.float32)
        ps_paths = [str(Path(args.soft_pseudo_dir) / f) for f in ps_df.iloc[:, 0].astype(str)]
        keep = ps_probs.max(1) >= args.pseudo_conf
        if args.pseudo_exclude_covid and not args.bv_only:
            keep = keep & (ps_probs[:, 3] < 0.05) & (ps_probs.argmax(1) != 3)
        if val_df is not None:
            val_md5 = {_md5(p) for p in val_df.path}
            keep = keep & np.array([_md5(p) not in val_md5 for p in ps_paths])
        if args.bv_only:
            # keep samples predicted as bacteria or virus with high combined B+V prob
            bv_sum = ps_probs[:, 1] + ps_probs[:, 2]
            keep = keep & (bv_sum >= 0.85)
            ps_t = np.stack([ps_probs[:, 1] / np.maximum(bv_sum, 1e-6),
                             ps_probs[:, 2] / np.maximum(bv_sum, 1e-6)], axis=1)
        else:
            ps_t = ps_probs
        idx_k = np.where(keep)[0]
        print(f"[soft_pseudo] +{len(idx_k)} images (weight={args.pseudo_weight})", flush=True)
        tr_paths = tr_paths + [ps_paths[i] for i in idx_k]
        tr_targets = np.concatenate([tr_targets, ps_t[idx_k]], axis=0)
        tr_sample_w = np.concatenate([tr_sample_w, np.full(len(idx_k), args.pseudo_weight, np.float32)], axis=0)

    print(f"[data] train={len(tr_paths)} (labeled={len(train_df)}) val={0 if val_df is None else len(val_df)} "
          f"num_classes={num_classes}", flush=True)

    pre = make_preprocessor(cfg, args.cache_dir)
    if pre.cache_dir is not None:
        t0 = time.time()
        all_p = tr_paths + ([] if val_df is None else val_df.path.tolist())
        warm_cache(pre, all_p, workers=max(4, args.workers))
        print(f"[cache] {len(all_p)} images ready in {pre.cache_dir} ({time.time()-t0:.0f}s)", flush=True)

    train_ds = SoftCXRDataset(tr_paths, tr_targets, tr_sample_w, pre,
                              build_train_transform(args.img_size, args.aug, args.hflip))
    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = None
    if val_df is not None:
        val_ds = CXRDataset(val_df, pre, build_eval_transform(args.img_size, "orig"))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    model = build_model(args.model, not args.no_pretrained, args.drop_rate, args.drop_path,
                        num_classes=num_classes)
    if args.freeze_layers > 0:
        import re
        pat = re.compile(r"(?:encoder\.layer|encoder\.layers|blocks|layers)\.(\d+)\.")
        n_frz = 0
        for pname, p in model.named_parameters():
            if any(k in pname for k in ("embeddings", "patch_embed", "pos_embed", "cls_token")):
                p.requires_grad_(False); n_frz += 1
            else:
                m = pat.search(pname)
                if m and int(m.group(1)) < args.freeze_layers:
                    p.requires_grad_(False); n_frz += 1
        print(f"[freeze] froze {n_frz} tensors (bottom {args.freeze_layers} layers + embeddings)", flush=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = ModelEMA(model, args.ema) if args.ema > 0 else None

    c_weights = class_weights(tr_y, num_classes, args.class_weight_power).to(device)
    print(f"[loss] class_weights={np.round(c_weights.cpu().numpy(), 3)}", flush=True)

    param_groups = build_llrd_param_groups(model, args.lr, args.weight_decay,
                                           args.layer_decay, args.head_lr_mult)
    lrs = sorted({g["lr"] for g in param_groups})
    print(f"[optim] {len(param_groups)} groups, LR range [{lrs[0]:.2e} .. {lrs[-1]:.2e}] "
          f"eff_bs={args.batch_size * args.grad_accum}", flush=True)
    optimizer = torch.optim.AdamW(param_groups)
    opt_steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    scheduler = cosine_with_warmup(optimizer, int(args.warmup_epochs * opt_steps_per_epoch),
                                   args.epochs * opt_steps_per_epoch)

    log_rows = []
    best_score = -1e9
    best_probs = None
    top_ckpts: list[tuple[float, int, dict]] = []  # (score, epoch, cpu_state_dict)
    last_ckpts: list[dict] = []  # for SWA in merge_val mode

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, target, sw) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            logp = F.log_softmax(logits.float(), dim=-1)
            loss_vec = -(target * logp * c_weights).sum(-1) * sw
            loss = loss_vec.mean() / args.grad_accum
            scaler.scale(loss).backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)

            running += loss_vec.sum().item()
            seen += x.size(0)

        row = {"epoch": epoch, "lr": optimizer.param_groups[-1]["lr"], "train_loss": running / seen}
        eval_model = ema.module if ema is not None else model

        if val_loader is not None:
            probs, labels, val_loss = evaluate(eval_model, val_loader, device, amp_dtype, args.bv_only)
            if args.bv_only:
                bv_auc = float(roc_auc_score(labels, probs[:, 1]))
                bv_acc = float((probs.argmax(1) == labels).mean())
                from sklearn.metrics import f1_score
                bv_f1 = float(f1_score(labels, probs.argmax(1), average="macro"))
                score = 0.6 * bv_auc + 0.4 * bv_f1 - 0.05 * val_loss
                row.update(val_loss=val_loss, acc=bv_acc, macro_f1=bv_f1, bv_auc=bv_auc, score=score)
                msg = f"val_loss {val_loss:.4f} acc {bv_acc:.4f} bvF1 {bv_f1:.4f} bvAUC {bv_auc:.4f} score {score:.4f}"
            else:
                m = compute_metrics(labels, probs.argmax(1))
                bv = np.isin(labels, [1, 2])
                bv_auc = float(roc_auc_score((labels[bv] == 2).astype(int),
                                             probs[bv, 2] / (probs[bv, 1] + probs[bv, 2] + 1e-12)))
                # Composite score: stable across folds (reduces 9-COVID-19 variance & penalizes overfitting!)
                score = 0.55 * m["macro_f1"] + 0.45 * bv_auc - 0.10 * val_loss
                row.update(val_loss=val_loss, bv_auc=bv_auc, score=score, **m)
                msg = (f"val_loss {val_loss:.4f} macroF1 {m['macro_f1']:.4f} bvAUC {bv_auc:.4f} score {score:.4f} | "
                       + " ".join(f"{c}:{m['f1_' + c]:.3f}" for c in CLASSES))

            improved = score > best_score + 1e-6
            if improved:
                best_score = score
                best_probs = probs
                torch.save({"model": eval_model.state_dict(), "args": cfg, "epoch": epoch,
                            "score": score}, run_dir / "best.pt")
                msg += " *"

            # Keep top-K epochs on CPU for Model Soup
            if args.soup_k > 1:
                if len(top_ckpts) < args.soup_k or score > top_ckpts[-1][0]:
                    sd_cpu = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
                    top_ckpts.append((score, epoch, sd_cpu))
                    top_ckpts.sort(key=lambda t: -t[0])
                    top_ckpts = top_ckpts[:args.soup_k]
        else:
            msg = ""
            if args.soup_k > 1 and epoch > args.epochs - args.soup_k:
                last_ckpts.append({k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()})

        row["time"] = time.time() - t0
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run_dir / "log.csv", index=False)
        print(f"[ep {epoch:02d}/{args.epochs}] tr_loss {row['train_loss']:.4f} | {msg} ({row['time']:.0f}s)", flush=True)

    # Wrap up
    if val_loader is None:
        if last_ckpts:
            soup_sd = average_state_dicts(last_ckpts)
            torch.save({"model": soup_sd, "args": cfg, "epoch": args.epochs}, run_dir / "soup.pt")
            torch.save({"model": soup_sd, "args": cfg, "epoch": args.epochs}, run_dir / "last.pt")
            print(f"[done] saved SWA soup ({len(last_ckpts)} epochs) -> {run_dir / 'last.pt'}", flush=True)
        else:
            torch.save({"model": eval_model.state_dict(), "args": cfg, "epoch": args.epochs}, run_dir / "last.pt")
        return run_dir

    tta_modes = [t.strip() for t in args.tta.split(",") if t.strip()]
    # Build Model Soup if soup_k > 1 and compare soup vs single best!
    if len(top_ckpts) > 1:
        soup_epochs = [e for _, e, _ in top_ckpts]
        soup_sd = average_state_dicts([sd for _, _, sd in top_ckpts])
        torch.save({"model": soup_sd, "args": cfg, "epochs": soup_epochs}, run_dir / "soup.pt")
        eval_model.load_state_dict(soup_sd)
        soup_probs = evaluate_tta(eval_model, val_df, pre, args.img_size, ["orig"], device, amp_dtype,
                                  args.batch_size * 2, args.workers)
        soup_tta = evaluate_tta(eval_model, val_df, pre, args.img_size, tta_modes, device, amp_dtype,
                                args.batch_size * 2, args.workers)
        np.save(run_dir / "val_probs_soup.npy", soup_probs)
        np.save(run_dir / "val_probs_soup_tta.npy", soup_tta)

        labels = val_df.label.to_numpy()
        if not args.bv_only:
            m_best = compute_metrics(labels, best_probs.argmax(1))
            m_soup = compute_metrics(labels, soup_probs.argmax(1))
            m_stta = compute_metrics(labels, soup_tta.argmax(1))
            bv = np.isin(labels, [1, 2])
            auc_b = roc_auc_score((labels[bv] == 2).astype(int), best_probs[bv, 2] / (best_probs[bv, 1] + best_probs[bv, 2] + 1e-12))
            auc_s = roc_auc_score((labels[bv] == 2).astype(int), soup_tta[bv, 2] / (soup_tta[bv, 1] + soup_tta[bv, 2] + 1e-12))
            print(f"\n[soup epochs={soup_epochs}] best={m_best['macro_f1']:.4f}(bvAUC={auc_b:.4f}) "
                  f"soup={m_soup['macro_f1']:.4f} soup+TTA={m_stta['macro_f1']:.4f}(bvAUC={auc_s:.4f})", flush=True)
        else:
            auc_b = roc_auc_score(labels, best_probs[:, 1])
            auc_s = roc_auc_score(labels, soup_tta[:, 1])
            print(f"\n[soup epochs={soup_epochs}] best bvAUC={auc_b:.4f} -> soup+TTA bvAUC={auc_s:.4f}", flush=True)
        # Save soup+TTA as val_probs.npy so summarize.py / stack.py directly use the best calibrated OOF!
        np.save(run_dir / "val_probs_best.npy", best_probs)
        np.save(run_dir / "val_probs.npy", soup_tta)
    else:
        np.save(run_dir / "val_probs.npy", best_probs)

    print(f"[done] run dir: {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
