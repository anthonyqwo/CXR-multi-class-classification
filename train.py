"""Train a CXR 4-class classifier (normal / bacteria / virus / COVID-19).

Examples
--------
# official train/val split
python train.py --train_dir data/train_images --train_csv data/train_data.csv \
                --val_dir data/val_images --val_csv data/val_data.csv

# 5-fold CV on train+val (run --fold 0..4), recommended because val has only ~7 COVID-19 cases
python train.py ... --folds 5 --fold 0

# final model on train+val, no validation
python train.py ... --merge_val --epochs 20
"""
from __future__ import annotations

import argparse
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
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as T

from dataset import (CLASSES, NUM_CLASSES, CXRDataset, build_eval_transform, build_train_transform,
                     make_preprocessor, read_split, warm_cache)
from losses import SoftFocalLoss, class_counts, class_weights, sample_weights
from models import ModelEMA, build_model
from utils import compute_metrics, format_confusion, seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--train_dir", required=True)
    g.add_argument("--train_csv", default=None, help="omit for folder-per-class layout")
    g.add_argument("--val_dir", default=None)
    g.add_argument("--val_csv", default=None)
    g.add_argument("--folds", type=int, default=0, help=">0: stratified K-fold on train+val")
    g.add_argument("--fold", type=int, default=0)
    g.add_argument("--merge_val", action="store_true", help="train on train+val, no validation")
    g.add_argument("--pseudo_dir", default=None, help="unlabeled images with pseudo labels (e.g. test)")
    g.add_argument("--pseudo_csv", default=None,
                   help="one-hot CSV of pseudo labels for --pseudo_dir; added to the training part only")

    g = p.add_argument_group("preprocessing / augmentation")
    g.add_argument("--img_size", type=int, default=384)
    g.add_argument("--clahe", type=float, default=0.0, help="CLAHE clip limit, 0 = off (try 2.0)")
    g.add_argument("--crop_border", type=float, default=0.0, help="crop this fraction from each side")
    g.add_argument("--aug", choices=["none", "light", "strong"], default="light")
    g.add_argument("--hflip", action="store_true", help="random horizontal flip (anatomically questionable)")
    g.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha, 0 = off")
    g.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha, 0 = off")
    g.add_argument("--mix_prob", type=float, default=0.5)

    g = p.add_argument_group("model")
    g.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    g.add_argument("--no_pretrained", action="store_true")
    g.add_argument("--drop_rate", type=float, default=0.2)
    g.add_argument("--drop_path", type=float, default=0.1)

    g = p.add_argument_group("loss / imbalance")
    g.add_argument("--focal_gamma", type=float, default=0.0, help="0 = cross-entropy")
    g.add_argument("--label_smoothing", type=float, default=0.05)
    g.add_argument("--class_weight_power", type=float, default=0.5,
                   help="loss weight = (inverse freq)^power; 0 = off")
    g.add_argument("--sampler_power", type=float, default=0.0,
                   help="oversampling with weight = count^-power; 0 = off, 0.5 = sqrt, 1 = balanced")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=25)
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.05)
    g.add_argument("--warmup_epochs", type=float, default=1.0)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--ema", type=float, default=0.0, help="EMA decay, 0 = off (try 0.999)")
    g.add_argument("--patience", type=int, default=0, help="early stopping patience, 0 = off")
    g.add_argument("--no_amp", action="store_true")
    g.add_argument("--amp_dtype", choices=["auto", "bf16", "fp16"], default="auto",
                   help="auto = bf16 when the GPU supports it; use fp16 on older GPUs such as the T4")

    g = p.add_argument_group("misc")
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--cache_dir", default="cache", help="resized-image cache; '' disables")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--val_list", default=None,
                   help="val_split.csv of an earlier run: use exactly its images as the validation fold "
                        "(K-fold splits can differ between machines / library versions)")
    g.add_argument("--split_seed", type=int, default=None,
                   help="seed for the K-fold split (default: --seed); keep it fixed to vary only the training seed")
    g.add_argument("--out", default="runs")
    g.add_argument("--tag", default="")
    return p.parse_args(argv)


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def add_pseudo(args, train_df, val_df):
    """Append pseudo-labeled images (e.g. confident test predictions) to the training part.
    Images byte-identical to a validation image are dropped so no label reaches the val fold."""
    if not args.pseudo_csv:
        return train_df
    ps = read_split(args.pseudo_dir, args.pseudo_csv)
    if (ps.label < 0).any():
        raise ValueError("--pseudo_csv rows must carry a label")
    if val_df is not None:
        val_md5 = {_md5(p) for p in val_df.path}
        keep = np.array([_md5(p) not in val_md5 for p in ps.path])
        if (~keep).any():
            print(f"[pseudo] dropped {(~keep).sum()} images identical to a validation image")
        ps = ps[keep]
    print(f"[pseudo] +{len(ps)} pseudo-labeled images: "
          + " ".join(f"{c}={(ps.label == i).sum()}" for i, c in enumerate(CLASSES)))
    return pd.concat([train_df, ps], ignore_index=True)


def make_splits(args):
    train_df, val_df = _make_splits(args)
    return add_pseudo(args, train_df, val_df), val_df


def _make_splits(args):
    train_df = read_split(args.train_dir, args.train_csv)
    val_df = read_split(args.val_dir, args.val_csv) if args.val_dir else None
    if (train_df.label < 0).any() or (val_df is not None and (val_df.label < 0).any()):
        raise ValueError("Training/validation data must be labeled")

    if args.folds > 0:
        full = pd.concat([train_df, val_df], ignore_index=True) if val_df is not None else train_df
        # Byte-identical images exist in the data; keep each duplicate group inside one fold so the
        # same picture never sits in both the training and validation part (would inflate CV).
        groups = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in full.path]
        n_dup = len(groups) - len(set(groups))
        sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed if args.split_seed is not None else args.seed)
        tr_idx, va_idx = list(sgkf.split(full, full.label, groups))[args.fold]
        if args.val_list:  # reproduce a split made elsewhere
            key = lambda q: "/".join(str(q).replace("\\", "/").split("/")[-2:])
            wanted = {key(q) for q in pd.read_csv(args.val_list).path}
            is_val = np.array([key(q) in wanted for q in full.path])
            if is_val.sum() != len(wanted):
                raise ValueError(f"--val_list: matched {is_val.sum()} of {len(wanted)} images")
            tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
        print(f"[cv] fold {args.fold}/{args.folds}, {n_dup} duplicate images grouped with their twin")
        return full.iloc[tr_idx].reset_index(drop=True), full.iloc[va_idx].reset_index(drop=True)
    if args.merge_val:
        full = pd.concat([train_df, val_df], ignore_index=True) if val_df is not None else train_df
        return full, None
    if val_df is None:
        raise ValueError("Provide --val_dir, or use --folds / --merge_val")
    return train_df, val_df


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
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


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if not args.no_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if args.amp_dtype != "auto":
            amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    # ---------------- run directory ----------------
    name = datetime.now().strftime("%m%d-%H%M%S") + "_" + args.model.split(".")[0]
    if args.folds > 0:
        name += f"_f{args.fold}of{args.folds}"
    if args.merge_val:
        name += "_full"
    if args.tag:
        name += "_" + args.tag
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # ---------------- data ----------------
    train_df, val_df = make_splits(args)
    counts = class_counts(train_df.label.to_numpy(), NUM_CLASSES).astype(int)
    print(f"[data] train={len(train_df)} " + " ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    if val_df is not None:
        vc = class_counts(val_df.label.to_numpy(), NUM_CLASSES).astype(int)
        print(f"[data] val={len(val_df)} " + " ".join(f"{c}={n}" for c, n in zip(CLASSES, vc)))
        val_df.to_csv(run_dir / "val_split.csv", index=False)

    pre = make_preprocessor(vars(args), args.cache_dir)
    if pre.cache_dir is not None:
        t0 = time.time()
        paths = train_df.path.tolist() + ([] if val_df is None else val_df.path.tolist())
        warm_cache(pre, paths, workers=max(4, args.workers))
        print(f"[cache] {len(paths)} images ready in {pre.cache_dir} ({time.time() - t0:.0f}s)")
    train_ds = CXRDataset(train_df, pre, build_train_transform(args.img_size, args.aug, args.hflip))
    sampler = None
    if args.sampler_power > 0:
        sw = sample_weights(train_df.label.to_numpy(), NUM_CLASSES, args.sampler_power)
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), replacement=True)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, drop_last=True, **loader_kw)
    val_loader = None
    if val_df is not None:
        val_ds = CXRDataset(val_df, pre, build_eval_transform(args.img_size))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    mixer = None
    mixers = []
    if args.mixup > 0:
        mixers.append(T.MixUp(alpha=args.mixup, num_classes=NUM_CLASSES))
    if args.cutmix > 0:
        mixers.append(T.CutMix(alpha=args.cutmix, num_classes=NUM_CLASSES))
    if mixers:
        mixer = T.RandomChoice(mixers)

    # ---------------- model / loss / optim ----------------
    model = build_model(args.model, not args.no_pretrained, args.drop_rate, args.drop_path)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = ModelEMA(model, args.ema) if args.ema > 0 else None

    weight = class_weights(train_df.label.to_numpy(), NUM_CLASSES, args.class_weight_power)
    print("[loss] class weights:", " ".join(f"{c}={w:.2f}" for c, w in zip(CLASSES, weight.tolist())))
    criterion = SoftFocalLoss(NUM_CLASSES, args.focal_gamma, weight, args.label_smoothing).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, int(args.warmup_epochs * steps_per_epoch),
                                   args.epochs * steps_per_epoch)

    # ---------------- training loop ----------------
    log_rows = []
    best = {"macro_f1": -1.0, "loss": float("inf"), "epoch": -1}
    best_probs = None
    bad_epochs = 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target = y
            if mixer is not None and rng.random() < args.mix_prob:
                x, target = mixer(x, y)
            x = x.to(memory_format=torch.channels_last)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = criterion(logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            running += loss.item() * x.size(0)
            seen += x.size(0)

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": running / seen}
        eval_model = ema.module if ema is not None else model

        if val_loader is not None:
            probs, labels, val_loss = evaluate(eval_model, val_loader, device, amp_dtype)
            m = compute_metrics(labels, probs.argmax(1))
            row.update(val_loss=val_loss, **m)
            improved = (m["macro_f1"] > best["macro_f1"] + 1e-6 or
                        (abs(m["macro_f1"] - best["macro_f1"]) <= 1e-6 and val_loss < best["loss"]))
            if improved:
                best = {"macro_f1": m["macro_f1"], "loss": val_loss, "epoch": epoch}
                best_probs = probs
                torch.save({"model": eval_model.state_dict(), "args": vars(args), "epoch": epoch,
                            "metrics": m}, run_dir / "best.pt")
                bad_epochs = 0
            else:
                bad_epochs += 1
            msg = (f"val_loss {val_loss:.4f} acc {m['acc']:.4f} macroF1 {m['macro_f1']:.4f} | "
                   + " ".join(f"{c}:{m['f1_' + c]:.3f}" for c in CLASSES) + (" *" if improved else ""))
        else:
            msg = ""
        row["time"] = time.time() - t0
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run_dir / "log.csv", index=False)
        print(f"[ep {epoch:02d}/{args.epochs}] train_loss {row['train_loss']:.4f} | {msg} ({row['time']:.0f}s)",
              flush=True)

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[early stop] no improvement for {args.patience} epochs")
            break

    # ---------------- wrap up ----------------
    if val_loader is None:
        torch.save({"model": eval_model.state_dict(), "args": vars(args), "epoch": epoch}, run_dir / "last.pt")
        print(f"[done] saved {run_dir / 'last.pt'}")
        return run_dir

    labels = val_df.label.to_numpy()
    np.save(run_dir / "val_probs.npy", best_probs)
    preds = best_probs.argmax(1)
    report = (f"best epoch {best['epoch']}  macro-F1 {best['macro_f1']:.4f}  val_loss {best['loss']:.4f}\n\n"
              + json.dumps(compute_metrics(labels, preds), indent=2) + "\n\n" + format_confusion(labels, preds))
    (run_dir / "val_report.txt").write_text(report)
    print("\n" + report)
    print(f"\n[done] run dir: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
