"""Predict the test set and write a Kaggle submission (one-hot CSV).

Supports ensembling several runs, test-time augmentation and optional per-class
decision-bias tuning on the runs' validation / out-of-fold predictions.

Example
-------
python predict.py --runs runs/1113-xxxx_convnext_tiny_f0of5 runs/... \
                  --test_dir data/test_images --test_csv test_data_sample.csv \
                  --tta orig,zoom --tune_bias --out submissions/convnext_cv5.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CLASSES, CXRDataset, build_eval_transform, make_preprocessor, read_split
from models import build_model
from utils import compute_metrics, format_confusion, tune_class_bias


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories from train.py")
    p.add_argument("--test_dir", required=True)
    p.add_argument("--test_csv", required=True, help="defines output row order (e.g. test_data_sample.csv)")
    p.add_argument("--tta", default="orig", help="comma list of: orig, zoom, hflip")
    p.add_argument("--tune_bias", action="store_true",
                   help="tune per-class log-prob offsets on validation / OOF predictions")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cache_dir", default="cache", help="resized-image cache; '' disables")
    p.add_argument("--ckpt", default="auto", help="checkpoint filename: auto (soup.pt > best.pt > last.pt), best.pt, soup.pt, last.pt")
    p.add_argument("--out", default="submission.csv")
    return p.parse_args()


def load_run(run_dir: Path, device, ckpt_name: str = "auto"):
    if ckpt_name == "auto":
        for cand in ("soup.pt", "best.pt", "last.pt"):
            if (run_dir / cand).exists():
                ckpt_path = run_dir / cand
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    else:
        ckpt_path = run_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    num_classes = int(cfg.get("num_classes", len(CLASSES)))
    model = build_model(cfg["model"], pretrained=False, drop_rate=0.0, drop_path_rate=0.0,
                        num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval().to(memory_format=torch.channels_last)
    return model, cfg, ckpt_path


@torch.no_grad()
def predict_probs(model, cfg, df, tta, device, batch_size, workers, cache_dir=None):
    pre = make_preprocessor(cfg, cache_dir)
    out = []
    for mode in tta:
        ds = CXRDataset(df, pre, build_eval_transform(cfg["img_size"], mode))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        probs = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                probs.append(model(x).float().softmax(1).cpu())
        out.append(torch.cat(probs).numpy())
    return np.mean(out, axis=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta = [t.strip() for t in args.tta.split(",") if t.strip()]
    test_df = read_split(args.test_dir, args.test_csv)

    test_probs, val_parts = [], []
    for run in args.runs:
        run_dir = Path(run)
        model, cfg, ckpt_path = load_run(run_dir, device, args.ckpt)
        print(f"[run] {ckpt_path}  model={cfg['model']} img={cfg['img_size']}")
        test_probs.append(predict_probs(model, cfg, test_df, tta, device, args.batch_size, args.workers,
                                        args.cache_dir))
        if args.tune_bias:
            val_csv = run_dir / "val_split.csv"
            if not val_csv.exists():
                raise FileNotFoundError(f"{val_csv} missing (run trained with --merge_val?)")
            vdf = pd.read_csv(val_csv)
            vp = predict_probs(model, cfg, vdf, tta, device, args.batch_size, args.workers, args.cache_dir)
            val_parts.append((vdf, vp))
            m = compute_metrics(vdf.label.to_numpy(), vp.argmax(1))
            print(f"       val macro-F1 (with TTA) {m['macro_f1']:.4f}")
        del model
        torch.cuda.empty_cache()

    probs = np.mean(test_probs, axis=0)
    bias = np.zeros(len(CLASSES))

    if args.tune_bias:
        # Average every held-out prediction per image. This covers one val set shared by several
        # models, disjoint K-fold OOF, and several models x K folds (OOF of the model ensemble).
        long = pd.concat([pd.DataFrame(p, columns=CLASSES).assign(filename=v.filename.values,
                                                                  label=v.label.values)
                          for v, p in val_parts], ignore_index=True)
        n_pred = long.groupby("filename").size()
        if n_pred.nunique() > 1:
            print(f"[warn] val images are covered by {sorted(n_pred.unique())} runs each; "
                  "bias tuning mixes ensemble sizes")
        agg = long.groupby("filename", sort=True).agg({**{c: "mean" for c in CLASSES}, "label": "first"})
        vprobs = agg[CLASSES].to_numpy()
        labels = agg.label.to_numpy()
        bias, before, after = tune_class_bias(vprobs, labels)
        print(f"[bias] {len(labels)} val samples: macro-F1 {before:.4f} -> {after:.4f}  "
              + " ".join(f"{c}={b:+.2f}" for c, b in zip(CLASSES, bias)))
        print(format_confusion(labels, (np.log(np.clip(vprobs, 1e-8, 1)) + bias).argmax(1)))

    preds = (np.log(np.clip(probs, 1e-8, 1)) + bias).argmax(1)
    sub = pd.DataFrame({"new_filename": test_df.filename})
    for i, c in enumerate(CLASSES):
        sub[c] = (preds == i).astype(int)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    np.save(out.with_suffix(".probs.npy"), probs)
    out.with_suffix(".json").write_text(json.dumps(
        {"runs": args.runs, "tta": tta, "bias": bias.tolist(),
         "pred_distribution": {c: int((preds == i).sum()) for i, c in enumerate(CLASSES)}}, indent=2))
    print(f"[done] wrote {out}  prediction counts: "
          + " ".join(f"{c}={int((preds == i).sum())}" for i, c in enumerate(CLASSES)))


if __name__ == "__main__":
    main()
