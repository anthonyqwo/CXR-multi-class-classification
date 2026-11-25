"""Retrain a train_v2.py 5-fold tag on all train+val data (--merge_val) with optimal SWA epochs,
predict test set with 3-view TTA, and blend 50/50 with the 5-fold test probabilities!"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_full_tag(tag: str, epochs_override: int | None = None):
    f0 = sorted(glob.glob(f"runs/*_f0of5_{tag}"))[-1]
    cfg = json.loads((Path(f0) / "config.json").read_text())
    args = []
    skip = {"folds", "fold", "merge_val", "tag", "out", "split_seed", "num_classes"}
    for k, v in cfg.items():
        if k in skip or v is None:
            continue
        if k == "epochs" and epochs_override is not None:
            v = epochs_override
        if isinstance(v, bool):
            if v:
                args.append(f"--{k}")
        else:
            args += [f"--{k}", str(v)]
    full_tag = f"full_{tag}"
    args += ["--merge_val", "--tag", full_tag]

    existing = [d for d in sorted(glob.glob(f"runs/*_full_{full_tag}")) if (Path(d) / "last.pt").exists()]
    if existing:
        run_dir = existing[-1]
        print(f"[skip] full train {tag} already exists: {run_dir}", flush=True)
    else:
        cmd = [sys.executable, "train_v2.py", *args]
        print(f"\n[full_train_v2] {' '.join(cmd)}", flush=True)
        log_p = Path("runs") / f"{full_tag}.log"
        with open(log_p, "w", encoding="utf-8") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[ERR] {full_tag} failed ({r.returncode})", flush=True)
            sys.exit(r.returncode)
        run_dir = sorted(glob.glob(f"runs/*_full_{full_tag}"))[-1]

    tta = cfg.get("tta", "orig,zoom5,zoom")
    full_out = Path("submissions/per_tag_v2_full") / f"{tag}.csv"
    full_out.parent.mkdir(parents=True, exist_ok=True)
    if not full_out.with_suffix(".probs.npy").exists():
        if cfg.get("bv_only", False):
            import torch
            sys.path.insert(0, ".")
            from dataset import read_split
            from predict import load_run, predict_probs
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_df = read_split("data/test_images", "test_data_sample.csv")
            tta_list = [t.strip() for t in tta.split(",") if t.strip()]
            model, rcfg, _ = load_run(Path(run_dir), device, "auto")
            tp = predict_probs(model, rcfg, test_df, tta_list, device, 32, 6, "cache")
            np.save(full_out.with_suffix(".probs.npy"), tp)
            print(f"[predict_full_bv] saved {full_out.with_suffix('.probs.npy')} shape={tp.shape}", flush=True)
            del model
            torch.cuda.empty_cache()
        else:
            cmd = [
                sys.executable, "predict.py",
                "--runs", run_dir,
                "--test_dir", "data/test_images",
                "--test_csv", "test_data_sample.csv",
                "--tta", tta,
                "--ckpt", "auto",
                "--out", str(full_out),
            ]
            print(f"[predict_full] {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                sys.exit(r.returncode)

    # 50/50 blend with 5-fold TTA test probs
    fold_p = np.load(Path("submissions/per_tag_v2") / f"{tag}.probs.npy")
    full_p = np.load(full_out.with_suffix(".probs.npy"))
    blend_dir = Path("submissions/per_tag_v2_blend")
    blend_dir.mkdir(parents=True, exist_ok=True)
    np.save(blend_dir / f"{tag}.probs.npy", 0.5 * fold_p + 0.5 * full_p)
    print(f"[blend] saved {blend_dir / (tag + '.probs.npy')}", flush=True)


def main():
    # Usage: python tools/full_train_v2.py tag1[:epochs] tag2[:epochs] ...
    for spec in sys.argv[1:]:
        if ":" in spec:
            tag, ep = spec.split(":")
            run_full_tag(tag, int(ep))
        else:
            run_full_tag(spec, None)


if __name__ == "__main__":
    main()
