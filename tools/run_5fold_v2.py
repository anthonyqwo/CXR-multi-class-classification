"""Run all 5 folds of a train_v2.py experiment, report pooled OOF metrics, and generate TTA test probs."""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from utils import compute_metrics


def main():
    # Usage: python tools/run_5fold_v2.py <tag> [extra train_v2.py args...]
    tag = sys.argv[1]
    extra = sys.argv[2:]
    bv_only = "--bv_only" in extra
    tta = "orig,zoom5,zoom"
    for i, a in enumerate(extra):
        if a == "--tta" and i + 1 < len(extra):
            tta = extra[i + 1]

    run_dirs = []
    for f in range(5):
        existing = [
            d for d in sorted(glob.glob(f"runs/*_f{f}of5_{tag}"))
            if (Path(d) / "val_probs.npy").exists()
        ]
        if existing:
            print(f"[skip] {tag} fold {f} already done: {existing[-1]}", flush=True)
            run_dirs.append(existing[-1])
            continue
        cmd = [sys.executable, "train_v2.py", "--folds", "5", "--fold", str(f), "--tag", tag, *extra]
        print(f"\n[run] fold {f}/5: {' '.join(cmd)}", flush=True)
        log_path = Path("runs") / f"{tag}_f{f}.log"
        with open(log_path, "w", encoding="utf-8") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[ERR] {tag} fold {f} failed with code {r.returncode}. Tail of {log_path}:", flush=True)
            print("\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-30:]), flush=True)
            sys.exit(r.returncode)
        done = sorted(glob.glob(f"runs/*_f{f}of5_{tag}"))[-1]
        run_dirs.append(done)
        # Print summary line from fold
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for ln in lines[-5:]:
            print("  " + ln, flush=True)

    # Pooled OOF summary
    parts_p, parts_y = [], []
    for d in run_dirs:
        v = pd.read_csv(Path(d) / "val_split.csv")
        p = np.load(Path(d) / "val_probs.npy")
        parts_p.append(p)
        parts_y.append(v.label.to_numpy())
    probs = np.concatenate(parts_p, axis=0)
    labels = np.concatenate(parts_y, axis=0)
    if bv_only:
        auc = roc_auc_score(labels, probs[:, 1])
        acc = (probs.argmax(1) == labels).mean()
        print(f"\n=== POOLED 5-FOLD OOF ({tag}, bv_only): bvAUC={auc:.4f} bvAcc={acc:.4f} ===", flush=True)
    else:
        m = compute_metrics(labels, probs.argmax(1))
        bv = np.isin(labels, [1, 2])
        bv_auc = roc_auc_score((labels[bv] == 2).astype(int),
                               probs[bv, 2] / (probs[bv, 1] + probs[bv, 2] + 1e-12))
        print(f"\n=== POOLED 5-FOLD OOF ({tag}): macroF1={m['macro_f1']:.4f} bvAUC={bv_auc:.4f} "
              f"[N:{m['f1_normal']:.4f} B:{m['f1_bacteria']:.4f} V:{m['f1_virus']:.4f} C:{m['f1_COVID-19']:.4f}] ===",
              flush=True)

    # Predict test set with TTA using soup.pt (auto picks soup.pt > best.pt)
    out_csv = Path("submissions/per_tag_v2") / f"{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if bv_only:
        # Custom binary test inference
        import torch
        from dataset import CXRDataset, build_eval_transform, make_preprocessor, read_split
        from predict import load_run, predict_probs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_df = read_split("data/test_images", "test_data_sample.csv")
        tta_list = [t.strip() for t in tta.split(",") if t.strip()]
        t_probs = []
        for d in run_dirs:
            model, cfg, ckpt_p = load_run(Path(d), device, "auto")
            t_probs.append(predict_probs(model, cfg, test_df, tta_list, device, 32, 6, "cache"))
            del model
            torch.cuda.empty_cache()
        tp = np.mean(t_probs, axis=0)
        np.save(out_csv.with_suffix(".probs.npy"), tp)
        print(f"[done] saved binary test probs -> {out_csv.with_suffix('.probs.npy')} shape={tp.shape}", flush=True)
    else:
        cmd = [
            sys.executable, "predict.py",
            "--runs", *run_dirs,
            "--test_dir", "data/test_images",
            "--test_csv", "test_data_sample.csv",
            "--tta", tta,
            "--ckpt", "auto",
            "--out", str(out_csv),
        ]
        print(f"[predict] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
