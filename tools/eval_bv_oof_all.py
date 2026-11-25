"""Run strictly OOF 3-view TTA inference of a --bv_only 5-fold tag on ALL 4 classes of each fold's val set,
so OOF and test have identical feature distributions for Normal/COVID images when stacking via --bv_tags."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from dataset import read_split
from predict import load_run, predict_probs
from stack import load_folds, load_oof
from summarize import load_runs


def main():
    tag = sys.argv[1]
    tta = [t.strip() for t in (sys.argv[2] if len(sys.argv) > 2 else "orig,zoom5,zoom").split(",")]
    df_runs = load_runs(Path("runs"))
    df_runs = df_runs[df_runs.complete & (df_runs.folds == 5)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
    ref_oof = load_oof(df_runs, "e36b_maira518_clean_llrd")
    ref_fold = load_folds(df_runs, "e36b_maira518_clean_llrd").to_numpy()

    tr_df = read_split("data/train_images", "data/train_data.csv")
    va_df = read_split("data/val_images", "data/val_data.csv")
    all_df = pd.concat([tr_df, va_df], ignore_index=True)
    all_df["fname"] = all_df["filename"].astype(str)
    all_df = all_df.set_index("fname").loc[ref_oof.index].reset_index()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f in range(5):
        run_dir = Path(sorted(glob.glob(f"runs/*_f{f}of5_{tag}"))[-1])
        out_p = run_dir / "val_probs_all.npy"
        out_csv = run_dir / "val_split_all.csv"
        if out_p.exists() and out_csv.exists():
            print(f"[skip] fold {f} already done", flush=True)
            continue
        sub_df = all_df[ref_fold == f].copy().reset_index(drop=True)
        model, cfg, ckpt_p = load_run(run_dir, device, "auto")
        probs = predict_probs(model, cfg, sub_df, tta, device, 32, 6, "cache")
        np.save(out_p, probs)
        sub_df[["filename", "label"]].to_csv(out_csv, index=False)
        print(f"[fold {f}] saved {out_p} shape={probs.shape}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
