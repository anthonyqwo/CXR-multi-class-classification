"""Summarise all training runs into one comparison table (handy for the report).

python summarize.py                 # every tag, every fold
python summarize.py --folds 0 4     # compare tags only on folds they all share
python summarize.py --csv runs/summary.csv

Per tag it reports the mean over folds of: best-epoch macro-F1 / per-class F1,
last-epoch macro-F1, and best epoch. When a tag has all K folds, the pooled
out-of-fold (OOF) macro-F1 is reported as well -- the most reliable number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import CLASSES
from utils import compute_metrics


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(root.iterdir()):
        cfg_p, log_p = d / "config.json", d / "log.csv"
        if not (cfg_p.exists() and log_p.exists()):
            continue
        cfg = json.loads(cfg_p.read_text())
        log = pd.read_csv(log_p)
        if "macro_f1" not in log.columns or log.empty:
            continue  # merge_val runs have no validation
        complete = len(log) == cfg["epochs"] or cfg.get("patience", 0) > 0
        best = log.sort_values(["macro_f1", "val_loss"], ascending=[False, True]).iloc[0]
        rows.append({
            "tag": cfg.get("tag") or "-", "fold": cfg["fold"] if cfg["folds"] else -1,  # -1 = official val
            "folds": cfg["folds"], "model": cfg["model"].split(".")[0], "img": cfg["img_size"],
            "best_f1": best.macro_f1, "best_ep": int(best.epoch), "last_f1": log.macro_f1.iloc[-1],
            **{f"f1_{c}": best.get(f"f1_{c}", np.nan) for c in CLASSES},
            "epochs_done": len(log), "complete": complete, "dir": str(d),
        })
    return pd.DataFrame(rows)


def pooled_oof(group: pd.DataFrame):
    if group.folds.iloc[0] == 0 or group.fold.nunique() != group.folds.iloc[0]:
        return np.nan
    parts = [(pd.read_csv(Path(d) / "val_split.csv"), np.load(Path(d) / "val_probs.npy"))
             for d in group.dir if (Path(d) / "val_probs.npy").exists()]
    if len(parts) != group.folds.iloc[0]:
        return np.nan
    labels = np.concatenate([v.label.to_numpy() for v, _ in parts])
    probs = np.concatenate([p for _, p in parts])
    return compute_metrics(labels, probs.argmax(1))["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--folds", nargs="*", type=int, help="restrict to these CV folds")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    df = load_runs(Path(args.runs))
    if df.empty:
        print("no runs found"); return
    df = df[df.complete]
    if args.folds:
        df = df[df.fold.isin(args.folds)]
    # keep the latest run if a (tag, fold) was trained twice
    df = df.sort_values("dir").groupby(["tag", "fold"], as_index=False).last()

    agg = df.groupby("tag").agg(
        model=("model", "first"), img=("img", "first"), n_folds=("fold", "nunique"),
        best_f1=("best_f1", "mean"), best_f1_std=("best_f1", "std"), last_f1=("last_f1", "mean"),
        best_ep=("best_ep", "mean"), **{f"f1_{c}": (f"f1_{c}", "mean") for c in CLASSES},
    )
    agg["oof_f1"] = [pooled_oof(df[df.tag == t]) for t in agg.index]
    agg = agg.sort_values("best_f1", ascending=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("== per run ==")
    print(df.drop(columns=["dir", "complete", "folds", "epochs_done"]).round(4).to_string(index=False))
    print("\n== per tag (mean over folds) ==")
    print(agg.round(4).to_string())
    if args.csv:
        agg.round(4).to_csv(args.csv)
        print(f"\nsaved {args.csv}")


if __name__ == "__main__":
    main()
