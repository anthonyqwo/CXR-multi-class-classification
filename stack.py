"""Stack several 5-fold CV models with a class-balanced logistic regression.

Each model ("tag") contributes its pooled out-of-fold (OOF) softmax probabilities on train+val
and its test probabilities (the average of its 5 fold models, from predict.py without
--tune_bias). The meta-model sees the concatenated log-probabilities. It is a better-calibrated
alternative to "average + per-class bias", and its score is estimated with nested CV:
the meta-model is re-fit on 4/5 of the OOF set and scored on the held-out 1/5.

python predict.py --runs runs/*of5_<tag> --tta orig --out submissions/per_tag/<tag>.csv   # per tag
python stack.py --tags e2_clahe e4_effb3 e6_effv2s_clahe --C 0.01 --out submissions/stack.csv

Optional nearest-neighbour features (--knn_feats, from embed.py): for every image, the labels of
its most similar labelled images in a frozen embedding space, weighted by similarity. For a
labelled image the neighbours come from the other folds only, so the features are out-of-fold
like the model probabilities; for a test image they come from all labelled images. Images of
the same patient are very similar and almost always share a label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from dataset import CLASSES
from summarize import load_runs
from utils import compute_metrics, format_confusion


def load_oof(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if "~" in tag:  # "a~b": average the OOF probabilities of several seeds of the same model (same folds)
        parts = [load_oof(df, t) for t in tag.split("~")]
        out = parts[0].copy()
        cols = [c for c in out.columns if c != "label"]
        out[cols] = np.mean([p[cols].to_numpy() for p in parts], axis=0)
        return out
    runs = df[df.tag == tag]
    if runs.fold.nunique() != runs.folds.iloc[0]:
        raise ValueError(f"{tag}: needs all {runs.folds.iloc[0]} folds, found {sorted(runs.fold)}")
    parts = []
    for d in runs.dir:
        v = pd.read_csv(Path(d) / "val_split.csv")
        parts.append(pd.DataFrame(np.load(Path(d) / "val_probs.npy"), index=v.filename)
                     .assign(label=v.label.values))
    return pd.concat(parts).sort_index()


def load_folds(df: pd.DataFrame, tag: str) -> pd.Series:
    """file name -> validation fold, for one 5-fold tag."""
    runs = df[df.tag == tag.split("~")[0]]
    return pd.concat([pd.Series(int(f), index=pd.read_csv(Path(d) / "val_split.csv").filename)
                      for d, f in zip(runs.dir, runs.fold)]).sort_index()


def knn_features(sim: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """sim: (queries, labelled) cosine similarities, excluded neighbours set to -1.

    Three similarity-weighted label votes (nearest 1, 5 and 20 neighbours) as log-fractions,
    plus the similarity of the nearest neighbour.
    """
    order = np.argsort(-sim, axis=1)[:, :20]
    s, lab = np.take_along_axis(sim, order, 1), labels[order]
    out = []
    for k, temp in ((1, 1.0), (5, 0.05), (20, 0.05)):
        w = np.exp((s[:, :k] - s[:, :1]) / temp)
        votes = np.stack([(w * (lab[:, :k] == c)).sum(1) for c in range(n_classes)], 1)
        out.append(np.log(votes / votes.sum(1, keepdims=True) + 1e-2))
    return np.concatenate(out + [s[:, :1]], axis=1)


def load_knn(path: str, labelled: pd.Index, test_names: list[str], y: np.ndarray, fold: np.ndarray,
             n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """`path` may join several .npz files with "+": their cosine similarities are averaged."""
    sim_lab, sim_test = 0, 0
    parts = path.split("+")
    for part in parts:
        z = np.load(part)
        feats = pd.DataFrame(z["feats"], index=z["names"].astype(str))
        lab, test = feats.loc[labelled].to_numpy(np.float32), feats.loc[test_names].to_numpy(np.float32)
        mean = lab.mean(0)
        lab, test = lab - mean, test - mean
        lab /= np.linalg.norm(lab, axis=1, keepdims=True)
        test /= np.linalg.norm(test, axis=1, keepdims=True)
        sim_lab = sim_lab + lab @ lab.T / len(parts)
        sim_test = sim_test + test @ lab.T / len(parts)
    for k in np.unique(fold):  # a labelled image may only use neighbours from the other folds
        sim_lab[np.ix_(fold == k, fold == k)] = -1
    return knn_features(sim_lab, y, n_classes), knn_features(sim_test, y, n_classes)


def features(probs: list[np.ndarray]) -> np.ndarray:
    return np.log(np.concatenate(probs, axis=1) + 1e-7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--test_probs_dir", default="submissions/per_tag")
    ap.add_argument("--test_csv", default="test_data_sample.csv")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=5, help="nested-CV repeats for the estimate")
    ap.add_argument("--knn_feats", nargs="*", default=[], help=".npz files from embed.py; adds neighbour-label features (a.npz+b.npz averages their similarities)")
    ap.add_argument("--knn_gate", type=float, default=0.0,
                    help=">0: also add the neighbour votes multiplied by max(0, similarity - knn_gate)")
    ap.add_argument("--knn_scale", type=float, default=0.3, help="weight of the neighbour features in the meta-model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_runs(Path(args.runs))
    df = df[df.complete & (df.folds > 0)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
    oofs = [load_oof(df, t) for t in args.tags]
    for o in oofs[1:]:
        assert (o.index == oofs[0].index).all(), "tags were trained on different fold splits"
    y = oofs[0].label.to_numpy()
    X = features([o[list(range(len(CLASSES)))].to_numpy() for o in oofs])
    test_df = pd.read_csv(args.test_csv)
    Xt = features([np.load(Path(args.test_probs_dir) / f"{t}.probs.npy") for t in args.tags])
    if args.knn_feats:
        fold = load_folds(df, args.tags[0])
        assert (fold.index == oofs[0].index).all()
        for path in args.knn_feats:
            a, b = load_knn(path, oofs[0].index, test_df.iloc[:, 0].astype(str).tolist(), y, fold.to_numpy(), len(CLASSES))
            if args.knn_gate > 0:  # extra copy of the votes that only counts when the nearest neighbour is very similar
                a, b = (np.concatenate([f, f[:, :-1] * np.clip(f[:, -1:] - args.knn_gate, 0, None) * 5], 1) for f in (a, b))
            X, Xt = np.concatenate([X, a * args.knn_scale], 1), np.concatenate([Xt, b * args.knn_scale], 1)

    def meta():
        return LogisticRegression(C=args.C, max_iter=5000, class_weight="balanced")

    avg = np.mean([o[list(range(len(CLASSES)))].to_numpy() for o in oofs], axis=0)
    print(f"[oof] simple average macro-F1 {compute_metrics(y, avg.argmax(1))['macro_f1']:.4f}")
    scores, pred = [], np.zeros(len(y), int)
    for seed in range(args.repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            pred[te] = meta().fit(X[tr], y[tr]).predict(X[te])
        scores.append(compute_metrics(y, pred)["macro_f1"])
    print(f"[oof] stacked macro-F1 (nested CV, {args.repeats} repeats) "
          f"{np.mean(scores):.4f} +- {np.std(scores):.4f}")
    print(format_confusion(y, pred))

    model = meta().fit(X, y)
    probs = model.predict_proba(Xt)
    preds = probs.argmax(1)
    sub = pd.DataFrame({"new_filename": test_df.iloc[:, 0]})
    for i, c in enumerate(CLASSES):
        sub[c] = (preds == i).astype(int)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    np.save(out.with_suffix(".probs.npy"), probs)
    out.with_suffix(".json").write_text(json.dumps({
        "tags": args.tags, "C": args.C, "knn_feats": args.knn_feats, "knn_scale": args.knn_scale, "knn_gate": args.knn_gate, "nested_cv_macro_f1": float(np.mean(scores)),
        "pred_distribution": {c: int((preds == i).sum()) for i, c in enumerate(CLASSES)}}, indent=2))
    print(f"[done] wrote {out}  prediction counts: "
          + " ".join(f"{c}={int((preds == i).sum())}" for i, c in enumerate(CLASSES)))


if __name__ == "__main__":
    main()
