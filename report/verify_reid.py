"""Independent check of the Re-ID numbers quoted in the report. Run from the project root:

    python report/verify_reid.py

For every bacteria / virus image it finds the most similar labelled image in the OTHER folds and reports how often
the two labels agree, for the most confident 50 / 30 / 15 % of images (ranked by that top-1 similarity). It also
reports the bacteria-vs-virus AUC of a plain similarity-weighted 15-NN vote. Results go to report/figs/reid_check.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from stack import load_folds, load_oof
from stack_v2 import DEFAULT_REID_SPECS, compute_whitened_sims, load_emb_pair
from stack_v3_spatial import BANKS, REID_V4_SEP, REID_V5_JOINT, REID_V5_SEP, gpu_whiten_pair
from summarize import load_runs

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = load_runs(Path("runs"))
df = df[df.complete & (df.folds == 5)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
o0 = load_oof(df, "e2_clahe")
names, y = o0.index, o0.label.to_numpy()
fold = load_folds(df, "e2_clahe").to_numpy()
test_names = pd.read_csv("test_data_sample.csv").iloc[:, 0].astype(str).tolist()
n = len(y)
bv = np.isin(y, [1, 2])


def report(S, label):
    """S[i, j] = similarity of image i to labelled image j; same-fold entries must already be very negative."""
    order = np.argsort(-S, axis=1)[:, :15]
    s = np.take_along_axis(S, order, 1)
    lab = y[order]
    top1, nn = s[:, 0], lab[:, 0]
    idx = np.where(bv)[0]
    rank = idx[np.argsort(-top1[idx])]
    out = {"label": label, "all": float((nn[idx] == y[idx]).mean())}
    for frac in (0.5, 0.3, 0.15):
        k = int(round(frac * len(idx)))
        sel = rank[:k]
        out[f"top{int(frac * 100)}"] = float((nn[sel] == y[sel]).mean())
        out[f"n{int(frac * 100)}"] = k
        out[f"wrong{int(frac * 100)}"] = int((nn[sel] != y[sel]).sum())
    w = np.exp((s - s[:, :1]) / 0.05)
    pv = (w * (lab == 2)).sum(1) / np.maximum((w * np.isin(lab, [1, 2])).sum(1), 1e-9)
    out["knn15_bv_auc"] = float(roc_auc_score((y[idx] == 2).astype(int), pv[idx]))
    print(f"{label:34s} all {out['all']:.4f} | top50 {out['top50']:.4f} top30 {out['top30']:.4f} "
          f"(wrong {out['wrong30']}/{out['n30']}) top15 {out['top15']:.4f} | 15-NN bvAUC {out['knn15_bv_auc']:.4f}", flush=True)
    return out


def mask_same_fold(S):
    S = S.copy()
    for k in range(5):
        S[np.ix_(fold == k, fold == k)] = -10.0
    return S


res = []

# 1) stage 1: frozen RAD-DINO, mean-centred cosine similarity (stack.py --knn_feats)
z = np.load("features/rad-dino.npz")
f = pd.DataFrame(z["feats"], index=z["names"].astype(str)).loc[names].to_numpy(np.float32)
f = f - f.mean(0)
f /= np.linalg.norm(f, axis=1, keepdims=True)
res.append(report(mask_same_fold(f @ f.T), "RAD-DINO cosine (stage 1)"))

# 2) stack_v2: seven encoders, PCA-256 soft whitening (power 0.7)
S7 = compute_whitened_sims(DEFAULT_REID_SPECS, names, test_names, fold, power=0.7)[0]
res.append(report(mask_same_fold(S7), "7 encoders, PCA-256 whitening"))

# 3) stack_v3_spatial: mid-layer spatial embeddings, high-dimensional whitening, with and without CSLS
cache = {k: load_emb_pair(p, fk, names, test_names) for k, (p, fk) in BANKS.items()}


def build(sep_w, joint_w, d_sep, d_joint, mix_joint, csls=True, k_hub=15, beta=0.5):
    S = np.zeros((n, n), np.float32)
    Str = {k: np.zeros((int((fold != k).sum()),) * 2, np.float32) for k in range(5)}
    tw = sum(w for _, w in sep_w)
    for bn, w in sep_w:
        wn = (1.0 - mix_joint) * w / tw
        Fl = cache[bn][0]
        for k in range(5):
            tr, te = fold != k, fold == k
            Ztr, Zte = gpu_whiten_pair(Fl[tr], Fl[te], d_sep, dev)
            S[np.ix_(te, tr)] += wn * (Zte @ Ztr.T).cpu().numpy()
            Str[k] += wn * (Ztr @ Ztr.T).cpu().numpy()
    if mix_joint > 0:
        tj = sum(w for _, w in joint_w)
        for k in range(5):
            tr, te = fold != k, fold == k
            ptr, pte = [], []
            for bn, w in joint_w:
                Fl = cache[bn][0]
                mu = Fl[tr].mean(0, keepdims=True)
                a = Fl[tr] - mu
                a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
                b = Fl[te] - mu
                b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
                ptr.append(a * np.sqrt(w / tj))
                pte.append(b * np.sqrt(w / tj))
            Ztr, Zte = gpu_whiten_pair(np.hstack(ptr), np.hstack(pte), d_joint, dev)
            S[np.ix_(te, tr)] += mix_joint * (Zte @ Ztr.T).cpu().numpy()
            Str[k] += mix_joint * (Ztr @ Ztr.T).cpu().numpy()
    out = np.full_like(S, -10.0)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        B = S[np.ix_(te, tr)]
        if csls:
            np.fill_diagonal(Str[k], -1.0)
            r_te = np.sort(B, axis=1)[:, -k_hub:].mean(1, keepdims=True)
            r_tr = np.sort(Str[k], axis=1)[:, -k_hub:].mean(1, keepdims=True).T
            B = B - beta * (r_te + r_tr)
        out[np.ix_(te, tr)] = B
    return out


LAST_ONLY = [("rad-sp2-l12", 0.5), ("maira-sp2-l12", 0.5)]
MID_ONLY = [("rad-sp2-l9", 0.5), ("maira-sp2-l9", 0.5)]
res.append(report(build(LAST_ONLY, LAST_ONLY, 2048, 3072, 0.0, csls=False), "2x2 last layer (L12), no CSLS"))
res.append(report(build(MID_ONLY, MID_ONLY, 2048, 3072, 0.0, csls=False), "2x2 middle layer (L9), no CSLS"))
res.append(report(build(MID_ONLY, MID_ONLY, 2048, 3072, 0.0, csls=True), "2x2 middle layer (L9), CSLS"))
res.append(report(build(REID_V4_SEP, REID_V4_SEP, 2048, 3072, 0.50), "Re-ID v4 (stack_v9)"))
res.append(report(build(REID_V5_SEP, REID_V5_JOINT, 2560, 3072, 0.35, csls=False), "Re-ID v5 without CSLS"))
res.append(report(build(REID_V5_SEP, REID_V5_JOINT, 2560, 3072, 0.35), "Re-ID v5 (stack_v10)"))

Path("report/figs/reid_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
