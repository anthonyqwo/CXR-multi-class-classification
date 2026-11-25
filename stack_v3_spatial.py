"""Production Re-ID v4/v5 Spatial-Biometric Meta-Stacker (Public LB 0.93640 #1, 0% Pseudo-Labels).

Combines:
1. 11-12 clean 5-fold + full-data SWA CXR vision models + 2 binary Bacteria-vs-Virus specialists (e38, e44)
2. GPU Dual-Gram Shrinkage-ZCA Whitening (p=1.0, reg=0.005) + Scale-Matched Exact CSLS Hubness Reduction
   across multi-layer 3x3 & 2x2 spatial quadrant biometric embeddings (RAD-DINO L6/L9, MAIRA-2 L6/L9,
   XraySigLIP-L L16/L20, DINOv2-Large L16, RAD-DINO-lc, Xray-DINOv2).
3. Strictly OOF per-model temperature calibration + 8-foundation/zero-shot/radiomics frozen probe
4. 4-family NBV meta-ensemble (LR + 4-seed LightGBM + 2-seed HistGBDT + 5-seed OOF-calibrated MLP)
   + adaptive CSLS transductive peer smoothing
5. Decoupled clean binary COVID-19 head (5 clean CNNs + similarity margin + prior-matched quantile 0.010152)

Usage:
  python stack_v3_spatial.py
Generates:
  - submissions/stack_v9_clean11_bv2_reidv4.csv       (Re-ID v4 2x2 spatial, LB 0.93445)
  - submissions/stack_v10_clean11_bv2_reidv5.csv      (Re-ID v5 3x3+2x2 spatial, LB 0.93636)
  - submissions/stack_v10_v9_superblend.csv           (Champion superblend, LB 0.93640 #1)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from dataset import CLASSES
from stack import features, load_folds, load_oof
from stack_v2 import (
    DEFAULT_REID_SPECS,
    build_mega_frozen_probe,
    calibrate_pair,
    compute_whitened_sims,
    covid_knn_features,
    extra_model_features,
    load_emb_pair,
    load_test_prob,
)
from summarize import load_runs
from utils import compute_metrics

TAGS11 = [
    "e2_clahe", "e4_effb3", "e6_effv2s_clahe", "e8_effv2s_clahe448", "e15_effv2m448",
    "e33_effv2m448_lc_rad3ch", "e34_xraysiglip512_clean", "e35_swinb384_rad3ch",
    "e36_raddino518_clean_llrd", "e36b_maira518_clean_llrd", "e42_convnextb384_rad3ch_clean",
]
TAGS11_MP = TAGS11[:-2] + ["e36b_maira518_clean_llrd~e37_maira518_lc_clean_llrd", "e42_convnextb384_rad3ch_clean"]
COVID_TAGS = ["e2_clahe", "e4_effb3", "e6_effv2s_clahe", "e8_effv2s_clahe448", "e15_effv2m448"]
TEST_DIRS = ["submissions/per_tag_v2_blend", "submissions/per_tag_v2", "submissions/per_tag_blend_e27", "submissions/per_tag"]

BANKS = {
    "rad-sp3-l6": ("features/rad-dino-sp3.npz", "feats_l6"),
    "rad-sp3-l9": ("features/rad-dino-sp3.npz", "feats_l9"),
    "rad-sp2-l9": ("features/rad-dino-spatial.npz", "feats_mid"),
    "rad-sp2-l12": ("features/rad-dino-spatial.npz", "feats_last"),
    "maira-sp3-l6": ("features/maira2-sp3.npz", "feats_l6"),
    "maira-sp3-l9": ("features/maira2-sp3.npz", "feats_l9"),
    "maira-sp2-l9": ("features/maira2-spatial.npz", "feats_mid"),
    "maira-sp2-l12": ("features/maira2-spatial.npz", "feats_last"),
    "sig-sp2-l16": ("features/xraysiglip-spatial.npz", "feats_l16"),
    "sig-sp2-l20": ("features/xraysiglip-spatial.npz", "feats_l20"),
    "d2l-sp2-l16": ("features/dinov2l-spatial.npz", "feats_l16"),
    "rad-lc": ("features/rad-dino-lc.npz", "feats"),
    "xray-d2": ("features/xray-dinov2-224.npz", "feats"),
    "xray-sig": ("features/xray-siglip-l384.npz", "feats"),
    "anatomy": ("features/cxr_anatomy.npz", "feats"),
    "d2-large": ("features/dinov2-large.npz", "feats"),
}

REID_V4_SEP = [
    ("rad-sp2-l9", 0.24), ("rad-sp2-l12", 0.14), ("maira-sp2-l9", 0.22), ("maira-sp2-l12", 0.12),
    ("rad-lc", 0.08), ("xray-d2", 0.07), ("xray-sig", 0.07), ("anatomy", 0.03), ("d2-large", 0.03),
]
REID_V5_SEP = [
    ("rad-sp3-l6", 0.10), ("rad-sp3-l9", 0.15), ("rad-sp2-l9", 0.10), ("rad-sp2-l12", 0.05),
    ("maira-sp3-l6", 0.10), ("maira-sp3-l9", 0.15), ("maira-sp2-l9", 0.10), ("maira-sp2-l12", 0.05),
    ("sig-sp2-l16", 0.07), ("sig-sp2-l20", 0.04), ("d2l-sp2-l16", 0.04), ("rad-lc", 0.03), ("xray-d2", 0.02),
]
REID_V5_JOINT = [
    ("rad-sp2-l9", 0.24), ("maira-sp2-l9", 0.22), ("sig-sp2-l16", 0.16), ("sig-sp2-l20", 0.10),
    ("d2l-sp2-l16", 0.14), ("rad-lc", 0.07), ("xray-d2", 0.07),
]


def gpu_whiten_pair(Ftr_np: np.ndarray, Fte_np: np.ndarray, d_max: int, device, power: float = 1.0, reg: float = 0.005):
    Xtr = torch.from_numpy(Ftr_np).to(device=device, dtype=torch.float32)
    Xte = torch.from_numpy(Fte_np).to(device=device, dtype=torch.float32)
    mu = Xtr.mean(0, keepdim=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    ntr, D = Xtr.shape
    if D <= ntr:
        C = (Xtr.T @ Xtr) / (ntr - 1)
        evals, evecs = torch.linalg.eigh(C)
        d = min(d_max, D)
        evals, evecs = evals[-d:].flip(0), evecs[:, -d:].flip(1)
        scale = (torch.clamp(evals, min=1e-6) + reg * evals.mean()).pow(-power / 2.0)
        Ztr, Zte = (Xtr @ evecs) * scale, (Xte @ evecs) * scale
    else:
        G = (Xtr @ Xtr.T) / (ntr - 1)
        evals, U = torch.linalg.eigh(G)
        d = min(d_max, ntr - 1)
        evals, U = evals[-d:].flip(0), U[:, -d:].flip(1)
        pos = torch.clamp(evals, min=1e-7)
        V = (Xtr.T @ U) / torch.sqrt(pos * (ntr - 1))
        scale = (pos + reg * pos.mean()).pow(-power / 2.0)
        Ztr, Zte = (Xtr @ V) * scale, (Xte @ V) * scale
    Ztr = Ztr / torch.clamp(Ztr.norm(dim=1, keepdim=True), min=1e-8)
    Zte = Zte / torch.clamp(Zte.norm(dim=1, keepdim=True), min=1e-8)
    return Ztr, Zte


def rich_knn_adaptive(Sm: np.ndarray, labels: np.ndarray, g1_th: float, g2_th: float) -> np.ndarray:
    order = np.argsort(-Sm, axis=1)[:, :30]
    s = np.take_along_axis(Sm, order, axis=1)
    lab = labels[order]
    out = []
    for k, temp in ((1, 1.0), (3, 0.025), (5, 0.035), (10, 0.045), (25, 0.05)):
        w = np.exp((s[:, :k] - s[:, :1]) / temp)
        votes = np.stack([(w * (lab[:, :k] == c)).sum(1) for c in range(4)], axis=1)
        out.append(np.log(votes / votes.sum(1, keepdims=True) + 1e-2))
    mpc1, mpc3 = [], []
    for c in range(4):
        sc = Sm[:, labels == c]
        top3 = np.sort(sc, axis=1)[:, -3:]
        mpc1.append(top3[:, -1:])
        mpc3.append(top3.mean(1, keepdims=True))
    mpc1, mpc3 = np.hstack(mpc1), np.hstack(mpc3)
    bv_v = np.concatenate(out[:3], axis=1)
    g1 = bv_v * np.clip(s[:, :1] - g1_th, 0, None) * 8.0
    g2 = bv_v * np.clip(s[:, :1] - g2_th, 0, None) * 14.0
    stats = np.concatenate([
        s[:, :1], s[:, :3].mean(1, keepdims=True), s[:, :1] - s[:, 1:2],
        mpc1, (mpc1 - mpc1.mean(1, keepdims=True)) * 10.0, (mpc3 - mpc3.mean(1, keepdims=True)) * 10.0,
        (mpc1[:, 2:3] - mpc1[:, 1:2]) * 12.0, (mpc3[:, 2:3] - mpc3[:, 1:2]) * 12.0,
    ], axis=1)
    return np.concatenate(out + [g1, g2, stats], axis=1)


def peer_smooth_adaptive(P: np.ndarray, S_peer: np.ndarray, th: float, hi: float, gamma: float = 0.60) -> np.ndarray:
    W = np.clip(S_peer - th, 0.0, None) ** 2
    rs = W.sum(1, keepdims=True)
    peer = np.where(rs > 1e-8, (W @ P) / np.maximum(rs, 1e-8), P)
    a = gamma * np.clip((S_peer.max(1, keepdims=True) - th) / max(hi - th, 1e-6), 0.0, 1.0)
    return (1.0 - a) * P + a * peer


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_runs(Path("runs"))
    df = df[df.complete & (df.folds == 5)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
    test_df = pd.read_csv("test_data_sample.csv")
    test_names = test_df.iloc[:, 0].astype(str).tolist()

    o0 = load_oof(df, TAGS11[0])
    names = o0.index
    y = o0.label.to_numpy()
    fold = load_folds(df, TAGS11[0]).to_numpy()
    n, nt = len(y), len(test_names)
    bv, all_nbv = np.isin(y, [1, 2]), y < 3

    cache = {k: load_emb_pair(p, fk, names, test_names) for k, (p, fk) in BANKS.items()}
    sim_cache = {}

    def get_bank(bname: str, d_max: int):
        key = (bname, d_max)
        if key in sim_cache:
            return sim_cache[key]
        Fl, Ft = cache[bname]
        sc = np.zeros((n, n), np.float32); ss = np.zeros((n, n), np.float32); sb = {}
        for k in range(5):
            tr, te = fold != k, fold == k
            Ztr, Zte = gpu_whiten_pair(Fl[tr], Fl[te], d_max, device)
            sc[np.ix_(te, tr)] = (Zte @ Ztr.T).cpu().numpy()
            ss[np.ix_(te, te)] = (Zte @ Zte.T).cpu().numpy()
            sb[k] = (Ztr @ Ztr.T).cpu().numpy()
        Zl, Zt = gpu_whiten_pair(Fl, Ft, d_max, device)
        sim_cache[key] = (sc, ss, sb, (Zt @ Zl.T).cpu().numpy(), (Zt @ Zt.T).cpu().numpy(), (Zl @ Zl.T).cpu().numpy())
        return sim_cache[key]

    def build_reid(sep_w, joint_w, d_sep, d_joint, mix_joint, k_hub=15, beta=0.5):
        S_c = np.zeros((n, n), np.float32); S_s = np.zeros((n, n), np.float32)
        S_tr = {k: np.zeros((int((fold != k).sum()), int((fold != k).sum())), np.float32) for k in range(5)}
        S_tl = np.zeros((nt, n), np.float32); S_tt = np.zeros((nt, nt), np.float32); S_ll = np.zeros((n, n), np.float32)
        tw = sum(w for _, w in sep_w)
        for bn, w in sep_w:
            wn = (1.0 - mix_joint) * (w / tw)
            sc, ss, sb, stl, stt, sll = get_bank(bn, d_sep)
            S_c += wn * sc; S_s += wn * ss; S_tl += wn * stl; S_tt += wn * stt; S_ll += wn * sll
            for k in range(5): S_tr[k] += wn * sb[k]
        if mix_joint > 0:
            tj = sum(w for _, w in joint_w)
            for k in range(5):
                tr, te = fold != k, fold == k
                ptr, pte = [], []
                for bn, w in joint_w:
                    Fl, _ = cache[bn]
                    mu = Fl[tr].mean(0, keepdims=True)
                    Ftr = Fl[tr] - mu; Ftr /= np.maximum(np.linalg.norm(Ftr, axis=1, keepdims=True), 1e-8)
                    Fte = Fl[te] - mu; Fte /= np.maximum(np.linalg.norm(Fte, axis=1, keepdims=True), 1e-8)
                    ptr.append(Ftr * np.sqrt(w / tj)); pte.append(Fte * np.sqrt(w / tj))
                Ztr, Zte = gpu_whiten_pair(np.hstack(ptr), np.hstack(pte), d_joint, device)
                S_c[np.ix_(te, tr)] += mix_joint * (Zte @ Ztr.T).cpu().numpy()
                S_s[np.ix_(te, te)] += mix_joint * (Zte @ Zte.T).cpu().numpy()
                S_tr[k] += mix_joint * (Ztr @ Ztr.T).cpu().numpy()
            pl, pt = [], []
            for bn, w in joint_w:
                Fl, Ft = cache[bn]
                mu = Fl.mean(0, keepdims=True)
                Flc = Fl - mu; Flc /= np.maximum(np.linalg.norm(Flc, axis=1, keepdims=True), 1e-8)
                Ftc = Ft - mu; Ftc /= np.maximum(np.linalg.norm(Ftc, axis=1, keepdims=True), 1e-8)
                pl.append(Flc * np.sqrt(w / tj)); pt.append(Ftc * np.sqrt(w / tj))
            Zl, Zt = gpu_whiten_pair(np.hstack(pl), np.hstack(pt), d_joint, device)
            S_tl += mix_joint * (Zt @ Zl.T).cpu().numpy()
            S_tt += mix_joint * (Zt @ Zt.T).cpu().numpy()
            S_ll += mix_joint * (Zl @ Zl.T).cpu().numpy()
        Sc = np.full_like(S_c, -10.0); Ss = np.full_like(S_s, -10.0)
        for k in range(5):
            tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
            np.fill_diagonal(S_tr[k], -1.0)
            B = S_c[np.ix_(te, tr)]
            r_te = np.sort(B, axis=1)[:, -k_hub:].mean(1, keepdims=True)
            r_tr = np.sort(S_tr[k], axis=1)[:, -k_hub:].mean(1, keepdims=True).T
            Sc[np.ix_(te, tr)] = B - beta * (r_te + r_tr)
            Pblk = S_s[np.ix_(te, te)] - beta * (r_te + r_te.T)
            np.fill_diagonal(Pblk, -10.0)
            Ss[np.ix_(te, te)] = Pblk
        np.fill_diagonal(S_ll, -1.0)
        r_t = np.sort(S_tl, axis=1)[:, -k_hub:].mean(1, keepdims=True)
        r_l = np.sort(S_ll, axis=1)[:, -k_hub:].mean(1, keepdims=True).T
        Stl = S_tl - beta * (r_t + r_l)
        Stt = S_tt - beta * (r_t + r_t.T)
        np.fill_diagonal(Stt, -10.0)
        return Sc, Ss, Stl, Stt

    # Proven clean COVID head (0.9670) & Probe_v2
    S7_c, _, S7_tl, _, rc7 = compute_whitened_sims(DEFAULT_REID_SPECS, names, test_names, fold, power=0.7)
    cov_oofs = [load_oof(df, t) for t in COVID_TAGS]
    X_cov = np.hstack([features([o[[0, 1, 2, 3]].to_numpy() for o in cov_oofs]), covid_knn_features(S7_c, y)])
    Xt_cov = np.hstack([features([load_test_prob(t, TEST_DIRS) for t in COVID_TAGS]), covid_knn_features(S7_tl, y)])
    oof_cov = np.zeros(n)
    for k in range(5):
        tr, te = fold != k, fold == k
        oof_cov[te] = LogisticRegression(C=0.10, max_iter=5000, class_weight={0: 1.0, 1: 10.0}).fit(
            X_cov[tr], (y[tr] == 3).astype(int)).predict_proba(X_cov[te])[:, 1]
    t_cov = LogisticRegression(C=0.10, max_iter=5000, class_weight={0: 1.0, 1: 10.0}).fit(
        X_cov, (y == 3).astype(int)).predict_proba(Xt_cov)[:, 1]
    p_mega_o, p_mega_t = build_mega_frozen_probe(rc7, y, fold, nt, lab_names=names, test_names=test_names, v2=True)

    def load_bv(tag):
        rb = df[df.tag == tag]
        pf = np.full((n, 2), 0.5, np.float32)
        for d in rb.dir:
            v = pd.read_csv(Path(d) / "val_split_all.csv")
            pf[names.get_indexer(v.filename.astype(str))] = np.load(Path(d) / "val_probs_all.npy")
        return np.log(np.clip(pf, 1e-6, 1.0)), np.log(np.clip(load_test_prob(tag, TEST_DIRS), 1e-6, 1.0))

    bv_pairs = [load_bv("e38_maira518_bv_clean"), load_bv("e44_raddino518_bv_clean")]

    def save_sub(Po, Pt, label, out_path):
        pc = Po[:, 3]; pn = Po[:, :3] / np.maximum(1 - pc[:, None], 1e-9)
        pr = pn.argmax(1)
        for k in range(5):
            tr, te = fold != k, fold == k
            te_idx = np.where(te)[0]
            pr[te_idx[np.argsort(-pc[te_idx])[:int(round((y[tr] == 3).mean() * te.sum()))]]] = 3
        m = compute_metrics(y, pr)
        auc = roc_auc_score((y[bv] == 2).astype(int), pn[bv, 2] / (pn[bv, 1] + pn[bv, 2] + 1e-12))
        ll = log_loss(y, np.clip(Po / np.maximum(Po.sum(1, keepdims=True), 1e-9), 1e-7, 1.0))
        tc = Pt[:, 3]; tn = Pt[:, :3] / np.maximum(1 - tc[:, None], 1e-9)
        tp = tn.argmax(1); tp[np.argsort(-tc)[:int(round(0.010152 * nt))]] = 3
        cnt = {c: int((tp == i).sum()) for i, c in enumerate(CLASSES)}
        print(f"[{label:30s}] F1={m['macro_f1']:.4f} bvAUC={auc:.4f} LL={ll:.4f} | N:{m['f1_normal']:.4f} B:{m['f1_bacteria']:.4f} V:{m['f1_virus']:.4f} C:{m['f1_COVID-19']:.4f} | test={list(cnt.values())}", flush=True)
        out = Path(out_path)
        sub = pd.DataFrame({"new_filename": test_df.iloc[:, 0]})
        for i, c in enumerate(CLASSES):
            sub[c] = (tp == i).astype(int)
        sub.to_csv(out, index=False)
        np.save(out.with_suffix(".probs.npy"), Pt)
        np.save(out.with_suffix(".oof.npy"), Po)
        out.with_suffix(".json").write_text(json.dumps({
            "label": label, "strict_5fold_macro_f1": float(m["macro_f1"]),
            "strict_5fold_bv_auc": float(auc), "strict_5fold_ll": float(ll),
            "per_class_f1": {c: float(m["f1_" + c]) for c in CLASSES}, "pred_distribution": cnt,
        }, indent=2))

    def fit_stack(tags, Sc, Ss, Stl, Stt):
        s1 = Sc.max(1)
        g1, g2, hi = float(np.quantile(s1, 0.55)), float(np.quantile(s1, 0.80)), float(np.quantile(s1, 0.96))
        K_o, K_t = rich_knn_adaptive(Sc, y, g1, g2), rich_knn_adaptive(Stl, y, g1, g2)
        cal_p = [calibrate_pair(load_oof(df, t)[[0, 1, 2, 3]].to_numpy(), load_test_prob(t, TEST_DIRS), y, fold) for t in tags]
        om, tm = [p[0] for p in cal_p], [p[1] for p in cal_p]
        X_nbv = np.hstack([features(om), np.log(np.clip(np.mean(om, axis=0), 1e-7, 1.0)),
                           *[b[0] for b in bv_pairs], np.log(np.clip(p_mega_o, 1e-7, 1.0)),
                           0.25 * K_o, extra_model_features(om)])
        Xt_nbv = np.hstack([features(tm), np.log(np.clip(np.mean(tm, axis=0), 1e-7, 1.0)),
                            *[b[1] for b in bv_pairs], np.log(np.clip(p_mega_t, 1e-7, 1.0)),
                            0.25 * K_t, extra_model_features(tm)])
        o_lr = np.zeros((n, 3)); o_lgb = np.zeros((n, 3)); o_hgb = np.zeros((n, 3)); o_mlp = np.zeros((n, 3))
        for k in range(5):
            tr_nbv, te = (fold != k) & all_nbv, fold == k
            o_lr[te] = LogisticRegression(C=0.008, max_iter=5000).fit(X_nbv[tr_nbv], y[tr_nbv]).predict_proba(X_nbv[te])
            o_lgb[te] = np.mean([
                lgb.LGBMClassifier(n_estimators=220, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                                   subsample=0.8, colsample_bytree=0.45, reg_alpha=2.0, reg_lambda=12.0,
                                   random_state=s, verbose=-1).fit(X_nbv[tr_nbv], y[tr_nbv]).predict_proba(X_nbv[te])
                for s in (42, 7, 123, 2026)
            ], axis=0)
            o_hgb[te] = np.mean([
                HistGradientBoostingClassifier(max_iter=180, learning_rate=0.03, max_leaf_nodes=7,
                                               min_samples_leaf=40, l2_regularization=12.0, random_state=s).fit(
                    X_nbv[tr_nbv], y[tr_nbv]).predict_proba(X_nbv[te])
                for s in (42, 7)
            ], axis=0)
            sc = StandardScaler().fit(X_nbv[tr_nbv])
            o_mlp[te] = np.mean([
                MLPClassifier(hidden_layer_sizes=(32,), alpha=5.0, max_iter=300, early_stopping=True,
                              n_iter_no_change=15, random_state=s).fit(sc.transform(X_nbv[tr_nbv]), y[tr_nbv]).predict_proba(sc.transform(X_nbv[te]))
                for s in (42, 7, 123, 2026, 999)
            ], axis=0)
        t_lr = LogisticRegression(C=0.008, max_iter=5000).fit(X_nbv[all_nbv], y[all_nbv]).predict_proba(Xt_nbv)
        t_lgb = np.mean([
            lgb.LGBMClassifier(n_estimators=220, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                               subsample=0.8, colsample_bytree=0.45, reg_alpha=2.0, reg_lambda=12.0,
                               random_state=s, verbose=-1).fit(X_nbv[all_nbv], y[all_nbv]).predict_proba(Xt_nbv)
            for s in (42, 7, 123, 2026)
        ], axis=0)
        t_hgb = np.mean([
            HistGradientBoostingClassifier(max_iter=180, learning_rate=0.03, max_leaf_nodes=7,
                                           min_samples_leaf=40, l2_regularization=12.0, random_state=s).fit(
                X_nbv[all_nbv], y[all_nbv]).predict_proba(Xt_nbv)
            for s in (42, 7)
        ], axis=0)
        sc = StandardScaler().fit(X_nbv[all_nbv])
        t_mlp = np.mean([
            MLPClassifier(hidden_layer_sizes=(32,), alpha=5.0, max_iter=300, early_stopping=True,
                          n_iter_no_change=15, random_state=s).fit(sc.transform(X_nbv[all_nbv]), y[all_nbv]).predict_proba(sc.transform(Xt_nbv))
            for s in (42, 7, 123, 2026, 999)
        ], axis=0)
        co = np.zeros_like(o_mlp); taus = []
        for k in range(5):
            tr_m = (fold != k) & all_nbv
            lp = np.log(np.clip(o_mlp[tr_m], 1e-7, 1.0))
            tau = float(minimize(lambda t: log_loss(y[tr_m], softmax(lp / t[0], axis=1)), [1.0], bounds=[(0.2, 5.0)]).x[0])
            taus.append(tau)
            co[fold == k] = softmax(np.log(np.clip(o_mlp[fold == k], 1e-7, 1.0)) / tau, axis=1)
        o_mlp, t_mlp = co, softmax(np.log(np.clip(t_mlp, 1e-7, 1.0)) / float(np.mean(taus)), axis=1)
        o_nbv = 0.25 * o_lr + 0.35 * o_lgb + 0.25 * o_hgb + 0.15 * o_mlp
        t_nbv = 0.25 * t_lr + 0.35 * t_lgb + 0.25 * t_hgb + 0.15 * t_mlp
        o_sm = o_nbv.copy()
        for k in range(5):
            te = np.where(fold == k)[0]
            o_sm[te] = peer_smooth_adaptive(o_nbv[te], Ss[np.ix_(te, te)], g1, hi, 0.60)
        t_sm = peer_smooth_adaptive(t_nbv, Stt, g1, hi, 0.60)
        Po = np.zeros((n, 4)); Po[:, 3] = oof_cov; Po[:, :3] = (1.0 - oof_cov[:, None]) * o_sm
        Pt = np.zeros((nt, 4)); Pt[:, 3] = t_cov; Pt[:, :3] = (1.0 - t_cov[:, None]) * t_sm
        return Po, Pt

    Sc4, Ss4, Stl4, Stt4 = build_reid(REID_V4_SEP, REID_V4_SEP, 2048, 3072, 0.50)
    Po9, Pt9 = fit_stack(TAGS11, Sc4, Ss4, Stl4, Stt4)
    save_sub(Po9, Pt9, "stack_v9_clean11_bv2_reidv4", "submissions/stack_v9_clean11_bv2_reidv4.csv")

    Sc5, Ss5, Stl5, Stt5 = build_reid(REID_V5_SEP, REID_V5_JOINT, 2560, 3072, 0.35)
    Po10, Pt10 = fit_stack(TAGS11, Sc5, Ss5, Stl5, Stt5)
    save_sub(Po10, Pt10, "stack_v10_clean11_bv2_reidv5", "submissions/stack_v10_clean11_bv2_reidv5.csv")
    Po10mp, Pt10mp = fit_stack(TAGS11_MP, Sc5, Ss5, Stl5, Stt5)
    save_sub(Po10mp, Pt10mp, "stack_v10_clean11mp_bv2_reidv5", "submissions/stack_v10_clean11mp_bv2_reidv5.csv")

    Po_sup = 0.50 * Po10 + 0.25 * Po10mp + 0.25 * Po9
    Pt_sup = 0.50 * Pt10 + 0.25 * Pt10mp + 0.25 * Pt9
    save_sub(Po_sup, Pt_sup, "stack_v10_v9_superblend", "submissions/stack_v10_v9_superblend.csv")
    print(f"All spatial stacks reproduced in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
