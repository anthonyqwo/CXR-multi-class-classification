"""Stacking v2 (upgraded): Decoupled Meta-Ensemble with:
- Optional per-model strictly OOF temperature calibration (--calibrate)
- Strict OOF Separate + Joint PCA-Whitened 7-Foundation Patient Re-ID (--reid_hybrid)
- 3-Meta-Learner NBV Head (Balanced LR + Multi-Seed LightGBM + Multi-Seed HistGBDT)
- Transductive Same-Patient Peer Smoothing (strictly held-out fold peers on OOF, test-test peers on test)
- Clean Calibrated Binary COVID-19 Head with similarity margin + optional prior-matched quantile (--covid_prior)

Zero data leakage: every PCA, temperature scalar, frozen foundation probe, k-NN lookup,
and meta-learner is fit strictly on fold != k when scoring fold k.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from dataset import CLASSES
from stack import features, load_folds, load_oof
from summarize import load_runs
from utils import compute_metrics, format_confusion

DEFAULT_REID_SPECS = [
    ("features/rad-dino.npz", "feats", 256, 0.25),
    ("features/rad-dino-lc.npz", "feats", 256, 0.15),
    ("features/rad-dino-maira-2.npz", "feats", 256, 0.18),
    ("features/xray-dinov2-224.npz", "feats", 256, 0.14),
    ("features/xray-siglip-l384.npz", "feats", 256, 0.14),
    ("features/cxr_anatomy.npz", "feats", 128, 0.08),
    ("features/dinov2-base.npz", "feats", 256, 0.06),
]


def load_emb_pair(path: str, key: str, lab_names: pd.Index, test_names: list[str]):
    z = np.load(path)
    idx_map = pd.Series(np.arange(len(z["names"])), index=z["names"].astype(str))
    Fl = z[key][idx_map.loc[lab_names].to_numpy()].astype(np.float32)
    Ft = z[key][idx_map.loc[test_names].to_numpy()].astype(np.float32)
    return Fl, Ft


def compute_whitened_sims(specs, lab_names, test_names, fold, power=0.7):
    """Strictly OOF per-model PCA-whitened multi-foundation cosine similarity matrices."""
    n = len(lab_names)
    S_cross = np.zeros((n, n), dtype=np.float32)
    S_same = np.zeros((n, n), dtype=np.float32)
    S_tl = np.zeros((len(test_names), n), dtype=np.float32)
    S_tt = np.zeros((len(test_names), len(test_names)), dtype=np.float32)
    raw_cache = {}

    for path, key, dim, w in specs:
        Fl, Ft = load_emb_pair(path, key, lab_names, test_names)
        raw_cache[path] = (Fl, Ft)

        sc = np.full((n, n), -1.0, dtype=np.float32)
        ss = np.full((n, n), -1.0, dtype=np.float32)
        for k in np.unique(fold):
            tr = fold != k
            te = fold == k
            mu = Fl[tr].mean(0, keepdims=True)
            d = min(dim, Fl.shape[1], int(tr.sum()) - 1)
            pca = PCA(n_components=d, random_state=42).fit(Fl[tr] - mu)
            ev = np.maximum(pca.explained_variance_, 1e-6)
            Ztr = pca.transform(Fl[tr] - mu) / (ev ** (power / 2.0))
            Zte = pca.transform(Fl[te] - mu) / (ev ** (power / 2.0))
            Ztr /= np.maximum(np.linalg.norm(Ztr, axis=1, keepdims=True), 1e-8)
            Zte /= np.maximum(np.linalg.norm(Zte, axis=1, keepdims=True), 1e-8)
            sc[np.ix_(te, tr)] = Zte @ Ztr.T
            s_block = Zte @ Zte.T
            np.fill_diagonal(s_block, -1.0)
            ss[np.ix_(te, te)] = s_block

        mu = Fl.mean(0, keepdims=True)
        d = min(dim, Fl.shape[1], n - 1)
        pca = PCA(n_components=d, random_state=42).fit(Fl - mu)
        ev = np.maximum(pca.explained_variance_, 1e-6)
        Zl = pca.transform(Fl - mu) / (ev ** (power / 2.0))
        Zt = pca.transform(Ft - mu) / (ev ** (power / 2.0))
        Zl /= np.maximum(np.linalg.norm(Zl, axis=1, keepdims=True), 1e-8)
        Zt /= np.maximum(np.linalg.norm(Zt, axis=1, keepdims=True), 1e-8)

        S_cross += w * sc
        S_same += w * ss
        S_tl += w * (Zt @ Zl.T)
        S_tt += w * (Zt @ Zt.T)

    for k in np.unique(fold):
        S_cross[np.ix_(fold == k, fold == k)] = -1.0
        for k2 in np.unique(fold):
            if k != k2:
                S_same[np.ix_(fold == k, fold == k2)] = -1.0
    np.fill_diagonal(S_same, -1.0)
    np.fill_diagonal(S_tt, -1.0)
    return S_cross, S_same, S_tl, S_tt, raw_cache


def compute_joint_whitened_sims(specs, raw_cache, fold, dim=768, power=0.65):
    """Strictly OOF Joint PCA-whitening across concatenated L2-normalized multi-foundation vectors."""
    first_l, first_t = next(iter(raw_cache.values()))
    n, nt = len(first_l), len(first_t)
    Sj_c = np.full((n, n), -1.0, dtype=np.float32)
    Sj_s = np.full((n, n), -1.0, dtype=np.float32)

    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        ptr, pte = [], []
        for path, _, _, w in specs:
            Fl, _ = raw_cache[path]
            mu = Fl[tr].mean(0, keepdims=True)
            Ftr = Fl[tr] - mu; Ftr /= np.maximum(np.linalg.norm(Ftr, axis=1, keepdims=True), 1e-8)
            Fte = Fl[te] - mu; Fte /= np.maximum(np.linalg.norm(Fte, axis=1, keepdims=True), 1e-8)
            ptr.append(Ftr * np.sqrt(w)); pte.append(Fte * np.sqrt(w))
        Atr, Ate = np.hstack(ptr), np.hstack(pte)
        d = min(dim, Atr.shape[1], int(tr.sum()) - 1)
        pca = PCA(n_components=d, random_state=42).fit(Atr)
        ev = np.maximum(pca.explained_variance_, 1e-6)
        Ztr = pca.transform(Atr) / (ev ** (power / 2.0))
        Zte = pca.transform(Ate) / (ev ** (power / 2.0))
        Ztr /= np.maximum(np.linalg.norm(Ztr, axis=1, keepdims=True), 1e-8)
        Zte /= np.maximum(np.linalg.norm(Zte, axis=1, keepdims=True), 1e-8)
        Sj_c[np.ix_(te, tr)] = Zte @ Ztr.T
        ss = Zte @ Zte.T; np.fill_diagonal(ss, -1.0)
        Sj_s[np.ix_(te, te)] = ss

    pl, pt = [], []
    for path, _, _, w in specs:
        Fl, Ft = raw_cache[path]
        mu = Fl.mean(0, keepdims=True)
        Flc = Fl - mu; Flc /= np.maximum(np.linalg.norm(Flc, axis=1, keepdims=True), 1e-8)
        Ftc = Ft - mu; Ftc /= np.maximum(np.linalg.norm(Ftc, axis=1, keepdims=True), 1e-8)
        pl.append(Flc * np.sqrt(w)); pt.append(Ftc * np.sqrt(w))
    Al, At = np.hstack(pl), np.hstack(pt)
    d = min(dim, Al.shape[1], n - 1)
    pca = PCA(n_components=d, random_state=42).fit(Al)
    ev = np.maximum(pca.explained_variance_, 1e-6)
    Zl = pca.transform(Al) / (ev ** (power / 2.0))
    Zt = pca.transform(At) / (ev ** (power / 2.0))
    Zl /= np.maximum(np.linalg.norm(Zl, axis=1, keepdims=True), 1e-8)
    Zt /= np.maximum(np.linalg.norm(Zt, axis=1, keepdims=True), 1e-8)
    Sj_tl = Zt @ Zl.T
    Sj_tt = Zt @ Zt.T; np.fill_diagonal(Sj_tt, -1.0)
    return Sj_c, Sj_s, Sj_tl, Sj_tt


def rich_knn_features(Sm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Multi-scale similarity-weighted votes, per-class max-sim, and whitened-scale similarity gates."""
    order = np.argsort(-Sm, axis=1)[:, :30]
    s = np.take_along_axis(Sm, order, axis=1)
    lab = labels[order]

    out = []
    for k, temp in ((1, 1.0), (3, 0.03), (5, 0.04), (10, 0.05), (25, 0.05)):
        w = np.exp((s[:, :k] - s[:, :1]) / temp)
        votes = np.stack([(w * (lab[:, :k] == c)).sum(1) for c in range(4)], axis=1)
        out.append(np.log(votes / votes.sum(1, keepdims=True) + 1e-2))

    mpc1, mpc3 = [], []
    for c in range(4):
        sc = Sm[:, labels == c]
        top3 = np.sort(sc, axis=1)[:, -3:]
        mpc1.append(top3[:, -1:])
        mpc3.append(top3.mean(1, keepdims=True))
    mpc1 = np.hstack(mpc1)
    mpc3 = np.hstack(mpc3)
    mrel1 = mpc1 - mpc1.mean(1, keepdims=True)
    mrel3 = mpc3 - mpc3.mean(1, keepdims=True)

    bv_v = np.concatenate(out[:3], axis=1)
    g30 = bv_v * np.clip(s[:, :1] - 0.30, 0, None) * 6.0
    g42 = bv_v * np.clip(s[:, :1] - 0.42, 0, None) * 10.0
    bv_margin = (mpc1[:, 2:3] - mpc1[:, 1:2]) * 10.0
    bv_margin3 = (mpc3[:, 2:3] - mpc3[:, 1:2]) * 10.0
    stats = np.concatenate([
        s[:, :1], s[:, :3].mean(1, keepdims=True), s[:, :1] - s[:, 1:2],
        mpc1, mrel1 * 8.0, mrel3 * 8.0, bv_margin, bv_margin3,
    ], axis=1)
    return np.concatenate(out + [g30, g42, stats], axis=1)


def covid_knn_features(Sm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    s_cov = Sm[:, labels == 3]
    s_non = Sm[:, labels < 3]
    cov_top1 = s_cov.max(1, keepdims=True)
    cov_top3 = np.sort(s_cov, axis=1)[:, -3:].mean(1, keepdims=True)
    non_top1 = s_non.max(1, keepdims=True)
    non_top3 = np.sort(s_non, axis=1)[:, -3:].mean(1, keepdims=True)
    margin1 = cov_top1 - non_top1
    margin3 = cov_top3 - non_top3
    strong_non = np.clip(non_top1 - 0.35, 0, None) * np.clip(-margin1, 0, None) * 10.0
    return np.hstack([cov_top1, cov_top3, non_top1, non_top3, margin1 * 10.0, margin3 * 10.0, strong_non])


def build_mega_frozen_probe(raw_cache, y, fold, n_test, lab_names=None, test_names=None, v2: bool = False, v3: bool = False):
    """Strictly OOF multi-foundation frozen embedding linear probe
    (v1: 6-emb bal; v2: 8-emb+ZS+Anat calibrated; v3: 10-emb+ZS+Anat+36 XRV clinical pathologies)."""
    keys = [
        "features/rad-dino.npz", "features/rad-dino-lc.npz", "features/rad-dino-maira-2.npz",
        "features/xray-dinov2-224.npz", "features/xray-siglip-l384.npz", "features/cxr_anatomy.npz",
    ]
    use_v2 = v2 or v3
    if use_v2:
        extra_embs = ["features/rad-dino-maira-2-lc.npz", "features/xray-siglip-lc.npz"]
        if v3:
            extra_embs += ["features/xrv-densenet121-all.npz", "features/xrv-resnet50-all.npz"]
        keys = keys + extra_embs
        for ek in extra_embs:
            if ek not in raw_cache:
                raw_cache[ek] = load_emb_pair(ek, "feats", lab_names, test_names)
        zs_l, zs_t = load_emb_pair("features/xray-siglip-l384.npz", "zs_sims", lab_names, test_names)
        rad_l, rad_t = load_emb_pair("features/cxr_anatomy.npz", "rad", lab_names, test_names)
        demo_l, demo_t = load_emb_pair("features/cxr_anatomy.npz", "demo", lab_names, test_names)
        anat_l = np.hstack([rad_l, demo_l])
        anat_t = np.hstack([rad_t, demo_t])
        if v3:
            pd121_l, pd121_t = load_emb_pair("features/xrv-densenet121-all.npz", "pathologies", lab_names, test_names)
            pr50_l, pr50_t = load_emb_pair("features/xrv-resnet50-all.npz", "pathologies", lab_names, test_names)
            path_l = np.hstack([pd121_l, pr50_l])
            path_t = np.hstack([pd121_t, pr50_t])
    n = len(y)
    p_oof = np.zeros((n, 4), dtype=np.float32)
    p_test = np.zeros((n_test, 4), dtype=np.float32)
    cw = None if use_v2 else "balanced"
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        ptr, pte, pt = [], [], []
        for ek in keys:
            Fl, Ft = raw_cache[ek]
            mu = Fl[tr].mean(0, keepdims=True)
            Ftr = Fl[tr] - mu; Ftr /= np.maximum(np.linalg.norm(Ftr, axis=1, keepdims=True), 1e-8)
            Fte = Fl[te] - mu; Fte /= np.maximum(np.linalg.norm(Fte, axis=1, keepdims=True), 1e-8)
            Ftt = Ft - mu; Ftt /= np.maximum(np.linalg.norm(Ftt, axis=1, keepdims=True), 1e-8)
            ptr.append(Ftr); pte.append(Fte); pt.append(Ftt)
        if use_v2:
            sc_z = StandardScaler().fit(zs_l[tr])
            sc_a = StandardScaler().fit(anat_l[tr])
            ptr += [sc_z.transform(zs_l[tr]) * 0.15, sc_a.transform(anat_l[tr]) * 0.10]
            pte += [sc_z.transform(zs_l[te]) * 0.15, sc_a.transform(anat_l[te]) * 0.10]
            pt += [sc_z.transform(zs_t) * 0.15, sc_a.transform(anat_t) * 0.10]
        if v3:
            sc_p = StandardScaler().fit(path_l[tr])
            ptr.append(sc_p.transform(path_l[tr]) * 0.12)
            pte.append(sc_p.transform(path_l[te]) * 0.12)
            pt.append(sc_p.transform(path_t) * 0.12)
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight=cw).fit(np.hstack(ptr), y[tr])
        p_oof[te] = clf.predict_proba(np.hstack(pte))
        p_test += clf.predict_proba(np.hstack(pt)) / len(np.unique(fold))
    return p_oof, p_test


def extra_model_features(mats: list[np.ndarray]) -> np.ndarray:
    """Strictly per-sample cross-model consensus, B-vs-V log-odds stats, and entropy."""
    arr = np.stack(mats, axis=0)  # (M, N, 4)
    bv_ratio = arr[:, :, 2] / (arr[:, :, 1] + arr[:, :, 2] + 1e-7)
    bv_clip = np.clip(bv_ratio, 1e-5, 1.0 - 1e-5)
    bv_logodds = np.log(bv_clip / (1.0 - bv_clip))
    geo = np.exp(np.log(np.clip(arr, 1e-7, 1.0)).mean(0))
    geo /= np.maximum(geo.sum(1, keepdims=True), 1e-9)
    ent = -(arr * np.log(np.clip(arr, 1e-7, 1.0))).sum(2).mean(0, keepdims=True).T
    return np.hstack([
        bv_logodds.mean(0, keepdims=True).T,
        bv_logodds.std(0, keepdims=True).T,
        np.median(bv_logodds, axis=0)[:, None],
        bv_ratio.max(0, keepdims=True).T - bv_ratio.min(0, keepdims=True).T,
        np.log(np.clip(geo, 1e-7, 1.0)),
        ent,
    ])


def load_test_prob(tag: str, dirs: list[str]) -> np.ndarray:
    if "~" in tag:
        return np.mean([load_test_prob(t, dirs) for t in tag.split("~")], axis=0)
    for d in dirs:
        p = Path(d) / f"{tag}.probs.npy"
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"Could not find {tag}.probs.npy in {dirs}")


def calibrate_pair(P_oof: np.ndarray, P_test: np.ndarray, y: np.ndarray, fold: np.ndarray):
    """Strictly OOF temperature calibration for one model's OOF & test probabilities."""
    Pc_oof = np.zeros_like(P_oof)
    taus = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        lp_tr = np.log(np.clip(P_oof[tr], 1e-7, 1.0))
        res = minimize(lambda tau: log_loss(y[tr], softmax(lp_tr / tau[0], axis=1)), [1.0], bounds=[(0.2, 5.0)])
        tau = float(res.x[0])
        taus.append(tau)
        Pc_oof[te] = softmax(np.log(np.clip(P_oof[te], 1e-7, 1.0)) / tau, axis=1)
    tau_full = float(np.mean(taus))
    Pc_test = softmax(np.log(np.clip(P_test, 1e-7, 1.0)) / tau_full, axis=1)
    return Pc_oof, Pc_test


def peer_smooth(P: np.ndarray, S_peer: np.ndarray, thresh: float = 0.35, gamma: float = 0.45) -> np.ndarray:
    W = np.clip(S_peer - thresh, 0.0, None) ** 2
    rs = W.sum(1, keepdims=True)
    peer = np.where(rs > 1e-8, (W @ P) / np.maximum(rs, 1e-8), P)
    max_s = S_peer.max(1, keepdims=True)
    a = gamma * np.clip((max_s - thresh) / (0.65 - thresh), 0.0, 1.0)
    return (1.0 - a) * P + a * peer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="main 4-class tags for NBV head")
    ap.add_argument("--covid_tags", nargs="+", default=None, help="clean non-PL tags for COVID head")
    ap.add_argument("--bv_tags", nargs="*", default=[], help="optional binary bacteria-vs-virus specialist tags")
    ap.add_argument("--test_probs_dirs", nargs="+",
                    default=["submissions/per_tag_v2_blend", "submissions/per_tag_v2",
                             "submissions/per_tag_blend_e27", "submissions/per_tag"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--test_csv", default="test_data_sample.csv")
    ap.add_argument("--calibrate", action="store_true", help="strictly OOF temperature-calibrate each model")
    ap.add_argument("--reid_hybrid", action="store_true", help="50/50 Separate + Joint PCA-whitened 7-emb Re-ID")
    ap.add_argument("--probe_v2", action="store_true", help="8-emb + zero-shot + radiomics/demo calibrated frozen probe")
    ap.add_argument("--probe_v3", action="store_true", help="10-emb (+XRV d121/r50 + 36 pathologies) + ZS + anat probe")
    ap.add_argument("--extra_model_feats", action="store_true", help="add cross-model B/V log-odds stats, geomean, entropy")
    ap.add_argument("--use_hgb", action="store_true", help="3-meta-learner NBV blend: 45%% LR + 35%% LGBM + 20%% HGBDT")
    ap.add_argument("--use_mlp", action="store_true", help="4-meta-learner NBV blend: 25%% LR + 35%% LGBM + 25%% HGB + 15%% cal-MLP")
    ap.add_argument("--use_et", action="store_true", help="5-meta-learner NBV blend: 20%% LR + 30%% LGBM + 20%% HGB + 15%% cal-MLP + 15%% cal-ET")
    ap.add_argument("--nbv_cw", choices=["balanced", "soft", "none"], default="balanced",
                    help="NBV class weighting: balanced, soft ({0:1,1:1,2:1.35}), or none")
    ap.add_argument("--lr_weight", type=float, default=0.50)
    ap.add_argument("--covid_cw", type=float, default=10.0)
    ap.add_argument("--covid_C", type=float, default=0.10)
    ap.add_argument("--peer_thresh", type=float, default=0.35)
    ap.add_argument("--peer_gamma", type=float, default=0.45)
    ap.add_argument("--peer_steps", type=int, default=1)
    ap.add_argument("--covid_prior", type=float, default=0.0,
                    help=">0: match test COVID-19 prior (e.g. 0.010152 from spec.pdf); OOF strictly uses fold train prior")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_runs(Path(args.runs))
    df = df[df.complete & (df.folds == 5)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
    test_df = pd.read_csv(args.test_csv)
    test_names = test_df.iloc[:, 0].astype(str).tolist()

    oofs = [load_oof(df, t) for t in args.tags]
    names = oofs[0].index
    for o in oofs[1:]:
        assert (o.index == names).all()
    y = oofs[0].label.to_numpy()
    fold = load_folds(df, args.tags[0]).to_numpy()
    n = len(y)
    bv = np.isin(y, [1, 2])

    oof_mats = [o[[0, 1, 2, 3]].to_numpy() for o in oofs]
    test_mats = [load_test_prob(t, args.test_probs_dirs) for t in args.tags]
    if args.calibrate:
        cal_pairs = [calibrate_pair(po, pt, y, fold) for po, pt in zip(oof_mats, test_mats)]
        oof_mats = [p[0] for p in cal_pairs]
        test_mats = [p[1] for p in cal_pairs]
    X_models = features(oof_mats)
    Xt_models = features(test_mats)
    avg_oof = np.mean(oof_mats, axis=0)
    avg_test = np.mean(test_mats, axis=0)

    X_bv_list, Xt_bv_list = [], []
    for bt in args.bv_tags:
        runs_b = df[df.tag == bt]
        p_full = np.full((n, 2), 0.5, dtype=np.float32)
        for d in runs_b.dir:
            dp = Path(d)
            if (dp / "val_probs_all.npy").exists() and (dp / "val_split_all.csv").exists():
                v = pd.read_csv(dp / "val_split_all.csv")
                vp = np.load(dp / "val_probs_all.npy")
            else:
                v = pd.read_csv(dp / "val_split.csv")
                vp = np.load(dp / "val_probs.npy")
            idx = names.get_indexer(v.filename.astype(str))
            p_full[idx] = vp
        X_bv_list.append(np.log(np.clip(p_full, 1e-6, 1.0)))
        Xt_bv_list.append(np.log(np.clip(load_test_prob(bt, args.test_probs_dirs), 1e-6, 1.0)))

    cov_tags = args.covid_tags or args.tags
    cov_oofs = [load_oof(df, t) for t in cov_tags]
    X_cov_models = features([o[[0, 1, 2, 3]].to_numpy() for o in cov_oofs])
    Xt_cov_models = features([load_test_prob(t, args.test_probs_dirs) for t in cov_tags])

    S_cross, S_same, S_tl, S_tt, raw_cache = compute_whitened_sims(
        DEFAULT_REID_SPECS, names, test_names, fold, power=0.7
    )
    if args.reid_hybrid:
        Sj_c, Sj_s, Sj_tl, Sj_tt = compute_joint_whitened_sims(DEFAULT_REID_SPECS, raw_cache, fold, dim=768, power=0.65)
        S_nbv_c = 0.5 * S_cross + 0.5 * Sj_c
        S_nbv_s = 0.5 * S_same + 0.5 * Sj_s
        S_nbv_tl = 0.5 * S_tl + 0.5 * Sj_tl
        S_nbv_tt = 0.5 * S_tt + 0.5 * Sj_tt
        for kf in np.unique(fold):
            S_nbv_c[np.ix_(fold == kf, fold == kf)] = -1.0
        np.fill_diagonal(S_nbv_s, -1.0)
        np.fill_diagonal(S_nbv_tt, -1.0)
    else:
        S_nbv_c, S_nbv_s, S_nbv_tl, S_nbv_tt = S_cross, S_same, S_tl, S_tt

    K_oof = rich_knn_features(S_nbv_c, y)
    K_test = rich_knn_features(S_nbv_tl, y)
    C_oof = covid_knn_features(S_cross, y)
    C_test = covid_knn_features(S_tl, y)
    p_mega, p_mega_t = build_mega_frozen_probe(
        raw_cache, y, fold, len(test_names), lab_names=names, test_names=test_names,
        v2=args.probe_v2, v3=args.probe_v3,
    )

    extra_o = [extra_model_features(oof_mats)] if args.extra_model_feats else []
    extra_t = [extra_model_features(test_mats)] if args.extra_model_feats else []
    X_nbv = np.hstack([X_models, np.log(np.clip(avg_oof, 1e-7, 1.0)), *X_bv_list,
                       np.log(np.clip(p_mega, 1e-7, 1.0)), 0.25 * K_oof, *extra_o])
    Xt_nbv = np.hstack([Xt_models, np.log(np.clip(avg_test, 1e-7, 1.0)), *Xt_bv_list,
                        np.log(np.clip(p_mega_t, 1e-7, 1.0)), 0.25 * K_test, *extra_t])
    X_cov = np.hstack([X_cov_models, C_oof])
    Xt_cov = np.hstack([Xt_cov_models, C_test])

    oof_nbv_lr = np.zeros((n, 3))
    oof_nbv_lgb = np.zeros((n, 3))
    oof_nbv_hgb = np.zeros((n, 3))
    oof_nbv_mlp = np.zeros((n, 3))
    oof_nbv_et = np.zeros((n, 3))
    oof_cov = np.zeros(n)

    nbv_cw = "balanced" if args.nbv_cw == "balanced" else ({0: 1.0, 1: 1.0, 2: 1.35} if args.nbv_cw == "soft" else None)

    def make_lgb(seed):
        return lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.03, num_leaves=7, min_child_samples=40,
            subsample=0.8, colsample_bytree=0.45, reg_alpha=2.0, reg_lambda=12.0,
            class_weight=nbv_cw, random_state=seed, verbose=-1
        )

    def make_hgb(seed):
        return HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=40,
            l2_regularization=12.0, class_weight=nbv_cw, random_state=seed
        )

    def make_mlp(seed):
        return MLPClassifier(
            hidden_layer_sizes=(32,), alpha=5.0, max_iter=300, early_stopping=True,
            n_iter_no_change=15, random_state=seed
        )

    def make_et(seed):
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=12, max_features=0.35,
            class_weight=nbv_cw, random_state=seed, n_jobs=-1
        )

    need_hgb = args.use_hgb or args.use_mlp or args.use_et
    need_mlp = args.use_mlp or args.use_et

    for k in range(5):
        tr, te = fold != k, fold == k
        tr_nbv = tr & (y < 3)
        oof_nbv_lr[te] = LogisticRegression(C=0.008, max_iter=5000, class_weight=nbv_cw).fit(
            X_nbv[tr_nbv], y[tr_nbv]).predict_proba(X_nbv[te])
        oof_nbv_lgb[te] = np.mean([
            make_lgb(s).fit(X_nbv[tr_nbv], y[tr_nbv]).predict_proba(X_nbv[te])
            for s in (42, 7, 123, 2026)
        ], axis=0)
        if need_hgb:
            oof_nbv_hgb[te] = np.mean([
                make_hgb(s).fit(X_nbv[tr_nbv], y[tr_nbv]).predict_proba(X_nbv[te])
                for s in (42, 7)
            ], axis=0)
        if need_mlp:
            sc = StandardScaler().fit(X_nbv[tr_nbv])
            Xtr_s, Xte_s = sc.transform(X_nbv[tr_nbv]), sc.transform(X_nbv[te])
            oof_nbv_mlp[te] = np.mean([
                make_mlp(s).fit(Xtr_s, y[tr_nbv]).predict_proba(Xte_s)
                for s in (42, 7, 123, 2026, 999)
            ], axis=0)
        if args.use_et:
            oof_nbv_et[te] = np.mean([
                make_et(s).fit(X_nbv[tr_nbv], y[tr_nbv]).predict_proba(X_nbv[te])
                for s in (42, 7, 123)
            ], axis=0)
        oof_cov[te] = LogisticRegression(
            C=args.covid_C, max_iter=5000, class_weight={0: 1.0, 1: args.covid_cw}
        ).fit(X_cov[tr], (y[tr] == 3).astype(int)).predict_proba(X_cov[te])[:, 1]

    all_nbv = y < 3
    t_nbv_lr = LogisticRegression(C=0.008, max_iter=5000, class_weight=nbv_cw).fit(
        X_nbv[all_nbv], y[all_nbv]).predict_proba(Xt_nbv)
    t_nbv_lgb = np.mean([
        make_lgb(s).fit(X_nbv[all_nbv], y[all_nbv]).predict_proba(Xt_nbv)
        for s in (42, 7, 123, 2026)
    ], axis=0)
    if need_hgb:
        t_nbv_hgb = np.mean([
            make_hgb(s).fit(X_nbv[all_nbv], y[all_nbv]).predict_proba(Xt_nbv)
            for s in (42, 7)
        ], axis=0)

    def cal_nbv_pair(po, pt):
        co = np.zeros_like(po)
        taus = []
        for k in np.unique(fold):
            tr_m = (fold != k) & all_nbv
            lp_tr = np.log(np.clip(po[tr_m], 1e-7, 1.0))
            tau = float(minimize(lambda t: log_loss(y[tr_m], softmax(lp_tr / t[0], axis=1)), [1.0], bounds=[(0.2, 5.0)]).x[0])
            taus.append(tau)
            co[fold == k] = softmax(np.log(np.clip(po[fold == k], 1e-7, 1.0)) / tau, axis=1)
        ct = softmax(np.log(np.clip(pt, 1e-7, 1.0)) / float(np.mean(taus)), axis=1)
        return co, ct

    if need_mlp:
        sc = StandardScaler().fit(X_nbv[all_nbv])
        Xtr_s, Xte_s = sc.transform(X_nbv[all_nbv]), sc.transform(Xt_nbv)
        t_nbv_mlp = np.mean([
            make_mlp(s).fit(Xtr_s, y[all_nbv]).predict_proba(Xte_s)
            for s in (42, 7, 123, 2026, 999)
        ], axis=0)
        oof_nbv_mlp, t_nbv_mlp = cal_nbv_pair(oof_nbv_mlp, t_nbv_mlp)

    if args.use_et:
        t_nbv_et = np.mean([
            make_et(s).fit(X_nbv[all_nbv], y[all_nbv]).predict_proba(Xt_nbv)
            for s in (42, 7, 123)
        ], axis=0)
        oof_nbv_et, t_nbv_et = cal_nbv_pair(oof_nbv_et, t_nbv_et)

    if args.use_et:
        oof_nbv = 0.20 * oof_nbv_lr + 0.30 * oof_nbv_lgb + 0.20 * oof_nbv_hgb + 0.15 * oof_nbv_mlp + 0.15 * oof_nbv_et
        t_nbv = 0.20 * t_nbv_lr + 0.30 * t_nbv_lgb + 0.20 * t_nbv_hgb + 0.15 * t_nbv_mlp + 0.15 * t_nbv_et
    elif args.use_mlp:
        oof_nbv = 0.25 * oof_nbv_lr + 0.35 * oof_nbv_lgb + 0.25 * oof_nbv_hgb + 0.15 * oof_nbv_mlp
        t_nbv = 0.25 * t_nbv_lr + 0.35 * t_nbv_lgb + 0.25 * t_nbv_hgb + 0.15 * t_nbv_mlp
    elif args.use_hgb:
        oof_nbv = 0.45 * oof_nbv_lr + 0.35 * oof_nbv_lgb + 0.20 * oof_nbv_hgb
        t_nbv = 0.45 * t_nbv_lr + 0.35 * t_nbv_lgb + 0.20 * t_nbv_hgb
    else:
        oof_nbv = args.lr_weight * oof_nbv_lr + (1.0 - args.lr_weight) * oof_nbv_lgb
        t_nbv = args.lr_weight * t_nbv_lr + (1.0 - args.lr_weight) * t_nbv_lgb

    oof_nbv_sm = oof_nbv.copy()
    for _ in range(args.peer_steps):
        nxt = oof_nbv_sm.copy()
        for k in range(5):
            te = np.where(fold == k)[0]
            nxt[te] = peer_smooth(oof_nbv_sm[te], S_nbv_s[np.ix_(te, te)], args.peer_thresh, args.peer_gamma)
        oof_nbv_sm = nxt

    oof_probs = np.zeros((n, 4), dtype=np.float64)
    oof_probs[:, 3] = oof_cov
    oof_probs[:, :3] = (1.0 - oof_cov[:, None]) * oof_nbv_sm
    oof_preds = oof_nbv_sm.argmax(1)
    if args.covid_prior > 0:
        for k in range(5):
            tr, te = fold != k, fold == k
            k_top = int(round((y[tr] == 3).mean() * te.sum()))
            te_idx = np.where(te)[0]
            oof_preds[te_idx[np.argsort(-oof_cov[te_idx])[:k_top]]] = 3
    else:
        oof_preds[oof_cov >= 0.5] = 3

    m = compute_metrics(y, oof_preds)
    bv_auc = roc_auc_score((y[bv] == 2).astype(int),
                           oof_nbv_sm[bv, 2] / (oof_nbv_sm[bv, 1] + oof_nbv_sm[bv, 2] + 1e-12))
    ll = log_loss(y, np.clip(oof_probs, 1e-7, 1.0))
    print(f"[OOF strict 5-fold] macroF1={m['macro_f1']:.4f} bvAUC={bv_auc:.4f} LL={ll:.4f} | "
          f"N:{m['f1_normal']:.4f} B:{m['f1_bacteria']:.4f} V:{m['f1_virus']:.4f} C:{m['f1_COVID-19']:.4f}")
    print(format_confusion(y, oof_preds))

    t_nbv_sm = t_nbv.copy()
    for _ in range(args.peer_steps):
        t_nbv_sm = peer_smooth(t_nbv_sm, S_nbv_tt, args.peer_thresh, args.peer_gamma)
    t_cov = LogisticRegression(
        C=args.covid_C, max_iter=5000, class_weight={0: 1.0, 1: args.covid_cw}
    ).fit(X_cov, (y == 3).astype(int)).predict_proba(Xt_cov)[:, 1]

    t_probs = np.zeros((len(test_names), 4), dtype=np.float64)
    t_probs[:, 3] = t_cov
    t_probs[:, :3] = (1.0 - t_cov[:, None]) * t_nbv_sm
    t_preds = t_nbv_sm.argmax(1)
    if args.covid_prior > 0:
        k_top = int(round(args.covid_prior * len(test_names)))
        t_preds[np.argsort(-t_cov)[:k_top]] = 3
    else:
        t_preds[t_cov >= 0.5] = 3

    sub = pd.DataFrame({"new_filename": test_df.iloc[:, 0]})
    for i, c in enumerate(CLASSES):
        sub[c] = (t_preds == i).astype(int)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    np.save(out.with_suffix(".probs.npy"), t_probs)
    np.save(out.with_suffix(".oof.npy"), oof_probs)
    meta_info = {
        "tags": args.tags,
        "covid_tags": cov_tags,
        "bv_tags": args.bv_tags,
        "calibrate": args.calibrate,
        "reid_hybrid": args.reid_hybrid,
        "probe_v2": args.probe_v2,
        "probe_v3": args.probe_v3,
        "extra_model_feats": args.extra_model_feats,
        "use_hgb": args.use_hgb,
        "use_mlp": args.use_mlp,
        "use_et": args.use_et,
        "nbv_cw": args.nbv_cw,
        "peer_thresh": args.peer_thresh,
        "peer_gamma": args.peer_gamma,
        "peer_steps": args.peer_steps,
        "covid_prior": args.covid_prior,
        "strict_5fold_macro_f1": float(m["macro_f1"]),
        "strict_5fold_bv_auc": float(bv_auc),
        "strict_5fold_ll": float(ll),
        "per_class_f1": {c: float(m["f1_" + c]) for c in CLASSES},
        "pred_distribution": {c: int((t_preds == i).sum()) for i, c in enumerate(CLASSES)},
    }
    out.with_suffix(".json").write_text(json.dumps(meta_info, indent=2))
    print(f"[done] wrote {out} | test pred counts: {meta_info['pred_distribution']}")
    print("Top 15 test COVID-19 probs:")
    for idx in np.argsort(-t_cov)[:15]:
        print(f"  {test_names[idx]:>10}: p_cov={t_cov[idx]:.4f} -> {CLASSES[t_preds[idx]]}")


if __name__ == "__main__":
    main()
