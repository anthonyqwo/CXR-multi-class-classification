"""Evaluate and generate final probability-blended champion submissions from strictly OOF-verified stacks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from dataset import CLASSES
from stack import load_folds, load_oof
from summarize import load_runs
from utils import compute_metrics, format_confusion

df = load_runs(Path("runs"))
df = df[df.complete & (df.folds == 5)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
o0 = load_oof(df, "e2_clahe")
y = o0.label.to_numpy()
fold = load_folds(df, "e2_clahe").to_numpy()
bv = np.isin(y, [1, 2])
test_df = pd.read_csv("test_data_sample.csv")

CANDS = [
    "stack_v3_pure_clean10_none",
    "stack_v4_clean10_mlp",
    "stack_v5_clean11_bv_mlp",
    "stack_v5_clean10_mairapair_bv_mlp",
    "stack_v5_clean10_bv_mlp",
    "stack_v5_mega13_bv_mlp",
]
oofs = {c: np.load(f"submissions/{c}.oof.npy") for c in CANDS}
tests = {c: np.load(f"submissions/{c}.probs.npy") for c in CANDS}


def score_and_optionally_save(P_oof, P_test, name, save_path=None):
    p_cov = P_oof[:, 3]
    p_nbv = P_oof[:, :3] / np.maximum(1.0 - p_cov[:, None], 1e-9)
    preds = p_nbv.argmax(1)
    for k in range(5):
        tr, te = fold != k, fold == k
        k_top = int(round((y[tr] == 3).mean() * te.sum()))
        te_idx = np.where(te)[0]
        preds[te_idx[np.argsort(-p_cov[te_idx])[:k_top]]] = 3
    m = compute_metrics(y, preds)
    auc = roc_auc_score((y[bv] == 2).astype(int), p_nbv[bv, 2] / (p_nbv[bv, 1] + p_nbv[bv, 2] + 1e-12))
    ll = log_loss(y, np.clip(P_oof / np.maximum(P_oof.sum(1, keepdims=True), 1e-9), 1e-7, 1.0))

    # Test predictions (exact same prior-matched COVID quantile 0.010152)
    tc = P_test[:, 3]
    tn = P_test[:, :3] / np.maximum(1.0 - tc[:, None], 1e-9)
    tp = tn.argmax(1)
    k_top_t = int(round(0.010152 * len(test_df)))
    tp[np.argsort(-tc)[:k_top_t]] = 3
    t_cnt = {c: int((tp == i).sum()) for i, c in enumerate(CLASSES)}

    print(
        f"{name:38s} F1={m['macro_f1']:.4f} bvAUC={auc:.4f} LL={ll:.4f} | "
        f"N:{m['f1_normal']:.4f} B:{m['f1_bacteria']:.4f} V:{m['f1_virus']:.4f} C:{m['f1_COVID-19']:.4f} | test={list(t_cnt.values())}"
    )

    if save_path:
        sub = pd.DataFrame({"new_filename": test_df.iloc[:, 0]})
        for i, c in enumerate(CLASSES):
            sub[c] = (tp == i).astype(int)
        out = Path(save_path)
        sub.to_csv(out, index=False)
        np.save(out.with_suffix(".probs.npy"), P_test)
        np.save(out.with_suffix(".oof.npy"), P_oof)
        meta = {
            "name": name,
            "strict_5fold_macro_f1": float(m["macro_f1"]),
            "strict_5fold_bv_auc": float(auc),
            "strict_5fold_ll": float(ll),
            "per_class_f1": {c: float(m["f1_" + c]) for c in CLASSES},
            "pred_distribution": t_cnt,
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"  -> saved {out}")
        print(format_confusion(y, preds))
    return tp


print("=== Individual Stacks ===")
t_preds = {}
for c in CANDS:
    t_preds[c] = score_and_optionally_save(oofs[c], tests[c], c)

print("\n=== Final Champion Probability Blends ===")
# Blend 1: Pure Clean v5 Trio (0% pseudo-labels, uses all 11 clean 4-class + 6 SWA + e38 BV specialist)
p_oof_clean = (
    0.45 * oofs["stack_v5_clean10_bv_mlp"]
    + 0.35 * oofs["stack_v5_clean10_mairapair_bv_mlp"]
    + 0.20 * oofs["stack_v5_clean11_bv_mlp"]
)
p_test_clean = (
    0.45 * tests["stack_v5_clean10_bv_mlp"]
    + 0.35 * tests["stack_v5_clean10_mairapair_bv_mlp"]
    + 0.20 * tests["stack_v5_clean11_bv_mlp"]
)
tp_clean = score_and_optionally_save(
    p_oof_clean, p_test_clean, "CHAMPION_PURE_CLEAN_V5_BLEND",
    save_path="submissions/stack_v5_champion_clean_blend.csv",
)

# Blend 2: Clean + Mega13 Hybrid Blend (adds legacy PL CNN+ViT diversity on NBV, still 100% clean COVID head)
p_oof_hyb = (
    0.40 * oofs["stack_v5_clean10_bv_mlp"]
    + 0.30 * oofs["stack_v5_clean10_mairapair_bv_mlp"]
    + 0.30 * oofs["stack_v5_mega13_bv_mlp"]
)
p_test_hyb = (
    0.40 * tests["stack_v5_clean10_bv_mlp"]
    + 0.30 * tests["stack_v5_clean10_mairapair_bv_mlp"]
    + 0.30 * tests["stack_v5_mega13_bv_mlp"]
)
tp_hyb = score_and_optionally_save(
    p_oof_hyb, p_test_hyb, "CHAMPION_HYBRID_V5_BLEND",
    save_path="submissions/stack_v5_champion_hybrid_blend.csv",
)

ref_lb = pd.read_csv("submissions/stack_v4_clean10_mlp.csv")[CLASSES].to_numpy().argmax(1)
print("\n=== Diffs vs current #1 (stack_v4_clean10_mlp, LB 0.92635) ===")
for nm, tp in [
    ("stack_v5_clean10_bv_mlp", t_preds["stack_v5_clean10_bv_mlp"]),
    ("stack_v5_champion_clean_blend", tp_clean),
    ("stack_v5_champion_hybrid_blend", tp_hyb),
    ("stack_v5_mega13_bv_mlp", t_preds["stack_v5_mega13_bv_mlp"]),
]:
    print(f"  {nm:35s} differs on {int((tp != ref_lb).sum()):3d}/1182 test images")
