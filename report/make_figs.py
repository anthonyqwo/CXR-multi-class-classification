"""Figures and numbers for the report. Run from the project root:

    python report/verify_reid.py     # writes report/figs/reid_check.json (needs the GPU for a minute)
    python report/make_figs.py

Inputs that are snapshots rather than recomputed: report/figs/submissions.csv (Kaggle submission history with the
cross-validation numbers recorded in each submission description).
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from stack import features, load_folds, load_knn, load_oof
from summarize import load_runs

OUT = Path("report/figs")
OUT.mkdir(parents=True, exist_ok=True)
K = ["normal", "bacteria", "virus", "COVID-19"]
KZ = ["正常", "細菌", "病毒", "COVID-19"]
BLUE, LIGHT, ORANGE, GREY, INK, MUTED = "#2a78d6", "#8fbdf0", "#eb6834", "#b9b8b0", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "Microsoft JhengHei", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c0", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 220, "savefig.bbox": "tight", "figure.facecolor": "white",
})
P = [0, 1, 2, 3]
NUM = {}

# ---- 1. one image per class: original / CLAHE / lung crop + CLAHE
tr = pd.read_csv("data/train_data.csv")
boxes = pd.read_csv("features/lung_boxes.csv").set_index("filename")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
fig, ax = plt.subplots(3, 4, figsize=(7.2, 5.2))
for j, c in enumerate(K):
    f = tr[tr[c] == 1].new_filename.iloc[3]
    raw = cv2.imdecode(np.fromfile(f"data/train_images/{f}", np.uint8), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(raw, (448, 448), interpolation=cv2.INTER_AREA)
    row = boxes.loc[f]
    h, w = raw.shape[:2]
    x1, y1, x2, y2 = float(row.x1) * w, float(row.y1) * h, float(row.x2) * w, float(row.y2) * h
    bw, bh = x2 - x1, y2 - y1
    lc = raw[max(0, int(y1 - 0.12 * bh)):min(h, int(y2 + 0.12 * bh)), max(0, int(x1 - 0.12 * bw)):min(w, int(x2 + 0.12 * bw))]
    lc = cv2.resize(lc, (448, 448), interpolation=cv2.INTER_AREA)
    for i, im in enumerate((g, clahe.apply(g), clahe.apply(lc))):
        ax[i, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[i, j].axis("off")
    ax[0, j].set_title(KZ[j], fontsize=10, color=INK)
for yy, t in ((0.80, "原始影像"), (0.50, "CLAHE 後"), (0.20, "肺部裁切 + CLAHE")):
    fig.text(0.005, yy, t, rotation=90, va="center", color=MUTED)
fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.035)
fig.savefig(OUT / "samples.png")
plt.close(fig)

# ---- 2. every submission and the running best
sub = pd.read_csv(OUT / "submissions.csv")
best = sub.score.cummax()
fig, ax = plt.subplots(figsize=(7.4, 3.2))
ax.axvspan(31.5, len(sub) + 0.7, color="#f1f6fd", zorder=0)
ax.step(sub.n, best, where="post", color=BLUE, lw=2, zorder=2)
ax.scatter(sub.n, sub.score, s=16, color=GREY, zorder=3, linewidths=0)
ax.scatter(sub.n[sub.score == best], sub.score[sub.score == best], s=22, color=BLUE, zorder=4, linewidths=0)
marks = {1: ("單一模型 0.8433", (10, -3), "left"), 9: ("堆疊 0.8779", (-5, 5), "right"), 23: ("加入 RAD-DINO 0.8898", (-4, 7), "right"),
         30: ("最近鄰特徵 0.8945", (-5, 8), "right"), 32: ("白化相似度 + 雙頭 0.9126", (-6, 2), "right"),
         33: ("COVID-19 張數對齊 0.9230", (-6, 6), "right"), 44: ("高維白化 + CSLS 0.9301", (-6, 5), "right"),
         48: ("中間層空間特徵 0.9364", (-6, 7), "right")}
for n_, (txt, off, ha) in marks.items():
    ax.annotate(txt, (n_, sub.score[n_ - 1]), textcoords="offset points", xytext=off, ha=ha, fontsize=7.6, color=INK)
ax.text(16, 0.9525, "第一階段（11/12–11/17）", ha="center", fontsize=8, color=MUTED)
ax.text(40, 0.9525, "第二階段（11/19–11/22）", ha="center", fontsize=8, color=MUTED)
ax.set_xlim(0, len(sub) + 0.7)
ax.set_ylim(0.838, 0.958)
ax.set_xlabel("上傳次序（共 48 次）")
ax.set_ylabel("排行榜 macro-F1")
ax.grid(axis="x", visible=False)
ax.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2), plt.Line2D([], [], color=GREY, marker="o", ls="", ms=4)],
          labels=["截至當時的最高分", "每一次上傳"], loc="lower right", frameon=False, fontsize=8)
fig.savefig(OUT / "public.png")
plt.close(fig)

# ---- 3. cross-validation vs leaderboard for the second-stage submissions
s2 = sub[sub.n >= 32].copy()
s2.loc[s2.file == "stack_v4_clean10_mlp.csv", "cv_f1"] = 0.9141   # from submissions/stack_v4_clean10_mlp.json
s2 = s2.dropna(subset=["cv_f1"])
NUM["cv_lb_spearman"] = float(s2.cv_f1.corr(s2.score, method="spearman"))
NUM["cv_lb_n"] = int(len(s2))
fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.scatter(s2.cv_f1, s2.score, s=30, color=BLUE, zorder=3, edgecolor="white", linewidths=1)
lab = {"stack_v2_reid7_decoupled.csv": ("v2", (5, -3)), "stack_v2_reid7_prior.csv": ("v2 + 張數對齊", (6, -3)),
       "stack_v8_clean10_reidv3_csls.csv": ("v8（10 個模型）", (6, -8)), "stack_v8_clean11_bv2_reidv3_csls.csv": ("v8", (6, 2)),
       "stack_v9_clean11_bv2_reidv4.csv": ("v9", (-6, 3)), "stack_v9_clean11_bv2_reidv4_s2_wv95.csv": ("v9 變體", (6, -8)),
       "stack_v10_clean11_bv2_reidv5.csv": ("v10", (-6, 2)), "stack_v10_v9_superblend.csv": ("最終", (5, 3))}
for _, r in s2.iterrows():
    if r.file in lab:
        t, off = lab[r.file]
        ax.annotate(t, (r.cv_f1, r.score), textcoords="offset points", xytext=off, ha="left" if off[0] > 0 else "right",
                    fontsize=7.6, color=INK)
ax.set_xlabel("5 折交叉驗證 macro-F1")
ax.set_ylabel("排行榜 macro-F1")
fig.savefig(OUT / "cv_vs_lb.png")
plt.close(fig)

# ---- 4. nearest-neighbour label agreement of each similarity (bacteria / virus images)
rc = {r["label"]: r for r in json.loads((OUT / "reid_check.json").read_text(encoding="utf-8"))}
stages = [("RAD-DINO cosine (stage 1)", "RAD-DINO\n餘弦相似度\n（第一階段）"), ("7 encoders, PCA-256 whitening", "7 種編碼器\nPCA 白化"),
          ("2x2 last layer (L12), no CSLS", "最後一層\n2×2 分區\n高維白化"), ("2x2 middle layer (L9), no CSLS", "中間層\n2×2 分區\n高維白化"),
          ("Re-ID v5 without CSLS", "多層多編碼器\n（無 CSLS）"), ("Re-ID v5 (stack_v10)", "多層多編碼器\n+ CSLS（最終）")]
x = np.arange(len(stages))
fig, ax = plt.subplots(figsize=(7.4, 3.0))
for off, key, col, name in ((-0.2, "all", LIGHT, "全部細菌／病毒影像（3,419 張）"), (0.2, "top30", BLUE, "相似度最高的 30%（1,026 張）")):
    v = [100 * rc[s][key] for s, _ in stages]
    ax.bar(x + off, v, 0.38, color=col, label=name)
    for xi, vi in zip(x + off, v):
        ax.text(xi, vi + 0.5, f"{vi:.1f}", ha="center", fontsize=7.6, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([t for _, t in stages], fontsize=7.6)
ax.set_ylim(60, 108)
ax.set_ylabel("最近鄰標籤相同比例 (%)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", frameon=False, fontsize=7.8, ncol=2)
fig.savefig(OUT / "reid.png")
plt.close(fig)

# ---- 5. every model that matters: macro-F1, bacteria-vs-virus AUC, log-loss
df = load_runs(Path("runs"))
df = df[df.complete & (df.folds == 5)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
FINAL = ["e2_clahe", "e4_effb3", "e6_effv2s_clahe", "e8_effv2s_clahe448", "e15_effv2m448", "e33_effv2m448_lc_rad3ch",
         "e34_xraysiglip512_clean", "e35_swinb384_rad3ch", "e36_raddino518_clean_llrd", "e36b_maira518_clean_llrd",
         "e37_maira518_lc_clean_llrd", "e42_convnextb384_rad3ch_clean"]
OLD = ["e11_effv2s448_pl", "e23_effv2m448_pl", "e27_raddino518_pl", "e28_maira518_pl"]
NAMES = {"e2_clahe": "e2 ConvNeXt-T 384", "e4_effb3": "e4 EffNet-B3 384", "e6_effv2s_clahe": "e6 EffNetV2-S 384",
         "e8_effv2s_clahe448": "e8 EffNetV2-S 448", "e15_effv2m448": "e15 EffNetV2-M 448",
         "e33_effv2m448_lc_rad3ch": "e33 EffNetV2-M 448 肺部裁切", "e34_xraysiglip512_clean": "e34 XraySigLIP-L 512",
         "e35_swinb384_rad3ch": "e35 Swin-B 384", "e36_raddino518_clean_llrd": "e36 RAD-DINO 518",
         "e36b_maira518_clean_llrd": "e36b MAIRA-2 518", "e37_maira518_lc_clean_llrd": "e37 MAIRA-2 518 肺部裁切",
         "e42_convnextb384_rad3ch_clean": "e42 ConvNeXt-B 384", "e11_effv2s448_pl": "e11 EffNetV2-S 448（偽標籤）",
         "e23_effv2m448_pl": "e23 EffNetV2-M 448（偽標籤）", "e27_raddino518_pl": "e27 RAD-DINO 518（偽標籤）",
         "e28_maira518_pl": "e28 MAIRA-2 518（偽標籤）"}
rows = []
for t in FINAL + OLD:
    o = load_oof(df, t)
    yt, pt = o.label.to_numpy(), o[P].to_numpy()
    b = np.isin(yt, [1, 2])
    rows.append(dict(tag=t, name=NAMES[t], final=t in FINAL, f1=f1_score(yt, pt.argmax(1), average="macro"),
                     auc=roc_auc_score((yt[b] == 2).astype(int), pt[b, 2] / (pt[b, 1] + pt[b, 2] + 1e-12)),
                     ll=log_loss(yt, np.clip(pt, 1e-7, 1.0)), covid_fp=int(((pt.argmax(1) == 3) & (yt != 3)).sum()),
                     bv_acc=float((pt.argmax(1) == yt)[b].mean())))
m = pd.DataFrame(rows).sort_values("f1").reset_index(drop=True)
for bt in ("e38_maira518_bv_clean", "e44_raddino518_bv_clean"):
    ps, ys = [], []
    for d in df[df.tag == bt].dir:
        ps.append(np.load(Path(d) / "val_probs.npy"))
        ys.append(pd.read_csv(Path(d) / "val_split.csv").label.to_numpy())
    ps, ys = np.concatenate(ps), np.concatenate(ys)
    NUM[bt] = dict(auc=float(roc_auc_score(ys, ps[:, 1])), acc=float((ps.argmax(1) == ys).mean()), n=int(len(ys)))
m.to_csv(OUT / "models.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 4.3), sharey=True)
cols = [BLUE if f else GREY for f in m.final]
for ax, key, title, lim in zip(axs, ("f1", "auc", "ll"), ("macro-F1（越高越好）", "細菌對病毒 AUC（越高越好）", "log-loss（越低越好）"),
                               ((0.84, 0.895), (0.84, 0.92), (0.38, 0.68))):
    ax.barh(m.name, m[key], color=cols, height=0.68)
    for i, v in enumerate(m[key]):
        ax.text(v + (lim[1] - lim[0]) * 0.015, i, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_xlim(*lim)
    ax.set_title(title, fontsize=8.6, color=INK)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.6)
    ax.tick_params(axis="x", labelsize=7.4)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=GREY)],
           labels=["最終方法使用", "第一階段使用，最終未採用"], loc="upper center", ncol=2, frameon=False, fontsize=8,
           bbox_to_anchor=(0.55, 1.0))
fig.subplots_adjust(wspace=0.08)
fig.savefig(OUT / "models.png")
plt.close(fig)

# ---- 6. validation curves of the two training recipes on fold 0 (MAIRA-2)
lo = pd.read_csv(Path(sorted(glob.glob("runs/*f0of5_e28_maira518_pl"))[-1]) / "log.csv")
ln = pd.read_csv(Path(sorted(glob.glob("runs/*f0of5_e36b_maira518_clean_llrd"))[-1]) / "log.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5))
for ax, col, yl in zip(axs, ("val_loss", "macro_f1"), ("驗證損失", "驗證 macro-F1")):
    ax.plot(lo.epoch, lo[col], color=ORANGE, lw=2, label="train.py：全部層同一學習率，25 週期")
    ax.plot(ln.epoch, ln[col], color=BLUE, lw=2, label="train_v2.py：分層學習率，10 週期")
    ax.set_xlabel("週期")
    ax.set_ylabel(yl)
axs[0].legend(frameon=False, fontsize=7.4)
fig.tight_layout()
fig.savefig(OUT / "curves.png")
plt.close(fig)

# ---- 7. out-of-fold predictions: best single model, first-stage stack, final stack
ref = load_oof(df, "e2_clahe")
names, y = ref.index, ref.label.to_numpy()
fold = load_folds(df, "e2_clahe").to_numpy()
bv = np.isin(y, [1, 2])


def stats(probs, preds):
    pb = probs[bv, 2] / (probs[bv, 1] + probs[bv, 2] + 1e-12)
    cm = confusion_matrix(y, preds)
    return dict(macro=float(f1_score(y, preds, average="macro")), per_class=[float(v) for v in f1_score(y, preds, average=None)],
                bv_auc=float(roc_auc_score((y[bv] == 2).astype(int), pb)),
                ll=float(log_loss(y, np.clip(probs / probs.sum(1, keepdims=True), 1e-7, 1.0))),
                bv_acc=float((preds == y)[bv].mean()), bv_swaps=int(cm[1, 2] + cm[2, 1]), errors=int((preds != y).sum()),
                covid_fp=int(cm[:3, 3].sum()), covid_fn=int(cm[3, :3].sum()), cm=cm.tolist(),
                per_fold=[float(f1_score(y[fold == k], preds[fold == k], average="macro")) for k in range(5)])


def quantile_preds(po):
    """Final decision rule: arg-max over normal / bacteria / virus, then the top COVID-19 scores of each fold."""
    pc = po[:, 3]
    pred = (po[:, :3] / np.maximum(1 - pc[:, None], 1e-9)).argmax(1)
    half = pred.copy()
    half[pc >= 0.5] = 3
    for k in range(5):
        te = np.where(fold == k)[0]
        pred[te[np.argsort(-pc[te])[:int(round((y[fold != k] == 3).mean() * len(te)))]]] = 3
    return pred, half


single = load_oof(df, "e36b_maira518_clean_llrd").loc[names][P].to_numpy()
NUM["single_e36b"] = stats(single, single.argmax(1))

S1 = ["e2_clahe", "e4_effb3", "e6_effv2s_clahe", "e8_effv2s_clahe448", "e11_effv2s448_pl", "e23_effv2m448_pl"]
X1 = features([load_oof(df, t).loc[names][P].to_numpy() for t in S1]
              + [np.mean([load_oof(df, t).loc[names][P].to_numpy() for t in ("e27_raddino518_pl", "e28_maira518_pl")], 0)])
test_names = pd.read_csv("test_data_sample.csv").iloc[:, 0].astype(str).tolist()
Xk, _ = load_knn("features/rad-dino.npz", names, test_names, y, fold, 4)
X1 = np.concatenate([X1, 0.3 * Xk], 1)
p1 = np.zeros((len(y), 4))
for k in range(5):
    p1[fold == k] = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced").fit(
        X1[fold != k], y[fold != k]).predict_proba(X1[fold == k])
NUM["stage1_stack"] = stats(p1, p1.argmax(1))

for key, f in (("final", "stack_v10_v9_superblend"), ("v10", "stack_v10_clean11_bv2_reidv5"), ("v9", "stack_v9_clean11_bv2_reidv4"),
               ("v2_prior", "stack_v2_reid7_prior")):
    po = np.load(f"submissions/{f}.oof.npy")
    pred, half = quantile_preds(po)
    NUM[key] = stats(po, pred)
    NUM[key]["macro_threshold_0.5"] = float(f1_score(y, half, average="macro"))
NUM["class_counts"] = np.bincount(y).tolist()
tp = pd.read_csv("submissions/stack_v10_v9_superblend.csv").iloc[:, 1:].to_numpy().argmax(1)
NUM["test_pred_counts"] = np.bincount(tp, minlength=4).tolist()
(OUT / "stack_numbers.json").write_text(json.dumps(NUM, indent=1, ensure_ascii=False), encoding="utf-8")
for k in ("single_e36b", "stage1_stack", "v2_prior", "v9", "v10", "final"):
    s = NUM[k]
    print(f"{k:13s} F1 {s['macro']:.4f} {np.round(s['per_class'], 4)} bvAUC {s['bv_auc']:.4f} LL {s['ll']:.4f} "
          f"bvAcc {s['bv_acc']:.4f} swaps {s['bv_swaps']} errors {s['errors']} covidFP {s['covid_fp']} FN {s['covid_fn']}")
print("test counts", NUM["test_pred_counts"], "| CV-LB spearman", round(NUM["cv_lb_spearman"], 3), "n", NUM["cv_lb_n"])

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
for ax, (title, st) in zip(axs, (("單一最佳模型（e36b MAIRA-2）", NUM["single_e36b"]), ("最終堆疊", NUM["final"]))):
    cm = np.array(st["cm"])
    frac = cm / cm.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9, color="white" if frac[i, j] > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(KZ)
    ax.set_yticks(range(4))
    ax.set_yticklabels(KZ)
    ax.set_xlabel("預測")
    ax.set_title(f"{title}\nmacro-F1 {st['macro']:.4f}", fontsize=9.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].set_ylabel("實際")
fig.tight_layout()
fig.savefig(OUT / "confusion.png")
plt.close(fig)
print("figures written to", OUT)
