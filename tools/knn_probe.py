"""CPU: does a nearest-labelled-neighbour signal (frozen RAD-DINO embeddings) add to the stack?"""
import sys, glob, json, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
S = Path(sys.argv[1]); K = ["normal", "bacteria", "virus", "COVID-19"]
tr = pd.read_csv("data/train_data.csv"); va = pd.read_csv("data/val_data.csv")
names = list(tr.new_filename) + list(va.new_filename); n = len(names)
y = np.concatenate([tr[K].values.argmax(1), va[K].values.argmax(1)])
F = np.load(S / "raddino/feats.npy").astype(np.float32); F -= F[:n].mean(0); F /= np.linalg.norm(F, axis=1, keepdims=True)
fold = pd.Series(-1, index=names); oofs = {}
tags = ["e2_clahe","e4_effb3","e6_effv2s_clahe","e8_effv2s_clahe448","e11_effv2s448_pl","e23_effv2m448_pl","e27_raddino518_pl","e28_maira518_pl"]
for t in tags:
    parts = []
    for d in sorted(glob.glob(f"runs/*of5_{t}")):
        v = pd.read_csv(d + "/val_split.csv"); fold[v.filename.values] = json.load(open(d + "/config.json"))["fold"]
        parts.append(pd.DataFrame(np.load(d + "/val_probs.npy"), index=v.filename))
    oofs[t] = pd.concat(parts).loc[names].to_numpy()
fold = fold.to_numpy(); assert (fold >= 0).all()
Sim = F[:n] @ F[:n].T
for k in range(5): Sim[np.ix_(fold == k, fold == k)] = -1   # neighbours only from other folds
order = np.argsort(-Sim, 1)[:, :20]; sims = np.take_along_axis(Sim, order, 1); labs = y[order]
print("nearest other-fold neighbour: label agreement by similarity")
for lo, hi in [(0.9, 1.1), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0, 0.5)]:
    m = (sims[:, 0] >= lo) & (sims[:, 0] < hi); bv = m & np.isin(y, [1, 2])
    if m.sum(): print(f"  sim [{lo},{hi}) n={m.sum():5d} agree {np.mean(labs[m, 0] == y[m]):.3f} | bact/virus n={bv.sum()} agree {np.mean(labs[bv, 0] == y[bv]) if bv.any() else 0:.3f}")
def knn_feats(sims, labs):
    out = []
    for k, T in [(1, 1), (5, 0.05), (20, 0.05)]:
        w = np.exp((sims[:, :k] - sims[:, :1]) / T); v = np.stack([(w * (labs[:, :k] == c)).sum(1) for c in range(4)], 1)
        out.append(np.log(v / v.sum(1, keepdims=True) + 1e-2))
    return np.concatenate(out + [sims[:, :1]], 1)
Xk = knn_feats(sims, labs)
avg = (oofs["e27_raddino518_pl"] + oofs["e28_maira518_pl"]) / 2
base = np.log(np.concatenate([oofs[t] for t in tags[:6]] + [avg], 1) + 1e-7)
def ncv(X, C=0.01):
    r = []
    for seed in range(5):
        p = np.zeros(n, int)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            p[b] = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[a], y[a]).predict(X[b])
        r.append(f1_score(y, p, average="macro"))
    return f"{np.mean(r):.4f}±{np.std(r):.4f}"
print("stack base            ", ncv(base))
for sc in (1.0, 0.3):
    print(f"stack base + knn (x{sc})", ncv(np.concatenate([base, Xk * sc], 1)))
print("knn only              ", ncv(Xk, 1.0))
