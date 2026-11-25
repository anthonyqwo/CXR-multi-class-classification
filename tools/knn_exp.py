"""CPU experiments on the nearest-neighbour features of stack.py (nested CV on the OOF set).

python tools/knn_exp.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from stack import features, knn_features, load_folds, load_oof
from summarize import load_runs

TAGS = ["e2_clahe", "e4_effb3", "e6_effv2s_clahe", "e8_effv2s_clahe448", "e11_effv2s448_pl", "e23_effv2m448_pl",
        "e27_raddino518_pl~e28_maira518_pl"]
P = [0, 1, 2, 3]
df = load_runs(Path("runs"))
df = df[df.complete & (df.folds > 0)].sort_values("dir").groupby(["tag", "fold"], as_index=False).last()
oofs = [load_oof(df, t) for t in TAGS]
names = oofs[0].index
y = oofs[0].label.to_numpy()
fold = load_folds(df, TAGS[0]).to_numpy()
probs = [o[P].to_numpy() for o in oofs]
X = features(probs)
avg = np.mean(probs, 0)
n = len(y)


def emb(name):
    z = np.load(f"features/{name}.npz")
    f = pd.DataFrame(z["feats"], index=z["names"].astype(str)).loc[names].to_numpy(np.float32)
    f = f - f.mean(0)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def other_fold(sim):
    s = sim.copy()
    for k in range(5):
        s[np.ix_(fold == k, fold == k)] = -1
    return s


def same_fold(sim):
    s = np.full_like(sim, -1)
    for k in range(5):
        ix = np.ix_(fold == k, fold == k)
        s[ix] = sim[ix]
    np.fill_diagonal(s, -1)
    return s


def peer_features(sim, p, ks=(1, 5), temp=0.05):
    """Similarity-weighted mean of the model probabilities of unlabelled peers (same fold = stand-in for test-test)."""
    order = np.argsort(-sim, 1)[:, :max(ks)]
    s = np.take_along_axis(sim, order, 1)
    out = []
    for k in ks:
        w = np.exp((s[:, :k] - s[:, :1]) / temp)
        out.append(np.log((w[:, :, None] * p[order[:, :k]]).sum(1) / w.sum(1, keepdims=True) + 1e-2))
    return np.concatenate(out + [s[:, :1]], 1)


def ncv(X, C=0.01, reps=5):
    r = []
    for seed in range(reps):
        p = np.zeros(n, int)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            p[b] = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[a], y[a]).predict(X[b])
        r.append(f1_score(y, p, average="macro"))
    return f"{np.mean(r):.4f} ±{np.std(r):.4f}"


E = {k: emb(k) for k in ("rad-dino", "rad-dino-maira-2", "dinov2-base")}
S = {k: v @ v.T for k, v in E.items()}
K = {k: knn_features(other_fold(s), y, 4) for k, s in S.items()}
print("base (no knn)                    ", ncv(X), flush=True)
for k in K:
    print(f"knn {k:28s} ", ncv(np.concatenate([X, 0.3 * K[k]], 1)), flush=True)
print("knn rad-dino + maira (separate)  ", ncv(np.concatenate([X, 0.3 * K["rad-dino"], 0.3 * K["rad-dino-maira-2"]], 1)), flush=True)
print("knn all three (separate)         ", ncv(np.concatenate([X] + [0.3 * v for v in K.values()], 1)), flush=True)
for combo in (("rad-dino", "rad-dino-maira-2"), ("rad-dino", "dinov2-base"), ("rad-dino", "rad-dino-maira-2", "dinov2-base")):
    sm = np.mean([S[c] for c in combo], 0)
    kc = knn_features(other_fold(sm), y, 4)
    print(f"knn mean-sim {'+'.join(c[:8] for c in combo):20s}", ncv(np.concatenate([X, 0.3 * kc], 1)), flush=True)
    for sc in (0.2, 0.5):
        print(f"    scale {sc}", ncv(np.concatenate([X, sc * kc], 1)), flush=True)

# transductive peers: unlabelled neighbours with their model probabilities
for name in ("rad-dino",):
    pf = peer_features(same_fold(S[name]), avg)
    for sc in (0.3, 1.0):
        print(f"peers only ({name}) x{sc}            ", ncv(np.concatenate([X, sc * pf], 1)), flush=True)
        print(f"knn + peers ({name}) x{sc}           ", ncv(np.concatenate([X, 0.3 * K[name], sc * pf], 1)), flush=True)
