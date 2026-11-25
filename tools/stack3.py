import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from summarize import load_runs
from utils import compute_metrics, tune_class_bias
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
df = load_runs(Path("runs")); df = df[df.complete & (df.folds == 5)]
df = df.sort_values("dir").groupby(["tag","fold"], as_index=False).last()
def load(tag):
    if "~" in tag:  # seed-averaged tag: a~b
        os_ = [load(t) for t in tag.split("~")]; o = os_[0].copy(); o[[0,1,2,3]] = np.mean([x[[0,1,2,3]].to_numpy() for x in os_], 0); return o
    parts=[]
    for d in df[df.tag==tag].dir:
        v = pd.read_csv(Path(d)/"val_split.csv"); p = np.load(Path(d)/"val_probs.npy")
        parts.append(pd.DataFrame(p, index=v.filename).assign(label=v.label.values))
    o = pd.concat(parts).sort_index(); assert len(o)==4726, tag; return o
sets = [s.split(",") for s in sys.argv[1:]]
oof = {t: load(t) for t in set(sum(sets, []))}; y = next(iter(oof.values())).label.to_numpy()
for tags in sets:
    Ps = [oof[t][[0,1,2,3]].to_numpy() for t in tags]; avg = np.mean(Ps,0)
    X = np.log(np.concatenate(Ps, 1) + 1e-7); r = {"avg+bias": [], "stack C=.01": [], "stack C=.03": []}
    for seed in range(5):
        pr = {k: np.zeros(len(y), int) for k in r}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            b,_,_ = tune_class_bias(avg[tr], y[tr]); pr["avg+bias"][te] = (np.log(avg[te]+1e-9)+b).argmax(1)
            for C in (0.01, 0.03): pr[f"stack C={str(C)[1:]}"][te] = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X[tr], y[tr]).predict(X[te])
        for k in r: r[k].append(compute_metrics(y, pr[k])["macro_f1"])
    print(f"{'+'.join(tags)}\n   " + "  ".join(f"{k} {np.mean(v):.4f}±{np.std(v):.4f}" for k, v in r.items()), flush=True)
