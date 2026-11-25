"""Frozen-feature probe: RAD-DINO (CXR-pretrained DINOv2 ViT-B/14) embeddings -> logistic regression on our 5 folds."""
import sys, glob, json, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, ".")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

OUT = Path(sys.argv[1]); NAME = sys.argv[2] if len(sys.argv) > 2 else "microsoft/rad-dino"; SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 518
K = ["normal", "bacteria", "virus", "COVID-19"]

class DS(torch.utils.data.Dataset):
    def __init__(s, paths, mean, std): s.p, s.m, s.s = paths, np.array(mean, np.float32), np.array(std, np.float32)
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        im = Image.open(s.p[i]).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        a = (np.asarray(im, np.float32) / 255 - s.m) / s.s
        return torch.from_numpy(a.transpose(2, 0, 1))

def main():
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(NAME); model = AutoModel.from_pretrained(NAME).cuda().eval().half()
    tr = pd.read_csv("data/train_data.csv"); va = pd.read_csv("data/val_data.csv"); te = pd.read_csv("test_data_sample.csv")
    paths = ([f"data/train_images/{f}" for f in tr.new_filename] + [f"data/val_images/{f}" for f in va.new_filename]
             + [f"data/test_images/{f}" for f in te.iloc[:, 0]])
    names = list(tr.new_filename) + list(va.new_filename)
    y_all = np.concatenate([tr[K].values.argmax(1), va[K].values.argmax(1)])
    dl = torch.utils.data.DataLoader(DS(paths, proc.image_mean, proc.image_std), batch_size=16, num_workers=6)
    feats = []
    with torch.no_grad():
        for i, x in enumerate(dl):
            h = model(pixel_values=x.cuda().half()).last_hidden_state.float()
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy())
            if i % 50 == 0: print(i, len(dl), flush=True)
    F = np.concatenate(feats); np.save(OUT / "feats.npy", F)
    n = len(names); Fl, Ft = F[:n], F[n:]
    # same folds as the CNN runs: read val_split.csv of an existing 5-fold tag
    fold = pd.Series(-1, index=names)
    for d in sorted(glob.glob("runs/*of5_e8_effv2s_clahe448")):
        k = json.load(open(d + "/config.json"))["fold"]; fold[pd.read_csv(d + "/val_split.csv").filename.values] = k
    fold = fold.to_numpy(); assert (fold >= 0).all()
    for C in (0.01, 0.1, 1.0):
        oof = np.zeros((n, 4)); tp = np.zeros((len(Ft), 4))
        for k in range(5):
            sc = StandardScaler().fit(Fl[fold != k])
            lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Fl[fold != k]), y_all[fold != k])
            oof[fold == k] = lr.predict_proba(sc.transform(Fl[fold == k])); tp += lr.predict_proba(sc.transform(Ft)) / 5
        f = f1_score(y_all, oof.argmax(1), average=None)
        print(f"C={C}: OOF macro-F1 {f.mean():.4f}  per-class {np.round(f, 3)}", flush=True)
        pd.DataFrame(oof, index=names).assign(label=y_all).to_pickle(OUT / f"oof_C{C}.pkl"); np.save(OUT / f"test_C{C}.npy", tp)

if __name__ == "__main__":
    main()
