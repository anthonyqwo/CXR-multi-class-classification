"""CPU-only: how similar are test images to labeled images (same-patient near-duplicates)?"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

K = ["normal", "bacteria", "virus", "COVID-19"]
def load_csv(csv, d):
    df = pd.read_csv(csv); df["path"] = [str(Path(d) / f) for f in df.new_filename]
    if all(k in df for k in K): df["label"] = df[K].values.argmax(1)
    return df
tr = load_csv("data/train_data.csv", "data/train_images"); va = load_csv("data/val_data.csv", "data/val_images")
lab = pd.concat([tr, va], ignore_index=True); te = load_csv("test_data_sample.csv", "data/test_images")

def thumb(p):
    im = Image.open(p).convert("L"); w, h = im.size
    m = int(min(w, h) * 0.05); im = im.crop((m, m, w - m, h - m)).resize((48, 48), Image.BILINEAR)
    a = np.asarray(im, np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-6); return a / np.linalg.norm(a)
with ThreadPoolExecutor(12) as ex:
    L = np.stack(list(ex.map(thumb, lab.path))); T = np.stack(list(ex.map(thumb, te.path)))
y = lab.label.to_numpy()

S = L @ L.T; np.fill_diagonal(S, -1)
nn = S.argmax(1); s = S.max(1)
print("labeled -> nearest other labeled image (pixel cosine of 48x48 thumbnails)")
for lo, hi in [(0.999, 1.01), (0.98, 0.999), (0.95, 0.98), (0.9, 0.95), (0.85, 0.9), (0, 0.85)]:
    m = (s >= lo) & (s < hi)
    if m.sum(): print(f"  sim [{lo:.3f},{hi:.3f}): n={m.sum():5d}  same label {np.mean(y[nn[m]] == y[m]):.3f}"
                      f"  | among bacteria/virus: n={np.isin(y[m],[1,2]).sum()} agree {np.mean((y[nn[m]]==y[m])[np.isin(y[m],[1,2])]) if np.isin(y[m],[1,2]).any() else float('nan'):.3f}")
St = T @ L.T; st = St.max(1)
print("test -> nearest labeled image")
for lo, hi in [(0.999, 1.01), (0.98, 0.999), (0.95, 0.98), (0.9, 0.95), (0.85, 0.9), (0, 0.85)]:
    print(f"  sim [{lo:.3f},{hi:.3f}): n={((st >= lo) & (st < hi)).sum()}")
np.save(sys.argv[1] if len(sys.argv) > 1 else "nn_test.npy", np.stack([St.argmax(1), st], 1))
