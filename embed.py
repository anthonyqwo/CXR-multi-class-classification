"""Extract frozen image embeddings (no fine-tuning) for the nearest-neighbour features of stack.py.

Every train, val and test image is resized to a square, normalised with the encoder's own
statistics and passed once through a Hugging Face encoder. The embedding is the concatenation of
the CLS token and the mean patch token. Output: one .npz with `names` (file names) and `feats`.

python embed.py --model microsoft/rad-dino --out features/rad-dino.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, size, mean, std):
        self.paths, self.size = paths, size
        self.mean, self.std = np.array(mean, np.float32), np.array(std, np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        a = (np.asarray(im, np.float32) / 255 - self.mean) / self.std
        return torch.from_numpy(a.transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/rad-dino", help="Hugging Face encoder name")
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--train_dir", default="data/train_images")
    ap.add_argument("--train_csv", default="data/train_data.csv")
    ap.add_argument("--val_dir", default="data/val_images")
    ap.add_argument("--val_csv", default="data/val_data.csv")
    ap.add_argument("--test_dir", default="data/test_images")
    ap.add_argument("--test_csv", default="test_data_sample.csv")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda().eval().half()

    names, paths = [], []
    for d, csv in ((args.train_dir, args.train_csv), (args.val_dir, args.val_csv), (args.test_dir, args.test_csv)):
        files = pd.read_csv(csv).iloc[:, 0].astype(str).tolist()
        names += files
        paths += [str(Path(d) / f) for f in files]
    if len(set(names)) != len(names):
        raise ValueError("file names must be unique across train, val and test")

    loader = torch.utils.data.DataLoader(Images(paths, args.img_size, proc.image_mean, proc.image_std),
                                         batch_size=args.batch_size, num_workers=args.workers)
    feats = []
    with torch.no_grad():
        for i, x in enumerate(loader):
            h = model(pixel_values=x.cuda().half()).last_hidden_state.float()
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy())
            if i % 50 == 0:
                print(f"[embed] batch {i}/{len(loader)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names), feats=np.concatenate(feats).astype(np.float32))
    print(f"[done] wrote {out}  {len(names)} images")


if __name__ == "__main__":
    main()
