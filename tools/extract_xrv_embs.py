"""Extract frozen TorchXRayVision CXR embeddings + 18 clinical pathology predictions
(densenet121-res224-all and resnet50-res512-all) across all 5,908 train+val+test images."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, ".")
from dataset import read_split


class XRVDataset(Dataset):
    def __init__(self, paths: list[str], img_size: int):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.paths[i])
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        arr = xrv.datasets.normalize(img.astype(np.float32), 255.0)
        return torch.from_numpy(arr[None, :, :])


@torch.no_grad()
def extract_one(model_kind: str, weights: str, img_size: int, out_path: str, paths: list[str], names: np.ndarray):
    out = Path(out_path)
    if out.exists():
        print(f"[skip] {out} already exists", flush=True)
        return
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.models.DenseNet(weights=weights) if model_kind == "d121" else xrv.models.ResNet(weights=weights)
    model = model.to(device).eval()

    ds = XRVDataset(paths, img_size)
    bs = 64 if img_size <= 256 else 24
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)

    feats_list, path_list = [], []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = model.features2(xb) if model_kind == "d121" else model.features(xb)
            p = model(xb)
        feats_list.append(f.float().cpu().numpy())
        path_list.append(p.float().cpu().numpy())

    feats = np.concatenate(feats_list, axis=0).astype(np.float32)
    pathos = np.concatenate(path_list, axis=0).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        names=names,
        feats=feats,
        pathologies=pathos,
        pathology_names=np.array(model.pathologies),
    )
    print(f"[done] {out} feats={feats.shape} pathologies={pathos.shape} ({time.time()-t0:.1f}s)", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    tr = read_split("data/train_images", "data/train_data.csv")
    va = read_split("data/val_images", "data/val_data.csv")
    te = read_split("data/test_images", "test_data_sample.csv")
    all_df = pd.concat([tr, va, te], ignore_index=True)
    paths = all_df["path"].tolist()
    names = np.array(all_df["filename"].astype(str).tolist())
    print(f"Extracting XRV features for {len(paths)} images...", flush=True)
    extract_one("d121", "densenet121-res224-all", 224, "features/xrv-densenet121-all.npz", paths, names)
    extract_one("r50", "resnet50-res512-all", 512, "features/xrv-resnet50-all.npz", paths, names)


if __name__ == "__main__":
    main()
