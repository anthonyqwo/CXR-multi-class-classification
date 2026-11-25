"""Extract spatially-structured biometric CXR embeddings (CLS + 2x2 spatial quadrant patch means
+ intermediate layer-9 spatial means) from RAD-DINO and MAIRA-2 across all 5,908 images.
Preserves left/right clavicle, ribcage, heart silhouette, and diaphragm geometry for same-patient Re-ID!"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, ".")
from dataset import read_split


class RawImgDS(Dataset):
    def __init__(self, paths: list[str], proc):
        self.paths = paths
        self.proc = proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return px


def spatial_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Concatenate [CLS, 2x2 quadrant means of patch tokens] -> (B, 5*D)."""
    cls_tok = hidden[:, 0, :]  # (B, D)
    patches = hidden[:, 1:, :]  # (B, N, D)
    b, n_tok, d = patches.shape
    g = int(round(math.sqrt(n_tok)))
    if g * g != n_tok:
        # Handle register tokens if any: take last g*g tokens
        g = int(math.isqrt(n_tok))
        patches = patches[:, -g * g:, :]
    grid = patches.reshape(b, g, g, d)
    h = g // 2
    q00 = grid[:, :h, :h, :].mean(dim=(1, 2))
    q01 = grid[:, :h, h:, :].mean(dim=(1, 2))
    q10 = grid[:, h:, :h, :].mean(dim=(1, 2))
    q11 = grid[:, h:, h:, :].mean(dim=(1, 2))
    return torch.cat([cls_tok, q00, q01, q10, q11], dim=1)


@torch.no_grad()
def extract_model(model_id: str, out_path: str, paths: list[str], names: np.ndarray, bs: int = 32):
    out = Path(out_path)
    if out.exists():
        print(f"[skip] {out} already exists", flush=True)
        return
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    ds = RawImgDS(paths, proc)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)

    f_last, f_mid = [], []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out_m = model(pixel_values=xb, output_hidden_states=True)
            hs = out_m.hidden_states
            # Last layer (index -1) and mid-late layer (~75% depth, e.g. layer 9 of 12)
            mid_idx = max(1, (len(hs) * 3) // 4)
            f_last.append(spatial_pool(hs[-1]).float().cpu().numpy())
            f_mid.append(spatial_pool(hs[mid_idx]).float().cpu().numpy())

    last_arr = np.concatenate(f_last, axis=0).astype(np.float16)
    mid_arr = np.concatenate(f_mid, axis=0).astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=names, feats_last=last_arr, feats_mid=mid_arr)
    print(f"[done] {out} last={last_arr.shape} mid={mid_arr.shape} ({time.time()-t0:.1f}s)", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    tr = read_split("data/train_images", "data/train_data.csv")
    va = read_split("data/val_images", "data/val_data.csv")
    te = read_split("data/test_images", "test_data_sample.csv")
    all_df = pd.concat([tr, va, te], ignore_index=True)
    paths = all_df["path"].tolist()
    names = np.array(all_df["filename"].astype(str).tolist())
    print(f"Extracting 2x2 spatial + mid-layer biometric Re-ID embeddings for {len(paths)} images...", flush=True)
    extract_model("microsoft/rad-dino", "features/rad-dino-spatial.npz", paths, names, bs=32)
    extract_model("microsoft/rad-dino-maira-2", "features/maira2-spatial.npz", paths, names, bs=32)


if __name__ == "__main__":
    main()
