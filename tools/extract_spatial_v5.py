"""Extract multi-layer 3x3 & 2x2 spatial biometric CXR embeddings for Re-ID v5:
- RAD-DINO (518px): 3x3 spatial grid at Layer 6 & Layer 9 (6,912 dims each, fp16)
- MAIRA-2  (518px): 3x3 spatial grid at Layer 6 & Layer 9 (6,912 dims each, fp16)
- StanfordAIMI XraySigLIP ViT-L/16 (512px, 24 layers): 2x2 spatial grid at Layer 16 & Layer 20 (4,096 dims each, fp16)
- Meta DINOv2-Large (518px, 24 layers): 2x2 spatial grid at Layer 16 (5,120 dims, fp16)
"""
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
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

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
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


def grid_pool(hidden: torch.Tensor, g_out: int, has_cls: bool = True) -> torch.Tensor:
    """Pool patch grid into g_out x g_out spatial zones (+ CLS if has_cls)."""
    patches = hidden[:, 1:, :] if has_cls else hidden
    b, n_tok, d = patches.shape
    g = int(round(math.sqrt(n_tok)))
    if g * g != n_tok:
        g = int(math.isqrt(n_tok))
        patches = patches[:, -g * g:, :]
    grid = patches.reshape(b, g, g, d).permute(0, 3, 1, 2)  # (B, D, g, g)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (g_out, g_out))  # (B, D, g_out, g_out)
    flat = pooled.flatten(1)  # (B, D * g_out * g_out)
    if has_cls:
        return torch.cat([hidden[:, 0, :], flat], dim=1)
    return flat


@torch.no_grad()
def extract_vit(model_id: str, out_path: str, paths: list[str], names: np.ndarray,
                layer_specs: list[tuple[str, int, int]], is_siglip: bool = False, bs: int = 32):
    out = Path(out_path)
    if out.exists():
        print(f"[skip] {out} already exists", flush=True)
        return
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(model_id)
    if is_siglip:
        model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()
    else:
        model = AutoModel.from_pretrained(model_id).to(device).eval()

    ds = RawImgDS(paths, proc)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=6, pin_memory=True)

    buf = {k: [] for k, _, _ in layer_specs}
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hs = model(pixel_values=xb, output_hidden_states=True).hidden_states
            for k, l_idx, g_out in layer_specs:
                buf[k].append(grid_pool(hs[l_idx], g_out, has_cls=not is_siglip).float().cpu().numpy().astype(np.float16))

    arrays = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=names, **arrays)
    shapes = {k: v.shape for k, v in arrays.items()}
    print(f"[done] {out} {shapes} ({time.time()-t0:.1f}s)", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    tr = read_split("data/train_images", "data/train_data.csv")
    va = read_split("data/val_images", "data/val_data.csv")
    te = read_split("data/test_images", "test_data_sample.csv")
    all_df = pd.concat([tr, va, te], ignore_index=True)
    paths = all_df["path"].tolist()
    names = np.array(all_df["filename"].astype(str).tolist())
    print(f"Extracting Re-ID v5 multi-layer spatial biometric embeddings for {len(paths)} images...", flush=True)

    extract_vit("microsoft/rad-dino", "features/rad-dino-sp3.npz", paths, names,
                [("feats_l6", 6, 3), ("feats_l9", 9, 3)], is_siglip=False, bs=32)
    extract_vit("microsoft/rad-dino-maira-2", "features/maira2-sp3.npz", paths, names,
                [("feats_l6", 6, 3), ("feats_l9", 9, 3)], is_siglip=False, bs=32)
    extract_vit("StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", "features/xraysiglip-spatial.npz", paths, names,
                [("feats_l16", 16, 2), ("feats_l20", 20, 2)], is_siglip=True, bs=16)
    extract_vit("facebook/dinov2-large", "features/dinov2l-spatial.npz", paths, names,
                [("feats_l16", 16, 2)], is_siglip=False, bs=16)


if __name__ == "__main__":
    main()
