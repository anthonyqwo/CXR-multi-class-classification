"""Data loading, preprocessing and augmentation for the CXR 4-class task.

Two input layouts are supported:
  1. CSV + image folder: CSV has `new_filename` plus one-hot columns
     `normal, bacteria, virus, COVID-19` (empty columns = unlabeled test set).
  2. Folder-per-class: <dir>/<class_name>/*.jpeg  (class names case-insensitive).
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

CLASSES = ["normal", "bacteria", "virus", "COVID-19"]
NUM_CLASSES = len(CLASSES)
IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_FOLDER_ALIASES = {
    "normal": 0,
    "bacteria": 1, "bacterial": 1,
    "virus": 2, "viral": 2,
    "covid-19": 3, "covid19": 3, "covid": 3,
}


# --------------------------------------------------------------------------- #
# Split loading
# --------------------------------------------------------------------------- #
def read_split(img_dir: str, csv_path: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns [filename, path, label] (label=-1 if unknown)."""
    img_dir = Path(img_dir)
    if csv_path:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        if "new_filename" not in cols:
            raise ValueError(f"{csv_path}: missing 'new_filename' column")
        fn_col = cols["new_filename"]
        class_cols = [cols.get(c.lower()) for c in CLASSES]
        if any(c is None for c in class_cols):
            labels = np.full(len(df), -1)
        else:
            onehot = df[class_cols].apply(pd.to_numeric, errors="coerce")
            if onehot.isna().to_numpy().all():
                labels = np.full(len(df), -1)  # unlabeled (test template)
            else:
                arr = onehot.fillna(0).to_numpy()
                bad = (arr.sum(1) != 1).sum()
                if bad:
                    raise ValueError(f"{csv_path}: {bad} rows are not valid one-hot labels")
                labels = arr.argmax(1)
        return pd.DataFrame({
            "filename": df[fn_col].astype(str),
            "path": [str(img_dir / f) for f in df[fn_col].astype(str)],
            "label": labels.astype(int),
        })

    rows = []
    subdirs = sorted(p for p in img_dir.iterdir() if p.is_dir())
    for sub in subdirs:
        key = sub.name.lower()
        if key not in _FOLDER_ALIASES:
            raise ValueError(f"Unknown class folder '{sub.name}' in {img_dir}")
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                rows.append((f.name, str(f), _FOLDER_ALIASES[key]))
    if not subdirs:  # flat folder of unlabeled images
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                rows.append((f.name, str(f), -1))
    if not rows:
        raise ValueError(f"No images found in {img_dir}")
    return pd.DataFrame(rows, columns=["filename", "path", "label"])


# --------------------------------------------------------------------------- #
# Preprocessing (deterministic, applied to train/val/test alike)
# --------------------------------------------------------------------------- #
def load_gray(path: str) -> np.ndarray:
    # np.fromfile + imdecode handles non-ASCII Windows paths, unlike cv2.imread.
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot decode image: {path}")
    return img


class Preprocessor:
    """path -> decode -> optional lung crop / border crop -> optional letterbox -> resize -> optional CLAHE / rad3ch.

    With `cache_dir`, the expensive part (decoding a 1-2k px JPEG and resizing) is done
    once and stored as a small lossless PNG; later epochs / runs read the PNG instead.
    The cache key includes the source path, size, mtime, load_size, crop_border, letterbox,
    and lung_crop settings, so changing any of them never reuses a stale entry. CLAHE and
    rad3ch are applied after the cache, so they can be switched on/off without rebuilding.
    """

    def __init__(self, load_size: int, clahe_clip: float = 0.0, crop_border: float = 0.0,
                 cache_dir: str | None = None, rad3ch: bool = False, letterbox: bool = False,
                 lung_crop_csv: str | None = None, lung_margin: float = 0.12):
        self.load_size = load_size
        self.clahe_clip = clahe_clip
        self.crop_border = crop_border
        self.rad3ch = rad3ch
        self.letterbox = letterbox
        self.lung_crop_csv = lung_crop_csv
        self.lung_margin = lung_margin
        self._lung_boxes: dict[str, tuple[float, float, float, float]] | None = None
        if lung_crop_csv:
            bdf = pd.read_csv(lung_crop_csv)
            self._lung_boxes = {
                str(r.filename): (float(r.y1), float(r.x1), float(r.y2), float(r.x2))
                for r in bdf.itertuples(index=False)
            }
        tag = f"s{load_size}_c{crop_border:g}"
        if letterbox:
            tag += "_lb"
        if lung_crop_csv:
            tag += f"_lc{lung_margin:g}"
        self.cache_dir = Path(cache_dir) / tag if cache_dir else None
        self._clahe = None  # created lazily: cv2 objects are not picklable
        self._clahe2 = None
        self._clahe4 = None

    def _cache_path(self, path: str) -> Path:
        st = os.stat(path)
        key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
        return self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}.png"

    def _decode_resize(self, path: str) -> np.ndarray:
        img = load_gray(path)
        if self._lung_boxes is not None:
            fn = Path(path).name
            if fn in self._lung_boxes:
                y1, x1, y2, x2 = self._lung_boxes[fn]
                h, w = img.shape
                bh, bw = y2 - y1, x2 - x1
                my, mx = bh * self.lung_margin, bw * self.lung_margin
                iy1 = max(0, int((y1 - my) * h))
                iy2 = min(h, int((y2 + my) * h))
                ix1 = max(0, int((x1 - mx) * w))
                ix2 = min(w, int((x2 + mx) * w))
                if (iy2 - iy1) >= 32 and (ix2 - ix1) >= 32:
                    img = img[iy1:iy2, ix1:ix2]
        elif self.crop_border > 0:
            h, w = img.shape
            dy, dx = int(h * self.crop_border), int(w * self.crop_border)
            img = img[dy:h - dy, dx:w - dx]
        if self.letterbox:
            h, w = img.shape
            side = max(h, w)
            pad_top = (side - h) // 2
            pad_bot = side - h - pad_top
            pad_left = (side - w) // 2
            pad_right = side - w - pad_left
            img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=0)
        return cv2.resize(img, (self.load_size, self.load_size), interpolation=cv2.INTER_AREA)

    def load_resized(self, path: str) -> np.ndarray:
        if self.cache_dir is None:
            return self._decode_resize(path)
        cp = self._cache_path(path)
        if cp.exists():
            img = cv2.imdecode(np.fromfile(str(cp), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape == (self.load_size, self.load_size):
                return img
        img = self._decode_resize(path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:  # write-then-rename so concurrent workers/runs never see a partial file
            tmp = cp.with_name(f"{cp.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            buf.tofile(str(tmp))
            try:
                os.replace(tmp, cp)
            except OSError:  # Windows: target locked by a concurrent reader -> already cached
                tmp.unlink(missing_ok=True)
        return img

    def __call__(self, path: str) -> np.ndarray:
        img = self.load_resized(path)
        if self.rad3ch:
            if self._clahe2 is None:
                self._clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                self._clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            return np.stack([img, self._clahe2.apply(img), self._clahe4.apply(img)], axis=-1)
        if self.clahe_clip > 0:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            img = self._clahe.apply(img)
        return img


def make_preprocessor(cfg: dict, cache_dir: str | None = None) -> Preprocessor:
    """Shared by train / val / test so every split is preprocessed identically.
    Images are stored ~15% larger than img_size so random crops keep detail."""
    return Preprocessor(
        int(cfg["img_size"] * 1.15),
        cfg.get("clahe", 0.0),
        cfg.get("crop_border", 0.0),
        cache_dir or None,
        rad3ch=bool(cfg.get("rad3ch", False)),
        letterbox=bool(cfg.get("letterbox", False)),
        lung_crop_csv=cfg.get("lung_crop", None),
        lung_margin=float(cfg.get("lung_margin", 0.12)),
    )


def warm_cache(pre: Preprocessor, paths: list[str], workers: int = 8):
    """Fill the cache in parallel (threads: cv2 releases the GIL). Returns #images processed."""
    if pre.cache_dir is None:
        return 0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for _ in ex.map(pre.load_resized, paths, chunksize=16):
            pass
    return len(paths)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def _to_tensor_norm():
    return [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def build_train_transform(img_size: int, aug: str = "light", hflip: bool = False):
    if aug == "none":
        ops = [T.Resize((img_size, img_size), antialias=True)]
        post = []
    elif aug == "light":
        ops = [
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomAffine(degrees=7, translate=(0.03, 0.03)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
        post = []
    elif aug == "strong":
        ops = [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
            T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.95, 1.05), shear=5),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ColorJitter(brightness=0.25, contrast=0.25),
        ]
        post = [T.RandomErasing(p=0.25, scale=(0.02, 0.1))]
    else:
        raise ValueError(f"Unknown aug level: {aug}")
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    return T.Compose(ops + _to_tensor_norm() + post)


def build_eval_transform(img_size: int, mode: str = "orig"):
    """Deterministic transform. `mode` doubles as the TTA view name."""
    if mode == "orig":
        ops = [T.Resize((img_size, img_size), antialias=True)]
    elif mode == "zoom":  # slight center zoom (~10%)
        ops = [T.Resize((int(img_size * 1.1), int(img_size * 1.1)), antialias=True), T.CenterCrop(img_size)]
    elif mode == "zoom5":  # gentler center zoom (~5%)
        ops = [T.Resize((int(img_size * 1.05), int(img_size * 1.05)), antialias=True), T.CenterCrop(img_size)]
    elif mode == "rot_p4":
        ops = [T.Resize((img_size, img_size), antialias=True), T.RandomRotation((4, 4))]
    elif mode == "rot_m4":
        ops = [T.Resize((img_size, img_size), antialias=True), T.RandomRotation((-4, -4))]
    elif mode == "hflip":
        ops = [T.Resize((img_size, img_size), antialias=True), T.RandomHorizontalFlip(p=1.0)]
    else:
        raise ValueError(f"Unknown TTA mode: {mode}")
    return T.Compose(ops + _to_tensor_norm())


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: Preprocessor, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy()
        self.pre = preprocessor
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.pre(self.paths[idx])
        img = Image.fromarray(img).convert("RGB")  # 3 channels for ImageNet-pretrained backbones
        return self.transform(img), int(self.labels[idx])
