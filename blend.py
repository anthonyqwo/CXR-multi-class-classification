"""Blend the test probabilities of each tag for stack.py: 5-fold mean and full-data model, half each.

For every tag it reads <fold_dir>/<tag>.probs.npy (predict.py on the 5 fold checkpoints) and
<full_dir>/<tag>.probs.npy (predict.py on the model trained with --merge_val) and writes their
mean to <out_dir>/<tag>.probs.npy. A tag written as `tagA~tagB` is the mean of the blended
probabilities of its parts, which is what stack.py expects for seed- or model-averaged inputs.

python blend.py --tags e2_clahe e23_effv2m448_pl "e27_raddino518_pl~e28_maira518_pl" \
                --out_dir submissions/per_tag_blend
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--fold_dir", default="submissions/per_tag")
    ap.add_argument("--full_dir", default="submissions/per_tag_full")
    ap.add_argument("--full_weight", type=float, default=0.5, help="weight of the full-data model")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def blend(tag: str) -> np.ndarray:
        fold = np.load(Path(args.fold_dir) / f"{tag}.probs.npy")
        full = np.load(Path(args.full_dir) / f"{tag}.probs.npy")
        return (1 - args.full_weight) * fold + args.full_weight * full

    for tag in args.tags:
        probs = np.mean([blend(t) for t in tag.split("~")], axis=0)
        np.save(out / f"{tag}.probs.npy", probs)
        print(f"[blend] {tag}: {probs.shape[0]} images -> {out / (tag + '.probs.npy')}")


if __name__ == "__main__":
    main()
