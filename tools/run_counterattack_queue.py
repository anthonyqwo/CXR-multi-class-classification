"""Overnight GPU Counterattack Queue to decisively retake #1 (>0.930+):
1. e38_maira518_bv_clean: Full-Data SWA 50/50 blend
2. e42_convnextb384_rad3ch_clean: 100% clean ConvNeXt-Base 384px (rad3ch + LLRD + Soup + SWA)
3. e44_raddino518_bv_clean: 100% clean RAD-DINO 518px Binary B-vs-V Specialist (+ all-4-class OOF + SWA)
4. e40_maira518_softdistill: MAIRA-2 518px Soft-Distilled from 0.92663 zero-PL teacher (COVID excluded!) + SWA
5. e41_raddino518_softdistill: RAD-DINO 518px Soft-Distilled from 0.92663 zero-PL teacher (COVID excluded!) + SWA
"""
from __future__ import annotations

import subprocess
import sys
import time

PY = sys.executable

STEPS = [
    ("1. Full-Data SWA for e38_maira518_bv_clean", [
        PY, "tools/full_train_v2.py", "e38_maira518_bv_clean:7",
    ]),
    ("2a. 5-Fold Clean ConvNeXt-Base 384 rad3ch (e42)", [
        PY, "tools/run_5fold_v2.py", "e42_convnextb384_rad3ch_clean",
        "--model", "convnext_base.fb_in22k_ft_in1k_384",
        "--img_size", "384", "--rad3ch",
        "--epochs", "14", "--batch_size", "8", "--grad_accum", "2",
        "--lr", "8e-5", "--layer_decay", "0.85", "--head_lr_mult", "4.0",
        "--drop_path", "0.2", "--drop_rate", "0.25",
        "--label_smoothing", "0.05", "--class_weight_power", "0.5",
        "--soup_k", "3", "--tta", "orig,zoom5,zoom",
    ]),
    ("2b. Full-Data SWA for e42_convnextb384_rad3ch_clean", [
        PY, "tools/full_train_v2.py", "e42_convnextb384_rad3ch_clean:12",
    ]),
    ("3a. 5-Fold Clean RAD-DINO 518 Binary B-vs-V Specialist (e44)", [
        PY, "tools/run_5fold_v2.py", "e44_raddino518_bv_clean",
        "--bv_only", "--model", "hfdinov2_microsoft--rad-dino", "--img_size", "518",
        "--epochs", "9", "--batch_size", "8", "--grad_accum", "2",
        "--lr", "4e-5", "--layer_decay", "0.8", "--head_lr_mult", "5.0",
        "--freeze_layers", "2", "--drop_path", "0.15", "--drop_rate", "0.25",
        "--label_smoothing", "0.05", "--class_weight_power", "0.0",
        "--soup_k", "3", "--tta", "orig,zoom5,zoom",
    ]),
    ("3b. Full 4-class OOF for e44_raddino518_bv_clean", [
        PY, "tools/eval_bv_oof_all.py", "e44_raddino518_bv_clean",
    ]),
    ("3c. Full-Data SWA for e44_raddino518_bv_clean", [
        PY, "tools/full_train_v2.py", "e44_raddino518_bv_clean:7",
    ]),
    ("4a. 5-Fold Soft-Distilled MAIRA-2 518 (e40, COVID excluded)", [
        PY, "tools/run_5fold_v2.py", "e40_maira518_softdistill",
        "--model", "hfdinov2_microsoft--rad-dino-maira-2", "--img_size", "518",
        "--soft_pseudo_dir", "data/test_images",
        "--soft_pseudo_csv", "test_data_sample.csv",
        "--soft_pseudo_npy", "submissions/stack_v5_clean10_bv_mlp.probs.npy",
        "--pseudo_exclude_covid", "--pseudo_weight", "0.5",
        "--epochs", "10", "--batch_size", "8", "--grad_accum", "2",
        "--lr", "4e-5", "--layer_decay", "0.8", "--head_lr_mult", "5.0",
        "--freeze_layers", "2", "--drop_path", "0.15", "--drop_rate", "0.25",
        "--label_smoothing", "0.05", "--class_weight_power", "0.5",
        "--soup_k", "3", "--tta", "orig,zoom5,zoom",
    ]),
    ("4b. Full-Data SWA for e40_maira518_softdistill", [
        PY, "tools/full_train_v2.py", "e40_maira518_softdistill:8",
    ]),
    ("5a. 5-Fold Soft-Distilled RAD-DINO 518 (e41, COVID excluded)", [
        PY, "tools/run_5fold_v2.py", "e41_raddino518_softdistill",
        "--model", "hfdinov2_microsoft--rad-dino", "--img_size", "518",
        "--soft_pseudo_dir", "data/test_images",
        "--soft_pseudo_csv", "test_data_sample.csv",
        "--soft_pseudo_npy", "submissions/stack_v5_clean10_bv_mlp.probs.npy",
        "--pseudo_exclude_covid", "--pseudo_weight", "0.5",
        "--epochs", "10", "--batch_size", "8", "--grad_accum", "2",
        "--lr", "4e-5", "--layer_decay", "0.8", "--head_lr_mult", "5.0",
        "--freeze_layers", "2", "--drop_path", "0.15", "--drop_rate", "0.25",
        "--label_smoothing", "0.05", "--class_weight_power", "0.5",
        "--soup_k", "3", "--tta", "orig,zoom5,zoom",
    ]),
    ("5b. Full-Data SWA for e41_raddino518_softdistill", [
        PY, "tools/full_train_v2.py", "e41_raddino518_softdistill:8",
    ]),
]


def main():
    t_all = time.time()
    for title, cmd in STEPS:
        t0 = time.time()
        print(f"\n==================== {title} ====================", flush=True)
        r = subprocess.run(cmd)
        dt = (time.time() - t0) / 60.0
        if r.returncode != 0:
            print(f"[FAILED] {title} exited with code {r.returncode} after {dt:.1f} min", flush=True)
            sys.exit(r.returncode)
        print(f"[OK] {title} finished in {dt:.1f} min", flush=True)
    print(f"\nALL COUNTERATTACK STAGES COMPLETE in {(time.time()-t_all)/60.0:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
