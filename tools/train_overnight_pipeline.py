"""Sequential overnight GPU training queue for diverse, clean 5-fold models.
Each experiment runs all 5 folds via tools/run_5fold_v2.py (auto-resumes finished folds)
and writes TTA test probabilities to submissions/per_tag_v2/<tag>.probs.npy.
"""
import subprocess
import sys

EXPERIMENTS = [
    # 1. Swin-Base-384 + 3-channel radiological fusion (fast ~1.2h, brand-new hierarchical ViT family!)
    (
        "e35_swinb384_rad3ch",
        [
            "--model", "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
            "--img_size", "384", "--rad3ch",
            "--epochs", "15", "--batch_size", "8", "--grad_accum", "2",
            "--lr", "8e-5", "--layer_decay", "0.80", "--head_lr_mult", "4.0",
            "--drop_rate", "0.25", "--drop_path", "0.20", "--soup_k", "3",
            "--tta", "orig,zoom5,zoom",
        ],
    ),
    # 2. StanfordAIMI XraySigLIP ViT-L/16 512px (304M Vision-Language CXR foundation model, CLEAN!)
    (
        "e34_xraysiglip512_clean",
        [
            "--model", "hfsiglip_StanfordAIMI--XraySigLIP__vit-l-16-siglip-384__webli",
            "--img_size", "512", "--freeze_layers", "10",
            "--epochs", "10", "--batch_size", "4", "--grad_accum", "4",
            "--lr", "4e-5", "--layer_decay", "0.85", "--head_lr_mult", "5.0",
            "--drop_rate", "0.25", "--soup_k", "3",
            "--tta", "orig,zoom5,zoom",
        ],
    ),
    # 3. Clean RAD-DINO 518px with LLRD + DropPath + Multi-Sample Dropout + Soup + TTA (ZERO poisoned PL!)
    (
        "e36_raddino518_clean_llrd",
        [
            "--model", "hfdinov2_microsoft--rad-dino",
            "--img_size", "518", "--freeze_layers", "2",
            "--epochs", "10", "--batch_size", "8", "--grad_accum", "2",
            "--lr", "4e-5", "--layer_decay", "0.80", "--head_lr_mult", "5.0",
            "--drop_rate", "0.25", "--drop_path", "0.15", "--soup_k", "3",
            "--tta", "orig,zoom5,zoom",
        ],
    ),
    # 4. ConvNeXtV2-Base 384 FCMAE + 3-channel radiological fusion (CLEAN!)
    (
        "e37_convnextv2b384_rad3ch",
        [
            "--model", "convnextv2_base.fcmae_ft_in22k_in1k_384",
            "--img_size", "384", "--rad3ch",
            "--epochs", "15", "--batch_size", "8", "--grad_accum", "2",
            "--lr", "1e-4", "--layer_decay", "0.80", "--head_lr_mult", "4.0",
            "--drop_rate", "0.25", "--drop_path", "0.20", "--soup_k", "3",
            "--tta", "orig,zoom5,zoom",
        ],
    ),
    # 5. Dedicated Bacteria-vs-Virus Binary Specialist on XraySigLIP-512!
    (
        "e39_bv_xraysiglip512",
        [
            "--model", "hfsiglip_StanfordAIMI--XraySigLIP__vit-l-16-siglip-384__webli",
            "--bv_only", "--img_size", "512", "--freeze_layers", "12",
            "--epochs", "8", "--batch_size", "4", "--grad_accum", "4",
            "--lr", "4e-5", "--layer_decay", "0.85", "--head_lr_mult", "5.0",
            "--drop_rate", "0.25", "--soup_k", "3",
            "--tta", "orig,zoom5,zoom",
        ],
    ),
]


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for tag, args in EXPERIMENTS:
        if only and tag not in only:
            continue
        print(f"\n================ STARTING {tag} ================", flush=True)
        cmd = [sys.executable, "tools/run_5fold_v2.py", tag, *args]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[WARN] {tag} exited with code {r.returncode}, continuing to next experiment...", flush=True)


if __name__ == "__main__":
    main()
