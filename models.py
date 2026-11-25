"""Model construction (timm backbones) and a minimal EMA implementation."""
from __future__ import annotations

import copy
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows without Developer Mode

import timm
import torch
import torch.nn as nn

from dataset import NUM_CLASSES


def build_model(name: str, pretrained: bool = True, drop_rate: float = 0.2,
                drop_path_rate: float = 0.1, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Any timm model name works, e.g.
    convnext_tiny.fb_in22k_ft_in1k, efficientnet_b3.ra2_in1k, resnet50.a1_in1k,
    densenet121.tv_in1k, swin_base_patch4_window12_384.ms_in22k_ft_in1k,
    eva02_base_patch14_448.mim_in22k_ft_in22k_in1k, convnextv2_base.fcmae_ft_in22k_in1k_384.
    Also supports xrv_*, hf_* (legacy), hfdinov2_* (MSD+drop_path), hfsiglip_* (SigLIP vision tower).
    """
    if name.startswith("xrv_"):
        return XRVClassifier(name[len("xrv_"):], drop_rate, num_classes=num_classes)
    if name.startswith("hfsiglip_"):
        return HFSiglipClassifier(name[len("hfsiglip_"):].replace("--", "/"), drop_rate,
                                  num_classes=num_classes)
    if name.startswith("hfdinov2_"):
        return HFDinoV2Classifier(name[len("hfdinov2_"):].replace("--", "/"), drop_rate,
                                  drop_path_rate=drop_path_rate, num_classes=num_classes)
    if name.startswith("hf_"):  # e.g. hf_microsoft--rad-dino ("--" stands for "/")
        return HFDinoClassifier(name[len("hf_"):].replace("--", "/"), drop_rate, num_classes=num_classes)
    kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    try:
        return timm.create_model(name, drop_path_rate=drop_path_rate, **kwargs)
    except TypeError:  # architectures without stochastic depth
        return timm.create_model(name, **kwargs)


class MultiSampleDropoutHead(nn.Module):
    """LayerNorm + 5-branch multi-sample dropout linear head for better calibration and stability."""

    def __init__(self, in_dim: int, num_classes: int, drop_rate: float = 0.2, n_samples: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drops = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(n_samples)])
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if not self.training:
            return self.fc(x)
        return torch.stack([self.fc(d(x)) for d in self.drops], dim=0).mean(0)


class XRVClassifier(nn.Module):
    """Chest-X-ray-pretrained backbone from TorchXRayVision (pip install torchxrayvision).

    Model names: xrv_densenet121-res224-all, xrv_resnet50-res512-all, ...
    The shared data pipeline gives a 3-channel ImageNet-normalised tensor of a grayscale image;
    channel 0 is mapped back to [0, 1] and then to the [-1024, 1024] range these weights expect.
    The xrv wrappers resize every input to their training resolution, so we call the raw
    backbone directly to keep our own resolution. The weights are always loaded (they are
    cached after the first download) and then overwritten by our checkpoint at inference.
    """

    def __init__(self, weights: str, drop_rate: float = 0.2, num_classes: int = NUM_CLASSES):
        super().__init__()
        import torchxrayvision as xrv
        if weights.startswith("densenet"):
            self.backbone, dim, self.pool_relu = xrv.models.DenseNet(weights=weights).features, 1024, True
        elif weights.startswith("resnet"):
            net = xrv.models.ResNet(weights=weights).model
            net.fc = nn.Identity()
            self.backbone, dim, self.pool_relu = net, 2048, False
        else:
            raise ValueError(f"unknown torchxrayvision weights {weights!r}")
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(dim, num_classes))

    def forward(self, x):
        from dataset import IMAGENET_MEAN, IMAGENET_STD
        x = x[:, :1] * IMAGENET_STD[0] + IMAGENET_MEAN[0]  # back to [0, 1]
        f = self.backbone((x * 2 - 1) * 1024)
        if self.pool_relu:  # densenet features end before the final ReLU and pooling
            f = torch.nn.functional.adaptive_avg_pool2d(torch.relu(f), 1).flatten(1)
        return self.head(f)


class HFDinoClassifier(nn.Module):
    """DINOv2-style ViT from the Hugging Face hub (pip install transformers), e.g. RAD-DINO,
    a ViT-B/14 pretrained on chest X-rays. Input sizes must be multiples of 14 (518 native).

    The shared pipeline gives an ImageNet-normalised tensor; it is re-normalised with the
    model's own statistics. The head sees the CLS token and the mean patch token.
    """

    def __init__(self, repo: str, drop_rate: float = 0.2, num_classes: int = NUM_CLASSES):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel
        from dataset import IMAGENET_MEAN, IMAGENET_STD
        self.backbone = AutoModel.from_pretrained(repo)
        if os.environ.get("LAB3_GRAD_CKPT", "1") != "0":  # set LAB3_GRAD_CKPT=0 on GPUs with >= 15 GB
            self.backbone.gradient_checkpointing_enable()  # fits 518 px in 8 GB
        proc = AutoImageProcessor.from_pretrained(repo)
        im, isd = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        m, sd = torch.tensor(proc.image_mean).view(1, 3, 1, 1), torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.register_buffer("scale", isd / sd, persistent=False)
        self.register_buffer("shift", (im - m) / sd, persistent=False)
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(self.backbone.config.hidden_size * 2, num_classes))

    def forward(self, x):
        h = self.backbone(pixel_values=x * self.scale + self.shift).last_hidden_state
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1))


class HFDinoV2Classifier(nn.Module):
    """Upgraded DINOv2/RAD-DINO classifier with DropPath and Multi-Sample Dropout head."""

    def __init__(self, repo: str, drop_rate: float = 0.25, drop_path_rate: float = 0.15,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModel
        from dataset import IMAGENET_MEAN, IMAGENET_STD
        cfg = AutoConfig.from_pretrained(repo)
        if hasattr(cfg, "drop_path_rate"):
            cfg.drop_path_rate = drop_path_rate
        self.backbone = AutoModel.from_pretrained(repo, config=cfg)
        if os.environ.get("LAB3_GRAD_CKPT", "1") != "0":
            self.backbone.gradient_checkpointing_enable()
        proc = AutoImageProcessor.from_pretrained(repo)
        im, isd = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        m, sd = torch.tensor(proc.image_mean).view(1, 3, 1, 1), torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.register_buffer("scale", isd / sd, persistent=False)
        self.register_buffer("shift", (im - m) / sd, persistent=False)
        self.head = MultiSampleDropoutHead(self.backbone.config.hidden_size * 2, num_classes, drop_rate)

    def forward(self, x):
        h = self.backbone(pixel_values=x * self.scale + self.shift).last_hidden_state
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1))


class HFSiglipClassifier(nn.Module):
    """Vision-only classifier wrapping a Hugging Face SigLIP vision tower (e.g. XraySigLIP ViT-L/16)."""

    def __init__(self, repo: str, drop_rate: float = 0.25, num_classes: int = NUM_CLASSES):
        super().__init__()
        from transformers import AutoImageProcessor, SiglipVisionModel
        from dataset import IMAGENET_MEAN, IMAGENET_STD
        self.backbone = SiglipVisionModel.from_pretrained(repo)
        if os.environ.get("LAB3_GRAD_CKPT", "1") != "0":
            self.backbone.gradient_checkpointing_enable()
        proc = AutoImageProcessor.from_pretrained(repo)
        im, isd = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        m, sd = torch.tensor(proc.image_mean).view(1, 3, 1, 1), torch.tensor(proc.image_std).view(1, 3, 1, 1)
        self.register_buffer("scale", isd / sd, persistent=False)
        self.register_buffer("shift", (im - m) / sd, persistent=False)
        dim = self.backbone.config.hidden_size
        self.head = MultiSampleDropoutHead(dim * 2, num_classes, drop_rate)

    def forward(self, x):
        out = self.backbone(pixel_values=x * self.scale + self.shift)
        feat = torch.cat([out.pooler_output, out.last_hidden_state.mean(1)], dim=1)
        return self.head(feat)


def build_llrd_param_groups(model: nn.Module, lr: float, weight_decay: float,
                            layer_decay: float = 1.0, head_lr_mult: float = 1.0):
    """Build AdamW parameter groups with optional Layer-wise LR Decay (LLRD) and no decay on norms/biases."""
    import re
    if layer_decay >= 0.999 and head_lr_mult <= 1.001:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]

    # Discover maximum block depth across common HF & timm naming conventions
    layer_pat = re.compile(r"(?:encoder\.layer|encoder\.layers|blocks|layers|stages)\.(\d+)\.")
    max_layer = 0
    for n, _ in model.named_parameters():
        m = layer_pat.search(n)
        if m:
            max_layer = max(max_layer, int(m.group(1)) + 1)
    num_layers = max(max_layer, 1)

    groups: dict[tuple[float, float], list] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower()
        wd = 0.0 if is_no_decay else weight_decay

        if n.startswith(("head.", "fc.", "classifier.")) or ".head." in n:
            scale = head_lr_mult
        elif any(k in n for k in ("embeddings", "patch_embed", "stem", "conv_stem", "pos_embed", "cls_token")):
            scale = layer_decay ** (num_layers + 1)
        else:
            m = layer_pat.search(n)
            if m:
                lid = int(m.group(1)) + 1
                scale = layer_decay ** (num_layers + 1 - lid)
            else:
                # final norm / pooler / neck
                scale = 1.0

        key = (round(lr * scale, 10), wd)
        groups.setdefault(key, []).append(p)

    return [{"params": ps, "lr": glr, "weight_decay": gwd} for (glr, gwd), ps in groups.items()]


class ModelEMA:
    """Exponential moving average of model weights (with warm-up on the decay)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src, alpha=1 - d)
            else:
                v.copy_(src)
