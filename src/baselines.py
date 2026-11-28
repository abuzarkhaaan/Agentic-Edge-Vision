from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def mobilenetv3_small(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = tvm.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
    model.classifier[-1] = nn.Linear(1024, num_classes)
    return model


def efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = tvm.efficientnet_b0(weights="DEFAULT" if pretrained else None)
    model.classifier[-1] = nn.Linear(1280, num_classes)
    return model


def deit_tiny_with_adapters(num_classes: int = 2, r_adapt: int = 4) -> nn.Module:
    import timm
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True,
                              num_classes=num_classes)
    for name, p in model.named_parameters():
        if not any(x in name for x in ["head", "norm", "pre_logits"]):
            p.requires_grad_(False)
    d = model.embed_dim
    model.adapter_A = nn.Parameter(torch.randn(d, r_adapt) * 0.01)
    model.adapter_B = nn.Parameter(torch.zeros(r_adapt, d))
    return model


class EarlyExitDeiT(nn.Module):
    def __init__(self, num_classes: int = 2, exit_threshold: float = 0.25):
        super().__init__()
        import timm
        backbone          = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        self.patch_embed  = backbone.patch_embed
        self.cls_token    = backbone.cls_token
        self.pos_embed    = backbone.pos_embed
        self.blocks       = backbone.blocks
        self.norm         = backbone.norm
        d                 = backbone.embed_dim
        self.exit_head_4  = nn.Linear(d, num_classes)
        self.exit_head_8  = nn.Linear(d, num_classes)
        self.final_head   = nn.Linear(d, num_classes)
        self.threshold    = exit_threshold

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)
        return -(p * (p + 1e-9).log()).sum(-1) / math.log(logits.shape[-1])

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            feat = self.norm(x)[:, 0]
            if i == 3:
                logits = self.exit_head_4(feat)
                if self._entropy(logits).max() < self.threshold:
                    return logits, "exit_layer4"
            if i == 7:
                logits = self.exit_head_8(feat)
                if self._entropy(logits).max() < self.threshold:
                    return logits, "exit_layer8"

        return self.final_head(self.norm(x)[:, 0]), "final"


class EdgeCloudSplitDeiT(nn.Module):
    SPLIT_LAYER = 6

    def __init__(self, num_classes: int = 2):
        super().__init__()
        import timm
        backbone         = timm.create_model("deit_tiny_patch16_224", pretrained=True)
        self.patch_embed = backbone.patch_embed
        self.cls_token   = backbone.cls_token
        self.pos_embed   = backbone.pos_embed
        self.edge_blocks = backbone.blocks[:self.SPLIT_LAYER]
        self.cloud_blocks = backbone.blocks[self.SPLIT_LAYER:]
        self.norm        = backbone.norm
        self.head        = nn.Linear(backbone.embed_dim, num_classes)

    def edge_forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.edge_blocks:
            x = blk(x)
        return x

    def cloud_forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.cloud_blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cloud_forward(self.edge_forward(x))


class ConfidenceOffloadingDeiT(nn.Module):
    def __init__(self, num_classes: int = 2, offload_threshold: float = 0.50):
        super().__init__()
        import timm
        self.backbone  = timm.create_model("deit_tiny_patch16_224",
                                           pretrained=True, num_classes=num_classes)
        self.threshold = offload_threshold

    def forward(self, x: torch.Tensor):
        logits       = self.backbone(x)
        conf         = F.softmax(logits, dim=-1).max(dim=-1).values
        offload_mask = conf < self.threshold
        return logits, offload_mask
