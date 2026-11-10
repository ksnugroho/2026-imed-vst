"""
model.py — iMED-VST
-------------------
Video Swin Transformer + Temporal Depth Enhancement (TD) untuk Video FER.

Backbone: torchvision swin3d_t / swin3d_s (Kinetics-400 pretrained)
  swin3d_t: 28M param, feat_dim=768
  swin3d_s: 49M param, feat_dim=768

TD module (~4.7M param) disisipkan SEBELUM avgpool:
  [B, D'=8, H'=7, W'=7, C=768]
  → spatial avg pool → [B, 8, 768]
  → TransformerEncoderLayer → [B, 8, 768]
  → temporal mean pool → [B, 768]
  → head

Paper: Liu et al., "Video Swin Transformer", CVPR 2022.
TD ref: "Video Swin Transformers in Pain Detection", 2024.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import (
    swin3d_t, Swin3D_T_Weights,
    swin3d_s, Swin3D_S_Weights,
)

_BACKBONE_REGISTRY = {
    "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
    "swin3d_s": (swin3d_s, Swin3D_S_Weights.DEFAULT),
}


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss untuk menangani class imbalance FER.
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

    Tidak pakai label_smoothing (focal sudah regularisasi implisit).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        # alpha = class weights [num_classes], di-register sebagai buffer agar ikut .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per sample dengan class weights
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# Temporal Depth Enhancement (TD) Module
# ---------------------------------------------------------------------------

class TemporalEnhancement(nn.Module):
    """
    Temporal aggregation module disisipkan setelah backbone norm.

    Input:  [B, D', H', W', C]   (output Swin3D setelah norm)
    Output: [B, C]

    Langkah:
      1. Spatial avg pool H'×W' → [B, D', C]
      2. Encoder (tergantung encoder_type):
           "transformer" → (opsional CLS/PE) + TransformerEncoder
           "gru"         → Bidirectional GRU, concat last hidden
           "none"        → tidak ada encoder (langsung pool)
      3. Pool: "mean" | "max" | "cls"

    Args:
      encoder_type : "transformer" | "gru" | "none"
      pool_type    : "mean" | "max" | "cls"
        - cls hanya valid dengan encoder_type="transformer"
        - gru encoder selalu output last hidden (pool_type diabaikan)
      num_layers   : jumlah layer (hanya untuk encoder_type="transformer")
      use_pos_enc  : sinusoidal PE (hanya untuk encoder_type="transformer")
      dim_ffn      : FFN dim (hanya untuk encoder_type="transformer")
    """

    def __init__(
        self,
        feat_dim: int = 768,
        nhead: int = 8,
        dim_ffn: int = 1536,
        dropout: float = 0.1,
        num_layers: int = 1,
        use_pos_enc: bool = False,
        pool_type: str = "mean",        # "mean" | "max" | "cls"
        encoder_type: str = "transformer",  # "transformer" | "gru" | "none"
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.pool_type    = pool_type

        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=nhead,
                dim_feedforward=dim_ffn, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.use_pos_enc = use_pos_enc
            if use_pos_enc:
                self.register_buffer("pos_enc", self._sinusoidal(feat_dim, max_len=32))
            if pool_type == "cls":
                self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
                nn.init.trunc_normal_(self.cls_token, std=0.02)

        elif encoder_type in ("gru", "gru_attn"):
            # Bidirectional GRU: hidden × 2 directions = feat_dim
            self.gru = nn.GRU(
                feat_dim, feat_dim // 2,
                batch_first=True, bidirectional=True,
                num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
            )
            if encoder_type == "gru_attn":
                # Learnable attention over all GRU outputs (vs last hidden state)
                self.gru_attn_proj = nn.Linear(feat_dim, 1)

        # encoder_type="none" → tidak ada parameter tambahan

        print(f"  [TD] enc={encoder_type} pool={pool_type} "
              f"layers={num_layers} ffn={dim_ffn} pos_enc={use_pos_enc}")

    @staticmethod
    def _sinusoidal(d_model: int, max_len: int = 32) -> torch.Tensor:
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        # Spatial avg pool → [B, D, C]
        x = x.reshape(B, D, H * W, C).mean(dim=2)

        if self.encoder_type == "transformer":
            if self.pool_type == "cls":
                cls = self.cls_token.expand(B, -1, -1)
                x   = torch.cat([cls, x], dim=1)   # [B, D+1, C]
            if self.use_pos_enc:
                x = x + self.pos_enc[:, :x.size(1), :]
            x = self.encoder(x)                    # [B, D(+1), C]
            if self.pool_type == "cls":
                return x[:, 0, :]                  # [B, C] — CLS token
            if self.pool_type == "max":
                return x.max(dim=1).values         # [B, C] — temporal max
            return x.mean(dim=1)                   # [B, C] — temporal mean

        if self.encoder_type in ("gru", "gru_attn"):
            out, h = self.gru(x)                               # out:[B,D,C] h:[2,B,C//2]
            if self.encoder_type == "gru_attn":
                w = F.softmax(self.gru_attn_proj(out), dim=1) # [B, D, 1]
                return (w * out).sum(dim=1)                    # [B, C]
            return torch.cat([h[0], h[1]], dim=-1)             # [B, C] — last hidden

        # encoder_type == "none"
        if self.pool_type == "max":
            return x.max(dim=1).values             # [B, C] — max pool
        return x.mean(dim=1)                       # [B, C] — mean pool


# ---------------------------------------------------------------------------
# VideoSwinFER
# ---------------------------------------------------------------------------

class VideoSwinFER(nn.Module):
    """
    Video Swin Transformer fine-tuned untuk Facial Expression Recognition.

    Struktur internal (dipecah agar TD bisa disisipkan):
      self.features  — PatchEmbed + SwinTransformer3D blocks  → [B, D', H', W', C]
      self.norm      — LayerNorm
      [self.td]      — TemporalEnhancement (jika use_td=True)
      [avgpool+flatten] — global pool (jika use_td=False)
      self.head      — LayerNorm → Dropout → Linear(feat_dim, num_classes)

    Input:  [B, 3, T, H, W]
    Output: [B, num_classes]
    """

    def __init__(
        self,
        backbone: str = "swin3d_t",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_head: float = 0.5,
        use_td: bool = False,
        td_config: dict | None = None,
    ):
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(f"backbone harus: {list(_BACKBONE_REGISTRY)}")

        build_fn, weights = _BACKBONE_REGISTRY[backbone]
        self._backbone_name = backbone

        model = build_fn(weights=weights if pretrained else None)
        if pretrained:
            print(f"[VideoSwinFER] pretrained Kinetics-400: {backbone}")

        self.feat_dim = model.head.in_features   # 768 untuk swin3d_t/s

        # Pisahkan komponen backbone (sesuai urutan forward torchvision Swin3D)
        self.patch_embed = model.patch_embed     # raw video → patch tokens
        self.pos_drop    = model.pos_drop        # positional dropout
        self.features    = model.features        # Swin3D blocks
        self.norm        = model.norm            # LayerNorm
        self.avgpool     = model.avgpool         # AdaptiveAvgPool3d(1)
        # flatten dilakukan via torch.flatten(x, 1) — bukan nn.Module

        self.use_td = use_td
        if use_td:
            td_cfg = td_config or {}
            self.td = TemporalEnhancement(
                feat_dim     = self.feat_dim,
                nhead        = td_cfg.get("nhead",        8),
                dim_ffn      = td_cfg.get("dim_ffn",      1536),
                dropout      = td_cfg.get("dropout",      0.1),
                num_layers   = td_cfg.get("num_layers",   1),
                use_pos_enc  = td_cfg.get("use_pos_enc",  False),
                pool_type    = td_cfg.get("pool_type",    "mean"),
                encoder_type = td_cfg.get("encoder_type", "transformer"),
            )
            td_params = sum(p.numel() for p in self.td.parameters()) / 1e6
            print(f"[VideoSwinFER] TD module aktif ({td_params:.1f}M param tambahan)")

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout_head),
            nn.Linear(self.feat_dim, num_classes),
        )
        self.num_classes = num_classes

        total = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[VideoSwinFER] feat_dim={self.feat_dim} | classes={num_classes} | "
              f"total={total:.1f}M params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        x = self.patch_embed(x)  # [B, D', H', W', C_0]  — patch embedding
        x = self.pos_drop(x)
        x = self.features(x)     # [B, D', H', W', C]    — Swin3D blocks
        x = self.norm(x)         # [B, D', H', W', C]

        if self.use_td:
            x = self.td(x)                       # [B, C]
        else:
            x = x.permute(0, 4, 1, 2, 3)        # [B, C, D', H', W']
            x = self.avgpool(x)                  # [B, C, 1, 1, 1]
            x = torch.flatten(x, 1)              # [B, C]

        return self.head(x)      # [B, num_classes]

    def freeze_backbone(self, freeze: bool = True):
        """Bekukan features + norm; biarkan TD + head tetap trainable."""
        for name, param in self.named_parameters():
            is_td   = self.use_td and name.startswith("td.")
            is_head = name.startswith("head.")
            if not (is_td or is_head):
                param.requires_grad = not freeze
        status = "FROZEN" if freeze else "UNFROZEN"
        print(f"[VideoSwinFER] backbone {status}")

    def replace_head(self, num_classes: int):
        """Ganti Linear terakhir di head (untuk two-stage step 4)."""
        self.head[-1] = nn.Linear(self.feat_dim, num_classes)
        self.num_classes = num_classes
        print(f"[VideoSwinFER] head diganti → {num_classes} kelas")

    def get_param_groups(self, lr: float, backbone_lr_scale: float = 0.1) -> list[dict]:
        """
        Param groups untuk layer-wise LR:
          - backbone (features + norm): lr × backbone_lr_scale
          - head:                       lr  (random init)
          - td (jika aktif):            lr  (random init)
        """
        backbone_params, head_params, td_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head."):
                head_params.append(param)
            elif self.use_td and name.startswith("td."):
                td_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr * backbone_lr_scale, "name": "backbone"})
        if head_params:
            groups.append({"params": head_params, "lr": lr, "name": "head"})
        if td_params:
            groups.append({"params": td_params, "lr": lr, "name": "td"})
        return groups


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> VideoSwinFER:
    return VideoSwinFER(
        backbone     = cfg["model"]["backbone"],
        num_classes  = cfg["dataset"]["num_classes"],
        pretrained   = cfg["model"].get("pretrained", True),
        dropout_head = cfg["model"].get("dropout_head", 0.5),
        use_td       = cfg["model"].get("use_td", False),
        td_config    = cfg["model"].get("td_config"),
    )


def build_loss(cfg: dict, class_weights: torch.Tensor | None) -> nn.Module:
    """Buat CE atau Focal loss sesuai config."""
    loss_type = cfg["train"].get("loss_type", "ce")

    if loss_type == "focal":
        return FocalLoss(
            gamma=cfg["train"].get("focal_gamma", 2.0),
            alpha=class_weights,
        )
    # default: cross entropy
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["train"].get("label_smoothing", 0.1),
    )
