"""
train.py — iMED-VST
--------------------
Training loop untuk satu run (single train/val split).
Dipakai untuk debug / single-fold test.
LOSO lengkap → pakai run_loso.py.

Cara pakai:
  python train.py --config config.yaml --debug
  python train.py --config config.yaml --test_subject 1 --synthetic
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from model   import build_model, build_loss
from dataset import (
    IMEDVideoDataset, SyntheticVideoDataset,
    VideoTransform, build_weighted_sampler,
)
from metrics import FERMetrics


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="config.yaml")
    p.add_argument("--test_subject", type=int, default=None)
    p.add_argument("--backbone",        default=None)
    p.add_argument("--loss_type",       default=None, choices=["ce", "focal"])
    p.add_argument("--use_td",          action="store_true")
    p.add_argument("--num_frames",      type=int, default=None)
    p.add_argument("--frame_interval",  type=int, default=None)
    p.add_argument("--epochs",          type=int, default=None)
    p.add_argument("--debug",           action="store_true")
    p.add_argument("--synthetic",       action="store_true")
    p.add_argument("--resume",          default=None)
    return p.parse_args()


def load_config(args) -> dict:
    cfg_path = BASE_DIR / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if args.test_subject:   cfg["dataset"]["test_subject"]   = args.test_subject
    if args.backbone:       cfg["model"]["backbone"]         = args.backbone
    if args.loss_type:      cfg["train"]["loss_type"]        = args.loss_type
    if args.use_td:         cfg["model"]["use_td"]           = True
    if args.num_frames:     cfg["dataset"]["num_frames"]     = args.num_frames
    if args.frame_interval: cfg["dataset"]["frame_interval"] = args.frame_interval
    if args.epochs:         cfg["train"]["epochs"]           = args.epochs
    return cfg


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict, synthetic: bool = False):
    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg["augment"]

    train_tf = VideoTransform(
        img_size         = ds_cfg["img_size"],
        is_train         = True,
        color_jitter     = aug_cfg.get("color_jitter", 0.4),
        random_grayscale = aug_cfg.get("random_grayscale", 0.1),
        flip_prob        = aug_cfg.get("flip_horizontal", 0.5),
    )
    val_tf = VideoTransform(img_size=ds_cfg["img_size"], is_train=False)

    if synthetic:
        train_ds = SyntheticVideoDataset(
            num_samples=160, num_classes=ds_cfg["num_classes"],
            num_frames=ds_cfg["num_frames"], img_size=ds_cfg["img_size"],
            class_names=ds_cfg.get("class_names"),
        )
        val_ds = SyntheticVideoDataset(
            num_samples=40, num_classes=ds_cfg["num_classes"],
            num_frames=ds_cfg["num_frames"], img_size=ds_cfg["img_size"],
            class_names=ds_cfg.get("class_names"),
        )
        return train_ds, val_ds

    kw = dict(
        root           = ds_cfg["root"],
        split_mode     = ds_cfg.get("split_mode", "loso"),
        test_subject   = ds_cfg.get("test_subject", 1),
        emotion_set    = ds_cfg.get("emotion_set", "basic"),
        num_frames     = ds_cfg["num_frames"],
        frame_interval = ds_cfg["frame_interval"],
        train_ratio    = ds_cfg.get("train_ratio", 0.8),
        seed           = cfg["train"]["seed"],
    )
    train_ds = IMEDVideoDataset(split="train", transform=train_tf, **kw)
    val_ds   = IMEDVideoDataset(split="val",   transform=val_tf,   **kw)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg["train"]
    param_groups = model.get_param_groups(
        lr=float(tr["lr"]),
        backbone_lr_scale=0.1,
    )
    return torch.optim.AdamW(param_groups, weight_decay=tr["weight_decay"])


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    tr = cfg["train"]
    total_steps  = tr["epochs"] * steps_per_epoch
    warmup_steps = tr["warmup_epochs"] * steps_per_epoch

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=tr["min_lr"],
    )
    if warmup_steps == 0:
        # No warmup: stage 2 two-stage or warmup_epochs=0 config
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# Train / Eval loops (dipakai juga oleh run_loso.py)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, loss_fn, device, grad_clip,
    amp_dtype=torch.float16, debug=False,
) -> dict:
    model.train()
    total_loss = correct = total = 0

    for i, (clips, labels) in enumerate(loader):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(clips)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if debug and i >= 1:
            break

    return {"loss": total_loss / (i + 1), "top1": correct / total * 100}


@torch.no_grad()
def evaluate(
    model, loader, loss_fn, metrics: FERMetrics, device,
    amp_dtype=torch.float16, debug=False,
) -> dict:
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for i, (clips, labels) in enumerate(loader):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(clips)
            loss   = loss_fn(logits, labels)

        total_loss += loss.item()
        metrics.update(logits.float(), labels)

        if debug and i >= 1:
            break

    m = metrics.compute()
    return {"loss": total_loss / (i + 1), **m}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args)
    set_seed(cfg["train"]["seed"])

    device    = torch.device(f"cuda:{cfg['hardware']['device']}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    amp_label = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
    print(f"\n[✓] Device: {device} | AMP: {amp_label}")

    train_ds, val_ds = build_datasets(cfg, synthetic=args.synthetic)

    sampler = build_weighted_sampler(train_ds) if cfg["train"]["use_weighted_loss"] else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg["dataset"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval"]["batch_size"],
        shuffle=False, num_workers=cfg["dataset"]["num_workers"], pin_memory=True,
    )

    model = build_model(cfg).to(device)

    class_weights = train_ds.class_weights().to(device) if cfg["train"]["use_weighted_loss"] else None
    loss_fn = build_loss(cfg, class_weights)
    # Focal Loss sudah register buffer, tapi pastikan di device yang benar
    if hasattr(loss_fn, "alpha") and loss_fn.alpha is not None:
        loss_fn = loss_fn.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = torch.cuda.amp.GradScaler(
        enabled=cfg["hardware"]["mixed_precision"] and device.type == "cuda"
    )

    start_epoch = 0
    best_uar    = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_uar    = ckpt.get("best_uar", 0.0)
        print(f"[✓] Resume dari {args.resume} (epoch {start_epoch})")

    log_dir = PROJECT_ROOT / cfg["logging"]["log_dir"] / "train_debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics = FERMetrics(cfg["dataset"]["class_names"])

    bb = cfg["model"]["backbone"]
    td = "TD" if cfg["model"].get("use_td") else "noTD"
    lt = cfg["train"].get("loss_type", "ce").upper()
    cls = cfg["dataset"]["emotion_set"]
    print(f"\n{'='*65}")
    print(f"  {bb} | {td} | {lt} | {cls} | LOSO subj={cfg['dataset'].get('test_subject',1)}")
    print(f"{'='*65}\n")

    eval_every = cfg["eval"]["eval_every_n_epochs"]

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, device, cfg["train"]["clip_grad"],
            amp_dtype=amp_dtype, debug=args.debug,
        )

        do_eval = (epoch + 1) % eval_every == 0 or epoch == cfg["train"]["epochs"] - 1
        if do_eval:
            val_m  = evaluate(model, val_loader, loss_fn, metrics, device,
                              amp_dtype=amp_dtype, debug=args.debug)
            uar    = val_m["uar"]
            is_best = uar > best_uar
            if is_best:
                best_uar = uar
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "best_uar": best_uar, "cfg": cfg},
                           log_dir / "best.pt")

            elapsed = time.time() - t0
            mk = "★" if is_best else " "
            print(
                f"[{epoch+1:3d}/{cfg['train']['epochs']}]{mk} "
                f"loss={train_m['loss']:.4f} top1={train_m['top1']:.1f}% | "
                f"WAR={val_m['war']:.2f}% UAR={uar:.2f}% "
                f"MacroF1={val_m['macro_f1']:.2f}% WF1={val_m['weighted_f1']:.2f}% | "
                f"{elapsed:.0f}s [best UAR={best_uar:.2f}%]"
            )
            if is_best:
                print(metrics.summary())
        else:
            print(f"[{epoch+1:3d}/{cfg['train']['epochs']}]  "
                  f"loss={train_m['loss']:.4f} top1={train_m['top1']:.1f}% | "
                  f"{time.time()-t0:.0f}s")

        if args.debug:
            print("[DEBUG] Berhenti setelah 1 epoch.")
            break

    print(f"\n[✓] Selesai. Best UAR: {best_uar:.2f}%")


if __name__ == "__main__":
    main()
