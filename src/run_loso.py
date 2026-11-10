"""
run_loso.py — iMED-VST
------------------------
Leave-One-Subject-Out (LOSO) cross-validation.
Mendukung Step 1/2/3/4/5 dari EXPERIMENT_DESIGN.md.

Step 1a: python run_loso.py --backbone swin3d_t --run_name loso_step1_swin3d_t_7cls
Step 1b: python run_loso.py --backbone swin3d_s --run_name loso_step1_swin3d_s_7cls
Step 2b: python run_loso.py --loss_type focal --run_name loso_step2_focal_7cls
Step 3:  python run_loso.py --use_td --run_name loso_step3_td_7cls
Step 4a: python run_loso.py --emotion_set all --use_td \
             --init_checkpoint_dir outputs/runs/step3_td_gru_7cls \
             --run_name step4a_twostage_19cls
Step 4b: python run_loso.py --emotion_set all --use_td \
             --run_name loso_step4b_direct_19cls
Step 5:  python run_step5_td_ablation.py   (atau gunakan --td_variant secara manual)
         python run_loso.py --use_td --td_variant td_pe --run_name step5_td_pe

TD variants (Step 5):
  td_base  1 layer, no PE, ffn=1536, mean  (= Step 3, baseline)
  td_pe    + sinusoidal positional encoding
  td_2l    2 layers
  td_4x    ffn=3072 (4×)
  td_cls   CLS token pool
  td_full  2L + PE + ffn=3072 + CLS

Single fold (cek cepat):
  python run_loso.py --fold 1 --run_name debug

Output:
  outputs/results/<run_name>.csv
  outputs/results/<run_name>_summary.txt
  outputs/runs/<run_name>/fold_01_best.pt ... fold_15_best.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import wandb

BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

WANDB_ENTITY  = "bdsrc-team"
WANDB_PROJECT = "2026-imed-vst-v2"
sys.path.insert(0, str(BASE_DIR))

# TD variants untuk Step 5 ablation
TD_VARIANTS: dict[str, dict] = {
    # Transformer encoder variants
    "td_base":    {"encoder_type": "transformer", "num_layers": 1, "use_pos_enc": False, "dim_ffn": 1536, "pool_type": "mean"},
    "td_pe":      {"encoder_type": "transformer", "num_layers": 1, "use_pos_enc": True,  "dim_ffn": 1536, "pool_type": "mean"},
    "td_2l":      {"encoder_type": "transformer", "num_layers": 2, "use_pos_enc": False, "dim_ffn": 1536, "pool_type": "mean"},
    "td_4x":      {"encoder_type": "transformer", "num_layers": 1, "use_pos_enc": False, "dim_ffn": 3072, "pool_type": "mean"},
    "td_cls":     {"encoder_type": "transformer", "num_layers": 1, "use_pos_enc": False, "dim_ffn": 1536, "pool_type": "cls"},
    "td_full":    {"encoder_type": "transformer", "num_layers": 2, "use_pos_enc": True,  "dim_ffn": 3072, "pool_type": "cls"},
    "td_pe_cls":  {"encoder_type": "transformer", "num_layers": 1, "use_pos_enc": True,  "dim_ffn": 1536, "pool_type": "cls"},
    # Alternative encoder variants (for ablation of encoder_type choice)
    "td_maxpool": {"encoder_type": "none",        "num_layers": 1, "use_pos_enc": False, "dim_ffn": 1536, "pool_type": "max"},
    "td_gru":     {"encoder_type": "gru",         "num_layers": 1, "use_pos_enc": False, "dim_ffn": 1536, "pool_type": "mean"},
    "td_gru_attn":{"encoder_type": "gru_attn",    "num_layers": 1, "use_pos_enc": False, "dim_ffn": 1536, "pool_type": "mean"},
}

from model   import VideoSwinFER, build_loss
from dataset import IMEDVideoDataset, VideoTransform, build_weighted_sampler
from metrics import FERMetrics
from train   import (
    set_seed, build_optimizer, build_scheduler,
    train_one_epoch, evaluate,
)


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def run_fold(
    cfg: dict,
    fold: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    fold_ckpt_path: Path,
    init_checkpoint_dir: Path | None = None,
    run_name: str = "loso",
    freeze_first_epochs: int = 0,
    permute_train_labels: bool = False,
) -> dict | None:
    """
    Latih dan evaluasi satu LOSO fold.

    init_checkpoint_dir: jika diisi (step 4a), load fold_{fold:02d}_best.pt
                         dari direktori tersebut untuk two-stage training.
    permute_train_labels: jika True, acak label training (untuk Step 7 identity leakage test).
    """
    set_seed(cfg["train"]["seed"] + fold)

    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg["augment"]
    tr_cfg  = cfg["train"]

    # W&B init — satu run per fold
    td_config    = cfg["model"].get("td_config")
    td_variant   = cfg["model"].get("td_variant", "td_base" if cfg["model"].get("use_td") else "none")
    wandb.init(
        entity  = WANDB_ENTITY,
        project = WANDB_PROJECT,
        name    = f"{run_name}_fold{fold:02d}",
        group   = run_name,
        config  = {
            "backbone":       cfg["model"]["backbone"],
            "use_td":         cfg["model"].get("use_td", False),
            "td_variant":     td_variant,
            "td_encoder_type": (td_config or {}).get("encoder_type", "transformer"),
            "td_num_layers":   (td_config or {}).get("num_layers",  1),
            "td_use_pos_enc":  (td_config or {}).get("use_pos_enc", False),
            "td_dim_ffn":      (td_config or {}).get("dim_ffn",     1536),
            "td_pool_type":    (td_config or {}).get("pool_type",   "mean"),
            "loss_type":      tr_cfg.get("loss_type", "ce"),
            "emotion_set":    ds_cfg["emotion_set"],
            "num_classes":    ds_cfg["num_classes"],
            "num_frames":     ds_cfg["num_frames"],
            "frame_interval": ds_cfg["frame_interval"],
            "window_fr":      ds_cfg["num_frames"] * ds_cfg["frame_interval"],
            "fold":           fold,
            "two_stage":      init_checkpoint_dir is not None,
            "freeze_first_epochs": freeze_first_epochs,
            "epochs":         tr_cfg["epochs"],
            "lr":             tr_cfg["lr"],
            "batch_size":     tr_cfg["batch_size"],
            "eval_protocol":  "LOSO",
        },
        reinit = True,
    )

    train_tf = VideoTransform(
        img_size=ds_cfg["img_size"], is_train=True,
        color_jitter=aug_cfg.get("color_jitter", 0.4),
        random_grayscale=aug_cfg.get("random_grayscale", 0.1),
        flip_prob=aug_cfg.get("flip_horizontal", 0.5),
    )
    val_tf = VideoTransform(img_size=ds_cfg["img_size"], is_train=False)

    kw = dict(
        root           = ds_cfg["root"],
        split_mode     = "loso",
        test_subject   = fold,
        emotion_set    = ds_cfg.get("emotion_set", "basic"),
        num_frames     = ds_cfg["num_frames"],
        frame_interval = ds_cfg["frame_interval"],
    )
    # train: 13 subjek | val: 1 subjek (checkpoint selection) | test: 1 subjek (pelaporan)
    # val dan test selalu berbeda subjek — tidak ada leakage.
    train_ds = IMEDVideoDataset(split="train", transform=train_tf, **kw)
    val_ds   = IMEDVideoDataset(split="val",   transform=val_tf,   **kw)
    test_ds  = IMEDVideoDataset(split="test",  transform=val_tf,   **kw)

    if len(test_ds) == 0:
        print(f"  [SKIP] Fold {fold}: subject {fold} tidak ada di dataset")
        return None

    # Step 7: Acak label training — breaks emotion-appearance correlation
    if permute_train_labels:
        rng = np.random.default_rng(cfg["train"]["seed"] + fold + 9999)
        perm_labels = [s["label"] for s in train_ds.samples]
        rng.shuffle(perm_labels)
        for s, lbl in zip(train_ds.samples, perm_labels):
            s["label"] = lbl
        print(f"  [Step7] Fold {fold}: training labels PERMUTED")

    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=tr_cfg["batch_size"],
        sampler=sampler, shuffle=False,
        num_workers=ds_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval"]["batch_size"],
        shuffle=False, num_workers=ds_cfg["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["eval"]["batch_size"],
        shuffle=False, num_workers=ds_cfg["num_workers"], pin_memory=True,
    )

    # Step 4: dua-stage flag ditentukan sebelum build model
    is_two_stage = init_checkpoint_dir is not None

    # --- Model ---
    # Untuk two-stage: build dulu dengan 7 kelas (sesuai checkpoint step 3),
    # baru load weights, lalu replace head ke target num_classes.
    # Jika langsung build 19 kelas lalu load checkpoint 7 kelas → size mismatch.
    init_num_classes = 7 if is_two_stage else ds_cfg["num_classes"]
    model = VideoSwinFER(
        backbone     = cfg["model"]["backbone"],
        num_classes  = init_num_classes,
        pretrained   = cfg["model"].get("pretrained", True),
        dropout_head = cfg["model"].get("dropout_head", 0.5),
        use_td       = cfg["model"].get("use_td", False),
        td_config    = cfg["model"].get("td_config"),
    ).to(device)

    if is_two_stage:
        ckpt_path = Path(init_checkpoint_dir) / f"fold_{fold:02d}_best.pt"
        if not ckpt_path.exists():
            print(f"  [ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location=device)
        # Pastikan backbone/use_td/td_config sesuai checkpoint
        ckpt_cfg = ckpt.get("cfg", {})
        # Gunakan variabel lokal — jangan mutasi cfg karena dipakai bersama
        # oleh semua fold dalam satu seed loop
        eff_backbone  = ckpt_cfg.get("model", {}).get("backbone",  cfg["model"]["backbone"])
        eff_use_td    = ckpt_cfg.get("model", {}).get("use_td",    cfg["model"].get("use_td", False))
        eff_td_config = ckpt_cfg.get("model", {}).get("td_config", cfg["model"].get("td_config"))
        if eff_backbone != cfg["model"]["backbone"]:
            print(f"  [Step4] backbone dari checkpoint: {eff_backbone} (beda dari config)")
        if eff_use_td != cfg["model"].get("use_td", False):
            print(f"  [Step4] use_td dari checkpoint: {eff_use_td} (beda dari config)")
        # Rebuild model dengan config dari checkpoint (tanpa mutasi cfg)
        model = VideoSwinFER(
            backbone     = eff_backbone,
            num_classes  = init_num_classes,
            pretrained   = False,
            dropout_head = cfg["model"].get("dropout_head", 0.5),
            use_td       = eff_use_td,
            td_config    = eff_td_config,
        ).to(device)
        model.load_state_dict(ckpt["model"])   # strict=True, 7-class cocok
        print(f"  [Step4] Loaded step3 checkpoint: {ckpt_path}")
        model.replace_head(ds_cfg["num_classes"])  # 7 → 19
        model = model.to(device)

    # --- Loss ---
    class_weights = train_ds.class_weights().to(device) if tr_cfg["use_weighted_loss"] else None
    loss_fn = build_loss(cfg, class_weights)
    if hasattr(loss_fn, "alpha") and loss_fn.alpha is not None:
        loss_fn = loss_fn.to(device)

    # --- Freeze setup (two-stage atau freeze_first_epochs) ---
    freeze_epochs = (
        tr_cfg.get("freeze_backbone_epochs", 0) if is_two_stage else freeze_first_epochs
    )

    # Stage 1: backbone frozen → hanya TD + head trainable
    if freeze_epochs > 0:
        model.freeze_backbone(True)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = torch.cuda.amp.GradScaler(
        enabled=cfg["hardware"]["mixed_precision"] and device.type == "cuda"
    )
    metrics   = FERMetrics(ds_cfg["class_names"])

    best_uar        = 0.0
    best_war        = 0.0
    best_macro_p    = 0.0
    best_macro_f1   = 0.0
    best_weighted_p = 0.0
    best_weighted_r = 0.0
    best_weighted_f1= 0.0
    best_per_class  = {}
    best_cm         = []
    best_epoch      = 0
    eval_every      = cfg["eval"]["eval_every_n_epochs"]
    t_start         = time.time()

    for epoch in range(tr_cfg["epochs"]):
        # Two-stage: unfreeze backbone setelah freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.freeze_backbone(False)
            # Rebuild optimizer dengan semua param (backbone LR = 10% head LR)
            optimizer = build_optimizer(model, cfg)
            # Rebuild scheduler sisa epoch
            remaining_epochs = tr_cfg["epochs"] - epoch
            scheduler = build_scheduler(
                optimizer,
                {**cfg, "train": {**tr_cfg, "epochs": remaining_epochs, "warmup_epochs": 0}},
                len(train_loader),
            )
            scaler = torch.cuda.amp.GradScaler(
                enabled=cfg["hardware"]["mixed_precision"] and device.type == "cuda"
            )
            print(f"  [Stage2 start] Epoch {epoch+1}: backbone unfrozen, fine-tune end-to-end")

        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, device, tr_cfg["clip_grad"], amp_dtype=amp_dtype,
        )

        log = {"epoch": epoch + 1, "train/loss": train_m["loss"], "train/top1": train_m["top1"]}

        do_eval = (epoch + 1) % eval_every == 0 or epoch == tr_cfg["epochs"] - 1
        if do_eval:
            val_m = evaluate(model, val_loader, loss_fn, metrics, device, amp_dtype=amp_dtype)
            log.update({
                "val/war":         val_m["war"],
                "val/uar":         val_m["uar"],
                "val/macro_p":     val_m["macro_p"],
                "val/macro_f1":    val_m["macro_f1"],
                "val/weighted_p":  val_m["weighted_p"],
                "val/weighted_r":  val_m["weighted_r"],
                "val/weighted_f1": val_m["weighted_f1"],
                **{f"val/recall/{c}": v for c, v in val_m["per_class_recall"].items()},
            })
            is_new_best = val_m["uar"] > best_uar
            if is_new_best:
                best_uar         = val_m["uar"]
                best_war         = val_m["war"]
                best_macro_p     = val_m["macro_p"]
                best_macro_f1    = val_m["macro_f1"]
                best_weighted_p  = val_m["weighted_p"]
                best_weighted_r  = val_m["weighted_r"]
                best_weighted_f1 = val_m["weighted_f1"]
                best_epoch       = epoch + 1
                best_per_class   = val_m.get("per_class_recall", {})
                best_cm          = val_m.get("confusion_matrix", [])
                torch.save({
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "best_uar": best_uar,
                    "fold":     fold,
                    "cfg":      cfg,
                }, fold_ckpt_path)

            if do_eval:
                elapsed_s = int(time.time() - t_start)
                mk = "★" if is_new_best else " "
                print(
                    f"  [Fold{fold:02d}|ep{epoch+1:3d}/{tr_cfg['epochs']}]{mk} "
                    f"loss={train_m['loss']:.3f} "
                    f"WAR={val_m['war']:.1f}% UAR={val_m['uar']:.1f}% "
                    f"MacroF1={val_m['macro_f1']:.1f}% WF1={val_m['weighted_f1']:.1f}% "
                    f"[best={best_uar:.1f}%] {elapsed_s}s"
                )

        wandb.log(log)

    elapsed = round((time.time() - t_start) / 60, 1)

    # --- Evaluasi final pada test subject (bebas leakage) ---
    # Load checkpoint terbaik (dipilih berdasarkan val subject), lalu eval pada test subject.
    best_ckpt = torch.load(fold_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_m = evaluate(model, test_loader, loss_fn, metrics, device, amp_dtype=amp_dtype)

    print(
        f"  Fold {fold:2d} | [val] UAR={best_uar:.2f}%  "
        f"[test] WAR={test_m['war']:.2f}% UAR={test_m['uar']:.2f}% "
        f"MacroF1={test_m['macro_f1']:.2f}% WF1={test_m['weighted_f1']:.2f}% "
        f"ep={best_epoch} {elapsed}mnt"
    )

    # W&B summary — test metrics (bukan val)
    wandb.run.summary["best/val_uar"]      = best_uar
    wandb.run.summary["best/war"]          = test_m["war"]
    wandb.run.summary["best/uar"]          = test_m["uar"]
    wandb.run.summary["best/macro_p"]      = test_m["macro_p"]
    wandb.run.summary["best/macro_f1"]     = test_m["macro_f1"]
    wandb.run.summary["best/weighted_p"]   = test_m["weighted_p"]
    wandb.run.summary["best/weighted_r"]   = test_m["weighted_r"]
    wandb.run.summary["best/weighted_f1"]  = test_m["weighted_f1"]
    wandb.run.summary["best/epoch"]        = best_epoch
    for cls, v in test_m.get("per_class_recall", {}).items():
        wandb.run.summary[f"recall/{cls}"] = v
    # Confusion matrix (test subject)
    test_cm = test_m.get("confusion_matrix", [])
    if test_cm:
        class_names = ds_cfg["class_names"]
        cm_table = wandb.Table(columns=["true\\pred"] + class_names)
        for i, row in enumerate(test_cm):
            cm_table.add_data(class_names[i], *row)
        wandb.log({"best/confusion_matrix": cm_table})
    wandb.finish()

    return {
        "fold":           fold,
        "test_subject":   fold,
        "train_clips":    len(train_ds),
        "val_clips":      len(val_ds),
        "test_clips":     len(test_ds),
        "best_WAR":       test_m["war"],
        "best_UAR":       test_m["uar"],
        "best_MacroP":    test_m["macro_p"],
        "best_MacroF1":   test_m["macro_f1"],
        "best_WeightedP": test_m["weighted_p"],
        "best_WeightedR": test_m["weighted_r"],
        "best_WF1":       test_m["weighted_f1"],
        "best_epoch":     best_epoch,
        "val_UAR":        best_uar,
        "train_min":      elapsed,
        **{f"recall_{k}": v for k, v in test_m.get("per_class_recall", {}).items()},
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], class_names: list[str], out_dir: Path, run_name: str):
    wars    = [r["best_WAR"]       for r in results]
    uars    = [r["best_UAR"]       for r in results]
    mps     = [r.get("best_MacroP",    0.0) for r in results]
    mf1s    = [r["best_MacroF1"]   for r in results]
    wps     = [r.get("best_WeightedP", 0.0) for r in results]
    wrs     = [r.get("best_WeightedR", 0.0) for r in results]
    wf1s    = [r["best_WF1"]       for r in results]

    lines = [
        f"LOSO Summary — {run_name}",
        f"Folds: {len(results)}/15",
        f"Mean WAR        : {np.mean(wars):.2f}% ± {np.std(wars):.2f}%",
        f"Mean UAR        : {np.mean(uars):.2f}% ± {np.std(uars):.2f}%",
        f"Mean Macro-P    : {np.mean(mps):.2f}% ± {np.std(mps):.2f}%",
        f"Mean Macro-F1   : {np.mean(mf1s):.2f}% ± {np.std(mf1s):.2f}%",
        f"Mean Weighted-P : {np.mean(wps):.2f}% ± {np.std(wps):.2f}%",
        f"Mean Weighted-R : {np.mean(wrs):.2f}% ± {np.std(wrs):.2f}%",
        f"Mean W-F1       : {np.mean(wf1s):.2f}% ± {np.std(wf1s):.2f}%",
        "",
        "Per-fold:",
    ]
    for r in results:
        lines.append(
            f"  Fold {r['fold']:2d}: WAR={r['best_WAR']:.2f}% UAR={r['best_UAR']:.2f}% "
            f"MacroP={r.get('best_MacroP',0):.2f}% MacroF1={r['best_MacroF1']:.2f}% "
            f"WP={r.get('best_WeightedP',0):.2f}% WR={r.get('best_WeightedR',0):.2f}% "
            f"WF1={r['best_WF1']:.2f}%"
        )
    lines += ["", "Per-class recall (mean ± std):"]
    for cls in class_names:
        key = f"recall_{cls}"
        vals = [r[key] for r in results if key in r]
        if vals:
            marker = " ←" if np.mean(vals) < 20.0 else ""
            lines.append(f"  {cls:25s}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%{marker}")

    summary_text = "\n".join(lines)
    print(f"\n{'='*65}")
    print(summary_text)
    print(f"{'='*65}")

    summary_path = out_dir / f"{run_name}_summary.txt"
    summary_path.write_text(summary_text)
    print(f"\n[✓] Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def print_multiseed_summary(
    all_seed_results: dict[int, list[dict]],
    class_names: list[str],
    out_dir: Path,
    run_name: str,
):
    """
    Aggregate LOSO results across multiple seeds.
    Per fold: average metrics across seeds → then mean ± std across folds.
    """
    seeds = sorted(all_seed_results)
    all_folds = sorted(set(
        r["fold"]
        for seed_results in all_seed_results.values()
        for r in seed_results
    ))

    metric_keys = ["best_WAR", "best_UAR", "best_MacroF1", "best_WF1",
                   "best_MacroP", "best_WeightedP", "best_WeightedR"]
    recall_keys = [f"recall_{c}" for c in class_names]

    agg_results = []
    for fold in all_folds:
        fold_rows = [
            r for sr in all_seed_results.values() for r in sr if r["fold"] == fold
        ]
        if not fold_rows:
            continue
        avg: dict = {"fold": fold, "test_subject": fold}
        for key in metric_keys + recall_keys:
            vals = [r[key] for r in fold_rows if key in r]
            if vals:
                avg[key] = round(float(np.mean(vals)), 4)
        agg_results.append(avg)

    print(f"\n{'='*65}")
    print(f"  Multi-seed aggregate — {run_name}  (seeds={seeds})")
    print(f"{'='*65}")
    for seed in seeds:
        rs = all_seed_results[seed]
        if not rs:
            continue
        w = [r["best_WAR"] for r in rs]
        u = [r["best_UAR"] for r in rs]
        print(f"  Seed {seed:4d}: WAR={np.mean(w):.2f}%±{np.std(w):.2f}%  "
              f"UAR={np.mean(u):.2f}%±{np.std(u):.2f}%  ({len(rs)} folds)")

    if agg_results:
        print_summary(agg_results, class_names, out_dir, f"{run_name}_agg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",               default="config.yaml")
    p.add_argument("--run_name",             default=None,
                   help="Nama output (misal: loso_step1_swin3d_t_7cls)")
    p.add_argument("--fold",                 type=int, default=None,
                   help="Jalankan hanya fold ini (1-15)")
    p.add_argument("--start_fold",           type=int, default=1)
    p.add_argument("--backbone",             default=None)
    p.add_argument("--loss_type",            default=None, choices=["ce", "focal"])
    p.add_argument("--use_td",               action="store_true")
    p.add_argument("--emotion_set",          default=None, choices=["basic", "all"])
    p.add_argument("--num_frames",           type=int, default=None,
                   help="Override T dari Step 0 (misal: 16)")
    p.add_argument("--frame_interval",       type=int, default=None,
                   help="Override interval dari Step 0 (misal: 4)")
    p.add_argument("--init_checkpoint_dir",  default=None,
                   help="Step 4a: direktori berisi fold_XX_best.pt dari step 3")
    p.add_argument("--td_variant",           default=None,
                   choices=list(TD_VARIANTS),
                   help="Step 5: pilih arsitektur TD (td_base/td_pe/td_2l/td_4x/td_cls/td_full)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seed(s) untuk multi-seed robustness (contoh: --seeds 42 123 456). "
                        "Default: pakai seed dari config (single run).")
    p.add_argument("--freeze_first_epochs", type=int, default=0,
                   help="Bekukan backbone N epoch pertama (single-stage training, tanpa "
                        "init_checkpoint_dir). 0 = tidak freeze (default).")
    args = p.parse_args()

    with open(BASE_DIR / args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.backbone:       cfg["model"]["backbone"]         = args.backbone
    if args.loss_type:      cfg["train"]["loss_type"]        = args.loss_type
    if args.use_td:         cfg["model"]["use_td"]           = True
    if args.num_frames:     cfg["dataset"]["num_frames"]     = args.num_frames
    if args.frame_interval: cfg["dataset"]["frame_interval"] = args.frame_interval
    if args.td_variant:
        cfg["model"]["use_td"]      = True
        cfg["model"]["td_config"]   = TD_VARIANTS[args.td_variant]
        cfg["model"]["td_variant"]  = args.td_variant
    if args.emotion_set:
        cfg["dataset"]["emotion_set"] = args.emotion_set
        if args.emotion_set == "all":
            cfg["dataset"]["num_classes"] = 19
            cfg["dataset"]["class_names"] = [
                "angrily-disgusted", "angrily-surprised", "angry",
                "disgust", "disgustedly-surprised",
                "fear", "fear-angry", "fear-disgust", "fear-surprise",
                "happy", "happy-disgust", "happy-surprise",
                "neutral",
                "sad", "sad-angry", "sad-disgust", "sad-fear", "sad-surprise",
                "surprise",
            ]
        else:
            cfg["dataset"]["num_classes"] = 7
            cfg["dataset"]["class_names"] = [
                "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
            ]

    # Derive run_name dari config jika tidak diisi
    if args.run_name is None:
        bb  = cfg["model"]["backbone"]
        td  = "td" if cfg["model"].get("use_td") else "notd"
        lt  = cfg["train"].get("loss_type", "ce")
        cls = cfg["dataset"]["emotion_set"]
        args.run_name = f"loso_{bb}_{td}_{lt}_{cls}"

    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    amp_label = "bf16" if amp_dtype == torch.bfloat16 else "fp16"

    bb   = cfg["model"]["backbone"]
    td   = "TD" if cfg["model"].get("use_td") else "noTD"
    lt   = cfg["train"].get("loss_type", "ce").upper()
    es   = cfg["dataset"]["emotion_set"]
    nc   = cfg["dataset"]["num_classes"]
    s4   = f" | step4 init: {args.init_checkpoint_dir}" if args.init_checkpoint_dir else ""

    print(f"\n[✓] Device  : {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print(f"[✓] AMP     : {amp_label}")
    print(f"[✓] Run     : {args.run_name}")
    print(f"[✓] Config  : {bb} | {td} | {lt} | {es} ({nc} kelas){s4}")
    print(f"[✓] T={cfg['dataset']['num_frames']} interval={cfg['dataset']['frame_interval']} "
          f"window={cfg['dataset']['num_frames']*cfg['dataset']['frame_interval']}fr")

    seeds       = args.seeds if args.seeds else [cfg["train"]["seed"]]
    multi_seed  = len(seeds) > 1
    results_dir = PROJECT_ROOT / "outputs" / "results"
    results_dir.mkdir(exist_ok=True)
    init_ckpt_dir = Path(args.init_checkpoint_dir) if args.init_checkpoint_dir else None
    fold_range    = [args.fold] if args.fold else range(args.start_fold, 16)

    if multi_seed:
        print(f"[✓] Multi-seed: {seeds}")

    all_seed_results: dict[int, list[dict]] = {}

    for seed in seeds:
        cfg_s = copy.deepcopy(cfg)
        cfg_s["train"]["seed"] = seed
        seed_suffix   = f"_s{seed}" if multi_seed else ""
        seed_run_name = f"{args.run_name}{seed_suffix}"

        runs_dir = PROJECT_ROOT / "outputs" / "runs" / seed_run_name
        csv_path = results_dir / f"{seed_run_name}.csv"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Step 4a two-stage: auto-derive seed-specific init_checkpoint_dir.
        # Jika user memberi "outputs/runs/step3_td_gru_7cls_s42" atau "outputs/runs/step3_td_gru_7cls",
        # kode akan mencoba "…_s{seed}" untuk seed aktif.
        if init_ckpt_dir is not None and multi_seed:
            base_name = re.sub(r"_s\d+$", "", init_ckpt_dir.name)
            candidate = init_ckpt_dir.parent / f"{base_name}_s{seed}"
            if candidate.exists():
                effective_init_ckpt: Path | None = candidate
            else:
                effective_init_ckpt = init_ckpt_dir
                print(f"  [WARN] Seed {seed}: init ckpt {candidate} tidak ada, "
                      f"fallback ke {init_ckpt_dir}")
        else:
            effective_init_ckpt = init_ckpt_dir

        # Baca fold yang sudah selesai untuk seed ini
        existing: dict[int, dict] = {}
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    fid = int(row["fold"])
                    existing[fid] = {
                        k: (int(v) if k in ("fold", "test_subject", "train_clips", "val_clips", "best_epoch")
                            else float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v)
                        for k, v in row.items()
                    }
            if existing:
                print(f"[✓] Seed {seed}: melanjutkan ({len(existing)} fold ada)")

        print(f"\n{'='*65}")
        print(f"  LOSO — {seed_run_name}  (seed={seed})")
        print(f"{'='*65}\n")

        for fold in fold_range:
            if fold in existing:
                r = existing[fold]
                print(f"  Fold {fold:2d} [SKIP] WAR={r['best_WAR']:.2f}% UAR={r['best_UAR']:.2f}%")
                continue

            fold_ckpt_path = runs_dir / f"fold_{fold:02d}_best.pt"
            print(f"\n--- Fold {fold:2d} / 15 (test subject {fold}) ---")

            result = run_fold(
                cfg_s, fold, device, amp_dtype,
                fold_ckpt_path=fold_ckpt_path,
                init_checkpoint_dir=effective_init_ckpt,
                run_name=seed_run_name,
                freeze_first_epochs=args.freeze_first_epochs,
            )
            if result is None:
                continue

            existing[fold] = result

            all_r = sorted(existing.values(), key=lambda r: r["fold"])
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_r[0].keys()), extrasaction="ignore")
                w.writeheader()
                w.writerows(all_r)
            print(f"  [✓] CSV: {csv_path}")

        seed_results = sorted(existing.values(), key=lambda r: r["fold"])
        all_seed_results[seed] = seed_results
        if seed_results:
            print_summary(seed_results, cfg_s["dataset"]["class_names"],
                          results_dir, seed_run_name)

    if multi_seed:
        print_multiseed_summary(all_seed_results, cfg["dataset"]["class_names"],
                                results_dir, args.run_name)


if __name__ == "__main__":
    main()
