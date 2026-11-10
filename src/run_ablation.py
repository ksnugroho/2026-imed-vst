"""
run_ablation.py — iMED-VST
---------------------------
Temporal window ablation study untuk Video Swin Transformer.

Research question:
  "Bagaimana konfigurasi temporal window (T, interval) mempengaruhi
   pengenalan facial expression dari video?"

Grid eksperimen (5 config, random split 80/20, 7-class basic):
  Config  T   interval  Window(frames)  Backbone
  ──────  ─── ────────  ──────────────  ──────────
  A        8     2           16         swin3d_t
  B        8     4           32         swin3d_t
  C       16     2           32         swin3d_t
  D       16     4           64         swin3d_t
  E       32     2           64         swin3d_t

Backbone dikunci di swin3d_t untuk isolasi variable temporal.
Backbone comparison dilakukan di Step 1 (LOSO, lebih valid).

Cara pakai:
  python run_ablation.py                   # jalankan semua
  python run_ablation.py --only D          # satu config saja
  python run_ablation.py --skip_done       # lanjut yang belum selesai

Output:
  results/ablation_temporal.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
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

from model   import VideoSwinFER, build_loss
from dataset import IMEDVideoDataset, VideoTransform, build_weighted_sampler
from metrics import FERMetrics
from train   import set_seed, build_optimizer, build_scheduler, train_one_epoch, evaluate


# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------

ABLATION_GRID = [
    # id   T   interval  backbone     description
    ("A",  8,     2,    "swin3d_t",  "narrow-dense  (window=16fr)"),
    ("B",  8,     4,    "swin3d_t",  "medium-sparse (window=32fr)"),
    ("C", 16,     2,    "swin3d_t",  "medium-dense  (window=32fr)"),
    ("D", 16,     4,    "swin3d_t",  "wide-sparse   (window=64fr)"),
    ("E", 32,     2,    "swin3d_t",  "wide-dense    (window=64fr)"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_base_cfg() -> dict:
    with open(BASE_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def make_cfg(base: dict, T: int, interval: int, backbone: str, run_id: str) -> dict:
    """Override temporal + backbone config untuk satu ablation run."""
    cfg = copy.deepcopy(base)
    cfg["dataset"]["num_frames"]     = T
    cfg["dataset"]["frame_interval"] = interval
    cfg["dataset"]["split_mode"]     = "random"   # ablation pakai random split
    cfg["dataset"]["emotion_set"]    = "basic"     # 7-class, isolasi variable temporal
    cfg["dataset"]["num_classes"]    = 7
    cfg["dataset"]["class_names"]    = [
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
    ]
    cfg["model"]["backbone"] = backbone
    cfg["model"]["use_td"]   = False               # ablation tanpa TD
    cfg["train"]["loss_type"]        = "ce"        # ablation pakai CE
    cfg["train"]["label_smoothing"]  = 0.1
    return cfg


def build_loaders(cfg: dict):
    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg["augment"]

    train_tf = VideoTransform(
        img_size=ds_cfg["img_size"], is_train=True,
        color_jitter=aug_cfg.get("color_jitter", 0.4),
        random_grayscale=aug_cfg.get("random_grayscale", 0.1),
        flip_prob=aug_cfg.get("flip_horizontal", 0.5),
    )
    val_tf = VideoTransform(img_size=ds_cfg["img_size"], is_train=False)

    kw = dict(
        root           = ds_cfg["root"],
        split_mode     = "random",
        train_ratio    = ds_cfg.get("train_ratio", 0.8),
        emotion_set    = ds_cfg["emotion_set"],
        num_frames     = ds_cfg["num_frames"],
        frame_interval = ds_cfg["frame_interval"],
        seed           = cfg["train"]["seed"],
    )
    train_ds = IMEDVideoDataset(split="train", transform=train_tf, **kw)
    val_ds   = IMEDVideoDataset(split="val",   transform=val_tf,   **kw)

    sampler = build_weighted_sampler(train_ds) if cfg["train"].get("use_weighted_loss") else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        sampler=sampler, shuffle=(sampler is None),
        num_workers=ds_cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval"]["batch_size"],
        shuffle=False, num_workers=ds_cfg["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader, train_ds


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_one(
    cfg: dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    run_id: str,
    desc: str,
    run_dir: Path,
) -> dict:
    set_seed(cfg["train"]["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["train"]["seed"]
    wandb.init(
        entity  = WANDB_ENTITY,
        project = WANDB_PROJECT,
        name    = f"step0_{run_id}_s{seed}",
        group   = "step0_temporal_ablation",
        config  = {
            "backbone":       cfg["model"]["backbone"],
            "T":              cfg["dataset"]["num_frames"],
            "frame_interval": cfg["dataset"]["frame_interval"],
            "window_fr":      cfg["dataset"]["num_frames"] * cfg["dataset"]["frame_interval"],
            "emotion_set":    cfg["dataset"]["emotion_set"],
            "num_classes":    cfg["dataset"]["num_classes"],
            "loss_type":      cfg["train"]["loss_type"],
            "epochs":         cfg["train"]["epochs"],
            "lr":             cfg["train"]["lr"],
            "batch_size":     cfg["train"]["batch_size"],
            "seed":           seed,
            "split":          "random_80_20",
        },
        reinit = True,
    )

    train_loader, val_loader, train_ds = build_loaders(cfg)

    model = VideoSwinFER(
        backbone     = cfg["model"]["backbone"],
        num_classes  = cfg["dataset"]["num_classes"],
        pretrained   = cfg["model"].get("pretrained", True),
        dropout_head = cfg["model"].get("dropout_head", 0.5),
        use_td       = False,
    ).to(device)

    class_weights = train_ds.class_weights().to(device) if cfg["train"]["use_weighted_loss"] else None
    loss_fn = build_loss(cfg, class_weights)
    if hasattr(loss_fn, "alpha") and loss_fn.alpha is not None:
        loss_fn = loss_fn.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = torch.cuda.amp.GradScaler(
        enabled=cfg["hardware"]["mixed_precision"] and device.type == "cuda"
    )
    metrics   = FERMetrics(cfg["dataset"]["class_names"])

    best_uar         = 0.0
    best_war         = 0.0
    best_macro_p     = 0.0
    best_macro_f1    = 0.0
    best_weighted_p  = 0.0
    best_weighted_r  = 0.0
    best_weighted_f1 = 0.0
    best_epoch       = 0
    best_per_class   = {}
    best_cm          = []
    eval_every       = cfg["eval"]["eval_every_n_epochs"]
    t_start          = time.time()

    for epoch in range(cfg["train"]["epochs"]):
        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, device, cfg["train"]["clip_grad"], amp_dtype=amp_dtype,
        )

        log = {"epoch": epoch + 1, "train/loss": train_m["loss"], "train/top1": train_m["top1"]}

        do_eval = (epoch + 1) % eval_every == 0 or epoch == cfg["train"]["epochs"] - 1
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
                    "epoch": epoch, "model": model.state_dict(),
                    "best_uar": best_uar, "cfg": cfg,
                }, run_dir / "best.pt")

            elapsed = int(time.time() - t_start)
            mk = "★" if is_new_best else " "
            print(
                f"  [{run_id}|ep{epoch+1:3d}]{mk} "
                f"loss={train_m['loss']:.3f} WAR={val_m['war']:.1f}% "
                f"UAR={val_m['uar']:.1f}% MacroF1={val_m['macro_f1']:.1f}% "
                f"[best={best_uar:.1f}%] {elapsed}s"
            )

        wandb.log(log)

    total_min = round((time.time() - t_start) / 60, 1)

    wandb.run.summary["best/war"]         = best_war
    wandb.run.summary["best/uar"]         = best_uar
    wandb.run.summary["best/macro_p"]     = best_macro_p
    wandb.run.summary["best/macro_f1"]    = best_macro_f1
    wandb.run.summary["best/weighted_p"]  = best_weighted_p
    wandb.run.summary["best/weighted_r"]  = best_weighted_r
    wandb.run.summary["best/weighted_f1"] = best_weighted_f1
    wandb.run.summary["best/epoch"]       = best_epoch
    for cls, v in best_per_class.items():
        wandb.run.summary[f"recall/{cls}"] = v
    if best_cm:
        class_names = cfg["dataset"]["class_names"]
        cm_table = wandb.Table(columns=["true\\pred"] + class_names)
        for i, row in enumerate(best_cm):
            cm_table.add_data(class_names[i], *row)
        wandb.log({"best/confusion_matrix": cm_table})
    wandb.finish()

    return {
        "config_id":        run_id,
        "description":      desc,
        "backbone":         cfg["model"]["backbone"],
        "T":                cfg["dataset"]["num_frames"],
        "interval":         cfg["dataset"]["frame_interval"],
        "window_fr":        cfg["dataset"]["num_frames"] * cfg["dataset"]["frame_interval"],
        "best_epoch":       best_epoch,
        "best_WAR":         best_war,
        "best_UAR":         best_uar,
        "best_MacroP":      best_macro_p,
        "best_MacroF1":     best_macro_f1,
        "best_WeightedP":   best_weighted_p,
        "best_WeightedR":   best_weighted_r,
        "best_WF1":         best_weighted_f1,
        "train_min":        total_min,
        **{f"recall_{k}": v for k, v in best_per_class.items()},
    }


# ---------------------------------------------------------------------------
# Multi-seed summary
# ---------------------------------------------------------------------------

def print_multiseed_ablation_summary(
    seed_results: dict[int, dict[str, dict]],
    results_dir: Path,
):
    """
    Agregasi multi-seed Step 0.
    seed_results: {seed → {config_id → row_dict}}
    Untuk setiap config_id: mean ± std metrik across seeds.
    """
    seeds = sorted(seed_results)
    all_config_ids = sorted(set(
        cid for sr in seed_results.values() for cid in sr
    ), key=lambda x: ABLATION_GRID[[r[0] for r in ABLATION_GRID].index(x)][1]
       if x in [r[0] for r in ABLATION_GRID] else 99)

    metric_keys = ["best_WAR", "best_UAR", "best_MacroF1", "best_WF1"]

    lines = [
        "=" * 90,
        f"Step 0 — Temporal Window Ablation (Multi-seed, seeds={seeds})",
        f"{'ID':<5} {'T':>3} {'int':>4} {'win':>5}  "
        f"{'WAR':>14} {'UAR':>14} {'MacroF1':>14} {'WF1':>14}",
        "─" * 90,
    ]

    agg_rows = []
    best_uar = -1
    best_cfg = None

    for run_id, T, interval, backbone, desc in ABLATION_GRID:
        if run_id not in all_config_ids:
            continue
        per_seed_rows = [seed_results[s][run_id] for s in seeds if run_id in seed_results[s]]
        if not per_seed_rows:
            continue

        agg = {"config_id": run_id, "T": T, "interval": interval,
               "window_fr": T * interval, "backbone": backbone, "description": desc}
        for key in metric_keys:
            vals = [r[key] for r in per_seed_rows if key in r]
            agg[key]             = round(float(np.mean(vals)), 4) if vals else 0.0
            agg[f"{key}_std"]    = round(float(np.std(vals)),  4) if vals else 0.0
        agg_rows.append(agg)

        if agg["best_UAR"] > best_uar:
            best_uar = agg["best_UAR"]
            best_cfg = run_id

        mark = " ← best" if run_id == best_cfg else ""
        lines.append(
            f"  {run_id:<5} {T:>3} {interval:>4} {T*interval:>4}fr  "
            + "  ".join(
                f"{agg[k]:6.2f}%±{agg[k+'_std']:.2f}%"
                for k in metric_keys
            ) + mark
        )

    lines.append("=" * 90)
    text = "\n".join(lines)
    print(f"\n{text}\n")

    csv_agg = results_dir / "ablation_temporal_agg.csv"
    if agg_rows:
        with open(csv_agg, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)
        print(f"[✓] Aggregated CSV: {csv_agg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only",      default=None,
                   help="Jalankan hanya config ini (A/B/C/D/E)")
    p.add_argument("--skip_done", action="store_true",
                   help="Skip config yang sudah ada best.pt")
    p.add_argument("--seeds",     type=int, nargs="+", default=None,
                   help="Seed(s) untuk robustness (default: seed dari config). "
                        "Step 0 umumnya cukup single seed — ini opsional.")
    args = p.parse_args()

    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    amp_label = "bf16" if amp_dtype == torch.bfloat16 else "fp16"

    print(f"\n[✓] Device  : {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print(f"[✓] AMP     : {amp_label}")
    print(f"[✓] Grid    : {len(ABLATION_GRID)} configs")

    base_cfg    = load_base_cfg()
    seeds       = args.seeds if args.seeds else [base_cfg["train"]["seed"]]
    multi_seed  = len(seeds) > 1
    results_dir = PROJECT_ROOT / "outputs" / "results"
    results_dir.mkdir(exist_ok=True)

    if multi_seed:
        print(f"[✓] Multi-seed: {seeds}")

    grid = [(rid, T, iv, bb, desc) for rid, T, iv, bb, desc in ABLATION_GRID
            if args.only is None or rid == args.only]

    # seed_results_map[seed][config_id] = row_dict
    seed_results_map: dict[int, dict[str, dict]] = {}

    for seed in seeds:
        seed_suffix = f"_s{seed}" if multi_seed else ""
        csv_path    = results_dir / f"ablation_temporal{seed_suffix}.csv"

        # Baca hasil yang sudah ada untuk seed ini
        existing: dict[str, dict] = {}
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    existing[row["config_id"]] = {
                        k: (float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v)
                        for k, v in row.items()
                    }
            if existing:
                print(f"[✓] Seed {seed}: {len(existing)} config sudah ada")

        all_results = list(existing.values())

        for i, (run_id, T, interval, backbone, desc) in enumerate(grid, 1):
            run_dir = PROJECT_ROOT / "outputs" / "runs" / f"ablation_{run_id}{seed_suffix}"

            if args.skip_done and (run_dir / "best.pt").exists():
                print(f"[{i}/{len(grid)}] SKIP {run_id} (seed={seed}) — best.pt sudah ada")
                continue
            if run_id in existing:
                r = existing[run_id]
                print(f"[{i}/{len(grid)}] SKIP {run_id} (seed={seed}) — "
                      f"ada di CSV (UAR={r.get('best_UAR', '?'):.2f}%)")
                continue

            cfg = make_cfg(base_cfg, T, interval, backbone, run_id)
            cfg["train"]["seed"] = seed

            print(f"\n{'─'*65}")
            print(f"  [{i}/{len(grid)}] Config {run_id} (seed={seed}): {desc}")
            print(f"  T={T}  interval={interval}  window={T*interval}fr  backbone={backbone}")
            print(f"{'─'*65}")

            row = run_one(cfg, device, amp_dtype, run_id, desc, run_dir)
            all_results.append(row)
            existing[run_id] = row

            # Simpan CSV per seed setelah setiap config
            fieldnames = list(all_results[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_results)
            print(f"  [✓] CSV: {csv_path}")

        seed_results_map[seed] = existing

    if multi_seed and seed_results_map:
        print_multiseed_ablation_summary(seed_results_map, results_dir)

    # Summary table single-seed (atau seed terakhir jika multi)
    last_seed = seeds[-1]
    sortable  = [r for r in seed_results_map.get(last_seed, {}).values()
                 if isinstance(r.get("best_UAR"), (int, float))]
    if sortable:
        print(f"\n{'='*90}")
        print("  TABLE I — Temporal Window Ablation")
        print(f"{'='*90}")
        print(f"  {'ID':<5} {'T':>3} {'int':>4} {'win':>5} {'backbone':<12} "
              f"{'WAR':>7} {'UAR':>7} {'MacroP':>7} {'MacroF1':>8} {'WP':>7} {'WR':>7} {'WF1':>7}")
        print(f"  {'─'*5} {'─'*3} {'─'*4} {'─'*5} {'─'*12} "
              f"{'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")
        best_uar_val = max(r["best_UAR"] for r in sortable)
        for r in sorted(sortable, key=lambda x: -x["best_UAR"]):
            mark = " ← best" if r["best_UAR"] == best_uar_val else ""
            print(
                f"  {r['config_id']:<5} {int(r['T']):>3} {int(r['interval']):>4} "
                f"{int(r['window_fr']):>4}fr {r['backbone']:<12} "
                f"{r['best_WAR']:>6.2f}% {r['best_UAR']:>6.2f}% "
                f"{r.get('best_MacroP',0):>6.2f}% {r['best_MacroF1']:>7.2f}% "
                f"{r.get('best_WeightedP',0):>6.2f}% {r.get('best_WeightedR',0):>6.2f}% "
                f"{r['best_WF1']:>6.2f}%{mark}"
            )
        print(f"\n[✓] Hasil: {csv_path}")


if __name__ == "__main__":
    main()
