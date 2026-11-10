"""
run_random_split_exp.py — iMED-VST
-------------------------------------
Identity Leakage Controlled Experiment (Reviewer Group B).

Tujuan: Buktikan bahwa hasil LOSO yang lebih rendah dari prior work bukan
karena model lemah, melainkan karena prior work menggunakan random split yang
membocorkan identitas subjek (train/test = subjek yang sama — "identity leakage").

Desain eksperimen:
  Kondisi A (LOSO):          train=13 subjek, val=1, test=1 → subjek disjoint
  Kondisi B (Random split):  60/20/20 clip-level random split → subjek bisa overlap

  Model, hyperparameter, dan jumlah epoch IDENTIK di kedua kondisi.
  Perbedaan satu-satunya: apakah identitas subjek bocor ke test set.

Hipotesis: Kondisi B akan menghasilkan WAR jauh lebih tinggi (mungkin 70–80%+)
           versus Kondisi A (~40–50%), membuktikan bahwa angka tinggi pada prior
           work disebabkan identity leakage, bukan keunggulan arsitektur.

Cara pakai:
  # Jalankan kedua kondisi (3 seeds masing-masing)
  conda run -n ksn_cv_task python run_random_split_exp.py

  # Hanya kondisi random
  conda run -n ksn_cv_task python run_random_split_exp.py --condition random

  # Hanya kondisi LOSO (re-pakai hasil run_loso.py jika sudah ada)
  conda run -n ksn_cv_task python run_random_split_exp.py --condition loso \
      --loso_csv outputs/results/step3_td_gru_7cls_s42.csv

Output:
  results/identity_leakage_random.csv       (3 baris, 1 per seed)
  results/identity_leakage_loso.csv         (15 baris, 1 per fold) — jika dijalankan
  results/identity_leakage_comparison.txt   (tabel perbandingan)
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
import yaml

BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from model   import VideoSwinFER, build_loss
from dataset import IMEDVideoDataset, VideoTransform, build_weighted_sampler
from metrics import FERMetrics
from train   import (
    set_seed, build_optimizer, build_scheduler,
    train_one_epoch, evaluate,
)
from torch.utils.data import DataLoader

WANDB_ENTITY  = "bdsrc-team"
WANDB_PROJECT = "2026-imed-vst-v2"

RANDOM_SEEDS = [42, 123, 456]   # 3 seeds → 3 independent random-split runs


# ---------------------------------------------------------------------------
# Random-split training run (single seed)
# ---------------------------------------------------------------------------

def run_random_single(
    base_cfg: dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
) -> dict | None:
    """
    Kondisi B: satu run dengan 60/20/20 clip-level random split.
    Subjek BISA muncul di train DAN test — ini yang disebut identity leakage.
    Arsitektur, LR, epoch, augmentasi IDENTIK dengan kondisi LOSO.
    """
    import wandb

    cfg     = copy.deepcopy(base_cfg)
    cfg["train"]["seed"] = seed
    set_seed(seed)

    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg["augment"]
    tr_cfg  = cfg["train"]
    run_name = f"identity_leakage_random_s{seed}"

    # Transforms — sama persis dengan run_loso.py
    train_tf = VideoTransform(
        img_size         = ds_cfg["img_size"],
        is_train         = True,
        color_jitter     = aug_cfg.get("color_jitter", 0.4),
        random_grayscale = aug_cfg.get("random_grayscale", 0.1),
        flip_prob        = aug_cfg.get("flip_horizontal", 0.5),
    )
    val_tf = VideoTransform(img_size=ds_cfg["img_size"], is_train=False)

    # 60/20/20 stratified random split (subject-leaked)
    kw = dict(
        root           = ds_cfg["root"],
        split_mode     = "random",
        emotion_set    = ds_cfg.get("emotion_set", "basic"),
        num_frames     = ds_cfg["num_frames"],
        frame_interval = ds_cfg["frame_interval"],
        train_ratio    = 0.6,   # triggers 3-way split in dataset._split_random
        seed           = seed,
    )
    train_ds = IMEDVideoDataset(split="train", transform=train_tf, **kw)
    val_ds   = IMEDVideoDataset(split="val",   transform=val_tf,   **kw)
    test_ds  = IMEDVideoDataset(split="test",  transform=val_tf,   **kw)

    if len(test_ds) == 0:
        print(f"  [SKIP] Random seed {seed}: test set kosong")
        return None

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

    wandb.init(
        entity  = WANDB_ENTITY,
        project = WANDB_PROJECT,
        name    = run_name,
        group   = "identity_leakage",
        config  = {
            "condition":   "random_60_20_20",
            "seed":        seed,
            "emotion_set": ds_cfg.get("emotion_set"),
            "num_classes": ds_cfg["num_classes"],
            "backbone":    cfg["model"]["backbone"],
            "use_td":      cfg["model"].get("use_td", False),
            "td_variant":  cfg["model"].get("td_variant", "none"),
            "epochs":      tr_cfg["epochs"],
            "train_clips": len(train_ds),
            "val_clips":   len(val_ds),
            "test_clips":  len(test_ds),
        },
        reinit = True,
    )

    model = VideoSwinFER(
        backbone     = cfg["model"]["backbone"],
        num_classes  = ds_cfg["num_classes"],
        pretrained   = cfg["model"].get("pretrained", True),
        dropout_head = cfg["model"].get("dropout_head", 0.5),
        use_td       = cfg["model"].get("use_td", False),
        td_config    = cfg["model"].get("td_config"),
    ).to(device)

    class_weights = train_ds.class_weights().to(device) if tr_cfg["use_weighted_loss"] else None
    loss_fn = build_loss(cfg, class_weights)
    if hasattr(loss_fn, "alpha") and loss_fn.alpha is not None:
        loss_fn = loss_fn.to(device)

    metrics = FERMetrics(ds_cfg["class_names"])

    # Tidak ada two-stage — harus identik dengan kondisi LOSO (run_fold tanpa
    # init_checkpoint_dir, yang juga freeze_epochs=0). Config freeze_backbone_epochs
    # hanya berlaku untuk Step 4a two-stage, bukan untuk eksperimen leakage ini.
    freeze_epochs = 0

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = torch.cuda.amp.GradScaler(
        enabled=cfg["hardware"]["mixed_precision"] and device.type == "cuda"
    )

    best_val_uar = 0.0
    best_epoch   = 0
    runs_dir     = PROJECT_ROOT / "outputs" / "runs" / "identity_leakage_random"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path    = runs_dir / f"seed_{seed}_best.pt"
    t_start      = time.time()
    eval_every   = cfg["eval"]["eval_every_n_epochs"]

    for epoch in range(tr_cfg["epochs"]):
        # Unfreeze backbone at stage 2
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.freeze_backbone(False)
            optimizer = build_optimizer(model, cfg)
            remaining = tr_cfg["epochs"] - epoch
            scheduler = build_scheduler(
                optimizer,
                {**cfg, "train": {**tr_cfg, "epochs": remaining, "warmup_epochs": 0}},
                len(train_loader),
            )
            scaler = torch.cuda.amp.GradScaler(
                enabled=cfg["hardware"]["mixed_precision"] and device.type == "cuda"
            )
            print(f"  [Stage2 epoch {epoch+1}] backbone unfrozen")

        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, device, tr_cfg["clip_grad"], amp_dtype=amp_dtype,
        )

        log = {"epoch": epoch + 1, "train/loss": train_m["loss"], "train/top1": train_m["top1"]}

        do_eval = (epoch + 1) % eval_every == 0 or epoch == tr_cfg["epochs"] - 1
        if do_eval:
            val_m = evaluate(model, val_loader, loss_fn, metrics, device, amp_dtype=amp_dtype)
            log.update({"val/war": val_m["war"], "val/uar": val_m["uar"],
                        "val/macro_f1": val_m["macro_f1"]})
            is_new_best = val_m["uar"] > best_val_uar
            if is_new_best:
                best_val_uar = val_m["uar"]
                best_epoch   = epoch + 1
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "best_uar": best_val_uar, "cfg": cfg}, ckpt_path)
            mk = "★" if is_new_best else " "
            print(f"  [ep{epoch+1:3d}/{tr_cfg['epochs']}]{mk} "
                  f"WAR={val_m['war']:.1f}% UAR={val_m['uar']:.1f}% "
                  f"[best_val={best_val_uar:.1f}%]")

        wandb.log(log)

    elapsed = round((time.time() - t_start) / 60, 1)

    # Evaluate best checkpoint on held-out test set
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, loss_fn, metrics, device, amp_dtype=amp_dtype)

    print(f"  Seed {seed} | [val] UAR={best_val_uar:.2f}%  "
          f"[test] WAR={test_m['war']:.2f}% UAR={test_m['uar']:.2f}% "
          f"MacroF1={test_m['macro_f1']:.2f}%  {elapsed}mnt")

    wandb.run.summary.update({
        "best/val_uar": best_val_uar, "best/war": test_m["war"],
        "best/uar": test_m["uar"], "best/macro_f1": test_m["macro_f1"],
        "best/epoch": best_epoch,
    })
    wandb.finish()

    return {
        "condition":  "random",
        "seed":       seed,
        "best_epoch": best_epoch,
        "train_clips": len(train_ds),
        "val_clips":   len(val_ds),
        "test_clips":  len(test_ds),
        "best_WAR":    test_m["war"],
        "best_UAR":    test_m["uar"],
        "best_MacroF1": test_m["macro_f1"],
        "best_WF1":    test_m["weighted_f1"],
        "val_UAR":     best_val_uar,
        "train_min":   elapsed,
    }


# ---------------------------------------------------------------------------
# LOSO condition — re-use run_loso machinery
# ---------------------------------------------------------------------------

def run_loso_condition(
    base_cfg: dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    seeds: list[int],
    loso_csv: str | None = None,
) -> list[dict]:
    """
    Kondisi A: LOSO multi-seed (re-pakai run_loso.run_fold).
    Jika loso_csv ada, muat dari sana (skip training ulang).
    """
    from run_loso import run_fold, print_summary, print_multiseed_summary

    if loso_csv:
        path = Path(loso_csv)
        if path.exists():
            rows = list(csv.DictReader(open(path)))
            return [
                {k: (int(v) if k in ("fold", "test_subject", "train_clips", "val_clips", "best_epoch")
                     else float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v)
                 for k, v in row.items()}
                for row in rows
            ]
        print(f"[WARN] loso_csv tidak ditemukan: {loso_csv}. Training dari awal.")

    results_dir = PROJECT_ROOT / "outputs" / "results"
    results_dir.mkdir(exist_ok=True)
    all_seed_results: dict[int, list[dict]] = {}
    multi_seed = len(seeds) > 1

    for seed in seeds:
        cfg_s = copy.deepcopy(base_cfg)
        cfg_s["train"]["seed"] = seed
        suffix = f"_s{seed}" if multi_seed else ""
        run_name = f"identity_leakage_loso{suffix}"

        runs_dir = PROJECT_ROOT / "outputs" / "runs" / run_name
        csv_path = results_dir / f"{run_name}.csv"
        runs_dir.mkdir(parents=True, exist_ok=True)

        existing: dict[int, dict] = {}
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    fid = int(row["fold"])
                    existing[fid] = {
                        k: (int(v) if k in ("fold","test_subject","train_clips","val_clips","best_epoch")
                            else float(v) if v.replace(".","",1).lstrip("-").isdigit() else v)
                        for k, v in row.items()
                    }
            if existing:
                print(f"[✓] Seed {seed}: {len(existing)} fold sudah ada")

        for fold in range(1, 16):
            if fold in existing:
                continue
            result = run_fold(cfg_s, fold, device, amp_dtype,
                              fold_ckpt_path=runs_dir / f"fold_{fold:02d}_best.pt",
                              run_name=run_name)
            if result is None:
                continue
            existing[fold] = result
            all_r = sorted(existing.values(), key=lambda r: r["fold"])
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_r[0].keys()), extrasaction="ignore")
                w.writeheader()
                w.writerows(all_r)

        seed_results = sorted(existing.values(), key=lambda r: r["fold"])
        all_seed_results[seed] = seed_results
        if seed_results:
            print_summary(seed_results, cfg_s["dataset"]["class_names"], results_dir, run_name)

    # Flatten all seeds → list of fold results (for comparison)
    all_results = [r for rs in all_seed_results.values() for r in rs]
    if multi_seed and all_seed_results:
        import numpy as np
        print_multiseed_summary(all_seed_results, base_cfg["dataset"]["class_names"],
                                results_dir, "identity_leakage_loso")
    return all_results


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def print_comparison(loso_results: list[dict], random_results: list[dict], out_dir: Path):
    def stats(results, key):
        vals = [r[key] for r in results if key in r]
        if not vals:
            return "  —  ", "  —  "
        return f"{np.mean(vals):6.2f}%", f"±{np.std(vals):.2f}%"

    lines = [
        "=" * 72,
        "Identity Leakage Controlled Experiment — iMED-VST",
        "Kondisi A: LOSO (subjek disjoint)  |  Kondisi B: Random 60/20/20",
        "=" * 72,
        f"{'Condition':<26} {'n':>4}  {'WAR':>12}  {'UAR':>12}  {'MacroF1':>12}",
        "-" * 72,
    ]

    def row(name, results):
        n = len(results)
        war_m, war_s = stats(results, "best_WAR")
        uar_m, uar_s = stats(results, "best_UAR")
        mf1_m, mf1_s = stats(results, "best_MacroF1")
        return (f"  {name:<24} {n:>4}  "
                f"{war_m}{war_s}  {uar_m}{uar_s}  {mf1_m}{mf1_s}")

    if loso_results:
        lines.append(row("LOSO (disjoint)", loso_results))
    if random_results:
        lines.append(row("Random 60/20/20 (leaked)", random_results))

    if loso_results and random_results:
        loso_war = np.mean([r["best_WAR"] for r in loso_results])
        rand_war = np.mean([r["best_WAR"] for r in random_results])
        lines += [
            "-" * 72,
            f"  Δ WAR (random − LOSO) = {rand_war - loso_war:+.2f}%",
            "  Selisih ini adalah estimasi besarnya identity leakage bias",
            "  pada prior work yang menggunakan random split.",
        ]

    lines.append("=" * 72)
    text = "\n".join(lines)
    print(f"\n{text}\n")
    out_path = out_dir / "identity_leakage_comparison.txt"
    out_path.write_text(text)
    print(f"[✓] Comparison: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--condition", choices=["loso", "random", "both"], default="both")
    p.add_argument("--seeds",     type=int, nargs="+", default=None,
                   help="Seeds (default: 42 123 456)")
    p.add_argument("--loso_csv",  default=None,
                   help="Path ke CSV LOSO yang sudah ada (skip re-training kondisi A)")
    return p.parse_args()


def main():
    args   = parse_args()
    seeds  = args.seeds if args.seeds else RANDOM_SEEDS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)

    with open(BASE_DIR / args.config) as f:
        base_cfg = yaml.safe_load(f)

    # Konfigurasi IDENTIK dengan Step 3/4 main experiment (td_cls, 7-class)
    base_cfg["model"]["use_td"]    = True
    base_cfg["model"]["td_config"] = {
        "encoder_type": "gru",
        "num_layers":   1,
        "use_pos_enc":  False,
        "dim_ffn":      1536,
        "pool_type":    "mean",
    }
    base_cfg["model"]["td_variant"]   = "td_gru"
    base_cfg["dataset"]["emotion_set"] = "basic"
    base_cfg["dataset"]["num_classes"] = 7
    base_cfg["dataset"]["class_names"] = [
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
    ]

    results_dir = PROJECT_ROOT / "outputs" / "results"
    results_dir.mkdir(exist_ok=True)

    loso_results   = []
    random_results = []

    if args.condition in ("loso", "both"):
        print("\n[=] Kondisi A: LOSO (disjoint subjek)")
        loso_results = run_loso_condition(base_cfg, device, amp_dtype, seeds,
                                          loso_csv=args.loso_csv)

    if args.condition in ("random", "both"):
        print(f"\n[=] Kondisi B: Random 60/20/20 (seeds={seeds})")
        csv_path = results_dir / "identity_leakage_random.csv"
        existing: list[dict] = []
        if csv_path.exists():
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    existing.append({
                        k: (int(v) if k in ("seed", "train_clips", "val_clips", "test_clips", "best_epoch")
                            else float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v)
                        for k, v in row.items()
                    })
        done_seeds = {int(r["seed"]) for r in existing}

        for seed in seeds:
            if seed in done_seeds:
                print(f"  Seed {seed} [SKIP] sudah ada")
                continue
            res = run_random_single(base_cfg, device, amp_dtype, seed)
            if res:
                existing.append(res)
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(existing[0].keys()))
                    w.writeheader()
                    w.writerows(existing)
                print(f"[✓] CSV: {csv_path}")

        random_results = existing

    print_comparison(loso_results, random_results, results_dir)


if __name__ == "__main__":
    main()
