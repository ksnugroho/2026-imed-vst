"""
run_step5_td_ablation.py — iMED-VST
--------------------------------------
Step 5: Ablasi arsitektur TD (Temporal Depth Enhancement) module.

Tujuan: Identifikasi masalah arsitektur TD pada Step 3 dan cari konfigurasi optimal.

Masalah yang diinvestigasi:
  1. Tidak ada positional encoding → permutation-invariant
  2. Hanya 1 layer TransformerEncoder → terlalu dangkal
  3. FFN 1536 = 2× feat_dim → lebih kecil dari standar (4×=3072)
  4. Mean pool → tidak ada pembobotan temporal

Variants:
  td_base  1L | no PE | ffn=1536 | mean   → Step 3 baseline (ulang)
  td_pe    1L | PE    | ffn=1536 | mean   → fix permutation issue
  td_2l    2L | no PE | ffn=1536 | mean   → lebih dalam
  td_4x    1L | no PE | ffn=3072 | mean   → FFN lebih besar
  td_cls   1L | no PE | ffn=1536 | CLS    → temporal weighting
  td_full  2L | PE    | ffn=3072 | CLS    → semua perbaikan

Setup: swin3d_s + CE loss + 7-class (sama dengan Step 1-3 agar comparable)

Cara pakai:
  Jalankan semua variant:
    conda run -n ksn_cv_task python run_step5_td_ablation.py

  Jalankan satu variant saja:
    conda run -n ksn_cv_task python run_step5_td_ablation.py --variant td_pe

  Lanjutkan dari variant tertentu:
    conda run -n ksn_cv_task python run_step5_td_ablation.py --start_variant td_4x

Output:
  results/step5_<variant>.csv
  results/step5_<variant>_summary.txt
  runs/step5_<variant>/fold_XX_best.pt
  results/step5_td_ablation_comparison.txt   (ringkasan semua variant)
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

from run_loso import run_fold, print_summary, TD_VARIANTS


WANDB_ENTITY  = "bdsrc-team"
WANDB_PROJECT = "2026-imed-vst-v2"

# Urutan variant — dari paling sederhana ke paling kompleks
VARIANT_ORDER = ["td_base", "td_pe", "td_2l", "td_4x", "td_cls", "td_full", "td_pe_cls",
                 "td_maxpool", "td_gru"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--variant",       default=None,
                   choices=list(TD_VARIANTS),
                   help="Jalankan satu variant saja")
    p.add_argument("--start_variant", default=None,
                   choices=list(TD_VARIANTS),
                   help="Mulai dari variant ini (skip sebelumnya)")
    p.add_argument("--fold",          type=int, default=None,
                   help="Debug: jalankan hanya fold ini")
    return p.parse_args()


def run_variant(
    variant_name: str,
    td_config: dict,
    base_cfg: dict,
    device: torch.device,
    amp_dtype: torch.dtype,
    fold_arg: int | None = None,
) -> list[dict]:
    """Jalankan semua fold untuk satu TD variant."""
    cfg = copy.deepcopy(base_cfg)

    # Override config untuk Step 5
    cfg["model"]["backbone"]   = "swin3d_s"
    cfg["model"]["pretrained"] = True
    cfg["model"]["use_td"]     = True
    cfg["model"]["td_config"]  = td_config
    cfg["model"]["td_variant"] = variant_name
    cfg["train"]["loss_type"]  = "ce"
    cfg["dataset"]["emotion_set"] = "basic"
    cfg["dataset"]["num_classes"] = 7
    cfg["dataset"]["class_names"] = [
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
    ]

    run_name    = f"step5_{variant_name}"
    results_dir = PROJECT_ROOT / "outputs" / "results"
    runs_dir    = PROJECT_ROOT / "outputs" / "runs" / run_name
    results_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{run_name}.csv"

    # Baca fold yang sudah ada
    existing: dict[int, dict] = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                fid = int(row["fold"])
                existing[fid] = {
                    k: (int(v)   if k in ("fold","test_subject","train_clips","val_clips","best_epoch")
                        else float(v) if v.replace(".","",1).lstrip("-").isdigit() else v)
                    for k, v in row.items()
                }
        if existing:
            print(f"    [✓] Melanjutkan ({len(existing)} fold sudah ada)")

    enc = td_config.get("encoder_type", "transformer")
    td_str = (f"enc={enc} "
              f"L={td_config.get('num_layers',1)} "
              f"PE={'Y' if td_config.get('use_pos_enc',False) else 'N'} "
              f"ffn={td_config.get('dim_ffn',1536)} "
              f"pool={td_config.get('pool_type','mean')}")
    print(f"\n{'='*65}")
    print(f"  STEP 5 | {variant_name} | {td_str}")
    print(f"  run: {run_name}")
    print(f"{'='*65}")

    fold_range = [fold_arg] if fold_arg else range(1, 16)

    for fold in fold_range:
        if fold in existing:
            r = existing[fold]
            print(f"  Fold {fold:2d} [SKIP] WAR={r['best_WAR']:.2f}% UAR={r['best_UAR']:.2f}%")
            continue

        fold_ckpt_path = runs_dir / f"fold_{fold:02d}_best.pt"
        print(f"\n--- Fold {fold:2d} / 15 ---")

        result = run_fold(
            cfg, fold, device, amp_dtype,
            fold_ckpt_path=fold_ckpt_path,
            init_checkpoint_dir=None,
            run_name=run_name,
        )
        if result is None:
            continue

        existing[fold] = result

        all_r = sorted(existing.values(), key=lambda r: r["fold"])
        fieldnames = list(all_r[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_r)

    all_results = sorted(existing.values(), key=lambda r: r["fold"])
    if all_results:
        print_summary(all_results, cfg["dataset"]["class_names"], results_dir, run_name)
    return all_results


def print_comparison(all_variant_results: dict[str, list[dict]], out_dir: Path):
    """Cetak tabel perbandingan semua variant."""
    lines = [
        "=" * 80,
        "Step 5 — TD Ablation Comparison",
        f"{'Variant':<12} {'WAR':>8} {'UAR':>8} {'MacroF1':>10} {'W-F1':>8}  Config",
        "-" * 80,
    ]

    variant_desc = {
        "td_base":    "transformer 1L no-PE ffn=1536 mean  ← Step3 baseline",
        "td_pe":      "transformer 1L +PE  ffn=1536 mean",
        "td_2l":      "transformer 2L no-PE ffn=1536 mean",
        "td_4x":      "transformer 1L no-PE ffn=3072 mean",
        "td_cls":     "transformer 1L no-PE ffn=1536 CLS",
        "td_full":    "transformer 2L +PE  ffn=3072 CLS   ← all improvements",
        "td_pe_cls":  "transformer 1L +PE  ffn=1536 CLS",
        "td_maxpool": "no-encoder  max pool  ← baseline temporal aggregation",
        "td_gru":     "BiGRU 1L    ← recurrent alternative",
    }

    for vname in VARIANT_ORDER:
        results = all_variant_results.get(vname)
        if not results:
            lines.append(f"  {vname:<12} {'—':>8} {'—':>8} {'—':>10} {'—':>8}  (tidak ada data)")
            continue
        wars  = [r["best_WAR"]     for r in results]
        uars  = [r["best_UAR"]     for r in results]
        mf1s  = [r["best_MacroF1"] for r in results]
        wf1s  = [r["best_WF1"]     for r in results]
        desc  = variant_desc.get(vname, "")
        lines.append(
            f"  {vname:<12} {np.mean(wars):7.2f}% {np.mean(uars):7.2f}% "
            f"{np.mean(mf1s):9.2f}% {np.mean(wf1s):7.2f}%  {desc}"
        )

    lines.append("=" * 80)
    text = "\n".join(lines)
    print(f"\n{text}\n")

    comp_path = out_dir / "step5_td_ablation_comparison.txt"
    comp_path.write_text(text)
    print(f"[✓] Comparison: {comp_path}")


def main():
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)

    with open(BASE_DIR / args.config) as f:
        base_cfg = yaml.safe_load(f)

    print(f"\n[✓] Device : {torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'}")
    print(f"[✓] AMP    : {'bf16' if amp_dtype == torch.bfloat16 else 'fp16'}")

    if args.variant:
        variants_to_run = [args.variant]
    elif args.start_variant:
        idx = VARIANT_ORDER.index(args.start_variant)
        variants_to_run = VARIANT_ORDER[idx:]
    else:
        variants_to_run = VARIANT_ORDER

    print(f"[✓] Variants: {variants_to_run}\n")

    results_dir = PROJECT_ROOT / "outputs" / "results"
    all_variant_results: dict[str, list[dict]] = {}

    # Muat hasil yang sudah ada dari variant yang tidak dijalankan
    for vname in VARIANT_ORDER:
        csv_path = results_dir / f"step5_{vname}.csv"
        if csv_path.exists() and vname not in variants_to_run:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            all_variant_results[vname] = [
                {k: (int(v)   if k in ("fold","test_subject","train_clips","val_clips","best_epoch")
                     else float(v) if v.replace(".","",1).lstrip("-").isdigit() else v)
                 for k, v in row.items()}
                for row in rows
            ]

    for vname in variants_to_run:
        t0 = time.time()
        results = run_variant(
            variant_name = vname,
            td_config    = TD_VARIANTS[vname],
            base_cfg     = base_cfg,
            device       = device,
            amp_dtype    = amp_dtype,
            fold_arg     = args.fold,
        )
        all_variant_results[vname] = results
        elapsed = round((time.time() - t0) / 60, 1)
        print(f"\n[✓] {vname} selesai dalam {elapsed} menit\n")

    if not args.fold:
        print_comparison(all_variant_results, results_dir)


if __name__ == "__main__":
    main()
