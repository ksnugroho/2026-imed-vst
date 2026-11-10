"""
run_step7_identity_leakage.py — Step 7: Identity Leakage Test
--------------------------------------------------------------
Label permutation test untuk membuktikan model belajar dari pola emosi,
bukan dari korelasi spurious (identitas subjek, bias dataset, dll.).

Methodology:
  - Config identik dengan baseline (swin3d_s, no-TD, seed=42)
  - Per fold: label training DIACAK secara random sebelum training
  - Val dan test label TIDAK diacak (tetap label asli)
  - Jika model genuinely belajar emosi:
      WAR permuted ≈ 14.3% (chance level = 100/7%)
  - Jika ada identity leakage:
      WAR permuted >> 14.3% (model bisa prediksi dari visual saja)

Komparasi:
  - Baseline seed=42 WAR : 59.84%
  - Permuted WAR         : diharapkan ~14.3%

Output:
  results/step7_permuted_s42.csv
  results/step7_permuted_s42_summary.txt
  results/step7_comparison.txt
"""

import copy
import csv
import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml

BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from run_loso import run_fold, print_summary, WANDB_ENTITY, WANDB_PROJECT
from train import set_seed


def main():
    # --- Config: identik dengan baseline step 1 ---
    with open(BASE_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["backbone"] = "swin3d_s"
    cfg["model"]["use_td"]   = False
    cfg["train"]["seed"]     = 42

    run_name = "step7_permuted_s42"

    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    results_dir  = PROJECT_ROOT / "outputs" / "results"
    results_dir.mkdir(exist_ok=True)
    ckpt_dir     = PROJECT_ROOT / "outputs" / "runs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Step 7 — Identity Leakage Test (Label Permutation)")
    print(f"  Config : swin3d_s, no-TD, seed=42")
    print(f"  Labels : PERMUTED per fold (train only)")
    print(f"  Expect : WAR ≈ 14.3% (random chance for 7 classes)")
    print(f"{'='*65}\n")

    # --- CSV header ---
    class_names = cfg["dataset"]["class_names"]
    fieldnames  = [
        "fold", "test_subject", "train_clips", "val_clips", "test_clips",
        "best_WAR", "best_UAR", "best_MacroP", "best_MacroF1",
        "best_WeightedP", "best_WeightedR", "best_WF1",
        "best_epoch", "val_UAR", "train_min",
        *[f"recall_{c}" for c in class_names],
    ]
    csv_path = results_dir / f"{run_name}.csv"
    csvfile  = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    results = []
    for fold in range(1, 16):
        fold_ckpt = ckpt_dir / f"fold_{fold:02d}_best.pt"
        result = run_fold(
            cfg                 = copy.deepcopy(cfg),
            fold                = fold,
            device              = device,
            amp_dtype           = amp_dtype,
            fold_ckpt_path      = fold_ckpt,
            run_name            = run_name,
            permute_train_labels= True,      # ← kunci Step 7
        )
        if result is None:
            continue
        writer.writerow(result)
        csvfile.flush()
        results.append(result)
        print(f"  → Fold {fold:2d} done: WAR={result['best_WAR']:.2f}%")

    csvfile.close()
    print(f"\n[✓] CSV: {csv_path}")

    print_summary(results, class_names, results_dir, run_name)

    # --- Comparison report ---
    wars_perm = [r["best_WAR"] for r in results]
    mean_perm = np.mean(wars_perm)
    std_perm  = np.std(wars_perm)

    # Load baseline seed=42 results
    baseline_wars = []
    baseline_path = results_dir / "step1_swin3d_s_7cls_s42.csv"
    try:
        with open(baseline_path) as f:
            for row in csv.DictReader(f):
                baseline_wars.append(float(row["best_WAR"]))
        mean_base = np.mean(baseline_wars)
    except FileNotFoundError:
        mean_base = None

    chance = 100.0 / 7

    lines = [
        "=" * 60,
        "Step 7 — Identity Leakage Test: Comparison",
        "=" * 60,
        "",
        f"Random chance (7 kelas)  : {chance:.2f}%",
        f"Baseline WAR (seed=42)   : {mean_base:.2f}%" if mean_base else "Baseline: N/A",
        f"Permuted WAR (seed=42)   : {mean_perm:.2f}% ± {std_perm:.2f}%",
        "",
        f"WAR per fold (permuted)  : {[f'{w:.1f}' for w in wars_perm]}",
        "",
    ]
    if mean_perm < chance * 1.5:
        lines.append(
            f"→ PASS: Permuted WAR ({mean_perm:.2f}%) mendekati chance level ({chance:.2f}%)."
        )
        lines.append(
            "  Tidak ada bukti identity leakage. Model belajar dari emosi, bukan bias spurious."
        )
    else:
        lines.append(
            f"→ PERINGATAN: Permuted WAR ({mean_perm:.2f}%) jauh di atas chance level ({chance:.2f}%)."
        )
        lines.append(
            "  Kemungkinan ada identity leakage atau bias spurious dalam dataset."
        )

    comparison_text = "\n".join(lines)
    print(f"\n{comparison_text}")

    comp_path = results_dir / "step7_comparison.txt"
    comp_path.write_text(comparison_text + "\n")
    print(f"\n[✓] Comparison: {comp_path}")


if __name__ == "__main__":
    main()
