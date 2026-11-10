"""
stat_significance.py — iMED-VST
---------------------------------
Wilcoxon signed-rank test untuk membandingkan performa antar LOSO variant.

Digunakan untuk menjawab reviewer: "apakah perbedaan WAR/UAR antar variant
secara statistik signifikan?"

Cara pakai:
  python stat_significance.py                         # default: td_base vs td_cls
  python stat_significance.py --a td_base --b td_gru  # pilih variant
  python stat_significance.py --all_pairs              # semua pasangan dari VARIANT_ORDER

Output:
  Statistik (W-statistic, p-value, effect size r) per metrik.
  Interpretasi otomatis (significant / not significant).

Referensi:
  Wilcoxon, F. (1945). Individual comparisons by ranking methods.
  r = Z / sqrt(N), N = jumlah fold (15).
  r: 0.1=small, 0.3=medium, 0.5=large (Cohen, 1988).
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

BASE_DIR     = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent


def load_fold_metrics(csv_path: Path) -> dict[str, list[float]]:
    """Baca CSV hasil fold, return dict metrik → list per-fold."""
    rows = list(csv.DictReader(open(csv_path)))
    metrics: dict[str, list[float]] = {}
    for key in ("best_WAR", "best_UAR", "best_MacroF1", "best_WF1"):
        try:
            metrics[key] = [float(r[key]) for r in rows]
        except (KeyError, ValueError):
            pass
    return metrics


def wilcoxon_test(a: list[float], b: list[float]) -> dict:
    """Wilcoxon signed-rank test dengan effect size r = Z/sqrt(N)."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("[ERROR] scipy diperlukan: conda run -n ksn_cv_task pip install scipy")
        sys.exit(1)

    diffs = [x - y for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]

    if len(nonzero) == 0:
        return {"W": 0.0, "p": 1.0, "r": 0.0, "N": len(a)}

    stat, p = wilcoxon(a, b, alternative="two-sided")
    N = len(a)
    # Approximate Z from p-value for effect size calculation
    # Clamp p away from 0/1 to prevent norm.ppf returning ±inf
    from scipy.stats import norm
    p_safe = float(np.clip(p, 1e-300, 1 - 1e-15))
    z = norm.ppf(1 - p_safe / 2)
    r = min(abs(z) / np.sqrt(N), 1.0)  # r ∈ [0, 1]
    return {"W": stat, "p": p, "r": r, "N": N}


def interpret(p: float, r: float) -> str:
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    if r >= 0.5:
        size = "large"
    elif r >= 0.3:
        size = "medium"
    elif r >= 0.1:
        size = "small"
    else:
        size = "negligible"
    return f"{sig} (p={p:.4f}, r={r:.3f} [{size} effect])"


def compare_pair(name_a: str, name_b: str, results_dir: Path, verbose: bool = True):
    """Bandingkan dua variant, cetak hasil."""

    def find_csv(name: str) -> Path | None:
        # Cari di step5_ prefix atau langsung nama run
        for prefix in (f"step5_{name}", f"loso_{name}", name):
            p = results_dir / f"{prefix}.csv"
            if p.exists():
                return p
        return None

    path_a = find_csv(name_a)
    path_b = find_csv(name_b)

    if path_a is None:
        print(f"[WARN] CSV tidak ditemukan untuk: {name_a}")
        return None
    if path_b is None:
        print(f"[WARN] CSV tidak ditemukan untuk: {name_b}")
        return None

    metrics_a = load_fold_metrics(path_a)
    metrics_b = load_fold_metrics(path_b)

    shared_metrics = [m for m in metrics_a if m in metrics_b]
    if not shared_metrics:
        print(f"[WARN] Tidak ada metrik bersama antara {name_a} dan {name_b}")
        return None

    na = len(metrics_a[shared_metrics[0]])
    nb = len(metrics_b[shared_metrics[0]])
    if na != nb:
        print(f"[WARN] Jumlah fold berbeda: {name_a}={na}, {name_b}={nb}")
        return None

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Wilcoxon Signed-Rank Test  |  n={na} fold")
        print(f"  A: {name_a}  vs  B: {name_b}")
        print(f"  CSV_A: {path_a.name}")
        print(f"  CSV_B: {path_b.name}")
        print(f"{'='*65}")
        header = f"  {'Metric':<14} {'Mean_A':>8} {'Mean_B':>8} {'Δ':>8}  Result"
        print(header)
        print(f"  {'-'*63}")

    summary = {}
    for metric in shared_metrics:
        a_vals = metrics_a[metric]
        b_vals = metrics_b[metric]
        res = wilcoxon_test(a_vals, b_vals)
        label = metric.replace("best_", "")
        delta = np.mean(b_vals) - np.mean(a_vals)
        interp = interpret(res["p"], res["r"])
        if verbose:
            print(f"  {label:<14} {np.mean(a_vals):8.3f} {np.mean(b_vals):8.3f} "
                  f"{delta:+8.3f}  {interp}")
        summary[metric] = {**res, "mean_a": np.mean(a_vals), "mean_b": np.mean(b_vals),
                           "delta": delta, "significant": res["p"] < 0.05}

    if verbose:
        print(f"{'='*65}\n")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Wilcoxon signed-rank test antar LOSO variant")
    p.add_argument("--a", default="td_base", help="Variant A (baseline)")
    p.add_argument("--b", default="td_cls",  help="Variant B (proposed)")
    p.add_argument("--results_dir", default="results", help="Folder CSV (relatif dari outputs/)")
    p.add_argument("--all_pairs", action="store_true",
                   help="Uji semua pasangan dari VARIANT_ORDER")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = PROJECT_ROOT / "outputs" / args.results_dir

    # Import VARIANT_ORDER dari run_loso
    sys.path.insert(0, str(BASE_DIR))
    try:
        from run_step5_td_ablation import VARIANT_ORDER
    except ImportError:
        VARIANT_ORDER = ["td_base", "td_pe", "td_2l", "td_4x", "td_cls",
                         "td_full", "td_pe_cls", "td_maxpool", "td_gru"]

    if args.all_pairs:
        print("\n[Wilcoxon] Menguji semua pasangan variant dari VARIANT_ORDER\n")
        pairs = list(combinations(VARIANT_ORDER, 2))
        sig_pairs = []
        for a, b in pairs:
            res = compare_pair(a, b, results_dir, verbose=False)
            if res is None:
                continue
            war_res = res.get("best_WAR", {})
            sig = war_res.get("significant", False)
            delta = war_res.get("delta", 0.0)
            p_val = war_res.get("p", 1.0)
            r_val = war_res.get("r", 0.0)
            marker = "* " if sig else "  "
            print(f"{marker}{a:<14} vs {b:<14}  "
                  f"ΔWAR={delta:+.3f}%  p={p_val:.4f}  r={r_val:.3f}")
            if sig:
                sig_pairs.append((a, b, delta, p_val, r_val))

        print(f"\n[Summary] {len(sig_pairs)}/{len(pairs)} pasangan signifikan (WAR, p<0.05):")
        for a, b, d, p, r in sig_pairs:
            print(f"  {a} vs {b}  Δ={d:+.3f}%  p={p:.4f}  r={r:.3f}")
    else:
        compare_pair(args.a, args.b, results_dir, verbose=True)


if __name__ == "__main__":
    main()
