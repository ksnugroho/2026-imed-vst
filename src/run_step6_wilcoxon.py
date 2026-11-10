"""
run_step6_wilcoxon.py — Step 6: Wilcoxon Signed-Rank Test
----------------------------------------------------------
Membandingkan proposed vs baseline pada 7-class LOSO.

Methodology:
  - Paired comparison per (seed, fold) → 45 pairs (3 seeds × 15 folds)
  - Wilcoxon signed-rank test, one-tailed (H1: proposed > baseline)
  - α = 0.05
  - Juga report: mean diff, % folds where proposed > baseline

Usage:
  python run_step6_wilcoxon.py
  python run_step6_wilcoxon.py --baseline step1_swin3d_s_7cls --proposed step3_td_gru_7cls

Output: results/step6_wilcoxon.txt
"""

import argparse
import csv
import os
from pathlib import Path
from scipy.stats import wilcoxon, norm
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = str(PROJECT_ROOT / "outputs" / "results")
SEEDS   = [42, 123, 456]

def load_war_per_fold(prefix, seed):
    path = os.path.join(RESULTS, f"{prefix}_s{seed}.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [float(r["best_WAR"]) for r in rows]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="step1_swin3d_s_7cls")
    parser.add_argument("--proposed", default="step3_td_gru_7cls")
    args = parser.parse_args()

    baseline_all, proposed_all = [], []

    for seed in SEEDS:
        b = load_war_per_fold(args.baseline, seed)
        g = load_war_per_fold(args.proposed,  seed)
        assert len(b) == len(g) == 15, f"seed={seed}: fold count mismatch"
        baseline_all.extend(b)
        proposed_all.extend(g)

    baseline = np.array(baseline_all)
    proposed = np.array(proposed_all)
    diff     = proposed - baseline

    stat, p_two = wilcoxon(proposed, baseline, alternative="two-sided")
    _, p_one    = wilcoxon(proposed, baseline, alternative="greater")

    n         = len(diff)

    # Effect size r = |Z| / sqrt(N), Z derived from two-sided p
    p_safe = float(np.clip(p_two, 1e-300, 1 - 1e-15))
    z_score = norm.ppf(1 - p_safe / 2)
    r_effect = float(min(abs(z_score) / np.sqrt(n), 1.0))
    effect_label = ("large" if r_effect >= 0.5 else
                    "medium" if r_effect >= 0.3 else
                    "small" if r_effect >= 0.1 else "negligible")
    mean_base = baseline.mean()
    mean_prop = proposed.mean()
    mean_diff = diff.mean()
    std_diff  = diff.std()
    n_better  = (diff > 0).sum()
    n_worse   = (diff < 0).sum()
    n_equal   = (diff == 0).sum()

    lines = [
        "=" * 60,
        "Step 6 — Wilcoxon Signed-Rank Test",
        f"  Baseline  : {args.baseline}",
        f"  Proposed  : {args.proposed}",
        "  Setup     : LOSO 7-class, 3 seeds × 15 folds = 45 pairs",
        "=" * 60,
        "",
        f"Mean WAR (baseline) : {mean_base:.2f}%",
        f"Mean WAR (proposed) : {mean_prop:.2f}%",
        f"Mean difference     : {mean_diff:+.2f}% ± {std_diff:.2f}%",
        "",
        f"proposed > baseline : {n_better}/{n} folds ({100*n_better/n:.1f}%)",
        f"proposed < baseline : {n_worse}/{n} folds ({100*n_worse/n:.1f}%)",
        f"tie                 : {n_equal}/{n} folds",
        "",
        f"Wilcoxon statistic  : {stat:.4f}",
        f"p-value (two-sided) : {p_two:.4f}",
        f"p-value (one-sided, proposed > baseline) : {p_one:.4f}",
        f"Effect size r       : {r_effect:.3f} ({effect_label})",
        "",
    ]

    alpha = 0.05
    if p_one < alpha:
        lines.append(f"→ SIGNIFICANT (p={p_one:.4f} < α={alpha}): proposed significantly outperforms baseline.")
    else:
        lines.append(f"→ NOT significant (p={p_one:.4f} ≥ α={alpha}): proposed does not significantly outperform baseline.")

    lines += ["", "Per-seed breakdown:"]
    for seed in SEEDS:
        b = load_war_per_fold(args.baseline, seed)
        g = load_war_per_fold(args.proposed,  seed)
        d = np.array(g) - np.array(b)
        lines.append(f"  seed={seed}: baseline={np.mean(b):.2f}%  proposed={np.mean(g):.2f}%  "
                     f"diff={d.mean():+.2f}%  better={(d>0).sum()}/15")

    out = "\n".join(lines)
    print(out)

    outpath = os.path.join(RESULTS, "step6_wilcoxon.txt")
    with open(outpath, "w") as f:
        f.write(out + "\n")
    print(f"\nSaved → {outpath}")

if __name__ == "__main__":
    main()
