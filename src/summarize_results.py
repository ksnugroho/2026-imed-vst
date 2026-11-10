"""
summarize_results.py — iMED-VST
--------------------------------
Agregasi semua hasil eksperimen V2 ke satu CSV.

Output: outputs/all_results_summary.csv

Kolom: step, experiment, WAR_mean, WAR_std, UAR_mean, UAR_std,
        MacroF1_mean, MacroF1_std, WF1_mean, WF1_std, notes

Cara pakai:
  cd <project_root>
  python src/summarize_results.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "outputs" / "results"
OUTPUT_CSV   = PROJECT_ROOT / "outputs" / "all_results_summary.csv"

FIELDNAMES = [
    "step", "experiment",
    "WAR_mean", "WAR_std",
    "UAR_mean", "UAR_std",
    "MacroF1_mean", "MacroF1_std",
    "WF1_mean", "WF1_std",
    "notes",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val: str) -> str:
    """Convert percentage string to decimal ratio with 4 decimal places (64.13 → 0.6413)."""
    if val == "":
        return ""
    try:
        return f"{float(val) / 100:.4f}"
    except ValueError:
        return val


def _parse_mean_std(line: str) -> tuple[str, str]:
    """'64.13% ± 16.65%' → ('64.1300', '16.6500')"""
    m = re.search(r"([\d.]+)%\s*±\s*([\d.]+)%", line)
    if not m:
        return ("", "")
    return _fmt(m.group(1)), _fmt(m.group(2))


def _parse_agg_summary(path: Path) -> dict | None:
    """
    Parse *_agg_summary.txt.
    Returns dict with WAR/UAR/MacroF1/WF1 mean & std, or None if unreadable.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None

    result = {}
    for line in text.splitlines():
        if "Mean WAR" in line:
            result["WAR_mean"], result["WAR_std"] = _parse_mean_std(line)
        elif "Mean UAR" in line:
            result["UAR_mean"], result["UAR_std"] = _parse_mean_std(line)
        elif "Mean Macro-F1" in line:
            result["MacroF1_mean"], result["MacroF1_std"] = _parse_mean_std(line)
        elif "Mean W-F1" in line:
            result["WF1_mean"], result["WF1_std"] = _parse_mean_std(line)

    if not result.get("WAR_mean"):
        return None
    return result


# ---------------------------------------------------------------------------
# Step 0 — Temporal window ablation (ablation_temporal_agg.csv)
# ---------------------------------------------------------------------------

def collect_step0(rows: list) -> None:
    path = RESULTS_DIR / "ablation_temporal_agg.csv"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return

    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "step":        "step0_temporal",
                "experiment":  f"ablation_{r['config_id']}_{r['description'].split()[0]}",
                "WAR_mean":    _fmt(r["best_WAR"]),
                "WAR_std":     _fmt(r["best_WAR_std"]),
                "UAR_mean":    _fmt(r["best_UAR"]),
                "UAR_std":     _fmt(r["best_UAR_std"]),
                "MacroF1_mean": _fmt(r["best_MacroF1"]),
                "MacroF1_std": _fmt(r["best_MacroF1_std"]),
                "WF1_mean":    _fmt(r["best_WF1"]),
                "WF1_std":     _fmt(r["best_WF1_std"]),
                "notes":       r["description"],
            })
    print(f"  [OK] step0: {len([r for r in rows if r['step']=='step0_temporal'])} configs")


# ---------------------------------------------------------------------------
# Single-seed _summary.txt (exp_seqlen, step5b, etc.)
# ---------------------------------------------------------------------------

def collect_single_seed_summaries(rows: list) -> None:
    added = 0
    for key, (step_label, notes) in SINGLE_SEED_SUMMARY_MAP.items():
        path = RESULTS_DIR / f"{key}_summary.txt"
        data = _parse_agg_summary(path)
        if data is None:
            print(f"  [WARN] Could not parse {path.name}")
            continue
        rows.append({
            "step":         step_label,
            "experiment":   key,
            "WAR_mean":     data.get("WAR_mean", ""),
            "WAR_std":      data.get("WAR_std", ""),
            "UAR_mean":     data.get("UAR_mean", ""),
            "UAR_std":      data.get("UAR_std", ""),
            "MacroF1_mean": data.get("MacroF1_mean", ""),
            "MacroF1_std":  data.get("MacroF1_std", ""),
            "WF1_mean":     data.get("WF1_mean", ""),
            "WF1_std":      data.get("WF1_std", ""),
            "notes":        notes,
        })
        added += 1
    print(f"  [OK] single-seed summaries: {added} experiments")


# ---------------------------------------------------------------------------
# Inference timing (inference_timing.txt)
# ---------------------------------------------------------------------------

def collect_inference_timing(rows: list) -> None:
    path = RESULTS_DIR / "inference_timing.txt"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return

    text = path.read_text()
    # Pattern: "swin3d_s  td_gru (proposed)   52.2  49.5  44.23  0.05  22.6"
    pattern = re.compile(
        r"^\s*(swin3d_\S+\s+\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )
    added = 0
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name, params, backbone_p, ms, ms_std, clips_s = m.groups()
        name = name.strip()
        rows.append({
            "step":         "inference_timing",
            "experiment":   name.replace(" ", "_").replace("(", "").replace(")", ""),
            "WAR_mean":     "",
            "WAR_std":      "",
            "UAR_mean":     "",
            "UAR_std":      "",
            "MacroF1_mean": "",
            "MacroF1_std":  "",
            "WF1_mean":     "",
            "WF1_std":      "",
            "notes":        f"params={params}M ms/clip={ms}±{ms_std} clips/s={clips_s}",
        })
        added += 1
    print(f"  [OK] inference timing: {added} models")


# ---------------------------------------------------------------------------
# Steps 1–4 and misc — *_agg_summary.txt
# ---------------------------------------------------------------------------

STEP_MAP = {
    "step1_swin3d_t_7cls":           ("step1_backbone", "swin3d_t 7cls"),
    "step1_swin3d_s_7cls":           ("step1_backbone", "swin3d_s 7cls"),
    "step2_focal_7cls":              ("step2_loss",     "focal loss 7cls"),
    "step3_td_gru_7cls":             ("step3_td",       "td_gru 7cls"),
    "step3_td_gru_freeze15_7cls":    ("step3_td",       "td_gru freeze15 7cls"),
    "step4a_twostage_19cls":         ("step4a_19cls",   "two-stage 19cls"),
    "step4b_direct_notd_19cls":      ("step4b_19cls",   "direct no-TD 19cls"),
    "step4b_direct_td_gru_19cls":    ("step4b_19cls",   "direct td_gru 19cls"),
    "step5_td_full_ext":             ("exp_td_full_ext",   "td_full direct fine-tune multi-seed"),
    "exp_frozen_backbone_td_gru":    ("exp_frozen",     "frozen backbone td_gru"),
}

# Experiment yang punya _summary.txt (single-seed, bukan agg) — dikumpulkan terpisah
SINGLE_SEED_SUMMARY_MAP = {
    "exp_seqlen_D16_td_gru":        ("exp_seqlen",     "seqlen D=16 td_gru"),
    "exp_seqlen_D4_td_gru":         ("exp_seqlen",     "seqlen D=4 td_gru"),
    "step5b_td_gru_attn_freeze15":  ("step5b",         "td_gru_attn freeze15 single-seed"),
}


def collect_agg_summaries(rows: list) -> None:
    files = sorted(RESULTS_DIR.glob("*_agg_summary.txt"))
    added = 0
    for path in files:
        # Derive key from filename: strip _agg_summary.txt
        key = path.name.replace("_agg_summary.txt", "")
        step_label, notes = STEP_MAP.get(key, (f"other_{key}", ""))

        data = _parse_agg_summary(path)
        if data is None:
            print(f"  [WARN] Could not parse {path.name}")
            continue

        rows.append({
            "step":         step_label,
            "experiment":   key,
            "WAR_mean":     data.get("WAR_mean", ""),
            "WAR_std":      data.get("WAR_std", ""),
            "UAR_mean":     data.get("UAR_mean", ""),
            "UAR_std":      data.get("UAR_std", ""),
            "MacroF1_mean": data.get("MacroF1_mean", ""),
            "MacroF1_std":  data.get("MacroF1_std", ""),
            "WF1_mean":     data.get("WF1_mean", ""),
            "WF1_std":      data.get("WF1_std", ""),
            "notes":        notes,
        })
        added += 1

    print(f"  [OK] agg_summary files: {added} experiments")


# ---------------------------------------------------------------------------
# Step 5 — TD ablation comparison table (step5_td_ablation_comparison.txt)
# ---------------------------------------------------------------------------

def collect_step5_comparison(rows: list) -> None:
    path = RESULTS_DIR / "step5_td_ablation_comparison.txt"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return

    text = path.read_text()
    # Pattern: "  td_base    61.59%   60.00%  51.17%  52.52%  <config desc>"
    pattern = re.compile(
        r"^\s+(\S+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+(.+)$"
    )
    added = 0
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        variant, war, uar, mf1, wf1, config_desc = m.groups()
        # Strip trailing notes (← ...)
        config_desc = re.sub(r"\s*←.*$", "", config_desc).strip()
        rows.append({
            "step":         "step5_td_ablation",
            "experiment":   f"step5_{variant}",
            "WAR_mean":     _fmt(war),
            "WAR_std":      "",
            "UAR_mean":     _fmt(uar),
            "UAR_std":      "",
            "MacroF1_mean": _fmt(mf1),
            "MacroF1_std":  "",
            "WF1_mean":     _fmt(wf1),
            "WF1_std":      "",
            "notes":        config_desc,
        })
        added += 1

    print(f"  [OK] step5 comparison: {added} variants")


# ---------------------------------------------------------------------------
# Step 6 — Wilcoxon (step6_wilcoxon.txt)
# ---------------------------------------------------------------------------

def collect_step6_wilcoxon(rows: list) -> None:
    path = RESULTS_DIR / "step6_wilcoxon.txt"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return

    text = path.read_text()

    war_base = war_gru = p_one = r_effect = verdict = ""

    for line in text.splitlines():
        if "Mean WAR (baseline)" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                war_base = m.group(1)
        elif "Mean WAR (proposed)" in line or "Mean WAR (td_gru)" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                war_gru = m.group(1)
        elif "p-value (one-sided" in line:
            m = re.search(r":\s*([\d.]+)", line)
            if m:
                p_one = m.group(1)
        elif "Effect size r" in line:
            m = re.search(r":\s*([\d.]+)\s*\(([^)]+)\)", line)
            if m:
                r_effect = f"{m.group(1)} ({m.group(2)})"
        elif line.startswith("→"):
            verdict = line.lstrip("→ ").strip()

    notes = f"p_one={p_one}; r={r_effect}; {verdict}"
    rows.append({
        "step":         "step6_wilcoxon",
        "experiment":   "step6_wilcoxon",
        "WAR_mean":     _fmt(war_gru),
        "WAR_std":      "",
        "UAR_mean":     "",
        "UAR_std":      "",
        "MacroF1_mean": "",
        "MacroF1_std":  "",
        "WF1_mean":     "",
        "WF1_std":      "",
        "notes":        notes,
    })
    rows.append({
        "step":         "step6_wilcoxon",
        "experiment":   "step6_baseline",
        "WAR_mean":     _fmt(war_base),
        "WAR_std":      "",
        "UAR_mean":     "",
        "UAR_std":      "",
        "MacroF1_mean": "",
        "MacroF1_std":  "",
        "WF1_mean":     "",
        "WF1_std":      "",
        "notes":        "swin3d_s no-TD baseline",
    })
    print(f"  [OK] step6 wilcoxon (p={p_one})")


# ---------------------------------------------------------------------------
# Step 7 — Identity leakage (identity_leakage_comparison.txt)
# ---------------------------------------------------------------------------

def collect_step7_leakage(rows: list) -> None:
    path = RESULTS_DIR / "identity_leakage_comparison.txt"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return

    text = path.read_text()

    # Lines like: "  LOSO (disjoint)   15   65.71%±21.45%   63.81%±20.78%   56.32%±24.02%"
    pattern = re.compile(
        r"^\s+(.+?)\s{2,}(\d+)\s+"
        r"([\d.]+)%±([\d.]+)%\s+"
        r"([\d.]+)%±([\d.]+)%\s+"
        r"([\d.]+)%±([\d.]+)%"
    )
    added = 0
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        condition, n, war_m, war_s, uar_m, uar_s, mf1_m, mf1_s = m.groups()
        label = "LOSO" if "LOSO" in condition else "random_split"
        rows.append({
            "step":         "step7_identity_leakage",
            "experiment":   f"step7_{label}",
            "WAR_mean":     _fmt(war_m),
            "WAR_std":      _fmt(war_s),
            "UAR_mean":     _fmt(uar_m),
            "UAR_std":      _fmt(uar_s),
            "MacroF1_mean": _fmt(mf1_m),
            "MacroF1_std":  _fmt(mf1_s),
            "WF1_mean":     "",
            "WF1_std":      "",
            "notes":        condition.strip(),
        })
        added += 1

    print(f"  [OK] step7 identity leakage: {added} conditions")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows: list[dict] = []

    print("Parsing results...")
    collect_step0(rows)
    collect_agg_summaries(rows)
    collect_single_seed_summaries(rows)
    collect_step5_comparison(rows)
    collect_step6_wilcoxon(rows)
    collect_step7_leakage(rows)
    collect_inference_timing(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal rows: {len(rows)}")
    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
