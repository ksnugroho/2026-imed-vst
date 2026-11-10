"""
metrics.py — iMED-VST
----------------------
Semua metrics FER: WAR, UAR, Macro-P/R/F1, Weighted-P/R/F1.

WAR (Weighted Average Recall) = overall accuracy
UAR (Unweighted Average Recall) = Macro-Recall = mean recall per kelas
Keduanya dilaporkan per fold dan Mean ± Std across 15 folds (LOSO).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class FERMetrics:
    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_preds  = []
        self.all_labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        preds = logits.argmax(dim=1).cpu().numpy()
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())

    def compute(self) -> dict:
        y_pred = np.array(self.all_preds)
        y_true = np.array(self.all_labels)
        n_cls  = len(self.class_names)
        labels = list(range(n_cls))

        war        = accuracy_score(y_true, y_pred)
        macro_p    = precision_score(y_true, y_pred, average="macro",    zero_division=0, labels=labels)
        uar        = recall_score   (y_true, y_pred, average="macro",    zero_division=0, labels=labels)
        macro_f1   = f1_score       (y_true, y_pred, average="macro",    zero_division=0, labels=labels)
        weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        weighted_r = recall_score   (y_true, y_pred, average="weighted", zero_division=0, labels=labels)
        weighted_f1= f1_score       (y_true, y_pred, average="weighted", zero_division=0, labels=labels)

        # Per-class recall dari confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        per_class_recall = {}
        for i, cls in enumerate(self.class_names):
            tp = cm[i, i]
            total = cm[i, :].sum()
            per_class_recall[cls] = round(float(tp / (total + 1e-9)) * 100, 2)

        return {
            "war":        round(war        * 100, 2),
            "uar":        round(uar        * 100, 2),
            "macro_p":    round(macro_p    * 100, 2),
            "macro_f1":   round(macro_f1   * 100, 2),
            "weighted_p": round(weighted_p * 100, 2),
            "weighted_r": round(weighted_r * 100, 2),
            "weighted_f1":round(weighted_f1* 100, 2),
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm.tolist(),
        }

    def summary(self) -> str:
        m = self.compute()
        lines = [
            f"  WAR={m['war']:.2f}%  UAR={m['uar']:.2f}%  "
            f"Macro-F1={m['macro_f1']:.2f}%  W-F1={m['weighted_f1']:.2f}%",
            "  Per-class recall:",
        ]
        for cls, r in m["per_class_recall"].items():
            marker = " ←" if r < 20.0 else ""
            lines.append(f"    {cls:25s}: {r:5.2f}%{marker}")
        return "\n".join(lines)

    def sklearn_report(self) -> str:
        return classification_report(
            self.all_labels, self.all_preds,
            target_names=self.class_names,
            zero_division=0,
        )
