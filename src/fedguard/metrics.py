"""Evaluation metrics (plan section 14.1). Macro-F1 is the main number; accuracy is never reported alone.

Multi-class metrics come from the 10-class predictions. Binary metrics (benign vs attack) are derived
from the same predictions: any attack class counts as "attack" (paper O5 asks for both views).
Class 0 is always Benign.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

BENIGN = 0


def classification_metrics(y_true, y_pred, classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.arange(len(classes))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    benign = y_true == BENIGN
    attack_true, attack_pred = ~benign, y_pred != BENIGN
    tp = int((attack_true & attack_pred).sum()); fn = int((attack_true & ~attack_pred).sum())
    fp = int((benign & attack_pred).sum());       tn = int((benign & ~attack_pred).sum())
    det_rate = tp / max(tp + fn, 1)
    bin_prec = tp / max(tp + fp, 1)

    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "benign_fpr": fp / max(fp + tn, 1),                    # normal traffic flagged as an attack
        "attack_detection_rate": det_rate,                     # binary recall: attack flagged as any attack
        "binary_f1": 2 * bin_prec * det_rate / max(bin_prec + det_rate, 1e-12),
        "per_class": {c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for c, p, r, f, s in zip(classes, prec, rec, f1, support)},
        "confusion_matrix": cm.tolist(),
    }


def honest_rejection_rate(weights, is_attacker):
    """Share of honest clients whose trust weight is below half of a fair share (0.5 / K)."""
    weights, is_attacker = np.asarray(weights, float), np.asarray(is_attacker, bool)
    honest = ~is_attacker
    if not honest.any():
        return 0.0
    return float(np.mean(weights[honest] < 0.5 / len(weights)))


def attacker_weight_share(weights, is_attacker):
    """Total weight the aggregator gave to attackers."""
    return float(np.asarray(weights, float)[np.asarray(is_attacker, bool)].sum())
