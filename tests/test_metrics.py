import numpy as np
import pytest
from sklearn.metrics import f1_score

from fedguard.metrics import attacker_weight_share, classification_metrics, honest_rejection_rate

CLASSES = ["Benign", "DoS", "Worms"]


def test_perfect_predictions():
    y = np.array([0, 0, 1, 2, 1])
    m = classification_metrics(y, y, CLASSES)
    assert m["macro_f1"] == m["accuracy"] == m["attack_detection_rate"] == 1.0
    assert m["benign_fpr"] == 0.0


def test_lazy_all_benign_model_has_high_accuracy_but_low_macro_f1():
    y = np.array([0] * 96 + [1] * 3 + [2])                       # 96% benign, like our data
    m = classification_metrics(y, np.zeros_like(y), CLASSES)
    assert m["accuracy"] == pytest.approx(0.96)
    assert m["macro_f1"] < 0.34                                  # the accuracy trap from paper ref [22]
    assert m["attack_detection_rate"] == 0.0


def test_values_match_sklearn_and_binary_view():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 0, 1, 2, 2, 0])
    m = classification_metrics(y, p, CLASSES)
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["benign_fpr"] == pytest.approx(1 / 4)               # one benign row flagged as DoS
    assert m["attack_detection_rate"] == pytest.approx(3 / 4)    # DoS->Worms still counts as "attack"
    assert np.array(m["confusion_matrix"]).sum() == len(y)
    assert m["per_class"]["Worms"]["support"] == 2


def test_missing_class_in_predictions_does_not_crash():
    m = classification_metrics(np.array([0, 1, 2]), np.array([0, 0, 0]), CLASSES)
    assert m["per_class"]["Worms"]["recall"] == 0.0


def test_hrr_and_attacker_share():
    w = np.array([0.30, 0.30, 0.02, 0.38, 0.0])
    att = np.array([False, False, False, False, True])
    assert honest_rejection_rate(w, att) == pytest.approx(1 / 4)  # 0.02 < 0.5/5
    assert attacker_weight_share(w, att) == 0.0
