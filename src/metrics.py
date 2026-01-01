from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(id2label))))
    m_f1 = float(f1_score(y_true, y_pred, average="macro"))
    bacc = float(balanced_accuracy_score(y_true, y_pred))

    # recall = diag / row_sum
    recalls: Dict[str, float] = {}
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        recalls[id2label[i]] = float(cm[i, i] / denom) if denom > 0 else 0.0

    mean_recall = float(np.mean(list(recalls.values()))) if recalls else 0.0

    return {
        "macro_f1": m_f1,
        "balanced_accuracy": bacc,
        "mean_recall": mean_recall,
        "per_class_recall": recalls,
        "confusion_matrix": cm.tolist(),
    }
