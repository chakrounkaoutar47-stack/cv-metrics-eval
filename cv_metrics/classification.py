"""Métriques de classification d'images.

Convention : les classes sont des entiers 0..K-1.
"""

from __future__ import annotations

import numpy as np


def _safe_div(num, den):
    """Division qui renvoie 0 quand le dénominateur est nul (ex. aucune prédiction d'une classe)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def confusion_matrix(y_true, y_pred, n_classes: int | None = None) -> np.ndarray:
    """Matrice de confusion K x K : ligne = vraie classe, colonne = classe prédite."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(y_true, y_pred) -> float:
    """Part des images correctement classées : (bonnes réponses) / (total)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, n_classes: int | None = None, average: str | None = "macro"):
    """Précision, rappel et F1.

    Pour chaque classe c :
        précision = TP / (TP + FP)   -> « quand le modèle dit c, a-t-il raison ? »
        rappel    = TP / (TP + FN)   -> « parmi les vrais c, combien en trouve-t-il ? »
        F1        = 2 * P * R / (P + R)

    average :
        None    -> un tableau par classe
        "macro" -> moyenne simple des classes (chaque classe pèse pareil)
        "micro" -> on somme TP/FP/FN de toutes les classes avant de calculer
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    if average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    if average is None:
        return precision, recall, f1
    if average == "macro":
        return float(precision.mean()), float(recall.mean()), float(f1.mean())
    if average == "micro":
        return float(precision), float(recall), float(f1)
    raise ValueError(f"average inconnu : {average!r}")


def top_k_accuracy(y_true, scores, k: int = 5) -> float:
    """La bonne classe fait-elle partie des k classes les plus probables ?

    scores : tableau (N, K) de scores ou de probabilités.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    top_k = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean(np.any(top_k == y_true[:, None], axis=1)))
