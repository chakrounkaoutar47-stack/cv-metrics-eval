"""Métriques de détection d'objets : IoU, courbe précision-rappel, AP et mAP.

Format des données (le même que torchmetrics) :
    gts   : liste d'images, chacune {"boxes": (N, 4), "labels": (N,)}
    preds : liste d'images, chacune {"boxes": (M, 4), "labels": (M,), "scores": (M,)}
Les boîtes sont au format (x1, y1, x2, y2) en pixels.

Le calcul suit le protocole COCO : appariement glouton par score décroissant,
AP interpolée sur 101 points de rappel, mAP moyennée sur les classes
(et éventuellement sur plusieurs seuils d'IoU : 0.50, 0.55, ..., 0.95).
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

COCO_IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)


def box_iou(boxes_a, boxes_b) -> np.ndarray:
    """IoU entre deux ensembles de boîtes -> matrice (N, M).

    IoU = aire(intersection) / aire(union)
    """
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _match_image(iou: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Apparie les prédictions (déjà triées par score décroissant) aux vérités terrain.

    Chaque prédiction prend la vérité terrain libre de plus forte IoU, si IoU >= seuil.
    Une vérité terrain ne peut servir qu'une fois : un doublon devient un faux positif.
    Renvoie un booléen par prédiction : True = vrai positif.
    """
    n_pred, n_gt = iou.shape
    gt_used = np.zeros(n_gt, dtype=bool)
    is_tp = np.zeros(n_pred, dtype=bool)
    for p in range(n_pred):
        best_iou, best_g = iou_threshold, -1
        for g in range(n_gt):
            if not gt_used[g] and iou[p, g] >= best_iou:
                best_iou, best_g = iou[p, g], g
        if best_g >= 0:
            gt_used[best_g] = True
            is_tp[p] = True
    return is_tp


def precision_recall_curve(
    gts: list[dict],
    preds: list[dict],
    label: int,
    iou_threshold: float = 0.5,
    iou_fn: Callable = box_iou,
    key: str = "boxes",
):
    """Courbe précision-rappel d'une classe, sur tout le jeu de données.

    On trie toutes les prédictions de la classe par score décroissant, puis on
    « baisse le seuil de confiance » une prédiction à la fois.
    Renvoie (precision, recall, scores, n_gt).
    """
    scores, tps = [], []
    n_gt = 0
    for gt, pred in zip(gts, preds):
        g_mask = np.asarray(gt["labels"]) == label
        p_mask = np.asarray(pred["labels"]) == label
        n_gt += int(g_mask.sum())

        p_scores = np.asarray(pred["scores"], dtype=float)[p_mask]
        order = np.argsort(-p_scores, kind="mergesort")
        p_items = np.asarray(pred[key])[p_mask][order]
        g_items = np.asarray(gt[key])[g_mask]

        if len(p_items) == 0:
            continue
        if len(g_items) == 0:
            is_tp = np.zeros(len(p_items), dtype=bool)
        else:
            is_tp = _match_image(iou_fn(p_items, g_items), iou_threshold)
        scores.extend(p_scores[order].tolist())
        tps.extend(is_tp.tolist())

    scores = np.asarray(scores, dtype=float)
    tps = np.asarray(tps, dtype=bool)
    order = np.argsort(-scores, kind="mergesort")
    scores, tps = scores[order], tps[order]

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(~tps)
    recall = cum_tp / max(n_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    return precision, recall, scores, n_gt


def average_precision(precision, recall, method: str = "coco") -> float:
    """Aire sous la courbe précision-rappel (après lissage).

    On remplace d'abord chaque précision par la meilleure précision obtenue à un
    rappel supérieur ou égal (enveloppe décroissante), puis :
        "coco" : moyenne de la précision sur 101 niveaux de rappel 0, 0.01, ..., 1
        "voc"  : aire exacte sous l'enveloppe (Pascal VOC 2010+)
    """
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    if precision.size == 0:
        return 0.0

    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    if method == "coco":
        levels = np.linspace(0, 1, 101)
        idx = np.searchsorted(recall, levels, side="left")
        sampled = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
        return float(sampled.mean())
    if method == "voc":
        r = np.concatenate([[0.0], recall])
        return float(np.sum((r[1:] - r[:-1]) * envelope))
    raise ValueError(f"method inconnue : {method!r}")


def mean_average_precision(
    gts: list[dict],
    preds: list[dict],
    iou_thresholds: Iterable[float] = (0.5,),
    method: str = "coco",
    iou_fn: Callable = box_iou,
    key: str = "boxes",
) -> dict:
    """mAP = moyenne des AP sur les classes (et sur les seuils d'IoU).

    Exemples :
        iou_thresholds=(0.5,)            -> mAP@0.5 (Pascal VOC)
        iou_thresholds=COCO_IOU_THRESHOLDS -> mAP@[.5:.95] (métrique principale de COCO)
    Les classes absentes de la vérité terrain sont ignorées, comme dans COCO.
    """
    iou_thresholds = [float(t) for t in iou_thresholds]
    labels = sorted({int(l) for gt in gts for l in np.asarray(gt["labels"]).ravel()})

    per_class = {}
    for label in labels:
        aps = []
        for thr in iou_thresholds:
            p, r, _, _ = precision_recall_curve(gts, preds, label, thr, iou_fn, key)
            aps.append(average_precision(p, r, method))
        per_class[label] = aps

    table = np.array([per_class[l] for l in labels]) if labels else np.zeros((0, len(iou_thresholds)))
    return {
        "map": float(table.mean()) if table.size else 0.0,
        "ap_per_class": {l: float(np.mean(v)) for l, v in per_class.items()},
        "map_per_threshold": dict(zip(iou_thresholds, table.mean(axis=0).tolist())) if table.size else {},
    }
