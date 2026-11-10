"""Métriques de segmentation sémantique et d'instance.

Segmentation sémantique : gt et pred sont des cartes d'étiquettes (H, W) d'entiers 0..K-1.
Segmentation d'instance : chaque objet est un masque binaire (H, W) ; on réutilise
la mAP de la détection en remplaçant l'IoU des boîtes par l'IoU des masques.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .detection import mean_average_precision


def _valid(gt, ignore_index):
    gt = np.asarray(gt)
    return np.ones(gt.shape, dtype=bool) if ignore_index is None else gt != ignore_index


def pixel_accuracy(gt, pred, ignore_index: int | None = None) -> float:
    """Part des pixels bien classés. Simple, mais trompeuse si une classe domine l'image."""
    gt, pred = np.asarray(gt), np.asarray(pred)
    valid = _valid(gt, ignore_index)
    return float(np.mean(gt[valid] == pred[valid]))


def _tp_fp_fn(gt, pred, n_classes, ignore_index):
    gt, pred = np.asarray(gt), np.asarray(pred)
    valid = _valid(gt, ignore_index)
    g, p = gt[valid].ravel(), pred[valid].ravel()
    tp = np.bincount(g[g == p], minlength=n_classes)[:n_classes].astype(float)
    gt_count = np.bincount(g, minlength=n_classes)[:n_classes].astype(float)
    pred_count = np.bincount(p, minlength=n_classes)[:n_classes].astype(float)
    return tp, pred_count - tp, gt_count - tp


def iou_per_class(gt, pred, n_classes: int, ignore_index: int | None = None) -> np.ndarray:
    """IoU de chaque classe = TP / (TP + FP + FN), comptés en pixels.

    NaN pour une classe absente à la fois de gt et de pred (elle ne compte pas dans la moyenne).
    """
    tp, fp, fn = _tp_fp_fn(gt, pred, n_classes, ignore_index)
    union = tp + fp + fn
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)


def mean_iou(gt, pred, n_classes: int, ignore_index: int | None = None) -> float:
    """mIoU = moyenne des IoU par classe : chaque classe pèse pareil, quelle que soit sa taille."""
    return float(np.nanmean(iou_per_class(gt, pred, n_classes, ignore_index)))


def dice_per_class(gt, pred, n_classes: int, ignore_index: int | None = None) -> np.ndarray:
    """Dice = 2 TP / (2 TP + FP + FN) = 2|A ∩ B| / (|A| + |B|).

    Lien avec l'IoU : Dice = 2 IoU / (1 + IoU). Le Dice est toujours >= IoU.
    """
    tp, fp, fn = _tp_fp_fn(gt, pred, n_classes, ignore_index)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)


def mean_dice(gt, pred, n_classes: int, ignore_index: int | None = None) -> float:
    return float(np.nanmean(dice_per_class(gt, pred, n_classes, ignore_index)))


def mask_iou(masks_a, masks_b) -> np.ndarray:
    """IoU entre deux ensembles de masques binaires -> matrice (N, M)."""
    a = np.asarray(masks_a, dtype=bool).reshape(len(masks_a), -1).astype(float)
    b = np.asarray(masks_b, dtype=bool).reshape(len(masks_b), -1).astype(float)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def mask_mean_average_precision(
    gts: list[dict],
    preds: list[dict],
    iou_thresholds: Iterable[float] = (0.5,),
    method: str = "coco",
) -> dict:
    """mask mAP de la segmentation d'instance.

    gts   : liste d'images {"masks": (N, H, W), "labels": (N,)}
    preds : liste d'images {"masks": (M, H, W), "labels": (M,), "scores": (M,)}
    Même protocole que la détection, mais l'IoU est calculée pixel par pixel sur les masques.
    """
    return mean_average_precision(gts, preds, iou_thresholds, method, iou_fn=mask_iou, key="masks")
