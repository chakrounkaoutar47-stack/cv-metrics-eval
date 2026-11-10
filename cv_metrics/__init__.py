"""cv_metrics : métriques d'évaluation pour la vision par ordinateur, écrites from scratch avec NumPy.

Trois familles de tâches :
- classification  : accuracy, précision, rappel, F1, matrice de confusion, top-k
- détection       : IoU de boîtes, courbe précision-rappel, AP, mAP
- segmentation    : pixel accuracy, IoU par classe, mIoU, Dice, mask AP
"""

from .classification import (
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    top_k_accuracy,
)
from .detection import (
    average_precision,
    box_iou,
    mean_average_precision,
    precision_recall_curve,
)
from .segmentation import (
    dice_per_class,
    iou_per_class,
    mask_iou,
    mean_dice,
    mean_iou,
    pixel_accuracy,
    mask_mean_average_precision,
)

__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "top_k_accuracy",
    "average_precision",
    "box_iou",
    "mean_average_precision",
    "precision_recall_curve",
    "dice_per_class",
    "iou_per_class",
    "mask_iou",
    "mean_dice",
    "mean_iou",
    "pixel_accuracy",
    "mask_mean_average_precision",
]
