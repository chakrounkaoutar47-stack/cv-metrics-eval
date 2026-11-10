"""Exemple 3 : métriques de segmentation sur la scène de rue.

Sémantique : on compare la carte prédite à la vérité terrain pixel par pixel
(pixel accuracy, IoU par classe, mIoU, Dice).
Instance : on compare des masques d'objets avec des scores (mask AP).

Lancer : python examples/03_segmentation.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from cv_metrics import (  # noqa: E402
    dice_per_class,
    iou_per_class,
    mask_iou,
    mask_mean_average_precision,
    mean_dice,
    mean_iou,
    pixel_accuracy,
)
from cv_metrics.detection import COCO_IOU_THRESHOLDS  # noqa: E402
from scene import CLASSES, PIETON, THING_CLASSES, colorize, ground_truth, prediction  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
K = len(CLASSES)

gt_sem, gt_inst = ground_truth()
pred_sem, pred_inst = prediction()

# ---------------------------------------------------------------- Sémantique
pa = pixel_accuracy(gt_sem, pred_sem)
ious = iou_per_class(gt_sem, pred_sem, K)
dices = dice_per_class(gt_sem, pred_sem, K)
miou = mean_iou(gt_sem, pred_sem, K)
mdice = mean_dice(gt_sem, pred_sem, K)

print("=== Segmentation sémantique ===")
print(f"Pixel accuracy : {pa:.3f}")
share = np.bincount(gt_sem.ravel(), minlength=K) / gt_sem.size
for c, name in enumerate(CLASSES):
    print(f"  {name:<9} part de l'image={share[c]:6.1%}  IoU={ious[c]:.3f}  Dice={dices[c]:.3f}")
print(f"mIoU  : {miou:.3f}")
print(f"mDice : {mdice:.3f}")

# ---------------------------------------------------------------- Instance
# Le modèle d'instance fusionne les piétons 1 et 2 en un seul objet (erreur fréquente).
gt_masks = np.array([m for _, _, m in gt_inst])
gt_labels = np.array([c for c, _, _ in gt_inst])
pred_named = {name: (cls, m) for cls, name, m in pred_inst}
merged = pred_named["P1"][1] | pred_named["P2"][1]
INSTANCE_PREDS = [
    ("voiture 1", 3, 0.97, pred_named["V1"][1]),
    ("voiture 2", 3, 0.93, pred_named["V2"][1]),
    ("piétons 1+2 fusionnés", PIETON, 0.85, merged),
    ("piéton 3 incomplet", PIETON, 0.80, pred_named["P3"][1]),
    ("feu sans poteau", 5, 0.72, pred_named["F1"][1]),
]
p_masks = np.array([m for *_, m in INSTANCE_PREDS])
inst_pred = {"masks": p_masks, "labels": np.array([c for _, c, _, _ in INSTANCE_PREDS]),
             "scores": np.array([s for _, _, s, _ in INSTANCE_PREDS])}
inst_gt = {"masks": gt_masks, "labels": gt_labels}

print("\n=== Segmentation d'instance : IoU de masque ===")
miou_mat = mask_iou(p_masks, gt_masks)
inst_rows = []
for i, (name, cls, score, _) in enumerate(INSTANCE_PREDS):
    same = gt_labels == cls
    j = int(np.argmax(np.where(same, miou_mat[i], -1)))
    inst_rows.append({"prediction": name, "score": score, "mask_iou": float(miou_mat[i, j]), "gt": gt_inst[j][1]})
    print(f"  {name:<24} score={score:.2f}  IoU masque={miou_mat[i, j]:.3f} avec {gt_inst[j][1]}")

res50 = mask_mean_average_precision([inst_gt], [inst_pred], [0.5])
res_coco = mask_mean_average_precision([inst_gt], [inst_pred], COCO_IOU_THRESHOLDS)
for cls, ap in res50["ap_per_class"].items():
    print(f"  mask AP@0.5 {THING_CLASSES[cls]:<8} = {ap:.3f}")
print(f"mask mAP@0.5      = {res50['map']:.3f}")
print(f"mask mAP@[.5:.95] = {res_coco['map']:.3f}")

# ---------------------------------------------------------------- Figures
def save(img, name):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(OUT / name)
    plt.close(fig)


save(colorize(gt_sem), "segmentation_gt.png")
save(colorize(pred_sem), "segmentation_pred.png")
errors = np.ones(gt_sem.shape + (3,)) * 0.96
errors[gt_sem != pred_sem] = [0.71, 0.33, 0.11]  # pixels faux en orange foncé
save(errors, "segmentation_errors.png")

palette = ["#2F5DA8", "#1F9E8F", "#D9463E", "#F2B13A", "#8E5BC4", "#5BAE4F"]


def instance_image(masks):
    img = np.ones(gt_sem.shape + (3,)) * 0.92
    for k, m in enumerate(masks):
        h = palette[k % len(palette)]
        img[m] = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return img


save(instance_image(gt_masks), "instance_gt.png")
save(instance_image([p_masks[i] for i in (0, 1, 2, 3, 4)]), "instance_pred.png")

results = {
    "pixel_accuracy": pa,
    "class_share": {n: float(share[c]) for c, n in enumerate(CLASSES)},
    "iou_per_class": {n: float(ious[c]) for c, n in enumerate(CLASSES)},
    "dice_per_class": {n: float(dices[c]) for c, n in enumerate(CLASSES)},
    "miou": miou,
    "mdice": mdice,
    "instance": inst_rows,
    "mask_ap50_per_class": {THING_CLASSES[c]: v for c, v in res50["ap_per_class"].items()},
    "mask_map50": res50["map"],
    "mask_map_coco": res_coco["map"],
}
(OUT / "segmentation_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nFigures et résultats écrits dans {OUT}")
