"""Exemple 2 : métriques de détection sur la scène de rue.

Un détecteur fictif renvoie 8 boîtes : de bonnes détections, un doublon,
une boîte mal placée et une fausse alerte. On calcule l'IoU, la courbe
précision-rappel, l'AP par classe, la mAP@0.5 et la mAP@[.5:.95] (COCO).

Lancer : python examples/02_detection.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from cv_metrics import box_iou, mean_average_precision, precision_recall_curve  # noqa: E402
from cv_metrics.detection import COCO_IOU_THRESHOLDS  # noqa: E402
from scene import PIETON, THING_CLASSES, VOITURE, colorize, ground_truth, mask_to_box  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- Vérité terrain
semantic, instances = ground_truth()
gt_names = [name for _, name, _ in instances]
gt = {
    "boxes": np.array([mask_to_box(m) for _, _, m in instances], dtype=float),
    "labels": np.array([cls for cls, _, _ in instances]),
}

# ---------------------------------------------------------------- Prédictions du détecteur fictif
# (classe, score, boîte x1 y1 x2 y2, commentaire)
DETECTIONS = [
    (VOITURE, 0.96, [64, 254, 236, 352], "voiture 1, bien placée"),
    (VOITURE, 0.91, [380, 255, 600, 385], "voiture 2, boîte un peu large"),
    (PIETON, 0.88, [267, 157, 293, 247], "piéton 1, bien placé"),
    (PIETON, 0.80, [595, 150, 625, 250], "piéton 3, boîte correcte"),
    (5, 0.75, [343, 64, 375, 255], "feu, bien placé"),
    (PIETON, 0.70, [455, 160, 480, 250], "fausse alerte sur le trottoir"),
    (PIETON, 0.62, [300, 165, 330, 250], "piéton 2, boîte trop décalée"),
    (VOITURE, 0.40, [40, 240, 200, 340], "doublon de la voiture 1"),
]
pred = {
    "labels": np.array([d[0] for d in DETECTIONS]),
    "scores": np.array([d[1] for d in DETECTIONS]),
    "boxes": np.array([d[2] for d in DETECTIONS], dtype=float),
}

# ---------------------------------------------------------------- IoU de chaque détection
print("=== IoU de chaque détection avec la meilleure vérité terrain de sa classe ===")
iou = box_iou(pred["boxes"], gt["boxes"])
det_rows = []
for i, (cls, score, box, note) in enumerate(DETECTIONS):
    same = gt["labels"] == cls
    best = int(np.argmax(np.where(same, iou[i], -1)))
    det_rows.append({"classe": THING_CLASSES[cls], "score": score, "iou": float(iou[i, best]),
                     "gt": gt_names[best], "note": note})
    print(f"  {THING_CLASSES[cls]:<8} score={score:.2f}  IoU={iou[i, best]:.2f} avec {gt_names[best]:<3} -> {note}")

# ---------------------------------------------------------------- AP et mAP
res50 = mean_average_precision([gt], [pred], iou_thresholds=[0.5])
res_coco = mean_average_precision([gt], [pred], iou_thresholds=COCO_IOU_THRESHOLDS)

print("\n=== AP par classe (IoU >= 0.5) ===")
for cls, ap in res50["ap_per_class"].items():
    print(f"  {THING_CLASSES[cls]:<8} AP = {ap:.3f}")
print(f"mAP@0.5        = {res50['map']:.3f}")
print(f"mAP@[.5:.95]   = {res_coco['map']:.3f}   (moyenne sur 10 seuils d'IoU)")
for thr, m in res_coco["map_per_threshold"].items():
    print(f"   seuil {thr:.2f} -> mAP {m:.3f}")

# ---------------------------------------------------------------- Courbe précision-rappel (piéton)
p, r, s, n_gt = precision_recall_curve([gt], [pred], PIETON, 0.5)
print("\n=== Courbe précision-rappel, classe piéton (3 vérités terrain) ===")
for k in range(len(p)):
    print(f"  seuil score >= {s[k]:.2f} : précision={p[k]:.2f}  rappel={r[k]:.2f}")

DARK, ACCENT, BLUE = "#14213D", "#B5541B", "#2F5DA8"
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
rr = np.concatenate([[0], r])
pp = np.concatenate([[1], p])
ax.step(rr, pp, where="pre", color=BLUE, lw=2.5, label="Précision-rappel (piéton)")
envelope = np.maximum.accumulate(p[::-1])[::-1]
ax.fill_between(np.concatenate([[0], r]), np.concatenate([[envelope[0]], envelope]),
                step="pre", alpha=0.15, color=ACCENT, label=f"AP = aire = {res50['ap_per_class'][PIETON]:.2f}")
for k in range(len(p)):
    ax.plot(r[k], p[k], "o", color=DARK)
    ax.annotate(f"{s[k]:.2f}", (r[k], p[k]), textcoords="offset points", xytext=(8, -4), fontsize=9, color=DARK)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Rappel")
ax.set_ylabel("Précision")
ax.legend(loc="lower left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "detection_pr_curve.png")

# ---------------------------------------------------------------- Image avec boîtes
natural = {0: "#CFE3F2", 1: "#D9CFC1", 2: "#6B7280", 3: "#5B7DB1", 4: "#2F3E46", 5: "#1F2937"}
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
ax.imshow(colorize(semantic, natural))
for box in gt["boxes"]:
    ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                   fill=False, ec="#2F9E77", lw=2, ls="--"))
matched = {(d["gt"], d["classe"]) for d in det_rows if d["iou"] >= 0.5}
seen = set()
for (cls, score, box, _), row in zip(DETECTIONS, det_rows):
    key = (row["gt"], row["classe"])
    ok = row["iou"] >= 0.5 and key not in seen
    if ok:
        seen.add(key)
    color = "#F2A541" if ok else "#D6336C"
    ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec=color, lw=2))
    ax.text(box[0], box[1] - 4, f"{score:.2f}", color="white", fontsize=7,
            bbox=dict(facecolor=color, edgecolor="none", pad=1))
ax.axis("off")
fig.tight_layout(pad=0)
fig.savefig(OUT / "detection_boxes.png")

results = {
    "detections": det_rows,
    "ap50_per_class": {THING_CLASSES[c]: v for c, v in res50["ap_per_class"].items()},
    "map50": res50["map"],
    "map_coco": res_coco["map"],
    "map_per_threshold": {f"{k:.2f}": v for k, v in res_coco["map_per_threshold"].items()},
    "pr_pieton": {"precision": p.tolist(), "recall": r.tolist(), "scores": s.tolist()},
}
(OUT / "detection_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nFigures et résultats écrits dans {OUT}")
