"""Tests : nos implémentations NumPy doivent donner les mêmes chiffres que les références.

- classification : scikit-learn
- détection et segmentation d'instance : pycocotools (l'évaluateur officiel de COCO)
- segmentation sémantique : scikit-learn (jaccard_score / f1_score sur les pixels)

Lancer : pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from cv_metrics import (  # noqa: E402
    accuracy,
    average_precision,
    box_iou,
    confusion_matrix,
    dice_per_class,
    iou_per_class,
    mask_mean_average_precision,
    mean_average_precision,
    pixel_accuracy,
    precision_recall_f1,
    top_k_accuracy,
)
from cv_metrics.detection import COCO_IOU_THRESHOLDS  # noqa: E402

sk = pytest.importorskip("sklearn.metrics")


# ------------------------------------------------------------------ classification
def test_classification_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 4, 500))

    assert np.array_equal(confusion_matrix(y_true, y_pred, 4), sk.confusion_matrix(y_true, y_pred))
    assert accuracy(y_true, y_pred) == pytest.approx(sk.accuracy_score(y_true, y_pred))
    for avg in ["macro", "micro"]:
        p, r, f = precision_recall_f1(y_true, y_pred, 4, average=avg)
        assert p == pytest.approx(sk.precision_score(y_true, y_pred, average=avg))
        assert r == pytest.approx(sk.recall_score(y_true, y_pred, average=avg))
        assert f == pytest.approx(sk.f1_score(y_true, y_pred, average=avg))


def test_top_k():
    rng = np.random.default_rng(1)
    scores = rng.random((200, 10))
    y = rng.integers(0, 10, 200)
    assert top_k_accuracy(y, scores, 3) == pytest.approx(sk.top_k_accuracy_score(y, scores, k=3))


# ------------------------------------------------------------------ détection
def test_box_iou_simple():
    # deux carrés 2x2 décalés de 1 : intersection 2, union 6
    assert box_iou([[0, 0, 2, 2]], [[1, 0, 3, 2]])[0, 0] == pytest.approx(2 / 6)
    assert box_iou([[0, 0, 1, 1]], [[2, 2, 3, 3]])[0, 0] == 0


def test_average_precision_perfect_and_empty():
    assert average_precision([1.0], [1.0]) == pytest.approx(1.0)
    assert average_precision([], []) == 0.0


def _coco_eval(gts, preds, iou_type="bbox"):
    """Fait tourner pycocotools et renvoie (mAP@[.5:.95], mAP@0.5)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils

    labels = sorted({int(l) for g in gts for l in g["labels"]})
    images, anns, dets = [], [], []
    for img_id, (g, p) in enumerate(zip(gts, preds), start=1):
        h, w = (g["masks"].shape[1:] if iou_type == "segm" else (1000, 1000))
        images.append({"id": img_id, "height": int(h), "width": int(w)})
        for k, lab in enumerate(g["labels"]):
            ann = {"id": len(anns) + 1, "image_id": img_id, "category_id": int(lab), "iscrowd": 0}
            if iou_type == "bbox":
                x1, y1, x2, y2 = g["boxes"][k]
                ann.update(bbox=[x1, y1, x2 - x1, y2 - y1], area=float((x2 - x1) * (y2 - y1)))
            else:
                rle = mask_utils.encode(np.asfortranarray(g["masks"][k].astype(np.uint8)))
                ann.update(segmentation=rle, area=float(g["masks"][k].sum()),
                           bbox=list(mask_utils.toBbox(rle)))
            anns.append(ann)
        for k, lab in enumerate(p["labels"]):
            det = {"image_id": img_id, "category_id": int(lab), "score": float(p["scores"][k])}
            if iou_type == "bbox":
                x1, y1, x2, y2 = p["boxes"][k]
                det["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            else:
                rle = mask_utils.encode(np.asfortranarray(p["masks"][k].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode()
                det["segmentation"] = rle
            dets.append(det)

    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": anns,
                       "categories": [{"id": int(l)} for l in labels]}
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats[0], ev.stats[1]


def _random_detection_set(seed, n_images=8, n_classes=3):
    rng = np.random.default_rng(seed)
    gts, preds = [], []
    for _ in range(n_images):
        n = rng.integers(1, 6)
        xy = rng.uniform(0, 800, (n, 2))
        wh = rng.uniform(30, 150, (n, 2))
        g_boxes = np.hstack([xy, xy + wh])
        g_labels = rng.integers(1, n_classes + 1, n)
        # prédictions = vérités bruitées + quelques fausses alertes
        noise = rng.normal(0, 12, g_boxes.shape)
        keep = rng.random(n) < 0.85
        p_boxes = (g_boxes + noise)[keep]
        p_labels = g_labels[keep]
        m = rng.integers(0, 3)
        fxy = rng.uniform(0, 800, (m, 2))
        p_boxes = np.vstack([p_boxes, np.hstack([fxy, fxy + rng.uniform(30, 150, (m, 2))])])
        p_labels = np.concatenate([p_labels, rng.integers(1, n_classes + 1, m)])
        gts.append({"boxes": g_boxes, "labels": g_labels})
        preds.append({"boxes": p_boxes, "labels": p_labels, "scores": rng.random(len(p_labels))})
    return gts, preds


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_map_matches_pycocotools(seed):
    pytest.importorskip("pycocotools")
    gts, preds = _random_detection_set(seed)
    coco_map, coco_ap50 = _coco_eval(gts, preds, "bbox")
    ours = mean_average_precision(gts, preds, COCO_IOU_THRESHOLDS)["map"]
    ours50 = mean_average_precision(gts, preds, [0.5])["map"]
    assert ours == pytest.approx(coco_map, abs=1e-6)
    assert ours50 == pytest.approx(coco_ap50, abs=1e-6)


# ------------------------------------------------------------------ segmentation
def test_semantic_matches_sklearn():
    from scene import CLASSES, ground_truth, prediction

    gt, _ = ground_truth()
    pred, _ = prediction()
    k = len(CLASSES)
    g, p = gt.ravel(), pred.ravel()
    assert pixel_accuracy(gt, pred) == pytest.approx(sk.accuracy_score(g, p))
    assert np.allclose(iou_per_class(gt, pred, k), sk.jaccard_score(g, p, average=None))
    # Dice = F1 calculé sur les pixels
    assert np.allclose(dice_per_class(gt, pred, k), sk.f1_score(g, p, average=None))


def test_dice_iou_relation():
    from scene import CLASSES, ground_truth, prediction

    gt, _ = ground_truth()
    pred, _ = prediction()
    iou = iou_per_class(gt, pred, len(CLASSES))
    dice = dice_per_class(gt, pred, len(CLASSES))
    assert np.allclose(dice, 2 * iou / (1 + iou))


def test_mask_map_matches_pycocotools():
    pytest.importorskip("pycocotools")
    from scene import ground_truth, prediction

    _, gt_inst = ground_truth()
    _, pred_inst = prediction()
    gts = [{"masks": np.array([m for *_, m in gt_inst]), "labels": np.array([c for c, *_ in gt_inst])}]
    rng = np.random.default_rng(0)
    preds = [{"masks": np.array([m for *_, m in pred_inst]),
              "labels": np.array([c for c, *_ in pred_inst]),
              "scores": rng.random(len(pred_inst))}]
    coco_map, coco_ap50 = _coco_eval(gts, preds, "segm")
    assert mask_mean_average_precision(gts, preds, COCO_IOU_THRESHOLDS)["map"] == pytest.approx(coco_map, abs=1e-6)
    assert mask_mean_average_precision(gts, preds, [0.5])["map"] == pytest.approx(coco_ap50, abs=1e-6)
