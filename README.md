# Métriques d'évaluation pour la vision par ordinateur

Devoir *ML Evaluation*, Master Data & IA, INPT (2026). Auteure : Kaoutar Chakroun.

Ce dépôt implémente **from scratch avec NumPy** les métriques d'évaluation de trois tâches de vision :

| Tâche | Métriques implémentées |
|---|---|
| Classification d'images | matrice de confusion, accuracy, précision, rappel, F1 (par classe, macro, micro), top-k accuracy |
| Détection d'objets | IoU de boîtes, courbe précision-rappel, AP (COCO 101 points et VOC), mAP@0.5, mAP@[.5:.95] |
| Segmentation (sémantique et d'instance) | pixel accuracy, IoU par classe, mIoU, Dice, IoU de masques, mask mAP |

Chaque implémentation est **vérifiée par des tests** contre les références du domaine : scikit-learn pour la classification et la segmentation sémantique, **pycocotools** (l'évaluateur officiel de COCO) pour la détection et la segmentation d'instance.

Tous les exemples utilisent la même image fil rouge : une scène de rue synthétique (ciel, trottoir, route, 2 voitures, 3 piétons, 1 feu tricolore), avec une prédiction volontairement imparfaite.

---

## Structure

```
cv-metrics-eval/
├── cv_metrics/
│   ├── __init__.py          # point d'entrée du package
│   ├── classification.py    # métriques de classification
│   ├── detection.py         # IoU, courbe PR, AP, mAP
│   └── segmentation.py      # pixel accuracy, IoU, mIoU, Dice, mask mAP
├── examples/
│   ├── scene.py             # scène de rue synthétique (vérité terrain + prédiction)
│   ├── 01_classification.py
│   ├── 02_detection.py
│   └── 03_segmentation.py
├── tests/
│   └── test_metrics.py      # comparaison avec scikit-learn et pycocotools
├── outputs/                 # figures et résultats JSON générés par les exemples
├── demo.ipynb               # notebook de démonstration (les 3 familles de métriques + tests)
└── requirements.txt
```

## Installation et exécution

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python examples/01_classification.py
python examples/02_detection.py
python examples/03_segmentation.py

pytest -q          # 10 tests

jupyter notebook demo.ipynb   # démo pas à pas
```

Chaque exemple affiche ses résultats dans le terminal et écrit une figure (`.png`) et un fichier de résultats (`.json`) dans `outputs/`.

---

## Documentation du code

### `cv_metrics/classification.py`

Les classes sont des entiers `0..K-1`.

| Fonction | Rôle |
|---|---|
| `confusion_matrix(y_true, y_pred, n_classes=None)` | Matrice K × K : ligne = vraie classe, colonne = classe prédite. |
| `accuracy(y_true, y_pred)` | Part des exemples bien classés. |
| `precision_recall_f1(y_true, y_pred, n_classes=None, average="macro")` | Précision, rappel et F1. `average=None` renvoie un tableau par classe, `"macro"` la moyenne simple des classes, `"micro"` le calcul sur les TP/FP/FN cumulés. |
| `top_k_accuracy(y_true, scores, k=5)` | La bonne classe est-elle parmi les k scores les plus élevés ? `scores` est un tableau (N, K). |

Formules, pour une classe c :

- précision = TP / (TP + FP) : quand le modèle dit « c », a-t-il raison ?
- rappel = TP / (TP + FN) : parmi les vrais « c », combien sont trouvés ?
- F1 = 2 · P · R / (P + R) : moyenne harmonique, faible dès que l'un des deux est faible.

Une division par zéro (classe jamais prédite, par exemple) renvoie 0, comme `zero_division=0` dans scikit-learn.

### `cv_metrics/detection.py`

Format des données, identique à celui de torchmetrics :

```python
gts   = [{"boxes": array (N, 4), "labels": array (N,)}, ...]                       # une entrée par image
preds = [{"boxes": array (M, 4), "labels": array (M,), "scores": array (M,)}, ...]
```

Les boîtes sont au format `(x1, y1, x2, y2)`.

| Fonction | Rôle |
|---|---|
| `box_iou(boxes_a, boxes_b)` | Matrice (N, M) des IoU = aire(intersection) / aire(union). |
| `precision_recall_curve(gts, preds, label, iou_threshold=0.5)` | Courbe précision-rappel d'une classe sur tout le jeu. Renvoie `(precision, recall, scores, n_gt)`. |
| `average_precision(precision, recall, method="coco")` | Aire sous la courbe après lissage. `"coco"` : moyenne sur 101 niveaux de rappel ; `"voc"` : aire exacte (Pascal VOC 2010+). |
| `mean_average_precision(gts, preds, iou_thresholds=(0.5,), method="coco")` | mAP moyennée sur les classes et les seuils. Renvoie `{"map", "ap_per_class", "map_per_threshold"}`. |
| `COCO_IOU_THRESHOLDS` | Les 10 seuils 0.50, 0.55, …, 0.95 de la métrique principale de COCO. |

Protocole (celui de COCO) :

1. Pour chaque classe, on trie toutes les prédictions par score décroissant.
2. Chaque prédiction est appariée à la vérité terrain libre de plus forte IoU, si l'IoU dépasse le seuil : c'est un **vrai positif**. Sinon, ou si l'objet a déjà été trouvé (doublon), c'est un **faux positif**. Les objets jamais trouvés sont des **faux négatifs**.
3. En descendant la liste, on calcule précision et rappel cumulés : c'est la courbe précision-rappel.
4. On lisse la courbe (enveloppe décroissante), puis l'**AP** est la précision moyenne sur 101 niveaux de rappel.
5. La **mAP** est la moyenne des AP sur les classes ; **mAP@[.5:.95]** fait en plus la moyenne sur les 10 seuils d'IoU, ce qui récompense une localisation précise.

### `cv_metrics/segmentation.py`

Segmentation sémantique : `gt` et `pred` sont des cartes d'étiquettes (H, W).

| Fonction | Rôle |
|---|---|
| `pixel_accuracy(gt, pred, ignore_index=None)` | Part des pixels bien classés. |
| `iou_per_class(gt, pred, n_classes, ignore_index=None)` | IoU de chaque classe = TP / (TP + FP + FN), en pixels. NaN si la classe est absente des deux cartes. |
| `mean_iou(...)` | mIoU : moyenne des IoU par classe (chaque classe pèse pareil, quelle que soit sa taille). |
| `dice_per_class(...)`, `mean_dice(...)` | Dice = 2 TP / (2 TP + FP + FN). On a toujours Dice = 2 · IoU / (1 + IoU). |

Segmentation d'instance : chaque objet est un masque binaire.

| Fonction | Rôle |
|---|---|
| `mask_iou(masks_a, masks_b)` | Matrice (N, M) des IoU entre masques, calculée pixel par pixel. |
| `mask_mean_average_precision(gts, preds, iou_thresholds=(0.5,))` | Même protocole que la mAP de détection, avec l'IoU des masques à la place de celle des boîtes. Les entrées ont une clé `"masks"` (N, H, W) au lieu de `"boxes"`. |

`ignore_index` permet d'exclure une étiquette « non annotée » (souvent 255) de tous les calculs sémantiques.

### `examples/scene.py`

Construit la scène 640 × 400 à partir de primitives (rectangles, cercles) :

- `ground_truth()` et `prediction()` renvoient `(carte_sémantique, instances)`, où `instances` est une liste de `(classe, nom, masque)` ;
- la prédiction contient des erreurs typiques : voitures trop larges de 4 px, jambes des piétons oubliées, tête du piéton 3 oubliée, poteau du feu oublié, frontière trottoir / route décalée de 6 px ;
- `mask_to_box(mask)` donne la boîte englobante d'un masque, `colorize(label_map)` produit une image RGB.

---

## Résultats des exemples

### 1. Classification

Scènes en 3 classes (60 images) :

| | urbain | autoroute | campagne |
|---|---|---|---|
| précision | 0,871 | 0,800 | 0,778 |
| rappel | 0,900 | 0,800 | 0,700 |
| F1 | 0,885 | 0,800 | 0,737 |

Accuracy = 0,833 ; F1 macro = 0,807.

Piège du déséquilibre (950 radios saines, 50 malades) :

| Modèle | Accuracy | Rappel (malade) | F1 (malade) |
|---|---|---|---|
| répond toujours « sain » | 0,950 | 0,000 | 0,000 |
| modèle utile | 0,960 | 0,800 | 0,667 |

L'accuracy ne distingue presque pas les deux modèles ; le rappel et le F1 de la classe rare, si.

### 2. Détection

8 détections sur la scène : 5 correctes, 1 doublon, 1 boîte trop décalée (IoU 0,30), 1 fausse alerte.

| | voiture | piéton | feu |
|---|---|---|---|
| AP@0.5 | 1,000 | 0,663 | 1,000 |

mAP@0.5 = 0,888 ; mAP@[.5:.95] = 0,523. L'écart montre que les boîtes sont trouvées mais pas toujours finement ajustées.

### 3. Segmentation

Sémantique :

| | ciel | trottoir | route | voiture | piéton | feu |
|---|---|---|---|---|---|---|
| part de l'image | 48,3 % | 11,4 % | 26,9 % | 10,6 % | 1,9 % | 0,9 % |
| IoU | 0,994 | 0,844 | 0,892 | 0,854 | 0,656 | 0,677 |
| Dice | 0,997 | 0,916 | 0,943 | 0,921 | 0,792 | 0,807 |

Pixel accuracy = 0,960, mais mIoU = 0,820 : les petites classes (piéton, feu) sont mal segmentées alors qu'elles ne pèsent presque rien dans la pixel accuracy.

Instance : le modèle fusionne les piétons 1 et 2 en un seul masque (IoU 0,42 : faux positif, et deux piétons manqués). Mask AP@0.5 : voiture 1,000, piéton 0,168, feu 1,000 ; mask mAP@0.5 = 0,723 ; mask mAP@[.5:.95] = 0,381.

---

## Validation

`tests/test_metrics.py` compare chaque fonction à une implémentation de référence :

- classification : `sklearn.metrics` (matrice de confusion, accuracy, précision, rappel, F1 macro et micro, top-k) ;
- détection : `pycocotools.COCOeval` sur 3 jeux aléatoires (mAP@[.5:.95] et mAP@0.5 identiques à 10⁻⁶ près) ;
- segmentation sémantique : `jaccard_score` (IoU) et `f1_score` (Dice) de scikit-learn sur les pixels ;
- segmentation d'instance : `COCOeval` en mode `segm`.

## Références

- Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014 (protocole mAP@[.5:.95]).
- Everingham et al., *The Pascal Visual Object Classes (VOC) Challenge*, IJCV 2010 (AP et mAP@0.5).
- Kirillov et al., *Panoptic Segmentation*, CVPR 2019 (métrique PQ).
