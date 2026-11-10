"""Exemple 1 : métriques de classification.

Partie A - classification de scènes en 3 classes (urbain, autoroute, campagne) sur 60 images.
Partie B - le piège du déséquilibre : 1000 radios dont 50 malades.

Lancer : python examples/01_classification.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cv_metrics import accuracy, confusion_matrix, precision_recall_f1  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def from_confusion(cm):
    """Reconstruit (y_true, y_pred) à partir d'une matrice de confusion voulue."""
    y_true, y_pred = [], []
    for t, row in enumerate(cm):
        for p, n in enumerate(row):
            y_true += [t] * n
            y_pred += [p] * n
    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------- Partie A
CLASSES = ["urbain", "autoroute", "campagne"]
TARGET_CM = [
    [27, 2, 1],   # 30 vraies scènes urbaines
    [3, 16, 1],   # 20 vraies autoroutes
    [1, 2, 7],    # 10 vraies campagnes
]
y_true, y_pred = from_confusion(TARGET_CM)

cm = confusion_matrix(y_true, y_pred, 3)
acc = accuracy(y_true, y_pred)
p_cls, r_cls, f1_cls = precision_recall_f1(y_true, y_pred, 3, average=None)
p_macro, r_macro, f1_macro = precision_recall_f1(y_true, y_pred, 3, average="macro")

print("=== A. Scènes (3 classes, 60 images) ===")
print("Matrice de confusion (ligne = vraie classe, colonne = prédite) :\n", cm)
print(f"Accuracy          : {acc:.3f}")
for c, name in enumerate(CLASSES):
    print(f"  {name:<10} précision={p_cls[c]:.3f}  rappel={r_cls[c]:.3f}  F1={f1_cls[c]:.3f}")
print(f"Macro  P/R/F1     : {p_macro:.3f} / {r_macro:.3f} / {f1_macro:.3f}")

# ---------------------------------------------------------------- Partie B
n_sains, n_malades = 950, 50
truth = np.array([0] * n_sains + [1] * n_malades)          # 1 = malade

naif = np.zeros_like(truth)                                 # répond toujours « sain »
utile = truth.copy()
utile[n_sains:n_sains + 10] = 0                             # rate 10 malades
utile[:30] = 1                                              # 30 fausses alertes

print("\n=== B. Déséquilibre : 950 sains / 50 malades ===")
imbalance = {}
for name, pred in [("naif", naif), ("utile", utile)]:
    p, r, f = precision_recall_f1(truth, pred, 2, average=None)
    imbalance[name] = {"accuracy": accuracy(truth, pred), "precision": p[1], "rappel": r[1], "f1": f[1]}
    print(f"Modèle {name:<6}: accuracy={accuracy(truth, pred):.3f}  "
          f"précision(malade)={p[1]:.3f}  rappel(malade)={r[1]:.3f}  F1(malade)={f[1]:.3f}")

# ---------------------------------------------------------------- Figure
fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200)
ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "#14213D")
ax.set_xticks(range(3), CLASSES)
ax.set_yticks(range(3), CLASSES)
ax.set_xlabel("Classe prédite")
ax.set_ylabel("Vraie classe")
ax.set_title(f"Accuracy = {acc:.2f}   F1 macro = {f1_macro:.2f}")
fig.tight_layout()
fig.savefig(OUT / "classification_confusion.png")

results = {
    "confusion_matrix": cm.tolist(),
    "accuracy": acc,
    "per_class": {n: {"precision": float(p_cls[c]), "recall": float(r_cls[c]), "f1": float(f1_cls[c])}
                  for c, n in enumerate(CLASSES)},
    "macro": {"precision": p_macro, "recall": r_macro, "f1": f1_macro},
    "imbalance": {k: {m: float(v) for m, v in d.items()} for k, d in imbalance.items()},
}
(OUT / "classification_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nFigure et résultats écrits dans {OUT}")
