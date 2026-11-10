"""Scène de rue synthétique servant de fil rouge à tous les exemples.

Une image 640 x 400 : ciel, trottoir, route, 2 voitures, 3 piétons, 1 feu tricolore.
On construit :
- la vérité terrain (carte sémantique, masques d'instance, boîtes) ;
- une « prédiction » volontairement imparfaite, pour que les métriques aient
  quelque chose à mesurer (voitures trop grosses, jambes oubliées, etc.).
"""

from __future__ import annotations

import numpy as np

H, W = 400, 640
CLASSES = ["ciel", "trottoir", "route", "voiture", "piéton", "feu"]
CIEL, TROTTOIR, ROUTE, VOITURE, PIETON, FEU = range(6)
# Classes « objets » (things) utilisées en détection et en segmentation d'instance
THING_CLASSES = {VOITURE: "voiture", PIETON: "piéton", FEU: "feu"}

CLASS_COLORS = {
    CIEL: "#9CC9E8",
    TROTTOIR: "#E9D8A6",
    ROUTE: "#8D939C",
    VOITURE: "#2F5DA8",
    PIETON: "#D9463E",
    FEU: "#E8C547",
}

# Chaque objet = (classe, nom, liste de primitives). Ordre = ordre de dessin (le dernier masque les autres).
GT_OBJECTS = [
    (PIETON, "P1", [("circle", 280, 168, 10), ("rect", 269, 180, 22, 38), ("rect", 272, 216, 16, 30)]),
    (PIETON, "P2", [("circle", 302, 164, 10), ("rect", 291, 176, 22, 40), ("rect", 294, 214, 16, 32)]),
    (PIETON, "P3", [("circle", 610, 166, 10), ("rect", 599, 178, 22, 40), ("rect", 602, 216, 16, 30)]),
    (FEU, "F1", [("rect", 356, 120, 6, 130), ("rect", 346, 66, 26, 60)]),
    (VOITURE, "V1", [("rect", 95, 250, 100, 40), ("rect", 60, 280, 170, 50),
                     ("circle", 100, 332, 15), ("circle", 190, 332, 15)]),
    (VOITURE, "V2", [("rect", 430, 268, 110, 40), ("rect", 390, 300, 190, 55),
                     ("circle", 435, 357, 15), ("circle", 535, 357, 15)]),
]


def _grow(prims, d):
    """Agrandit chaque primitive de d pixels (simule un masque « qui bave »)."""
    out = []
    for p in prims:
        if p[0] == "rect":
            _, x, y, w, h = p
            out.append(("rect", x - d, y - d, w + 2 * d, h + 2 * d))
        else:
            _, cx, cy, r = p
            out.append(("circle", cx, cy, r + d))
    return out


def _by_name(name):
    return next(o for o in GT_OBJECTS if o[1] == name)


# Prédiction de segmentation : erreurs typiques d'un vrai modèle
PRED_OBJECTS = [
    (PIETON, "P1", _by_name("P1")[2][:2]),             # jambes oubliées
    (PIETON, "P2", _by_name("P2")[2][:2]),             # jambes oubliées
    (PIETON, "P3", _by_name("P3")[2][1:2]),            # tête et jambes oubliées
    (FEU, "F1", _by_name("F1")[2][1:]),                # poteau oublié
    (VOITURE, "V1", _grow(_by_name("V1")[2], 4)),      # contours trop larges
    (VOITURE, "V2", _grow(_by_name("V2")[2], 4)),
]
GT_SIDEWALK_END = 250
PRED_SIDEWALK_END = 256  # frontière trottoir / route décalée de 6 px


def _prim_mask(p):
    yy, xx = np.mgrid[0:H, 0:W]
    if p[0] == "rect":
        _, x, y, w, h = p
        return (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
    _, cx, cy, r = p
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _shape_mask(prims):
    m = np.zeros((H, W), dtype=bool)
    for p in prims:
        m |= _prim_mask(p)
    return m


def render(objects, sidewalk_end):
    """Renvoie (carte sémantique (H, W), liste de (classe, nom, masque visible))."""
    semantic = np.full((H, W), CIEL, dtype=np.int64)
    semantic[200:sidewalk_end] = TROTTOIR
    semantic[sidewalk_end:] = ROUTE

    owner = np.full((H, W), -1)
    for i, (cls, _, prims) in enumerate(objects):
        m = _shape_mask(prims)
        semantic[m] = cls
        owner[m] = i
    instances = [(cls, name, owner == i) for i, (cls, name, _) in enumerate(objects)]
    return semantic, instances


def ground_truth():
    return render(GT_OBJECTS, GT_SIDEWALK_END)


def prediction():
    return render(PRED_OBJECTS, PRED_SIDEWALK_END)


def mask_to_box(mask):
    """Boîte englobante (x1, y1, x2, y2) d'un masque binaire."""
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def colorize(label_map, colors=CLASS_COLORS):
    """Carte d'étiquettes -> image RGB (pour l'affichage)."""
    rgb = np.zeros(label_map.shape + (3,))
    for cls, hex_color in colors.items():
        rgb[label_map == cls] = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return rgb
