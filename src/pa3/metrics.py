"""Métricas: F1 macro (principal), acurácia, F1 por classe, matriz de confusão."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def per_class_table(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    return pd.DataFrame({"label": classes, "precision": p, "recall": r, "f1": f, "support": s})


def plot_confusion(y_true, y_pred, classes: list[str], path, title: str = "") -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fração da linha (recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm
