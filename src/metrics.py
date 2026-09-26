import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    ACTIVE_PYTHON_MODEL,
    CRITICAL_CLASSES,
    METRICS_DIR,
    PYTHON_MODELS,
    display_name,
)



def load_metrics(model_name):
    settings = PYTHON_MODELS.get(model_name)

    if not settings:
        return None

    path = METRICS_DIR / settings["metrics_file"]

    if not os.path.exists(path):
        return None

    try:
        with open(path) as metrics_file:
            metrics = json.load(metrics_file)
    except (ValueError, OSError):
        return None

    if not isinstance(metrics, dict):
        return None

    return metrics


def critical_recall(metrics):
    per_class = metrics.get("per_class", {})
    rows = []

    for class_id in CRITICAL_CLASSES:
        entry = per_class.get(class_id)

        if not isinstance(entry, dict):
            continue

        rows.append(
            {
                "class_id": class_id,
                "name": display_name(class_id),
                "recall": entry.get("recall"),
            }
        )

    return rows


def per_class_rows(metrics):
    per_class = metrics.get("per_class", {})
    rows = []

    for class_id, entry in per_class.items():
        if not isinstance(entry, dict):
            continue

        rows.append(
            {
                "class_id": class_id,
                "name": display_name(class_id),
                "precision": entry.get("precision"),
                "recall": entry.get("recall"),
                "f1": entry.get("f1_score", entry.get("f1")),
                "support": entry.get("support"),
            }
        )

    rows.sort(key=lambda row: row["name"])
    return rows


def all_model_metrics():
    models = []

    for name in PYTHON_MODELS:
        metrics = load_metrics(name)

        models.append(
            {
                "name": name,
                "is_active": name == ACTIVE_PYTHON_MODEL,
                "available": metrics is not None,
                "accuracy": metrics.get("accuracy") if metrics else None,
                "macro_f1": metrics.get("macro_f1") if metrics else None,
                "precision": metrics.get("precision") if metrics else None,
                "recall": metrics.get("recall") if metrics else None,
                "f1": metrics.get("f1_score") if metrics else None,
            }
        )

    return models


def active_model_metrics():
    metrics = load_metrics(ACTIVE_PYTHON_MODEL)

    if metrics is None:
        return {
            "available": False,
            "message": (
                f"No saved metrics for '{ACTIVE_PYTHON_MODEL}'. "
                f"Run the training notebook, it writes the metrics file into reports/."
            ),
        }

    return {
        "available": True,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "macro_f1": metrics.get("macro_f1"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "labels": [display_name(label) for label in metrics.get("labels", [])],
        "per_class": per_class_rows(metrics),
        "critical_recall": critical_recall(metrics),
        "best_params": metrics.get("best_params"),
        "trained_at": metrics.get("trained_at"),
        "noise_robustness": metrics.get("noise_robustness"),
    }
