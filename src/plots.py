import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    CLASSES,
    CRITICAL_CLASSES,
    METADATA_DIR,
    METRICS_DIR,
    PYTHON_MODELS,
    display_name,
)

# Every figure the report needs is drawn from the saved metrics file, so plots
# can be regenerated at any time without retraining:
#   python src/plots.py

PLOT_DIR = METRICS_DIR / "plots"

# The SRS accuracy targets, drawn on the charts as a reference line.
ACCURACY_TARGET = 0.85
MACRO_F1_TARGET = 0.80
CRITICAL_RECALL_TARGET = 0.85

GREEN = "#238b6c"
RED = "#b3261e"
GREY = "#8a949c"
BLUE = "#2a6f97"
ORANGE = "#c2570d"


def plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)
    return PLOT_DIR


def save(figure, name):
    path = plot_dir() / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("saved plot", path)
    return path


def short_names(labels):
    return [display_name(label).replace(" or ", "/").replace("Person Asking for ", "") for label in labels]


def bar_colours(values, target):
    return [GREEN if value >= target else RED for value in values]


def plot_overall_metrics(metrics, model_name):
    names = ["accuracy", "precision", "recall", "f1_score", "macro_f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1 score", "Macro F1"]
    values = [metrics.get(name) or 0.0 for name in names]

    figure, axes = plt.subplots(figsize=(7, 4))

    targets = [ACCURACY_TARGET, 0, 0, 0, MACRO_F1_TARGET]
    colours = []
    for value, target in zip(values, targets):
        colours.append(GREEN if target == 0 or value >= target else RED)

    bars = axes.bar(labels, values, color=colours)

    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                  f"{value:.3f}", ha="center", fontsize=9)

    axes.axhline(ACCURACY_TARGET, color=GREY, linestyle="--", linewidth=1)
    axes.text(-0.45, ACCURACY_TARGET + 0.012, f"accuracy target {ACCURACY_TARGET}",
              ha="left", fontsize=8, color=GREY)

    axes.axhline(MACRO_F1_TARGET, color=GREY, linestyle=":", linewidth=1)
    axes.text(-0.45, MACRO_F1_TARGET - 0.045, f"macro F1 target {MACRO_F1_TARGET}",
              ha="left", fontsize=8, color=GREY)

    axes.set_ylim(0, 1.08)
    axes.set_ylabel("Score")
    axes.set_title(f"{model_name} — overall test results")
    axes.grid(axis="y", alpha=0.25)

    return save(figure, f"{model_name}_overall_metrics")


def plot_confusion_matrix(metrics, model_name, normalise=False):
    matrix = metrics.get("confusion_matrix")
    labels = metrics.get("labels") or CLASSES

    if not matrix:
        return None

    matrix = np.array(matrix, dtype=float)

    if normalise:
        totals = matrix.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        shown = matrix / totals
        value_format = "{:.2f}"
        title = f"{model_name} — confusion matrix, share of each true class"
        suffix = "confusion_matrix_normalised"
    else:
        shown = matrix
        value_format = "{:.0f}"
        title = f"{model_name} — confusion matrix, clip counts"
        suffix = "confusion_matrix"

    names = short_names(labels)

    figure, axes = plt.subplots(figsize=(9, 7.5))
    image = axes.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() or 1)

    axes.set_xticks(range(len(names)))
    axes.set_yticks(range(len(names)))
    axes.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes.set_yticklabels(names, fontsize=9)
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("True class")
    axes.set_title(title)

    limit = (shown.max() or 1) / 2

    for row in range(shown.shape[0]):
        for column in range(shown.shape[1]):
            value = shown[row, column]
            axes.text(column, row, value_format.format(value), ha="center", va="center",
                      fontsize=8, color="white" if value > limit else "#17202a")

    figure.colorbar(image, ax=axes, shrink=0.8)

    return save(figure, f"{model_name}_{suffix}")


def plot_per_class_metrics(metrics, model_name):
    per_class = metrics.get("per_class")

    if not per_class:
        return None

    labels = [label for label in CLASSES if label in per_class]
    names = short_names(labels)

    precision = [per_class[label].get("precision", 0.0) for label in labels]
    recall = [per_class[label].get("recall", 0.0) for label in labels]
    f1 = [per_class[label].get("f1_score", 0.0) for label in labels]

    positions = np.arange(len(labels))
    width = 0.27

    figure, axes = plt.subplots(figsize=(10, 4.5))
    axes.bar(positions - width, precision, width, label="Precision", color=BLUE)
    axes.bar(positions, recall, width, label="Recall", color=GREEN)
    axes.bar(positions + width, f1, width, label="F1", color=ORANGE)

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("Score")
    axes.set_title(f"{model_name} — per-class performance on the test split")
    axes.legend(fontsize=9)
    axes.grid(axis="y", alpha=0.25)

    return save(figure, f"{model_name}_per_class_metrics")


# The five critical classes the SRS singles out, each against the 0.85 target.
def plot_critical_recall(metrics, model_name):
    per_class = metrics.get("per_class") or {}
    critical = metrics.get("critical_recall") or {}

    values = []
    labels = []

    for label in CRITICAL_CLASSES:
        if label in critical:
            values.append(float(critical[label]))
            labels.append(label)
        elif label in per_class:
            values.append(float(per_class[label].get("recall", 0.0)))
            labels.append(label)

    if not values:
        return None

    names = short_names(labels)

    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    bars = axes.bar(names, values, color=bar_colours(values, CRITICAL_RECALL_TARGET))

    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                  f"{value:.3f}", ha="center", fontsize=9)

    axes.axhline(CRITICAL_RECALL_TARGET, color=GREY, linestyle="--", linewidth=1.2)
    axes.text(-0.45, CRITICAL_RECALL_TARGET + 0.012,
              f"target {CRITICAL_RECALL_TARGET}", ha="left", fontsize=8, color=GREY)

    axes.set_ylim(0, 1.08)
    axes.set_ylabel("Recall")
    axes.set_title(f"{model_name} — recall on the critical classes")
    axes.grid(axis="y", alpha=0.25)
    plt.setp(axes.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    return save(figure, f"{model_name}_critical_recall")


def plot_validation_vs_test(metrics, model_name):
    validation = metrics.get("validation")

    if not validation:
        return None

    labels = ["Accuracy", "Macro F1", "F1 score"]
    val_values = [validation.get("accuracy", 0.0), validation.get("macro_f1", 0.0),
                  validation.get("f1_score", 0.0)]
    test_values = [metrics.get("accuracy", 0.0), metrics.get("macro_f1", 0.0),
                   metrics.get("f1_score", 0.0)]

    positions = np.arange(len(labels))
    width = 0.35

    figure, axes = plt.subplots(figsize=(6.5, 4))
    axes.bar(positions - width / 2, val_values, width, label="Validation", color=BLUE)
    axes.bar(positions + width / 2, test_values, width, label="Test", color=GREEN)

    for position, value in zip(positions - width / 2, val_values):
        axes.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)

    for position, value in zip(positions + width / 2, test_values):
        axes.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylim(0, 1.08)
    axes.set_ylabel("Score")
    axes.set_title(f"{model_name} — validation against test, to show generalisation")
    axes.legend(fontsize=9)
    axes.grid(axis="y", alpha=0.25)

    return save(figure, f"{model_name}_validation_vs_test")


# Only the CNN saves a per-epoch history.
def plot_training_curves(metrics, model_name):
    history = metrics.get("history")

    if not history or "loss" not in history:
        return None

    epochs = range(1, len(history["loss"]) + 1)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    left.plot(epochs, history["loss"], label="Training", color=BLUE)
    if "val_loss" in history:
        left.plot(epochs, history["val_loss"], label="Validation", color=ORANGE)
    left.set_xlabel("Epoch")
    left.set_ylabel("Loss")
    left.set_title("Loss per epoch")
    left.legend(fontsize=9)
    left.grid(alpha=0.25)

    if "accuracy" in history:
        right.plot(epochs, history["accuracy"], label="Training", color=BLUE)
    if "val_accuracy" in history:
        right.plot(epochs, history["val_accuracy"], label="Validation", color=ORANGE)

    right.axhline(ACCURACY_TARGET, color=GREY, linestyle="--", linewidth=1)
    right.text(1, ACCURACY_TARGET + 0.012,
               f"target {ACCURACY_TARGET}", ha="left", fontsize=8, color=GREY)
    right.set_xlabel("Epoch")
    right.set_ylabel("Accuracy")
    right.set_title("Accuracy per epoch")
    right.legend(fontsize=9)
    right.grid(alpha=0.25)

    figure.suptitle(f"{model_name} — training history")

    return save(figure, f"{model_name}_training_curves")


# Evidence of the hyperparameter tuning pass: what each candidate scored.
def plot_search_results(metrics, model_name):
    results = metrics.get("search_results")

    if not results:
        return None

    scores = [float(row.get("val_accuracy", 0.0)) for row in results]
    labels = [f"#{number}" for number in range(1, len(results) + 1)]

    best = max(scores) if scores else 0.0
    colours = [GREEN if score == best else GREY for score in scores]

    figure, axes = plt.subplots(figsize=(7.5, 4))
    bars = axes.bar(labels, scores, color=colours)

    for bar, score in zip(bars, scores):
        axes.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
                  f"{score:.3f}", ha="center", fontsize=8)

    axes.set_ylim(0, 1.05)
    axes.set_xlabel("Candidate")
    axes.set_ylabel("Best validation accuracy")
    axes.set_title(f"{model_name} — hyperparameter search, scored on validation")
    axes.grid(axis="y", alpha=0.25)

    return save(figure, f"{model_name}_hyperparameter_search")


def plot_noise_robustness(metrics, model_name):
    robustness = metrics.get("noise_robustness")

    if not robustness:
        return None

    labels = list(robustness.keys())
    values = [float(robustness[label]) for label in labels]

    clean = metrics.get("accuracy")

    figure, axes = plt.subplots(figsize=(7.5, 4))

    if clean is not None:
        labels = ["no added noise"] + labels
        values = [float(clean)] + values

    axes.plot(range(len(values)), values, marker="o", color=BLUE)

    for index, value in enumerate(values):
        axes.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)

    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels([label.replace("gaussian noise ", "") for label in labels],
                         rotation=20, ha="right", fontsize=9)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("Accuracy")
    axes.set_title(f"{model_name} — accuracy as noise is added to the test clips")
    axes.grid(alpha=0.25)

    return save(figure, f"{model_name}_noise_robustness")


# Compares every model that has a saved metrics file.
def plot_model_comparison():
    names = []
    accuracy = []
    macro_f1 = []

    for model_name, settings in PYTHON_MODELS.items():
        path = METRICS_DIR / settings["metrics_file"]

        if not os.path.exists(path):
            continue

        try:
            with open(path) as handle:
                metrics = json.load(handle)
        except (ValueError, OSError):
            continue

        names.append(model_name)
        accuracy.append(float(metrics.get("accuracy") or 0.0))
        macro_f1.append(float(metrics.get("macro_f1") or 0.0))

    if not names:
        print("no metrics files found, skipping the model comparison plot")
        return None

    order = np.argsort(accuracy)
    names = [names[i] for i in order]
    accuracy = [accuracy[i] for i in order]
    macro_f1 = [macro_f1[i] for i in order]

    positions = np.arange(len(names))
    height = 0.35

    figure, axes = plt.subplots(figsize=(8, 0.9 * len(names) + 2))
    axes.barh(positions + height / 2, accuracy, height, label="Test accuracy", color=GREEN)
    axes.barh(positions - height / 2, macro_f1, height, label="Test macro F1", color=BLUE)

    for position, value in zip(positions + height / 2, accuracy):
        axes.text(value + 0.008, position, f"{value:.3f}", va="center", fontsize=8)

    for position, value in zip(positions - height / 2, macro_f1):
        axes.text(value + 0.008, position, f"{value:.3f}", va="center", fontsize=8)

    axes.axvline(ACCURACY_TARGET, color=GREY, linestyle="--", linewidth=1)
    axes.text(ACCURACY_TARGET + 0.005, len(names) - 0.6,
              f"accuracy target {ACCURACY_TARGET}", fontsize=8, color=GREY)

    axes.set_yticks(positions)
    axes.set_yticklabels(names)
    axes.set_xlim(0, 1.1)
    axes.set_xlabel("Score")
    axes.set_title("Python models compared on the test split")
    axes.legend(fontsize=9, loc="lower right")
    axes.grid(axis="x", alpha=0.25)

    return save(figure, "model_comparison")


# Dataset composition, read from the statistics file prepare_dataset.py writes.
def plot_dataset_distribution():
    path = METADATA_DIR / "dataset_statistics.csv"

    if not os.path.exists(path):
        print("no dataset_statistics.csv found, skipping the dataset plot")
        return None

    import csv

    labels = []
    training = []
    validation = []
    testing = []

    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                labels.append(row["class_label"])
                training.append(int(row["training"]))
                validation.append(int(row["validation"]))
                testing.append(int(row["testing"]))
    except (ValueError, OSError, KeyError):
        return None

    if not labels:
        return None

    names = short_names(labels)
    positions = np.arange(len(labels))

    figure, axes = plt.subplots(figsize=(10, 4.5))
    axes.bar(names, training, label="Training", color=GREEN)
    axes.bar(names, validation, bottom=training, label="Validation", color=BLUE)
    bottom = [a + b for a, b in zip(training, validation)]
    axes.bar(names, testing, bottom=bottom, label="Testing", color=ORANGE)

    for position, total in zip(positions, [a + b for a, b in zip(bottom, testing)]):
        axes.text(position, total + 3, str(total), ha="center", fontsize=8)

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    axes.set_ylabel("Original clips")
    axes.set_title("Dataset composition by class and split")
    axes.legend(fontsize=9)
    axes.grid(axis="y", alpha=0.25)

    return save(figure, "dataset_distribution")


# One example spectrogram per class, taken from the cached CNN inputs. Useful in
# the report to show what the network actually sees.
def plot_class_spectrograms():
    path = METADATA_DIR / "train_spectrograms.npz"

    if not os.path.exists(path):
        print("no cached spectrograms found, skipping the example spectrogram plot")
        return None

    try:
        data = np.load(path, allow_pickle=True)
        images, labels = data["X"], data["y"]
    except Exception:
        return None

    figure, axes_grid = plt.subplots(2, 5, figsize=(14, 5))

    for index, class_label in enumerate(CLASSES):
        axes = axes_grid[index // 5][index % 5]
        matches = np.where(labels == class_label)[0]

        if len(matches) == 0:
            axes.axis("off")
            continue

        axes.imshow(images[matches[0]][:, :, 0], origin="lower", aspect="auto", cmap="magma")
        axes.set_title(display_name(class_label), fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])

    figure.suptitle("Log-mel spectrogram the CNN reads, one example per class")
    figure.tight_layout()

    return save(figure, "example_spectrograms")


# Called at the end of a training run. A failing figure must never fail the run,
# so each one is guarded.
def make_plots(metrics, model_name):
    made = []

    jobs = [
        plot_overall_metrics,
        plot_per_class_metrics,
        plot_critical_recall,
        plot_validation_vs_test,
        plot_training_curves,
        plot_search_results,
        plot_noise_robustness,
    ]

    for job in jobs:
        try:
            path = job(metrics, model_name)
        except Exception as error:
            print(f"could not draw {job.__name__}: {error}")
            continue

        if path:
            made.append(path)

    for job in (plot_confusion_matrix,):
        for normalise in (False, True):
            try:
                path = job(metrics, model_name, normalise)
            except Exception as error:
                print(f"could not draw {job.__name__}: {error}")
                continue

            if path:
                made.append(path)

    for job in (plot_model_comparison, plot_dataset_distribution):
        try:
            path = job()
        except Exception as error:
            print(f"could not draw {job.__name__}: {error}")
            continue

        if path:
            made.append(path)

    print(f"\n{len(made)} plot(s) written to {PLOT_DIR}")
    return made


def load_metrics_file(model_name):
    settings = PYTHON_MODELS.get(model_name)

    if not settings:
        return None

    path = METRICS_DIR / settings["metrics_file"]

    if not os.path.exists(path):
        return None

    try:
        with open(path) as handle:
            return json.load(handle)
    except (ValueError, OSError):
        return None


# Regenerates every figure from the saved metrics files, without retraining.
def make_all_plots():
    made = []

    for model_name in PYTHON_MODELS:
        metrics = load_metrics_file(model_name)

        if metrics is None:
            print(f"no metrics file for {model_name}, skipping")
            continue

        print(f"\n--- {model_name} ---")
        made.extend(make_plots(metrics, model_name))

    for job in (plot_class_spectrograms,):
        try:
            path = job()
        except Exception as error:
            print(f"could not draw {job.__name__}: {error}")
            continue

        if path:
            made.append(path)

    return made


if __name__ == "__main__":
    made = make_all_plots()
    print(f"\ntotal figures: {len(made)}")
    print(f"folder: {PLOT_DIR}")
