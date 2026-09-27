import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    CLASSES,
    SUPPORTED_EXTENSIONS,
    CRITICAL_CLASSES,
    DATASET_DIR,
    LABEL_ENCODER_FILE,
    METADATA_DIR,
    METRICS_DIR,
    PYTHON_MODELS,
    PYTHON_MODEL_DIR,
    TARGET_SAMPLE_RATE,
    display_name,
)
from feature_extraction.features import (
    FEATURE_NAMES,
    extract_all_features,
    extract_features_from_samples,
    features_to_vector,
)

TRAIN_DIR = DATASET_DIR / "training"
AUGMENTED_TRAIN_DIR = DATASET_DIR / "augmented_training"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "testing"

MAX_CANDIDATES = 40

PROGRESS_EVERY = 50


def list_audio_files(class_dir):
    if not os.path.isdir(class_dir):
        return []

    return [
        f for f in sorted(os.listdir(class_dir))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]


def count_audio_files(directory):
    counts = {}

    for class_label in CLASSES:
        counts[class_label] = len(list_audio_files(os.path.join(directory, class_label)))

    return counts


def describe_dataset(use_augmented=True):
    splits = [
        ("training", TRAIN_DIR),
        ("augmented training", AUGMENTED_TRAIN_DIR),
        ("validation", VAL_DIR),
        ("testing", TEST_DIR),
    ]

    if not use_augmented:
        splits = [row for row in splits if row[0] != "augmented training"]

    print("Dataset on disk")
    print(f"  dataset folder: {DATASET_DIR}")
    print()
    print(f"  {'class':24}" + "".join(f"{name:>20}" for name, _ in splits))

    totals = {name: 0 for name, _ in splits}
    counts_by_split = {name: count_audio_files(path) for name, path in splits}

    for class_label in CLASSES:
        line = f"  {class_label:24}"

        for name, _ in splits:
            value = counts_by_split[name][class_label]
            totals[name] += value
            line += f"{value:>20}"

        print(line)

    print(f"  {'TOTAL':24}" + "".join(f"{totals[name]:>20}" for name, _ in splits))
    print()

    missing = [name for name, _ in splits if totals[name] == 0]

    for name in missing:
        print(f"  WARNING: the {name} split is empty.")

    return totals


def build_dataset(split_dir, extra_dir=None, cache_name=None, use_cache=True):
    if cache_name and use_cache:
        cached = load_cached_features(cache_name)

        if cached is not None:
            print(f"loaded {len(cached)} cached rows from {cache_path(cache_name)}")
            return cached

    rows = []
    labels = []

    directories = [split_dir]

    if extra_dir is not None and os.path.isdir(extra_dir):
        directories.append(extra_dir)

    for directory in directories:
        for class_label in CLASSES:
            class_dir = os.path.join(directory, class_label)

            if not os.path.isdir(class_dir):
                print("missing folder:", class_dir)
                continue

            files = list_audio_files(class_dir)

            done = 0

            for file_name in files:
                file_path = os.path.join(class_dir, file_name)

                try:
                    features = extract_all_features(file_path)
                except Exception as error:
                    print("failed:", file_path, error)
                    continue

                rows.append(features_to_vector(features))
                labels.append(class_label)
                done += 1

                if done % PROGRESS_EVERY == 0:
                    print(f"  {class_label}: {done}/{len(files)} files", flush=True)

            print(f"{class_label} -> {done}/{len(files)} files done", flush=True)

    frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
    frame["label"] = labels

    if cache_name and not frame.empty:
        save_cached_features(cache_name, frame)

    return frame


def cache_path(cache_name):
    return METADATA_DIR / f"{cache_name}_features.csv"


def save_cached_features(cache_name, frame):
    os.makedirs(METADATA_DIR, exist_ok=True)
    frame.to_csv(cache_path(cache_name), index=False)
    print(f"cached {len(frame)} rows to {cache_path(cache_name)}", flush=True)


def load_cached_features(cache_name):
    path = cache_path(cache_name)

    if not os.path.exists(path):
        return None

    try:
        frame = pd.read_csv(path)
    except Exception:
        return None

    expected = FEATURE_NAMES + ["label"]

    if list(frame.columns) != expected:
        print(f"cache {path} does not match the current features, rebuilding")
        return None

    if frame.empty:
        return None

    return frame


def prepare_features(train_df, val_df, test_df):
    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    X_train = train_df[FEATURE_NAMES].values
    y_train = encoder.transform(train_df["label"].values)

    X_val = val_df[FEATURE_NAMES].values
    y_val = encoder.transform(val_df["label"].values)

    X_test = test_df[FEATURE_NAMES].values
    y_test = encoder.transform(test_df["label"].values)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler, encoder


def evaluate(model, X, y, encoder, label):
    predictions = model.predict(X)

    print(f"\n{label} accuracy:", round(float(accuracy_score(y, predictions)), 4))
    print(f"{label} macro f1:", round(float(f1_score(y, predictions, average='macro', zero_division=0)), 4))
    print(classification_report(y, predictions, target_names=encoder.classes_, zero_division=0))

    return predictions


def count_candidates(param_grid):
    total = 1

    for values in param_grid.values():
        total = total * len(values)

    return total


def tune_on_validation(estimator, param_grid, X_train, y_train, X_val, y_val, scoring, n_jobs=-1):
    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])

    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val)),
    ])

    splitter = PredefinedSplit(test_fold)

    candidates = count_candidates(param_grid)

    if candidates > MAX_CANDIDATES:
        print(
            f"{candidates} parameter combinations is too many for one runtime, "
            f"sampling {MAX_CANDIDATES} of them at random instead"
        )
        search = RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=MAX_CANDIDATES,
            cv=splitter,
            scoring=scoring,
            refit=False,
            n_jobs=n_jobs,
            verbose=2,
            random_state=42,
        )
    else:
        print(f"searching all {candidates} parameter combinations")
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=splitter,
            scoring=scoring,
            refit=False,
            n_jobs=n_jobs,
            verbose=2,
        )

    search.fit(X_search, y_search)

    print("best params:", search.best_params_)
    print(f"best validation {scoring}:", round(float(search.best_score_), 4))

    # Refit the winning settings on the training split alone.
    model = clone(estimator).set_params(**search.best_params_)
    model.fit(X_train, y_train)

    return model, search.best_params_, float(search.best_score_)


def noise_robustness(model, scaler, encoder, test_dir=TEST_DIR, noise_levels=(0.005, 0.02, 0.05)):
    import librosa

    results = {}

    file_paths = []
    true_labels = []

    for class_label in CLASSES:
        class_dir = os.path.join(test_dir, class_label)

        if not os.path.isdir(class_dir):
            continue

        for file_name in list_audio_files(class_dir):
            file_paths.append(os.path.join(class_dir, file_name))
            true_labels.append(class_label)

    if not file_paths:
        return {}

    rng = np.random.default_rng(42)

    for level in noise_levels:
        vectors = []
        kept_labels = []

        for file_path, class_label in zip(file_paths, true_labels):
            try:
                samples, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
                noisy = samples + rng.normal(0, level, size=len(samples)).astype(np.float32)
                features = extract_features_from_samples(noisy, sr)
            except Exception:
                continue

            vectors.append(features_to_vector(features))
            kept_labels.append(class_label)

        if not vectors:
            continue

        predictions = model.predict(scaler.transform(vectors))
        accuracy = accuracy_score(encoder.transform(kept_labels), predictions)
        results[f"gaussian noise sd={level}"] = round(float(accuracy), 4)
        print(f"  noise sd={level}: accuracy {accuracy:.4f}", flush=True)

    return results


def split_metrics(y_true, y_pred, encoder):
    labels = list(range(len(encoder.classes_)))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=encoder.classes_,
        output_dict=True, zero_division=0,
    )

    per_class = {}
    for class_label in encoder.classes_:
        entry = report.get(class_label, {})
        per_class[class_label] = {
            "precision": round(entry.get("precision", 0.0), 4),
            "recall": round(entry.get("recall", 0.0), 4),
            "f1_score": round(entry.get("f1-score", 0.0), 4),
            "support": int(entry.get("support", 0)),
        }

    critical = {
        class_label: per_class[class_label]["recall"]
        for class_label in CRITICAL_CLASSES
        if class_label in per_class
    }

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": per_class,
        "critical_recall": critical,
    }


def save_metrics(
    model_name,
    y_test,
    test_predictions,
    encoder,
    y_val=None,
    val_predictions=None,
    best_params=None,
    best_validation_score=None,
    robustness=None,
    search_notes=None,
):
    os.makedirs(METRICS_DIR, exist_ok=True)

    settings = PYTHON_MODELS[model_name]

    metrics = {
        "model": model_name,
        "version": settings["version"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_vector_length": len(FEATURE_NAMES),
        "labels": list(encoder.classes_),
        "evaluated_on": "test split",
        "best_params": best_params or {},
        "best_validation_score": (
            round(float(best_validation_score), 4) if best_validation_score is not None else None
        ),
        "search": search_notes or "",
        "noise_robustness": robustness or {},
    }

    metrics.update(split_metrics(y_test, test_predictions, encoder))

    if y_val is not None and val_predictions is not None and len(y_val):
        metrics["validation"] = split_metrics(y_val, val_predictions, encoder)

    metrics_path = METRICS_DIR / settings["metrics_file"]

    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nsaved metrics to", metrics_path)

    print("\ntest results")
    print(f"  accuracy  {metrics['accuracy']}")
    print(f"  macro f1  {metrics['macro_f1']}")
    print("  critical-class recall")

    for class_label, value in metrics["critical_recall"].items():
        flag = "" if value >= 0.85 else "   <- below the 0.85 target"
        print(f"    {display_name(class_label):24} {value}{flag}")

    return metrics


def save_model(model_name, model, scaler, encoder):
    os.makedirs(PYTHON_MODEL_DIR, exist_ok=True)

    settings = PYTHON_MODELS[model_name]

    joblib.dump(model, PYTHON_MODEL_DIR / settings["model_file"])
    joblib.dump(scaler, PYTHON_MODEL_DIR / settings["scaler_file"])
    joblib.dump(encoder, PYTHON_MODEL_DIR / LABEL_ENCODER_FILE)

    print("saved model to", PYTHON_MODEL_DIR / settings["model_file"])
    print("saved scaler to", PYTHON_MODEL_DIR / settings["scaler_file"])
    print("saved label encoder to", PYTHON_MODEL_DIR / LABEL_ENCODER_FILE)


def build_all_splits(use_augmented=True, use_cache=True):
    print(f"\nbuilding training features from {TRAIN_DIR}")

    if use_augmented:
        print(f"including augmented training clips from {AUGMENTED_TRAIN_DIR}")

    train_df = build_dataset(
        TRAIN_DIR,
        AUGMENTED_TRAIN_DIR if use_augmented else None,
        cache_name="train",
        use_cache=use_cache,
    )

    print(f"\nbuilding validation features from {VAL_DIR}")
    val_df = build_dataset(VAL_DIR, cache_name="val", use_cache=use_cache)

    print(f"\nbuilding test features from {TEST_DIR}")
    test_df = build_dataset(TEST_DIR, cache_name="test", use_cache=use_cache)

    return train_df, val_df, test_df


def run_training(
    model_name,
    estimator,
    param_grid,
    scoring="f1_macro",
    use_augmented=True,
    use_cache=True,
    measure_robustness=True,
    n_jobs=-1,
):
    if model_name not in PYTHON_MODELS:
        raise ValueError(
            f"'{model_name}' is not in PYTHON_MODELS in config/config.py. "
            f"Known models: {', '.join(PYTHON_MODELS)}"
        )

    describe_dataset(use_augmented)

    train_df, val_df, test_df = build_all_splits(use_augmented, use_cache)

    if train_df.empty or val_df.empty or test_df.empty:
        print("\nThe training, validation and test splits must all contain audio.")
        print("Prepare the dataset first:")
        print("  python data_preparation/prepare_dataset.py")
        print("  python augmentation/augmentation.py")
        return None

    print(f"\nfeature rows -> train {len(train_df)}, validation {len(val_df)}, test {len(test_df)}")
    print(f"feature vector length: {len(FEATURE_NAMES)}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, encoder = prepare_features(
        train_df, val_df, test_df
    )

    candidates = count_candidates(param_grid)
    search_notes = (
        f"{'randomised search over ' + str(MAX_CANDIDATES) + ' of ' if candidates > MAX_CANDIDATES else 'full grid search over '}"
        f"{candidates} combinations, selected on the validation split"
    )

    print(f"\ntuning {model_name} on the validation split")
    model, best_params, best_score = tune_on_validation(
        estimator, param_grid, X_train, y_train, X_val, y_val, scoring, n_jobs
    )

    val_predictions = evaluate(model, X_val, y_val, encoder, "validation")

    test_predictions = evaluate(model, X_test, y_test, encoder, "test")

    robustness = {}

    if measure_robustness:
        print("\nmeasuring noise robustness on the test split")
        robustness = noise_robustness(model, scaler, encoder)

    save_model(model_name, model, scaler, encoder)
    metrics = save_metrics(
        model_name,
        y_test,
        test_predictions,
        encoder,
        y_val=y_val,
        val_predictions=val_predictions,
        best_params=best_params,
        best_validation_score=best_score,
        robustness=robustness,
        search_notes=search_notes,
    )

    # Figures for the report. A failing figure never fails the training run.
    try:
        from src import plots
        plots.make_plots(metrics, model_name)
    except Exception as error:
        print("could not draw the plots:", error)

    return model
