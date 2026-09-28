"""Train a classifier head on frozen YAMNet embeddings.

Run with --extract once to build the per-segment embedding cache, then rerun
without it to reuse that cache. Every candidate is fitted on the training split
only, validation picks the winner, and the test split is scored once, after the
choice has been made. Nothing already in python_models/ is overwritten: this
writes its own yamnet_transfer artifacts and leaves the Keras yamnet head from
src/yamnet_training.py alone.

    python src/yamnet_transfer.py --extract      build the cache, then train
    python src/yamnet_transfer.py --extract-only build the cache and stop
    python src/yamnet_transfer.py                train from the cache
    python src/yamnet_transfer.py --activate     train, then make it the active
                                                 Python model
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.preprocessing import prepare_segments
from audio_preprocessing.validation import ValidationStatus, validate_audio_signal
from config.config import (
    CLASSES,
    LABEL_ENCODER_FILE,
    METRICS_DIR,
    PROJECT_ROOT,
    PYTHON_MODEL_DIR,
    PYTHON_MODELS,
    display_name,
)
from feature_extraction.embeddings import (
    EMBEDDING_LENGTH,
    EMBEDDING_MEAN_LENGTH,
    FEATURE_VERSION,
    YAMNET_DIR,
    build_embedding,
    download_yamnet,
    load_yamnet,
    yamnet_is_downloaded,
)
from src.training import (
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    list_audio_files,
    split_metrics,
)

MODEL_NAME = "yamnet_transfer"
SEED = 7
PROGRESS_EVERY = 100

RUN_DIR = PROJECT_ROOT / "data" / MODEL_NAME

# One small .npy per segment. That is thousands of tiny writes, which is slow on
# a network filesystem such as a mounted Google Drive, so the cache can be sent
# to fast local storage while the outputs below stay with the project.
CACHE_DIR = Path(os.environ.get("YAMNET_TRANSFER_CACHE") or RUN_DIR / "cache")
FEATURES_FILE = RUN_DIR / "features.npz"
MANIFEST_FILE = RUN_DIR / "manifest.json"
EXCLUDED_FILE = RUN_DIR / "excluded.json"
VALIDATION_FILE = RUN_DIR / "validation_results.json"
SELECTION_FILE = PYTHON_MODEL_DIR / "selection.json"
SELECTION_BACKUP = PYTHON_MODEL_DIR / "selection_before_transfer.json"

# The dataset is already split on disk, so these are read, never re-split.
SPLIT_DIRS = {"train": TRAIN_DIR, "validation": VAL_DIR, "test": TEST_DIR}


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def file_hash(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def collect_sources():
    """Every source file in the three split folders, with the guards that make
    the later numbers trustworthy."""
    rows = []

    for split_name, split_dir in SPLIT_DIRS.items():
        for class_label in CLASSES:
            class_dir = os.path.join(split_dir, class_label)
            filenames = list_audio_files(class_dir)

            if not filenames:
                raise ValueError(
                    f"No readable audio for '{class_label}' in the {split_name} split "
                    f"({class_dir}). All ten classes must be present in every split."
                )

            for filename in filenames:
                path = os.path.join(class_dir, filename)
                rows.append({
                    "source_path": path,
                    "filename": filename,
                    "class_label": class_label,
                    "dataset_split": split_name,
                    "source_hash": file_hash(path),
                })

    # The same audio under two labels would make the labels meaningless.
    labels_by_hash = {}
    for row in rows:
        previous = labels_by_hash.setdefault(row["source_hash"], row["class_label"])

        if previous != row["class_label"]:
            raise ValueError(
                f"The same audio is labelled both '{previous}' and "
                f"'{row['class_label']}': {row['source_path']}"
            )

    # The same audio either side of a split boundary would leak the answer.
    splits_by_hash = {}
    for row in rows:
        splits_by_hash.setdefault(row["source_hash"], set()).add(row["dataset_split"])

    leaked = [h for h, names in splits_by_hash.items() if len(names) > 1]

    if leaked:
        examples = [r["source_path"] for r in rows if r["source_hash"] in leaked[:3]]
        raise ValueError(
            f"{len(leaked)} source file(s) appear in more than one split. "
            f"For example: {examples[:4]}"
        )

    return rows


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def require_yamnet():
    """Fail here rather than after grinding through every file: without YAMNet
    each embedding comes back as None and the whole run is wasted."""
    if not yamnet_is_downloaded():
        print("downloading YAMNet...", flush=True)

        try:
            download_yamnet()
        except ImportError as error:
            raise SystemExit(
                "YAMNet needs TensorFlow to download: %s\n"
                "Install the pinned dependencies first: pip install -r requirements.txt" % error
            )

    # Present on disk is not the same as loadable, so it is actually loaded.
    if load_yamnet() is None:
        raise SystemExit(
            "YAMNet is on disk at %s but could not be loaded, so no embeddings can be\n"
            "produced. This usually means TensorFlow is missing from the environment:\n"
            "    pip install -r requirements.txt\n"
            "Note that the pinned tensorflow==2.16.2 has no wheel for Python %s."
            % (YAMNET_DIR, platform.python_version())
        )


def cache_key(source_hash, segment_index):
    raw = f"{FEATURE_VERSION}:{source_hash}:{segment_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def extract():
    os.makedirs(CACHE_DIR, exist_ok=True)
    require_yamnet()

    rows = collect_sources()
    print(f"{len(rows)} source files across {len(SPLIT_DIRS)} splits", flush=True)

    features, labels, splits, audio_ids, manifest, excluded = [], [], [], [], [], []
    reused = 0

    for number, row in enumerate(rows, 1):
        try:
            segments, sr, _ = prepare_segments(row["source_path"])
        except Exception as error:
            excluded.append({"filename": row["source_path"], "reason": str(error)})
            continue

        if not segments:
            excluded.append({"filename": row["source_path"], "reason": "no usable audio"})
            continue

        for index, segment in enumerate(segments):
            check = validate_audio_signal(np.asarray(segment))

            if check.status == ValidationStatus.INVALID:
                excluded.append({
                    "filename": f"{row['source_path']}#{index}",
                    "reason": check.message,
                })
                continue

            path = CACHE_DIR / f"{cache_key(row['source_hash'], index)}.npy"

            if path.exists():
                vector = np.load(path, allow_pickle=False)
                reused += 1
            else:
                vector = build_embedding(segment, sr)

                if vector is None:
                    excluded.append({
                        "filename": f"{row['source_path']}#{index}",
                        "reason": "YAMNet is not available",
                    })
                    continue

                np.save(path, vector)

            features.append(vector)
            labels.append(row["class_label"])
            splits.append(row["dataset_split"])
            audio_ids.append(row["source_hash"])
            manifest.append(dict(row, segment_index=index))

        if number % PROGRESS_EVERY == 0 or number == len(rows):
            print(
                f"embeddings: {number}/{len(rows)} sources, {len(features)} segments",
                flush=True,
            )

    if not features:
        raise ValueError("No embeddings were produced, so there is nothing to train on.")

    os.makedirs(RUN_DIR, exist_ok=True)
    np.savez_compressed(
        FEATURES_FILE,
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels),
        split=np.asarray(splits),
        audio_id=np.asarray(audio_ids),
        feature_version=np.asarray(FEATURE_VERSION),
    )

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    EXCLUDED_FILE.write_text(json.dumps(excluded, indent=2), encoding="utf-8")

    counts = {name: splits.count(name) for name in SPLIT_DIRS}
    print(f"\nsegments per split: {counts}", flush=True)
    print(f"reused from cache:  {reused}", flush=True)
    print(f"excluded segments:  {len(excluded)}", flush=True)
    print("saved", FEATURES_FILE, flush=True)


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def candidates():
    """Each entry is (name, view, estimator). "mean" uses the mean half of the
    embedding, "combined" uses mean and max together."""
    for c in (0.1, 1, 10):
        yield f"logistic_c{c}", "mean", make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, class_weight="balanced"),
        )

    for c in (1, 10, 100):
        yield f"svm_mean_c{c}", "mean", make_pipeline(
            Normalizer(),
            SVC(C=c, gamma="scale", class_weight="balanced",
                probability=True, random_state=SEED),
        )

    for c in (1, 10, 100):
        yield f"svm_combined_c{c}", "combined", make_pipeline(
            StandardScaler(),
            SVC(C=c, gamma="scale", class_weight="balanced",
                probability=True, random_state=SEED),
        )


def view_of(x, view):
    return x[:, :EMBEDDING_MEAN_LENGTH] if view == "mean" else x


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def load_features():
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"{FEATURES_FILE} does not exist. Run with --extract first."
        )

    with np.load(FEATURES_FILE, allow_pickle=False) as source:
        x = source["x"]
        y = source["y"]
        split = source["split"]
        audio_id = source["audio_id"]
        cached_version = str(source["feature_version"])

    if cached_version != FEATURE_VERSION:
        raise ValueError(
            f"The cache was built with feature version '{cached_version}' but the code "
            f"is on '{FEATURE_VERSION}'. Re-run with --extract."
        )

    if x.ndim != 2 or x.shape[1] != EMBEDDING_LENGTH:
        raise ValueError(
            f"Expected embeddings of shape (n, {EMBEDDING_LENGTH}), got {x.shape}."
        )

    if not np.isfinite(x).all():
        raise ValueError("The cached embeddings contain non-finite values.")

    return x, y, split, audio_id


def check_splits(y, split, audio_id):
    masks = {name: split == name for name in SPLIT_DIRS}

    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"The {name} split has no segments.")

        missing = set(CLASSES) - set(y[mask])

        if missing:
            raise ValueError(f"The {name} split is missing: {sorted(missing)}")

    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = set(audio_id[masks[first]]) & set(audio_id[masks[second]])

        if shared:
            raise ValueError(
                f"{len(shared)} source file(s) appear in both the {first} and {second} "
                f"splits, so those scores would be inflated."
            )

    return masks


def train(activate=False):
    x, y, split, audio_id = load_features()
    masks = check_splits(y, split, audio_id)

    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    print(f"segments: " + ", ".join(
        f"{name} {int(mask.sum())}" for name, mask in masks.items()), flush=True)
    print(f"sources:  " + ", ".join(
        f"{name} {len(set(audio_id[mask]))}" for name, mask in masks.items()), flush=True)
    print()

    os.makedirs(RUN_DIR, exist_ok=True)

    results = {}
    best = None

    for name, view, model in candidates():
        values = view_of(x, view)

        print(f"training {name} ({view}) ...", flush=True)
        model.fit(values[masks["train"]], y[masks["train"]])

        # Scored the way the application predicts: argmax over the probabilities.
        probabilities = model.predict_proba(values[masks["validation"]])
        predicted = model.classes_[probabilities.argmax(axis=1)]

        validation = split_metrics(
            encoder.transform(y[masks["validation"]]),
            encoder.transform(predicted),
            encoder,
        )
        results[name] = {"view": view, **validation}

        print(
            f"  validation accuracy {validation['accuracy']:.4f}  "
            f"macro f1 {validation['macro_f1']:.4f}",
            flush=True,
        )

        if best is None or validation["macro_f1"] > best[0]:
            best = (validation["macro_f1"], name, view, model)

        VALIDATION_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    _, name, view, model = best
    print(f"\nwinner on validation macro f1: {name} ({view})", flush=True)

    # The test split is touched here for the first time, after the choice.
    values = view_of(x, view)
    probabilities = model.predict_proba(values[masks["test"]])
    predicted = model.classes_[probabilities.argmax(axis=1)]

    test = split_metrics(
        encoder.transform(y[masks["test"]]),
        encoder.transform(predicted),
        encoder,
    )

    save_artifacts(model, name, view, encoder, results[name], test, masks, audio_id)

    if activate:
        activate_model(name, view)

    print("\ntest results")
    print(f"  accuracy  {test['accuracy']}")
    print(f"  macro f1  {test['macro_f1']}")
    print("  critical-class recall")

    for class_label, value in test["critical_recall"].items():
        flag = "" if value >= 0.85 else "   <- below the 0.85 target"
        print(f"    {display_name(class_label):24} {value}{flag}")

    return test


def save_artifacts(model, name, view, encoder, validation, test, masks, audio_id):
    os.makedirs(PYTHON_MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    settings = PYTHON_MODELS[MODEL_NAME]

    # Carried on the model so inference slices the embedding the same way.
    model.feature_version_ = FEATURE_VERSION
    model.feature_view_ = view

    model_path = PYTHON_MODEL_DIR / settings["model_file"]
    joblib.dump(model, model_path)
    print("saved model to", model_path, flush=True)

    encoder_path = PYTHON_MODEL_DIR / LABEL_ENCODER_FILE
    if not encoder_path.exists():
        joblib.dump(encoder, encoder_path)
        print("saved label encoder to", encoder_path, flush=True)

    report = {
        "model": MODEL_NAME,
        "version": settings["version"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": name,
        "feature_view": view,
        "feature_version": FEATURE_VERSION,
        "feature_vector_length": EMBEDDING_LENGTH if view == "combined" else EMBEDDING_MEAN_LENGTH,
        "labels": list(encoder.classes_),
        "pretrained_model": "https://tfhub.dev/google/yamnet/1",
        "selection_metric": "validation macro_f1",
        "prediction_rule": "probability argmax",
        "augmentation": "none; original segments only",
        "seed": SEED,
        "evaluated_on": "test split",
        "segment_counts": {name_: int(mask.sum()) for name_, mask in masks.items()},
        "source_counts": {name_: len(set(audio_id[mask])) for name_, mask in masks.items()},
        "feature_archive_sha256": file_hash(FEATURES_FILE),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "validation": validation,
    }
    report.update(test)

    metrics_path = METRICS_DIR / settings["metrics_file"]
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved metrics to", metrics_path, flush=True)


def activate_model(name, view):
    if SELECTION_FILE.exists() and not SELECTION_BACKUP.exists():
        SELECTION_BACKUP.write_bytes(SELECTION_FILE.read_bytes())
        print("backed up the previous selection to", SELECTION_BACKUP, flush=True)

    SELECTION_FILE.write_text(json.dumps({
        "selected": MODEL_NAME,
        "candidate": name,
        "feature_view": view,
        "feature_version": FEATURE_VERSION,
        "selection_metric": "validation macro_f1",
        "selected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    print(f"\n'{MODEL_NAME}' is now the active Python model ({SELECTION_FILE}).", flush=True)
    print("delete that file to fall back to ACTIVE_PYTHON_MODEL in config.py.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true",
                        help="build the embedding cache, then train")
    parser.add_argument("--extract-only", action="store_true",
                        help="build the embedding cache and stop")
    parser.add_argument("--activate", action="store_true",
                        help="make the trained model the active Python model")
    args = parser.parse_args()

    if args.extract or args.extract_only:
        extract()

    if not args.extract_only:
        train(args.activate)
