import json
import os
import random
import sys
from datetime import datetime

import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    CLASSES,
    LABEL_ENCODER_FILE,
    METADATA_DIR,
    METRICS_DIR,
    PYTHON_MODELS,
    PYTHON_MODEL_DIR,
)
from feature_extraction.spectrogram import INPUT_SHAPE, spectrogram_from_file
from src.training import (
    AUGMENTED_TRAIN_DIR,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    describe_dataset,
    list_audio_files,
    split_metrics,
)

MODEL_NAME = "cnn"

# Spectrograms are large, so they are cached as .npy next to the feature caches.
# A disconnect then costs nothing but the training time.
CACHE_DIR = METADATA_DIR

PROGRESS_EVERY = 200

# The search space for the tuning pass. Kept small on purpose: every candidate
# trains a network, so a wide grid would not finish on a free Colab runtime.
SEARCH_SPACE = {
    "base_filters": [16, 32],
    "dropout": [0.2, 0.3, 0.4],
    "dense_units": [128, 256],
    "learning_rate": [0.0003, 0.001, 0.003],
    "batch_size": [32, 64],
}

SEARCH_CANDIDATES = 8
TUNING_EPOCHS = 18
FINAL_EPOCHS = 60
SEED = 42


def cache_path(name):
    return CACHE_DIR / f"{name}_spectrograms.npz"


def save_cache(name, X, y):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path(name), X=X, y=y)
    print(f"cached {len(X)} spectrograms to {cache_path(name)}", flush=True)


def load_cache(name):
    path = cache_path(name)

    if not os.path.exists(path):
        return None, None

    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None, None

    X, y = data["X"], data["y"]

    if X.shape[1:] != INPUT_SHAPE:
        print(f"cache {path} has shape {X.shape[1:]}, expected {INPUT_SHAPE}, rebuilding")
        return None, None

    print(f"loaded {len(X)} cached spectrograms from {path}", flush=True)
    return X, y


def build_split(split_dir, extra_dir=None, cache_name=None, use_cache=True):
    if cache_name and use_cache:
        X, y = load_cache(cache_name)

        if X is not None:
            return X, y

    directories = [split_dir]

    if extra_dir is not None and os.path.isdir(extra_dir):
        directories.append(extra_dir)

    images = []
    labels = []

    for directory in directories:
        for class_label in CLASSES:
            class_dir = os.path.join(directory, class_label)
            files = list_audio_files(class_dir)

            if not files:
                print("missing or empty folder:", class_dir)
                continue

            done = 0

            for file_name in files:
                file_path = os.path.join(class_dir, file_name)

                try:
                    images.append(spectrogram_from_file(file_path))
                except Exception as error:
                    print("failed:", file_path, error)
                    continue

                labels.append(class_label)
                done += 1

                if done % PROGRESS_EVERY == 0:
                    print(f"  {class_label}: {done}/{len(files)}", flush=True)

            print(f"{class_label} -> {done}/{len(files)} done", flush=True)

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels)

    if cache_name and len(X):
        save_cache(cache_name, X, y)

    return X, y


def build_all(use_augmented=True, use_cache=True):
    print(f"\nbuilding training spectrograms from {TRAIN_DIR}")

    if use_augmented:
        print(f"including augmented clips from {AUGMENTED_TRAIN_DIR}")

    X_train, y_train = build_split(
        TRAIN_DIR,
        AUGMENTED_TRAIN_DIR if use_augmented else None,
        cache_name="train",
        use_cache=use_cache,
    )

    print(f"\nbuilding validation spectrograms from {VAL_DIR}")
    X_val, y_val = build_split(VAL_DIR, cache_name="val", use_cache=use_cache)

    print(f"\nbuilding test spectrograms from {TEST_DIR}")
    X_test, y_test = build_split(TEST_DIR, cache_name="test", use_cache=use_cache)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_keras():
    try:
        import keras
        return keras
    except ImportError:
        from tensorflow import keras
        return keras


# Four convolution blocks, then global pooling instead of a big flatten, which
# keeps the parameter count down on a small dataset. Batch normalisation after
# each convolution, and dropout that grows with depth.
def build_model(base_filters, dropout, dense_units, learning_rate):
    keras = get_keras()
    layers = keras.layers

    model = keras.Sequential(name="sonicsentinel_cnn")
    model.add(layers.Input(shape=INPUT_SHAPE))

    filters = base_filters

    for block in range(4):
        model.add(layers.Conv2D(filters, (3, 3), padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))

        model.add(layers.Conv2D(filters, (3, 3), padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))

        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(dropout * (0.5 + 0.25 * block)))

        filters = filters * 2

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout + 0.1))
    model.add(layers.Dense(len(CLASSES), activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def callbacks_for(patience, monitor="val_loss"):
    keras = get_keras()

    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=max(patience // 2, 2), min_lr=1e-6, verbose=1
        ),
    ]


# Rarer classes get a larger weight, the same idea as class_weight="balanced"
# in the scikit-learn models.
def class_weights_for(y_encoded):
    counts = np.bincount(y_encoded, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(CLASSES) * counts)
    return {index: float(value) for index, value in enumerate(weights)}


def sample_candidates(count=SEARCH_CANDIDATES):
    rng = random.Random(SEED)
    seen = set()
    candidates = []

    while len(candidates) < count and len(seen) < 10 * count:
        candidate = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
        key = tuple(sorted(candidate.items()))

        if key in seen:
            continue

        seen.add(key)
        candidates.append(candidate)

    return candidates


# The tuning pass. Every candidate is trained on the training split only and
# scored on the validation split; the test split is never touched here.
def tune(X_train, y_train, X_val, y_val, weights, candidates, epochs=TUNING_EPOCHS):
    results = []
    best = None
    best_accuracy = -1.0

    for number, candidate in enumerate(candidates, start=1):
        print("\n" + "-" * 70, flush=True)
        print(f"candidate {number}/{len(candidates)}: {candidate}", flush=True)
        print("-" * 70, flush=True)

        keras = get_keras()
        keras.utils.set_random_seed(SEED)

        model = build_model(
            candidate["base_filters"],
            candidate["dropout"],
            candidate["dense_units"],
            candidate["learning_rate"],
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=candidate["batch_size"],
            class_weight=weights,
            callbacks=callbacks_for(patience=5),
            verbose=2,
        )

        accuracy = float(max(history.history["val_accuracy"]))
        results.append({"params": candidate, "val_accuracy": round(accuracy, 4)})
        print(f"candidate {number} best validation accuracy: {accuracy:.4f}", flush=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = candidate

        del model
        keras.backend.clear_session()

    print("\nbest candidate:", best)
    print(f"best validation accuracy during tuning: {best_accuracy:.4f}")

    return best, best_accuracy, results


def save_model(model):
    os.makedirs(PYTHON_MODEL_DIR, exist_ok=True)
    path = PYTHON_MODEL_DIR / PYTHON_MODELS[MODEL_NAME]["model_file"]
    model.save(path)
    print("saved model to", path)


def save_metrics(y_test, test_predictions, y_val, val_predictions, encoder,
                 best_params, best_validation, search_results, epochs_run, history=None):
    os.makedirs(METRICS_DIR, exist_ok=True)

    settings = PYTHON_MODELS[MODEL_NAME]

    metrics = {
        "model": MODEL_NAME,
        "version": settings["version"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_shape": list(INPUT_SHAPE),
        "labels": list(encoder.classes_),
        "evaluated_on": "test split",
        "best_params": best_params or {},
        "best_validation_score": round(float(best_validation), 4),
        "search": (
            f"random search over {len(search_results)} of "
            f"{count_space()} combinations, selected on the validation split"
        ),
        "search_results": search_results,
        "epochs_run": epochs_run,
        # Per-epoch curves, so the training run can be plotted for the report.
        "history": history or {},
        "noise_robustness": {},
    }

    metrics.update(split_metrics(y_test, test_predictions, encoder))
    metrics["validation"] = split_metrics(y_val, val_predictions, encoder)

    path = METRICS_DIR / settings["metrics_file"]

    with open(path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nsaved metrics to", path)
    return metrics


def count_space():
    total = 1

    for values in SEARCH_SPACE.values():
        total = total * len(values)

    return total


def load_encoder():
    path = PYTHON_MODEL_DIR / LABEL_ENCODER_FILE

    # The same encoder the other models use, so the class ordering matches.
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(CLASSES)
    os.makedirs(PYTHON_MODEL_DIR, exist_ok=True)
    joblib.dump(encoder, path)
    print("wrote a new label encoder to", path)

    return encoder


def run_training(use_augmented=True, use_cache=True, candidates=SEARCH_CANDIDATES,
                 tuning_epochs=TUNING_EPOCHS, final_epochs=FINAL_EPOCHS):
    keras = get_keras()

    describe_dataset(use_augmented)

    X_train, y_train, X_val, y_val, X_test, y_test = build_all(use_augmented, use_cache)

    if not len(X_train) or not len(X_val) or not len(X_test):
        print("\nThe training, validation and test splits must all contain audio.")
        print("Prepare the dataset first:")
        print("  python data_preparation/prepare_dataset.py")
        print("  python augmentation/augmentation.py")
        return None

    print(f"\nspectrograms -> train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")
    print(f"input shape: {INPUT_SHAPE}")

    encoder = load_encoder()

    y_train_encoded = encoder.transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    weights = class_weights_for(y_train_encoded)

    print("\ntuning the network on the validation split")
    best_params, best_validation, search_results = tune(
        X_train, y_train_encoded, X_val, y_val_encoded, weights,
        sample_candidates(candidates), tuning_epochs,
    )

    print("\n" + "=" * 70)
    print("retraining the best candidate for longer")
    print("=" * 70)

    keras.utils.set_random_seed(SEED)

    model = build_model(
        best_params["base_filters"],
        best_params["dropout"],
        best_params["dense_units"],
        best_params["learning_rate"],
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train_encoded,
        validation_data=(X_val, y_val_encoded),
        epochs=final_epochs,
        batch_size=best_params["batch_size"],
        class_weight=weights,
        callbacks=callbacks_for(patience=10),
        verbose=2,
    )

    epochs_run = len(history.history["loss"])

    val_predictions = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print(f"\nvalidation accuracy: {float(np.mean(val_predictions == y_val_encoded)):.4f}")

    # The test split is scored once, here, and is never tuned against.
    test_predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"test accuracy: {float(np.mean(test_predictions == y_test_encoded)):.4f}")

    save_model(model)
    metrics = save_metrics(
        y_test_encoded, test_predictions, y_val_encoded, val_predictions,
        encoder, best_params, best_validation, search_results, epochs_run,
        history={key: [float(v) for v in values] for key, values in history.history.items()},
    )

    print("\ntest results")
    print(f"  accuracy  {metrics['accuracy']}")
    print(f"  macro f1  {metrics['macro_f1']}")
    print("  critical-class recall")

    for class_label, value in metrics["critical_recall"].items():
        flag = "" if value >= 0.85 else "   <- below the 0.85 target"
        print(f"    {class_label:24} {value}{flag}")

    try:
        from src import plots
        plots.make_plots(metrics, MODEL_NAME)
        plots.plot_class_spectrograms()
    except Exception as error:
        print("could not draw the plots:", error)

    return model
