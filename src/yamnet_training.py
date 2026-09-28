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
from feature_extraction.embeddings import (
    EMBEDDING_LENGTH,
    embedding_from_file,
    yamnet_status,
)
from src.training import (
    AUGMENTED_TRAIN_DIR,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    describe_dataset,
    list_audio_files,
    split_metrics,
)

MODEL_NAME = "yamnet"

PROGRESS_EVERY = 200

# The head is small, so we can afford a few candidates.
SEARCH_SPACE = {
    "hidden_units": [128, 256, 512],
    "dropout": [0.2, 0.3, 0.5],
    "learning_rate": [0.0003, 0.001, 0.003],
    "batch_size": [32, 64],
}

SEARCH_CANDIDATES = 8
TUNING_EPOCHS = 30
FINAL_EPOCHS = 80
SEED = 42


def cache_path(name):
    return METADATA_DIR / f"{name}_embeddings.npz"


def save_cache(name, X, y):
    os.makedirs(METADATA_DIR, exist_ok=True)
    np.savez_compressed(cache_path(name), X=X, y=y)
    print("cached", len(X), "embeddings to", cache_path(name), flush=True)


def load_cache(name):
    path = cache_path(name)

    if not os.path.exists(path):
        return None, None

    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None, None

    X = data["X"]
    y = data["y"]

    # Rebuild if the embedding length changed.
    if X.shape[1] != EMBEDDING_LENGTH:
        print("cache", path, "has the wrong length, rebuilding")
        return None, None

    print("loaded", len(X), "cached embeddings from", path, flush=True)
    return X, y


def build_split(split_dir, extra_dir=None, cache_name=None, use_cache=True):
    if cache_name and use_cache:
        X, y = load_cache(cache_name)

        if X is not None:
            return X, y

    directories = [split_dir]

    if extra_dir is not None and os.path.isdir(extra_dir):
        directories.append(extra_dir)

    rows = []
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
                    embedding = embedding_from_file(file_path)
                except Exception as error:
                    print("failed:", file_path, error)
                    continue

                if embedding is None:
                    print("YAMNet is not available, stopping")
                    return np.asarray([]), np.asarray([])

                rows.append(embedding)
                labels.append(class_label)
                done = done + 1

                if done % PROGRESS_EVERY == 0:
                    print(" ", class_label, done, "/", len(files), flush=True)

            print(class_label, "->", done, "/", len(files), "done", flush=True)

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels)

    if cache_name and len(X):
        save_cache(cache_name, X, y)

    return X, y


def build_all(use_augmented=True, use_cache=True):
    print("\nbuilding training embeddings from", TRAIN_DIR)

    if use_augmented:
        print("including augmented clips from", AUGMENTED_TRAIN_DIR)

    X_train, y_train = build_split(
        TRAIN_DIR,
        AUGMENTED_TRAIN_DIR if use_augmented else None,
        cache_name="train",
        use_cache=use_cache,
    )

    print("\nbuilding validation embeddings from", VAL_DIR)
    X_val, y_val = build_split(VAL_DIR, cache_name="val", use_cache=use_cache)

    print("\nbuilding test embeddings from", TEST_DIR)
    X_test, y_test = build_split(TEST_DIR, cache_name="test", use_cache=use_cache)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_keras():
    try:
        import keras
        return keras
    except ImportError:
        from tensorflow import keras
        return keras


# YAMNet stays frozen, so we only train this small head on top.
def build_model(hidden_units, dropout, learning_rate):
    keras = get_keras()
    layers = keras.layers

    model = keras.Sequential(name="sonicsentinel_yamnet_head")
    model.add(layers.Input(shape=(EMBEDDING_LENGTH,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(hidden_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(hidden_units // 2, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(CLASSES), activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def callbacks_for(patience):
    keras = get_keras()

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(patience // 2, 2), min_lr=1e-6, verbose=1
        ),
    ]


# Rarer classes get a bigger weight.
def class_weights_for(y_encoded):
    counts = np.bincount(y_encoded, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(CLASSES) * counts)

    result = {}

    for index in range(len(CLASSES)):
        result[index] = float(weights[index])

    return result


def count_space():
    total = 1

    for values in SEARCH_SPACE.values():
        total = total * len(values)

    return total


def sample_candidates(count=SEARCH_CANDIDATES):
    rng = random.Random(SEED)
    seen = []
    candidates = []
    tries = 0

    while len(candidates) < count and tries < 10 * count:
        tries = tries + 1
        candidate = {}

        for key in SEARCH_SPACE:
            candidate[key] = rng.choice(SEARCH_SPACE[key])

        key = str(sorted(candidate.items()))

        if key in seen:
            continue

        seen.append(key)
        candidates.append(candidate)

    return candidates


# Candidates are scored on validation, never on test.
def tune(X_train, y_train, X_val, y_val, weights, candidates, epochs):
    keras = get_keras()
    results = []
    best = None
    best_accuracy = -1.0

    number = 0

    for candidate in candidates:
        number = number + 1
        print("\n" + "-" * 70, flush=True)
        print("candidate", number, "of", len(candidates), ":", candidate, flush=True)
        print("-" * 70, flush=True)

        keras.utils.set_random_seed(SEED)

        model = build_model(
            candidate["hidden_units"],
            candidate["dropout"],
            candidate["learning_rate"],
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=candidate["batch_size"],
            class_weight=weights,
            callbacks=callbacks_for(patience=6),
            verbose=2,
        )

        accuracy = float(max(history.history["val_accuracy"]))
        results.append({"params": candidate, "val_accuracy": round(accuracy, 4)})
        print("candidate", number, "best validation accuracy:", round(accuracy, 4), flush=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = candidate

        del model
        keras.backend.clear_session()

    print("\nbest candidate:", best)
    print("best validation accuracy during tuning:", round(best_accuracy, 4))

    return best, best_accuracy, results


# Same noise test the other models get, but on embeddings.
def noise_robustness(model, encoder, levels=(0.005, 0.02, 0.05)):
    import librosa

    from feature_extraction.embeddings import build_embedding

    file_paths = []
    true_labels = []

    for class_label in CLASSES:
        class_dir = os.path.join(TEST_DIR, class_label)

        for file_name in list_audio_files(class_dir):
            file_paths.append(os.path.join(class_dir, file_name))
            true_labels.append(class_label)

    if not file_paths:
        return {}

    results = {}
    rng = np.random.default_rng(SEED)

    for level in levels:
        rows = []
        kept = []

        for file_path, class_label in zip(file_paths, true_labels):
            try:
                samples, sr = librosa.load(file_path, sr=16000, mono=True)
                noisy = samples + rng.normal(0, level, size=len(samples)).astype(np.float32)
                embedding = build_embedding(noisy, sr)
            except Exception:
                continue

            if embedding is None:
                continue

            rows.append(embedding)
            kept.append(class_label)

        if not rows:
            continue

        predictions = np.argmax(model.predict(np.asarray(rows), verbose=0), axis=1)
        accuracy = float(np.mean(predictions == encoder.transform(kept)))
        results["gaussian noise sd=" + str(level)] = round(accuracy, 4)
        print("  noise sd=", level, "accuracy", round(accuracy, 4), flush=True)

    return results


def save_model(model):
    os.makedirs(PYTHON_MODEL_DIR, exist_ok=True)
    path = PYTHON_MODEL_DIR / PYTHON_MODELS[MODEL_NAME]["model_file"]
    model.save(path)
    print("saved model to", path)


def load_encoder():
    path = PYTHON_MODEL_DIR / LABEL_ENCODER_FILE

    # Reuse the shared encoder so the class order matches the other models.
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


def save_metrics(y_test, test_predictions, y_val, val_predictions, encoder,
                 best_params, best_validation, search_results, epochs_run,
                 history, robustness):
    os.makedirs(METRICS_DIR, exist_ok=True)

    settings = PYTHON_MODELS[MODEL_NAME]

    metrics = {
        "model": MODEL_NAME,
        "version": settings["version"],
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_length": EMBEDDING_LENGTH,
        "base_model": "YAMNet, frozen",
        "labels": list(encoder.classes_),
        "evaluated_on": "test split",
        "best_params": best_params or {},
        "best_validation_score": round(float(best_validation), 4),
        "search": "random search over " + str(len(search_results)) + " of " + str(count_space()) + " combinations, selected on the validation split",
        "search_results": search_results,
        "epochs_run": epochs_run,
        "history": history or {},
        "noise_robustness": robustness or {},
    }

    metrics.update(split_metrics(y_test, test_predictions, encoder))
    metrics["validation"] = split_metrics(y_val, val_predictions, encoder)

    path = METRICS_DIR / settings["metrics_file"]

    with open(path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nsaved metrics to", path)
    return metrics


def run_training(use_augmented=True, use_cache=True, candidates=SEARCH_CANDIDATES,
                 tuning_epochs=TUNING_EPOCHS, final_epochs=FINAL_EPOCHS,
                 measure_robustness=True):
    keras = get_keras()

    status = yamnet_status()
    print(status["message"])

    describe_dataset(use_augmented)

    X_train, y_train, X_val, y_val, X_test, y_test = build_all(use_augmented, use_cache)

    if not len(X_train) or not len(X_val) or not len(X_test):
        print("\nThe training, validation and test splits must all contain audio.")
        print("Prepare the dataset first:")
        print("  python data_preparation/prepare_dataset.py")
        print("  python augmentation/augmentation.py")
        return None

    print("\nembeddings -> train", len(X_train), "validation", len(X_val), "test", len(X_test))
    print("embedding length:", EMBEDDING_LENGTH)

    encoder = load_encoder()

    y_train_encoded = encoder.transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    weights = class_weights_for(y_train_encoded)

    print("\ntuning the head on the validation split")
    best_params, best_validation, search_results = tune(
        X_train, y_train_encoded, X_val, y_val_encoded, weights,
        sample_candidates(candidates), tuning_epochs,
    )

    print("\n" + "=" * 70)
    print("retraining the best candidate for longer")
    print("=" * 70)

    keras.utils.set_random_seed(SEED)

    model = build_model(
        best_params["hidden_units"],
        best_params["dropout"],
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
        callbacks=callbacks_for(patience=12),
        verbose=2,
    )

    epochs_run = len(history.history["loss"])

    val_predictions = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nvalidation accuracy:", round(float(np.mean(val_predictions == y_val_encoded)), 4))

    # The test split is scored once here.
    test_predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("test accuracy:", round(float(np.mean(test_predictions == y_test_encoded)), 4))

    robustness = {}

    if measure_robustness:
        print("\nmeasuring noise robustness on the test split")
        robustness = noise_robustness(model, encoder)

    save_model(model)

    plain_history = {}

    for key in history.history:
        plain_history[key] = [float(value) for value in history.history[key]]

    metrics = save_metrics(
        y_test_encoded, test_predictions, y_val_encoded, val_predictions, encoder,
        best_params, best_validation, search_results, epochs_run,
        plain_history, robustness,
    )

    print("\ntest results")
    print("  accuracy ", metrics["accuracy"])
    print("  macro f1 ", metrics["macro_f1"])
    print("  critical-class recall")

    for class_label in metrics["critical_recall"]:
        value = metrics["critical_recall"][class_label]
        flag = "" if value >= 0.85 else "   <- below the 0.85 target"
        print("   ", class_label, value, flag)

    try:
        from src import plots
        plots.make_plots(metrics, MODEL_NAME)
    except Exception as error:
        print("could not draw the plots:", error)

    return model
