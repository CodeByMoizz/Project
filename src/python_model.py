import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    ACTIVE_PYTHON_MODEL,
    CLASSES,
    LABEL_ENCODER_FILE,
    PYTHON_MODEL_DIR,
    PYTHON_MODELS,
    selected_model_name,
)
from feature_extraction.features import (
    FEATURE_VECTOR_LENGTH,
    extract_features_from_samples,
    features_to_vector,
)
from feature_extraction.embeddings import (
    EMBEDDING_LENGTH,
    EMBEDDING_MEAN_LENGTH,
    build_embedding,
)
from feature_extraction.spectrogram import build_spectrogram

# Loaded once and kept, because loading a model per request is slow.
_loaded = {}


# selection.json wins when src/yamnet_transfer.py --activate has written one,
# otherwise the ACTIVE_PYTHON_MODEL default in config.py stands.
def active_model_name():
    return selected_model_name() or ACTIVE_PYTHON_MODEL


def active_settings():
    return PYTHON_MODELS[active_model_name()]


def model_kind():
    return active_settings().get("kind", "features")


def model_paths():
    settings = active_settings()

    scaler_file = settings.get("scaler_file")

    return {
        "model": PYTHON_MODEL_DIR / settings["model_file"],
        # A spectrogram model has no scaler, the spectrogram is already scaled.
        "scaler": PYTHON_MODEL_DIR / scaler_file if scaler_file else None,
        "encoder": PYTHON_MODEL_DIR / LABEL_ENCODER_FILE,
        "version": settings["version"],
    }


def model_status():
    paths = model_paths()

    missing = [
        name for name in ("model", "scaler")
        if paths[name] is not None and not os.path.exists(paths[name])
    ]

    if missing:
        return {
            "available": False,
            "name": active_model_name(),
            "version": paths["version"],
            "message": (
                f"The '{active_model_name()}' model is not trained yet. "
                f"Missing file(s): {', '.join(str(paths[name].name) for name in missing)}. "
                f"Put the trained files in {PYTHON_MODEL_DIR.name}/ and reload."
            ),
        }

    return {
        "available": True,
        "name": active_model_name(),
        "version": paths["version"],
        "message": f"Active model: {active_model_name()} ({paths['version']}).",
    }


def load_model():
    if _loaded.get("name") == active_model_name():
        return _loaded

    status = model_status()

    if not status["available"]:
        return None

    try:
        import joblib
    except ImportError:
        return None

    paths = model_paths()
    kind = model_kind()

    if kind in ("spectrogram", "embedding"):
        model = load_keras_model(paths["model"])
        scaler = None

        if model is None:
            return None
    elif kind == "embedding_sklearn":
        # The pipeline carries its own scaler, so only the model is loaded.
        try:
            model = joblib.load(paths["model"])
        except Exception:
            return None

        scaler = None
    else:
        try:
            model = joblib.load(paths["model"])
            scaler = joblib.load(paths["scaler"])
        except Exception:
            return None

    encoder = None
    if os.path.exists(paths["encoder"]):
        try:
            encoder = joblib.load(paths["encoder"])
        except Exception:
            encoder = None

    _loaded.clear()
    _loaded.update(
        {
            "name": active_model_name(),
            "version": paths["version"],
            "kind": kind,
            "model": model,
            "scaler": scaler,
            "encoder": encoder,
        }
    )

    return _loaded


# TensorFlow is only needed for the CNN, so it is imported here rather than at
# the top. A missing TensorFlow gives a clear message instead of breaking the
# whole application.
def load_keras_model(path):
    try:
        import keras
    except ImportError:
        try:
            from tensorflow import keras
        except ImportError:
            return None

    try:
        return keras.models.load_model(path)
    except Exception:
        return None


def label_for_index(loaded, index):
    encoder = loaded.get("encoder")

    if encoder is not None:
        try:
            return str(encoder.inverse_transform([index])[0])
        except Exception:
            pass

    model = loaded["model"]
    classes = getattr(model, "classes_", None)

    if loaded.get("kind") in ("spectrogram", "embedding"):
        classes = None

    if classes is not None and index < len(classes):
        value = classes[index]

        if isinstance(value, (int,)) or str(value).isdigit():
            position = int(value)
            if position < len(CLASSES):
                return CLASSES[position]

        return str(value)

    if index < len(CLASSES):
        return CLASSES[index]

    return "unknown"


def empty_scores():
    return {name: 0.0 for name in CLASSES}


def predict_from_features(loaded, samples, sr):
    try:
        features = extract_features_from_samples(samples, sr)
        vector = features_to_vector(features)
    except Exception:
        return None

    if len(vector) != FEATURE_VECTOR_LENGTH:
        return None

    try:
        scaled = loaded["scaler"].transform([vector])
        return loaded["model"].predict_proba(scaled)[0]
    except Exception:
        return None


def predict_from_spectrogram(loaded, samples, sr):
    import numpy as np

    try:
        spectrogram = build_spectrogram(samples, sr)
    except Exception:
        return None

    try:
        batch = np.expand_dims(spectrogram, axis=0)
        return loaded["model"].predict(batch, verbose=0)[0]
    except Exception:
        return None


def predict_from_embedding(loaded, samples, sr):
    import numpy as np

    embedding = build_embedding(samples, sr)

    if embedding is None or len(embedding) != EMBEDDING_LENGTH:
        return None

    try:
        batch = np.expand_dims(embedding, axis=0)
        return loaded["model"].predict(batch, verbose=0)[0]
    except Exception:
        return None


def predict_from_embedding_sklearn(loaded, samples, sr):
    embedding = build_embedding(samples, sr)

    if embedding is None or len(embedding) != EMBEDDING_LENGTH:
        return None

    model = loaded["model"]

    # The head may have been trained on the mean half only, so the embedding is
    # sliced exactly as it was at training time.
    if getattr(model, "feature_view_", "combined") == "mean":
        embedding = embedding[:EMBEDDING_MEAN_LENGTH]

    try:
        return model.predict_proba([embedding])[0]
    except Exception:
        return None


# Returns confidence scores for all ten classes, or None when no model is
# loaded, so callers can show a message instead of failing.
def get_prediction(samples, sr):
    loaded = load_model()

    if loaded is None:
        return None

    if loaded.get("kind") == "spectrogram":
        probabilities = predict_from_spectrogram(loaded, samples, sr)
    elif loaded.get("kind") == "embedding":
        probabilities = predict_from_embedding(loaded, samples, sr)
    elif loaded.get("kind") == "embedding_sklearn":
        probabilities = predict_from_embedding_sklearn(loaded, samples, sr)
    else:
        probabilities = predict_from_features(loaded, samples, sr)

    if probabilities is None:
        return None

    scores = empty_scores()

    for index, probability in enumerate(probabilities):
        label = label_for_index(loaded, index)

        if label in scores:
            scores[label] = float(probability)

    return scores
