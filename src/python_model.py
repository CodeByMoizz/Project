import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    ACTIVE_PYTHON_MODEL,
    CLASSES,
    LABEL_ENCODER_FILE,
    PYTHON_MODEL_DIR,
    active_model_config,
)
from feature_extraction.features import (
    FEATURE_VECTOR_LENGTH,
    extract_features_from_samples,
    features_to_vector,
)

# Loaded once and kept, because loading a model per request is slow.
_loaded = {}


def model_paths():
    settings = active_model_config()

    return {
        "model": PYTHON_MODEL_DIR / settings["model_file"],
        "scaler": PYTHON_MODEL_DIR / settings["scaler_file"],
        "encoder": PYTHON_MODEL_DIR / LABEL_ENCODER_FILE,
        "version": settings["version"],
    }


def model_status():
    paths = model_paths()

    missing = [
        name for name in ("model", "scaler") if not os.path.exists(paths[name])
    ]

    if missing:
        return {
            "available": False,
            "name": ACTIVE_PYTHON_MODEL,
            "version": paths["version"],
            "message": (
                f"The '{ACTIVE_PYTHON_MODEL}' model is not trained yet. "
                f"Missing file(s): {', '.join(str(paths[name].name) for name in missing)}. "
                f"Put the trained files in {PYTHON_MODEL_DIR.name}/ and reload."
            ),
        }

    return {
        "available": True,
        "name": ACTIVE_PYTHON_MODEL,
        "version": paths["version"],
        "message": f"Active model: {ACTIVE_PYTHON_MODEL} ({paths['version']}).",
    }


def load_model():
    if _loaded.get("name") == ACTIVE_PYTHON_MODEL:
        return _loaded

    status = model_status()

    if not status["available"]:
        return None

    try:
        import joblib
    except ImportError:
        return None

    paths = model_paths()

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
            "name": ACTIVE_PYTHON_MODEL,
            "version": paths["version"],
            "model": model,
            "scaler": scaler,
            "encoder": encoder,
        }
    )

    return _loaded


def label_for_index(loaded, index):
    encoder = loaded.get("encoder")

    if encoder is not None:
        try:
            return str(encoder.inverse_transform([index])[0])
        except Exception:
            pass

    model = loaded["model"]
    classes = getattr(model, "classes_", None)

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


# Returns confidence scores for all ten classes, or None when no model is
# loaded, so callers can show a message instead of failing.
def get_prediction(samples, sr):
    loaded = load_model()

    if loaded is None:
        return None

    try:
        features = extract_features_from_samples(samples, sr)
        vector = features_to_vector(features)
    except Exception:
        return None

    if len(vector) != FEATURE_VECTOR_LENGTH:
        return None

    try:
        scaled = loaded["scaler"].transform([vector])
        probabilities = loaded["model"].predict_proba(scaled)[0]
    except Exception:
        return None

    scores = empty_scores()

    for index, probability in enumerate(probabilities):
        label = label_for_index(loaded, index)

        if label in scores:
            scores[label] = float(probability)

    return scores
