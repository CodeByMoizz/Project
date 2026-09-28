import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.preprocessing import prepare_segments
from config.config import PYTHON_MODEL_DIR, TARGET_SAMPLE_RATE

# YAMNet reads a 16 kHz mono waveform and returns one embedding per frame.
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_DIR = PYTHON_MODEL_DIR / "yamnet"

# We pool the frames with mean and max, so the length is 1024 + 1024.
EMBEDDING_LENGTH = 2048

# Loaded once and kept, because loading YAMNet is slow.
_yamnet = {}


def yamnet_is_downloaded():
    return os.path.exists(YAMNET_DIR / "saved_model.pb")


# Downloads YAMNet once so the application can run without internet.
def download_yamnet():
    import tensorflow as tf
    import tensorflow_hub as hub

    os.makedirs(PYTHON_MODEL_DIR, exist_ok=True)
    model = hub.load(YAMNET_URL)
    tf.saved_model.save(model, str(YAMNET_DIR))
    print("saved YAMNet to", YAMNET_DIR)
    return YAMNET_DIR


def load_yamnet():
    if "model" in _yamnet:
        return _yamnet["model"]

    try:
        import tensorflow as tf
    except ImportError:
        return None

    if yamnet_is_downloaded():
        try:
            model = tf.saved_model.load(str(YAMNET_DIR))
        except Exception:
            return None
    else:
        try:
            import tensorflow_hub as hub
            model = hub.load(YAMNET_URL)
        except Exception:
            return None

    _yamnet["model"] = model
    return model


def yamnet_status():
    if yamnet_is_downloaded():
        return {"available": True, "message": f"YAMNet loaded from {YAMNET_DIR}."}

    return {
        "available": False,
        "message": "YAMNet is not downloaded. Run: python feature_extraction/embeddings.py",
    }


# Every embedding in the project is built here, so training matches inference.
def build_embedding(samples, sr=TARGET_SAMPLE_RATE):
    model = load_yamnet()

    if model is None:
        return None

    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    samples = np.nan_to_num(samples)

    scores, frames, spectrogram = model(samples)
    frames = np.asarray(frames)

    if frames.size == 0:
        return np.zeros(EMBEDDING_LENGTH, dtype=np.float32)

    mean_part = np.mean(frames, axis=0)
    max_part = np.max(frames, axis=0)

    embedding = np.concatenate([mean_part, max_part])
    return embedding.astype(np.float32)


# A file is preprocessed first, so dataset clips and uploads travel one path.
def embedding_from_file(file_path):
    segments, sr, info = prepare_segments(file_path)

    if not segments:
        raise ValueError(f"No usable audio in {file_path}")

    return build_embedding(segments[0], sr)


if __name__ == "__main__":
    if yamnet_is_downloaded():
        print("YAMNet is already in", YAMNET_DIR)
    else:
        download_yamnet()
